"""Expert datasets: what an agent saw and did on public seeds, as an NPZ file and a card.

`expert_dataset` flies an agent (the `OracleAgent` by default) on the train or validation seeds
of a scenario and keeps the observations, actions, rewards and episode ends, never the true
state. `ExpertDataset.save` writes the arrays as an NPZ file whose bytes depend only on the
data and the zlib build, and the dataset card beside it, a markdown file whose front matter
records the provenance. The test split is refused before anything is read, so no hidden seed
or key can reach a dataset.
"""

from __future__ import annotations

import hashlib
import io
import logging
import operator
import platform
import subprocess
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import scipy

from .. import __version__
from ..agents.base import CONTINUOUS_MODES, Agent, agent_generator
from ..scenarios import make
from ..scenarios.seeds import SPLITS, derive_seed, split_key
from ..spacecraft import PerfectNav
from .expert import OracleAgent

log = logging.getLogger(__name__)

ARRAYS = (
    "observations",
    "actions",
    "rewards",
    "terminations",
    "truncations",
    "episode_starts",
    "seeds",
    "agent_seeds",
)
CARD_FIELDS = (
    "vbar_version",
    "git_revision",
    "scenario_ids",
    "split",
    "indices",
    "agent",
    "oracle_config",
    "numpy_version",
    "scipy_version",
    "highs_version",
    "platform",
    "sha256",
    "e1_summary",
)
BEHAVIOUR = "deterministic behaviour, no action probabilities"
# Zip entries carry a modification time; a fixed one keeps the file's bytes a function of the
# data. 1980-01-01 is the earliest time the format stores.
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_ZIP_LEVEL = 6
_CARD_BODY = f"""
# Expert dataset

Episodes flown by the agent named in the front matter on the public seeds of the listed
scenario and split. Arrays, one row per decision unless stated:

- `observations` (steps + episodes, d): every observation the agent received, the first of each
  episode included, in the scenario's observation units (Hill frame, m and m/s first).
- `actions` (steps, 3): the agent's commands in the scenario's action mode.
- `rewards` (steps,), `terminations` and `truncations` (steps,): what the environment returned.
- `episode_starts` (episodes,): the first step of each episode; episode i has its observations
  from row episode_starts[i] + i.
- `seeds` and `agent_seeds` (episodes,): the episode and agent seeds as 32 hex digits.

No true state, start state or hidden seed is stored. Resetting the scenario with an episode
seed and stepping the stored actions reproduces the stored observations and rewards.

The data have {BEHAVIOUR}. Each action is the only action the expert takes in its state, so
offline value-based reinforcement learning sees no other action and has no coverage to estimate
their values from; behaviour cloning and DAgger, which need only the expert's action, are
unaffected. Agents trained on these data name the dataset and its SHA-256 in their results.

Files are identical byte for byte when rebuilt from the same seeds, release and library
versions on the same platform. On another platform the linear programs can resolve ties
between plans of equal cost differently, so actions and delta-v may differ.
"""


def _hex(seed: int) -> str:
    return f"{seed:032x}"


def _index_ranges(indices: list[int]) -> str:
    """Indices as runs of consecutive values, "0-9, 12"."""
    runs: list[list[int]] = []
    for k in indices:
        if runs and k == runs[-1][1] + 1:
            runs[-1][1] = k
        else:
            runs.append([k, k])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _git_revision() -> str:
    """The commit of the source tree this package runs from, "-dirty" if tracked files
    changed, or "unknown" when it is not run from a checkout of its own repository."""
    root = Path(__file__).resolve().parents[3]
    # An installed package can sit inside an unrelated repository; only a checkout whose
    # root holds this source tree names the code that ran.
    if not (root / ".git").exists() or not (root / "src" / "vbar").is_dir():
        return "unknown"
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v2", "--branch", "-uno"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    lines = listed.stdout.splitlines()
    head = [line.split()[2] for line in lines if line.startswith("# branch.oid ")]
    if not head or head[0] == "(initial)":
        return "unknown"
    dirty = any(not line.startswith("#") for line in lines)
    return head[0] + ("-dirty" if dirty else "")


def _highs_version() -> str:
    """The HiGHS version scipy's `linprog` runs, or "unknown" when scipy does not expose it."""
    # A private module whose layout changes between scipy releases.
    try:
        from scipy.optimize._highspy import _core

        return (
            f"{_core.HIGHS_VERSION_MAJOR}.{_core.HIGHS_VERSION_MINOR}.{_core.HIGHS_VERSION_PATCH}"
        )
    except (ImportError, AttributeError):
        return "unknown"


def _platform() -> str:
    return (
        f"{platform.system()} {platform.release()} {platform.machine()}, "
        f"{platform.python_implementation()} {platform.python_version()}"
    )


def _summary(e1: list[float]) -> str:
    values = np.array(e1)
    return (
        f"mean {values.mean():.9g} m/s, min {values.min():.9g} m/s, max {values.max():.9g} m/s "
        f"over {len(values)} episodes"
    )


def _oracle_config(agent: Agent) -> str:
    if not isinstance(agent, OracleAgent):
        return "none"
    settings = [f"{f.name}={getattr(agent.config, f.name)!r}" for f in fields(agent.config)]
    replan = "none" if agent.replan_steps is None else str(agent.replan_steps)
    return ", ".join([*settings, f"replan_steps={replan}"])


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    lines = text.split("\n")
    if lines[0] != "---" or "---" not in lines[1:]:
        raise ValueError("the dataset card has no front matter")
    end = lines.index("---", 1)
    card = {}
    for line in lines[1:end]:
        key, colon, value = line.partition(":")
        if not colon:
            raise ValueError(f"malformed card line {line!r}")
        card[key] = value.strip()
    return card, "\n".join(lines[end + 1 :])


@dataclass(frozen=True, eq=False)
class ExpertDataset:
    """Episodes of one agent on public seeds of one scenario, and their provenance.

    Parameters
    ----------
    observations : numpy.ndarray, shape (steps + episodes, d)
        Every observation the agent received, each episode's first included, in the scenario's
        observation units (Hill frame, m and m/s first).
    actions : numpy.ndarray, shape (steps, 3)
        The agent's commands in the scenario's action mode.
    rewards : numpy.ndarray, shape (steps,)
        Rewards the environment returned.
    terminations, truncations : numpy.ndarray of bool, shape (steps,)
        Episode ends the environment returned; each episode's last step has one of them.
    episode_starts : numpy.ndarray of int64, shape (episodes,)
        First step of each episode; episode i's observations start at row
        `episode_starts[i] + i`.
    seeds, agent_seeds : numpy.ndarray of str, shape (episodes,)
        Episode and agent seeds as 32 lowercase hex digits.
    card : mapping of str to str
        Front matter of the dataset card (`CARD_FIELDS`) without "sha256", which `save`
        computes from the file.

    Raises
    ------
    ValueError
        If the arrays' shapes or episode ends disagree, or a card field is missing or is not
        one line of text without leading or trailing spaces.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    episode_starts: np.ndarray
    seeds: np.ndarray
    agent_seeds: np.ndarray
    card: Mapping[str, str]

    def __post_init__(self):
        steps, episodes = len(self.actions), len(self.seeds)
        if episodes == 0 or len(self.agent_seeds) != episodes:
            raise ValueError("a dataset needs one seed and one agent seed per episode")
        if len(self.observations) != steps + episodes or self.actions.shape != (steps, 3):
            raise ValueError("observations and actions do not match the episodes")
        if not self.rewards.shape == self.terminations.shape == self.truncations.shape == (steps,):
            raise ValueError("rewards and episode ends need one entry per step")
        starts = np.r_[self.episode_starts, steps]
        if (
            self.episode_starts.shape != (episodes,)
            or starts[0] != 0
            or np.any(np.diff(starts) < 1)
        ):
            raise ValueError("episode_starts must start at 0 and increase")
        ended = self.terminations | self.truncations
        if not np.array_equal(np.flatnonzero(ended), starts[1:] - 1):
            raise ValueError("each episode must end at its last step and only there")
        missing = set(CARD_FIELDS) - {"sha256"} - set(self.card)
        if missing:
            raise ValueError(f"the card misses {sorted(missing)}")
        # Each value is one front matter line and is read back stripped.
        for key, value in self.card.items():
            if not isinstance(value, str) or len(value.splitlines()) > 1 or value != value.strip():
                raise ValueError(f"card field {key!r} must be one line without outer spaces")
        object.__setattr__(self, "card", MappingProxyType(dict(self.card)))

    def save(self, path: str | Path) -> None:
        """Write the arrays to `path` and the dataset card to `path` with the suffix ".md".

        Parameters
        ----------
        path : str or pathlib.Path
            Destination of the NPZ file, ending in ".npz".

        Raises
        ------
        ValueError
            If `path` does not end in ".npz".
        """
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("the dataset path must end in .npz")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in ARRAYS:
                member = io.BytesIO()
                np.lib.format.write_array(member, getattr(self, name), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o644 << 16
                archive.writestr(info, member.getvalue(), compresslevel=_ZIP_LEVEL)
        blob = buffer.getvalue()
        card = {**self.card, "sha256": hashlib.sha256(blob).hexdigest()}
        head = "\n".join(f"{key}: {card[key]}" for key in CARD_FIELDS)
        path.write_bytes(blob)
        path.with_suffix(".md").write_bytes(f"---\n{head}\n---\n{_CARD_BODY}".encode())

    @classmethod
    def load(cls, path: str | Path) -> ExpertDataset:
        """Read a dataset written by `save`, checking the file against the card's SHA-256.

        Parameters
        ----------
        path : str or pathlib.Path
            The NPZ file; its card is `path` with the suffix ".md".

        Returns
        -------
        ExpertDataset
            The dataset.

        Raises
        ------
        ValueError
            If the file's SHA-256 differs from the card's, an array is missing or the arrays
            or card are inconsistent.
        """
        path = Path(path)
        blob = path.read_bytes()
        card, _ = _front_matter(path.with_suffix(".md").read_text(encoding="utf-8"))
        if card.get("sha256") != hashlib.sha256(blob).hexdigest():
            raise ValueError(f"{path.name} does not match the SHA-256 in its card")
        del card["sha256"]
        with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
            missing = set(ARRAYS) - set(npz.files)
            if missing:
                raise ValueError(f"{path.name} misses {sorted(missing)}")
            arrays = {name: npz[name] for name in ARRAYS}
        return cls(**arrays, card=card)


def expert_dataset(
    scenario_id: str,
    split: str,
    indices: Iterable[int],
    *,
    agent: Agent | None = None,
    **config: Any,
) -> ExpertDataset:
    """Fly an agent on public seeds of a scenario and keep what it saw and did.

    Episode k flies `derive_seed(split_key(split), scenario_id, k)` with the agent seed of the
    "agent" stream, the seeds `seeds(scenario_id, split, ...)` gives. The agent gets what
    `vbar.run_episode` gives it: the view, its own random stream, the observations and the
    agent-safe info.

    Parameters
    ----------
    scenario_id : str
        Scenario name, name with variants or full identifier, as `make` takes it; also the
        name the seeds are keyed by.
    split : {"train", "validation"}
        Public seed split.
    indices : iterable of int
        Episode indices, distinct and not negative, in the order flown.
    agent : Agent, optional
        The agent; an `OracleAgent` with the default settings when omitted.
    **config : Any
        `EnvConfig` overrides passed to `make`.

    Returns
    -------
    ExpertDataset
        The episodes and their card.

    Raises
    ------
    ValueError
        If `split` is "test" or unknown, `indices` is empty, repeats an index or holds a
        negative one, the scenario's navigation is not perfect, its action mode is not
        continuous, or the agent returns a command that is not 3 numbers.
    """
    if split == "test":
        raise ValueError("expert datasets use the public train and validation splits, not test")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected 'train' or 'validation'")
    order = [operator.index(k) for k in indices]
    if not order or len(set(order)) != len(order) or min(order) < 0:
        raise ValueError("indices must be distinct, not negative and at least one")
    env = make(scenario_id, **config)
    # With noisy navigation the expert plans from an estimate, and solver ties between nearly
    # equal plans can turn into different trajectories on different platforms.
    if type(env.sensor) is not PerfectNav:
        raise ValueError("expert datasets need perfect navigation, not a :nav variant")
    if env.cfg.action_mode not in CONTINUOUS_MODES:
        raise ValueError(
            f"expert datasets need a continuous action mode, not {env.cfg.action_mode}"
        )
    agent = OracleAgent() if agent is None else agent
    key = split_key(split)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminations: list[bool] = []
    truncations: list[bool] = []
    starts: list[int] = []
    seeds: list[str] = []
    agent_seeds: list[str] = []
    e1: list[float] = []
    for k in order:
        seed = derive_seed(key, scenario_id, k)
        agent_seed = derive_seed(key, scenario_id, k, "agent")
        obs, _ = env.reset(seed=seed)
        agent.reset(env.agent_view(), agent_generator(agent_seed))
        starts.append(len(actions))
        terminated = truncated = False
        while not (terminated or truncated):
            # The agent may change the arrays it is given or returns; the dataset keeps copies.
            observations.append(np.array(obs, dtype=float))
            action = np.array(agent.act(obs, env.agent_info()), dtype=float)
            if action.shape != (3,):
                raise ValueError(f"the agent returned a command of shape {action.shape}, not (3,)")
            actions.append(action.copy())
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
        observations.append(np.array(obs, dtype=float))
        seeds.append(_hex(seed))
        agent_seeds.append(_hex(agent_seed))
        e1.append(env.dv_l2)
        log.info("episode %d of %s: %d steps, E1 %.6g m/s", k, scenario_id, env.steps, env.dv_l2)

    card = {
        "vbar_version": __version__,
        "git_revision": _git_revision(),
        "scenario_ids": env.agent_view().scenario_id,
        "split": split,
        "indices": _index_ranges(order),
        "agent": agent.name,
        "oracle_config": _oracle_config(agent),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "highs_version": _highs_version(),
        "platform": _platform(),
        "e1_summary": _summary(e1),
    }
    return ExpertDataset(
        observations=np.array(observations, dtype=float),
        actions=np.array(actions, dtype=float).reshape(-1, 3),
        rewards=np.array(rewards, dtype=float),
        terminations=np.array(terminations, dtype=bool),
        truncations=np.array(truncations, dtype=bool),
        episode_starts=np.array(starts, dtype=np.int64),
        seeds=np.array(seeds, dtype="<U32"),
        agent_seeds=np.array(agent_seeds, dtype="<U32"),
        card=card,
    )


__all__ = ["ARRAYS", "BEHAVIOUR", "CARD_FIELDS", "ExpertDataset", "expert_dataset"]
