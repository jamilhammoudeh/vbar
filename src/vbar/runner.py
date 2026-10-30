"""Run episodes and collect benchmark metrics."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

import numpy as np

from .agents.base import agent_generator
from .metrics.episode import FilterPair, record_pair
from .metrics.filter import paired_filter_metrics
from .safety import FilteredAgent
from .scenarios.base import check_count


@dataclass(frozen=True, kw_only=True)
class RealTime:
    """The simulation clock runs while the agent decides.

    Each command takes effect `time_scale` times the agent's wall-clock decision time (s) after
    the observation it answers, plus any actuation delay; meanwhile the previous command stays
    on. `latencies` (wall s, one per decision) replace the measured times, to repeat a recorded
    run exactly.
    """

    time_scale: float = 1.0
    latencies: tuple[float, ...] | None = None

    def __post_init__(self):
        if not (np.isfinite(self.time_scale) and self.time_scale >= 0.0):
            raise ValueError("time_scale must be finite and not negative")


class ScoredRun:
    """One scored episode, stepped by its caller; the reward never leaves it.

    A `seed` or `agent_seed` not given is drawn from the operating system (128 bits) and returned
    in the result, so the run can be repeated. The agent stream is seeded from a hash of
    `agent_seed` alone. An agent can still find a small `agent_seed` by trying candidates against
    its generator, and a small `seed` by matching its first observation, so scored comparisons
    use drawn seeds, and `agent_seed` may not equal `seed`. Environments built for training
    (`training=True`) are refused.
    """

    def __init__(
        self,
        env,
        seed: int | None = None,
        agent_seed: int | None = None,
        real_time: RealTime | None = None,
    ):
        config = getattr(getattr(env, "unwrapped", env), "cfg", None)
        if getattr(config, "training", False):
            raise ValueError("scored runs need an environment built with training=False")
        if seed is not None and agent_seed == seed:
            raise ValueError("agent_seed must differ from seed")
        self.env = env
        self.seed = secrets.randbits(128) if seed is None else seed
        self.agent_seed = secrets.randbits(128) if agent_seed is None else agent_seed
        self.real_time = real_time
        self.obs, self.info = env.reset(seed=self.seed)
        self.states = [self.info["true_state"]]
        self.actions: list[np.ndarray] = []
        self.latencies: list[float] = []
        self.done = False
        self.started = time.perf_counter()

    def agent_rng(self) -> np.random.Generator:
        return agent_generator(self.agent_seed)

    def step(self, action, wall_s: float = 0.0, latency_s: float | None = None) -> np.ndarray:
        """Fly `action` and return the next observation.

        `wall_s` is the agent's decision time; `latency_s` delays the command's effect, in
        real-time runs only.
        """
        if self.done:
            raise RuntimeError("the episode has ended")
        # A copy: an agent may reuse one array for every command.
        action = np.array(action, dtype=float)
        if latency_s is None:
            self.obs, _, terminated, truncated, self.info = self.env.step(action)
        else:
            self.obs, _, terminated, truncated, self.info = self.env.step(
                action, latency_s=latency_s
            )
        self.states.append(self.info["true_state"])
        self.actions.append(action)
        self.latencies.append(wall_s)
        self.done = terminated or truncated
        return self.obs

    def result(self, agent_name: str) -> dict:
        info = self.info
        return {
            "agent": agent_name,
            "seed": self.seed,
            "agent_seed": self.agent_seed,
            # The clock setting changes the trajectory, so it is part of the record.
            "real_time": None
            if self.real_time is None
            else {
                "time_scale": self.real_time.time_scale,
                "replayed": self.real_time.latencies is not None,
            },
            "captured": info["captured"],
            "keep_out_violated": info["keep_out_violated"],
            "violation_steps": info["violation_steps"],
            "min_range_to_target_m": info["min_range_to_target_m"],
            "final_dist_to_hold_m": info["dist_to_hold_m"],
            "dv_used_mps": info["dv_used_mps"],
            "time_s": info["time_s"],
            "completed": self.done,
            "wall_time_s": time.perf_counter() - self.started,
            "latencies_s": np.array(self.latencies),
            "states": np.array(self.states),
            "actions": np.array(self.actions),
        }


def run_episode(
    env,
    agent,
    seed: int | None = None,
    agent_seed: int | None = None,
    real_time: RealTime | None = None,
) -> dict:
    """Run one episode. Returns final metrics plus the true trajectory and the actions.

    The agent gets the scenario's `AgentView`, a random stream seeded from a hash of
    `agent_seed`, the observations and the agent-safe info; the true states come from `info`.
    Seeds not given are drawn (128 bits) and returned in the result; small seeds can be guessed
    by the agent, and `agent_seed` may not equal `seed` (`ScoredRun`). Environments built for
    training (`training=True`) are refused. `latencies_s` holds the
    agent's wall-clock time per decision and `reset_s` its time in `reset`; with `real_time` the
    clock runs while it decides.
    """
    episode = ScoredRun(env, seed, agent_seed, real_time)
    t0 = time.perf_counter()
    agent.reset(env.agent_view(), episode.agent_rng())
    reset_s = time.perf_counter() - t0
    replay = None if real_time is None else real_time.latencies
    while not episode.done:
        t0 = time.perf_counter()
        action = agent.act(episode.obs, env.agent_info())
        wall = time.perf_counter() - t0
        latency = None
        if real_time is not None:
            k = len(episode.actions)
            if replay is not None:
                if k >= len(replay):
                    raise ValueError(f"real_time.latencies ran out at decision {k}")
                wall = replay[k]
            latency = real_time.time_scale * wall
        episode.step(action, wall, latency)
    result = episode.result(agent.name)
    # Work moved from `act` into `reset` is still the agent's compute.
    result["reset_s"] = reset_s
    # Any agent, a subclass included, can set `interventions`; only FilteredAgent's own count holds.
    if type(agent) is FilteredAgent:
        result["filter_interventions"] = agent.interventions
    return result


def run_many(env, agent, seeds, agent_seeds=None) -> list[dict]:
    """Run one episode per seed, with one `agent_seeds` entry per seed if given."""
    seeds = list(seeds)
    if agent_seeds is None:
        agent_seeds = [None] * len(seeds)
    return [run_episode(env, agent, s, a) for s, a in zip(seeds, agent_seeds, strict=True)]


def run_pairs(
    env, make_agent, safety_filter, seeds=None, agent_seeds=None, count: int | None = None
) -> tuple[list[FilterPair], dict]:
    """Fly each seed with the filter on and off and score the pairs.

    Parameters
    ----------
    env : BaseEnv
        The scenario, built with ``training=False``.
    make_agent : callable
        Returns a new agent on each call; each run gets its own.
    safety_filter : SafetyFilter
        The filter of the on runs.
    seeds, agent_seeds : sequence of int, optional
        One episode seed and one agent seed per pair; drawn (128 bits) when not given.
    count : int, optional
        The number of pairs when `seeds` is not given, at least 1.

    Returns
    -------
    list of FilterPair
        The pairs in seed order.
    dict
        `vbar.metrics.paired_filter_metrics` over them: F3, F4 and F8p.

    Raises
    ------
    ValueError
        If neither `seeds` nor `count` is given, `count` is not an integer of at least 1, or a
        pair is refused by `record_pair`.
    """
    if seeds is None:
        if count is None:
            raise ValueError("give seeds or count")
        seeds = [None] * check_count("count", count, 1)
    seeds = list(seeds)
    if agent_seeds is None:
        agent_seeds = [None] * len(seeds)
    pairs = [
        record_pair(env, make_agent, safety_filter, seed, agent_seed)
        for seed, agent_seed in zip(seeds, agent_seeds, strict=True)
    ]
    return pairs, paired_filter_metrics([p.on for p in pairs], [p.off for p in pairs])
