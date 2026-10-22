"""Episodes as metrics read them, and how to record one.

Metrics are functions of a whole recorded episode: the scenario's public description, the true
start state and every step record, with the agent's own command, the filter's result and the
agent's wall time filled in. Nothing here is passed to the agent. `record_pair` flies one seed
with a safety filter on and off, for the paired filter metrics F3, F4 and F8p.
"""

from __future__ import annotations

import secrets
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..agents.base import AgentView, agent_generator
from ..constraints import Constraint
from ..safety import FilteredAgent, PassThroughFilter
from ..scenarios.base import StepRecord
from ..scenarios.targets import Misbriefed, TargetMode

PAIR_ORDERS = ("on", "off")


@dataclass(frozen=True, kw_only=True)
class Episode:
    """One run: public spec, truth and step records.

    `start` is the true state at t = 0 (m, m/s, Hill). `hard` names the constraints whose
    violation fails the run. `isp` (s), `drive` ("accel" or "thrust") and the view's `mass_kg`
    and `thrust_limit` describe the actuator; `dynamics`, `body`, `altitude_m` and `inclination`
    (rad) the propagation model.
    `agent_log` holds the agent's own per-decision log when it keeps one (LLM agents), and
    `first_obs` the observation the first decision was made on. `scenario_id` is the identifier
    of what was flown, which can differ from the one in `view` where the briefing hides the
    target's true mode. `outcome` is the environment's verdict on a goal the metrics cannot
    rebuild from the trajectory (`BaseEnv.task_outcome`; the inspection's coverage), else None.
    `reset_s` is the agent's wall time in `reset` (s). `target_misbriefed` is True when the target
    followed a motion other than the one `view` describes; the recorder reads it from the
    environment's true target mode, and neither the view nor the agent's info carries it.
    `target` is the scenario's true target attitude mode (it may differ from the briefing in the
    goal), for metrics that judge geometry on the target; None for environments without one.
    """

    view: AgentView
    start: np.ndarray
    records: tuple[StepRecord, ...]
    constraints: tuple[Constraint, ...]
    hard: frozenset[str]
    terminated: bool
    truncated: bool
    isp: float
    drive: str = "accel"
    dynamics: str = "cw"
    body: str = "earth"
    altitude_m: float = 500e3
    inclination: float = 0.0
    agent_log: tuple[Mapping[str, Any], ...] = ()
    first_obs: np.ndarray | None = None
    seed: int | None = None
    agent_seed: int | None = None
    scenario_id: str | None = None
    reset_s: float = 0.0
    outcome: Mapping[str, Any] | None = None
    target_misbriefed: bool = False
    target: TargetMode | None = None

    @property
    def duration(self) -> float:
        """Episode length T (s)."""
        return self.records[-1].t if self.records else 0.0

    @property
    def states(self) -> np.ndarray:
        """True states at t = 0 and after every step, shape (steps + 1, 6)."""
        return np.array([self.start, *(r.true_state for r in self.records)])

    @property
    def times(self) -> np.ndarray:
        """Times (s) of `states`."""
        return np.array([0.0, *(r.t for r in self.records)])

    def constraint(self, name: str) -> Constraint | None:
        return next((c for c in self.constraints if c.name == name), None)


def _agent_log(agent, decisions: int) -> tuple[Mapping[str, Any], ...]:
    """The agent's per-decision `log`, or () when it has none or its length does not fit."""
    for holder in (agent, getattr(agent, "agent", None)):
        log = getattr(holder, "log", None)
        if log is None:
            continue
        if len(log) != decisions:
            warnings.warn(
                f"{type(holder).__name__}.log has {len(log)} entries for {decisions} decisions, "
                "so it is not read as a per-decision log",
                stacklevel=3,
            )
            return ()
        return tuple(dict(entry) for entry in log)
    return ()


def record_episode(env, agent, seed: int | None = None, agent_seed: int | None = None) -> Episode:
    """Run one episode and return everything metrics need.

    The agent sees what `vbar.run_episode` gives it: the view, its own random stream seeded from
    a hash of `agent_seed`, the observations and the agent-safe info. Seeds not given are drawn
    (128 bits) and recorded, and `agent_seed` may not equal `seed`. A `FilteredAgent`'s raw
    command and `FilterResult` go into each record, and the agent's latency there excludes the
    filter's measured wall time; other agents' attributes are not read.
    """
    config = getattr(getattr(env, "unwrapped", env), "cfg", None)
    if getattr(config, "training", False):
        raise ValueError("scored runs need an environment built with training=False")
    if seed is not None and agent_seed == seed:
        raise ValueError("agent_seed must differ from seed")
    if seed is None:
        seed = secrets.randbits(128)
    if agent_seed is None:
        agent_seed = secrets.randbits(128)
    obs, info = env.reset(seed=seed)
    start = np.array(info["true_state"], dtype=float)
    first_obs = np.array(obs, dtype=float)
    t0 = time.perf_counter()
    agent.reset(env.agent_view(), agent_generator(agent_seed))
    reset_s = time.perf_counter() - t0
    records = []
    terminated = truncated = False
    while not (terminated or truncated):
        t0 = time.perf_counter()
        action = np.asarray(agent.act(obs, env.agent_info()), dtype=float)
        latency = time.perf_counter() - t0
        # Only FilteredAgent's own bookkeeping is trusted: any agent, a subclass included, could
        # set these names.
        filtered = type(agent) is FilteredAgent
        verdict = agent.last_result if filtered else None
        command = agent.last_command if filtered else action
        filter_time = agent.last_filter_s if filtered else 0.0
        obs, _, terminated, truncated, info = env.step(action)
        records.append(
            replace(
                env.record,
                command=np.array(command, dtype=float),
                filter_result=verdict,
                agent_latency_s=latency - filter_time,
            )
        )
    cfg = env.cfg
    soft = set(getattr(env, "soft_constraints", ()))
    target = getattr(getattr(env, "unwrapped", env), "target", None)
    return Episode(
        view=env.agent_view(),
        start=start,
        records=tuple(records),
        constraints=tuple(env.constraints),
        hard=frozenset(c.name for c in env.constraints if c.name not in soft),
        terminated=bool(terminated),
        truncated=bool(truncated),
        isp=cfg.isp,
        drive=cfg.drive,
        dynamics=cfg.dynamics,
        body=cfg.body,
        altitude_m=cfg.altitude_m,
        inclination=cfg.inclination,
        agent_log=_agent_log(agent, len(records)),
        first_obs=first_obs,
        seed=seed,
        agent_seed=agent_seed,
        scenario_id=env.scenario_id,
        reset_s=reset_s,
        outcome=getattr(env, "task_outcome", lambda: None)(),
        target_misbriefed=isinstance(target, Misbriefed) and target.actual != target.briefed,
        target=target,
    )


@dataclass(frozen=True)
class FilterPair:
    """One seed flown with the filter on and off.

    Attributes
    ----------
    on : Episode
        The run with the filter.
    off : Episode
        The run with the pass-through filter, the same episode and agent seeds.
    filter_name : str
        The name of the filter of the on run.
    first : str
        Which run was flown first, "on" or "off", drawn at random.
    """

    on: Episode
    off: Episode
    filter_name: str
    first: str


def record_pair(
    env,
    make_agent: Callable[[], object],
    safety_filter,
    seed: int | None = None,
    agent_seed: int | None = None,
) -> FilterPair:
    """Fly one episode with `safety_filter` and once without it, from the same seeds.

    Parameters
    ----------
    env : BaseEnv
        The scenario, built with ``training=False``; `reset` rebuilds every random stream from the
        seed, so both runs start from the same state with the same noise.
    make_agent : callable
        Returns a new agent on each call. Each run gets its own, so nothing the agent remembers
        from one run reaches the other.
    safety_filter : SafetyFilter
        The filter of the on run.
    seed, agent_seed : int, optional
        The episode and agent seeds shared by both runs; drawn (128 bits) when not given, and
        `agent_seed` may not equal `seed`.

    Returns
    -------
    FilterPair
        Both episodes, the filter's name and which run came first. The order is drawn at random,
        so a remote agent cannot rely on seeing the filtered run first.

    Raises
    ------
    ValueError
        If `make_agent` returns the same object twice, `agent_seed` equals `seed`, or the
        environment is built for training.
    """
    if seed is not None and agent_seed == seed:
        raise ValueError("agent_seed must differ from seed")
    if seed is None:
        seed = secrets.randbits(128)
    if agent_seed is None:
        agent_seed = secrets.randbits(128)
        while agent_seed == seed:
            agent_seed = secrets.randbits(128)
    agent_on, agent_off = make_agent(), make_agent()
    if agent_on is agent_off:
        raise ValueError("make_agent must return a new agent on every call")
    runs = {
        "on": FilteredAgent(agent_on, safety_filter),
        "off": FilteredAgent(agent_off, PassThroughFilter()),
    }
    first = PAIR_ORDERS[secrets.randbits(1)]
    order = (first, "off" if first == "on" else "on")
    episodes = {name: record_episode(env, runs[name], seed, agent_seed) for name in order}
    return FilterPair(episodes["on"], episodes["off"], safety_filter.name, first)
