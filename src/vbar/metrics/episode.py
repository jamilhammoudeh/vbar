"""Episodes as metrics read them, and how to record one.

Metrics are functions of a whole recorded episode: the scenario's public description, the true
start state and every step record, with the agent's own command, the filter's result and the
agent's wall time filled in. Nothing here is passed to the agent.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..agents.base import AgentView, agent_generator
from ..constraints import Constraint
from ..safety import FilteredAgent
from ..scenarios.base import StepRecord


@dataclass(frozen=True, kw_only=True)
class Episode:
    """One run: public spec, truth and step records.

    `start` is the true state at t = 0 (m, m/s, Hill). `hard` names the constraints whose
    violation fails the run. `isp` (s) and the view's `mass_kg` and `thrust_limit` describe the
    actuator; `dynamics`, `body`, `altitude_m` and `inclination` (rad) the propagation model.
    `agent_log` holds the agent's own per-decision log when it keeps one (LLM agents), and
    `first_obs` the observation the first decision was made on.
    """

    view: AgentView
    start: np.ndarray
    records: tuple[StepRecord, ...]
    constraints: tuple[Constraint, ...]
    hard: frozenset[str]
    terminated: bool
    truncated: bool
    isp: float
    dynamics: str = "cw"
    body: str = "earth"
    altitude_m: float = 500e3
    inclination: float = 0.0
    agent_log: tuple[Mapping[str, Any], ...] = ()
    first_obs: np.ndarray | None = None
    seed: int | None = None
    agent_seed: int | None = None

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


def _agent_log(agent) -> tuple[Mapping[str, Any], ...]:
    for holder in (agent, getattr(agent, "agent", None)):
        log = getattr(holder, "log", None)
        if log is not None:
            return tuple(dict(entry) for entry in log)
    return ()


def record_episode(env, agent, seed: int | None = None, agent_seed: int | None = None) -> Episode:
    """Run one episode and return everything metrics need.

    The agent sees what `vbar.run_episode` gives it: the view, its own random stream seeded by
    `agent_seed` (drawn from the operating system when not given), the observations and the
    agent-safe info. A `FilteredAgent`'s raw command and `FilterResult` go into each record, and
    the agent's latency there excludes the filter's measured wall time; other agents' attributes
    are not read.
    """
    if agent_seed is None:
        agent_seed = secrets.randbits(128)
    obs, info = env.reset(seed=seed)
    start = np.array(info["true_state"], dtype=float)
    first_obs = np.array(obs, dtype=float)
    agent.reset(env.agent_view(), agent_generator(agent_seed))
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
    return Episode(
        view=env.agent_view(),
        start=start,
        records=tuple(records),
        constraints=tuple(env.constraints),
        hard=frozenset(c.name for c in env.constraints if c.name not in soft),
        terminated=bool(terminated),
        truncated=bool(truncated),
        isp=cfg.isp,
        dynamics=cfg.dynamics,
        body=cfg.body,
        altitude_m=cfg.altitude_m,
        inclination=cfg.inclination,
        agent_log=_agent_log(agent),
        first_obs=first_obs,
        seed=seed,
        agent_seed=agent_seed,
    )
