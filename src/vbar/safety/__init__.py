"""Safety filters wrap an agent and modify unsafe commands before they are applied.

Only the pass-through filter (filter off) exists so far. A filter returns either the action or a
`FilterResult`; `FilteredAgent` counts interventions, the decisions where the filter changed the
agent's command, and keeps the last command and result for the step record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..agents.base import AgentView

FILTER_STATUSES = ("feasible", "infeasible_certified", "solver_failed")


@dataclass(frozen=True)
class FilterResult:
    """One filter decision.

    `status` is "feasible", "infeasible_certified" (proven infeasible by a certificate) or
    "solver_failed" (no answer and no proof, a filter defect); `slack` is the constraint violation
    the returned action accepts, `conflicting` names the constraints that could not all hold, and
    `compute_s` is the filter's wall time (s).

    Filters that know more report it: `active` that a safety row was active (a positive
    multiplier, or a line search that stopped short) even if the action did not move,
    `recovering` whether the state was already outside the filter's safe set,
    `fallback_level` which answer was used (0 the converged solution, 1 the previous plan shifted,
    2 the declared backup), `certificate` the kind of guarantee and `certificate_horizon` how long
    it holds (s, inf for an invariant set), and `iterations` the solver's count. None means not
    reported.
    """

    action: np.ndarray
    intervened: bool
    status: str = "feasible"
    slack: float = 0.0
    conflicting: tuple[str, ...] = ()
    compute_s: float = 0.0
    active: bool = False
    recovering: bool | None = None
    fallback_level: int | None = None
    certificate: str | None = None
    certificate_horizon: float | None = None
    iterations: int | None = None

    def __post_init__(self):
        if self.status not in FILTER_STATUSES:
            raise ValueError(f"unknown filter status {self.status!r}, expected {FILTER_STATUSES}")


class PassThroughFilter:
    """Does not change the action. The 'safety filter off' condition."""

    name = "none"

    def reset(self, view: AgentView) -> None:
        pass

    def filter(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        return action


class FilteredAgent:
    """Agent + safety filter, presented as a single agent.

    After each `act`, `last_command` is the agent's own command, `last_result` the filter's
    `FilterResult` and `last_filter_s` the filter's measured wall time (s). A filter that returns
    a bare action gets a result with status "feasible", `intervened` when any component moved by
    more than `tol`, and its measured wall time.
    """

    def __init__(self, agent, safety_filter, tol: float = 1e-6):
        self.agent = agent
        self.filter = safety_filter
        self.tol = tol
        self.name = f"{agent.name}+{safety_filter.name}"

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        self.agent.reset(view, rng)
        self.filter.reset(view)
        self.interventions = 0
        self.decisions = 0
        self.last_command: np.ndarray | None = None
        self.last_result: FilterResult | None = None
        self.last_filter_s = 0.0

    def act(self, obs, info):
        raw = np.asarray(self.agent.act(obs, info), dtype=float)
        t0 = time.perf_counter()
        out = self.filter.filter(obs, raw)
        elapsed = time.perf_counter() - t0
        if isinstance(out, FilterResult):
            result = out
        else:
            safe = np.asarray(out, dtype=float)
            moved = bool(np.max(np.abs(safe - raw)) > self.tol)
            result = FilterResult(safe, moved, compute_s=elapsed)
        self.decisions += 1
        self.interventions += int(result.intervened)
        self.last_command = raw
        self.last_result = result
        self.last_filter_s = elapsed
        return np.asarray(result.action, dtype=float)


__all__ = ["FILTER_STATUSES", "FilterResult", "FilteredAgent", "PassThroughFilter"]
