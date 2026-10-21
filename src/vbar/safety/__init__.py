"""Safety filters wrap an agent and modify unsafe commands before they are applied.

Only the pass-through filter (filter off) exists so far. A filter returns a `FilterResult`, whose
`intervened` is the filter's own report; `FilteredAgent` counts those interventions and keeps the
last command and result for the step record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from ..agents.base import CONTINUOUS_MODES, AgentView

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
        if self.fallback_level not in (None, 0, 1, 2):
            raise ValueError(f"fallback level {self.fallback_level!r} is not 0, 1 or 2")
        if self.certificate_horizon is not None and not self.certificate_horizon > 0.0:
            raise ValueError("a certificate horizon must be positive (inf for an invariant set)")


def run_filter(safety_filter, obs: np.ndarray, command: np.ndarray) -> tuple[FilterResult, float]:
    """The filter's decision on `command` and its measured wall time.

    The filter gets copies, so it cannot rewrite the caller's command or observation. The
    measured time replaces the filter's own `compute_s`, so a filter cannot report its latency
    lower than it was.

    Parameters
    ----------
    safety_filter : SafetyFilter
        Any object with `filter(obs, action)` returning a `FilterResult`.
    obs : np.ndarray
        The observation the agent acted on.
    command : np.ndarray
        The agent's command, in the action mode's units.

    Returns
    -------
    FilterResult
        The filter's result with `compute_s` set to the measured wall time.
    float
        The wall time of the `filter` call (s).

    Raises
    ------
    TypeError
        If the filter returns anything but a `FilterResult`, a bare action included.
    """
    t0 = time.perf_counter()
    out = safety_filter.filter(np.array(obs, dtype=float), np.array(command, dtype=float))
    elapsed = time.perf_counter() - t0
    if not isinstance(out, FilterResult):
        raise TypeError(f"filter returned {type(out).__name__}, expected FilterResult")
    return replace(out, compute_s=elapsed), elapsed


class PassThroughFilter:
    """Does not change the action. The 'safety filter off' condition."""

    name = "none"

    def reset(self, view: AgentView) -> None:
        pass

    def filter(self, obs: np.ndarray, action: np.ndarray) -> FilterResult:
        return FilterResult(action, intervened=False)


class FilteredAgent:
    """Agent + safety filter, presented as a single agent.

    After each `act`, `last_command` is the agent's own command, `last_result` the filter's
    `FilterResult` and `last_filter_s` the filter's measured wall time (s); `interventions` counts
    the decisions the filter reported as `intervened` and `decisions` all of them.

    Parameters
    ----------
    agent : Agent
        The wrapped agent; its commands go to the filter.
    safety_filter : SafetyFilter
        Has `reset(view)` and `filter(obs, action)` returning a `FilterResult`.

    Raises
    ------
    ValueError
        At `reset`, for an action mode that is not continuous.
    TypeError
        At `act`, when the filter returns anything but a `FilterResult`.
    """

    def __init__(self, agent, safety_filter):
        self.agent = agent
        self.filter = safety_filter
        self.name = f"{agent.name}+{safety_filter.name}"

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        if view.action_mode not in CONTINUOUS_MODES:
            raise ValueError(f"filters need a continuous action mode, not {view.action_mode!r}")
        self.agent.reset(view, rng)
        self.filter.reset(view)
        self.interventions = 0
        self.decisions = 0
        self.last_command: np.ndarray | None = None
        self.last_result: FilterResult | None = None
        self.last_filter_s = 0.0

    def act(self, obs, info):
        # Own copies, so a filter that writes into its input cannot rewrite the recorded command.
        raw = np.array(self.agent.act(obs, info), dtype=float)
        result, elapsed = run_filter(self.filter, obs, raw)
        self.decisions += 1
        self.interventions += int(result.intervened)
        self.last_command = raw
        self.last_result = result
        self.last_filter_s = elapsed
        return np.asarray(result.action, dtype=float)


__all__ = ["FILTER_STATUSES", "FilterResult", "FilteredAgent", "PassThroughFilter", "run_filter"]
