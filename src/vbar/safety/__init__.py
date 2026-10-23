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
# Multipliers may be negative by this fraction of the largest, the rounding of the solver.
_PROOF_SIGN_TOL = 1e-12
# b . lam must exceed the support of the actuator set by this fraction of
# max(a_max, max_i |b_i| / |a_i|) times sum_i lam_i |a_i|, the margin the solver certifies with.
_PROOF_MARGIN = 1e-10
PROOF_LAYOUTS = ("box", "ball")


def _read_only(value, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    array = np.array(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class InfeasibilityProof:
    """A Farkas certificate that rows a_i . u >= b_i cannot all hold under the thrust limit.

    Parameters
    ----------
    normals : array_like, shape (k, 3)
        Row normals a_i on the command (m/s^2), Hill frame; each row keeps its own units:
        unitless for barrier and speed rows, s for end-of-step speed cuts.
    bounds : array_like, shape (k,)
        Right-hand sides b_i: m/s^2 for barrier and speed rows, m/s for speed cuts.
    multipliers : array_like, shape (k,)
        The certificate lam, nonnegative.
    layout : str
        The thrust limit, "box" (|u_i| <= a_max) or "ball" (|u|_2 <= a_max).
    a_max : float
        The thrust limit, m/s^2, positive.
    names : tuple of str, length k
        The rows' names.

    Raises
    ------
    ValueError
        If the shapes disagree, a value is not finite, `layout` is unknown or `a_max` is not
        positive.

    Notes
    -----
    Every u in the actuator set U has lam . (A u) <= max over U of (A^T lam) . u, the support
    a_max |A^T lam|_1 of a box or a_max |A^T lam|_2 of a ball. So b . lam above that support
    with lam >= 0 shows that no u in U meets every row (Boyd and Vandenberghe, *Convex
    Optimization*, sec. 5.8.3). The arrays are read-only copies.
    """

    normals: np.ndarray
    bounds: np.ndarray
    multipliers: np.ndarray
    layout: str
    a_max: float
    names: tuple[str, ...]

    def __post_init__(self):
        normals = _read_only(self.normals, None, "normals")
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError("normals must have shape (k, 3)")
        k = normals.shape[0]
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "bounds", _read_only(self.bounds, (k,), "bounds"))
        object.__setattr__(self, "multipliers", _read_only(self.multipliers, (k,), "multipliers"))
        names = tuple(self.names)
        if len(names) != k or not all(isinstance(name, str) for name in names):
            raise ValueError("names must be k strings")
        object.__setattr__(self, "names", names)
        if self.layout not in PROOF_LAYOUTS:
            raise ValueError(f"unknown layout {self.layout!r}, expected {PROOF_LAYOUTS}")
        a_max = float(self.a_max)
        if not (np.isfinite(a_max) and a_max > 0.0):
            raise ValueError("a_max must be finite and positive")
        object.__setattr__(self, "a_max", a_max)

    def verify(self) -> bool:
        """Return whether the certificate proves the rows infeasible under the thrust limit.

        Returns
        -------
        bool
            True when lam >= 0 to 1e-12 of its largest entry and b . lam exceeds the support of
            the actuator set by 1e-10 of max(a_max, max_i |b_i| / |a_i|) sum_i lam_i |a_i|, the
            relative margin `vbar.solvers.project` certifies with. False otherwise, including for
            an all-zero certificate.
        """
        lam = self.multipliers
        if lam.size == 0:
            return False
        largest = float(np.max(np.abs(lam)))
        if not largest > 0.0 or np.any(lam < -_PROOF_SIGN_TOL * largest):
            return False
        lam = np.maximum(lam, 0.0)
        norms = np.linalg.norm(self.normals, axis=1)
        w = self.normals.T @ lam
        support = self.a_max * float(
            np.sum(np.abs(w)) if self.layout == "box" else np.linalg.norm(w)
        )
        unit = np.where(norms > 0.0, norms, 1.0)
        scale = max(self.a_max, float(np.max(np.abs(self.bounds) / unit)))
        weight = float(lam @ np.where(norms > 0.0, norms, 1.0))
        return float(self.bounds @ lam) - support > _PROOF_MARGIN * scale * weight


@dataclass(frozen=True)
class FilterResult:
    """One filter decision.

    `status` is "feasible", "infeasible_certified" (proven infeasible by a certificate) or
    "solver_failed" (no answer and no proof, a filter defect); `slack` is the constraint violation
    the returned action accepts, `conflicting` names the constraints that could not all hold, and
    `compute_s` is the filter's wall time (s). A filter that prepares its next decision in a
    `prepare()` call has that call's wall time in `prep_s` and the `filter` call's in
    `feedback_s` (s); `run_filter` measures all three, and `prep_s` stays None for a filter
    without `prepare`.

    Filters that know more report it: `active` that a safety row was active (a positive
    multiplier, or a line search that stopped short) even if the action did not move,
    `recovering` whether the state was already outside the filter's safe set,
    `fallback_level` which answer was used (0 the converged solution, 1 the previous plan shifted,
    2 the declared backup), `certificate` the kind of guarantee and `certificate_horizon` how long
    it holds (s, inf for an invariant set), and `iterations` the solver's count. None means not
    reported. An "infeasible_certified" decision may carry its `proof`, which anyone can check
    with `InfeasibilityProof.verify`; no other status may.
    """

    action: np.ndarray
    intervened: bool
    status: str = "feasible"
    slack: float = 0.0
    conflicting: tuple[str, ...] = ()
    compute_s: float = 0.0
    prep_s: float | None = None
    feedback_s: float | None = None
    active: bool = False
    recovering: bool | None = None
    fallback_level: int | None = None
    certificate: str | None = None
    certificate_horizon: float | None = None
    iterations: int | None = None
    proof: InfeasibilityProof | None = None

    def __post_init__(self):
        if self.status not in FILTER_STATUSES:
            raise ValueError(f"unknown filter status {self.status!r}, expected {FILTER_STATUSES}")
        if self.fallback_level not in (None, 0, 1, 2):
            raise ValueError(f"fallback level {self.fallback_level!r} is not 0, 1 or 2")
        if self.certificate_horizon is not None and not self.certificate_horizon > 0.0:
            raise ValueError("a certificate horizon must be positive (inf for an invariant set)")
        if self.proof is not None and self.status != "infeasible_certified":
            raise ValueError("only an infeasible_certified result carries a proof")
        if self.proof is not None and not isinstance(self.proof, InfeasibilityProof):
            raise TypeError("proof must be an InfeasibilityProof")


def run_filter(safety_filter, obs: np.ndarray, command: np.ndarray) -> tuple[FilterResult, float]:
    """The filter's decision on `command` and its measured wall time.

    A filter with a `prepare()` method has it called first, with no arguments, then
    `filter(obs, action)`. One clock runs from the lookup of `prepare` to the return of `filter`
    without a gap: `prep_s` covers the lookup and the `prepare` call, `feedback_s` the copies of
    the inputs, the `filter` call and, for a filter without `prepare`, the lookup. The filter gets
    those copies, so it cannot rewrite the caller's command or observation. The measured times
    replace whatever the filter reported, so a filter cannot report its latency lower or higher
    than it was. Work done outside these calls, in `reset` or on another thread, is not measured.

    Parameters
    ----------
    safety_filter : SafetyFilter
        Any object with `filter(obs, action)` returning a `FilterResult`, and optionally
        `prepare()`.
    obs : np.ndarray
        The observation the agent acted on.
    command : np.ndarray
        The agent's command, in the action mode's units.

    Returns
    -------
    FilterResult
        The filter's result with `prep_s` the wall time of `prepare` (None without it),
        `feedback_s` that of `filter` and `compute_s` their sum (s).
    float
        The same `compute_s` (s).

    Raises
    ------
    TypeError
        If the filter returns anything but a `FilterResult`, a bare action or a subclass
        included; a subclass could rewrite the measured times when the result is rebuilt.
    """
    t0 = time.perf_counter()
    prepare = getattr(safety_filter, "prepare", None)
    prep_s = None
    if prepare is not None:
        prepare()
        t1 = time.perf_counter()
        prep_s = t1 - t0
        t0 = t1
    out = safety_filter.filter(np.array(obs, dtype=float), np.array(command, dtype=float))
    feedback_s = time.perf_counter() - t0
    if type(out) is not FilterResult:
        raise TypeError(f"filter returned {type(out).__name__}, expected FilterResult")
    compute_s = feedback_s if prep_s is None else prep_s + feedback_s
    return replace(out, prep_s=prep_s, feedback_s=feedback_s, compute_s=compute_s), compute_s


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
    `FilterResult` and `last_filter_s` the filter's measured wall time, `prepare` included (s);
    `interventions` counts the decisions the filter reported as `intervened` and `decisions` all
    of them.

    Parameters
    ----------
    agent : Agent
        The wrapped agent; its commands go to the filter.
    safety_filter : SafetyFilter
        Has `reset(view)` and `filter(obs, action)` returning a `FilterResult`, and optionally
        `prepare()`, called before each `filter`.

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


__all__ = [
    "FILTER_STATUSES",
    "PROOF_LAYOUTS",
    "FilterResult",
    "FilteredAgent",
    "InfeasibilityProof",
    "PassThroughFilter",
    "run_filter",
]
