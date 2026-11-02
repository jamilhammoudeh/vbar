"""Safety filters wrap an agent and modify unsafe commands before they are applied.

This module holds what every filter shares: `FilterResult`, the decision a filter returns, and
`InfeasibilityProof`, the certificate it may carry; `run_filter`, which calls a filter and times
it; `FilteredAgent`, which presents an agent and a filter as one agent; and `PassThroughFilter`,
the filter-off condition. A filter's `FilterResult` has `intervened`, the filter's own report;
`FilteredAgent` counts those interventions and keeps the last command and result for the step
record.

The filters are in submodules: `cbf` (`CbfFilter`, `SoftCbfFilter`, `BackupFilter`), `predictive`
(`PredictiveFilter`), `corridor` (`CorridorCamFilter`) and `rta` (the four run time assurance
filters). `barriers`, `sampled`, `projections` and `tightening` hold the barriers, the held-step
check, the projections and the error margins they are built from, and `audit` flies a filter
around a nominal agent and judges the run.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from ..agents.base import CONTINUOUS_MODES, AgentView

FILTER_STATUSES = ("feasible", "infeasible_certified", "solver_failed")
CERTIFICATE_KINDS = ("nominal", "bounded-W", "chance-epsilon", "finite-horizon")
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
    reported. The kind is one of `CERTIFICATE_KINDS`: "nominal" for the model without errors,
    "bounded-W" for every error in the declared sets, "chance-epsilon" when a declared set is left
    with a stated probability, and "finite-horizon" for a guarantee up to a finite horizon only;
    a kind and its horizon are given together or not at all. An "infeasible_certified" decision
    may carry its `proof`, which anyone can check with `InfeasibilityProof.verify`; no other
    status may.
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
        # The exact type, so a str subclass cannot compare equal to a kind it is not.
        if self.certificate is not None and (
            type(self.certificate) is not str or self.certificate not in CERTIFICATE_KINDS
        ):
            raise ValueError(
                f"unknown certificate {self.certificate!r}, expected {CERTIFICATE_KINDS}"
            )
        if (self.certificate is None) != (self.certificate_horizon is None):
            raise ValueError("a certificate and its horizon are given together")
        if self.certificate_horizon is not None and not self.certificate_horizon > 0.0:
            raise ValueError("a certificate horizon must be positive (inf for an invariant set)")
        if self.certificate == "finite-horizon" and math.isinf(self.certificate_horizon):
            raise ValueError("a finite-horizon certificate needs a finite horizon")
        if self.proof is not None and self.status != "infeasible_certified":
            raise ValueError("only an infeasible_certified result carries a proof")
        if self.proof is not None and not isinstance(self.proof, InfeasibilityProof):
            raise TypeError("proof must be an InfeasibilityProof")


# The exact type each field of a filter's result must have. A subclass or a numpy scalar runs its
# own code when it is converted or compared, which would move the filter's work past the clock.
_NONE = type(None)
_PLAIN_FIELDS: dict[str, tuple[type, ...]] = {
    "intervened": (bool, np.bool_),
    "status": (str,),
    "slack": (float,),
    "conflicting": (tuple,),
    "compute_s": (float,),
    "prep_s": (float, _NONE),
    "feedback_s": (float, _NONE),
    "active": (bool, np.bool_),
    "recovering": (bool, np.bool_, _NONE),
    "fallback_level": (int, _NONE),
    "certificate": (str, _NONE),
    "certificate_horizon": (float, _NONE),
    "iterations": (int, _NONE),
}
_NUMBER_TYPES = (bool, int, float)


def _require_exact(value, kinds: tuple[type, ...], name: str) -> None:
    # `is` on the types, since comparing them with `in` would run a metaclass's `__eq__`.
    if not any(type(value) is kind for kind in kinds):
        expected = " or ".join(sorted({kind.__name__ for kind in kinds}))
        raise TypeError(f"filter result {name} is {type(value).__name__}, expected {expected}")


def _float_array(value, name: str) -> np.ndarray:
    """A new float array of `value`, converted here so that the conversion is timed.

    The copy is explicit because an `__array__` may ignore copy=True and hand back a buffer the
    filter keeps writing to. The dtype is read before any float conversion, since astype parses
    strings and bytes; an object array, which numpy makes of integers beyond 64 bits, passes only
    when every element is exactly a bool, int or float.
    """
    try:
        array = np.asarray(value)
        kind = array.dtype.kind
        if kind in "biuf" or (
            kind == "O"
            and all(any(type(item) is plain for plain in _NUMBER_TYPES) for item in array.flat)
        ):
            return array.astype(float)
    except (TypeError, ValueError, OverflowError) as err:
        raise ValueError(f"filter {name} does not convert to floats") from err
    # Complex is refused, not cast: astype drops the imaginary part and only warns.
    what = "complex" if kind == "c" else "not numeric"
    raise ValueError(f"filter {name} is {what}")


def _plain_proof(proof) -> InfeasibilityProof:
    _require_exact(proof, (InfeasibilityProof,), "proof")
    _require_exact(proof.layout, (str,), "proof layout")
    _require_exact(proof.a_max, (float,), "proof a_max")
    _require_exact(proof.names, (tuple,), "proof names")
    for name in proof.names:
        _require_exact(name, (str,), "proof name")
    return InfeasibilityProof(
        normals=_float_array(proof.normals, "proof normals"),
        bounds=_float_array(proof.bounds, "proof bounds"),
        multipliers=_float_array(proof.multipliers, "proof multipliers"),
        layout=proof.layout,
        a_max=proof.a_max,
        names=proof.names,
    )


def run_filter(safety_filter, obs: np.ndarray, command: np.ndarray) -> tuple[FilterResult, float]:
    """The filter's decision on `command` and its measured wall time.

    A filter with a `prepare()` method has it called first, with no arguments, then
    `filter(obs, action)`. One clock runs from the lookup of `prepare` to the return of `filter`
    without a gap: `prep_s` covers the lookup and the `prepare` call, `feedback_s` the copies of
    the inputs, the `filter` call, the conversion and checks of the returned result and, for a
    filter without `prepare`, the lookup. The filter gets those copies, so it cannot rewrite the
    caller's command or observation. The measured times replace whatever the filter reported, so a
    filter cannot report its latency lower or higher than it was, nor defer its work to an object
    in the result that computes when it is converted or compared. Inside the clock every field of
    the result is checked for its exact type, never a subclass and, for the float and int fields,
    never a numpy scalar; the flags may be `bool` or `numpy.bool_`. The action and the arrays of a
    proof become new float arrays, and the result is rebuilt from those plain values, so it holds
    nothing left to compute and nothing the filter can write to later.

    The clock measures the filter's calls, the conversion of what it returns and the release of
    the objects it returns and does not keep, so their finalizers are timed. Work a filter does
    outside them, in `reset`, on another thread or in the finalizer of an object it keeps or leaves
    in a reference cycle, is not measured; time an untrusted filter in its own process.

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
        The filter's result as plain values, with `action` a float array of the command's shape,
        the flags `bool`, `prep_s` the wall time of `prepare` (None without it), `feedback_s` that
        of `filter` and `compute_s` their sum (s).
    float
        The same `compute_s` (s).

    Raises
    ------
    TypeError
        If the filter returns anything but a `FilterResult`, a subclass included; a subclass could
        compute in the reads of its fields. Also if a field of the result, or the layout, `a_max`
        or names of its proof, is not of its exact type, or its proof is not an
        `InfeasibilityProof` itself.
    ValueError
        If the action or an array of the proof is not numeric (its dtype is not boolean, integer
        or float: a string, bytes, a complex value, a mixed list, or an object array with an
        element that is not a plain bool, int or float), does not convert to floats (a dict) or
        holds an integer too large for a float; if the action is not finite or not of the
        command's shape; or if the rebuilt result fails the checks of `FilterResult`.
    """
    t0 = time.perf_counter()
    prepare = getattr(safety_filter, "prepare", None)
    prep_s = None
    if prepare is not None:
        prepare()
        # Released inside the clock, so its finalizer is timed when the filter does not keep it.
        del prepare
        t1 = time.perf_counter()
        prep_s = t1 - t0
        t0 = t1
    out = safety_filter.filter(np.array(obs, dtype=float), np.array(command, dtype=float))
    if type(out) is not FilterResult:
        raise TypeError(f"filter returned {type(out).__name__}, expected FilterResult")
    fields = {}
    for name, kinds in _PLAIN_FIELDS.items():
        fields[name] = getattr(out, name)
        _require_exact(fields[name], kinds, name)
    for name in fields["conflicting"]:
        _require_exact(name, (str,), "conflicting name")
    for flag in ("intervened", "active", "recovering"):
        if fields[flag] is not None:
            fields[flag] = bool(fields[flag])
    action = _float_array(out.action, "action")
    if action.shape != np.shape(command):
        raise ValueError(f"filter action has shape {action.shape}, expected {np.shape(command)}")
    if not np.all(np.isfinite(action)):
        raise ValueError("filter action is not finite")
    proof = None if out.proof is None else _plain_proof(out.proof)
    # The measured times cannot go in before the clock stops, so they are set on this copy of
    # plain values after it.
    plain = FilterResult(action=action, proof=proof, **fields)
    # The last reference to what the filter returned. Letting go here runs the finalizers of the
    # parts it does not keep inside the clock, not when the frame ends after it.
    del out
    feedback_s = time.perf_counter() - t0
    compute_s = feedback_s if prep_s is None else prep_s + feedback_s
    return (
        replace(plain, prep_s=prep_s, feedback_s=feedback_s, compute_s=compute_s),
        compute_s,
    )


class PassThroughFilter:
    """Does not change the action. The 'safety filter off' condition."""

    name = "none"

    def reset(self, view: AgentView) -> None:
        pass

    def filter(self, obs: np.ndarray, action: np.ndarray) -> FilterResult:
        return FilterResult(action, intervened=False)


class FilteredAgent:
    """Agent + safety filter, presented as a single agent.

    After each `act`, `last_command` is the agent's own command, `last_result` the `FilterResult`
    that `run_filter` rebuilt from the filter's and `last_filter_s` the filter's measured wall
    time, `prepare` included (s); `interventions` counts the decisions the filter reported as
    `intervened` and `decisions` all of them. Only the rebuilt plain values are read, never the
    filter's own objects, so what the filter returns cannot compute after the clock has stopped.
    The clock measures the filter's calls and the conversion of what it returns; work it does
    outside them, on another thread or in the finalizer of an object it keeps or leaves in a
    reference cycle, is not measured,
    so an untrusted filter is timed in its own process; the finalizers of objects it returns and
    does not keep are timed.

    Parameters
    ----------
    agent : Agent
        The wrapped agent; its commands go to the filter. An agent with an `applied(action)`
        method is told the command that was flown after each decision, for a state estimate that
        predicts from its own commands.
    safety_filter : SafetyFilter
        Has `reset(view)` and `filter(obs, action)` returning a `FilterResult`, and optionally
        `prepare()`, called before each `filter`.

    Raises
    ------
    ValueError
        At `reset`, for an action mode that is not continuous.
    TypeError
        At `act`, when the filter returns anything but a `FilterResult` or a field of it is not of
        its exact type (see `run_filter`).
    ValueError
        At `act`, when the filter's result fails the checks of `FilterResult` or its action is
        not numeric, is too large for a float, or is not a finite float array of the command's
        shape (see `run_filter`).
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
        applied = getattr(self.agent, "applied", None)
        if applied is not None:
            applied(result.action)
        return result.action


__all__ = [
    "CERTIFICATE_KINDS",
    "FILTER_STATUSES",
    "PROOF_LAYOUTS",
    "FilterResult",
    "FilteredAgent",
    "InfeasibilityProof",
    "PassThroughFilter",
    "run_filter",
]
