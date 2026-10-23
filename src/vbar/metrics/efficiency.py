"""Efficiency metrics E1 to E6: delta-v, propellant, burns, duty cycle and chatter.

All of them read the executed acceleration, the spacecraft's `Actuation` segments, so delays,
the minimum impulse bit, run-out and safety filters are counted as flown, not as commanded.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..agents.base import AgentView
from ..conventions import G0
from .episode import Episode
from .mission import success

# A run this far below its certified lower bound is reported; closer is rounding.
_BOUND_TOL = 1e-9
# Delivered acceleration this far above a bound's limit voids it; closer is rounding.
_ACCEL_TOL = 1e-12


def _segments(ep: Episode) -> Iterator[tuple[int, float, np.ndarray]]:
    """(step index, duration s, Hill acceleration m/s^2) of every executed segment."""
    for k, rec in enumerate(ep.records):
        for duration, accel in rec.actuation.segments:
            yield k, float(duration), np.asarray(accel, dtype=float)


def delta_v(ep: Episode) -> float:
    """E1: sum of |a|_2 tau over the executed segments, as the actuator integrated it (m/s)."""
    return float(sum(rec.actuation.dv_l2 for rec in ep.records))


def delta_v_axes(ep: Episode) -> float:
    """E1b: the same with the 1-norm, the cost of separate thrusters per axis (m/s)."""
    return float(sum(rec.actuation.dv_l1 for rec in ep.records))


def layout_delta_v(ep: Episode) -> float:
    """Delta-v in the norm the thruster layout spends propellant on: E1b for box, E1 for ball."""
    return delta_v_axes(ep) if ep.view.thrust_limit == "box" else delta_v(ep)


def propellant_mass(ep: Episode) -> float:
    """E2: m0 (1 - exp(-dv / (Isp g0))) with the layout's delta-v (kg)."""
    if ep.view.mass_kg is None:
        raise ValueError("propellant mass needs the initial mass")
    return -ep.view.mass_kg * math.expm1(-layout_delta_v(ep) / (ep.isp * G0))


def burn_count(ep: Episode) -> int:
    """E4: number of maximal runs of consecutive steps with any executed thrust."""
    thrusting = np.zeros(len(ep.records), dtype=bool)
    for k, _, accel in _segments(ep):
        thrusting[k] |= bool(np.any(accel))
    return int(np.sum(thrusting[1:] & ~thrusting[:-1]) + (thrusting[0] if len(thrusting) else 0))


def duty_cycle(ep: Episode) -> float:
    """E5: time with any executed thrust divided by the episode length."""
    on = sum(duration for _, duration, accel in _segments(ep) if np.any(accel))
    return on / ep.duration if ep.duration > 0.0 else 0.0


def chatter(ep: Episode) -> float:
    """E6: sign changes of each axis between consecutive nonzero accelerations, per second."""
    accels = np.array([accel for _, _, accel in _segments(ep)]).reshape(-1, 3)
    changes = 0
    for axis in accels.T:
        signs = np.sign(axis[axis != 0.0])
        changes += int(np.sum(signs[1:] != signs[:-1]))
    return changes / ep.duration if ep.duration > 0.0 else 0.0


@dataclass(frozen=True)
class FuelBounds:
    """Bounds on the least delta-v from a start state to the scenario's success set.

    Both bounds are for the same success set, time window and actuator norm (Boyd and
    Vandenberghe, Convex Optimization, sections 5.1 and 5.5). A lower bound is a theorem only for
    the dynamics model it was computed on and for acceleration histories within the limit it
    assumed; for any other run it is a model bound, since the true optimum there can lie on
    either side of it.

    Parameters
    ----------
    lower : float
        Certified (dual-feasible) lower bound (m/s).
    upper : float
        Best known feasible cost (m/s); inf when no plan was verified.
    norm : {"l1", "l2"}
        Actuator norm: "l1" for box thrusters, "l2" for a ball.
    rung : str
        `Episode.dynamics` name of the model the bounds are computed on, such as "cw".
    accel_limit : float
        Acceleration `lower` assumes the run never exceeds (m/s^2, Hill frame), per axis for
        "l1" and in the 2-norm for "l2". The default inf states a bound that holds for any
        acceleration, as an impulsive one does.
    impulsive_lower : float or None
        Lower bound for impulsive control on the same rung (m/s), which holds for any
        acceleration; used in place of `lower` for a run that exceeded `accel_limit`.

    Raises
    ------
    ValueError
        For an unknown norm, bounds outside 0 <= lower <= upper, a negative or NaN
        `accel_limit`, or a negative or non-finite `impulsive_lower` or one above `upper`.
    """

    lower: float
    upper: float
    norm: str
    rung: str
    accel_limit: float = math.inf
    impulsive_lower: float | None = None

    def __post_init__(self):
        if self.norm not in ("l1", "l2"):
            raise ValueError(f"unknown norm {self.norm!r}, expected l1 or l2")
        if not 0.0 <= self.lower <= self.upper:
            raise ValueError("bounds must satisfy 0 <= lower <= upper")
        if not self.accel_limit >= 0.0:
            raise ValueError("accel_limit must be a nonnegative number or inf")
        if self.impulsive_lower is not None and not (
            math.isfinite(self.impulsive_lower) and self.impulsive_lower >= 0.0
        ):
            raise ValueError("impulsive_lower must be finite and nonnegative")
        if self.impulsive_lower is not None and self.impulsive_lower > self.upper:
            raise ValueError("impulsive_lower must not exceed upper")


class Oracle(Protocol):
    """Gives fuel bounds for a start state."""

    def bounds(self, view: AgentView, start: np.ndarray) -> FuelBounds: ...


def _peak_accel(ep: Episode) -> float:
    """Largest delivered acceleration in the record (m/s^2): per axis for box, 2-norm for ball.

    Under the constant-thrust drive the acceleration rises over each burn, which the record keeps
    as two half-burn segments whose second is at least the burn's initial acceleration, so a
    step's segments times its mass ratio cover the acceleration at the step's final mass.
    """
    order = np.inf if ep.view.thrust_limit == "box" else 2
    ratios = np.ones(len(ep.records))
    if ep.drive == "thrust":
        if ep.view.mass_kg is None:
            raise ValueError("the constant-thrust drive needs the initial mass")
        burned = np.cumsum([0.0, *(rec.actuation.propellant_kg for rec in ep.records)])
        mass = ep.view.mass_kg - burned
        ratios = mass[:-1] / mass[1:]
    return max(
        (
            ratio * float(np.linalg.norm(accel, ord=order))
            for ratio, rec in zip(ratios, ep.records, strict=True)
            for _, accel in rec.actuation.segments
        ),
        default=0.0,
    )


def _ratio(dv: float, bound: float) -> float:
    """Delta-v over a bound, inf for a zero bound."""
    return dv / bound if bound > 0.0 else math.inf


def optimality_gap(ep: Episode, bounds: FuelBounds | None) -> dict[str, float | bool | str]:
    """E3: delta-v in the actuator's norm over the lower bound, and over the upper.

    The lower bound is certified for a run on the bound's rung whose delivered acceleration,
    read from the record and never from the agent, stayed within `accel_limit` (1e-12 relative).
    The ratio is then at least 1 by theorem, and a successful run below it means a simulator or
    metric bug and is flagged. A run that exceeded the limit is held to `impulsive_lower`
    instead, when given. A lower bound not certified for the run gives the model ratio, which
    may fall below 1 and is never flagged.

    Parameters
    ----------
    ep : Episode
        The recorded run.
    bounds : FuelBounds or None
        Bounds for the run's start state, in the norm of its thruster layout.

    Returns
    -------
    dict
        "optimality_gap" (E3), "optimality_gap_model" (E3m) and "optimality_gap_upper" (E3u),
        each NaN for a failed run, without bounds or where it does not apply, E3u also for an
        infinite upper bound; "below_lower_bound" (E3f), "bound_certified" (E3c) and
        "bound_rung" (E3r).

    Raises
    ------
    ValueError
        When the bounds' norm does not match the thruster layout.
    """
    gap: dict[str, float | bool | str] = {
        "optimality_gap": math.nan,
        "optimality_gap_model": math.nan,
        "optimality_gap_upper": math.nan,
        "below_lower_bound": False,
        "bound_certified": False,
        "bound_rung": "",
    }
    if bounds is None:
        return gap
    expected = "l1" if ep.view.thrust_limit == "box" else "l2"
    if bounds.norm != expected:
        raise ValueError(f"{ep.view.thrust_limit} thrusters need {expected} bounds")
    lower_certified = False
    certified_lower = None
    if bounds.rung == ep.dynamics:
        limit = bounds.accel_limit
        lower_certified = limit == math.inf or _peak_accel(ep) <= limit * (1.0 + _ACCEL_TOL)
        certified_lower = bounds.lower if lower_certified else bounds.impulsive_lower
    gap.update(bound_certified=certified_lower is not None, bound_rung=bounds.rung)
    if not success(ep):
        return gap
    dv = layout_delta_v(ep)
    if certified_lower is not None:
        gap["optimality_gap"] = _ratio(dv, certified_lower)
        gap["below_lower_bound"] = bool(dv < certified_lower * (1.0 - _BOUND_TOL))
    if not lower_certified:
        gap["optimality_gap_model"] = _ratio(dv, bounds.lower)
    if bounds.upper < math.inf:
        gap["optimality_gap_upper"] = _ratio(dv, bounds.upper)
    return gap


def efficiency_metrics(ep: Episode) -> dict[str, float | int]:
    """E1, E1b, E2, E4, E5 and E6 for one episode."""
    return {
        "delta_v": delta_v(ep),
        "delta_v_axes": delta_v_axes(ep),
        "propellant_mass": propellant_mass(ep),
        "burn_count": burn_count(ep),
        "duty_cycle": duty_cycle(ep),
        "chatter": chatter(ep),
    }
