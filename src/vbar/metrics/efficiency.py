"""Efficiency metrics: delta-v, propellant, burns, duty cycle and chatter (METRICS E1 to E6).

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
    """Bounds on the least delta-v (m/s) from a start state to the scenario's success set.

    `lower` must be certified (a dual-feasible bound), `upper` the best known feasible cost, both
    for the same success set, time window and actuator norm: "l1" for box thrusters, "l2" for a
    ball (Boyd and Vandenberghe, Convex Optimization, sections 5.1 and 5.5).
    """

    lower: float
    upper: float
    norm: str

    def __post_init__(self):
        if self.norm not in ("l1", "l2"):
            raise ValueError(f"unknown norm {self.norm!r}, expected l1 or l2")
        if not 0.0 <= self.lower <= self.upper:
            raise ValueError("bounds must satisfy 0 <= lower <= upper")


class Oracle(Protocol):
    """Gives fuel bounds for a start state."""

    def bounds(self, view: AgentView, start: np.ndarray) -> FuelBounds: ...


def optimality_gap(ep: Episode, bounds: FuelBounds | None) -> dict[str, float | bool]:
    """E3: delta-v in the actuator's norm over the certified lower bound, and over the upper.

    The ratio to the lower bound is at least 1 by theorem when the bound matches the success set,
    time window and norm; a successful run below it means a simulator or metric bug and is
    flagged. Both ratios are NaN for failed runs and when no bounds are given.
    """
    gap: dict[str, float | bool] = {
        "optimality_gap": math.nan,
        "optimality_gap_upper": math.nan,
        "below_lower_bound": False,
    }
    if bounds is None:
        return gap
    expected = "l1" if ep.view.thrust_limit == "box" else "l2"
    if bounds.norm != expected:
        raise ValueError(f"{ep.view.thrust_limit} thrusters need {expected} bounds")
    if not success(ep):
        return gap
    dv = layout_delta_v(ep)
    gap["below_lower_bound"] = bool(dv < bounds.lower * (1.0 - _BOUND_TOL))
    gap["optimality_gap"] = dv / bounds.lower if bounds.lower > 0.0 else math.inf
    gap["optimality_gap_upper"] = dv / bounds.upper if bounds.upper > 0.0 else math.inf
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
