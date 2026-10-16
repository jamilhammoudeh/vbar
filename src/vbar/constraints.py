"""Constraints on the relative state and their minimum margin over continuous time.

A constraint's margin is >= 0 where it holds and < 0 where it is violated, in the units of the
constraint; it is the robustness of "always margin >= 0". Within a segment of constant thrust
the state is a smooth function of time that the dynamics model can evaluate anywhere, so the
minimum margin over the segment is found from evenly spaced samples and then refined by a
bounded scalar minimization next to the lowest sample.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import minimize_scalar

# Refinement stops when the bracket on the time of the minimum is this narrow (s); a margin
# that falls at 10 m/s is then off by at most 0.1 mm.
_TIME_TOL = 1e-5


class Constraint(Protocol):
    name: str

    def margin(self, state: np.ndarray, t: float | np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class KeepOut:
    """Stay outside a sphere of `radius` (m) around the target; margin rho - radius (m)."""

    radius: float
    name: str = "keep_out"

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        return np.linalg.norm(np.asarray(state)[..., :3], axis=-1) - self.radius


@dataclass(frozen=True)
class SegmentMinimum:
    """Lowest margin of one constraint over a segment and its time (s from episode start)."""

    t: float
    margin: float


def segment_minimum(
    constraint: Constraint,
    state_at: Callable[[float], np.ndarray],
    times: np.ndarray,
    states: np.ndarray,
    t0: float = 0.0,
) -> SegmentMinimum:
    """Minimum margin of `constraint` over a segment.

    `times` (s from the segment start, increasing) and `states` are samples of the segment, and
    `state_at(tau)` gives the state `tau` s into it; `t0` is the segment's start time.
    """
    margins = constraint.margin(states, t0 + times)
    k = int(np.argmin(margins))
    best = SegmentMinimum(t0 + float(times[k]), float(margins[k]))
    lo, hi = times[max(k - 1, 0)], times[min(k + 1, len(times) - 1)]
    if hi > lo:
        found = minimize_scalar(
            lambda tau: float(constraint.margin(state_at(tau), t0 + tau)),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": _TIME_TOL},
        )
        if found.fun < best.margin:
            best = SegmentMinimum(t0 + float(found.x), float(found.fun))
    return best
