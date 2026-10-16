"""Constraints on the relative state and their minimum margin over continuous time.

A constraint's margin is >= 0 where it holds and < 0 where it is violated, in the units of the
constraint, as in signal temporal logic. Within a segment of constant thrust
the state is a smooth function of time that the dynamics model can evaluate anywhere, so the
minimum margin over the segment is found from evenly spaced samples and refined by a bounded
scalar minimization.

Local refinement finds the minimum but cannot prove there is no lower one between samples, so
each segment also gets a certified lower bound from a Lipschitz argument: on a sample interval
of length h whose margin changes at most L per second, the margin never falls below
(m_i + m_{i+1} - L h) / 2, which is at least min_i m_i - L h / 2. Every interval whose bound is
more than the constraint's `resolution` below the best value found is refined, so the reported
minimum is within `resolution` of the true one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from scipy.optimize import minimize_scalar

# Refinement stops when the bracket on the time of the minimum is this narrow (s); a margin
# that falls at 10 m/s is then off by at most 0.1 mm.
_TIME_TOL = 1e-5
# Safety factor on the CW gravity-gradient and Coriolis terms in the acceleration bound; it
# covers the two-body nonlinearity and J2 at separations below 1% of the orbit radius.
_NATURAL_MARGIN = 2.0


class Constraint(Protocol):
    @property
    def name(self) -> str: ...

    resolution: ClassVar[float]  # margin units; how closely the minimum is resolved

    def margin(self, state: np.ndarray, t: float | np.ndarray) -> np.ndarray: ...

    def rate_bound(self, speed: float, accel: float) -> float:
        """Bound on |d margin / dt| given bounds on |v| (m/s) and |dv/dt| (m/s^2)."""
        ...


@dataclass(frozen=True)
class KeepOut:
    """Stay outside a sphere of `radius` (m) around the target; margin rho - radius (m)."""

    radius: float
    name: str = "keep_out"
    resolution: ClassVar[float] = 1e-3

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        return np.linalg.norm(np.asarray(state)[..., :3], axis=-1) - self.radius

    def rate_bound(self, speed: float, accel: float) -> float:
        # |d rho / dt| = |r . v| / rho <= |v|.
        return speed


@dataclass(frozen=True)
class SegmentMinimum:
    """Lowest margin of one constraint over a segment and a certified lower bound on it.

    `t` is the time of the lowest margin found, in s from the episode start.
    """

    t: float
    margin: float
    lower_bound: float


def interval_bounds(
    states: np.ndarray, h: float, thrust: float, n: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds on |v| (m/s) and |dv/dt| (m/s^2) over each interval between consecutive samples.

    `states` are samples `h` s apart, `thrust` (m/s^2) the thrust acceleration's magnitude and
    `n` (rad/s) the mean motion. Between samples |v| <= mean of the end speeds + A h / 2 and
    rho <= larger end range + v h / 2, with A = thrust + c (3 n^2 rho + 2 n v); solved for v.
    The solution needs g = c h (1.5 n^2 h + 2 n) / 2 < 1; for longer intervals both bounds are
    infinite.
    """
    rho = np.linalg.norm(states[:, :3], axis=1)
    speed = np.linalg.norm(states[:, 3:], axis=1)
    rho_end = np.maximum(rho[:-1], rho[1:])
    speed_mid = 0.5 * (speed[:-1] + speed[1:])
    c = _NATURAL_MARGIN
    gain = 0.5 * h * c * (1.5 * n * n * h + 2.0 * n)
    if gain >= 1.0:
        unbounded = np.full(len(rho_end), np.inf)
        return unbounded, unbounded
    v = (speed_mid + 0.5 * h * (thrust + 3.0 * c * n * n * rho_end)) / (1.0 - gain)
    accel = thrust + c * (3.0 * n * n * (rho_end + 0.5 * h * v) + 2.0 * n * v)
    return v, accel


def segment_minimum(
    constraint: Constraint,
    state_at: Callable[[float], np.ndarray],
    times: np.ndarray,
    states: np.ndarray,
    thrust: float,
    n: float,
    t0: float = 0.0,
) -> SegmentMinimum:
    """Minimum margin of `constraint` over a segment, refined and certified.

    `times` (s from the segment start, evenly spaced) and `states` are samples of the segment,
    `state_at(tau)` gives the state `tau` s into it, `thrust` (m/s^2) is the magnitude of the
    segment's thrust acceleration, `n` the mean motion and `t0` the segment's start time.
    """
    margins = constraint.margin(states, t0 + times)
    k = int(np.argmin(margins))
    best_t, best = float(times[k]), float(margins[k])
    if len(times) < 2 or times[-1] <= times[0]:
        return SegmentMinimum(t0 + best_t, best, best)
    h = float(times[1] - times[0])
    speed, accel = interval_bounds(states, h, thrust, n)
    rates = np.array([constraint.rate_bound(v, a) for v, a in zip(speed, accel, strict=True)])
    bounds = 0.5 * (margins[:-1] + margins[1:] - rates * h)
    for i in np.argsort(bounds):
        if bounds[i] >= best - constraint.resolution:
            break
        found = minimize_scalar(
            lambda tau: float(constraint.margin(state_at(tau), t0 + tau)),
            bounds=(times[i], times[i + 1]),
            method="bounded",
            options={"xatol": _TIME_TOL},
        )
        if found.fun < best:
            best_t, best = float(found.x), float(found.fun)
    return SegmentMinimum(t0 + best_t, best, min(float(bounds.min()), best))
