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
more than the constraint's `resolution` below the best value found is refined with a bounded
Brent search, which finds the true minimum when the refined interval holds a single dip.
Refinement does not tighten the bound, which can stay up to L h / 2 below the reported minimum.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from scipy.optimize import minimize_scalar

# Refinement stops when the bracket on the time of the minimum is this narrow (s); a margin
# that falls at 10 m/s is then off by at most 0.1 mm.
_BRACKET_TOL = 1e-5
# Safety factor on the CW gravity-gradient and Coriolis terms in the acceleration bound; it
# covers the two-body nonlinearity and J2 at separations below 1% of the orbit radius.
_NATURAL_MARGIN = 2.0


class Constraint(Protocol):
    @property
    def name(self) -> str: ...

    resolution: ClassVar[float]  # margin units; refinement threshold below the best value

    def margin(self, state: np.ndarray, t: float | np.ndarray) -> np.ndarray: ...

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        """Bound on |d margin / dt| given |v| <= speed (m/s), |dv/dt| <= accel (m/s^2).

        `rho_min` (m) is a lower bound on the range over the same interval.
        """
        ...


@dataclass(frozen=True)
class KeepOut:
    """Stay outside a sphere of `radius` (m) around the target; margin rho - radius (m)."""

    radius: float
    name: str = "keep_out"
    resolution: ClassVar[float] = 1e-3

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        return np.linalg.norm(np.asarray(state)[..., :3], axis=-1) - self.radius

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        # |d rho / dt| = |r . v| / rho <= |v|.
        return speed


def closing_speed(state: np.ndarray) -> np.ndarray:
    """max(0, -rho_dot) (m/s), the speed at which the range falls; |v| at the origin."""
    state = np.asarray(state)
    r, v = state[..., :3], state[..., 3:6]
    rho = np.linalg.norm(r, axis=-1)
    speed = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_dot = np.where(rho > 0.0, np.sum(r * v, axis=-1) / rho, -speed)
    return np.maximum(-rho_dot, 0.0)


def _closing_rate_bound(speed: float, accel: float, rho_min: float) -> float:
    # rho_ddot = (|v|^2 - rho_dot^2) / rho + r_hat . dv/dt, so |rho_ddot| <= |v|^2 / rho + |dv/dt|.
    if rho_min <= 0.0:
        return np.inf
    return speed * speed / rho_min + accel


@dataclass(frozen=True)
class SpeedLimit:
    """Speed no higher than v0 + v1 rho; margin v0 + v1 rho - |v| (m/s).

    The distance-dependent form of the run time assurance docking benchmark (Dunlap et al.,
    "Comparing Run Time Assurance Approaches for Safe Spacecraft Docking", 2021, eq. 14,
    v0 = 0.2 m/s, v1 = 4 n).
    """

    v0: float
    v1: float
    name: str = "speed_limit"
    resolution: ClassVar[float] = 1e-5

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        state = np.asarray(state)
        rho = np.linalg.norm(state[..., :3], axis=-1)
        return self.v0 + self.v1 * rho - np.linalg.norm(state[..., 3:6], axis=-1)

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        return abs(self.v1) * speed + accel


@dataclass(frozen=True)
class ClosingSpeedLimit:
    """Closing speed no higher than v0 + v1 rho; margin v0 + v1 rho - max(0, -rho_dot) (m/s).

    Only the speed toward the target counts, so moving across the line of sight or away is free.
    """

    v0: float
    v1: float
    name: str = "closing_speed_limit"
    resolution: ClassVar[float] = 1e-5

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        state = np.asarray(state)
        rho = np.linalg.norm(state[..., :3], axis=-1)
        return self.v0 + self.v1 * rho - closing_speed(state)

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        return abs(self.v1) * speed + _closing_rate_bound(speed, accel, rho_min)


@dataclass(frozen=True)
class ApproachCone:
    """Stay inside a cone with its apex at `apex` (m, Hill) while within `active_range` of it.

    `axis` points from the apex toward the approach side and `half_angle` (rad, below pi/2)
    is the cone's half-angle; the corridor starts at the mating point (Fehse, Automated
    Rendezvous and Docking of Spacecraft, 2003, sections 2.4.2 and 5.6). With d = r - apex,
    rho = |d| and phi the angle between d and the axis, the signed distance to the cone is
    rho sin(half_angle - phi) while phi < half_angle + pi/2 and -rho beyond, where the apex is
    the nearest point of the cone (Boyd and Vandenberghe, Convex Optimization, exercise 8.3).
    The margin (m) is the robustness of "rho <= active_range implies inside the cone",
    max(rho - active_range, signed distance), so it does not jump where the cone turns on.
    """

    half_angle: float
    axis: tuple[float, float, float] = (0.0, -1.0, 0.0)
    apex: tuple[float, float, float] = (0.0, 0.0, 0.0)
    active_range: float = np.inf
    name: str = "approach_cone"
    resolution: ClassVar[float] = 1e-3

    def __post_init__(self):
        if not 0.0 < self.half_angle < 0.5 * np.pi:
            raise ValueError("half_angle must be between 0 and pi/2")
        if not np.linalg.norm(self.axis) > 0.0:
            raise ValueError("axis must be nonzero")

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        d = np.asarray(state)[..., :3] - np.asarray(self.apex, dtype=float)
        axis = np.asarray(self.axis, dtype=float)
        rho = np.linalg.norm(d, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_phi = np.where(rho > 0.0, d @ axis / (rho * np.linalg.norm(axis)), 1.0)
        phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))
        inside = rho * np.sin(self.half_angle - phi)
        distance = np.where(phi < self.half_angle + 0.5 * np.pi, inside, -rho)
        return np.maximum(rho - self.active_range, distance)

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        # A signed distance to a convex set changes no faster than the point moves, and so
        # does the maximum of two such functions.
        return speed


@dataclass(frozen=True)
class SunExclusion:
    """Keep the Sun out of a sensor looking from the chaser at the target; margin in rad.

    The margin is angle(-r, s) - half_angle, with s the unit Sun direction in Hill axes. The Sun
    is fixed in inertial space, so in the Hill frame, which turns at `rate` (rad/s, the mean
    motion n of a circular target orbit) about z, s(t) = Rz(-rate t) `sun_direction`.
    """

    half_angle: float
    sun_direction: tuple[float, float, float]
    rate: float
    name: str = "sun_exclusion"
    resolution: ClassVar[float] = 1e-5

    def sun_at(self, t: float | np.ndarray) -> np.ndarray:
        """Unit Sun direction in Hill axes at `t` s, shape (..., 3)."""
        s0 = np.asarray(self.sun_direction, dtype=float)
        s0 = s0 / np.linalg.norm(s0)
        angle = -self.rate * np.asarray(t, dtype=float)
        c, s = np.cos(angle), np.sin(angle)
        return np.stack([c * s0[0] - s * s0[1], s * s0[0] + c * s0[1], np.full_like(c, s0[2])], -1)

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        look = -np.asarray(state)[..., :3]
        sun = self.sun_at(t)
        angle = np.arctan2(np.linalg.norm(np.cross(look, sun), axis=-1), np.sum(look * sun, -1))
        return angle - self.half_angle

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        # The line of sight turns at most |v| / rho and the Sun direction at `rate`.
        if rho_min <= 0.0:
            return np.inf
        return speed / rho_min + abs(self.rate)


# Forward coast after thrust stops on a V-bar approach at speed v is kappa v / n, at
# n t = arccos(3/4) (Fehse, eq. 3.28 and fig. 3.13).
_VBAR_COAST = np.sqrt(7.0) - 3.0 * np.arccos(0.75)


@dataclass(frozen=True)
class PassiveSpeed:
    """Closing speed below which a thrust failure still misses structure `extent` m out; m/s.

    Envelopes derived from Fehse (Automated Rendezvous and Docking of Spacecraft, 2003) for
    straight-line approaches at mean motion `n` (rad/s). On the V-bar ("vbar") a chaser whose
    thrust stops at closing speed v coasts forward kappa v / n with
    kappa = sqrt(7) - 3 arccos(3/4) = 0.47755 (eq. 3.28, fig. 3.13), so it stays clear while
    v <= n (rho - extent) / kappa. On the R-bar ("rbar") its closest radial approach from range
    h is 4 h - sqrt(9 h^2 + v^2 / n^2) (eq. 3.22), clear while v <= n sqrt((4h - extent)^2 - 9h^2)
    for h > extent (section 5.7.2 flies inside this envelope). The margin is that speed minus the
    closing speed max(0, -rho_dot); the range rho stands for the distance along the approach line.
    """

    axis: str
    extent: float
    n: float
    name: str = "passive_speed"
    resolution: ClassVar[float] = 1e-5

    def __post_init__(self):
        if self.axis not in ("vbar", "rbar"):
            raise ValueError(f"unknown approach axis {self.axis!r}, expected vbar or rbar")

    def speed_envelope(self, rho: np.ndarray) -> np.ndarray:
        """Highest passively safe closing speed (m/s) at range `rho` (m)."""
        rho = np.asarray(rho, dtype=float)
        if self.axis == "vbar":
            return self.n * (rho - self.extent) / _VBAR_COAST
        reach = (4.0 * rho - self.extent) ** 2 - 9.0 * rho**2
        return np.where(rho > self.extent, self.n * np.sqrt(np.maximum(reach, 0.0)), 0.0)

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        state = np.asarray(state)
        rho = np.linalg.norm(state[..., :3], axis=-1)
        return self.speed_envelope(rho) - closing_speed(state)

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        closing = _closing_rate_bound(speed, accel, rho_min)
        if self.axis == "vbar":
            return self.n / _VBAR_COAST * speed + closing
        if rho_min <= self.extent:
            return np.inf
        # d envelope / d rho = n f' / (2 sqrt f), f = 7 h^2 - 8 h d + d^2, falls with h above d.
        h, d = rho_min, self.extent
        slope = self.n * (14.0 * h - 8.0 * d) / (2.0 * np.sqrt(7.0 * h * h - 8.0 * h * d + d * d))
        return slope * speed + closing


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


def range_lower_bounds(states: np.ndarray, h: float, speed: np.ndarray) -> np.ndarray:
    """Lower bound on the range (m) over each interval between consecutive samples.

    With |d rho / dt| <= `speed` on an interval of length `h`, rho stays above both
    rho_i - speed (t - t_i) and rho_{i+1} - speed (t_{i+1} - t), so above where they meet.
    """
    rho = np.linalg.norm(states[:, :3], axis=1)
    with np.errstate(invalid="ignore"):
        lower = 0.5 * (rho[:-1] + rho[1:] - speed * h)
    return np.maximum(np.nan_to_num(lower, nan=0.0, neginf=0.0), 0.0)


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
    rho_min = range_lower_bounds(states, h, speed)
    rates = np.array(
        [
            constraint.rate_bound(v, a, rho_min=r)
            for v, a, r in zip(speed, accel, rho_min, strict=True)
        ]
    )
    bounds = 0.5 * (margins[:-1] + margins[1:] - rates * h)
    for i in np.argsort(bounds):
        if bounds[i] >= best - constraint.resolution:
            break
        found = minimize_scalar(
            lambda tau: float(constraint.margin(state_at(tau), t0 + tau)),
            bounds=(times[i], times[i + 1]),
            method="bounded",
            options={"xatol": _BRACKET_TOL},
        )
        if found.fun < best:
            best_t, best = float(found.x), float(found.fun)
    return SegmentMinimum(t0 + best_t, best, min(float(bounds.min()), best))
