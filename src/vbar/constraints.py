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
from typing import Any, ClassVar, Protocol, cast

import numpy as np
from scipy.optimize import brentq, minimize_scalar

# Refinement stops when the bracket on the time of the minimum is this narrow (s); a margin
# that falls at 10 m/s is then off by at most 0.1 mm.
_BRACKET_TOL = 1e-5
# Allowance past a GO time, relative above 1 s; summed step times drift from it by a few ulp.
_GO_TOL = 1e-9
# Violation boundaries are located to this many seconds.
_ROOT_TOL = 1e-9
# Batched refinement narrows its bracket to this many seconds; near a dip the margin's error is
# second order in it, well below the 1e-5 s Brent search's.
_GOLDEN_TOL = 1e-6
_INVPHI = 0.5 * (np.sqrt(5.0) - 1.0)
# Safety factor on the CW gravity-gradient and Coriolis terms in the acceleration bound; it
# covers the two-body nonlinearity and J2 at separations below 1% of the orbit radius.
_NATURAL_MARGIN = 2.0


class Constraint(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def resolution(self) -> float:
        """Margin units; intervals whose bound is this far below the best value are refined."""
        ...

    def margin(self, state: np.ndarray, t: float | np.ndarray) -> np.ndarray: ...

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        """Bound on |d margin / dt| given |v| <= speed (m/s), |dv/dt| <= accel (m/s^2).

        `rho_min` (m) is a lower bound on the range over the same interval.
        """
        ...


def _vector3(value, name: str) -> tuple[float, float, float]:
    """`value` as three floats, so a constraint given a list or array shares nothing mutable."""
    vector = [float(c) for c in value]
    if len(vector) != 3:
        raise ValueError(f"{name} must have three components")
    return vector[0], vector[1], vector[2]


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
    # Written elementwise so the batched minimum can pass arrays.
    rho = np.asarray(rho_min, dtype=float)
    positive = rho > 0.0
    bound = np.where(positive, speed * speed / np.where(positive, rho, 1.0) + accel, np.inf)
    return cast(float, bound if bound.ndim else float(bound))


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

    def allowed_speed(self, rho: np.ndarray) -> np.ndarray:
        """Highest allowed closing speed (m/s) at range `rho` (m)."""
        return self.v0 + self.v1 * np.asarray(rho, dtype=float)

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        state = np.asarray(state)
        rho = np.linalg.norm(state[..., :3], axis=-1)
        return self.allowed_speed(rho) - closing_speed(state)

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        return abs(self.v1) * speed + _closing_rate_bound(speed, accel, rho_min)


@dataclass(frozen=True)
class GateTable:
    """Closing speed below a table of range-rate gates; margin allowed - max(0, -rho_dot) (m/s).

    `gates` are (range m, closing speed m/s) pairs, the form the Shuttle flew (Goodman, History of
    Space Shuttle Rendezvous, 2011, table 24.3 and figs 14.12, 25.17, 25.22). The allowed speed is
    linearly interpolated between gates and held at the nearest gate beyond the table; a speed of
    0 is a hold.
    """

    gates: tuple[tuple[float, float], ...]
    name: str = "gate_table"
    resolution: ClassVar[float] = 1e-5

    def __post_init__(self):
        # Stored as float pairs so a table given as lists is still hashable.
        gates = tuple((float(r), float(v)) for r, v in self.gates)
        object.__setattr__(self, "gates", gates)
        ranges = sorted(r for r, _ in gates)
        if len(ranges) < 2 or any(b <= a for a, b in zip(ranges, ranges[1:], strict=False)):
            raise ValueError("gates need at least two distinct ranges")
        if not all(0.0 <= r < np.inf and 0.0 <= v < np.inf for r, v in gates):
            raise ValueError("gate ranges and speeds must be finite and not negative")

    def _table(self) -> tuple[np.ndarray, np.ndarray]:
        table = np.array(sorted(self.gates), dtype=float)
        return table[:, 0], table[:, 1]

    def allowed_speed(self, rho: np.ndarray) -> np.ndarray:
        """Highest allowed closing speed (m/s) at range `rho` (m)."""
        ranges, speeds = self._table()
        return np.interp(np.asarray(rho, dtype=float), ranges, speeds)

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        state = np.asarray(state)
        rho = np.linalg.norm(state[..., :3], axis=-1)
        return self.allowed_speed(rho) - closing_speed(state)

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        ranges, speeds = self._table()
        slope = float(np.max(np.abs(np.diff(speeds) / np.diff(ranges))))
        return slope * speed + _closing_rate_bound(speed, accel, rho_min)


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
        object.__setattr__(self, "axis", _vector3(self.axis, "axis"))
        object.__setattr__(self, "apex", _vector3(self.apex, "apex"))
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

    def __post_init__(self):
        object.__setattr__(self, "sun_direction", _vector3(self.sun_direction, "sun_direction"))

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
class ZoneEntry:
    """Stay out of an ellipsoid up to `go_time` (s); margin a_min (q - 1) in m.

    q = |D^-1 (r - center)| with D the diagonal of `semi_axes` (m, Hill axes), and a_min the
    smallest semi-axis. Since |q(x) - q(y)| <= |x - y| / a_min, the margin is 1-Lipschitz in
    position and below the distance to the ellipsoid outside it, exactly that distance for a
    sphere. The ISS approach ellipsoid is 2000 m along-track and 1000 m otherwise (Fehse 5.6).
    After `go_time` the margin is infinite; the GO event must fall on a step boundary, since a
    jump inside a segment is not covered by the rate bound.
    """

    semi_axes: tuple[float, float, float]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    go_time: float = np.inf
    name: str = "zone_entry"
    resolution: ClassVar[float] = 1e-3

    def __post_init__(self):
        object.__setattr__(self, "semi_axes", _vector3(self.semi_axes, "semi_axes"))
        object.__setattr__(self, "center", _vector3(self.center, "center"))
        if not min(self.semi_axes) > 0.0:
            raise ValueError("semi_axes must be positive")

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        offset = np.asarray(state)[..., :3] - np.asarray(self.center, dtype=float)
        q = np.linalg.norm(offset / np.asarray(self.semi_axes, dtype=float), axis=-1)
        # Finite up to and including GO, so the step ending at GO still sees the zone; the
        # environment's time is a sum of steps, so a GO on a boundary may be passed by rounding.
        until = self.go_time + _GO_TOL * max(1.0, abs(self.go_time))
        return np.where(np.asarray(t) <= until, min(self.semi_axes) * (q - 1.0), np.inf)

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
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
    states: np.ndarray, h: float, thrust: float | np.ndarray, n: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds on |v| (m/s) and |dv/dt| (m/s^2) over each interval between consecutive samples.

    `states` are samples `h` s apart, `thrust` (m/s^2) the thrust acceleration's magnitude and
    `n` (rad/s) the mean motion. Between samples |v| <= mean of the end speeds + A h / 2 and
    rho <= larger end range + v h / 2, with A = thrust + c (3 n^2 rho + 2 n v); solved for v.
    The solution needs g = c h (1.5 n^2 h + 2 n) / 2 < 1; for longer intervals both bounds are
    infinite. Stacked trajectories (..., samples, 6) take a `thrust` that broadcasts against
    (..., samples - 1).
    """
    rho = np.linalg.norm(states[..., :3], axis=-1)
    speed = np.linalg.norm(states[..., 3:], axis=-1)
    rho_end = np.maximum(rho[..., :-1], rho[..., 1:])
    speed_mid = 0.5 * (speed[..., :-1] + speed[..., 1:])
    c = _NATURAL_MARGIN
    gain = 0.5 * h * c * (1.5 * n * n * h + 2.0 * n)
    if gain >= 1.0:
        unbounded = np.full(rho_end.shape, np.inf)
        return unbounded, unbounded
    v = (speed_mid + 0.5 * h * (thrust + 3.0 * c * n * n * rho_end)) / (1.0 - gain)
    accel = thrust + c * (3.0 * n * n * (rho_end + 0.5 * h * v) + 2.0 * n * v)
    return v, accel


def range_lower_bounds(states: np.ndarray, h: float, speed: np.ndarray) -> np.ndarray:
    """Lower bound on the range (m) over each interval between consecutive samples.

    With |d rho / dt| <= `speed` on an interval of length `h`, rho stays above both
    rho_i - speed (t - t_i) and rho_{i+1} - speed (t_{i+1} - t), so above where they meet.
    """
    rho = np.linalg.norm(states[..., :3], axis=-1)
    with np.errstate(invalid="ignore"):
        lower = 0.5 * (rho[..., :-1] + rho[..., 1:] - speed * h)
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


def segment_minima(
    constraint: Constraint,
    state_at: Callable[[np.ndarray, np.ndarray], np.ndarray],
    times: np.ndarray,
    states: np.ndarray,
    thrust: np.ndarray,
    n: float,
    t0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`segment_minimum` for a batch of segments sampled at the same `times`.

    `states` is (N, m, 6), `thrust` and `t0` are (N,), and `state_at(rows, tau)` returns the
    states `tau` s into the segments `rows`, both (A,), as (A, 6). The constraint's `margin` and
    `rate_bound` must accept arrays. Intervals are refined by the same rule, in order of their
    bound until one is within `resolution` of the best value. Every interval that rule could
    refine is searched at once, with a golden-section search to 1e-6 s instead of Brent's
    method, and the rule is then replayed on the results, so each minimum comes from the
    intervals `segment_minimum` would refine and agrees with it to the accuracy of the two
    searches. Returns the times (s from the episode start), margins and lower bounds, each (N,).
    """
    rows = np.arange(len(states))
    margins = constraint.margin(states, t0[:, None] + times)
    k = np.argmin(margins, axis=1)
    best_t, best = times[k].astype(float), margins[rows, k].astype(float)
    h = float(times[1] - times[0])
    speed, accel = interval_bounds(states, h, thrust[:, None], n)
    rho_min = range_lower_bounds(states, h, speed)
    # The protocol is written for floats; batched use needs a vectorized rate bound.
    rate_bound = cast(Any, constraint.rate_bound)
    rates = np.broadcast_to(
        np.asarray(rate_bound(speed, accel, rho_min=rho_min), dtype=np.float64), speed.shape
    )
    bounds = 0.5 * (margins[:, :-1] + margins[:, 1:] - rates * h)
    # The best value only falls during refinement, so these are all the intervals it can visit.
    owner, column = np.nonzero(bounds < (best - constraint.resolution)[:, None])
    if owner.size:
        found_t = np.zeros(bounds.shape)
        found = np.full(bounds.shape, np.inf)

        def margin_at(tau: np.ndarray) -> np.ndarray:
            return constraint.margin(state_at(owner, tau), t0[owner] + tau)

        found_t[owner, column], found[owner, column] = _golden(
            margin_at, times[column], times[column + 1]
        )
        active = np.ones(len(states), dtype=bool)
        for visit in np.argsort(bounds, axis=1).T:
            active &= bounds[rows, visit] < best - constraint.resolution
            if not active.any():
                break
            better = active & (found[rows, visit] < best)
            best = np.where(better, found[rows, visit], best)
            best_t = np.where(better, found_t[rows, visit], best_t)
    return t0 + best_t, best, np.minimum(bounds.min(axis=1), best)


def _golden(
    f: Callable[[np.ndarray], np.ndarray], a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum of `f` on each [a, b] by golden-section search, all rows at once."""
    c, d = b - _INVPHI * (b - a), a + _INVPHI * (b - a)
    fc, fd = f(c), f(d)
    best_x, best = np.where(fc < fd, c, d), np.minimum(fc, fd)
    width = float(np.max(b - a))
    for _ in range(max(0, int(np.ceil(np.log(_GOLDEN_TOL / width) / np.log(_INVPHI))))):
        left = fc < fd
        a, b = np.where(left, a, c), np.where(left, d, b)
        x = np.where(left, b - _INVPHI * (b - a), a + _INVPHI * (b - a))
        fx = f(x)
        c, fc, d, fd = (
            np.where(left, x, d),
            np.where(left, fx, fd),
            np.where(left, c, x),
            np.where(left, fc, fx),
        )
        better = fx < best
        best_x, best = np.where(better, x, best_x), np.where(better, fx, best)
    return best_x, best


def sampled_negative_intervals(
    margin_at: Callable[[float], float],
    times: np.ndarray,
    margins: np.ndarray,
    rates: np.ndarray,
) -> list[tuple[float, float]]:
    """Intervals (s) where a margin is negative, from evenly spaced samples and rate bounds.

    `times` (s) and `margins` are the samples, `margin_at(t)` the margin at any time between
    them, and `rates` bound |d margin / dt| on each sample interval. Sign changes are found by
    Brent's method to 1e-9 s; an interval whose ends share a sign is searched for a dip below zero,
    or a rise above it, only when its rate bound allows one, which assumes at most one dip per
    interval. The environment's violation intervals and the metrics' both come from here.
    """
    found: list[tuple[float, float]] = []
    h = float(times[1] - times[0])

    def root(a: float, b: float) -> float:
        return float(brentq(margin_at, a, b, xtol=_ROOT_TOL))

    def extremum(f: Callable[[float], float], a: float, b: float) -> tuple[float, float]:
        best = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": _ROOT_TOL})
        return float(best.x), float(best.fun)

    for i in range(len(times) - 1):
        t0, t1, a, b = float(times[i]), float(times[i + 1]), margins[i], margins[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            # An infinite margin (a constraint switched off) leaves at most an instant below zero.
            continue
        if a < 0.0 and b < 0.0:
            if 0.5 * (a + b + rates[i] * h) > 0.0:
                t_peak, minus_peak = extremum(lambda t: -margin_at(t), t0, t1)
                if minus_peak <= 0.0:
                    found += [(t0, root(t0, t_peak)), (root(t_peak, t1), t1)]
                    continue
            found.append((t0, t1))
        elif a < 0.0:
            found.append((t0, root(t0, t1)))
        elif b < 0.0:
            found.append((root(t0, t1), t1))
        elif 0.5 * (a + b - rates[i] * h) < 0.0:
            t_min, low = extremum(margin_at, t0, t1)
            if low < 0.0:
                found.append((root(t0, t_min), root(t_min, t1)))
    return found


def violation_intervals(
    constraint: Constraint,
    state_at: Callable[[float], np.ndarray],
    times: np.ndarray,
    states: np.ndarray,
    minimum: SegmentMinimum,
    t0: float = 0.0,
    thrust: float = 0.0,
    n: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """Intervals (s from the episode start) inside a segment where the margin is negative.

    Takes the samples `segment_minimum` was given and its result, plus the segment's thrust
    acceleration (m/s^2) and the mean motion (rad/s) for the rate bounds; a segment whose
    certified lower bound is not negative has none. The search is `sampled_negative_intervals`,
    and touching intervals are joined.
    """
    if not minimum.lower_bound < 0.0 or len(times) < 2:
        return ()
    h = float(times[1] - times[0])
    speed, accel = interval_bounds(states, h, thrust, n)
    rho_min = range_lower_bounds(states, h, speed)
    bounds = zip(speed, accel, rho_min, strict=True)
    rates = np.array([constraint.rate_bound(v, a, rho_min=r) for v, a, r in bounds])

    def margin_at(t: float) -> float:
        return float(constraint.margin(state_at(t - t0), t))

    margins = np.asarray(constraint.margin(states, t0 + times), dtype=float)
    joined: list[tuple[float, float]] = []
    for lo, hi in sampled_negative_intervals(margin_at, t0 + times, margins, rates):
        if joined and lo <= joined[-1][1] + _ROOT_TOL:
            joined[-1] = (joined[-1][0], max(joined[-1][1], hi))
        else:
            joined.append((lo, hi))
    return tuple(joined)
