"""Control barrier functions for the keep-out sphere, speed limits and an approach cone.

Each barrier h is a function of the Hill-frame state x = (r, v) (m, m/s) that is nonnegative on
its safe set. A barrier with a `row` method gives the linear condition on the commanded
acceleration u (m/s^2) that keeps h from decreasing faster than its class-K function allows,
as a normal a and right-hand side b of a . u >= b, given the natural acceleration a_cw(x) of
the dynamics. Position barriers have relative degree 2 and use the exponential control barrier
function condition h'' + (p1 + p2) h' + p1 p2 h >= 0 with real poles p1, p2 > 0 (Ames et al.,
*Control Barrier Functions: Theory and Applications*, ECC 2019, sec. IV, Def. 7); speed barriers
have relative degree 1 and use h' >= -gamma h (sec. II, Def. 2).

`PassiveBarrier` and `BrakingBarrier` are evading-maneuver barriers (sec. III, eq. 12): h is the
margin that a fixed backup command, free drift or full outward braking, keeps from x, so that
command is always safe on the set h >= 0 (Theorem 4). They have no row and are enforced by a
line search toward the backup on the exact held segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from ..constraints import KeepOut, interval_bounds
from ..dynamics.cw import cw_discrete
from ..solvers.qp import ACTUATOR_LAYOUTS, ActuatorSet

# Below this speed (m/s) the direction of v is undefined and the speed row is dropped; the
# barrier value is then at least v0, far from its boundary.
_MIN_SPEED = 1e-9
# Below this secular drift rate (m/s) the relative orbit is treated as closed: the drift minimum
# repeats every orbit and has no unique minimiser.
_MIN_DRIFT = 1e-9
# The drift minimiser is located to this many seconds.
_TAU_TOL = 1e-9
# Phases of one relative orbit on which the terminal tube of the free drift is bounded.
_PSI_STEP = 2.0 * math.pi / 1024
_COS_PSI = np.cos(_PSI_STEP * np.arange(1024))
_SIN_PSI = np.sin(_PSI_STEP * np.arange(1024))


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _poles(poles) -> tuple[float, float]:
    """Two real positive poles; complex poles break the comparison step of the ECBF proof."""
    values = tuple(poles)
    if len(values) != 2:
        raise ValueError("poles must be two values")
    if any(isinstance(p, complex) or np.iscomplexobj(p) for p in values):
        raise ValueError("poles must be real and positive")
    p1, p2 = (float(p) for p in values)
    if not all(math.isfinite(p) and p > 0.0 for p in (p1, p2)):
        raise ValueError("poles must be real and positive")
    return p1, p2


def _vector3(value, name: str) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be three finite numbers")
    return float(vector[0]), float(vector[1]), float(vector[2])


def _state(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape != (6,) or not np.all(np.isfinite(x)):
        raise ValueError("x must be six finite numbers")
    return x


def _accel(a_cw) -> np.ndarray:
    return np.array(_vector3(a_cw, "a_cw"))


def _range_terms(x: np.ndarray) -> tuple[float, np.ndarray, float, float]:
    """rho, rhat, rho_dot and the centripetal term (|v|^2 - rho_dot^2) / rho."""
    r, v = x[:3], x[3:]
    rho = float(np.linalg.norm(r))
    if rho == 0.0:
        raise ValueError("the range is zero, its direction is undefined")
    rhat = r / rho
    rho_dot = float(rhat @ v)
    cent = max(0.0, float(v @ v) - rho_dot * rho_dot) / rho
    return rho, rhat, rho_dot, cent


@dataclass(frozen=True, kw_only=True)
class KeepOutEcbf:
    """Exponential control barrier function h = rho - R for the keep-out sphere.

    Parameters
    ----------
    radius : float
        The barrier radius R, m, positive; the keep-out radius plus any buffer.
    poles : tuple of float
        The closed-loop poles p1, p2 of h, 1/s, real and positive.

    Raises
    ------
    ValueError
        If `radius` is not positive or a pole is complex, not finite or not positive.

    Notes
    -----
    With h' = rho_dot and h'' = (|v|^2 - rho_dot^2) / rho + rhat . (a_cw + u), the condition
    h'' + (p1 + p2) h' + p1 p2 h >= 0 is the row
    rhat . u >= -(|v|^2 - rho_dot^2) / rho - rhat . a_cw - (p1 + p2) rho_dot - p1 p2 h
    (Ames et al., sec. IV, eq. 19 with K = (p1 p2, p1 + p2)). It keeps h >= 0 in continuous
    time from starts in `contains`. The poles must be real: the comparison step after eq. 18
    fails for complex poles.
    """

    radius: float
    poles: tuple[float, float]

    def __post_init__(self):
        object.__setattr__(self, "radius", _positive(self.radius, "radius"))
        object.__setattr__(self, "poles", _poles(self.poles))

    @property
    def gains(self) -> tuple[float, float]:
        """The gains (p1 p2, p1 + p2) on h and h', 1/s^2 and 1/s."""
        p1, p2 = self.poles
        return p1 * p2, p1 + p2

    def value(self, x) -> float:
        """Return h = rho - R, m, at the Hill-frame state `x` (m, m/s)."""
        return float(np.linalg.norm(_state(x)[:3])) - self.radius

    def contains(self, x) -> bool:
        """Return whether `x` lies in the set h >= 0 and rho_dot + max(p1, p2) h >= 0.

        Parameters
        ----------
        x : array_like, shape (6,)
            Hill-frame state, m and m/s.

        Returns
        -------
        bool
            True on the intersection of the barrier's zero superlevel set and the first
            intermediate set, from which the continuous-time condition keeps h >= 0. The pole
            order along the chain is free, so the faster pole sets the second condition
            (Ames et al., Theorem 8, with the sign of its eigenvalue condition reversed).

        Raises
        ------
        ValueError
            If `x` is not six finite numbers or its range is zero.
        """
        rho, _, rho_dot, _ = _range_terms(_state(x))
        h = rho - self.radius
        return h >= 0.0 and rho_dot + max(self.poles) * h >= 0.0

    def row(self, x, a_cw) -> tuple[np.ndarray, float]:
        """Return the barrier row a . u >= b at the state `x`.

        Parameters
        ----------
        x : array_like, shape (6,)
            Hill-frame state, m and m/s.
        a_cw : array_like, shape (3,)
            Natural acceleration of the dynamics at `x`, m/s^2, Hill frame.

        Returns
        -------
        a : ndarray, shape (3,)
            The unit range direction rhat.
        b : float
            The right-hand side, m/s^2.

        Raises
        ------
        ValueError
            If `x` or `a_cw` is malformed or the range is zero.
        """
        rho, rhat, rho_dot, cent = _range_terms(_state(x))
        k0, k1 = self.gains
        b = -cent - float(rhat @ _accel(a_cw)) - k1 * rho_dot - k0 * (rho - self.radius)
        return rhat, b


@dataclass(frozen=True, kw_only=True)
class SpeedCbf:
    """First-order barrier s = v0 + v1 rho - |v| for a range-dependent speed limit.

    Parameters
    ----------
    v0 : float
        The allowed speed at the target, m/s, positive.
    v1 : float
        The allowed speed's growth with range, 1/s, nonnegative.
    gamma : float
        The class-K gain of s' >= -gamma s, 1/s, positive.

    Raises
    ------
    ValueError
        If a parameter is out of range or not finite.

    Notes
    -----
    s' = v1 rho_dot - vhat . (a_cw + u), so s' >= -gamma s is the row
    -vhat . u >= vhat . a_cw - v1 rho_dot - gamma s. The speed has relative degree 1 in u.
    """

    v0: float
    v1: float
    gamma: float

    def __post_init__(self):
        object.__setattr__(self, "v0", _positive(self.v0, "v0"))
        object.__setattr__(self, "v1", _nonnegative(self.v1, "v1"))
        object.__setattr__(self, "gamma", _positive(self.gamma, "gamma"))

    def value(self, x) -> float:
        """Return s = v0 + v1 rho - |v|, m/s, at the Hill-frame state `x` (m, m/s)."""
        x = _state(x)
        return self.v0 + self.v1 * float(np.linalg.norm(x[:3])) - float(np.linalg.norm(x[3:]))

    def contains(self, x) -> bool:
        """Return whether s >= 0 at the Hill-frame state `x` (m, m/s)."""
        return self.value(x) >= 0.0

    def row(self, x, a_cw) -> tuple[np.ndarray, float] | None:
        """Return the barrier row a . u >= b at the state `x`, or None below 1e-9 m/s.

        Parameters
        ----------
        x : array_like, shape (6,)
            Hill-frame state, m and m/s.
        a_cw : array_like, shape (3,)
            Natural acceleration of the dynamics at `x`, m/s^2, Hill frame.

        Returns
        -------
        tuple of (ndarray, float) or None
            The normal -vhat and the right-hand side, m/s^2. None when |v| < 1e-9 m/s, where
            vhat is undefined and s >= v0 > 0. With v1 > 0 the range must be nonzero.

        Raises
        ------
        ValueError
            If `x` or `a_cw` is malformed, or v1 > 0 and the range is zero.
        """
        x = _state(x)
        a_cw = _accel(a_cw)
        v = x[3:]
        speed = float(np.linalg.norm(v))
        if speed < _MIN_SPEED:
            return None
        vhat = v / speed
        rho_dot = _range_terms(x)[2] if self.v1 > 0.0 else 0.0
        s = self.v0 + self.v1 * float(np.linalg.norm(x[:3])) - speed
        return -vhat, float(vhat @ a_cw) - self.v1 * rho_dot - self.gamma * s


@dataclass(frozen=True, kw_only=True)
class ClosingSpeedCbf:
    """First-order barrier s = v0 + v1 rho - c on the closing speed c = max(0, -rho_dot).

    Parameters
    ----------
    v0 : float
        The allowed closing speed at the target, m/s, positive.
    v1 : float
        The allowed closing speed's growth with range, 1/s, nonnegative.
    gamma : float
        The class-K gain of s' >= -gamma s, 1/s, positive.

    Raises
    ------
    ValueError
        If a parameter is out of range or not finite.

    Notes
    -----
    While closing, c = -rho_dot and c' = -(|v|^2 - rho_dot^2) / rho - rhat . (a_cw + u), so
    s' >= -gamma s is the row rhat . u >= -(|v|^2 - rho_dot^2) / rho - rhat . a_cw - v1 rho_dot
    - gamma s. While opening, s >= v0 > 0 and the row is dropped.
    """

    v0: float
    v1: float
    gamma: float

    def __post_init__(self):
        object.__setattr__(self, "v0", _positive(self.v0, "v0"))
        object.__setattr__(self, "v1", _nonnegative(self.v1, "v1"))
        object.__setattr__(self, "gamma", _positive(self.gamma, "gamma"))

    def value(self, x) -> float:
        """Return s = v0 + v1 rho - max(0, -rho_dot), m/s, at the state `x` (m, m/s)."""
        rho, _, rho_dot, _ = _range_terms(_state(x))
        return self.v0 + self.v1 * rho - max(0.0, -rho_dot)

    def contains(self, x) -> bool:
        """Return whether s >= 0 at the Hill-frame state `x` (m, m/s)."""
        return self.value(x) >= 0.0

    def row(self, x, a_cw) -> tuple[np.ndarray, float] | None:
        """Return the barrier row a . u >= b while closing, or None when rho_dot >= 0.

        Parameters
        ----------
        x : array_like, shape (6,)
            Hill-frame state, m and m/s.
        a_cw : array_like, shape (3,)
            Natural acceleration of the dynamics at `x`, m/s^2, Hill frame.

        Returns
        -------
        tuple of (ndarray, float) or None
            The normal rhat and the right-hand side, m/s^2, or None while not closing.

        Raises
        ------
        ValueError
            If `x` or `a_cw` is malformed or the range is zero.
        """
        rho, rhat, rho_dot, cent = _range_terms(_state(x))
        a_cw = _accel(a_cw)
        if rho_dot >= 0.0:
            return None
        s = self.v0 + self.v1 * rho + rho_dot
        return rhat, -cent - float(rhat @ a_cw) - self.v1 * rho_dot - self.gamma * s


@dataclass(frozen=True, kw_only=True)
class ConeEcbf:
    """Exponential control barrier function h = a . (r - p) - cos(theta) |r - p| for a cone.

    Parameters
    ----------
    half_angle : float
        The cone half-angle theta, rad, in (0, pi/2).
    axis : array_like, shape (3,)
        The cone axis a, Hill frame, nonzero; stored as a unit vector.
    apex : array_like, shape (3,)
        The apex p, m, Hill frame.
    active_range : float
        The barrier is enforced within this distance of the apex, m, positive.
    poles : tuple of float
        The closed-loop poles p1, p2 of h, 1/s, real and positive.

    Raises
    ------
    ValueError
        If a parameter is out of range, not finite or `axis` is zero.

    Notes
    -----
    With d = r - p, h' = a . v - cos(theta) dhat . v and
    h'' = g . (a_cw + u) - cos(theta) |v_perp|^2 / |d|, where g = a - cos(theta) dhat is the
    gradient and v_perp the part of v normal to d. On the cone surface |g| = sin(theta), so h is
    the signed distance to the surface scaled by sin(theta) there. The gradient is undefined at
    the apex, and with a lateral speed v_lat the curvature term needs more than a ball limit
    a_max gives within `infeasible_radius` of it.
    """

    half_angle: float
    axis: tuple[float, float, float]
    apex: tuple[float, float, float]
    active_range: float
    poles: tuple[float, float]

    def __post_init__(self):
        theta = float(self.half_angle)
        if not (math.isfinite(theta) and 0.0 < theta < 0.5 * math.pi):
            raise ValueError("half_angle must be in (0, pi/2) rad")
        axis = np.array(_vector3(self.axis, "axis"))
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise ValueError("axis must be nonzero")
        object.__setattr__(self, "half_angle", theta)
        object.__setattr__(self, "axis", _vector3(axis / norm, "axis"))
        object.__setattr__(self, "apex", _vector3(self.apex, "apex"))
        object.__setattr__(self, "active_range", _positive(self.active_range, "active_range"))
        object.__setattr__(self, "poles", _poles(self.poles))

    @property
    def gains(self) -> tuple[float, float]:
        """The gains (p1 p2, p1 + p2) on h and h', 1/s^2 and 1/s."""
        p1, p2 = self.poles
        return p1 * p2, p1 + p2

    def _terms(self, x: np.ndarray) -> tuple[float, np.ndarray, float, float]:
        """h, the gradient g, h' and the curvature term cos(theta) |v_perp|^2 / |d|."""
        d = x[:3] - np.array(self.apex)
        dist = float(np.linalg.norm(d))
        if dist == 0.0:
            raise ValueError("the state is at the apex, where the cone barrier is undefined")
        a = np.array(self.axis)
        cos_t = math.cos(self.half_angle)
        dhat = d / dist
        v = x[3:]
        v_radial = float(dhat @ v)
        g = a - cos_t * dhat
        h = float(a @ d) - cos_t * dist
        curvature = cos_t * max(0.0, float(v @ v) - v_radial * v_radial) / dist
        return h, g, float(g @ v), curvature

    def value(self, x) -> float:
        """Return h, m, at the Hill-frame state `x` (m, m/s); 0 at the apex."""
        d = _state(x)[:3] - np.array(self.apex)
        return float(np.array(self.axis) @ d) - math.cos(self.half_angle) * float(np.linalg.norm(d))

    def gradient(self, x) -> np.ndarray:
        """Return the position gradient of h, a - cos(theta) dhat (unitless).

        Raises
        ------
        ValueError
            If `x` is malformed or at the apex.
        """
        return self._terms(_state(x))[1]

    def contains(self, x) -> bool:
        """Return whether h >= 0 and h' + max(p1, p2) h >= 0 at the state `x` (m, m/s).

        Raises
        ------
        ValueError
            If `x` is malformed or at the apex.
        """
        h, _, h_dot, _ = self._terms(_state(x))
        return h >= 0.0 and h_dot + max(self.poles) * h >= 0.0

    def row(self, x, a_cw) -> tuple[np.ndarray, float] | None:
        """Return the barrier row a . u >= b within `active_range` of the apex, else None.

        Parameters
        ----------
        x : array_like, shape (6,)
            Hill-frame state, m and m/s.
        a_cw : array_like, shape (3,)
            Natural acceleration of the dynamics at `x`, m/s^2, Hill frame.

        Returns
        -------
        tuple of (ndarray, float) or None
            The gradient g and the right-hand side
            cos(theta) |v_perp|^2 / |d| - g . a_cw - (p1 + p2) h' - p1 p2 h, m/s^2. Outside the
            cone the row drives h back up. None beyond `active_range` or at the apex.

        Raises
        ------
        ValueError
            If `x` or `a_cw` is malformed.
        """
        x = _state(x)
        a_cw = _accel(a_cw)
        dist = float(np.linalg.norm(x[:3] - np.array(self.apex)))
        if dist == 0.0 or dist > self.active_range:
            return None
        h, g, h_dot, curvature = self._terms(x)
        k0, k1 = self.gains
        return g, curvature - float(g @ a_cw) - k1 * h_dot - k0 * h

    def infeasible_radius(self, v_lat: float, a_max: float) -> float:
        """Return the distance from the apex inside which the row cannot be met, m.

        Parameters
        ----------
        v_lat : float
            Speed normal to the line of sight from the apex, m/s, nonnegative.
        a_max : float
            The thrust limit, m/s^2, positive.

        Returns
        -------
        float
            cot(theta) v_lat^2 / a_max. On the surface with h = h' = 0 the row needs
            g . u >= cos(theta) v_lat^2 / |d| and a ball of radius a_max offers at most
            a_max sin(theta); a box contains that ball, so the radius holds for both.

        Raises
        ------
        ValueError
            If `v_lat` is negative or `a_max` is not positive.
        """
        v_lat = _nonnegative(v_lat, "v_lat")
        a_max = _positive(a_max, "a_max")
        return v_lat * v_lat / (math.tan(self.half_angle) * a_max)


@dataclass(frozen=True, kw_only=True)
class PassiveBarrier:
    """Free-drift barrier: the certified lowest keep-out margin of the unforced motion.

    Parameters
    ----------
    radius : float
        The keep-out radius R, m, positive.
    n : float
        Mean motion of the target orbit, rad/s, positive.
    grid : float
        Spacing of the drift samples, s, positive; short enough that the in-step bounds of
        `vbar.constraints.interval_bounds` exist (below about 350 s at 500 km).
    t_cap : float, optional
        The longest drift horizon, s, at least `grid`; ten orbits, 20 pi / n, by default.
    terminal : bool
        False certifies the drift up to the horizon T(x); True certifies it for all time, through
        a terminal set that the drift must reach within t_cap.

    Raises
    ------
    ValueError
        If a parameter is out of range or `grid` is too long for the in-step bounds.
    TypeError
        If `terminal` is not a bool.

    Notes
    -----
    h_p(x) = min over tau >= 0 of |r_drift(tau)| - R with zero thrust (Ames et al., sec. III,
    eq. 12, with the backup beta = 0, which every thrust limit contains). In the
    Clohessy-Wiltshire solution the along-track position is y0 - 2 vx0 / n
    + (6 x0 + 4 vy0 / n) sin(n tau) + (2 vx0 / n) cos(n tau) - (6 n x0 + 3 vy0) tau, whose
    periodic part is at most B = 6 |x0| + |y0| + 4 |vx0| / n + 4 |vy0| / n. Past
    (R + B) / |6 n x0 + 3 vy0| the range exceeds R, so without `terminal` the sign of the minimum
    is decided on the horizon T(x) = min(t_cap, (R + B) / |6 n x0 + 3 vy0| + grid). Samples
    `grid` s apart are certified on every interval with the first- and second-order bounds that
    `vbar.constraints.segment_minimum` uses, without refinement. When the horizon is t_cap the
    sign holds for t_cap only: on a closed relative orbit the minimum repeats every orbit.

    With `terminal`, the same solution reads r = (x_c + b cos psi, y_c - 2 b sin psi,
    a_z cos(psi + delta)) with psi = n tau + alpha, x_c = 4 x0 + 2 vy0 / n, y_c = y0 - 2 vx0 / n,
    b = |(3 x0 + 2 vy0 / n, vx0 / n)| and a_z = |(z0, vz0 / n)|. Free drift keeps x_c, b, a_z
    and delta and moves y_c at -1.5 n x_c. The terminal set holds the states whose radial and
    cross-track tube, min over psi of |(x_c + b cos psi, a_z cos(psi + delta))|, is at least R
    (with x_c = 0 exactly, the lowest range of the closed orbit may be used instead), and those
    whose y_c recedes or stands still with |y_c| - 2 b - R >= 0. The tube is bounded below on
    1024 phases by the sampled minimum less curvature * step^2 / 8. Both parts depend on the
    invariants or on a |y_c| that does not fall, so their margins hold for all time. Any other
    state is in the set when a drift sample within t_cap lies in the far part and every sample
    interval before it is certified; h_p is then the lowest of those bounds and the terminal
    margin, and at most -1 mm when the drift reaches no terminal state within t_cap. When
    `grid` divides a step dt, the samples of the state one drift step later are samples of the
    state now, so h_p does not fall over the step and free drift keeps the set.
    """

    radius: float
    n: float
    grid: float = 5.0
    t_cap: float | None = None
    terminal: bool = False
    _keep_out: KeepOut = field(init=False, repr=False, compare=False)
    _tau: np.ndarray = field(init=False, repr=False, compare=False)
    _phi: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        radius = _positive(self.radius, "radius")
        n = _positive(self.n, "n")
        grid = _positive(self.grid, "grid")
        t_cap = 20.0 * math.pi / n if self.t_cap is None else _positive(self.t_cap, "t_cap")
        if t_cap < grid:
            raise ValueError("t_cap must be at least grid")
        if not isinstance(self.terminal, bool):
            raise TypeError("terminal must be a bool")
        speed, _ = interval_bounds(np.zeros((2, 6)), grid, 0.0, n)
        if not np.all(np.isfinite(speed)):
            raise ValueError("grid is too long for the in-step bounds at this mean motion")
        tau = grid * np.arange(math.ceil(t_cap / grid) + 1)
        tau.flags.writeable = False
        # Stored as (6, samples, 6), so every drift sample of a state is one contraction over
        # the leading axis of a contiguous slice.
        phi = np.ascontiguousarray(np.transpose(cw_discrete(n, tau)[0], (2, 0, 1)))
        phi.flags.writeable = False
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "t_cap", t_cap)
        object.__setattr__(self, "_keep_out", KeepOut(radius))
        object.__setattr__(self, "_tau", tau)
        object.__setattr__(self, "_phi", phi)

    @property
    def _cap(self) -> float:
        return float(self.t_cap)  # type: ignore[arg-type]

    def _horizons(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Horizons T (s) and secular drift rates |6 n x + 3 vy| (m/s) of states (k, 6)."""
        n = self.n
        drift = np.abs(6.0 * n * x[:, 0] + 3.0 * x[:, 4])
        bound = (
            6.0 * np.abs(x[:, 0])
            + np.abs(x[:, 1])
            + 4.0 * np.abs(x[:, 3]) / n
            + 4.0 * np.abs(x[:, 4]) / n
        )
        reach = self.radius + bound
        # Comparing products avoids dividing by a vanishing drift rate.
        decided = drift * (self._cap - self.grid) > reach
        t = np.where(decided, reach / np.where(decided, drift, 1.0) + self.grid, self._cap)
        return np.minimum(t, self._cap), drift

    def horizon(self, x) -> tuple[float, bool]:
        """Return the drift horizon, s, and whether the sign of `value` holds for all time.

        Parameters
        ----------
        x : array_like, shape (6,)
            Hill-frame state, m and m/s.

        Returns
        -------
        T : float
            Without `terminal`, min(t_cap, (R + B) / |6 n x + 3 vy| + grid). With it, the time of
            the first drift sample in the terminal set, 0 for a terminal state, or the last
            sample checked when no terminal state is reached.
        proven : bool
            Without `terminal`, True when T < t_cap: the drift stays outside the keep-out after
            T. With it, True when the drift reaches the terminal set by T.

        Raises
        ------
        ValueError
            If `x` is not six finite numbers.
        """
        x = _state(x)
        if self.terminal:
            _, last, proven = self._terminal(x)
            return float(self._tau[last]), proven
        t, _ = self._horizons(x[None, :])
        return float(t[0]), bool(t[0] < self._cap)

    def _samples(self, x: np.ndarray) -> np.ndarray:
        """Drift samples (m, 6) of the state `x` from 0 to its horizon, `grid` s apart."""
        t, _ = self._horizons(x[None, :])
        m = min(math.ceil(float(t[0]) / self.grid - 1e-9) + 1, len(self._tau))
        return np.tensordot(x, self._phi[:, :m], axes=(0, 0))

    def _sampled_minimum(self, states: np.ndarray) -> float:
        """The lowest certified bound over the sample intervals of `states` and at the first, m."""
        h = self.grid
        speed, accel = interval_bounds(states, h, 0.0, self.n)
        margins = self._keep_out.margin(states)
        bounds = np.maximum(
            0.5 * (margins[:-1] + margins[1:] - speed * h),
            self._keep_out.second_order_bounds(states, h, speed, accel),
        )
        return float(min(bounds.min(initial=np.inf), margins[0]))

    def _far(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Far margins |y_c| - 2 b - R (m) of states (k, 6) and whether each y_c recedes."""
        n = self.n
        x_c = 4.0 * states[:, 0] + 2.0 * states[:, 4] / n
        y_c = states[:, 1] - 2.0 * states[:, 3] / n
        b = np.hypot(3.0 * states[:, 0] + 2.0 * states[:, 4] / n, states[:, 3] / n)
        # dy_c/dt = -1.5 n x_c, so |y_c| does not fall when x_c y_c <= 0.
        return np.abs(y_c) - 2.0 * b - self.radius, x_c * y_c <= 0.0

    def _tube(self, x: np.ndarray) -> float:
        """A lower bound on the keep-out margin of the drift for all time from (x, z), m.

        With x_c = 0 exactly the orbit is closed, and the bound on its full range is used when
        it is higher.
        """
        n = self.n
        x_c = 4.0 * x[0] + 2.0 * x[4] / n
        p, q = 3.0 * x[0] + 2.0 * x[4] / n, x[3] / n
        b = math.hypot(p, q)
        a_z = math.hypot(x[2], x[5] / n)
        delta = math.atan2(-x[5] / n, x[2]) - math.atan2(-q, -p)
        cross = a_z * (_COS_PSI * math.cos(delta) - _SIN_PSI * math.sin(delta))
        rho2 = (x_c + b * _COS_PSI) ** 2 + cross**2
        # Between phases a function with |f''| <= curvature lies at most curvature step^2 / 8
        # below the lower of its two end values.
        dip = _PSI_STEP * _PSI_STEP / 8.0
        curvature = 2.0 * abs(x_c) * b + 2.0 * b * b + 2.0 * a_z * a_z
        lowest = float(rho2.min()) - curvature * dip
        if x_c == 0.0:
            y_c = x[1] - 2.0 * x[3] / n
            full = rho2 + (y_c - 2.0 * b * _SIN_PSI) ** 2
            curvature += 4.0 * abs(y_c) * b + 8.0 * b * b
            lowest = max(lowest, float(full.min()) - curvature * dip)
        return math.sqrt(max(lowest, 0.0)) - self.radius

    def _terminal(self, x: np.ndarray) -> tuple[float, int, bool]:
        """The value with `terminal` (m), the last sample checked and whether it is terminal."""
        tube = self._tube(x)
        far, recedes = self._far(x[None, :])
        if tube >= 0.0 or (recedes[0] and far[0] >= 0.0):
            return max(tube, float(far[0]) if recedes[0] else -math.inf), 0, True
        n = self.n
        x_c = 4.0 * x[0] + 2.0 * x[4] / n
        y_c = x[1] - 2.0 * x[3] / n
        b = math.hypot(3.0 * x[0] + 2.0 * x[4] / n, x[3] / n)
        rate = 1.5 * n * abs(x_c)
        # Time until |y_c| - 2 b - R reaches zero while receding; an approaching y_c first
        # passes through zero.
        if rate == 0.0:
            reach = math.inf
        elif recedes[0]:
            reach = -float(far[0]) / rate
        else:
            reach = (abs(y_c) + 2.0 * b + self.radius) / rate
        m = min(math.ceil(min(reach, self._cap) / self.grid) + 1, len(self._tau))
        states = np.tensordot(x, self._phi[:, :m], axes=(0, 0))
        far, recedes = self._far(states)
        hits = np.flatnonzero(recedes & (far >= 0.0))
        if hits.size == 0:
            # No terminal state within t_cap, so nothing certifies the drift beyond it.
            return min(self._sampled_minimum(states), -KeepOut.resolution), m - 1, False
        k = int(hits[0])
        tail = max(float(far[k]), tube)
        return min(self._sampled_minimum(states[: k + 1]), tail), k, True

    def _value(self, x: np.ndarray) -> float:
        if self.terminal:
            return self._terminal(x)[0]
        return self._sampled_minimum(self._samples(x))

    def value(self, x) -> float | np.ndarray:
        """Return the certified free-drift margin h_p, m.

        Parameters
        ----------
        x : array_like, shape (6,) or (k, 6)
            Hill-frame states, m and m/s.

        Returns
        -------
        float or ndarray, shape (k,)
            A certified lower bound on min |r_drift| - R. Without `terminal`, the lowest bound
            over the sample intervals up to the horizon: h_p >= 0 certifies the drift up to the
            horizon, and for all time when `horizon` reports it proven. With `terminal`,
            h_p >= 0 certifies the drift for all time. A negative value bounds the deepest
            incursion before the horizon from below, or is at most -1 mm when no terminal state
            is reached within t_cap.

        Raises
        ------
        ValueError
            If `x` is not a finite array of states.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim not in (1, 2) or x.shape[-1] != 6 or not np.all(np.isfinite(x)):
            raise ValueError("x must be finite states of six numbers")
        if x.ndim == 1:
            return self._value(x)
        # Horizons differ by up to fifty times between states, so each gets its own samples.
        return np.array([self._value(state) for state in x])

    def contains(self, x) -> bool:
        """Return whether the certified drift margin of `x` (m, m/s) is nonnegative."""
        return bool(self.value(_state(x)) >= 0.0)

    def gradient(self, x) -> np.ndarray:
        """Return the gradient of the exact drift minimum by the envelope theorem.

        Parameters
        ----------
        x : array_like, shape (6,)
            Hill-frame state, m and m/s.

        Returns
        -------
        ndarray, shape (6,)
            Phi_r(tau*)^T rhat(tau*), in m per m and m per m/s, with tau* the minimiser of the
            drift range, found from the samples and refined by a bounded scalar search.

        Raises
        ------
        ValueError
            If the relative orbit is closed (|6 n x + 3 vy| < 1e-9 m/s), where the minimiser is
            not unique, if tau* is at 0 or at the horizon end, where the minimum is not
            stationary, or if the drift passes through the target.
        """
        x = _state(x)
        _, drift = self._horizons(x[None, :])
        if drift[0] < _MIN_DRIFT:
            raise ValueError("closed relative orbit, the drift minimiser is not unique")
        rho = np.linalg.norm(self._samples(x)[:, :3], axis=1)
        k = int(np.argmin(rho))
        if k == 0 or k == len(rho) - 1:
            raise ValueError("the drift minimum is at the start or the end of the horizon")
        found = minimize_scalar(
            lambda tau: float(np.linalg.norm(cw_discrete(self.n, tau)[0][:3] @ x)),
            bounds=(float(self._tau[k - 1]), float(self._tau[k + 1])),
            method="bounded",
            options={"xatol": _TAU_TOL},
        )
        phi_r = cw_discrete(self.n, float(found.x))[0][:3]
        r = phi_r @ x
        rho_min = float(np.linalg.norm(r))
        if rho_min == 0.0:
            raise ValueError("the drift passes through the target")
        return phi_r.T @ (r / rho_min)

    def backup(self, x, actuator: ActuatorSet) -> np.ndarray:
        """Return the backup command, zero thrust (m/s^2), which every actuator set contains."""
        _state(x)
        return np.zeros(3)


def braking_deceleration(
    *,
    a_max: float,
    layout: str,
    v_cap: float,
    dt: float,
    radius: float,
    rho_cap: float,
    n: float,
) -> float:
    """Return the braking acceleration that held full braking delivers in a region.

    Parameters
    ----------
    a_max : float
        The thrust limit, m/s^2, positive.
    layout : str
        "box" or "ball".
    v_cap : float
        The largest speed of the region, m/s, positive.
    dt : float
        The hold time of a command, s, positive.
    radius : float
        The keep-out radius R, m, positive.
    rho_cap : float
        The largest range of the region, m, positive.
    n : float
        Mean motion, rad/s, positive.

    Returns
    -------
    float
        a_max (1 - k v_cap dt / R) - (3 n^2 rho_cap + 2 n v_cap), m/s^2, with k = sqrt(3) for a
        box and 1 for a ball. Outside the keep-out rhat turns by at most v_cap t / R within a
        step, so a held beta keeps rhat . beta >= a_max - |beta| v_cap dt / R, and the CW terms
        take at most 3 n^2 rho + 2 n |v| of it. Zero or negative when no braking barrier exists
        for these values.

    Raises
    ------
    ValueError
        If a parameter is not positive or `layout` is unknown.
    """
    if layout not in ACTUATOR_LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}, expected {ACTUATOR_LAYOUTS}")
    a_max = _positive(a_max, "a_max")
    v_cap = _positive(v_cap, "v_cap")
    dt = _positive(dt, "dt")
    radius = _positive(radius, "radius")
    rho_cap = _positive(rho_cap, "rho_cap")
    n = _positive(n, "n")
    k = math.sqrt(3.0) if layout == "box" else 1.0
    held = a_max * (1.0 - k * v_cap * dt / radius)
    return held - (3.0 * n * n * rho_cap + 2.0 * n * v_cap)


@dataclass(frozen=True, kw_only=True)
class BrakingBarrier:
    """Braking barrier h_b = rho - R - max(0, -rho_dot)^2 / (2 a_b) for the keep-out sphere.

    Parameters
    ----------
    radius : float
        The keep-out radius R, m, positive.
    a_b : float
        The braking acceleration the barrier assumes, m/s^2, positive.
    rho_cap : float
        The largest range of the region where `a_b` is valid, m, above `radius`.
    v_cap : float
        The largest speed of that region, m/s, positive.

    Raises
    ------
    ValueError
        If a parameter is out of range or not finite.

    Notes
    -----
    The keep-out analogue of the lane-keeping barrier of Ames et al., sec. V.B. Under full
    outward braking beta along rhat, rhat . beta >= a_max, and while closing at c = -rho_dot,
    h_b' = rho_dot + (c / a_b) rho_ddot >= rho_dot + c = 0 as long as rho_ddot >= a_b, which
    `braking_deceleration` makes hold in the region rho <= rho_cap, |v| <= v_cap. L_g h_b
    vanishes at rho_dot = 0, so a first-order row on h_b puts no condition on u there and fails
    under sample and hold; the barrier has no row and is enforced by a line search toward
    `backup`.
    """

    radius: float
    a_b: float
    rho_cap: float
    v_cap: float

    def __post_init__(self):
        radius = _positive(self.radius, "radius")
        rho_cap = _positive(self.rho_cap, "rho_cap")
        if rho_cap <= radius:
            raise ValueError("rho_cap must exceed radius")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "a_b", _positive(self.a_b, "a_b"))
        object.__setattr__(self, "rho_cap", rho_cap)
        object.__setattr__(self, "v_cap", _positive(self.v_cap, "v_cap"))

    def value(self, x) -> float:
        """Return h_b, m, at the Hill-frame state `x` (m, m/s)."""
        rho, _, rho_dot, _ = _range_terms(_state(x))
        c = max(0.0, -rho_dot)
        return rho - self.radius - c * c / (2.0 * self.a_b)

    def contains(self, x) -> bool:
        """Return whether h_b >= 0 inside the region rho <= rho_cap, |v| <= v_cap.

        Raises
        ------
        ValueError
            If `x` is malformed or its range is zero.
        """
        x = _state(x)
        rho = float(np.linalg.norm(x[:3]))
        speed = float(np.linalg.norm(x[3:]))
        return self.value(x) >= 0.0 and rho <= self.rho_cap and speed <= self.v_cap

    def horizon(self, x) -> tuple[float, bool]:
        """Return (inf, True): the braking argument holds for all time inside the region."""
        _state(x)
        return math.inf, True

    def backup(self, x, actuator: ActuatorSet) -> np.ndarray:
        """Return full outward braking, m/s^2.

        Parameters
        ----------
        x : array_like, shape (6,)
            Hill-frame state, m and m/s.
        actuator : ActuatorSet
            The thrust limit.

        Returns
        -------
        ndarray, shape (3,)
            Along rhat in both layouts: a_max rhat / max_i |rhat_i| for a box, the longest radial
            command in it, and a_max rhat for a ball. rhat . beta >= a_max, and |beta| is at
            most sqrt(3) a_max (box) or a_max (ball), the bound `braking_deceleration` allows
            for the turn of rhat within a step. A command with a part across the line of sight,
            such as a_max sign(rhat), would raise the speed and carry the state out of the
            region where the barrier holds.

        Raises
        ------
        ValueError
            If `x` is malformed or its range is zero.
        """
        _, rhat, _, _ = _range_terms(_state(x))
        if actuator.layout == "box":
            return actuator.a_max * rhat / float(np.max(np.abs(rhat)))
        return actuator.a_max * rhat
