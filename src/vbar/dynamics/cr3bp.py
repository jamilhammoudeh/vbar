"""Circular restricted three-body problem (CR3BP): equations, Lagrange points, relative motion.

Nondimensional synodic frame (Koon, Lo, Marsden and Ross, Dynamical Systems, the Three-Body
Problem and Space Mission Design, sec. 2.3): origin at the barycentre, x from the larger primary
to the smaller, y along the smaller primary's motion, z along the primaries' angular momentum;
primaries at (-mu, 0, 0) and (1 - mu, 0, 0); unit distance, angular rate and total mass.
Equations of motion (their eq 2.3.8):

    x'' - 2 y' = dOmega/dx,  y'' + 2 x' = dOmega/dy,  z'' = dOmega/dz,
    Omega = (x^2 + y^2) / 2 + (1 - mu) / r1 + mu / r2.

The frame has the orientation and Coriolis sign of the Hill frame of the smaller primary's
orbit, and at mu = 0 the linearization about (1, 0, 0) is CW with n = 1. Jacobi constants are
Szebehely's, C = 2 Omega - v^2; Koon et al.'s convention adds mu (1 - mu) (their eq 2.3.14),
which puts L4 and L5 at exactly 3. Coriolis terms do no work, so the Jacobi constant cannot
detect a sign error in them; the inertial-frame cross-check can.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from .bodies import ThreeBodySystem
from .cw import CWDiscretizer
from .models import CWDynamics, integrate_segment, point_mass_difference

_CORIOLIS = np.array([[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def primaries(mu: float) -> tuple[tuple[float, np.ndarray], ...]:
    """(mass, position) of the larger and the smaller primary."""
    return ((1.0 - mu, np.array([-mu, 0.0, 0.0])), (mu, np.array([1.0 - mu, 0.0, 0.0])))


def potential(r: ArrayLike, mu: float) -> np.ndarray:
    """Omega = (x^2 + y^2) / 2 + (1 - mu) / r1 + mu / r2 at positions `r` (..., 3)."""
    r = np.asarray(r, dtype=float)
    omega = 0.5 * (r[..., 0] ** 2 + r[..., 1] ** 2)
    for mass, p in primaries(mu):
        if mass:
            omega = omega + mass / np.linalg.norm(r - p, axis=-1)
    return omega


def acceleration(r: ArrayLike, v: ArrayLike, mu: float) -> np.ndarray:
    """Acceleration in the synodic frame at positions `r` and velocities `v`, (..., 3)."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.stack([r[..., 0] + 2.0 * v[..., 1], r[..., 1] - 2.0 * v[..., 0], 0.0 * r[..., 2]], -1)
    for mass, p in primaries(mu):
        if mass:
            d = r - p
            d2 = np.sum(d * d, axis=-1, keepdims=True)
            a = a - mass * d / (d2 * np.sqrt(d2))
    return a


def rhs(t: float, state: np.ndarray, mu: float) -> np.ndarray:
    """Time derivative of a state [x, y, z, vx, vy, vz]."""
    return np.concatenate([state[3:6], acceleration(state[0:3], state[3:6], mu)])


def jacobi_constant(state: ArrayLike, mu: float) -> np.ndarray:
    """Szebehely's Jacobi constant 2 Omega - v^2 of states (..., 6)."""
    state = np.asarray(state, dtype=float)
    v = state[..., 3:6]
    return 2.0 * potential(state[..., 0:3], mu) - np.sum(v * v, axis=-1)


def potential_hessian(r: ArrayLike, mu: float) -> np.ndarray:
    """Second derivatives of Omega at `r` (..., 3), shape (..., 3, 3)."""
    r = np.asarray(r, dtype=float)
    hessian = np.zeros(r.shape + (3,))
    hessian[..., 0, 0] = hessian[..., 1, 1] = 1.0
    for mass, p in primaries(mu):
        if mass:
            d = r - p
            d2 = np.sum(d * d, axis=-1)[..., None, None]
            dd = d[..., :, None] * d[..., None, :]
            hessian = hessian + mass * (3.0 * dd / d2 - np.eye(3)) / (d2 * np.sqrt(d2))
    return hessian


def jacobian(r: ArrayLike, mu: float) -> np.ndarray:
    """Matrix A of the variational equations, d(state)/dt = A d(state), at `r`, (..., 6, 6)."""
    hessian = potential_hessian(r, mu)
    a = np.zeros(hessian.shape[:-2] + (6, 6))
    a[..., 0:3, 3:6] = np.eye(3)
    a[..., 3:6, 0:3] = hessian
    a[..., 3:6, 3:6] = _CORIOLIS
    return a


def lagrange_points(mu: float) -> np.ndarray:
    """Lagrange points L1 to L5, shape (5, 3).

    L1 lies between the primaries, L2 beyond the smaller, L3 beyond the larger, and L4 leads
    the smaller primary by 60 deg.
    """

    def slope(x: float) -> float:
        return float(acceleration([x, 0.0, 0.0], np.zeros(3), mu)[0])

    # dOmega/dx runs from -inf to +inf between neighbouring singularities, so each interval
    # holds exactly one root (Koon et al. sec. 2.5).
    tiny = 1e-12
    x1 = brentq(slope, -mu + tiny, 1.0 - mu - tiny, xtol=1e-15, rtol=1e-15)
    x2 = brentq(slope, 1.0 - mu + tiny, 2.0, xtol=1e-15, rtol=1e-15)
    x3 = brentq(slope, -2.0, -mu - tiny, xtol=1e-15, rtol=1e-15)
    h = math.sqrt(3.0) / 2.0
    return np.array(
        [
            [x1, 0.0, 0.0],
            [x2, 0.0, 0.0],
            [x3, 0.0, 0.0],
            [0.5 - mu, h, 0.0],
            [0.5 - mu, -h, 0.0],
        ]
    )


def collinear_coefficient(mu: float, point: int, n: int) -> float:
    """Legendre coefficient c_n of the potential's expansion about L1 or L2 (`point` 1 or 2).

    Koon et al. eq 6.3.1 with gamma the point's distance from the smaller primary; c2 sets the
    linear dynamics, c3 and c4 Richardson's third-order halo solution.
    """
    x = lagrange_points(mu)[point - 1, 0]
    gamma = abs(x - (1.0 - mu))
    if point == 1:
        return (mu + (-1) ** n * (1.0 - mu) * (gamma / (1.0 - gamma)) ** (n + 1)) / gamma**3
    if point == 2:
        return (-1) ** n * (mu + (1.0 - mu) * (gamma / (1.0 + gamma)) ** (n + 1)) / gamma**3
    raise ValueError("collinear coefficients are defined here for L1 and L2 only")


def libration_frequencies(c2: float) -> tuple[float, float, float, float]:
    """Saddle rate lambda, in-plane and out-of-plane frequencies, and kappa at a collinear point.

    Koon et al. eqs 6.4.1 to 6.4.3: lambda^2 and wp^2 are (c2 - 2 +- sqrt(9 c2^2 - 8 c2)) / 2
    with the signs taken to make them positive, wv^2 = c2, kappa = (wp^2 + 1 + 2 c2) / (2 wp),
    the ratio of y to x amplitude of the in-plane oscillation.
    """
    root = math.sqrt(9.0 * c2 * c2 - 8.0 * c2)
    lam = math.sqrt(0.5 * (c2 - 2.0 + root))
    wp = math.sqrt(0.5 * (2.0 - c2 + root))
    return lam, wp, math.sqrt(c2), (wp * wp + 1.0 + 2.0 * c2) / (2.0 * wp)


class CR3BPDynamics:
    """Chaser motion relative to a target in the CR3BP, in synodic axes, SI at the boundary.

    The target follows its own CR3BP trajectory from the nondimensional `target_state`; the
    chaser is an offset driven by the difference of the primaries' gravity, computed without
    cancellation (`point_mass_difference`), plus the linear Coriolis and centrifugal terms,
    and integrated with tolerances of its own: integrating two absolute states and differencing
    them closes a halo orbit only to centimetres. The relative state is chaser minus target in
    synodic axes, in m and m/s, with the velocity as seen in the synodic frame, and the
    acceleration is held constant in synodic axes. `n` is the synodic frame's rate in rad/s;
    a target in the CR3BP has no mean motion of its own.
    """

    name = "cr3bp"

    def __init__(self, system: ThreeBodySystem, target_state: ArrayLike, rtol: float = 1e-12):
        self.system = system
        self.target_start = np.array(target_state, dtype=float)
        self.rtol = rtol
        self.n = 1.0 / system.time
        self._scale = np.repeat([system.length, system.velocity], 3)
        # The same absolute tolerances as the two-body model, 1e-6 m and 1e-9 m/s on the
        # target and 1e-9 m and 1e-12 m/s on the offset, in nondimensional units.
        self._atol = np.concatenate([[1e-6] * 3, [1e-9] * 3, [1e-9] * 3, [1e-12] * 3]) / np.tile(
            self._scale, 2
        )
        self._y = np.zeros(12)
        self._elapsed = 0.0
        self._sol = None
        self._step: float | None = None

    def reset(self, rel_state: ArrayLike) -> None:
        offset = np.asarray(rel_state, dtype=float) / self._scale
        self._y = np.concatenate([self.target_start, offset])
        self._elapsed = 0.0
        self._sol = None
        self._step = None

    def _rhs(self, t: float, y: np.ndarray, accel: np.ndarray) -> np.ndarray:
        r_t, v_t, dr, dv = y[0:3], y[3:6], y[6:9], y[9:12]
        mu = self.system.mu
        da = np.array([dr[0] + 2.0 * dv[1], dr[1] - 2.0 * dv[0], 0.0]) + accel
        for mass, p in primaries(mu):
            if mass:
                da = da + point_mass_difference(r_t - p, dr, mass)
        return np.concatenate([v_t, acceleration(r_t, v_t, mu), dv, da])

    def advance(
        self, accel: ArrayLike, duration: float, n_samples: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Hold `accel` (m/s^2, synodic axes) for `duration` s.

        Returns the final relative state and, if `n_samples > 0`, the states at `n_samples`
        evenly spaced times in [0, duration], endpoints included, shape (n_samples, 6).
        """
        a = np.asarray(accel, dtype=float) / self.system.acceleration
        tau = duration / self.system.time
        sol, self._step = integrate_segment(
            self._rhs, self._y, tau, (a,), self.rtol, self._atol, self._step
        )
        self._sol = sol
        self._y = sol.y[:, -1]
        self._elapsed += tau
        samples = None
        if n_samples > 0:
            samples = sol.sol(np.linspace(0.0, tau, n_samples))[6:12].T * self._scale
        return self._y[6:12] * self._scale, samples

    def state_in_last_segment(self, tau: float) -> np.ndarray:
        """State `tau` seconds into the most recent segment."""
        if self._sol is None:
            raise RuntimeError("no segment has been advanced since reset")
        return self._sol.sol(tau / self.system.time)[6:12] * self._scale

    def target_state(self) -> np.ndarray:
        """Target's nondimensional synodic state."""
        return self._y[0:6].copy()

    def chaser_state(self) -> np.ndarray:
        """Chaser's nondimensional synodic state."""
        return self._y[0:6] + self._y[6:12]


def _hold_terms(mu: float, s: np.ndarray) -> tuple[np.ndarray, ...]:
    """C, S, K, D, E of exp(A s) with A^2 = mu: cosh(sqrt(mu) s), its integrals and tails.

    S = int C, K = (C - 1) / mu = int S, D = (S - s) / mu, E = (K - s^2 / 2) / mu, all power
    series in y = mu s^2 with coefficients 1 / (2k + j)!, valid for either sign of mu. Series
    below |y| = 1, where the closed forms of D and E cancel; above it they lose at most two
    digits of terms that are then small beside C, S and K.
    """
    y = mu * s * s
    small = np.abs(y) < 1.0
    series = [sum(y**k / math.factorial(2 * k + j) for k in range(12)) for j in range(5)]
    x = np.sqrt(np.abs(y))
    x_safe = np.where(small, 1.0, x)
    if mu > 0.0:
        c, sn = np.cosh(x_safe), np.sinh(x_safe)
        k = 2.0 * np.sinh(0.5 * x_safe) ** 2
        d, e = sn - x_safe, c - 1.0 - 0.5 * x_safe**2
    else:
        c, sn = np.cos(x_safe), np.sin(x_safe)
        k = 2.0 * np.sin(0.5 * x_safe) ** 2
        d, e = x_safe - sn, c - 1.0 + 0.5 * x_safe**2
    closed = [c, sn / x_safe, k / x_safe**2, d / x_safe**3, e / x_safe**4]
    powers = [1.0, s, s**2, s**3, s**4]
    return tuple(
        p * np.where(small, ser, cl) for p, ser, cl in zip(powers, series, closed, strict=True)
    )


def libration_discrete(
    n: float, c2: float, dt: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold matrices of the linear dynamics at a collinear point.

    x'' - 2 n y' - (1 + 2 c2) n^2 x = a_x,  y'' + 2 n x' + (c2 - 1) n^2 y = a_y,
    z'' + c2 n^2 z = a_z (Koon et al. eq 6.4.1, n the synodic rate); c2 = 1 is CW. The in-plane
    matrix satisfies (A^2 - lambda^2)(A^2 + wp^2) = 0, so exp(A t) = C(A^2) + A S(A^2) is a
    two-point interpolation in A^2, exact also at c2 = 1 where CW's zero eigenvalue is
    defective. Same returns as `vbar.dynamics.cw_discrete`.
    """
    s = n * np.asarray(dt, dtype=float)
    lam, wp, wv, _ = libration_frequencies(c2)
    mu1, mu2 = lam * lam, -wp * wp
    delta = mu1 - mu2
    _, _, k1, d1, e1 = _hold_terms(mu1, s)
    _, _, k2, d2, e2 = _hold_terms(mu2, s)
    # Interpolation weights on A^2 and I for C, S and K; the differences that would cancel
    # are written as sums of like-signed terms.
    c_a, c_b = (mu1 * k1 - mu2 * k2) / delta, 1.0 + mu1 * mu2 * (k2 - k1) / delta
    s_a, s_b = (mu1 * d1 - mu2 * d2) / delta, s + mu1 * mu2 * (d2 - d1) / delta
    k_a, k_b = (mu1 * e1 - mu2 * e2) / delta, 0.5 * s * s + mu1 * mu2 * (e2 - e1) / delta
    a = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0 + 2.0 * c2, 0.0, 0.0, 2.0],
            [0.0, 1.0 - c2, -2.0, 0.0],
        ]
    )
    powers = [np.eye(4), a, a @ a, a @ a @ a]

    def combine(*weights: np.ndarray) -> np.ndarray:
        return sum(w[..., None, None] * m for w, m in zip(weights, powers, strict=True))

    phi_in = combine(c_b, s_b, c_a, s_a)
    gamma_in = combine(s_b, k_b, s_a, k_a)[..., :, 2:4]
    zx = wv * s
    sin_z, cos_z = np.sin(zx), np.cos(zx)
    shape = s.shape
    phi = np.zeros(shape + (6, 6))
    gamma = np.zeros(shape + (6, 3))
    plane = [0, 1, 3, 4]
    phi[(..., *np.ix_(plane, plane))] = phi_in
    gamma[(..., *np.ix_(plane, [0, 1]))] = gamma_in
    phi[..., 2, 2], phi[..., 2, 5] = cos_z, sin_z / wv
    phi[..., 5, 2], phi[..., 5, 5] = -wv * sin_z, cos_z
    gamma[..., 2, 2] = 2.0 * np.sin(0.5 * zx) ** 2 / wv**2
    gamma[..., 5, 2] = sin_z / wv
    # From time in 1/n and velocity in units of n to seconds and m/s.
    scale = np.array([1.0, 1.0, 1.0, n, n, n])
    return phi * scale[:, None] / scale[None, :], gamma * scale[:, None] / n**2


class LibrationDiscretizer(CWDiscretizer):
    """Caches `libration_discrete` matrices by duration, keyed to the nanosecond."""

    def __init__(self, n: float, c2: float):
        super().__init__(n)
        self.c2 = c2

    def _discrete(self, dt: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return libration_discrete(self.n, self.c2, dt)


class LibrationPointDynamics(CWDynamics):
    """Relative motion about a target held at a collinear Lagrange point, linear and exact.

    The constant-coefficient linearization of the CR3BP at L1 or L2 in synodic axes, SI units,
    with the acceleration held in synodic axes and the exact zero-order hold of
    `libration_discrete`. `n` is the synodic frame's rate (rad/s). The motion has a saddle,
    e-folding in 1.5 days at Earth-Moon L1 and 2 days at L2, and two oscillations. With c2 = 1
    it is CW. Left out against `CR3BPDynamics` with its target at the point: terms of second
    order in the separation.
    """

    name = "libration"

    def __init__(self, n: float, c2: float):
        super().__init__(n)
        self.c2 = c2
        self.disc = LibrationDiscretizer(n, c2)

    @classmethod
    def at_point(cls, system: ThreeBodySystem, point: int) -> LibrationPointDynamics:
        """The model about L1 or L2 (`point` 1 or 2) of `system`."""
        return cls(1.0 / system.time, collinear_coefficient(system.mu, point, 2))
