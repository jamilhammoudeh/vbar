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
from .models import integrate_segment, point_mass_difference

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
