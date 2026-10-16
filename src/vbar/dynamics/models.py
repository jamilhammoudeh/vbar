"""Relative-motion models behind one interface, so a scenario can change fidelity.

Every model holds the chaser's state relative to the target in the target's Hill frame
([x, y, z, vx, vy, vz], see `vbar.conventions`) and advances it under an acceleration held
constant in the Hill frame over a segment.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from ..conventions import hill_frame, hill_to_inertial, inertial_to_hill
from .bodies import Body, get_body
from .cw import CWDiscretizer, mean_motion


class Dynamics(Protocol):
    name: str
    n: float  # rad/s, mean motion of the target's reference orbit

    def reset(self, rel_state: ArrayLike) -> None: ...

    def advance(
        self, accel_hill: ArrayLike, duration: float, n_samples: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None]: ...

    def state_in_last_segment(self, tau: float) -> np.ndarray: ...


class CWDynamics:
    """Clohessy-Wiltshire motion propagated with the exact zero-order-hold solution."""

    name = "cw"

    def __init__(self, n: float):
        self.n = n
        self.disc = CWDiscretizer(n)
        self._state = np.zeros(6)
        self._segment: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self, rel_state: ArrayLike) -> None:
        self._state = np.array(rel_state, dtype=float)
        self._segment = None

    def advance(
        self, accel_hill: ArrayLike, duration: float, n_samples: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Hold `accel_hill` (m/s^2) for `duration` s.

        Returns the final state and, if `n_samples > 0`, the states at `n_samples` evenly
        spaced times in [0, duration], endpoints included, shape (n_samples, 6).
        """
        accel = np.array(accel_hill, dtype=float)
        x0 = self._state
        self._segment = (x0, accel)
        samples = None
        if n_samples > 0:
            phis, gammas = self.disc.sample_matrices(duration, n_samples)
            samples = phis @ x0 + gammas @ accel
        phi, gamma = self.disc.matrices(duration)
        self._state = phi @ x0 + gamma @ accel
        return self._state.copy(), samples

    def state_in_last_segment(self, tau: float) -> np.ndarray:
        """State `tau` seconds into the most recent segment."""
        if self._segment is None:
            raise RuntimeError("no segment has been advanced since reset")
        x0, accel = self._segment
        phi, gamma = self.disc.matrices(tau)
        return phi @ x0 + gamma @ accel


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1, keepdims=True)


def _j2_acceleration(r: np.ndarray, body: Body) -> np.ndarray:
    r2 = _dot(r, r)
    k = 1.5 * body.j2 * body.mu * body.radius**2 / (r2 * r2 * np.sqrt(r2))
    return k * r * (5.0 * r[..., 2:3] ** 2 / r2 - np.array([1.0, 1.0, 3.0]))


def gravity(r: ArrayLike, body: Body, j2: bool = False) -> np.ndarray:
    """Inertial gravitational acceleration (m/s^2) at `r`: point mass, optionally plus J2.

    `r` is body-centred inertial with z along the spin axis; it may be stacked as (..., 3).
    """
    r = np.asarray(r, dtype=float)
    r2 = _dot(r, r)
    g = -body.mu * r / (r2 * np.sqrt(r2))
    if j2 and body.j2:
        g = g + _j2_acceleration(r, body)
    return g


def differential_gravity(r: ArrayLike, dr: ArrayLike, body: Body, j2: bool = False) -> np.ndarray:
    """Gravity at `r + dr` minus gravity at `r` (m/s^2), accurate to rounding for any `dr`.

    Subtracting two 8 m/s^2 accelerations directly leaves 1e-15 m/s^2 of rounding, a 1e-9
    relative error at 1 m of separation. The point-mass part uses Battin's f(q) from Encke's
    method instead (Battin, An Introduction to the Mathematics and Methods of Astrodynamics);
    the J2 part is a direct difference, whose rounding is J2 times smaller.
    """
    r = np.asarray(r, dtype=float)
    dr = np.asarray(dr, dtype=float)
    r_c = r + dr
    q = (_dot(dr, dr) + 2.0 * _dot(dr, r)) / _dot(r, r)
    # f(q) = (|r + dr| / |r|)^3 - 1, written without subtracting nearly equal numbers.
    f = q * (3.0 + 3.0 * q + q * q) / (1.0 + (1.0 + q) ** 1.5)
    r_c2 = _dot(r_c, r_c)
    dg = -body.mu / (r_c2 * np.sqrt(r_c2)) * (dr - f * r)
    if j2 and body.j2:
        dg = dg + _j2_acceleration(r_c, body) - _j2_acceleration(r, body)
    return dg


def circular_orbit(body: Body, altitude: float, inclination: float = 0.0):
    """Inertial position and velocity at the ascending node (on +x) of a circular orbit.

    `altitude` in m above the equatorial radius, `inclination` in rad.
    """
    a = body.radius + altitude
    speed = np.sqrt(body.mu / a)
    r = np.array([a, 0.0, 0.0])
    v = speed * np.array([0.0, np.cos(inclination), np.sin(inclination)])
    return r, v


# Absolute tolerances for the integrated blocks [r_target, v_target, dr, dv] in m and m/s.
# The offset blocks get their own small tolerances so relative motion is resolved well below
# a millimetre whatever the orbit size.
_ATOL = np.repeat([1e-6, 1e-9, 1e-9, 1e-12], 3)
_RTOL = 1e-11


class TwoBodyDynamics:
    """Nonlinear two-body relative motion, optionally with J2, as relative Cowell integration.

    The target follows its own orbit and the chaser is carried as an inertial offset
    (dr, dv) driven by the differential gravity g(r_t + dr) - g(r_t), so the small relative
    motion is integrated with its own tolerances instead of being lost in 7000 km positions.
    The target starts on a circular orbit at its ascending node. The Hill frame of the
    output uses the frame's exact rotation, including the out-of-plane rate J2 causes.
    """

    def __init__(
        self,
        body: str | Body = "earth",
        altitude: float = 500e3,
        inclination: float = 0.0,
        j2: bool = False,
    ):
        self.body = get_body(body)
        self.altitude = altitude
        self.inclination = inclination
        self.j2 = j2
        self.name = "twobody_j2" if j2 else "twobody"
        self.n = mean_motion(self.body.radius + altitude, self.body.mu)
        self._y = np.zeros(12)
        self._sol = None
        self._step: float | None = None

    def reset(self, rel_state: ArrayLike) -> None:
        r_t, v_t = circular_orbit(self.body, self.altitude, self.inclination)
        dr, dv = hill_to_inertial(r_t, v_t, rel_state, self._gravity(r_t))
        self._y = np.concatenate([r_t, v_t, dr, dv])
        self._sol = None
        self._step = None

    def _gravity(self, r: np.ndarray) -> np.ndarray:
        return gravity(r, self.body, self.j2)

    def _rhs(self, t: float, y: np.ndarray, accel_hill: np.ndarray) -> np.ndarray:
        r_t, v_t, dr, dv = y[0:3], y[3:6], y[6:9], y[9:12]
        C, _ = hill_frame(r_t, v_t)
        dg = differential_gravity(r_t, dr, self.body, self.j2)
        return np.concatenate([v_t, self._gravity(r_t), dv, dg + C @ accel_hill])

    def _to_hill(self, y: np.ndarray) -> np.ndarray:
        r_t = y[..., 0:3]
        return inertial_to_hill(r_t, y[..., 3:6], y[..., 6:9], y[..., 9:12], self._gravity(r_t))

    def advance(
        self, accel_hill: ArrayLike, duration: float, n_samples: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Hold `accel_hill` (m/s^2) for `duration` s. Same returns as `CWDynamics.advance`."""
        accel = np.array(accel_hill, dtype=float)
        # Restarting every decision step from scipy's initial step guess costs about five times
        # the evaluations. Start from the largest step of the last segment instead, allowed to
        # grow by the step-size controller's limit of 10: a segment's final step is cut short to
        # land on its end, so on its own it would pin later segments to two steps.
        span = abs(duration)
        first_step = None if self._step is None or span == 0.0 else min(10.0 * self._step, span)
        sol = solve_ivp(
            self._rhs,
            (0.0, duration),
            self._y,
            method="DOP853",
            args=(accel,),
            rtol=_RTOL,
            atol=_ATOL,
            dense_output=True,
            first_step=first_step,
        )
        if not sol.success:
            raise RuntimeError(f"two-body integration failed: {sol.message}")
        self._sol = sol
        self._y = sol.y[:, -1]
        if span > 0.0:
            self._step = float(np.max(np.abs(np.diff(sol.t))))
        samples = None
        if n_samples > 0:
            samples = self._to_hill(sol.sol(np.linspace(0.0, duration, n_samples)).T)
        return self._to_hill(self._y), samples

    def state_in_last_segment(self, tau: float) -> np.ndarray:
        """State `tau` seconds into the most recent segment."""
        if self._sol is None:
            raise RuntimeError("no segment has been advanced since reset")
        return self._to_hill(self._sol.sol(tau))

    def target_state(self) -> np.ndarray:
        """Target inertial [r, v] in m and m/s."""
        return self._y[0:6].copy()

    def chaser_state(self) -> np.ndarray:
        """Chaser inertial [r, v] in m and m/s."""
        return self._y[0:6] + self._y[6:12]


def make_dynamics(
    kind: str, body: str | Body = "earth", altitude: float = 500e3, inclination: float = 0.0
) -> Dynamics:
    """Build a model by name: "cw", "twobody" or "twobody_j2"."""
    b = get_body(body)
    if kind == "cw":
        return CWDynamics(mean_motion(b.radius + altitude, b.mu))
    if kind in ("twobody", "twobody_j2"):
        return TwoBodyDynamics(b, altitude, inclination, j2=kind == "twobody_j2")
    raise ValueError(f"unknown dynamics {kind!r}, expected cw, twobody or twobody_j2")
