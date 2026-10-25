"""Relative-motion models behind one interface, so a scenario can change fidelity.

Every model holds the chaser's state relative to the target in the target's Hill frame
([x, y, z, vx, vy, vz], see `vbar.conventions`) and advances it under an acceleration held
constant in the Hill frame over a segment.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from ..conventions import hill_frame, hill_to_inertial, inertial_to_hill
from .bodies import Body, get_body
from .cw import CWDiscretizer, cw_system, mean_motion
from .kepler import state_from_elements
from .mean_elements import mean_to_osculating


class Dynamics(Protocol):
    name: str
    n: float  # rad/s, mean motion of the target's reference orbit

    def reset(self, rel_state: ArrayLike) -> None: ...

    def advance(
        self, accel_hill: ArrayLike, duration: float, n_samples: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None]: ...

    def state_in_last_segment(self, tau: float) -> np.ndarray: ...


class Perturbation(Protocol):
    """A force beyond gravity acting on both spacecraft, such as drag or solar pressure."""

    name: str

    def accelerations(
        self, r: np.ndarray, v: np.ndarray, dr: np.ndarray, dv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inertial accelerations (m/s^2) of the target and of the chaser minus the target's.

        The target is at (r, v) and the chaser at (r + dr, v + dv); inputs may be stacked as
        (..., 3).
        """
        ...


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


# A duration or time this far (s) from a whole number of steps still counts as one; segment
# times built by summing steps land a rounding step either side.
_NODE_TOL = 1e-9


class EulerCWDynamics:
    """Clohessy-Wiltshire equations stepped by forward Euler, a plant defined only at its nodes.

    Each step is x + h (A x + B a) with A, B from `cw_system(n)`, the discrete model of the run
    time assurance docking benchmark (Dunlap et al., 2021). The plant has no state between
    nodes, so every duration must be a whole number of steps.

    Parameters
    ----------
    n : float
        Mean motion of the target's circular orbit (rad/s).
    step_s : float
        Euler step h (s), positive.

    Raises
    ------
    ValueError
        If `step_s` is not positive and finite.
    """

    name = "euler1"
    step_s = 1.0

    def __init__(self, n: float, step_s: float = 1.0):
        if not (np.isfinite(step_s) and step_s > 0.0):
            raise ValueError("step_s must be positive and finite")
        self.n = n
        self.step_s = step_s
        self._a, self._b = cw_system(n)
        self._state = np.zeros(6)
        self._nodes: np.ndarray | None = None

    def reset(self, rel_state: ArrayLike) -> None:
        self._state = np.array(rel_state, dtype=float)
        self._nodes = None

    def _steps(self, duration: float) -> int:
        k = round(duration / self.step_s)
        if k < 0 or abs(duration - k * self.step_s) > _NODE_TOL:
            raise ValueError(f"duration {duration!r} s is not a whole number of steps")
        return k

    def advance(
        self, accel_hill: ArrayLike, duration: float, n_samples: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Hold `accel_hill` (m/s^2, Hill frame) for `duration` s of Euler steps.

        Parameters
        ----------
        accel_hill : array_like, shape (3,)
            Acceleration held over every step (m/s^2, Hill frame).
        duration : float
            Whole number of steps times `step_s` (s).
        n_samples : int
            When positive, the nodes are returned; their count is the number of steps plus
            one whatever `n_samples` asks for, since the plant has no state between them.

        Returns
        -------
        state : ndarray, shape (6,)
            Hill state after the last step (m, m/s).
        nodes : ndarray, shape (k + 1, 6) or None
            The state at every node, the start included, or None when `n_samples` is 0.

        Raises
        ------
        ValueError
            If `duration` is not a whole number of steps.
        """
        accel = np.array(accel_hill, dtype=float)
        k = self._steps(duration)
        h = self.step_s
        nodes = np.empty((k + 1, 6))
        nodes[0] = x = self._state
        for j in range(1, k + 1):
            x = x + h * (self._a @ x + self._b @ accel)
            nodes[j] = x
        self._nodes = nodes
        self._state = nodes[-1].copy()
        return self._state.copy(), (nodes.copy() if n_samples > 0 else None)

    def state_in_last_segment(self, tau: float) -> np.ndarray:
        """State at the node `tau` s into the most recent segment.

        Raises
        ------
        RuntimeError
            If no segment has been advanced since reset.
        ValueError
            If `tau` is not a node of that segment.
        """
        if self._nodes is None:
            raise RuntimeError("no segment has been advanced since reset")
        k = round(tau / self.step_s)
        if not 0 <= k < len(self._nodes) or abs(tau - k * self.step_s) > _NODE_TOL:
            raise ValueError(f"an Euler plant has no state at {tau!r} s, only at its nodes")
        return self._nodes[k].copy()


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


def point_mass_difference(r: ArrayLike, dr: ArrayLike, mu: float) -> np.ndarray:
    """Point-mass gravity at `r + dr` minus at `r`, accurate to rounding for any `dr`.

    Subtracting two 8 m/s^2 accelerations directly leaves 1e-15 m/s^2 of rounding, a 1e-9
    relative error at 1 m of separation. Battin's f(q) from Encke's method avoids it (Battin,
    An Introduction to the Mathematics and Methods of Astrodynamics). Any consistent units.
    """
    r = np.asarray(r, dtype=float)
    dr = np.asarray(dr, dtype=float)
    r_c = r + dr
    q = (_dot(dr, dr) + 2.0 * _dot(dr, r)) / _dot(r, r)
    # f(q) = (|r + dr| / |r|)^3 - 1, written without subtracting nearly equal numbers.
    f = q * (3.0 + 3.0 * q + q * q) / (1.0 + (1.0 + q) ** 1.5)
    r_c2 = _dot(r_c, r_c)
    return -mu / (r_c2 * np.sqrt(r_c2)) * (dr - f * r)


def differential_gravity(r: ArrayLike, dr: ArrayLike, body: Body, j2: bool = False) -> np.ndarray:
    """Gravity at `r + dr` minus gravity at `r` (m/s^2), accurate to rounding for any `dr`.

    The point-mass part is `point_mass_difference`; the J2 part is a direct difference, whose
    rounding is J2 times smaller.
    """
    dg = point_mass_difference(r, dr, body.mu)
    if j2 and body.j2:
        r = np.asarray(r, dtype=float)
        r_c = r + np.asarray(dr, dtype=float)
        dg = dg + _j2_acceleration(r_c, body) - _j2_acceleration(r, body)
    return dg


def gravity_gradient(r: ArrayLike, body: Body, j2: bool = False) -> np.ndarray:
    """Jacobian of `gravity` with respect to position (1/s^2), shape (..., 3, 3)."""
    r = np.asarray(r, dtype=float)
    r2 = np.sum(r * r, axis=-1)[..., None, None]
    rr = r[..., :, None] * r[..., None, :]
    eye = np.eye(3)
    grad = -body.mu / (r2 * np.sqrt(r2)) * (eye - 3.0 * rr / r2)
    if j2 and body.j2:
        # Derivative of 1.5 J2 mu R^2 (5 r_i z^2 / r^7 - c_i r_i / r^5), c = (1, 1, 3).
        r5 = r2 * r2 * np.sqrt(r2)
        z = r[..., 2][..., None, None]
        c = np.array([1.0, 1.0, 3.0])
        z_part = (
            z * z / (r5 * r2) * eye
            + 2.0 * z / (r5 * r2) * r[..., :, None] * np.array([0.0, 0.0, 1.0])
            - 7.0 * z * z / (r5 * r2 * r2) * rr
        )
        c_part = -np.diag(c) / r5 + 5.0 * c[:, None] * rr / (r5 * r2)
        grad = grad + 1.5 * body.j2 * body.mu * body.radius**2 * (5.0 * z_part + c_part)
    return grad


def _hill_rate_derivative(
    r: np.ndarray, v: np.ndarray, a: np.ndarray, a_dot: np.ndarray
) -> np.ndarray:
    """Time derivative of the `hill_frame` angular velocity along the orbit (r, v, a, a_dot)."""
    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec, axis=-1)
    z = h_vec / h[..., None]
    r_norm = np.linalg.norm(r, axis=-1)
    r_norm_dot = np.sum(r * v, axis=-1) / r_norm
    h_vec_dot = np.cross(r, a)
    h_dot = np.sum(z * h_vec_dot, axis=-1)
    z_dot = (h_vec_dot - h_dot[..., None] * z) / h[..., None]
    a_z = np.sum(a * z, axis=-1)
    # omega = (|r| (a . z) / h, 0, h / |r|^2) in Hill axes; differentiate each factor.
    w_x_dot = (
        r_norm_dot * a_z + r_norm * (np.sum(a_dot * z, axis=-1) + np.sum(a * z_dot, axis=-1))
    ) / h - r_norm * a_z * h_dot / h**2
    w_z_dot = h_dot / r_norm**2 - 2.0 * h * r_norm_dot / r_norm**3
    return np.stack([w_x_dot, np.zeros_like(w_x_dot), w_z_dot], axis=-1)


def relative_acceleration(
    rel_state: ArrayLike, target_state: ArrayLike, body: Body, j2: bool = False
) -> np.ndarray:
    """Hill-frame acceleration (m/s^2) of a coasting chaser relative to its target.

    Parameters
    ----------
    rel_state : array_like, shape (..., 6)
        Chaser state [x, y, z, vx, vy, vz] in the target's Hill frame (m, m/s).
    target_state : array_like, shape (..., 6)
        Target inertial position and velocity (m, m/s), body-centred with z along the spin
        axis; broadcasts against `rel_state`.
    body : Body
        Central body.
    j2 : bool
        Add the body's J2 to the gravity of both spacecraft.

    Returns
    -------
    numpy.ndarray, shape (..., 3)
        Second time derivative of the Hill position with no thrust: the gravity difference
        rotated into the Hill axes, minus the Coriolis, centripetal and Euler terms of the
        rotating frame. It is the acceleration `TwoBodyDynamics` integrates between burns
        and differs from the Clohessy-Wiltshire terms by the second-order gravity remainder
        and, with J2, by the frame's out-of-plane rate and its derivative.

    References
    ----------
    Schaub and Junkins, Analytical Mechanics of Space Systems, eq 13.12 (the exact relative
    equations of motion in the rotating frame).
    """
    rel = np.asarray(rel_state, dtype=float)
    target = np.asarray(target_state, dtype=float)
    r_t, v_t = target[..., :3], target[..., 3:]
    a_t = gravity(r_t, body, j2)
    a_t_dot = np.einsum("...ij,...j->...i", gravity_gradient(r_t, body, j2), v_t)
    C, omega = hill_frame(r_t, v_t, a_t)
    omega_dot = _hill_rate_derivative(r_t, v_t, a_t, a_t_dot)
    rho, rho_dot = rel[..., :3], rel[..., 3:]
    da = differential_gravity(r_t, np.einsum("...ij,...j->...i", C, rho), body, j2)
    return (
        np.einsum("...ji,...j->...i", C, da)
        - 2.0 * np.cross(omega, rho_dot)
        - np.cross(omega, np.cross(omega, rho))
        - np.cross(omega_dot, rho)
    )


def circular_orbit(body: Body, altitude: float, inclination: float = 0.0):
    """Inertial position and velocity at the ascending node (on +x) of a circular orbit.

    `altitude` in m above the equatorial radius, `inclination` in rad.
    """
    a = body.radius + altitude
    speed = np.sqrt(body.mu / a)
    r = np.array([a, 0.0, 0.0])
    v = speed * np.array([0.0, np.cos(inclination), np.sin(inclination)])
    return r, v


def integrate_segment(
    rhs: Callable[..., np.ndarray],
    y: np.ndarray,
    duration: float,
    args: tuple,
    rtol: float,
    atol: np.ndarray,
    step: float | None,
) -> tuple[Any, float | None]:
    """One DOP853 segment with dense output, started from the previous segment's step.

    Returns the solution and the step to start the next segment with.
    """
    # Restarting every decision step from scipy's initial step guess costs about five times
    # the evaluations. Start from the largest step of the last segment instead, allowed to
    # grow by the step-size controller's limit of 10: a segment's final step is cut short to
    # land on its end, so on its own it would pin later segments to two steps.
    span = abs(duration)
    first_step = None if step is None or span == 0.0 else min(10.0 * step, span)
    sol = solve_ivp(
        rhs,
        (0.0, duration),
        y,
        method="DOP853",
        args=args,
        rtol=rtol,
        atol=atol,
        dense_output=True,
        first_step=first_step,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    if span > 0.0:
        step = float(np.max(np.abs(np.diff(sol.t))))
    return sol, step


# Absolute tolerances for the integrated blocks [r_target, v_target, dr, dv] in m and m/s.
# The offset blocks get their own small tolerances so relative motion is resolved well below
# a millimetre whatever the orbit size.
_ATOL = np.repeat([1e-6, 1e-9, 1e-9, 1e-12], 3)
# Steps for central differences of perturbing forces (m, m/s); drag and solar pressure vary
# over tens of kilometres and kilometres per second, so the truncation error is below 1e-9.
_FD_STEP = np.repeat([1.0, 1e-3], 3)


class TwoBodyDynamics:
    """Nonlinear two-body relative motion, optionally with J2, as relative Cowell integration.

    The target follows its own orbit and the chaser is carried as an inertial offset
    (dr, dv) driven by the differential gravity g(r_t + dr) - g(r_t), so the small relative
    motion is integrated with its own tolerances instead of being lost in 7000 km positions.
    The target starts at the ascending node (on +x) of a circular orbit, or from classical
    `elements` [a, e, i, raan, argp, M] (m, rad) when they are given, which then replace
    `altitude` and `inclination`. With J2 the orbit is circular in first-order mean elements,
    or `elements` are mean elements, so the target's osculating state carries J2's
    short-period terms rather than a spurious eccentricity. `n` is the Keplerian mean motion
    of the mean semi-major axis either way; under J2 the target's mean argument of latitude
    advances faster, by (3/2) J2 (R/a)^2 (4 cos^2 i - 1) n. The Hill frame of the output
    uses the frame's exact rotation, including the out-of-plane rate J2 causes.
    `rtol` is the integrator's relative tolerance; at the default the relative state is
    within about 1e-9 of its own size after one orbit. `perturbations` add forces such as drag
    and solar pressure (`vbar.dynamics.perturbations`); the target follows its own and the
    chaser's offset feels the difference.
    """

    def __init__(
        self,
        body: str | Body = "earth",
        altitude: float = 500e3,
        inclination: float = 0.0,
        j2: bool = False,
        rtol: float = 1e-11,
        perturbations: Sequence[Perturbation] = (),
        elements: ArrayLike | None = None,
    ):
        self.body = get_body(body)
        self.elements = None if elements is None else np.array(elements, dtype=float)
        if self.elements is not None:
            altitude = float(self.elements[0]) - self.body.radius
            inclination = float(self.elements[2])
        self.altitude = altitude
        self.inclination = inclination
        self.j2 = j2
        self.rtol = rtol
        self.perturbations = tuple(perturbations)
        self.name = ("twobody_j2" if j2 else "twobody") + "".join(
            "_" + p.name for p in self.perturbations
        )
        self.n = mean_motion(self.body.radius + altitude, self.body.mu)
        self._y = np.zeros(12)
        self._sol = None
        self._step: float | None = None

    def _target_start(self) -> tuple[np.ndarray, np.ndarray]:
        if self.elements is None and not self.j2:
            return circular_orbit(self.body, self.altitude, self.inclination)
        elements = self.elements
        if elements is None:
            elements = np.array([self.body.radius + self.altitude, 0, self.inclination, 0, 0, 0])
        if self.j2:
            elements = mean_to_osculating(elements, self.body)
        return state_from_elements(elements, self.body.mu)

    def reset(self, rel_state: ArrayLike) -> None:
        r_t, v_t = self._target_start()
        accel = self._target_acceleration(np.concatenate([r_t, v_t, np.zeros(6)]))
        dr, dv = hill_to_inertial(r_t, v_t, rel_state, accel)
        self._y = np.concatenate([r_t, v_t, dr, dv])
        self._sol = None
        self._step = None

    def reset_inertial(self, target: ArrayLike, chaser: ArrayLike) -> None:
        """Start from given inertial states instead of the configured orbit.

        `n`, `altitude` and `inclination` keep the configured orbit's values.

        Parameters
        ----------
        target, chaser : array_like
            Inertial [r, v] (m, m/s), shape (6,), in the body-centred frame with z along the
            spin axis that the gravity and perturbation models use.

        Raises
        ------
        ValueError
            If a state does not have six finite entries or the target is at the body's centre.
        """
        states = [np.array(s, dtype=float) for s in (target, chaser)]
        if any(s.shape != (6,) or not np.all(np.isfinite(s)) for s in states):
            raise ValueError("target and chaser must be six finite numbers each")
        if not np.linalg.norm(states[0][:3]) > 0.0:
            raise ValueError("the target must not be at the body's centre")
        self._y = np.concatenate([states[0], states[1] - states[0]])
        self._sol = None
        self._step = None

    def _gravity(self, r: np.ndarray) -> np.ndarray:
        return gravity(r, self.body, self.j2)

    def _target_acceleration(self, y: np.ndarray) -> np.ndarray:
        r_t, v_t = y[..., 0:3], y[..., 3:6]
        accel = self._gravity(r_t)
        for p in self.perturbations:
            accel = accel + p.accelerations(r_t, v_t, y[..., 6:9], y[..., 9:12])[0]
        return accel

    def _rhs(self, t: float, y: np.ndarray, accel_hill: np.ndarray) -> np.ndarray:
        r_t, v_t, dr, dv = y[0:3], y[3:6], y[6:9], y[9:12]
        a_t = self._gravity(r_t)
        da = differential_gravity(r_t, dr, self.body, self.j2)
        # The Hill axes cost as much as the gravity; coasting segments do not need them.
        if accel_hill.any():
            da = da + hill_frame(r_t, v_t)[0] @ accel_hill
        for p in self.perturbations:
            target, relative = p.accelerations(r_t, v_t, dr, dv)
            a_t = a_t + target
            da = da + relative
        return np.concatenate([v_t, a_t, dv, da])

    def _to_hill(self, y: np.ndarray) -> np.ndarray:
        return inertial_to_hill(
            y[..., 0:3], y[..., 3:6], y[..., 6:9], y[..., 9:12], self._target_acceleration(y)
        )

    def advance(
        self, accel_hill: ArrayLike, duration: float, n_samples: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Hold `accel_hill` (m/s^2) for `duration` s. Same returns as `CWDynamics.advance`."""
        accel = np.array(accel_hill, dtype=float)
        sol, self._step = integrate_segment(
            self._rhs, self._y, duration, (accel,), self.rtol, _ATOL, self._step
        )
        self._sol = sol
        self._y = sol.y[:, -1]
        samples = None
        if n_samples > 0:
            samples = self._to_hill(sol.sol(np.linspace(0.0, duration, n_samples)).T)
        return self._to_hill(self._y), samples

    def state_in_last_segment(self, tau: float) -> np.ndarray:
        """State `tau` seconds into the most recent segment."""
        if self._sol is None:
            raise RuntimeError("no segment has been advanced since reset")
        return self._to_hill(self._sol.sol(tau))

    def transition_matrices(
        self, accel_hill: ArrayLike, duration: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hill-frame transition and input matrices of the segment `advance` would fly next.

        phi = d x(t + duration) / d x(t) and gamma = d x(t + duration) / d accel_hill for the
        relative Hill state x, from the variational equations of this same model integrated
        along the chaser's own trajectory (Montenbruck and Gill, Satellite Orbits, sec. 7.2).
        Gravity's gradient is analytic, the perturbations' come from central differences. The
        model's state is not changed.
        """
        accel = np.array(accel_hill, dtype=float)
        n = self.n
        atol_phi = 1e-12 * np.outer([1, 1, 1, n, n, n], [1, 1, 1, 1 / n, 1 / n, 1 / n])
        atol_gamma = 1e-12 * np.outer([1 / n**2] * 3 + [1 / n] * 3, [1, 1, 1])
        sol = solve_ivp(
            self._variational_rhs,
            (0.0, duration),
            np.concatenate([self._y, np.eye(6).ravel(), np.zeros(18)]),
            method="DOP853",
            args=(accel,),
            rtol=self.rtol,
            atol=np.concatenate([_ATOL, atol_phi.ravel(), atol_gamma.ravel()]),
        )
        if not sol.success:
            raise RuntimeError(f"variational integration failed: {sol.message}")
        end = sol.y[:, -1]
        to_hill_0, from_hill_0 = self._hill_maps(self._y)
        to_hill_1, _ = self._hill_maps(end[:12])
        phi = to_hill_1 @ end[12:48].reshape(6, 6) @ from_hill_0
        return phi, to_hill_1 @ end[48:66].reshape(6, 3)

    def _relative_jacobian(self, y: np.ndarray) -> np.ndarray:
        """d(dv/dt) / d(dr, dv) for the chaser's offset at state `y`, shape (3, 6)."""
        r_t, v_t, dr, dv = y[0:3], y[3:6], y[6:9], y[9:12]
        jac = np.zeros((3, 6))
        jac[:, :3] = gravity_gradient(r_t + dr, self.body, self.j2)
        if self.perturbations:
            # All twelve shifted offsets in one stacked evaluation per force.
            steps = np.diag(_FD_STEP)
            offsets = np.concatenate([dr, dv]) + np.concatenate([steps, -steps])
            for p in self.perturbations:
                shifted = p.accelerations(r_t, v_t, offsets[:, :3], offsets[:, 3:])[1]
                jac += ((shifted[:6] - shifted[6:]) / (2.0 * _FD_STEP[:, None])).T
        return jac

    def _variational_rhs(self, t: float, y: np.ndarray, accel_hill: np.ndarray) -> np.ndarray:
        a = np.zeros((6, 6))
        a[0:3, 3:6] = np.eye(3)
        a[3:6, :] = self._relative_jacobian(y[:12])
        b = np.zeros((6, 3))
        b[3:6, :], _ = hill_frame(y[0:3], y[3:6])
        phi = y[12:48].reshape(6, 6)
        gamma = y[48:66].reshape(6, 3)
        return np.concatenate(
            [self._rhs(t, y[:12], accel_hill), (a @ phi).ravel(), (a @ gamma + b).ravel()]
        )

    def _hill_maps(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linear maps from the inertial offset (dr, dv) to the Hill state and back."""
        C, omega = hill_frame(y[0:3], y[3:6], self._target_acceleration(y))
        w = np.array(
            [[0.0, -omega[2], omega[1]], [omega[2], 0.0, -omega[0]], [-omega[1], omega[0], 0.0]]
        )
        to_hill = np.zeros((6, 6))
        to_hill[:3, :3] = to_hill[3:, 3:] = C.T
        to_hill[3:, :3] = -w @ C.T
        from_hill = np.zeros((6, 6))
        from_hill[:3, :3] = from_hill[3:, 3:] = C
        from_hill[3:, :3] = C @ w
        return to_hill, from_hill

    def target_state(self) -> np.ndarray:
        """Target inertial [r, v] in m and m/s."""
        return self._y[0:6].copy()

    def chaser_state(self) -> np.ndarray:
        """Chaser inertial [r, v] in m and m/s."""
        return self._y[0:6] + self._y[6:12]


def make_dynamics(
    kind: str, body: str | Body = "earth", altitude: float = 500e3, inclination: float = 0.0
) -> Dynamics:
    """Build a model by name: "cw", "euler1", "twobody" or "twobody_j2".

    "euler1" is `EulerCWDynamics` with 1 s steps at the Keplerian mean motion of `altitude`.
    """
    b = get_body(body)
    if kind == "cw":
        return CWDynamics(mean_motion(b.radius + altitude, b.mu))
    if kind == "euler1":
        return EulerCWDynamics(mean_motion(b.radius + altitude, b.mu), step_s=1.0)
    if kind in ("twobody", "twobody_j2"):
        return TwoBodyDynamics(b, altitude, inclination, j2=kind == "twobody_j2")
    raise ValueError(f"unknown dynamics {kind!r}, expected cw, euler1, twobody or twobody_j2")
