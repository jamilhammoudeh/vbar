"""Relative-motion models between CW and the nonlinear two-body model.

All hold the chaser's state relative to the target in the target's Hill frame
([x, y, z, vx, vy, vz], see `vbar.conventions`) and satisfy the `Dynamics` protocol, with an
acceleration held constant in Hill axes over each segment. Their errors against the nonlinear
model scale differently with the separation rho: two-body nonlinearity left out by a linear model
grows as rho^2, J2 left out as rho, the target's eccentricity left out as e rho (Alfriend et al.,
Spacecraft Formation Flying, sec. 8.3).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from ..conventions import hill_frame
from .bodies import Body, get_body
from .cw import CWDiscretizer, mean_motion
from .models import TwoBodyDynamics, gravity_gradient

# Hill in-plane [x, y, vx, vy] to Yamanaka and Ankersen's [x, z, vx, vz], whose x is along-track
# and z points at the central body (the ISS-style LVLH axes of `vbar.conventions`).
_HILL_TO_YA = np.array(
    [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0]]
)
_IN_PLANE = [0, 1, 3, 4]
# Gauss-Legendre rule for the forced response, on pieces of at most 1/32 orbit, shortened
# further on eccentric orbits by the peak rate of the true anomaly over its mean.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(12)


def true_anomaly(M: ArrayLike, e: float) -> np.ndarray:
    """True anomaly (rad) at mean anomaly `M` (rad, any shape) on an orbit of eccentricity `e`."""
    M = np.asarray(M, dtype=float)
    M_rev = M % math.tau
    # As in `vbar.dynamics.kepler.solve_kepler`, vectorized.
    E = np.full_like(M_rev, math.pi) if e >= 0.8 else M_rev
    for _ in range(50):
        dE = (E - e * np.sin(E) - M_rev) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.max(np.abs(dE), initial=0.0) < 1e-12:
            break
    else:
        raise RuntimeError("kepler's equation did not converge")
    # f - E without wrapping, so f stays in E's revolution.
    f = E + 2.0 * np.arctan2(e * np.sin(E), 1.0 + np.sqrt(1.0 - e * e) - e * np.cos(E))
    return f + (M - M_rev)


def ya_in_plane(f: ArrayLike, e: float, J: ArrayLike) -> np.ndarray:
    """Fundamental matrix of the in-plane Tschauner-Hempel equations, (..., 4, 4).

    Yamanaka and Ankersen, "New State Transition Matrix for Relative Motion on an Arbitrary
    Elliptical Orbit", JGCD 25(1), 2002, in the paper's frame and order: [x, z, x', z'] with x
    along-track, z toward the central body, each coordinate scaled by rho = 1 + e cos f and
    differentiated with respect to the true anomaly f. J = k^2 (t - t_ref) with
    k^2 = mu^2 / h^3, for any reference time.
    """
    f = np.asarray(f, dtype=float)
    J = np.broadcast_to(np.asarray(J, dtype=float), f.shape)
    rho = 1.0 + e * np.cos(f)
    s, c = rho * np.sin(f), rho * np.cos(f)
    ds = np.cos(f) + e * np.cos(2.0 * f)
    dc = -(np.sin(f) + e * np.sin(2.0 * f))
    o, i = np.zeros_like(f), np.ones_like(f)
    psi = np.array(
        [
            [i, -c * (1.0 + 1.0 / rho), s * (1.0 + 1.0 / rho), 3.0 * rho**2 * J],
            [o, s, c, 2.0 - 3.0 * e * s * J],
            [o, 2.0 * s, 2.0 * c - e, 3.0 * (1.0 - 2.0 * e * s * J)],
            [o, ds, dc, -3.0 * e * (ds * J + s / rho**2)],
        ]
    )
    return np.moveaxis(psi, (0, 1), (-2, -1))


def ya_in_plane_inverse(f: ArrayLike, e: float, J: ArrayLike) -> np.ndarray:
    """Inverse of `ya_in_plane`, (..., 4, 4), in closed form.

    At J = 0 it is the paper's pseudo-initial-value matrix. A shift of J adds J (3, -3e, 0, 0)
    in the first three columns' coordinates to the fourth column, a nilpotent update whose
    inverse is exact.
    """
    f = np.asarray(f, dtype=float)
    J = np.broadcast_to(np.asarray(J, dtype=float), f.shape)
    rho = 1.0 + e * np.cos(f)
    s, c = rho * np.sin(f), rho * np.cos(f)
    o = np.zeros_like(f)
    start = np.array(
        [
            [
                1.0 - e * e + o,
                3.0 * e * s * (1.0 / rho + 1.0 / rho**2),
                -e * s * (1.0 + 1.0 / rho),
                2.0 - e * c,
            ],
            [o, -3.0 * s * (1.0 / rho + e * e / rho**2), s * (1.0 + 1.0 / rho), c - 2.0 * e],
            [o, -3.0 * (c / rho + e), c * (1.0 + 1.0 / rho) + e, -s],
            [o, 3.0 * rho + e * e - 1.0, -(rho**2), e * s],
        ]
    ) / (1.0 - e * e)
    start = np.moveaxis(start, (0, 1), (-2, -1))
    shifted = start.copy()
    shifted[..., 0, :] -= 3.0 * J[..., None] * start[..., 3, :]
    shifted[..., 1, :] += 3.0 * e * J[..., None] * start[..., 3, :]
    return shifted


class YamanakaAnkersenDynamics:
    """Linear relative motion about an elliptic Keplerian target (Yamanaka and Ankersen 2002).

    Exact for the linearized two-body problem at any eccentricity below 1; at e = 0 it is CW.
    The free motion uses the closed-form Yamanaka-Ankersen solution; the response to an
    acceleration held in Hill axes, which has no closed form, is integrated by Gauss-Legendre
    quadrature to rounding. The target's orbit has semi-major axis `semi_major_axis` (m),
    eccentricity `eccentricity`, and mean anomaly `mean_anomaly` (rad) at reset; its
    orientation does not enter. `n` is its mean motion. Left out against the nonlinear model:
    two-body nonlinearity, about rho^2 / a per orbit.
    """

    name = "ya"

    def __init__(
        self,
        semi_major_axis: float,
        eccentricity: float,
        mean_anomaly: float = 0.0,
        body: str | Body = "earth",
    ):
        self.body = get_body(body)
        self.a = semi_major_axis
        self.e = eccentricity
        self.mean_anomaly = mean_anomaly
        self.n = mean_motion(semi_major_axis, self.body.mu)
        h = math.sqrt(self.body.mu * semi_major_axis * (1.0 - eccentricity**2))
        self.k2 = self.body.mu**2 / h**3
        self._t = 0.0
        self._state = np.zeros(6)
        self._segment: tuple[float, np.ndarray, np.ndarray] | None = None

    def reset(self, rel_state: ArrayLike) -> None:
        self._t = 0.0
        self._state = np.array(rel_state, dtype=float)
        self._segment = None

    def _anomaly(self, t: ArrayLike) -> np.ndarray:
        return true_anomaly(self.mean_anomaly + self.n * np.asarray(t, dtype=float), self.e)

    def _scaling(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Maps from Hill states to scaled states and back, (..., 6, 6) each."""
        rho = 1.0 + self.e * np.cos(f)
        e_sin = self.e * np.sin(f)
        eye = np.eye(3)
        to_scaled = np.zeros(f.shape + (6, 6))
        to_scaled[..., :3, :3] = rho[..., None, None] * eye
        to_scaled[..., 3:, :3] = -e_sin[..., None, None] * eye
        to_scaled[..., 3:, 3:] = (1.0 / (self.k2 * rho))[..., None, None] * eye
        to_hill = np.zeros(f.shape + (6, 6))
        to_hill[..., :3, :3] = (1.0 / rho)[..., None, None] * eye
        to_hill[..., 3:, :3] = (self.k2 * e_sin)[..., None, None] * eye
        to_hill[..., 3:, 3:] = (self.k2 * rho)[..., None, None] * eye
        return to_scaled, to_hill

    def _fundamental(self, f: np.ndarray, J: np.ndarray) -> np.ndarray:
        """Fundamental matrix of the scaled equations in Hill order, (..., 6, 6)."""
        psi = np.zeros(f.shape + (6, 6))
        psi[(..., *np.ix_(_IN_PLANE, _IN_PLANE))] = _HILL_TO_YA.T @ ya_in_plane(f, self.e, J)
        cos_f, sin_f = np.cos(f), np.sin(f)
        psi[..., 2, 2], psi[..., 2, 5] = cos_f, sin_f
        psi[..., 5, 2], psi[..., 5, 5] = -sin_f, cos_f
        return psi

    def _fundamental_inverse(self, f: np.ndarray, J: np.ndarray) -> np.ndarray:
        inverse = np.zeros(f.shape + (6, 6))
        inverse[(..., *np.ix_(_IN_PLANE, _IN_PLANE))] = (
            ya_in_plane_inverse(f, self.e, J) @ _HILL_TO_YA
        )
        cos_f, sin_f = np.cos(f), np.sin(f)
        inverse[..., 2, 2], inverse[..., 2, 5] = cos_f, -sin_f
        inverse[..., 5, 2], inverse[..., 5, 5] = sin_f, cos_f
        return inverse

    def matrices(self, t0: float, durations: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Transition and input matrices from `t0` (s since reset) over each of `durations`.

        Returns phi (..., 6, 6) and gamma (..., 6, 3) with state(t0 + d) = phi state(t0) +
        gamma accel for an acceleration held in Hill axes.
        """
        d = np.asarray(durations, dtype=float)
        f0 = self._anomaly(t0)
        to_scaled0, _ = self._scaling(f0)
        start = self._fundamental_inverse(f0, np.zeros(())) @ to_scaled0
        f = self._anomaly(t0 + d)
        _, to_hill = self._scaling(f)
        carry = to_hill @ self._fundamental(f, self.k2 * d)
        phi = carry @ start
        gamma = carry @ self._forced_integral(t0, d)
        return phi, gamma

    def _forced_integral(self, t0: float, d: np.ndarray) -> np.ndarray:
        """Integral over [t0, t0 + d] of the inverse fundamental matrix times the input map.

        Evaluated between consecutive sorted durations and summed, so many sample times share
        one pass of the quadrature.
        """
        flat = d.ravel()
        order = np.argsort(flat)
        edges = np.concatenate([[0.0], flat[order]])
        per_orbit = 32.0 * (1.0 + self.e) ** 2 / (1.0 - self.e**2) ** 1.5
        pieces = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            count = max(1, math.ceil(per_orbit * self.n * abs(hi - lo) / math.tau))
            bounds = np.linspace(lo, hi, count + 1)
            half = 0.5 * np.diff(bounds)
            tau = (0.5 * (bounds[:-1] + bounds[1:]))[:, None] + half[:, None] * _GL_NODES
            weights = (half[:, None] * _GL_WEIGHTS).ravel()
            tau = tau.ravel()
            f = self._anomaly(t0 + tau)
            rho = 1.0 + self.e * np.cos(f)
            inputs = np.zeros(tau.shape + (6, 3))
            inputs[:, 3:, :] = (1.0 / (self.k2 * rho))[:, None, None] * np.eye(3)
            integrand = self._fundamental_inverse(f, self.k2 * tau) @ inputs
            pieces.append(np.tensordot(weights, integrand, axes=1))
        cumulative = np.cumsum(np.array(pieces), axis=0)
        out = np.empty_like(cumulative)
        out[order] = cumulative
        return out.reshape(d.shape + (6, 3))

    def advance(
        self, accel_hill: ArrayLike, duration: float, n_samples: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Hold `accel_hill` (m/s^2) for `duration` s. Same returns as `CWDynamics.advance`."""
        accel = np.array(accel_hill, dtype=float)
        x0 = self._state
        self._segment = (self._t, x0, accel)
        samples = None
        if n_samples > 0:
            phis, gammas = self.matrices(self._t, np.linspace(0.0, duration, n_samples))
            samples = phis @ x0 + gammas @ accel
        phi, gamma = self.matrices(self._t, duration)
        self._state = phi @ x0 + gamma @ accel
        self._t += duration
        return self._state.copy(), samples

    def state_in_last_segment(self, tau: float) -> np.ndarray:
        """State `tau` seconds into the most recent segment."""
        if self._segment is None:
            raise RuntimeError("no segment has been advanced since reset")
        t0, x0, accel = self._segment
        phi, gamma = self.matrices(t0, tau)
        return phi @ x0 + gamma @ accel

    def make_bounded(self, rel_state: ArrayLike) -> np.ndarray:
        """`rel_state` with its along-track velocity changed so the motion repeats every orbit.

        The coefficient of the secular solution (the J column) must vanish; at e = 0 this is
        CW's vy = -2 n x. Applies at the model's current time.
        """
        state = np.array(rel_state, dtype=float)
        f0 = self._anomaly(self._t)
        to_scaled, _ = self._scaling(f0)
        secular = self._fundamental_inverse(f0, np.zeros(()))[_IN_PLANE[3]]
        weights = secular @ to_scaled
        state[4] -= (weights @ state) / weights[4]
        return state


def hill_to_curvilinear(rel_state: ArrayLike, radius: float) -> np.ndarray:
    """Curvilinear relative state about a circular target orbit of `radius` (m).

    [r - r0, r0 theta, r0 phi] and their rates: the chaser's radius minus the target's, and
    arc lengths at the target's radius along the orbit (angle theta from the target in its
    orbit plane) and across it (latitude phi above that plane). Inputs may be stacked (..., 6).
    """
    s = np.asarray(rel_state, dtype=float)
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    vx, vy, vz = s[..., 3], s[..., 4], s[..., 5]
    px = radius + x
    rho2 = px * px + y * y
    r = np.sqrt(rho2 + z * z)
    rho = np.sqrt(rho2)
    r_dot = (px * vx + y * vy + z * vz) / r
    theta_dot = (px * vy - y * vx) / rho2
    phi_dot = (vz * r - z * r_dot) / (r * rho)
    # r - r0 without cancelling two 7000 km numbers.
    dr = (2.0 * radius * x + x * x + y * y + z * z) / (r + radius)
    return np.stack(
        [
            dr,
            radius * np.arctan2(y, px),
            radius * np.arcsin(z / r),
            r_dot,
            radius * theta_dot,
            radius * phi_dot,
        ],
        axis=-1,
    )


def curvilinear_to_hill(curvilinear: ArrayLike, radius: float) -> np.ndarray:
    """Inverse of `hill_to_curvilinear`."""
    c = np.asarray(curvilinear, dtype=float)
    r = radius + c[..., 0]
    theta, phi = c[..., 1] / radius, c[..., 2] / radius
    r_dot, theta_dot, phi_dot = c[..., 3], c[..., 4] / radius, c[..., 5] / radius
    radial, along, normal = _spherical_axes(theta, phi)
    # cos(phi) cos(theta) - 1 without cancellation, so small offsets keep full precision.
    cp = np.cos(phi)
    x = c[..., 0] * radial[..., 0] - 2.0 * radius * (
        cp * np.sin(0.5 * theta) ** 2 + np.sin(0.5 * phi) ** 2
    )
    position = np.stack([x, r * radial[..., 1], r * radial[..., 2]], axis=-1)
    velocity = (
        r_dot[..., None] * radial
        + (r * np.cos(phi) * theta_dot)[..., None] * along
        + (r * phi_dot)[..., None] * normal
    )
    return np.concatenate([position, velocity], axis=-1)


def _spherical_axes(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, ...]:
    """Local radial, along-track and normal unit vectors in Hill axes, (..., 3) each."""
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    radial = np.stack([cp * ct, cp * st, sp], axis=-1)
    along = np.stack([-st, ct, np.zeros_like(ct)], axis=-1)
    normal = np.stack([-sp * ct, -sp * st, cp], axis=-1)
    return radial, along, normal


class CurvilinearCWDynamics:
    """CW motion in curvilinear coordinates about a circular target orbit.

    The CW equations hold to first order in the radius difference and the along-track and
    cross-track arc lengths as well as in Cartesian Hill coordinates (Alfriend et al.,
    Spacecraft Formation Flying, eqs 5.79 to 5.88), but curvilinear coordinates follow the
    orbit's curvature: a chaser on the target's circle is exactly stationary, and a V-bar
    offset D, which Cartesian CW gets wrong by 6 pi D^2 / a per orbit, is right to O(D^4).
    Other geometries keep second-order errors of their own; a radial offset X started with
    CW's bounded velocity drifts -12 pi X^2 / a per orbit here, against two-body's 3 pi X^2 / a
    and Cartesian CW's zero. States in and out are Cartesian Hill states, mapped exactly.
    The acceleration is resolved along the chaser's local radial, along-track and normal axes
    at the start of each segment and held there; those axes turn against the Hill axes by the
    change of the chaser's angle from the target, 1e-6 rad over 10 m.
    """

    name = "cw_curvilinear"

    def __init__(self, radius: float, body: str | Body = "earth"):
        self.radius = radius
        self.n = mean_motion(radius, get_body(body).mu)
        self.disc = CWDiscretizer(self.n)
        self._curvilinear = np.zeros(6)
        self._segment: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self, rel_state: ArrayLike) -> None:
        self._curvilinear = hill_to_curvilinear(rel_state, self.radius)
        self._segment = None

    def advance(
        self, accel_hill: ArrayLike, duration: float, n_samples: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Hold `accel_hill` (m/s^2) for `duration` s. Same returns as `CWDynamics.advance`."""
        x0 = self._curvilinear
        theta, phi = x0[1] / self.radius, x0[2] / self.radius
        accel = np.array(accel_hill, dtype=float)
        local = np.array([accel @ axis for axis in _spherical_axes(theta, phi)])
        self._segment = (x0, local)
        samples = None
        if n_samples > 0:
            phis, gammas = self.disc.sample_matrices(duration, n_samples)
            samples = curvilinear_to_hill(phis @ x0 + gammas @ local, self.radius)
        phi_d, gamma = self.disc.matrices(duration)
        self._curvilinear = phi_d @ x0 + gamma @ local
        return curvilinear_to_hill(self._curvilinear, self.radius), samples

    def state_in_last_segment(self, tau: float) -> np.ndarray:
        """State `tau` seconds into the most recent segment."""
        if self._segment is None:
            raise RuntimeError("no segment has been advanced since reset")
        x0, local = self._segment
        phi, gamma = self.disc.matrices(tau)
        return curvilinear_to_hill(phi @ x0 + gamma @ local, self.radius)


class LinearizedTwoBodyDynamics(TwoBodyDynamics):
    """Relative motion linearized about the target's own orbit, with or without J2.

    The chaser's inertial offset obeys d2(dr)/dt2 = G(r_t) dr + C a, with G the gravity
    gradient (point mass and J2) along the target's numerically integrated orbit and C the Hill
    axes, mapped to the Hill frame exactly: the exact first-order model, linear and time-varying
    under J2. With J2 off it is CW on a circular orbit and Yamanaka-Ankersen on an elliptic one.
    Left out against the nonlinear model: the second-order terms, which dominate the linear J2
    model's error (Alfriend et al., Spacecraft Formation Flying, sec. 7.4); one-orbit
    along-track drift -(9 pi rho^2 / 4a)(2 + cos 2 alpha) for a projected circular orbit of
    radius rho and phase alpha (their eq 5.64). Arguments as for `TwoBodyDynamics`, without
    perturbations.
    """

    def __init__(
        self,
        body: str | Body = "earth",
        altitude: float = 500e3,
        inclination: float = 0.0,
        j2: bool = True,
        rtol: float = 1e-11,
        elements: ArrayLike | None = None,
    ):
        super().__init__(body, altitude, inclination, j2, rtol, elements=elements)
        self.name = "linear_j2" if j2 else "linear_twobody"

    def _rhs(self, t: float, y: np.ndarray, accel_hill: np.ndarray) -> np.ndarray:
        r_t, v_t, dr, dv = y[0:3], y[3:6], y[6:9], y[9:12]
        dg = gravity_gradient(r_t, self.body, self.j2) @ dr
        if accel_hill.any():
            dg = dg + hill_frame(r_t, v_t)[0] @ accel_hill
        return np.concatenate([v_t, self._gravity(r_t), dv, dg])

    def _relative_jacobian(self, y: np.ndarray) -> np.ndarray:
        jac = np.zeros((3, 6))
        jac[:, :3] = gravity_gradient(y[0:3], self.body, self.j2)
        return jac
