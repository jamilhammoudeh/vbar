"""Torque-free rigid-body attitude in closed form.

A rigid body free of torques keeps its rotational energy T and its angular momentum H, fixed in
inertial space. With principal moments I1 < I2 < I3 and the effective inertia D = H^2 / (2 T),
the body rate circulates about the major axis when D > I2 (short-axis mode) and about the minor
axis when D < I2 (long-axis mode); D = I2 is the separatrix through the unstable spin about the
intermediate axis. In either mode the body rate is a pair of Jacobi elliptic functions,

    w = (s A1 cn(tau), sigma A2 sn(tau), s A3 dn(tau)),  tau = tau0 + lambda t,

written about the axis it circulates about (Landau and Lifshitz, Mechanics, 3rd ed., section 37;
Scheeres, Orbital Motion in Strongly Perturbed Environments, 2012, section 3.2). The attitude
follows from 3-1-3 angles about H: theta and psi are algebraic in the body components of H, and
the precession angle phi integrates to an incomplete elliptic integral of the third kind,

    phi(t) = H t / I3 + H (I3 - I1) / (I3 I1 lambda) [Pi(n; am tau) - Pi(n; am tau0)],
    n = -I3 (I2 - I1) / (I1 (I3 - I2)),

evaluated with Carlson's symmetric integrals (DLMF 19.25), whose characteristic has the
opposite sign to the one in Scheeres' appendix F. Attitudes are stored inertially: `matrix`
returns R_n_b, which takes body components to inertial ones (`v_n = R_n_b v_b`), and rates are
in body axes.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
from scipy.special import ellipj, ellipk, ellipkinc, elliprf, elliprj

# Relabels a body frame with ascending moments so the circulation axis of a long-axis tumble is
# the third axis: e1' = e3, e2' = -e2, e3' = e1, a right-handed frame.
_SWAP = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def principal_axes(inertia: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Principal moments and a right-handed set of principal axes of an inertia tensor.

    An eigensolver may return a left-handed set of eigenvectors; the last one is flipped when
    it does (Schaub and Junkins, Analytical Mechanics of Space Systems, 4th ed., example 4.2).

    Parameters
    ----------
    inertia : array_like
        Inertia tensor about the centre of mass (kg m^2), body axes, shape (3, 3), symmetric
        and positive definite.

    Returns
    -------
    moments : numpy.ndarray
        Principal moments (kg m^2), ascending, shape (3,).
    axes : numpy.ndarray
        Principal axes as the columns of a rotation matrix (determinant +1), body axes, so that
        `axes.T @ inertia @ axes` is diagonal, shape (3, 3).

    Raises
    ------
    ValueError
        If the tensor is not a symmetric 3 x 3 matrix with positive eigenvalues.
    """
    tensor = np.asarray(inertia, dtype=float)
    if tensor.shape != (3, 3) or not np.allclose(tensor, tensor.T, rtol=1e-12, atol=0.0):
        raise ValueError("inertia must be a symmetric 3 x 3 matrix")
    moments, axes = np.linalg.eigh(tensor)
    if not moments[0] > 0.0:
        raise ValueError("inertia must be positive definite")
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] = -axes[:, 2]
    return moments, axes


def _carlson_pi(n: float, phi: np.ndarray, m: float) -> np.ndarray:
    """Pi(n; phi | m) = int_0^phi dt / ((1 - n sin^2 t) sqrt(1 - m sin^2 t)), |phi| <= pi / 2."""
    s, c = np.sin(phi), np.cos(phi)
    s2 = s * s
    delta = 1.0 - m * s2
    return s * elliprf(c * c, delta, 1.0) + n / 3.0 * s * s2 * elliprj(
        c * c, delta, 1.0, 1.0 - n * s2
    )


def _about_z(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    out = np.zeros((*np.shape(c), 3, 3))
    out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1] = c, -s, s, c
    out[..., 2, 2] = 1.0
    return out


class TorqueFreeMotion:
    """The closed-form attitude and body rate of a rigid body free of torques.

    Parameters
    ----------
    moments : array_like
        Principal moments about the body x, y and z axes (kg m^2), shape (3,), positive, in any
        order; the body axes are principal axes.
    omega : array_like
        Body rate at t = 0 (rad/s), body axes, shape (3,).
    attitude : array_like
        Quaternion q_n_b at t = 0, scalar last, shape (4,), nonzero; normalised here.

    Raises
    ------
    ValueError
        If a moment is not positive, the inputs have the wrong shapes, or the motion lies on the
        separatrix (D = I2 exactly, with distinct moments and a rate off the principal axes),
        which has no periodic solution.

    Notes
    -----
    A rate along a principal axis (or any rate of a body with three equal moments) is a pure
    spin about a fixed axis, taken separately; every other motion has the closed form of the
    module docstring, exact but for rounding: the elliptic functions are good to about 1e-15
    relative, and the phase tau = tau0 + lambda t carries the rounding of a number that grows
    with t, so the attitude error grows like 1e-16 lambda t rad.
    """

    def __init__(self, moments: ArrayLike, omega: ArrayLike, attitude: ArrayLike = (0, 0, 0, 1)):
        moments = np.asarray(moments, dtype=float)
        omega = np.asarray(omega, dtype=float)
        q = np.asarray(attitude, dtype=float)
        if moments.shape != (3,) or omega.shape != (3,) or q.shape != (4,):
            raise ValueError("moments and omega are 3-vectors and attitude a quaternion")
        if not np.all(moments > 0.0):
            raise ValueError("principal moments must be positive")
        if not np.linalg.norm(q) > 0.0:
            raise ValueError("attitude must be a nonzero quaternion")
        self.moments = moments
        self.omega0 = omega
        self.r0 = Rotation.from_quat(q).as_matrix()
        h_body = moments * omega
        self.momentum = self.r0 @ h_body
        self.h = float(np.linalg.norm(h_body))
        self.energy = 0.5 * float(np.sum(moments * omega**2))
        self.spin = not np.any(np.cross(h_body, omega))
        if self.spin:
            return
        # Body axes relabelled with ascending moments, right-handed (one sign flip when the
        # relabelling is odd).
        order = np.argsort(moments, kind="stable")
        relabel = np.eye(3)[order]
        if np.linalg.det(relabel) < 0.0:
            relabel[1] = -relabel[1]
        i_s = moments[order]
        w_s = relabel @ omega
        # sign(D - I2) without the cancellation of H^2 - 2 T I2.
        side = i_s[0] * (i_s[0] - i_s[1]) * w_s[0] ** 2 + i_s[2] * (i_s[2] - i_s[1]) * w_s[2] ** 2
        if side == 0.0:
            raise ValueError("the motion lies on the separatrix (D = I2), which is not periodic")
        self.long_axis = side < 0.0
        if self.long_axis:
            relabel = _SWAP @ relabel
            i_s = i_s[::-1].copy()
            w_s = _SWAP @ w_s
        self.relabel = relabel
        i1, i2, i3 = i_s
        w1, w2, w3 = w_s
        # g3 = 2 T I3 - H^2 and g1 = H^2 - 2 T I1, each a sum of terms of one sign.
        g3 = i1 * (i3 - i1) * w1**2 + i2 * (i3 - i2) * w2**2
        g1 = i2 * (i2 - i1) * w2**2 + i3 * (i3 - i1) * w3**2
        self.amplitudes = np.sqrt(
            [g3 / (i1 * (i3 - i1)), g3 / (i2 * (i3 - i2)), g1 / (i3 * (i3 - i1))]
        )
        self.rate = math.sqrt((i3 - i2) * g1 / (i1 * i2 * i3))
        self.m = float((i2 - i1) * g3 / ((i3 - i2) * g1))
        self.sign = 1.0 if w3 >= 0.0 else -1.0
        self.sigma = 1.0 if i3 > i1 else -1.0
        self.quarter = float(ellipk(self.m))
        self.n_char = -i3 * (i2 - i1) / (i1 * (i3 - i2))
        self.phi_scale = self.h * (i3 - i1) / (i3 * i1 * self.rate)
        self.pi_half = float(
            elliprf(0.0, 1.0 - self.m, 1.0)
            + self.n_char / 3.0 * elliprj(0.0, 1.0 - self.m, 1.0, 1.0 - self.n_char)
        )
        a1, a2, _ = self.amplitudes
        am0 = math.atan2(self.sigma * w2 / a2, self.sign * w1 / a1)
        turns = round(am0 / math.pi)
        rest = am0 - turns * math.pi
        self.tau0 = 2.0 * self.quarter * turns + float(ellipkinc(rest, self.m))
        self.i_core = i_s
        self.pi0 = float(self._pi_of(np.array(self.tau0)))
        # The frame of H: its z axis along H, its x axis where phi = 0 puts it at t = 0.
        core0 = self.r0 @ relabel.T
        self.frame = core0 @ self._nutation(w_s[None, :])[0].T

    @property
    def effective_inertia(self) -> float:
        """D = H^2 / (2 T) (kg m^2), between the least and the greatest moment."""
        return self.h**2 / (2.0 * self.energy) if self.energy > 0.0 else float(self.moments[0])

    @property
    def period(self) -> float:
        """Period of the body rate (s): 4 K(m) / lambda, or infinity for a pure spin."""
        return math.inf if self.spin else 4.0 * self.quarter / self.rate

    def _reduce(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Whole half periods j and the remainder r = tau - 2 K j, |r| <= K."""
        turns = np.round(tau / (2.0 * self.quarter))
        return turns, tau - 2.0 * self.quarter * turns

    def _pi_of(self, tau: np.ndarray) -> np.ndarray:
        """Pi(n; am tau | m), continued through every half period."""
        turns, rest = self._reduce(tau)
        phase = ellipj(rest, self.m)[3]
        return 2.0 * turns * self.pi_half + _carlson_pi(self.n_char, phase, self.m)

    def _core_rate(self, t: np.ndarray) -> np.ndarray:
        tau = self.tau0 + self.rate * t
        turns, rest = self._reduce(tau)
        sn, cn, dn, _ = ellipj(rest, self.m)
        flip = np.where(turns % 2.0 == 0.0, 1.0, -1.0)
        a1, a2, a3 = self.amplitudes
        return np.stack(
            [self.sign * a1 * cn * flip, self.sigma * a2 * sn * flip, self.sign * a3 * dn], axis=-1
        )

    def _nutation(self, w: np.ndarray) -> np.ndarray:
        """Rx(theta) Rz(psi) from the body components of H, h = I w / H, shape (..., 3, 3)."""
        h = w * self.i_core / self.h
        s = np.hypot(h[..., 0], h[..., 1])
        sp, cp = h[..., 0] / s, h[..., 1] / s
        out = np.zeros((*np.shape(s), 3, 3))
        out[..., 0, 0], out[..., 0, 1] = cp, -sp
        out[..., 1, 0], out[..., 1, 1], out[..., 1, 2] = h[..., 2] * sp, h[..., 2] * cp, -s
        out[..., 2, :] = h
        return out

    def omega(self, t: ArrayLike) -> np.ndarray:
        """Body rate at `t`.

        Parameters
        ----------
        t : array_like
            Time since t = 0 (s), any shape.

        Returns
        -------
        numpy.ndarray
            Body rate (rad/s), body axes, shape (*t.shape, 3).
        """
        t = np.asarray(t, dtype=float)
        if self.spin:
            return np.broadcast_to(self.omega0, (*t.shape, 3)).copy()
        return np.einsum("ji,...j->...i", self.relabel, self._core_rate(t))

    def precession(self, t: ArrayLike) -> np.ndarray:
        """phi(t), the angle about H of the circulation axis's 3-1-3 node, from phi(0) = 0.

        Parameters
        ----------
        t : array_like
            Time since t = 0 (s), any shape.

        Returns
        -------
        numpy.ndarray
            Angle (rad), not wrapped, shape of `t`.

        Raises
        ------
        ValueError
            For a pure spin, which has no 3-1-3 node.
        """
        if self.spin:
            raise ValueError("a pure spin has no precession angle")
        t = np.asarray(t, dtype=float)
        tau = self.tau0 + self.rate * t
        return self.h * t / self.i_core[2] + self.phi_scale * (self._pi_of(tau) - self.pi0)

    def matrix(self, t: ArrayLike) -> np.ndarray:
        """Attitude R_n_b at `t`.

        Parameters
        ----------
        t : array_like
            Time since t = 0 (s), any shape.

        Returns
        -------
        numpy.ndarray
            Rotation matrices from body to inertial axes, shape (*t.shape, 3, 3).
        """
        t = np.asarray(t, dtype=float)
        if self.spin:
            rate = float(np.linalg.norm(self.omega0))
            if rate == 0.0:
                return np.broadcast_to(self.r0, (*t.shape, 3, 3)).copy()
            turn = Rotation.from_rotvec(np.multiply.outer(rate * t, self.momentum / self.h))
            return turn.as_matrix().reshape(*t.shape, 3, 3) @ self.r0
        nutation = self._nutation(self._core_rate(t))
        return self.frame @ _about_z(self.precession(t)) @ nutation @ self.relabel

    def angular_velocity(self, t: ArrayLike) -> np.ndarray:
        """Body rate in inertial axes, R_n_b w, at `t`.

        Parameters
        ----------
        t : array_like
            Time since t = 0 (s), any shape.

        Returns
        -------
        numpy.ndarray
            Angular velocity (rad/s), inertial axes, shape (*t.shape, 3).
        """
        return np.einsum("...ij,...j->...i", self.matrix(t), self.omega(t))

    def rate_bound(self) -> float:
        """Bound on the body rate's magnitude (rad/s).

        The rate itself for a pure spin; otherwise the least of sqrt(2 T / I_min), H / I_min and
        the norm of the three amplitudes, each of which bounds one component of the closed form.
        """
        if self.spin:
            return float(np.linalg.norm(self.omega0))
        least = float(self.moments.min())
        bound = min(math.sqrt(2.0 * self.energy / least), self.h / least)
        return min(bound, float(np.linalg.norm(self.amplitudes)))

    def acceleration_bound(self) -> float:
        """Bound on the body rate's derivative, |dw/dt| in body axes (rad/s^2).

        Euler's equations give dw_i/dt = (I_j - I_k) / I_i w_j w_k, and each |w_j| is at most its
        amplitude; a pure spin has none.
        """
        if self.spin:
            return 0.0
        i1, i2, i3 = self.i_core
        a1, a2, a3 = self.amplitudes
        terms = ((i2 - i3) / i1 * a2 * a3, (i3 - i1) / i2 * a3 * a1, (i1 - i2) / i3 * a1 * a2)
        return float(np.linalg.norm(terms))

    @classmethod
    def from_tumble(
        cls,
        moments: ArrayLike,
        effective_inertia: float,
        rate: float,
        phase: float = 0.0,
        momentum_direction: ArrayLike = (0.0, 0.0, 1.0),
        spin_angle: float = 0.0,
    ) -> TorqueFreeMotion:
        """The motion with the given tumble parameters rather than a body rate.

        The tumble is set by the inertia, D and omega_l = 2 T / H = H / D (Scheeres, section
        3.2.8): its shape by D between the least and greatest moment, its pace by omega_l, its
        start by the polhode `phase`, and its orientation by the inertial direction of H and the
        angle of the body about it.

        Parameters
        ----------
        moments : array_like
            Principal moments about the body axes (kg m^2), shape (3,), positive.
        effective_inertia : float
            D = H^2 / (2 T) (kg m^2), strictly between the least and greatest moment and not the
            intermediate one (equal to a distinct extreme moment gives a pure spin about it).
        rate : float
            omega_l = H / D (rad/s), positive.
        phase : float
            Fraction of the body-rate period at t = 0, in [0, 1).
        momentum_direction : array_like
            Direction of H in inertial axes, shape (3,), nonzero.
        spin_angle : float
            Rotation of the body about H at t = 0 (rad).

        Returns
        -------
        TorqueFreeMotion
            The motion, with its rate and attitude at t = 0.

        Raises
        ------
        ValueError
            If D is outside the moments' range or equals the intermediate moment, or the rate
            is not positive.
        """
        moments = np.asarray(moments, dtype=float)
        order = np.argsort(moments, kind="stable")
        i_s = moments[order]
        d = float(effective_inertia)
        if not i_s[0] <= d <= i_s[2] or not rate > 0.0:
            raise ValueError("D must lie between the least and greatest moment and rate be > 0")
        relabel = np.eye(3)[order]
        if np.linalg.det(relabel) < 0.0:
            relabel[1] = -relabel[1]
        h = d * rate
        two_t = h * rate
        if d in (i_s[0], i_s[2]):
            axis = 0 if d == i_s[0] else 2
            w_s = np.zeros(3)
            w_s[axis] = h / i_s[axis]
        else:
            if d == i_s[1]:
                raise ValueError("D equal to the intermediate moment is the separatrix")
            core = relabel.copy()
            i_c = i_s.copy()
            if d < i_s[1]:
                core = _SWAP @ core
                i_c = i_c[::-1].copy()
            i1, i2, i3 = i_c
            g3 = two_t * i3 - h * h
            g1 = h * h - two_t * i1
            a1 = math.sqrt(g3 / (i1 * (i3 - i1)))
            a2 = math.sqrt(g3 / (i2 * (i3 - i2)))
            a3 = math.sqrt(g1 / (i3 * (i3 - i1)))
            m = (i2 - i1) * g3 / ((i3 - i2) * g1)
            sigma = 1.0 if i3 > i1 else -1.0
            sn, cn, dn, _ = ellipj(4.0 * float(ellipk(m)) * phase, m)
            w_c = np.array([a1 * cn, sigma * a2 * sn, a3 * dn])
            w_s = (core @ relabel.T).T @ w_c
        omega = relabel.T @ w_s
        # The body attitude that puts H along `momentum_direction`, turned by `spin_angle`.
        z = np.asarray(momentum_direction, dtype=float)
        z = z / np.linalg.norm(z)
        h_body = moments * omega / h
        to_h = _align(h_body, z)
        about = Rotation.from_rotvec(spin_angle * z).as_matrix()
        return cls(moments, omega, Rotation.from_matrix(about @ to_h).as_quat())


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A rotation matrix taking the unit vector `a` to the unit vector `b`."""
    axis = np.cross(a, b)
    s, c = float(np.linalg.norm(axis)), float(a @ b)
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        # Half a turn about any axis normal to a.
        normal = np.cross(a, (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0))
        return Rotation.from_rotvec(math.pi * normal / np.linalg.norm(normal)).as_matrix()
    return Rotation.from_rotvec(axis / s * math.atan2(s, c)).as_matrix()


__all__ = ["TorqueFreeMotion", "principal_axes"]
