"""Error sets on the Hill-frame state and the margins they add to safety constraints.

An error set E bounds e = x - x_hat, the difference between the true state and the state a
filter acts on (m and m/s, Hill frame, order x, y, z, vx, vy, vz). A row a . x >= b holds for
every x in x_hat + E when a . x_hat >= b + h_E(a), with the support function h_E(a) = max over
e in E of a . e (Boyd and Vandenberghe, *Convex Optimization*, sec. 4.3.1). Every set here is
symmetric, so h_E(-a) = h_E(a).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

# A covariance may be asymmetric or have negative eigenvalues by this fraction of its largest
# entry, the rounding of a propagated covariance; beyond it the matrix is refused.
_COV_TOL = 1e-12
_VERTICES = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))
_VERTICES.flags.writeable = False


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def _covariance(cov) -> np.ndarray:
    """Return a validated 6x6 covariance with its eigenvalues clipped at zero."""
    cov = np.array(cov, dtype=float)
    if cov.shape != (6, 6) or not np.all(np.isfinite(cov)):
        raise ValueError("cov must be a finite 6x6 matrix")
    scale = float(np.max(np.abs(cov)))
    if np.max(np.abs(cov - cov.T)) > _COV_TOL * scale:
        raise ValueError("cov must be symmetric")
    cov = 0.5 * (cov + cov.T)
    eigenvalues, vectors = np.linalg.eigh(cov)
    if eigenvalues[0] < -_COV_TOL * scale:
        raise ValueError("cov must be positive semidefinite")
    return (vectors * np.clip(eigenvalues, 0.0, None)) @ vectors.T


@dataclass(frozen=True, kw_only=True)
class BoxError:
    """Per-axis bounds |e_i| <= r_pos on position and |e_i| <= r_vel on velocity.

    Parameters
    ----------
    r_pos : float
        Bound on each position component, m, nonnegative.
    r_vel : float
        Bound on each velocity component, m/s, nonnegative.

    Raises
    ------
    ValueError
        If a bound is negative or not finite.
    """

    r_pos: float
    r_vel: float

    def __post_init__(self):
        object.__setattr__(self, "r_pos", _nonnegative(self.r_pos, "r_pos"))
        object.__setattr__(self, "r_vel", _nonnegative(self.r_vel, "r_vel"))


@dataclass(frozen=True, kw_only=True)
class BallError:
    """Separate balls |e_pos|_2 <= r_pos and |e_vel|_2 <= r_vel.

    Parameters
    ----------
    r_pos : float
        Radius of the position error ball, m, nonnegative.
    r_vel : float
        Radius of the velocity error ball, m/s, nonnegative.

    Raises
    ------
    ValueError
        If a radius is negative or not finite.
    """

    r_pos: float
    r_vel: float

    def __post_init__(self):
        object.__setattr__(self, "r_pos", _nonnegative(self.r_pos, "r_pos"))
        object.__setattr__(self, "r_vel", _nonnegative(self.r_vel, "r_vel"))


@dataclass(frozen=True, kw_only=True)
class EllipsoidError:
    """The ellipsoid E = {P z : |z|_2 <= 1}.

    Parameters
    ----------
    P : array_like, shape (6, 6)
        The shape matrix, rows in m (position) and m/s (velocity); singular for a flat
        ellipsoid.
    exceedance : float
        The probability per draw that the error lies outside E, in [0, 1]; 0 for a hard bound.

    Raises
    ------
    ValueError
        If `P` is not a finite 6x6 matrix or `exceedance` is outside [0, 1].
    """

    P: np.ndarray
    exceedance: float = 0.0

    def __post_init__(self):
        P = np.array(self.P, dtype=float)
        if P.shape != (6, 6) or not np.all(np.isfinite(P)):
            raise ValueError("P must be a finite 6x6 matrix")
        exceedance = float(self.exceedance)
        if not 0.0 <= exceedance <= 1.0:
            raise ValueError("exceedance must be in [0, 1]")
        P.flags.writeable = False
        object.__setattr__(self, "P", P)
        object.__setattr__(self, "exceedance", exceedance)

    @classmethod
    def from_gaussian(cls, cov, k: float) -> EllipsoidError:
        """Build the k-sigma ellipsoid of a zero-mean Gaussian error.

        Parameters
        ----------
        cov : array_like, shape (6, 6)
            The error covariance S, symmetric positive semidefinite, m^2, m^2/s and m^2/s^2.
        k : float
            The size of the ellipsoid in standard deviations, positive.

        Returns
        -------
        EllipsoidError
            P = k S^(1/2), the set {e : e^T S^-1 e <= k^2} for a nonsingular S, with
            `exceedance` = P(chi-square with 6 degrees of freedom > k^2).

        Raises
        ------
        ValueError
            If `cov` is not a symmetric positive semidefinite 6x6 matrix or `k` is not positive.

        Notes
        -----
        e^T S^-1 e of a Gaussian error is chi-square with 6 degrees of freedom. For a singular S
        it has as many degrees of freedom as the rank of S, and its tail beyond k^2 is smaller,
        so the stated exceedance is an upper bound then.
        """
        k = _positive(k, "k")
        eigenvalues, vectors = np.linalg.eigh(_covariance(cov))
        root = (vectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ vectors.T
        return cls(P=k * root, exceedance=float(chi2.sf(k**2, 6)))


ErrorSet = BoxError | BallError | EllipsoidError


def halfspace_margin(a, error: ErrorSet) -> float:
    """Return the support function h_E(a), the tightening of a row a . x >= b.

    Parameters
    ----------
    a : array_like, shape (6,)
        The row normal on the state, in row units per m (position) and per m/s (velocity).
    error : BoxError, BallError or EllipsoidError
        The error set E.

    Returns
    -------
    float
        max over e in E of a . e, in row units: r_pos |a_pos|_1 + r_vel |a_vel|_1 for a box,
        r_pos |a_pos|_2 + r_vel |a_vel|_2 for two balls, |P^T a|_2 for an ellipsoid
        (Boyd and Vandenberghe, sec. 4.3.1 and eq. 4.37).

    Raises
    ------
    ValueError
        If `a` is not a finite vector of length 6.
    TypeError
        If `error` is not one of the error sets.
    """
    a = np.asarray(a, dtype=float)
    if a.shape != (6,) or not np.all(np.isfinite(a)):
        raise ValueError("a must be a finite vector of length 6")
    match error:
        case BoxError():
            return error.r_pos * float(np.sum(np.abs(a[:3]))) + error.r_vel * float(
                np.sum(np.abs(a[3:]))
            )
        case BallError():
            return error.r_pos * float(np.linalg.norm(a[:3])) + error.r_vel * float(
                np.linalg.norm(a[3:])
            )
        case EllipsoidError():
            return float(np.linalg.norm(error.P.T @ a))
    raise TypeError(f"unknown error set {type(error).__name__}")


def chance_margin(a, cov, eta: float) -> float:
    """Return the margin that makes a row hold with probability eta under Gaussian error.

    Parameters
    ----------
    a : array_like, shape (6,)
        The row normal on the state, in row units per m and per m/s.
    cov : array_like, shape (6, 6)
        The covariance S of the zero-mean Gaussian error, symmetric positive semidefinite.
    eta : float
        The required probability, in [0.5, 1).

    Returns
    -------
    float
        Phi^-1(eta) |S^(1/2) a|_2, in row units: a . x >= b holds with probability eta when
        a . x_hat >= b plus this margin (Boyd and Vandenberghe, eq. 4.38).

    Raises
    ------
    ValueError
        If `eta` is outside [0.5, 1), where the constraint is not convex or cannot be met, or
        `a` or `cov` is malformed.
    """
    eta = float(eta)
    if not 0.5 <= eta < 1.0:
        raise ValueError("eta must be in [0.5, 1)")
    a = np.asarray(a, dtype=float)
    if a.shape != (6,) or not np.all(np.isfinite(a)):
        raise ValueError("a must be a finite vector of length 6")
    variance = max(0.0, float(a @ _covariance(cov) @ a))
    return float(norm.ppf(eta)) * math.sqrt(variance)


def keep_out_margin(phi_r, error: ErrorSet) -> float:
    """Return the largest range change max over e in E of |Phi_r e|_2.

    The range |r| is 1-Lipschitz in r and the Clohessy-Wiltshire map is linear, so with
    r = Phi_r x the true range is at least the range computed from x_hat minus this margin.

    Parameters
    ----------
    phi_r : array_like, shape (k, 6)
        The rows of the state transition matrix that give position, in m per m and m per m/s.
    error : BoxError, BallError or EllipsoidError
        The error set E.

    Returns
    -------
    float
        The margin in m. Exact for a box (the maximum over its 64 vertices, where a convex
        function on a polytope peaks) and for an ellipsoid (sigma_max(Phi_r P)). For two balls it
        is the bound r_pos sigma_max(Phi_r,pos) + r_vel sigma_max(Phi_r,vel), exact when the
        two blocks share their top left singular vector and otherwise above the maximum.

    Raises
    ------
    ValueError
        If `phi_r` is not a finite matrix with 6 columns.
    TypeError
        If `error` is not one of the error sets.
    """
    phi_r = np.asarray(phi_r, dtype=float)
    if phi_r.ndim != 2 or phi_r.shape[1] != 6 or not np.all(np.isfinite(phi_r)):
        raise ValueError("phi_r must be a finite matrix with 6 columns")
    match error:
        case BoxError():
            radii = np.array([error.r_pos] * 3 + [error.r_vel] * 3)
            return float(np.max(np.linalg.norm((_VERTICES * radii) @ phi_r.T, axis=1)))
        case BallError():
            return error.r_pos * float(np.linalg.norm(phi_r[:, :3], 2)) + error.r_vel * float(
                np.linalg.norm(phi_r[:, 3:], 2)
            )
        case EllipsoidError():
            return float(np.linalg.norm(phi_r @ error.P, 2))
    raise TypeError(f"unknown error set {type(error).__name__}")


def ecbf_bias(da_max: float, poles: tuple[float, float]) -> float:
    """Return the depth of the riding offset of an exponential CBF under a persistent bias.

    Parameters
    ----------
    da_max : float
        The largest unmodelled acceleration along the barrier gradient, m/s^2, nonnegative.
    poles : tuple of float
        The two real closed-loop poles p1, p2 of the barrier, 1/s, positive.

    Returns
    -------
    float
        da_max / (p1 p2), in m: h'' + (p1 + p2) h' + p1 p2 h = -da_max settles at
        h = -da_max / (p1 p2), so the barrier must be moved out by this amount.

    Raises
    ------
    ValueError
        If `da_max` is negative or a pole is not positive.
    """
    da_max = _nonnegative(da_max, "da_max")
    if len(poles) != 2:
        raise ValueError("poles must be two values")
    p1, p2 = (_positive(p, "poles") for p in poles)
    return da_max / (p1 * p2)


def first_order_bias(da_max: float, gamma: float) -> float:
    """Return the depth of the riding offset of a first-order CBF under a persistent bias.

    Parameters
    ----------
    da_max : float
        The largest unmodelled rate of the barrier value, in barrier units per s, nonnegative.
    gamma : float
        The class-K gain of h' >= -gamma h, 1/s, positive.

    Returns
    -------
    float
        da_max / gamma: h' = -gamma h - da_max settles at h = -da_max / gamma.

    Raises
    ------
    ValueError
        If `da_max` is negative or `gamma` is not positive.
    """
    return _nonnegative(da_max, "da_max") / _positive(gamma, "gamma")
