"""Closed-form Euclidean projections and the exact projection out of a keep-out ellipsoid.

The halfspace, box, ball and second-order cone projections are those of Boyd and Vandenberghe,
*Convex Optimization*, sec. 8.1.1 and exercise 8.3(c). `keep_out_projection` solves the nonconvex
problem min |u - u_nom|^2 subject to |G u + h| >= R exactly: it has one indefinite quadratic
constraint, so its duality gap is zero (appendix B.1) and the optimum follows from a
one-dimensional search on the multiplier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

# The search variable mu = 1 - lam sigma_max^2 is located to this many ulp of itself.
_MU_RTOL = 4.0 * np.finfo(float).eps
# Below this fraction of R, the component of G u_nom + h along the top left singular vectors is
# indistinguishable from rounding and the problem is solved as the hard case.
_HARD_TOL = 1e-12


def halfspace(u, a, b: float) -> np.ndarray:
    """Project onto the halfspace a . u >= b.

    Parameters
    ----------
    u : array_like, shape (k,)
        The point to project.
    a : array_like, shape (k,)
        The halfspace normal, nonzero.
    b : float
        The offset, in units of a . u.

    Returns
    -------
    ndarray, shape (k,)
        u + max(0, b - a . u) a / |a|^2 (Boyd and Vandenberghe, sec. 8.1.1).

    Raises
    ------
    ValueError
        If the shapes differ or `a` is zero.
    """
    u = np.array(u, dtype=float)
    a = np.asarray(a, dtype=float)
    if u.ndim != 1 or a.shape != u.shape:
        raise ValueError("u and a must be vectors of one length")
    aa = float(a @ a)
    if aa == 0.0:
        raise ValueError("a must be nonzero")
    return u + max(0.0, b - float(a @ u)) / aa * a


def box(u, a_max: float) -> np.ndarray:
    """Project onto the box |u_i| <= a_max.

    Parameters
    ----------
    u : array_like, shape (k,)
        The point to project.
    a_max : float
        The half-width of the box on every axis, positive.

    Returns
    -------
    ndarray, shape (k,)
        u clipped to [-a_max, a_max] on every axis.

    Raises
    ------
    ValueError
        If `a_max` is not finite and positive.
    """
    if not (math.isfinite(a_max) and a_max > 0.0):
        raise ValueError("a_max must be finite and positive")
    return np.clip(np.asarray(u, dtype=float), -a_max, a_max)


def ball(u, a_max: float) -> np.ndarray:
    """Project onto the ball |u|_2 <= a_max.

    Parameters
    ----------
    u : array_like, shape (k,)
        The point to project.
    a_max : float
        The radius of the ball, positive.

    Returns
    -------
    ndarray, shape (k,)
        u, or u scaled onto the sphere of radius `a_max` when it lies outside.

    Raises
    ------
    ValueError
        If `a_max` is not finite and positive.
    """
    if not (math.isfinite(a_max) and a_max > 0.0):
        raise ValueError("a_max must be finite and positive")
    u = np.array(u, dtype=float)
    norm = float(np.linalg.norm(u))
    return u * (a_max / norm) if norm > a_max else u


def second_order_cone(x, t: float) -> tuple[np.ndarray, float]:
    """Project (x, t) onto the second-order cone |x|_2 <= t.

    Parameters
    ----------
    x : array_like, shape (k,)
        The vector part.
    t : float
        The scalar part.

    Returns
    -------
    x : ndarray, shape (k,)
        The vector part of the projection.
    t : float
        The scalar part of the projection.

    Notes
    -----
    Boyd and Vandenberghe, exercise 8.3(c): the apex (0, 0) when |x| <= -t, (x, t) itself when
    |x| <= t, and (1/2)(1 + t/|x|)(x, |x|) otherwise. Membership needs t >= 0 as well as
    t^2 >= |x|^2, so the negative cone {|x| <= -t} maps to the apex.
    """
    x = np.array(x, dtype=float)
    t = float(t)
    norm = float(np.linalg.norm(x))
    if norm <= -t:
        return np.zeros_like(x), 0.0
    if norm <= t:
        return x, t
    scale = 0.5 * (1.0 + t / norm)
    return scale * x, scale * norm


@dataclass(frozen=True, kw_only=True)
class KeepOutProjection:
    """The result of `keep_out_projection`.

    Attributes
    ----------
    u : ndarray, shape (n,)
        The nearest command to u_nom with |G u + h| >= R; on the boundary unless u_nom was
        already outside.
    lam : float
        The multiplier of the constraint, in [0, 1/sigma_max(G)^2]; 0 when u_nom was outside.
    hard_case : bool
        True when G u_nom + h has no component along the top left singular vectors of G and the
        other directions cannot reach R, so lam = 1/sigma_max(G)^2 and `u` has a component along
        the top right singular vector. The optimum is then not unique: the opposite sign of that
        component is optimal too.
    """

    u: np.ndarray
    lam: float
    hard_case: bool


def keep_out_projection(u_nom, G, h, radius: float) -> KeepOutProjection:
    """Find the nearest command whose image G u + h lies outside the sphere of radius R.

    Solves min |u - u_nom|^2 subject to |G u + h|_2 >= R exactly, the projection out of a keep-out
    region when the position after one held step is G u + h.

    Parameters
    ----------
    u_nom : array_like, shape (n,)
        The nominal command.
    G : array_like, shape (m, n)
        The map from command to position (for example m per m/s^2), nonzero.
    h : array_like, shape (m,)
        The position with zero command, in the units of `radius`.
    radius : float
        The keep-out radius R, positive.

    Returns
    -------
    KeepOutProjection
        The optimal command, its multiplier and whether the hard case occurred.

    Raises
    ------
    ValueError
        If the shapes disagree, a value is not finite, R is not positive, or G is zero while
        u_nom is inside.

    Notes
    -----
    With the multiplier lam, stationarity gives u(lam) = (I - lam G^T G)^-1 (u_nom + lam G^T h),
    and the dual is concave on lam in [0, 1/sigma_max(G)^2) (Boyd and Vandenberghe, appendix B.1).
    With G = U S V^T and z = U^T (G u_nom + h), the image is w_i(lam) = z_i / (1 - lam s_i^2), so
    |G u(lam) + h| increases with lam and the boundary is crossed once. The search runs on
    mu = 1 - lam s_1^2 in (0, 1], where the top term is z_1 / mu without cancellation, by Brent's
    method to 4 ulp of mu. When z has no component along the top singular directions and
    the rest of the image stays inside R as lam -> 1/s_1^2 (the trust-region hard case), lam is
    that limit and the missing range is supplied by a component along the top right singular
    vector, the null space of I - lam G^T G.

    The guarantee holds for this one constraint only; adding a thrust limit makes the problem a
    general nonconvex QP with two constraints, where the duality gap need not be zero.
    """
    u_nom = np.array(u_nom, dtype=float)
    G = np.array(G, dtype=float)
    h = np.array(h, dtype=float)
    radius = float(radius)
    if u_nom.ndim != 1 or G.ndim != 2 or G.shape[1] != u_nom.shape[0] or h.shape != G.shape[:1]:
        raise ValueError("expected u_nom (n,), G (m, n) and h (m,)")
    if not (np.all(np.isfinite(u_nom)) and np.all(np.isfinite(G)) and np.all(np.isfinite(h))):
        raise ValueError("u_nom, G and h must be finite")
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError("radius must be finite and positive")
    w0 = G @ u_nom + h
    if np.linalg.norm(w0) >= radius:
        return KeepOutProjection(u=u_nom, lam=0.0, hard_case=False)
    left, s, right_t = np.linalg.svd(G)
    if s[0] == 0.0:
        raise ValueError("G is zero, the keep-out cannot be left")
    z = left.T @ w0
    k = len(s)
    z_range, z_perp = z[:k], z[k:]
    r = (s / s[0]) ** 2
    top = s == s[0]
    z_top = float(np.linalg.norm(z_range[top]))
    # The image at the hard-case limit lam = 1/s_1^2, without the top singular directions.
    w_rest = np.concatenate([np.zeros(np.count_nonzero(top)), z_range[~top] / (1.0 - r[~top])])
    rest = math.hypot(float(np.linalg.norm(w_rest)), float(np.linalg.norm(z_perp)))

    def image(mu: float) -> np.ndarray:
        # At mu = 0 the top terms are 0 / 0 with z_top = 0; their limit along the path is 0.
        den = (1.0 - r) + mu * r
        return np.divide(z_range, den, out=np.zeros(k), where=den > 0.0)

    if z_top <= _HARD_TOL * radius and rest <= radius:
        w_range = np.where(top, z_range, w_rest)
        w_range[0] = 0.0
        reach = math.sqrt(max(0.0, radius**2 - float(w_range @ w_range) - float(z_perp @ z_perp)))
        w_range[0] = math.copysign(reach, z_range[0])
        lam = 1.0 / s[0] ** 2
        hard_case = True
    else:

        def image_norm(mu: float) -> float:
            return math.hypot(float(np.linalg.norm(image(mu))), float(np.linalg.norm(z_perp)))

        # |w(mu)| >= z_top / mu, so at mu = z_top / (2 R) the image is outside; with z_top = 0 the
        # image at mu = 0 has norm `rest`, which exceeds R on this branch.
        mu_lo = z_top / (2.0 * radius)
        mu = brentq(
            lambda m: image_norm(m) - radius,
            mu_lo,
            1.0,
            xtol=_MU_RTOL * max(mu_lo, np.finfo(float).tiny),
            rtol=_MU_RTOL,
        )
        w_range = image(mu)
        lam = (1.0 - mu) / s[0] ** 2
        hard_case = False
    nonzero = s > 0.0
    du = np.zeros(len(u_nom))
    du[:k][nonzero] = (w_range - z_range)[nonzero] / s[nonzero]
    return KeepOutProjection(u=u_nom + right_t.T @ du, lam=lam, hard_case=hard_case)
