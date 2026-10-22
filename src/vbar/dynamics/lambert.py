"""Lambert's problem in universal variables: the conic joining two positions in a given time."""

from __future__ import annotations

import math
import sys

import numpy as np
from numpy.typing import ArrayLike

from .kepler import _stumpff

_EPS = sys.float_info.epsilon
_Z_MAX = 4.0 * math.pi**2
# Estimated relative rounding in the velocities above which a transfer is refused: half the
# 1e-9 the round trips are validated to, since the estimate counts roundings, not their signs.
_ROUNDING_LIMIT = 5e-10
# Below this z the two leading terms of F, each of order cosh(sqrt(-z))^(1/4) times A^(3/2),
# cancel to their rounding, and a root there would stand for a transfer far faster than the
# orbital time scale of the positions.
_Z_SEARCH_FLOOR = -1e4


def _stumpff_derivatives(z: float, c2: float, c3: float) -> tuple[float, float]:
    """dc2/dz and dc3/dz; the closed forms divide a cancelling difference by z near z = 0."""
    if abs(z) < 0.1:
        dc2 = -sum((k + 1) * (-z) ** k / math.factorial(2 * k + 4) for k in range(8))
        dc3 = -sum((k + 1) * (-z) ** k / math.factorial(2 * k + 5) for k in range(8))
        return dc2, dc3
    return (1.0 - z * c3 - 2.0 * c2) / (2.0 * z), (c2 - 3.0 * c3) / (2.0 * z)


def _half_angle_terms(z: float) -> tuple[float, float]:
    """q(z) = sin^2(sqrt(z) / 4) continued to -sinh^2(sqrt(-z) / 4) for z < 0, and dq/dz."""
    if z == 0.0:
        return 0.0, 1.0 / 16.0
    if z > 0.0:
        s = math.sqrt(z)
        return math.sin(0.25 * s) ** 2, math.sin(0.5 * s) / (8.0 * s)
    s = math.sqrt(-z)
    return -(math.sinh(0.25 * s) ** 2), math.sinh(0.5 * s) / (8.0 * s)


def lambert(
    r1: ArrayLike,
    r2: ArrayLike,
    tof: float,
    mu: float,
    *,
    prograde: bool = True,
    normal: ArrayLike = (0.0, 0.0, 1.0),
    tol: float = 1e-10,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocities at `r1` and `r2` of the single-revolution conic that joins them in `tof` s.

    Universal-variable formulation after Curtis, Orbital Mechanics for Engineering Students,
    algorithm 5.2, and Bate, Mueller and White, Fundamentals of Astrodynamics, ch. 5.

    Parameters
    ----------
    r1, r2 : array_like, shape (3,)
        Inertial positions (m) at departure and arrival.
    tof : float
        Time of flight (s), positive.
    mu : float
        Gravitational parameter (m^3/s^2) of the central body.
    prograde : bool
        Sweep the transfer angle counterclockwise about `normal` (True) or clockwise (False).
    normal : array_like, shape (3,)
        Reference direction that fixes the sense of the transfer; the orbit normal of a
        prograde transfer has a positive component along it.
    tol : float
        Stop when a Newton step changes y, the universal-variable quantity the velocities are
        built from, by less than `tol` of itself.
    max_iter : int
        Iterations allowed on the root of the time-of-flight equation.

    Returns
    -------
    v1, v2 : ndarray, shape (3,)
        Inertial velocities (m/s) at `r1` and `r2`.

    Raises
    ------
    ValueError
        Positions collinear with the centre (|sin of the transfer angle| below 1e-12), `normal`
        in the transfer plane, `tof` not positive, any input zero or not finite, or a transfer
        too short for rounding to leave the velocities within 1e-9 (messages starting with
        "tof too short").
    RuntimeError
        No root within `max_iter` iterations.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    normal = np.asarray(normal, dtype=float)
    for name, vector in (("r1", r1), ("r2", r2), ("normal", normal)):
        if vector.shape != (3,):
            raise ValueError(f"{name} must have shape (3,)")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be finite")
    tof = float(tof)
    mu = float(mu)
    if not 0.0 < tof < math.inf:
        raise ValueError("tof must be positive and finite")
    if not 0.0 < mu < math.inf:
        raise ValueError("mu must be positive and finite")
    n1 = float(np.linalg.norm(r1))
    n2 = float(np.linalg.norm(r2))
    if n1 == 0.0 or n2 == 0.0:
        raise ValueError("r1 and r2 must be nonzero")
    cross = np.cross(r1, r2)
    sin_short = float(np.linalg.norm(cross)) / (n1 * n2)
    if sin_short < 1e-12:
        raise ValueError("r1 and r2 are collinear with the centre")
    sense = float(cross @ normal)
    if sense == 0.0:
        raise ValueError("normal lies in the transfer plane")
    # The short way round is the arc below pi; the long way is its complement to 2 pi.
    short = math.atan2(sin_short, float(r1 @ r2) / (n1 * n2))
    sign = -1.0 if (sense < 0.0) == prograde else 1.0
    # A = sin(dtheta) sqrt(r1 r2 / (1 - cos dtheta)) through the half angle of the short arc,
    # and y = r1 + r2 + A (z S - 1) / sqrt(C) through the identity
    # (z S - 1) / sqrt(C) = -sqrt(2) cos(sqrt(z) / 2) and the half-angle formulas, so that
    # y = offset + scale q(z) adds two terms of which only one depends on z. The plain forms
    # lose digits in 1 - cos dtheta on short arcs and in y whenever it is small against r1 + r2.
    h = math.cos(0.5 * short)
    A = sign * math.sqrt(2.0 * n1 * n2) * h
    geometric_mean = math.sqrt(n1 * n2)
    half_quarter = math.sin(0.25 * short) ** 2 if sign > 0.0 else math.cos(0.25 * short) ** 2
    offset = (math.sqrt(n1) - math.sqrt(n2)) ** 2 + 4.0 * geometric_mean * half_quarter
    scale = 4.0 * geometric_mean * sign * h
    sqrt_mu_tof = math.sqrt(mu) * tof

    def residual_and_slope(z: float) -> tuple[float, float, float, float]:
        """(F(z), dF/dz, y(z), dy/dz); F and the slopes are NaN where y <= 0."""
        q, dq = _half_angle_terms(z)
        y = offset + scale * q
        if y <= 0.0:
            return math.nan, math.nan, y, math.nan
        dy = scale * dq
        c2, c3 = _stumpff(z)
        dc2, dc3 = _stumpff_derivatives(z, c2, c3)
        ratio = y / c2
        residual = ratio**1.5 * c3 + A * math.sqrt(y) - sqrt_mu_tof
        slope = (
            1.5 * math.sqrt(ratio) * (dy - ratio * dc2) / c2 * c3
            + ratio**1.5 * dc3
            + 0.5 * A * dy / math.sqrt(y)
        )
        return residual, slope, y, dy

    # F rises with z wherever y > 0 and tends to +inf at 4 pi^2, where C(z) vanishes. For A > 0
    # the admissible range starts where y crosses zero; for A < 0, y stays positive and F falls
    # toward -sqrt(mu) tof. Either way every z with y <= 0 or F < 0 lies below the root. The
    # search doubles down from -1 because the roots of ordinary transfers lie between -10 and
    # 4 pi^2, so the bracket starts tight where the bisection fallback needs it.
    hi = math.nextafter(_Z_MAX, 0.0)
    lo = -1.0
    while True:
        residual, _, y, _ = residual_and_slope(lo)
        if y <= 0.0 or residual < 0.0:
            break
        lo *= 2.0
        if lo < _Z_SEARCH_FLOOR:
            raise ValueError("tof too short: the root lies below z = -1e4")
    z = 0.0
    # Size of the Newton step that produced z and |F| where it started; inf after a bisection.
    newton_step = residual_before = math.inf
    for _ in range(max_iter):
        residual, slope, y, dy = residual_and_slope(z)
        if y <= 0.0 or residual < 0.0:
            lo = max(lo, z)
        else:
            hi = min(hi, z)
        # The velocities are built from y, so the stop asks that the last Newton step changed
        # y by less than tol of itself. A tolerance on z alone fails on short transfers, where
        # the root sits within 1e-8 of the zero of y and a step of that size leaves F at half
        # its size. The stop also waits for the evaluation, since the step can still cross the
        # zero of y, and asks for an admissible point.
        if y > 0.0 and newton_step * abs(dy) < tol * y:
            break
        if hi - lo <= 4.0 * math.ulp(max(abs(lo), abs(hi))):
            # F is at its rounding floor and the bracket has closed on the root; its upper end
            # is on the side where y > 0.
            z = hi
            break
        newton = z - residual / slope if y > 0.0 and 0.0 < slope < math.inf else math.nan
        # Safeguarded Newton (Press et al., Numerical Recipes, 3rd ed., sec. 9.4), with the
        # progress measured on F: bisect when a step leaves the bracket or the last Newton step
        # did not halve |F|. Measuring the steps instead rejects every Newton step when the root
        # sits at one end of the bracket, since the step then equals the last bisection. A point
        # where y <= 0 has no slope and only tightens the bracket. F rises with z, so a slope
        # that is not positive and finite is rounding noise, as on fast long-way transfers
        # where F is a small difference of large terms, and the step bisects.
        if lo < newton < hi and abs(residual) <= 0.5 * residual_before:
            step, newton_step, residual_before = newton, abs(newton - z), abs(residual)
        else:
            step, newton_step, residual_before = 0.5 * (lo + hi), math.inf, math.inf
        z = step
    else:
        raise RuntimeError("lambert's equation did not converge")
    _, _, y, dy = residual_and_slope(z)
    if y <= 0.0:
        raise ValueError("transfer angle too close to a full revolution to resolve")
    # Short transfers push the root to where y is the small remainder of its two terms, and
    # fast long-way transfers to where F is the small remainder of its two. Each term carries a
    # few roundings, counted as four, and y also moves by dy over the ulp of z; the error in y
    # reaches the velocities through g at half its relative size, and the error in F through
    # the time of flight.
    q, _ = _half_angle_terms(z)
    y_rounding = 4.0 * _EPS * max(offset, abs(scale * q)) + abs(dy) * math.ulp(z)
    f_term = abs(A) * math.sqrt(y)
    f_rounding = 4.0 * _EPS * max(f_term, abs(sqrt_mu_tof - f_term))
    if y_rounding / (2.0 * y) + f_rounding / sqrt_mu_tof > _ROUNDING_LIMIT:
        raise ValueError("tof too short to resolve the velocities to 1e-9")
    f = 1.0 - y / n1
    g = A * math.sqrt(y / mu)
    g_dot = 1.0 - y / n2
    return (r2 - f * r1) / g, (g_dot * r2 - r1) / g
