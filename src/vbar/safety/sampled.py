"""Certified keep-out check of a held command and a line search toward a backup command.

A filter that commands u at the start of a step of length dt holds it for the whole step. The
continuous-time barrier conditions say nothing about the state between samples, so the check
here bounds the keep-out margin over the held segment of the exact Clohessy-Wiltshire solution,
and `line_search` moves the command toward a backup that is known to keep the barrier, the
evading maneuver of Ames et al., *Control Barrier Functions: Theory and Applications*, ECC 2019,
sec. III.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from ..constraints import KeepOut, _rounding_allowances, interval_bounds, range_lower_bounds
from ..dynamics.cw import CWDiscretizer
from .tightening import ErrorSet, keep_out_margin


@lru_cache(maxsize=8)
def _discretizer(n: float) -> CWDiscretizer:
    """One matrix cache per mean motion, shared by every call."""
    return CWDiscretizer(n)


def _finite(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be {shape[0]} finite numbers")
    return array


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def held_segment_bound(
    x,
    u,
    dt: float,
    m: int,
    radius: float,
    n: float,
    error: ErrorSet | None = None,
    da_max: float = 0.0,
) -> float:
    """Return a certified lower bound on the keep-out margin over a held step.

    Parameters
    ----------
    x : array_like, shape (6,)
        Hill-frame state at the step start, m and m/s; the estimate when `error` is given.
    u : array_like, shape (3,)
        The commanded acceleration, held for the step, m/s^2, Hill frame.
    dt : float
        The step length, s, positive.
    m : int
        Number of samples over the step, endpoints included, at least 2.
    radius : float
        The keep-out radius R, m, positive.
    n : float
        Mean motion, rad/s, positive.
    error : BoxError, BallError or EllipsoidError, optional
        Bound on the error of `x`, m and m/s; none by default.
    da_max : float
        Bound on the execution error of the held command, a vector of norm at most `da_max`
        constant over the step, m/s^2, nonnegative.

    Returns
    -------
    float
        A lower bound on |r(t)| - R, m, over the step for every true start in x + E and every
        execution error. -inf when the step is too long for the in-step bounds.

    Raises
    ------
    ValueError
        If an argument is malformed or out of range.

    Notes
    -----
    The samples are phi(t_i) x + gamma(t_i) u, exact for the CW model. Each interval gets the
    larger of the first-order bound (m_i + m_{i+1} - |v|_max h) / 2 and the second-order bound
    on |r|^2, as in `vbar.constraints.segment_minimum` without its refinement. The deviation of
    the true trajectory is a CW trajectory from e in E under the constant execution error, so
    at a sample its position is at most g_i = keep_out_margin(Phi_r(t_i), E)
    + da_max sigma_max(Gamma_r(t_i)), and its velocity at most the same with the velocity rows.
    Its position grows at most at its speed W between samples and its speed at most at
    3 n^2 |r| + 2 n |v| + da_max, which over the step gives
    W <= (max gv_i + h (3 n^2 max g_i + da_max) / 2) / (1 - n h - 3 n^2 h^2 / 4); each
    interval's bound is lowered by (g_i + g_{i+1} + W h) / 2.
    """
    x = _finite(x, (6,), "x")
    u = _finite(u, (3,), "u")
    dt = _positive(dt, "dt")
    if isinstance(m, bool) or int(m) != m or m < 2:
        raise ValueError("m must be an integer of at least 2")
    m = int(m)
    keep_out = KeepOut(_positive(radius, "radius"))
    n = _positive(n, "n")
    da_max = float(da_max)
    if not (math.isfinite(da_max) and da_max >= 0.0):
        raise ValueError("da_max must be finite and nonnegative")
    phis, gammas = _discretizer(n).sample_matrices(dt, m)
    states = phis @ x + gammas @ u
    h = dt / (m - 1)
    speed, accel = interval_bounds(states, h, float(np.linalg.norm(u)), n)
    if not np.all(np.isfinite(speed)):
        return -math.inf
    margins = keep_out.margin(states)
    bounds = np.maximum(
        0.5 * (margins[:-1] + margins[1:] - speed * h),
        keep_out.second_order_bounds(states, h, speed, accel),
    )
    rho_min = range_lower_bounds(states, h, speed)
    bounds = bounds - _rounding_allowances(keep_out, dt, states, speed, accel, rho_min)
    if error is None and da_max == 0.0:
        return float(bounds.min())
    return float((bounds - _tightening(n, dt, m, error, da_max)).min())


# Error sets are declared once per filter and checked every step; the key holds the set's id,
# and the entry keeps the set itself so that a reused id cannot match another set.
_TIGHTENING: dict[tuple, tuple[ErrorSet | None, np.ndarray]] = {}
_TIGHTENING_ENTRIES = 256


def _tightening(n: float, dt: float, m: int, error: ErrorSet | None, da_max: float) -> np.ndarray:
    """The bound on the deviation's range over each sample interval, m."""
    key = (n, dt, m, id(error), da_max)
    cached = _TIGHTENING.get(key)
    if cached is not None and cached[0] is error:
        return cached[1]
    phis, gammas = _discretizer(n).sample_matrices(dt, m)
    h = dt / (m - 1)
    g_r = da_max * np.linalg.norm(gammas[:, :3, :], ord=2, axis=(1, 2))
    g_v = da_max * np.linalg.norm(gammas[:, 3:, :], ord=2, axis=(1, 2))
    if error is not None:
        g_r = g_r + np.array([keep_out_margin(phi[:3], error) for phi in phis])
        g_v = g_v + np.array([keep_out_margin(phi[3:], error) for phi in phis])
    denominator = 1.0 - n * h - 0.75 * n * n * h * h
    if denominator <= 0.0:
        tightening = np.full(m - 1, np.inf)
    else:
        w = (float(g_v.max()) + 0.5 * h * (3.0 * n * n * float(g_r.max()) + da_max)) / denominator
        tightening = 0.5 * (g_r[:-1] + g_r[1:] + w * h)
    tightening.flags.writeable = False
    if len(_TIGHTENING) >= _TIGHTENING_ENTRIES:
        _TIGHTENING.clear()
    _TIGHTENING[key] = (error, tightening)
    return tightening


class LineSearch(NamedTuple):
    """The result of `line_search`.

    Attributes
    ----------
    u : ndarray, shape (3,)
        The accepted command theta u_c + (1 - theta) beta, m/s^2.
    theta : float
        The weight on the candidate, in [0, 1]; 0 returns the backup unchecked.
    checks : int
        Calls of `accept` made: 1 when the candidate passed, else 1 + `iterations`.
    """

    u: np.ndarray
    theta: float
    checks: int


def line_search(
    accept: Callable[[np.ndarray], bool], u_c, beta, iterations: int = 12
) -> LineSearch:
    """Return the command nearest the candidate, toward a backup, that `accept` passes.

    Parameters
    ----------
    accept : callable
        accept(u) is True when the held command u (m/s^2) passes the filter's checks.
    u_c : array_like, shape (3,)
        The candidate command, m/s^2.
    beta : array_like, shape (3,)
        The backup command, m/s^2, safe by the barrier's construction.
    iterations : int
        Bisection steps on theta after the candidate fails, nonnegative.

    Returns
    -------
    LineSearch
        theta = 1 when `accept(u_c)`; otherwise the largest theta found by bisection on
        [0, 1] with u = theta u_c + (1 - theta) beta, to 2^-iterations.

    Raises
    ------
    ValueError
        If `u_c` or `beta` is not three finite numbers or `iterations` is negative.

    Notes
    -----
    Bisection finds the largest accepted theta only when the accepted set is an interval
    [0, theta*]; otherwise it returns an accepted theta that need not be the largest. The backup
    itself is never checked here, since its safety is the barrier's theorem.
    """
    u_c = _finite(u_c, (3,), "u_c")
    beta = _finite(beta, (3,), "beta")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if accept(u_c):
        return LineSearch(u_c, 1.0, 1)
    lo, hi = 0.0, 1.0
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        if accept(mid * u_c + (1.0 - mid) * beta):
            lo = mid
        else:
            hi = mid
    return LineSearch(lo * u_c + (1.0 - lo) * beta, lo, 1 + int(iterations))
