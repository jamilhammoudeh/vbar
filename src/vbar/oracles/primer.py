"""Lawden's primer vector and weak-duality lower bounds on CW minimum-fuel transfers.

For constraint times t_1 < ... < t_J and multipliers nu_j, the primer is
p(t) = sum over t_j > t of B^T Phi(t_j - t)^T nu_j with B = [0; I], and every acceleration
history with |a(t)| <= a in the actuator norm that puts the chaser in the capture ball at every
t_j spends at least

    D(nu) - a * integral over [0, t_J] of max(0, |p(t)|_* - 1) dt,
    D(nu) = sum_j nu_j^T (x_g - Phi(t_j) x0) - rho_r |nu_j,r|_2 - rho_v |nu_j,v|_2,

on exact CW motion, where |.|_* is the dual of the actuator norm: l_inf per axis for box thrusters
(cost l1), l2 for a ball (Boyd and Vandenberghe, Convex Optimization, 2004, section 5.1.3 and
example 3.26). Without an acceleration limit the bound is D(nu) / sup |p|_* (D. F. Lawden, Optimal
Trajectories for Space Navigation, 1963). Both hold for any nu; the integral and the supremum are
bounded over cells from above, so the result holds between evaluation points too.
"""

from __future__ import annotations

import math
import operator

import numpy as np
from numpy.typing import ArrayLike

from ..agents.base import AgentView
from ..dynamics import cw_discrete
from ..metrics.mission import goal_state

NORMS = {"box": "l1", "ball": "l2"}
# Default cell density when `cells` is not given. The Lipschitz constants are a few times n times
# the primer's size, so the slack L h / 2 is a few times n h / 2 of it, with h = dt / 8.
CELLS_PER_STEP = 8
# Rounding floor of the bound: the evaluated terms carry a few ulps each.
_FLOOR = 64 * np.finfo(float).eps
_EXCESS_INFLATION = 1.0 + 1e-12


def primer(n: float, nu: ArrayLike, tau: ArrayLike) -> np.ndarray:
    """Primer vector B^T Phi(tau)^T nu of a CW multiplier, tau before its constraint time.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    nu : array_like, shape (6,)
        Multiplier of the Hill state: position part in 1/s, velocity part dimensionless, so
        that nu^T x is in m/s.
    tau : array_like
        Time left to the constraint time, t_j - t (s), any shape.

    Returns
    -------
    numpy.ndarray, shape tau.shape + (3,)
        Primer in Hill axes (dimensionless), equal to the velocity part of `nu` at tau = 0.
    """
    phi, _ = cw_discrete(n, np.asarray(tau, dtype=float))
    return np.swapaxes(phi[..., :, 3:6], -1, -2) @ np.asarray(nu, dtype=float)


def _lipschitz(n: float, nu: np.ndarray) -> np.ndarray:
    """Per-axis bounds (1/s) on |dp_i/dt| over all t, from the derivatives of the closed form."""
    rx, ry, rz, vx, vy, vz = np.abs(nu)
    return np.array(
        [
            rx + 2 * ry + n * vx + 2 * n * vy,
            2 * rx + 7 * ry + 2 * n * vx + 4 * n * vy,
            rz + n * vz,
        ]
    )


def _cell_tops(
    n: float, nus: np.ndarray, times: np.ndarray, norm: str, cells: int
) -> list[tuple[float, np.ndarray]]:
    """Cell width (s) and, per cell, a certified upper bound of |p|_* (l1) or |p|_2 (l2).

    Cells align with the constraint times, where p jumps; on [t_(i-1), t_i] the primers of
    t_i, ..., t_J act, evaluated at tau = 0 on the right edge for the left limit at t_i.
    """
    tops = []
    lo = 0.0
    for i, hi in enumerate(times):
        m = max(1, round(cells * (hi - lo) / times[-1]))
        grid = np.linspace(lo, hi, m + 1)
        p = np.zeros((m + 1, 3))
        lip = np.zeros(3)
        for nu, tj in zip(nus[i:], times[i:], strict=True):
            p += primer(n, nu, tj - grid)
            lip += _lipschitz(n, nu)
        if norm == "l1":
            size, slope = np.abs(p), lip
        else:
            # | |p(s)|_2 - |p(t)|_2 | <= |p(s) - p(t)|_2 <= |L|_2 |s - t|.
            size = np.linalg.norm(p, axis=1, keepdims=True)
            slope = np.linalg.norm(lip, keepdims=True)
        h = (hi - lo) / m
        tops.append((h, np.maximum(size[:-1], size[1:]) + 0.5 * slope * h))
        lo = hi
    return tops


def dual_bound(
    n: float,
    view: AgentView,
    start: ArrayLike,
    nus: ArrayLike,
    times: ArrayLike,
    *,
    norm: str,
    accel: float | None = None,
    cells: int | None = None,
) -> float:
    """Certified lower bound (m/s) on the delta-v from `start` into the capture ball at `times`.

    Valid for every acceleration history with |a_i(t)| <= `accel` per Hill axis ("l1", box
    thrusters, delta-v in the 1-norm) or |a(t)|_2 <= `accel` ("l2", a ball, delta-v in the
    2-norm) on [0, t_J], on exact CW motion with no other forcing, and for any multipliers. The
    capture ball is closed: |r - r_g| <= capture_radius and |v - v_g| <= capture_speed.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s), equal to `view.n` to 1e-12 relative.
    view : AgentView
        Scenario description; its goal gives `hold_point` (m, Hill), `capture_radius` (m),
        `capture_speed` (m/s) and optionally `goal_velocity` (m/s, Hill), its `thrust_limit`
        the actuator layout and its `max_accel` the default `accel`.
    start : array_like, shape (6,)
        Hill state at t = 0 (m, m/s).
    nus : array_like, shape (J, 6)
        One multiplier per constraint time, in the units dual to the Hill state.
    times : array_like, shape (J,)
        Constraint times (s), positive and strictly increasing.
    norm : {"l1", "l2"}
        Actuator norm: "l1" for box thrusters, "l2" for a ball; it must match `view.thrust_limit`.
    accel : float, optional
        Acceleration limit (m/s^2) in that norm, `view.max_accel` by default; `numpy.inf` gives
        the impulsive bound D(nu) / sup |p|_*.
    cells : int, optional
        Cells over [0, t_J] for the certified integral and supremum, shared among the intervals
        between constraint times by length (at least one each). Default 8 per `view.dt`.

    Returns
    -------
    float
        The lower bound (m/s); it can be negative for poor multipliers.

    Raises
    ------
    ValueError
        If `norm` is unknown or does not match the layout, the goal has no hold point and capture
        ball, the inputs are not finite or of the wrong shape, `times` is not positive and
        strictly increasing, `n`, `accel` or `cells` is not positive, or `n` differs from
        `view.n` by more than 1e-12 relative.
    """
    if norm not in NORMS.values():
        raise ValueError(f"unknown norm {norm!r}, expected 'l1' or 'l2'")
    if NORMS.get(view.thrust_limit) != norm:
        raise ValueError(f"norm {norm!r} does not match the {view.thrust_limit!r} thrusters")
    if not (math.isfinite(n) and n > 0.0):
        raise ValueError("n must be a positive finite number")
    if not math.isclose(n, view.n, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("n does not match the view's mean motion")
    goal = view.goal
    if not {"hold_point", "capture_radius", "capture_speed"} <= goal.keys():
        raise ValueError("the goal has no hold point and capture ball")
    x0 = np.asarray(start, dtype=float)
    if x0.shape != (6,) or not np.all(np.isfinite(x0)):
        raise ValueError("start must be 6 finite numbers")
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) == 0 or not np.all(np.isfinite(t)):
        raise ValueError("times must be a non-empty sequence of finite numbers")
    if t[0] <= 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be positive and strictly increasing")
    nu = np.asarray(nus, dtype=float)
    if nu.shape != (len(t), 6) or not np.all(np.isfinite(nu)):
        raise ValueError("nus must be finite with shape (len(times), 6)")
    a = view.max_accel if accel is None else accel
    if not a > 0.0:
        raise ValueError("accel must be positive")
    if cells is None:
        cells = max(1, math.ceil(CELLS_PER_STEP * t[-1] / view.dt))
    cells = operator.index(cells)
    if cells < 1:
        raise ValueError("cells must be positive")

    r_g, v_g = goal_state(goal)
    phi, _ = cw_discrete(n, t)
    terms = np.concatenate(
        [
            np.einsum("ji,ji->j", nu, np.r_[r_g, v_g] - phi @ x0),
            -goal["capture_radius"] * np.linalg.norm(nu[:, :3], axis=1),
            -goal["capture_speed"] * np.linalg.norm(nu[:, 3:], axis=1),
        ]
    )
    floor = _FLOOR * np.abs(terms).sum()
    tops = _cell_tops(n, nu, t, norm, cells)
    if math.isinf(a):
        sup = max(float(top.max()) for _, top in tops) * _EXCESS_INFLATION
        # Only all-zero multipliers give a zero supremum, and then D(nu) is zero too.
        return float((terms.sum() - floor) / sup) if sup > 0.0 else 0.0
    excess = sum(h * float(np.maximum(0.0, top - 1.0).sum()) for h, top in tops)
    excess *= _EXCESS_INFLATION
    return float(terms.sum() - a * excess - floor - _FLOOR * a * excess)
