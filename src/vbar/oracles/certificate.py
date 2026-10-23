"""Primer-vector certificates of impulsive CW plans and the indirect finite-thrust optimum.

An impulsive plan that reaches its target is fuel-optimal in the actuator norm when a terminal
multiplier nu gives a primer p(t) = B^T Phi(t_f - t)^T nu that equals the burn direction at every
burn and stays within 1 in the dual norm on [0, t_f] (D. F. Lawden, Optimal Trajectories for
Space Navigation, 1963). On linear dynamics these conditions are sufficient (Longuski, Guzman and
Prussing, Optimal Control with Aerospace Applications, 2014, section 10.5): the plan's cost then
equals the impulsive lower bound D(nu) / sup |p|_*. The supremum is bounded over cells by a
second-order Taylor bound and refined where it is loose, so it holds between evaluation points.

With an acceleration limit the optimum thrusts at full level where |p|_* exceeds 1 and coasts
elsewhere; `indirect_optimum` finds the multiplier whose switching control lands on the target by
maximizing the concave dual function of `vbar.oracles.primer`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from ..dynamics import cw_discrete, cw_system

_NORMS = ("l1", "l2")
# Certificate tolerance: on sup |p|_* above 1, on the burn directions, and on the terminal miss
# relative to the size of the terms that make up the final state.
PRIMER_TOL = 1e-9
_EPS = float(np.finfo(float).eps)
_FLOOR = 64 * _EPS
# Refinement stops when the certified supremum is within this of the largest sampled value,
# relative to the larger of 1 and that value.
_SUP_GAP = 1e-10
_SUP_ROUNDS = 80
_CELLS = 256
# Exchange rounds of the multiplier LPs, the samples searched for new maxima in each, and the
# excess over the LP's level, relative to the larger of 1 and the level, that adds a row.
_EXCHANGE_ROUNDS = 20
_EXCHANGE_CELLS = 2048
_EXCHANGE_TOL = 1e-10
# HiGHS's smallest primal and dual feasibility tolerances.
_LP_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
# Singular values below this fraction of the largest leave a multiplier direction free.
_RANK_TOL = 1e-10
_CUBE = np.array(
    [
        (i, j, k)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if (i, j, k) != (0, 0, 0)
    ],
    dtype=float,
)
_BALL_DIRECTIONS = _CUBE / np.linalg.norm(_CUBE, axis=1, keepdims=True)
# Gauss-Legendre nodes per piece of a ball thrust arc, and the longest piece in n t.
_GAUSS = np.polynomial.legendre.leggauss(24)
_PIECE = 0.25
# The indirect solve stops at this terminal miss relative to the distance to cover, and fails
# above 1e3 times it.
_INDIRECT_ITERATIONS = 200
_INDIRECT_TOL = 1e-12


@dataclass(frozen=True)
class PrimerCertificate:
    """Lawden's conditions checked on an impulsive plan, with the bound they imply.

    Parameters
    ----------
    optimal : bool
        True when the plan reaches its target and the primer of `nu` equals the burn direction at
        every burn and stays within 1 + 1e-9 in the dual norm on [0, t_f]; the plan is then
        fuel-optimal in the actuator norm on CW motion.
    primer_max : float
        Certified upper bound of sup |p(t)|_* over [0, t_f] for `nu` (dimensionless): the largest
        component for box thrusters, the 2-norm for one engine.
    lower : float
        Impulsive lower bound D(nu) / `primer_max` (m/s) on the delta-v from the start to the
        target in the actuator norm, valid for any acceleration history; 0 when negative.
    cost : float
        Delta-v of the plan (m/s): the sum of 1-norms (box) or 2-norms (one engine) of the
        impulses.
    hamiltonian_spread : float
        Spread of H = -nu^T Phi(t_f - t) A x along the coasts of the plan, relative to the
        largest of the products |nu_i Phi_ij A_jk x_k| summed over the indices. H is constant
        on each coast and steps by p'(t_j)^T dv_j at a burn, so the spread is at rounding level
        when the primer is stationary along every interior burn.
    nu : numpy.ndarray, shape (6,)
        Terminal multiplier: position part in 1/s, velocity part dimensionless.
    reason : str
        First condition that fails: "miss" (the plan does not reach the target), "direction" (no
        multiplier puts the primer along every burn), "primer_max" (the least primer peak found
        exceeds 1 + 1e-9); empty when `optimal`.
    """

    optimal: bool
    primer_max: float
    lower: float
    cost: float
    hamiltonian_spread: float
    nu: np.ndarray
    reason: str


def _scale(n: float) -> np.ndarray:
    """Multiplier units: nu = scale * omega makes every primer coefficient of order one."""
    return np.array([n, n, n, 1.0, 1.0, 1.0])


def _sample(n: float, nu: np.ndarray, tf: float, t: np.ndarray) -> np.ndarray:
    """Primer, its first two time derivatives and their rounding bounds at times t, (m, 6, 3)."""
    a_mat, b_mat = cw_system(n)
    ab = a_mat @ b_mat
    aab = a_mat @ ab
    phi, _ = cw_discrete(n, tf - t)
    lam = np.einsum("mij,i->mj", phi, nu)
    size = np.einsum("mij,i->mj", np.abs(phi), np.abs(nu))
    # d/dt Phi(tf - t) = -Phi(tf - t) A, so the k-th derivative is (-1)^k (A^k B)^T lam.
    return np.stack(
        [
            lam[:, 3:],
            -lam @ ab,
            lam @ aab,
            _FLOOR * size[:, 3:],
            _FLOOR * size @ np.abs(ab),
            _FLOOR * size @ np.abs(aab),
        ],
        axis=1,
    )


def _amplitudes(n: float, nu: np.ndarray) -> np.ndarray:
    """Bounds over all t on |p_i'|, |p_i''| and |p_i'''| (rows), from the closed form.

    The second and higher derivatives are sinusoids in n tau; p_y' adds the constant -3 nu_y.
    """
    nx, ny, nz, vx, vy, vz = nu
    ax = math.hypot(nx - 2 * n * vy, 2 * ny + n * vx)
    az = math.hypot(nz, n * vz)
    tail = np.array([ax, 2 * ax, az])
    return np.array([[ax, 2 * ax + 3 * abs(ny), az], n * tail, n * n * tail]) * (1 + 1e-12)


def _functions(norm: str, s: np.ndarray) -> np.ndarray:
    """Smooth functions whose largest value is |p|_inf (l1) or |p|_2^2 (l2), (6, m, F).

    Rows: value, first and second time derivatives, then a rounding bound for each.
    """
    p, p1, p2, e0, e1, e2 = np.moveaxis(s, 1, 0)
    if norm == "l1":
        g = np.stack([p, p1, p2])
        e = np.stack([e0, e1, e2])
        return np.concatenate([np.concatenate([g, -g], axis=2), np.concatenate([e, e], axis=2)])
    f = np.stack([(p * p).sum(1), 2 * (p * p1).sum(1), 2 * (p1 * p1 + p * p2).sum(1)])
    ap, ap1, ap2 = np.abs(p), np.abs(p1), np.abs(p2)
    ef = np.stack(
        [
            (2 * ap * e0 + e0 * e0).sum(1),
            2 * (ap1 * e0 + ap * e1 + e0 * e1).sum(1),
            2 * (2 * ap1 * e1 + ap2 * e0 + ap * e2 + e1 * e1 + e0 * e2).sum(1),
        ]
    )
    return np.concatenate([f, ef + 4 * _EPS * np.abs(f)])[..., None]


def _third(norm: str, s: np.ndarray, amp: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Per-cell bound on the third time derivative of each function, (cells, F)."""
    if norm == "l1":
        return np.tile(amp[2], 2)[None, :]

    def within(k: int) -> np.ndarray:
        size = np.abs(s[:, k])
        return np.linalg.norm(0.5 * (size[:-1] + size[1:]) + 0.5 * amp[k] * h, axis=1)

    # (|p|^2)''' = 2 (3 p' . p'' + p . p'''), bounded by Cauchy-Schwarz on each cell.
    return (2 * (3 * within(1) * within(2) + within(0) * np.linalg.norm(amp[2])))[:, None]


def _rise(g: np.ndarray, slope: np.ndarray, curv: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Largest value over s in [0, h] of g + slope s + curv s^2 / 2."""
    concave = curv < 0.0
    vertex = np.clip(np.divide(slope, -curv, out=np.zeros_like(g), where=concave), 0.0, h)
    vertex = np.where(concave, vertex, 0.0)
    far = g + h * (slope + 0.5 * curv * h)
    return np.maximum(np.maximum(g, far), g + vertex * (slope + 0.5 * curv * vertex))


def _cell_bounds(
    norm: str, s: np.ndarray, amp: np.ndarray, h: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Function values (m, F), and per cell an upper and a lower bound of each function.

    With g''' bounded by K3 on a cell, g'' stays below the mean of its end values plus K3 h / 2,
    and Taylor's theorem from either end bounds g by a quadratic; the smaller of the two maxima
    is the cell's top. Rounding bounds are added to the top and subtracted from the bottom.
    """
    g, g1, g2, e0, e1, e2 = _functions(norm, s)
    k3 = _third(norm, s, amp, h)
    mean2 = 0.5 * (g2[:-1] + g2[1:])
    up, down = mean2 + 0.5 * k3 * h, -mean2 + 0.5 * k3 * h
    top = np.minimum(_rise(g[:-1], g1[:-1], up, h), _rise(g[1:], -g1[1:], up, h))
    bottom = -np.minimum(_rise(-g[:-1], -g1[:-1], down, h), _rise(-g[1:], g1[1:], down, h))
    err = [np.maximum(e[:-1], e[1:]) for e in (e0, e1, e2, np.abs(g), np.abs(g1))]
    slack = err[0] + h * err[1] + 0.5 * h * h * err[2]
    slack += 4 * _EPS * (err[3] + h * err[4] + h * h * (np.abs(mean2) + k3 * h))
    return g, top + slack, bottom - slack, g1, mean2, k3


def _dual_sup(n: float, nu: np.ndarray, tf: float, norm: str, knots: np.ndarray) -> float:
    """Certified upper bound of sup |p(t)|_* on [0, tf], within _SUP_GAP of a sampled value."""
    amp = _amplitudes(n, nu)
    t = np.unique(np.r_[np.linspace(0.0, tf, _CELLS + 1), knots])
    s = _sample(n, nu, tf, t)
    smallest = 16 * _EPS * tf
    for _ in range(_SUP_ROUNDS):
        h = np.diff(t)[:, None]
        g, top, *_ = _cell_bounds(norm, s, amp, h)
        seen = float(g.max())
        cell_top = top.max(axis=1)
        loose = (cell_top > seen + _SUP_GAP * max(1.0, seen)) & (h[:, 0] > smallest)
        if not loose.any():
            break
        mid = 0.5 * (t[:-1] + t[1:])[loose]
        t = np.r_[t, mid]
        s = np.concatenate([s, _sample(n, nu, tf, mid)])
        order = np.argsort(t)
        t, s = t[order], s[order]
    upper = max(float(cell_top.max()), seen, 0.0)
    return math.sqrt(upper) if norm == "l2" else upper


def _primer_rows(n: float, tf: float, t: np.ndarray) -> np.ndarray:
    """Primer at times t as a linear map of the scaled multiplier, (m, 3, 6)."""
    phi, _ = cw_discrete(n, tf - t)
    return np.swapaxes(phi[:, :, 3:6], 1, 2) * _scale(n)


def _slope_rows(n: float, tf: float, t: np.ndarray) -> np.ndarray:
    """Primer rate over n at times t as a linear map of the scaled multiplier, (m, 3, 6)."""
    a_mat, b_mat = cw_system(n)
    phi, _ = cw_discrete(n, tf - t)
    return -np.swapaxes(phi @ (a_mat @ b_mat), 1, 2) * _scale(n) / n


def _affine_family(rows: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares solution of rows w = rhs, its largest residual, and a null-space basis."""
    if len(rows) == 0:
        return np.zeros(6), 0.0, np.eye(6)
    u, sv, vt = np.linalg.svd(rows)
    rank = int((sv > _RANK_TOL * sv[0]).sum()) if sv[0] > 0.0 else 0
    w = vt[:rank].T @ ((u[:, :rank].T @ rhs) / sv[:rank])
    return w, float(np.abs(rows @ w - rhs).max()), vt[rank:].T


def _new_maxima(
    n: float, omega: np.ndarray, tf: float, norm: str, level: float, knots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of the dual-norm functions above `level`, with their dual directions."""
    nu = _scale(n) * omega
    t = np.unique(np.r_[np.linspace(0.0, tf, _EXCHANGE_CELLS + 1), knots])
    g = _functions(norm, _sample(n, nu, tf, t))[0]
    up = np.r_[np.full((1, g.shape[1]), True), g[1:] >= g[:-1]]
    down = np.r_[g[:-1] >= g[1:], np.full((1, g.shape[1]), True)]
    k, f = np.nonzero(up & down)
    at = t[k]
    lo, hi = t[np.maximum(k - 1, 0)], t[np.minimum(k + 1, len(t) - 1)]
    idx = np.arange(len(at))
    for _ in range(4):
        _, g1, g2 = _functions(norm, _sample(n, nu, tf, at))[:3]
        g1, g2 = g1[idx, f], g2[idx, f]
        step = np.divide(g1, -g2, out=np.zeros_like(g1), where=g2 < 0.0)
        at = np.clip(at + step, lo, hi)
    s = _sample(n, nu, tf, at)
    value = _functions(norm, s)[0][idx, f]
    if norm == "l2":
        value = np.sqrt(value)
    keep = value > level + _EXCHANGE_TOL * max(1.0, level)
    if norm == "l1":
        dirs = np.r_[np.eye(3), -np.eye(3)][f[keep]]
    else:
        p = s[keep, 0]
        dirs = p / np.linalg.norm(p, axis=1, keepdims=True)
    return at[keep], dirs


def _room(n: float, t: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Weight min(1, (n dist(t, knots))^2) of the margin below 1, zero at the burns."""
    if len(knots) == 0:
        return np.ones_like(t)
    gap = np.abs(t[:, None] - knots[None, :]).min(axis=1)
    return np.minimum(1.0, (n * gap) ** 2)


def _exchange(
    n: float,
    tf: float,
    norm: str,
    base: np.ndarray,
    null: np.ndarray,
    slope_ub: np.ndarray,
    knots: np.ndarray,
    *,
    margin: bool,
) -> tuple[np.ndarray, float] | None:
    """Scaled multiplier on base + null w chosen by a grid LP refined by an exchange method.

    With `margin` the LP maximizes e subject to d^T p(t) + e room(t) <= 1, which keeps the
    primer inside the unit ball of the dual norm with the most room away from the burns; else
    it minimizes z subject to d^T p(t) <= z, the least peak. Both hold over the dual directions
    d and slope_ub omega <= 0. Local maxima of the solution's primer above the LP's level join
    the rows until none is left (Hettich and Kortanek, Semi-infinite programming, SIAM Review,
    1993). Returns the multiplier and e or z; None when the LP fails.
    """
    k = null.shape[1]
    t = np.unique(np.r_[np.linspace(0.0, tf, _CELLS + 1), knots])
    dirs = np.r_[np.eye(3), -np.eye(3)] if norm == "l1" else _BALL_DIRECTIONS

    def rows(t: np.ndarray, dirs: np.ndarray, per_time: bool):
        prim = _primer_rows(n, tf, t)
        if per_time:
            lin = np.einsum("mk,mkj->mj", dirs, prim)
            at = t
        else:
            lin = np.einsum("dk,mkj->mdj", dirs, prim).reshape(-1, 6)
            at = np.repeat(t, len(dirs))
        extra = _room(n, at, knots) if margin else -np.ones(len(at))
        rhs = (1.0 if margin else 0.0) - lin @ base
        return np.c_[lin @ null, extra], rhs

    a_ub, b_ub = rows(t, dirs, False)
    if len(slope_ub):
        a_ub = np.r_[a_ub, np.c_[slope_ub @ null, np.zeros(len(slope_ub))]]
        b_ub = np.r_[b_ub, -slope_ub @ base]
    cost = np.r_[np.zeros(k), -1.0 if margin else 1.0]
    bounds = [(None, None)] * k + [(None, 1.0 if margin else None)]
    found = None
    for _ in range(_EXCHANGE_ROUNDS):
        res = linprog(
            cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=_LP_OPTIONS
        )
        if res.status != 0:
            return found
        found = (base + null @ res.x[:k], float(res.x[k]))
        level = 1.0 if margin else found[1]
        at, extra = _new_maxima(n, found[0], tf, norm, level, knots)
        if len(at) == 0:
            break
        more, rhs = rows(at, extra, True)
        a_ub, b_ub = np.r_[a_ub, more], np.r_[b_ub, rhs]
    return found


def _burn_rows(
    n: float, tf: float, times: np.ndarray, dvs: np.ndarray, norm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Burn rows in the scaled multiplier: directions and targets, rates, end rates (<= 0)."""
    burning = np.any(dvs != 0.0, axis=1)
    t, dv = times[burning], dvs[burning]
    prim = _primer_rows(n, tf, t)
    rate = _slope_rows(n, tf, t)
    if norm == "l1":
        j, i = np.nonzero(dv)
        sign = np.sign(dv[j, i])
        direction, target = prim[j, i], sign
        slope = sign[:, None] * rate[j, i]
        at = t[j]
    else:
        u = dv / np.linalg.norm(dv, axis=1, keepdims=True)
        direction, target = prim.reshape(-1, 6), u.ravel()
        slope = np.einsum("mk,mkj->mj", u, rate)
        at = t
    # |p|_* cannot rise above 1 on either side of an interior burn, nor into the interval at
    # the ends: the rate along the burn is zero inside, at most 0 at t = 0, at least 0 at tf.
    inner = (at > 0.0) & (at < tf)
    ends = np.r_[slope[at == 0.0], -slope[at == tf]]
    return direction, target, slope[inner], ends


def _coast_spread(
    n: float, nu: np.ndarray, tf: float, x0: np.ndarray, times: np.ndarray, dvs: np.ndarray
) -> float:
    """Spread of the coast Hamiltonian -nu^T Phi(tf - t) A x relative to its largest term."""
    a_mat, _ = cw_system(n)
    stops = np.r_[times, tf]
    t_list, x_list = [], []
    x, now = x0.copy(), 0.0
    for stop, dv in zip(stops, np.r_[dvs, np.zeros((1, 3))], strict=True):
        if stop > now:
            span = np.linspace(now, stop, 3)
            phi, _ = cw_discrete(n, span - now)
            t_list.append(span)
            x_list.append(phi @ x)
            x = x_list[-1][-1].copy()
        x[3:] += dv
        now = stop
    if not t_list:
        return 0.0
    t = np.concatenate(t_list)
    states = np.concatenate(x_list)
    phi, _ = cw_discrete(n, tf - t)
    h = -np.einsum("i,mij,jk,mk->m", nu, phi, a_mat, states)
    terms = np.einsum("i,mij,jk,mk->m", np.abs(nu), np.abs(phi), np.abs(a_mat), np.abs(states))
    largest = float(terms.max())
    return float(np.ptp(h) / largest) if largest > 0.0 else 0.0


def _check_norm(norm: str) -> None:
    """Refuse a norm other than "l1" and "l2"."""
    if norm not in _NORMS:
        raise ValueError(f"unknown norm {norm!r}, expected 'l1' or 'l2'")


def _state(x: ArrayLike, name: str) -> np.ndarray:
    """Hill state as 6 finite floats."""
    v = np.asarray(x, dtype=float)
    if v.shape != (6,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be 6 finite numbers")
    return v


def _positive(value: float, name: str) -> float:
    """A positive finite float."""
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def certify_impulsive(
    n: float,
    start: ArrayLike,
    impulses: Iterable[tuple[float, ArrayLike]],
    target: ArrayLike,
    *,
    tf: float,
    norm: str,
) -> PrimerCertificate:
    """Check Lawden's conditions on an impulsive CW plan in the actuator's dual norm.

    The multiplier is fitted to the plan's burns: the primer must equal sign(dv_i) on each
    burning axis ("l1", box thrusters) or dv / |dv|_2 ("l2", one engine). Among the multipliers
    that fit, the one kept has the least primer peak found: when the burns leave it free, a grid
    LP with the primer rate zero at interior burns, refined by an exchange method, chooses it.
    The plan is certified when it reaches the target to 1e-9 of the size of the terms of its
    final state and the certified peak is at most 1 + 1e-9.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    start : array_like, shape (6,)
        Hill state at t = 0 (m, m/s).
    impulses : iterable of (float, array_like) pairs
        Burn times (s, strictly increasing within [0, tf]) and velocity changes (m/s, Hill
        axes), applied instantly.
    target : array_like, shape (6,)
        Hill state required at tf (m, m/s).
    tf : float
        Final time (s).
    norm : {"l1", "l2"}
        Actuator norm: "l1" for box thrusters (cost sum of |dv|_1, dual norm max |p_i|), "l2"
        for one engine (cost sum of |dv|_2, dual norm |p|_2).

    Returns
    -------
    PrimerCertificate
        The verdict, the fitted multiplier, the certified primer peak and the lower bound.

    Raises
    ------
    ValueError
        If `norm` is unknown, `n` or `tf` is not positive and finite, a state or impulse is not
        finite or of the wrong shape, the burn times are not strictly increasing within
        [0, tf], or an impulse norm, the plan cost, the propagated final state or the terminal
        miss overflows.
    """
    _check_norm(norm)
    n = _positive(n, "n")
    tf = _positive(tf, "tf")
    x0, xg = _state(start, "start"), _state(target, "target")
    pairs = [(float(t), np.asarray(dv, dtype=float)) for t, dv in impulses]
    if any(dv.shape != (3,) for _, dv in pairs):
        raise ValueError("each impulse must be a (time, 3-vector) pair")
    times = np.array([t for t, _ in pairs])
    dvs = np.array([dv for _, dv in pairs]).reshape(len(pairs), 3)
    if not (np.all(np.isfinite(dvs)) and np.all(np.isfinite(times))):
        raise ValueError("impulse times and velocity changes must be finite")
    if len(times) and (times[0] < 0.0 or times[-1] > tf):
        raise ValueError("burn times must lie within [0, tf]")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("burn times must be strictly increasing")

    order = 1 if norm == "l1" else 2
    # Finite inputs can still overflow below; an infinite miss would compare as not above an
    # infinite size and pass as reached, so each result is checked and numpy's warnings muted.
    with np.errstate(over="ignore", invalid="ignore"):
        burn_sizes = np.linalg.norm(dvs, ord=order, axis=1)
        if not np.all(np.isfinite(burn_sizes)):
            raise ValueError("impulse norms must be finite")
        cost = float(burn_sizes.sum())
        if not math.isfinite(cost):
            raise ValueError("plan cost must be finite")
        phi_tf, _ = cw_discrete(n, tf)
        drift = phi_tf @ x0
        kicks = np.einsum("mij,mj->mi", cw_discrete(n, tf - times)[0][:, :, 3:6], dvs)
        final = drift + kicks.sum(axis=0)
        if not np.all(np.isfinite(final)):
            raise ValueError("propagated final state must be finite")
        # Velocities over n, so the miss compares metres with metres.
        hill = np.r_[np.ones(3), np.full(3, 1.0 / n)]
        size = float(
            np.linalg.norm(hill * drift)
            + np.linalg.norm(hill * xg)
            + np.linalg.norm(hill * kicks, axis=1).sum()
        )
        miss = float(np.linalg.norm(hill * (final - xg)))
        if not (math.isfinite(size) and math.isfinite(miss)):
            raise ValueError("terminal miss and final state size must be finite")
    missed = miss > PRIMER_TOL * size

    direction, signs, slope_eq, slope_ub = _burn_rows(n, tf, times, dvs, norm)
    base, residual, null = _affine_family(direction, signs)
    aligned = residual <= PRIMER_TOL

    def peak(omega: np.ndarray) -> float:
        return _dual_sup(n, _scale(n) * omega, tf, norm, times)

    def least(base: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, float]:
        best = (base, peak(base))
        if null.shape[1]:
            found = _exchange(n, tf, norm, base, null, np.zeros((0, 6)), times, margin=False)
            if found is not None:
                top = peak(found[0])
                if top < best[1]:
                    best = (found[0], top)
        return best

    # First look for a certifying multiplier: the primer rate vanishes at interior burns and
    # points inward at the ends. If none exists, keep the least peak among those that fit.
    omega, top = base, math.inf
    if aligned and (len(slope_eq) or len(slope_ub)):
        base_a, residual_a, null_a = _affine_family(
            np.r_[direction, slope_eq], np.r_[signs, np.zeros(len(slope_eq))]
        )
        if residual_a <= PRIMER_TOL:
            candidate: np.ndarray | None = base_a
            if null_a.shape[1]:
                found = _exchange(n, tf, norm, base_a, null_a, slope_ub, times, margin=True)
                # A negative margin means no multiplier keeps even the grid values within 1.
                if found is not None:
                    candidate = found[0] if found[1] > -_EXCHANGE_TOL else None
            if candidate is not None:
                omega, top = candidate, peak(candidate)
    if top > 1.0 + PRIMER_TOL:
        omega, top = least(base, null) if aligned else (base, peak(base))

    nu = _scale(n) * omega
    d = xg - drift
    dual = float(nu @ d) - _FLOOR * float(np.abs(nu * d).sum())
    lower = max(0.0, dual / top) if top > 0.0 else 0.0
    if missed:
        reason = "miss"
    elif not aligned:
        reason = "direction"
    elif top > 1.0 + PRIMER_TOL:
        reason = "primer_max"
    else:
        reason = ""
    return PrimerCertificate(
        optimal=reason == "",
        primer_max=top,
        lower=lower,
        cost=cost,
        hamiltonian_spread=_coast_spread(n, nu, tf, x0, times, dvs),
        nu=nu,
        reason=reason,
    )


def hamiltonian(
    n: float,
    nu: ArrayLike,
    tf: float,
    t: ArrayLike,
    x: ArrayLike,
    a: ArrayLike,
    *,
    norm: str,
) -> np.ndarray:
    """Hamiltonian H = |a| - nu^T Phi(tf - t) (A x + B a) of the minimum-fuel CW problem.

    The costate is -Phi(tf - t)^T nu. Along an optimal switching control H is constant: between
    switches the time derivatives of its two terms cancel, and at a switch the step in |a| is
    matched by the step in nu^T Phi B a because |p|_* = 1 there.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    nu : array_like, shape (6,)
        Terminal multiplier: position part in 1/s, velocity part dimensionless.
    tf : float
        Final time (s).
    t : array_like, shape (m,)
        Times (s).
    x : array_like, shape (m, 6)
        Hill states at `t` (m, m/s).
    a : array_like, shape (m, 3)
        Hill accelerations at `t` (m/s^2).
    norm : {"l1", "l2"}
        Cost of the acceleration: "l1" for box thrusters, "l2" for one engine.

    Returns
    -------
    numpy.ndarray, shape (m,)
        H at each time (m/s^2).

    Raises
    ------
    ValueError
        If `norm` is unknown, `n` is not positive and finite, `tf`, `nu`, `t`, `x` or `a` is
        not finite, or the shapes do not match.
    """
    _check_norm(norm)
    n = _positive(n, "n")
    nu = np.asarray(nu, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    x = np.asarray(x, dtype=float).reshape(-1, 6)
    a = np.asarray(a, dtype=float).reshape(-1, 3)
    if nu.shape != (6,) or not (len(t) == len(x) == len(a)):
        raise ValueError("nu must have 6 entries and t, x, a one row per time")
    if not (math.isfinite(tf) and all(np.all(np.isfinite(v)) for v in (nu, t, x, a))):
        raise ValueError("tf, nu, t, x and a must be finite")
    a_mat, b_mat = cw_system(n)
    costate = -np.einsum("i,mij->mj", nu, cw_discrete(n, tf - t)[0])
    size = np.abs(a).sum(axis=1) if norm == "l1" else np.linalg.norm(a, axis=1)
    return (
        size
        + np.einsum("mj,ji,mi->m", costate, a_mat, x)
        + np.einsum("mj,ji,mi->m", costate, b_mat, a)
    )


def _crossings(n: float, nu: np.ndarray, tf: float, norm: str) -> list[tuple]:
    """Thrust arcs (t0, t1, function) where a dual-norm function exceeds 1, and the end slopes.

    Cells are bisected until each function is certified below 1, above 1 or monotone on each;
    a monotone cell holds at most one crossing, found by safeguarded Newton.
    """
    amp = _amplitudes(n, nu)
    t = np.linspace(0.0, tf, _CELLS + 1)
    s = _sample(n, nu, tf, t)
    smallest = 16 * _EPS * tf
    for _ in range(_SUP_ROUNDS):
        h = np.diff(t)[:, None]
        _, top, bottom, g1, mean2, k3 = _cell_bounds(norm, s, amp, h)
        curv = np.abs(mean2) + 0.5 * k3 * h
        mean1 = 0.5 * (g1[:-1] + g1[1:])
        monotone = (mean1 - 0.5 * curv * h > 0.0) | (mean1 + 0.5 * curv * h < 0.0)
        open_cell = ~((top < 1.0) | (bottom > 1.0) | monotone)
        split = open_cell.any(axis=1) & (h[:, 0] > smallest)
        if not split.any():
            break
        mid = 0.5 * (t[:-1] + t[1:])[split]
        t = np.r_[t, mid]
        s = np.concatenate([s, _sample(n, nu, tf, mid)])
        order = np.argsort(t)
        t, s = t[order], s[order]
    above = _functions(norm, s)[0] > 1.0
    cell, f = np.nonzero(above[:-1] != above[1:])
    lo, hi = t[cell].copy(), t[cell + 1].copy()
    rising = ~above[cell, f]
    root = 0.5 * (lo + hi)
    idx = np.arange(len(root))
    for _ in range(100):
        value, slope, _, noise = _functions(norm, _sample(n, nu, tf, root))[:4]
        value, slope, noise = value[idx, f] - 1.0, slope[idx, f], noise[idx, f]
        past = (value > 0.0) == rising
        hi, lo = np.where(past, root, hi), np.where(past, lo, root)
        step = root - np.divide(value, slope, out=np.full_like(value, np.inf), where=slope != 0.0)
        # A crossing at a cell end leaves the Newton step on the bracket's edge.
        inside = (step >= lo) & (step <= hi)
        new = np.where(inside, step, 0.5 * (lo + hi))
        done = (np.abs(new - root) <= 4 * _EPS * tf) | (np.abs(value) <= noise)
        root = np.where(np.abs(value) <= noise, root, new)
        if done.all():
            break
    slope = _functions(norm, _sample(n, nu, tf, root))[1][idx, f]
    arcs = []
    for k in range(above.shape[1]):
        mine = f == k
        edges, rates = root[mine], slope[mine]
        # The ends of [0, tf] are fixed, not crossings: their rate entries are 0.
        if above[0, k]:
            edges, rates = np.r_[0.0, edges], np.r_[0.0, rates]
        if len(edges) % 2:
            edges, rates = np.r_[edges, tf], np.r_[rates, 0.0]
        for j in range(0, len(edges), 2):
            arcs.append((edges[j], edges[j + 1], k, rates[j], rates[j + 1]))
    return arcs


def _switching(
    n: float, nu: np.ndarray, tf: float, norm: str, accel: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Final-state reach, Jacobian of the reach in nu, and cost of the switching control."""
    reach, jac, cost = np.zeros(6), np.zeros((6, 6)), 0.0
    for t0, t1, k, rate0, rate1 in _crossings(n, nu, tf, norm):
        cost += accel * (t1 - t0)
        if norm == "l1":
            i, sign = k % 3, 1.0 if k < 3 else -1.0
            phi, gam = cw_discrete(n, np.array([tf - t1, t1 - t0, tf - t0]))
            reach += accel * sign * (phi[0] @ gam[1][:, i])
            # A crossing at t* moves by -c^T dnu / (sign p_i'(t*)), c = Phi(tf - t*) B e_i;
            # rate0 and rate1 are the slopes of sign p_i there, 0 at the ends of [0, tf].
            for rate, col, side in (
                (rate1, phi[0][:, 3 + i], 1.0),
                (rate0, phi[2][:, 3 + i], -1.0),
            ):
                if rate != 0.0:
                    jac -= side * accel * np.outer(col, col) / rate
            continue
        nodes, weights = _GAUSS
        pieces = max(1, math.ceil(n * (t1 - t0) / _PIECE))
        edges = np.linspace(t0, t1, pieces + 1)
        half = 0.5 * np.diff(edges)[:, None]
        at = (0.5 * (edges[:-1] + edges[1:])[:, None] + half * nodes).ravel()
        w = (half * weights).ravel()
        cols = cw_discrete(n, tf - at)[0][:, :, 3:6]
        p = np.einsum("mji,j->mi", cols, nu)
        size = np.linalg.norm(p, axis=1)
        u = p / size[:, None]
        reach += accel * np.einsum("m,mji,mi->j", w, cols, u)
        proj = (np.eye(3) - u[:, :, None] * u[:, None, :]) / size[:, None, None]
        jac += accel * np.einsum("m,mji,mik,mlk->jl", w, cols, proj, cols)
        # At a crossing |p| = 1 and t* moves by -2 (C p)^T dnu / (|p|^2)'.
        ends = cw_discrete(n, np.array([tf - t1, tf - t0]))[0][:, :, 3:6]
        for rate, col, side in ((rate1, ends[0], 1.0), (rate0, ends[1], -1.0)):
            if rate != 0.0:
                cp = col @ (col.T @ nu)
                jac -= side * 2 * accel * np.outer(cp, cp) / rate
    return reach, jac, cost


def indirect_optimum(
    n: float,
    start: ArrayLike,
    target: ArrayLike,
    tf: float,
    *,
    accel: float,
    norm: str,
    nu0: ArrayLike,
) -> tuple[float, np.ndarray]:
    """Minimum-fuel finite-thrust CW transfer by the switching control of a terminal multiplier.

    The control thrusts at full level a sign(p_i) on each axis where |p_i| > 1 ("l1", box
    thrusters) or a p / |p|_2 where |p|_2 > 1 ("l2", one engine), with the primer p of the
    multiplier nu. The multiplier that lands on the target maximizes the concave dual function
    nu^T d - a E(nu) of `dual_bound`, whose gradient is the terminal miss; it is found by
    Levenberg-Marquardt steps on that function with the analytic Hessian, switching times by
    safeguarded Newton on certified monotone cells.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    start : array_like, shape (6,)
        Hill state at t = 0 (m, m/s).
    target : array_like, shape (6,)
        Hill state required at `tf` (m, m/s).
    tf : float
        Final time (s).
    accel : float
        Acceleration limit (m/s^2) per axis ("l1") or in the 2-norm ("l2").
    norm : {"l1", "l2"}
        Actuator norm.
    nu0 : array_like, shape (6,)
        Starting multiplier, such as the equality multipliers of a step-grid LP.

    Returns
    -------
    cost : float
        Delta-v of the switching control (m/s): `accel` times the summed thrust time.
    nu : numpy.ndarray, shape (6,)
        The multiplier: position part in 1/s, velocity part dimensionless.

    Raises
    ------
    ValueError
        If `norm` is unknown, `n`, `tf` or `accel` is not positive and finite, or a state or
        `nu0` is not 6 finite numbers.
    RuntimeError
        If the terminal miss does not fall below 1e-9 of the distance to cover (velocities over
        n), as when the target is out of reach at this acceleration.
    """
    _check_norm(norm)
    n = _positive(n, "n")
    tf = _positive(tf, "tf")
    accel = _positive(accel, "accel")
    x0, xg = _state(start, "start"), _state(target, "target")
    scale = _scale(n)
    omega = _state(nu0, "nu0") / scale
    hill = np.r_[np.ones(3), np.full(3, 1.0 / n)]
    d = xg - cw_discrete(n, tf)[0] @ x0
    distance = float(np.linalg.norm(hill * d))
    if distance == 0.0:
        return 0.0, np.zeros(6)

    def evaluate(omega: np.ndarray):
        nu = scale * omega
        reach, jac, cost = _switching(n, nu, tf, norm, accel)
        # nu^T d - a E(nu) = nu^T (d - reach) + cost for the switching control of nu.
        return float(nu @ (d - reach)) + cost, reach, jac, cost

    value, reach, jac, cost = evaluate(omega)
    damping = None
    for _ in range(_INDIRECT_ITERATIONS):
        miss = float(np.linalg.norm(hill * (reach - d)))
        if miss <= _INDIRECT_TOL * distance:
            break
        grad = scale * (d - reach)
        hess = scale[:, None] * jac * scale
        if damping is None:
            damping = 1e-6 * max(float(np.abs(np.diag(hess)).max()), float(np.abs(grad).max()))
        while damping < 1e30:
            step = np.linalg.solve(hess + damping * np.eye(6), grad)
            trial = evaluate(omega + step)
            if trial[0] > value or np.linalg.norm(hill * (trial[1] - d)) < miss:
                omega, (value, reach, jac, cost) = omega + step, trial
                damping /= 4
                break
            damping *= 4
        else:
            break
    if np.linalg.norm(hill * (reach - d)) > 1e3 * _INDIRECT_TOL * distance:
        raise RuntimeError("indirect solve did not reach the target")
    return cost, scale * omega
