"""Fuel oracle on the CW model: certified lower bounds and verified plans, at a fixed or free
final time.

The least delta-v that puts the chaser in the capture ball at given step boundaries is a linear
program on the zero-order-hold step grid (Boyd and Vandenberghe, Convex Optimization, 2004,
sections 4.3 and 6.1). For box thrusters the program is exact. For a ball the 2-norm is bracketed
by one shared set of directions: from outside by Kelley's cutting planes s_k >= d^T u_k (Kelley,
1960), from inside by the convex hull of the same directions, inside which every plan is within
the thrust ball. The multipliers of the membership rows go through the primer bound of
`dual_bound`, so the lower bounds hold in continuous time whatever the solver returned; the upper
bound is the cost of a plan checked by exact propagation. A free final time takes the least bound
over every window of boundaries a success must pass through, and `DualBank` moves multipliers
between windows by shifting their constraint times, which any multipliers allow (Boyd and
Vandenberghe, section 5.1.3).
"""

from __future__ import annotations

import math
import operator
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.optimize import OptimizeResult, linprog

from ..agents.base import AgentView, hold_point
from ..constraints import KeepOut, interval_bounds, segment_minima
from ..dynamics import cw_discrete
from .primer import _EXCESS_INFLATION, _FLOOR, NORMS, _lipschitz, dual_bound

if TYPE_CHECKING:
    from ..metrics.efficiency import FuelBounds
    from ..metrics.episode import Episode

# HiGHS's default primal and dual feasibility tolerance.
_FEASIBILITY_TOL = 1e-7
# A cut that binds at no step for this many solves in a row leaves the set; Kelley adds it back
# if a step needs it again.
_IDLE_SOLVES = 3
# Samples per step of the keep-out and dwell checks, as the environment samples its constraint
# checks.
_SAMPLES = 20
_FIRST_CUTS = np.vstack(
    [
        np.eye(3),
        -np.eye(3),
        np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) / math.sqrt(3),
    ]
)


@dataclass(frozen=True, kw_only=True)
class OracleConfig:
    """Settings of `FuelOracle` and `DualBank`.

    Parameters
    ----------
    cells_per_step : int
        Primer cells per step for the certified excess integral and supremum of `dual_bound`.
    arrival_candidates : int
        Arrival steps tried for the upper bound at a free final time, spread evenly from the
        last admissible step down to step 1; the window where the lower bound is least is tried
        too. Any choice gives a valid upper bound.
    aim : float
        Fraction of the capture radius and speed the plan aims inside, in (0, 1); half the rest
        is the margin the propagated plan must keep.
    cut_rounds : int
        Most LP solves of the outer Kelley loop for the ball layout.
    cut_tol : float
        The Kelley loop stops when (|u_k|_2 - s_k) / |u_k|_2 is below this at every step, s_k the
        cutting-plane value of |u_k|_2.
    inner_directions : int
        Fibonacci directions added to the cut directions whose convex hull bounds the ball plan.
    bank_stride : int
        Windows between the LPs a `DualBank` solves per start; the last window is always solved.
    max_refinements : int
        Most window LPs a free-final-time `FuelOracle.solve` adds after its strided bank, each at
        the window whose bound is least; 0 keeps the bank's bound.

    Raises
    ------
    ValueError
        If `cells_per_step`, `arrival_candidates`, `cut_rounds` or `bank_stride` is below 1,
        `inner_directions` or `max_refinements` is negative, `aim` is outside (0, 1) or
        `cut_tol` is not positive and finite.
    """

    cells_per_step: int = 8
    arrival_candidates: int = 16
    aim: float = 0.9
    cut_rounds: int = 20
    cut_tol: float = 1e-4
    inner_directions: int = 64
    bank_stride: int = 10
    max_refinements: int = 8

    def __post_init__(self):
        for name in ("cells_per_step", "arrival_candidates", "cut_rounds", "bank_stride"):
            if operator.index(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("inner_directions", "max_refinements"):
            if operator.index(getattr(self, name)) < 0:
                raise ValueError(f"{name} must not be negative")
        if not 0.0 < self.aim < 1.0:
            raise ValueError("aim must lie in (0, 1)")
        if not (math.isfinite(self.cut_tol) and self.cut_tol > 0.0):
            raise ValueError("cut_tol must be positive and finite")


@dataclass(frozen=True)
class OracleSolution:
    """Bounds on the least delta-v of a success, and the plan behind the upper bound.

    Parameters
    ----------
    bounds : FuelBounds
        `finite_lower`, `upper` and `impulsive_lower` in the layout's norm, rung "cw", with
        `accel_limit` the view's `max_accel`.
    finite_lower : float
        Certified lower bound (m/s) for acceleration histories within `max_accel` in the actuator
        norm, never below `impulsive_lower`.
    impulsive_lower : float
        Certified lower bound (m/s) for any acceleration history.
    finite_optimum : float
        For a fixed arrival step, the optimum (m/s) of the outer LP on the step grid: the capture
        ball relaxed to its bounding box and, for a ball layout, the 2-norm to its cutting
        planes; inf when that LP is infeasible, NaN when HiGHS failed on it. No acceleration
        history held over each step and within the limit costs less. NaN at a free final time:
        the least optimum over the windows would need every window's LP, and the bank solves
        only some of them.
    upper : float
        Cost (m/s) of `plan`; inf when no plan was verified or `unmodelled` is not empty.
    plan : numpy.ndarray or None, shape (arrival_step + hold, 3)
        Hill accelerations (m/s^2), each held for one step, within `max_accel`, zero in thrust
        inhibit windows, within (1 + aim) / 2 of the capture radius and speed at the end of step
        `arrival_step` and, with a dwell, throughout the next hold = ceil(hold_time / dt) steps,
        certified between samples; if keep-out is hard, outside it throughout. None when no
        plan passed.
    arrival_step : int or None
        For a fixed arrival step, that step; at a free final time, the arrival step of `plan`,
        None without a plan.
    unmodelled : tuple of str
        Hard constraints the plan was not checked against.
    """

    bounds: FuelBounds
    finite_lower: float
    impulsive_lower: float
    finite_optimum: float
    upper: float
    plan: np.ndarray | None
    arrival_step: int | None
    unmodelled: tuple[str, ...]


@dataclass(frozen=True)
class _Window:
    """LP data for membership at the consecutive step boundaries first..steps, in scaled units:
    w = u / a, positions over L, velocities over L n, cost over L n.

    `response[d]` is the scaled state change d steps after a scaled input held over one step,
    `miss[i]` the scaled goal state less the free motion at boundary first + i, and `drift` the
    scaled change of the goal state over one step of free motion.
    """

    first: int
    steps: int
    response: np.ndarray
    miss: np.ndarray
    step: np.ndarray
    drift: np.ndarray
    ball: np.ndarray
    scale: np.ndarray
    cost: float
    inhibited: np.ndarray


@dataclass(frozen=True)
class _Solution:
    """An LP solution in scaled units: optimum, inputs w (k, 3), the cost the LP charges each
    step s (k,), and the multipliers of the membership rows (boundaries, 6)."""

    value: float
    w: np.ndarray
    s: np.ndarray
    nu: np.ndarray


def _fibonacci(count: int) -> np.ndarray:
    """`count` unit vectors spread over the sphere on a Fibonacci lattice."""
    i = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / count)
    azimuth = math.pi * (1.0 + math.sqrt(5.0)) * i
    return np.c_[np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)]


class _Undecided(RuntimeError):
    """HiGHS ended an LP with no verdict on optimality or feasibility after every retry."""


# scipy's status for a HiGHS model status with no verdict, such as 15 (unknown).
_NO_VERDICT = 4
# Bounds on the residuals of an accepted solve, relative to 1 + the largest right-hand side,
# bound or cost. HiGHS meets its feasibility tolerances of 1e-7 on its own scaling of the LP, so
# the residuals of solves it calls optimal reach 6e-8 on the scenario LPs; ten times its
# tolerance passes those and rejects a solve whose multipliers do not fit the costs.
_PRIMAL_TOL = 1e-6
_DUAL_TOL = 1e-6
# Bound on |c^T x - dual objective| / (1 + |c^T x|), HiGHS's own tolerance on its relative
# objective error. Solves HiGHS calls optimal show 8e-11 or less on the scenario LPs, and the
# dual simplex end it could not decide on window 301 of approach-vbar validation start 19 was
# 5.8e-4.
_GAP_TOL = 1e-7


def _certified(problem: Mapping[str, Any], res: OptimizeResult) -> bool:
    """Whether `res` is optimal for `problem` by its own residuals, whatever HiGHS's status.

    The multipliers are first given their dual signs (bound marginals nonnegative at lower and
    nonpositive at upper bounds, inequality marginals nonpositive, zero at an infinite bound), so
    a wrong sign shows in the stationarity residual c - A_ub^T z - A_eq^T y - lower - upper; then
    the primal residuals, that residual and the gap between c^T x and the dual objective must be
    within `_PRIMAL_TOL`, `_DUAL_TOL` and `_GAP_TOL` (Boyd and Vandenberghe, Convex Optimization,
    2004, section 5.5.3: feasibility and zero gap make both points optimal).
    """
    c = np.asarray(problem["c"], dtype=float)
    x = res.x
    if x is None:
        return False
    # scipy writes an infinite bound as None, which a float array holds as NaN.
    given = problem.get("bounds")
    # scipy reads no bounds as x >= 0.
    given = (0.0, None) if given is None else given
    bounds = np.broadcast_to(np.array(given, dtype=float), (len(c), 2))
    low = np.where(np.isnan(bounds[:, 0]), -np.inf, bounds[:, 0])
    high = np.where(np.isnan(bounds[:, 1]), np.inf, bounds[:, 1])
    at_low, at_high = np.isfinite(low), np.isfinite(high)
    lower = np.where(at_low, np.maximum(res.lower.marginals, 0.0), 0.0)
    upper = np.where(at_high, np.minimum(res.upper.marginals, 0.0), 0.0)
    finite = np.r_[np.abs(low[at_low]), np.abs(high[at_high])]
    primal = [np.max(np.r_[low - x, x - high, 0.0]) / (1.0 + np.max(finite, initial=0.0))]
    stationarity = c - lower - upper
    dual = lower[at_low] @ low[at_low] + upper[at_high] @ high[at_high]
    if problem.get("A_ub") is not None:
        a, b = problem["A_ub"], np.asarray(problem["b_ub"], dtype=float)
        z = np.minimum(res.ineqlin.marginals, 0.0)
        over = np.max(np.maximum(a @ x - b, 0.0), initial=0.0)
        primal.append(over / (1.0 + np.max(np.abs(b), initial=0.0)))
        stationarity = stationarity - a.T @ z
        dual += b @ z
    if problem.get("A_eq") is not None:
        a, b = problem["A_eq"], np.asarray(problem["b_eq"], dtype=float)
        y = res.eqlin.marginals
        primal.append(
            np.max(np.abs(a @ x - b), initial=0.0) / (1.0 + np.max(np.abs(b), initial=0.0))
        )
        stationarity = stationarity - a.T @ y
        dual += b @ y
    value = float(c @ x)
    return bool(
        max(primal) <= _PRIMAL_TOL
        and np.max(np.abs(stationarity)) <= _DUAL_TOL * (1.0 + np.max(np.abs(c)))
        and abs(value - dual) <= _GAP_TOL * (1.0 + abs(value))
    )


def _highs(presolve: bool, **problem: object) -> OptimizeResult:
    """`linprog` by HiGHS, accepted only if `_certified`; an end with no verdict or an optimum
    that fails the check is retried by the interior point method, then by dual simplex with
    presolve switched, and a result no attempt certified comes back at status `_NO_VERDICT`.

    Dual simplex at both presolve settings ended window 301 of a 360-step approach-vbar start at
    model status 15, and the interior point method solved it.
    """
    attempts = (("highs", presolve), ("highs-ipm", presolve), ("highs-ds", not presolve))
    for method, pre in attempts:
        res = linprog(**problem, method=method, options={"presolve": pre})
        if res.status == 0 and not _certified(problem, res):
            res = OptimizeResult(
                {**res, "status": _NO_VERDICT, "message": "optimum failed the certificate check"}
            )
        if res.status != _NO_VERDICT:
            break
    return res


def _checked(status: int, message: str, infeasible: tuple[int, ...]) -> bool:
    """Whether HiGHS solved the LP: False for the statuses that mean infeasible, raise otherwise."""
    if status in infeasible:
        return False
    if status == _NO_VERDICT:
        raise _Undecided(f"HiGHS failed: {message}")
    if status != 0:
        raise RuntimeError(f"HiGHS failed: {message}")
    return True


def _first_rows(win: _Window) -> np.ndarray:
    """Scaled state at the first boundary as rows (6, 3 first) in the scaled inputs before it."""
    k = win.first
    return win.response[k - 1 :: -1].transpose(1, 0, 2).reshape(6, 3 * k)


def _rows(win: _Window) -> tuple[np.ndarray, np.ndarray]:
    """Scaled states at every boundary of the window as dense rows (6 J, 3 steps) in the scaled
    inputs, and the misses (6 J,)."""
    count = len(win.miss)
    lag = (win.first + np.arange(count))[:, None] - 1 - np.arange(win.steps)[None, :]
    blocks = np.where((lag >= 0)[..., None, None], win.response[np.maximum(lag, 0)], 0.0)
    return blocks.transpose(0, 2, 1, 3).reshape(6 * count, 3 * win.steps), win.miss.ravel()


def _terminal_rows(
    reach: np.ndarray, miss: np.ndarray, ball: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rows and right-hand side of |reach w - miss|_i <= ball_i, upper rows first, with the
    half-widths `ball` (6,) repeated for every boundary."""
    half = np.tile(ball, len(miss) // 6)
    return np.vstack([reach, -reach]), np.r_[miss + half, half - miss]


def _box_lp(win: _Window, ball: np.ndarray, limit: bool, inhibit: bool) -> _Solution | None:
    """min sum |w_k|_1 with |w_k,i| <= 1 (if `limit`) and the scaled error e at every boundary
    of the window within the box of half-widths `ball`, with w = w+ - w-; None if infeasible.

    The error at the first boundary is a dense row block in the inputs; each later one follows
    by one step, e' = A e + B w + drift, so the rows stay sparse however long the window. The
    multipliers are the reduced costs of the error variables, which carry the box.
    """
    k, steps, count = win.first, win.steps, len(win.miss)
    width = 3 * steps
    blocks = [sparse.csr_matrix(np.hstack([-_first_rows(win), np.zeros((6, width - 3 * k))]))]
    if count > 1:
        later = sparse.kron(sparse.identity(count - 1), -win.response[0])
        blocks.append(sparse.hstack([sparse.csr_matrix((6 * (count - 1), 3 * k)), later]))
    inputs = sparse.vstack(blocks, format="csr")
    errors = sparse.identity(6 * count) - sparse.kron(sparse.eye(count, k=-1), win.step)
    top = np.full(width, 1.0 if limit else np.inf)
    if inhibit:
        top[np.repeat(win.inhibited, 3)] = 0.0
    half = np.tile(ball, count)
    res = _highs(
        True,
        c=np.r_[np.full(2 * width, win.cost), np.zeros(6 * count)],
        A_eq=sparse.hstack([inputs, -inputs, errors], format="csr"),
        b_eq=np.r_[-win.miss[0], np.tile(win.drift, count - 1)],
        bounds=np.c_[np.r_[np.zeros(2 * width), -half], np.r_[top, top, half]],
    )
    if not _checked(res.status, res.message, (2,)):
        return None
    plus, minus = res.x[:width].reshape(-1, 3), res.x[width : 2 * width].reshape(-1, 3)
    # The upper bound's marginal is the sensitivity to e <= ball, the lower bound's that to
    # -e <= ball with the sign flipped.
    nu = (res.upper.marginals + res.lower.marginals)[2 * width :].reshape(count, 6)
    return _Solution(value=float(res.fun), w=plus - minus, s=(plus + minus).sum(axis=1), nu=nu)


def _cut_lp(
    win: _Window, ball: np.ndarray, cuts: np.ndarray, limit: bool
) -> tuple[_Solution, np.ndarray] | None:
    """min sum s_k with s_k >= d^T w_k over the cuts, s_k <= 1 if `limit`, and the boundary
    boxes of half-widths `ball`; with it, which cuts bind at some step. None if infeasible.

    Every bound is a row, so the LP is solved through its dual, min b^T y with A^T y = -c and
    y >= 0, which has one row per variable rather than one per cut and step; the inputs are the
    dual's equality multipliers, and a cut binds where its dual variable is positive.
    """
    k, m = win.steps, len(cuts)
    eye = sparse.identity(k, format="csr")
    blocks = [sparse.hstack([sparse.kron(eye, cuts), sparse.kron(eye, -np.ones((m, 1)))])]
    rhs = [np.zeros(k * m)]
    if limit:
        blocks.append(sparse.hstack([sparse.csr_matrix((k, 3 * k)), eye]))
        rhs.append(np.ones(k))
    reach, miss = _rows(win)
    rows, bound = _terminal_rows(np.hstack([reach, np.zeros((len(miss), k))]), miss, ball)
    blocks.append(sparse.csr_matrix(rows))
    rhs.append(bound)
    # Presolve takes about 40% of these solves and removes almost nothing from them.
    res = _highs(
        False,
        c=np.concatenate(rhs),
        A_eq=sparse.vstack(blocks, format="csr").T,
        b_eq=-np.r_[np.zeros(3 * k), np.full(k, win.cost)],
        bounds=(0.0, None),
    )
    # An unbounded dual is an infeasible primal.
    if not _checked(res.status, res.message, (2, 3)):
        return None
    x = res.eqlin.marginals
    # The primal marginal of row i is -y_i.
    y = res.x[-2 * len(miss) :]
    solution = _Solution(
        value=-float(res.fun),
        w=x[: 3 * k].reshape(k, 3),
        s=x[3 * k :],
        nu=(y[len(miss) :] - y[: len(miss)]).reshape(-1, 6),
    )
    return solution, (res.x[: k * m].reshape(k, m) > 0.0).any(axis=0)


def _hull_lp(win: _Window, ball: np.ndarray, directions: np.ndarray) -> _Solution | None:
    """min sum_kd lambda_kd with w_k = sum_d lambda_kd d, sum_d lambda_kd <= 1 and lambda >= 0,
    zero on inhibited steps, and the boundary boxes of half-widths `ball`; None if infeasible.

    Each w_k lies in the convex hull of the unit directions and the origin, so
    |w_k|_2 <= sum_d lambda_kd <= 1: the plan is within the thrust ball and costs at most the
    LP value.
    """
    k, m = win.steps, len(directions)
    reach, miss = _rows(win)
    along = np.einsum("ikc,dc->ikd", reach.reshape(-1, k, 3), directions).reshape(-1, k * m)
    rows, bound = _terminal_rows(along, miss, ball)
    top = np.where(np.repeat(win.inhibited, m), 0.0, np.inf)
    # Presolve takes half of this solve and removes almost nothing.
    res = _highs(
        False,
        c=np.full(k * m, win.cost),
        A_ub=sparse.vstack(
            [sparse.csr_matrix(rows), sparse.kron(sparse.identity(k), np.ones((1, m)))],
            format="csr",
        ),
        b_ub=np.r_[bound, np.ones(k)],
        bounds=np.c_[np.zeros(k * m), top],
    )
    if not _checked(res.status, res.message, (2,)):
        return None
    weights = res.x.reshape(k, m)
    marginals = res.ineqlin.marginals
    return _Solution(
        value=float(res.fun),
        w=weights @ directions,
        s=weights.sum(axis=1),
        nu=(marginals[: len(miss)] - marginals[len(miss) : 2 * len(miss)]).reshape(-1, 6),
    )


def _kelley(
    win: _Window, ball: np.ndarray, config: OracleConfig, limit: bool
) -> tuple[_Solution | None, np.ndarray]:
    """The outer ball LP and its cut set, refined by cuts at the directions of w_k until s_k is
    within `cut_tol` of |w_k|_2 at every step or `cut_rounds` LPs are solved."""
    cuts = _FIRST_CUTS
    idle = np.zeros(len(cuts), dtype=int)
    lp = None
    for _ in range(config.cut_rounds):
        solved = _cut_lp(win, ball, cuts, limit)
        if solved is None:
            return None, cuts
        lp, binding = solved
        size = np.linalg.norm(lp.w, axis=1)
        gap = size - lp.s
        # A gap within HiGHS's feasibility tolerance is no gap a cut can close.
        short = (gap >= config.cut_tol * size) & (gap > _FEASIBILITY_TOL)
        if not short.any():
            break
        idle = np.where(binding, 0, idle + 1)
        keep = idle < _IDLE_SOLVES
        keep[: len(_FIRST_CUTS)] = True
        cuts = np.vstack([cuts[keep], lp.w[short] / size[short, None]])
        idle = np.r_[idle[keep], np.zeros(int(short.sum()), dtype=int)]
    return lp, cuts


def _outer(
    win: _Window, norm: str, config: OracleConfig
) -> tuple[_Solution | None, float, np.ndarray]:
    """The outer LP within the thrust limit, or without it if that is infeasible or HiGHS fails
    on a ball layout's LP within it; its optimum (m/s) within the limit, inf if infeasible and
    NaN if HiGHS failed; and the final cuts of a ball layout."""
    cuts = _FIRST_CUTS
    optimum = math.inf
    for limit in (True, False):
        if norm == "l1":
            lp = _box_lp(win, win.ball, limit, inhibit=False)
        else:
            try:
                lp, cuts = _kelley(win, win.ball, config, limit)
            except RuntimeError:
                # HiGHS reports a solve error instead of unboundedness on the dual of some
                # windows that have no plan within the limit. The LP without the limit is a
                # relaxation, its optimum no larger, and its multipliers give a valid bound
                # through the primer bound like any others; a failure there is raised.
                if not limit:
                    raise
                lp, optimum = None, math.nan
        if lp is not None:
            return lp, lp.value * win.scale[3] if limit else optimum, cuts
    return None, optimum, cuts


def _multipliers(lp: _Solution, win: _Window) -> np.ndarray:
    """The LP's multipliers in the units of `dual_bound`, one row per boundary."""
    # Objective and rows are scaled: d(m/s) / d(x_i) = (L n / s_i) d(objective) / d(row).
    return lp.nu * win.scale[3] / win.scale


def _landed_safely(
    view: AgentView,
    start: np.ndarray,
    plan: np.ndarray,
    goal: np.ndarray,
    rho: np.ndarray,
    slack: float,
    keep_out: float | None,
    arrival: int,
) -> bool:
    """Whether the plan, propagated exactly, is within `slack` times the capture radius and
    speed of the goal at the end of step `arrival` and throughout every later step, and stays
    outside the keep-out sphere throughout (`segment_minima`'s certified bound between samples).

    Between two samples h apart, a state within the ball at both ends departs from it by at most
    M h^2 / 8, M a bound on the second derivative: the acceleration bound of `interval_bounds`
    for the position, and 3 n^2 |v| + 2 n |a| for the velocity, the thrust being constant over a
    step.
    """
    phi, gamma = cw_discrete(view.n, view.dt)
    states = np.empty((len(plan) + 1, 6))
    states[0] = start
    for i, u in enumerate(plan):
        states[i + 1] = phi @ states[i] + gamma @ u
    miss = states[arrival] - goal
    if not (
        np.linalg.norm(miss[:3]) <= slack * rho[0] and np.linalg.norm(miss[3:]) <= slack * rho[1]
    ):
        return False
    if keep_out is None and arrival == len(plan):
        return True
    taus = np.linspace(0.0, view.dt, _SAMPLES)
    phis, gammas = cw_discrete(view.n, taus)
    begins = states[:-1]
    samples = np.einsum("tij,kj->kti", phis, begins) + np.einsum("tij,kj->kti", gammas, plan)
    thrust = np.linalg.norm(plan, axis=1)
    if arrival < len(plan):
        n, h = view.n, float(taus[1])
        held = samples[arrival:]
        speed, accel = interval_bounds(held, h, thrust[arrival:, None], n)
        jerk = 3.0 * n * n * speed + 2.0 * n * accel
        errors = np.stack(
            [
                np.linalg.norm(held[..., :3] - goal[:3], axis=-1),
                np.linalg.norm(held[..., 3:] - goal[3:], axis=-1),
            ]
        )
        reach = np.maximum(errors[..., :-1], errors[..., 1:]) + np.stack([accel, jerk]) * h**2 / 8
        if not np.all(reach <= slack * rho[:, None, None]):
            return False
    if keep_out is None:
        return True

    def state_at(rows: np.ndarray, tau: np.ndarray) -> np.ndarray:
        p, g = cw_discrete(view.n, tau)
        return np.einsum("aij,aj->ai", p, begins[rows]) + np.einsum("aij,aj->ai", g, plan[rows])

    _, _, lowest = segment_minima(
        KeepOut(float(keep_out)),
        state_at,
        taus,
        samples,
        thrust,
        view.n,
        view.dt * np.arange(len(plan)),
    )
    return bool(np.all(lowest >= 0.0))


def _goal_state(view: AgentView) -> tuple[np.ndarray, np.ndarray]:
    """Goal state (m, m/s, Hill) and the capture radius and speed, after the checks every bound
    needs."""
    if view.thrust_limit not in NORMS:
        raise ValueError(f"unknown thrust limit {view.thrust_limit!r}")
    for name in ("n", "dt", "max_accel"):
        number = getattr(view, name)
        if not (math.isfinite(number) and number > 0.0):
            raise ValueError(f"{name} must be positive and finite")
    r_g = hold_point(view, "fuel oracle")
    goal = view.goal
    if not {"capture_radius", "capture_speed"} <= goal.keys():
        raise ValueError("the goal has no capture ball")
    rho = np.array([goal["capture_radius"], goal["capture_speed"]], dtype=float)
    if not (np.all(np.isfinite(rho)) and np.all(rho >= 0.0)):
        raise ValueError("capture radius and speed must be finite and nonnegative")
    hold_time = goal.get("hold_time", 0.0)
    if not (math.isfinite(hold_time) and hold_time >= 0.0):
        raise ValueError("hold_time must be finite and nonnegative")
    if math.floor(hold_time / view.dt) > view.max_steps:
        raise ValueError("the dwell is longer than the episode")
    target = goal.get("target")
    if isinstance(target, Mapping) and target.get("burns"):
        raise ValueError("a manoeuvring target adds an acceleration the CW bound does not model")
    x_g = np.r_[r_g, np.asarray(goal.get("goal_velocity", (0.0, 0.0, 0.0)), dtype=float)]
    if x_g.shape != (6,) or not np.all(np.isfinite(x_g)):
        raise ValueError("hold point and goal velocity must be 3 finite numbers each")
    return x_g, rho


def _dwell(view: AgentView) -> tuple[int, int]:
    """Boundaries a success is certainly in the closed capture ball at, and the steps a plan
    holds the ball after it arrives.

    A dwell is a time interval of length hold_time inside the ball, so its closure holds
    floor(hold_time / dt) boundaries, none for a dwell shorter than a step; without a dwell the
    success is one boundary in the ball. The environment counts a dwell from the boundary where
    it starts, so a plan holds the ball for ceil(hold_time / dt) steps.
    """
    hold_time = view.goal.get("hold_time", 0.0)
    if hold_time <= 0.0:
        return 1, 0
    return math.floor(hold_time / view.dt), math.ceil(hold_time / view.dt)


def _start(start: ArrayLike) -> np.ndarray:
    """The start as 6 finite numbers."""
    x0 = np.asarray(start, dtype=float)
    if x0.shape != (6,) or not np.all(np.isfinite(x0)):
        raise ValueError("start must be 6 finite numbers")
    return x0


def _window(
    view: AgentView, x0: np.ndarray, x_g: np.ndarray, rho: np.ndarray, first: int, count: int
) -> _Window:
    """Scaled LP data for membership at the `count` boundaries from the end of step `first`."""
    n, a = view.n, view.max_accel
    # Each term scales with the start, goal and ball, so scaling all of them and the thrust
    # limit by the same factor leaves the scaled LP unchanged.
    length = max(
        np.linalg.norm(x0[:3]),
        np.linalg.norm(x_g[:3]),
        rho[0],
        np.linalg.norm(x0[3:]) / n,
        np.linalg.norm(x_g[3:]) / n,
        rho[1] / n,
    )
    if length == 0.0:
        length = 1.0
    scale = np.r_[np.full(3, length), np.full(3, length * n)]
    steps = first + count - 1
    powers, gamma = cw_discrete(n, view.dt * np.arange(steps + 1))
    # From the closed form at each duration, so long windows carry no product rounding.
    response = powers[:steps] @ gamma[1] * a / scale[:, None]
    starts = view.dt * np.arange(steps)
    inhibited = np.zeros(steps, dtype=bool)
    for t_on, t_off in view.goal.get("thrust_inhibit", ()):
        # Window edges lie on step boundaries; the tolerance absorbs the sum of steps.
        inhibited |= (starts < t_off - 1e-6) & (starts + view.dt > t_on + 1e-6)
    return _Window(
        first=first,
        steps=steps,
        response=response,
        miss=(x_g - powers[first:] @ x0) / scale,
        step=powers[1] / scale[:, None] * scale,
        drift=(powers[1] @ x_g - x_g) / scale,
        ball=np.repeat(rho, 3) / scale,
        scale=scale,
        cost=view.dt * a / (length * n),
        inhibited=inhibited,
    )


def _hard(view: AgentView, hard: Collection[str] | None) -> tuple[float | None, tuple[str, ...]]:
    """The keep-out radius (m) if keep-out is hard, and the hard constraints a plan is not
    checked against."""
    listed = {c["type"]: c for c in view.constraints}
    names = set(listed) if hard is None else set(hard)
    keep_out = listed["keep_out"]["radius"] if "keep_out" in names & listed.keys() else None
    return keep_out, tuple(sorted(names - ({"keep_out"} if keep_out is not None else set())))


class FuelOracle:
    """Fuel bounds into the capture ball on exact CW motion (`Oracle` protocol).

    A success puts the chaser in the closed capture ball at m consecutive step boundaries
    k..k+m-1, m = floor(hold_time / dt) with a dwell and 1 without one. For one window the outer
    LP relaxes the ball to its bounding box, |x_i - x_g,i| <= rho per component, at each of its
    boundaries, and has no path constraints; for a ball layout it also replaces |u_k|_2 by
    Kelley cuts, one set shared by all steps, started from the 6 axes and 8 diagonals, with cuts
    that bind nowhere for three solves dropped. Its multipliers give both lower bounds through
    `dual_bound`; if it is infeasible within the thrust limit, or HiGHS fails on a ball layout's
    LP within it, they come from the LP without the limit, and if that is infeasible too, both
    lower bounds are 0. A free final time takes the
    least bound over every window that fits in the episode, from a one-start `DualBank` refined
    until the least window has its own LP or `max_refinements` LPs are added; a dwell shorter
    than a step gives 0.

    The inner LP aims inside the box of half-width aim rho / sqrt(3) at the arrival boundary and,
    with a dwell, at the ceil(hold_time / dt) boundaries after it, and zeroes inhibited steps;
    for a ball it draws each u_k from the convex hull of a final cut set and `inner_directions`
    Fibonacci directions. Its plan is propagated exactly and accepted if it stays within
    (1 + aim) / 2 of the capture radius and speed from the arrival boundary to the end of the
    dwell, between samples included, and, when keep-out is hard, never enters it. At a free
    final time the upper bound is the cheapest accepted plan over the arrival candidates. A zero
    capture radius or speed leaves no margin to verify, so `upper` is then inf.

    Parameters
    ----------
    config : OracleConfig, optional
        Settings; the defaults when omitted.
    """

    def __init__(self, *, config: OracleConfig | None = None):
        self.config = OracleConfig() if config is None else config

    def bounds(self, view: AgentView, start: ArrayLike) -> FuelBounds:
        """Bounds at a free final time with every listed constraint hard.

        Parameters
        ----------
        view : AgentView
            Scenario description (see `solve`).
        start : array_like, shape (6,)
            Hill state at t = 0 (m, m/s).

        Returns
        -------
        FuelBounds
            `solve(view, start).bounds`.

        Raises
        ------
        ValueError
            As `solve`.
        RuntimeError
            As `solve`.
        """
        return self.solve(view, start).bounds

    def solve(
        self,
        view: AgentView,
        start: ArrayLike,
        *,
        arrival_step: int | None = None,
        hard: Collection[str] | None = None,
    ) -> OracleSolution:
        """Lower and upper bounds on the least delta-v of a success.

        Parameters
        ----------
        view : AgentView
            Scenario description: mean motion `n` (rad/s), `dt` (s), `max_steps`, `max_accel`
            (m/s^2, per axis for "box", 2-norm for "ball"), `thrust_limit`, a hold or berthing
            goal with `hold_point` (m, Hill), `capture_radius` (m), `capture_speed` (m/s) and
            optionally `goal_velocity` (m/s, Hill), `hold_time` (s) and `thrust_inhibit` windows
            (s from the start), and the listed constraints, of which only "keep_out" is checked.
        start : array_like, shape (6,)
            Hill state at t = 0 (m, m/s).
        arrival_step : int, optional
            Step, 1 to `max_steps`, at whose end the chaser is in the ball and the dwell
            starts: the lower bounds are for membership at that boundary and the next
            floor(hold_time / dt) - 1 within the episode. A free final time when omitted.
        hard : collection of str, optional
            Names of the hard constraints; every listed constraint when None.

        Returns
        -------
        OracleSolution
            The bounds, the outer LP optimum and the plan.

        Raises
        ------
        ValueError
            If the thrust layout is unknown, `n`, `dt` or `max_accel` is not positive and finite,
            the goal is not a hold or berthing or has no hold point and capture ball, the
            capture radius or speed is negative or not finite, `hold_time` is negative, not
            finite or longer than the episode, the briefed target burns, the hold point or goal
            velocity is not 3 finite numbers, `start` is not 6 finite numbers, or
            `arrival_step` is outside 1 to `max_steps`.
        RuntimeError
            If HiGHS fails for a reason other than infeasibility.
        """
        x_g, rho = _goal_state(view)
        x0 = _start(start)
        count, hold = _dwell(view)
        steps = view.max_steps
        if arrival_step is not None and not 1 <= operator.index(arrival_step) <= steps:
            raise ValueError(f"arrival_step must lie in 1 to {steps}")
        norm = NORMS[view.thrust_limit]
        keep_out, unmodelled = _hard(view, hard)

        if arrival_step is None:
            finite_lower, impulsive_lower, cut_sets, least = self._free_lower(view, x0, count)
            finite_optimum = math.nan
            last = steps - hold
            arrivals = np.rint(np.linspace(last, 1, self.config.arrival_candidates)).astype(int)
            if least is not None:
                arrivals = np.r_[arrivals, min(least, last)]
            candidates = [int(k) for k in np.unique(arrivals)] if last >= 1 else []
        else:
            k = operator.index(arrival_step)
            win = _window(view, x0, x_g, rho, k, min(max(count, 1), steps - k + 1))
            lp, finite_optimum, cuts = _outer(win, norm, self.config)
            finite_lower = impulsive_lower = 0.0
            if lp is not None:
                nus = _multipliers(lp, win)
                times = view.dt * np.arange(k, win.steps + 1)
                cells = self.config.cells_per_step * win.steps
                impulsive_lower = max(
                    0.0,
                    dual_bound(view.n, view, x0, nus, times, norm=norm, accel=np.inf, cells=cells),
                )
                # nu over its certified primer peak has no excess, so its finite-thrust bound
                # is the impulsive one.
                finite_lower = max(
                    impulsive_lower,
                    dual_bound(view.n, view, x0, nus, times, norm=norm, cells=cells),
                )
            cut_sets = {k: cuts}
            # A plan is verified on its own, so only a proved infeasible window skips it.
            candidates = [k] if finite_optimum != math.inf and k + hold <= steps else []

        plan, upper, arrival = None, math.inf, None
        if np.all(rho > 0.0):
            order = 1 if norm == "l1" else 2
            for k in candidates:
                nearest = min(cut_sets, key=lambda j: (abs(j - k), j), default=None)
                cuts = _FIRST_CUTS if nearest is None else cut_sets[nearest]
                found = self._plan(view, x0, x_g, rho, k, hold, cuts, norm, keep_out)
                if found is None:
                    continue
                cost = float(view.dt * np.linalg.norm(found, ord=order, axis=1).sum())
                if cost < upper:
                    plan, upper, arrival = found, cost, k
        if unmodelled:
            upper = math.inf
        if arrival_step is not None:
            arrival = operator.index(arrival_step)

        # Importing the metrics package pulls scipy.stats, interpolate and ndimage into every
        # oracle import, so it waits until bounds are built.
        from ..metrics.efficiency import FuelBounds

        bounds = FuelBounds(
            lower=finite_lower,
            upper=upper,
            norm=norm,
            rung="cw",
            accel_limit=view.max_accel,
            impulsive_lower=impulsive_lower,
        )
        return OracleSolution(
            bounds=bounds,
            finite_lower=finite_lower,
            impulsive_lower=impulsive_lower,
            finite_optimum=finite_optimum,
            upper=upper,
            plan=plan,
            arrival_step=arrival,
            unmodelled=unmodelled,
        )

    def _free_lower(
        self, view: AgentView, x0: np.ndarray, count: int
    ) -> tuple[float, float, dict[int, np.ndarray], int | None]:
        """Finite-thrust and impulsive lower bounds (m/s) at a free final time, the cut sets of
        the windows solved, and the window whose bound is least (None for the trivial bound).

        The bank starts with every `bank_stride`-th window and solves the window whose bound is
        least until that window has its own LP, at most `max_refinements` times.
        """
        if count == 0:
            return 0.0, 0.0, {}, None
        bank = DualBank(view, config=self.config)
        cut_sets = {k: bank._solve(x0, k) for k in bank._strided()}
        refinements = 0
        while True:
            finite, impulsive = bank._window_bounds(x0)
            least = int(np.argmin(finite)) + 1
            if least in cut_sets or refinements == self.config.max_refinements:
                break
            cut_sets[least] = bank._solve(x0, least)
            refinements += 1
        return max(0.0, float(finite.min())), max(0.0, float(impulsive.min())), cut_sets, least

    def _plan(
        self,
        view: AgentView,
        x0: np.ndarray,
        x_g: np.ndarray,
        rho: np.ndarray,
        arrival: int,
        hold: int,
        cuts: np.ndarray,
        norm: str,
        keep_out: float | None,
    ) -> np.ndarray | None:
        """The inner LP's plan (m/s^2) arriving at the end of step `arrival` and holding the ball
        for `hold` steps, if it passes verification, else None."""
        aim = self.config.aim
        win = _window(view, x0, x_g, rho, arrival, hold + 1)
        ball = aim * win.ball / math.sqrt(3.0)
        if norm == "l1":
            lp = _box_lp(win, ball, limit=True, inhibit=True)
        else:
            directions = np.vstack([cuts, _fibonacci(self.config.inner_directions)])
            lp = _hull_lp(win, ball, directions)
        if lp is None:
            return None
        # HiGHS meets bounds to its tolerance; the plan meets them exactly and is then checked.
        if norm == "l1":
            w = np.clip(lp.w, -1.0, 1.0)
        else:
            w = lp.w / np.maximum(1.0, np.linalg.norm(lp.w, axis=1))[:, None]
        plan = view.max_accel * w
        if not _landed_safely(view, x0, plan, x_g, rho, 0.5 * (1.0 + aim), keep_out, arrival):
            return None
        return plan


class DualBank:
    """Multipliers of window LPs for a set of starts, moved to every window by a time shift.

    A success is in the closed capture ball at m consecutive boundaries k..k+m-1 (see
    `FuelOracle`), for some window k from 1 to max_steps - m + 1. Each entry is the multipliers
    of one window LP from one start. Moved to window k2, they apply at the boundaries
    k2..k2+m-1: the primer depends only on the time before the last of them, so the excess
    integral and the primer peak of window k2 are a prefix of one profile per entry, on the
    cells of `dual_bound`, and the displacement term is affine in the start. The bound of a
    window is the largest over the entries of the finite-thrust and impulsive bounds; a bound
    from any multipliers is valid (Boyd and Vandenberghe, Convex Optimization, 2004, section
    5.1.3), so every bound holds for any start, in the bank or not, whatever the LPs returned.
    A dwell shorter than a step gives every window the bound 0.

    Parameters
    ----------
    view : AgentView
        Scenario description, as for `FuelOracle.solve`.
    config : OracleConfig, optional
        `cells_per_step`, `bank_stride` and, for a ball layout, the Kelley settings; the
        defaults when omitted.

    Attributes
    ----------
    uncertified : tuple of int
        Rows of the starts given to `build` that added no multipliers because HiGHS ended one of
        their window LPs with no verdict after its retries, in increasing order; empty for a
        bank not built by `build`.

    Raises
    ------
    ValueError
        As `FuelOracle.solve` for the view.
    """

    def __init__(self, view: AgentView, *, config: OracleConfig | None = None):
        self.view = view
        self.config = OracleConfig() if config is None else config
        self._goal, self._rho = _goal_state(view)
        self._norm = NORMS[view.thrust_limit]
        self._count, _ = _dwell(view)
        steps = view.max_steps
        self._windows = steps - self._count + 1 if self._count > 0 else steps
        self._powers, _ = cw_discrete(view.n, view.dt * np.arange(steps + 1))
        cells = self.config.cells_per_step
        self._cell = view.dt / cells
        tau = view.dt * np.arange(steps)[:, None] + self._cell * np.arange(cells + 1)
        phi, _ = cw_discrete(view.n, tau)
        # Primer of a multiplier tau before its constraint time: B^T Phi(tau)^T nu.
        self._basis = np.swapaxes(phi[..., :, 3:6], -1, -2)
        self._entries: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        self._entries = []
        self.uncertified: tuple[int, ...] = ()

    @classmethod
    def build(
        cls, view: AgentView, starts: ArrayLike, *, config: OracleConfig | None = None
    ) -> DualBank:
        """A bank of the LPs of every `bank_stride`-th window, and the last, for each start.

        A window LP that HiGHS ends with no verdict is retried by the interior point method and
        by dual simplex with presolve switched. A start whose LP is still undecided adds no
        multipliers, those of its earlier windows included, and its row goes to `uncertified`;
        the other starts are solved as before.

        Parameters
        ----------
        view : AgentView
            Scenario description, as for `FuelOracle.solve`.
        starts : array_like, shape (N, 6)
            Hill states at t = 0 (m, m/s).
        config : OracleConfig, optional
            Settings; the defaults when omitted.

        Returns
        -------
        DualBank
            The bank.

        Raises
        ------
        ValueError
            As `FuelOracle.solve` for the view, or if `starts` is not N rows of 6 finite numbers.
        RuntimeError
            If HiGHS fails for a reason other than infeasibility or an end with no verdict.
        """
        bank = cls(view, config=config)
        points = np.asarray(starts, dtype=float)
        if points.ndim != 2:
            raise ValueError("starts must have shape (N, 6)")
        if bank._count == 0:
            return bank
        uncertified = []
        for i, row in enumerate(points):
            x0 = _start(row)
            kept = len(bank._entries)
            try:
                for k in bank._strided():
                    bank._solve(x0, k)
            except _Undecided:
                del bank._entries[kept:]
                uncertified.append(i)
        bank.uncertified = tuple(uncertified)
        return bank

    def bounds(self, start: ArrayLike) -> FuelBounds:
        """Lower bounds at a free final time: the least window bound.

        Parameters
        ----------
        start : array_like, shape (6,)
            Hill state at t = 0 (m, m/s).

        Returns
        -------
        FuelBounds
            `lower` max(0, least of `window_lowers`), `impulsive_lower` the same for the
            impulsive bounds, `upper` inf, rung "cw" and `accel_limit` the view's `max_accel`.

        Raises
        ------
        ValueError
            If `start` is not 6 finite numbers.
        """
        x0 = _start(start)
        finite_lower = impulsive_lower = 0.0
        if self._count > 0:
            finite, impulsive = self._window_bounds(x0)
            finite_lower = max(0.0, float(finite.min()))
            impulsive_lower = max(0.0, float(impulsive.min()))
        from ..metrics.efficiency import FuelBounds

        return FuelBounds(
            lower=finite_lower,
            upper=math.inf,
            norm=self._norm,
            rung="cw",
            accel_limit=self.view.max_accel,
            impulsive_lower=impulsive_lower,
        )

    def window_lowers(self, start: ArrayLike) -> np.ndarray:
        """Finite-thrust lower bound (m/s) of each window k = 1..max_steps.

        Parameters
        ----------
        start : array_like, shape (6,)
            Hill state at t = 0 (m, m/s).

        Returns
        -------
        numpy.ndarray, shape (max_steps,)
            The bound of window k at index k - 1, which may be negative; inf for a window that
            ends after the episode, which no success has; 0 throughout for a dwell shorter than
            a step and for the windows of a bank without entries.

        Raises
        ------
        ValueError
            If `start` is not 6 finite numbers.
        """
        x0 = _start(start)
        if self._count == 0:
            return np.zeros(self.view.max_steps)
        lows = np.full(self.view.max_steps, np.inf)
        lows[: self._windows] = self._window_bounds(x0)[0]
        return lows

    def _strided(self) -> list[int]:
        """Every `bank_stride`-th window from 1, and the last."""
        return sorted({*range(1, self._windows + 1, self.config.bank_stride), self._windows})

    def _solve(self, x0: np.ndarray, first: int) -> np.ndarray:
        """Adds the multipliers of window `first`'s outer LP from `x0`, if it has any; returns the
        final cut set of a ball layout."""
        win = _window(self.view, x0, self._goal, self._rho, first, self._count)
        lp, _, cuts = _outer(win, self._norm, self.config)
        if lp is not None:
            self._add(_multipliers(lp, win))
        return cuts

    def _add(self, nus: np.ndarray) -> None:
        """Stores, for every window, the constant and gain of the displacement term and the
        excess and peak of multipliers for consecutive boundaries."""
        n, m, w = self.view.n, self._count, self._windows
        steps, h = self.view.max_steps, self._cell
        # Offset o counts steps back from the last boundary; on step s of the time before it,
        # multiplier o acts for s >= o, at tau = 0 on the step's lower edge (the left limit).
        late = nus[::-1]
        terms = np.einsum("scij,oj->scoi", self._basis, late)
        p = np.zeros((steps, terms.shape[1], 3))
        for o in range(m):
            p[o:] += terms[: steps - o, :, o]
        lips = np.cumsum([_lipschitz(n, nu) for nu in late], axis=0)
        slope = lips[np.minimum(np.arange(steps), m - 1)]
        if self._norm == "l1":
            size = np.abs(p)
        else:
            # | |p(s)|_2 - |p(t)|_2 | <= |p(s) - p(t)|_2 <= |L|_2 |s - t|.
            size = np.linalg.norm(p, axis=-1, keepdims=True)
            slope = np.linalg.norm(slope, axis=-1, keepdims=True)
        top = np.maximum(size[:, :-1], size[:, 1:]) + 0.5 * h * slope[:, None, :]
        excess = np.cumsum(h * np.maximum(0.0, top - 1.0).sum(axis=(1, 2))) * _EXCESS_INFLATION
        peak = np.maximum.accumulate(top.max(axis=(1, 2))) * _EXCESS_INFLATION
        flows = self._powers[np.arange(1, w + 1)[:, None] + np.arange(m)]
        support = (
            self._rho[0] * np.linalg.norm(nus[:, :3], axis=1).sum()
            + self._rho[1] * np.linalg.norm(nus[:, 3:], axis=1).sum()
        )
        self._entries.append(
            (
                float((nus @ self._goal).sum() - support),
                float((np.abs(nus) @ np.abs(self._goal)).sum() + support),
                np.einsum("wjab,ja->wb", flows, nus),
                np.einsum("wjab,ja->wb", np.abs(flows), np.abs(nus)),
                excess[m - 1 : m - 1 + w],
                peak[m - 1 : m - 1 + w],
            )
        )

    def _window_bounds(self, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per window, the largest bound (m/s) over the entries, and the largest impulsive one.

        Each is the bound of `dual_bound` at the shifted times, with the same inflation of the
        excess and the peak and a rounding floor over the absolute value of every product in the
        displacement term, at least the floor `dual_bound` takes.
        """
        if not self._entries:
            zero = np.zeros(self._windows)
            return zero, zero
        const, const_abs, gain, gain_abs, excess, peak = (
            np.array(column) for column in zip(*self._entries, strict=True)
        )
        a = self.view.max_accel
        displacement = const[:, None] - gain @ x0
        floor = _FLOOR * (const_abs[:, None] + gain_abs @ np.abs(x0))
        finite = displacement - a * excess - floor - _FLOOR * a * excess
        safe = np.where(peak > 0.0, peak, 1.0)
        impulsive = np.where(peak > 0.0, (displacement - floor) / safe, 0.0)
        return np.maximum(finite, impulsive).max(axis=0), impulsive.max(axis=0)


def bounds_for(ep: Episode, oracle: FuelOracle | None = None) -> FuelBounds:
    """Fuel bounds for a recorded episode: free final time from its start, with its hard
    constraints.

    Parameters
    ----------
    ep : Episode
        The recorded episode; its `view`, `start` (truth, Hill, m and m/s) and `hard` are read.
    oracle : FuelOracle, optional
        The oracle; one with the default settings when omitted.

    Returns
    -------
    FuelBounds
        `oracle.solve(ep.view, ep.start, hard=ep.hard).bounds`, rung "cw" whatever the episode's
        dynamics, so `optimality_gap` certifies it only for a CW run whose target moved as
        briefed (`Episode.target_misbriefed` False).

    Raises
    ------
    ValueError
        As `FuelOracle.solve`.
    RuntimeError
        As `FuelOracle.solve`.
    """
    oracle = FuelOracle() if oracle is None else oracle
    return oracle.solve(ep.view, ep.start, hard=ep.hard).bounds
