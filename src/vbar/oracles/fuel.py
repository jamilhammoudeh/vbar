"""Fixed-window fuel oracle on the CW model: certified lower bounds and verified plans.

The least delta-v that puts the chaser in the capture ball at one step boundary is a linear
program on the zero-order-hold step grid (Boyd and Vandenberghe, Convex Optimization, 2004,
sections 4.3 and 6.1). For box thrusters the program is exact. For a ball the 2-norm is bracketed
by one shared set of directions: from outside by Kelley's cutting planes s_k >= d^T u_k (Kelley,
1960), from inside by the convex hull of the same directions, inside which every plan is within
the thrust ball. The terminal multipliers go through `dual_bound`, so the lower bounds hold in
continuous time whatever the solver returned; the upper bound is the cost of a plan checked by
exact propagation.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.optimize import linprog

from ..agents.base import AgentView, hold_point
from ..constraints import KeepOut, segment_minima
from ..dynamics import cw_discrete
from .primer import NORMS, dual_bound

if TYPE_CHECKING:
    from ..metrics.efficiency import FuelBounds

# HiGHS's default primal and dual feasibility tolerance.
_FEASIBILITY_TOL = 1e-7
# A cut that binds at no step for this many solves in a row leaves the set; Kelley adds it back
# if a step needs it again.
_IDLE_SOLVES = 3
# Keep-out samples per step, as the environment samples its constraint checks.
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
    """Settings of `FuelOracle`.

    Parameters
    ----------
    cells_per_step : int
        Primer cells per step for the certified excess integral and supremum of `dual_bound`.
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

    Raises
    ------
    ValueError
        If `cells_per_step` or `cut_rounds` is below 1, `inner_directions` is negative, `aim` is
        outside (0, 1) or `cut_tol` is not positive and finite.
    """

    cells_per_step: int = 8
    aim: float = 0.9
    cut_rounds: int = 20
    cut_tol: float = 1e-4
    inner_directions: int = 64

    def __post_init__(self):
        for name in ("cells_per_step", "cut_rounds"):
            if operator.index(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if operator.index(self.inner_directions) < 0:
            raise ValueError("inner_directions must not be negative")
        if not 0.0 < self.aim < 1.0:
            raise ValueError("aim must lie in (0, 1)")
        if not (math.isfinite(self.cut_tol) and self.cut_tol > 0.0):
            raise ValueError("cut_tol must be positive and finite")


@dataclass(frozen=True)
class OracleSolution:
    """Bounds on the least delta-v into the capture ball at one step, and the plan behind them.

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
        Optimum (m/s) of the outer LP on the step grid: the capture ball relaxed to its bounding
        box and, for a ball layout, the 2-norm to its cutting planes; inf when that LP is
        infeasible. No acceleration history held over each step and within the limit costs less.
    upper : float
        Cost (m/s) of `plan`; inf when no plan was verified, `unmodelled` is not empty, or the
        goal asks for a dwell (`hold_time` > 0), which the plan does not hold.
    plan : numpy.ndarray or None, shape (arrival_step, 3)
        Hill accelerations (m/s^2), each held for one step, within `max_accel`, zero in thrust
        inhibit windows, landing within (1 + aim) / 2 of the capture radius and speed and, if
        keep-out is hard, outside it throughout; None when no plan passed.
    arrival_step : int
        The step at whose end the chaser must be in the capture ball.
    unmodelled : tuple of str
        Hard constraints the plan was not checked against.
    """

    bounds: FuelBounds
    finite_lower: float
    impulsive_lower: float
    finite_optimum: float
    upper: float
    plan: np.ndarray | None
    arrival_step: int
    unmodelled: tuple[str, ...]


@dataclass(frozen=True)
class _Window:
    """The fixed-window LP data in scaled units: w = u / a, positions over L, velocities over
    L n, cost over L n."""

    steps: int
    reach: np.ndarray
    miss: np.ndarray
    ball: np.ndarray
    scale: np.ndarray
    cost: float
    inhibited: np.ndarray


@dataclass(frozen=True)
class _Solution:
    """An LP solution in scaled units: optimum, inputs w (k, 3), the cost the LP charges each
    step s (k,), and the terminal multipliers (6,)."""

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


def _checked(status: int, message: str, infeasible: tuple[int, ...]) -> bool:
    """Whether HiGHS solved the LP: False for the statuses that mean infeasible, raise otherwise."""
    if status in infeasible:
        return False
    if status != 0:
        raise RuntimeError(f"HiGHS failed: {message}")
    return True


def _terminal_rows(
    win: _Window, reach: np.ndarray, ball: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rows and right-hand side of |reach w - miss|_i <= ball_i, upper rows first."""
    return np.vstack([reach, -reach]), np.r_[win.miss + ball, ball - win.miss]


def _box_lp(win: _Window, ball: np.ndarray, limit: bool, inhibit: bool) -> _Solution | None:
    """min sum |w_k|_1 with |w_k,i| <= 1 (if `limit`) and the terminal box of half-widths `ball`,
    with w = w+ - w-; None if infeasible."""
    width = 3 * win.steps
    rows, rhs = _terminal_rows(win, np.hstack([win.reach, -win.reach]), ball)
    top = np.full(width, 1.0 if limit else np.inf)
    if inhibit:
        top[np.repeat(win.inhibited, 3)] = 0.0
    res = linprog(
        np.full(2 * width, win.cost),
        A_ub=rows,
        b_ub=rhs,
        bounds=np.c_[np.zeros(2 * width), np.r_[top, top]],
        method="highs",
    )
    if not _checked(res.status, res.message, (2,)):
        return None
    m = res.ineqlin.marginals
    plus, minus = res.x[:width].reshape(-1, 3), res.x[width:].reshape(-1, 3)
    return _Solution(
        value=float(res.fun), w=plus - minus, s=(plus + minus).sum(axis=1), nu=m[:6] - m[6:]
    )


def _cut_lp(
    win: _Window, ball: np.ndarray, cuts: np.ndarray, limit: bool
) -> tuple[_Solution, np.ndarray] | None:
    """min sum s_k with s_k >= d^T w_k over the cuts, s_k <= 1 if `limit`, and the terminal box
    of half-widths `ball`; with it, which cuts bind at some step. None if infeasible.

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
    rows, bound = _terminal_rows(win, np.hstack([win.reach, np.zeros((6, k))]), ball)
    blocks.append(sparse.csr_matrix(rows))
    rhs.append(bound)
    res = linprog(
        np.concatenate(rhs),
        A_eq=sparse.vstack(blocks, format="csr").T,
        b_eq=-np.r_[np.zeros(3 * k), np.full(k, win.cost)],
        bounds=(0.0, None),
        method="highs",
        # Presolve takes about 40% of these solves and removes almost nothing from them.
        options={"presolve": False},
    )
    # An unbounded dual is an infeasible primal.
    if not _checked(res.status, res.message, (2, 3)):
        return None
    x = res.eqlin.marginals
    # The primal marginal of row i is -y_i.
    y = res.x[-12:]
    solution = _Solution(
        value=-float(res.fun), w=x[: 3 * k].reshape(k, 3), s=x[3 * k :], nu=y[6:] - y[:6]
    )
    return solution, (res.x[: k * m].reshape(k, m) > 0.0).any(axis=0)


def _hull_lp(win: _Window, ball: np.ndarray, directions: np.ndarray) -> _Solution | None:
    """min sum_kd lambda_kd with w_k = sum_d lambda_kd d, sum_d lambda_kd <= 1 and lambda >= 0,
    zero on inhibited steps, and the terminal box of half-widths `ball`; None if infeasible.

    Each w_k lies in the convex hull of the unit directions and the origin, so
    |w_k|_2 <= sum_d lambda_kd <= 1: the plan is within the thrust ball and costs at most the
    LP value.
    """
    k, m = win.steps, len(directions)
    along = np.einsum("ikc,dc->ikd", win.reach.reshape(6, k, 3), directions).reshape(6, k * m)
    rows, bound = _terminal_rows(win, along, ball)
    top = np.where(np.repeat(win.inhibited, m), 0.0, np.inf)
    res = linprog(
        np.full(k * m, win.cost),
        A_ub=sparse.vstack(
            [sparse.csr_matrix(rows), sparse.kron(sparse.identity(k), np.ones((1, m)))],
            format="csr",
        ),
        b_ub=np.r_[bound, np.ones(k)],
        bounds=np.c_[np.zeros(k * m), top],
        method="highs",
        # Presolve takes half of this solve and removes almost nothing.
        options={"presolve": False},
    )
    if not _checked(res.status, res.message, (2,)):
        return None
    weights = res.x.reshape(k, m)
    marginals = res.ineqlin.marginals
    return _Solution(
        value=float(res.fun),
        w=weights @ directions,
        s=weights.sum(axis=1),
        nu=marginals[:6] - marginals[6:12],
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


def _landed_safely(
    view: AgentView,
    start: np.ndarray,
    plan: np.ndarray,
    goal: np.ndarray,
    rho: np.ndarray,
    slack: float,
    keep_out: float | None,
) -> bool:
    """Whether the plan, propagated exactly, ends within `slack` times the capture radius and
    speed of the goal and stays outside the keep-out sphere throughout (`segment_minima`'s
    certified bound between samples)."""
    phi, gamma = cw_discrete(view.n, view.dt)
    states = np.empty((len(plan) + 1, 6))
    states[0] = start
    for i, u in enumerate(plan):
        states[i + 1] = phi @ states[i] + gamma @ u
    miss = states[-1] - goal
    if not (
        np.linalg.norm(miss[:3]) <= slack * rho[0] and np.linalg.norm(miss[3:]) <= slack * rho[1]
    ):
        return False
    if keep_out is None:
        return True
    taus = np.linspace(0.0, view.dt, _SAMPLES)
    phis, gammas = cw_discrete(view.n, taus)
    begins = states[:-1]
    samples = np.einsum("tij,kj->kti", phis, begins) + np.einsum("tij,kj->kti", gammas, plan)

    def state_at(rows: np.ndarray, tau: np.ndarray) -> np.ndarray:
        p, g = cw_discrete(view.n, tau)
        return np.einsum("aij,aj->ai", p, begins[rows]) + np.einsum("aij,aj->ai", g, plan[rows])

    _, _, lowest = segment_minima(
        KeepOut(float(keep_out)),
        state_at,
        taus,
        samples,
        np.linalg.norm(plan, axis=1),
        view.n,
        view.dt * np.arange(len(plan)),
    )
    return bool(np.all(lowest >= 0.0))


class FuelOracle:
    """Fuel bounds into the capture ball at a fixed step on exact CW motion (`Oracle` protocol).

    The outer LP relaxes the capture ball to its bounding box, |x_i - x_g,i| <= rho per
    component, and has no path constraints; for a ball layout it also replaces |u_k|_2 by
    Kelley cuts, one set shared by all steps, started from the 6 axes and 8 diagonals, with cuts
    that bind nowhere for three solves dropped. Its optimum is `finite_optimum` and its terminal
    multipliers give both lower bounds through `dual_bound`; if it is infeasible within the
    thrust limit, the multipliers come from the LP without the limit, and if that is infeasible
    too, both lower bounds are 0. The inner LP aims inside the box of half-width
    aim rho / sqrt(3) and zeroes inhibited steps; for a ball it draws each u_k from the convex
    hull of the final cut set and `inner_directions` Fibonacci directions. Its plan is propagated
    exactly and accepted if it lands within (1 + aim) / 2 of the capture radius and speed and,
    when keep-out is hard, never enters it. A zero capture radius or speed leaves no margin to
    verify and a dwell is not held, so `upper` is then inf.

    Parameters
    ----------
    config : OracleConfig, optional
        Settings; the defaults when omitted.
    """

    def __init__(self, *, config: OracleConfig | None = None):
        self.config = OracleConfig() if config is None else config

    def bounds(self, view: AgentView, start: np.ndarray) -> FuelBounds:
        """Bounds for arrival at the last step of the episode, with every listed constraint hard.

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
        """Lower and upper bounds on the least delta-v into the capture ball at `arrival_step`.

        Parameters
        ----------
        view : AgentView
            Scenario description: mean motion `n` (rad/s), `dt` (s), `max_steps`, `max_accel`
            (m/s^2, per axis for "box", 2-norm for "ball"), `thrust_limit`, a hold or berthing
            goal with `hold_point` (m, Hill), `capture_radius` (m), `capture_speed` (m/s) and
            optionally `goal_velocity` (m/s, Hill) and `thrust_inhibit` windows (s from the
            start), and the listed constraints, of which only "keep_out" is checked.
        start : array_like, shape (6,)
            Hill state at t = 0 (m, m/s).
        arrival_step : int, optional
            Step, 1 to `max_steps`, at whose end the chaser is in the ball; `max_steps` by
            default.
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
            capture radius or speed is negative or not finite, the hold point or goal velocity
            is not 3 finite numbers, `start` is not 6 finite numbers, or `arrival_step` is
            outside 1 to `max_steps`.
        RuntimeError
            If HiGHS fails for a reason other than infeasibility.
        """
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
        x0 = np.asarray(start, dtype=float)
        if x0.shape != (6,) or not np.all(np.isfinite(x0)):
            raise ValueError("start must be 6 finite numbers")
        k = view.max_steps if arrival_step is None else operator.index(arrival_step)
        if not 1 <= k <= view.max_steps:
            raise ValueError(f"arrival_step must lie in 1 to {view.max_steps}")
        x_g = np.r_[r_g, np.asarray(goal.get("goal_velocity", (0.0, 0.0, 0.0)), dtype=float)]
        if x_g.shape != (6,) or not np.all(np.isfinite(x_g)):
            raise ValueError("hold point and goal velocity must be 3 finite numbers each")

        win = _window(view, x0, x_g, rho, k)
        norm = NORMS[view.thrust_limit]
        cuts = _FIRST_CUTS
        finite_optimum = math.inf
        lp = None
        for limit in (True, False):
            if norm == "l1":
                lp = _box_lp(win, win.ball, limit, inhibit=False)
            else:
                lp, cuts = _kelley(win, win.ball, self.config, limit)
            if lp is not None:
                if limit:
                    finite_optimum = lp.value * win.scale[3]
                break
        finite_lower = impulsive_lower = 0.0
        if lp is not None:
            # Objective and rows are scaled: d(m/s) / d(x_i) = (L n / s_i) d(objective) / d(row).
            nu = (lp.nu * win.scale[3] / win.scale)[None]
            times = [k * view.dt]
            cells = self.config.cells_per_step * k
            impulsive_lower = max(
                0.0, dual_bound(view.n, view, x0, nu, times, norm=norm, accel=np.inf, cells=cells)
            )
            # nu over its certified primer peak has no excess, so its finite-thrust bound is the
            # impulsive one.
            finite_lower = max(
                impulsive_lower, dual_bound(view.n, view, x0, nu, times, norm=norm, cells=cells)
            )

        listed = {c["type"]: c for c in view.constraints}
        names = set(listed) if hard is None else set(hard)
        keep_out = listed["keep_out"]["radius"] if "keep_out" in names & listed.keys() else None
        unmodelled = tuple(sorted(names - ({"keep_out"} if keep_out is not None else set())))

        plan = None
        if math.isfinite(finite_optimum) and np.all(rho > 0.0):
            plan = self._plan(view, x0, x_g, rho, win, cuts, norm, keep_out)
        upper = math.inf
        # A plan that only arrives is no success for a goal that asks for a dwell.
        if plan is not None and not unmodelled and not goal.get("hold_time", 0.0) > 0.0:
            order = 1 if norm == "l1" else 2
            upper = float(view.dt * np.linalg.norm(plan, ord=order, axis=1).sum())

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
            arrival_step=k,
            unmodelled=unmodelled,
        )

    def _plan(
        self,
        view: AgentView,
        x0: np.ndarray,
        x_g: np.ndarray,
        rho: np.ndarray,
        win: _Window,
        cuts: np.ndarray,
        norm: str,
        keep_out: float | None,
    ) -> np.ndarray | None:
        """The inner LP's plan (m/s^2) if it passes verification, else None."""
        aim = self.config.aim
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
        if not _landed_safely(view, x0, plan, x_g, rho, 0.5 * (1.0 + aim), keep_out):
            return None
        return plan


def _window(view: AgentView, x0: np.ndarray, x_g: np.ndarray, rho: np.ndarray, k: int) -> _Window:
    """Scaled LP data for arrival at step k."""
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
    powers, gamma = cw_discrete(n, view.dt * np.arange(k + 1))
    # Response at step k to w_j held over step j: Phi((k - 1 - j) dt) Gamma, from the closed form
    # at each duration so long windows carry no product rounding.
    response = (powers[k - 1 :: -1] @ gamma[1]).transpose(1, 0, 2).reshape(6, 3 * k)
    starts = view.dt * np.arange(k)
    inhibited = np.zeros(k, dtype=bool)
    for t_on, t_off in view.goal.get("thrust_inhibit", ()):
        # Window edges lie on step boundaries; the tolerance absorbs the sum of steps.
        inhibited |= (starts < t_off - 1e-6) & (starts + view.dt > t_on + 1e-6)
    return _Window(
        steps=k,
        reach=response * a / scale[:, None],
        miss=(x_g - powers[k] @ x0) / scale,
        ball=np.repeat(rho, 3) / scale,
        scale=scale,
        cost=view.dt * a / (length * n),
        inhibited=inhibited,
    )
