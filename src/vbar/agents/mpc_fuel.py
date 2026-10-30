"""Fuel-optimal MPC on the exact zero-order-hold CW model, and the fuel MPC baseline agent.

The plan minimizes the 1-norm delta-v to rest at the hold point, sum_k dt |u_k|_1, under a box
thrust limit. That linear program can have a whole face of minimizers (Rawlings, Mayne and
Diehl, Model Predictive Control, 2nd ed., exercise 2.19 and section 7.7), so a small strictly
convex term makes the applied move unique, and the plan is found through the six multipliers of
the terminal equality. With a fuel-only stage cost the receding-horizon controller is economic
MPC (section 2.8): it has no stability theorem and need not arrive when its horizon says. Hill
frame, SI units.
"""

from __future__ import annotations

import functools
import math
import operator
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from ..constraints import KeepOut, interval_bounds
from ..dynamics import CWDiscretizer, cw_discrete
from .base import AgentView, action_scale, hold_point
from .lqr import steady_state_input

MODES = ("receding", "shrinking")
# Stop rule on the terminal equality, relative to |rhs| + |G| |u|: six decades above the 4e-16
# that converged plans of 284 and 568 steps reach, and far below any navigation error.
STOP_TOL = 1e-10
# Arrival within 1 cm of the hold point, a fiftieth of the smallest capture radius in the
# catalogue (0.5 m); the logged flown arrival time uses it.
ARRIVAL_M = 0.01
# Keep-out samples per step, as the environment checks its constraints.
VIOLATION_SAMPLES = 20
# Levenberg term relative to the trace of the generalized Hessian plus the residual norm; it
# only orients the step in directions the Hessian does not see, since the line search sets the
# length.
LEVENBERG = 1e-12
# Rounding allowance on the directional derivative of the dual, relative to its terms.
SLOPE_TOL = 1e-12
# Rounding of |g' mu| / dt in units of eps |g|' |mu| / dt: a six-term dot product and a divide.
PRICE_ROUNDING = 8.0
# A warm start gets a quarter of the iterations before the cold start from the linear
# program's multipliers takes over: on the receding transfers of the tests, warm starts that
# converge take at most 23 steps, those that stall would hold the whole cap, and cold starts
# converge in at most 5.
WARM_SHARE = 4


@dataclass(frozen=True, kw_only=True)
class FuelMPCConfig:
    """Horizon and solver settings of `FuelMPCAgent`.

    Parameters
    ----------
    mode : str
        "receding": every decision plans to arrive one horizon later. "shrinking": the arrival
        time is fixed at the first decision and the horizon shrinks to it; the hold input
        follows.
    horizon_s : float or None
        Horizon (s), rounded to whole steps; None takes half the orbit period.
    tie_break : float
        Weight of the strictly convex term, sum_k dt tie_break |u_k|^2 / (2 a_max), relative to
        the fuel. It makes the plan unique. A plan's fuel exceeds the optimum by at most its
        run-time bound (`FuelPlan.fuel_lower`), about tie_break away from the goal.
    max_iter : int
        Updates of the multipliers allowed per plan (see `fuel_plan`).

    Raises
    ------
    ValueError
        If `mode` is unknown, `horizon_s` is not positive and finite, `tie_break` is not
        positive and finite, or `max_iter` is below 1.
    TypeError
        If `max_iter` is not an integer.
    """

    mode: str = "receding"
    horizon_s: float | None = None
    tie_break: float = 1e-6
    max_iter: int = 100

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}, expected one of {MODES}")
        if self.horizon_s is not None and not (
            math.isfinite(self.horizon_s) and self.horizon_s > 0.0
        ):
            raise ValueError("horizon_s must be positive and finite")
        _check_settings(self.tie_break, self.max_iter)


@dataclass(frozen=True)
class FuelPlan:
    """A fuel plan and how the solver ended.

    Attributes
    ----------
    u : numpy.ndarray, shape (steps, 3)
        Hill accelerations (m/s^2) held over each step, first step first.
    fuel : float
        1-norm delta-v of `u` (m/s), sum_k dt |u_k|_1.
    nu : numpy.ndarray, shape (6,)
        Multipliers of the terminal equality: the change of the regularized cost per unit of
        the terminal state, (m/s per m, m/s per m/s).
    status : str
        "converged"; "iteration_cap"; or "infeasible" when no plan within the box reaches the
        goal in time, proved by HiGHS or by a direction along which the dual rises without
        bound. Only a converged plan meets the goal.
    iterations : int
        Updates of the multipliers: Newton steps, plus one for the jump to the linear program's
        multipliers on a cold start.
    fuel_lower : float
        Lower bound (m/s) on the optimum of the linear program without the tie-break: its dual
        function at `nu`, less the rounding of that sum. For any multipliers it bounds the
        optimum from below, so a converged plan's fuel exceeds the optimum by at most
        fuel - fuel_lower; this is the plan's fuel guarantee.
    """

    u: np.ndarray
    fuel: float
    nu: np.ndarray
    status: str
    iterations: int
    fuel_lower: float


def _check_settings(tie_break: float, max_iter: int) -> None:
    if not (math.isfinite(tie_break) and tie_break > 0.0):
        raise ValueError("tie_break must be positive and finite")
    if operator.index(max_iter) < 1:
        raise ValueError("max_iter must be at least 1")


@functools.lru_cache(maxsize=8)
def _horizon_matrices(n: float, dt: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """G (6, 3 steps), with x_N = Phi^N x_0 + G u for u in stage order, and Phi^N; read-only.

    Cached because a receding horizon plans with the same matrices at every decision.
    """
    phi, gamma = cw_discrete(n, dt)
    cols = np.empty((steps, 6, 3))
    cols[-1] = gamma
    for k in range(steps - 2, -1, -1):
        cols[k] = phi @ cols[k + 1]
    g = cols.transpose(1, 0, 2).reshape(6, 3 * steps)
    phi_n = phi @ np.linalg.matrix_power(phi, steps - 1)
    g.flags.writeable = False
    phi_n.flags.writeable = False
    return g, phi_n


def _moves(w: np.ndarray, dt: float, a_max: float, tie_break: float) -> np.ndarray:
    """Minimizer of dt (|u| + tie_break u^2 / (2 a_max)) - w u over |u| <= a_max, per entry."""
    excess = np.abs(w) / dt - 1.0
    return np.copysign(np.clip(excess * (a_max / tie_break), 0.0, a_max), w)


def _pieces(w: np.ndarray, dt: float, tie_break: float) -> np.ndarray:
    """Which piece of `_moves` each entry is on: 0 and 4 saturated, 2 zero, 1 and 3 between."""
    return np.digitize(w, dt * np.array([-1.0 - tie_break, -1.0, 1.0, 1.0 + tie_break]))


def _line_search(
    w: np.ndarray, c: np.ndarray, slope0: float, rd: float, dt: float, a_max: float, tau: float
) -> float:
    """Step along a dual direction to the maximum, or inf when the dual rises without bound.

    Along w + t c the slope of the dual, rd - c . u(t), is piecewise linear and falls with t,
    so its root lies between two consecutive breakpoints of the moves and is found exactly.
    """

    def slope(t: float) -> float:
        return float(rd - c @ _moves(w + t * c, dt, a_max, tau))

    moving = c != 0.0
    levels = dt * np.array([-1.0 - tau, -1.0, 1.0, 1.0 + tau])
    breaks = ((levels[None, :] - w[moving, None]) / c[moving, None]).ravel()
    breaks = np.sort(breaks[breaks > 0.0])
    if breaks.size == 0:
        return math.inf
    last = slope(float(breaks[-1]))
    if last > 0.0:
        # Past the last breakpoint every move is constant, so the slope stays at `last`.
        if last > SLOPE_TOL * (abs(rd) + a_max * float(np.sum(np.abs(c)))):
            return math.inf
        return float(breaks[-1])
    lo, hi = -1, breaks.size - 1
    slope_lo, slope_hi = slope0, last
    while hi - lo > 1:
        mid = (lo + hi) // 2
        s = slope(float(breaks[mid]))
        if s > 0.0:
            lo, slope_lo = mid, s
        else:
            hi, slope_hi = mid, s
    t_lo = 0.0 if lo < 0 else float(breaks[lo])
    t_hi = float(breaks[hi])
    return t_lo + slope_lo * (t_hi - t_lo) / (slope_lo - slope_hi)


def fuel_plan(
    n: float,
    dt: float,
    x0: ArrayLike,
    x_goal: ArrayLike,
    steps: int,
    a_max: float,
    *,
    tie_break: float = 1e-6,
    max_iter: int = 100,
    warm: ArrayLike | None = None,
) -> FuelPlan:
    """Fuel-optimal plan to a state in a fixed number of steps on the exact ZOH CW model.

    Minimizes sum_k dt (|u_k|_1 + tie_break |u_k|^2 / (2 a_max)) subject to x_N = x_goal and
    |u_ki| <= a_max. Given the multipliers nu of the terminal equality, each move is a clip in
    closed form, so the dual is a concave, continuously differentiable function of six
    variables with a unique maximizer of the moves. It is maximized by Newton steps with a
    generalized Hessian, a Levenberg term where that is singular, and an exact line search.
    `warm` gets a quarter of `max_iter`; a cold start, or a warm one that did not converge,
    starts from the multipliers of the linear program without the tie-break (HiGHS dual
    simplex). Whenever the full Newton step keeps every move on its piece, the moves strictly
    inside the box take the minimum-norm change onto the terminal rows, which is that step
    taken in the moves without amplifying the rounding of nu by a_max / tie_break; moves too
    small for the prices to resolve are corrected the same way within the prices' rounding. The
    plan stops when |G u - rhs| <= 1e-10 (|rhs| + |G| |u|), Frobenius norm for G, or at the
    cap. The guarantee on fuel is the run-time one: the plan's fuel exceeds the linear
    program's optimum by at most fuel - fuel_lower, from the program's dual at `nu`. Measured,
    the relative bound is about tie_break for plans 1e-5 m or more from the goal and grows to
    1e-3 at 1e-9 m, where the moves fall below the rounding of the prices.

    Parameters
    ----------
    n : float
        Mean motion of the target's circular orbit (rad/s).
    dt : float
        Step over which each move is held (s).
    x0, x_goal : array_like, shape (6,)
        Start and terminal Hill states (m, m/s).
    steps : int
        Number of steps N, at least 2.
    a_max : float
        Box limit per Hill axis (m/s^2).
    tie_break : float
        Weight of the strictly convex term relative to the fuel.
    max_iter : int
        Updates of the multipliers allowed, at least 1.
    warm : array_like, shape (6,), optional
        Multipliers to start from, such as `nu` of the previous plan.

    Returns
    -------
    FuelPlan
        Moves, fuel, multipliers, status and iteration count.

    Raises
    ------
    ValueError
        If `steps` is below 2 (three moves cannot meet six terminal equalities, so the
        multipliers are not unique), `dt` or `a_max` is not positive and finite, `tie_break`
        is not positive and finite, `max_iter` is below 1, or a state is not six finite numbers.
    TypeError
        If `max_iter` is not an integer.
    """
    if steps < 2:
        raise ValueError("a fuel plan needs at least 2 steps")
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be positive and finite")
    if not (math.isfinite(a_max) and a_max > 0.0):
        raise ValueError("a_max must be positive and finite")
    _check_settings(tie_break, max_iter)
    x0 = np.asarray(x0, dtype=float)
    x_goal = np.asarray(x_goal, dtype=float)
    for state in (x0, x_goal):
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("states must be six finite numbers")

    g_si, phi_n = _horizon_matrices(n, dt, steps)
    dual = _Dual(g_si, x_goal - phi_n @ x0, dt, a_max)
    budget = max_iter
    nu, u, status = np.zeros(6), np.zeros(3 * steps), ""
    if warm is not None:
        share = max(1, max_iter // WARM_SHARE)
        nu, u, status, used = dual.ascend(tie_break, np.asarray(warm, dtype=float), share)
        budget -= used
    if status not in ("converged", "infeasible"):
        if not np.any(dual.rhs):
            nu, u, status = np.zeros(6), np.zeros(3 * steps), "converged"
        elif budget > 0:
            # The jump to the linear program's multipliers is the cold start's first update.
            budget -= 1
            start = dual.lp_multipliers()
            if start is None:
                nu, u, status = np.zeros(6), np.zeros(3 * steps), "infeasible"
            else:
                nu, u, status, used = dual.ascend(tie_break, start, budget)
                budget -= used
    moves = u.reshape(steps, 3)
    return FuelPlan(
        u=moves,
        fuel=float(dt * np.sum(np.abs(moves))),
        nu=nu,
        status=status,
        iterations=max_iter - budget,
        fuel_lower=dual.lp_lower(nu),
    )


class _Dual:
    """The dual of the regularized fuel problem over the six terminal multipliers."""

    def __init__(self, g: np.ndarray, rhs: np.ndarray, dt: float, a_max: float):
        self.g_si, self.rhs_si = g, rhs
        # Rows of unit norm put position and velocity rows on one scale for the Newton system.
        self.row_scale = 1.0 / np.linalg.norm(g, axis=1)
        self.g = g * self.row_scale[:, None]
        self.rhs = rhs * self.row_scale
        self.g_norm, self.rhs_norm = float(np.linalg.norm(g)), float(np.linalg.norm(rhs))
        self.dt, self.a_max = dt, a_max

    def lp_multipliers(self) -> np.ndarray | None:
        """Multipliers of the plan without the tie-break, by HiGHS; None when it is infeasible.

        The program is solved for a right-hand side of unit norm with the moves and the box
        scaled alike, which leaves the multipliers unchanged and makes the absolute tolerances
        of HiGHS relative to the plan.
        """
        size = float(np.linalg.norm(self.rhs))
        found = linprog(
            np.full(2 * self.g.shape[1], self.dt),
            A_eq=np.hstack([self.g, -self.g]),
            b_eq=self.rhs / size,
            bounds=(0.0, self.a_max / size),
            method="highs-ds",
        )
        if found.status == 2:
            return None
        if found.status != 0:
            # HiGHS stopped without an answer; Newton steps from zero may still find one.
            return np.zeros(6)
        return found.eqlin.marginals * self.row_scale

    def lp_lower(self, nu: np.ndarray) -> float:
        """Dual value of the linear program at `nu` less its rounding, a bound below its optimum.

        min dt |u|_1 over |u_i| <= a_max and G u = rhs is at least
        rhs . nu - a_max sum_i max(0, |g_i . nu| - dt) for every nu (weak duality).
        """
        w = self.g_si.T @ nu
        over = np.maximum(np.abs(w) - self.dt, 0.0)
        value = float(self.rhs_si @ nu) - self.a_max * float(np.sum(over))
        eps = np.finfo(float).eps
        # Only terms within their rounding of the price, or above it, can be off; the rest are
        # exactly zero.
        price_rounding = PRICE_ROUNDING * eps * (np.abs(self.g_si).T @ np.abs(nu))
        counted = np.abs(w) > self.dt - price_rounding
        rounding = (
            PRICE_ROUNDING * eps * float(np.abs(self.rhs_si) @ np.abs(nu))
            + self.a_max * float(np.sum(price_rounding[counted]))
            + w.size * eps * self.a_max * float(np.sum(over))
        )
        return value - rounding

    def met(self, u: np.ndarray) -> bool:
        """Whether `u` meets the stop rule on the terminal equality."""
        residual = float(np.linalg.norm(self.rhs_si - self.g_si @ u))
        return residual <= STOP_TOL * (self.rhs_norm + self.g_norm * float(np.linalg.norm(u)))

    def ascend(
        self, tau: float, nu: np.ndarray, budget: int
    ) -> tuple[np.ndarray, np.ndarray, str, int]:
        """Maximize the dual at tie-break `tau` from multipliers `nu` in at most `budget` steps.

        Returns the multipliers, the moves (stage order, flattened), the status and the steps
        used.
        """
        g, rhs, dt, a_max = self.g, self.rhs, self.dt, self.a_max
        gain = a_max / (tau * dt)
        mu = nu / self.row_scale
        used = 0
        while True:
            w = g.T @ mu
            u = _moves(w, dt, a_max, tau)
            if self.met(u):
                return mu * self.row_scale, u, "converged", used
            if used == budget:
                return mu * self.row_scale, u, "iteration_cap", used
            used += 1
            residual = rhs - g @ u
            pieces = _pieces(w, dt, tau)
            inside = (pieces == 1) | (pieces == 3)
            g_in = g[:, inside]
            curvature, basis = np.linalg.eigh(gain * (g_in @ g_in.T))
            projected = basis.T @ residual
            # The pseudo-inverse with the cutoff of numpy.linalg.lstsq.
            kept = curvature > 6 * np.finfo(float).eps * max(float(curvature[-1]), 0.0)
            step = basis[:, kept] @ (projected[kept] / curvature[kept])
            c = g.T @ step
            if np.array_equal(_pieces(w + c, dt, tau), pieces):
                corrected = u.copy()
                corrected[inside] += gain * c[inside]
                corrected = np.clip(corrected, -a_max, a_max)
                if self.met(corrected):
                    return (mu + step) * self.row_scale, corrected, "converged", used
            rounded = self._round_off(w, u, mu, residual, tau)
            if rounded is not None and self.met(rounded):
                return mu * self.row_scale, rounded, "converged", used
            damping = LEVENBERG * (float(np.sum(curvature)) + float(np.linalg.norm(residual)))
            direction = basis @ (projected / (np.maximum(curvature, 0.0) + damping))
            t = _line_search(
                w,
                g.T @ direction,
                float(residual @ direction),
                float(rhs @ direction),
                dt,
                a_max,
                tau,
            )
            if math.isinf(t):
                return mu * self.row_scale, u, "infeasible", used
            mu = mu + t * direction

    def _round_off(
        self, w: np.ndarray, u: np.ndarray, mu: np.ndarray, residual: np.ndarray, tau: float
    ) -> np.ndarray | None:
        """Moves put on the terminal rows by changing prices within their rounding, or None.

        Moves below about eps a_max / tau sit inside the rounding of |w| / dt - 1, so their
        pieces jitter and Newton steps cannot settle them. The minimum-norm correction onto the
        rows over the moves at or inside the fuel price, up to that rounding, is accepted when
        every corrected move keeps the sign of its price and is optimal at a price within the
        rounding of its own: it is then the exact plan for prices that differ from those of `mu`
        by rounding alone.
        """
        dt, a_max = self.dt, self.a_max
        excess = np.abs(w) / dt - 1.0
        slack = PRICE_ROUNDING * np.finfo(float).eps * (np.abs(self.g).T @ np.abs(mu)) / dt
        support = (excess > -slack) & (excess < tau)
        # A shortcut implied by the checks below: each move may change by at most 2 slack_i
        # in price, (2 a_max / tau) slack_i in size, and rows of unit norm give |G_S| <= sqrt 6,
        # so no allowed change removes a larger residual.
        reach = (2.0 * a_max / tau) * float(np.linalg.norm(slack[support])) * math.sqrt(6.0)
        if not np.any(support) or float(np.linalg.norm(residual)) > reach:
            return None
        g_s = self.g[:, support]
        moved = u[support] + g_s.T @ _pseudo_solve(g_s @ g_s.T, residual)
        size = moved * np.sign(w[support])
        if np.any(size < 0.0) or np.any(size > a_max):
            return None
        if np.any(np.abs(size * (tau / a_max) - excess[support]) > slack[support]):
            return None
        corrected = u.copy()
        corrected[support] = moved
        return corrected


def _pseudo_solve(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Minimum-norm least-squares solution of m x = v for a symmetric positive semidefinite m."""
    curvature, basis = np.linalg.eigh(m)
    projected = basis.T @ v
    # The cutoff of numpy.linalg.lstsq.
    kept = curvature > 6 * np.finfo(float).eps * max(float(curvature[-1]), 0.0)
    return basis[:, kept] @ (projected[kept] / curvature[kept])


class FuelMPCAgent:
    """Fuel-optimal MPC to the hold point on the CW model, box thrust limit only.

    Each decision plans `fuel_plan` from the observation to rest at the hold point and applies
    the first move. "receding" plans one horizon ahead at every decision; "shrinking" fixes the
    arrival one horizon after the first decision, applies the plan's last move when one step is
    left and the hold input after the arrival. A plan that did not converge is never applied:
    the agent flies the previous plan shifted by one step, with the hold input appended, or zero
    thrust when it has none. Constraints are not modelled; the keep-out margins of the flown
    plan are logged.

    `log` holds one entry per decision and `fallbacks` the count of decisions by level (0 a plan
    solved at this decision, 1 the previous plan shifted, 2 zero thrust). Both are the agent's
    own bookkeeping, which the recorder and metrics never read as scores.

    Parameters
    ----------
    config : FuelMPCConfig, optional
        Horizon and solver settings; the defaults of `FuelMPCConfig` when omitted.
    name : str
        Name in results.
    """

    def __init__(self, config: FuelMPCConfig | None = None, name: str = "fuel-mpc"):
        self.config = FuelMPCConfig() if config is None else config
        self.name = name
        self.log: list[dict] = []
        self.fallbacks = [0, 0, 0]

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Set the horizon, hold point and hold input for the view's orbit and step.

        Parameters
        ----------
        view : AgentView
            Scenario description; uses `n`, `dt`, `max_accel`, `thrust_limit`, `action_mode`,
            `mass_kg`, `goal["hold_point"]` (Hill frame, m) and the keep-out constraint.
        rng : numpy.random.Generator, optional
            Unused; the law is deterministic.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, the action mode takes
            no continuous command, the thrust limit is not a box, the horizon is shorter than 2
            steps, or the hold point needs more than `max_accel` of steady thrust.
        """
        hold = hold_point(view, "fuel MPC agent")
        self.scale = action_scale(view)
        if view.thrust_limit == "ball":
            raise ValueError(
                "a ball thrust limit makes the fuel plan a second-order cone program; the fuel "
                "MPC takes a box limit"
            )
        if view.thrust_limit != "box":
            raise ValueError(f"unknown thrust limit {view.thrust_limit!r}")
        cfg = self.config
        horizon_s = math.pi / view.n if cfg.horizon_s is None else cfg.horizon_s
        self.steps = round(horizon_s / view.dt)
        if self.steps < 2:
            raise ValueError("the fuel MPC horizon must span at least 2 steps")
        self.view = view
        self.phi, self.gamma = cw_discrete(view.n, view.dt)
        self.goal = np.concatenate([hold, np.zeros(3)])
        self.us = steady_state_input(self.phi, self.gamma, self.goal)
        if np.max(np.abs(self.us)) > view.max_accel:
            raise ValueError(
                f"hold point needs {np.max(np.abs(self.us)):.2e} m/s^2 of steady thrust, above "
                f"max_accel {view.max_accel}"
            )
        radius = next((c["radius"] for c in view.constraints if c["type"] == "keep_out"), None)
        self.keep_out = None if radius is None else KeepOut(float(radius))
        self.disc = CWDiscretizer(view.n)
        self.log = []
        self.fallbacks = [0, 0, 0]
        self.arrival_s: float | None = None
        self.flown_arrival_s: float | None = None
        self.tail: np.ndarray | None = None
        self.nu: np.ndarray | None = None

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Plan from the observation and command the plan's first move.

        Parameters
        ----------
        obs : numpy.ndarray
            Observation whose first six entries are the Hill state (m, m/s).
        info : dict
            Agent-safe information; only `time_s` (s since the episode start) is read.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Command in the view's action mode.
        """
        view, cfg = self.view, self.config
        t = float(info["time_s"])
        x = np.asarray(obs, dtype=float)[:6]
        if self.flown_arrival_s is None and np.linalg.norm(x[:3] - self.goal[:3]) < ARRIVAL_M:
            self.flown_arrival_s = t
        if cfg.mode == "receding":
            arrival = t + self.steps * view.dt
        else:
            if self.arrival_s is None:
                self.arrival_s = t + self.steps * view.dt
            arrival = self.arrival_s
        left = round((arrival - t) / view.dt)

        start = time.perf_counter()
        warm, fuel, excess, lower = False, None, None, None
        certified: bool | None = None
        iterations = 0
        plan_u: np.ndarray | None = None
        if left >= 2:
            warm = self.nu is not None
            plan = fuel_plan(
                view.n,
                view.dt,
                x,
                self.goal,
                left,
                view.max_accel,
                tie_break=cfg.tie_break,
                max_iter=cfg.max_iter,
                warm=self.nu,
            )
            status, iterations = plan.status, plan.iterations
            if plan.status == "converged":
                plan_u, fuel, self.nu = plan.u, plan.fuel, plan.nu
                lower = plan.fuel_lower
                cost = view.dt * float(np.sum(np.abs(plan.u)))
                # The bound holds only when the lower bound is positive and below the cost; a
                # lower bound above the cost means the plan or the bound is wrong, and is logged.
                certified = cost == 0.0 or 0.0 < lower <= cost
                if cost == 0.0:
                    excess = 0.0
                elif certified:
                    excess = cost / lower - 1.0
        else:
            status = "last_move" if left == 1 else "after_arrival"
        if plan_u is not None:
            level = 0
        elif self.tail is not None:
            level = 1
            plan_u = self.tail if len(self.tail) else self.us[None, :]
        else:
            level = 2
            plan_u = np.zeros((1, 3))
        u = np.clip(plan_u[0], -view.max_accel, view.max_accel)
        self.tail = None if level == 2 else plan_u[1:]
        elapsed = time.perf_counter() - start
        self.fallbacks[level] += 1

        margin, bound = self._margins(x, plan_u) if level < 2 else (None, None)
        self.log.append(
            {
                "time_s": t,
                "status": status,
                "iterations": iterations,
                "elapsed_s": elapsed,
                "level": level,
                "fallback": status not in ("converged", "last_move", "after_arrival"),
                "warm_start": warm,
                "solver": "fuel dual newton",
                "tolerance": STOP_TOL,
                "planned_arrival_s": arrival,
                "flown_arrival_s": self.flown_arrival_s,
                "planned_fuel_mps": fuel,
                "fuel_lower_mps": lower,
                "fuel_excess_bound": excess,
                "fuel_bound_certified": certified,
                "predicted_margin": margin,
                "certified_bound": bound,
            }
        )
        return u / view.max_accel * self.scale

    def _margins(self, x: np.ndarray, moves: np.ndarray) -> tuple[float | None, float | None]:
        """Lowest sampled keep-out margin over the plan from `x`, and a certified lower bound (m).

        Every step is sampled at the environment's rate; the bound is the first-order one of
        `vbar.constraints.segment_minimum` between samples, -inf where the samples are too far
        apart for it. None, None without a keep-out constraint.
        """
        if self.keep_out is None:
            return None, None
        starts = np.empty((len(moves), 6))
        starts[0] = x
        for k in range(1, len(moves)):
            starts[k] = self.phi @ starts[k - 1] + self.gamma @ moves[k - 1]
        dt = self.view.dt
        phis, gammas = self.disc.sample_matrices(dt, VIOLATION_SAMPLES)
        samples = np.einsum("sij,kj->ksi", phis, starts) + np.einsum("sij,kj->ksi", gammas, moves)
        margins = self.keep_out.margin(samples)
        h = dt / (VIOLATION_SAMPLES - 1)
        speed, _ = interval_bounds(samples, h, np.linalg.norm(moves, axis=1)[:, None], self.view.n)
        # |d rho / dt| <= |v|, the keep-out rate bound.
        bounds = 0.5 * (margins[:, :-1] + margins[:, 1:] - speed * h)
        margin = float(margins.min())
        return margin, min(float(bounds.min()), margin)
