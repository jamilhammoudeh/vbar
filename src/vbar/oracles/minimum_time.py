"""Minimum-time oracle on the CW model: the fewest held steps into the capture ball.

A step k is reached when a plan of accelerations, each held for one step with |a_i| <= a per
Hill axis, puts the chaser in the capture ball at the end of step k, checked by exact
propagation. A step is unreachable when a multiplier nu satisfies

    D(nu) > a * integral over [0, k dt] of |p(t)|_1 dt,
    D(nu) = nu^T (x_g - Phi(k dt) x0) - rho_r |nu_r|_2 - rho_v |nu_v|_2,

with p the primer of nu: any acceleration history within the box, held per step or not, moves
nu^T x by at most the right side, while entering the ball needs at least D(nu) (a theorem of
alternatives: Boyd and Vandenberghe, Convex Optimization, 2004, section 5.8). The integral is
bounded from above over cells, so the proof holds between evaluation points. A phase I linear
program on the step grid proposes the multiplier; the proof never depends on the solver.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from ..agents.base import AgentView, hold_point
from ..dynamics import cw_discrete
from .fuel import OracleConfig
from .primer import CELLS_PER_STEP, NORMS, _lipschitz, primer

# Cell refinement stops here, four times the 32 per step that proves 141 one-second steps short
# of a 100 m rest-to-rest move; the Lipschitz slack halves with each doubling.
_MAX_CELLS_PER_STEP = 128
# The tightest tolerances HiGHS accepts; a capture ball can be 1e-8 of the scaled distance.
_LP_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
# Scaled phase I miss above which no plan can land, ten times the LP's tolerance.
_REACHED = 1e-9
# Least-squares corrections of the LP plan; clipping after one can leave a residual for the next.
_POLISH_ROUNDS = 3
# Rounding floor of the margin: the evaluated terms carry a few ulps each.
_FLOOR = 64 * np.finfo(float).eps
_INTEGRAL_INFLATION = 1.0 + 1e-12


@dataclass(frozen=True)
class MinimumTime:
    """The fewest steps into the capture ball, with the proof for every smaller step.

    Parameters
    ----------
    steps : int or None
        Smallest step at whose end `plan` puts the chaser in the capture ball; None when no step
        up to `max_steps` was reached.
    certified : bool
        True when every step below `steps` (every step of the episode, when `steps` is None) is
        proved unreachable by a multiplier that `verify_infeasible` accepts.
    plan : numpy.ndarray or None, shape (steps, 3)
        Hill accelerations (m/s^2), each held for one step, within `max_accel` per axis; None
        when `steps` is None.
    uncertified_steps : tuple of int
        Steps below `steps` that are neither reached by a verified plan nor proved unreachable.
    """

    steps: int | None
    certified: bool
    plan: np.ndarray | None
    uncertified_steps: tuple[int, ...]


def _checked(view: AgentView, start: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start state, goal state (m, m/s) and capture radius and speed, after validation."""
    if view.thrust_limit not in NORMS:
        raise ValueError(f"unknown thrust limit {view.thrust_limit!r}")
    if view.thrust_limit != "box":
        raise ValueError("the minimum-time oracle supports box thrusters only")
    for name in ("n", "dt", "max_accel"):
        number = getattr(view, name)
        if not (math.isfinite(number) and number > 0.0):
            raise ValueError(f"{name} must be positive and finite")
    r_g = hold_point(view, "minimum-time oracle")
    goal = view.goal
    if not {"capture_radius", "capture_speed"} <= goal.keys():
        raise ValueError("the goal has no capture ball")
    if goal.get("thrust_inhibit"):
        raise ValueError("the minimum-time oracle does not model thrust inhibit windows")
    rho = np.array([goal["capture_radius"], goal["capture_speed"]], dtype=float)
    if not (np.all(np.isfinite(rho)) and np.all(rho >= 0.0)):
        raise ValueError("capture radius and speed must be finite and nonnegative")
    x0 = np.asarray(start, dtype=float)
    if x0.shape != (6,) or not np.all(np.isfinite(x0)):
        raise ValueError("start must be 6 finite numbers")
    x_g = np.r_[r_g, np.asarray(goal.get("goal_velocity", (0.0, 0.0, 0.0)), dtype=float)]
    if x_g.shape != (6,) or not np.all(np.isfinite(x_g)):
        raise ValueError("hold point and goal velocity must be 3 finite numbers each")
    return x0, x_g, rho


def _primer_integral(n: float, nu: np.ndarray, t_f: float, cells: int) -> tuple[float, float]:
    """Certified upper bound and trapezoid estimate of the integral of |p|_1 over [0, t_f]."""
    size = np.abs(primer(n, nu, t_f - np.linspace(0.0, t_f, cells + 1)))
    h = t_f / cells
    tops = np.maximum(size[:-1], size[1:]) + 0.5 * h * _lipschitz(n, nu)
    trapezoid = 0.5 * h * float((size[:-1] + size[1:]).sum())
    return h * float(tops.sum()) * _INTEGRAL_INFLATION, trapezoid


def _unreachable(
    view: AgentView,
    x0: np.ndarray,
    x_g: np.ndarray,
    rho: np.ndarray,
    nu: np.ndarray,
    step: int,
    cells_per_step: int,
    max_cells_per_step: int,
) -> bool:
    """Whether `nu` proves the capture ball out of reach at the end of `step`, with cells doubled
    from `cells_per_step` up to `max_cells_per_step` until it does."""
    t_f = step * view.dt
    phi, _ = cw_discrete(view.n, t_f)
    terms = np.array(
        [
            nu @ (x_g - phi @ x0),
            -rho[0] * np.linalg.norm(nu[:3]),
            -rho[1] * np.linalg.norm(nu[3:]),
        ]
    )
    d, floor = float(terms.sum()), _FLOOR * float(np.abs(terms).sum())
    a = view.max_accel
    per_step = cells_per_step
    while True:
        upper, estimate = _primer_integral(view.n, nu, t_f, per_step * step)
        if d - a * upper - floor - _FLOOR * a * upper > 0.0:
            return True
        # Refining closes the gap to the integral, which the trapezoid rule estimates; a
        # multiplier with no estimated margin is not worth more cells.
        if 2 * per_step > max_cells_per_step or not d - a * estimate > floor:
            return False
        per_step *= 2


def _step_grid(
    view: AgentView, x0: np.ndarray, x_g: np.ndarray, rho: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scaled reach (6, 3k) of the held inputs over the box, miss and ball half-widths, and the
    state scale, for arrival at the end of step k."""
    t_f = k * view.dt
    # Velocities scale by the window length rather than 1/n, which is useless as n goes to 0.
    length = max(
        np.linalg.norm(x0[:3]),
        np.linalg.norm(x_g[:3]),
        rho[0],
        np.linalg.norm(x0[3:]) * t_f,
        np.linalg.norm(x_g[3:]) * t_f,
        rho[1] * t_f,
    )
    if length == 0.0:
        length = 1.0
    scale = np.r_[np.full(3, length), np.full(3, length / t_f)]
    powers, gamma = cw_discrete(view.n, view.dt * np.arange(k + 1))
    # Response at step k to an input held over step j: Phi((k - 1 - j) dt) Gamma, from the
    # closed form at each duration so long windows carry no product rounding.
    response = (powers[k - 1 :: -1] @ gamma[1]).transpose(1, 0, 2).reshape(6, 3 * k)
    return (
        response * view.max_accel / scale[:, None],
        (x_g - powers[k] @ x0) / scale,
        np.repeat(rho, 3) / scale,
        scale,
    )


def _phase_one(reach: np.ndarray, miss: np.ndarray, ball: np.ndarray) -> tuple[float, np.ndarray]:
    """Least s with |reach w - miss|_i <= ball_i + s and |w_i| <= 1, and its multiplier on the
    terminal rows, in the scaled units."""
    width = reach.shape[1]
    ones = np.ones((6, 1))
    res = linprog(
        np.r_[np.zeros(width), 1.0],
        A_ub=np.block([[reach, -ones], [-reach, -ones]]),
        b_ub=np.r_[miss + ball, ball - miss],
        bounds=np.r_[np.tile([-1.0, 1.0], (width, 1)), [[0.0, np.inf]]],
        method="highs",
        options=_LP_OPTIONS,
    )
    if res.status != 0:
        raise RuntimeError(f"HiGHS failed: {res.message}")
    m = res.ineqlin.marginals
    return float(res.fun), m[:6] - m[6:]


def _least_fuel(reach: np.ndarray, miss: np.ndarray, ball: np.ndarray) -> np.ndarray | None:
    """Scaled inputs of least 1-norm within |w_i| <= 1 landing in the box of half-widths `ball`;
    None if there are none."""
    width = reach.shape[1]
    res = linprog(
        np.ones(2 * width),
        A_ub=np.block([[reach, -reach], [-reach, reach]]),
        b_ub=np.r_[miss + ball, ball - miss],
        bounds=(0.0, 1.0),
        method="highs",
        options=_LP_OPTIONS,
    )
    if res.status == 2:
        return None
    if res.status != 0:
        raise RuntimeError(f"HiGHS failed: {res.message}")
    # HiGHS meets bounds to its tolerance; the plan meets them exactly and is then checked.
    w = np.clip(res.x[:width] - res.x[width:], -1.0, 1.0)
    # HiGHS drops matrix entries below 1e-9, such as the cross-axis coupling when n t is small,
    # so its landing can miss a small ball. Least squares on the inputs inside the box aims the
    # exact reach at the centre.
    for _ in range(_POLISH_ROUNDS):
        free = np.abs(w) < 1.0 - 1e-9
        if not free.any():
            break
        w[free] += np.linalg.lstsq(reach[:, free], miss - reach @ w, rcond=None)[0]
        w = np.clip(w, -1.0, 1.0)
    return w


def _lands(
    view: AgentView,
    x0: np.ndarray,
    x_g: np.ndarray,
    rho: np.ndarray,
    plan: np.ndarray,
    slack: float,
) -> bool:
    """Whether the plan, propagated exactly step by step, ends strictly within `slack` times the
    capture radius and speed of the goal."""
    phi, gamma = cw_discrete(view.n, view.dt)
    x = x0
    for u in plan:
        x = phi @ x + gamma @ u
    return bool(
        np.linalg.norm(x[:3] - x_g[:3]) < slack * rho[0]
        and np.linalg.norm(x[3:] - x_g[3:]) < slack * rho[1]
    )


def verify_infeasible(view: AgentView, start: ArrayLike, nu: ArrayLike, step: int) -> bool:
    """Whether `nu` proves that no acceleration within the box reaches the capture ball at `step`.

    True when D(nu) > a times a certified upper bound of the integral of |p(t)|_1 over
    [0, step dt], p the primer of `nu` and D(nu) the support margin of the capture ball (module
    docstring). Then no acceleration history with |a_i(t)| <= `max_accel` per Hill axis on
    [0, step dt], held per step or not, puts the chaser in the closed capture ball at
    t = step dt, on exact CW motion with no other forcing. The integral is bounded over cells
    of dt / 8, halved down to dt / 128 while the trapezoid estimate of the integral leaves a
    margin.

    Parameters
    ----------
    view : AgentView
        Scenario description: mean motion `n` (rad/s), `dt` (s), `max_steps`, `max_accel`
        (m/s^2 per axis), `thrust_limit` "box", and a hold or berthing goal with `hold_point`
        (m, Hill), `capture_radius` (m), `capture_speed` (m/s) and optionally `goal_velocity`
        (m/s, Hill).
    start : array_like, shape (6,)
        Hill state at t = 0 (m, m/s).
    nu : array_like, shape (6,)
        Multiplier of the Hill state at t = step dt, in the units of `dual_bound`: position
        part in 1/s, velocity part dimensionless. Any positive multiple gives the same answer.
    step : int
        Step, 1 to `max_steps`, at whose end the ball is claimed out of reach.

    Returns
    -------
    bool
        True if the proof holds; False says nothing about the step.

    Raises
    ------
    ValueError
        If the thrust layout is not "box", `n`, `dt` or `max_accel` is not positive and finite,
        the goal is not a hold or berthing, has no hold point and capture ball or has thrust
        inhibit windows, the capture radius or speed is negative or not finite, the hold point
        or goal velocity is not 3 finite numbers, `start` or `nu` is not 6 finite numbers, or
        `step` is outside 1 to `max_steps`.
    """
    x0, x_g, rho = _checked(view, start)
    nu = np.asarray(nu, dtype=float)
    if nu.shape != (6,) or not np.all(np.isfinite(nu)):
        raise ValueError("nu must be 6 finite numbers")
    step = operator.index(step)
    if not 1 <= step <= view.max_steps:
        raise ValueError(f"step must lie in 1 to {view.max_steps}")
    return _unreachable(view, x0, x_g, rho, nu, step, CELLS_PER_STEP, _MAX_CELLS_PER_STEP)


def minimum_time(
    view: AgentView, start: ArrayLike, *, config: OracleConfig | None = None
) -> MinimumTime:
    """The fewest steps into the capture ball on exact CW motion with box thrusters.

    Steps are tried in order from 1. At each, a multiplier that proved an earlier step
    unreachable is tried first at `cells_per_step`; otherwise a phase I LP on the zero-order-hold
    step grid finds the least violation s of the capture ball's bounding box,
    |x_k,i - x_g,i| <= rho + s per component, with |u_i| <= `max_accel`. If s is above the LP's
    accuracy no held plan lands, and its terminal multiplier is checked as by
    `verify_infeasible`, with cells refined from `cells_per_step` per step. Otherwise the least
    1-norm plan aiming inside aim rho / sqrt(3) per component is propagated exactly and accepted
    if it ends within (1 + aim) / 2 of the capture radius and speed. A step that is neither
    proved unreachable nor reached by an accepted plan goes to `uncertified_steps`.

    The answer is the arrival step: a dwell (`hold_time`) and path constraints such as keep-out
    are not part of the problem, so `plan` may violate them, and the steps proved unreachable
    stay unreachable with them.

    Parameters
    ----------
    view : AgentView
        Scenario description, as for `verify_infeasible`.
    start : array_like, shape (6,)
        Hill state at t = 0 (m, m/s).
    config : OracleConfig, optional
        Its `cells_per_step` and `aim` are used; the defaults when omitted.

    Returns
    -------
    MinimumTime
        The arrival step, its plan, and whether every smaller step is proved unreachable.

    Raises
    ------
    ValueError
        As `verify_infeasible`, for `view` and `start`.
    RuntimeError
        If HiGHS fails for a reason other than infeasibility.
    """
    config = OracleConfig() if config is None else config
    x0, x_g, rho = _checked(view, start)
    slack = 0.5 * (1.0 + config.aim)
    top = max(config.cells_per_step, _MAX_CELLS_PER_STEP)
    uncertified: list[int] = []
    proof = None
    for k in range(1, view.max_steps + 1):
        # The multiplier of an earlier step usually proves the next steps too, which saves the
        # LP until the margin runs out.
        if proof is not None and _unreachable(
            view, x0, x_g, rho, proof, k, config.cells_per_step, config.cells_per_step
        ):
            continue
        reach, miss, ball, scale = _step_grid(view, x0, x_g, rho, k)
        violation, multiplier = _phase_one(reach, miss, ball)
        if violation > _REACHED:
            # Rows were divided by the scale, so the physical multiplier is divided by it too.
            nu = multiplier / scale
            if _unreachable(view, x0, x_g, rho, nu, k, config.cells_per_step, top):
                proof = nu
            else:
                uncertified.append(k)
            continue
        w = _least_fuel(reach, miss, config.aim * ball / math.sqrt(3.0))
        if w is not None:
            plan = view.max_accel * w.reshape(k, 3)
            if _lands(view, x0, x_g, rho, plan, slack):
                return MinimumTime(k, not uncertified, plan, tuple(uncertified))
        uncertified.append(k)
    return MinimumTime(None, not uncertified, None, tuple(uncertified))
