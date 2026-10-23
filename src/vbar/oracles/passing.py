"""Passing-side fuel oracle: a verified upper bound when the keep-out sphere lies across the
transfer.

Outside a ball in three dimensions every path is homotopic to every other, so the classes searched
are passing sides: unit directions e in the plane normal to the line of approach. Per side, the
keep-out sphere |r| >= R is replaced at each step boundary j by the halfspace q_j^T r_j >= R + m,
q_j a unit vector. The first q_j points at the keep-out-free plan's position pushed along e out
to 1.5 R; each later LP takes q_j = r_j / |r_j| from the previous iterate. A halfspace with unit q
lies outside the sphere (|r| >= q^T r), and it linearizes the convex |r| at the previous iterate
(Boyd and Vandenberghe, Convex Optimization, 2004, eq. 3.2), which therefore stays feasible: the
LP value never rises. The halfspaces hold at step boundaries only, so each iterate's plan is
propagated exactly and checked against the sphere itself between samples before it counts.
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

from ..agents.base import AgentView
from ..dynamics import cw_discrete
from .fuel import (
    _FIRST_CUTS,
    FuelOracle,
    OracleConfig,
    _checked,
    _fibonacci,
    _goal_state,
    _hard,
    _landed_safely,
    _start,
    _window,
)
from .primer import NORMS

if TYPE_CHECKING:
    from ..metrics.efficiency import FuelBounds

# The first halfspaces point at the keep-out-free positions pushed out to this many radii.
_PUSH = 1.5
# Default margin of the halfspaces over the keep-out radius, as a fraction of the radius.
_MARGIN = 0.02


@dataclass(frozen=True)
class PassingSide:
    """The iterates of one passing side.

    Parameters
    ----------
    direction : numpy.ndarray, shape (3,)
        Unit vector (Hill) normal to the line of approach, the side the plan passes on.
    costs : tuple of float
        Cost (m/s, in the layout's norm) of each iterate's plan; empty if the first LP is
        infeasible.
    verified : tuple of bool
        Whether each iterate's plan passed verification against the capture ball and the
        keep-out sphere between samples.
    upper : float
        Least cost (m/s) of a verified iterate; inf when none passed.
    plan : numpy.ndarray or None, shape (arrival_step + hold, 3)
        Hill accelerations (m/s^2) of that iterate, each held for one step; None when none
        passed.
    """

    direction: np.ndarray
    costs: tuple[float, ...]
    verified: tuple[bool, ...]
    upper: float
    plan: np.ndarray | None


@dataclass(frozen=True)
class PassingSolution:
    """Bounds on the least delta-v of a success that stays outside the keep-out sphere.

    Parameters
    ----------
    bounds : FuelBounds
        `finite_lower`, `upper` and `impulsive_lower` in the layout's norm, rung "cw", with
        `accel_limit` the view's `max_accel`.
    finite_lower : float
        The fuel oracle's certified lower bound (m/s) at a free final time for acceleration
        histories within `max_accel`; it relaxes the keep-out sphere.
    impulsive_lower : float
        The same for any acceleration history (m/s).
    upper : float
        Cost (m/s) of `plan`; inf when no plan was verified or `unmodelled` is not empty.
    plan : numpy.ndarray or None, shape (arrival_step + hold, 3)
        Hill accelerations (m/s^2), each held for one step, verified as `FuelOracle` verifies
        its plans and, if keep-out is hard, outside the sphere throughout; None without one.
    arrival_step : int or None
        Step at whose end `plan` arrives in the capture ball; None without a plan.
    sides : tuple of PassingSide
        The sides searched; empty when the keep-out-free plan needs no detour, keep-out is soft
        or not listed, `unmodelled` is not empty, or no keep-out-free plan was found.
    unmodelled : tuple of str
        Hard constraints the plan was not checked against.
    """

    bounds: FuelBounds
    finite_lower: float
    impulsive_lower: float
    upper: float
    plan: np.ndarray | None
    arrival_step: int | None
    sides: tuple[PassingSide, ...]
    unmodelled: tuple[str, ...]


def _directions(x0: np.ndarray, x_g: np.ndarray, count: int) -> np.ndarray:
    """`count` unit vectors (count, 3) at equal angles in the plane normal to the line of
    approach, from u = w x d, w the orbit normal made orthogonal to d, towards w."""
    d = x_g[:3] - x0[:3]
    size = np.linalg.norm(d)
    # A start on the hold point has no line; V-bar stands in for it.
    d = d / size if size > 0.0 else np.array([0.0, 1.0, 0.0])
    w = np.array([0.0, 0.0, 1.0]) - d[2] * d
    if np.linalg.norm(w) < 1e-6:
        # A line along the orbit normal: the radial axis takes its place.
        w = np.array([1.0, 0.0, 0.0]) - d[0] * d
    w = w / np.linalg.norm(w)
    u = np.cross(w, d)
    theta = 2.0 * math.pi * np.arange(count) / count
    return np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * w


def _pushed(r: np.ndarray, e: np.ndarray, radius: float) -> np.ndarray:
    """Positions (K, 3) inside `radius` moved along the unit vector e out to it."""
    rho = np.linalg.norm(r, axis=1)
    along = r @ e
    # Clipped where |r| >= radius, whose rows are kept as they are.
    shift = np.sqrt(np.maximum(along**2 - rho**2 + radius**2, 0.0)) - along
    return np.where((rho < radius)[:, None], r + shift[:, None] * e, r)


class _SideLP:
    """The inner LP of `FuelOracle` at one arrival step, in state-space form with the scaled
    state at every boundary 1..K as variables, so that halfspace rows on each position are
    sparse. Only the halfspace rows change between sides and iterations."""

    def __init__(
        self,
        view: AgentView,
        x0: np.ndarray,
        x_g: np.ndarray,
        rho: np.ndarray,
        arrival: int,
        hold: int,
        config: OracleConfig,
    ):
        win = _window(view, x0, x_g, rho, arrival, hold + 1)
        steps = win.steps
        self.view, self.win, self.steps = view, win, steps
        self.norm = NORMS[view.thrust_limit]
        if self.norm == "l1":
            # w = w+ - w-, each in [0, 1].
            self.mix = np.hstack([np.eye(3), -np.eye(3)])
        else:
            # w = sum_d lambda_d d over the cut seeds and Fibonacci directions, sum lambda <= 1,
            # so |w|_2 <= 1: the plan is within the thrust ball and costs at most the LP value.
            self.mix = np.vstack([_FIRST_CUTS, _fibonacci(config.inner_directions)]).T
        width = self.mix.shape[1]
        self.states, self.inputs = 6 * steps, width * steps
        step = sparse.csr_matrix(win.step)
        self.a_eq = sparse.hstack(
            [
                sparse.identity(self.states) - sparse.kron(sparse.eye(steps, k=-1), step),
                -sparse.kron(sparse.identity(steps), win.response[0] @ self.mix),
            ],
            format="csr",
        )
        self.b_eq = np.zeros(self.states)
        self.b_eq[:6] = win.step @ (x0 / win.scale)
        lo = np.r_[np.full(self.states, -np.inf), np.zeros(self.inputs)]
        top = np.where(np.repeat(win.inhibited, width), 0.0, 1.0 if self.norm == "l1" else np.inf)
        hi = np.r_[np.full(self.states, np.inf), top]
        half = config.aim * win.ball / math.sqrt(3.0)
        goal = x_g / win.scale
        lo[6 * (arrival - 1) : self.states] = np.tile(goal - half, hold + 1)
        hi[6 * (arrival - 1) : self.states] = np.tile(goal + half, hold + 1)
        self.bounds = np.c_[lo, hi]
        self.cost = np.r_[np.zeros(self.states), np.full(self.inputs, win.cost)]
        # Position columns of every boundary, and for a ball the sum lambda <= 1 rows.
        self.columns = (6 * np.arange(steps)[:, None] + np.arange(3)).ravel()
        self.budget = None
        if self.norm == "l2":
            self.budget = sparse.hstack(
                [
                    sparse.csr_matrix((steps, self.states)),
                    sparse.kron(sparse.identity(steps), np.ones((1, width))),
                ],
                format="csr",
            )

    def solve(self, q: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Plan (K, 3, m/s^2) and boundary positions (K, 3, m) of the LP with the rows
        q_j^T r_j >= level at every boundary j, q (K, 3) unit; None if infeasible."""
        steps, scale = self.steps, self.win.scale[0]
        rows = sparse.csr_matrix(
            (-q.ravel(), self.columns, 3 * np.arange(steps + 1)),
            shape=(steps, self.states + self.inputs),
        )
        rhs = np.full(steps, -level / scale)
        if self.budget is not None:
            rows = sparse.vstack([rows, self.budget], format="csr")
            rhs = np.r_[rhs, np.ones(steps)]
        res = linprog(
            self.cost,
            A_ub=rows,
            b_ub=rhs,
            A_eq=self.a_eq,
            b_eq=self.b_eq,
            bounds=self.bounds,
            method="highs",
        )
        if not _checked(res.status, res.message, (2,)):
            return None
        w = res.x[self.states :].reshape(steps, -1) @ self.mix.T
        # HiGHS meets bounds to its tolerance; the plan meets them exactly and is then checked.
        if self.norm == "l1":
            w = np.clip(w, -1.0, 1.0)
        else:
            w = w / np.maximum(1.0, np.linalg.norm(w, axis=1))[:, None]
        positions = res.x[: self.states].reshape(steps, 6)[:, :3] * scale
        return self.view.max_accel * w, positions


class PassingSideOracle:
    """Fuel bounds into the capture ball when the keep-out sphere lies across the transfer
    (`Oracle` protocol).

    The lower bounds are `FuelOracle`'s at a free final time, which relax the keep-out sphere:
    every success that stays outside it is a success of the relaxed problem, whichever side it
    passes on, so they hold for all of them. The upper bound starts from the keep-out-free plan
    of `FuelOracle`; if that plan stays outside the sphere it is the answer. Otherwise, at the
    same arrival step, each side runs up to `iterations` LPs: `FuelOracle`'s inner LP plus the
    halfspace q_j^T r_j >= R + margin at every step boundary j, with q_j from the keep-out-free
    position pushed along the side's direction out to 1.5 R on the first LP and from the
    previous iterate's position after it. Each iterate's plan is propagated exactly and verified
    as `FuelOracle` verifies its plans, keep-out included between samples, and the upper bound
    is the cheapest verified iterate over the sides.

    Parameters
    ----------
    config : OracleConfig, optional
        Settings of the fuel oracle and of the side LPs (`aim`, and `inner_directions` for a
        ball layout); the defaults when omitted.
    sides : int
        Number of side directions, at equal angles in the plane normal to the line of approach.
    iterations : int
        LPs per side.
    margin : float, optional
        Height (m) of the halfspaces above the keep-out radius R, any finite value; 0.02 R when
        omitted.

    Raises
    ------
    ValueError
        If `sides` or `iterations` is below 1 or `margin` is not finite.
    """

    def __init__(
        self,
        *,
        config: OracleConfig | None = None,
        sides: int = 8,
        iterations: int = 5,
        margin: float | None = None,
    ):
        if operator.index(sides) < 1:
            raise ValueError("sides must be at least 1")
        if operator.index(iterations) < 1:
            raise ValueError("iterations must be at least 1")
        if margin is not None and not math.isfinite(margin):
            raise ValueError("margin must be finite")
        self.config = OracleConfig() if config is None else config
        self.sides = sides
        self.iterations = iterations
        self.margin = margin

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
        self, view: AgentView, start: ArrayLike, *, hard: Collection[str] | None = None
    ) -> PassingSolution:
        """Lower and upper bounds on the least delta-v of a success that avoids keep-out.

        Parameters
        ----------
        view : AgentView
            Scenario description, as for `FuelOracle.solve`; the keep-out sphere is centred on
            the target with the radius (m) of its "keep_out" constraint.
        start : array_like, shape (6,)
            Hill state at t = 0 (m, m/s).
        hard : collection of str, optional
            Names of the hard constraints; every listed constraint when None.

        Returns
        -------
        PassingSolution
            The bounds, the plan and the sides searched.

        Raises
        ------
        ValueError
            As `FuelOracle.solve`.
        RuntimeError
            If HiGHS fails for a reason other than infeasibility.
        """
        x_g, rho = _goal_state(view)
        x0 = _start(start)
        keep_out, unmodelled = _hard(view, hard)
        names = {c["type"] for c in view.constraints} if hard is None else set(hard)
        if keep_out is not None:
            names.discard("keep_out")
        relaxed = FuelOracle(config=self.config).solve(view, x0, hard=frozenset(names))

        def solution(
            upper: float,
            plan: np.ndarray | None,
            arrival: int | None,
            sides: tuple[PassingSide, ...] = (),
        ) -> PassingSolution:
            from ..metrics.efficiency import FuelBounds

            bounds = FuelBounds(
                lower=relaxed.finite_lower,
                upper=upper,
                norm=relaxed.bounds.norm,
                rung="cw",
                accel_limit=view.max_accel,
                impulsive_lower=relaxed.impulsive_lower,
            )
            return PassingSolution(
                bounds=bounds,
                finite_lower=relaxed.finite_lower,
                impulsive_lower=relaxed.impulsive_lower,
                upper=upper,
                plan=plan,
                arrival_step=arrival,
                sides=sides,
                unmodelled=unmodelled,
            )

        plan, k = relaxed.plan, relaxed.arrival_step
        if unmodelled or plan is None or k is None:
            return solution(math.inf, None, None)
        slack = 0.5 * (1.0 + self.config.aim)
        if keep_out is None or _landed_safely(view, x0, plan, x_g, rho, slack, keep_out, k):
            return solution(relaxed.upper, plan, k)

        lp = _SideLP(view, x0, x_g, rho, k, len(plan) - k, self.config)
        level = keep_out + (_MARGIN * keep_out if self.margin is None else self.margin)
        phi, gamma = cw_discrete(view.n, view.dt)
        x, positions = x0, np.empty((len(plan), 3))
        for i, u in enumerate(plan):
            x = phi @ x + gamma @ u
            positions[i] = x[:3]
        sides = tuple(
            self._side(view, x0, x_g, rho, k, lp, positions, e, keep_out, level)
            for e in _directions(x0, x_g, self.sides)
        )
        best = min(sides, key=lambda side: side.upper)
        if best.plan is None:
            return solution(math.inf, None, None, sides)
        return solution(best.upper, best.plan, k, sides)

    def _side(
        self,
        view: AgentView,
        x0: np.ndarray,
        x_g: np.ndarray,
        rho: np.ndarray,
        arrival: int,
        lp: _SideLP,
        positions: np.ndarray,
        e: np.ndarray,
        keep_out: float,
        level: float,
    ) -> PassingSide:
        """The iterates of the side `e`, from the keep-out-free positions."""
        order = 1 if lp.norm == "l1" else 2
        slack = 0.5 * (1.0 + self.config.aim)
        r = _pushed(positions, e, _PUSH * keep_out)
        q = r / np.linalg.norm(r, axis=1, keepdims=True)
        costs: list[float] = []
        verified: list[bool] = []
        upper, best = math.inf, None
        for _ in range(self.iterations):
            found = lp.solve(q, level)
            if found is None:
                break
            plan, r = found
            cost = float(view.dt * np.linalg.norm(plan, ord=order, axis=1).sum())
            ok = _landed_safely(view, x0, plan, x_g, rho, slack, keep_out, arrival)
            costs.append(cost)
            verified.append(ok)
            if ok and cost < upper:
                upper, best = cost, plan
            size = np.linalg.norm(r, axis=1, keepdims=True)
            # A position at the origin has no direction; its previous row still holds there.
            q = np.where(size > 0.0, r / np.where(size > 0.0, size, 1.0), q)
        return PassingSide(
            direction=e, costs=tuple(costs), verified=tuple(verified), upper=upper, plan=best
        )
