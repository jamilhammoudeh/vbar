"""Model predictive control on the zero-order-hold CW model: terminal set, horizon QP and agent.

The terminal set is the maximal output admissible set of a stable closed loop, computed by the
algorithm of Gilbert and Tan, "Linear systems with state and control constraints: the theory and
application of maximal output admissible sets", IEEE Transactions on Automatic Control 36(9),
1991, as used for terminal sets in Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed.,
section 2.5.4 and exercise 2.5.

The horizon problem is the LQR regulator's stage cost about the hold point with the DARE terminal
cost (Rawlings, Mayne and Diehl, sections 2.5 and 8.8), the thrust box, and keep-out rows at
samples within each step, linearized at a reference plan and softened by an exact penalty (Boyd
and Vandenberghe, Convex Optimization, exercise 5.16). `MPCAgent` solves it at every decision
and applies the first move, with fallbacks when the solver gives no plan. Hill frame, SI units.
"""

from __future__ import annotations

import dataclasses
import math
import time
import warnings
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from ..constraints import KeepOut, interval_bounds, segment_minimum
from ..dynamics import cw_discrete
from ..solvers import IPMSettings, StageQP, riccati_solve, solve_horizon
from .base import AgentView, action_scale, hold_point
from .lqr import ALIGNMENT_TOL, LQRConfig, RegulatorModel, regulator_model, steady_state_input

# Within 1e-12 of the unit circle the determination index grows past any cap (2.3e12 steps per
# decade of decay), and a closed loop with a mode on the circle has no finite description.
STABILITY_TOL = 1e-12
# A row whose maximum over the set equals its bound comes back from the LP above the bound by
# rounding, which would add rows the set does not need. The margin is relative to the bound.
REDUNDANCY_TOL = 1e-9

TERMINALS = ("cost", "set")
INSTEP_BOUNDS = ("first_order", "second_order")
SIDES = ("auto", "+x", "-x", "+z", "-z")
# Rows per stage ahead of the keep-out rows: the six thrust bounds and the slack's bound.
BOX_AND_SLACK_ROWS = 7
# The slack input is sigma = s + 1 m with the bound row sigma >= 1 m. On the row s >= 0,
# `solve_horizon` stops with "solver_failed" even at the optimum: the row's only term is |s|,
# which goes to zero there, so its complementarity test has no scale to pass against.
SLACK_OFFSET_M = 1.0
# Largest sum of the keep-out multipliers of one stage in the hard problem, 205.958 per m, over
# ten approach-vbar starts and ten passes 55 to 90 m from the target (default weights, horizon
# 20), divided by that model's mean position weight per step, 0.1000046 per m^2. Multiplying by
# the model's own weight lets the default penalty follow the cost scale.
PENALTY_LENGTH_M = 2059.49
PENALTY_FACTOR = 10.0
# Curvature of the slack's cost relative to its price per metre of the keep-out radius: small
# against the price, and enough to keep the input block positive definite.
SLACK_CURVATURE = 1e-6
# A Riccati plan whose rows hold up to rounding of their terms is the constrained optimum.
ROW_ROUNDING = 1e-12
# HiGHS meets the scaled rows to a feasibility tolerance of 1e-7, so an extent of the terminal
# set read from an LP is enlarged by ten times that before it bounds the allowance.
EXTENT_MARGIN = 1e-6
TERMINAL_CACHE_SIZE = 8
_TERMINAL_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
# Samples of the held step in the keep-out certificate, as the environment checks constraints.
VIOLATION_SAMPLES = 20
# Passing waypoints sit 1.2 R from the target, 60 m for the 50 m sphere of the catalogue;
# a side is contested when the straight segment to the hold point passes closer than that.
WAYPOINT_RADIUS = 1.2
# A reference leg to a waypoint ends within 0.2 R of it, where the leg to the hold point starts.
WAYPOINT_REACHED = 0.2
# A keep-out row asks only for the range of its reference sample, so reference states nearer
# than 1.1 R are moved radially out to 1.1 R before the rows are built: the margin covers the
# range a reference loses within a step at closing speeds up to 0.1 R / dt.
REFERENCE_RADIUS = 1.1
# The warm start's slack sits 1 mm above zero. The shifted plan is its own reference, so it meets
# its rows to rounding where a row asks only for the reference's range; started on them, the
# interior-point method's first Newton system takes multipliers over slacks of 1e-14 and fails to
# factor ("solver_failed" at iteration 0), which this start avoids. It moves only the starting
# point passed as `warm`; the problem's rows, and the slack bound sigma >= 1 m, are unchanged.
WARM_SLACK_M = 1e-3
# Below this sine of the angle between the radial axis and the segment to the hold point, the
# radial direction across the segment is set by navigation noise, and the side is chosen across
# track instead.
PARALLEL_TOL = 0.1


def admissible_set(
    f: ArrayLike, h: ArrayLike, bound: ArrayLike, *, t_max: int = 200
) -> tuple[np.ndarray, int]:
    """Maximal admissible set of x+ = F x under the output rows |h_i x| <= bound_i.

    The set {x : |h_i F^t x| <= bound_i for all i and all t >= 0} is described by the rows
    h F^t for t = 0, ..., t*, where t* is the first t at which every row of h F^(t+1) is
    redundant over the rows so far (Gilbert and Tan, 1991, algorithm 3.2). Redundancy is
    checked by one linear program per candidate row, in coordinates where each row is divided
    by its bound, so the result does not depend on the units of the bound.

    Parameters
    ----------
    f : array_like, shape (n, n)
        Closed-loop transition matrix of one step, spectral radius below 1.
    h : array_like, shape (p, n)
        Output rows, in the units of `bound` per unit of state.
    bound : array_like, shape (p,)
        Positive bound on |h_i x| for each output row.
    t_max : int, optional
        Largest determination index accepted.

    Returns
    -------
    rows : numpy.ndarray, shape ((t* + 1) p, n)
        The rows h, h F, ..., h F^t* stacked by time; row j has the bound bound[j mod p].
    t_star : int
        Determination index t*.

    Raises
    ------
    ValueError
        If the shapes disagree, f or h is not finite, a bound is not positive and finite, a
        row of h is zero, t_max is negative, the spectral radius of F is 1 - 1e-12 or more,
        or t* exceeds t_max. A pair (F, h) unobservable only up to rounding is refused at the
        cap; write it in exactly unobservable coordinates instead.
    RuntimeError
        If a redundancy LP ends without an optimum or a proof of unboundedness, or the set
        cannot be certified: for each row of h F^(t*+1), multipliers on independent returned
        rows, solved in exact rational arithmetic, must bound its maximum over the set by its
        bound times 1 + 1e-9. A row with no exact multipliers lies outside the span of the
        returned rows, where the set is unbounded; it is kept and the iteration goes on, as
        after an unbounded LP.
    """
    f = np.asarray(f, dtype=float)
    h = np.asarray(h, dtype=float)
    bound = np.asarray(bound, dtype=float)
    if f.ndim != 2 or f.shape[0] != f.shape[1]:
        raise ValueError("f must be a square matrix")
    if h.ndim != 2 or h.shape[1] != f.shape[0]:
        raise ValueError("h must have one column per state of f")
    if bound.shape != (h.shape[0],):
        raise ValueError("bound must have one entry per row of h")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(h)):
        raise ValueError("f and h must be finite")
    if not np.all(np.isfinite(bound) & (bound > 0.0)):
        raise ValueError("bound must be positive and finite")
    if not np.all(np.any(h != 0.0, axis=1)):
        raise ValueError("h has a row of zeros")
    if t_max < 0:
        raise ValueError("t_max must be nonnegative")
    if not np.max(np.abs(np.linalg.eigvals(f))) < 1.0 - STABILITY_TOL:
        raise ValueError("terminal set needs a stable closed loop")

    rows = [h]
    scaled = [h / bound[:, None]]
    for t in range(t_max + 1):
        current = np.vstack(scaled)
        maximizers = _maximizers(scaled[-1] @ f, current)
        if maximizers is not None:
            stacked = np.vstack(rows)
            limits = np.tile(bound, t + 1)
            checks = list(zip(rows[-1] @ f, bound, maximizers, strict=True))
            if all(_certified(row, limit, stacked, limits, x) for row, limit, x in checks):
                return stacked, t
            if all(_multipliers(row, stacked, limits, x) is not None for row, _, x in checks):
                raise RuntimeError("terminal set could not be certified")
        rows.append(rows[-1] @ f)
        scaled.append(scaled[-1] @ f)
    raise ValueError("terminal set is not finitely determined within t_max steps")


def _maximizers(candidates: np.ndarray, rows: np.ndarray) -> list[np.ndarray] | None:
    """Maximizer over {x : |rows x| <= 1} of each candidate, or None once one is not redundant."""
    # Gains in SI units span many decades between position and velocity columns, and rows decay
    # as F^t. Columns are scaled to unit maximum and every row, candidate included, to unit
    # norm, with the right-hand sides scaled to match.
    column = np.max(np.abs(rows), axis=0)
    column[column == 0.0] = 1.0
    a = rows / column
    size = np.linalg.norm(a, axis=1)
    a = a[size > 0.0] / size[size > 0.0, None]
    b = 1.0 / size[size > 0.0]
    # The set is symmetric about the origin, so the maximum of -row . x equals that of row . x
    # and one LP covers both signs.
    a_ub = np.vstack([a, -a])
    b_ub = np.concatenate([b, b])
    maximizers = []
    for row in candidates:
        c = row / column
        c_size = np.linalg.norm(c)
        if c_size == 0.0:
            maximizers.append(np.zeros(len(row)))
            continue
        # HiGHS presolve on these nearly parallel rows has reported an unbounded program as
        # infeasible and stopped 5e-5 below the optimum of another, so it is off. The origin
        # is feasible, so a reported infeasibility is a numerical failure, retried by interior
        # point.
        lp = linprog(
            -c / c_size,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=(None, None),
            method="highs-ds",
            options={"presolve": False},
        )
        if lp.status == 2:
            lp = linprog(-c / c_size, A_ub=a_ub, b_ub=b_ub, bounds=(None, None), method="highs-ipm")
        if lp.status == 3:
            return None
        if lp.status != 0:
            raise RuntimeError(f"terminal set LP failed: {lp.message}")
        if -lp.fun * c_size > 1.0 + REDUNDANCY_TOL:
            return None
        maximizers.append(lp.x / column)
    return maximizers


def _certified(
    row: np.ndarray, limit: float, rows: np.ndarray, limits: np.ndarray, x: np.ndarray
) -> bool:
    """Whether |row . z| <= limit (1 + REDUNDANCY_TOL) is proved over {z : |rows_j z| <= limits_j}.

    Any multipliers lam with sum_j lam_j rows_j = row bound the maximum by sum_j |lam_j|
    limits_j. False when the row has no exact multipliers on the rows.
    """
    found = _multipliers(row, rows, limits, x)
    if found is None:
        return False
    basis, lam = found
    proved = sum(
        (abs(lam_j) * Fraction(float(limits[j])) for lam_j, j in zip(lam, basis, strict=True)),
        Fraction(0),
    )
    return proved <= Fraction(float(limit)) * (1 + Fraction(REDUNDANCY_TOL))


def _multipliers(
    row: np.ndarray, rows: np.ndarray, limits: np.ndarray, x: np.ndarray
) -> tuple[list[int], list[Fraction]] | None:
    """Exact multipliers expressing `row` on independent rows; None off the rows' span.

    The rows are taken by their activity at the LP maximizer `x` up to a maximal independent
    set, and the multipliers solved in rational arithmetic on the float entries, so a bound
    built on them does not rest on the LP's tolerances.
    """
    activity = np.abs(rows @ x) / limits
    # Rank is judged with unit columns, since position gains can sit 1e-9 below velocity gains;
    # scaling columns leaves the multipliers unchanged.
    column = np.max(np.abs(rows), axis=0)
    column[column == 0.0] = 1.0
    unit = rows / column
    basis: list[int] = []
    for j in np.argsort(-activity, kind="stable"):
        if np.linalg.matrix_rank(unit[[*basis, j]]) > len(basis):
            basis.append(int(j))
            if len(basis) == rows.shape[1]:
                break
    lam = _exact_solve(rows[basis].T, row)
    return None if lam is None else (basis, lam)


def _exact_solve(m: np.ndarray, v: np.ndarray) -> list[Fraction] | None:
    """Exact solution y of m y = v, m of full column rank, in rational arithmetic.

    None when the system has no exact solution or m is rank deficient in exact arithmetic.
    """
    unknowns = m.shape[1]
    aug = [
        [Fraction(float(e)) for e in m_row] + [Fraction(float(v_i))]
        for m_row, v_i in zip(m, v, strict=True)
    ]
    for k in range(unknowns):
        pivot = next((i for i in range(k, len(aug)) if aug[i][k] != 0), None)
        if pivot is None:
            return None
        aug[k], aug[pivot] = aug[pivot], aug[k]
        for i in range(len(aug)):
            if i != k and aug[i][k] != 0:
                ratio = aug[i][k] / aug[k][k]
                aug[i] = [e - ratio * e_k for e, e_k in zip(aug[i], aug[k], strict=True)]
    # Equations past the pivots have no coefficients left; a nonzero right side is a residual.
    if any(aug[i][unknowns] != 0 for i in range(unknowns, len(aug))):
        return None
    return [aug[k][unknowns] / aug[k][k] for k in range(unknowns)]


@dataclass(frozen=True, kw_only=True)
class MPCConfig:
    """Horizon, terminal ingredients, keep-out rows and solver settings of the hold-point MPC.

    Parameters
    ----------
    regulator : LQRConfig
        Weights, stage-cost convention and rate weight of the regulator whose stage cost and
        DARE solution the problem uses (see `regulator_model`).
    horizon : int
        Steps N in the horizon, at least 1.
    terminal : {"cost", "set"}
        "cost": the terminal cost beta P alone. "set": the same cost and, as hard terminal rows,
        the maximal admissible set of the LQR loop under the thrust box and the hold-point
        halfspace.
    beta : float
        Terminal cost inflation, at least 1 (Rawlings, Mayne and Diehl, theorem 2.41). Only
        beta = 1 makes the unconstrained first move the LQR law.
    t_max : int
        Largest determination index of the terminal set (see `admissible_set`), at least 0.
    row_samples : int
        Keep-out samples per step, both ends included, at least 2.
    instep_bound : {"first_order", "second_order"}
        Tightening of each keep-out row for the range lost between samples: V h / 2, or
        (V h)^2 / (8 R) + A h^2 / 8, with V (m/s) and A (m/s^2) from `interval_bounds` on the
        reference samples, h (s) the sample interval and R (m) the keep-out radius.
    penalty : float or None
        Price rho of one metre of keep-out slack per stage, in units of the stage cost per m;
        None takes 10 times the largest keep-out multiplier sum found on approach-vbar, scaled
        by the model's position weight.
    side : {"auto", "+x", "-x", "+z", "-z"}
        Side of the keep-out sphere on which a controller passes; "auto" lets it choose.
        `mpc_problem` does not read it.
    side_hysteresis : int
        Failed decisions in a row before an "auto" side is chosen again, at least 1.
        `mpc_problem` does not read it.
    settings : IPMSettings or None
        Interior-point settings of each solve; None takes `IPMSettings()`.

    Raises
    ------
    ValueError
        If a count is below its minimum or not an integer, `beta` is below 1 or not finite,
        `penalty` is not positive and finite, or a choice is not one of its options.
    """

    regulator: LQRConfig = field(default_factory=LQRConfig)
    horizon: int = 20
    terminal: str = "cost"
    beta: float = 1.0
    t_max: int = 200
    row_samples: int = 5
    instep_bound: str = "first_order"
    penalty: float | None = None
    side: str = "auto"
    side_hysteresis: int = 3
    settings: IPMSettings | None = None

    def __post_init__(self):
        counts = (("horizon", 1), ("t_max", 0), ("row_samples", 2), ("side_hysteresis", 1))
        for name, least in counts:
            value = getattr(self, name)
            if not (isinstance(value, int) and value >= least):
                raise ValueError(f"{name} must be an integer of at least {least}")
        if self.terminal not in TERMINALS:
            raise ValueError(f"terminal must be one of {TERMINALS}")
        if not (math.isfinite(self.beta) and self.beta >= 1.0):
            raise ValueError("beta must be finite and at least 1")
        if self.instep_bound not in INSTEP_BOUNDS:
            raise ValueError(f"instep_bound must be one of {INSTEP_BOUNDS}")
        if self.penalty is not None and not (math.isfinite(self.penalty) and self.penalty > 0.0):
            raise ValueError("penalty must be positive and finite")
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}")


@dataclass(frozen=True)
class MPCPlan:
    """A horizon plan and how it was found.

    Attributes
    ----------
    x : numpy.ndarray, shape (N + 1, nz), or None
        Planned augmented states, x[0] the initial state; None unless converged.
    u : numpy.ndarray, shape (N, 3), or None
        Planned Hill accelerations (m/s^2); None unless converged.
    slack : numpy.ndarray, shape (N,), or None
        Keep-out slack of each stage (m); a positive entry is a predicted violation. None
        unless converged.
    status : str
        "converged", or the status of `solve_horizon`: "iteration_cap", "infeasible_certified"
        or "solver_failed".
    path : {"riccati", "ipm"}
        "riccati" when the plan without rows meets every row, else "ipm".
    iterations : int
        Interior-point iterations, 0 on the Riccati path.
    conflicting : tuple of (int, int)
        (stage, row) pairs of the infeasibility certificate, stage N for terminal rows; ()
        unless the status is "infeasible_certified".
    elapsed_s : float
        Wall-clock time of `mpc_solve`, both paths included (s).
    """

    x: np.ndarray | None
    u: np.ndarray | None
    slack: np.ndarray | None
    status: str
    path: str
    iterations: int
    conflicting: tuple[tuple[int, int], ...]
    elapsed_s: float


def mpc_problem(
    model: RegulatorModel,
    view: AgentView,
    z0: ArrayLike,
    reference: tuple[ArrayLike, ArrayLike],
    config: MPCConfig,
) -> StageQP:
    """Horizon QP of the hold-point MPC from the augmented state z0.

    The inputs are the three Hill accelerations and a keep-out slack s (m). Stage k costs
    1/2 (z_k - zs, u_k - us)^T H (z_k - zs, u_k - us) + rho s_k + rho_q s_k^2 / 2, with H the
    regulator's stage weight [[q, s], [s^T, r]], and the terminal cost is
    1/2 beta (x_N - zs)^T P (x_N - zs) with P the DARE solution: half the LQR cost, so with no
    binding row and beta = 1 the first move is the LQR law for every N. The rows of each stage
    are the thrust box (on a ball limit the box inscribed in it, max_accel / sqrt 3 per axis),
    the slack bound and `row_samples` keep-out rows n.r + s >= min(R + b, |r_ref|) at evenly
    spaced times within the step, n = r_ref / |r_ref| at the reference plan's samples and b the
    `instep_bound` tightening. A sample at the origin, and the first sample of the first step
    (the current state), get an inert row 0 >= -1. With terminal "set" the terminal rows are
    the maximal admissible set of the LQR loop.

    Parameters
    ----------
    model : RegulatorModel
        `regulator_model(view, config.regulator)`.
    view : AgentView
        Scenario description; uses `n`, `dt`, `max_accel`, `thrust_limit` and the largest
        "keep_out" radius in `constraints` (m, about the target). Other constraint types are
        not modelled.
    z0 : array_like, shape (nz,)
        Augmented state at the decision (m, m/s, and m/s^2 for pending or last commands).
    reference : (x_ref, u_ref)
        Plan the keep-out rows are linearized at: augmented states, shape (N + 1, nz), and Hill
        accelerations (m/s^2), shape (N, 3). The rows never cut the reference itself.
    config : MPCConfig
        Horizon, terminal ingredients, keep-out rows and penalty.

    Returns
    -------
    StageQP
        States z (nz) and inputs (u, sigma), sigma = s + 1 m the slack offset by a metre;
        7 + `row_samples` rows per stage (six box rows, the slack bound sigma >= 1 m, then the
        keep-out rows in time order); P = beta P_dare and p = -P zs; terminal rows only for
        terminal "set".

    Raises
    ------
    ValueError
        If the thrust limit is unknown; z0 or the reference has the wrong shape or is not
        finite; "row_samples too few for the in-step bound at this step" when the bound between
        samples is infinite at h = dt / (row_samples - 1); or, for terminal "set", a message
        starting "terminal set": no thrust margin at the hold point, no room for the hold-point
        halfspace outside the keep-out sphere, a set unbounded in position or velocity, or an
        error of `admissible_set`.
    """
    if view.thrust_limit not in ("box", "ball"):
        raise ValueError(f"unknown thrust limit {view.thrust_limit!r}")
    steps = config.horizon
    nz = len(model.zs)
    z0 = np.array(z0, dtype=float)
    x_ref, u_ref = (np.array(part, dtype=float) for part in reference)
    if z0.shape != (nz,):
        raise ValueError(f"z0 must have shape ({nz},)")
    if x_ref.shape != (steps + 1, nz) or u_ref.shape != (steps, 3):
        raise ValueError(f"reference must have shapes ({steps + 1}, {nz}) and ({steps}, 3)")
    if not all(np.all(np.isfinite(part)) for part in (z0, x_ref, u_ref)):
        raise ValueError("z0 and the reference must be finite")
    a_box = view.max_accel if view.thrust_limit == "box" else view.max_accel / math.sqrt(3.0)
    radii = [float(c["radius"]) for c in view.constraints if c["type"] == "keep_out"]
    radius = max(radii) if radii else None

    weight = np.block([[model.q, model.s], [model.s.T, model.r]])
    if config.penalty is None:
        rho = PENALTY_FACTOR * PENALTY_LENGTH_M * float(np.mean(np.diag(model.q)[:3]))
    else:
        rho = config.penalty
    # Without a keep-out sphere the slack enters no row, and any positive curvature serves.
    rho_q = SLACK_CURVATURE * rho / (1.0 if radius is None else radius)
    q_stage = np.zeros((steps, nz + 4, nz + 4))
    q_stage[:, : nz + 3, : nz + 3] = weight
    q_stage[:, -1, -1] = rho_q
    q_lin = np.zeros((steps, nz + 4))
    q_lin[:, : nz + 3] = -weight @ np.concatenate([model.zs, model.us])
    q_lin[:, -1] = rho - rho_q * SLACK_OFFSET_M
    b_stage = np.zeros((steps, nz, 4))
    b_stage[..., :3] = model.gamma

    c_rows = np.zeros((steps, BOX_AND_SLACK_ROWS + config.row_samples, nz + 4))
    e_rows = np.full(c_rows.shape[:2], -1.0)
    for i in range(3):
        c_rows[:, 2 * i, nz + i] = 1.0
        c_rows[:, 2 * i + 1, nz + i] = -1.0
        e_rows[:, 2 * i : 2 * i + 2] = -a_box
    c_rows[:, 6, -1] = 1.0
    e_rows[:, 6] = SLACK_OFFSET_M
    if radius is not None:
        rows, bounds = _keep_out_rows(model, view, x_ref, u_ref, config, radius, a_box)
        c_rows[:, BOX_AND_SLACK_ROWS:] = rows
        e_rows[:, BOX_AND_SLACK_ROWS:] = bounds

    c_n = e_n = None
    if config.terminal == "set":
        c_n, e_n = _cached_terminal_rows(model, view, config.t_max, radius, a_box)
    p_terminal = config.beta * model.p
    return StageQP(
        x0=z0,
        A=np.broadcast_to(model.phi, (steps, nz, nz)),
        B=b_stage,
        c=np.zeros((steps, nz)),
        Q=q_stage,
        q=q_lin,
        P=p_terminal,
        p=-p_terminal @ model.zs,
        C=c_rows,
        e=e_rows,
        C_N=c_n,
        e_N=e_n,
    )


def _keep_out_rows(
    model: RegulatorModel,
    view: AgentView,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    config: MPCConfig,
    radius: float,
    a_box: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep-out rows (N, m, nz + 4) and their bounds (N, m) at the samples of each step."""
    steps, nz = len(u_ref), x_ref.shape[1]
    m = config.row_samples
    h = view.dt / (m - 1)
    phis, gammas = cw_discrete(view.n, np.linspace(0.0, view.dt, m))
    # With pending slots the command acting during step k is the oldest slot of z_k, so the
    # command being issued does not move that step's samples.
    delayed = bool(model.slots)
    acting = x_ref[:steps, 6:9] if delayed else u_ref
    samples = np.einsum("jab,kb->kja", phis, x_ref[:steps, :6])
    samples += np.einsum("jab,kb->kja", gammas, acting)
    speed, accel = interval_bounds(samples, h, math.sqrt(3.0) * a_box, view.n)
    if not np.all(np.isfinite(speed)):
        raise ValueError("row_samples too few for the in-step bound at this step")
    # A sample takes the larger bound of the two intervals it ends.
    speed = np.maximum(
        np.pad(speed, ((0, 0), (1, 0)), "edge"), np.pad(speed, ((0, 0), (0, 1)), "edge")
    )
    accel = np.maximum(
        np.pad(accel, ((0, 0), (1, 0)), "edge"), np.pad(accel, ((0, 0), (0, 1)), "edge")
    )
    if config.instep_bound == "first_order":
        tightening = speed * h / 2.0
    else:
        tightening = (speed * h) ** 2 / (8.0 * radius) + accel * h**2 / 8.0

    r_ref = samples[..., :3]
    size = np.linalg.norm(r_ref, axis=-1)
    live = size > 0.0
    live[0, 0] = False
    normal = np.zeros_like(r_ref)
    normal[live] = r_ref[live] / size[live, None]
    rows = np.zeros((steps, m, nz + 4))
    rows[..., :6] = np.einsum("kja,jab->kjb", normal, phis[:, :3, :])
    acting_column = 6 if delayed else nz
    rows[..., acting_column : acting_column + 3] = np.einsum(
        "kja,jab->kjb", normal, gammas[:, :3, :]
    )
    rows[..., -1] = 1.0
    rows[~live] = 0.0
    bounds = np.where(live, np.minimum(radius + tightening, size) + SLACK_OFFSET_M, -1.0)
    return rows, bounds


def _cached_terminal_rows(
    model: RegulatorModel, view: AgentView, t_max: int, radius: float | None, a_box: float
) -> tuple[np.ndarray, np.ndarray]:
    """`_terminal_rows`, read-only, kept for the last few models and steps."""
    # A controller builds a problem from one model at every decision, and the set costs two
    # Gilbert-Tan runs and six LPs, 0.3 s at 1 s steps. Sharing is safe: the rows are a pure
    # function of the key (the bytes of every array they are computed from and the scalars),
    # so any two callers with equal inputs get bitwise equal rows; entries are read-only and
    # StageQP copies them; failures are not stored.
    key = (
        *(a.tobytes() for a in (model.phi, model.gamma, model.k, model.xs, model.us, model.zs)),
        len(model.zs),
        view.dt,
        view.n,
        t_max,
        radius,
        a_box,
    )
    if key not in _TERMINAL_CACHE:
        if len(_TERMINAL_CACHE) >= TERMINAL_CACHE_SIZE:
            _TERMINAL_CACHE.clear()
        c_n, e_n = _terminal_rows(model, view, t_max, radius, a_box)
        c_n.flags.writeable = False
        e_n.flags.writeable = False
        _TERMINAL_CACHE[key] = (c_n, e_n)
    return _TERMINAL_CACHE[key]


def _terminal_rows(
    model: RegulatorModel, view: AgentView, t_max: int, radius: float | None, a_box: float
) -> tuple[np.ndarray, np.ndarray]:
    """Terminal rows C_N z >= e_N of the maximal admissible set of the LQR loop.

    The output rows are the thrust box on u = us - K (z - zs), |K_i (z - zs)| <= a_box -
    |us_i|, and with a keep-out sphere the hold-point halfspace n_h.r >= R + b_h written about
    the hold point, |n_h.(r - r_hold)| <= |r_hold| - R - b_h. The rows hold at samples dt
    apart, and b_h = A dt^2 / 8 covers what a linear row can lose between them, A from
    `interval_bounds` at the set's largest range and speed with sqrt(3) a_box of thrust. The
    set depends on b_h, so b_h is taken from the set built with the allowance at the hold point
    at rest; that set contains the final one, whose own allowance is therefore no larger.
    """
    nz = len(model.zs)
    f = model.phi - model.gamma @ model.k
    thrust_room = a_box - np.abs(model.us)
    if not np.all(thrust_room > 0.0):
        raise ValueError("terminal set has no thrust margin at the hold point")
    if radius is None:
        rows, t_star = admissible_set(f, model.k, thrust_room, t_max=t_max)
        limits = np.tile(thrust_room, t_star + 1)
    else:
        distance = float(np.linalg.norm(model.xs[:3]))
        thrust = math.sqrt(3.0) * a_box
        hold_row = np.zeros(nz)
        if distance > 0.0:
            hold_row[:3] = model.xs[:3] / distance
        h = np.vstack([model.k, hold_row])

        def hold_set(allowance: float) -> tuple[np.ndarray, np.ndarray]:
            room = distance - radius - allowance
            if not (math.isfinite(room) and room > 0.0):
                raise ValueError(
                    "terminal set has no room between the keep-out sphere and the hold point"
                )
            bound = np.append(thrust_room, room)
            rows, t_star = admissible_set(f, h, bound, t_max=t_max)
            return rows, np.tile(bound, t_star + 1)

        rows, limits = hold_set(_hold_allowance(view, distance, 0.0, thrust))
        extent = _extents(rows, limits) * (1.0 + EXTENT_MARGIN)
        reach = distance + float(np.linalg.norm(extent[:3]))
        speed = float(np.linalg.norm(extent[3:]))
        rows, limits = hold_set(_hold_allowance(view, reach, speed, thrust))
    offset = rows @ model.zs
    return np.vstack([rows, -rows]), np.concatenate([offset - limits, -offset - limits])


def _hold_allowance(view: AgentView, range_m: float, speed_mps: float, thrust: float) -> float:
    """Range (m) a linear row can lose between samples dt apart, A dt^2 / 8; inf if unbounded."""
    state = np.array([range_m, 0.0, 0.0, speed_mps, 0.0, 0.0])
    _, accel = interval_bounds(np.stack([state, state]), view.dt, thrust, view.n)
    return float(accel[0]) * view.dt**2 / 8.0


def _extents(rows: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Largest |x_i| over {x : |rows x| <= limits} for the six Hill-state components."""
    # Scaled as in `_maximizers`: rows in units of their bound, unit column maxima, unit row
    # norms.
    a = rows / limits[:, None]
    column = np.max(np.abs(a), axis=0)
    column[column == 0.0] = 1.0
    a = a / column
    size = np.linalg.norm(a, axis=1)
    a, b = a / size[:, None], 1.0 / size
    a_ub, b_ub = np.vstack([a, -a]), np.concatenate([b, b])
    extent = np.empty(6)
    for i in range(6):
        cost = np.zeros(rows.shape[1])
        cost[i] = -1.0
        lp = linprog(
            cost,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=(None, None),
            method="highs-ds",
            options={"presolve": False},
        )
        if lp.status == 3:
            raise ValueError("terminal set is unbounded in position or velocity")
        if lp.status != 0:
            raise RuntimeError(f"terminal set LP failed: {lp.message}")
        extent[i] = -lp.fun / column[i]
    return extent


def mpc_solve(
    qp: StageQP,
    settings: IPMSettings | None = None,
    warm: tuple[np.ndarray, np.ndarray] | None = None,
) -> MPCPlan:
    """Solve a problem of `mpc_problem`, by the Riccati recursion when no row binds.

    The plan without rows and with zero slack comes first; if it meets every row up to the
    rounding of the row's terms it is the optimum of the full problem, since the slack's cost is
    lowest at zero and couples to nothing. Otherwise `solve_horizon` solves the full problem with
    the given arguments and its status passes through; only a converged solve gives a plan.

    Parameters
    ----------
    qp : StageQP
        A problem from `mpc_problem`: four inputs, the last the slack offset by 1 m.
    settings : IPMSettings, optional
        Interior-point settings; the defaults of `solve_horizon` when None.
    warm : (x, u), optional
        Interior-point start: states (N + 1, nz) and inputs (N, 4) with the offset slack.

    Returns
    -------
    MPCPlan
        The plan, its slack in metres, the status, the path and the time taken.

    Raises
    ------
    ValueError
        If the problem does not have four inputs, an input block is not positive definite, or
        `warm` has the wrong shapes.
    """
    started = time.perf_counter()
    steps, nz, nu = qp.B.shape
    if nu != 4:
        raise ValueError("the problem must carry the slack as a fourth input")
    free = dataclasses.replace(
        qp,
        B=qp.B[..., :3],
        Q=qp.Q[:, : nz + 3, : nz + 3],
        q=qp.q[:, : nz + 3],
        C=None,
        e=None,
        C_N=None,
        e_N=None,
    )
    x, u, _ = riccati_solve(free)
    if _rows_hold(qp, x, u):
        elapsed = time.perf_counter() - started
        return MPCPlan(x, u, np.zeros(steps), "converged", "riccati", 0, (), elapsed)
    sol = solve_horizon(qp, settings=settings, warm=warm)
    elapsed = time.perf_counter() - started
    if sol.status != "converged" or sol.x is None or sol.u is None:
        return MPCPlan(
            None, None, None, sol.status, "ipm", sol.iterations, sol.conflicting, elapsed
        )
    slack = sol.u[:, 3] - SLACK_OFFSET_M
    return MPCPlan(sol.x, sol.u[:, :3], slack, sol.status, "ipm", sol.iterations, (), elapsed)


def _rows_hold(qp: StageQP, x: np.ndarray, u: np.ndarray) -> bool:
    """Whether the plan (x, u) at zero slack meets every row up to the rounding of its terms."""
    z = np.concatenate([x[:-1], u, np.full((len(u), 1), SLACK_OFFSET_M)], axis=1)
    if qp.C is not None and qp.e is not None:
        value = np.einsum("kij,kj->ki", qp.C, z) - qp.e
        size = np.einsum("kij,kj->ki", np.abs(qp.C), np.abs(z)) + np.abs(qp.e)
        if np.any(value < -ROW_ROUNDING * size):
            return False
    if qp.C_N is not None and qp.e_N is not None:
        value_n = qp.C_N @ x[-1] - qp.e_N
        size_n = np.abs(qp.C_N) @ np.abs(x[-1]) + np.abs(qp.e_N)
        if np.any(value_n < -ROW_ROUNDING * size_n):
            return False
    return True


class MPCAgent:
    """Hold-point MPC on the CW model with keep-out rows, a passing side and fallbacks.

    Each decision builds the horizon problem of `mpc_problem` at the observed state and solves
    it with `mpc_solve` (Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed., sections
    2.5 and 8.8). The keep-out rows are linearized at the previous plan shifted by one step, with
    the LQR move at its last state clipped to the box appended, and flown from the observed
    state; an interior-point solve starts from that plan with 1 mm of slack. With no plan the
    reference is the clipped LQR rollout or, when the straight segment to the hold point passes
    within 1.2 R of the target, a rollout through a waypoint 1.2 R out on the chosen side. In
    every reference, states nearer than 1.1 R to the target are moved radially out to 1.1 R: a
    row asks only for its reference sample's range, so a reference inside the sphere would keep
    later plans there. Pending commands and the rate weight augment the state as for `LQRAgent`,
    and where no row binds the command is the `LQRAgent` command of the same regulator.

    Passing side: with `side="auto"`, a contested decision with no side yet, or after
    `side_hysteresis` decisions in a row at level 1 or 2, solves the problems of both waypoints
    (+-x across the segment, or +-z when the radial axis is within about 6 degrees of it) from
    the solver's own start and commits to the lower converged objective for the episode. A
    fixed side skips the comparison.

    Fallback levels: 0, the plan solved now; 1, the shifted plan, when the solver gives no plan
    and one exists; 2, the LQR law clipped per axis (box) or scaled onto the ball, when none
    does. An unconverged iterate is never applied. The applied step is certified against the
    keep-out sphere by `segment_minimum` over 20 samples of the held step on the CW model; a
    level-0 plan with a negative bound is solved once more with its first step's keep-out rows
    raised by the shortfall plus the keep-out resolution, and applied either way. Commands are
    issued in the view's action mode.

    `log` holds one entry per decision: `time_s`; `status`, `path` and `conflicting` of the
    solve applied (of the first side's when no side converged); `iterations` of every solve of
    the decision and `elapsed_s`, its wall-clock time (s); `solver` and `tolerance`; `level` and
    `fallback`; `warm_start`, whether an interior-point solve started from the shifted plan;
    `resolved`, whether the certificate asked for a second solve; `certified_bound`, the
    certified lower bound on the keep-out margin over the applied step (m); `certified`, that
    bound nonnegative and no slack above the keep-out resolution; `predicted_margin`, the least
    keep-out margin of the plan at its row samples (m), None at level 2; `max_slack`, the
    largest slack of a plan solved now (m), None at levels 1 and 2; `side`, the committed side or
    None; `in_xf`, whether the plan ends in the terminal set, None at level 2 or when the set
    could not be built. Without a keep-out constraint the keep-out entries are None and
    `certified` is True. `fallbacks` counts decisions by level and `plan` is the (states,
    commands) plan followed, None after level 2. They are the agent's own bookkeeping, which the
    recorder and metrics never read as scores.

    Parameters
    ----------
    config : MPCConfig, optional
        Horizon problem, passing side and solver settings; the defaults of `MPCConfig` when
        omitted.
    name : str
        Name in results.
    """

    def __init__(self, config: MPCConfig | None = None, name: str = "mpc"):
        self.config = MPCConfig() if config is None else config
        self.name = name
        self.log: list[dict] = []
        self.fallbacks = [0, 0, 0]
        self.plan: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Build the regulator model and the terminal set, and check the horizon problem.

        Parameters
        ----------
        view : AgentView
            Scenario description; uses `n`, `dt`, `max_accel`, `thrust_limit`, `action_mode`,
            `mass_kg`, `goal["hold_point"]` (Hill frame, m), the keep-out constraints and the
            observation names.
        rng : numpy.random.Generator, optional
            Unused; the controller is deterministic.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, the action mode takes
            no continuous command, the thrust limit is unknown, the hold point needs more than
            `max_accel` of steady thrust, the bound between row samples is infinite
            ("row_samples too few"), or, with terminal "set", the terminal set cannot be built
            (see `mpc_problem`).

        Warns
        -----
        UserWarning
            With terminal "cost", when the terminal set cannot be built; `in_xf` is then None.
        """
        cfg = self.config
        hold_point(view, "MPC agent")
        self.scale = action_scale(view)
        model = regulator_model(view, cfg.regulator)
        steps = cfg.horizon
        # Errors of the in-step bound, and of a terminal row set, show here, not mid-episode.
        at_hold = (np.tile(model.zs, (steps + 1, 1)), np.tile(model.us, (steps, 1)))
        mpc_problem(model, view, model.zs, at_hold, cfg)
        a_box = view.max_accel if view.thrust_limit == "box" else view.max_accel / math.sqrt(3.0)
        radii = [float(c["radius"]) for c in view.constraints if c["type"] == "keep_out"]
        radius = max(radii) if radii else None
        self.terminal_rows: tuple[np.ndarray, np.ndarray] | None = None
        try:
            self.terminal_rows = _cached_terminal_rows(model, view, cfg.t_max, radius, a_box)
        except (ValueError, RuntimeError) as err:
            # With terminal "set" mpc_problem has raised the same error above.
            warnings.warn(f"{err}; terminal set membership is not logged", stacklevel=2)
        settings = IPMSettings() if cfg.settings is None else cfg.settings
        self.model: RegulatorModel = model
        self.view = view
        self.a_box = a_box
        self.radius = radius
        self.keep_out = None if radius is None else KeepOut(radius)
        self.cw = cw_discrete(view.n, view.dt)
        self.row_steps = cw_discrete(view.n, np.linspace(0.0, view.dt, cfg.row_samples))
        self.check_times = np.linspace(0.0, view.dt, VIOLATION_SAMPLES)
        self.check_steps = cw_discrete(view.n, self.check_times)
        self.tolerance = max(settings.tol_feas, settings.tol_gap)
        self.feasibility = settings.tol_feas
        self.side = None if cfg.side == "auto" else cfg.side
        self.failures = 0
        self.last = np.zeros(3)
        self.plan = None
        self.log = []
        self.fallbacks = [0, 0, 0]

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """First move of the horizon plan at the observed state, or a fallback.

        Parameters
        ----------
        obs : numpy.ndarray
            Observation in the order of the view's names: the Hill state (m, m/s) first, then
            the pending-command slots if enabled.
        info : dict
            Agent-safe information; only `time_s` (s since the episode start) is read.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Command in the view's action mode.

        Raises
        ------
        ValueError
            If a pending command's wait is not a whole number of steps.
        """
        started = time.perf_counter()
        cfg, steps = self.config, self.config.horizon
        t = float(info["time_s"])
        z0 = self._state(obs)
        shifted = None if self.plan is None else self._shifted(z0)
        contested = self._contested(z0[:3])
        choose = (
            contested
            and cfg.side == "auto"
            and (self.side is None or self.failures >= cfg.side_hysteresis)
        )
        if choose:
            # Both sides start cold: the shifted plan belongs to one of them at most.
            solves = []
            for label in self._side_labels(z0[:3]):
                reference = self._lifted(self._reference(z0, self._waypoint(label, z0[:3])))
                solves.append((label, *self._solve(z0, reference, None)))
            ranked = [
                (self._objective(qp, found), i)
                for i, (_, qp, found) in enumerate(solves)
                if found.x is not None
            ]
            label, qp, plan = solves[min(ranked)[1] if ranked else 0]
            if ranked:
                self.side = label
            warm_start = False
            iterations = sum(found.iterations for _, _, found in solves)
        else:
            warm = None
            if shifted is not None:
                reference = self._lifted(shifted)
                slack = np.full((steps, 1), SLACK_OFFSET_M + WARM_SLACK_M)
                warm = (shifted[0], np.concatenate([shifted[1], slack], axis=1))
            elif contested and self.side is not None:
                reference = self._lifted(self._reference(z0, self._waypoint(self.side, z0[:3])))
            else:
                reference = self._lifted(self._reference(z0, None))
            qp, plan = self._solve(z0, reference, warm)
            warm_start = warm is not None and plan.path == "ipm"
            iterations = plan.iterations

        x_plan: np.ndarray | None = None
        u_plan: np.ndarray | None = None
        if plan.x is not None and plan.u is not None:
            level = 0
            x_plan, u_plan = plan.x, plan.u
        elif shifted is not None:
            level = 1
            x_plan, u_plan = shifted
        else:
            level = 2
        u = self._first_move(z0, u_plan)
        bound = self._certificate(z0, u)
        resolved = False
        if level == 0 and bound is not None and bound < 0.0:
            resolved = True
            raised = self._raised(qp, -bound)
            iterations += raised.iterations
            if raised.x is not None and raised.u is not None:
                plan = raised
                x_plan, u_plan = raised.x, raised.u
                u = self._first_move(z0, u_plan)
                bound = self._certificate(z0, u)

        max_slack = None
        if level == 0 and plan.slack is not None:
            max_slack = max(0.0, float(np.max(plan.slack)))
        margin = in_xf = None
        if x_plan is not None and u_plan is not None:
            if self.radius is not None:
                margin = self._predicted_margin(x_plan, u_plan)
            if self.terminal_rows is not None:
                c_n, e_n = self.terminal_rows
                end = x_plan[-1]
                size = np.abs(c_n) @ np.abs(end) + np.abs(e_n)
                in_xf = bool(np.all(c_n @ end - e_n >= -self.feasibility * size))
        certified = (bound is None or bound >= 0.0) and (
            max_slack is None or max_slack <= KeepOut.resolution
        )

        self.plan = None if x_plan is None or u_plan is None else (x_plan, u_plan)
        self.failures = 0 if level == 0 else self.failures + 1
        self.last = u
        self.fallbacks[level] += 1
        self.log.append(
            {
                "time_s": t,
                "status": plan.status,
                "iterations": iterations,
                "elapsed_s": time.perf_counter() - started,
                "solver": "mpc_solve",
                "tolerance": self.tolerance,
                "level": level,
                "fallback": level > 0,
                "path": plan.path,
                "warm_start": warm_start,
                "resolved": resolved,
                "certified": certified,
                "predicted_margin": margin,
                "certified_bound": bound,
                "max_slack": max_slack,
                "side": self.side,
                "in_xf": in_xf,
                "conflicting": plan.conflicting,
            }
        )
        return u / self.view.max_accel * self.scale

    def _state(self, obs: np.ndarray) -> np.ndarray:
        """Augmented state of the regulator model from the observation, as `LQRAgent` builds it."""
        view, model = self.view, self.model
        obs = np.asarray(obs, dtype=float)
        parts = [obs[:6]]
        if model.slots:
            pending = obs[model.slots]
            steps = pending[:, 0] / view.dt
            if np.any(np.abs(steps - np.round(steps)) > ALIGNMENT_TOL):
                raise ValueError("pending commands are not aligned to the decision step")
            # Commands are echoed in the action units of their issue, which are this agent's.
            parts.append((pending[:, 1:] * (view.max_accel / self.scale)).ravel())
        elif model.own_last:
            parts.append(self.last)
        return np.concatenate(parts)

    def _solve(
        self,
        z0: np.ndarray,
        reference: tuple[np.ndarray, np.ndarray],
        warm: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[StageQP, MPCPlan]:
        """The horizon problem at z0 linearized at `reference`, and its plan."""
        qp = mpc_problem(self.model, self.view, z0, reference, self.config)
        return qp, mpc_solve(qp, settings=self.config.settings, warm=warm)

    def _raised(self, qp: StageQP, shortfall: float) -> MPCPlan:
        """Solve of `qp` with the first step's keep-out rows raised by `shortfall` (m)."""
        assert qp.e is not None
        e = qp.e.copy()
        rows = e[0, BOX_AND_SLACK_ROWS:]
        # Inert rows have a negative bound and stay inert.
        rows[rows > 0.0] += shortfall + KeepOut.resolution
        return mpc_solve(dataclasses.replace(qp, e=e), settings=self.config.settings)

    def _shifted(self, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The plan shifted by one step with the clipped LQR move at its end, flown from z0."""
        assert self.plan is not None
        model = self.model
        x_prev, u_prev = self.plan
        end = np.clip(model.us - model.k @ (x_prev[-1] - model.zs), -self.a_box, self.a_box)
        u = np.vstack([u_prev[1:], end])
        x = np.empty_like(x_prev)
        x[0] = z0
        for k, u_k in enumerate(u):
            x[k + 1] = model.phi @ x[k] + model.gamma @ u_k
        return x, u

    def _reference(
        self, z0: np.ndarray, waypoint: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clipped LQR rollout from z0 to the hold point, via the waypoint (m) if one is given."""
        model, steps = self.model, self.config.horizon
        targets = [(model.zs, model.us)]
        if waypoint is not None:
            xw = np.concatenate([waypoint, np.zeros(3)])
            uw = steady_state_input(*self.cw, xw)
            targets.insert(0, (np.concatenate([xw, np.tile(uw, (len(model.zs) - 6) // 3)]), uw))
        x = np.empty((steps + 1, len(z0)))
        u = np.empty((steps, 3))
        x[0] = z0
        for k in range(steps):
            if waypoint is not None and len(targets) > 1:
                assert self.radius is not None
                if np.linalg.norm(x[k, :3] - waypoint) < WAYPOINT_REACHED * self.radius:
                    targets.pop(0)
            z_target, u_target = targets[0]
            u[k] = np.clip(u_target - model.k @ (x[k] - z_target), -self.a_box, self.a_box)
            x[k + 1] = model.phi @ x[k] + model.gamma @ u[k]
        return x, u

    def _lifted(self, reference: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """The reference with its later states nearer than 1.1 R moved radially out to 1.1 R."""
        if self.radius is None:
            return reference
        x, u = reference
        x = x.copy()
        size = np.linalg.norm(x[:, :3], axis=1)
        near = (size < REFERENCE_RADIUS * self.radius) & (size > 0.0)
        near[0] = False
        x[near, :3] *= (REFERENCE_RADIUS * self.radius / size[near])[:, None]
        return x, u

    def _contested(self, r: np.ndarray) -> bool:
        """Whether the segment from r to the hold point passes within 1.2 R of the target."""
        if self.radius is None:
            return False
        d = self.model.xs[:3] - r
        length = float(d @ d)
        if length == 0.0:
            return False
        s = min(max(-float(r @ d) / length, 0.0), 1.0)
        return float(np.linalg.norm(r + s * d)) < WAYPOINT_RADIUS * self.radius

    def _across(self, axis: int, r: np.ndarray) -> np.ndarray:
        """Component of a Hill axis orthogonal to the segment from r to the hold point."""
        d = self.model.xs[:3] - r
        d = d / np.linalg.norm(d)
        unit = np.eye(3)[axis]
        return unit - (unit @ d) * d

    def _side_labels(self, r: np.ndarray) -> tuple[str, str]:
        """The two sides compared from r: across the segment along x, or along z."""
        if np.linalg.norm(self._across(0, r)) >= PARALLEL_TOL:
            return ("+x", "-x")
        return ("+z", "-z")

    def _waypoint(self, label: str, r: np.ndarray) -> np.ndarray | None:
        """Passing waypoint (m) of a side; None when the side's axis lies along the segment."""
        assert self.radius is not None
        across = self._across(0 if label[1] == "x" else 2, r)
        size = float(np.linalg.norm(across))
        if size < PARALLEL_TOL:
            return None
        sign = 1.0 if label[0] == "+" else -1.0
        return sign * WAYPOINT_RADIUS * self.radius * across / size

    @staticmethod
    def _objective(qp: StageQP, plan: MPCPlan) -> float:
        """Objective of `qp` at a converged plan."""
        assert plan.x is not None and plan.u is not None and plan.slack is not None
        sigma = plan.slack + SLACK_OFFSET_M
        z = np.concatenate([plan.x[:-1], plan.u, sigma[:, None]], axis=1)
        stage = 0.5 * np.einsum("ki,kij,kj->", z, qp.Q, z) + np.einsum("ki,ki->", qp.q, z)
        end = plan.x[-1]
        return float(stage + 0.5 * end @ qp.P @ end + qp.p @ end)

    def _first_move(self, z0: np.ndarray, u_plan: np.ndarray | None) -> np.ndarray:
        """Command (m/s^2): the plan's first move, or the LQR law at z0 without a plan.

        The plan's move is clipped to the box, which moves an interior-point solution by at most
        the solver's feasibility tolerance. The LQR law is clipped per axis (box) or scaled onto
        the ball, as `LQRAgent` does.
        """
        if u_plan is not None:
            return np.clip(u_plan[0], -self.a_box, self.a_box)
        model, limit = self.model, self.view.max_accel
        u = model.us - model.k @ (z0 - model.zs)
        if self.view.thrust_limit == "box":
            return np.clip(u, -limit, limit)
        size = float(np.linalg.norm(u))
        return u * (limit / size) if size > limit else u

    def _certificate(self, z0: np.ndarray, u: np.ndarray) -> float | None:
        """Certified lower bound (m) on the keep-out margin over the held step on the CW model."""
        if self.keep_out is None:
            return None
        n = self.view.n
        x = z0[:6]
        acting = z0[6:9] if self.model.slots else u
        phis, gammas = self.check_steps
        states = phis @ x + gammas @ acting

        def state_at(tau: float) -> np.ndarray:
            phi, gamma = cw_discrete(n, tau)
            return phi @ x + gamma @ acting

        found = segment_minimum(
            self.keep_out, state_at, self.check_times, states, float(np.linalg.norm(acting)), n
        )
        return float(found.lower_bound)

    def _predicted_margin(self, x: np.ndarray, u: np.ndarray) -> float:
        """Least keep-out margin (m) of a plan at the row samples of its steps."""
        assert self.radius is not None
        acting = x[:-1, 6:9] if self.model.slots else u
        phis, gammas = self.row_steps
        r = np.einsum("jab,kb->kja", phis[:, :3, :6], x[:-1, :6])
        r += np.einsum("jab,kb->kja", gammas[:, :3, :], acting)
        return float(np.min(np.linalg.norm(r, axis=-1))) - self.radius
