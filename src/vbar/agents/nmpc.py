"""Nonlinear MPC by direct multiple shooting and real-time iteration on a declared model.

The horizon is cut into one segment per step with a node state at each segment start (direct
multiple shooting, Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed., sections 8.5.2
and 8.6.3). Each decision linearizes the declared relative-motion model along the nodes and
controls of the previous plan shifted by one step, with the node Jacobians from the model's
variational equations, and solves one horizon QP: one Gauss-Newton step per decision, the
real-time iteration of section 8.9.2, eq. 8.60. Stage cost, terminal cost, thrust box and
keep-out rows are those of `vbar.agents.mpc`. Hill frame, SI units.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from ..conventions import hill_to_inertial
from ..dynamics import cw_discrete, get_body, mean_motion
from ..dynamics.models import TwoBodyDynamics
from ..solvers import IPMSettings, solve_horizon
from .base import AgentView, action_scale, hold_point
from .lqr import RegulatorModel, regulator_model
from .lyapunov import DeclaredModel
from .mpc import SLACK_OFFSET_M, MPCConfig, mpc_problem, mpc_solve

DECLARED_DYNAMICS = ("cw", "twobody", "twobody_j2")
# A decision time within this fraction of a step of the one the shifted plan was made for counts
# as that time; environments accumulate their clock and can differ from k dt by rounding.
SCHEDULE_TOL = 1e-9
# Statuses after which the same problem is solved again from the solver's own start. A warm
# start with an input on its bound can give "solver_failed" where a cold start converges; a
# certified infeasibility is a property of the problem, not of the start.
COLD_RETRY = ("iteration_cap", "solver_failed")


class _ShootingModel(TwoBodyDynamics):
    """Two-body model whose chaser can be put at a Hill state beside a given target state."""

    def place(self, target: np.ndarray, rel_state: np.ndarray) -> None:
        """Set the target's inertial [r, v] (m, m/s) and the chaser's Hill state (m, m/s)."""
        # TwoBodyDynamics has no public way to place the chaser beside a given target state:
        # `reset` always restarts the target at the episode start. This sets the private `_y`
        # (target and inertial offset) and clears `_sol` and `_step`, as `reset` does.
        y = np.concatenate([target, np.zeros(6)])
        dr, dv = hill_to_inertial(target[:3], target[3:], rel_state, self._target_acceleration(y))
        self._y = np.concatenate([target, dr, dv])
        # As after `reset`: the next segment starts from the integrator's own step guess.
        self._sol = None
        self._step = None


def _check_dynamics(model: DeclaredModel) -> None:
    if model.dynamics not in DECLARED_DYNAMICS:
        raise ValueError(f"unknown dynamics {model.dynamics!r}, expected cw, twobody or twobody_j2")


def shooting_linearization(
    model: DeclaredModel,
    t0: float,
    nodes: ArrayLike,
    controls: ArrayLike,
    dt: float,
    *,
    argument_of_latitude: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearization of the declared model along a multiple-shooting plan.

    Segment k starts at node s_k at time t0 + k dt and holds the Hill acceleration q_k for dt.
    With F_k the state the declared model reaches at its end, A_k = dF_k/ds_k and
    B_k = dF_k/dq_k, the linearized segment is s_(k+1) = A_k s_k + B_k q_k + c_k with
    c_k = F_k - A_k s_k - B_k q_k, exact at the nodes. On "cw" A_k and B_k are the exact
    zero-order-hold step and c_k is zero. On "twobody" and "twobody_j2" the target follows the
    declared circular orbit from the episode start, where its argument of latitude is
    `argument_of_latitude` (mean, with J2), and is propagated alone to each node time; the
    chaser is placed at s_k beside it and A_k, B_k come from the variational equations of the
    same model (`TwoBodyDynamics.transition_matrices`), F_k from `advance`. No finite
    differences of the integrator are taken.

    Parameters
    ----------
    model : DeclaredModel
        Declared relative-motion model: dynamics, altitude (m), inclination (rad) and body.
    t0 : float
        Time of the first node since the episode start (s), at least 0.
    nodes : array_like, shape (N + 1, 6)
        Node Hill states s_0, ..., s_N (m, m/s). The last node enters no segment.
    controls : array_like, shape (N, 3)
        Hill accelerations q_k held over each segment (m/s^2), N at least 1.
    dt : float
        Segment length (s), positive.
    argument_of_latitude : float, optional
        Target's argument of latitude at the episode start (rad); ignored on "cw".

    Returns
    -------
    A : numpy.ndarray, shape (N, 6, 6)
        Segment transition matrices on the Hill state.
    B : numpy.ndarray, shape (N, 6, 3)
        Segment input matrices (m and m/s per m/s^2).
    c : numpy.ndarray, shape (N, 6)
        Affine terms (m, m/s).

    Raises
    ------
    ValueError
        If the shapes disagree, an input is not finite, `t0` is negative, `dt` is not positive,
        or the declared dynamics is unknown.
    RuntimeError
        If an integration of the declared model fails.
    """
    nodes = np.array(nodes, dtype=float)
    controls = np.array(controls, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != 3 or len(controls) < 1:
        raise ValueError("controls must have shape (N, 3) with N at least 1")
    steps = len(controls)
    if nodes.shape != (steps + 1, 6):
        raise ValueError(f"nodes must have shape ({steps + 1}, 6)")
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(controls))):
        raise ValueError("nodes and controls must be finite")
    if not (math.isfinite(t0) and t0 >= 0.0):
        raise ValueError("t0 must be finite and not negative")
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be positive and finite")
    if not math.isfinite(argument_of_latitude):
        raise ValueError("argument_of_latitude must be finite")
    _check_dynamics(model)
    body = get_body(model.body)
    radius = body.radius + model.altitude_m
    if model.dynamics == "cw":
        phi, gamma = cw_discrete(mean_motion(radius, body.mu), dt)
        return (
            np.repeat(phi[None], steps, axis=0),
            np.repeat(gamma[None], steps, axis=0),
            np.zeros((steps, 6)),
        )

    j2 = model.dynamics == "twobody_j2"
    elements = [radius, 0.0, model.inclination, 0.0, 0.0, argument_of_latitude]
    # The target coasts on its own model, so the node times do not depend on the plan.
    clock = _ShootingModel(body, j2=j2, elements=elements)
    clock.reset(np.zeros(6))
    if t0 > 0.0:
        clock.advance(np.zeros(3), t0)
    segment = _ShootingModel(body, j2=j2, elements=elements)
    a = np.empty((steps, 6, 6))
    b = np.empty((steps, 6, 3))
    c = np.empty((steps, 6))
    for k in range(steps):
        if k > 0:
            clock.advance(np.zeros(3), dt)
        segment.place(clock.target_state(), nodes[k])
        a[k], b[k] = segment.transition_matrices(controls[k], dt)
        end, _ = segment.advance(controls[k], dt)
        c[k] = end - a[k] @ nodes[k] - b[k] @ controls[k]
    return a, b, c


@dataclass(frozen=True, kw_only=True)
class NMPCConfig:
    """Horizon problem and declared model of `NMPCAgent`.

    Parameters
    ----------
    mpc : MPCConfig
        Weights, horizon, terminal ingredients, keep-out rows, penalty and solver settings, as
        for the linear MPC; `side` and `side_hysteresis` are not read.
    model : DeclaredModel
        Relative-motion model the plan is shot on, from the public scenario fields.
    argument_of_latitude : float
        Target's argument of latitude at the episode start (rad), a declared assumption: the
        scenario does not publish it. Circular two-body motion does not depend on it; with J2 the
        relative motion does.

    Raises
    ------
    ValueError
        If `argument_of_latitude` is not finite.
    """

    mpc: MPCConfig = field(default_factory=MPCConfig)
    model: DeclaredModel = field(default_factory=DeclaredModel)
    argument_of_latitude: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.argument_of_latitude):
            raise ValueError("argument_of_latitude must be finite")


class NMPCAgent:
    """Hold-point MPC on a declared nonlinear model by the real-time iteration.

    Each decision has two phases (Rawlings, Mayne and Diehl, section 8.9.2). Preparation:
    `shooting_linearization` of the declared model along the previous plan shifted by one step
    (its last move the LQR law on CW, clipped to the box), and the horizon QP of `mpc_problem`
    with its keep-out rows linearized at the same plan and the CW dynamics replaced by the
    linearized segments; none of this reads the new observation. Feedback: the QP from the
    observed state by `solve_horizon`, warm-started at the shifted plan and solved again cold
    when that start gives "iteration_cap" or "solver_failed". At the first decision, and after
    a decision with no plan, the linearization point is the plan of the linear MPC on CW from
    the observed state, or the clipped LQR rollout when that plan fails.

    A converged QP gives level 0 and its first move. Otherwise the shifted plan's first move is
    applied (level 1) when a plan exists, and the LQR law clipped to the thrust limit (level 2)
    when none does. The controls applied or planned are rolled out from the observed state
    through the linearized segments; the second state of that rollout is the logged prediction
    and the first node of the next plan. Commands are issued in the view's action mode.

    `log` holds one entry per decision: `time_s`; `status` of the solve applied and
    `warm_status` of the warm-started one; `cold_retry`, `warm_start`, `iterations` (both
    solves), `solver` and `tolerance`; `level` and `fallback`; `reference` ("shifted",
    "linear_mpc" or "lqr"); `max_slack_m`, the largest keep-out slack of a converged plan (a
    positive value is a predicted violation); `continuity`, max_k |F_k - s_(k+1)| of the plan
    linearized (m and m/s); `predicted`, the predicted next Hill state; `declared_phase`, the
    declared argument of latitude on "twobody_j2" and None otherwise; `prep_s` and `feedback_s`
    (s). `fallbacks` counts decisions by level. Both are the agent's own bookkeeping, which the
    recorder and metrics never read as scores.

    Parameters
    ----------
    config : NMPCConfig, optional
        Horizon problem and declared model; the defaults of `NMPCConfig` when omitted.
    name : str
        Name in results.
    """

    def __init__(self, config: NMPCConfig | None = None, name: str = "nmpc"):
        self.config = NMPCConfig() if config is None else config
        self.name = name
        self.log: list[dict] = []
        self.fallbacks = [0, 0, 0]

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Build the regulator model and check the horizon problem for the view.

        Parameters
        ----------
        view : AgentView
            Scenario description; uses `n`, `dt`, `max_accel`, `thrust_limit`, `action_mode`,
            `mass_kg`, `goal["hold_point"]` (Hill frame, m), the keep-out constraints and the
            observation names. The orbit of the shooting model comes from the declared model.
        rng : numpy.random.Generator, optional
            Unused; the controller is deterministic.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, the action mode takes
            no continuous command, the declared dynamics is unknown, the thrust layout is
            unknown, the hold point needs more than `max_accel` of steady thrust, the regulator
            has a rate weight, the observation carries pending commands, or the horizon problem
            cannot be built (see `mpc_problem`).
        """
        cfg = self.config
        hold_point(view, "NMPC agent")
        self.scale = action_scale(view)
        _check_dynamics(cfg.model)
        if cfg.mpc.regulator.rate_weight > 0.0:
            raise ValueError("the NMPC agent takes no rate weight")
        model = regulator_model(view, cfg.mpc.regulator)
        if model.slots:
            raise ValueError("the NMPC agent does not model pending commands")
        steps = cfg.mpc.horizon
        # Errors of the terminal set and of the in-step bound show here rather than mid-episode.
        mpc_problem(
            model,
            view,
            model.zs,
            (np.tile(model.zs, (steps + 1, 1)), np.tile(model.us, (steps, 1))),
            cfg.mpc,
        )
        self.model: RegulatorModel = model
        self.view = view
        self.a_box = view.max_accel if view.thrust_limit == "box" else view.max_accel / math.sqrt(3)
        settings = IPMSettings() if cfg.mpc.settings is None else cfg.mpc.settings
        self.tolerance = max(settings.tol_feas, settings.tol_gap)
        self.plan: tuple[np.ndarray, np.ndarray] | None = None
        self.plan_time = math.nan
        self.log = []
        self.fallbacks = [0, 0, 0]

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Command of one real-time iteration at the observed state.

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

        Raises
        ------
        RuntimeError
            If an integration of the declared model fails.
        """
        cfg, view, model = self.config, self.view, self.model
        t = float(info["time_s"])
        x = np.asarray(obs, dtype=float)[:6].copy()
        steps = cfg.mpc.horizon

        cold_s = 0.0
        on_schedule = abs(t - self.plan_time) <= SCHEDULE_TOL * view.dt
        if self.plan is not None and on_schedule:
            nodes, controls = self.plan
            reference = "shifted"
        else:
            started = time.perf_counter()
            nodes, controls, reference = self._cold_reference(x)
            cold_s = time.perf_counter() - started
        has_plan = reference == "shifted"

        started = time.perf_counter()
        a, b, c = shooting_linearization(
            cfg.model, t, nodes, controls, view.dt, argument_of_latitude=cfg.argument_of_latitude
        )
        ends = np.einsum("kij,kj->ki", a, nodes[:-1]) + np.einsum("kij,kj->ki", b, controls) + c
        continuity = float(np.max(np.abs(ends - nodes[1:])))
        b_slack = np.zeros((steps, 6, 4))
        b_slack[..., :3] = b
        qp = dataclasses.replace(
            mpc_problem(model, view, nodes[0], (nodes, controls), cfg.mpc), A=a, B=b_slack, c=c
        )
        prep_s = time.perf_counter() - started

        started = time.perf_counter()
        qp = dataclasses.replace(qp, x0=x)
        warm_u = np.concatenate([controls, np.full((steps, 1), SLACK_OFFSET_M)], axis=1)
        solution = solve_horizon(qp, settings=cfg.mpc.settings, warm=(nodes, warm_u))
        warm_status, iterations = solution.status, solution.iterations
        cold_retry = warm_status in COLD_RETRY
        if cold_retry:
            solution = solve_horizon(qp, settings=cfg.mpc.settings)
            iterations += solution.iterations

        max_slack = None
        if solution.status == "converged" and solution.u is not None:
            level = 0
            planned = np.clip(solution.u[:, :3], -self.a_box, self.a_box)
            max_slack = float(np.max(solution.u[:, 3]) - SLACK_OFFSET_M)
        elif has_plan:
            level = 1
            planned = controls
        else:
            level = 2
            planned = None

        if planned is not None:
            states = _rollout(x, planned, a, b, c)
            u = planned[0]
            last = self._terminal_move(states[-1])
            self.plan = (
                np.vstack([states[1:], model.phi @ states[-1] + model.gamma @ last]),
                np.vstack([planned[1:], last]),
            )
            self.plan_time = t + view.dt
            predicted = states[1].copy()
        else:
            u = self._clipped_lqr(x)
            predicted = a[0] @ x + b[0] @ u + c[0]
            self.plan = None
            self.plan_time = math.nan
        feedback_s = time.perf_counter() - started + cold_s

        self.fallbacks[level] += 1
        self.log.append(
            {
                "time_s": t,
                "status": solution.status,
                "warm_status": warm_status,
                "cold_retry": cold_retry,
                "warm_start": not cold_retry,
                "iterations": iterations,
                "solver": "rti",
                "tolerance": self.tolerance,
                "level": level,
                "fallback": level > 0,
                "reference": reference,
                "max_slack_m": max_slack,
                "continuity": continuity,
                "predicted": predicted,
                "declared_phase": (
                    cfg.argument_of_latitude if cfg.model.dynamics == "twobody_j2" else None
                ),
                "prep_s": prep_s,
                "feedback_s": feedback_s,
            }
        )
        return u / view.max_accel * self.scale

    def _cold_reference(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Linearization point without a plan: the linear MPC plan, else the LQR rollout."""
        cfg, model = self.config, self.model
        steps = cfg.mpc.horizon
        at_rest = (np.tile(x, (steps + 1, 1)), np.zeros((steps, 3)))
        linear = mpc_solve(mpc_problem(model, self.view, x, at_rest, cfg.mpc), cfg.mpc.settings)
        if linear.x is not None and linear.u is not None:
            return linear.x, np.clip(linear.u, -self.a_box, self.a_box), "linear_mpc"
        nodes = np.empty((steps + 1, 6))
        controls = np.empty((steps, 3))
        nodes[0] = x
        for k in range(steps):
            controls[k] = self._terminal_move(nodes[k])
            nodes[k + 1] = model.phi @ nodes[k] + model.gamma @ controls[k]
        return nodes, controls, "lqr"

    def _terminal_move(self, x: np.ndarray) -> np.ndarray:
        """LQR move (m/s^2) at `x` clipped to the QP's box, which appends a step to a plan."""
        model = self.model
        return np.clip(model.us - model.k @ (x - model.zs), -self.a_box, self.a_box)

    def _clipped_lqr(self, x: np.ndarray) -> np.ndarray:
        """LQR move (m/s^2) at `x` clipped per axis (box) or scaled onto the ball."""
        model, limit = self.model, self.view.max_accel
        u = model.us - model.k @ (x - model.zs)
        if self.view.thrust_limit == "box":
            return np.clip(u, -limit, limit)
        size = float(np.linalg.norm(u))
        return u * (limit / size) if size > limit else u


def _rollout(
    x: np.ndarray, controls: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """States (N + 1, 6) of `controls` from `x` through the linearized segments."""
    states = np.empty((len(controls) + 1, 6))
    states[0] = x
    for k, q in enumerate(controls):
        states[k + 1] = a[k] @ states[k] + b[k] @ q + c[k]
    return states
