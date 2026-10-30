"""The hold-point MPC problem on the zero-order-hold CW model and its known answers.

`mpc_problem` builds the horizon QP of the MPC baseline as a `StageQP`: the stage cost of the LQR
regulator on the deviation from the hold point (zs, us), the DARE terminal cost, the thrust box,
keep-out rows at sub-step samples linearized at a reference plan and softened by a slack input,
and with terminal "set" the maximal admissible set of the LQR loop (Rawlings, Mayne and Diehl,
Model Predictive Control, 2nd ed., sections 2.5 and 8.8; Boyd and Vandenberghe, Convex
Optimization, section 11.7). `mpc_solve` takes the Riccati solution when no row binds and the
interior-point solution otherwise. References marked "independent computation" come from a
bounded-variable least-squares solve of the condensed problem, from Van Loan's sampled cost, from
a dense scan of the CW coast, or from a separate stage builder, not from the module under test.
Every case is for a 500 km circular orbit of the WGS 84 Earth and a 0.02 m/s^2 limit per axis;
the step is 10 s unless stated. "Discrete weights" are 10 m, 0.01 m/s and 0.02 m/s^2 costing 1,
summed once per step: Q = diag(1e-2 x3, 1e4 x3), R = 2500 I.
"""

import dataclasses

import numpy as np
import pytest
from scipy.optimize import linprog, lsq_linear

from vbar.agents import mpc
from vbar.agents.base import AgentView
from vbar.agents.lqr import LQRAgent, LQRConfig, regulator_model, steady_state_input
from vbar.constraints import interval_bounds
from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.scenarios import make
from vbar.solvers import HorizonSolution, IPMSettings, riccati_solve, solve_horizon

MPCConfig = mpc.MPCConfig
mpc_problem = mpc.mpc_problem
mpc_solve = mpc.mpc_solve

N = mean_motion(EARTH.radius + 500e3)
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
VBAR_HOLD = (0.0, -100.0, 0.0)
STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")
DISCRETE = LQRConfig(pos_scale_m=10.0, vel_scale_mps=0.01, stage_cost="discrete")
TIGHT = IPMSettings(tol_feas=1e-10, tol_gap=1e-10)
ALONG = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
# Largest along-track offset at rest from the hold point inside the maximal admissible set of the
# discrete-weight LQR loop under the box, and under the box inscribed in the ball (a_max / sqrt 3);
# independent computation.
REACH_BOX = 201.509568598
REACH_BALL = 116.341603675


def hand_view(dt=DT, hold=VBAR_HOLD, radius=None, limit="box", max_accel=A_MAX, slots=0):
    """An AgentView built without an environment, in the "accel" action mode."""
    pending = tuple(f"pending_{k}_{c}" for k in range(slots) for c in "txyz")
    return AgentView(
        scenario_id="hand-made",
        dt=dt,
        max_steps=360,
        n=N,
        action_mode="accel",
        max_accel=max_accel,
        thrust_limit=limit,
        mass_kg=500.0,
        goal={"hold_point": hold, "capture_radius": 5.0, "capture_speed": 0.05},
        constraints=() if radius is None else ({"type": "keep_out", "radius": radius},),
        observation=STATE_NAMES + pending,
    )


def lqr_plan(model, z0, steps, a_box=None):
    """Rollout of u = us - K (z - zs) on the model, clipped per axis at a_box when given."""
    x = np.empty((steps + 1, len(model.zs)))
    u = np.empty((steps, 3))
    x[0] = z0
    for k in range(steps):
        u[k] = model.us - model.k @ (x[k] - model.zs)
        if a_box is not None:
            u[k] = np.clip(u[k], -a_box, a_box)
        x[k + 1] = model.phi @ x[k] + model.gamma @ u[k]
    return x, u


def waypoint_plan(model, z0, steps, waypoint):
    """Clipped LQR rollout toward a waypoint held at rest: a plan the hard rows always admit."""
    xw = np.concatenate([waypoint, np.zeros(3)])
    uw = steady_state_input(model.phi, model.gamma, xw)
    x = np.empty((steps + 1, 6))
    u = np.empty((steps, 3))
    x[0] = z0
    for k in range(steps):
        u[k] = np.clip(uw - model.k @ (x[k] - xw), -A_MAX, A_MAX)
        x[k + 1] = model.phi @ x[k] + model.gamma @ u[k]
    return x, u


def outward_plan(model, z0, steps, radius, margin=5.0):
    """Clipped LQR rollout with each position closer than radius + margin pushed out radially.

    The plain rollout passes through the keep-out sphere, where the rows only ask to stay as far
    out as the reference; pushed out, the rows bind where the LQR would cut in.
    """
    x, u = lqr_plan(model, z0, steps, A_MAX)
    size = np.linalg.norm(x[:, :3], axis=1)
    inside = size < radius + margin
    x[inside, :3] *= ((radius + margin) / size[inside])[:, None]
    return x, u


def slack_rows(qp):
    """Mask (N, m) of the slack's own bound row in each stage: no coefficient but the slack's."""
    nz = qp.A.shape[1]
    return (np.abs(qp.C[..., : nz + 3]).sum(axis=-1) == 0.0) & (qp.C[..., nz + 3] != 0.0)


def slack_zero(qp):
    """Value of the slack input at zero slack, from its bound row (one per stage)."""
    mask = slack_rows(qp)
    assert np.all(mask.sum(axis=1) == 1), "each stage has one bound row on the slack"
    values = qp.e[mask] / qp.C[..., -1][mask]
    assert np.all(values == values[0])
    return float(values[0])


def keep_out_rows(qp):
    """Mask (N, m) of keep-out rows: a slack coefficient and a state coefficient."""
    nz = qp.A.shape[1]
    return (qp.C[..., nz + 3] != 0.0) & (np.abs(qp.C[..., :nz]).sum(axis=-1) > 0.0)


def with_slack(qp, u, slack_m=0.0):
    """Inputs (N, 4) of the problem from commands (N, 3) and a slack in metres."""
    return np.column_stack([u, np.full(len(u), slack_zero(qp) + slack_m)])


def objective(qp, x, u_full):
    z = np.concatenate([x[:-1], u_full], axis=1)
    stage = 0.5 * np.einsum("ki,kij,kj->", z, qp.Q, z) + np.sum(qp.q * z)
    return float(stage + 0.5 * x[-1] @ qp.P @ x[-1] + qp.p @ x[-1])


def deviation_cost(qp, model, x, u):
    """Cost of a plan on its deviation from the hold point, at zero slack.

    The objective of a StageQP leaves out constants, so the cost is its difference from the
    objective of the plan that sits at the hold point.
    """
    steps = len(u)
    hold_x = np.tile(model.zs, (steps + 1, 1))
    hold_u = np.tile(model.us, (steps, 1))
    return objective(qp, x, with_slack(qp, u)) - objective(qp, hold_x, with_slack(qp, hold_u))


def row_margins(qp, x, u_full):
    """C_k z_k - e_k over the stages and C_N x_N - e_N, each relative to the row's terms."""
    z = np.concatenate([x[:-1], u_full], axis=1)
    value = np.einsum("kij,kj->ki", qp.C, z) - qp.e
    size = np.einsum("kij,kj->ki", np.abs(qp.C), np.abs(z)) + np.abs(qp.e)
    margins = [(value / np.maximum(size, 1e-300)).ravel()]
    if qp.C_N is not None and len(qp.C_N):
        value_n = qp.C_N @ x[-1] - qp.e_N
        size_n = np.abs(qp.C_N) @ np.abs(x[-1]) + np.abs(qp.e_N)
        margins.append(value_n / np.maximum(size_n, 1e-300))
    return np.concatenate(margins)


def hard_problem(qp):
    """The same problem with the slack fixed at zero: no slack input and no slack bound row."""
    nz = qp.A.shape[1]
    sigma0 = slack_zero(qp)
    keep = ~slack_rows(qp)
    count = int(keep[0].sum())
    c = np.stack([qp.C[k][keep[k]] for k in range(len(qp.C))])
    e = np.stack([qp.e[k][keep[k]] for k in range(len(qp.e))]) - c[..., nz + 3] * sigma0
    assert c.shape[1] == count
    return dataclasses.replace(
        qp,
        B=qp.B[..., :3],
        Q=qp.Q[:, : nz + 3, : nz + 3],
        q=qp.q[:, : nz + 3] + qp.Q[:, : nz + 3, nz + 3] * sigma0,
        C=c[..., : nz + 3],
        e=e,
    )


def penalty_weight(qp):
    """The slack's cost per metre at zero slack: the exact-penalty weight rho."""
    sigma0 = slack_zero(qp)
    return float(qp.q[0, -1] + qp.Q[0, -1, -1] * sigma0)


# Configuration


def test_config_is_frozen_and_keyword_only():
    config = MPCConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.horizon = 5
    with pytest.raises(TypeError):
        MPCConfig(LQRConfig())


def test_config_defaults_are_the_documented_ones():
    config = MPCConfig()
    assert config.regulator == LQRConfig()
    assert (config.horizon, config.terminal, config.beta, config.t_max) == (20, "cost", 1.0, 200)
    assert (config.row_samples, config.instep_bound) == (5, "first_order")
    assert (config.penalty, config.side, config.side_hysteresis) == (None, "auto", 3)
    assert config.settings is None


@pytest.mark.parametrize(
    "field",
    [
        {"horizon": 0},
        {"terminal": "equality"},
        {"beta": 0.5},
        {"t_max": -1},
        {"row_samples": 1},
        {"instep_bound": "zeroth_order"},
        {"penalty": 0.0},
        {"penalty": -1.0},
        {"side": "+y"},
        {"side_hysteresis": 0},
    ],
)
def test_config_refuses_values_outside_their_range(field):
    with pytest.raises(ValueError):
        MPCConfig(**field)


# Problem layout


@pytest.mark.parametrize("radius", [None, 50.0])
def test_slack_is_a_fourth_input_outside_the_dynamics_and_the_command_cost(radius):
    view = hand_view(radius=radius)
    model = regulator_model(view, DISCRETE)
    z0 = model.zs + 300.0 * ALONG
    config = MPCConfig(regulator=DISCRETE, horizon=7)
    qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 7, A_MAX), config)
    nz = len(model.zs)
    assert qp.B.shape == (7, nz, 4)
    np.testing.assert_array_equal(qp.A, np.broadcast_to(model.phi, (7, nz, nz)))
    np.testing.assert_array_equal(qp.B[..., :3], np.broadcast_to(model.gamma, (7, nz, 3)))
    assert np.all(qp.B[..., 3] == 0.0) and np.all(qp.c == 0.0)
    np.testing.assert_array_equal(qp.x0, z0)
    # The slack is coupled to nothing in the cost, so the command block is the regulator's.
    assert np.all(qp.Q[:, : nz + 3, nz + 3] == 0.0)
    assert np.all(qp.Q[:, nz + 3, nz + 3] > 0.0)
    assert penalty_weight(qp) > 0.0


@pytest.mark.parametrize("radius", [None, 50.0])
@pytest.mark.parametrize("row_samples", [2, 5])
def test_every_stage_has_box_slack_and_keep_out_rows_with_absent_ones_inert(radius, row_samples):
    view = hand_view(radius=radius)
    model = regulator_model(view, DISCRETE)
    z0 = model.zs + 300.0 * ALONG
    config = MPCConfig(regulator=DISCRETE, horizon=4, row_samples=row_samples)
    qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 4, A_MAX), config)
    assert qp.C.shape == (4, 7 + row_samples, len(model.zs) + 4)
    assert np.all(slack_rows(qp).sum(axis=1) == 1)
    empty = np.all(qp.C == 0.0, axis=-1)
    # A row with no coefficients must hold strictly, or no interior point exists.
    assert np.all(qp.e[empty] < 0.0)
    if radius is None:
        assert not keep_out_rows(qp).any()
    else:
        # The first stage skips the sample at its start, the current state.
        assert keep_out_rows(qp).sum(axis=1).tolist() == [row_samples - 1] + [row_samples] * 3
    assert qp.C_N is None or len(qp.C_N) == 0


def test_reference_of_the_wrong_length_is_refused():
    view = hand_view(radius=50.0)
    model = regulator_model(view, DISCRETE)
    z0 = model.zs + 300.0 * ALONG
    x, u = lqr_plan(model, z0, 6, A_MAX)
    with pytest.raises(ValueError):
        mpc_problem(model, view, z0, (x, u), MPCConfig(regulator=DISCRETE, horizon=5))


def test_terminal_cost_is_beta_times_the_dare_cost_about_the_hold_point():
    view = hand_view()
    model = regulator_model(view, DISCRETE)
    z0 = model.zs + 300.0 * ALONG
    for beta in (1.0, 2.0):
        config = MPCConfig(regulator=DISCRETE, horizon=3, beta=beta)
        qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 3), config)
        np.testing.assert_allclose(qp.P, beta * model.p, rtol=1e-12, atol=0.0)
        # The terminal cost is lowest at the hold point.
        np.testing.assert_allclose(qp.P @ model.zs + qp.p, 0.0, atol=1e-12 * np.abs(qp.p).max())


# Unconstrained: the first move is the LQR law for every horizon (Rawlings et al. 2.5.1)

FREE_CASES = {
    "discrete": (DISCRETE, 0),
    "sampled": (LQRConfig(pos_scale_m=10.0, vel_scale_mps=0.01, stage_cost="sampled"), 0),
    "delay": (LQRConfig(pos_scale_m=10.0, vel_scale_mps=0.01, stage_cost="sampled"), 1),
    "rate": (LQRConfig(pos_scale_m=10.0, vel_scale_mps=0.01, rate_weight=0.1), 0),
}


def free_starts(model):
    """50 states drawn with scales 200, 200, 50 m and 0.1 m/s about the hold point (seed 0),
    followed by pending or last commands of 0.01 m/s^2 scale (seed 1)."""
    rng = np.random.default_rng(0)
    x0 = (rng.normal(size=(6, 50)) * np.array([[200.0], [200.0], [50.0], [0.1], [0.1], [0.1]])).T
    extra = np.random.default_rng(1).normal(size=(50, len(model.zs) - 6)) * 0.01
    return np.hstack([x0 + model.zs[:6], extra])


def free_view(slots):
    # A thrust limit far above every command, with the weights of a 0.02 m/s^2 limit: no row
    # binds and the LQR agent does not clip.
    return hand_view(max_accel=10.0, slots=slots)


@pytest.mark.parametrize("half", [0, 1])
@pytest.mark.parametrize("steps", [1, 2, 5, 20, 60])
@pytest.mark.parametrize("case", list(FREE_CASES))
def test_unconstrained_first_move_is_the_lqr_law_for_every_horizon(case, steps, half):
    # The 50 states run in two halves, under a second each at 60 steps.
    weights, slots = FREE_CASES[case]
    weights = dataclasses.replace(weights, accel_scale_mps2=A_MAX)
    view = free_view(slots)
    model = regulator_model(view, weights)
    config = MPCConfig(regulator=weights, horizon=steps)
    worst = 0.0
    for z0 in free_starts(model)[25 * half : 25 * half + 25]:
        qp = mpc_problem(model, view, z0, lqr_plan(model, z0, steps), config)
        _, u, _ = riccati_solve(qp)
        law = model.us - model.k @ (z0 - model.zs)
        worst = max(worst, np.max(np.abs(u[0, :3] - law)) / np.max(np.abs(law)))
    # The independent condensed solve agrees to 7.5e-13; 1e-10 leaves rounding room.
    assert worst <= 1e-10


@pytest.mark.parametrize("case", ["sampled", "delay"])
def test_unconstrained_first_move_equals_the_lqr_agent_command(case):
    weights, slots = FREE_CASES[case]
    weights = dataclasses.replace(weights, accel_scale_mps2=A_MAX)
    view = free_view(slots)
    model = regulator_model(view, weights)
    agent = LQRAgent(weights)
    agent.reset(view)
    config = MPCConfig(regulator=weights, horizon=5)
    for z0 in free_starts(model)[:10]:
        qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 5), config)
        _, u, _ = riccati_solve(qp)
        # The pending command waits one step and is echoed in the action units (m/s^2 here).
        obs = z0[:6] if not slots else np.concatenate([z0[:6], [view.dt], z0[6:9]])
        command = agent.act(obs, {"time_s": 0.0})
        np.testing.assert_allclose(u[0, :3], command, rtol=0.0, atol=1e-10 * np.abs(command).max())


def test_mpc_takes_the_riccati_path_and_the_lqr_move_when_no_row_binds():
    view = free_view(0)
    weights = dataclasses.replace(DISCRETE, accel_scale_mps2=A_MAX)
    model = regulator_model(view, weights)
    z0 = free_starts(model)[0]
    qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 20), MPCConfig(regulator=weights))
    plan = mpc_solve(qp)
    law = model.us - model.k @ (z0 - model.zs)
    assert (plan.status, plan.path, plan.iterations) == ("converged", "riccati", 0)
    np.testing.assert_allclose(plan.u[0], law, rtol=0.0, atol=1e-10 * np.abs(law).max())
    assert plan.u.shape == (20, 3) and plan.x.shape == (21, 6)
    np.testing.assert_array_equal(plan.slack, np.zeros(20))


# Box rows inside and outside the terminal set (Rawlings et al. 2.5.4)


def box_problem(limit, offset_m, steps=30):
    view = hand_view(limit=limit)
    model = regulator_model(view, DISCRETE)
    z0 = model.zs + offset_m * ALONG
    a_box = A_MAX if limit == "box" else A_MAX / np.sqrt(3.0)
    config = MPCConfig(regulator=DISCRETE, horizon=steps)
    return model, mpc_problem(model, view, z0, lqr_plan(model, z0, steps, a_box), config)


@pytest.mark.parametrize(("limit", "reach"), [("box", REACH_BOX), ("ball", REACH_BALL)])
@pytest.mark.parametrize("fraction", [0.5, 0.99])
def test_mpc_equals_the_lqr_inside_the_terminal_set(limit, reach, fraction):
    model, qp = box_problem(limit, fraction * reach)
    law = model.us - model.k @ (qp.x0 - model.zs)
    shortcut = mpc_solve(qp)
    assert shortcut.path == "riccati"
    np.testing.assert_allclose(shortcut.u[0], law, rtol=0.0, atol=1e-10 * A_MAX)
    sol = solve_horizon(qp)
    assert sol.status == "converged"
    # The interior-point plan meets the LQR move to its tolerance.
    np.testing.assert_allclose(sol.u[0, :3], law, rtol=0.0, atol=1e-6 * A_MAX)


@pytest.mark.parametrize(
    ("limit", "reach", "multiple", "printed", "reference"),
    [
        ("box", REACH_BOX, 3.0, 0.040, 4.0000000000e-02),
        ("box", REACH_BOX, 20.0, 0.380, 3.8000000000e-01),
        ("ball", REACH_BALL, 3.0, 0.023, 2.3094010768e-02),
        ("ball", REACH_BALL, 20.0, 0.219, 2.1939310229e-01),
    ],
)
def test_box_rows_cut_the_lqr_move_outside_the_terminal_set(
    limit, reach, multiple, printed, reference
):
    # Reference |u_MPC - u_LQR|_inf from the condensed problem by BVLS, N = 30 (independent
    # computation); the ball cases are the box values scaled by 1 / sqrt 3.
    model, qp = box_problem(limit, multiple * reach)
    law = model.us - model.k @ (qp.x0 - model.zs)
    sol = solve_horizon(qp, TIGHT)
    assert sol.status == "converged"
    gap = np.max(np.abs(sol.u[0, :3] - law))
    assert round(gap, 3) == printed
    assert gap == pytest.approx(reference, rel=1e-6)
    plan = mpc_solve(qp, settings=TIGHT)
    assert (plan.status, plan.path) == ("converged", "ipm")
    assert plan.iterations > 0
    np.testing.assert_allclose(plan.u, sol.u[:, :3], rtol=0.0, atol=1e-12)


def test_ball_limit_rows_are_the_inscribed_box():
    _, qp = box_problem("ball", 0.5 * REACH_BALL, steps=3)
    x = np.tile(qp.x0, (4, 1))
    inner = A_MAX / np.sqrt(3.0)
    for sign in (1.0, -1.0):
        inside = with_slack(qp, np.full((3, 3), sign * inner * (1.0 - 1e-9)))
        assert np.min(row_margins(qp, x, inside)[: qp.C.size // qp.C.shape[-1]]) >= 0.0
        for axis in range(3):
            u = np.zeros((3, 3))
            u[:, axis] = sign * inner * (1.0 + 1e-6)
            assert np.min(row_margins(qp, x, with_slack(qp, u))) < 0.0


# Exact constrained cost and its horizon independence (Rawlings et al. Prop 2.18)

# 3 x REACH_BOX = 604.528706 m along-track from the hold point, at rest. The plan reaches the
# terminal set within 5 steps, so the cost is the infinite-horizon constrained value for every
# horizon from 5 on (independent computation, BVLS, to 1e-12 for N = 5 to 120).
FAR_OFFSET = -3.0 * REACH_BOX
FAR_VALUE = 1.855711116137e5
FAR_LOWER = 1.845593979241e5  # 1/2 dz' P dz, the unconstrained value


def condensed_value(model, dz0, steps):
    """V_N^0 of the box problem by BVLS on the condensed deviation problem."""
    nz = len(model.zs)
    sx = np.vstack([np.linalg.matrix_power(model.phi, i) for i in range(steps + 1)])
    su = np.zeros((nz * (steps + 1), 3 * steps))
    for i in range(1, steps + 1):
        for j in range(i):
            block = np.linalg.matrix_power(model.phi, i - 1 - j) @ model.gamma
            su[nz * i : nz * (i + 1), 3 * j : 3 * (j + 1)] = block
    qb = np.kron(np.eye(steps + 1), model.q)
    qb[-nz:, -nz:] = model.p
    h = su.T @ qb @ su + np.kron(np.eye(steps), model.r)
    h = 0.5 * (h + h.T)
    f = su.T @ qb @ sx
    chol = np.linalg.cholesky(h)
    bounds = (np.tile(-A_MAX - model.us, steps), np.tile(A_MAX - model.us, steps))
    sol = lsq_linear(
        chol.T, -np.linalg.solve(chol, f @ dz0), bounds=bounds, method="bvls", tol=1e-15
    )
    du = sol.x
    return 0.5 * du @ h @ du + dz0 @ f.T @ du + 0.5 * dz0 @ sx.T @ qb @ sx @ dz0


def far_problem(steps):
    view = hand_view()
    model = regulator_model(view, DISCRETE)
    z0 = model.zs + FAR_OFFSET * ALONG
    config = MPCConfig(regulator=DISCRETE, horizon=steps)
    return model, mpc_problem(model, view, z0, lqr_plan(model, z0, steps, A_MAX), config)


@pytest.mark.parametrize("steps", [5, 10, 20, 40])
def test_constrained_cost_is_the_infinite_horizon_value_for_every_horizon(steps):
    model, qp = far_problem(steps)
    sol = solve_horizon(qp, TIGHT)
    assert sol.status == "converged"
    value = deviation_cost(qp, model, sol.x, sol.u[:, :3])
    assert value == pytest.approx(FAR_VALUE, rel=1e-8)
    assert value == pytest.approx(condensed_value(model, qp.x0 - model.zs, steps), rel=1e-8)
    dz = qp.x0 - model.zs
    assert 0.5 * dz @ qp.P @ dz == pytest.approx(FAR_LOWER, rel=1e-10)
    assert value > FAR_LOWER


def test_constrained_cost_does_not_rise_with_the_horizon():
    values = []
    for steps in (5, 10, 20, 40):
        model, qp = far_problem(steps)
        sol = solve_horizon(qp, TIGHT)
        values.append(deviation_cost(qp, model, sol.x, sol.u[:, :3]))
    for longer, shorter in zip(values[1:], values[:-1], strict=True):
        assert longer <= shorter * (1.0 + 1e-10)


def two_step_gain(qp, model):
    """First-move gain G of u_0 = us + G (z0 - zs), column by column from the Riccati solve."""
    nz = len(model.zs)
    gain = np.empty((3, nz))
    base = riccati_solve(dataclasses.replace(qp, x0=model.zs))[1][0, :3]
    for i in range(nz):
        _, u, _ = riccati_solve(dataclasses.replace(qp, x0=model.zs + np.eye(nz)[i]))
        gain[:, i] = u[0, :3] - base
    return gain


def test_dare_terminal_cost_gives_the_lqr_loop_at_two_steps_and_none_does_not():
    # Closed-loop spectral radius of the two-step first move: 0.9900499039 with the DARE
    # terminal cost (the LQR loop), 0.9999508854 with none (independent computation).
    view = free_view(0)
    weights = dataclasses.replace(DISCRETE, accel_scale_mps2=A_MAX)
    model = regulator_model(view, weights)
    qp = mpc_problem(
        model, view, model.zs, lqr_plan(model, model.zs, 2), MPCConfig(regulator=weights, horizon=2)
    )
    for problem, radius in (
        (qp, 0.99005),
        (dataclasses.replace(qp, P=np.zeros_like(qp.P), p=np.zeros_like(qp.p)), 0.99995),
    ):
        gain = two_step_gain(problem, model)
        loop = model.phi + model.gamma @ gain
        assert np.max(np.abs(np.linalg.eigvals(loop))) == pytest.approx(radius, abs=1e-5)


@pytest.mark.parametrize(("dt", "twice_value"), [(10.0, 116.65796914060), (60.0, 122.99249125509)])
def test_sampled_cost_objective_is_the_van_loan_infinite_horizon_value(dt, twice_value):
    # Qc = diag(1e-4 x3, 1 x3), Rc = 1e3 I, xs = 0, five steps under the DARE terminal cost:
    # x0' P x0 by Van Loan's sampled cost (independent computation).
    view = hand_view(dt=dt, hold=(0.0, 0.0, 0.0), max_accel=10.0)
    weights = LQRConfig(
        pos_scale_m=100.0, vel_scale_mps=1.0, accel_scale_mps2=1e3**-0.5, stage_cost="sampled"
    )
    model = regulator_model(view, weights)
    z0 = np.array([50.0, -80.0, 10.0, 0.02, -0.01, 0.005])
    qp = mpc_problem(
        model, view, z0, lqr_plan(model, z0, 5), MPCConfig(regulator=weights, horizon=5)
    )
    x, u, _ = riccati_solve(qp)
    assert 2.0 * deviation_cost(qp, model, x, u[:, :3]) == pytest.approx(twice_value, rel=1e-10)


# Exact penalty: soft keep-out rows give the hard solution (Boyd and Vandenberghe, exercise
# 5.16; Rawlings et al. 2.4)

# Ten approach-vbar starts (LQR reference) and ten passes 55 to 90 m from the target whose hard
# solution leans on a keep-out row (reference toward a waypoint 70 m out on their side); default
# weights, horizon 20. The largest keep-out multiplier sum of a stage is 205.96 per m, at the
# twelfth state (independent stage builder and solve).
PENALTY_STATES = np.array(
    [
        [
            15.26574058878,
            337.3083479253,
            -13.09327602046,
            -0.02510064688242,
            0.09748899803729,
            0.02655125452143,
        ],
        [
            69.7295722018,
            331.9853821542,
            17.9917661164,
            -0.07540552502256,
            -0.08965415360324,
            0.07003826936176,
        ],
        [
            -196.4416777858,
            591.5070248741,
            32.70030258191,
            0.05704219186527,
            -0.09038330259949,
            -0.05851214675861,
        ],
        [
            139.9462524381,
            372.997924662,
            12.74679337031,
            -0.07554352023781,
            -0.06273273626443,
            -0.0005464174517053,
        ],
        [
            103.7866307289,
            415.9428817378,
            -39.3119921431,
            0.07323721570298,
            -0.07206821808517,
            -0.01206844849754,
        ],
        [
            34.73978937779,
            327.6598009586,
            -8.194583411611,
            0.07739289816532,
            -0.009172859682775,
            0.007849422501041,
        ],
        [
            188.1956608563,
            310.3932545734,
            39.47901878878,
            -0.02762889571425,
            -0.03759179651813,
            0.07625783888979,
        ],
        [
            -173.8013007318,
            546.6620487644,
            32.77676184267,
            0.07894296822011,
            -0.08804854651403,
            -0.05223888108912,
        ],
        [
            110.061446084,
            374.7400351317,
            -49.46868413724,
            0.03852392420589,
            0.05995393479322,
            0.09472498804671,
        ],
        [
            -139.7343482366,
            300.7819899712,
            34.13605920365,
            -0.08163817527417,
            -0.04514588633633,
            -0.009647458664799,
        ],
        [
            -69.16747117862,
            8.115969861182,
            1.828288469033,
            0.003688862736436,
            -0.225066874007,
            -0.002013635641871,
        ],
        [
            -66.39168436862,
            31.349882371,
            -2.736781726068,
            0.01310081690593,
            -0.2255979788427,
            0.01339892474394,
        ],
        [
            56.98996573084,
            -3.100020084095,
            3.51782080411,
            -0.0003331135297592,
            -0.2690240908358,
            0.00123254454426,
        ],
        [
            -58.17536118728,
            -16.32561677586,
            -2.864238477086,
            -0.01167127590123,
            -0.3059442861668,
            0.009289190050243,
        ],
        [
            75.61110408222,
            35.60351626227,
            7.584482844391,
            -0.01256527613171,
            -0.1579101892792,
            -0.01548755585497,
        ],
        [
            76.77320690105,
            28.84661062786,
            -0.6412802610979,
            0.003489817933861,
            -0.06255624030612,
            -0.01260517983255,
        ],
        [
            77.09709166346,
            43.8403233986,
            1.355673414697,
            0.0198388864091,
            -0.09140275410823,
            0.0148688773821,
        ],
        [
            78.08338863314,
            16.28068185937,
            5.936005980608,
            0.0113613379079,
            -0.2590439409427,
            -0.0183305183514,
        ],
        [
            -62.31620034418,
            6.610108794962,
            -5.281951670686,
            0.01789020204844,
            -0.2034103929961,
            0.007615035700755,
        ],
        [
            61.60249022801,
            -7.574183550494,
            3.913094528133,
            -0.006070184705124,
            -0.1768361963191,
            0.01391925640539,
        ],
    ]
)


def penalty_case(index, config=None):
    view = make("approach-vbar").agent_view()
    config = MPCConfig() if config is None else config
    model = regulator_model(view, config.regulator)
    z0 = PENALTY_STATES[index]
    if z0[1] > 150.0:
        reference = lqr_plan(model, z0, config.horizon, A_MAX)
    else:
        waypoint = np.array([np.sign(z0[0]) * 70.0, 0.0, 0.0])
        reference = waypoint_plan(model, z0, config.horizon, waypoint)
    return model, mpc_problem(model, view, z0, reference, config)


@pytest.mark.parametrize("index", range(20))
def test_default_penalty_gives_the_hard_solution(index):
    _, qp = penalty_case(index)
    hard = solve_horizon(hard_problem(qp), TIGHT)
    assert hard.status == "converged"
    soft = solve_horizon(qp, TIGHT)
    assert soft.status == "converged"
    np.testing.assert_allclose(soft.u[:, :3], hard.u, rtol=0.0, atol=1e-6 * A_MAX)
    assert np.max(soft.u[:, 3] - slack_zero(qp)) <= 1e-9


def test_penalty_below_the_keep_out_multipliers_leaves_slack():
    # At a hundredth of the default weight the slack is cheaper than the row it relaxes; the
    # independent solve gives every pass a slack, 8.6 m at most.
    _, default = penalty_case(11)
    config = MPCConfig(penalty=penalty_weight(default) / 100.0)
    _, qp = penalty_case(11, config)
    assert penalty_weight(qp) == pytest.approx(penalty_weight(default) / 100.0, rel=1e-12)
    soft = solve_horizon(qp, TIGHT)
    assert soft.status == "converged"
    assert np.max(soft.u[:, 3] - slack_zero(qp)) > 1e-6


def test_default_penalty_is_ten_times_the_largest_keep_out_multiplier_sum():
    # 10 x 205.958 per m for the default weights; the rule scales with the stage cost, and 5%
    # covers tightening details that move the multipliers.
    _, qp = penalty_case(11)
    assert penalty_weight(qp) == pytest.approx(10.0 * 205.958, rel=0.05)


# Keep-out rows between samples (Rawlings et al. 8.5.1)

# Zero-thrust coasts with both ends on the sphere, chord above the target (+x) moving toward
# +V-bar: (speed m/s, step s, radius m, start state, dip below the sphere between the ends in m);
# start states and dips from a dense scan of the exact CW coast (independent computation).
COASTS = [
    (
        0.2,
        10.0,
        50.0,
        [49.9899989998, -1.0, 0.0, -0.003132014410906, 0.1999884451041, 0.0],
        1.783106e-2,
    ),
    (
        1.0,
        10.0,
        50.0,
        [49.74937185533, -5.0, 0.0, -0.01198158955548, 0.9999557964932, 0.0],
        2.805822e-1,
    ),
    (
        0.2,
        5.0,
        2.0,
        [1.936491673104, -0.5, 0.0, -0.001124565912271, 0.1999979255807, 0.0],
        6.491404e-2,
    ),
]


@pytest.mark.parametrize("bound", ["first_order", "second_order"])
@pytest.mark.parametrize("row_samples", [2, 5])
@pytest.mark.parametrize("coast", range(3), ids=["slow", "fast", "small"])
def test_rows_reject_a_coast_that_dips_inside_between_samples(coast, row_samples, bound):
    speed, step, radius, start, dip = COASTS[coast]
    x0 = np.array(start)
    phi, _ = cw_discrete(N, step)
    x1 = phi @ x0
    for state in (x0, x1):
        assert np.linalg.norm(state[:3]) == pytest.approx(radius, rel=1e-12)
    # Rows only at the ends, tangent there with no tightening, accept the coast: the case a
    # plan checked at its samples alone would fly.
    assert x1[:3] @ (x1[:3] / np.linalg.norm(x1[:3])) >= radius * (1.0 - 1e-12)

    view = hand_view(dt=step, radius=radius)
    model = regulator_model(view, LQRConfig())
    # The reference is the same coast scaled out by 10%, clear of the sphere and its margin, so
    # each row asks for the full radius plus the tightening.
    reference = (np.vstack([1.1 * x0, 1.1 * x1]), np.zeros((1, 3)))
    config = MPCConfig(horizon=1, row_samples=row_samples, instep_bound=bound)
    qp = mpc_problem(model, view, x0, reference, config)
    margins = np.einsum(
        "ij,j->i", qp.C[0], np.concatenate([x0, with_slack(qp, np.zeros((1, 3)))[0]])
    )
    shortfall = -np.min(margins - qp.e[0])
    if row_samples == 2:
        # The end row asks for R + b with b at least the dip the samples miss.
        assert shortfall >= dip
    else:
        assert shortfall > 0.5 * dip


def test_keep_out_rows_of_a_delayed_model_ignore_the_command_being_issued():
    # With one step of delay the command issued at step k acts during step k + 1; the samples of
    # step k depend on the pending command, which is part of the state z_k.
    for slots, depends in ((1, False), (0, True)):
        view = hand_view(radius=50.0, slots=slots)
        model = regulator_model(view, LQRConfig())
        nz = len(model.zs)
        z0 = model.zs.copy()
        z0[:6] = [60.0, 40.0, 0.0, 0.0, -0.2, 0.0]
        qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 4, A_MAX), MPCConfig(horizon=4))
        rows = qp.C[keep_out_rows(qp)]
        assert len(rows)
        assert bool(np.any(rows[:, nz : nz + 3] != 0.0)) is depends
        if slots:
            assert np.any(rows[:, 6:9] != 0.0)


# Recursive feasibility of the shifted plan (Rawlings et al. 2.4.2, 2.7)

# 55 to 120 m from the target, under 0.05 m/s per axis; each first plan converges with the
# terminal set (independent stage builder and solve, which also found that rows linearized at
# the current position cut the shifted plan in 9 of the 20).
SHIFT_STATES = np.array(
    [
        [
            -13.88113568233,
            -74.97909224743,
            8.801111651927,
            0.03698965116962,
            -0.01089151934608,
            -0.00621181268772,
        ],
        [
            -72.66493875104,
            31.57515170167,
            -0.6666157095038,
            -0.02586478549153,
            -0.02428547514489,
            -0.03152684431656,
        ],
        [
            34.59663080962,
            -58.077489562,
            -6.600026276791,
            0.009090285176541,
            0.01042722960585,
            0.01468580270602,
        ],
        [
            -108.3815523715,
            36.10813462691,
            -5.876999362915,
            -0.02153793624892,
            -0.04832179476755,
            -0.03184460626373,
        ],
        [
            79.42390512002,
            -14.02938159961,
            -1.539947800857,
            0.01073306675715,
            -0.02769400145866,
            -0.03660128682006,
        ],
        [
            69.38263143415,
            31.99363310546,
            -4.100136640153,
            -0.003982069251004,
            0.02189886438895,
            0.03670248693354,
        ],
        [
            -14.25597775701,
            -56.60162818788,
            11.41175252973,
            -0.03812228907017,
            -0.01249782443017,
            0.0002254011905081,
        ],
        [
            103.5671180843,
            -0.7060926960568,
            -2.075356132805,
            -0.02238406473994,
            -0.005083483729893,
            0.0009974007901371,
        ],
        [
            63.26783402404,
            -66.41528413658,
            4.787790569414,
            -0.04098538213387,
            0.01110419644322,
            -0.004096240561857,
        ],
        [
            -63.42007177953,
            6.628468735808,
            -11.0984349136,
            -0.01276780373921,
            -0.009622031815782,
            -0.00112176014896,
        ],
        [
            -40.04481916746,
            -44.06258139729,
            -5.196628706013,
            0.03913654203826,
            0.01011123336801,
            0.01586733536551,
        ],
        [
            -87.98413103745,
            16.63263644303,
            -12.50498780526,
            -0.03158534211426,
            -0.01593156795648,
            -0.03299411813522,
        ],
        [
            19.09171881401,
            -83.9114180658,
            -6.507803032058,
            -0.01935433317076,
            0.02821136862164,
            0.04744879794872,
        ],
        [
            94.4251573452,
            9.878284403693,
            -1.921404663584,
            -0.03423079981204,
            -0.00657931057351,
            0.03904103702147,
        ],
        [
            -65.63631223387,
            -18.4279779084,
            12.22362125816,
            0.01294125124552,
            -0.005693463913446,
            0.01533146577009,
        ],
        [
            67.52120125574,
            30.64535044631,
            9.858795736097,
            -0.003337155009576,
            0.04267701067604,
            -0.03121813527061,
        ],
        [
            50.92317830923,
            -22.92894810815,
            10.93916579289,
            0.01378199386372,
            -0.04079521896294,
            0.00585596463676,
        ],
        [
            77.59668954302,
            19.88206220131,
            10.22552941065,
            0.02143498305245,
            0.03085003573623,
            0.01102123589440,
        ],
        [
            71.77186114038,
            -70.13039984727,
            2.401531188266,
            0.03725177982638,
            -0.01301773231952,
            -0.03333727686011,
        ],
        [
            -50.38606492593,
            25.40340257879,
            -1.455834017366,
            -0.04802757732929,
            0.01785507926857,
            0.02240895843062,
        ],
    ]
)


@pytest.mark.parametrize("index", range(20))
def test_shifted_plan_meets_every_row_of_the_next_problem(index):
    view = make("approach-vbar").agent_view()
    config = MPCConfig(terminal="set")
    model = regulator_model(view, config.regulator)
    z0 = SHIFT_STATES[index]
    qp = mpc_problem(model, view, z0, outward_plan(model, z0, 20, 50.0), config)
    sol = solve_horizon(qp)
    assert sol.status == "converged"
    x, u = sol.x, sol.u[:, :3]
    u_last = model.us - model.k @ (x[-1] - model.zs)
    shifted = (
        np.vstack([x[1:], model.phi @ x[-1] + model.gamma @ u_last]),
        np.vstack([u[1:], u_last]),
    )
    following = mpc_problem(model, view, x[1], shifted, config)
    assert following.C_N is not None and len(following.C_N)
    # Rows that hold by construction may miss by the solver's tolerance on the first plan.
    assert np.min(row_margins(following, shifted[0], with_slack(following, shifted[1]))) >= -1e-7


# Terminal set


def test_terminal_rows_are_an_invariant_set_clear_of_the_keep_out_sphere():
    view = make("approach-vbar").agent_view()
    config = MPCConfig(terminal="set")
    model = regulator_model(view, config.regulator)
    z0 = model.zs + 30.0 * ALONG
    qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 20, A_MAX), config)
    c_n, e_n = qp.C_N, qp.e_N
    assert c_n is not None and len(c_n)
    assert np.all(c_n @ model.zs > e_n)

    def lowest(row):
        lp = linprog(row, A_ub=-c_n, b_ub=-e_n, bounds=[(None, None)] * len(model.zs))
        assert lp.status == 0
        return lp.fun

    f = model.phi - model.gamma @ model.k
    shift = model.zs - f @ model.zs
    # Invariant under the LQR loop: every row holds at the image of every point of the set.
    for row, bound in zip(c_n, e_n, strict=True):
        assert lowest(row @ f) + row @ shift >= bound - 1e-9 * (abs(bound) + 1.0)
    # The LQR command stays in the box over the set.
    for i in range(3):
        for sign in (1.0, -1.0):
            command = sign * -model.k[i]
            top = -lowest(-command) + sign * (model.us[i] + model.k[i] @ model.zs)
            assert top <= A_MAX * (1.0 + 1e-9)
    # The hold-point halfspace keeps the samples clear of the sphere by the chord-to-arc allowance
    # between them, A dt^2 / 8 with sqrt(3) a_max of thrust: 1.4595 m at the largest range and
    # speed of the set itself (3894 m, 11.41 m/s; independent computation).
    n_h = np.zeros(len(model.zs))
    n_h[:3] = np.array(VBAR_HOLD) / np.linalg.norm(VBAR_HOLD)
    assert lowest(n_h) >= 50.0 + 1.4595


def test_terminal_cost_alone_has_no_terminal_rows():
    view = make("approach-vbar").agent_view()
    model = regulator_model(view, LQRConfig())
    z0 = model.zs + 30.0 * ALONG
    qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 20, A_MAX), MPCConfig())
    assert qp.C_N is None or len(qp.C_N) == 0


def terminal_build(view, config):
    """Terminal rows of one build, or the refusal message."""
    model = regulator_model(view, config.regulator)
    z0 = model.zs + 30.0 * ALONG
    try:
        qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 3, A_MAX), config)
    except ValueError as refusal:
        return str(refusal)
    return qp.C_N.copy(), qp.e_N.copy()


def same_build(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return a[0].shape == b[0].shape and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_stored_terminal_rows_cannot_be_written(monkeypatch):
    monkeypatch.setattr(mpc, "_TERMINAL_CACHE", {})
    terminal_build(hand_view(radius=50.0), MPCConfig(terminal="set", horizon=3))
    assert mpc._TERMINAL_CACHE
    for stored in mpc._TERMINAL_CACHE.values():
        for array in stored:
            with pytest.raises(ValueError):
                array[0] = 0.0


@pytest.mark.parametrize(
    "configs",
    [
        # Different weights give a different gain on the same orbit, step and hold point.
        (
            MPCConfig(terminal="set", horizon=3),
            MPCConfig(regulator=DISCRETE, terminal="set", horizon=3),
        ),
        # t* = 1 here, so a cap of 0 refuses the set that a cap of 200 builds.
        (MPCConfig(terminal="set", horizon=3), MPCConfig(terminal="set", horizon=3, t_max=0)),
    ],
    ids=["weights", "t_max"],
)
def test_interleaved_builds_give_the_terminal_rows_of_separate_builds(monkeypatch, configs):
    view = hand_view(radius=50.0)
    separate = []
    for config in configs:
        monkeypatch.setattr(mpc, "_TERMINAL_CACHE", {})
        separate.append(terminal_build(view, config))
    assert not same_build(*separate)
    monkeypatch.setattr(mpc, "_TERMINAL_CACHE", {})
    for config, expected in [
        *zip(configs, separate, strict=True),
        *zip(configs, separate, strict=True),
    ]:
        assert same_build(terminal_build(view, config), expected)


def test_a_reference_meets_its_own_keep_out_rows_inside_the_tightening():
    # Rows ask for min(R + b, |r_ref|), so a reference that grazes the sphere or dips inside it
    # between samples is still a plan for its own rows, which keeps the shifted plan feasible.
    cases = []
    for _, step, radius, start, _ in COASTS:
        x0 = np.array(start)
        phi, _ = cw_discrete(N, step)
        cases.append(
            (hand_view(dt=step, radius=radius), x0, (np.vstack([x0, phi @ x0]), np.zeros((1, 3))))
        )
    view = make("approach-vbar").agent_view()
    model = regulator_model(view, LQRConfig())
    for z0 in PENALTY_STATES[10:14]:
        cases.append((view, z0, lqr_plan(model, z0, 20, A_MAX)))
    for case_view, z0, (x_ref, u_ref) in cases:
        model = regulator_model(case_view, LQRConfig())
        config = MPCConfig(horizon=len(u_ref))
        qp = mpc_problem(model, case_view, z0, (x_ref, u_ref), config)
        assert np.min(row_margins(qp, x_ref, with_slack(qp, u_ref))) >= -1e-12


@pytest.mark.parametrize("thrust", [1.0, -1.0], ids=["speeding_up", "slowing_down"])
def test_a_sample_takes_the_tightening_of_both_intervals_it_ends(thrust):
    # A reference 500 m out, thrusting 1 m/s^2 along its velocity: the speed changes by 2.5 m/s
    # over each 2.5 s interval, so the two intervals at a sample need different tightenings.
    view = hand_view(radius=50.0)
    model = regulator_model(view, LQRConfig())
    phis, gammas = cw_discrete(N, np.linspace(0.0, DT, 5))
    x0 = np.array([500.0, 0.0, 0.0, 0.0, 10.0, 0.0])
    u_ref = np.array([[0.0, thrust, 0.0]] * 3)
    x_ref = np.empty((4, 6))
    x_ref[0] = x0
    for k in range(3):
        x_ref[k + 1] = phis[-1] @ x_ref[k] + gammas[-1] @ u_ref[k]
    qp = mpc_problem(model, view, x0, (x_ref, u_ref), MPCConfig(horizon=3))
    mask = keep_out_rows(qp)
    sigma0 = slack_zero(qp)
    h = DT / 4.0
    for k in range(3):
        samples = phis @ x_ref[k] + gammas @ u_ref[k]
        speed, _ = interval_bounds(samples, h, 0.0, N)
        tightening = qp.e[k][mask[k]] - qp.C[k][mask[k]][:, -1] * sigma0 - 50.0
        # Stage 0 skips its first sample; each later sample j is the end of intervals j - 1
        # and j, when they exist.
        for row, j in enumerate(range(1 if k == 0 else 0, 5)):
            needed = max(speed[i] for i in (j - 1, j) if 0 <= i < 4) * h / 2.0
            assert tightening[row] >= needed * (1.0 - 1e-12)


def test_a_riccati_plan_just_outside_the_box_goes_to_the_interior_point_solve():
    # 1.0001 times the terminal set's reach the LQR move asks 1.0001 a_max: the rows miss by
    # 5e-5 of their terms, far above rounding, so the Riccati plan is not the optimum.
    model, qp = box_problem("box", 1.0001 * REACH_BOX)
    plan = mpc_solve(qp)
    assert plan.path == "ipm"
    assert np.max(np.abs(plan.u)) <= A_MAX * (1.0 + 1e-8)


def terminal_row_problem(miss):
    """A free problem with one terminal row y_N >= e that the Riccati plan misses by `miss` of the
    row's terms |y_N| + |e|, and that plan."""
    view = free_view(0)
    weights = dataclasses.replace(DISCRETE, accel_scale_mps2=A_MAX)
    model = regulator_model(view, weights)
    z0 = free_starts(model)[0]
    qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 20), MPCConfig(regulator=weights))
    free = mpc_solve(qp)
    assert free.path == "riccati"
    # This plan ends 41 m behind the target, so the row's terms are far above rounding.
    y_n = free.x[-1, 1]
    assert abs(y_n) > 10.0
    row = np.zeros((1, qp.A.shape[1]))
    row[0, 1] = 1.0
    bound = np.array([y_n + 2.0 * miss * abs(y_n)])
    return dataclasses.replace(qp, C_N=row, e_N=bound), free, 2.0 * abs(y_n)


@pytest.mark.parametrize("miss", [1e-4, 1e-6, 1e-8, 1e-10])
def test_a_riccati_plan_that_misses_a_terminal_row_by_more_than_rounding_is_not_the_optimum(miss):
    qp, free, size = terminal_row_problem(miss)
    plan = mpc_solve(qp, settings=TIGHT)
    assert (plan.status, plan.path) == ("converged", "ipm")
    # The interior-point plan meets the row the Riccati plan missed.
    assert plan.x[-1, 1] >= qp.e_N[0] - 1e-9 * size
    assert plan.x[-1, 1] - free.x[-1, 1] > 0.5 * miss * size


@pytest.mark.parametrize("miss", [0.0, 1e-14])
def test_a_riccati_plan_that_misses_a_terminal_row_by_rounding_only_is_the_optimum(miss):
    qp, free, _ = terminal_row_problem(miss)
    plan = mpc_solve(qp, settings=TIGHT)
    assert (plan.status, plan.path) == ("converged", "riccati")
    np.testing.assert_array_equal(plan.u, free.u)


# Extreme steps: a problem whose first move is the LQR law, or a refusal

STEP_OUTCOMES = {
    # dt: (terminal "cost", terminal "set" with t_max = 20); None builds.
    1e-6: (None, "terminal set"),
    1e-3: (None, "terminal set"),
    1.0: (None, None),
    10.0: (None, None),
    60.0: (None, "terminal set"),
    600.0: (None, "terminal set"),
    1e4: ("row_samples too few", "terminal set|row_samples too few"),
}


@pytest.mark.parametrize("terminal", ["cost", "set"])
@pytest.mark.parametrize("dt", list(STEP_OUTCOMES))
def test_extreme_steps_build_the_lqr_problem_or_refuse(dt, terminal):
    # The in-step bound needs g = c h (1.5 n^2 h + 2 n) / 2 < 1 at h = dt / 4 (c = 2): 17.0 at
    # 1e4 s, 0.37 at 600 s. The set's loop has spectral radius 1 - 1e-9 at 1e-6 s and 1 - 1e-6 at
    # 1e-3 s, so t* exceeds 20 there; at 600 s the bound between samples on the hold row is
    # infinite (gain at h = dt above 1) and no hold-row set exists. At 60 s the set reaches 8.3 km
    # radially and 9.8 km cross-track, so the allowance between samples on the hold row is about
    # 103 m for any room left to it, above the 50 m between the hold point and the sphere.
    # t* = 5, 1 at 1, 10 s.
    expected = STEP_OUTCOMES[dt][terminal == "set"]
    view = hand_view(dt=dt, radius=50.0)
    model = regulator_model(view, DISCRETE)
    config = MPCConfig(regulator=DISCRETE, horizon=5, terminal=terminal, t_max=20)
    starts = free_starts(model)[:5]
    if expected is not None:
        with pytest.raises(ValueError, match=f"^({expected})"):
            mpc_problem(model, view, starts[0], lqr_plan(model, starts[0], 5), config)
        return
    for z0 in starts:
        qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 5), config)
        _, u, _ = riccati_solve(qp)
        law = model.us - model.k @ (z0 - model.zs)
        np.testing.assert_allclose(u[0, :3], law, rtol=0.0, atol=1e-9 * np.abs(law).max())


# Solver outcomes passed through


def stub_solution(status, qp, **fields):
    defaults = dict(
        x=None,
        u=None,
        status=status,
        iterations=7,
        objective=float("nan"),
        residual_primal=1.0,
        residual_dual=1.0,
        gap=1.0,
        multipliers=None,
        conflicting=((2, 3), (qp.B.shape[0], 0)) if status == "infeasible_certified" else (),
        elapsed_s=0.25,
    )
    defaults.update(fields)
    return HorizonSolution(**defaults)


@pytest.mark.parametrize("status", ["iteration_cap", "infeasible_certified", "solver_failed"])
def test_unconverged_solver_outcomes_pass_through_with_no_plan(monkeypatch, status):
    _, qp = box_problem("box", 3.0 * REACH_BOX, steps=10)
    calls = []

    def stub(problem, settings=None, warm=None):
        calls.append((problem, settings, warm))
        return stub_solution(status, problem)

    monkeypatch.setattr(mpc, "solve_horizon", stub)
    warm = (np.tile(qp.x0, (11, 1)), np.zeros((10, 4)))
    plan = mpc_solve(qp, settings=TIGHT, warm=warm)
    assert len(calls) == 1
    assert calls[0][0] is qp and calls[0][1] is TIGHT and calls[0][2] is warm
    assert (plan.status, plan.path, plan.iterations) == (status, "ipm", 7)
    assert plan.x is None and plan.u is None and plan.slack is None
    assert plan.conflicting == stub_solution(status, qp).conflicting


def test_converged_solver_plan_is_split_into_commands_and_slack_in_metres(monkeypatch):
    _, qp = box_problem("box", 3.0 * REACH_BOX, steps=10)
    x = np.tile(qp.x0, (11, 1))
    u = with_slack(qp, np.full((10, 3), 0.01), slack_m=0.5)

    def stub(problem, settings=None, warm=None):
        return stub_solution("converged", problem, x=x, u=u, iterations=9, objective=1.0)

    monkeypatch.setattr(mpc, "solve_horizon", stub)
    plan = mpc_solve(qp)
    assert (plan.status, plan.path, plan.iterations) == ("converged", "ipm", 9)
    np.testing.assert_array_equal(plan.x, x)
    np.testing.assert_array_equal(plan.u, u[:, :3])
    np.testing.assert_allclose(plan.slack, np.full(10, 0.5), rtol=1e-12)
    assert plan.conflicting == ()


def test_iteration_cap_returns_no_plan():
    _, qp = box_problem("box", 3.0 * REACH_BOX, steps=10)
    plan = mpc_solve(qp, settings=IPMSettings(max_iter=1))
    assert (plan.status, plan.path, plan.iterations) == ("iteration_cap", "ipm", 1)
    assert plan.x is None and plan.u is None and plan.slack is None


def test_terminal_set_out_of_reach_is_certified_infeasible():
    # From 500 m behind the hold the 0.02 m/s^2 box cannot reach the terminal set in 100 s.
    view = make("approach-vbar").agent_view()
    config = MPCConfig(terminal="set", horizon=10)
    model = regulator_model(view, config.regulator)
    z0 = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])
    qp = mpc_problem(model, view, z0, lqr_plan(model, z0, 10, A_MAX), config)
    plan = mpc_solve(qp)
    assert (plan.status, plan.path) == ("infeasible_certified", "ipm")
    assert plan.x is None and plan.u is None
    assert any(stage == 10 for stage, _ in plan.conflicting)
