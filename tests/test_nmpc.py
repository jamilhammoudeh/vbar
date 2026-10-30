"""Nonlinear MPC by direct multiple shooting and real-time iteration on a declared model.

Each decision linearizes the declared model at the nodes and controls of the shifted previous plan
(Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed., sections 8.5.2 and 8.6.3), with the
node Jacobians from the model's variational equations, and solves one horizon QP: the real-time
iteration of section 8.9.2, eq. 8.60, whose preparation needs no new measurement. References
marked "independent computation" come from a separate shooting code that places a fresh model at
each node, a dense KKT solve, or central differences of `TwoBodyDynamics.advance`, not from the
module under test. Every number is for a 500 km circular orbit of the WGS 84 Earth at
i = 51.6 deg, a 10 s step and a 0.02 m/s^2 box limit per axis unless stated.
"""

import dataclasses
import inspect
import math
from pathlib import Path

import gymnasium
import numpy as np
import pytest
from metric_cases import episode

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import nmpc
from vbar.agents.base import AgentView, agent_generator, command_fraction
from vbar.agents.lqr import LQRAgent, LQRConfig, regulator_model
from vbar.agents.lyapunov import DeclaredModel
from vbar.agents.mpc import MPCConfig, mpc_problem, mpc_solve
from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.dynamics.models import TwoBodyDynamics
from vbar.metrics import agent_metrics
from vbar.metrics.agent import solver_metrics
from vbar.scenarios import make
from vbar.solvers import HorizonSolution

A_ORBIT = EARTH.radius + 500e3
N = mean_motion(A_ORBIT)
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
INCLINATION = np.radians(51.6)
VBAR_HOLD = (0.0, -100.0, 0.0)
# 300 m behind the V-bar hold point at rest: the first moves ride the along-track thrust limit and
# the path stays 350 m clear of the 50 m keep-out sphere.
BEHIND = (0.0, -400.0, 0.0, 0.0, 0.0, 0.0)
INFO = {"time_s": 0.0}
CW = DeclaredModel(dynamics="cw")
TWO_BODY = DeclaredModel(dynamics="twobody", inclination=INCLINATION)
TWO_BODY_J2 = DeclaredModel(dynamics="twobody_j2", inclination=INCLINATION)
# Velocities in metres per radian of orbit and accelerations in metres per radian squared, so
# every Jacobian entry is of order one and one relative tolerance serves all of them.
SCALE = np.diag([1.0, 1.0, 1.0, 1.0 / N, 1.0 / N, 1.0 / N])


def config(model=CW, horizon=10, phase=0.0, **mpc):
    return nmpc.NMPCConfig(
        mpc=MPCConfig(horizon=horizon, **mpc), model=model, argument_of_latitude=phase
    )


def hand_view(mode="accel", keep_out=50.0, hold=VBAR_HOLD, limit="box", task=None):
    """An AgentView built without an environment."""
    goal = {"hold_point": hold, "capture_radius": 5.0, "capture_speed": 0.05}
    if task is not None:
        goal["task"] = task
    constraints = () if keep_out is None else ({"type": "keep_out", "radius": keep_out},)
    return AgentView(
        scenario_id="hand-made",
        dt=DT,
        max_steps=360,
        n=N,
        action_mode=mode,
        max_accel=A_MAX,
        thrust_limit=limit,
        mass_kg=500.0,
        goal=goal,
        constraints=constraints,
    )


def command(agent, view, x, time_s=0.0):
    """The agent's command (m/s^2, Hill)."""
    action = agent.act(np.asarray(x, dtype=float), {"time_s": time_s})
    return command_fraction(view, action) * A_MAX


def node_model(t, phase=0.0, j2=False):
    """A fresh model whose target is at argument of latitude phase + n t (independent placement)."""
    return TwoBodyDynamics(
        "earth", j2=j2, elements=[A_ORBIT, 0.0, INCLINATION, 0.0, 0.0, phase + N * t]
    )


def segment_end(t, s, q, h, phase=0.0, j2=False):
    m = node_model(t, phase, j2)
    m.reset(s)
    return m.advance(q, h)[0]


def implied_ends(a, b, c, nodes, controls):
    """F_k = A_k s_k + B_k q_k + c_k, the segment ends the linearization reproduces at its nodes."""
    return np.einsum("kij,kj->ki", a, nodes[:-1]) + np.einsum("kij,kj->ki", b, controls) + c


def fly(x0, controls, h):
    m = node_model(0.0)
    m.reset(x0)
    x = np.asarray(x0, dtype=float)
    for q in controls:
        x, _ = m.advance(q, h)
    return x


def cw_minimum_energy(x0, steps, h):
    """The CW plan minimizing sum h |q_k|^2 to the origin at rest, by least squares."""
    phi, gamma = cw_discrete(N, h)
    cols = np.hstack([np.linalg.matrix_power(phi, steps - 1 - k) @ gamma for k in range(steps)])
    q = np.linalg.lstsq(cols, -np.linalg.matrix_power(phi, steps) @ x0, rcond=None)[0]
    q = q.reshape(steps, 3)
    s = [np.asarray(x0, dtype=float)]
    for k in range(steps):
        s.append(phi @ s[-1] + gamma @ q[k])
    return np.array(s), q


def gauss_newton_step(x0, a, b, c, h):
    """Minimum-energy rendezvous on the linearized shooting constraints, by a dense KKT solve.

    min sum h |q_k|^2 subject to s_0 = x0, s_(k+1) = A_k s_k + B_k q_k + c_k and s_N = 0.
    """
    steps = len(a)
    ns, nq = 6 * (steps + 1), 3 * steps
    nv = ns + nq
    hess = np.zeros((nv, nv))
    hess[ns:, ns:] = 2.0 * h * np.eye(nq)
    rows = [np.eye(6, nv)]
    rhs = [np.asarray(x0, dtype=float)]
    for k in range(steps):
        row = np.zeros((6, nv))
        row[:, 6 * k : 6 * k + 6] = a[k]
        row[:, ns + 3 * k : ns + 3 * k + 3] = b[k]
        row[:, 6 * k + 6 : 6 * k + 12] = -np.eye(6)
        rows.append(row)
        rhs.append(-c[k])
    rows.append(np.eye(6, nv, ns - 6))
    rhs.append(np.zeros(6))
    eq = np.vstack(rows)
    kkt = np.block([[hess, eq.T], [eq, np.zeros((len(eq), len(eq)))]])
    sol = np.linalg.solve(kkt, np.concatenate([np.zeros(nv), *rhs]))
    return sol[:ns].reshape(steps + 1, 6), sol[ns:nv].reshape(steps, 3)


def one_iteration(x0, steps, h):
    s, q = cw_minimum_energy(x0, steps, h)
    a, b, c = nmpc.shooting_linearization(TWO_BODY, 0.0, s, q, h)
    return (s, q), gauss_newton_step(x0, a, b, c, h)


def continuity(nodes, controls, h):
    """max_k |F_k - s_(k+1)| with each segment flown on a model placed independently (m, m/s)."""
    ends = [segment_end(k * h, nodes[k], controls[k], h) for k in range(len(controls))]
    return float(np.max(np.abs(np.array(ends) - nodes[1:])))


START_2KM = np.array([100.0, -2000.0, 50.0, 0.05, 0.2, -0.02])


def test_cw_linearization_is_the_exact_zero_order_hold_step_with_no_affine_term():
    rng = np.random.default_rng(0)
    nodes = rng.normal(size=(4, 6)) * np.array([300.0] * 3 + [0.3] * 3)
    controls = rng.uniform(-A_MAX, A_MAX, size=(3, 3))
    a, b, c = nmpc.shooting_linearization(CW, 120.0, nodes, controls, DT)
    phi, gamma = cw_discrete(N, DT)
    assert a.shape == (3, 6, 6) and b.shape == (3, 6, 3) and c.shape == (3, 6)
    for k in range(3):
        assert np.max(np.abs(SCALE @ (a[k] - phi) @ np.linalg.inv(SCALE))) <= 1e-12
        assert np.max(np.abs(SCALE @ (b[k] - gamma))) * N**2 <= 1e-12
    # The affine term of an exact linear model is rounding of the 300 m states.
    assert np.max(np.abs(SCALE @ c.T)) <= 1e-12 * 300.0


@pytest.mark.parametrize("j2", [False, True])
def test_two_body_jacobians_equal_central_differences_of_the_flown_segment(j2):
    # Independent computation: steps of 1e-2 m, 1e-3 m/s and 1e-4 m/s^2 (1e-2 m over the step)
    # agree with the variational Jacobians to 1e-10 nondimensionally; the differences' own
    # truncation is (1e-2 m / 6878 km)^2.
    phase, h = 0.7, DT
    s0 = np.array([100.0, -2000.0, 50.0, 0.05, 0.2, -0.02])
    q0 = np.array([1e-4, -2e-4, 5e-5])
    if j2:
        # The mean elements of the J2 target are known in closed form only at the episode start.
        t0, k = 0.0, 0
        nodes, controls = np.array([s0, s0]), np.array([q0])
        model = TWO_BODY_J2
    else:
        # The second node checks that node k sits at t0 + k dt on the target's orbit.
        t0, k = 300.0, 1
        nodes, controls = np.array([s0 * 0.5, s0, s0]), np.array([q0 * 0.5, q0])
        model = TWO_BODY
    a, b, c = nmpc.shooting_linearization(model, t0, nodes, controls, h, argument_of_latitude=phase)
    t = t0 + k * h
    steps = np.array([1e-2] * 3 + [1e-2 / h] * 3)
    fd_a = np.zeros((6, 6))
    for i in range(6):
        e = np.zeros(6)
        e[i] = steps[i]
        hi = segment_end(t, s0 + e, q0, h, phase, j2)
        lo = segment_end(t, s0 - e, q0, h, phase, j2)
        fd_a[:, i] = (hi - lo) / (2.0 * steps[i])
    da = 1e-2 / h**2
    fd_b = np.zeros((6, 3))
    for i in range(3):
        e = np.zeros(3)
        e[i] = da
        fd_b[:, i] = (
            segment_end(t, s0, q0 + e, h, phase, j2) - segment_end(t, s0, q0 - e, h, phase, j2)
        ) / (2.0 * da)
    inv = np.linalg.inv(SCALE)
    a_nd, fd_a_nd = SCALE @ a[k] @ inv, SCALE @ fd_a @ inv
    b_nd, fd_b_nd = SCALE @ b[k] * N**2, SCALE @ fd_b * N**2
    assert np.max(np.abs(a_nd - fd_a_nd)) <= 1e-5 * np.max(np.abs(fd_a_nd))
    assert np.max(np.abs(b_nd - fd_b_nd)) <= 1e-5 * np.max(np.abs(fd_b_nd))
    # The affine term puts the linearization through the flown segment end at the node; the two
    # placements of the target differ by the integrator's 1e-10 m.
    end = implied_ends(a, b, c, nodes, controls)[k]
    assert np.max(np.abs(SCALE @ (end - segment_end(t, s0, q0, h, phase, j2)))) <= 1e-8


def test_two_body_jacobians_tend_to_cw_near_the_target_and_depart_from_it_at_400_m():
    # Independent computation: nondimensional distance from the CW step 4.8e-9 at 1 m and 1.9e-6
    # at 400 m, of order range / orbit radius (1.5e-7 and 5.8e-5).
    phi, gamma = cw_discrete(N, DT)
    inv = np.linalg.inv(SCALE)
    gap = {}
    for rho in (1.0, 400.0):
        nodes = np.array([[0.0, -rho, 0.0, 0.0, 0.0, 0.0]] * 2)
        a, b, _ = nmpc.shooting_linearization(TWO_BODY, 0.0, nodes, np.zeros((1, 3)), DT)
        gap[rho] = np.max(np.abs(SCALE @ (a[0] - phi) @ inv))
        assert np.max(np.abs(SCALE @ (b[0] - gamma))) * N**2 <= 1e-9
    assert gap[1.0] <= 1e-7
    assert gap[400.0] >= 5e-7


def test_circular_two_body_linearization_does_not_depend_on_the_target_phase():
    nodes = np.array([START_2KM, START_2KM])
    controls = np.array([[1e-4, -2e-4, 5e-5]])
    at_0 = nmpc.shooting_linearization(TWO_BODY, 50.0, nodes, controls, DT)
    at_2 = nmpc.shooting_linearization(
        TWO_BODY, 50.0, nodes, controls, DT, argument_of_latitude=2.0
    )
    inv = np.linalg.inv(SCALE)
    assert np.max(np.abs(SCALE @ (at_0[0][0] - at_2[0][0]) @ inv)) <= 1e-9
    assert np.max(np.abs(SCALE @ (at_0[1][0] - at_2[1][0]))) * N**2 <= 1e-9
    ends_0 = implied_ends(*at_0, nodes, controls)
    ends_2 = implied_ends(*at_2, nodes, controls)
    assert np.max(np.abs(SCALE @ (ends_0 - ends_2).T)) <= 1e-8


@pytest.mark.parametrize(
    "nodes, controls",
    [
        (np.zeros((3, 6)), np.zeros((3, 3))),
        (np.zeros((4, 6)), np.zeros((3, 2))),
        (np.zeros((1, 6)), np.zeros((0, 3))),
        (np.full((2, 6), np.nan), np.zeros((1, 3))),
    ],
)
def test_linearization_refuses_inconsistent_or_non_finite_plans(nodes, controls):
    with pytest.raises(ValueError):
        nmpc.shooting_linearization(TWO_BODY, 0.0, nodes, controls, DT)


def test_one_gauss_newton_step_from_the_cw_plan_closes_a_short_two_body_rendezvous():
    # Four 60 s segments from 2 km, minimum energy, to the origin at rest. Independent
    # computation: the CW plan flown on two-body misses by 0.017893 m and leaves 1.8e-3 of
    # continuity; one step leaves 1.2e-12 of continuity and an open-loop miss of 7.1e-11 m,
    # where B_k from the CW gamma leaves 8.1e-9 m.
    h = 60.0
    (s, q), (s1, q1) = one_iteration(START_2KM, 4, h)
    assert np.linalg.norm(fly(START_2KM, q, h)[:3]) == pytest.approx(0.017893, rel=1e-3)
    assert continuity(s1, q1, h) <= 1e-10
    assert np.linalg.norm(fly(START_2KM, q1, h)[:3]) <= 1e-9


@pytest.mark.slow
def test_one_gauss_newton_step_closes_the_half_orbit_two_body_rendezvous():
    # Twenty segments over half an orbit (141.924 s each), about 6 s of integration. Independent
    # computation: CW plan miss 3.990900 m; after one step continuity 1.0e-10 (the integrator's
    # floor, so the bound sits three times above it), open-loop miss 1.9e-8 m (1.7e-5 m with B_k
    # from the CW gamma), energy 0.07806% above the CW plan's.
    steps = 20
    h = math.pi / N / steps
    (s, q), (s1, q1) = one_iteration(START_2KM, steps, h)
    assert np.linalg.norm(fly(START_2KM, q, h)[:3]) == pytest.approx(3.9909, rel=1e-3)
    assert continuity(s1, q1, h) <= 3e-10
    assert np.linalg.norm(fly(START_2KM, q1, h)[:3]) <= 1e-7
    assert np.sum(q1**2) / np.sum(q**2) - 1.0 == pytest.approx(7.806e-4, rel=1e-3)


def test_config_is_frozen_and_keyword_only():
    with pytest.raises(TypeError):
        nmpc.NMPCConfig(MPCConfig())
    with pytest.raises(dataclasses.FrozenInstanceError):
        config().argument_of_latitude = 1.0
    assert isinstance(nmpc.NMPCConfig().mpc, MPCConfig)
    assert isinstance(nmpc.NMPCConfig().model, DeclaredModel)


@pytest.mark.parametrize("phase", [math.nan, math.inf])
def test_config_refuses_a_non_finite_phase(phase):
    with pytest.raises(ValueError):
        nmpc.NMPCConfig(argument_of_latitude=phase)


def test_unknown_declared_dynamics_is_refused_at_construction():
    with pytest.raises(ValueError, match="unknown dynamics"):
        DeclaredModel(dynamics="twobody_drag")


@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_discrete_and_impulsive_action_modes_are_refused_at_reset(mode):
    view = ApproachEnv(ApproachConfig(action_mode=mode)).agent_view()
    with pytest.raises(ValueError, match="takes no continuous command"):
        nmpc.NMPCAgent(config()).reset(view)


@pytest.mark.parametrize("task", ["dock", "inspect", "depart"])
def test_a_task_that_is_not_a_hold_is_refused_at_reset(task):
    with pytest.raises(ValueError, match=f"cannot fly the {task} task"):
        nmpc.NMPCAgent(config()).reset(hand_view(task=task))


def linear_mpc_move(view, cfg, x):
    """First move of the linear MPC of `vbar.agents.mpc` on the same view and settings."""
    model = regulator_model(view, cfg.mpc.regulator)
    steps = cfg.mpc.horizon
    reference = (np.tile(x, (steps + 1, 1)), np.zeros((steps, 3)))
    plan = mpc_solve(mpc_problem(model, view, x, reference, cfg.mpc), cfg.mpc.settings)
    assert plan.status == "converged"
    return plan.u[0]


def test_on_the_cw_model_every_decision_equals_the_linear_mpc_first_move():
    # Without a keep-out sphere the linear MPC does not depend on its reference plan, and on CW
    # the shooting constraints are its dynamics; both stop at the interior-point tolerance.
    view = hand_view(keep_out=None)
    cfg = config(model=CW, horizon=8)
    agent = nmpc.NMPCAgent(cfg)
    agent.reset(view)
    phi, gamma = cw_discrete(N, DT)
    x = np.array(BEHIND)
    saturated = 0
    for k in range(3):
        u = command(agent, view, x, k * DT)
        assert np.max(np.abs(u - linear_mpc_move(view, cfg, x))) <= 1e-6 * A_MAX
        saturated += abs(u[1]) >= A_MAX * (1 - 1e-6)
        entry = agent.log[-1]
        assert entry["level"] == 0
        # The shifted plan of a linear model is continuous up to rounding of 400 m states.
        assert entry["continuity"] <= 1e-9 * 400.0
        x = phi @ x + gamma @ u
        assert np.max(np.abs(entry["predicted"] - x)) <= 1e-9 * 400.0
    assert saturated >= 2


def test_near_the_hold_point_the_cw_command_is_the_lqr_law():
    view = hand_view(keep_out=None)
    agent = nmpc.NMPCAgent(config(model=CW, horizon=10))
    agent.reset(view)
    lqr = LQRAgent(LQRConfig())
    lqr.reset(view)
    x = np.array([*VBAR_HOLD, 0.0, 0.0, 0.0]) + np.array([0.5, -0.4, 0.3, 1e-4, -1e-4, 2e-4])
    assert np.max(np.abs(command(agent, view, x) - command(lqr, view, x))) <= 1e-6 * A_MAX


def failing_solver(status):
    def solve(qp, settings=None, warm=None):
        return HorizonSolution(
            x=None,
            u=None,
            status=status,
            iterations=60,
            objective=math.nan,
            residual_primal=1.0,
            residual_dual=1.0,
            gap=1.0,
            multipliers=None,
            conflicting=(),
            certificate=None,
            elapsed_s=0.0,
        )

    return solve


@pytest.mark.parametrize("status", ["iteration_cap", "solver_failed"])
def test_without_a_plan_a_failed_solve_falls_back_to_the_clipped_lqr_law(monkeypatch, status):
    view = hand_view(keep_out=None)
    agent = nmpc.NMPCAgent(config(model=CW, horizon=10))
    agent.reset(view)
    lqr = LQRAgent(LQRConfig())
    lqr.reset(view)
    monkeypatch.setattr(nmpc, "solve_horizon", failing_solver(status))
    x = np.array(BEHIND)
    u = command(agent, view, x)
    # 300 m from the hold point the LQR law asks for more than the along-track limit.
    assert np.max(np.abs(u - command(lqr, view, x))) <= 1e-12 * A_MAX
    assert abs(u[1]) == pytest.approx(A_MAX, rel=1e-12)
    assert agent.log[-1]["level"] == 2 and agent.log[-1]["fallback"] is True
    assert agent.fallbacks == [0, 0, 1]


@pytest.mark.parametrize("status", ["iteration_cap", "solver_failed"])
def test_with_a_plan_a_failed_solve_applies_the_shifted_plan(monkeypatch, status):
    view = hand_view(keep_out=None)
    cfg = config(model=CW, horizon=10)
    agent = nmpc.NMPCAgent(cfg)
    agent.reset(view)
    phi, gamma = cw_discrete(N, DT)
    x0 = np.array(BEHIND)
    u0 = command(agent, view, x0)
    # The linear MPC's second move, found independently; the shifted plan applies it next.
    model = regulator_model(view, cfg.mpc.regulator)
    reference = (np.tile(x0, (11, 1)), np.zeros((10, 3)))
    second = mpc_solve(mpc_problem(model, view, x0, reference, cfg.mpc)).u[1]
    monkeypatch.setattr(nmpc, "solve_horizon", failing_solver(status))
    x1 = phi @ x0 + gamma @ u0
    u1 = command(agent, view, x1, DT)
    assert np.max(np.abs(u1 - second)) <= 1e-6 * A_MAX
    assert [e["level"] for e in agent.log] == [0, 1]
    assert agent.log[-1]["fallback"] is True and agent.log[0]["fallback"] is False
    assert agent.fallbacks == [1, 1, 0]


def training_env(dynamics, seed=3, start=BEHIND):
    env = make("approach-vbar", dynamics=dynamics, training=True)
    obs, _ = env.reset(seed=seed, options={"start_state": start})
    return env, obs


def prediction_errors(agent, dynamics, decisions, seed=3):
    """Distances (m, m/s) of each decision's predicted next state from the next observation."""
    env, obs = training_env(dynamics, seed)
    agent.reset(env.agent_view(), agent_generator(seed + 1))
    errors = []
    for _ in range(decisions):
        action = agent.act(obs, env.agent_info())
        obs, *_ = env.step(action)
        gap = agent.log[-1]["predicted"] - obs[:6]
        errors.append((np.linalg.norm(gap[:3]), np.linalg.norm(gap[3:])))
    return np.array(errors), env


def test_matched_two_body_model_predicts_the_next_observation():
    # Three decisions of a two-step horizon on the two-body rung, about 1 s. Independent
    # computation: 1e-13 m at every decision; declared CW leaves 2.1e-6 m (next test).
    agent = nmpc.NMPCAgent(config(model=TWO_BODY, horizon=2))
    errors, _ = prediction_errors(agent, "twobody", 3)
    assert np.all(errors[:, 0] <= 1e-8)
    assert np.all(errors[:, 1] <= 1e-9)
    assert [e["level"] for e in agent.log] == [0, 0, 0]
    # A shifted plan of the linearized problem is continuous only to second order.
    assert all(0.0 < e["continuity"] <= 1e-4 for e in agent.log)


@pytest.mark.slow
def test_matched_two_body_model_predicts_the_next_observation_over_a_ten_step_horizon():
    # Five decisions of a ten-step horizon, about 7 s; independent computation 1e-13 m.
    agent = nmpc.NMPCAgent(config(model=TWO_BODY, horizon=10))
    errors, _ = prediction_errors(agent, "twobody", 5)
    assert np.all(errors[:, 0] <= 1e-8)
    assert [e["level"] for e in agent.log] == [0] * 5


def test_declared_cw_on_two_body_motion_mispredicts_by_the_gravity_remainder():
    # Independent computation: 2.1e-6 m at each of the first decisions from 400 m, the
    # second-order gravity terms CW drops (of order n^2 rho^2 / r dt^2 / 2 = 1.4e-6 m).
    agent = nmpc.NMPCAgent(config(model=CW, horizon=3))
    errors, _ = prediction_errors(agent, "twobody", 3)
    assert np.all(errors[:, 0] >= 5e-7)


def test_preparation_linearizes_at_the_shifted_plan_without_the_new_observation(monkeypatch):
    calls = []
    original = nmpc.shooting_linearization
    signature = inspect.signature(original)

    def recorded(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls[-1].append(
            {
                k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
                for k, v in bound.arguments.items()
            }
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(nmpc, "shooting_linearization", recorded)
    view = hand_view()
    phi, gamma = cw_discrete(N, DT)
    x0 = np.array(BEHIND)
    agents = []
    # The second observation differs by a metre between the two agents.
    for nudge in (0.0, 1.0):
        calls.append([])
        agent = nmpc.NMPCAgent(config(model=TWO_BODY, horizon=1))
        agent.reset(view)
        u0 = command(agent, view, x0)
        x1 = phi @ x0 + gamma @ u0 + np.array([nudge, 0.0, 0.0, 0.0, 0.0, 0.0])
        command(agent, view, x1, DT)
        agents.append(agent)
    plain, nudged = calls
    # One linearization per decision; one for the next decision may already have run.
    assert all(len(c) in (2, 3) for c in calls)
    for key in ("nodes", "controls"):
        assert np.array_equal(plain[1][key], nudged[1][key])
    assert plain[1]["t0"] == nudged[1]["t0"] == DT
    # The first node of the second linearization is the first plan's prediction, not x1.
    assert np.array_equal(plain[1]["nodes"][0], agents[0].log[0]["predicted"])
    log = agents[0].log
    assert all(e["prep_s"] > 0.0 and e["feedback_s"] > 0.0 for e in log)


def test_on_circular_two_body_the_command_does_not_depend_on_the_declared_phase():
    # Circular Keplerian relative motion has no preferred phase; independent computation: the
    # commands of phases 0 and 2 rad over three decisions of a five-step horizon differ by 1.5e-10
    # of the limit.
    view = hand_view()
    phi, gamma = cw_discrete(N, DT)
    runs = []
    for phase in (0.0, 2.0):
        agent = nmpc.NMPCAgent(config(model=TWO_BODY, horizon=1, phase=phase))
        agent.reset(view)
        x, us = np.array(BEHIND), []
        for k in range(2):
            us.append(command(agent, view, x, k * DT))
            x = phi @ x + gamma @ us[-1]
        runs.append(np.array(us))
        assert all(e["declared_phase"] is None for e in agent.log)
    assert np.max(np.abs(runs[0] - runs[1])) <= 1e-8 * A_MAX


def test_on_two_body_the_drawn_argument_of_latitude_does_not_change_the_commands():
    actions, draws = [], []
    for seed in (3, 4):
        env, obs = training_env("twobody", seed)
        draws.append(env.draw["argument_of_latitude"])
        agent = nmpc.NMPCAgent(config(model=TWO_BODY, horizon=1))
        agent.reset(env.agent_view(), agent_generator(20))
        run = []
        for _ in range(2):
            run.append(agent.act(obs, env.agent_info()))
            obs, *_ = env.step(run[-1])
        actions.append(np.array(run))
    assert draws[0] != draws[1]
    assert np.array_equal(actions[0], actions[1])


def test_on_j2_the_declared_phase_is_used_and_logged():
    # Independent computation: with the episode's drawn phase declared the prediction is good to
    # 1e-13 m; a quarter orbit off it misses by 1.2e-4 m, the J2 field seen from another latitude.
    env, _ = training_env("twobody_j2")
    drawn = float(env.draw["argument_of_latitude"])
    matched = nmpc.NMPCAgent(config(model=TWO_BODY_J2, horizon=1, phase=drawn))
    errors, _ = prediction_errors(matched, "twobody_j2", 2)
    assert np.all(errors[:, 0] <= 1e-8)
    assert all(e["declared_phase"] == drawn for e in matched.log)
    shifted = nmpc.NMPCAgent(config(model=TWO_BODY_J2, horizon=1, phase=drawn + math.pi / 2))
    errors, _ = prediction_errors(shifted, "twobody_j2", 2)
    assert np.all(errors[:, 0] >= 1e-5)
    assert all(e["declared_phase"] == drawn + math.pi / 2 for e in shifted.log)


def test_commands_are_equal_in_fractions_in_every_continuous_mode():
    views = {
        m: ApproachEnv(ApproachConfig(action_mode=m)).agent_view()
        for m in ("normalized", "accel", "thrust")
    }
    rng = np.random.default_rng(3)
    for _ in range(2):
        obs = np.array([*VBAR_HOLD, 0.0, 0.0, 0.0]) + rng.uniform(-1.0, 1.0, 6) * np.array(
            [100.0, 300.0, 50.0, 0.05, 0.05, 0.05]
        )
        fractions = {}
        for m, view in views.items():
            agent = nmpc.NMPCAgent(config(model=CW, horizon=5))
            agent.reset(view, np.random.default_rng(0))
            fractions[m] = command_fraction(view, agent.act(obs.copy(), dict(INFO)))
        ref = fractions["normalized"]
        assert np.max(np.abs(ref)) > 0.0
        for m in ("accel", "thrust"):
            assert np.max(np.abs(fractions[m] - ref)) <= 1e-12 * np.max(np.abs(ref))


A_START = (60.0, -300.0, -20.0, 0.01, 0.0, 0.0)


def actions(agent, start, steps):
    # The agent's declared two-body target is the state that must not leak between episodes or
    # agents; the CW rung keeps the environment's share of the time small.
    env = ApproachEnv(ApproachConfig(start_state=start, max_steps=steps))
    return run_episode(env, agent, seed=11, agent_seed=12)["actions"]


def test_reset_wipes_the_previous_episode():
    agent = nmpc.NMPCAgent(config(model=TWO_BODY, horizon=1))
    actions(agent, A_START, 2)
    reused = actions(agent, BEHIND, 2)
    fresh = actions(nmpc.NMPCAgent(config(model=TWO_BODY, horizon=1)), BEHIND, 2)
    assert np.array_equal(reused, fresh)
    assert len(agent.log) == 2 and sum(agent.fallbacks) == 2


def test_two_agents_from_one_config_on_interleaved_environments_match_separate_runs():
    cfg = config(model=TWO_BODY, horizon=1)
    starts = (A_START, BEHIND)
    envs = [ApproachEnv(ApproachConfig(start_state=s, max_steps=2)) for s in starts]
    agents = [nmpc.NMPCAgent(cfg), nmpc.NMPCAgent(cfg)]
    obs, logs = [], [[], []]
    for env, agent in zip(envs, agents, strict=True):
        o, _ = env.reset(seed=11)
        agent.reset(env.agent_view(), agent_generator(12))
        obs.append(o)
    for _ in range(2):
        for i, (env, agent) in enumerate(zip(envs, agents, strict=True)):
            a = agent.act(obs[i], env.agent_info())
            logs[i].append(a)
            obs[i], *_ = env.step(a)
    for i, start in enumerate(starts):
        alone = actions(nmpc.NMPCAgent(cfg), start, 2)
        assert np.array_equal(np.array(logs[i]), alone)


class ReadLog(dict):
    """An info mapping that records which keys are read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


def test_agent_runs_on_a_hand_made_view_and_reads_only_the_time():
    view = hand_view(mode="normalized")
    obs = np.array([5.0, -380.0, 3.0, 0.01, 0.0, -0.002])
    cfg = config(model=TWO_BODY_J2, horizon=1, phase=1.0)
    agent = nmpc.NMPCAgent(cfg)
    agent.reset(view)
    plain = agent.act(obs.copy(), {"time_s": 100.0})
    agent = nmpc.NMPCAgent(cfg)
    agent.reset(view)
    poisoned = ReadLog(time_s=100.0, true_state=np.full(6, 1e6), dv_used_mps=1e6)
    action = agent.act(obs.copy(), poisoned)
    assert poisoned.read <= {"time_s"}
    assert np.array_equal(action, plain)
    assert action.shape == (3,) and np.all(np.isfinite(action))


def held_objects(root, depth=4):
    """Every object reachable from `root` through attributes and containers, to `depth`."""
    seen, frontier = {id(root): root}, [root]
    for _ in range(depth):
        nxt = []
        for obj in frontier:
            if isinstance(obj, dict):
                children = list(obj.values())
            elif isinstance(obj, list | tuple | set | frozenset):
                children = list(obj)
            elif hasattr(obj, "__dict__") and not isinstance(obj, type):
                children = list(vars(obj).values())
            else:
                children = []
            for child in children:
                if id(child) not in seen and not isinstance(child, np.ndarray):
                    seen[id(child)] = child
                    nxt.append(child)
        frontier = nxt
    return list(seen.values())


def test_agent_holds_no_environment_object_and_replays_from_the_view_alone():
    cfg = config(model=TWO_BODY, horizon=1)
    env = ApproachEnv(
        ApproachConfig(start_state=BEHIND, max_steps=3, dynamics="twobody", inclination=INCLINATION)
    )
    o, _ = env.reset(seed=11)
    agent = nmpc.NMPCAgent(cfg)
    agent.reset(env.agent_view(), agent_generator(12))
    log = []
    for _ in range(3):
        info = env.agent_info()
        a = agent.act(o.copy(), info)
        log.append((o.copy(), info["time_s"], a))
        o, *_ = env.step(a)
    held = held_objects(agent)
    assert not any(isinstance(obj, gymnasium.Env) for obj in held)
    assert not any(obj is env.dynamics or obj is env.cfg for obj in held)
    replay = nmpc.NMPCAgent(cfg)
    replay.reset(dataclasses.replace(env.agent_view()))
    for obs, time_s, a in log:
        assert np.array_equal(replay.act(obs, {"time_s": time_s}), a)


def test_recorded_episode_keeps_one_log_entry_per_decision_with_valid_solver_fields():
    agent = nmpc.NMPCAgent(config(model=CW, horizon=5))
    ep = episode(agent, start_state=(0.0, -400.0, 30.0, 0.0, 0.0, 0.0), max_steps=10)
    assert len(ep.agent_log) == len(ep.records) == 10
    agent_metrics(ep)
    split = solver_metrics(ep)
    assert split["preparation_median"] > 0.0 and split["feedback_median"] > 0.0


def test_the_recorder_and_metrics_never_read_the_controller_log_fields():
    # The log is the agent's own bookkeeping; a scored count needs a channel the recorder trusts.
    package = Path(__file__).parents[1] / "src" / "vbar"
    for folder in ("metrics", "scenarios"):
        for source in (package / folder).rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            for key in ("fallbacks", "continuity", "declared_phase"):
                assert key not in text, (source, key)
