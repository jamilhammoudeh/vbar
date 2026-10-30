"""The hold-point MPC agent: its closed loop, warm starts, fallbacks, passing side and log.

`MPCAgent` solves the horizon problem of `mpc_problem` at each decision with `mpc_solve` and
applies the first move (Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed., sections
2.5 and 8.8). Where no row binds it is the LQR agent of the same regulator; an interior-point
solve starts from the previous plan shifted by one step with the terminal law appended; an
unconverged solve falls back to that shifted plan, or to the clipped LQR law when there is none;
at a start with the target in the way the passing side is chosen once per episode. References
marked "independent computation" come from LQR loops rolled out in the test or in a separate
script, from the LQR agent, or from a dense scan of the CW step, not from the agent under test.
Every case is for a 500 km circular orbit of the WGS 84 Earth, 10 s steps, a 0.02 m/s^2 limit per
axis and the 50 m keep-out sphere; the regulator weights are the defaults (10 m, 0.03 m/s, the
thrust limit, sampled cost) unless stated. Environment flights check the soft constraints at 2
samples per step, as in `test_lower_bound_check.py`; the keep-out minimum is refined in
continuous time either way. An interior-point decision takes 0.1 to 0.3 s on one machine, so
the fast tests fly at most a few such decisions and whole episodes are marked slow.
"""

import dataclasses
import functools
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from metric_cases import episode

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import mpc
from vbar.agents.base import AgentView, agent_generator, command_fraction
from vbar.agents.lqr import LQRAgent, LQRConfig, regulator_model
from vbar.agents.sign_check import sign_check
from vbar.constraints import KeepOut, segment_minimum
from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.metrics import record_episode
from vbar.metrics.agent import solver_metrics
from vbar.scenarios import SCENARIOS, make, seeds
from vbar.solvers import HorizonSolution, IPMSettings

MPCConfig = mpc.MPCConfig

N = mean_motion(EARTH.radius + 500e3)
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
KEEP_OUT = 50.0  # m
VBAR_HOLD = (0.0, -100.0, 0.0)
RBAR_HOLD = (-100.0, 0.0, 0.0)
# The radial input that holds a point 100 m below the target at rest, 3 n^2 100 m/s^2 (Clohessy-
# Wiltshire equilibrium); independent computation 3.6749087912e-4 m/s^2.
HOLD_INPUT = 3.0 * N**2 * 100.0
DISCRETE = LQRConfig(pos_scale_m=10.0, vel_scale_mps=0.01, stage_cost="discrete")
FAST = {"violation_samples": 2}
STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")
INFO = {"time_s": 0.0}
LOG_FIELDS = {
    "time_s",
    "status",
    "iterations",
    "elapsed_s",
    "level",
    "fallback",
    "path",
    "warm_start",
    "certified",
    "predicted_margin",
    "certified_bound",
    "max_slack",
    "side",
    "in_xf",
    "conflicting",
    "resolved",
}
# Starts on approach-rbar whose segment to the hold point passes 100 m from the target, so no
# passing side is contested. From FAR at rest the LQR asks for 0.145 m/s^2 radially, from MIXED
# for (0.073, -0.053, -0.007) m/s^2 and from DEEP for 0.218 m/s^2 radially, so the box binds
# (independent computation); with 10 steps of horizon the first 6, 3 and 9 decisions take the
# interior-point path.
FAR = (-500.0, 0.0, 0.0, 0.0, 0.0, 0.0)
MIXED = (-300.0, 150.0, 20.0, 0.0, 0.0, 0.0)
DEEP = (-700.0, 100.0, 20.0, 0.0, 0.0, 0.0)
# From rest here the unclipped LQR loop peaks at 0.008885 m/s^2 and stays 101.9 m or more from
# the target over 80 steps (independent computation): no row binds.
FREE = (-120.0, 25.0, 10.0, 0.0, 0.0, 0.0)


def make_agent(config=None, name=None):
    return mpc.MPCAgent(config) if name is None else mpc.MPCAgent(config, name=name)


def hand_view(mode="accel", limit="box", hold=VBAR_HOLD, dt=DT, task=None, keep_out=True):
    """An AgentView built without an environment."""
    goal = {"hold_point": hold, "capture_radius": 5.0, "capture_speed": 0.05}
    if task is not None:
        goal["task"] = task
    return AgentView(
        scenario_id="hand-made",
        dt=dt,
        max_steps=360,
        n=N,
        action_mode=mode,
        max_accel=A_MAX,
        thrust_limit=limit,
        mass_kg=500.0,
        goal=goal,
        constraints=({"type": "keep_out", "radius": KEEP_OUT},) if keep_out else (),
        observation=STATE_NAMES,
    )


def accel(view, action):
    """The commanded Hill acceleration (m/s^2) of an action."""
    return command_fraction(view, action) * view.max_accel


class Flight(NamedTuple):
    view: AgentView
    observations: np.ndarray
    actions: np.ndarray
    commands: np.ndarray  # m/s^2
    margins: np.ndarray  # continuous-time keep-out margin of each step (m)


def fly(agent, name, start, steps, seed=0, **config):
    """Fly `steps` decisions of a training environment from `start`."""
    env = make(name, training=True, **{**FAST, **config})
    view = env.agent_view()
    obs, _ = env.reset(seed=seed, options={"start_state": start})
    agent.reset(view, agent_generator(seed + 1))
    observations, issued, margins = [], [], []
    for _ in range(steps):
        observations.append(obs.copy())
        action = agent.act(obs, env.agent_info())
        issued.append(np.array(action, dtype=float))
        obs, _, terminated, truncated, _ = env.step(action)
        margins.append(env.record.segment.minima["keep_out"].margin)
        if terminated or truncated:
            break
    issued = np.array(issued)
    commands = np.array([accel(view, a) for a in issued])
    return Flight(view, np.array(observations), issued, commands, np.array(margins))


def lqr_command(view, obs, config=None):
    agent = LQRAgent(config)
    agent.reset(view, np.random.default_rng(0))
    return accel(view, agent.act(np.asarray(obs, dtype=float), dict(INFO)))


class Call(NamedTuple):
    decision: int
    qp: object
    settings: object
    warm: object
    solution: HorizonSolution


def scripted_solver(monkeypatch, outcomes, iterate=None):
    """Replace `solve_horizon` in the MPC module; `outcomes[d]` is the status to return at
    decision d, or None to solve for real. Decisions are told apart by the problem's initial
    state. With `iterate`, an unconverged outcome also carries that (x, u), which a solver must
    never return and an agent must never apply."""
    real = mpc.solve_horizon
    calls: list[Call] = []

    def solver(qp, settings=None, warm=None):
        decision = 0 if not calls else calls[-1].decision
        if calls and not np.array_equal(qp.x0, calls[-1].qp.x0):
            decision += 1
        status = outcomes[decision] if decision < len(outcomes) else None
        if status is None:
            solution = real(qp, settings=settings, warm=warm)
        else:
            x, u = (None, None) if iterate is None else iterate(qp)
            solution = HorizonSolution(
                x=x,
                u=u,
                status=status,
                iterations=7,
                objective=math.nan,
                residual_primal=1.0,
                residual_dual=1.0,
                gap=1.0,
                multipliers=None,
                elapsed_s=0.01,
            )
        calls.append(Call(decision, qp, settings, warm, solution))
        return solution

    monkeypatch.setattr(mpc, "solve_horizon", solver)
    return calls


def never_called(monkeypatch):
    def solver(qp, settings=None, warm=None):
        raise AssertionError("the interior-point solver was called")

    monkeypatch.setattr(mpc, "solve_horizon", solver)


# Construction


def test_agent_takes_its_config_and_name():
    agent = make_agent()
    assert agent.config == MPCConfig()
    assert agent.name == "mpc"
    config = MPCConfig(horizon=7)
    named = make_agent(config, name="mpc-7")
    assert named.config is config
    assert named.name == "mpc-7"


# Hold point and the LQR equivalence


def test_command_at_the_rbar_hold_point_is_the_radial_hold_input_on_the_riccati_path(monkeypatch):
    never_called(monkeypatch)
    agent = make_agent()
    flight = fly(agent, "approach-rbar", (*RBAR_HOLD, 0.0, 0.0, 0.0), 10)
    assert len(flight.commands) == 10
    for u in flight.commands:
        # The feedforward alone: rel 1e-10 of the hold input on every axis.
        assert np.max(np.abs(u - [HOLD_INPUT, 0.0, 0.0])) <= 1e-10 * HOLD_INPUT
    assert [e["path"] for e in agent.log] == ["riccati"] * 10
    assert [e["level"] for e in agent.log] == [0] * 10
    assert [e["iterations"] for e in agent.log] == [0] * 10
    x, u = agent.plan
    assert x.shape == (agent.config.horizon + 1, 6) and u.shape == (agent.config.horizon, 3)
    assert np.max(np.abs(x[:, :3] - RBAR_HOLD)) <= 1e-9
    assert np.max(np.abs(x[:, 3:])) <= 1e-11
    entry = agent.log[-1]
    # The plan sits 100 m from the target, 50 m outside the sphere, at rest.
    assert entry["predicted_margin"] == pytest.approx(50.0, abs=1e-9)
    assert 50.0 - 1e-3 <= entry["certified_bound"] <= 50.0 + 1e-9
    assert entry["certified"] is True
    assert entry["max_slack"] == 0.0
    assert entry["in_xf"] is True


def test_closed_loop_equals_the_lqr_agent_where_no_row_binds(monkeypatch):
    never_called(monkeypatch)
    agent = make_agent()
    flown = fly(agent, "approach-rbar", FREE, 15)
    reference = fly(LQRAgent(), "approach-rbar", FREE, 15)
    assert [e["path"] for e in agent.log] == ["riccati"] * 15
    scale = np.max(np.abs(reference.actions))
    # Same law from the same design; the Riccati plan and the gain round differently.
    assert np.max(np.abs(flown.actions - reference.actions)) <= 1e-10 * scale


DELAYED = {"delay_s": 10.0, "observation_channels": ("pending_commands",)}
# 150 m behind the hold point at rest; with the discrete weights and a one-step delay the
# unclipped LQR loop peaks at 0.014888 m/s^2 (independent DARE of the delayed model).
DELAY_START = (0.0, -250.0, 0.0, 0.0, 0.0, 0.0)
# The terminal set of the delay-augmented loop is not finitely determined within 200 steps, and
# the search there takes about 45 s on one machine; a cap of 0 refuses it at reset at once.
DELAYED_CONFIG = MPCConfig(regulator=DISCRETE, t_max=0)


@pytest.mark.filterwarnings("ignore:terminal set")
def test_delayed_closed_loop_equals_the_delay_augmented_lqr_agent(monkeypatch):
    never_called(monkeypatch)
    config = DELAYED_CONFIG
    agent = make_agent(config)
    flown = fly(agent, "approach-vbar", DELAY_START, 20, **DELAYED)
    reference = fly(LQRAgent(DISCRETE), "approach-vbar", DELAY_START, 20, **DELAYED)
    assert flown.view.observation[6:10] == (
        "pending_0_t",
        "pending_0_x",
        "pending_0_y",
        "pending_0_z",
    )
    assert [e["path"] for e in agent.log] == ["riccati"] * 20
    assert agent.plan[0].shape == (config.horizon + 1, 9)
    assert agent.log[0]["in_xf"] is None
    scale = np.max(np.abs(reference.actions))
    assert np.max(np.abs(flown.actions - reference.actions)) <= 1e-10 * scale


@pytest.mark.filterwarnings("ignore:terminal set")
def test_delay_off_the_step_grid_is_refused():
    # Nothing is pending at the first decision; the first command's 5 s wait shows at the second.
    with pytest.raises(ValueError, match="not aligned"):
        fly(
            make_agent(DELAYED_CONFIG),
            "approach-vbar",
            DELAY_START,
            2,
            delay_s=15.0,
            observation_channels=("pending_commands",),
        )


# Thrust limit and keep-out


def test_commands_stay_within_the_box_while_it_binds():
    agent = make_agent(MPCConfig(horizon=10))
    flight = fly(agent, "approach-rbar", FAR, 4)
    assert [e["path"] for e in agent.log] == ["ipm"] * 4
    assert [e["level"] for e in agent.log] == [0] * 4
    assert np.max(np.abs(flight.commands)) <= A_MAX * (1.0 + 1e-9)
    # The box binds: the radial command is at the limit.
    assert np.min(flight.commands[:, 0]) >= A_MAX * (1.0 - 1e-6)


def test_ball_limit_commands_stay_in_the_box_inscribed_in_the_ball():
    agent = make_agent(MPCConfig(horizon=10))
    flight = fly(agent, "approach-rbar", MIXED, 3, thrust_limit="ball")
    inscribed = A_MAX / math.sqrt(3.0)
    assert np.max(np.abs(flight.commands)) <= inscribed * (1.0 + 1e-9)
    assert np.max(np.abs(flight.commands)) >= inscribed * (1.0 - 1e-6)
    assert np.max(np.linalg.norm(flight.commands, axis=1)) <= A_MAX * (1.0 + 1e-12)


def plan_samples(view, x, u, samples):
    """Positions (N, samples, 3) of a plan's held steps at evenly spaced times within each."""
    phis, gammas = cw_discrete(view.n, np.linspace(0.0, view.dt, samples))
    r = np.einsum("jab,kb->kja", phis[:, :3, :6], x[:-1, :6])
    return r + np.einsum("jab,kb->kja", gammas[:, :3, :], u)


# Closing on the target from 80 m at 0.25 m/s: the clipped LQR plan of 20 steps dips to 1.669 m
# from the target centre (independent computation).
CLOSING = (3.0, 80.0, 0.0, 0.0, -0.25, 0.0)


def test_plan_from_a_closing_state_keeps_its_samples_outside_the_sphere():
    view = hand_view()
    agent = make_agent()
    agent.reset(view, np.random.default_rng(0))
    u0 = accel(view, agent.act(np.array(CLOSING), dict(INFO)))
    entry = agent.log[-1]
    assert (entry["status"], entry["path"], entry["level"]) == ("converged", "ipm", 0)
    assert entry["max_slack"] <= 1e-9
    x, u = agent.plan
    np.testing.assert_allclose(u0, u[0], rtol=0, atol=1e-12 * A_MAX)
    # Rows met with no slack put each of the row samples at the radius or beyond.
    ranges = np.linalg.norm(plan_samples(view, x, u, agent.config.row_samples), axis=-1)
    assert np.min(ranges) >= KEEP_OUT - 1e-6
    assert entry["predicted_margin"] == pytest.approx(np.min(ranges) - KEEP_OUT, abs=1e-6)
    # The held first step, scanned densely, is above the logged certified bound, which is >= 0.
    dense = np.linalg.norm(plan_samples(view, x[:2], u[:1], 2001), axis=-1)
    assert entry["certified"] is True
    assert 0.0 <= entry["certified_bound"] <= np.min(dense) - KEEP_OUT + 1e-9
    # Without the rows the same state is flown into the sphere.
    model = regulator_model(view, LQRConfig())
    xl = np.empty((21, 6))
    ul = np.empty((20, 3))
    xl[0] = CLOSING
    for k in range(20):
        ul[k] = np.clip(model.us - model.k @ (xl[k] - model.zs), -A_MAX, A_MAX)
        xl[k + 1] = model.phi @ xl[k] + model.gamma @ ul[k]
    assert np.min(np.linalg.norm(plan_samples(view, xl, ul, 401), axis=-1)) < KEEP_OUT


def held_step_bound(view, x, u):
    """`segment_minimum`'s certified lower bound on the keep-out margin of the held step (m), and
    the margins at its 20 samples and on a dense scan of 20001 (m)."""
    times = np.linspace(0.0, view.dt, 20)
    phis, gammas = cw_discrete(view.n, times)
    states = phis @ x + gammas @ u

    def state_at(tau):
        phi, gamma = cw_discrete(view.n, tau)
        return phi @ x + gamma @ u

    found = segment_minimum(
        KeepOut(KEEP_OUT), state_at, times, states, float(np.linalg.norm(u)), view.n
    )
    dense_phis, dense_gammas = cw_discrete(view.n, np.linspace(0.0, view.dt, 20001))
    dense = np.linalg.norm((dense_phis @ x + dense_gammas @ u)[:, :3], axis=1) - KEEP_OUT
    sampled = np.linalg.norm(states[:, :3], axis=1) - KEEP_OUT
    return float(found.lower_bound), sampled, dense


# 0.2 mm outside the sphere, passing it at 0.8 m/s 0.02 s before the closest approach. Every
# sample of the step, and the whole step, stay outside (at least 0.20 mm), but the bound between
# samples 0.53 s apart, (V h)^2 / (8 R) + A h^2 / 8, is larger than that margin, so the
# certified lower bound is -0.19 mm: the step cannot be certified.
GRAZING = (50.0002, 0.8 * 0.02, 0.0, 0.0, -0.8, 0.0)


def test_a_step_clear_at_every_sample_but_not_certified_between_them_is_solved_again_and_logged(
    monkeypatch,
):
    solves = []
    real = mpc.mpc_solve

    def recording(qp, settings=None, warm=None):
        plan = real(qp, settings=settings, warm=warm)
        solves.append((qp, plan))
        return plan

    monkeypatch.setattr(mpc, "mpc_solve", recording)
    view = hand_view()
    agent = make_agent(MPCConfig(side="+x"))
    agent.reset(view, np.random.default_rng(0))
    x = np.array(GRAZING)
    u = accel(view, agent.act(x.copy(), dict(INFO)))
    entry = agent.log[-1]
    assert len(solves) == 2
    (first_qp, first), (second_qp, second) = solves
    assert first.status == second.status == "converged"
    # The first plan's move fails the certificate, so the first step's keep-out rows are raised
    # by the shortfall plus the 1 mm resolution of the keep-out check, and nothing else changes.
    shortfall = -held_step_bound(view, x, np.clip(first.u[0], -A_MAX, A_MAX))[0]
    assert shortfall > 0.0
    raised = first_qp.e.copy()
    live = raised[0, 7:] > 0.0
    assert np.any(live)
    raised[0, 7:][live] += shortfall + KeepOut.resolution
    np.testing.assert_allclose(second_qp.e, raised, rtol=1e-15, atol=1e-12)
    for name in ("x0", "Q", "q", "P", "p", "C"):
        np.testing.assert_array_equal(getattr(second_qp, name), getattr(first_qp, name))
    # The second plan is applied, and its step is certified again.
    np.testing.assert_allclose(u, second.u[0], rtol=0, atol=1e-12 * A_MAX)
    bound, sampled, dense = held_step_bound(view, x, u)
    assert np.min(sampled) >= 0.0 and np.min(dense) >= 0.0
    assert bound < 0.0
    assert entry["certified_bound"] == pytest.approx(bound, abs=1e-12)
    assert entry["resolved"] is True
    assert entry["iterations"] == first.iterations + second.iterations
    assert (entry["level"], entry["path"]) == (0, "ipm")
    # No slack: the bound alone refuses the certificate.
    assert entry["max_slack"] <= 1e-6
    assert entry["certified"] is False


# Terminal set membership and refusals


def test_terminal_state_inside_and_outside_the_terminal_set_is_logged():
    agent = make_agent(MPCConfig(horizon=10))
    fly(agent, "approach-rbar", (*RBAR_HOLD, 0.0, 0.0, 0.0), 1)
    assert agent.log[-1]["in_xf"] is True
    # From 400 m beyond the hold point 100 s of thrust do not reach the set.
    fly(agent, "approach-rbar", FAR, 1)
    assert agent.log[-1]["in_xf"] is False


def test_terminal_set_failure_warns_with_terminal_cost_and_raises_with_terminal_set():
    # With the discrete weights at 60 s steps the set's allowance between samples, about 103 m,
    # leaves no room between the sphere and the hold point.
    view = hand_view(dt=60.0)
    cost = make_agent(MPCConfig(regulator=DISCRETE, horizon=5))
    with pytest.warns(UserWarning, match="terminal set"):
        cost.reset(view, np.random.default_rng(0))
    cost.act(np.array([0.0, -130.0, 0.0, 0.0, 0.0, 0.0]), dict(INFO))
    assert cost.log[-1]["in_xf"] is None
    hard = make_agent(MPCConfig(regulator=DISCRETE, horizon=5, terminal="set"))
    with pytest.raises(ValueError, match="terminal set"):
        hard.reset(view, np.random.default_rng(0))


def test_row_samples_too_few_for_the_step_are_refused_at_reset():
    view = hand_view(dt=1e4)
    with pytest.raises(ValueError, match="row_samples too few"):
        make_agent(MPCConfig(regulator=DISCRETE, horizon=5)).reset(view, np.random.default_rng(0))


# Warm starts and fallbacks


def lifted(x, radius=KEEP_OUT):
    """Reference states after the first moved radially out to 1.1 R when nearer than that."""
    x = x.copy()
    size = np.linalg.norm(x[1:, :3], axis=1)
    near = size < 1.1 * radius
    x[1:][near, :3] *= (1.1 * radius / size[near])[:, None]
    return x


def test_interior_point_solves_start_from_the_shifted_plan(monkeypatch):
    calls = scripted_solver(monkeypatch, [])
    agent = make_agent(MPCConfig(horizon=10))
    flight = fly(agent, "approach-rbar", DEEP, 4)
    model = regulator_model(flight.view, agent.config.regulator)
    assert [c.decision for c in calls] == [0, 1, 2, 3]
    assert calls[0].warm is None
    assert [e["warm_start"] for e in agent.log] == [False, True, True, True]
    for previous, call in zip(calls, calls[1:], strict=False):
        x_prev, u_prev = previous.solution.x, previous.solution.u
        z0 = call.qp.x0
        # Independent shift: the previous moves from the second on, then the terminal law at the
        # previous plan's end clipped to the box, flown from the observed state on the model.
        end = np.clip(model.us - model.k @ (x_prev[-1] - model.zs), -A_MAX, A_MAX)
        u_shift = np.vstack([u_prev[1:, :3], end])
        x_shift = np.empty((11, 6))
        x_shift[0] = z0
        for k in range(10):
            x_shift[k + 1] = model.phi @ x_shift[k] + model.gamma @ u_shift[k]
        x_warm, u_warm = call.warm
        assert x_warm.shape == (11, 6) and u_warm.shape == (10, 4)
        np.testing.assert_allclose(u_warm[:, :3], u_shift, rtol=0, atol=1e-12 * A_MAX)
        np.testing.assert_allclose(x_warm, x_shift, rtol=1e-12, atol=1e-9)
        # The slack input in its offset form, 1 m, plus 1 mm of slack so the start is interior.
        np.testing.assert_allclose(u_warm[:, 3], 1.001, rtol=1e-15, atol=0)
        # CW flown exactly: the shifted states agree with the previous plan to rounding.
        np.testing.assert_allclose(x_warm[:10], x_prev[1:], rtol=0, atol=1e-6)
        # The problem solved is the one linearized at the lifted shifted plan.
        expected = mpc.mpc_problem(model, flight.view, z0, (lifted(x_shift), u_shift), agent.config)
        for name in ("x0", "A", "B", "c", "Q", "q", "P", "p", "C", "e"):
            np.testing.assert_allclose(
                getattr(call.qp, name),
                getattr(expected, name),
                rtol=1e-12,
                atol=1e-12,
                err_msg=name,
            )
        assert call.qp.C_N is None and expected.C_N is None


def garbage_iterate(qp):
    steps, nz, nu = qp.B.shape
    return np.full((steps + 1, nz), 1e3), np.full((steps, nu), 1e3)


@pytest.mark.parametrize("iterate", [None, garbage_iterate], ids=["no-iterate", "iterate"])
def test_fallback_levels_follow_the_solver_outcomes_and_never_apply_an_unconverged_iterate(
    monkeypatch, iterate
):
    outcomes = ["iteration_cap", None, "iteration_cap", "solver_failed", None, "solver_failed"]
    calls = scripted_solver(monkeypatch, outcomes, iterate)
    agent = make_agent(MPCConfig(horizon=10))
    flight = fly(agent, "approach-rbar", DEEP, 6)
    assert [c.decision for c in calls] == list(range(6))
    expected = [
        "iteration_cap",
        "converged",
        "iteration_cap",
        "solver_failed",
        "converged",
        "solver_failed",
    ]
    assert [e["status"] for e in agent.log] == expected
    assert [e["level"] for e in agent.log] == [2, 0, 1, 1, 0, 1]
    assert [e["fallback"] for e in agent.log] == [True, False, True, True, False, True]
    assert [e["path"] for e in agent.log] == ["ipm"] * 6
    assert [e["iterations"] for e in agent.log] == [c.solution.iterations for c in calls]
    assert [agent.log[d]["iterations"] for d in (0, 2, 3, 5)] == [7, 7, 7, 7]
    assert agent.fallbacks == [2, 3, 1]
    tol = 1e-12 * A_MAX
    # No plan yet: the clipped LQR law of the same regulator.
    np.testing.assert_allclose(
        flight.commands[0], lqr_command(flight.view, flight.observations[0]), rtol=0, atol=tol
    )
    # A plan exists: its next moves, one per failed decision.
    first, second = calls[1].solution.u, calls[4].solution.u
    for k, move in ((1, first[0]), (2, first[1]), (3, first[2]), (4, second[0]), (5, second[1])):
        np.testing.assert_allclose(flight.commands[k], move[:3], rtol=0, atol=tol)
    assert np.max(np.abs(flight.commands)) <= A_MAX * (1.0 + 1e-9)


def test_solver_settings_of_the_config_reach_every_solve_and_no_plan_means_the_lqr_law(
    monkeypatch,
):
    calls = scripted_solver(monkeypatch, [])
    settings = IPMSettings(max_iter=1)
    agent = make_agent(MPCConfig(horizon=10, settings=settings))
    flight = fly(agent, "approach-rbar", FAR, 3)
    assert len(calls) == 3 and all(c.settings is settings for c in calls)
    assert [e["status"] for e in agent.log] == ["iteration_cap"] * 3
    assert [e["level"] for e in agent.log] == [2, 2, 2]
    assert agent.fallbacks == [0, 0, 3]
    assert agent.plan is None
    for obs, u in zip(flight.observations, flight.commands, strict=True):
        np.testing.assert_allclose(u, lqr_command(flight.view, obs), rtol=0, atol=1e-12 * A_MAX)


def test_terminal_set_out_of_reach_falls_back_to_the_lqr_law_with_the_certificate_logged():
    # From 500 m behind the hold point the box cannot reach the terminal set in 100 s.
    agent = make_agent(MPCConfig(horizon=10, terminal="set"))
    flight = fly(agent, "approach-vbar", (0.0, 400.0, 0.0, 0.0, 0.0, 0.0), 1)
    entry = agent.log[0]
    assert (entry["status"], entry["level"]) == ("infeasible_certified", 2)
    assert any(stage == 10 for stage, _ in entry["conflicting"])
    assert agent.fallbacks == [0, 0, 1]
    np.testing.assert_allclose(
        flight.commands[0],
        lqr_command(flight.view, flight.observations[0]),
        rtol=0,
        atol=1e-12 * A_MAX,
    )


# Passing side. On approach-hbar the start (0, 0, 150) m at rest is symmetric under the mirror
# (x, y) -> (-x, -y), which maps the CW equations, the cost, the box and the sphere onto
# themselves and the +x pass onto the -x pass, so only the navigation noise picks a side. On the
# V-bar the pair is not symmetric (Coriolis), and at 300 m with 10 steps no keep-out row binds
# within the horizon, so both sides give the same plan (independent computation).
SYMMETRIC = (0.0, 0.0, 150.0, 0.0, 0.0, 0.0)
SIDE_STEPS = 4


@functools.cache
def side_flight(seed):
    """Observations and logged sides of a committed agent on approach-hbar with noisy navigation.

    One interior-point decision takes about 0.2 s and the first makes two, so a flight of four
    takes 1 to 2 s; the side claim needs several decisions of one episode.
    """
    agent = make_agent()
    flight = fly(agent, "approach-hbar:nav1", SYMMETRIC, SIDE_STEPS, seed=seed)
    return flight.view, flight.observations, tuple(e["side"] for e in agent.log)


@pytest.mark.parametrize("seed", [0, 1])
def test_auto_side_is_committed_for_the_episode(seed):
    _, _, sides = side_flight(seed)
    assert len(sides) == SIDE_STEPS
    assert sides[0] in ("+x", "-x")
    assert set(sides) == {sides[0]}


def test_choosing_the_side_afresh_at_each_decision_would_change_it():
    # The mutant: a fresh agent at every observation of the committed flight. About 1.5 s.
    view, observations, _ = side_flight(0)
    fresh = []
    for k, obs in enumerate(observations[:3]):
        agent = make_agent()
        agent.reset(view, np.random.default_rng(0))
        agent.act(obs, {"time_s": DT * k})
        fresh.append(agent.log[-1]["side"])
    assert len(set(fresh)) == 2


MIRROR_A = np.array([1.0, 2.0, 150.0, 1e-3, 0.0, -1e-3])
MIRROR_B = MIRROR_A * [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0]


def test_mirrored_states_choose_mirrored_sides():
    view = make("approach-hbar").agent_view()
    sides = []
    for obs in (MIRROR_A, MIRROR_B):
        agent = make_agent()
        agent.reset(view, np.random.default_rng(0))
        agent.act(obs.copy(), dict(INFO))
        sides.append(agent.log[-1]["side"])
    assert sorted(sides) == ["+x", "-x"]


def test_fixed_sides_at_mirrored_states_give_mirrored_commands_with_one_solve_each(monkeypatch):
    calls = scripted_solver(monkeypatch, [])
    view = make("approach-hbar").agent_view()
    commands = []
    for obs, side in ((MIRROR_A, "+x"), (MIRROR_B, "-x")):
        agent = make_agent(MPCConfig(side=side))
        agent.reset(view, np.random.default_rng(0))
        commands.append(accel(view, agent.act(obs.copy(), dict(INFO))))
        assert agent.log[-1]["side"] == side
    assert len(calls) == 2
    mirrored = commands[1] * [-1.0, -1.0, 1.0]
    # Both solves meet the interior-point tolerance, 1e-8 of the problem's scale.
    np.testing.assert_allclose(commands[0], mirrored, rtol=0, atol=1e-6 * A_MAX)


def test_side_is_chosen_again_only_after_the_hysteresis_count_of_failed_decisions(monkeypatch):
    # Failed decisions 1, 2 (two in a row) and 4, 5, 6 (three, the default hysteresis): only
    # decision 7 compares both sides again. Perfect navigation, just off the symmetric start.
    fail = "iteration_cap"
    calls = scripted_solver(monkeypatch, [None, fail, fail, None, fail, fail, fail, None])
    agent = make_agent()
    fly(agent, "approach-hbar", (1.0, 0.0, 150.0, 0.0, 0.0, 0.0), 8)
    per_decision = [sum(c.decision == d for c in calls) for d in range(8)]
    assert per_decision == [2, 1, 1, 1, 1, 1, 1, 2]
    assert [e["level"] for e in agent.log] == [0, 1, 1, 0, 1, 1, 1, 0]


# Log


def test_every_decision_is_logged_with_its_fields_and_fallbacks_count_the_levels():
    # From 180 m below the target the box binds first and then lets go, so both paths occur.
    agent = make_agent(MPCConfig(horizon=10))
    fly(agent, "approach-rbar", (-180.0, 0.0, 0.0, 0.0, 0.0, 0.0), 12)
    log = agent.log
    assert len(log) == 12
    assert {e["path"] for e in log} == {"riccati", "ipm"}
    for k, entry in enumerate(log):
        assert set(entry) >= LOG_FIELDS
        assert entry["time_s"] == pytest.approx(DT * k, abs=1e-9)
        assert entry["status"] == "converged"
        assert entry["level"] == 0 and entry["fallback"] is False
        assert isinstance(entry["iterations"], int) and entry["iterations"] >= 0
        assert (entry["iterations"] == 0) == (entry["path"] == "riccati")
        assert entry["elapsed_s"] > 0.0
        assert entry["warm_start"] is (entry["path"] == "ipm" and k > 0 and log[k - 1]["level"] < 2)
        assert isinstance(entry["certified"], bool)
        assert math.isfinite(entry["predicted_margin"]) and math.isfinite(entry["certified_bound"])
        assert entry["max_slack"] >= 0.0
        assert entry["side"] is None
        assert entry["conflicting"] == ()
        assert entry["resolved"] is False
    assert agent.fallbacks == [12, 0, 0]
    assert sum(agent.fallbacks) == len(log)


def test_recorded_episode_keeps_one_log_entry_per_decision_and_the_solver_fields_are_read():
    ep = episode(
        make_agent(MPCConfig(horizon=10)),
        start_state=(60.0, -300.0, -20.0, 0.01, 0.0, 0.0),
        max_steps=6,
    )
    assert len(ep.agent_log) == len(ep.records) == 6
    solver = solver_metrics(ep)
    assert 0.0 <= solver["warm_start_rate"] <= 1.0
    assert solver["agent_fallback_count"] == 0.0
    assert math.isfinite(solver["solver_iterations_median"])


# Sign check


@pytest.mark.filterwarnings("ignore:terminal set")
def test_mpc_passes_the_sign_check_with_the_lqr_first_commands():
    # 50 m from rest the unclipped LQR loop peaks at 0.017906 m/s^2 and stays 254 m or more from
    # the target (independent computation), so each first command is the LQR one.
    # A cap of 0 skips building six terminal sets, which the first commands do not use.
    check = sign_check(make_agent(MPCConfig(t_max=0)), ApproachConfig())
    reference = sign_check(LQRAgent(), ApproachConfig())
    assert check.passed
    assert check.commands.keys() == reference.commands.keys()
    for label, fractions in check.commands.items():
        expected = np.array(reference.commands[label])
        assert np.max(np.abs(np.array(fractions) - expected)) <= 1e-10 * np.max(np.abs(expected))


# Agent protocol


@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_discrete_and_impulsive_action_modes_are_refused_at_reset(mode):
    view = ApproachEnv(ApproachConfig(action_mode=mode)).agent_view()
    with pytest.raises(ValueError, match="takes no continuous command"):
        make_agent().reset(view, np.random.default_rng(0))


@pytest.mark.parametrize("task", ["dock", "inspect", "depart"])
def test_a_task_that_is_not_a_hold_is_refused_at_reset(task):
    with pytest.raises(ValueError, match=f"cannot fly the {task} task"):
        make_agent().reset(hand_view(task=task), np.random.default_rng(0))


def test_a_hold_goal_without_a_hold_point_is_refused_at_reset():
    view = dataclasses.replace(hand_view(), goal={"capture_radius": 5.0})
    with pytest.raises(ValueError, match="this goal has none"):
        make_agent().reset(view, np.random.default_rng(0))


def test_commands_are_equal_in_fractions_in_every_continuous_mode():
    views = {
        m: make("approach-rbar", action_mode=m).agent_view()
        for m in ("normalized", "accel", "thrust")
    }
    # One state on the Riccati path, one where the box binds.
    for x in (FREE, MIXED):
        fractions = {}
        for m, view in views.items():
            agent = make_agent(MPCConfig(horizon=10))
            agent.reset(view, np.random.default_rng(0))
            fractions[m] = command_fraction(view, agent.act(np.array(x), dict(INFO)))
        ref = fractions["normalized"]
        assert np.max(np.abs(ref)) > 0.0
        for m in ("accel", "thrust"):
            assert np.max(np.abs(fractions[m] - ref)) <= 1e-12 * np.max(np.abs(ref))


# Two short V-bar episodes: A starts 200 m off the hold point, where the box binds; B starts
# 150 m beyond the target, which is in the way, so a side is chosen and a plan carried. Ten
# steps of horizon keep each decision near 0.1 s.
A_START = (60.0, -300.0, -20.0, 0.01, 0.0, 0.0)
B_START = (0.5, 150.0, 0.0, 0.0, 0.0, 0.0)
STEPS = 3
SHORT = MPCConfig(horizon=10)


def actions(agent, start):
    env = ApproachEnv(ApproachConfig(start_state=start, max_steps=STEPS, **FAST))
    return run_episode(env, agent, seed=11, agent_seed=12)["actions"]


@functools.cache
def fresh_b_actions():
    return np.array(actions(make_agent(SHORT), B_START))


def test_reset_wipes_the_previous_episode():
    agent = make_agent(SHORT)
    actions(agent, A_START)
    assert agent.plan is not None
    reused = actions(agent, B_START)
    assert np.array_equal(np.array(reused), fresh_b_actions())
    assert len(agent.log) == STEPS
    assert agent.log[0]["warm_start"] is False
    assert agent.log[0]["side"] in ("+x", "-x")
    assert sum(agent.fallbacks) == STEPS


def test_two_agents_from_one_config_on_interleaved_environments_match_separate_runs():
    starts = (A_START, B_START)
    envs = [ApproachEnv(ApproachConfig(start_state=s, max_steps=STEPS, **FAST)) for s in starts]
    agents = [make_agent(SHORT) for _ in starts]
    obs, logs = [], [[], []]
    for env, agent in zip(envs, agents, strict=True):
        o, _ = env.reset(seed=11)
        agent.reset(env.agent_view(), agent_generator(12))
        obs.append(o)
    for _ in range(STEPS):
        for i, (env, agent) in enumerate(zip(envs, agents, strict=True)):
            a = agent.act(obs[i], env.agent_info())
            logs[i].append(a)
            obs[i], *_ = env.step(a)
    assert np.array_equal(np.array(logs[1]), fresh_b_actions())
    alone = actions(make_agent(SHORT), A_START)
    assert np.array_equal(np.array(logs[0]), np.array(alone))


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
    view = hand_view(mode="normalized", hold=RBAR_HOLD)
    obs = np.array(MIXED)
    agent = make_agent(SHORT)
    agent.reset(view)
    plain = agent.act(obs.copy(), {"time_s": 100.0})
    agent.reset(view)
    poisoned = ReadLog(time_s=100.0, true_state=np.full(6, 1e6), dv_used_mps=1e6)
    action = agent.act(obs.copy(), poisoned)
    assert poisoned.read <= {"time_s"}
    assert np.array_equal(action, plain)
    assert action.shape == (3,) and np.all(np.isfinite(action))
    assert agent.log[-1]["time_s"] == 100.0


def test_the_recorder_and_metrics_never_read_the_controller_log_fields():
    # The log is the agent's own bookkeeping; a scored count needs a channel the recorder trusts.
    package = Path(__file__).parents[1] / "src" / "vbar"
    for folder in ("metrics", "scenarios"):
        for source in (package / folder).rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            for key in ("fallbacks", "certified_bound", "max_slack", "predicted_margin", "in_xf"):
                assert key not in text, (source, key)


# Whole episodes


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, max_steps",
    [("approach-rbar", 210), ("approach-vbar", 360), ("approach-hbar", 360)],
)
def test_mpc_completes_the_dwell_without_entering_the_keep_out_sphere(name, max_steps):
    # approach-rbar is cut to 2100 s, by which the LQR agent at the same weights completes the
    # dwell; on approach-vbar and approach-hbar the target is in the way and the LQR agent from
    # the same start enters the sphere. 10 to 40 s each on one machine.
    results = {}
    for label, agent in (("mpc", make_agent()), ("lqr", LQRAgent())):
        env = make(name, max_steps=max_steps, **FAST)
        obs, _ = env.reset(seed=0)
        agent.reset(env.agent_view(), agent_generator(1))
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(agent.act(obs, env.agent_info()))
            done = terminated or truncated
        results[label] = info
    assert results["mpc"]["success"]
    assert results["mpc"]["min_range_to_target_m"] >= KEEP_OUT
    if name != "approach-rbar":
        assert results["lqr"]["min_range_to_target_m"] < KEEP_OUT


@pytest.mark.slow
def test_pass_by_the_target_stays_outside_the_sphere_in_continuous_time_and_certifies_each_step():
    # Fifteen interior-point decisions near the sphere, about 13 s; the environment checks every
    # constraint at its 20 samples per step here.
    agent = make_agent(MPCConfig(horizon=10))
    flight = fly(agent, "approach-vbar", CLOSING, 15, violation_samples=20)
    reference = fly(LQRAgent(), "approach-vbar", CLOSING, 15, violation_samples=20)
    assert np.min(reference.margins) < 0.0
    assert np.min(flight.margins) >= 0.0
    for continuous, entry in zip(flight.margins, agent.log, strict=True):
        # Sound, and within the 1 mm resolution of the refined minimum.
        assert continuous >= entry["certified_bound"] - 1e-9
        assert continuous - entry["certified_bound"] <= 2e-3
        if entry["certified"]:
            assert entry["certified_bound"] >= 0.0


# Raw 1 m position noise (:nav1) on a controller that plans to ride the sphere at 50.4 m puts the
# agent inside it on most seeds, so that variant is left out of the claim.
SPHERE_CASES = [
    pytest.param(f"{base}:{variant}" if variant else base, index)
    for base in ("approach-vbar", "approach-hbar")
    for variant in (None, *SCENARIOS[base].variants)
    if variant != "nav1"
    for index in range(3)
]


@pytest.mark.slow
@pytest.mark.parametrize("name, index", SPHERE_CASES)
def test_mpc_completes_the_dwell_outside_the_sphere_on_every_variant_without_raw_noise(name, index):
    # The variants come from the registry and the seeds are the first three validation seeds, as
    # in the scenario audit.
    env = make(name, **FAST)
    seed, agent_seed = seeds(name, "validation", 3).pairs()[index]
    record_episode(env, make_agent(), seed=seed, agent_seed=agent_seed)
    info = env._info()
    assert info["success"], info
    assert not info["keep_out_violated"]
    assert info["min_range_to_target_m"] >= KEEP_OUT
