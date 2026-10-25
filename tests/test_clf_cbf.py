"""CLF-CBF QP controller: a relaxed Lyapunov row toward the hold point and a hard keep-out barrier.

The command minimizes 1/2 |u - us|^2 + (w / 2) delta^2 subject to the control Lyapunov row
dV/dt <= -decay V + delta, V = e'Pe with P the continuous Riccati solution of the regulator
weights, and the exponential keep-out barrier h = rho - R_f with real poles (p1, p2) as a hard
row (Ames, Coogan, Egerstedt, Notomista, Sreenath and Tabuada, "Control barrier functions:
theory and applications", ECC 2019, sections II.C and IV; the roots of the characteristic
polynomial are -p1 and -p2). References marked "independent computation" come from SLSQP on the
four variables (u, delta), from a continuous Riccati solve in the test, or from the barrier row
written out in the test, not from the module under test. CW cases are for a 500 km circular
orbit of the WGS 84 Earth, a 10 s step and a 0.02 m/s^2 thrust limit.
"""

import dataclasses
import functools
from pathlib import Path

import numpy as np
import pytest
from metric_cases import episode
from scipy.linalg import eigh, solve_continuous_are
from scipy.optimize import minimize

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import clf_cbf
from vbar.agents.base import AgentView, agent_generator, command_fraction
from vbar.agents.lqr import LQRConfig
from vbar.dynamics import EARTH, mean_motion
from vbar.scenarios import make
from vbar.solvers import ActuatorSet, Projection, Rows, project

N = mean_motion(EARTH.radius + 500e3)
DT = 10.0
A_MAX = 0.02  # m/s^2
KEEP_OUT = 50.0  # m
VBAR_HOLD = (0.0, -100.0, 0.0)
RBAR_HOLD = (-100.0, 0.0, 0.0)
HBAR_HOLD = (0.0, 0.0, -100.0)
INFO = {"time_s": 0.0}
B = np.vstack([np.zeros((3, 3)), np.eye(3)])
# Default LQRConfig weights: 10 m, 0.03 m/s, and the thrust limit as the acceleration scale.
QC = np.diag([10.0**-2] * 3 + [0.03**-2] * 3)
RC = np.eye(3) / A_MAX**2


def cw_a(n):
    a = np.zeros((6, 6))
    a[:3, 3:] = np.eye(3)
    a[3, 0] = 3 * n * n
    a[3, 4] = 2 * n
    a[4, 3] = -2 * n
    a[5, 2] = -n * n
    return a


@functools.cache
def riccati():
    """Continuous Riccati solution P and M = Qc + P B Rc^-1 B'P, by an independent solve."""
    p = solve_continuous_are(cw_a(N), B, QC, RC)
    return p, QC + p @ B @ np.linalg.solve(RC, B.T @ p)


def hold_input(hold):
    """The continuous input that holds a point at rest: -3 n^2 x radially, n^2 z normally."""
    return np.array([-3 * N * N * hold[0], 0.0, N * N * hold[2]])


def barrier_row(x, radius, poles, n=N):
    """r_hat . u >= b for h = rho - radius with h'' + (p1 + p2) h' + p1 p2 h >= 0 on CW."""
    r, v = np.asarray(x[:3], dtype=float), np.asarray(x[3:6], dtype=float)
    rho = np.linalg.norm(r)
    r_hat = r / rho
    rho_dot = r_hat @ v
    natural = np.array([3 * n * n * x[0] + 2 * n * x[4], -2 * n * x[3], -n * n * x[2]])
    p1, p2 = poles
    b = (
        -(v @ v - rho_dot**2) / rho
        - r_hat @ natural
        - (p1 + p2) * rho_dot
        - p1 * p2 * (rho - radius)
    )
    return r_hat, b


def hand_view(mode="accel", limit="box", hold=VBAR_HOLD, task=None):
    """An AgentView built without an environment."""
    goal = {"hold_point": hold, "capture_radius": 5.0, "capture_speed": 0.05}
    if task is not None:
        goal["task"] = task
    return AgentView(
        scenario_id="hand-made",
        dt=DT,
        max_steps=1000,
        n=N,
        action_mode=mode,
        max_accel=A_MAX,
        thrust_limit=limit,
        mass_kg=500.0,
        goal=goal,
        constraints=({"type": "keep_out", "radius": KEEP_OUT},),
    )


def command(agent, view, x):
    return command_fraction(view, agent.act(np.asarray(x, dtype=float), dict(INFO))) * A_MAX


def test_config_is_frozen_and_keyword_only():
    with pytest.raises(TypeError):
        clf_cbf.ClfCbfConfig(LQRConfig())
    with pytest.raises(dataclasses.FrozenInstanceError):
        clf_cbf.ClfCbfConfig().buffer_m = 2.0


def test_default_poles_buffer_and_clf_weight_are_the_ones_chosen_for_capture():
    # Ours, from a closed-loop sweep on approach-hbar: p_max dt = 0.5 at dt = 10 s stays below
    # 0.618, where a double-pole barrier under a held command first loses invariance on a double
    # integrator, and a weight of 1e-2 captured 10 of 10 dispersed starts where 1e-8 captured
    # none of 3.
    config = clf_cbf.ClfCbfConfig()
    assert config.poles == (0.02, 0.05)
    assert config.buffer_m == 1.0
    assert config.clf_weight == 1e-2
    assert config.regulator == LQRConfig()


@pytest.mark.parametrize(
    "fields",
    [
        {"poles": (0.0, 0.05)},
        {"poles": (-0.02, 0.05)},
        {"poles": (0.02, float("inf"))},
        {"poles": (0.02,)},
        {"buffer_m": -0.1},
        {"buffer_m": float("nan")},
        {"clf_weight": 0.0},
        {"clf_weight": -1.0},
    ],
)
def test_config_refuses_poles_that_are_not_positive_and_a_negative_buffer_or_weight(fields):
    with pytest.raises(ValueError):
        clf_cbf.ClfCbfConfig(**fields)


def test_clf_matrix_is_the_continuous_riccati_solution_of_the_regulator_weights():
    design = clf_cbf.clf_design(hand_view(), LQRConfig())
    p, _ = riccati()
    # Independent computation; two Riccati solves of the same pair agree near 1e-13 of P.
    assert np.max(np.abs(design.p - p)) <= 1e-9 * np.max(np.abs(p))
    assert np.array_equal(design.xs, np.array([*VBAR_HOLD, 0.0, 0.0, 0.0]))
    assert np.max(np.abs(design.us - hold_input(VBAR_HOLD))) <= 1e-15


def test_decay_rate_is_the_smallest_generalized_eigenvalue_of_the_closed_loop_dissipation():
    # dV/dt = -e'Me under the continuous LQR, so the largest rate it meets at every deviation
    # is the smallest eigenvalue of M e = lambda P e: 4.486407652117e-3 1/s, a time constant of
    # 223 s. lambda_min(M) / lambda_max(P) = 8.94e-6 1/s also holds but is 500 times looser.
    design = clf_cbf.clf_design(hand_view(), LQRConfig())
    p, m = riccati()
    assert design.decay == pytest.approx(eigh(m, p, eigvals_only=True).min(), rel=1e-9)
    assert design.decay == pytest.approx(4.486407652117e-3, rel=1e-9)


def deviations(count=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(count, 6)) * np.array([200.0, 200.0, 50.0, 0.1, 0.1, 0.1])


def test_clf_row_normal_is_twice_the_velocity_rows_of_p_times_the_deviation():
    design = clf_cbf.clf_design(hand_view(), LQRConfig())
    p, _ = riccati()
    xs = design.xs
    for e in deviations(20, seed=1):
        g, _ = clf_cbf.clf_row(design, xs + e)
        ref = 2 * (p @ e)[3:]
        assert np.max(np.abs(g - ref)) <= 1e-9 * np.max(np.abs(ref))


def test_continuous_lqr_command_meets_the_clf_row_without_relaxation_at_100_deviations():
    design = clf_cbf.clf_design(hand_view(), LQRConfig())
    p, m = riccati()
    for e in deviations():
        x = design.xs + e
        u = design.us - np.linalg.solve(RC, B.T @ p @ e)
        g, bound = clf_cbf.clf_row(design, x)
        # The slack is e'Me - decay e'Pe, at least 0.0034 of decay V on these draws.
        scale = abs(bound) + abs(g @ u) + design.decay * (e @ p @ e)
        assert bound - g @ u >= -1e-12 * scale


def test_one_and_a_half_times_the_decay_rate_rejects_the_lqr_command_at_some_deviation():
    # The rate is tight: 69 of the 100 draws fail at 1.5 times it (independent computation).
    design = clf_cbf.clf_design(hand_view(), LQRConfig())
    p, _ = riccati()
    failed = 0
    for e in deviations():
        u = design.us - np.linalg.solve(RC, B.T @ p @ e)
        g, bound = clf_cbf.clf_row(design, design.xs + e)
        failed += bound - 0.5 * design.decay * (e @ p @ e) - g @ u < 0.0
    assert failed >= 1


def test_clf_row_at_the_hold_point_at_rest_is_empty():
    design = clf_cbf.clf_design(hand_view(), LQRConfig())
    g, bound = clf_cbf.clf_row(design, design.xs)
    assert np.array_equal(g, np.zeros(3)) and bound == 0.0


def random_problem(rng, layout):
    """A command, a CLF row, 0 to 3 barrier rows through a feasible point, and a weight."""
    k = int(rng.integers(0, 4))
    if layout == "box":
        u_t = rng.uniform(-0.5, 0.5, 3) * A_MAX
    else:
        direction = rng.normal(size=3)
        u_t = direction / np.linalg.norm(direction) * rng.uniform(0, 0.5) * A_MAX
    a = rng.normal(size=(k, 3))
    a /= np.linalg.norm(a, axis=1)[:, None]
    through = rng.random(k) < 0.8
    b = a @ u_t - np.where(through, 0.0, rng.uniform(0.1, 0.5, k) * A_MAX)
    u_ref = u_t - (rng.uniform(0.0, 0.5, k) * through) @ a * A_MAX if k else u_t.copy()
    u_ref = u_ref + rng.normal(size=3) * 0.1 * A_MAX
    g = rng.normal(size=3)
    g /= np.linalg.norm(g)
    bound = g @ u_ref - rng.uniform(-0.2, 0.6) * A_MAX
    w = 10 ** rng.uniform(-1, 1)
    rows = Rows(a, b, tuple(f"barrier_{i}" for i in range(k)))
    return u_ref, g, bound, rows, ActuatorSet(A_MAX, layout), w


def four_variable_qp(u_ref, g, bound, rows, actuator, w):
    """min 1/2 |u - u_ref|^2 + w/2 delta^2, g.u <= bound + delta, rows, limit; by SLSQP."""
    s = actuator.a_max

    def cost(z):
        du = z[:3] - u_ref / s
        return 0.5 * du @ du + 0.5 * w * z[3] ** 2, np.r_[du, w * z[3]]

    cons = [
        {
            "type": "ineq",
            "fun": lambda z: bound / s + z[3] - g @ z[:3],
            "jac": lambda z: np.r_[-g, 1.0],
        }
    ]
    for a_i, b_i in zip(rows.a, rows.b, strict=True):
        cons.append(
            {
                "type": "ineq",
                "fun": lambda z, a_i=a_i, b_i=b_i: a_i @ z[:3] - b_i / s,
                "jac": lambda z, a_i=a_i: np.r_[a_i, 0.0],
            }
        )
    bounds = None
    if actuator.layout == "box":
        bounds = [(-1.0, 1.0)] * 3 + [(None, None)]
    else:
        cons.append(
            {
                "type": "ineq",
                "fun": lambda z: 1.0 - z[:3] @ z[:3],
                "jac": lambda z: np.r_[-2 * z[:3], 0.0],
            }
        )
    z0 = np.r_[np.clip(u_ref / s, -0.5, 0.5), 0.0]
    found = minimize(
        cost,
        z0,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-16, "maxiter": 500},
    )
    return found.x[:3] * s, found.x[3] * s


SEEDS = {"box": 12, "ball": 13}


@functools.cache
def problems(layout):
    rng = np.random.default_rng(SEEDS[layout])
    return tuple(random_problem(rng, layout) for _ in range(100))


@functools.cache
def solutions(layout, half):
    """The module's solve of each problem in one half, shared by the checks on it."""
    return tuple(
        clf_cbf.clf_cbf_qp(u_ref, g, bound, rows, actuator, clf_weight=w)
        for u_ref, g, bound, rows, actuator, w in problems(layout)[50 * half : 50 * (half + 1)]
    )


@pytest.mark.parametrize("layout", ["box", "ball"])
@pytest.mark.parametrize("half", [0, 1])
def test_dualized_solve_equals_the_four_variable_qp_at_100_problems(layout, half):
    # Fifty problems per test keep each under 2 s.
    part = problems(layout)[50 * half : 50 * (half + 1)]
    for problem, solution in zip(part, solutions(layout, half), strict=True):
        u_ref, g, bound, rows, actuator, w = problem
        u_qp, delta_qp = four_variable_qp(u_ref, g, bound, rows, actuator, w)
        # Independent computation: SLSQP meets an exact active-set enumeration to 5e-9 a_max.
        assert solution.status == "feasible"
        assert np.max(np.abs(solution.u - u_qp)) <= 1e-6 * A_MAX
        assert abs(solution.delta - delta_qp) <= 1e-6 * A_MAX


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_barrier_rows_stay_hard_while_the_clf_row_relaxes(layout):
    active_rows, relaxed = set(), 0
    every = solutions(layout, 0) + solutions(layout, 1)
    for problem, solution in zip(problems(layout), every, strict=True):
        _, g, bound, rows, _, w = problem
        assert np.all(rows.a @ solution.u >= rows.b - 1e-10 * A_MAX)
        assert g @ solution.u <= bound + solution.delta + 1e-10 * A_MAX
        assert solution.delta >= 0.0
        assert solution.multiplier == pytest.approx(w * solution.delta, rel=1e-9, abs=1e-15)
        if layout == "box":
            assert np.max(np.abs(solution.u)) <= A_MAX * (1 + 1e-12)
        else:
            assert np.linalg.norm(solution.u) <= A_MAX * (1 + 1e-12)
        active_rows.add(sum(name.startswith("barrier") for name in solution.projection.active))
        relaxed += solution.delta > 0.0
    # The draws give 0, 1, 2 and 3 active barrier rows, and the CLF row binds in 77 (box) and
    # 74 (ball) of them.
    assert active_rows == {0, 1, 2, 3}
    assert relaxed >= 50


def test_an_inactive_clf_row_and_no_rows_return_the_reference_command():
    u_ref = np.array([1e-3, -2e-3, 5e-3])
    g = np.array([0.0, 0.0, 1.0])
    rows = Rows(np.zeros((0, 3)), np.zeros(0), ())
    solution = clf_cbf.clf_cbf_qp(u_ref, g, 1.0, rows, ActuatorSet(A_MAX, "box"), clf_weight=1.0)
    assert np.max(np.abs(solution.u - u_ref)) <= 1e-15
    assert solution.delta == 0.0 and solution.multiplier == 0.0
    assert solution.status == "feasible"


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_barrier_rows_in_conflict_with_the_limit_give_the_least_violation_command(layout):
    # u_z >= 0.03 cannot hold at 0.02 m/s^2; the CLF row is dropped and the command is the
    # projection's least-violation point nearest the reference.
    u_ref = np.array([1e-3, -2e-3, 0.0])
    rows = Rows(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), np.array([0.03, -1.0]), ("a", "b"))
    actuator = ActuatorSet(A_MAX, layout)
    g = np.array([1.0, 0.0, 0.0])
    solution = clf_cbf.clf_cbf_qp(u_ref, g, -1.0, rows, actuator, clf_weight=1.0)
    reference = project(u_ref, rows, actuator)
    assert reference.status == "infeasible_certified"
    assert solution.status == "infeasible_certified"
    assert solution.projection.status == "infeasible_certified"
    assert "a" in solution.projection.conflicting
    assert np.max(np.abs(solution.u - reference.u)) <= 1e-12 * A_MAX
    assert solution.multiplier == 0.0
    # The relaxation reported is the CLF row's violation at the command applied.
    assert solution.delta == pytest.approx(max(0.0, g @ solution.u + 1.0), rel=1e-12)


@pytest.mark.parametrize("hold", [VBAR_HOLD, RBAR_HOLD, HBAR_HOLD])
def test_command_at_the_hold_point_at_rest_is_the_hold_input(hold):
    view = hand_view(hold=hold)
    agent = clf_cbf.ClfCbfAgent()
    agent.reset(view, np.random.default_rng(0))
    u = command(agent, view, [*hold, 0.0, 0.0, 0.0])
    # 3 n^2 100 = 3.6749087912e-4 m/s^2 radially for the R-bar hold, n^2 100 normally for H-bar.
    ref = hold_input(hold)
    assert np.max(np.abs(u - ref)) <= 1e-12 * max(np.max(np.abs(ref)), 1e-6)
    entry = agent.log[-1]
    assert entry["status"] == "feasible"
    assert entry["level"] == 0
    assert entry["cbf_active"] is False
    assert entry["delta"] == 0.0
    assert agent.fallbacks == [1, 0, 0]


@pytest.mark.parametrize("limit", ["box", "ball"])
def test_a_closing_state_the_limit_cannot_stop_gives_the_least_violation_command_and_logs_it(limit):
    # 5 m outside the 51 m barrier radius, closing at 1 m/s along the orbit normal: the row asks
    # for 6.607e-2 m/s^2 outward, past 0.02. The least-violation command is full thrust outward.
    view = hand_view(limit=limit, hold=HBAR_HOLD)
    agent = clf_cbf.ClfCbfAgent()
    agent.reset(view, np.random.default_rng(0))
    x = np.array([0.0, 0.0, 55.0, 0.0, 0.0, -1.0])
    _, b = barrier_row(x, KEEP_OUT + 1.0, (0.02, 0.05))
    assert b == pytest.approx(6.606737e-2, rel=1e-6)
    u = command(agent, view, x)
    assert np.max(np.abs(u - np.array([0.0, 0.0, A_MAX]))) <= 1e-12 * A_MAX
    assert agent.log[-1]["status"] == "infeasible_certified"
    assert agent.log[-1]["level"] == 1
    assert agent.fallbacks == [0, 1, 0]


def test_a_failed_projection_is_logged_and_its_command_applied(monkeypatch):
    failed = Projection(
        u=np.array([0.0, 1e-3, 0.0]),
        status="solver_failed",
        active=(),
        multipliers=np.zeros(1),
        objective=0.0,
        conflicting=(),
        certificate=None,
        phase1_margin=0.0,
    )
    monkeypatch.setattr(clf_cbf, "project", lambda *args, **kwargs: failed)
    view = hand_view()
    agent = clf_cbf.ClfCbfAgent()
    agent.reset(view, np.random.default_rng(0))
    u = command(agent, view, [0.0, 80.0, 0.0, 0.0, -0.2, 0.0])
    assert np.max(np.abs(u - failed.u)) <= 1e-15
    assert agent.log[-1]["status"] == "solver_failed"
    assert agent.log[-1]["level"] == 2
    assert agent.fallbacks == [0, 0, 1]


def failing_after_first_call(monkeypatch, u_failed):
    """Stub the module's `project`: exact on the first call, "solver_failed" on every later one."""
    calls = []

    def stub(u_nom, rows, actuator=None, weight=None):
        calls.append(np.array(u_nom, dtype=float))
        if len(calls) == 1:
            return project(u_nom, rows, actuator, weight)
        return Projection(
            u=np.array(u_failed),
            status="solver_failed",
            active=(),
            multipliers=np.zeros(len(rows.b)),
            objective=0.0,
            conflicting=(),
            certificate=None,
            phase1_margin=0.0,
        )

    monkeypatch.setattr(clf_cbf, "project", stub)
    return calls


def test_a_projection_that_fails_during_the_root_search_is_reported_not_hidden(monkeypatch):
    # The start projection is feasible and violates the Lyapunov row, so the search must project
    # again; that projection fails, and its status and command are what the solve returns.
    u_failed = np.array([0.0, 1e-3, -2e-3])
    calls = failing_after_first_call(monkeypatch, u_failed)
    u_ref = np.array([1e-3, 0.0, 0.0])
    g = np.array([1.0, 0.0, 0.0])
    rows = Rows(np.array([[0.0, 0.0, 1.0]]), np.array([-1.0]), ("keep_out",))
    solution = clf_cbf.clf_cbf_qp(u_ref, g, 0.0, rows, ActuatorSet(A_MAX, "box"), clf_weight=1.0)
    assert len(calls) >= 2
    assert solution.status == "solver_failed"
    assert solution.projection.status == "solver_failed"
    assert np.array_equal(solution.u, u_failed)
    assert solution.multiplier == 0.0


def test_an_agent_whose_projection_fails_during_the_root_search_logs_level_2(monkeypatch):
    view = hand_view()
    agent = clf_cbf.ClfCbfAgent()
    agent.reset(view, np.random.default_rng(0))
    u_failed = np.array([0.0, 1e-3, 0.0])
    calls = failing_after_first_call(monkeypatch, u_failed)
    # 280 m from the hold point at rest the hold input alone does not meet the Lyapunov row.
    u = command(agent, view, [5.0, -380.0, 3.0, 0.0, 0.0, 0.0])
    assert len(calls) >= 2
    assert np.max(np.abs(u - u_failed)) <= 1e-15
    entry = agent.log[-1]
    assert entry["status"] == "solver_failed" and entry["level"] == 2
    assert agent.fallbacks == [0, 0, 1]


def test_a_rate_weight_is_refused_by_the_design():
    # The CLF is designed on the six-state model; a rate weight needs the last command as state.
    with pytest.raises(ValueError, match="takes no rate weight"):
        clf_cbf.clf_design(hand_view(), LQRConfig(rate_weight=0.1))
    agent = clf_cbf.ClfCbfAgent(clf_cbf.ClfCbfConfig(regulator=LQRConfig(rate_weight=0.1)))
    with pytest.raises(ValueError, match="takes no rate weight"):
        agent.reset(hand_view())


def test_an_observation_with_pending_commands_is_refused_by_the_design():
    pending = tuple(f"pending_0_{c}" for c in "txyz")
    view = dataclasses.replace(hand_view(), observation=hand_view().observation + pending)
    with pytest.raises(ValueError, match="does not model pending commands"):
        clf_cbf.clf_design(view, LQRConfig())
    with pytest.raises(ValueError, match="does not model pending commands"):
        clf_cbf.ClfCbfAgent().reset(view)


def test_a_riccati_solve_that_raises_is_refused(monkeypatch):
    def singular(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(clf_cbf, "solve_continuous_are", singular)
    with pytest.raises(ValueError, match="riccati solution did not converge"):
        clf_cbf.clf_design(hand_view(), LQRConfig())


def test_a_riccati_solution_with_a_large_residual_is_refused(monkeypatch):
    # The identity is no solution of the scaled CW equation; its residual is of order 1.
    monkeypatch.setattr(clf_cbf, "solve_continuous_are", lambda *args, **kwargs: np.eye(6))
    with pytest.raises(ValueError, match="riccati solution did not converge"):
        clf_cbf.clf_design(hand_view(), LQRConfig())


def clf_only_command(view, x):
    """The command of the Lyapunov row alone under the view's limit, from the module's parts."""
    design = clf_cbf.clf_design(view, LQRConfig())
    g, bound = clf_cbf.clf_row(design, x)
    none = Rows(np.zeros((0, 3)), np.zeros(0), ())
    actuator = ActuatorSet(view.max_accel, view.thrust_limit)
    return clf_cbf.clf_cbf_qp(design.us, g, bound, none, actuator, clf_weight=1e-2).u


def test_a_view_without_a_keep_out_constraint_runs_the_lyapunov_row_alone():
    view = dataclasses.replace(hand_view(), constraints=())
    agent = clf_cbf.ClfCbfAgent()
    agent.reset(view)
    guarded = clf_cbf.ClfCbfAgent()
    guarded.reset(hand_view())
    # Far out, next to the barrier, and closing on it at 1 m/s.
    states = (
        [5.0, -380.0, 3.0, 0.01, 0.0, -0.002],
        [30.0, 45.0, 0.0, -0.2, -0.3, 0.0],
        [0.0, 55.0, 0.0, 0.0, -1.0, 0.0],
    )
    differs = 0
    for x in states:
        u = command(agent, view, x)
        assert np.max(np.abs(u - clf_only_command(view, x))) <= 1e-12 * A_MAX
        differs += np.max(np.abs(u - command(guarded, hand_view(), x))) > 1e-6 * A_MAX
    assert all(e["cbf_active"] is False for e in agent.log)
    assert all(e["held_bound"] is None for e in agent.log)
    assert {e["level"] for e in agent.log} == {0}
    # The barrier changes the command next to the target, so the rows really were absent.
    assert differs >= 1


def test_at_the_target_centre_there_is_no_barrier_row():
    # rho = 0 gives the barrier no direction; the command is the Lyapunov row's alone.
    view = hand_view(hold=HBAR_HOLD)
    agent = clf_cbf.ClfCbfAgent()
    agent.reset(view)
    x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    u = command(agent, view, x)
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u - clf_only_command(view, x))) <= 1e-12 * A_MAX
    entry = agent.log[-1]
    assert entry["cbf_active"] is False and entry["status"] == "feasible" and entry["level"] == 0
    # Inside the keep-out sphere the held step's certified margin is negative.
    assert entry["held_bound"] < 0.0


@functools.cache
def flown(name, start, steps, seed):
    """Per decision: the observed state, the command (m/s^2), the log entry, and the step's
    continuous and sampled keep-out minima from the environment (m)."""
    env = make(name, training=True)
    view = env.agent_view()
    agent = clf_cbf.ClfCbfAgent()
    obs, _ = env.reset(seed=seed, options={"start_state": start})
    agent.reset(view, agent_generator(seed + 1))
    steps_flown = []
    for _ in range(steps):
        x = obs[:6].copy()
        action = agent.act(obs, env.agent_info())
        u = command_fraction(view, action) * view.max_accel
        obs, _, terminated, truncated, _ = env.step(action)
        segment = env.record.segment
        sampled = float(np.min(np.linalg.norm(segment.states[:, :3], axis=1))) - KEEP_OUT
        steps_flown.append((x, u, dict(agent.log[-1]), segment.minima["keep_out"].margin, sampled))
        if terminated or truncated:
            break
    return tuple(steps_flown)


HEAD_ON = ("approach-hbar", (0.0, 0.0, 300.0, 0.0, 0.0, 0.0), 30)
# A pass whose closest approach falls between the environment's samples with the barrier active.
PASS = ("approach-hbar", (30.0, 0.0, 45.0, 0.0, 0.2, -0.3), 15)


@pytest.mark.parametrize("seed", [0, 1])
def test_head_on_hbar_approach_brakes_on_the_barrier_from_decision_20_as_recorded(seed):
    # Reference closed loop: every projection feasible, the barrier row active from decision
    # 20 to 29, the lowest continuous margin 11.341355447 m at the last step, no capture.
    steps = flown(*HEAD_ON, seed)
    assert len(steps) == 30
    assert {s[2]["status"] for s in steps} == {"feasible"}
    assert {s[2]["level"] for s in steps} == {0}
    active = [k for k, s in enumerate(steps) if s[2]["cbf_active"]]
    assert active == list(range(20, 30))
    assert min(s[3] for s in steps) == pytest.approx(11.341355447, abs=1e-6)


@pytest.mark.parametrize("case, seed", [(HEAD_ON, 0), (HEAD_ON, 1), (PASS, 0)])
def test_logged_held_step_bound_is_below_the_continuous_minimum_and_within_2_mm_of_it(case, seed):
    for _, _, entry, continuous, _ in flown(*case, seed):
        # Sound: the certified bound never exceeds the minimum over continuous time.
        assert continuous >= entry["held_bound"] - 1e-9
        # Tight: the reference certificate over 20 samples sits at most 1.1e-3 m below it.
        assert continuous - entry["held_bound"] <= 2e-3


def test_pass_keeps_the_certified_margin_between_samples_while_the_barrier_is_active():
    steps = flown(*PASS, 0)
    assert all(s[2]["cbf_active"] for s in steps)
    # Reference closed loop: the lowest continuous margin is 0.48936 m, and on some steps the
    # minimum falls up to 7.2e-5 m below the lowest of the environment's 20 samples, so a
    # sampled minimum logged in place of the bound would fail the soundness check.
    assert min(s[3] for s in steps) == pytest.approx(0.48936, abs=1e-5)
    assert max(s[4] - s[3] for s in steps) > 1e-6
    for _, _, entry, continuous, _ in steps:
        assert entry["held_bound"] >= 0.0
        assert continuous >= entry["held_bound"] - 1e-9


@pytest.mark.parametrize("case, seed", [(HEAD_ON, 0), (PASS, 0)])
def test_every_feasible_command_meets_the_barrier_row(case, seed):
    for x, u, entry, _, _ in flown(*case, seed):
        a, b = barrier_row(x, KEEP_OUT + 1.0, (0.02, 0.05))
        assert entry["status"] == "feasible"
        # Independent computation of the row; it holds to the projection's 1e-10 of a_max.
        assert a @ u >= b - 1e-9 * A_MAX
        if entry["cbf_active"]:
            assert abs(a @ u - b) <= 1e-9 * A_MAX


@pytest.mark.slow
def test_whole_head_on_episode_stalls_at_the_buffer_without_capture():
    # Exactly on the orbit normal the barrier correction is collinear with the CLF command and
    # CW gives no lateral push, so the chaser rides the barrier: the reference closed loop
    # stays 1.0 m outside the keep-out for the whole 360 decisions. About 40 s.
    steps = flown("approach-hbar", (0.0, 0.0, 300.0, 0.0, 0.0, 0.0), 360, 0)
    assert len(steps) == 360
    assert min(s[3] for s in steps) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.slow
def test_dispersed_hbar_episode_is_captured():
    # The reference closed loop captured 10 of 10 dispersed starts (seeds 1 to 10), seed 1 after
    # 234 decisions with the lowest margin 20.07 m. About 25 s.
    env = make("approach-hbar")
    agent = clf_cbf.ClfCbfAgent()
    obs, _ = env.reset(seed=1)
    agent.reset(env.agent_view(), agent_generator(2))
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(agent.act(obs, env.agent_info()))
        done = terminated or truncated
    assert info["success"]
    assert info["min_range_to_target_m"] >= KEEP_OUT


def test_a_ball_limit_is_flown_on_the_ball():
    view = hand_view(limit="ball")
    agent = clf_cbf.ClfCbfAgent()
    agent.reset(view, np.random.default_rng(0))
    u = command(agent, view, [80.0, 300.0, -60.0, 0.0, 0.0, 0.0])
    assert np.linalg.norm(u) <= A_MAX * (1 + 1e-12)
    assert np.linalg.norm(u) >= 0.5 * A_MAX


@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_discrete_and_impulsive_action_modes_are_refused_at_reset(mode):
    view = ApproachEnv(ApproachConfig(action_mode=mode)).agent_view()
    with pytest.raises(ValueError, match="takes no continuous command"):
        clf_cbf.ClfCbfAgent().reset(view, np.random.default_rng(0))


def hold_view(**goal):
    """The approach-hbar view with its goal entries replaced; None removes an entry."""
    view = make("approach-hbar").agent_view()
    merged = {**view.goal, **goal}
    return dataclasses.replace(view, goal={k: v for k, v in merged.items() if v is not None})


@pytest.mark.parametrize("task", ["dock", "inspect", "depart"])
def test_a_task_that_is_not_a_hold_is_refused_at_reset(task):
    with pytest.raises(ValueError, match=f"cannot fly the {task} task"):
        clf_cbf.ClfCbfAgent().reset(hold_view(task=task), np.random.default_rng(0))


def test_a_hold_goal_without_a_hold_point_is_refused_at_reset():
    with pytest.raises(ValueError, match="this goal has none"):
        clf_cbf.ClfCbfAgent().reset(hold_view(hold_point=None), np.random.default_rng(0))


def test_commands_are_equal_in_fractions_in_every_continuous_mode():
    views = {
        m: ApproachEnv(ApproachConfig(action_mode=m)).agent_view()
        for m in ("normalized", "accel", "thrust")
    }
    # Far from the target, next to the barrier, and closing on it.
    states = (
        [5.0, -380.0, 3.0, 0.01, 0.0, -0.002],
        [30.0, 45.0, 0.0, -0.2, -0.3, 0.0],
        [0.0, 60.0, 5.0, 0.0, -0.4, 0.0],
    )
    for x in states:
        obs = np.array(x)
        fractions = {}
        for m, view in views.items():
            agent = clf_cbf.ClfCbfAgent()
            agent.reset(view, np.random.default_rng(0))
            fractions[m] = command_fraction(view, agent.act(obs.copy(), dict(INFO)))
        ref = fractions["normalized"]
        assert np.max(np.abs(ref)) > 0.0
        for m in ("accel", "thrust"):
            assert np.max(np.abs(fractions[m] - ref)) <= 1e-12 * np.max(np.abs(ref))


# Both starts reach the barrier within the 10 steps; each decision, a root search over
# projections and the held-step certificate, took 16 to 19 ms on one machine at full CPU load,
# so the interleaving check, four runs, stays well under 2 s.
A_START = (0.0, 70.0, 0.0, 0.0, -0.3, 0.0)
B_START = (20.0, 50.0, 10.0, -0.1, -0.4, 0.0)
STEPS = 10


def actions(agent, start):
    env = ApproachEnv(ApproachConfig(start_state=start, max_steps=STEPS))
    return run_episode(env, agent, seed=11, agent_seed=12)["actions"]


def test_reset_wipes_the_previous_episode():
    agent = clf_cbf.ClfCbfAgent()
    actions(agent, A_START)
    reused = actions(agent, B_START)
    fresh = actions(clf_cbf.ClfCbfAgent(), B_START)
    assert np.array_equal(reused, fresh)
    assert len(agent.log) == STEPS
    assert any(e["cbf_active"] for e in agent.log)
    first = clf_cbf.ClfCbfAgent()
    actions(first, A_START)
    assert any(e["cbf_active"] for e in first.log)


def test_two_agents_from_one_config_on_interleaved_environments_match_separate_runs():
    config = clf_cbf.ClfCbfConfig()
    starts = (A_START, B_START)
    envs = [ApproachEnv(ApproachConfig(start_state=s, max_steps=STEPS)) for s in starts]
    agents = [clf_cbf.ClfCbfAgent(config) for _ in starts]
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
    for i, start in enumerate(starts):
        alone = actions(clf_cbf.ClfCbfAgent(config), start)
        assert np.array_equal(np.array(logs[i]), np.array(alone))


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
    obs = np.array([30.0, 45.0, 0.0, -0.2, -0.3, 0.0])
    agent = clf_cbf.ClfCbfAgent()
    agent.reset(view)
    plain = agent.act(obs.copy(), {"time_s": 100.0})
    agent.reset(view)
    poisoned = ReadLog(time_s=100.0, true_state=np.full(6, 1e6), dv_used_mps=1e6)
    action = agent.act(obs.copy(), poisoned)
    assert poisoned.read <= {"time_s"}
    assert np.array_equal(action, plain)
    assert action.shape == (3,) and np.all(np.isfinite(action))


def test_recorded_episode_keeps_one_log_entry_per_decision():
    ep = episode(clf_cbf.ClfCbfAgent(), start_state=A_START, max_steps=15)
    assert len(ep.agent_log) == len(ep.records) == 15
    assert all(np.isfinite(e["held_bound"]) for e in ep.agent_log)


def test_the_recorder_and_metrics_never_read_the_controller_log_fields():
    # The log is the agent's own bookkeeping; a scored count needs a channel the recorder trusts.
    package = Path(__file__).parents[1] / "src" / "vbar"
    for folder in ("metrics", "scenarios"):
        for source in (package / folder).rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            for key in ("held_bound", "cbf_active", "fallbacks"):
                assert key not in text, (source, key)
