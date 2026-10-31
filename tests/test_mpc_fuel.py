"""Fuel-optimal MPC on the exact zero-order-hold CW model.

The 1-norm regulator is a linear program (Rawlings, Mayne and Diehl, Model Predictive Control,
2nd ed., chapter 2, exercise 2.19); its minimizer can be a whole face, so the applied move is not
unique without a strictly convex term (chapter 7, section 7.7). With a fuel-only stage cost the
receding-horizon controller is economic MPC with no stability theorem (chapter 2, section 2.8),
and its arrival time is not its horizon; a shrinking horizon keeps the first plan optimal.
Every number is for a 500 km circular orbit of the WGS 84 Earth, a 10 s step and a 0.02 m/s^2
box limit per axis unless stated; references marked "independent computation" come from a
separate dual solver and HiGHS, not from the module under test.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from metric_cases import episode
from scipy.optimize import linprog

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import mpc_fuel
from vbar.agents.base import AgentView, agent_generator, command_fraction
from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.metrics import agent_metrics
from vbar.scenarios import make

N = mean_motion(EARTH.radius + 500e3)
PERIOD = 2 * np.pi / N
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
HOLD = (0.0, -100.0, 0.0)
GOAL = np.array([*HOLD, 0.0, 0.0, 0.0])
START = np.array([0.0, -500.0, 0.0, 0.0, 0.0, 0.0])
HALF_ORBIT_STEPS = 284  # round(PERIOD / 2 / DT) = round(283.8)
ARRIVED_M = 0.01  # arrival: within 1 cm of the hold point
INFO = {"time_s": 0.0}

# Time reversal of CW: (x, y, z, vx, vy, vz)(t) -> (x, -y, z, -vx, vy, -vz)(-t), with the
# input (ax, -ay, az).
REFLECT_STATE = np.diag([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
REFLECT_INPUT = np.diag([1.0, -1.0, 1.0])

# Cross-track case: start at the node with 3 cm/s out of plane, step one 96th of the period,
# 150 steps, 2e-3 m/s^2 per axis, terminal state at rest at the origin.
CROSS_DT = PERIOD / 96
CROSS_STEPS = 150
CROSS_A_MAX = 2e-3
CROSS_START = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.03])
CROSS_FUEL = 0.0300214291  # m/s, from an independent computation (HiGHS: 0.0300214291213)


def hand_view(mode="normalized", limit="box", hold=HOLD, task=None):
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
        constraints=({"type": "keep_out", "radius": 50.0},),
    )


def terminal_rows(n, dt, x0, x_goal, steps):
    """G and rhs of the terminal equality G u = rhs, u the moves in stage order (m/s^2)."""
    phi, gamma = cw_discrete(n, dt)
    cols = [gamma]
    for _ in range(steps - 1):
        cols.append(phi @ cols[-1])
    g = np.hstack(cols[::-1])
    rhs = x_goal - np.linalg.matrix_power(phi, steps) @ x0
    return g, rhs


def lp_bracket(n, dt, x0, x_goal, steps, a_max):
    """Bounds (m/s) on the open-loop 1-norm optimum, each checked independently of HiGHS.

    The LP is solved by HiGHS in scaled variables: rows of positions over the offset L and of
    velocities over n L, moves over the largest minimum-norm move, so the solver's absolute
    tolerances are relative to the plan. The upper bound is the fuel of HiGHS's own moves,
    checked to hold the box and the rows, so a reference that leans on the tolerances (moves
    scaled by a_max at offsets of 1e-6 m go below zero by 2e-9 m/s^2 and report 16% under the
    optimum) cannot pass; the lower bound is the LP dual function at HiGHS's multipliers
    (weak duality).
    """
    g, rhs = terminal_rows(n, dt, x0, x_goal, steps)
    length = np.linalg.norm(rhs[:3]) + np.linalg.norm(rhs[3:]) / n
    scale = np.r_[np.full(3, 1.0 / length), np.full(3, 1.0 / (n * length))]
    g_rows = g * scale[:, None]
    unit = np.max(np.abs(np.linalg.lstsq(g_rows, rhs * scale, rcond=None)[0]))
    res = linprog(
        np.full(6 * steps, dt),
        A_eq=np.hstack([g_rows, -g_rows]) * unit,
        b_eq=rhs * scale,
        bounds=(0.0, a_max / unit),
        method="highs-ipm",
    )
    assert res.status == 0
    assert np.all((res.x >= 0.0) & (res.x <= a_max / unit))
    u = unit * (res.x[: 3 * steps] - res.x[3 * steps :])
    assert np.max(np.abs(g @ u - rhs) * scale) <= 1e-12
    nu = res.eqlin.marginals * scale * unit
    lower = nu @ rhs - a_max * np.sum(np.maximum(np.abs(g.T @ nu) - dt, 0.0))
    upper = dt * np.sum(np.abs(u))
    assert lower <= upper * (1 + 1e-12)
    # Measured up to 3.2e-5 relative, at 1e-6 m from the goal; 1e-14 on plans of cm/s.
    assert upper - lower <= 1e-4 * upper
    return lower, upper


def lp_fuel(n, dt, x0, x_goal, steps, a_max):
    """The open-loop 1-norm optimum (m/s), as the verified upper end of `lp_bracket`."""
    return lp_bracket(n, dt, x0, x_goal, steps, a_max)[1]


def rollout(n, dt, x0, u):
    phi, gamma = cw_discrete(n, dt)
    x = np.array(x0, dtype=float)
    for uk in u:
        x = phi @ x + gamma @ uk
    return x


def fly(agent, view, x0, steps):
    """Closed loop on the exact CW model; returns states (steps + 1, 6) and inputs (m/s^2)."""
    phi, gamma = cw_discrete(view.n, view.dt)
    agent.reset(view, np.random.default_rng(0))
    states = [np.array(x0, dtype=float)]
    inputs = []
    for k in range(steps):
        action = agent.act(states[-1].copy(), {"time_s": k * view.dt})
        u = command_fraction(view, action) * view.max_accel
        inputs.append(u)
        states.append(phi @ states[-1] + gamma @ u)
    return np.array(states), np.array(inputs)


def distances(states):
    return np.linalg.norm(states[:, :3] - GOAL[:3], axis=1)


def test_config_is_frozen_and_keyword_only_with_the_stated_defaults():
    with pytest.raises(TypeError):
        mpc_fuel.FuelMPCConfig("receding")
    config = mpc_fuel.FuelMPCConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.mode = "shrinking"
    assert (config.mode, config.horizon_s) == ("receding", None)
    assert (config.tie_break, config.max_iter) == (1e-6, 100)


@pytest.mark.parametrize(
    "fields", [{"tie_break": 0.0}, {"tie_break": -1e-6}, {"mode": "fixed"}, {"max_iter": 0}]
)
def test_config_refuses_no_tie_break_an_unknown_mode_and_no_iterations(fields):
    with pytest.raises(ValueError):
        mpc_fuel.FuelMPCConfig(**fields)


def test_planner_refuses_no_tie_break():
    with pytest.raises(ValueError):
        mpc_fuel.fuel_plan(N, DT, START, GOAL, 20, A_MAX, tie_break=0.0)


def test_cross_track_fuel_is_the_lp_optimum_within_half_the_tie_break():
    p_star = lp_fuel(N, CROSS_DT, CROSS_START, np.zeros(6), CROSS_STEPS, CROSS_A_MAX)
    assert p_star == pytest.approx(CROSS_FUEL, rel=1e-9)
    plan = mpc_fuel.fuel_plan(N, CROSS_DT, CROSS_START, np.zeros(6), CROSS_STEPS, CROSS_A_MAX)
    assert plan.status == "converged"
    assert plan.u.shape == (CROSS_STEPS, 3)
    assert np.max(np.abs(plan.u)) <= CROSS_A_MAX
    assert plan.fuel == pytest.approx(CROSS_DT * np.sum(np.abs(plan.u)), rel=1e-12)
    # |u|^2 <= a_max |u| per axis, so the tie-broken plan costs at most tie_break / 2 more fuel
    # than the LP. The 1e-10 m/s covers the quoted rounding of the optimum and the stopping
    # residual: an independent computation lands 1.7e-10 relative below the HiGHS value.
    assert CROSS_FUEL - 1e-10 <= plan.fuel <= CROSS_FUEL * (1 + 5e-7) + 1e-10
    x_n = rollout(N, CROSS_DT, CROSS_START, plan.u)
    # 1e-9 of the motion's scale, 0.03 m/s and 0.03 / n = 27 m; an independent computation
    # ends 1.8e-9 m and 6.5e-12 m/s from rest.
    assert np.linalg.norm(x_n[:3]) <= 1e-9 * 0.03 / N
    assert np.linalg.norm(x_n[3:]) <= 1e-9 * 0.03


@pytest.mark.parametrize(
    "start",
    [CROSS_START, np.array([10.0, -50.0, 0.0, 0.0, 0.0, 0.03])],
    ids=["cross-track", "with-in-plane"],
)
def test_reversed_and_reflected_problem_returns_the_reversed_plan(start):
    # The time-reversed problem (reflected goal to reflected start) has the reversed plan as its
    # optimum, with the stages in the opposite order and the along-track axis reflected. On
    # this face of optima HiGHS returns vertices whose moves differ by 0.127 a_max between the
    # two formulations (first move 0 against -2.54e-4 m/s^2). Tie-broken plans from an
    # independent computation agree to 4.9e-9 a_max, the floor set by rounding in the interior
    # moves amplified by a_max / tie_break; one stopped at the residual rule with a poorer
    # search agreed to 1.3e-7 a_max, so 1e-6 a_max leaves room and sits five decades below the
    # gap between vertices.
    goal = np.zeros(6)
    forward = mpc_fuel.fuel_plan(N, CROSS_DT, start, goal, CROSS_STEPS, CROSS_A_MAX)
    backward = mpc_fuel.fuel_plan(
        N, CROSS_DT, REFLECT_STATE @ goal, REFLECT_STATE @ start, CROSS_STEPS, CROSS_A_MAX
    )
    assert forward.status == backward.status == "converged"
    reversed_plan = backward.u[::-1] @ REFLECT_INPUT
    assert np.max(np.abs(forward.u[0] - reversed_plan[0])) <= 1e-6 * CROSS_A_MAX
    assert np.max(np.abs(forward.u - reversed_plan)) <= 1e-6 * CROSS_A_MAX


def test_half_orbit_plan_is_the_lp_optimum_and_arrives():
    p_star = lp_fuel(N, DT, START, GOAL, HALF_ORBIT_STEPS, A_MAX)
    # An independent computation gives 0.222906697 m/s.
    assert p_star == pytest.approx(0.22291, abs=5e-6)
    plan = mpc_fuel.fuel_plan(N, DT, START, GOAL, HALF_ORBIT_STEPS, A_MAX)
    assert plan.status == "converged"
    assert p_star * (1 - 1e-9) <= plan.fuel <= p_star * (1 + 5e-7) + 1e-10
    x_n = rollout(N, DT, START, plan.u)
    # 1e-9 of the 400 m span; an independent computation reaches 5e-15.
    assert np.linalg.norm(x_n[:3] - GOAL[:3]) <= 1e-9 * 400.0


def test_a_single_step_plan_is_refused():
    # Three inputs cannot meet six terminal equalities, so the multipliers are not unique.
    with pytest.raises(ValueError):
        mpc_fuel.fuel_plan(N, DT, START, GOAL, 1, A_MAX)


@pytest.mark.parametrize(
    "dt, steps, x0, x_goal",
    [
        # 0.2 s: the reachable set is a_max T^2 / 4 = 2e-4 m wide per axis.
        (1e-3, 200, np.array([1e-4, 5e-5, -2e-5, 0.0, 0.0, 0.0]), np.zeros(6)),
        (600.0, 10, START, GOAL),
    ],
    ids=["1e-3 s", "600 s"],
)
def test_plans_at_extreme_steps_arrive(dt, steps, x0, x_goal):
    # Position error over the span, and velocity error times the duration over the span. An
    # independent computation reaches 4e-12 (1e-3 s) and, after a minimum-norm correction of
    # the interior moves onto the terminal rows, 1e-15 (600 s); recovering the moves from the
    # multipliers alone leaves 2.4e-7 at 600 s, since rounding in each interior move is
    # amplified by a_max / tie_break and the along-track column reaches 2.2e7 m per m/s^2.
    plan = mpc_fuel.fuel_plan(N, dt, x0, x_goal, steps, A_MAX)
    assert plan.status == "converged"
    x_n = rollout(N, dt, x0, plan.u)
    span = np.linalg.norm(x0[:3] - x_goal[:3])
    duration = steps * dt
    assert np.linalg.norm(x_n[:3] - x_goal[:3]) <= 1e-9 * span
    assert duration * np.linalg.norm(x_n[3:] - x_goal[3:]) <= 1e-9 * span


def test_iteration_cap_is_reported():
    # A cold start needs several Newton steps on this face of optima.
    plan = mpc_fuel.fuel_plan(
        N, CROSS_DT, CROSS_START, np.zeros(6), CROSS_STEPS, CROSS_A_MAX, max_iter=1
    )
    assert plan.status == "iteration_cap"
    assert plan.iterations == 1


def test_a_cold_start_counts_the_jump_to_the_lp_multipliers_as_its_first_iteration():
    # With one iteration allowed the multipliers are those of the LP without the tie-break, so
    # the LP dual function at them, nu . rhs - a_max sum max(0, |g_j . nu| - dt), equals the
    # LP optimum (strong duality); this holds for any optimal dual, whatever vertex is chosen.
    plan = mpc_fuel.fuel_plan(
        N, CROSS_DT, CROSS_START, np.zeros(6), CROSS_STEPS, CROSS_A_MAX, max_iter=1
    )
    assert (plan.status, plan.iterations) == ("iteration_cap", 1)
    g, rhs = terminal_rows(N, CROSS_DT, CROSS_START, np.zeros(6), CROSS_STEPS)
    price = np.abs(g.T @ plan.nu) - CROSS_DT
    dual = plan.nu @ rhs - CROSS_A_MAX * np.sum(np.maximum(price, 0.0))
    assert dual == pytest.approx(CROSS_FUEL, rel=1e-9)
    # The second iteration is the first Newton step: the multipliers move.
    second = mpc_fuel.fuel_plan(
        N, CROSS_DT, CROSS_START, np.zeros(6), CROSS_STEPS, CROSS_A_MAX, max_iter=2
    )
    assert second.iterations == 2
    assert not np.array_equal(second.nu, plan.nu)


def test_a_target_out_of_reach_in_the_horizon_is_reported_infeasible():
    # Two 10 s steps at 0.02 m/s^2 move the chaser at most a_max T^2 / 4 = 2 m per axis.
    plan = mpc_fuel.fuel_plan(N, DT, START, GOAL, 2, A_MAX)
    assert plan.status == "infeasible"
    # 600 s reaches 1800 m per axis, so 10 km is out of reach from a cold and a warm start.
    far = GOAL + np.array([0.0, -1e4, 0.0, 0.0, 0.0, 0.0])
    near = mpc_fuel.fuel_plan(N, DT, NEAR, GOAL, 60, A_MAX)
    for warm in (None, near.nu):
        assert mpc_fuel.fuel_plan(N, DT, far, GOAL, 60, A_MAX, warm=warm).status == "infeasible"


def test_an_infeasible_plan_is_replaced_by_the_shifted_previous_plan():
    view = hand_view()
    first_plan = mpc_fuel.fuel_plan(N, DT, NEAR, GOAL, 60, A_MAX)
    agent = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(horizon_s=60 * DT))
    agent.reset(view, np.random.default_rng(0))
    agent.act(NEAR.copy(), {"time_s": 0.0})
    far = GOAL + np.array([0.0, -1e4, 0.0, 0.0, 0.0, 0.0])
    second = command_fraction(view, agent.act(far, {"time_s": DT})) * A_MAX
    assert [e["status"] for e in agent.log] == ["converged", "infeasible"]
    assert [e["level"] for e in agent.log] == [0, 1]
    assert agent.log[1]["fallback"] is True
    assert np.max(np.abs(second - first_plan.u[1])) <= 1e-12 * A_MAX


# A near-goal state (1e-6 m and 1.1e-9 m/s off the V-bar hold) whose 60-step plan, warm-started
# from the plan out of NEAR, has moves inside the rounding of the prices. Found by a search over
# offsets; whether the default 100 iterations end there depends on the platform's BLAS (OpenBLAS
# does not converge in 1000, Accelerate converges), so the test sets its cap explicitly.
CAPPED = np.array(
    [
        4.929793712602455e-07,
        -100.00000083073692,
        2.585488326840085e-07,
        5.604269995577952e-11,
        1.0609022365390982e-09,
        3.1034699510234325e-10,
    ]
)


def test_a_near_goal_plan_at_the_cap_is_not_applied():
    view = hand_view()
    first_plan = mpc_fuel.fuel_plan(N, DT, NEAR, GOAL, 60, A_MAX)
    # The first plan converges in 4 updates of the 100 allowed. At max_iter=1 a warm start gets
    # one update, which leaves a relative residual of 0.3 against the stop at 1e-10, so the cap
    # is reached whatever the rounding.
    capped = mpc_fuel.fuel_plan(N, DT, CAPPED, GOAL, 60, A_MAX, max_iter=1, warm=first_plan.nu)
    assert capped.status == "iteration_cap"
    assert np.any(capped.u[0] != first_plan.u[1])
    agent = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(horizon_s=60 * DT))
    agent.reset(view, np.random.default_rng(0))
    agent.act(NEAR.copy(), {"time_s": 0.0})
    agent.config = dataclasses.replace(agent.config, max_iter=1)
    second = command_fraction(view, agent.act(CAPPED.copy(), {"time_s": DT})) * A_MAX
    assert [e["status"] for e in agent.log] == ["converged", "iteration_cap"]
    assert [e["level"] for e in agent.log] == [0, 1]
    assert np.array_equal(second, first_plan.u[1])


# Offsets of 1e-6 and 1e-7 m from the goal, where the moves fall to the rounding of the prices
# and the planner corrects them within that rounding, and one 1e-6 m case, found by a search,
# where accepting moves against the sign of their price would cost 44% above the optimum.
ROUNDING_CASES = [
    (steps, size, seed) for steps in (60, 284) for size in (1e-6, 1e-7) for seed in range(4)
]
SIGN_CASE = np.array(
    [
        -4.6959913142010467e-07,
        -1.6439004837529294e-08,
        -8.82726693200922e-07,
        9.727431872308824e-10,
        -5.04921956669873e-10,
        -1.5425338391626794e-10,
    ]
)


def near_goal_offset(size, seed):
    d = np.random.default_rng(seed).normal(size=6)
    d[:3] *= size / np.linalg.norm(d[:3])
    d[3:] *= size * N / np.linalg.norm(d[3:])
    return d


def test_a_near_goal_plan_costs_no_more_than_its_certified_excess_bound_above_the_optimum():
    # 1e-6 m off the goal over 284 steps the moves fall to the rounding of the prices, and the
    # plan costs 2.5806668e-9 m/s, 1.86% above a feasible plan checked from its own moves
    # (2.5335205e-9 m/s), so only the run-time bound fuel / fuel_lower - 1 (1.89% here) holds.
    # Against an independent lower bound on the optimum, from the LP dual at HiGHS's
    # multipliers, the excess must stay within that bound.
    plan = mpc_fuel.fuel_plan(N, DT, SIGN_CASE, np.zeros(6), 284, A_MAX)
    assert plan.status == "converged"
    lower, upper = lp_bracket(N, DT, SIGN_CASE, np.zeros(6), 284, A_MAX)
    bound = plan.fuel / plan.fuel_lower - 1.0
    assert plan.fuel / lower - 1.0 <= bound
    assert plan.fuel >= upper * (1 - 1e-4)


@pytest.mark.parametrize("steps", [60, 284])
@pytest.mark.parametrize("seed", range(3))
def test_a_plan_from_a_millimetre_off_the_goal_costs_within_half_the_tie_break(steps, seed):
    # At 1e-3 m the moves sit well above the rounding of the prices, so the exact minimizer's
    # bound, fuel within tie_break / 2 of the optimum, holds for the computed plan.
    x0 = near_goal_offset(1e-3, seed)
    plan = mpc_fuel.fuel_plan(N, DT, x0, np.zeros(6), steps, A_MAX)
    assert plan.status == "converged"
    _, upper = lp_bracket(N, DT, x0, np.zeros(6), steps, A_MAX)
    assert plan.fuel <= upper * (1 + 5e-7 + 1e-9)


@pytest.mark.parametrize(
    "dt, x0, x_goal, steps, a_max",
    [
        (CROSS_DT, CROSS_START, np.zeros(6), CROSS_STEPS, CROSS_A_MAX),
        (DT, START, GOAL, HALF_ORBIT_STEPS, A_MAX),
        (DT, SIGN_CASE, np.zeros(6), 284, A_MAX),
    ],
    ids=["cross-track", "half-orbit", "near-goal"],
)
def test_fuel_lower_bounds_the_lp_optimum_and_the_plan(dt, x0, x_goal, steps, a_max):
    # Weak duality: the LP dual at any multipliers, less its rounding, is at most the optimum,
    # which is at most the fuel of any plan that meets the rows.
    plan = mpc_fuel.fuel_plan(N, dt, x0, x_goal, steps, a_max)
    assert plan.status == "converged"
    _, upper = lp_bracket(N, dt, x0, x_goal, steps, a_max)
    assert 0.0 < plan.fuel_lower <= upper
    assert plan.fuel_lower <= plan.fuel
    # So fuel / fuel_lower - 1 bounds the plan's excess over the optimum.
    assert plan.fuel / upper - 1.0 <= plan.fuel / plan.fuel_lower - 1.0


def test_logged_fuel_excess_bound_is_the_plan_over_its_lower_bound():
    plan = mpc_fuel.fuel_plan(N, DT, NEAR, GOAL, 60, A_MAX)
    agent = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(horizon_s=60 * DT))
    agent.reset(hand_view(), np.random.default_rng(0))
    agent.act(NEAR.copy(), {"time_s": 0.0})
    entry = agent.log[0]
    assert entry["planned_fuel_mps"] == plan.fuel
    assert entry["fuel_excess_bound"] == pytest.approx(plan.fuel / plan.fuel_lower - 1.0, rel=1e-12)
    # The tie-break's 5e-7 plus the gap left by the multipliers: an independent computation
    # gives 6.2e-7 here, 4.5e-7 on the half orbit and 9.6e-7 on the cross-track case.
    assert 0.0 <= entry["fuel_excess_bound"] <= 2e-6


def test_round_off_is_not_tried_on_plans_far_above_the_rounding(monkeypatch):
    # From zero multipliers the half-orbit plan's residuals are far beyond what a change of
    # prices within their rounding can remove, so no correction is solved; without that
    # shortcut an independent run solved 24.
    calls = []
    real = mpc_fuel._pseudo_solve

    def counted(m, v):
        calls.append(1)
        return real(m, v)

    monkeypatch.setattr(mpc_fuel, "_pseudo_solve", counted)
    plan = mpc_fuel.fuel_plan(N, DT, START, GOAL, HALF_ORBIT_STEPS, A_MAX, warm=np.zeros(6))
    assert plan.status == "converged"
    assert calls == []


def test_plans_corrected_within_price_rounding_stay_feasible_and_optimal(monkeypatch):
    # The goal at the origin keeps offsets of 1e-7 m clear of the rounding of the position.
    # Each converged plan must hold the box and the terminal rows, every move must be the
    # optimal clip at the returned multipliers up to the rounding of its price (an independent
    # sweep stays within 3.9 of that rounding, against 361 when the price check is dropped),
    # and its fuel must stay within 10% of the LP optimum: the sweep finds at most 1.9%, at
    # the sign case, and 44% there when moves of the wrong sign are accepted.
    accepted = []
    real = mpc_fuel._Dual._round_off

    def counted(self, *args):
        rounded = real(self, *args)
        accepted.append(rounded is not None)
        return rounded

    monkeypatch.setattr(mpc_fuel._Dual, "_round_off", counted)
    cases = [(steps, near_goal_offset(size, seed)) for steps, size, seed in ROUNDING_CASES]
    cases.append((284, SIGN_CASE))
    converged = 0
    for steps, x0 in cases:
        plan = mpc_fuel.fuel_plan(N, DT, x0, np.zeros(6), steps, A_MAX)
        if plan.status != "converged":
            continue
        converged += 1
        g, rhs = terminal_rows(N, DT, x0, np.zeros(6), steps)
        u = plan.u.ravel()
        assert np.max(np.abs(u)) <= A_MAX
        assert np.linalg.norm(g @ u - rhs) <= 1e-9 * np.linalg.norm(rhs)
        w = g.T @ plan.nu
        clip = np.copysign(np.clip((np.abs(w) / DT - 1.0) * (A_MAX / 1e-6), 0.0, A_MAX), w)
        rounding = (A_MAX / 1e-6) * np.finfo(float).eps * (np.abs(g).T @ np.abs(plan.nu)) / DT
        assert np.all(np.abs(u - clip) <= 16 * rounding)
        # The sign case's optimum is about 2.5e-9 m/s, where solver rounding moves the plan by a few
        # tenths of that (2.86e-9 on Linux with the oldest pins), so 1e-9 m/s is allowed on top.
        assert DT * np.sum(np.abs(u)) <= 1.1 * lp_fuel(N, DT, x0, np.zeros(6), steps, A_MAX) + 1e-9
    assert converged >= len(cases) - 2
    assert any(accepted)


def test_default_horizon_is_half_an_orbit_in_whole_steps():
    agent = mpc_fuel.FuelMPCAgent()
    agent.reset(hand_view(), np.random.default_rng(0))
    agent.act(START.copy(), {"time_s": 0.0})
    assert agent.log[0]["planned_arrival_s"] == HALF_ORBIT_STEPS * DT


def test_receding_arrival_moves_with_time_and_shrinking_arrival_is_fixed_at_reset():
    obs = START.copy()
    receding = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(mode="receding"))
    shrinking = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(mode="shrinking"))
    for agent in (receding, shrinking):
        agent.reset(hand_view(), np.random.default_rng(0))
        for t in (0.0, 10.0, 1000.0):
            agent.act(obs, {"time_s": t})
    assert [e["planned_arrival_s"] for e in receding.log] == [2840.0, 2850.0, 3840.0]
    assert [e["planned_arrival_s"] for e in shrinking.log] == [2840.0] * 3


NEAR = np.array([0.0, -160.0, 0.0, 0.0, 0.0, 0.0])


def test_shrinking_horizon_flies_the_open_loop_optimum_and_arrives_on_time():
    # 60 decisions; an independent computation arrives 3.4e-10 m from the hold point after 60
    # steps (0.535 m after 59) at 7.8e-10 relative above the LP optimum, and a receding
    # 60-step horizon is still 19 m away. 0.9 s with that computation.
    steps = 60
    config = mpc_fuel.FuelMPCConfig(mode="shrinking", horizon_s=steps * DT)
    agent = mpc_fuel.FuelMPCAgent(config)
    states, inputs = fly(agent, hand_view(), NEAR, steps + 5)
    dv = DT * np.sum(np.abs(inputs))
    p_star = lp_fuel(N, DT, NEAR, GOAL, steps, A_MAX)
    assert p_star * (1 - 1e-9) <= dv <= p_star * (1 + 5e-7) + 1e-10
    dist = distances(states)
    assert dist[steps - 1] > ARRIVED_M
    assert np.all(dist[steps:] < ARRIVED_M)
    assert np.all(inputs[steps:] == 0.0)
    assert all(e["planned_arrival_s"] == steps * DT for e in agent.log)
    assert agent.log[steps - 1]["flown_arrival_s"] is None
    assert agent.log[-1]["flown_arrival_s"] == steps * DT


def test_after_a_shrinking_arrival_the_agent_holds_the_r_bar_point_with_the_hold_input():
    # On the R-bar at x = -100 m the hold needs a_x = -3 n^2 x = 300 n^2 = 3.7e-4 m/s^2.
    steps = 60
    view = hand_view(hold=(-100.0, 0.0, 0.0))
    config = mpc_fuel.FuelMPCConfig(mode="shrinking", horizon_s=steps * DT)
    agent = mpc_fuel.FuelMPCAgent(config)
    start = np.array([-160.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    states, inputs = fly(agent, view, start, steps + 5)
    hold_input = np.array([300.0 * N**2, 0.0, 0.0])
    # ZOH holds the equilibrium exactly, so the discrete hold input is the continuous one.
    assert np.max(np.abs(inputs[steps:] - hold_input)) <= 1e-9 * hold_input[0]
    dist = np.linalg.norm(states[:, :3] - np.array([-100.0, 0.0, 0.0]), axis=1)
    assert np.all(dist[steps:] < ARRIVED_M)
    statuses = [e["status"] for e in agent.log]
    levels = [e["level"] for e in agent.log]
    assert statuses[steps - 1 :] == ["last_move"] + ["after_arrival"] * 5
    assert levels[steps - 1 :] == [1] * 6
    assert all(e["fallback"] is False for e in agent.log[steps - 1 :])


@pytest.mark.slow
def test_half_orbit_shrinking_horizon_flies_the_open_loop_optimum():
    # 294 decisions on up to 284 steps: 3.5 s with an independent dual solver.
    agent = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(mode="shrinking"))
    states, inputs = fly(agent, hand_view(), START, HALF_ORBIT_STEPS + 10)
    dv = DT * np.sum(np.abs(inputs))
    p_star = lp_fuel(N, DT, START, GOAL, HALF_ORBIT_STEPS, A_MAX)
    # An independent computation gives 0.222906698 m/s, 4.3e-9 above the LP optimum.
    assert dv == pytest.approx(0.22291, abs=5e-6)
    assert p_star * (1 - 1e-9) <= dv <= p_star * (1 + 5e-7) + 1e-10
    dist = distances(states)
    # Independent computation: 0.556 m one step early, 2.5e-10 m on time, 5e-9 m ten steps on.
    assert dist[HALF_ORBIT_STEPS - 1] > ARRIVED_M
    assert np.all(dist[HALF_ORBIT_STEPS:] < ARRIVED_M)
    assert np.all(inputs[HALF_ORBIT_STEPS:] == 0.0)
    assert agent.log[-1]["flown_arrival_s"] == HALF_ORBIT_STEPS * DT


def test_receding_horizon_has_not_arrived_when_its_first_horizon_ends():
    # The same 60-step transfer as above with the arrival pushed back every step; an
    # independent computation is still 18.9 m from the hold point after 60 steps.
    steps = 60
    agent = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(mode="receding", horizon_s=steps * DT))
    states, _ = fly(agent, hand_view(), NEAR, steps + 1)
    assert distances(states)[steps] > 1.0
    assert [e["planned_arrival_s"] for e in agent.log] == [(k + steps) * DT for k in range(61)]
    assert all(e["flown_arrival_s"] is None for e in agent.log)


@pytest.mark.slow
def test_receding_half_orbit_horizon_arrives_late_and_spends_more_than_the_optimum():
    # Rawlings, Mayne and Diehl, chapter 2, section 2.8: with a fuel-only stage cost the
    # receding controller need not fly the plan it made. 425 decisions on 284 steps.
    # A plain LP controller taking whichever vertex HiGHS returned, and an independent
    # tie-broken computation, give the same figures: within 1 cm after 401 steps (0.7064
    # orbit) with 0.170431 m/s, against 0.13563 m/s for the LP optimum arriving then.
    steps = int(0.75 * PERIOD / DT)
    agent = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(mode="receding"))
    states, inputs = fly(agent, hand_view(), START, steps)
    dist = distances(states)
    assert np.any(dist < ARRIVED_M)
    arrival = int(np.argmax(dist < ARRIVED_M))
    assert arrival > HALF_ORBIT_STEPS
    assert arrival * DT / PERIOD == pytest.approx(0.706, abs=0.01)
    dv = DT * np.sum(np.abs(inputs))
    assert dv == pytest.approx(0.17043, abs=5e-6)
    p_same = lp_fuel(N, DT, START, GOAL, arrival, A_MAX)
    assert p_same == pytest.approx(0.13563, abs=5e-6)
    assert dv / p_same - 1.0 == pytest.approx(0.26, abs=0.03)
    assert dv > 1.05 * p_same
    assert agent.log[arrival - 1]["flown_arrival_s"] is None
    assert agent.log[arrival]["flown_arrival_s"] == arrival * DT


@pytest.mark.slow
def test_receding_one_orbit_horizon_arrives_after_its_horizon_near_the_optimum():
    # 571 decisions on 568 steps. An independent tie-broken computation, like the plain LP
    # controller, is within 1 cm after 569 steps (1.0023 orbit) with 0.047112 m/s, 0.3%
    # above the LP optimum for that arrival (0.0469733 m/s); the horizon alone costs 0.0472978.
    steps = 571
    agent = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(mode="receding", horizon_s=568 * DT))
    states, inputs = fly(agent, hand_view(), START, steps)
    dist = distances(states)
    assert np.any(dist < ARRIVED_M)
    arrival = int(np.argmax(dist < ARRIVED_M))
    assert arrival > 568
    assert arrival * DT / PERIOD == pytest.approx(1.002, abs=0.01)
    dv = DT * np.sum(np.abs(inputs))
    assert dv == pytest.approx(0.04711, abs=5e-6)
    p_same = lp_fuel(N, DT, START, GOAL, arrival, A_MAX)
    assert p_same == pytest.approx(0.04697, abs=5e-6)
    assert dv > p_same
    assert agent.log[arrival]["flown_arrival_s"] == arrival * DT


def test_an_unconverged_plan_is_never_applied():
    agent = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(max_iter=1))
    agent.reset(hand_view(), np.random.default_rng(0))
    for t in (0.0, 10.0, 20.0):
        action = agent.act(START.copy(), {"time_s": t})
        assert np.all(action == 0.0)
    assert [e["status"] for e in agent.log] == ["iteration_cap"] * 3
    assert [e["level"] for e in agent.log] == [2] * 3


def test_an_unconverged_plan_is_replaced_by_the_shifted_previous_plan(monkeypatch):
    plans = []
    real = mpc_fuel.fuel_plan

    def second_capped(*args, **kwargs):
        plan = real(*args, **kwargs)
        plans.append(plan)
        if len(plans) == 2:
            return dataclasses.replace(plan, status="iteration_cap")
        return plan

    monkeypatch.setattr(mpc_fuel, "fuel_plan", second_capped)
    view = hand_view()
    phi, gamma = cw_discrete(N, DT)
    agent = mpc_fuel.FuelMPCAgent()
    agent.reset(view, np.random.default_rng(0))
    first = command_fraction(view, agent.act(START.copy(), {"time_s": 0.0})) * A_MAX
    # A 2 m radial push off the plan: an independent computation's fresh first move then
    # differs from the shifted plan's by 0.019 a_max.
    x1 = phi @ START + gamma @ first + np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    second = command_fraction(view, agent.act(x1, {"time_s": DT})) * A_MAX
    assert len(plans) == 2
    assert np.max(np.abs(plans[1].u[0] - plans[0].u[1])) > 1e-3 * A_MAX
    assert np.max(np.abs(second - plans[0].u[1])) <= 1e-12 * A_MAX
    assert [e["level"] for e in agent.log] == [0, 1]
    assert agent.fallbacks[1] == 1


def test_a_ball_thrust_layout_is_refused_at_reset():
    with pytest.raises(ValueError, match="second-order cone"):
        mpc_fuel.FuelMPCAgent().reset(hand_view(limit="ball"), np.random.default_rng(0))


@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_discrete_and_impulsive_action_modes_are_refused_at_reset(mode):
    view = ApproachEnv(ApproachConfig(action_mode=mode)).agent_view()
    with pytest.raises(ValueError, match="takes no continuous command"):
        mpc_fuel.FuelMPCAgent().reset(view, np.random.default_rng(0))


@pytest.mark.parametrize("mode", ["receding", "shrinking"])
def test_commands_are_equal_in_fractions_in_every_continuous_mode(mode):
    views = {
        m: ApproachEnv(ApproachConfig(action_mode=m)).agent_view()
        for m in ("normalized", "accel", "thrust")
    }
    rng = np.random.default_rng(3)
    for _ in range(3):
        obs = START + rng.normal(size=6) * np.array([20.0] * 3 + [0.02] * 3)
        fractions = {}
        for m, view in views.items():
            agent = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(mode=mode, horizon_s=600.0))
            agent.reset(view, np.random.default_rng(0))
            fractions[m] = command_fraction(view, agent.act(obs.copy(), dict(INFO)))
        ref = fractions["normalized"]
        assert np.max(np.abs(ref)) > 0.0
        for m in ("accel", "thrust"):
            assert np.max(np.abs(fractions[m] - ref)) <= 1e-12 * np.max(np.abs(ref))


def hold_view(**goal):
    """The approach-vbar view with its goal entries replaced; None removes an entry."""
    view = make("approach-vbar").agent_view()
    merged = {**view.goal, **goal}
    return dataclasses.replace(view, goal={k: v for k, v in merged.items() if v is not None})


@pytest.mark.parametrize("task", ["dock", "inspect", "depart"])
def test_a_task_that_is_not_a_hold_is_refused_at_reset(task):
    with pytest.raises(ValueError, match=f"cannot fly the {task} task"):
        mpc_fuel.FuelMPCAgent().reset(hold_view(task=task), np.random.default_rng(0))


def test_a_hold_goal_without_a_hold_point_is_refused_at_reset():
    with pytest.raises(ValueError, match="this goal has none"):
        mpc_fuel.FuelMPCAgent().reset(hold_view(hold_point=None), np.random.default_rng(0))


A_START = (60.0, -300.0, -20.0, 0.01, 0.0, 0.0)
B_START = (0.0, -450.0, 30.0, 0.0, 0.0, 0.0)
STEPS = 30
# A 600 s horizon keeps these checks cheap; 0.02 m/s^2 reaches a_max T^2 / 4 = 1800 m in it.
SHORT = 60 * DT


def actions(agent, start):
    env = ApproachEnv(ApproachConfig(start_state=start, max_steps=STEPS))
    return run_episode(env, agent, seed=11, agent_seed=12)["actions"]


@pytest.mark.parametrize("mode", ["receding", "shrinking"])
def test_reset_wipes_the_previous_episode(mode):
    config = mpc_fuel.FuelMPCConfig(mode=mode, horizon_s=SHORT)
    agent = mpc_fuel.FuelMPCAgent(config)
    actions(agent, A_START)
    reused = actions(agent, B_START)
    fresh = actions(mpc_fuel.FuelMPCAgent(config), B_START)
    assert np.array_equal(reused, fresh)
    assert len(agent.log) == STEPS


def test_two_agents_from_one_config_on_interleaved_environments_match_separate_runs():
    config = mpc_fuel.FuelMPCConfig(mode="shrinking", horizon_s=SHORT)
    starts = (A_START, B_START)
    envs = [ApproachEnv(ApproachConfig(start_state=s, max_steps=STEPS)) for s in starts]
    agents = [mpc_fuel.FuelMPCAgent(config), mpc_fuel.FuelMPCAgent(config)]
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
        alone = actions(mpc_fuel.FuelMPCAgent(config), start)
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


@pytest.mark.parametrize("mode", ["receding", "shrinking"])
def test_agent_runs_on_a_hand_made_view_and_reads_only_the_time(mode):
    view = hand_view()
    obs = np.array([5.0, -380.0, 3.0, 0.01, 0.0, -0.002])
    agent = mpc_fuel.FuelMPCAgent(mpc_fuel.FuelMPCConfig(mode=mode))
    agent.reset(view)
    plain = agent.act(obs.copy(), {"time_s": 100.0})
    agent.reset(view)
    poisoned = ReadLog(time_s=100.0, true_state=np.full(6, 1e6), dv_used_mps=1e6)
    command = agent.act(obs.copy(), poisoned)
    assert poisoned.read <= {"time_s"}
    assert np.array_equal(command, plain)
    assert command.shape == (3,) and np.all(np.isfinite(command))


def test_recorded_episode_keeps_one_log_entry_per_decision_with_valid_solver_fields():
    # The recorder copies a per-decision log and the agent metrics check its solver fields.
    ep = episode(mpc_fuel.FuelMPCAgent(), start_state=START, max_steps=15)
    assert len(ep.agent_log) == len(ep.records) == 15
    assert all(
        e["planned_arrival_s"] == e_t + HALF_ORBIT_STEPS * DT
        for e, e_t in zip(ep.agent_log, [0.0, *(r.t for r in ep.records[:-1])], strict=True)
    )
    agent_metrics(ep)


def test_the_recorder_and_metrics_never_read_the_fallback_counts():
    # The counts are the agent's own bookkeeping; a scored count needs a channel the recorder
    # trusts.
    package = Path(__file__).parents[1] / "src" / "vbar"
    for folder in ("metrics", "scenarios"):
        for source in (package / folder).rglob("*.py"):
            assert "fallbacks" not in source.read_text(encoding="utf-8"), source
