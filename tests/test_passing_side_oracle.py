"""Passing-side oracle: a verified upper bound when the target sphere lies across the transfer.

Outside a ball in three dimensions every path is homotopic to every other, so the classes searched
are passing sides: unit directions e in the plane normal to the start-to-goal line, at equal angles
2 pi j / sides, the orbit normal z among them when the line lies in the orbit plane. Per side, the
keep-out-free plan, with every position inside 1.5 R pushed along e to 1.5 R, gives the first
halfspaces r_k^T q_k >= R + margin, q_k the unit position there; each later LP takes q_k from the
previous iterate. The previous iterate stays feasible, so the LP cost never rises, and each
halfspace lies outside the sphere (|r| >= r^T q by Cauchy-Schwarz): it linearizes the convex |r|
at the previous iterate (Boyd and Vandenberghe, Convex Optimization, 2004, eq. 3.2). The
halfspaces hold at samples only, so every iterate is propagated exactly and kept only if it stays
outside the sphere between samples too. The upper bound is the best verified side; the lower bound
is the fuel oracle's, which relaxes the keep-out sphere. A finite set of sides can miss the
optimum, which only widens the gap.

Scenario cases run on approach-vbar shortened from 360 to 120 steps: at full length the cheapest
keep-out-free plan from the nominal start already passes 90.5 m from the target, so the target is
in the way only on shorter transfers. Values marked "independent computation" come from a separate
state-space LP with the halfspaces at the step boundaries and a margin of 1 m, not from the module
under test.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from vbar.dynamics import cw_discrete
from vbar.metrics import FuelBounds, record_episode
from vbar.metrics.efficiency import delta_v_axes
from vbar.metrics.mission import captured, success
from vbar.metrics.safety import safety_metrics
from vbar.oracles.fuel import FuelOracle, OracleConfig
from vbar.oracles.passing import PassingSideOracle, _directions
from vbar.scenarios import HoldPointEnv, Maneuvering, get, make
from vbar.scenarios.spec import Fixed

KEEP_OUT = frozenset({"keep_out"})
RADIUS = 50.0
STEPS = 120
NOMINAL = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])
# Two arrival candidates and one bank LP keep each solve short; any choice gives valid bounds. On
# the 120-step nominal case the bound is the one the default settings give, 2.3890 m/s, as the
# least window is the last, which the bank always solves.
FAST = OracleConfig(arrival_candidates=2, bank_stride=100, max_refinements=0)
# Three LPs per side keep each solve short; every claim below holds from the third iterate.
ITERATIONS = 3

# Independent computation, nominal start, 120 steps: the cheapest keep-out-free plan arrives at
# step 60 (the last that leaves the 600 s dwell) for 2.5257353762 m/s and passes 48.49 m from the
# target. Four sides with a 1 m margin, third iterates: +x 2.5466298566 m/s, +-z 2.5495420949, -x
# 3.5826066652 (converged: 2.5466263779 on +x and +-z, 51.0 m least range, 3.5241798795 on -x); a
# 3 m margin gives 2.5641. The fuel oracle's lower bound with these settings is 2.3890, so the
# bracket is 6.2%.
RELAXED_COST = 2.5257353762
UPPER_CEILING = 2.60
BRACKET = 0.08


def scenario_view(steps=STEPS, **goal):
    view = make("vbar:approach-vbar:cw:box:dv:v1").agent_view()
    return replace(view, max_steps=steps, goal={**view.goal, **goal})


def hold_steps(view):
    return math.ceil(view.goal.get("hold_time", 0.0) / view.dt)


def path_samples(view, start, plan, samples):
    """States at `samples` + 1 points of every step of the plan, flown exactly."""
    taus = np.linspace(0.0, view.dt, samples + 1)
    phis, gammas = cw_discrete(view.n, taus)
    x, paths = np.asarray(start, dtype=float), []
    for u in plan:
        path = phis @ x + gammas @ u
        paths.append(path)
        x = path[-1]
    return np.array(paths)


def least_range(view, start, plan, samples=500):
    return np.linalg.norm(path_samples(view, start, plan, samples)[..., :3], axis=-1).min()


def assert_plan_lands_and_holds(view, start, plan, cost):
    """Within the thrust limit, costs `cost`, in the capture ball from the arrival boundary
    len(plan) - hold through the dwell at 50 samples per step."""
    plan = np.asarray(plan)
    k = len(plan) - hold_steps(view)
    assert 1 <= k and len(plan) <= view.max_steps
    assert np.all(np.abs(plan) <= view.max_accel * (1 + 1e-12))
    assert view.dt * np.abs(plan).sum() == pytest.approx(cost, rel=1e-12)
    paths = path_samples(view, start, plan, 50)
    assert captured(paths[k - 1, -1], view.goal)
    assert np.all(captured(paths[k:], view.goal))


def side_along(sol, axis):
    """The side whose direction is the unit vector `axis`."""
    found = [s for s in sol.sides if np.allclose(s.direction, axis, atol=1e-12)]
    assert len(found) == 1
    return found[0]


@pytest.fixture(scope="module")
def relaxed():
    """The fuel oracle's keep-out-free solve of the nominal case, shared by three tests."""
    return FuelOracle(config=FAST).solve(scenario_view(), NOMINAL, hard=frozenset())


@pytest.fixture(scope="module")
def passing():
    """One four-side solve of the nominal case, shared by eleven tests: the sides' LPs are the
    cost of the oracle, and a solve per test would put several of them over the time budget."""
    view = scenario_view()
    oracle = PassingSideOracle(config=FAST, sides=4, iterations=ITERATIONS)
    return view, oracle.solve(view, NOMINAL, hard=KEEP_OUT)


def test_the_target_is_in_the_way_of_the_nominal_vbar_transfer(relaxed):
    # Without this the cases below would not exercise a passing side.
    view = scenario_view()
    assert math.isinf(FuelOracle(config=FAST).solve(view, NOMINAL, hard=KEEP_OUT).upper)
    assert relaxed.upper == pytest.approx(RELAXED_COST, rel=1e-6)
    assert relaxed.arrival_step == 60
    assert least_range(view, NOMINAL, relaxed.plan) < RADIUS


def test_passing_bounds_bracket_the_vbar_transfer_with_the_fuel_oracle_lower_bound(
    passing, relaxed
):
    view, sol = passing
    assert isinstance(sol.bounds, FuelBounds)
    assert sol.bounds.rung == "cw" and sol.bounds.norm == "l1"
    assert sol.bounds.accel_limit == view.max_accel
    assert sol.unmodelled == ()
    # The lower bound relaxes the keep-out sphere, so it is the fuel oracle's to the bit.
    assert sol.finite_lower == relaxed.finite_lower
    assert sol.impulsive_lower == relaxed.impulsive_lower
    assert sol.bounds.lower == sol.finite_lower and sol.bounds.upper == sol.upper
    assert 0.0 < sol.finite_lower <= sol.upper < math.inf
    assert (sol.upper - sol.finite_lower) / sol.upper <= BRACKET
    # A plan at the keep-out-free cost passes through the sphere; the reported plan cannot.
    assert sol.upper > RELAXED_COST * (1 + 1e-3)
    assert sol.upper <= UPPER_CEILING
    assert sol.arrival_step == 60
    assert_plan_lands_and_holds(view, NOMINAL, sol.plan, sol.upper)
    assert least_range(view, NOMINAL, sol.plan) >= RADIUS


def nominal_spec():
    """approach-vbar from the nominal start with only its keep-out sphere: the closing-speed and
    passive-speed constraints are soft there and take most of the simulation time."""
    spec = get("approach-vbar")
    parameters = {**spec.parameters, "closing_limit": None, "passive_axis": None, "soft": ()}
    dispersions = {
        **spec.dispersions,
        "start_pos": Fixed(tuple(NOMINAL[:3])),
        "start_vel": Fixed(tuple(NOMINAL[3:])),
    }
    return replace(spec, parameters=parameters, dispersions=dispersions)


class OpenLoop:
    """Flies a fixed plan, then coasts."""

    name = "open-loop"

    def __init__(self, plan):
        self.plan = np.asarray(plan)

    def reset(self, view, rng=None):
        self.view = view
        self.step = 0

    def act(self, obs, info):
        u = self.plan[self.step] if self.step < len(self.plan) else np.zeros(3)
        self.step += 1
        return u / self.view.max_accel


def test_the_passing_plan_flown_open_loop_succeeds_outside_the_keep_out_sphere(passing):
    view, sol = passing
    env = HoldPointEnv(nominal_spec(), {"max_steps": STEPS})
    ep = record_episode(env, OpenLoop(sol.plan), seed=7, agent_seed=8)
    assert np.array_equal(ep.start, NOMINAL)
    assert ep.hard == KEEP_OUT
    assert success(ep)
    assert safety_metrics(ep)["min_margin"]["keep_out"] >= 0.0
    assert delta_v_axes(ep) == pytest.approx(sol.upper, rel=1e-9)


def test_the_keep_out_free_plan_flown_open_loop_enters_the_sphere(relaxed):
    # The same run with the relaxed plan fails, so the success above is the passing side's.
    env = HoldPointEnv(nominal_spec(), {"max_steps": STEPS})
    ep = record_episode(env, OpenLoop(relaxed.plan), seed=7, agent_seed=8)
    assert not success(ep)
    assert safety_metrics(ep)["min_margin"]["keep_out"] < 0.0


def test_sides_are_unit_directions_at_quarter_turns_normal_to_the_transfer_line(passing):
    _, sol = passing
    line = np.asarray(scenario_view().goal["hold_point"]) - NOMINAL[:3]
    line = line / np.linalg.norm(line)
    directions = np.array([s.direction for s in sol.sides])
    assert directions.shape == (4, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
    assert np.allclose(directions @ line, 0.0, atol=1e-12)
    # Four sides at quarter turns: each is orthogonal to two others and opposite the fourth.
    gram = directions @ directions.T
    assert np.allclose(np.sort(gram, axis=1), [[-1.0, 0.0, 0.0, 1.0]] * 4, atol=1e-12)
    side_along(sol, (0.0, 0.0, 1.0))
    side_along(sol, (0.0, 0.0, -1.0))


def test_the_plus_z_and_minus_z_sides_cost_the_same(passing):
    # CW out-of-plane motion is symmetric under z -> -z and the transfer lies in the orbit plane.
    # Independent computation: both reach 2.5495420949 m/s at the third iterate with a 1 m margin.
    _, sol = passing
    up, down = side_along(sol, (0.0, 0.0, 1.0)), side_along(sol, (0.0, 0.0, -1.0))
    assert math.isfinite(up.upper)
    assert down.upper == pytest.approx(up.upper, rel=1e-6)
    assert len(down.costs) == len(up.costs)
    assert np.allclose(down.costs, up.costs, rtol=1e-6, atol=0.0)


def test_iterations_never_raise_the_cost_of_a_side(passing):
    _, sol = passing
    for side in sol.sides:
        costs = np.asarray(side.costs)
        assert 1 <= len(costs) <= ITERATIONS
        assert len(side.verified) == len(costs)
        # Each LP's previous iterate is feasible for it; the slack is HiGHS's tolerance.
        assert np.all(costs[1:] <= costs[:-1] * (1 + 1e-9))
    # The +-z sides start at 2.97 m/s from the pushed plan and fall to 2.550 by the third iterate
    # (independent computation), so the iterations do work.
    up = side_along(sol, (0.0, 0.0, 1.0))
    assert up.costs[-1] < up.costs[0] * (1 - 1e-2)


def test_no_lower_bound_exceeds_a_verified_plan_of_any_side(passing):
    view, sol = passing
    verified = [s for s in sol.sides if math.isfinite(s.upper)]
    # Independent computation: all four sides verify with a 1 m margin.
    assert len(verified) >= 3
    for side in verified:
        assert sol.finite_lower <= side.upper * (1 + 1e-9)
        assert sol.impulsive_lower <= side.upper * (1 + 1e-9)
        assert side.upper == min(c for c, ok in zip(side.costs, side.verified, strict=True) if ok)
        assert sol.upper <= side.upper
    assert sol.upper == min(s.upper for s in sol.sides)


def test_every_verified_side_plan_clears_the_sphere_between_samples(passing):
    view, sol = passing
    for side in sol.sides:
        if math.isfinite(side.upper):
            assert_plan_lands_and_holds(view, NOMINAL, side.plan, side.upper)
            assert least_range(view, NOMINAL, side.plan, samples=1000) >= RADIUS
        else:
            assert side.plan is None


def test_halfspaces_inside_the_sphere_give_plans_that_fail_verification():
    # Halfspaces at R - 2 m: on the +x side the keep-out-free plan (48.49 m at closest) meets
    # every row, so each iterate is that plan; the -x iterates reach 49.57, 52.47 and 48.17 m
    # (independent computation). A verifier that trusted the halfspaces would report 2.5257.
    view = scenario_view()
    oracle = PassingSideOracle(config=FAST, sides=2, iterations=ITERATIONS, margin=-2.0)
    sol = oracle.solve(view, NOMINAL, hard=KEEP_OUT)
    near = side_along(sol, (1.0, 0.0, 0.0))
    assert near.costs and not any(near.verified)
    assert math.isinf(near.upper) and near.plan is None
    assert np.allclose(near.costs, RELAXED_COST, rtol=1e-6)
    for side in sol.sides:
        assert not side.verified[-1]
    assert not sol.upper <= RELAXED_COST * (1 + 1e-3)
    if math.isfinite(sol.upper):
        assert least_range(view, NOMINAL, sol.plan) >= RADIUS


def test_halfspaces_on_the_sphere_are_rejected_where_the_path_dips_between_samples():
    # With no margin the third +x iterate meets r_k^T q_k = R at the samples and reaches 49.99 m
    # between them (independent computation, rows at the step boundaries); a check at the
    # samples alone would accept it.
    view = scenario_view()
    oracle = PassingSideOracle(config=FAST, sides=2, iterations=ITERATIONS, margin=0.0)
    sol = oracle.solve(view, NOMINAL, hard=KEEP_OUT)
    near = side_along(sol, (1.0, 0.0, 0.0))
    assert not near.verified[-1]
    for side in sol.sides:
        if math.isfinite(side.upper):
            assert least_range(view, NOMINAL, side.plan, samples=1000) >= RADIUS
    assert sol.finite_lower <= sol.upper


# Independent computation: from (150, 200, 0) m at rest the keep-out-free plan arrives at step 60
# for 1.4303029426 m/s and passes no closer than 75.5 m.
CLEAR = np.array([150.0, 200.0, 0.0, 0.0, 0.0, 0.0])
CLEAR_COST = 1.4303029426


def test_a_target_out_of_the_way_gives_the_fuel_oracle_result():
    view = scenario_view()
    plain = FuelOracle(config=FAST).solve(view, CLEAR, hard=KEEP_OUT)
    assert plain.upper == pytest.approx(CLEAR_COST, rel=1e-6)
    assert least_range(view, CLEAR, plain.plan) > 70.0
    sol = PassingSideOracle(config=FAST).solve(view, CLEAR, hard=KEEP_OUT)
    assert sol.upper == pytest.approx(plain.upper, rel=1e-9)
    assert sol.finite_lower == plain.finite_lower
    assert sol.impulsive_lower == plain.impulsive_lower
    assert sol.arrival_step == plain.arrival_step
    assert_plan_lands_and_holds(view, CLEAR, sol.plan, sol.upper)


def test_a_soft_keep_out_gives_the_fuel_oracle_result():
    # Nothing has to pass the sphere, so the keep-out-free plan through it is an upper bound.
    view = scenario_view()
    plain = FuelOracle(config=FAST).solve(view, NOMINAL, hard=frozenset())
    sol = PassingSideOracle(config=FAST).solve(view, NOMINAL, hard=frozenset())
    assert sol.upper == pytest.approx(plain.upper, rel=1e-9)
    assert sol.upper == pytest.approx(RELAXED_COST, rel=1e-6)
    assert sol.finite_lower == plain.finite_lower
    assert sol.unmodelled == ()


def test_unmodelled_hard_constraints_void_the_upper_bound_and_bounds_is_solve():
    # Every listed constraint is hard by default; the closing-speed and passive-speed rules are
    # not checked by the plan.
    view = scenario_view()
    oracle = PassingSideOracle(config=FAST)
    sol = oracle.solve(view, CLEAR)
    assert math.isinf(sol.upper) and sol.plan is None
    assert sol.unmodelled == ("closing_speed_limit", "passive_speed")
    assert sol.finite_lower > 0.0
    assert oracle.bounds(view, CLEAR) == sol.bounds


def test_defaults_are_eight_sides_and_five_iterations():
    oracle = PassingSideOracle()
    assert oracle.sides == 8 and oracle.iterations == 5
    assert oracle.config == OracleConfig()


@pytest.mark.parametrize(
    "settings",
    [
        {"sides": 0},
        {"sides": -4},
        {"iterations": 0},
        {"margin": math.nan},
        {"margin": math.inf},
    ],
)
def test_bad_settings_raise(settings):
    with pytest.raises(ValueError):
        PassingSideOracle(**settings)


@pytest.mark.parametrize(
    "start", [np.zeros(5), np.r_[NOMINAL[:5], np.nan], np.r_[NOMINAL[:5], np.inf]]
)
def test_a_bad_start_raises(start):
    with pytest.raises(ValueError):
        PassingSideOracle(config=FAST).solve(scenario_view(), start, hard=KEEP_OUT)


def test_a_docking_goal_raises():
    # Success there is a contact event at a continuous time, not a hold at step boundaries.
    view = make("docking").agent_view()
    with pytest.raises(ValueError):
        PassingSideOracle(config=FAST).solve(view, np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0]))


def test_a_briefed_maneuvering_target_raises():
    view = scenario_view()
    target = Maneuvering(schedule=((900.0, 600.0, (0.0, 5e-4, 0.0)),)).public(view.n)
    view = replace(view, goal={**view.goal, "target": target})
    with pytest.raises(ValueError, match="manoeuv|maneuv"):
        PassingSideOracle(config=FAST).solve(view, NOMINAL, hard=KEEP_OUT)


def test_the_default_margin_on_a_50_m_sphere_is_one_metre(passing):
    # The first iterate of a side depends only on the keep-out-free plan, the side and the
    # margin; with no margin the +x side's first iterate costs 2.5467835 m/s, not 2.5616174.
    view, sol = passing
    explicit = PassingSideOracle(config=FAST, sides=4, iterations=1, margin=0.02 * RADIUS)
    first = explicit.solve(view, NOMINAL, hard=KEEP_OUT)
    assert [s.costs[0] for s in first.sides] == [s.costs[0] for s in sol.sides]


def test_the_default_margin_scales_with_the_keep_out_radius():
    # On an 80 m sphere the default is 1.6 m: the +x side's first iterate costs 3.3930340 m/s
    # with it and 3.3750074 with a fixed 1 m, so a margin that ignored the radius fails here.
    view = replace(scenario_view(), constraints=({"type": "keep_out", "radius": 80.0},))
    solves = [
        PassingSideOracle(config=FAST, sides=1, iterations=1, margin=margin).solve(
            view, NOMINAL, hard=KEEP_OUT
        )
        for margin in (None, 0.02 * 80.0)
    ]
    default, explicit = (sol.sides[0] for sol in solves)
    assert default.costs and default.costs == explicit.costs
    assert default.verified == explicit.verified


# The ball layout is checked on approach-vbar shortened to 60 steps without a dwell, with fewer
# cut rounds and directions: the fuel oracle's ball LPs with a 60-step dwell take tens of seconds.
# There the keep-out-free plan arrives at step 60 for 1.8463 m/s and passes 45.3 m from the
# target; two sides verify at 2.0458 (+x) and 3.4393 m/s (-x) at the second iterate.
BALL = OracleConfig(
    arrival_candidates=2, bank_stride=100, max_refinements=0, cut_rounds=4, inner_directions=16
)


def test_ball_side_plans_stay_in_the_thrust_ball_and_outside_the_sphere():
    view = replace(scenario_view(60, hold_time=0.0), thrust_limit="ball")
    relaxed = FuelOracle(config=BALL).solve(view, NOMINAL, hard=frozenset())
    assert least_range(view, NOMINAL, relaxed.plan) < RADIUS
    sol = PassingSideOracle(config=BALL, sides=2, iterations=2).solve(view, NOMINAL, hard=KEEP_OUT)
    assert sol.bounds.norm == "l2" and sol.bounds.rung == "cw"
    assert sol.finite_lower == relaxed.finite_lower
    assert sol.impulsive_lower == relaxed.impulsive_lower
    assert 0.0 < sol.finite_lower <= sol.upper < math.inf
    assert sol.upper == min(s.upper for s in sol.sides)
    verified = [s for s in sol.sides if math.isfinite(s.upper)]
    assert verified
    for side in verified:
        plan = np.asarray(side.plan)
        size = np.linalg.norm(plan, axis=1)
        assert np.all(size <= view.max_accel * (1 + 1e-12))
        assert view.dt * size.sum() == pytest.approx(side.upper, rel=1e-12)
        assert captured(path_samples(view, NOMINAL, plan, 1)[-1, -1], view.goal)
        assert least_range(view, NOMINAL, plan, samples=1000) >= RADIUS


def test_a_start_on_the_hold_point_takes_vbar_as_its_line_of_approach():
    # From the hold point at 3 m/s toward the target, stopping takes 225 m at 0.02 m/s^2 and the
    # chaser crosses the V-bar origin within 37 s, by when it can move at most 13.5 m sideways:
    # no plan stays out of the sphere, so none may be reported.
    start = np.array([0.0, -100.0, 0.0, 0.0, 3.0, 0.0])
    oracle = PassingSideOracle(config=FAST, sides=4, iterations=1)
    sol = oracle.solve(scenario_view(), start, hard=KEEP_OUT)
    directions = np.array([s.direction for s in sol.sides])
    assert directions.shape == (4, 3)
    assert np.allclose(directions[:, 1], 0.0, atol=1e-12)
    side_along(sol, (0.0, 0.0, 1.0))
    side_along(sol, (0.0, 0.0, -1.0))
    assert math.isinf(sol.upper) and sol.plan is None and sol.arrival_step is None
    assert 0.0 <= sol.finite_lower


def test_a_start_holding_at_the_hold_point_costs_nothing():
    start = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    sol = PassingSideOracle(config=FAST).solve(scenario_view(), start, hard=KEEP_OUT)
    assert sol.finite_lower == 0.0 and sol.upper == 0.0
    assert sol.sides == ()


def test_a_line_along_the_orbit_normal_takes_the_radial_axis_for_its_sides():
    start = np.array([0.0, -100.0, 300.0, 0.0, 0.0, 0.0])
    goal = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    directions = _directions(start, goal, 4)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
    assert np.allclose(directions[:, 2], 0.0, atol=1e-12)
    assert np.isclose(directions @ (1.0, 0.0, 0.0), 1.0, atol=1e-12).sum() == 1
    assert np.isclose(directions @ (-1.0, 0.0, 0.0), 1.0, atol=1e-12).sum() == 1
    # The line passes 100 m from the target, so the solve needs no detour and gives the bounds.
    view = scenario_view()
    sol = PassingSideOracle(config=FAST, sides=4, iterations=1).solve(view, start, hard=KEEP_OUT)
    assert 0.0 < sol.finite_lower <= sol.upper < math.inf
    assert least_range(view, start, sol.plan) >= RADIUS
