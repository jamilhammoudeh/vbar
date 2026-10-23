"""Free final time, the dwell and the dual bank of the CW fuel oracle.

A hold succeeds when the capture predicate holds for `hold_time` without a break, so a success is
in the capture ball at m = floor(hold_time / dt) consecutive step boundaries k..k+m-1 (at one
boundary k >= 1 without a dwell). `FuelOracle.solve(view, start)` without an arrival step bounds the
least delta-v over every window k that fits in the episode: the lower bound is the least window
bound, and the upper bound is the cost of a plan, flown exactly, that keeps the predicate from the
end of step k for the whole dwell. A bound from any multipliers is valid (Boyd and Vandenberghe,
Convex Optimization, 2004, section 5.1.3), so `DualBank` keeps the multipliers of window LPs for a
set of starts and moves each one to every window by shifting its constraint times; for fixed
multipliers and window the bound is affine in the start, which makes a bank bound instant.

Scenario cases run on shortened episodes (the full ones are benchmarks). Values marked "independent
computation" come from a separate implementation of the window LPs (scipy's HiGHS on the
state-space form) and of the cell bound, not from the module under test. There, a window's
"every-window bound" is the largest bound over the multipliers of every window's own LP shifted to
it, the most a start's own LPs give.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from vbar.agents.base import AgentView
from vbar.dynamics import cw_discrete, mean_motion
from vbar.dynamics.cw import R_EARTH
from vbar.metrics import FuelBounds, optimality_gap, record_episode
from vbar.metrics.mission import captured, success
from vbar.oracles.fuel import (
    _SAMPLES,
    DualBank,
    FuelOracle,
    OracleConfig,
    _landed_safely,
    bounds_for,
)
from vbar.oracles.primer import dual_bound
from vbar.scenarios import HoldPointEnv, Maneuvering, get, make

N_500 = mean_motion(R_EARTH + 500e3)
KEEP_OUT = frozenset({"keep_out"})


def make_view(n, accel, hold, *, dt, steps, radius, speed, constraints=(), **extra):
    return AgentView(
        scenario_id="oracle-test",
        dt=dt,
        max_steps=steps,
        n=n,
        action_mode="accel",
        max_accel=accel,
        thrust_limit="box",
        goal={"hold_point": hold, "capture_radius": radius, "capture_speed": speed, **extra},
        constraints=constraints,
    )


def scenario_view(name, steps, **goal):
    view = make(f"vbar:{name}:cw:box:dv:v1").agent_view()
    return replace(view, max_steps=steps, goal={**view.goal, **goal})


def nominal_start(name):
    draw = get(name).nominal()
    return np.r_[draw["start_pos"], draw["start_vel"]]


def dwell_steps(view):
    hold_time = view.goal.get("hold_time", 0.0)
    return math.floor(hold_time / view.dt) if hold_time > 0.0 else 0


def assert_valid_bounds(bounds, view):
    assert isinstance(bounds, FuelBounds)
    assert bounds.rung == "cw"
    assert bounds.norm == "l1"
    assert bounds.accel_limit == view.max_accel
    assert 0.0 <= bounds.lower <= bounds.upper
    if bounds.impulsive_lower is not None:
        assert bounds.impulsive_lower <= bounds.lower * (1 + 1e-9)


def assert_plan_holds_the_dwell(sol, view, start, samples=50):
    """The plan, flown exactly, is within the thrust limit, costs `upper`, and keeps the strict
    capture predicate at boundary `arrival_step` and, with a dwell, at every sample of the next
    floor(hold_time / dt) steps: the whole dwell from the end of step `arrival_step`."""
    plan = np.asarray(sol.plan)
    k, m = sol.arrival_step, dwell_steps(view)
    assert 1 <= k and k + m <= len(plan) <= view.max_steps
    assert np.all(np.abs(plan) <= view.max_accel * (1 + 1e-12))
    assert view.dt * np.abs(plan).sum() == pytest.approx(sol.upper, rel=1e-12)
    taus = np.linspace(0.0, view.dt, samples + 1)
    phis, gammas = cw_discrete(view.n, taus)
    x = np.asarray(start, dtype=float)
    for i, u in enumerate(plan[: k + m]):
        path = phis @ x + gammas @ u
        if i == k - 1:
            assert captured(path[-1], view.goal)
        elif i >= k:
            assert np.all(captured(path, view.goal))
        x = path[-1]


def least_range(view, start, plan, samples=50):
    taus = np.linspace(0.0, view.dt, samples + 1)
    phis, gammas = cw_discrete(view.n, taus)
    x, least = np.asarray(start, dtype=float), np.inf
    for u in plan:
        path = phis @ x + gammas @ u
        least = min(least, np.linalg.norm(path[:, :3], axis=1).min())
        x = path[-1]
    return least


# approach-rbar shortened from 360 to 100 steps: with the 600 s dwell the transfer from the
# nominal start (-400, 0, 0) m at rest has at most 400 s. Independent computation over every
# window: every-window bound 2.2669232883 m/s (least at the last window, 41), dwell-free bound
# 1.2434996264, best verified plan 2.4505454253 (window 40, least range 101.7 m), a bracket of
# 7.49%, most of it the inner box aim rho / sqrt(3) the plans aim at. The bracket narrows with
# time to transfer: 7.21% at 120 steps, 4.84% at 150 steps.
RBAR_STEPS = 100
RBAR_LOWER = 2.2669232883
RBAR_UPPER = 2.4505454253
RBAR_ARRIVE = 1.2434996264
RBAR_BRACKET = 0.08
# Fewer arrival candidates and LPs keep the scenario solves within the test budget; any choice
# gives valid bounds. In both scenario cases the optima lie at the last windows.
COARSE = OracleConfig(arrival_candidates=4, bank_stride=20)


@pytest.fixture(scope="module")
def rbar_solution():
    """One free-time solve of the approach-rbar nominal case, shared by three tests because a
    60-step dwell makes it one of the slowest solves in this module."""
    view = scenario_view("approach-rbar", RBAR_STEPS)
    start = nominal_start("approach-rbar")
    return view, start, FuelOracle(config=COARSE).solve(view, start, hard=KEEP_OUT)


def test_free_time_bracket_of_the_rbar_approach_with_its_dwell(rbar_solution):
    view, start, sol = rbar_solution
    assert_valid_bounds(sol.bounds, view)
    assert sol.unmodelled == ()
    assert math.isfinite(sol.upper)
    # A halved lower bound leaves a bracket above 50%.
    assert (sol.upper - sol.finite_lower) / sol.upper <= RBAR_BRACKET
    # No lower bound may exceed a verified plan, and no plan may cost less than a certified
    # lower bound.
    assert sol.finite_lower <= RBAR_UPPER * (1 + 1e-9)
    assert sol.upper >= RBAR_LOWER * (1 - 1e-9)
    assert_plan_holds_the_dwell(sol, view, start)
    assert least_range(view, start, sol.plan) > 50.0


def test_the_dwell_raises_the_free_time_lower_bound_of_the_rbar_approach(rbar_solution):
    view, start, sol = rbar_solution
    no_dwell = replace(view, goal={**view.goal, "hold_time": 0.0})
    arrive = FuelOracle(config=COARSE).solve(no_dwell, start)
    # The every-window bounds with and without the dwell differ by 1.02 m/s (with LPs every 10
    # windows, 2.2669 and 1.2030): a bound that ignored the dwell fails here.
    assert sol.finite_lower >= arrive.finite_lower
    assert sol.finite_lower >= RBAR_ARRIVE + 0.5
    assert arrive.finite_lower <= arrive.upper


def test_a_one_start_bank_with_the_dwell_is_sound_and_close_to_the_free_time_bound(rbar_solution):
    view, start, sol = rbar_solution
    bank = DualBank.build(view, [start])
    bounds = bank.bounds(start)
    assert_valid_bounds(bounds, view)
    assert bounds.lower <= sol.upper * (1 + 1e-9)
    assert bounds.lower >= 0.95 * sol.finite_lower
    lows = bank.window_lowers(start)
    assert lows.shape == (view.max_steps,)
    # Windows that would end after the episode admit no success.
    last = view.max_steps - dwell_steps(view) + 1
    assert np.all(np.isfinite(lows[:last]))
    assert np.all(np.isinf(lows[last:]) & (lows[last:] > 0))
    assert bounds.lower == max(0.0, lows.min())


# rbar-acquisition shortened from 1140 to 200 steps, from (0, -3000, 0) m at rest to 500 m below
# the target with a 300 s dwell. Independent computation over every window: every-window bound
# 4.1303658977 m/s (window 171, the last), dwell-free bound 3.1400200881, best verified plan
# 4.9221014980 (window 170, least range 497.4 m), a bracket of 16.09% against 2.2% without the
# dwell: the inner box aim rho / sqrt(3) keeps each component of the plan within 0.52 of the
# ball while it holds 500 m below the target against the radial gravity gradient.
ACQUISITION_STEPS = 200
ACQUISITION_LOWER = 4.1303658977
ACQUISITION_UPPER = 4.9221014980
ACQUISITION_ARRIVE = 3.1400200881
ACQUISITION_BRACKET = 0.17


def test_free_time_bracket_of_the_rbar_acquisition_with_its_dwell():
    view = scenario_view("rbar-acquisition", ACQUISITION_STEPS)
    start = nominal_start("rbar-acquisition")
    oracle = FuelOracle(config=COARSE)
    sol = oracle.solve(view, start, hard=KEEP_OUT)
    assert_valid_bounds(sol.bounds, view)
    assert math.isfinite(sol.upper)
    assert (sol.upper - sol.finite_lower) / sol.upper <= ACQUISITION_BRACKET
    assert sol.finite_lower <= ACQUISITION_UPPER * (1 + 1e-9)
    assert sol.upper >= ACQUISITION_LOWER * (1 - 1e-9)
    assert_plan_holds_the_dwell(sol, view, start)
    assert least_range(view, start, sol.plan) > 200.0
    arrive = oracle.solve(replace(view, goal={**view.goal, "hold_time": 0.0}), start)
    assert sol.finite_lower >= arrive.finite_lower
    assert sol.finite_lower >= ACQUISITION_ARRIVE + 0.5


def vbar_starts(count=20, seed=2026):
    spec = get("approach-vbar")
    rng = np.random.default_rng(seed)
    pos, vel = spec.dispersions["start_pos"], spec.dispersions["start_vel"]
    return np.array([np.r_[pos.sample(rng), vel.sample(rng)] for _ in range(count)])


@pytest.fixture(scope="module")
def vbar_bank():
    """One bank of 20 starts drawn from the approach-vbar dispersion, shared by four tests of five
    starts each, since its window LPs are the cost of a bank. The dwell is removed and the episode
    shortened to 60 steps, and LPs are kept every 20 steps, to fit 20 per-start solves in the test
    budget."""
    view = scenario_view("approach-vbar", 60, hold_time=0.0)
    starts = vbar_starts()
    return view, starts, DualBank.build(view, starts, config=OracleConfig(bank_stride=20))


@pytest.mark.parametrize("group", range(4))
def test_bank_bounds_of_dispersed_starts_are_sound_and_within_five_percent(vbar_bank, group):
    view, starts, bank = vbar_bank
    oracle = FuelOracle(config=OracleConfig(arrival_candidates=2))
    for start in starts[5 * group : 5 * group + 5]:
        # Keep-out is left out: the bound relaxes it, and the cheapest plan from a start behind
        # the target passes through the sphere.
        sol = oracle.solve(view, start, hard=frozenset())
        bounds = bank.bounds(start)
        assert_valid_bounds(bounds, view)
        assert math.isfinite(sol.upper)
        assert bounds.lower <= sol.upper * (1 + 1e-9)
        assert bounds.lower >= 0.95 * sol.finite_lower
        # The bank is not below the start's own bound in general: its other starts' multipliers
        # are valid too, and lift it up to 1.9% above the start's every-window bound on these
        # starts with LPs every 10 steps (independent computation).


HOLD_RBAR_500 = (-100.0, 0.0, 0.0)
# Free drift from Phi(340 s)^-1 x_g reaches this R-bar point at rest at the end of step 17 of 20 s
# and misses it by 0.0735 m and 0.00735 m/s one step earlier or later.
DRIFT_STEP = 17


def drift_view():
    return make_view(N_500, 0.02, HOLD_RBAR_500, dt=20.0, steps=30, radius=0.01, speed=1e-4)


def drift_start(view):
    phi, _ = cw_discrete(view.n, DRIFT_STEP * view.dt)
    return np.linalg.solve(phi, np.r_[HOLD_RBAR_500, 0.0, 0.0, 0.0])


def test_the_drift_case_is_free_at_one_step_only():
    view = drift_view()
    phis, _ = cw_discrete(view.n, view.dt * np.arange(view.max_steps + 1))
    inside = captured(phis @ drift_start(view), view.goal)
    assert np.flatnonzero(inside).tolist() == [DRIFT_STEP]


def test_a_bank_finds_a_free_window_between_its_lp_steps():
    view = drift_view()
    start = drift_start(view)
    # Every window's own LP: each window bound is positive except the free one, where it is at
    # most 0, so a bank that skipped window 17 would report a positive lower bound. The least
    # elsewhere is 0.0072022 at window 16 (independent computation).
    every = DualBank.build(view, [start], config=OracleConfig(bank_stride=1))
    lows = every.window_lowers(start)
    assert lows.shape == (30,)
    assert lows[DRIFT_STEP - 1] <= 0.0
    assert np.delete(lows, DRIFT_STEP - 1).min() >= 0.0072
    assert every.bounds(start).lower == 0.0
    # With LPs every 10 steps, window 17 is reached only by shifting multipliers to it; their
    # displacement term is zero there, so the bound is at most 0, and positive elsewhere
    # (0.0034944 at window 16, independent computation).
    strided = DualBank.build(view, [start])
    lows = strided.window_lowers(start)
    assert lows[DRIFT_STEP - 1] <= 0.0
    assert np.delete(lows, DRIFT_STEP - 1).min() > 0.0
    assert strided.bounds(start).lower == 0.0
    assert FuelOracle().solve(view, start).finite_lower == 0.0


def property_view(**extra):
    return make_view(
        N_500, 0.02, (0.0, 0.0, 0.0), dt=20.0, steps=30, radius=1.0, speed=0.01, **extra
    )


def random_starts(seed, count):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-200.0, 200.0, (count, 3)), rng.uniform(-0.2, 0.2, (count, 3))]


def test_shifted_multipliers_never_exceed_a_verified_plan_at_any_window():
    view = property_view()
    starts = random_starts(41, 3)
    bank = DualBank.build(view, starts[:2])
    # The first start is in the bank, the third is not. The comparison is with each window's
    # verified plan: the outer LP optimum relaxes the ball to its bounding box, and a ball bound
    # can exceed it.
    for start in starts[[0, 2]]:
        lows = bank.window_lowers(start)
        compared = 0
        # Every other window keeps the test short.
        for k in range(1, view.max_steps + 1, 2):
            sol = FuelOracle().solve(view, start, arrival_step=k)
            if math.isfinite(sol.upper):
                assert lows[k - 1] <= sol.upper * (1 + 1e-9)
                compared += 1
        assert compared >= 10
        assert np.sum(lows > 0.0) >= 20
        assert bank.bounds(start).lower == max(0.0, lows.min())


def test_free_time_lower_bound_is_below_every_fixed_arrival_plan():
    view = property_view()
    for start in random_starts(42, 2):
        free = FuelOracle().solve(view, start)
        assert_valid_bounds(free.bounds, view)
        assert free.finite_lower > 0.0
        uppers = [FuelOracle().solve(view, start, arrival_step=k).upper for k in range(3, 31, 3)]
        assert free.finite_lower <= min(uppers) * (1 + 1e-9)
        assert math.isfinite(free.upper)
        assert_plan_holds_the_dwell(free, view, start)


class RecordingBank(DualBank):
    """A bank that keeps the multipliers of each entry, one row per boundary of its window."""

    def __init__(self, view, *, config=None):
        super().__init__(view, config=config)
        self.multipliers = []

    def _add(self, nus):
        self.multipliers.append(nus)
        super()._add(nus)


@pytest.mark.parametrize("thrust_limit,norm", [("box", "l1"), ("ball", "l2")])
def test_bank_window_bound_is_the_dual_bound_at_the_shifted_times(thrust_limit, norm):
    # A 60 s dwell puts the success in the ball at 3 boundaries, so several multipliers act at
    # once and the cell slack L h / 2 takes its slope from all of them. The bank's rounding floor
    # covers every product, at least the floor of dual_bound, so it is never above it.
    view = replace(property_view(hold_time=60.0), thrust_limit=thrust_limit)
    config = OracleConfig(bank_stride=14)
    starts = random_starts(47, 2)
    bank = RecordingBank.build(view, starts[:1], config=config)
    m = dwell_steps(view)
    assert len(bank.multipliers) >= 2
    assert all(nus.shape == (m, 6) for nus in bank.multipliers)
    # The second start is not in the bank: the displacement term is affine in the start.
    for start in starts:
        lows = bank.window_lowers(start)
        for k in range(1, view.max_steps - m + 2):
            times = view.dt * (k + np.arange(m))
            cells = config.cells_per_step * (k + m - 1)
            expected = max(
                dual_bound(view.n, view, start, nus, times, norm=norm, accel=accel, cells=cells)
                for nus in bank.multipliers
                for accel in (None, np.inf)
            )
            assert lows[k - 1] <= expected
            assert lows[k - 1] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_no_lower_bound_exceeds_a_verified_plan_with_a_large_capture_ball():
    # With a 20 m, 0.05 m/s ball the support term rho |nu| is a large part of the dual bound:
    # without it the bank puts the bounds of windows 11 to 30 (second start) and 19 to 30 (first)
    # up to 10% above each window's verified plan, and the free-time lower bound above its plan.
    view = make_view(N_500, 0.02, (0.0, 0.0, 0.0), dt=20.0, steps=30, radius=20.0, speed=0.05)
    starts = random_starts(1, 2)
    bank = DualBank.build(view, starts)
    for start in starts:
        free = FuelOracle().solve(view, start)
        assert math.isfinite(free.upper)
        assert free.finite_lower <= free.upper * (1 + 1e-9)
        assert free.impulsive_lower <= free.upper * (1 + 1e-9)
        bounds = bank.bounds(start)
        assert bounds.lower <= free.upper * (1 + 1e-9)
        assert bounds.impulsive_lower <= free.upper * (1 + 1e-9)
        lows = bank.window_lowers(start)
        for k in range(10, view.max_steps + 1, 5):
            plan = FuelOracle().solve(view, start, arrival_step=k)
            assert math.isfinite(plan.upper)
            assert lows[k - 1] <= plan.upper * (1 + 1e-9)
            assert plan.finite_lower <= plan.upper * (1 + 1e-9)


@pytest.mark.parametrize(
    "hold_time,trivial", [(5.0, True), (9.999, True), (10.0, False), (0.0, False)]
)
def test_a_dwell_shorter_than_a_step_gives_the_trivial_lower_bound(hold_time, trivial):
    # A success that holds for less than a step may fall between two boundaries, so no boundary
    # is certain to be in the ball; from one step on, at least one is.
    view = make_view(
        N_500, 0.02, (0.0, 0.0, 0.0), dt=10.0, steps=30, radius=1.0, speed=0.01, hold_time=hold_time
    )
    start = random_starts(43, 1)[0]
    sol = FuelOracle().solve(view, start)
    bank = DualBank.build(view, [start]).bounds(start)
    assert_valid_bounds(sol.bounds, view)
    assert_valid_bounds(bank, view)
    if trivial:
        assert sol.finite_lower == 0.0 and sol.impulsive_lower == 0.0
        assert sol.bounds.lower == 0.0 and bank.lower == 0.0
    else:
        assert sol.finite_lower > 0.0 and bank.lower > 0.0


def maneuvering_view():
    target = Maneuvering(schedule=((900.0, 600.0, (0.0, 5e-4, 0.0)),)).public(N_500)
    return property_view(target=target)


@pytest.mark.parametrize(
    "call",
    [
        lambda view, x: FuelOracle().solve(view, x),
        lambda view, x: FuelOracle().solve(view, x, arrival_step=10),
        lambda view, x: FuelOracle().bounds(view, x),
        lambda view, x: DualBank.build(view, [x]),
    ],
)
def test_a_briefed_maneuvering_target_raises(call):
    # The target's burns are a relative acceleration the CW bound does not model.
    with pytest.raises(ValueError, match="manoeuv|maneuv"):
        call(maneuvering_view(), random_starts(44, 1)[0])


@pytest.mark.parametrize(
    "call",
    [
        lambda view, x: FuelOracle().solve(view, x),
        lambda view, x: FuelOracle().bounds(view, x),
        lambda view, x: DualBank.build(view, [x]),
    ],
)
def test_a_docking_goal_raises(call):
    # Success there is a contact event at a continuous time, not a hold at step boundaries.
    view = make("docking").agent_view()
    assert view.goal["task"] == "dock"
    with pytest.raises(ValueError):
        call(view, np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0]))


def test_bounds_is_the_free_time_solve_with_every_listed_constraint_hard():
    view = property_view(hold_time=60.0)
    start = random_starts(45, 1)[0]
    assert FuelOracle().bounds(view, start) == FuelOracle().solve(view, start).bounds


def test_config_defaults_and_refusals_for_arrival_candidates_and_bank_stride():
    config = OracleConfig()
    assert config.arrival_candidates == 16 and config.bank_stride == 10
    for field in ("arrival_candidates", "bank_stride"):
        with pytest.raises(ValueError):
            OracleConfig(**{field: 0})


def test_without_refinements_the_free_time_lower_bound_is_the_bank_bound():
    view = property_view(hold_time=60.0)
    start = random_starts(46, 1)[0]
    sol = FuelOracle(config=OracleConfig(max_refinements=0)).solve(view, start)
    bank = DualBank.build(view, [start]).bounds(start)
    assert sol.finite_lower == bank.lower
    assert sol.impulsive_lower == bank.impulsive_lower


def test_refinements_raise_the_free_time_lower_bound_but_not_above_the_plan():
    view = property_view(hold_time=60.0)
    start = random_starts(46, 1)[0]
    bank = DualBank.build(view, [start]).bounds(start)
    sol = FuelOracle().solve(view, start)
    assert math.isfinite(sol.upper)
    assert bank.lower <= sol.finite_lower <= sol.upper * (1 + 1e-9)
    # The strided bank gives 1.4913 m/s and the refined bound 1.5862, below the 1.7092 plan, so
    # a solve that skipped the refinements fails here.
    assert sol.finite_lower > bank.lower * (1 + 1e-3)


def test_max_refinements_defaults_to_eight_and_refuses_negative_values():
    assert OracleConfig().max_refinements == 8
    assert OracleConfig(max_refinements=0).max_refinements == 0
    with pytest.raises(ValueError):
        OracleConfig(max_refinements=-1)


def test_a_dwell_that_leaves_the_ball_between_samples_fails_verification():
    # Full thrust back toward the hold point through the dwell step: the range from it peaks
    # midway between two samples, 0.6001 m against 0.5973 m at the nearest samples, so a 0.599 m
    # ball holds every sample and is left between them.
    view = make_view(N_500, 0.02, (0.0, 0.0, 0.0), dt=20.0, steps=2, radius=0.599, speed=0.5)
    h = view.dt / (_SAMPLES - 1)
    at_arrival = np.array([-0.4, 0.0, 0.0, 0.02 * 9.5 * h, 0.0, 0.0])
    plan = np.array([[0.0, 0.0, 0.0], [-0.02, 0.0, 0.0]])
    phi, _ = cw_discrete(view.n, view.dt)
    start = np.linalg.solve(phi, at_arrival)
    phis, gammas = cw_discrete(view.n, np.linspace(0.0, view.dt, 20001))
    dwell = phis @ at_arrival + gammas @ plan[1]
    sampled_phis, sampled_gammas = cw_discrete(view.n, np.linspace(0.0, view.dt, _SAMPLES))
    sampled = sampled_phis @ at_arrival + sampled_gammas @ plan[1]
    assert np.all(captured(sampled, view.goal))
    assert not np.all(captured(dwell, view.goal))
    goal, rho = np.zeros(6), np.array([0.599, 0.5])
    assert not _landed_safely(view, start, plan, goal, rho, 1.0, None, 1)
    # The same plan is accepted by a ball that holds it throughout with the check's margin.
    assert _landed_safely(view, start, plan, goal, np.array([0.61, 0.5]), 1.0, None, 1)


# Fewer candidates and LPs keep each solve of the oracle run short; any choice gives valid bounds.
FAST = OracleConfig(arrival_candidates=4, bank_stride=20)


def approach_spec():
    """approach-rbar with a 100 s dwell and only its keep-out sphere: the closing-speed and
    passive-speed constraints are soft there and take most of the simulation time."""
    spec = get("approach-rbar")
    parameters = {
        **spec.parameters,
        "hold_time": 100.0,
        "closing_limit": None,
        "passive_axis": None,
        "soft": (),
    }
    return replace(spec, parameters=parameters)


class OpenLoopOracle:
    """Solves the oracle from its first observation (perfect navigation) and flies the plan."""

    name = "open-loop-oracle"

    def reset(self, view, rng=None):
        self.view = view
        self.plan = None
        self.step = 0

    def act(self, obs, info):
        if self.plan is None:
            self.solution = FuelOracle(config=FAST).solve(self.view, obs)
            self.plan = self.solution.plan
        u = self.plan[self.step] if self.step < len(self.plan) else np.zeros(3)
        self.step += 1
        return u / self.view.max_accel


@pytest.fixture(scope="module")
def oracle_run():
    """One recorded run and its bounds, shared by two tests."""
    env = HoldPointEnv(approach_spec(), {"max_steps": 50})
    agent = OpenLoopOracle()
    ep = record_episode(env, agent, seed=7, agent_seed=8)
    return ep, agent.solution, bounds_for(ep, oracle=FuelOracle(config=FAST))


def test_bounds_for_an_open_loop_oracle_run_bracket_its_delta_v(oracle_run):
    ep, flown, bounds = oracle_run
    assert ep.hard == KEEP_OUT
    assert success(ep)
    assert_valid_bounds(bounds, ep.view)
    gap = optimality_gap(ep, bounds)
    assert gap["bound_certified"] and not gap["below_lower_bound"]
    assert 1.0 <= gap["optimality_gap"] <= bounds.upper / bounds.lower * (1 + 1e-9)
    # The run flies the plan behind the upper bound and ends with the dwell.
    assert gap["optimality_gap_upper"] == pytest.approx(1.0, abs=1e-6)
    assert bounds.upper == pytest.approx(flown.upper, rel=1e-12)


def test_bounds_for_reads_the_hard_constraints_of_the_episode(oracle_run):
    ep, _, bounds = oracle_run
    speed_limit = {"type": "speed_limit", "v0": 0.2, "v1": 4 * N_500}
    listed = replace(ep, view=replace(ep.view, constraints=(*ep.view.constraints, speed_limit)))
    # The speed limit is listed but not hard for this run, so the plan is still an upper bound;
    # with every listed constraint hard it is unmodelled and voids the upper bound.
    assert bounds_for(listed, oracle=FuelOracle(config=FAST)) == bounds
    assert math.isinf(FuelOracle(config=FAST).bounds(listed.view, listed.start).upper)
