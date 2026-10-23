"""Fixed-window fuel oracle: certified lower bounds and verified plans on CW transfers.

`FuelOracle.solve(view, start, arrival_step=k)` bounds the least delta-v, in the actuator norm,
that puts the chaser in the capture ball at step k. The box layout is an LP on the zero-order-hold
step grid, solved exactly; the ball layout is bracketed by an outer cutting-plane LP and an inner
polyhedral LP. The lower bounds come from the LP duals through the primer-vector bound, so they
hold for continuous time; the upper bound is the cost of a plan checked by exact propagation.

Known answers: Fehse's tangential and radial V-bar transfers (Automated Rendezvous and Docking of
Spacecraft, 2003, eqs 3.29 to 3.32 and 3.37); Curtis, Orbital Mechanics for Engineering Students,
example 7.5; Alfriend, Vadali, Gurfil, How and Breger, Spacecraft Formation Flying, example 5.2.
The half-orbit transfer of x0 = (-200, -1500, 50) m at rest to (0, -100, 0) m at rest on a 6778 km
orbit with 2e-3 m/s^2 thrusters is a computed case, not a book example; the LP and its duality
follow Boyd and Vandenberghe, Convex Optimization, 2004, sections 4.3 and 5.1.3. Values marked
"independent computation" were computed with `cw_discrete`, scipy's HiGHS and a separate
implementation of the LPs and of the cell bound, not with the module under test.
"""

import math

import numpy as np
import pytest
from metric_cases import pd_episode

from vbar.agents.base import AgentView
from vbar.agents.cw_targeting import TwoImpulseBound, two_impulse
from vbar.dynamics import cw_discrete, cw_states, mean_motion
from vbar.dynamics.cw import R_EARTH
from vbar.metrics import FuelBounds
from vbar.metrics.mission import goal_errors
from vbar.oracles import fuel
from vbar.oracles.fuel import FuelOracle, OracleConfig

N_HALF = mean_motion(6778e3)
T_HALF = math.pi / N_HALF
X0_HALF = np.array([-200.0, -1500.0, 50.0, 0.0, 0.0, 0.0])
HOLD_HALF = (0.0, -100.0, 0.0)
A_HALF = 2e-3
# The 640-step ZOH optimum of the box problem, a feasible plan and so an upper bound on the
# continuous optimum (independent computation); the margin covers HiGHS's tolerances.
P640 = 0.8947292762
P640_MARGIN = 1e-6
# A 40-step ball plan reaching the target point with |u_k|_2 <= a on every step, from a 400-
# direction inner polyhedron (independent computation): no ball bound may exceed its cost.
BALL_PLAN = 0.8324776
# 40 steps of 1423 s: 500 cells per step make 20000 cells over the transfer, where the cell
# bound is within 1e-3 of the dual's continuous value.
FINE = OracleConfig(cells_per_step=500)

N_500 = mean_motion(R_EARTH + 500e3)


def make_view(n, accel, hold, layout, *, dt, steps, radius, speed, constraints=(), **extra):
    return AgentView(
        scenario_id="oracle-test",
        dt=dt,
        max_steps=steps,
        n=n,
        action_mode="accel",
        max_accel=accel,
        thrust_limit=layout,
        goal={"hold_point": hold, "capture_radius": radius, "capture_speed": speed, **extra},
        constraints=constraints,
    )


def half_view(layout="box", radius=0.0, speed=0.0):
    return make_view(
        N_HALF, A_HALF, HOLD_HALF, layout, dt=T_HALF / 40, steps=40, radius=radius, speed=speed
    )


def solve(view, start, *, config=None, arrival_step=None, hard=None):
    oracle = FuelOracle() if config is None else FuelOracle(config=config)
    step = view.max_steps if arrival_step is None else arrival_step
    return oracle.solve(view, np.asarray(start, dtype=float), arrival_step=step, hard=hard)


def order(view):
    return np.inf if view.thrust_limit == "box" else 2


def cost_order(view):
    return 1 if view.thrust_limit == "box" else 2


def fly(view, start, plan):
    """Exact ZOH propagation of a plan, one step at a time."""
    x = np.asarray(start, dtype=float)
    for u in plan:
        x = cw_states(view.n, np.array([view.dt]), x[None], np.asarray(u)[None])[0]
    return x


def assert_verified_plan(sol, view, start, step):
    """The plan lands in the capture ball at `step` with the aim margin, within the thrust
    limit, and costs `upper`."""
    plan = np.asarray(sol.plan)
    assert sol.arrival_step == step
    assert plan.shape == (step, 3)
    peaks = np.linalg.norm(plan, ord=order(view), axis=1)
    assert np.all(peaks <= view.max_accel * (1 + 1e-12))
    position, velocity = goal_errors(fly(view, start, plan), view.goal)
    aim = OracleConfig().aim
    # The plan aims inside aim times the ball; the rest of the margin is left for motion
    # between samples, so the landing point must keep at least half of it.
    assert position <= 0.5 * (1 + aim) * view.goal["capture_radius"]
    assert velocity <= 0.5 * (1 + aim) * view.goal["capture_speed"]
    cost = view.dt * np.linalg.norm(plan, ord=cost_order(view), axis=1).sum()
    assert cost == pytest.approx(sol.upper, rel=1e-12)


def assert_bounds_consistent(sol, view):
    bounds = sol.bounds
    assert isinstance(bounds, FuelBounds)
    assert bounds.rung == "cw"
    assert bounds.norm == {"box": "l1", "ball": "l2"}[view.thrust_limit]
    assert bounds.accel_limit == view.max_accel
    assert bounds.lower == sol.finite_lower
    assert bounds.upper == sol.upper
    assert bounds.impulsive_lower == sol.impulsive_lower
    # Dividing any multiplier by its certified primer peak makes the excess integral zero, so
    # the finite-thrust bound is never below the impulsive one.
    assert sol.finite_lower >= sol.impulsive_lower * (1 - 1e-9)
    assert sol.impulsive_lower >= 0.0
    assert sol.finite_lower <= sol.upper


def test_box_lp_reaches_the_zoh_optimum_of_the_half_orbit_transfer():
    sol = solve(half_view(), X0_HALF, config=FINE)
    assert sol.finite_optimum == pytest.approx(0.8977056, rel=1e-6)
    # The 40-step dual through the cell bound at 20000 cells gives 0.8921478; its continuous
    # value is 0.8930605 and the continuous optimum is below P640 (independent computation).
    # Reporting the ZOH optimum as the lower bound gives 0.8977, dropping the excess term
    # 0.9356, halving it 0.9139, and the 2-norm dual norm 0.3407: each leaves this window.
    assert 0.892140 <= sol.finite_lower <= P640 + P640_MARGIN
    assert sol.finite_lower < sol.finite_optimum
    assert sol.impulsive_lower > 0.0
    assert_bounds_consistent(sol, half_view())


def test_box_plan_is_the_exact_lp_optimum_into_a_capture_ball():
    view = half_view(radius=1.0, speed=1e-3)
    sol = solve(view, X0_HALF, config=FINE)
    # Outer LP (|x_i - x_g,i| <= rho per component) and the inner LP aimed at 0.9 rho / sqrt(3),
    # independent computation.
    assert sol.finite_optimum == pytest.approx(0.8901371173, rel=1e-6)
    assert sol.upper == pytest.approx(0.8937729083, rel=1e-6)
    assert_verified_plan(sol, view, X0_HALF, 40)
    # Cell bound 0.8867022; a 640-step plan inside the ball costs 0.8904004, so the continuous
    # optimum is below it; without the capture-ball support terms the bound reads 0.8921478.
    assert 0.886695 <= sol.finite_lower <= 0.8904004 + P640_MARGIN
    assert_bounds_consistent(sol, view)


def test_support_terms_of_a_wide_capture_ball_lower_the_bound():
    view = half_view(radius=20.0, speed=0.02)
    sol = solve(view, X0_HALF, config=FINE)
    # Cell bound 0.8010997, below the 0.8163351 of a 640-step plan inside the ball; leaving out
    # rho_r |nu_r| + rho_v |nu_v| gives 0.8677, above it (independent computation).
    assert 0.80105 <= sol.finite_lower <= 0.8163351 + P640_MARGIN
    assert sol.upper == pytest.approx(0.8242298458, rel=1e-6)
    assert_verified_plan(sol, view, X0_HALF, 40)


def test_ball_bracket_closes_on_the_half_orbit_transfer():
    view = half_view("ball", radius=0.01, speed=1e-5)
    sol = solve(view, X0_HALF, config=FINE)
    # The outer cutting-plane LP is a relaxation of the 40-step problem and the inner plan a
    # feasible point of it: 0.8305728 and 0.8326060 with the default 20 rounds and 64 inner
    # directions (independent computation), a bracket of 0.24%.
    assert sol.finite_optimum >= 0.8280
    assert sol.upper <= 0.8330
    assert (sol.upper - sol.finite_optimum) / sol.finite_optimum <= 0.005
    assert_verified_plan(sol, view, X0_HALF, 40)
    # The certified continuous-time bound: 0.8277623 from the cut LP's dual, 0.8267173 from a
    # 400-direction outer LP's dual (independent computation). The l_inf dual norm of the box
    # would give 0.8388, above a verified ball plan.
    assert 0.8265 <= sol.finite_lower <= BALL_PLAN
    assert (sol.upper - sol.finite_lower) / sol.finite_lower <= 0.008
    assert_bounds_consistent(sol, view)
    # The inner plan draws each step from the hull of the final cuts and 64 Fibonacci
    # directions; it closes the step-grid bracket to 0.024% (0.02408% measured).
    assert 0.0 <= (sol.upper - sol.finite_optimum) / sol.finite_optimum <= 2.45e-4


def test_inner_hull_vertices_lie_on_the_thrust_ball(monkeypatch):
    # Each step of the ball plan is a_max times a combination, with weights summing to at most
    # 1, of the directions handed to the inner LP; with every direction of 2-norm at most 1 the
    # plan is within the limit before the final rescaling, which would hide a vertex outside.
    seen = []
    original = fuel._hull_lp

    def spy(win, ball, directions):
        seen.append(np.array(directions))
        return original(win, ball, directions)

    monkeypatch.setattr(fuel, "_hull_lp", spy)
    view = half_view("ball", radius=0.01, speed=1e-5)
    sol = solve(view, X0_HALF)
    assert seen and math.isfinite(sol.upper)
    for directions in seen:
        assert len(directions) > OracleConfig().inner_directions
        vertices = view.max_accel * np.linalg.norm(directions, axis=1)
        assert np.all(vertices <= view.max_accel * (1 + 1e-12))


FEHSE_DY = 500.0
FEHSE = {
    # Fehse eqs 3.29 to 3.32: tangential impulses of n dy / (6 pi), one orbit apart.
    "tangential": (2 * math.pi / N_500, N_500 * FEHSE_DY / (3 * math.pi)),
    # Fehse eq 3.37: radial impulses of n dy / 4, half an orbit apart.
    "radial": (math.pi / N_500, N_500 * FEHSE_DY / 2),
}
FEHSE_START = np.array([0.0, -FEHSE_DY, 0.0, 0.0, 0.0, 0.0])


def fehse_view(transfer, layout="box", steps=720, radius=0.0, speed=0.0):
    tf, _ = FEHSE[transfer]
    return make_view(
        N_500, 1.0, (0.0, 0.0, 0.0), layout, dt=tf / steps, steps=steps, radius=radius, speed=speed
    )


@pytest.mark.parametrize(
    "transfer,zoh_excess,dual_shortfall",
    # On 720 steps the ZOH optimum lies 0.781% (tangential) and 0.324% (radial) above the
    # impulsive optimum, and the impulsive bound from its dual 0.576% and 0.230% below it
    # (independent computation): held burns lose to impulses at first order in n dt.
    [("tangential", 0.0079, 0.0060), ("radial", 0.0033, 0.0025)],
)
def test_box_bounds_bracket_fehse_impulsive_optima(transfer, zoh_excess, dual_shortfall):
    _, optimum = FEHSE[transfer]
    view = fehse_view(transfer)
    sol = solve(view, FEHSE_START)
    # Every acceleration history costs at least the impulsive optimum, and the ZOH plan is one.
    assert optimum <= sol.finite_optimum <= optimum * (1 + zoh_excess)
    assert optimum * (1 - dual_shortfall) <= sol.impulsive_lower <= optimum * (1 + 1e-12)
    assert sol.finite_lower >= sol.impulsive_lower * (1 - 1e-9)
    assert sol.finite_lower <= sol.finite_optimum * (1 + 1e-9)

    ball = fehse_view(transfer, radius=0.01, speed=1e-5)
    planned = solve(ball, FEHSE_START)
    assert planned.upper <= optimum * (1 + zoh_excess)
    assert planned.impulsive_lower <= optimum * (1 + 1e-12)
    assert_verified_plan(planned, ball, FEHSE_START, 720)


@pytest.mark.parametrize("transfer", ["tangential", "radial"])
def test_ball_bounds_stay_below_fehse_impulsive_optima(transfer):
    # Both transfers burn along one axis, so their 2-norm cost equals the 1-norm optimum and
    # bounds the ball's optimum from above.
    _, optimum = FEHSE[transfer]
    view = fehse_view(transfer, "ball", steps=40, radius=0.01, speed=1e-5)
    sol = solve(view, FEHSE_START)
    assert 0.0 < sol.impulsive_lower <= optimum * (1 + 1e-12)
    assert_bounds_consistent(sol, view)


# Curtis example 7.5: 300 km circular orbit with mu = 398600 km^3/s^2, chaser 2 km behind at
# rest, rendezvous in 1.49 h.
N_CURTIS = math.sqrt(398600.0e9 / 6678e3**3)
TF_CURTIS = 1.49 * 3600.0
X0_CURTIS = np.array([0.0, -2000.0, 0.0, 0.0, 0.0, 0.0])


def curtis_view(layout, steps):
    return make_view(
        N_CURTIS,
        0.02,
        (0.0, 0.0, 0.0),
        layout,
        dt=TF_CURTIS / steps,
        steps=steps,
        radius=1.0,
        speed=1e-3,
    )


def two_burn_plan(view, start):
    """Burns on the first and last steps solved to hit the hold point at rest exactly."""
    k = view.max_steps
    phi, gamma = cw_discrete(view.n, view.dt)
    first = np.linalg.matrix_power(phi, k - 1) @ gamma
    lhs = np.hstack([first, gamma])
    rhs = -np.linalg.matrix_power(phi, k) @ start
    u = np.linalg.solve(lhs, rhs)
    plan = np.zeros((k, 3))
    plan[0], plan[-1] = u[:3], u[3:]
    return plan


@pytest.mark.parametrize("layout,steps,impulsive_floor", [("box", 120, 0.249), ("ball", 40, 0.218)])
def test_curtis_bounds_stay_below_two_impulse_and_flown_costs(layout, steps, impulsive_floor):
    view = curtis_view(layout, steps)
    sol = solve(view, X0_CURTIS)
    dv1, dv2 = two_impulse(N_CURTIS, X0_CURTIS, np.zeros(3), TF_CURTIS)
    p = cost_order(view)
    impulses = np.linalg.norm(dv1, ord=p) + np.linalg.norm(dv2, ord=p)
    # The book's 0.2452 m/s (2-norm); 0.263461 in the 1-norm.
    assert impulses == pytest.approx({"box": 0.263461, "ball": 0.245230}[layout], abs=1e-6)
    assert sol.impulsive_lower <= impulses * (1 + 1e-12)
    assert TwoImpulseBound().bounds(view, X0_CURTIS).upper >= sol.impulsive_lower
    plan = two_burn_plan(view, X0_CURTIS)
    assert np.all(np.linalg.norm(plan, ord=order(view), axis=1) <= view.max_accel)
    flown = view.dt * np.linalg.norm(plan, ord=p, axis=1).sum()
    assert sol.finite_lower <= flown * (1 + 1e-9)
    # The bound from the LP dual: 0.2501043 (box, 120 steps) and 0.2193070 (ball, 40 steps),
    # independent computation.
    assert sol.impulsive_lower >= impulsive_floor
    if layout == "box":
        # The two-burn plan is a feasible point of the exact inner LP.
        assert sol.upper <= flown * (1 + 1e-9)
    assert_verified_plan(sol, view, X0_CURTIS, steps)
    assert_bounds_consistent(sol, view)


def test_multi_impulse_plan_beats_the_best_two_impulse_plan_of_alfriend_example_5_2():
    # Normalized units, n = 1; 500 steps over n T = 5 with a thrust limit far above need.
    view = make_view(1.0, 10.0, (0.0, 0.0, 0.0), "box", dt=0.01, steps=500, radius=1e-6, speed=1e-6)
    start = np.array([0.01, 0.02, 0.015, 0.001, 0.001, 0.001])
    sol = solve(view, start)
    two = TwoImpulseBound().bounds(view, start).upper
    # Least two-impulse 1-norm on the 400-point grid: 0.0457373; the impulsive 1-norm optimum
    # over n T = 5 is 0.0375601 with four to five impulses; the 500-step plan costs 0.0376931
    # (independent computation).
    assert two == pytest.approx(0.0457373, rel=1e-6)
    assert sol.upper < 0.85 * two
    assert sol.upper >= 0.0375601 * (1 - 1e-4)
    assert 0.0370 <= sol.impulsive_lower <= 0.0375601 * (1 + 1e-5)
    assert_verified_plan(sol, view, start, 500)


def test_flown_pd_run_spends_at_least_the_lower_bound_for_its_capture_step():
    ep = pd_episode()
    states = np.array([rec.true_state for rec in ep.records])
    position, velocity = goal_errors(states, ep.view.goal)
    goal = ep.view.goal
    inside = (position < goal["capture_radius"]) & (velocity < goal["capture_speed"])
    assert inside.any()
    first = int(np.flatnonzero(inside)[0])
    # records[k] holds the state after step k + 1; the PD agent's commands saturate per axis at
    # max_accel under the acceleration drive, which the finite-thrust bound assumes.
    peak = max(np.abs(a).max() for rec in ep.records for _, a in rec.actuation.segments)
    assert peak <= ep.view.max_accel * (1 + 1e-12)
    spent = sum(rec.actuation.dv_l1 for rec in ep.records[: first + 1])
    sol = solve(ep.view, ep.start, arrival_step=first + 1, hard=ep.hard)
    assert 0.0 < sol.finite_lower <= spent * (1 + 1e-9)
    assert sol.impulsive_lower <= spent * (1 + 1e-9)


def property_view(scale=1.0, accel=0.02, radius=1.0, speed=0.01, layout="box", **extra):
    return make_view(
        N_500,
        scale * accel,
        (0.0, 0.0, 0.0),
        layout,
        dt=20.0,
        steps=30,
        radius=scale * radius,
        speed=scale * speed,
        **extra,
    )


def random_starts(seed, count):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-200.0, 200.0, (count, 3)), rng.uniform(-0.2, 0.2, (count, 3))]


def test_bounds_scale_with_the_start_ball_and_thrust_limit():
    # CW is linear: scaling the start, the capture ball and the thrust limit by lam scales every
    # admissible plan, so each bound scales by lam exactly up to rounding.
    for x0 in random_starts(21, 10):
        base = solve(property_view(), x0)
        assert base.finite_lower > 0.0
        for lam in (0.01, 3.7):
            scaled = solve(property_view(lam), lam * x0)
            assert scaled.finite_lower == pytest.approx(lam * base.finite_lower, rel=1e-9)
            assert scaled.impulsive_lower == pytest.approx(lam * base.impulsive_lower, rel=1e-9)
            assert scaled.upper == pytest.approx(lam * base.upper, rel=1e-9)


def test_lower_bound_of_a_sum_is_below_the_sum_of_upper_bounds():
    # Plans for x1 and x2 with half the thrust and the ball add up to a plan for x1 + x2 with the
    # full thrust and twice the ball; with any thrust for the impulsive bound.
    starts = random_starts(22, 12)
    for x1, x2 in zip(starts[::2], starts[1::2], strict=True):
        whole = solve(property_view(accel=0.02, radius=2.0, speed=0.02), x1 + x2)
        halves = [solve(property_view(accel=0.01), x) for x in (x1, x2)]
        assert math.isfinite(halves[0].upper) and math.isfinite(halves[1].upper)
        assert whole.finite_lower <= (halves[0].upper + halves[1].upper) * (1 + 1e-9)
        full = [solve(property_view(), x) for x in (x1, x2)]
        assert whole.impulsive_lower <= (full[0].upper + full[1].upper) * (1 + 1e-9)


def test_bounds_are_symmetric_under_reversal_and_the_orbit_plane_mirror():
    # The box, the ball and the CW equations are unchanged by x -> -x and by z -> -z.
    mirror = np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
    for x0 in random_starts(23, 10):
        base = solve(property_view(), x0)
        for image in (-x0, mirror * x0):
            other = solve(property_view(), image)
            assert other.finite_lower == pytest.approx(base.finite_lower, rel=1e-9)
            assert other.impulsive_lower == pytest.approx(base.impulsive_lower, rel=1e-9)
            assert other.finite_optimum == pytest.approx(base.finite_optimum, rel=1e-9)


def test_bounds_are_convex_in_the_start_state():
    # The average of two plans is a plan for the average start, within the same thrust limit and
    # into the same ball; the LP value is convex in the start for the same reason.
    starts = random_starts(24, 16)
    for xa, xb in zip(starts[::2], starts[1::2], strict=True):
        a, b = solve(property_view(), xa), solve(property_view(), xb)
        mid = solve(property_view(), 0.5 * (xa + xb))
        assert mid.finite_lower <= 0.5 * (a.upper + b.upper) * (1 + 1e-9)
        mean = 0.5 * (a.finite_optimum + b.finite_optimum)
        assert mid.finite_optimum <= mean * (1 + 1e-9)


def test_every_plan_flown_exactly_lands_in_the_ball_and_costs_the_upper_bound():
    view = property_view()
    for x0 in random_starts(25, 8):
        for step in (10, 30):
            sol = solve(view, x0, arrival_step=step)
            assert_verified_plan(sol, view, x0, step)
            assert_bounds_consistent(sol, view)


def test_constraints_the_plan_was_not_checked_against_void_the_upper_bound():
    speed_limit = {"type": "speed_limit", "v0": 0.2, "v1": 4 * N_500}
    view = make_view(
        N_500,
        0.02,
        (0.0, 0.0, 0.0),
        "box",
        dt=20.0,
        steps=30,
        radius=1.0,
        speed=0.01,
        constraints=(speed_limit,),
    )
    x0 = random_starts(27, 1)[0]
    voided = solve(view, x0)
    assert voided.unmodelled == ("speed_limit",)
    assert math.isinf(voided.upper) and math.isinf(voided.bounds.upper)
    free = solve(view, x0, hard=frozenset())
    assert free.unmodelled == ()
    assert math.isfinite(free.upper)
    # The lower bound relaxes every path constraint, so it does not depend on them.
    assert voided.finite_lower == free.finite_lower > 0.0


def test_goal_without_a_capture_ball_raises():
    view = AgentView(
        scenario_id="oracle-test",
        dt=20.0,
        max_steps=30,
        n=N_500,
        action_mode="accel",
        max_accel=0.02,
        thrust_limit="box",
        goal={"hold_point": (0.0, 0.0, 0.0)},
    )
    with pytest.raises(ValueError):
        solve(view, random_starts(28, 1)[0])


@pytest.mark.parametrize("step", [0, 31])
def test_arrival_outside_the_episode_raises(step):
    with pytest.raises(ValueError):
        solve(property_view(), random_starts(29, 1)[0], arrival_step=step)


def dense_path(view, start, plan, samples=50):
    """States at `samples` points inside every step of a plan, by exact propagation."""
    taus = np.linspace(0.0, view.dt, samples)
    phis, gammas = cw_discrete(view.n, taus)
    x = np.asarray(start, dtype=float)
    path = []
    for u in plan:
        path.append(phis @ x + gammas @ u)
        x = path[-1][-1]
    return np.concatenate(path)


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_inhibited_steps_carry_no_thrust_and_leave_the_lower_bound_sound(layout):
    # Steps 10 to 19 start inside the window from 200 s to 400 s.
    inhibited = property_view(layout=layout, thrust_inhibit=((200.0, 400.0),))
    free = property_view(layout=layout)
    x0 = random_starts(30, 1)[0]
    held = solve(inhibited, x0)
    plain = solve(free, x0)
    assert math.isfinite(held.upper)
    assert np.all(held.plan[10:20] == 0.0)
    assert np.any(held.plan[:10] != 0.0) or np.any(held.plan[20:] != 0.0)
    assert_verified_plan(held, inhibited, x0, 30)
    # The lower bound admits thrust in the window, a relaxation: unchanged, and below the
    # inhibited plan.
    assert held.finite_lower == plain.finite_lower
    assert held.impulsive_lower == plain.impulsive_lower
    assert held.finite_lower <= held.upper
    if layout == "box":
        # The inhibited inner LP is the free one with ten steps fixed at zero.
        assert held.upper >= plain.upper * (1 - 1e-9)


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_every_step_inhibited_leaves_no_plan_and_the_relaxed_lower_bound(layout):
    view = property_view(layout=layout, thrust_inhibit=((0.0, 600.0),))
    x0 = random_starts(31, 1)[0]
    sol = solve(view, x0)
    # Free drift from this start does not reach the ball, so no admissible plan exists.
    position, _ = goal_errors(fly(view, x0, np.zeros((30, 3))), view.goal)
    assert position > view.goal["capture_radius"]
    assert sol.plan is None and math.isinf(sol.upper)
    plain = solve(property_view(layout=layout), x0)
    assert sol.finite_lower == plain.finite_lower > 0.0
    assert sol.finite_optimum == plain.finite_optimum
    # A start at rest on the hold point stays in the ball with no thrust at all.
    rest = solve(view, np.zeros(6))
    assert rest.upper == 0.0 and np.all(rest.plan == 0.0)
    assert rest.finite_lower == 0.0


def test_a_plan_through_the_keep_out_sphere_gives_no_upper_bound():
    keep_out = {"type": "keep_out", "radius": 50.0}
    view = make_view(
        N_500,
        0.02,
        (0.0, 100.0, 0.0),
        "box",
        dt=20.0,
        steps=30,
        radius=1.0,
        speed=0.01,
        constraints=(keep_out,),
    )
    x0 = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    free = solve(view, x0, hard=frozenset())
    # Without the check, the cheapest plan passes 29.5 m from the target, inside the sphere.
    ranges = np.linalg.norm(dense_path(view, x0, free.plan)[:, :3], axis=1)
    assert ranges.min() < 50.0
    assert math.isfinite(free.upper)
    checked = solve(view, x0)
    assert checked.unmodelled == ()
    assert checked.plan is None and math.isinf(checked.upper)
    assert checked.finite_lower == free.finite_lower > 0.0


def test_a_dwell_goal_gives_no_upper_bound_and_the_arrival_lower_bound():
    x0 = random_starts(32, 1)[0]
    dwell = solve(property_view(hold_time=100.0), x0)
    arrive = solve(property_view(), x0)
    # A dwell success is in the ball at the arrival step too, so the arrival bound holds; the
    # plan only arrives, so it is no upper bound for the dwell.
    assert math.isinf(dwell.upper) and math.isinf(dwell.bounds.upper)
    assert math.isfinite(dwell.finite_lower) and dwell.finite_lower > 0.0
    assert dwell.finite_lower == arrive.finite_lower
    assert dwell.finite_lower <= arrive.upper


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_an_arrival_too_soon_for_the_thrust_limit_has_no_plan_and_a_sound_bound(layout):
    # 200 m away with 40 s to go: 0.02 m/s^2 moves the chaser at most 16 m.
    view = property_view(layout=layout)
    x0 = np.array([0.0, -200.0, 0.0, 0.0, 0.0, 0.0])
    sol = solve(view, x0, arrival_step=2)
    assert math.isinf(sol.finite_optimum)
    assert sol.plan is None and math.isinf(sol.upper)
    # Two impulses 40 s apart reach the hold point; no acceleration history spends less than
    # the impulsive optimum, and the multipliers come from the LP without the thrust limit.
    dv1, dv2 = two_impulse(N_500, x0, np.zeros(3), 2 * view.dt)
    p = cost_order(view)
    impulses = np.linalg.norm(dv1, ord=p) + np.linalg.norm(dv2, ord=p)
    assert 0.0 < sol.impulsive_lower <= impulses * (1 + 1e-12)
    assert sol.finite_lower >= sol.impulsive_lower


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_a_window_no_input_reaches_gives_zero_bounds(layout):
    # One held step has three inputs for six terminal rows: even without a thrust limit the
    # step grid cannot reach the ball from 200 m, so no multipliers exist and both bounds are 0.
    sol = solve(property_view(layout=layout), [0.0, -200.0, 0.0, 0.0, 0.0, 0.0], arrival_step=1)
    assert sol.finite_lower == 0.0 and sol.impulsive_lower == 0.0
    assert math.isinf(sol.finite_optimum) and math.isinf(sol.upper) and sol.plan is None


@pytest.mark.parametrize("radius,speed", [(0.0, 0.01), (1.0, 0.0)])
def test_a_zero_capture_radius_or_speed_leaves_no_margin_for_a_verified_plan(radius, speed):
    # The capture predicate is strict, so a zero radius or speed admits no state at all.
    sol = solve(property_view(radius=radius, speed=speed), random_starts(33, 1)[0])
    assert math.isinf(sol.upper) and sol.plan is None
    assert sol.finite_lower > 0.0


def test_an_unknown_thrust_layout_raises():
    with pytest.raises(ValueError, match="thrust limit"):
        solve(property_view(layout="cone"), random_starts(34, 1)[0])


@pytest.mark.parametrize(
    "start",
    [
        [0.0, -200.0, 0.0, 0.0, 0.0],
        [0.0, -200.0, math.nan, 0.0, 0.0, 0.0],
        [0.0, -200.0, 0.0, math.inf, 0.0, 0.0],
    ],
)
def test_a_bad_start_state_raises(start):
    with pytest.raises(ValueError, match="start"):
        solve(property_view(), start)


@pytest.mark.parametrize(
    "goal",
    [
        {"hold_point": (0.0, math.nan, 0.0)},
        {"hold_point": (math.inf, 0.0, 0.0)},
        {"goal_velocity": (0.0, math.nan, 0.0)},
        {"goal_velocity": (0.0, 0.0, -math.inf)},
    ],
)
def test_a_non_finite_goal_state_raises(goal):
    view = property_view()
    view = AgentView(
        scenario_id=view.scenario_id,
        dt=view.dt,
        max_steps=view.max_steps,
        n=view.n,
        action_mode=view.action_mode,
        max_accel=view.max_accel,
        thrust_limit=view.thrust_limit,
        goal={**view.goal, **goal},
    )
    with pytest.raises(ValueError, match="hold point and goal velocity"):
        solve(view, random_starts(35, 1)[0])
