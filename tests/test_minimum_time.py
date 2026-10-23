"""Minimum-time oracle: the fewest held steps into the capture ball, with certified lower steps.

`minimum_time(view, start)` returns the smallest step k at whose end a plan of accelerations held
one step each, within `max_accel`, puts the chaser in the capture ball, verified by exact
propagation, and for every smaller step either a multiplier nu that `verify_infeasible` accepts
or the step in `uncertified_steps`. A multiplier proves a step unreachable when

    nu^T (x_g - Phi(T) x0) - rho_r |nu_r| - rho_v |nu_v| > a * integral over [0, T] of |p(t)|_*,

p the primer of nu and |.|_* the norm dual to the thrust limit (l1 for box thrusters): every
acceleration history within the limit, held per step or not, moves nu^T x by at most the right
side (Boyd and Vandenberghe, Convex Optimization, 2004, section 5.8, theorems of alternatives).

Known answers: with n = 1e-9 rad/s the in-plane motion over a few minutes is a double
integrator to 1e-12 m, so the rest-to-rest minimum time over d is the bang-bang 2 sqrt(d / a),
and the largest rest-to-rest move in k held steps of dt is a dt^2 floor(k^2 / 4). Along one axis
the primer of nu = (nu_y, nu_vy) is nu_y tau + nu_vy; with nu_vy = -nu_y T / 2 its integral is
nu_y T^2 / 4, so the multiplier proves T unreachable exactly when T < 2 sqrt(d / a). Values
marked "independent computation" come from scipy's matrix exponential of the CW system and scipy
HiGHS, not from the module under test.
"""

import math

import numpy as np
import pytest

from vbar.agents.base import AgentView
from vbar.dynamics import cw_discrete, cw_states, mean_motion
from vbar.dynamics.cw import R_EARTH
from vbar.metrics.mission import goal_errors
from vbar.oracles.minimum_time import MinimumTime, minimum_time, verify_infeasible

N_FREE = 1e-9
A = 0.02
POINT = 1e-6
D_FREE = 100.0
X0_FREE = np.array([0.0, -D_FREE, 0.0, 0.0, 0.0, 0.0])
N_500 = mean_motion(R_EARTH + 500e3)


def make_view(n, accel, *, dt, steps, radius=POINT, speed=POINT, layout="box", **goal):
    return AgentView(
        scenario_id="minimum-time-test",
        dt=dt,
        max_steps=steps,
        n=n,
        action_mode="accel",
        max_accel=accel,
        thrust_limit=layout,
        goal={
            "hold_point": (0.0, 0.0, 0.0),
            "capture_radius": radius,
            "capture_speed": speed,
            **goal,
        },
    )


def free_view(steps=200):
    return make_view(N_FREE, A, dt=1.0, steps=steps)


def switching_certificate(t, axes=(1,)):
    """Multiplier whose primer along each axis is tau - t / 2 (n -> 0): it proves a rest-to-rest
    move of d unreachable in t exactly when d > a t^2 / 4."""
    nu = np.zeros(6)
    for axis in axes:
        nu[axis] = 1.0
        nu[3 + axis] = -t / 2
    return nu


def fly(view, start, plan):
    """Exact ZOH propagation of a plan, one step at a time."""
    x = np.asarray(start, dtype=float)
    for u in plan:
        x = cw_states(view.n, np.array([view.dt]), x[None], np.asarray(u)[None])[0]
    return x


def assert_plan_lands(result, view, start):
    plan = np.asarray(result.plan)
    assert plan.shape == (result.steps, 3)
    assert np.all(np.abs(plan) <= view.max_accel * (1 + 1e-12))
    position, velocity = goal_errors(fly(view, start, plan), view.goal)
    assert position < view.goal["capture_radius"]
    assert velocity < view.goal["capture_speed"]


@pytest.fixture(scope="module")
def free_result():
    # One solve shared by the tests of the 100 m transfer: it proves 141 steps unreachable, the
    # costliest call in this module.
    return minimum_time(free_view(), X0_FREE)


def test_hundred_metres_rest_to_rest_need_142_one_second_steps(free_result):
    # 2 sqrt(100 / 0.02) = 141.42 s; 141 held steps move at most 0.02 * 70 * 71 = 99.40 m and
    # 142 move 100.82 m (independent computation).
    assert isinstance(free_result, MinimumTime)
    assert free_result.steps == math.ceil(2 * math.sqrt(D_FREE / A)) == 142
    assert free_result.certified is True
    assert free_result.uncertified_steps == ()
    assert free_result.plan is not None


def test_minimum_time_plan_flown_exactly_lands_on_the_point_within_the_box(free_result):
    assert_plan_lands(free_result, free_view(), X0_FREE)


def test_switching_multiplier_proves_141_steps_unreachable_and_not_142():
    view = free_view()
    # Margins 100 - 0.02 * 141^2 / 4 = 0.595 m and 100 - 0.02 * 142^2 / 4 = -0.82 m. The global
    # Lipschitz bound of the primer (7 |nu_y|) costs 1.76 m at 8 cells per step and 0.22 m at
    # 64 (independent computation), so the verifier must refine its cells to accept 141.
    assert verify_infeasible(view, X0_FREE, switching_certificate(141.0), 141) is True
    assert verify_infeasible(view, X0_FREE, switching_certificate(142.0), 142) is False


def test_verifier_rejects_every_random_multiplier_at_a_reachable_step():
    view = free_view()
    rng = np.random.default_rng(2026)
    scale = np.r_[np.ones(3), np.full(3, 71.0)]
    best = switching_certificate(142.0)
    near = best + 1e-2 * scale * rng.standard_normal((50, 6))
    broad = scale * rng.standard_normal((50, 6))
    sizes = 10.0 ** rng.uniform(-3.0, 3.0, size=(100, 1))
    for nu in np.vstack([near, broad]) * sizes:
        assert verify_infeasible(view, X0_FREE, nu, 142) is False
    # The same small perturbations of the multiplier for 141 steps still prove 141 unreachable,
    # so the sample sits on both sides of the verifier's threshold.
    tight = switching_certificate(141.0) + 1e-4 * scale * rng.standard_normal((10, 6))
    assert all(verify_infeasible(view, X0_FREE, nu, 141) for nu in tight)


def test_box_verifier_charges_the_primer_on_every_axis():
    # 100 m along x and along y at once: each axis is its own double integrator under box
    # thrusters, so 142 steps reach the target. Charging only the largest primer component
    # (the l_inf norm) would let the two-axis multiplier claim 200 m against 100.82 m of reach.
    view = free_view()
    x0 = np.array([-100.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    assert verify_infeasible(view, x0, switching_certificate(142.0, axes=(0, 1)), 142) is False
    assert verify_infeasible(view, x0, switching_certificate(141.0, axes=(0, 1)), 141) is True


def test_certificate_holds_under_positive_scaling_and_fails_when_negated():
    view = free_view()
    nu = switching_certificate(141.0)
    for size in (1e-6, 1e-2, 1e4):
        assert verify_infeasible(view, X0_FREE, size * nu, 141) is True
    assert verify_infeasible(view, X0_FREE, -nu, 141) is False
    assert verify_infeasible(view, X0_FREE, np.zeros(6), 141) is False


def test_thirty_metres_radial_need_t_min_over_dt_rounded_up():
    view = make_view(N_FREE, A, dt=2.0, steps=60)
    x0 = np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = minimum_time(view, x0)
    # 2 sqrt(30 / 0.02) = 77.46 s: 39 steps of 2 s move 30.40 m, 38 move 28.88 m (independent
    # computation).
    assert result.steps == math.ceil(2 * math.sqrt(30.0 / A) / 2.0) == 39
    assert result.certified is True
    assert result.uncertified_steps == ()
    assert_plan_lands(result, view, x0)


def test_real_gravity_result_brackets_the_minimum_between_certified_and_flown_steps():
    view = make_view(N_500, A, dt=10.0, steps=60)
    x0 = np.array([0.0, -700.0, 0.0, 0.0, 0.0, 0.0])
    result = minimum_time(view, x0)
    # The step-grid LP first reaches the point at 36 steps; a 2 s grid needs 358 s, so the
    # continuous minimum lies in (350, 358] s. The best multiplier for 350 s passes the cell rule
    # at 64 cells per step but not at 8 (independent computation).
    assert result.steps == 36
    assert result.certified is True
    assert result.uncertified_steps == ()
    assert_plan_lands(result, view, x0)


def test_a_step_only_continuous_thrust_reaches_stays_uncertified():
    # 22.04 m with 0.05 m/s^2 and 2 s steps: continuous thrust arrives in 2 sqrt(440.8) =
    # 41.990 s, before step 21 ends, but 21 held steps move at most 0.2 * 110 = 22.00 m. No
    # multiplier can prove step 21 unreachable, and the step grid cannot reach it either.
    view = make_view(N_FREE, 0.05, dt=2.0, steps=40)
    x0 = np.array([0.0, -22.04, 0.0, 0.0, 0.0, 0.0])
    result = minimum_time(view, x0)
    assert result.steps == 22
    assert result.uncertified_steps == (21,)
    assert result.certified is False
    assert_plan_lands(result, view, x0)


def test_an_episode_too_short_to_arrive_is_certified_without_a_plan():
    # 30 steps of 2 s move at most 0.08 * 225 = 18 m of the 30 m.
    view = make_view(N_FREE, A, dt=2.0, steps=30)
    result = minimum_time(view, np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert result.steps is None
    assert result.plan is None
    assert result.certified is True
    assert result.uncertified_steps == ()


def test_a_start_at_rest_in_the_capture_ball_arrives_at_the_first_step():
    view = make_view(N_500, A, dt=10.0, steps=20, radius=1.0, speed=0.01)
    result = minimum_time(view, np.zeros(6))
    assert result.steps == 1
    assert result.certified is True
    assert result.uncertified_steps == ()
    assert_plan_lands(result, view, np.zeros(6))


def test_an_unknown_thrust_layout_raises():
    view = make_view(N_FREE, A, dt=1.0, steps=10, layout="cone")
    with pytest.raises(ValueError):
        minimum_time(view, X0_FREE)


def test_a_ball_thrust_layout_is_refused_because_the_proof_charges_the_box_dual():
    view = make_view(N_FREE, A, dt=1.0, steps=10, layout="ball")
    with pytest.raises(ValueError, match="box thrusters only"):
        minimum_time(view, X0_FREE)
    with pytest.raises(ValueError, match="box thrusters only"):
        verify_infeasible(view, X0_FREE, switching_certificate(5.0), 5)


def test_a_goal_with_a_thrust_inhibit_window_is_refused():
    view = make_view(N_FREE, A, dt=1.0, steps=10, thrust_inhibit=((2.0, 4.0),))
    with pytest.raises(ValueError, match="thrust inhibit"):
        minimum_time(view, X0_FREE)
    with pytest.raises(ValueError, match="thrust inhibit"):
        verify_infeasible(view, X0_FREE, switching_certificate(5.0), 5)


def test_a_goal_without_a_capture_ball_raises():
    view = AgentView(
        scenario_id="minimum-time-test",
        dt=1.0,
        max_steps=10,
        n=N_FREE,
        action_mode="accel",
        max_accel=A,
        thrust_limit="box",
        goal={"hold_point": (0.0, 0.0, 0.0)},
    )
    with pytest.raises(ValueError):
        minimum_time(view, X0_FREE)
    with pytest.raises(ValueError):
        verify_infeasible(view, X0_FREE, switching_certificate(5.0), 5)


@pytest.mark.parametrize(
    "start",
    [
        [0.0, -100.0, 0.0, 0.0, 0.0],
        [0.0, math.nan, 0.0, 0.0, 0.0, 0.0],
        [0.0, -100.0, 0.0, math.inf, 0.0, 0.0],
    ],
)
def test_a_bad_start_state_raises(start):
    view = free_view(10)
    with pytest.raises(ValueError):
        minimum_time(view, start)
    with pytest.raises(ValueError):
        verify_infeasible(view, start, switching_certificate(5.0), 5)


@pytest.mark.parametrize("accel", [0.0, -0.02, math.inf, math.nan])
def test_a_thrust_limit_that_is_not_positive_and_finite_raises(accel):
    view = make_view(N_FREE, accel, dt=1.0, steps=10)
    with pytest.raises(ValueError):
        minimum_time(view, X0_FREE)


@pytest.mark.parametrize(
    "nu",
    [np.ones(5), np.ones((2, 6)), np.r_[1.0, math.nan, 0.0, 0.0, 0.0, 0.0], np.full(6, math.inf)],
)
def test_a_bad_multiplier_raises(nu):
    with pytest.raises(ValueError):
        verify_infeasible(free_view(10), X0_FREE, nu, 5)


@pytest.mark.parametrize("step", [0, -1, 11])
def test_a_step_outside_the_episode_raises(step):
    with pytest.raises(ValueError):
        verify_infeasible(free_view(10), X0_FREE, switching_certificate(5.0), step)


# 30 m along R-bar across the target, from 15 m above to 15 m below it: the same double
# integrator move as the 30 m case, 39 steps of 2 s, but the straight path crosses the origin.
X0_ACROSS = np.array([15.0, 0.0, 0.0, 0.0, 0.0, 0.0])
HOLD_ACROSS = (-15.0, 0.0, 0.0)


def across_view(**goal):
    return make_view(N_FREE, A, dt=2.0, steps=60, hold_point=HOLD_ACROSS, **goal)


@pytest.fixture(scope="module")
def across_result():
    return minimum_time(across_view(), X0_ACROSS)


def test_the_arrival_step_ignores_the_dwell_time(across_result):
    result = minimum_time(across_view(hold_time=600.0), X0_ACROSS)
    assert across_result.steps == 39
    assert result.steps == across_result.steps
    assert result.certified is across_result.certified is True


def test_the_arrival_step_ignores_an_active_keep_out_sphere(across_result):
    view = AgentView(
        scenario_id="minimum-time-test",
        dt=2.0,
        max_steps=60,
        n=N_FREE,
        action_mode="accel",
        max_accel=A,
        thrust_limit="box",
        goal={**across_view().goal, "keep_out": 5.0},
        constraints=({"type": "keep_out", "radius": 5.0},),
    )
    result = minimum_time(view, X0_ACROSS)
    assert result.steps == across_result.steps == 39
    assert result.certified is True
    # The plan flies through the sphere: 39 steps at up to 0.78 m/s cover at most 1.6 m each,
    # so some step ends within 5 m of the target on the way across.
    x = X0_ACROSS
    ranges = []
    for u in result.plan:
        x = fly(view, x, [u])
        ranges.append(np.linalg.norm(x[:3]))
    assert min(ranges) < 5.0


def test_a_primer_that_vanishes_at_every_cell_edge_is_still_charged_between_them():
    # One step of four orbits with nu = (0, 0, n, 0, 0, 0): the primer is sin(n tau), zero at
    # every edge of the 8 cells. Thrust along z at a sign(sin(n tau)) ends at z = (a / n) times
    # the integral of |sin(n tau)|, 16 a / n^2, with vz = 0, so this hold point is reached by
    # continuous thrust and no multiplier may prove it out of reach.
    n = 1e-3
    view = make_view(n, A, dt=8 * math.pi / n, steps=1, hold_point=(0.0, 0.0, 16 * A / n**2))
    assert verify_infeasible(view, np.zeros(6), np.array([0.0, 0.0, n, 0.0, 0.0, 0.0]), 1) is False


def test_the_switching_multiplier_cannot_prove_the_flown_bang_bang_endpoint_of_3000_steps():
    # 1500 steps each way reach a dt^2 k^2 / 4 = 45000 m, the continuous bang-bang distance, so
    # the switching multiplier's margin is zero up to the capture ball. Its cell bound exceeds
    # the integral by 8.4e-4 of it at 8 cells per step (independent computation), so a verifier
    # that scales the bound by 0.999 accepts this step.
    k = 3000
    half = np.array([0.0, A, 0.0])
    phi, gamma = cw_discrete(N_FREE, k / 2)
    x_k = phi @ (gamma @ half) - gamma @ half
    view = make_view(
        N_FREE, A, dt=1.0, steps=k, hold_point=tuple(x_k[:3]), goal_velocity=tuple(x_k[3:])
    )
    assert verify_infeasible(view, np.zeros(6), switching_certificate(float(k)), k) is False


@pytest.mark.parametrize(
    ("radius", "speed", "boost"),
    [
        # Rest to rest, 70 steps each way and one coasting step, ends 0.6 m short at rest.
        (1.0, POINT, 0.0),
        # Thrust 0.6 / 1.41 of a on the middle step: the extra 1.41 m per unit of it lands on
        # the point at 0.0085 m/s.
        (1e-3, 0.05, 0.6 / 1.41),
    ],
)
def test_a_multiplier_cannot_prove_a_capture_ball_that_a_flown_plan_enters(radius, speed, boost):
    # The switching multiplier's margin for 141 steps is 0.595 m before the ball terms; a radius
    # of 1 m or a speed of 0.05 m/s against |nu_v| = 70.5 s takes it below zero.
    view = make_view(N_FREE, A, dt=1.0, steps=200, radius=radius, speed=speed)
    plan = np.zeros((141, 3))
    plan[:70, 1] = A
    plan[70, 1] = boost * A
    plan[71:, 1] = -A
    position, velocity = goal_errors(fly(view, X0_FREE, plan), view.goal)
    assert position < radius
    assert velocity < speed
    assert verify_infeasible(view, X0_FREE, switching_certificate(141.0), 141) is False


def test_a_capture_ball_of_zero_radius_is_never_entered_so_no_step_is_reported():
    # The capture test is strict, so no state is inside a ball of zero radius and speed. Steps
    # 1 to 38 move at most 28.88 m of the 30 m and stay proved; 39 and 40 reach the point
    # exactly and can be neither proved nor flown into the ball.
    view = make_view(N_FREE, A, dt=2.0, steps=40, radius=0.0, speed=0.0)
    result = minimum_time(view, np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert result.steps is None
    assert result.plan is None
    assert result.certified is False
    assert result.uncertified_steps == (39, 40)


def test_a_wide_capture_ball_is_entered_with_every_input_within_the_limit():
    # 5 m along V-bar with 0.05 m/s^2 and 2 s steps: 10 steps rest to rest move exactly 5 m.
    # 9 steps cannot cover the 4.7 m to the ball's edge even with continuous thrust and a final
    # speed of 0.01 m/s, 0.05 * 18^2 / 4 + 0.01 * 18 / 2 = 4.14 m. Aiming the plan at the
    # ball's centre pushes some inputs past the limit unless they are clipped.
    view = make_view(N_FREE, 0.05, dt=2.0, steps=20, radius=0.3, speed=0.01)
    x0 = np.array([0.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    result = minimum_time(view, x0)
    assert result.steps == 10
    assert result.certified is True
    assert result.uncertified_steps == ()
    assert_plan_lands(result, view, x0)


def test_an_episode_ending_at_the_uncertified_odd_step_is_not_certified():
    # The 22.04 m case cut at 21 steps: step 21 is neither reached nor proved, so the episode's
    # answer "no arrival" is not certified.
    view = make_view(N_FREE, 0.05, dt=2.0, steps=21)
    result = minimum_time(view, np.array([0.0, -22.04, 0.0, 0.0, 0.0, 0.0]))
    assert result.steps is None
    assert result.certified is False
    assert result.uncertified_steps == (21,)
