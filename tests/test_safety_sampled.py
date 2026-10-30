import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vbar.constraints import KeepOut, segment_minimum
from vbar.dynamics import EARTH, CWDiscretizer, cw_discrete, cw_states, mean_motion
from vbar.safety.sampled import held_segment_bound, line_search
from vbar.safety.tightening import BallError, BoxError, keep_out_margin

N = mean_motion(EARTH.radius + 500e3)
R = 50.0
M = 20
FINITE = {"allow_nan": False, "allow_infinity": False, "allow_subnormal": False}

# Passing at 60 m, grazing at about 51 m, still closing at the step end, and already inside.
CASES = [
    (np.array([60.0, 30.0, 0.0, 0.0, -1.0, 0.0]), np.array([0.0, 0.005, 0.0]), 60.0),
    (np.array([51.0, 8.0, 3.0, 0.02, -0.8, -0.05]), np.array([0.01, -0.02, 0.0]), 20.0),
    (np.array([20.0, 120.0, -15.0, -0.05, -0.6, 0.02]), np.array([0.0, 0.01, 0.0]), 10.0),
    (np.array([10.0, 40.0, 0.0, 0.1, 0.0, 0.0]), np.zeros(3), 5.0),
]


def certified_reference(x, u, dt, m=M):
    """segment_minimum's certified bound on the exact sample-and-hold samples of the step."""
    phi, gamma = CWDiscretizer(N).sample_matrices(dt, m)
    states = phi @ x + gamma @ u

    def state_at(tau):
        return cw_states(N, np.array([tau]), x[None], u[None])[0]

    found = segment_minimum(
        KeepOut(R), state_at, np.linspace(0.0, dt, m), states, float(np.linalg.norm(u)), N
    )
    return found.lower_bound


def dense_margin(x, u, dt, count=2001):
    phi, gamma = cw_discrete(N, np.linspace(0.0, dt, count))
    states = phi @ x + gamma @ u
    return np.linalg.norm(states[:, :3], axis=1) - R, np.linalg.norm(states[:, 3:], axis=1)


def bound(x, u, dt, **kwargs):
    return held_segment_bound(x, u, dt, M, R, N, **kwargs)


@pytest.mark.parametrize(("x", "u", "dt"), CASES)
def test_held_segment_bound_is_segment_minimum_certified_bound(x, u, dt):
    # The same samples and the same bounds, so equality is exact: a regression check that the
    # held step is judged by the environment's own certificate.
    assert bound(x, u, dt) == certified_reference(x, u, dt)


@pytest.mark.parametrize(("x", "u", "dt"), CASES)
def test_held_segment_bound_is_valid_and_tight_against_a_dense_exact_step(x, u, dt):
    margin, speed = dense_margin(x, u, dt)
    value = bound(x, u, dt)
    assert value <= margin.min() + 1e-9
    # Within the first-order loss over one sample interval.
    assert value >= margin.min() - speed.max() * dt / (M - 1) - 1e-9


@settings(max_examples=25, deadline=None)
@given(
    r=st.lists(st.floats(-300.0, 300.0, **FINITE), min_size=3, max_size=3),
    v=st.lists(st.floats(-0.5, 0.5, **FINITE), min_size=3, max_size=3),
    u=st.lists(st.floats(-0.02, 0.02, **FINITE), min_size=3, max_size=3),
    dt=st.sampled_from([1.0, 2.0, 5.0, 10.0, 20.0]),
)
def test_held_segment_bound_never_exceeds_the_true_step_minimum(r, v, u, dt):
    x, u = np.array(r + v), np.array(u)
    margin, _ = dense_margin(x, u, dt)
    assert bound(x, u, dt) <= margin.min() + 1e-9 * max(1.0, float(np.linalg.norm(x[:3])))


def closing_case():
    """A step that closes the whole way, so the lowest sample interval is the last one."""
    return np.array([0.0, 120.0, 0.0, 0.0, -0.8, 0.0]), np.array([0.0, 0.005, 0.0]), 10.0


@pytest.mark.parametrize(
    "error",
    [BallError(r_pos=1.0, r_vel=0.01), BoxError(r_pos=0.5, r_vel=0.005)],
)
def test_navigation_error_lowers_the_bound_by_the_keep_out_tightening(error):
    x, u, dt = closing_case()
    phi = cw_discrete(N, np.linspace(0.0, dt, M))[0]
    position = [keep_out_margin(p[:3], error) for p in phi]
    velocity = [keep_out_margin(p[3:], error) for p in phi]
    drop = bound(x, u, dt) - bound(x, u, dt, error=error)
    # At least the range error at the start of the lowest interval, at most the largest range
    # error plus its rate over one interval.
    assert drop >= position[-2] * (1 - 1e-12)
    assert drop <= max(position) + max(velocity) * dt / (M - 1) + 1e-12


def test_navigation_tightened_bound_holds_for_every_state_in_the_error_box():
    x, u, dt = closing_case()
    error = BoxError(r_pos=0.5, r_vel=0.005)
    value = bound(x, u, dt, error=error)
    signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * 6)).reshape(6, -1).T
    for e in signs * np.array([0.5] * 3 + [0.005] * 3):
        assert value <= dense_margin(x + e, u, dt, count=401)[0].min() + 1e-9


def test_execution_error_lowers_the_bound_by_the_held_response():
    x, u, dt = closing_case()
    da_max = 2e-4
    gamma = cw_discrete(N, np.linspace(0.0, dt, M))[1]
    position = [np.linalg.norm(g[:3], 2) * da_max for g in gamma]
    velocity = [np.linalg.norm(g[3:], 2) * da_max for g in gamma]
    drop = bound(x, u, dt) - bound(x, u, dt, da_max=da_max)
    assert drop >= position[-2] * (1 - 1e-12)
    assert drop <= max(position) + max(velocity) * dt / (M - 1) + 1e-12
    rng = np.random.default_rng(5)
    for _ in range(20):
        du = rng.standard_normal(3)
        du *= da_max / np.linalg.norm(du)
        assert bound(x, u, dt, da_max=da_max) <= dense_margin(x, u + du, dt, 401)[0].min() + 1e-9


def test_larger_error_sets_give_lower_bounds():
    x, u, dt = closing_case()
    values = [bound(x, u, dt, error=BallError(r_pos=r, r_vel=0.01 * r)) for r in (0.0, 0.5, 2.0)]
    assert values[0] == pytest.approx(bound(x, u, dt), rel=1e-12)
    assert values[0] > values[1] > values[2]


# Line search toward a backup (Ames et al., Control Barrier Functions: Theory and Applications,
# ECC 2019, sec. III): u = theta u_c + (1 - theta) beta, theta = 1 tried first, then bisection.
U_C = np.array([0.02, 0.0, 0.0])
BETA = np.array([0.0, -0.02, 0.0])


class Scripted:
    """Acceptance by theta, recovered from the command; records every theta it is asked about."""

    def __init__(self, rule):
        self.rule = rule
        self.asked = []

    def __call__(self, u):
        theta = float(u[0] / U_C[0])
        assert np.allclose(u, theta * U_C + (1 - theta) * BETA, rtol=0, atol=1e-15)
        self.asked.append(theta)
        return self.rule(theta)


def test_line_search_keeps_the_candidate_when_it_is_accepted():
    accept = Scripted(lambda theta: True)
    found = line_search(accept, U_C, BETA)
    assert found.theta == 1.0
    assert np.array_equal(found.u, U_C)
    assert found.checks == 1
    assert accept.asked == [1.0]


def test_line_search_falls_back_to_the_backup_when_nothing_else_is_accepted():
    accept = Scripted(lambda theta: False)
    found = line_search(accept, U_C, BETA)
    assert found.theta == 0.0
    assert np.array_equal(found.u, BETA)
    assert found.checks == 13
    # The backup's own safety is the barrier's theorem, so theta = 0 is never asked about.
    assert 0.0 not in accept.asked


def test_line_search_bisects_twelve_times_to_the_largest_dyadic_below_a_monotone_threshold():
    accept = Scripted(lambda theta: theta <= 0.3)
    found = line_search(accept, U_C, BETA)
    # 12 halvings of [0, 1] end on a multiple of 2^-12; floor(0.3 * 4096) = 1228.
    assert found.theta == 1228 / 4096
    assert np.allclose(found.u, found.theta * U_C + (1 - found.theta) * BETA, rtol=0, atol=1e-15)
    assert found.checks == 13
    assert accept.asked[:3] == [1.0, 0.5, 0.25]


def test_line_search_iterations_set_the_resolution():
    accept = Scripted(lambda theta: theta <= 0.3)
    found = line_search(accept, U_C, BETA, iterations=3)
    assert found.theta == 0.25
    assert found.checks == 4


def test_line_search_need_not_find_the_largest_accepted_theta_when_acceptance_is_not_monotone():
    # Accepted below 0.1 and on [0.6, 0.7]; the first midpoint 0.5 fails, so bisection never
    # visits the upper window.
    def window(theta):
        return theta <= 0.1 or 0.6 <= theta <= 0.7

    accept = Scripted(window)
    found = line_search(accept, U_C, BETA)
    assert found.theta == 409 / 4096
    assert window(0.65)
    assert found.theta < 0.6


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_held_segment_bound_refuses_a_sample_count_that_is_not_finite(value):
    with pytest.raises(ValueError):
        held_segment_bound(*CASES[0], value, R, N)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_line_search_refuses_an_iteration_count_that_is_not_finite(value):
    with pytest.raises(ValueError):
        line_search(Scripted(lambda theta: True), U_C, BETA, iterations=value)
