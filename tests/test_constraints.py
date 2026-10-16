import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from vbar.constraints import (
    KeepOut,
    interval_bounds,
    range_lower_bounds,
    segment_minimum,
    violation_intervals,
)
from vbar.dynamics import EARTH, CWDynamics, TwoBodyDynamics, cw_rhs, cw_stm, mean_motion

N = mean_motion(EARTH.radius + 500e3)
KEEP_OUT = KeepOut(50.0)


def closest_approach(state, accel, duration):
    """Minimum range over a segment from an independent reference.

    The closed-form STM when coasting, DOP853 at rtol 1e-13 when thrusting; the minimum is where
    r . v changes sign.
    """
    if np.any(accel):
        sol = solve_ivp(
            cw_rhs,
            (0.0, duration),
            state,
            method="DOP853",
            args=(N, accel),
            rtol=1e-13,
            atol=1e-14,
            dense_output=True,
        )

    def at(t):
        return sol.sol(t) if np.any(accel) else cw_stm(N, t) @ state

    ts = np.linspace(0.0, duration, 2001)
    ranges = [np.linalg.norm(at(t)[:3]) for t in ts]
    k = int(np.argmin(ranges))
    if k in (0, len(ts) - 1):
        return ranges[k]
    t = brentq(lambda t: at(t)[:3] @ at(t)[3:], ts[k - 1], ts[k + 1], xtol=1e-12)
    return np.linalg.norm(at(t)[:3])


def detected_minimum(state, accel, duration, samples=20):
    m = CWDynamics(N)
    m.reset(state)
    _, sampled = m.advance(accel, duration, n_samples=samples)
    times = np.linspace(0.0, duration, samples)
    thrust = float(np.linalg.norm(accel))
    found = segment_minimum(KEEP_OUT, m.state_in_last_segment, times, sampled, thrust, N)
    return found, float(KEEP_OUT.margin(sampled).min())


def test_keep_out_margin_is_range_minus_radius():
    states = np.array([[30.0, 40.0, 0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]])
    assert np.allclose(KEEP_OUT.margin(states), [0.0, -40.0])


def test_fast_flyby_minimum_is_found_between_samples():
    state = np.array([0.0, 60.0, 5.0, 0.0, -12.0, 0.0])
    found, sampled = detected_minimum(state, np.zeros(3), 10.0)
    exact = closest_approach(state, np.zeros(3), 10.0) - 50.0
    assert found.margin == pytest.approx(exact, abs=1e-3)
    # The best of 20 samples misses the passage by most of a metre.
    assert sampled - exact > 0.1


@pytest.mark.parametrize("seed", range(20))
def test_minimum_range_matches_the_exact_cw_trajectory_to_a_millimetre(seed):
    rng = np.random.default_rng(seed)
    position = rng.normal(size=3)
    position *= rng.uniform(20.0, 200.0) / np.linalg.norm(position)
    speed = rng.uniform(0.5, 20.0)
    direction = -position / np.linalg.norm(position) + rng.normal(scale=0.3, size=3)
    state = np.r_[position, speed * direction / np.linalg.norm(direction)]
    accel = rng.uniform(-0.02, 0.02, 3) if seed % 2 else np.zeros(3)
    duration = rng.uniform(5.0, 30.0)
    found, _ = detected_minimum(state, accel, duration)
    assert found.margin == pytest.approx(closest_approach(state, accel, duration) - 50.0, abs=1e-3)


def test_minimum_at_a_segment_end_is_the_end_sample():
    state = np.array([0.0, 300.0, 0.0, 0.0, -1.0, 0.0])
    found, sampled = detected_minimum(state, np.zeros(3), 10.0)
    assert found.margin == sampled
    assert found.t == pytest.approx(10.0)


def random_flyby(seed):
    rng = np.random.default_rng(seed)
    position = rng.normal(size=3)
    position *= rng.uniform(20.0, 200.0) / np.linalg.norm(position)
    direction = -position / np.linalg.norm(position) + rng.normal(scale=0.3, size=3)
    state = np.r_[position, rng.uniform(0.5, 20.0) * direction / np.linalg.norm(direction)]
    accel = rng.uniform(-0.02, 0.02, 3) if seed % 2 else np.zeros(3)
    return state, accel, rng.uniform(5.0, 30.0)


@pytest.mark.parametrize("seed", range(20))
def test_certified_bound_lies_below_the_exact_minimum_within_l_h_over_2(seed):
    state, accel, duration = random_flyby(seed)
    found, _ = detected_minimum(state, accel, duration)
    exact = closest_approach(state, accel, duration) - 50.0
    assert found.lower_bound <= exact <= found.margin + 1e-9
    # Never looser than the plan's form min_i m_i - L h / 2 with the largest interval L.
    m = CWDynamics(N)
    m.reset(state)
    _, sampled = m.advance(accel, duration, n_samples=20)
    h = duration / 19
    speed, _ = interval_bounds(sampled, h, float(np.linalg.norm(accel)), N)
    assert found.lower_bound >= KEEP_OUT.margin(sampled).min() - speed.max() * h / 2 - 1e-12


def test_bound_flags_a_dip_between_samples_and_refinement_finds_it():
    # A 12 m/s pass 5 m from the target, sampled only at the ends of a 10 s segment.
    state = np.array([0.0, 60.0, 5.0, 0.0, -12.0, 0.0])
    found, sampled = detected_minimum(state, np.zeros(3), 10.0, samples=2)
    assert sampled > 0.0 and found.lower_bound < 0.0
    assert found.margin == pytest.approx(
        closest_approach(state, np.zeros(3), 10.0) - 50.0, abs=1e-3
    )


@pytest.mark.parametrize("seed", range(6))
def test_speed_bounds_and_certificate_hold_on_two_body_motion(seed):
    rng = np.random.default_rng(seed)
    m = TwoBodyDynamics(inclination=0.9, j2=bool(seed % 2))
    position = rng.normal(size=3)
    position *= rng.uniform(100.0, 5000.0) / np.linalg.norm(position)
    state = np.r_[position, rng.uniform(-5.0, 5.0, 3)]
    accel = rng.uniform(-0.02, 0.02, 3)
    m.reset(state)
    duration = 60.0
    _, sampled = m.advance(accel, duration, n_samples=20)
    times = np.linspace(0.0, duration, 20)
    found = segment_minimum(
        KEEP_OUT, m.state_in_last_segment, times, sampled, 0.02 * np.sqrt(3), m.n
    )
    dense = np.array([m.state_in_last_segment(t) for t in np.linspace(0.0, duration, 3801)])
    assert found.lower_bound <= KEEP_OUT.margin(dense).min() <= found.margin + 1e-6
    speed, _ = interval_bounds(sampled, times[1], 0.02 * np.sqrt(3), m.n)
    dense_speed = np.linalg.norm(dense[:, 3:], axis=1)
    for i in range(19):
        assert dense_speed[200 * i : 200 * (i + 1) + 1].max() <= speed[i]


def test_coarse_sampling_gives_no_finite_bound_rather_than_a_wrong_one():
    # 1000 s between samples is past the 350 s at which the speed bound stops closing at 500 km;
    # the bound used to come out above the true minimum here (212.4 m against 208.3 m).
    m = CWDynamics(N)
    m.reset([0.0, 400.0, 0.0, 0.3, 0.0, 0.0])
    _, sampled = m.advance(np.zeros(3), 2000.0, n_samples=3)
    times = np.linspace(0.0, 2000.0, 3)
    speed, accel = interval_bounds(sampled, times[1], 0.0, N)
    assert np.all(np.isinf(speed)) and np.all(np.isinf(accel))
    found = segment_minimum(KEEP_OUT, m.state_in_last_segment, times, sampled, 0.0, N)
    dense = np.array([m.state_in_last_segment(t) for t in np.linspace(0.0, 2000.0, 20001)])
    exact = KEEP_OUT.margin(dense).min()
    assert found.lower_bound == -np.inf
    assert found.margin == pytest.approx(exact, abs=1e-3)


def test_flat_margin_is_not_refined_past_the_resolution():
    # On the V-bar the range is constant; every interval's bound is within 1 mm of the samples.
    found, sampled = detected_minimum(np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), 10.0)
    assert found.margin == sampled
    assert 0.0 < found.margin - found.lower_bound < KEEP_OUT.resolution


@pytest.mark.parametrize("seed", range(10))
def test_range_bound_lies_below_the_dense_range_on_every_interval(seed):
    state, accel, duration = random_flyby(seed)
    m = CWDynamics(N)
    m.reset(state)
    _, sampled = m.advance(accel, duration, n_samples=20)
    h = duration / 19
    speed, _ = interval_bounds(sampled, h, float(np.linalg.norm(accel)), N)
    lower = range_lower_bounds(sampled, h, speed)
    dense = np.array([m.state_in_last_segment(t) for t in np.linspace(0.0, duration, 1901)])
    dense_range = np.linalg.norm(dense[:, :3], axis=1)
    for i in range(19):
        assert dense_range[100 * i : 100 * (i + 1) + 1].min() >= lower[i]


def test_range_bound_is_zero_when_the_speed_bound_is_infinite():
    states = np.array([[0.0, 400.0, 0.0, 0.3, 0.0, 0.0], [0.0, 500.0, 0.0, 0.3, 0.0, 0.0]])
    assert range_lower_bounds(states, 1000.0, np.array([np.inf]))[0] == 0.0


def test_bounds_on_stacked_trajectories_equal_the_bounds_on_each():
    trajectories = []
    for seed in range(4):
        state, _, _ = random_flyby(seed)
        m = CWDynamics(N)
        m.reset(state)
        trajectories.append(m.advance(np.zeros(3), 10.0, n_samples=20)[1])
    stacked = np.array(trajectories)
    thrust = np.array([[0.0], [0.01], [0.02], [0.03]])
    speed, accel = interval_bounds(stacked, 10.0 / 19, thrust, N)
    lower = range_lower_bounds(stacked, 10.0 / 19, speed)
    for k, samples in enumerate(trajectories):
        one_speed, one_accel = interval_bounds(samples, 10.0 / 19, float(thrust[k, 0]), N)
        assert np.array_equal(speed[k], one_speed) and np.array_equal(accel[k], one_accel)
        assert np.array_equal(lower[k], range_lower_bounds(samples, 10.0 / 19, one_speed))


def exact_crossings(state, duration):
    """Times a coasting CW trajectory crosses the 50 m sphere, from the closed-form STM."""
    ts = np.linspace(0.0, duration, 4001)
    margins = [np.linalg.norm((cw_stm(N, t) @ state)[:3]) - 50.0 for t in ts]
    return [
        brentq(lambda t: np.linalg.norm((cw_stm(N, t) @ state)[:3]) - 50.0, ts[i], ts[i + 1])
        for i in range(len(ts) - 1)
        if (margins[i] < 0.0) != (margins[i + 1] < 0.0)
    ]


def detected_violations(state, duration, samples=20):
    m = CWDynamics(N)
    m.reset(state)
    _, sampled = m.advance(np.zeros(3), duration, n_samples=samples)
    times = np.linspace(0.0, duration, samples)
    found = segment_minimum(KEEP_OUT, m.state_in_last_segment, times, sampled, 0.0, N, 100.0)
    return violation_intervals(KEEP_OUT, m.state_in_last_segment, times, sampled, found, 100.0)


def test_violation_interval_is_bounded_by_the_exact_entry_and_exit():
    state = np.array([3.0, 100.0, -2.0, 0.0, -10.0, 0.0])
    ((start, end),) = detected_violations(state, 20.0)
    entry, exit_ = exact_crossings(state, 20.0)
    assert start - 100.0 == pytest.approx(entry, abs=1e-8)
    assert end - 100.0 == pytest.approx(exit_, abs=1e-8)


def test_violation_between_two_samples_is_found_from_the_refined_minimum():
    state = np.array([0.0, 60.0, 5.0, 0.0, -12.0, 0.0])
    ((start, end),) = detected_violations(state, 10.0, samples=2)
    entry, exit_ = exact_crossings(state, 10.0)
    assert (start - 100.0, end - 100.0) == pytest.approx((entry, exit_), abs=1e-8)


def test_violation_open_at_either_end_of_the_segment_runs_to_that_end():
    leaving = np.array([0.0, 10.0, 0.0, 0.0, 10.0, 0.0])
    ((start, end),) = detected_violations(leaving, 10.0)
    assert start == 100.0
    assert end - 100.0 == pytest.approx(exact_crossings(leaving, 10.0)[0], abs=1e-8)
    staying = np.array([0.0, 20.0, 0.0, 0.0, 0.1, 0.0])
    assert detected_violations(staying, 10.0) == ((100.0, 110.0),)
    clear = np.array([0.0, 300.0, 0.0, 0.0, -1.0, 0.0])
    assert detected_violations(clear, 10.0) == ()
