import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from vbar.constraints import KeepOut, segment_minimum
from vbar.dynamics import EARTH, CWDynamics, cw_rhs, cw_stm, mean_motion

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
    found = segment_minimum(KEEP_OUT, m.state_in_last_segment, times, sampled)
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
