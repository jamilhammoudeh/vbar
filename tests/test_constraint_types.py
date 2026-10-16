import numpy as np
import pytest

from vbar.constraints import ClosingSpeedLimit, SpeedLimit, closing_speed, segment_minimum
from vbar.dynamics import EARTH, CWDynamics, mean_motion

N = mean_motion(EARTH.radius + 500e3)
# Dunlap et al. 2021, eq. 14: v0 = 0.2 m/s, v1 = 4 n.
SPEED = SpeedLimit(0.2, 4 * N)
CLOSING = ClosingSpeedLimit(0.2, 4 * N)


def flyby(seed):
    rng = np.random.default_rng(seed)
    position = rng.normal(size=3)
    position *= rng.uniform(20.0, 200.0) / np.linalg.norm(position)
    direction = -position / np.linalg.norm(position) + rng.normal(scale=0.5, size=3)
    state = np.r_[position, rng.uniform(0.1, 3.0) * direction / np.linalg.norm(direction)]
    accel = rng.uniform(-0.02, 0.02, 3) if seed % 2 else np.zeros(3)
    return state, accel, rng.uniform(5.0, 30.0)


def found_and_dense(constraint, seed, samples=20):
    state, accel, duration = flyby(seed)
    m = CWDynamics(N)
    m.reset(state)
    _, sampled = m.advance(accel, duration, n_samples=samples)
    times = np.linspace(0.0, duration, samples)
    thrust = float(np.linalg.norm(accel))
    found = segment_minimum(constraint, m.state_in_last_segment, times, sampled, thrust, N)
    grid = np.linspace(0.0, duration, 4001)
    dense = np.array([m.state_in_last_segment(t) for t in grid])
    return found, constraint.margin(dense, grid)


def test_speed_limit_margin_is_the_allowed_minus_the_actual_speed():
    state = np.array([0.0, -100.0, 0.0, 0.3, 0.4, 0.0])
    assert SPEED.margin(state) == pytest.approx(0.2 + 400 * N - 0.5)


def test_closing_speed_counts_only_motion_toward_the_target():
    approaching = np.array([0.0, -100.0, 0.0, 0.1, 0.3, 0.0])
    receding = np.array([0.0, -100.0, 0.0, 0.1, -0.3, 0.0])
    crossing = np.array([0.0, -100.0, 0.0, 0.5, 0.0, 0.0])
    assert closing_speed(approaching) == pytest.approx(0.3)
    assert closing_speed(receding) == 0.0
    assert closing_speed(crossing) == 0.0
    assert CLOSING.margin(approaching) == pytest.approx(0.2 + 400 * N - 0.3)


def test_closing_speed_rate_bound_needs_a_range_bound():
    assert CLOSING.rate_bound(1.0, 0.0) == np.inf
    assert CLOSING.rate_bound(1.0, 0.01, rho_min=10.0) == pytest.approx(4 * N + 0.1 + 0.01)


@pytest.mark.parametrize("constraint", [SPEED, CLOSING], ids=lambda c: c.name)
@pytest.mark.parametrize("seed", range(12))
def test_certificate_and_refined_minimum_bracket_the_dense_minimum(constraint, seed):
    found, dense = found_and_dense(constraint, seed)
    assert np.isfinite(found.lower_bound)
    assert found.lower_bound <= dense.min() + 1e-12
    # 4001 samples over at most 30 s miss a smooth minimum by far less than 1e-5 m/s.
    assert found.margin <= dense.min() + 1e-6
    assert found.margin >= found.lower_bound
