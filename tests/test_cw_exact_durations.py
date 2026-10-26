import numpy as np
import pytest

from vbar.dynamics import EARTH, CWDynamics, cw_discrete, mean_motion

N = mean_motion(EARTH.radius + 500e3)
X0 = np.array([100.0, 50.0, 0.0, 1.0, -2.0, 0.5])


@pytest.mark.parametrize("dt", [4e-10, 5e-10, 1.4e-9, 0.1 + 4e-10])
def test_a_step_moves_the_state_by_its_exact_duration(dt):
    # The displacement is about 1e-9 m per 5e-10 s; 1e-12 m is a hundred ulp of a 100 m range.
    model = CWDynamics(N)
    model.reset(X0)
    got, _ = model.advance(np.zeros(3), dt)
    phi, _ = cw_discrete(N, dt)
    assert np.linalg.norm(got[:3] - (phi @ X0)[:3]) <= 1e-12


@pytest.mark.parametrize("tau", [1e-10, 2.5e-9, 3.3333333333])
def test_state_in_last_segment_is_at_the_exact_time_asked_for(tau):
    model = CWDynamics(N)
    model.reset(X0)
    model.advance(np.zeros(3), 10.0)
    phi, _ = cw_discrete(N, tau)
    assert np.linalg.norm(model.state_in_last_segment(tau)[:3] - (phi @ X0)[:3]) <= 1e-12


@pytest.mark.parametrize("steps", [30000, pytest.param(300000, marks=pytest.mark.slow)])
def test_many_third_of_a_second_steps_drift_only_by_rounding(steps):
    # Each step's product rounds by a few eps of the state and CW growth is secular, so the sum
    # stays below steps eps |r|; repeating the exact one-third-second matrices gave 7.5e-9 m at
    # 3e4 steps and 1.0e-7 m at 3e5, against bounds of 3.9e-7 m and 3.6e-5 m.
    model = CWDynamics(N)
    model.reset(X0)
    for _ in range(steps):
        state, _ = model.advance(np.zeros(3), 1.0 / 3.0)
    phi, _ = cw_discrete(N, steps / 3.0)
    r = (phi @ X0)[:3]
    drift = np.linalg.norm(state[:3] - r)
    assert drift <= steps * np.finfo(float).eps * max(np.linalg.norm(r), np.linalg.norm(X0[:3]))
