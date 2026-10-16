"""Validation of the CW dynamics. If these fail, every result is wrong."""

import numpy as np
import pytest

from vbar.dynamics import R_EARTH, cw_stm, mean_motion, propagate

N = mean_motion(R_EARTH + 500e3)
PERIOD = 2 * np.pi / N


def test_mean_motion_leo_period():
    # A 500 km circular orbit has a period of about 94.6 minutes.
    assert PERIOD / 60 == pytest.approx(94.6, abs=0.1)


def test_stm_identity_at_zero():
    assert np.allclose(cw_stm(N, 0.0), np.eye(6))


def test_stm_composition():
    # Propagating t1 then t2 must equal propagating t1 + t2.
    t1, t2 = 700.0, 1900.0
    assert np.allclose(cw_stm(N, t2) @ cw_stm(N, t1), cw_stm(N, t1 + t2), atol=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_numerical_matches_exact_solution(seed):
    # The core check: numerical integration agrees with the closed-form
    # solution to sub-millimeter accuracy over a full orbit.
    rng = np.random.default_rng(seed)
    s0 = np.r_[rng.uniform(-2000, 2000, 3), rng.uniform(-2, 2, 3)]
    numeric = propagate(s0, N, PERIOD)
    exact = cw_stm(N, PERIOD) @ s0
    assert np.max(np.abs(numeric[:3] - exact[:3])) < 1e-3  # meters
    assert np.max(np.abs(numeric[3:] - exact[3:])) < 1e-6  # m/s


def test_vbar_point_is_equilibrium():
    # A chaser sitting still on the V-bar (along-track) stays there forever.
    s0 = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(propagate(s0, N, PERIOD), s0, atol=1e-6)


def test_closed_ellipse_is_periodic():
    # With vy0 = -2 n x0 there is no along-track drift: after one orbit the
    # chaser is back where it started (a "football" orbit around the target).
    x0 = 100.0
    s0 = np.array([x0, 0.0, 0.0, 0.0, -2 * N * x0, 0.0])
    assert np.allclose(propagate(s0, N, PERIOD), s0, atol=1e-5)


def test_rbar_offset_drifts():
    # A chaser placed below the target (x < 0) with zero velocity drifts
    # along-track: lower orbit = faster. Classic CW behavior.
    s0 = np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s1 = propagate(s0, N, PERIOD)
    assert s1[1] > 0  # moved ahead in +y
    # Exact drift per orbit is 12*pi*|x0| (from the closed-form solution).
    assert s1[1] == pytest.approx(12 * np.pi * 100.0, rel=1e-6)


def test_constant_thrust_changes_velocity():
    # Over a short burn, delta-v is approximately accel * time.
    s0 = np.zeros(6)
    a = np.array([0.0, 0.01, 0.0])
    s1 = propagate(s0, N, 1.0, a)
    assert s1[4] == pytest.approx(0.01, rel=1e-3)
