"""Steady-state Kalman filter and bias detectability on the exact zero-order-hold CW model.

The detectability test of constant biases and disturbances is Lemma 1.8 of Rawlings, Mayne and
Diehl, Model Predictive Control, 2nd ed., chapter 1. The covariance update is the Joseph form
(Crassidis and Junkins, Optimal Estimation of Dynamic Systems, section 3.3). Every number is for
a 500 km circular orbit of the WGS 84 Earth and a 10 s step: n = 1.106783e-3 rad/s.
"""

import numpy as np
import pytest

from vbar.agents.lqg import bias_detectable, kalman_gain
from vbar.dynamics import EARTH, cw_discrete, mean_motion

N = mean_motion(EARTH.radius + 500e3)
DT = 10.0
PHI, GAMMA = cw_discrete(N, DT)
CHANNELS = ("x", "y", "z", "vx", "vy", "vz")
C_FULL = np.eye(6)
# Start of the recursion: 10 m and 0.1 m/s one-sigma.
P0 = np.diag([100.0] * 3 + [1e-2] * 3)


def noise(sigma_accel=1e-6, sigma_pos=1.0, sigma_vel=1e-3):
    """Process covariance of a white acceleration held per step, and the measurement covariance."""
    qd = sigma_accel**2 * GAMMA @ GAMMA.T
    rv = np.diag([sigma_pos**2] * 3 + [sigma_vel**2] * 3)
    return qd, rv


def joseph_recursion(phi, c, qd, rv, p0, steps):
    """Time-varying filter from p0: gain, then the Joseph update, then the prediction."""
    identity = np.eye(len(phi))
    p_minus = p0
    for _ in range(steps):
        gain = p_minus @ c.T @ np.linalg.inv(c @ p_minus @ c.T + rv)
        a = identity - gain @ c
        p_plus = a @ p_minus @ a.T + gain @ rv @ gain.T
        p_next = phi @ p_plus @ phi.T + qd
        p_minus, last = p_next, (gain, p_minus, p_plus)
    return last


def relative(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


@pytest.mark.parametrize("channel", ["x", "z", "vx", "vy", "vz"])
def test_constant_bias_on_one_channel_is_detectable_except_along_track_position(channel):
    cd = C_FULL[:, [CHANNELS.index(channel)]]
    assert bias_detectable(PHI, C_FULL, np.zeros((6, 1)), cd) is True


def test_along_track_position_bias_is_not_detectable():
    # A constant y offset is itself a CW equilibrium (rank 6 of 7), so it looks like the state.
    cd = C_FULL[:, [1]]
    assert bias_detectable(PHI, C_FULL, np.zeros((6, 1)), cd) is False


def test_biases_on_all_six_channels_are_not_detectable_together():
    # Rank 11 of 12; dropping the along-track position bias leaves 11 of 11.
    assert bias_detectable(PHI, C_FULL, np.zeros((6, 6)), C_FULL) is False
    others = C_FULL[:, [0, 2, 3, 4, 5]]
    assert bias_detectable(PHI, C_FULL, np.zeros((6, 5)), others) is True


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_constant_acceleration_on_one_axis_is_detectable(axis):
    assert bias_detectable(PHI, C_FULL, GAMMA[:, [axis]], np.zeros((6, 1))) is True


def test_constant_acceleration_on_all_three_axes_is_detectable():
    assert bias_detectable(PHI, C_FULL, GAMMA, np.zeros((6, 3))) is True


def test_steady_state_gain_is_the_limit_of_the_joseph_recursion():
    qd, rv = noise()
    gain, p_minus, p_plus = kalman_gain(PHI, C_FULL, qd, rv)
    assert gain.shape == (6, 6)
    gain_k, p_minus_k, p_plus_k = joseph_recursion(PHI, C_FULL, qd, rv, P0, 3000)
    # Riccati solve against 3000 steps of the recursion: two paths, observed 2e-13 apart.
    assert relative(gain, gain_k) <= 1e-9
    assert relative(p_minus, p_minus_k) <= 1e-9
    assert relative(p_plus, p_plus_k) <= 1e-9


def test_bias_augmented_gain_is_the_limit_of_the_joseph_recursion():
    qd, rv = noise()
    # Radial position bias as a random walk of 1e-3 of the 1 m position sigma per step.
    phi_a = np.eye(7)
    phi_a[:6, :6] = PHI
    c_a = np.hstack([C_FULL, C_FULL[:, [0]]])
    qd_a = np.zeros((7, 7))
    qd_a[:6, :6] = qd
    qd_a[6, 6] = 1e-3**2
    gain, p_minus, _ = kalman_gain(phi_a, c_a, qd_a, rv)
    assert gain.shape == (7, 6)
    p0 = np.diag([100.0] * 3 + [1e-2] * 3 + [25.0])
    # The slow bias mode needs 20000 steps to settle: observed 1.2e-12 apart there, 6e-10 at
    # 10000 steps.
    gain_k, p_minus_k, _ = joseph_recursion(phi_a, c_a, qd_a, rv, p0, 20000)
    assert relative(gain, gain_k) <= 1e-9
    assert relative(p_minus, p_minus_k) <= 1e-9


@pytest.mark.parametrize(
    "sigmas",
    [
        {},
        {"sigma_pos": 1e-3, "sigma_vel": 1e-6},
        {"sigma_accel": 1e-3},
    ],
    ids=["nav1", "cheap-measurements", "loud-process"],
)
def test_steady_state_is_a_fixed_point_of_one_filter_cycle(sigmas):
    qd, rv = noise(**sigmas)
    gain, p_minus, p_plus = kalman_gain(PHI, C_FULL, qd, rv)
    # Each identity recomputes one quantity from the others; observed 2e-15 relative.
    expected_gain = p_minus @ C_FULL.T @ np.linalg.inv(C_FULL @ p_minus @ C_FULL.T + rv)
    assert relative(gain, expected_gain) <= 1e-9
    a = np.eye(6) - gain @ C_FULL
    assert relative(p_plus, a @ p_minus @ a.T + gain @ rv @ gain.T) <= 1e-9
    assert relative(p_minus, PHI @ p_plus @ PHI.T + qd) <= 1e-9


@pytest.mark.parametrize(
    "sigmas",
    [
        {},
        {"sigma_pos": 1e-3, "sigma_vel": 1e-6},
        {"sigma_accel": 1e-3},
    ],
    ids=["nav1", "cheap-measurements", "loud-process"],
)
def test_covariances_are_symmetric_and_positive_definite(sigmas):
    qd, rv = noise(**sigmas)
    _, p_minus, p_plus = kalman_gain(PHI, C_FULL, qd, rv)
    for p in (p_minus, p_plus):
        assert np.max(np.abs(p - p.T)) <= 1e-12 * np.max(np.abs(p))
        # Every state is measured with finite noise and excited by the process noise, so the
        # covariance is definite.
        assert np.linalg.eigvalsh(p)[0] > 0.0


def test_zero_process_noise_is_refused():
    _, rv = noise()
    with pytest.raises(ValueError, match="process noise must be positive"):
        kalman_gain(PHI, C_FULL, np.zeros((6, 6)), rv)


def test_process_noise_that_misses_an_oscillating_axis_is_refused():
    # Without noise on the normal axis its undamped oscillation (eigenvalues on the unit circle)
    # is never excited, so the steady-state gain on it would be zero.
    _, rv = noise()
    in_plane = GAMMA[:, :2]
    with pytest.raises(ValueError):
        kalman_gain(PHI, C_FULL, 1e-12 * in_plane @ in_plane.T, rv)
