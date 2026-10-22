"""Disturbance-feasibility audit of the hold scenarios (Rawlings, Mayne and Diehl, Model
Predictive Control, 2nd ed., sections 2.9 and 3.1): a hold box must contain, with margin, the
dispersion the declared disturbances force on a good controller, or the task is unsolvable at
the claimed rate."""

import math

import numpy as np
import pytest
from scenario_references import reference_for
from scipy.linalg import solve_discrete_are

from vbar import run_episode
from vbar.dynamics import cw_discrete
from vbar.scenarios import make


def lqr_gain(n, dt, pos_weight=10.0, vel_weight=0.1, accel_weight=0.02):
    """Discrete LQR gain for the Clohessy-Wiltshire step, Bryson weights (m, m/s, m/s^2)."""
    phi, gamma = cw_discrete(n, dt)
    q = np.diag([1 / pos_weight**2] * 3 + [1 / vel_weight**2] * 3)
    r = np.eye(3) / accel_weight**2
    p = solve_discrete_are(phi, gamma, q, r)
    return np.linalg.solve(gamma.T @ p @ gamma + r, gamma.T @ p @ phi)


def stationary_dispersion(n, dt, sigma_pos, sigma_vel):
    """1-sigma per axis (m, m/s) of a hold under the LQR with raw noisy measurements fed back."""
    phi, gamma = cw_discrete(n, dt)
    k = lqr_gain(n, dt)
    closed = phi - gamma @ k
    noise = np.diag([sigma_pos**2] * 3 + [sigma_vel**2] * 3)
    cov = np.zeros((6, 6))
    for _ in range(20000):
        cov = closed @ cov @ closed.T + (gamma @ k) @ noise @ (gamma @ k).T
    return np.sqrt(np.diag(cov))


def test_nav1_into_the_reference_lqr_gives_the_audited_stationary_dispersion():
    # The audit figure (Rawlings, Mayne and Diehl, sec. 2.9): raw :nav1 noise (1 m, 1 mm/s) into
    # the 10 m / 0.1 m/s
    # LQR gives a stationary 1-sigma of 0.22 m and 8.2 mm/s per axis at a 10 s step.
    env = make("approach-vbar")
    sigma = stationary_dispersion(env.n, env.cfg.dt, 1.0, 1e-3)
    assert np.allclose(sigma[:3], 0.22, atol=0.03)
    assert np.allclose(sigma[3:], 8.2e-3, atol=1e-3)


@pytest.mark.parametrize(
    "name",
    [
        "approach-vbar",
        "approach-rbar",
        "approach-hbar",
        "closing",
        "rbar-acquisition",
        "orbt-rbar",
        "rpm-hold",
        "torva",
    ],
)
def test_the_hold_box_contains_the_three_sigma_dispersion_with_margin(name):
    # The per-axis 3-sigma of the stationary dispersion (0.66 m, 0.025 m/s) sits inside the
    # capture ball and the capture speed with a factor of 1.5 to spare, and the 3-sigma norm over
    # the three axes (1.13 m, 0.043 m/s) is still inside; the speed criterion is the tight one.
    env = make(name)
    goal = env.agent_view().goal
    sigma = stationary_dispersion(env.n, env.cfg.dt, 1.0, 1e-3)
    assert 3.0 * np.max(sigma[:3]) * 1.5 < goal["capture_radius"]
    assert 3.0 * np.max(sigma[3:]) * 1.5 < goal["capture_speed"], (name, sigma)
    assert 3.0 * math.sqrt(np.sum(sigma[:3] ** 2)) < goal["capture_radius"]
    assert 3.0 * math.sqrt(np.sum(sigma[3:] ** 2)) < goal["capture_speed"]


def test_the_grapple_box_is_declared_for_a_tighter_sensor_than_nav1():
    # 35 ft station keeping at 1 m and 0.01 m/s is not reachable through 1 m white noise: the
    # 3-sigma speed dispersion alone is 0.043 m/s. The spec says so and its nav1 variant is
    # a hardness track, not a solvable one.
    env = make("hst-rbar-grapple")
    sigma = stationary_dispersion(env.n, env.cfg.dt, 1.0, 1e-3)
    assert 3.0 * math.sqrt(np.sum(sigma[3:] ** 2)) > env.agent_view().goal["capture_speed"]


def test_an_along_track_position_bias_cannot_be_told_from_a_hold_offset():
    # Rawlings, Mayne and Diehl, lemma 1.8: a constant along-track position bias is undetectable on
    # the Clohessy-Wiltshire model with full-state measurements (rank 6 of 7), so a biased y
    # reads as another V-bar equilibrium; biases on the other channels are detectable.
    env = make("approach-vbar")
    phi, _ = cw_discrete(env.n, env.cfg.dt)
    for axis, detectable in ((0, True), (1, False), (2, True), (3, True), (4, True), (5, True)):
        a = np.zeros((7, 7))
        a[:6, :6] = phi
        a[6, 6] = 1.0
        c = np.zeros((6, 7))
        c[:, :6] = np.eye(6)
        c[axis, 6] = 1.0
        rows = [c @ np.linalg.matrix_power(a, k) for k in range(7)]
        rank = np.linalg.matrix_rank(np.vstack(rows))
        assert bool(rank == 7) == detectable


@pytest.mark.slow
def test_a_minimum_impulse_bit_leaves_the_hold_inside_the_box():
    # With a 1 N s minimum impulse (2 mm/s quanta on 500 kg) the reference hold still completes
    # its dwell inside the 5 m, 0.05 m/s box.
    env = make("approach-rbar", min_impulse=1.0)
    run_episode(env, reference_for("approach-rbar"), seed=3, agent_seed=4)
    assert env._info()["success"]
