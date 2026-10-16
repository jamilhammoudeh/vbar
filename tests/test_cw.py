"""Validation of the CW dynamics. If these fail, every result is wrong."""

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

from vbar.dynamics import (
    R_EARTH,
    CWDiscretizer,
    cw_discrete,
    cw_stm,
    cw_system,
    mean_motion,
    propagate,
)

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


def random_state(rng):
    return np.r_[rng.uniform(-2000, 2000, 3), rng.uniform(-2, 2, 3)]


@pytest.mark.parametrize("seed", range(10))
def test_exact_cw_matches_closed_form_over_one_orbit(seed):
    s0 = random_state(np.random.default_rng(seed))
    phi, _ = cw_discrete(N, PERIOD)
    exact = cw_stm(N, PERIOD) @ s0
    s1 = phi @ s0
    assert np.max(np.abs(s1[:3] - exact[:3])) < 1e-9
    assert np.max(np.abs(s1[3:] - exact[3:])) < 1e-12


@pytest.mark.parametrize("dt", [1.0, 10.0, 100.0, 1000.0])
def test_exact_cw_matches_van_loan_exponential(dt):
    # Both matrices from one exponential of [[A, B], [0, 0]] dt (Van Loan, 1978). Durations
    # stay short because scipy's expm loses accuracy near one orbit before version 1.18.
    A, B = cw_system(N)
    M = np.zeros((9, 9))
    M[:6, :6] = A
    M[:6, 6:] = B
    E = expm(M * dt)
    phi, gamma = cw_discrete(N, dt)
    assert np.max(np.abs(phi - E[:6, :6])) < 1e-12 * np.max(np.abs(E[:6, :6]))
    assert np.max(np.abs(gamma - E[:6, 6:])) < 1e-12 * np.max(np.abs(E[:6, 6:]))


@pytest.mark.parametrize("dt", [1.0, 10.0, 1000.0, PERIOD])
def test_exact_cw_thrust_matrix_is_integral_of_stm(dt):
    _, gamma = cw_discrete(N, dt)
    integral, _ = quad_vec(lambda s: cw_stm(N, s)[:, 3:], 0.0, dt, epsabs=0.0, epsrel=1e-14)
    assert np.max(np.abs(gamma - integral)) < 1e-12 * np.max(np.abs(integral))


@pytest.mark.parametrize("seed", range(5))
def test_exact_cw_matches_dop853_over_one_orbit(seed):
    rng = np.random.default_rng(seed)
    s0 = random_state(rng)
    # Small enough that one orbit of thrust keeps the chaser within tens of km, where
    # DOP853 near its tightest rtol resolves 1e-9 m.
    accel = rng.uniform(-1e-3, 1e-3, 3)
    phi, gamma = cw_discrete(N, PERIOD)
    s1 = phi @ s0 + gamma @ accel
    numeric = propagate(s0, N, PERIOD, accel, rtol=1e-13, atol=1e-14)
    assert np.max(np.abs(s1[:3] - numeric[:3])) < 1e-9
    assert np.max(np.abs(s1[3:] - numeric[3:])) < 1e-12


@pytest.mark.parametrize("seed", range(10))
def test_stepped_exact_cw_stays_at_rounding_floor(seed):
    s0 = random_state(np.random.default_rng(seed))
    dt = 10.0
    steps = int(PERIOD // dt)
    phi, _ = cw_discrete(N, dt)
    path = [s0]
    for _ in range(steps):
        path.append(phi @ path[-1])
    path = np.array(path)
    exact = cw_stm(N, steps * dt) @ s0
    scale = np.max(np.abs(path[:, :3]))
    assert np.max(np.abs(path[-1, :3] - exact[:3])) < 1e-12 * scale
    assert np.max(np.abs(path[-1, 3:] - exact[3:])) < 1e-12 * N * scale


@pytest.mark.parametrize("seed", range(3))
def test_stepped_exact_cw_matches_dop853_with_thrust(seed):
    rng = np.random.default_rng(seed)
    dt = 10.0
    phi, gamma = cw_discrete(N, dt)
    exact = numeric = random_state(rng)
    peak = np.max(np.abs(exact[:3]))
    for _ in range(int(PERIOD // dt)):
        accel = rng.uniform(-0.02, 0.02, 3)
        exact = phi @ exact + gamma @ accel
        numeric = propagate(numeric, N, dt, accel, rtol=1e-13, atol=1e-14)
        peak = max(peak, np.max(np.abs(exact[:3])))
    assert np.max(np.abs(exact[:3] - numeric[:3])) < 1e-12 * peak
    assert np.max(np.abs(exact[3:] - numeric[3:])) < 1e-12 * N * peak


def test_exact_cw_rbar_offset_drifts_12_pi_x0_per_orbit():
    phi, _ = cw_discrete(N, PERIOD)
    s1 = phi @ np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert s1[1] == pytest.approx(12 * np.pi * 100.0, rel=1e-12)


def test_exact_cw_vbar_point_is_equilibrium():
    s0 = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])
    phi, _ = cw_discrete(N, PERIOD)
    assert np.allclose(phi @ s0, s0, rtol=0.0, atol=1e-12)


def test_sample_matrices_span_the_step():
    disc = CWDiscretizer(N)
    phis, gammas = disc.sample_matrices(10.0, 5)
    assert phis.shape == (5, 6, 6) and gammas.shape == (5, 6, 3)
    assert np.array_equal(phis[0], np.eye(6)) and np.array_equal(gammas[0], np.zeros((6, 3)))
    phi, gamma = disc.matrices(10.0)
    assert np.allclose(phis[-1], phi, rtol=1e-15, atol=0.0)
    assert np.allclose(gammas[-1], gamma, rtol=1e-15, atol=0.0)
    phi_mid, gamma_mid = cw_discrete(N, 5.0)
    assert np.allclose(phis[2], phi_mid, rtol=1e-15, atol=0.0)
    assert np.allclose(gammas[2], gamma_mid, rtol=1e-15, atol=0.0)
