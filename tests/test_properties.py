"""Property-based tests: invariants that must hold for every state, duration and orbit."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from vbar.conventions import hill_frame, hill_to_inertial, inertial_to_hill
from vbar.dynamics import EARTH, cw_discrete, cw_system, mean_motion
from vbar.dynamics.models import TwoBodyDynamics, gravity

N = mean_motion(EARTH.radius + 500e3)
PERIOD = 2 * np.pi / N
EPS = np.finfo(float).eps

durations = st.floats(min_value=1e-3, max_value=PERIOD)
positions = st.lists(st.floats(-2000.0, 2000.0), min_size=3, max_size=3)
velocities = st.lists(st.floats(-2.0, 2.0), min_size=3, max_size=3)
accels = st.lists(st.floats(-0.02, 0.02), min_size=3, max_size=3)
quaternions = st.lists(st.floats(-1.0, 1.0), min_size=4, max_size=4).filter(
    lambda q: np.linalg.norm(q) > 0.1
)


def jacobi(state, n):
    """CW Jacobi integral; constant along unforced motion."""
    x, _, z = state[:3]
    return 0.5 * state[3:] @ state[3:] - 1.5 * n**2 * x**2 + 0.5 * n**2 * z**2


@given(durations, durations)
def test_cw_transition_composes(t1, t2):
    phi1, gamma1 = cw_discrete(N, t1)
    phi2, gamma2 = cw_discrete(N, t2)
    phi12, gamma12 = cw_discrete(N, t1 + t2)
    # Rounding in a 6x6 product is bounded by a few eps times the product of the norms.
    scale = np.linalg.norm(phi1) * np.linalg.norm(phi2)
    assert np.max(np.abs(phi12 - phi2 @ phi1)) < 64 * EPS * scale
    gamma_scale = np.linalg.norm(phi2) * np.linalg.norm(gamma1) + np.linalg.norm(gamma2)
    assert np.max(np.abs(gamma12 - (phi2 @ gamma1 + gamma2))) < 64 * EPS * gamma_scale


@given(durations)
def test_cw_motion_is_time_reversible(t):
    # (x, y, z, vx, vy, vz, t) -> (x, -y, z, -vx, vy, -vz, -t) maps CW solutions to solutions.
    S = np.diag([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    phi, _ = cw_discrete(N, t)
    phi_back, _ = cw_discrete(N, -t)
    assert np.allclose(phi_back, S @ phi @ S, rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(phi_back @ phi - np.eye(6))) < 64 * EPS * np.linalg.norm(phi) ** 2


@given(durations)
def test_cw_transition_preserves_phase_volume(t):
    phi, _ = cw_discrete(N, t)
    assert np.linalg.det(phi) == pytest.approx(1.0, rel=1e-9)


@given(st.floats(min_value=1e-3, max_value=1000.0))
def test_cw_discretization_matches_van_loan_exponential(t):
    A, B = cw_system(N)
    M = np.zeros((9, 9))
    M[:6, :6] = A
    M[:6, 6:] = B
    E = expm(M * t)
    phi, gamma = cw_discrete(N, t)
    assert np.max(np.abs(phi - E[:6, :6])) < 1e-12 * np.max(np.abs(E[:6, :6]))
    assert np.max(np.abs(gamma - E[:6, 6:])) < 1e-12 * np.max(np.abs(E[:6, 6:]))


@given(positions, velocities, accels, durations)
def test_cw_jacobi_integral_changes_by_work_of_thrust(r0, v0, accel, t):
    s0 = np.r_[r0, v0]
    a = np.array(accel)
    phi, gamma = cw_discrete(N, t)
    s1 = phi @ s0 + gamma @ a
    # dJ/dt = v . a, so a constant acceleration changes J by a . (r1 - r0).
    expected = jacobi(s0, N) + a @ (s1[:3] - s0[:3])
    scale = max(s1[3:] @ s1[3:], N**2 * s1[:3] @ s1[:3], s0[3:] @ s0[3:], 1e-6)
    assert abs(jacobi(s1, N) - expected) < 1e-11 * scale


@given(positions, velocities, accels, durations, st.floats(-3.0, 3.0))
def test_cw_response_is_linear(r0, v0, accel, t, c):
    s0 = np.r_[r0, v0]
    a = np.array(accel)
    phi, gamma = cw_discrete(N, t)
    s1 = phi @ s0 + gamma @ a
    scaled = phi @ (c * s0) + gamma @ (c * a)
    assert np.allclose(scaled, c * s1, rtol=1e-12, atol=1e-12 * np.max(np.abs(s1)))


@settings(max_examples=15, deadline=None)
@given(positions, velocities, st.floats(0.0, np.pi), st.floats(100.0, PERIOD))
def test_twobody_conserves_energy_and_angular_momentum(r0, v0, inclination, t):
    m = TwoBodyDynamics(inclination=inclination)
    m.reset(np.r_[r0, v0])
    before = [m.target_state(), m.chaser_state()]
    m.advance(np.zeros(3), t)
    for s0, s1 in zip(before, [m.target_state(), m.chaser_state()], strict=True):
        energy = [0.5 * s[3:] @ s[3:] - EARTH.mu / np.linalg.norm(s[:3]) for s in (s0, s1)]
        assert abs(energy[1] - energy[0]) < 1e-10 * abs(energy[0])
        h0, h1 = np.cross(s0[:3], s0[3:]), np.cross(s1[:3], s1[3:])
        assert np.max(np.abs(h1 - h0)) < 1e-10 * np.linalg.norm(h0)


@settings(max_examples=15, deadline=None)
@given(positions, velocities, st.floats(0.0, np.pi), st.floats(100.0, PERIOD))
def test_twobody_j2_conserves_energy_and_polar_angular_momentum(r0, v0, inclination, t):
    m = TwoBodyDynamics(inclination=inclination, j2=True)
    m.reset(np.r_[r0, v0])

    def energy(s):
        r = np.linalg.norm(s[:3])
        sin2_lat = (s[2] / r) ** 2
        u_j2 = EARTH.mu * EARTH.j2 * EARTH.radius**2 / (2 * r**3) * (3 * sin2_lat - 1)
        return 0.5 * s[3:] @ s[3:] - EARTH.mu / r + u_j2

    before = [m.target_state(), m.chaser_state()]
    m.advance(np.zeros(3), t)
    for s0, s1 in zip(before, [m.target_state(), m.chaser_state()], strict=True):
        assert abs(energy(s1) - energy(s0)) < 1e-10 * abs(energy(s0))
        h_z = [np.cross(s[:3], s[3:])[2] for s in (s0, s1)]
        assert abs(h_z[1] - h_z[0]) < 1e-10 * np.linalg.norm(np.cross(s0[:3], s0[3:]))


@given(positions, velocities, quaternions, st.floats(0.0, np.pi), st.floats(0.0, 2 * np.pi))
def test_hill_state_is_invariant_under_inertial_rotation(r0, v0, q, inclination, u):
    a = EARTH.radius + 500e3
    speed = np.sqrt(EARTH.mu / a)
    r_t = a * np.array(
        [np.cos(u), np.cos(inclination) * np.sin(u), np.sin(inclination) * np.sin(u)]
    )
    v_t = speed * np.array(
        [-np.sin(u), np.cos(inclination) * np.cos(u), np.sin(inclination) * np.cos(u)]
    )
    accel = gravity(r_t, EARTH, j2=True)
    rel = np.r_[r0, v0]
    dr, dv = hill_to_inertial(r_t, v_t, rel, accel)
    R = Rotation.from_quat(q).as_matrix()
    rotated = inertial_to_hill(R @ r_t, R @ v_t, R @ dr, R @ dv, R @ accel)
    assert np.allclose(rotated[:3], rel[:3], rtol=0.0, atol=64 * EPS * 2000.0)
    assert np.allclose(rotated[3:], rel[3:], rtol=0.0, atol=64 * EPS * (2.0 + N * 2000.0))


@given(quaternions, st.floats(6.6e6, 4.2e7), st.floats(0.8, 1.3), st.floats(-0.3, 0.3))
def test_hill_axes_are_a_proper_rotation(q, radius, speed_ratio, flight_path):
    R = Rotation.from_quat(q).as_matrix()
    speed = speed_ratio * np.sqrt(EARTH.mu / radius)
    r = R @ np.array([radius, 0.0, 0.0])
    v = R @ (speed * np.array([np.sin(flight_path), np.cos(flight_path), 0.0]))
    C, omega = hill_frame(r, v)
    assert np.allclose(C.T @ C, np.eye(3), atol=1e-15)
    assert np.linalg.det(C) == pytest.approx(1.0, rel=1e-15)
    assert omega[2] > 0.0
