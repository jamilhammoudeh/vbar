"""Property-based tests: invariants that must hold for every state, duration and orbit."""

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from vbar.conventions import hill_frame, hill_to_inertial, inertial_to_hill
from vbar.dynamics import EARTH, cw_discrete, cw_system, mean_motion
from vbar.dynamics.kepler import lagrange_coefficients, state_from_elements
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
    # LU rounding on entries up to 2e4 s; the worst seen is 2e-14.
    assert abs(np.linalg.det(phi) - 1.0) < 1e-12


# The CW flow is symplectic in the canonical coordinates (r, v + omega x r). In the Hill state
# [r, v] that reads phi^T J' phi = J' with J' = [[2 [omega]x, I], [-I, 0]]; the standard J fails
# on correct code because of the Coriolis term (Schaub and Junkins; Scheeres, ch. 6).
OMEGA_CROSS = np.array([[0.0, -N, 0.0], [N, 0.0, 0.0], [0.0, 0.0, 0.0]])
J_HILL = np.block([[2 * OMEGA_CROSS, np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])


@given(durations)
def test_cw_transition_is_symplectic_in_canonical_coordinates(t):
    phi, _ = cw_discrete(N, t)
    bound = 64 * EPS * np.linalg.norm(phi) ** 2 * np.linalg.norm(J_HILL)
    assert np.max(np.abs(phi.T @ J_HILL @ phi - J_HILL)) < bound


@given(durations)
def test_cw_inverse_follows_from_the_symplectic_form(t):
    phi, _ = cw_discrete(N, t)
    phi_back, _ = cw_discrete(N, -t)
    inverse = np.linalg.solve(J_HILL, phi.T @ J_HILL)
    assert np.max(np.abs(inverse - phi_back)) < 64 * EPS * np.linalg.norm(phi) ** 2


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


@given(
    st.floats(6.8e6, 4.2e7),
    st.floats(0.0, 0.95),
    st.lists(st.floats(0.0, 2 * np.pi), min_size=4, max_size=4),
    st.floats(-2e5, 2e5),
)
def test_lagrange_coefficients_satisfy_f_gdot_minus_g_fdot_equals_one(a, e, angles, t):
    r0, v0 = state_from_elements([a, e, *angles], EARTH.mu)
    f, g, f_dot, g_dot = lagrange_coefficients(r0, v0, t, EARTH.mu)
    # g is the small difference t - chi^3 c3 / sqrt(mu), so it carries the rounding of t.
    scale = abs(f * g_dot) + abs(g * f_dot) + abs(t * f_dot)
    assert abs(f * g_dot - g * f_dot - 1.0) < 64 * EPS * scale


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


def rotating_frame_jacobi(state, a, n):
    """Rotating-frame Jacobi integral minus its value at the target, from a Hill state.

    The frame rotates with a circular orbit of radius `a`; the large terms cancel analytically.
    """
    x, y, z = state[:3]
    q = 2 * a * x + x * x + y * y + z * z
    radius = np.sqrt((a + x) ** 2 + y * y + z * z)
    return (
        0.5 * state[3:] @ state[3:]
        - 0.5 * n * n * (2 * a * x + x * x + y * y)
        + EARTH.mu * q / (a * radius * (a + radius))
    )


@settings(max_examples=15, deadline=None)
@given(positions, velocities, accels, st.floats(0.0, np.pi), st.floats(100.0, PERIOD))
@example([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, 100.0)
def test_twobody_jacobi_integral_changes_by_work_of_hill_thrust(r0, v0, accel, inclination, t):
    # A circular target's Hill frame rotates uniformly, so thrust held in it does work a . dr
    # on the rotating-frame Jacobi integral; a thrust frame or sign error breaks this.
    m = TwoBodyDynamics(inclination=inclination)
    s0 = np.r_[r0, v0]
    a = np.array(accel)
    m.reset(s0)
    s1, _ = m.advance(a, t)
    radius = EARTH.radius + m.altitude
    change = rotating_frame_jacobi(s1, radius, m.n) - rotating_frame_jacobi(s0, radius, m.n)
    work = a @ (s1[:3] - s0[:3])
    scale = max(s1[3:] @ s1[3:], m.n**2 * s1[:3] @ s1[:3], s0[3:] @ s0[3:], abs(work), 1e-6)
    # The n^2 a x terms that cancel inside the integral leave rounding of their own size.
    floor = 16 * EPS * m.n**2 * radius * (np.linalg.norm(s0[:3]) + np.linalg.norm(s1[:3]))
    assert abs(change - work) < 1e-10 * scale + floor


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
