import dataclasses
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vbar.dynamics import cr3bp
from vbar.dynamics.bodies import SUN_FOR_EARTH_MOON_KLMR as SUN

SYSTEM = SUN.system
MU = SYSTEM.mu
NRHO = np.array([0.9873849199042153, 0.0, 0.008369210996156513, 0.0, 1.6744665364075588, 0.0])
OFFSET = np.array([800.0, -1500.0, 400.0, 0.05, 0.02, -0.03])


def test_sun_rate_is_consistent_with_its_mass_and_distance():
    # Koon, Lo, Marsden and Ross, eqs 5.3.4 to 5.3.7: the Sun's synodic rate is 1 minus the
    # Earth-Moon barycentre's rate about it, sqrt((m_S + 1) / a_S^3); 0.92520 to five digits.
    assert 1 - math.sqrt((SUN.mass + 1) / SUN.distance**3) == pytest.approx(SUN.rate, abs=1e-5)


def test_massless_sun_leaves_the_cr3bp():
    quiet = dataclasses.replace(SUN, mass=0.0)
    bicircular = cr3bp.BicircularDynamics(quiet, NRHO, sun_angle=0.7)
    plain = cr3bp.CR3BPDynamics(SYSTEM, NRHO)
    for m in (bicircular, plain):
        m.reset(OFFSET)
    accel = [1e-5, 0.0, -2e-5]
    assert np.array_equal(bicircular.advance(accel, 86400.0)[0], plain.advance(accel, 86400.0)[0])


def sun_potential(state, t, sun_angle):
    """The Sun's potential m / |r - r_S| - m r . r_S / a^3 and its time derivative at fixed r."""
    angle = sun_angle - SUN.rate * t
    r_s = SUN.distance * np.array([math.cos(angle), math.sin(angle), 0.0])
    v_s = SUN.distance * SUN.rate * np.array([math.sin(angle), -math.cos(angle), 0.0])
    d = state[:3] - r_s
    r3 = SUN.distance**3
    potential = SUN.mass / np.linalg.norm(d) - SUN.mass * state[:3] @ r_s / r3
    rate = SUN.mass * (d @ v_s) / np.linalg.norm(d) ** 3 - SUN.mass * state[:3] @ v_s / r3
    return potential, rate


def test_jacobi_constant_changes_by_the_suns_time_dependence_and_the_thrust_work():
    # With the Sun's potential in it, C = 2 (Omega + Omega_S) - v^2 changes as
    # dC/dt = 2 dOmega_S/dt - 2 v . a, the partial derivative at fixed position (the
    # extended-Hamiltonian identity; Scheeres, Orbital Motion in Strongly Perturbed
    # Environments, sec. 5.1). Integrated here by Simpson's rule over 400 segments.
    m = cr3bp.BicircularDynamics(SUN, NRHO, sun_angle=1.1)
    m.reset(OFFSET)
    accel = np.array([2e-4, -1e-4, 5e-5])
    steps, span = 400, 0.4 * SYSTEM.time
    states = [m.chaser_state()]
    for _ in range(steps):
        m.advance(accel, span / steps)
        states.append(m.chaser_state())
    states = np.array(states)
    h = 0.4 / steps
    rates = np.array([sun_potential(s, k * h, 1.1)[1] for k, s in enumerate(states)])
    integral = h / 3 * (rates[0] + rates[-1] + 4 * rates[1:-1:2].sum() + 2 * rates[2:-1:2].sum())
    work = (accel / SYSTEM.acceleration) @ (states[-1, :3] - states[0, :3])
    jacobi = [
        cr3bp.jacobi_constant(states[k], MU) + 2 * sun_potential(states[k], k * h, 1.1)[0]
        for k in (0, steps)
    ]
    change = jacobi[1] - jacobi[0]
    assert abs(integral) > 1e-3
    assert change == pytest.approx(2 * integral - 2 * work, abs=1e-10)


def test_bicircular_equations_match_an_inertial_frame_integration():
    # In a non-rotating frame at the Earth-Moon barycentre the Earth and Moon turn at rate 1 and
    # the Sun at 1 - rate; the frame's acceleration toward the Sun enters as -m r_S / a^3.
    m = cr3bp.BicircularDynamics(SUN, NRHO, sun_angle=0.3)
    m.reset(np.zeros(6))
    m.advance(np.zeros(3), 0.5 * SYSTEM.time)
    inertial0 = np.r_[NRHO[:3], NRHO[3:] + np.cross([0.0, 0.0, 1.0], NRHO[:3])]

    def rhs(t, s):
        c, s_ = math.cos(t), math.sin(t)
        accel = np.zeros(3)
        for mass, p in cr3bp.primaries(MU):
            d = s[:3] - np.array([p[0] * c, p[0] * s_, 0.0])
            accel -= mass * d / np.linalg.norm(d) ** 3
        angle = 0.3 + (1 - SUN.rate) * t
        r_s = SUN.distance * np.array([math.cos(angle), math.sin(angle), 0.0])
        d = s[:3] - r_s
        accel -= SUN.mass * (d / np.linalg.norm(d) ** 3 + r_s / SUN.distance**3)
        return np.r_[s[3:], accel]

    end = solve_ivp(rhs, (0, 0.5), inertial0, "DOP853", rtol=1e-13, atol=1e-15).y[:, -1]
    c, s_ = math.cos(0.5), math.sin(0.5)
    rot = np.array([[c, s_, 0.0], [-s_, c, 0.0], [0.0, 0.0, 1.0]])
    r = rot @ end[:3]
    synodic = np.r_[r, rot @ end[3:] - np.cross([0.0, 0.0, 1.0], r)]
    assert np.max(np.abs(m.target_state() - synodic)) < 1e-10


def test_relative_bicircular_motion_matches_differenced_absolute_propagations():
    m = cr3bp.BicircularDynamics(SUN, NRHO, sun_angle=2.0)
    m.reset(OFFSET)
    accel = np.array([2e-5, -1e-5, 5e-6])
    final, _ = m.advance(accel, 86400.0)
    scale = np.repeat([SYSTEM.length, SYSTEM.velocity], 3)

    def absolute(t, s, a):
        angle = 2.0 - SUN.rate * t
        r_s = SUN.distance * np.array([math.cos(angle), math.sin(angle), 0.0])
        d = s[:3] - r_s
        sun = -SUN.mass * (d / np.linalg.norm(d) ** 3 + r_s / SUN.distance**3)
        return np.r_[s[3:], cr3bp.acceleration(s[:3], s[3:], MU) + sun + a]

    tau = 86400.0 / SYSTEM.time
    a = accel / SYSTEM.acceleration
    kw = dict(method="DOP853", rtol=1e-13, atol=1e-16)
    chaser = solve_ivp(absolute, (0, tau), NRHO + OFFSET / scale, args=(a,), **kw).y[:, -1]
    target = solve_ivp(absolute, (0, tau), NRHO, args=(np.zeros(3),), **kw).y[:, -1]
    expected = (chaser - target) * scale
    assert np.max(np.abs(final[:3] - expected[:3])) < 2e-4
    assert np.max(np.abs(final[3:] - expected[3:])) < 1e-9
