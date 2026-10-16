from decimal import Decimal, localcontext

import numpy as np
import pytest

from vbar.conventions import inertial_to_hill
from vbar.dynamics import cw_stm
from vbar.dynamics.bodies import EARTH, KERBIN
from vbar.dynamics.kepler import elements_from_state, propagate_kepler, state_from_elements
from vbar.dynamics.mean_elements import j2_secular_rates, mean_to_osculating, osculating_to_mean
from vbar.dynamics.models import (
    CWDynamics,
    TwoBodyDynamics,
    differential_gravity,
    gravity,
    make_dynamics,
)

ZERO = np.zeros(3)
EPS = np.finfo(float).eps


def exact_gravity_difference(r, dr, j2):
    """g(r + dr) - g(r) in 50-digit decimal arithmetic, rounded to float at the end."""
    with localcontext() as ctx:
        ctx.prec = 50
        mu, radius, j2_coef = (Decimal(x) for x in (EARTH.mu, EARTH.radius, EARTH.j2))

        def g(p):
            r2 = sum(c * c for c in p)
            r_norm = r2.sqrt()
            out = [-mu * c / (r2 * r_norm) for c in p]
            if j2:
                k = Decimal(3) / 2 * j2_coef * mu * radius**2 / (r2 * r2 * r_norm)
                z2 = 5 * p[2] ** 2 / r2
                out = [o + k * c * (z2 - w) for o, c, w in zip(out, p, (1, 1, 3), strict=True)]
            return out

        p = [Decimal(float(x)) for x in r]
        q = [a + Decimal(float(b)) for a, b in zip(p, dr, strict=True)]
        return np.array([float(a - b) for a, b in zip(g(q), g(p), strict=True)])


def state_at_range(rng, rho, n):
    """Random relative state with position norm `rho` and velocity norm `n rho`."""
    u = rng.normal(size=6)
    u[:3] *= rho / np.linalg.norm(u[:3])
    u[3:] *= n * rho / np.linalg.norm(u[3:])
    return u


def specific_energy(s, mu):
    return 0.5 * s[3:] @ s[3:] - mu / np.linalg.norm(s[:3])


@pytest.mark.parametrize("rho", [1e-3, 1e-2, 1.0, 1e3, 1e6])
def test_differential_gravity_is_exact_to_rounding_at_any_separation(rho):
    r = np.array([EARTH.radius + 500e3, 1234.5678, -987.654321])
    dr = rho * np.array([0.3, -0.8, 0.52]) / np.linalg.norm([0.3, -0.8, 0.52])
    exact = exact_gravity_difference(r, dr, j2=False)
    error = np.linalg.norm(differential_gravity(r, dr, EARTH) - exact)
    assert error < 4 * EPS * np.linalg.norm(exact)
    # The J2 part is differenced directly; its rounding is bounded by that of the J2 term.
    exact_j2 = exact_gravity_difference(r, dr, j2=True)
    j2_size = np.linalg.norm(gravity(r, EARTH, j2=True) - gravity(r, EARTH))
    error_j2 = np.linalg.norm(differential_gravity(r, dr, EARTH, j2=True) - exact_j2)
    assert error_j2 < 4 * EPS * (np.linalg.norm(exact_j2) + j2_size)


def test_direct_gravity_difference_loses_digits_at_short_range():
    r = np.array([EARTH.radius + 500e3, 1234.5678, -987.654321])
    dr = np.array([0.0, 1.0, 0.0])
    exact = exact_gravity_difference(r, dr, j2=False)
    naive = gravity(r + dr, EARTH) - gravity(r, EARTH)
    assert np.linalg.norm(naive - exact) > 1e5 * EPS * np.linalg.norm(exact)


def test_cw_dynamics_samples_match_closed_form():
    m = make_dynamics("cw")
    s0 = np.array([150.0, -400.0, 30.0, 0.05, 0.1, -0.02])
    m.reset(s0)
    final, samples = m.advance(ZERO, 100.0, n_samples=11)
    exact = np.array([cw_stm(m.n, t) @ s0 for t in np.linspace(0.0, 100.0, 11)])
    assert np.allclose(samples, exact, rtol=0.0, atol=1e-10)
    assert np.allclose(final, exact[-1], rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("kind", ["cw", "twobody_j2"])
def test_state_in_last_segment_matches_shorter_segment(kind):
    s0 = np.array([150.0, -400.0, 30.0, 0.05, 0.1, -0.02])
    accel = np.array([2e-3, -1e-3, 5e-4])
    long, short = make_dynamics(kind), make_dynamics(kind)
    long.reset(s0)
    long.advance(accel, 100.0)
    short.reset(s0)
    end, _ = short.advance(accel, 37.0)
    assert np.allclose(long.state_in_last_segment(37.0), end, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("kind", ["cw", "twobody"])
def test_state_in_last_segment_needs_a_segment(kind):
    m = make_dynamics(kind)
    m.reset(np.zeros(6))
    with pytest.raises(RuntimeError, match="no segment"):
        m.state_in_last_segment(1.0)


def test_twobody_decision_steps_cost_under_20_gravity_evaluations_each():
    m = TwoBodyDynamics(inclination=np.radians(51.6))
    calls = 0
    rhs = m._rhs

    def counted(*args):
        nonlocal calls
        calls += 1
        return rhs(*args)

    m._rhs = counted
    start = [0.0, -1000.0, 0.0, 0.0, 0.0, 0.0]
    m.reset(start)
    period = 2 * np.pi / m.n
    steps = 568  # about 10 s each
    for _ in range(steps):
        stepped, _ = m.advance(ZERO, period / steps, n_samples=20)
    # One DOP853 step with dense output is 16 evaluations; restarting from scipy's own initial
    # step guess took 84.
    assert calls < 20 * steps
    m.reset(start)
    continuous, _ = m.advance(ZERO, period)
    assert np.max(np.abs(stepped[:3] - continuous[:3])) < 1e-8


@pytest.mark.parametrize("kind", ["cw", "twobody_j2"])
def test_zero_and_backward_segments_are_exact(kind):
    m = make_dynamics(kind, inclination=0.9)
    s0 = np.array([150.0, -400.0, 30.0, 0.05, 0.1, -0.02])
    m.reset(s0)
    forward, _ = m.advance(ZERO, 30.0)
    same, _ = m.advance(ZERO, 0.0)
    assert np.array_equal(same, forward)
    back, _ = m.advance(ZERO, -30.0)
    assert np.allclose(back, s0, rtol=0.0, atol=1e-9)


def test_twobody_samples_are_states_within_the_segment():
    m = TwoBodyDynamics(j2=True, inclination=0.9)
    m.reset([150.0, -400.0, 30.0, 0.05, 0.1, -0.02])
    final, samples = m.advance([1e-3, 0.0, -1e-3], 60.0, n_samples=7)
    ts = np.linspace(0.0, 60.0, 7)
    single = np.array([m.state_in_last_segment(t) for t in ts])
    assert np.allclose(samples, single, rtol=0.0, atol=1e-12)
    assert np.allclose(final, samples[-1], rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("rho", [100.0, 1000.0, 5000.0])
@pytest.mark.parametrize("seed", range(2))
def test_twobody_matches_analytic_kepler_over_one_orbit(rho, seed):
    m = TwoBodyDynamics()
    m.reset(state_at_range(np.random.default_rng(seed), rho, m.n))
    target, chaser = m.target_state(), m.chaser_state()
    period = 2 * np.pi / m.n
    final, _ = m.advance(ZERO, period)
    r_t, v_t = propagate_kepler(target[:3], target[3:], period, EARTH.mu)
    r_c, v_c = propagate_kepler(chaser[:3], chaser[3:], period, EARTH.mu)
    expected = inertial_to_hill(r_t, v_t, r_c - r_t, v_c - v_t)
    assert np.max(np.abs(final[:3] - expected[:3])) < 1e-5
    assert np.max(np.abs(final[3:] - expected[3:])) < 1e-8


@pytest.mark.parametrize("inclination_deg", [0.0, 51.6])
def test_twobody_absolute_states_match_kepler_over_ten_orbits(inclination_deg):
    m = TwoBodyDynamics(inclination=np.radians(inclination_deg))
    m.reset([5000.0, 2000.0, -3000.0, 1.0, -2.0, 0.5])
    start = [m.target_state(), m.chaser_state()]
    duration = 10 * 2 * np.pi / m.n
    m.advance(ZERO, duration)
    for s0, s1 in zip(start, [m.target_state(), m.chaser_state()], strict=True):
        r, v = propagate_kepler(s0[:3], s0[3:], duration, EARTH.mu)
        assert np.linalg.norm(s1[:3] - r) < 1e-9 * np.linalg.norm(r)
        assert np.linalg.norm(s1[3:] - v) < 1e-9 * np.linalg.norm(v)


def hohmann_through_model(r1, r2):
    """Burns (m/s) of a Hohmann transfer flown by the chaser through the two-body model."""
    a_transfer = 0.5 * (r1 + r2)
    dv1 = np.sqrt(EARTH.mu * (2 / r1 - 1 / a_transfer)) - np.sqrt(EARTH.mu / r1)
    m = TwoBodyDynamics(altitude=r1 - EARTH.radius)
    m.reset([0.0, 0.0, 0.0, 0.0, dv1, 0.0])
    m.advance(ZERO, np.pi * np.sqrt(a_transfer**3 / EARTH.mu))
    s = m.chaser_state()
    r = np.linalg.norm(s[:3])
    assert r == pytest.approx(r2, rel=1e-11)
    assert abs(s[:3] @ s[3:] / r) < 1e-8
    return dv1, np.sqrt(EARTH.mu / r2) - np.linalg.norm(s[3:])


def test_hohmann_transfer_from_7000_to_7200_km_costs_105_5392_m_per_s():
    # Schaub and Junkins' worked example; their printed 104.351 m/s used a wrong circular speed.
    dv1, dv2 = hohmann_through_model(7000e3, 7200e3)
    assert dv1 == pytest.approx(52.9554, abs=5e-5)
    assert dv2 == pytest.approx(52.5838, abs=5e-5)
    assert dv1 + dv2 == pytest.approx(105.5392, abs=5e-5)


def test_hohmann_transfer_from_750_to_775_km_altitude_costs_13_08_m_per_s():
    # Montenbruck and Gill, Satellite Orbits, exercise 2.1.
    radius = 6378.137e3
    dv1, dv2 = hohmann_through_model(radius + 750e3, radius + 775e3)
    assert dv1 + dv2 == pytest.approx(13.08, abs=5e-3)


def pco_state(rho, alpha, n):
    """Projected circular orbit: (y, z) circle of radius rho, x = rho / 2 sin(n t + alpha)."""
    s, c = np.sin(alpha), np.cos(alpha)
    return [0.5 * rho * s, rho * c, rho * s, 0.5 * rho * n * c, -rho * n * s, rho * n * c]


# One-orbit along-track drift of two-body motion relative to CW, as a multiple of D^2 / a, for
# CW-bounded starts at range D (Alfriend et al., Spacecraft Formation Flying, eqs 5.63, 5.64
# and 5.73).
SECOND_ORDER_DRIFT = {
    "vbar_offset": (lambda d, n: [0.0, d, 0.0, 0.0, 0.0, 0.0], -6 * np.pi),
    "radial_offset": (lambda d, n: [d, 0.0, 0.0, 0.0, -2 * n * d, 0.0], 3 * np.pi),
    "pco_alpha_0": (lambda d, n: pco_state(d, 0.0, n), -27 * np.pi / 4),
    "pco_alpha_90": (lambda d, n: pco_state(d, np.pi / 2, n), -9 * np.pi / 4),
}


@pytest.mark.parametrize("geometry", SECOND_ORDER_DRIFT)
@pytest.mark.parametrize(
    "d, rtol",
    [
        # At 1 cm the drift is about 1e-8 of the range, close to the default integrator's
        # error of 1e-9 of the range per orbit, so the sub-metre cases run tighter.
        (0.01, 1e-13),
        (1.0, 1e-13),
        (100.0, 1e-11),
        (1000.0, 1e-11),
        (5000.0, 1e-11),
    ],
)
def test_twobody_minus_cw_drift_matches_second_order_coefficient(geometry, d, rtol):
    start, coefficient = SECOND_ORDER_DRIFT[geometry]
    twobody = TwoBodyDynamics(rtol=rtol)
    cw = CWDynamics(twobody.n)
    period = 2 * np.pi / twobody.n
    twobody.reset(start(d, twobody.n))
    cw.reset(start(d, twobody.n))
    nonlinear, _ = twobody.advance(ZERO, period)
    linear, _ = cw.advance(ZERO, period)
    a = EARTH.radius + twobody.altitude
    assert nonlinear[1] - linear[1] == pytest.approx(coefficient * d**2 / a, rel=1e-2)


def test_twobody_conserves_energy_over_ten_orbits():
    m = TwoBodyDynamics()
    m.reset([300.0, -800.0, 150.0, 0.1, -0.2, 0.05])
    e_target = specific_energy(m.target_state(), EARTH.mu)
    e_chaser = specific_energy(m.chaser_state(), EARTH.mu)
    m.advance(ZERO, 10 * 2 * np.pi / m.n)
    assert specific_energy(m.target_state(), EARTH.mu) == pytest.approx(e_target, rel=1e-10)
    assert specific_energy(m.chaser_state(), EARTH.mu) == pytest.approx(e_chaser, rel=1e-10)


def j2_energy(s):
    """Specific energy including the J2 potential, m^2/s^2."""
    r = np.linalg.norm(s[:3])
    u_j2 = EARTH.mu * EARTH.j2 * EARTH.radius**2 / (2 * r**3) * (3 * (s[2] / r) ** 2 - 1)
    return specific_energy(s, EARTH.mu) + u_j2


def test_twobody_j2_conserves_energy_and_polar_angular_momentum_over_ten_orbits():
    m = TwoBodyDynamics(inclination=np.radians(51.6), j2=True)
    m.reset([300.0, -800.0, 150.0, 0.1, -0.2, 0.05])
    start = [m.target_state(), m.chaser_state()]
    m.advance(ZERO, 10 * 2 * np.pi / m.n)
    for s0, s1 in zip(start, [m.target_state(), m.chaser_state()], strict=True):
        assert j2_energy(s1) == pytest.approx(j2_energy(s0), rel=1e-10)
        h0, h1 = np.cross(s0[:3], s0[3:]), np.cross(s1[:3], s1[3:])
        assert abs(h1[2] - h0[2]) < 1e-10 * np.linalg.norm(h0)


def mean_elements(state):
    return osculating_to_mean(elements_from_state(state[:3], state[3:], EARTH.mu), EARTH)


def wrapped(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@pytest.mark.parametrize("inclination_deg", [0.0, 51.6, 97.4])
def test_j2_target_is_circular_in_mean_elements(inclination_deg):
    m = TwoBodyDynamics(inclination=np.radians(inclination_deg), j2=True)
    m.reset(np.zeros(6))
    period = 2 * np.pi / m.n
    a = EARTH.radius + m.altitude
    for _ in range(5 * 8):
        mean = mean_elements(m.target_state())
        # An osculating circular start has a mean eccentricity of 8e-4 at 51.6 deg.
        assert mean[1] < 1e-5
        assert abs(mean[0] - a) < 50.0
        m.advance(ZERO, period / 8)


@pytest.mark.parametrize("inclination_deg", [28.5, 51.6, 97.4])
def test_j2_target_mean_raan_and_latitude_rates_match_secular_theory(inclination_deg):
    inclination = np.radians(inclination_deg)
    m = TwoBodyDynamics(inclination=inclination, j2=True)
    m.reset(np.zeros(6))
    duration = 10 * 2 * np.pi / m.n
    start = mean_elements(m.target_state())
    m.advance(ZERO, duration)
    end = mean_elements(m.target_state())
    a = EARTH.radius + m.altitude
    raan_dot, _, _ = j2_secular_rates(a, 0.0, inclination, EARTH)
    assert wrapped(end[3] - start[3]) / duration == pytest.approx(raan_dot, rel=1e-2)
    # The mean argument of latitude outruns the Keplerian n agents are given.
    excess = 1.5 * EARTH.j2 * (EARTH.radius / a) ** 2 * (4 * np.cos(inclination) ** 2 - 1)
    latitude = wrapped(end[4] + end[5] - start[4] - start[5]) + 10 * 2 * np.pi
    assert latitude / (m.n * duration) - 1 == pytest.approx(excess, rel=1e-2)


@pytest.mark.parametrize("inclination_deg", [30.0, 97.4])
def test_j2_apsidal_and_nodal_rates_match_secular_theory_on_an_eccentric_orbit(inclination_deg):
    # The chaser flies an e = 0.3 orbit through the same model, thousands of km from the
    # target; the argument of perigee is only well defined away from circular orbits.
    inclination = np.radians(inclination_deg)
    m = TwoBodyDynamics(inclination=inclination, j2=True)
    m.reset(np.zeros(6))
    target = m.target_state()
    mean = [10000e3, 0.3, inclination, 0.4, 0.9, 0.3]
    r, v = state_from_elements(mean_to_osculating(mean, EARTH), EARTH.mu)
    accel = gravity(target[:3], EARTH, j2=True)
    m.reset(inertial_to_hill(target[:3], target[3:], r - target[:3], v - target[3:], accel))
    duration = 5 * 2 * np.pi * np.sqrt(mean[0] ** 3 / EARTH.mu)
    start = mean_elements(m.chaser_state())
    m.advance(ZERO, duration)
    end = mean_elements(m.chaser_state())
    raan_dot, argp_dot, _ = j2_secular_rates(start[0], start[1], start[2], EARTH)
    # The first-order rates leave out J2^2 terms worth about 0.2%.
    assert wrapped(end[3] - start[3]) / duration == pytest.approx(raan_dot, rel=1e-2)
    assert wrapped(end[4] - start[4]) / duration == pytest.approx(argp_dot, rel=1e-2)


@pytest.mark.parametrize("t", [500.0, 1400.0, 2800.0])
def test_hill_velocity_under_j2_needs_the_exact_frame_rate(t):
    m = TwoBodyDynamics(inclination=np.radians(51.6), j2=True)
    m.reset([200.0, -1000.0, 300.0, 0.05, 0.1, -0.2])
    h = 1.0
    m.advance(ZERO, t - 2 * h)
    positions = [m.state_in_last_segment(t - 2 * h)[:3]]
    for k in range(4):
        state, _ = m.advance(ZERO, h)
        positions.append(state[:3])
        if k == 1:
            rel, target, chaser = state, m.target_state(), m.chaser_state()
    rate = (positions[0] - 8 * positions[1] + 8 * positions[3] - positions[4]) / (12 * h)
    assert np.max(np.abs(rel[3:] - rate)) < 1e-9
    # The Keplerian frame rate ignores the orbit plane turning under J2.
    keplerian = inertial_to_hill(
        target[:3], target[3:], chaser[:3] - target[:3], chaser[3:] - target[3:]
    )
    assert np.max(np.abs(keplerian[3:] - rate)) > 1e-4


@pytest.mark.parametrize("axis", range(3))
def test_twobody_thrust_acts_along_hill_axes(axis):
    accel = np.zeros(3)
    accel[axis] = 1e-3
    twobody = TwoBodyDynamics(inclination=0.9)
    cw = CWDynamics(twobody.n)
    twobody.reset(np.zeros(6))
    cw.reset(np.zeros(6))
    nonlinear, _ = twobody.advance(accel, 60.0)
    linear, _ = cw.advance(accel, 60.0)
    # A 60 s burn moves the chaser 1.8 m, where the nonlinear terms are below a micrometre.
    assert np.allclose(nonlinear, linear, rtol=0.0, atol=1e-6)


def test_make_dynamics_builds_each_model():
    assert make_dynamics("cw").name == "cw"
    assert make_dynamics("twobody").name == "twobody"
    kerbin = make_dynamics("twobody_j2", body="kerbin", altitude=150e3)
    assert kerbin.name == "twobody_j2"
    assert kerbin.n == pytest.approx(np.sqrt(KERBIN.mu / (KERBIN.radius + 150e3) ** 3))
    with pytest.raises(ValueError, match="unknown dynamics"):
        make_dynamics("nbody")
