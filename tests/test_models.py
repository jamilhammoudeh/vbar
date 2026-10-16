import numpy as np
import pytest

from vbar.conventions import inertial_to_hill
from vbar.dynamics import cw_stm
from vbar.dynamics.bodies import EARTH, KERBIN
from vbar.dynamics.kepler import propagate_kepler
from vbar.dynamics.models import CWDynamics, TwoBodyDynamics, make_dynamics

ZERO = np.zeros(3)


def state_at_range(rng, rho, n):
    """Random relative state with position norm `rho` and velocity norm `n rho`."""
    u = rng.normal(size=6)
    u[:3] *= rho / np.linalg.norm(u[:3])
    u[3:] *= n * rho / np.linalg.norm(u[3:])
    return u


def specific_energy(s, mu):
    return 0.5 * s[3:] @ s[3:] - mu / np.linalg.norm(s[:3])


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


def test_hohmann_transfer_from_7000_to_7200_km_costs_105_5392_m_per_s():
    # Schaub and Junkins' worked example; their printed 104.351 m/s used a wrong circular speed.
    r1, r2 = 7000e3, 7200e3
    a_transfer = 0.5 * (r1 + r2)
    dv1 = np.sqrt(EARTH.mu * (2 / r1 - 1 / a_transfer)) - np.sqrt(EARTH.mu / r1)
    m = TwoBodyDynamics(altitude=r1 - EARTH.radius)
    m.reset([0.0, 0.0, 0.0, 0.0, dv1, 0.0])
    m.advance(ZERO, np.pi * np.sqrt(a_transfer**3 / EARTH.mu))
    s = m.chaser_state()
    r = np.linalg.norm(s[:3])
    assert r == pytest.approx(r2, rel=1e-11)
    assert abs(s[:3] @ s[3:] / r) < 1e-8
    dv2 = np.sqrt(EARTH.mu / r2) - np.linalg.norm(s[3:])
    assert dv1 == pytest.approx(52.9554, abs=5e-5)
    assert dv2 == pytest.approx(52.5838, abs=5e-5)
    assert dv1 + dv2 == pytest.approx(105.5392, abs=5e-5)


def test_twobody_minus_cw_error_grows_quadratically_with_range():
    twobody = TwoBodyDynamics()
    cw = CWDynamics(twobody.n)
    period = 2 * np.pi / twobody.n
    shape = np.r_[1.0, 0.5, 0.3, twobody.n * np.array([0.2, -2.0, 0.4])]
    ranges = np.array([100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0])
    errors = []
    for rho in ranges:
        twobody.reset(rho * shape)
        cw.reset(rho * shape)
        nonlinear, _ = twobody.advance(ZERO, period)
        linear, _ = cw.advance(ZERO, period)
        errors.append(np.linalg.norm(nonlinear[:3] - linear[:3]))
    ratios = np.array(errors[1:]) / np.array(errors[:-1])
    assert np.all(np.abs(ratios - 4.0) < 0.1)
    slope = np.polyfit(np.log(ranges), np.log(errors), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.01)


def test_twobody_conserves_energy_over_ten_orbits():
    m = TwoBodyDynamics()
    m.reset([300.0, -800.0, 150.0, 0.1, -0.2, 0.05])
    e_target = specific_energy(m.target_state(), EARTH.mu)
    e_chaser = specific_energy(m.chaser_state(), EARTH.mu)
    m.advance(ZERO, 10 * 2 * np.pi / m.n)
    assert specific_energy(m.target_state(), EARTH.mu) == pytest.approx(e_target, rel=1e-10)
    assert specific_energy(m.chaser_state(), EARTH.mu) == pytest.approx(e_chaser, rel=1e-10)


@pytest.mark.parametrize("inclination_deg", [28.5, 51.6, 97.4])
def test_j2_mean_raan_drift_matches_analytic_rate(inclination_deg):
    inclination = np.radians(inclination_deg)
    m = TwoBodyDynamics(inclination=inclination, j2=True)
    m.reset(np.zeros(6))

    def raan():
        s = m.target_state()
        h = np.cross(s[:3], s[3:])
        return np.arctan2(h[0], -h[1])

    # Whole orbits end to end, so the short-period J2 terms largely cancel.
    duration = 10 * 2 * np.pi / m.n
    raan0 = raan()
    m.advance(ZERO, duration)
    a = EARTH.radius + m.altitude
    expected = -1.5 * m.n * EARTH.j2 * (EARTH.radius / a) ** 2 * np.cos(inclination)
    assert (raan() - raan0) / duration == pytest.approx(expected, rel=1e-2)


def test_hill_velocity_under_j2_needs_the_exact_frame_rate():
    m = TwoBodyDynamics(inclination=np.radians(51.6), j2=True)
    m.reset([200.0, -1000.0, 300.0, 0.05, 0.1, -0.2])
    h = 0.5
    m.advance(ZERO, 500.0)
    before = [m.state_in_last_segment(500.0 - k * h)[:3] for k in (2, 1)]
    target, chaser = m.target_state(), m.chaser_state()
    m.advance(ZERO, 500.0)
    rel = m.state_in_last_segment(0.0)
    after = [m.state_in_last_segment(k * h)[:3] for k in (1, 2)]
    rate = (before[0] - 8 * before[1] + 8 * after[0] - after[1]) / (12 * h)
    assert np.max(np.abs(rel[3:] - rate)) < 1e-8
    # The Keplerian frame rate ignores the orbit plane turning under J2.
    keplerian = inertial_to_hill(
        target[:3], target[3:], chaser[:3] - target[:3], chaser[3:] - target[3:]
    )
    assert np.max(np.abs(keplerian[3:] - rate)) > 5e-4


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
