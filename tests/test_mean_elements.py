import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from vbar.dynamics.bodies import EARTH, Body
from vbar.dynamics.kepler import elements_from_state, solve_kepler, state_from_elements
from vbar.dynamics.mean_elements import (
    j2_secular_rates,
    mean_to_osculating,
    osculating_to_mean,
)
from vbar.dynamics.models import gravity


def wrapped(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def true_anomaly(elements):
    e, M = elements[1], elements[5]
    E = solve_kepler(M, e)
    return math.atan2(math.sqrt(1 - e * e) * math.sin(E), math.cos(E) - e)


def test_secular_rates_reproduce_published_sun_synchronous_repeat_orbit():
    # Montenbruck and Gill, Satellite Orbits, exercise 2.5: sun-synchronous, 43 revolutions in
    # 3 days. The printed a and i reproduce with J2 = 1.0826e-3 or 1.08263e-3.
    body = Body("earth", 398600.4415e9, 6378137.0, 1.08263e-3)
    sun_rate = 2 * np.pi / (365.2422 * 86400.0)
    earth_rate = 7.2921158553e-5

    def conditions(x):
        a, i = x[0] * 1e6, x[1]
        raan_dot, argp_dot, M_dot = j2_secular_rates(a, 0.0, i, body)
        nodal_period = 2 * np.pi / (argp_dot + M_dot)
        nodal_day = 2 * np.pi / (earth_rate - raan_dot)
        return [(raan_dot - sun_rate) / sun_rate, 43 * nodal_period / (3 * nodal_day) - 1.0]

    a, i = fsolve(conditions, [7.15, np.radians(98.5)], xtol=1e-13)
    assert a * 1e6 == pytest.approx(7153.123e3, abs=0.5)
    assert np.degrees(i) == pytest.approx(98.498, abs=5e-4)


def test_mean_to_osculating_matches_published_example():
    # Alfriend et al., Spacecraft Formation Flying, example 3.1: mean a = 7100 km, i = 70 deg,
    # raan = 45 deg, q1 = q2 = 0.05, argument of latitude 0. The argument of latitude and q1
    # differ from the book in the fifth decimal, a second-order difference between the two
    # books' forms of the theory. Reproduced with Re = 6378.1363 km (the book's) and the EGM96
    # J2; WGS 84's J2 moves a by 3 cm.
    body = Body("earth", 398600.4415e9, 6378136.3, 1.0826267e-3)
    e = math.hypot(0.05, 0.05)
    argp = math.atan2(0.05, 0.05)
    E = math.atan2(math.sqrt(1 - e * e) * math.sin(-argp), e + math.cos(-argp))
    mean = [7100e3, e, math.radians(70.0), math.radians(45.0), argp, E - e * math.sin(E)]
    a, e_osc, i, raan, argp_osc, M = mean_to_osculating(mean, body)
    assert a == pytest.approx(7109.31795e3, abs=5e-3)
    assert i == pytest.approx(1.22196, abs=5e-6)
    assert raan == pytest.approx(0.78547, abs=5e-6)
    assert e_osc * math.sin(argp_osc) == pytest.approx(0.05003, abs=5e-6)
    assert e_osc * math.cos(argp_osc) == pytest.approx(0.05063, abs=2e-5)
    theta = wrapped(argp_osc + true_anomaly([a, e_osc, i, raan, argp_osc, M]))
    assert theta == pytest.approx(0.00005, abs=3e-5)


def test_round_trip_is_second_order_in_j2():
    rng = np.random.default_rng(0)
    for _ in range(300):
        mean = np.array(
            [
                rng.uniform(6.6e6, 7.4e6),
                rng.uniform(0.0, 0.01),
                np.radians(rng.uniform(0.0, 120.0)),
                *rng.uniform(0.0, 2 * np.pi, 3),
            ]
        )
        back = osculating_to_mean(mean_to_osculating(mean, EARTH), EARTH)
        # J2^2 a is about 8 m in low Earth orbit.
        assert abs(back[0] - mean[0]) < 100.0
        assert abs(back[1] - mean[1]) < 1e-5
        assert abs(back[2] - mean[2]) < 1e-6
        assert abs(wrapped(back[3:].sum() - mean[3:].sum())) < 1e-5


def test_map_is_finite_and_continuous_at_its_singular_inclinations():
    a = EARTH.radius + 500e3
    critical = math.acos(math.sqrt(0.2))
    for i in (0.0, critical):
        for e in (0.0, 0.01):
            mean = [a, e, i, 0.4, 0.9, 0.3]
            assert np.all(np.isfinite(osculating_to_mean(mean_to_osculating(mean, EARTH), EARTH)))
    # Circular orbits are mapped exactly through the critical inclination.
    at = mean_to_osculating([a, 0.0, critical, 0.4, 0.0, 0.3], EARTH)
    near = mean_to_osculating([a, 0.0, critical + 1e-9, 0.4, 0.0, 0.3], EARTH)
    assert np.allclose(at, near, rtol=1e-8, atol=1e-8)


def test_mean_elements_of_a_j2_orbit_drift_only_at_the_secular_rates():
    mean0 = np.array([7000e3, 0.05, np.radians(51.6), 0.4, 0.9, 0.3])
    r0, v0 = state_from_elements(mean_to_osculating(mean0, EARTH), EARTH.mu)
    period = 2 * np.pi * math.sqrt(mean0[0] ** 3 / EARTH.mu)
    times = np.linspace(0.0, 5 * period, 101)
    sol = solve_ivp(
        lambda t, y: np.r_[y[3:], gravity(y[:3], EARTH, j2=True)],
        (0.0, times[-1]),
        np.r_[r0, v0],
        method="DOP853",
        t_eval=times,
        rtol=1e-12,
        atol=1e-6,
    )
    rates = j2_secular_rates(mean0[0], mean0[1], mean0[2], EARTH)
    for t, y in zip(times, sol.y.T, strict=True):
        osculating = elements_from_state(y[:3], y[3:], EARTH.mu)
        mean = osculating_to_mean(osculating, EARTH)
        # Osculating a swings by 13 km here; the mean value stays within J2^2 a.
        assert abs(mean[0] - mean0[0]) < 50.0
        assert abs(mean[1] - mean0[1]) < 1e-5
        assert abs(mean[2] - mean0[2]) < 1e-6
        # The second-order secular terms the first-order rates leave out add about 1e-5 rad
        # per orbit.
        predicted = mean0[3:] + np.array(rates) * t
        assert np.max(np.abs(wrapped(mean[3:] - predicted))) < 2e-4
