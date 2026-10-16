import numpy as np
import pytest

from vbar.conventions import inertial_to_hill
from vbar.dynamics.bodies import EARTH
from vbar.dynamics.kepler import elements_from_state, propagate_kepler
from vbar.dynamics.mean_elements import osculating_to_mean
from vbar.dynamics.models import TwoBodyDynamics

ZERO = np.zeros(3)
ELLIPTIC = [7500e3, 0.1, np.radians(51.6), 0.3, 1.1, 2.0]


def test_elements_set_the_target_orbit_and_mean_motion():
    m = TwoBodyDynamics(elements=ELLIPTIC)
    m.reset(np.zeros(6))
    target = m.target_state()
    assert elements_from_state(target[:3], target[3:], EARTH.mu) == pytest.approx(
        ELLIPTIC, rel=1e-12
    )
    assert m.n == pytest.approx(np.sqrt(EARTH.mu / 7500e3**3), rel=1e-15)
    assert m.altitude == 7500e3 - EARTH.radius
    assert m.inclination == ELLIPTIC[2]


def test_elliptic_target_and_chaser_follow_kepler_over_one_orbit():
    m = TwoBodyDynamics(elements=ELLIPTIC)
    m.reset([300.0, -800.0, 150.0, 0.1, -0.2, 0.05])
    target, chaser = m.target_state(), m.chaser_state()
    period = 2 * np.pi / m.n
    final, _ = m.advance(ZERO, period)
    r_t, v_t = propagate_kepler(target[:3], target[3:], period, EARTH.mu)
    r_c, v_c = propagate_kepler(chaser[:3], chaser[3:], period, EARTH.mu)
    expected = inertial_to_hill(r_t, v_t, r_c - r_t, v_c - v_t)
    assert np.max(np.abs(final[:3] - expected[:3])) < 1e-5
    assert np.max(np.abs(final[3:] - expected[3:])) < 1e-8


def test_j2_elements_are_mean_elements():
    # A circular J2 target started at a mean argument of latitude of 45 deg.
    mean = [EARTH.radius + 500e3, 0.0, np.radians(51.6), 0.7, 0.0, np.radians(45.0)]
    m = TwoBodyDynamics(j2=True, elements=mean)
    m.reset(np.zeros(6))
    target = m.target_state()
    recovered = osculating_to_mean(elements_from_state(target[:3], target[3:], EARTH.mu), EARTH)
    # The first-order map round-trips to order J2^2, tens of metres in a.
    assert abs(recovered[0] - mean[0]) < 50.0
    assert recovered[1] < 1e-5
    assert recovered[3] == pytest.approx(mean[3], abs=1e-6)
    latitude = (recovered[4] + recovered[5]) % (2 * np.pi)
    assert latitude == pytest.approx(mean[5], abs=1e-5)
