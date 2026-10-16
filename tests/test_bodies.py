import numpy as np
import pytest

from vbar.dynamics import mean_motion
from vbar.dynamics.bodies import EARTH, KERBIN, get_body


def test_earth_j2_follows_from_the_wgs84_defining_constants():
    # Normal-field J2 of the ellipsoid with WGS 84's a, f, GM and spin rate (Moritz, Geodetic
    # Reference System 1980, closed form), so GM, radius and J2 are one matched set.
    a, f, gm, w = EARTH.radius, 1 / 298.257223563, EARTH.mu, 7.292115e-5
    e2 = f * (2 - f)
    e_prime = np.sqrt(e2 / (1 - e2))
    m = w**2 * a**2 * a * (1 - f) / gm
    q0 = 0.5 * ((1 + 3 / e_prime**2) * np.arctan(e_prime) - 3 / e_prime)
    assert EARTH.j2 == pytest.approx(e2 / 3 * (1 - 2 / 15 * m * e_prime / q0), rel=1e-11)
    assert EARTH.mu == 3.986004418e14
    assert EARTH.radius == 6378137.0


def test_kerbin_synchronous_orbit_period_is_its_sidereal_day():
    # KSP wiki: synchronous altitude 2863.33 km, sidereal rotation period 21549.425 s.
    n = mean_motion(KERBIN.radius + 2_863_330.0, KERBIN.mu)
    assert 2 * np.pi / n == pytest.approx(21_549.425, rel=1e-5)


def test_get_body_accepts_name_or_body():
    assert get_body("Earth") is EARTH
    assert get_body(KERBIN) is KERBIN


def test_unknown_body_is_rejected():
    with pytest.raises(ValueError, match="unknown body"):
        get_body("mun")
