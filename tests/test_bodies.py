import numpy as np
import pytest

from vbar.dynamics import mean_motion
from vbar.dynamics.bodies import EARTH, KERBIN, get_body


def test_earth_j2_matches_egm96_c20():
    c20 = -0.484165371736e-3  # EGM-96 fully normalized C20
    assert EARTH.j2 == pytest.approx(-np.sqrt(5.0) * c20, rel=1e-8)


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
