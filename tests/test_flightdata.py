"""Flight-data replay: GRACE-FO orbits of 2019-12-03 against the two-body, J2 and drag rungs.

The fixture is an excerpt of PO.DAAC collection GRACEFO_L1B_ASCII_GRAV_JPL_RL04
(doi:10.5067/GFL1B-ASJ04, CC0), written by examples/gracefo_excerpt.py. The error bounds below were
fixed before the flight data were read, from a budget of the forces the models leave out; the
derivation is in docs/flight-data.md. The ten-orbit claims are behind the `slow` marker.
"""

import datetime
import math
from pathlib import Path

import numpy as np
import pytest

from vbar.dynamics import EARTH, TwoBodyDynamics, cw_stm
from vbar.dynamics.kepler import elements_from_state
from vbar.dynamics.perturbations import Drag, HarrisPriester
from vbar.flightdata import (
    earth_fixed_to_inertial,
    error_envelope,
    gmst,
    hill_states,
    read_level1b,
    read_orbit_excerpt,
    replay,
    sun_direction,
    write_orbit_excerpt,
)

EXCERPT = Path(__file__).parent / "data" / "gracefo_2019-12-03.csv"
# The bound stated before the run: 30 m plus 66 m per orbit for the geopotential beyond J2
# (1.5 times the simulated envelope against EGM96 to degree 30), plus the response to a 5e-8
# m/s^2 bound on drag model error, lunar and solar tides and the frame's precession.
PER_ORBIT_M = 66.0
FLOOR_M = 30.0
FORCE_BOUND = 5e-8
# 601.2 kg launch mass over C_D 2.3 times the 0.955 m^2 front panel (handbook, 2019, table 5).
BALLISTIC = 601.2 / (2.3 * 0.9551567)


def gps_seconds(utc: datetime.datetime) -> float:
    """GPS seconds past 2000-01-01 12:00:00 GPS of a UTC time after 2017, when GPS - UTC = 18 s."""
    return (utc - datetime.datetime(2000, 1, 1, 12)).total_seconds() + 18.0


@pytest.fixture(scope="module")
def flown():
    excerpt = read_orbit_excerpt(EXCERPT)
    t = excerpt["t"] - excerpt["t"][0]
    theta0 = float(gmst(excerpt["t"][0]))
    states = []
    for key in ("target", "chaser"):
        r, v = earth_fixed_to_inertial(t, excerpt[key][:, :3], excerpt[key][:, 3:], theta0)
        states.append(np.hstack([r, v]))
    a = elements_from_state(states[0][0, :3], states[0][0, 3:], EARTH.mu)[0]
    n = math.sqrt(EARTH.mu / a**3)
    return {"excerpt": excerpt, "t": t, "target": states[0], "chaser": states[1], "n": n}


def bound(t, n):
    return FLOOR_M + error_envelope(t, n, PER_ORBIT_M, FORCE_BOUND)


def run(flown, dynamics, orbits):
    keep = flown["t"] <= orbits * 2.0 * math.pi / flown["n"] + 1e-6
    return replay(flown["t"][keep], flown["target"][keep], flown["chaser"][keep], dynamics)


def test_a_point_fixed_on_the_earth_turns_with_the_earth():
    t = np.array([0.0, 100.0])
    r = np.array([[EARTH.radius, 0.0, 0.0]] * 2)
    r_i, v_i = earth_fixed_to_inertial(t, r, np.zeros((2, 3)), theta0=0.3)
    w = EARTH.spin_rate
    angle = 0.3 + w * 100.0
    np.testing.assert_allclose(r_i[1], EARTH.radius * np.array([np.cos(angle), np.sin(angle), 0.0]))
    np.testing.assert_allclose(
        v_i[1], w * EARTH.radius * np.array([-np.sin(angle), np.cos(angle), 0.0]), rtol=1e-12
    )


def test_sidereal_angle_at_j2000_is_the_iau_1982_value():
    # The IAU 1982 definition puts Greenwich mean sidereal time at J2000.0 UT1 at
    # 18h 41m 50.54841s; in 2000 GPS ran 13 s ahead of UTC.
    hours = 18.0 + 41.0 / 60.0 + 50.54841 / 3600.0
    assert float(gmst(13.0, gps_minus_utc_s=13.0)) == pytest.approx(math.radians(15.0 * hours))


def test_sidereal_angle_returns_after_one_sidereal_day():
    t0 = gps_seconds(datetime.datetime(2019, 12, 3))
    # 86164.0905 s is the mean sidereal day; the linear expression turns 360 deg in it.
    change = float(gmst(t0 + 86164.0905) - gmst(t0))
    assert abs(math.remainder(change, 2.0 * math.pi)) < 1e-6


@pytest.mark.parametrize(
    "utc, declination_deg, right_ascension_deg",
    [
        # Solstice and equinox times from the US Naval Observatory's table of Earth's seasons.
        (datetime.datetime(2019, 12, 22, 4, 19), -23.44, 270.0),
        (datetime.datetime(2020, 3, 20, 3, 50), 0.0, 0.0),
        (datetime.datetime(2020, 6, 20, 21, 44), 23.44, 90.0),
    ],
)
def test_sun_direction_at_the_solstices_and_equinox(utc, declination_deg, right_ascension_deg):
    s = sun_direction(gps_seconds(utc))
    assert np.linalg.norm(s) == pytest.approx(1.0, rel=1e-12)
    # The low-precision formula is good to about 0.1 deg; the event times are to the minute.
    assert math.degrees(math.asin(s[2])) == pytest.approx(declination_deg, abs=0.05)
    ra = math.degrees(math.atan2(s[1], s[0])) % 360.0
    assert abs(math.remainder(ra - right_ascension_deg, 360.0)) < 0.1


def test_level1b_reader_takes_columns_from_the_header(tmp_path):
    path = tmp_path / "GNV1B_test.txt"
    path.write_text(
        "header:\n  variables:\n    - gps_time:\n        units: second\n    - GRACEFO_id:\n"
        "    - xpos:\n        units: m\n# End of YAML header\n100 C 1.5\n101 C -2.25\n",
        encoding="utf-8",
    )
    columns = read_level1b(path)
    np.testing.assert_array_equal(columns["gps_time"], [100.0, 101.0])
    np.testing.assert_array_equal(columns["xpos"], [1.5, -2.25])
    assert list(columns["GRACEFO_id"]) == ["C", "C"]


@pytest.mark.parametrize(
    "body",
    [
        "header:\n  variables:\n    - a:\n100\n",
        "header:\n  variables:\n    - a:\n    - b:\n# End of YAML header\n1 2 3\n",
    ],
)
def test_level1b_reader_refuses_a_broken_file(tmp_path, body):
    path = tmp_path / "broken.txt"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        read_level1b(path)


def test_excerpt_round_trip_keeps_the_written_precision(tmp_path):
    rng = np.random.default_rng(0)
    target = rng.uniform(-7e6, 7e6, (3, 6))
    chaser = rng.uniform(-7e6, 7e6, (3, 6))
    path = tmp_path / "excerpt.csv"
    write_orbit_excerpt(path, [1.0, 2.0, 3.0], target, chaser, [10.0, 20.0, 30.0], ["source"])
    back = read_orbit_excerpt(path)
    # Positions to 0.1 mm and velocities to 0.1 um/s.
    np.testing.assert_allclose(back["target"][:, :3], target[:, :3], atol=5e-5)
    np.testing.assert_allclose(back["chaser"][:, 3:], chaser[:, 3:], atol=5e-8)
    assert path.read_text(encoding="utf-8").startswith("# source\n")


def test_fixture_is_the_cited_day_at_two_minute_steps():
    excerpt = read_orbit_excerpt(EXCERPT)
    text = EXCERPT.read_text(encoding="utf-8")
    assert "doi:10.5067/GFL1B-ASJ04" in text and "CC0" in text
    assert excerpt["t"][0] == gps_seconds(datetime.datetime(2019, 12, 3)) - 18.0
    np.testing.assert_array_equal(np.diff(excerpt["t"]), 120.0)
    assert excerpt["t"][-1] - excerpt["t"][0] == 16 * 3600.0


def test_gnv1b_range_agrees_with_the_kband_range_to_millimetres():
    excerpt = read_orbit_excerpt(EXCERPT)
    gps_range = np.linalg.norm(excerpt["chaser"][:, :3] - excerpt["target"][:, :3], axis=1)
    residual = gps_range - excerpt["range"]
    # The K-band range carries an unknown constant bias, removed by the mean.
    rms = np.sqrt(np.mean((residual - residual.mean()) ** 2))
    assert rms < 0.01
    # Regression record of the excerpt, quoted in docs/flight-data.md.
    assert rms == pytest.approx(1.28e-3, abs=0.05e-3)


def test_separation_is_the_nominal_formation_and_far_beyond_cw():
    excerpt = read_orbit_excerpt(EXCERPT)
    rho = np.linalg.norm(excerpt["chaser"][0, :3] - excerpt["target"][0, :3])
    r = np.linalg.norm(excerpt["target"][0, :3])
    assert 170e3 < rho < 270e3
    # The curvature drop rho^2 / (2 r) is kilometres, far outside the CW model's reach.
    assert rho**2 / (2.0 * r) > 2e3


def test_j2_rung_stays_inside_the_stated_bound_over_two_orbits(flown):
    result = run(flown, TwoBodyDynamics(EARTH, j2=True), 2)
    assert np.all(result.position_error <= bound(result.t, flown["n"]))
    assert result.position_error[0] < 1e-6


def test_envelope_term_follows_the_short_time_limit():
    # For t much shorter than an orbit the velocity-to-position block is t I, so the force term
    # integrates to A t^2 / 2.
    n, t = 1.1e-3, 10.0
    assert error_envelope([t], n, 0.0, 1e-6)[0] == pytest.approx(0.5e-6 * t**2, rel=1e-4)
    period = 2.0 * math.pi / n
    assert error_envelope([period], n, 66.0, 0.0)[0] == pytest.approx(66.0, rel=1e-12)


def test_reset_inertial_refuses_a_bad_state():
    dynamics = TwoBodyDynamics(EARTH)
    with pytest.raises(ValueError):
        dynamics.reset_inertial(np.zeros(5), np.zeros(6))
    with pytest.raises(ValueError):
        dynamics.reset_inertial(np.full(6, np.nan), np.zeros(6))


def test_replay_refuses_unordered_times(flown):
    with pytest.raises(ValueError):
        replay([0.0, 0.0], flown["target"][:2], flown["chaser"][:2], TwoBodyDynamics(EARTH))


@pytest.mark.slow
def test_ten_orbit_replay_meets_every_stated_bound(flown):
    n = flown["n"]
    period = 2.0 * math.pi / n
    sun = sun_direction(float(flown["excerpt"]["t"][0]))
    drag = Drag(HarrisPriester(sun, 6.0), BALLISTIC, BALLISTIC)
    two_body = run(flown, TwoBodyDynamics(EARTH), 10)
    j2 = run(flown, TwoBodyDynamics(EARTH, j2=True), 10)
    j2_drag = run(flown, TwoBodyDynamics(EARTH, j2=True, perturbations=(drag,)), 10)
    orbits = j2.t / period
    envelope = bound(j2.t, n)
    for result in (j2, j2_drag):
        assert np.all(result.position_error <= envelope)
        assert np.polyfit(orbits, result.position_error, 1)[0] <= 95.0
    first = orbits <= 1.0 + 1e-9
    assert np.max(two_body.position_error[first]) >= 5.0 * np.max(j2.position_error[first])
    assert np.max(np.abs(j2_drag.position_error - j2.position_error)) <= 100.0
    one_orbit = np.argmin(np.abs(orbits - 1.0))
    cw = cw_stm(n, j2.t[one_orbit]) @ j2.flown[0]
    assert np.linalg.norm(cw[:3] - j2.flown[one_orbit, :3]) > 10e3
    kband = flown["excerpt"]["range"][: len(j2.t)]
    for result in (j2, j2_drag):
        modelled = np.linalg.norm(result.model[:, :3], axis=1)
        drift = np.abs((modelled - modelled[0]) - (kband - kband[0]))
        assert np.all(drift <= envelope)
    # Regression record of this run, quoted in docs/flight-data.md.
    j2_record = (52.96, 106.47, 254.17, 500.59)
    two_body_record = (2911.0, 5820.0, 14551.0, 29115.0)
    for k, j2_max, two_body_max in zip((1, 2, 5, 10), j2_record, two_body_record, strict=True):
        upto = orbits <= k + 1e-9
        assert np.max(j2.position_error[upto]) == pytest.approx(j2_max, abs=0.5)
        assert np.max(two_body.position_error[upto]) == pytest.approx(two_body_max, rel=0.01)
    assert np.polyfit(orbits, j2.position_error, 1)[0] == pytest.approx(49.98, rel=0.01)
    assert np.polyfit(orbits, two_body.position_error, 1)[0] == pytest.approx(2910.0, rel=0.01)
    assert np.min(envelope - j2.position_error) == pytest.approx(28.5, abs=0.5)
    assert np.interp(1.0, orbits, envelope) == pytest.approx(98.7, abs=0.5)
    assert np.interp(10.0, orbits, envelope) == pytest.approx(929.6, abs=0.5)
    assert j2.velocity_error[-1] == pytest.approx(0.0041, rel=0.05)
    drag_change = np.max(np.abs(j2_drag.position_error - j2.position_error))
    assert drag_change == pytest.approx(0.13, abs=0.05)
    worst_cw = max(
        np.linalg.norm((cw_stm(n, s) @ j2.flown[0])[:3] - x[:3])
        for s, x in zip(j2.t[first], j2.flown[first], strict=True)
    )
    assert worst_cw == pytest.approx(89429.0, rel=0.01)


def test_hill_states_of_identical_orbits_are_zero(flown):
    zero = hill_states(flown["target"][:3], flown["target"][:3])
    np.testing.assert_array_equal(zero, np.zeros((3, 6)))


GOOD = np.tile([7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0], (2, 1))


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: replay([0.0, math.inf], GOOD, GOOD, TwoBodyDynamics(EARTH)), "finite"),
        (
            lambda: replay([0.0, 60.0], GOOD, np.vstack([GOOD[0], np.full(6, np.nan)]), None),
            "finite",
        ),
        (lambda: error_envelope([-100.0], 1e-3, 66.0, 5e-8), "negative"),
        (lambda: error_envelope([100.0], 1e-3, 66.0, 5e-8, samples=1), "samples"),
        (lambda: error_envelope([100.0], 1e-3, -1.0, 5e-8), "per_orbit_m"),
        (lambda: error_envelope([100.0], 1e-3, 66.0, -5e-8), "accel_mps2"),
        (lambda: error_envelope([], 1e-3, 66.0, 5e-8), "at least one"),
        (lambda: error_envelope([math.nan], 1e-3, 66.0, 5e-8), "finite"),
        (lambda: error_envelope([100.0], math.nan, 66.0, 5e-8), "n must"),
        (lambda: error_envelope([100.0], 0.0, 66.0, 5e-8), "n must"),
        (lambda: earth_fixed_to_inertial([0.0], [7e6, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0), "shape"),
        (
            lambda: earth_fixed_to_inertial([0.0], [[7e6, 0.0, 0.0]], [[0.0, 0.0, 0.0]], math.nan),
            "theta0",
        ),
        (lambda: hill_states(GOOD, GOOD[:1]), "shape"),
        (lambda: hill_states(np.zeros((1, 6)), GOOD[:1]), "centre"),
        (lambda: gmst(math.nan), "finite"),
        (lambda: sun_direction(math.nan), "finite"),
    ],
)
def test_bad_flight_data_inputs_are_refused(call, message):
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize(
    "target, biased_range",
    [(GOOD[:1], [1.0, 2.0]), (GOOD[:, :5], [1.0, 2.0]), (GOOD, [1.0, math.nan])],
)
def test_a_bad_excerpt_is_refused_before_anything_is_written(tmp_path, target, biased_range):
    path = tmp_path / "excerpt.csv"
    with pytest.raises(ValueError):
        write_orbit_excerpt(path, [0.0, 1.0], target, GOOD, biased_range)
    assert not path.exists()


def test_an_excerpt_with_missing_fields_is_refused(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("# t\n" + ",".join(["1.0"] * 13) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="14"):
        read_orbit_excerpt(path)
