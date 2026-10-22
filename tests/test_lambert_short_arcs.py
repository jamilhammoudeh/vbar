import math

import numpy as np
import pytest

from vbar.dynamics.bodies import EARTH
from vbar.dynamics.kepler import propagate_kepler
from vbar.dynamics.lambert import lambert

MU = EARTH.mu
R = 6878e3
PERIOD = 2.0 * math.pi * math.sqrt(R**3 / MU)
R1 = np.array([R, 0.0, 0.0])


def on_circle(angle):
    return R * np.array([math.cos(angle), math.sin(angle), 0.0])


def assert_close(actual, expected, rel):
    assert np.linalg.norm(np.asarray(actual) - expected) <= rel * np.linalg.norm(expected)


def test_a_millisecond_arc_on_a_circle_is_refused_as_too_short():
    # y, the remainder of two terms of size r here, is 4e-6 m, below what rounding resolves;
    # the solver used to end where y <= 0 and fail in the square root.
    with pytest.raises(ValueError, match="tof too short"):
        lambert(R1, on_circle(1.0), 1e-3, MU)


@pytest.mark.parametrize("angle", [0.1, 1.0, 3.0])
def test_short_arcs_either_round_trip_or_are_refused(angle):
    # From 1e-7 to 1e-3 of the period the root crosses from below the rounding floor of y to
    # above it; every case either reproduces r2 and v2 through the independent Kepler propagator
    # to 1e-9 or raises the one refusal, never another exception.
    r2 = on_circle(angle)
    outcomes = set()
    for fraction in np.logspace(-7, -3, 25):
        tof = fraction * PERIOD
        try:
            v1, v2 = lambert(R1, r2, tof, MU)
        except ValueError as refusal:
            assert str(refusal).startswith("tof too short")
            outcomes.add("refused")
            continue
        r2_back, v2_back = propagate_kepler(R1, v1, tof, MU)
        assert_close(r2_back, r2, 1e-9)
        assert_close(v2_back, v2, 1e-9)
        outcomes.add("solved")
    assert outcomes == {"refused", "solved"}


@pytest.mark.parametrize("tof", [1.0, 1e-2])
def test_tiny_arcs_on_a_circle_keep_the_circular_velocity(tof):
    # The terms of y that cancel on a small arc are of size r theta^2 / 4, not r1 + r2, so a
    # 1e-2 s arc (0.1 mrad) is still exact to rounding rather than refused.
    a = EARTH.radius + 500e3
    n = math.sqrt(MU / a**3)
    f1, f2 = 0.3, 0.3 + n * tof
    r1 = a * np.array([math.cos(f1), math.sin(f1), 0.0])
    r2 = a * np.array([math.cos(f2), math.sin(f2), 0.0])
    v1, v2 = lambert(r1, r2, tof, MU)
    assert_close(v1, n * a * np.array([-math.sin(f1), math.cos(f1), 0.0]), 1e-9)
    assert_close(v2, n * a * np.array([-math.sin(f2), math.cos(f2), 0.0]), 1e-9)


def solved_or_refused(r1, r2, tof, prograde):
    """'solved' after a Kepler round trip to 1e-9, or 'refused'; any other exception escapes."""
    try:
        v1, v2 = lambert(r1, r2, tof, MU, prograde=prograde)
    except ValueError as refusal:
        assert str(refusal).startswith("tof too short")
        return "refused"
    r2_back, v2_back = propagate_kepler(r1, v1, tof, MU)
    assert_close(r2_back, r2, 1e-9)
    assert_close(v2_back, v2, 1e-9)
    return "solved"


def test_a_zero_newton_slope_is_solved_or_refused():
    # At 1e-8 of the period the time-of-flight slope rounds to exactly zero on this long-way
    # transfer; the solver used to divide by it.
    r1 = np.array([-25451061.621939126, 9964178.299544405, 6619581.630824058])
    r2 = np.array([-14536752.970729942, -15230073.59687847, -35492887.19255181])
    solved_or_refused(r1, r2, 0.000903994670478089, prograde=False)


def test_transfers_below_2e_8_of_the_period_are_solved_or_refused():
    # Random geometry from low orbit to beyond geostationary radius, both senses, times of flight
    # log-uniform from 1e-14 to 2e-8 of the departure circular period.
    rng = np.random.default_rng(20261021)
    for _ in range(200):
        r1, r2 = rng.normal(size=(2, 3))
        r1 *= rng.uniform(6.6e6, 4.5e7) / np.linalg.norm(r1)
        r2 *= rng.uniform(6.6e6, 4.5e7) / np.linalg.norm(r2)
        period = 2.0 * math.pi * math.sqrt(np.linalg.norm(r1) ** 3 / MU)
        tof = 10.0 ** rng.uniform(-14.0, math.log10(2e-8)) * period
        solved_or_refused(r1, r2, tof, prograde=bool(rng.integers(2)))


@pytest.mark.parametrize(
    ("r1", "r2", "tof"),
    [
        # Elliptic, 0.59 of the departure period.
        (
            [5174828.869165987, -376924.9306481218, -10815698.267246123],
            [12694638.207233906, 14666734.256223926, 4205243.073860812],
            7655.453308547914,
        ),
        # Hyperbolic, 4e-3 of the departure period, semi-major axis -5.6 km.
        (
            [26148883.836053137, 21089660.165988203, 1788658.1882246004],
            [-35322812.136641495, 31165.115815238245, 17597708.6091309],
            249.72898394162144,
        ),
    ],
)
def test_iteration_ending_on_a_closed_bracket_returns_the_transfer(r1, r2, tof):
    # F, the time-of-flight residual, is the difference of terms of size sqrt(mu) tof, so near
    # the root it sits at its rounding floor (about 1e-4 m^1.5 here) and no longer halves from
    # one Newton step to the next. Those steps are rejected, the bisections close the bracket to
    # a few ulp of z before a Newton step is small enough to stop on, and the solver returns the
    # bracket end where y > 0. Found by instrumenting the exit path over random problems.
    assert solved_or_refused(np.array(r1), np.array(r2), tof, prograde=True) == "solved"
