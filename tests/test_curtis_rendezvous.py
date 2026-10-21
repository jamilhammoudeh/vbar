"""Curtis, Orbital Mechanics for Engineering Students, section 7.5: two-impulse rendezvous.

First edition Examples 7.2 and 7.3; second and third editions Examples 7.4 and 7.5. Curtis's
comoving frame is the Hill frame here: x along the target's radius, y along its velocity,
z = x cross y, so a chaser behind the target has y < 0. Constants as in the book:
mu = 398600 km^3/s^2 and a 300 km circular target orbit, whose mean motion sqrt(mu / r^3) the
book's matrices use unrounded even where the text prints a rounded value.
"""

import numpy as np
import pytest

from vbar.agents.cw_targeting import cw_lambert, two_impulse
from vbar.dynamics import cw_discrete

MU = 398600.0e9
R = (6378.0 + 300.0) * 1e3
N = np.sqrt(MU / R**3)


def approx_printed(text_kmps):
    """A value in m/s within one unit of the last printed digit of a km/s figure, relative."""
    printed = float(text_kmps)
    unit = 10.0 ** -len(text_kmps.split(".")[1])
    return pytest.approx(printed * 1e3, rel=unit / abs(printed))


def test_curtis_example_7_5_two_impulses_from_two_kilometres_behind():
    # Example 7.5 (first edition 7.3): same orbit, 2 km behind, at rest relative, 1.49 h.
    # Each impulse prints as 0.1226 m/s and the total as 0.2452 m/s; the components are the
    # book's formulas evaluated at the exact mean motion.
    state = np.array([0.0, -2000.0, 0.0, 0.0, 0.0, 0.0])
    tf = 1.49 * 3600.0
    dv1, dv2 = two_impulse(N, state, np.zeros(3), tf)
    assert np.linalg.norm(dv1) == pytest.approx(0.1226, rel=1e-4 / 0.1226)
    assert np.linalg.norm(dv2) == pytest.approx(0.1226, rel=1e-4 / 0.1226)
    assert np.linalg.norm(dv1) + np.linalg.norm(dv2) == pytest.approx(0.2452, rel=1e-4 / 0.2452)
    assert dv1[0] == approx_printed("-9.4824e-6") and dv1[1] == approx_printed("-1.22248e-4")
    assert dv2[0] == approx_printed("-9.4824e-6") and dv2[1] == approx_printed("1.22248e-4")
    assert dv1[2] == 0.0 and dv2[2] == 0.0


def test_curtis_example_7_5_departure_and_arrival_velocities():
    tf = 1.49 * 3600.0
    v0, vf = cw_lambert(N, np.array([0.0, -2000.0, 0.0]), np.zeros(3), tf)
    assert v0[0] == approx_printed("-9.4824e-6") and v0[1] == approx_printed("-1.22248e-4")
    # The arrival velocity mirrors the departure one in x: delta-vf- = (+9.4824e-6, -1.22248e-4, 0).
    assert vf[0] == approx_printed("9.4824e-6") and vf[1] == approx_printed("-1.22248e-4")


def test_curtis_example_7_4_two_impulses_from_twenty_kilometres_in_eight_hours():
    # Example 7.4 (first edition 7.2): chaser at (20, 20, 20) km moving at (-0.02, 0.02, -0.005)
    # km/s, rendezvous in 8 h. The first edition's printed final impulse has a misprint and uses
    # a rounded mean motion; the later editions' figures are reproduced.
    state = np.array([20e3, 20e3, 20e3, -20.0, 20.0, -5.0])
    tf = 8.0 * 3600.0
    dv1, dv2 = two_impulse(N, state, np.zeros(3), tf)
    for value, text in zip(dv1, ("0.0293046", "-0.0667472", "0.0129834"), strict=True):
        assert value == approx_printed(text)
    for value, text in zip(dv2, ("0.0257978", "0.000470870", "0.0244767"), strict=True):
        assert value == approx_printed(text)
    assert np.linalg.norm(dv1) == approx_printed("0.0740440")
    assert np.linalg.norm(dv2) == approx_printed("0.0355649")
    assert np.linalg.norm(dv1) + np.linalg.norm(dv2) == approx_printed("0.109609")


def test_curtis_example_7_4_final_impulse_cancels_the_arrival_velocity():
    # Delta-vf = -delta-vf-: after the first impulse the chaser coasts to the target, and the
    # second impulse is the negative of the velocity it arrives with (eq 7.58).
    state = np.array([20e3, 20e3, 20e3, -20.0, 20.0, -5.0])
    tf = 8.0 * 3600.0
    dv1, dv2 = two_impulse(N, state, np.zeros(3), tf)
    _, vf = cw_lambert(N, state[:3], np.zeros(3), tf)
    np.testing.assert_allclose(dv2, -vf, rtol=1e-12, atol=0.0)
    arrival = cw_discrete(N, tf)[0] @ (state + np.r_[np.zeros(3), dv1])
    # Rounding floor of the solve and the transition product, relative to the impulse size.
    assert np.linalg.norm(arrival[:3]) <= 1e-9 * np.linalg.norm(state[:3])
    assert np.linalg.norm(arrival[3:] + dv2) <= 1e-12 * np.linalg.norm(dv2)
