"""Two-impulse Clohessy-Wiltshire targeting: known answers, the singular set and scaling."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import brentq, minimize_scalar

from vbar.agents.cw_targeting import conditioning, cw_lambert, two_impulse
from vbar.dynamics import EARTH, cw_discrete, cw_stm, mean_motion

N = mean_motion(EARTH.radius + 500e3)
REST = np.array([0.0, -300.0, 0.0, 0.0, 0.0, 0.0])
HOLD = np.array([0.0, -100.0, 0.0])

# Alfriend, Vadali, Gurfil, How and Breger, Spacecraft Formation Flying, example 5.2, in the
# book's normalized units: n = 1, lengths in orbit radii, so velocities are in units of n a.
EX52_R0 = np.array([0.01, 0.02, 0.015])
EX52_V0 = np.array([0.001, 0.001, 0.001])


def assert_printed(values, printed):
    """Each value within one unit of the last printed digit of its book value."""
    for value, text in zip(values, printed, strict=True):
        unit = 10.0 ** -len(text.split(".")[1])
        assert abs(value - float(text)) <= unit, (value, text)


def test_alfriend_example_5_2_two_impulse_rendezvous():
    # The printed transfer time (3.22 h, two orbits) is a misprint: the printed impulses
    # correspond to n tf = 2 rad. The printed second impulse is +v(tf) where the rendezvous
    # needs -v(tf); the values here are recomputed at the stated inputs with that sign.
    dv1, dv2 = two_impulse(1.0, np.r_[EX52_R0, EX52_V0], np.zeros(3), 2.0)
    assert_printed(dv1, ("-0.00180", "-0.01927", "0.005865"))
    assert_printed(dv2, ("0.00562", "-0.00173", "0.0165"))
    assert np.linalg.norm(dv1) + np.linalg.norm(dv2) == pytest.approx(0.03774, abs=5e-5)


def test_the_cheapest_transfer_of_example_5_2_lies_past_the_printed_transfer_time():
    # Over n tf in [0.05, 10], singular times skipped, the printed minimum total (0.0375 at the
    # printed time, n tf = 2 rad) is a misprint; recomputed at the stated inputs the minimum is
    # 0.03572 at n tf = 4.64, which a scan stopping at the printed time misses. A 400-point
    # grid brackets the minimum and a bounded search locates it.
    state = np.r_[EX52_R0, EX52_V0]

    def total(tf):
        try:
            dv1, dv2 = two_impulse(1.0, state, np.zeros(3), tf)
        except ValueError:
            return np.inf
        return np.linalg.norm(dv1) + np.linalg.norm(dv2)

    grid = np.linspace(0.05, 10.0, 400)
    totals = np.array([total(tf) for tf in grid])
    k = int(np.argmin(totals))
    found = minimize_scalar(total, bounds=(grid[k - 1], grid[k + 1]), method="bounded")
    assert found.fun == pytest.approx(0.03572, abs=1e-4)
    assert found.x == pytest.approx(4.64, abs=0.01)
    assert np.min(totals[grid <= 2.0]) > found.fun + 1e-3


def test_alfriend_example_9_2_short_transfer_departure_velocity():
    # Alfriend et al., example 9.2: a = 7170 km with mu = 398600.4418 km^3/s^2, from
    # (25, 25, 50) m to the origin in 0.01 orbits (60.42 s). The book prints four digits; the
    # fifth is recomputed at these constants.
    n = np.sqrt(3.986004418e14 / 7170e3**3)
    v0, _ = cw_lambert(n, np.array([25.0, 25.0, 50.0]), np.zeros(3), 0.01 * 2 * np.pi / n)
    assert v0 == pytest.approx([-0.38888, -0.43924, -0.82643], rel=1e-4)


def test_fehse_in_plane_targeting_reaches_the_end_point():
    # Fehse, Automated Rendezvous and Docking of Spacecraft, section 3.3.2, eqs 3.41 to 3.43:
    # from 3000 m to 300 m behind the target on the V-bar in 0.37 orbits at Fehse's
    # n = 1.131367e-3 rad/s. Fehse's along-track axis is the Hill y axis.
    n = 1.131367e-3
    tf = 0.37 * 2 * np.pi / n
    r0, rf = np.array([0.0, -3000.0, 0.0]), np.array([0.0, -300.0, 0.0])
    v0, vf = cw_lambert(n, r0, rf, tf)
    end = cw_discrete(n, tf)[0] @ np.r_[r0, v0]
    # The rounding floor of the solve and the transition product, 1e-9 of the 3000 m transfer.
    assert np.max(np.abs(end[:3] - rf)) < 3e-6
    np.testing.assert_allclose(vf, end[3:], rtol=1e-12, atol=1e-15)
    dv1, dv2 = two_impulse(n, np.r_[r0, np.zeros(3)], rf, tf)
    np.testing.assert_allclose(dv1, v0, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(dv2, -vf, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    ("measure", "bracket", "root"),
    [
        (0, (1.9, 2.1), 2.0),
        (0, (2.7, 2.9), 2.81346),
        (0, (4.8, 5.0), 4.89060),
        (1, (0.9, 1.1), 1.0),
    ],
)
def test_conditioning_vanishes_at_the_singular_transfer_times(measure, bracket, root):
    # Alfriend et al., p. 92: in plane at n tf = 2 k pi and at the roots of
    # 8 cos x + 3 x sin x = 8 (2.81346 pi, 4.89060 pi); across the track at n tf = k pi.
    n = 1.1e-3
    found = brentq(lambda x: conditioning(n, x * np.pi / n)[measure], *bracket)
    assert found == pytest.approx(root, abs=1e-5)


def test_conditioning_is_one_for_a_vanishing_transfer_and_positive_before_one_orbit():
    # g = (8 - 8 cos x - 3 x sin x) / x^2 = 1 + x^2 / 6 - x^4 / 72 + ... needs its 1 - cos
    # term free of cancellation: at n tf = 1e-6 the direct form is off by 4e-4, at 1e-4 by 2e-8.
    for tf in (1e-3, 1e-4 / N, 1e-2 / N):
        x = N * tf
        g, h = conditioning(N, tf)
        assert g == pytest.approx(1.0 + x * x / 6.0, rel=1e-9)
        assert h == pytest.approx(1.0 - x * x / 6.0, rel=1e-9)
    for n_tf in np.linspace(0.05, 2 * np.pi - 0.05, 200):
        g, h = conditioning(N, n_tf / N)
        assert g > 0.0 and -1.0 < h <= 1.0


@pytest.mark.parametrize("n_tf", [2 * np.pi, 2.81346 * np.pi, 4.89060 * np.pi, np.pi])
def test_cw_lambert_refuses_the_singular_transfer_times(n_tf):
    n = 1.1e-3
    with pytest.raises(ValueError, match="singular"):
        cw_lambert(n, REST[:3], HOLD, n_tf / n)


def test_cw_lambert_refuses_one_second_off_an_orbit_and_accepts_one_minute():
    n = 1.1e-3
    period = 2 * np.pi / n
    for offset in (-1.0, 1.0):
        with pytest.raises(ValueError, match="singular"):
            cw_lambert(n, REST[:3], HOLD, period + offset)
    for offset in (-60.0, 60.0):
        v0, _ = cw_lambert(n, REST[:3], HOLD, period + offset)
        assert np.all(np.isfinite(v0))


def test_the_singular_tolerance_is_a_keyword():
    # One minute past an orbit the in-plane measure is 0.031 and the cross-track one 0.0104.
    n = 1.1e-3
    tf = 2 * np.pi / n + 60.0
    cw_lambert(n, REST[:3], HOLD, tf, tol=1e-4)
    with pytest.raises(ValueError, match="singular"):
        cw_lambert(n, REST[:3], HOLD, tf, tol=0.05)


@pytest.mark.parametrize("tf", [0.0, -1.0])
def test_a_transfer_needs_a_positive_duration(tf):
    with pytest.raises(ValueError):
        cw_lambert(N, REST[:3], HOLD, tf)
    with pytest.raises(ValueError):
        two_impulse(N, REST, HOLD, tf)


@pytest.mark.parametrize("scale", [1e-2, 1e5])
def test_the_plan_scales_with_the_geometry_from_a_centimetre_to_a_hundred_kilometres(scale):
    state = np.array([0.3, -0.8, 0.2, 1e-4, -2e-4, 5e-5])
    target = np.array([0.1, 0.2, -0.05])
    dv1, dv2 = two_impulse(N, state, target, 1.0 / N)
    scaled1, scaled2 = two_impulse(N, scale * state, scale * target, 1.0 / N)
    # The plan is linear in the state and target; only rounding separates the two paths.
    np.testing.assert_allclose(scaled1, scale * dv1, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(scaled2, scale * dv2, rtol=1e-12, atol=0.0)


def test_a_millisecond_transfer_needs_the_displacement_over_the_time():
    r0 = np.array([0.3, -0.8, 0.2])
    dr = np.array([0.01, 0.01, 0.01])
    tf = 1e-3
    v0, _ = cw_lambert(N, r0, r0 + dr, tf)
    # The speed matches to second order in n tf; the Coriolis coupling turns the in-plane
    # direction at first order, n tf = 1.1e-6, so the components agree only to that.
    assert np.linalg.norm(v0) == pytest.approx(np.linalg.norm(dr) / tf, rel=1e-9)
    np.testing.assert_allclose(v0, dr / tf, rtol=2 * N * tf, atol=0.0)


@pytest.mark.parametrize("n_tf", [0.3, 1.0, 2.5])
def test_the_required_velocity_agrees_with_the_textbook_transition_matrix(n_tf):
    r0, v0 = np.array([30.0, -80.0, 20.0]), np.array([0.01, -0.02, 0.005])
    target = np.array([10.0, 20.0, -5.0])
    tf = n_tf / N
    phi = cw_stm(N, tf)
    expected = np.linalg.solve(phi[:3, 3:], target - phi[:3, :3] @ r0)
    required, vf = cw_lambert(N, r0, target, tf)
    # The same closed form with and without the cancellation-free 1 - cos terms.
    np.testing.assert_allclose(required, expected, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(vf, phi[3:, :3] @ r0 + phi[3:, 3:] @ required, rtol=1e-12, atol=0.0)
    dv1, dv2 = two_impulse(N, np.r_[r0, v0], target, tf)
    np.testing.assert_allclose(dv1, required - v0, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(dv2, -vf, rtol=1e-12, atol=0.0)


def test_the_radial_command_from_rest_changes_sign_at_the_root_of_4_sin_x_equals_3_x():
    # From rest the radial component of the first impulse is n (4 sin nt - 3 nt) dx / D with
    # D = 8 - 8 cos nt - 3 nt sin nt > 0 on (0, 2 pi), so it points at the goal only below the
    # root of 4 sin x = 3 x on (1, 2), 1.27570 (recomputed independently).
    target = REST[:3] + np.array([50.0, 0.0, 0.0])

    def radial(n_tf):
        return two_impulse(N, REST, target, n_tf / N)[0][0]

    assert radial(1.2) > 0.0 and radial(1.35) < 0.0
    assert brentq(radial, 1.0, 2.0) == pytest.approx(1.27570, abs=1e-5)


@pytest.mark.parametrize(("axis", "sign"), [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)])
def test_a_one_radian_transfer_starts_toward_the_goal_on_every_axis(axis, sign):
    target = REST[:3].copy()
    target[axis] += sign * 50.0
    dv1, _ = two_impulse(N, REST, target, 1.0 / N)
    assert sign * dv1[axis] > 0.0


positions = st.lists(st.floats(-2000.0, 2000.0, allow_subnormal=False), min_size=3, max_size=3)
velocities = st.lists(st.floats(-2.0, 2.0, allow_subnormal=False), min_size=3, max_size=3)


@settings(max_examples=60, deadline=None)
@given(positions, velocities, positions, st.floats(0.05, 1.5, allow_subnormal=False))
def test_the_two_impulses_bring_the_chaser_to_rest_at_the_target(r0, v0, target, n_tf):
    state, target = np.array(r0 + v0), np.array(target)
    tf = n_tf / N
    dv1, dv2 = two_impulse(N, state, target, tf)
    arrival = cw_discrete(N, tf)[0] @ (state + np.r_[np.zeros(3), dv1])
    # Rounding floor of the solve and the transition product, relative to the transfer's
    # scale; n tf stays below 1.5, where both conditioning measures are above 0.6.
    scale = max(np.linalg.norm(state[:3]), np.linalg.norm(target), np.linalg.norm(v0) * tf, 1.0)
    assert np.linalg.norm(arrival[:3] - target) <= 1e-9 * scale
    assert np.linalg.norm(arrival[3:] + dv2) <= 1e-9 * (np.linalg.norm(v0) + np.linalg.norm(dv1))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_scalars_and_vectors_are_refused(bad):
    n, tf = 1.1e-3, 900.0
    rest = np.zeros(6)
    with pytest.raises(ValueError, match="n must be"):
        conditioning(bad, tf)
    with pytest.raises(ValueError, match="tf must be"):
        cw_lambert(n, [0.0, -300.0, 0.0], [0.0, -100.0, 0.0], bad)
    with pytest.raises(ValueError, match="r0 must be"):
        cw_lambert(n, [bad, -300.0, 0.0], [0.0, -100.0, 0.0], tf)
    with pytest.raises(ValueError, match="rf must be"):
        cw_lambert(n, [0.0, -300.0, 0.0], [0.0, bad, 0.0], tf)
    with pytest.raises(ValueError, match="state must be"):
        two_impulse(n, np.r_[rest[:5], bad], [0.0, -100.0, 0.0], tf)
    with pytest.raises(ValueError, match="state must be"):
        two_impulse(n, rest[:3], [0.0, -100.0, 0.0], tf)
