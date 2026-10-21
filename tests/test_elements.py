"""Trajectory elements: Fehse's transfers and fly-arounds, forced motion and the twice-rate circle.

Known answers from Fehse, Automated Rendezvous and Docking of Spacecraft, sections 3.3 and 4.4,
at n = 1.131367e-3 rad/s (a 400 km circular orbit; the book used r = 6766 km), and from Goodman,
History of Space Shuttle Rendezvous (JSC-63400), p. 137, for the twice-orbital-rate circle.

Fehse's local orbital frame has x along the V-bar, y = -h and z toward Earth. `iss_to_hill` maps
it to the Hill frame used here (x radial out, y along-track, z = +h), so a point dz below the
target in Fehse's frame is Hill (-dz, 0, 0).
"""

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize_scalar

from vbar.agents.elements import (
    Element,
    cw_hohmann,
    finite_burn_duration,
    forced_motion_accel,
    radial_flyaround,
    radial_transfer,
    straight_line,
    straight_line_cost,
    tangential_flyaround,
    tangential_transfer,
    twice_rate_circle,
    twice_rate_cost,
)
from vbar.agents.planning import Impulse
from vbar.conventions import iss_to_hill
from vbar.dynamics import MU_EARTH, R_EARTH, cw_discrete, cw_rhs, mean_motion

N = 1.131367e-3
T = 2.0 * math.pi / N
DY = 2700.0  # Fehse's V-bar transfer from 3000 m to 300 m behind the target
DZ = 500.0
BELOW = iss_to_hill([0.0, 0.0, DZ])

# Goodman's twice-rate examples: 185 n.mi. circular, R = 600 ft.
N_TORVA = mean_motion(R_EARTH + 185.0 * 1852.0, MU_EARTH)
T_TORVA = 2.0 * math.pi / N_TORVA
R_TORVA = 600.0 * 0.3048


def coast(n, element, state, t0, t1):
    """State after flying from t0 to t1 under the element's acceleration, or coasting."""
    if t1 <= t0:
        return state
    if element.accel is None:
        return cw_discrete(n, t1 - t0)[0] @ state
    scale = np.linalg.norm(state[:3]) + 1.0
    sol = solve_ivp(
        lambda t, s: cw_rhs(t, s, n, element.accel(t, s)),
        (t0, t1),
        state,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13 * scale,
    )
    return sol.y[:, -1]


def fly(n, element, start, until=None):
    """State after flying `element` from `start`.

    With `until`, the state at that time before any later impulse planned there (the departure
    impulse at t = 0 is always applied); without, the state at the element's end with its last
    impulse applied.
    """
    end = element.duration if until is None else until
    state = np.array(start, dtype=float)
    t = 0.0
    for imp in sorted(element.impulses, key=lambda i: i.t):
        if imp.t > end or (until is not None and 0.0 < imp.t == end):
            break
        state = coast(n, element, state, t, imp.t)
        state[3:] += imp.dv
        t = imp.t
    return coast(n, element, state, t, end)


def total_impulse(element):
    return sum(float(np.linalg.norm(imp.dv)) for imp in element.impulses)


def rel(a, b):
    """Distance between two vectors relative to the second's norm."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b))


def test_tangential_transfer_burns_n_dy_over_6_pi_and_costs_n_dy_over_3_pi():
    # Fehse eqs 3.29 to 3.32: a +y impulse dvy moves the chaser by -6 pi dvy / n in one orbit.
    element = tangential_transfer(N, DY)
    first, second = sorted(element.impulses, key=lambda i: i.t)
    burn = N * DY / (6.0 * math.pi)
    assert first.t == 0.0 and second.t == pytest.approx(T, rel=1e-12)
    assert first.dv == pytest.approx([0.0, -burn, 0.0], rel=1e-12, abs=1e-15)
    assert second.dv == pytest.approx([0.0, burn, 0.0], rel=1e-12, abs=1e-15)
    # The book prints 0.33 m/s; 0.3241 is n dy / (3 pi) at the stated n.
    assert total_impulse(element) == pytest.approx(0.3241, abs=1e-4)


def test_radial_transfer_costs_3_pi_over_2_times_the_tangential_one():
    # Fehse eq 3.37: two radial burns of n dy / 4, half an orbit apart.
    element = radial_transfer(N, DY)
    first, second = sorted(element.impulses, key=lambda i: i.t)
    assert first.t == 0.0 and second.t == pytest.approx(T / 2.0, rel=1e-12)
    assert first.dv == pytest.approx([-N * DY / 4.0, 0.0, 0.0], rel=1e-12, abs=1e-15)
    assert total_impulse(element) == pytest.approx(1.5273, abs=1e-4)
    ratio = total_impulse(element) / total_impulse(tangential_transfer(N, DY))
    assert ratio == pytest.approx(1.5 * math.pi, rel=1e-9)


def test_cw_hohmann_burns_n_dx_over_4_twice_and_advances_3_pi_dx_over_4():
    # Raising by dx from a coelliptic orbit dx below (along-track speed 1.5 n dx); the lower
    # half-orbit carries the chaser ahead.
    dx = 500.0
    element = cw_hohmann(N, dx)
    for imp in element.impulses:
        assert imp.dv == pytest.approx([0.0, N * dx / 4.0, 0.0], rel=1e-12, abs=1e-15)
    start = np.array([-dx, -3000.0, 0.0, 0.0, 1.5 * N * dx, 0.0])
    end = fly(N, element, start)
    assert end[1] - start[1] == pytest.approx(0.75 * math.pi * dx, rel=1e-9)
    assert abs(end[0]) <= 1e-9 * dx


# Start states (Hill, m and m/s) and end points of each element. A fly-around to a point dz
# below starts behind the target: from rest at -3 pi dz / 4 with a retrograde burn (Fehse eq
# 3.33), or at -2 dz with a radial burn toward Earth (eqs 3.38, 3.39).
IMPULSIVE = {
    "tangential transfer": (
        lambda: tangential_transfer(N, DY),
        (0.0, -3000.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, -300.0, 0.0),
        DY,
    ),
    "radial transfer": (
        lambda: radial_transfer(N, DY),
        (0.0, -3000.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, -300.0, 0.0),
        DY,
    ),
    "cw hohmann": (
        lambda: cw_hohmann(N, DZ),
        (-DZ, -3000.0, 0.0, 0.0, 1.5 * N * DZ, 0.0),
        (0.0, -3000.0 + 0.75 * math.pi * DZ, 0.0),
        DZ,
    ),
    "tangential fly-around": (
        lambda: tangential_flyaround(N, DZ),
        (0.0, -0.75 * math.pi * DZ, 0.0, 0.0, 0.0, 0.0),
        tuple(BELOW),
        DZ,
    ),
    "radial fly-around": (
        lambda: radial_flyaround(N, DZ),
        (0.0, -2.0 * DZ, 0.0, 0.0, 0.0, 0.0),
        tuple(BELOW),
        DZ,
    ),
}


@pytest.mark.parametrize("name", IMPULSIVE)
def test_impulsive_element_ends_at_rest_at_its_end_point(name):
    factory, start, end_point, length = IMPULSIVE[name]
    element = factory()
    end = fly(N, element, start)
    # The rounding floor of a few transition products, 1e-9 of the transfer length; n times
    # that length is the element's speed scale.
    assert np.linalg.norm(end[:3] - end_point) <= 1e-9 * length
    assert np.linalg.norm(end[3:]) <= 1e-9 * N * length
    assert max(imp.t for imp in element.impulses) == pytest.approx(element.duration, rel=1e-12)


def test_elements_are_frozen_records_of_impulses():
    element = tangential_transfer(N, DY)
    assert isinstance(element, Element)
    assert all(isinstance(imp, Impulse) for imp in element.impulses)
    assert element.accel is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        element.duration = 0.0


def test_fly_arounds_cost_2_n_dz_and_3_n_dz():
    # Fehse eq 3.33 and eqs 3.38, 3.39 for a point 500 m below the target.
    tangential = sorted(tangential_flyaround(N, DZ).impulses, key=lambda i: i.t)
    assert tangential[0].dv == pytest.approx([0.0, -N * DZ / 4.0, 0.0], rel=1e-12, abs=1e-15)
    assert tangential[1].t == pytest.approx(T / 2.0, rel=1e-12)
    assert np.linalg.norm(tangential[1].dv) == pytest.approx(0.98995, abs=1e-5)  # 7 n dz / 4
    assert total_impulse(tangential_flyaround(N, DZ)) == pytest.approx(1.1314, abs=1e-4)
    radial = sorted(radial_flyaround(N, DZ).impulses, key=lambda i: i.t)
    assert radial[0].dv == pytest.approx([-N * DZ, 0.0, 0.0], rel=1e-12, abs=1e-15)
    assert radial[1].t == pytest.approx(T / 4.0, rel=1e-12)
    assert np.linalg.norm(radial[1].dv) == pytest.approx(2.0 * N * DZ, abs=1e-5)
    assert total_impulse(radial_flyaround(N, DZ)) == pytest.approx(1.697, abs=1e-4)


@pytest.mark.parametrize(
    ("duration", "cost", "ratio"), [(T, 7.082, 21.85), (T / 2.0, 8.054, 24.85)]
)
def test_straight_vbar_approach_costs_2_v_plus_2_n_dy(duration, cost, ratio):
    # Fehse eqs 3.44 to 3.48 over 3000 m to 300 m. The printed half-orbit ratio (44x) is a
    # misprint; 21.85 and 24.85 are recomputed at the stated constants.
    found = straight_line_cost(N, DY, duration)
    assert found == pytest.approx(cost, abs=1e-3)
    assert found / (N * DY / (3.0 * math.pi)) == pytest.approx(ratio, abs=1e-2)


def test_straight_line_element_holds_the_line_and_costs_what_the_formula_says():
    r0, r1 = np.array([0.0, -3000.0, 0.0]), np.array([0.0, -300.0, 0.0])
    speed = DY / T
    element = straight_line(N, r0, r1, speed)
    assert element.duration == pytest.approx(T, rel=1e-12)
    first, last = sorted(element.impulses, key=lambda i: i.t)
    assert first.dv == pytest.approx([0.0, speed, 0.0], rel=1e-12, abs=1e-15)
    assert last.dv == pytest.approx([0.0, -speed, 0.0], rel=1e-12, abs=1e-15)
    end = fly(N, element, np.r_[r0, 0.0, 0.0, 0.0])
    # DOP853 at rtol 1e-12 over one orbit; 1e-8 of the 2700 m path leaves it room.
    assert np.linalg.norm(end[:3] - r1) <= 1e-8 * DY
    assert np.linalg.norm(end[3:]) <= 1e-8 * speed
    # On the V-bar the forced acceleration is -2 n vy along x, so its integral is 2 n dy.
    mid = np.r_[0.0, -1500.0, 0.0, 0.0, speed, 0.0]
    assert element.accel(T / 2.0, mid) == pytest.approx([-2.0 * N * speed, 0.0, 0.0], rel=1e-12)

    def thrust(t):
        nominal = np.r_[r0 + (r1 - r0) * t / T, (r1 - r0) / T]
        return float(np.linalg.norm(element.accel(t, nominal)))

    flown = total_impulse(element) + quad(thrust, 0.0, T)[0]
    assert flown == pytest.approx(straight_line_cost(N, DY, T), rel=1e-9)


def test_forced_motion_reduces_to_fehse_on_the_vbar_and_rbar_and_holds_out_of_plane():
    # Fehse eqs 3.44 (V-bar), 3.49 to 3.53 (R-bar) and 3.58 (station keeping out of plane).
    vy, vx, x, z = 0.3, 0.4, -200.0, 30.0
    on_vbar = forced_motion_accel(
        N, np.array([0.0, -250.0, 0.0]), np.array([0.0, vy, 0.0]), np.zeros(3)
    )
    assert rel(on_vbar, [-2.0 * N * vy, 0.0, 0.0]) <= 1e-12
    on_rbar = forced_motion_accel(N, np.array([x, 0.0, 0.0]), np.array([vx, 0.0, 0.0]), np.zeros(3))
    assert rel(on_rbar, [-3.0 * N**2 * x, 2.0 * N * vx, 0.0]) <= 1e-12
    held = forced_motion_accel(N, np.array([0.0, -100.0, z]), np.zeros(3), np.zeros(3))
    assert rel(held, [0.0, 0.0, N**2 * z]) <= 1e-12


def second_derivative(path, t, h):
    """Sixth-order central difference: truncation (omega h)^6 / 560, about 1e-11 at h = 20 s."""
    weights = (1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90)
    return sum(w * path(t + k * h) for w, k in zip(weights, range(-3, 4), strict=True)) / h**2


@pytest.mark.parametrize("rate", [2.0, -2.0, 1.0, 0.5])
@pytest.mark.parametrize("tilt", [0.0, 0.7])
def test_forced_circle_acceleration_equals_the_differentiated_path_minus_cw_terms(rate, tilt):
    # Fehse eqs 3.77 to 3.80 for a 30 m circle; the tilt moves it out of the orbit plane.
    radius, omega = 30.0, rate * N
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, math.cos(tilt), math.sin(tilt)])

    def path(t):
        return radius * (math.cos(omega * t) * e1 + math.sin(omega * t) * e2)

    for t in (0.0, 400.0, 1300.0, 2900.0):
        r = path(t)
        v = radius * omega * (-math.sin(omega * t) * e1 + math.cos(omega * t) * e2)
        expected = second_derivative(path, t, 20.0) - cw_rhs(t, np.r_[r, v], N, np.zeros(3))[3:]
        found = forced_motion_accel(N, r, v, -(omega**2) * r)
        assert rel(found, expected) <= 1e-9


def test_fehse_twice_rate_circle_needs_only_the_rbar_hover_force():
    # Fehse's circle (-R cos 2nt, 0, R sin 2nt) in his frame is the clockwise twice-rate circle
    # in Hill (ahead, above, behind, below); eq 3.79 then has no along-track term (Goodman p. 137).
    radius, omega = 30.0, 2.0 * N
    for t in np.linspace(0.0, T / 2.0, 7):
        c, s = math.cos(omega * t), math.sin(omega * t)
        r = iss_to_hill([-radius * c, 0.0, radius * s])
        v = iss_to_hill([radius * omega * s, 0.0, radius * omega * c])
        found = forced_motion_accel(N, r, v, -(omega**2) * r)
        # Relative to the largest hover force, 3 n^2 R, since the force vanishes on the V-bar.
        hover = np.array([-3.0 * N**2 * r[0], 0.0, 0.0])
        assert np.linalg.norm(found - hover) <= 1e-12 * 3.0 * N**2 * radius
        assert v[1] + 2.0 * N * r[0] == pytest.approx(0.0, abs=1e-12 * N * radius)


def circle_state(n, radius, degrees):
    """State on the clockwise twice-rate circle, the angle from +x (above) toward +y (ahead)."""
    th = math.radians(degrees)
    x, y = radius * math.cos(th), radius * math.sin(th)
    return np.array([x, y, 0.0, 2.0 * n * y, -2.0 * n * x, 0.0])


def test_twice_rate_circle_starts_on_the_rbar_and_is_held_by_the_hover_force():
    element = twice_rate_circle(N_TORVA, R_TORVA)
    first = min(element.impulses, key=lambda i: i.t)
    assert first.t == 0.0
    assert first.dv == pytest.approx([0.0, 2.0 * N_TORVA * R_TORVA, 0.0], rel=1e-12, abs=1e-15)
    for degrees in (180.0, 150.0, 120.0, 90.0):
        t = math.radians(180.0 - degrees) / (2.0 * N_TORVA)
        state = circle_state(N_TORVA, R_TORVA, degrees)
        hover = np.array([-3.0 * N_TORVA**2 * state[0], 0.0, 0.0])
        gap = np.linalg.norm(element.accel(t, state) - hover)
        assert gap <= 1e-12 * 3.0 * N_TORVA**2 * R_TORVA


def range_extremes(n, state, period):
    """Least and greatest range over one period of free CW motion, refined by bounded Brent."""

    def rng(t):
        return float(np.linalg.norm((cw_discrete(n, t)[0] @ state)[:3]))

    ts = np.linspace(0.0, period, 2001)
    rs = np.array([rng(t) for t in ts])
    low, high = min(rs[0], rs[-1]), max(rs[0], rs[-1])
    for i in range(1, len(ts) - 1):
        bracket = (ts[i - 1], ts[i + 1])
        if rs[i] <= rs[i - 1] and rs[i] <= rs[i + 1]:
            low = min(low, minimize_scalar(rng, bounds=bracket, method="bounded").fun)
        if rs[i] >= rs[i - 1] and rs[i] >= rs[i + 1]:
            high = max(
                high, -minimize_scalar(lambda t: -rng(t), bounds=bracket, method="bounded").fun
            )
    return low, high


# Greatest range after thrust loss, in R. The two-decimal values 4.15 and 6.20 are rounded;
# these are from an independent computation at Goodman's constants (185 n.mi., 600 ft).
THRUST_LOSS = [(180.0, 2.0), (150.0, 4.145751), (120.0, 6.203627), (90.0, 7.0)]


@pytest.mark.parametrize(("degrees", "greatest"), THRUST_LOSS)
def test_thrust_loss_on_the_twice_rate_circle_never_comes_closer_than_its_radius(degrees, greatest):
    # Goodman p. 137: flown from +R-bar, the arc has vy + 2 n x = 0 everywhere, so free motion
    # after a thrust loss is a closed ellipse with no drift.
    element = twice_rate_circle(N_TORVA, R_TORVA)
    t = math.radians(180.0 - degrees) / (2.0 * N_TORVA)
    lost = fly(N_TORVA, element, [-R_TORVA, 0.0, 0.0, 0.0, 0.0, 0.0], until=t)
    assert lost[4] + 2.0 * N_TORVA * lost[0] == pytest.approx(0.0, abs=1e-9 * N_TORVA * R_TORVA)
    low, high = range_extremes(N_TORVA, lost, T_TORVA)
    assert low == pytest.approx(R_TORVA, abs=1e-3 * R_TORVA)
    assert high == pytest.approx(greatest * R_TORVA, abs=1e-3 * R_TORVA)


@pytest.mark.parametrize(("degrees", "cost"), [(90, 5.5), (180, 7.0), (270, 8.5), (360, 10.0)])
def test_twice_rate_arc_from_rest_to_rest_costs_4_n_r_plus_the_hover_integral(degrees, cost):
    # Start and stop impulses 2 n R each, plus (3 n R / 2) times the integral of |cos| over
    # the arc from +R-bar (Goodman p. 137, derived; the arc is cheaper than an inertial one only
    # beyond about 200 degrees).
    n, radius = N_TORVA, R_TORVA
    found = twice_rate_cost(n, radius, math.radians(degrees))
    assert found == pytest.approx(cost * n * radius, abs=1e-3 * n * radius)


def test_finite_burn_lasts_dv_over_the_acceleration():
    # Fehse eqs 3.61, 3.68: a CW Hohmann burn of n dz / 4 at 0.02 m/s^2 lasts n dz / (4 gamma).
    burn = cw_hohmann(N, DZ).impulses[0].dv
    assert finite_burn_duration(burn, 0.02) == pytest.approx(N * DZ / (4.0 * 0.02), rel=1e-12)
    assert finite_burn_duration(np.array([0.3, -0.4, 0.0]), 0.02) == pytest.approx(25.0, rel=1e-12)


@pytest.mark.parametrize("gamma", [0.0, -0.02, math.nan, math.inf])
def test_finite_burn_refuses_an_acceleration_that_is_not_positive_and_finite(gamma):
    with pytest.raises(ValueError):
        finite_burn_duration(np.array([0.1, 0.0, 0.0]), gamma)


@pytest.mark.parametrize("speed", [0.0, -0.1, math.nan])
def test_straight_line_refuses_a_speed_that_is_not_positive(speed):
    with pytest.raises(ValueError):
        straight_line(N, [0.0, -3000.0, 0.0], [0.0, -300.0, 0.0], speed)


LINE = np.array([0.3, -1.0, 0.2])
SCALED = {
    "tangential transfer": lambda s: tangential_transfer(N, s),
    "radial transfer": lambda s: radial_transfer(N, s),
    "cw hohmann": lambda s: cw_hohmann(N, s),
    "tangential fly-around": lambda s: tangential_flyaround(N, s),
    "radial fly-around": lambda s: radial_flyaround(N, s),
    "twice-rate circle": lambda s: twice_rate_circle(N, s),
    "straight line": lambda s: straight_line(N, s * LINE, -s * LINE, s * 0.01),
}
PROBE = np.array([0.7, -0.4, 0.2, 1e-3, -2e-3, 5e-4])


def assert_scaled(name, scale):
    one, big = SCALED[name](1.0), SCALED[name](scale)
    assert big.duration == pytest.approx(one.duration, rel=1e-12)
    assert len(big.impulses) == len(one.impulses)
    for a, b in zip(one.impulses, big.impulses, strict=True):
        assert b.t == pytest.approx(a.t, rel=1e-12, abs=1e-12 * one.duration)
        assert np.linalg.norm(b.dv - scale * a.dv) <= 1e-12 * scale * np.linalg.norm(a.dv)
    if one.accel is not None:
        for t in (0.25 * one.duration, 0.5 * one.duration):
            expected = scale * np.asarray(one.accel(t, PROBE))
            assert rel(big.accel(t, scale * PROBE), expected) <= 1e-12


@pytest.mark.parametrize("scale", [1e-2, 1e5])
@pytest.mark.parametrize("name", SCALED)
def test_elements_scale_linearly_from_a_centimetre_to_100_km(name, scale):
    assert_scaled(name, scale)


@pytest.mark.parametrize("scale", [1e-2, 1e5])
def test_element_costs_scale_linearly_with_length(scale):
    assert straight_line_cost(N, scale * DY, T) == pytest.approx(
        scale * straight_line_cost(N, DY, T), rel=1e-12
    )
    assert twice_rate_cost(N, scale, math.pi) == pytest.approx(
        scale * twice_rate_cost(N, 1.0, math.pi), rel=1e-12
    )


@settings(max_examples=40, deadline=None)
@given(
    st.sampled_from(sorted(SCALED)),
    st.floats(1e-2, 1e5, allow_nan=False, allow_infinity=False, allow_subnormal=False),
)
def test_elements_scale_linearly_at_any_length(name, scale):
    assert_scaled(name, scale)
