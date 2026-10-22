import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from vbar.agents.cw_targeting import two_impulse
from vbar.conventions import hill_frame, hill_to_inertial
from vbar.dynamics.bodies import EARTH
from vbar.dynamics.cw import mean_motion
from vbar.dynamics.kepler import elements_from_state, propagate_kepler
from vbar.dynamics.lambert import lambert
from vbar.dynamics.models import circular_orbit

MU = EARTH.mu
EPS = np.finfo(float).eps

# Montenbruck and Gill, Satellite Orbits, exercise 2.6, with the book's GM.
MU_BOOK = 398600.4415e9
R1_BOOK = np.array([11959.978e3, -16289.478e3, -5963.827e3])
R2_BOOK = np.array([39863.390e3, -13730.547e3, -4862.350e3])
TOF_BOOK = 9000.0

ORIENTATION = Rotation.from_euler("zxz", [0.7, 0.4, 1.9]).as_matrix()


def conic_state(p, e, f):
    """Inertial state at true anomaly f on the conic (p, e) rotated by ORIENTATION."""
    r = p / (1.0 + e * math.cos(f)) * np.array([math.cos(f), math.sin(f), 0.0])
    v = math.sqrt(MU / p) * np.array([-math.sin(f), e + math.cos(f), 0.0])
    return ORIENTATION @ r, ORIENTATION @ v


def time_since_perigee(p, e, f):
    if e < 1.0:
        a = p / (1.0 - e * e)
        E = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * math.tan(0.5 * f))
        return math.sqrt(a**3 / MU) * (E - e * math.sin(E))
    a = p / (e * e - 1.0)
    H = 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(0.5 * f))
    return math.sqrt(a**3 / MU) * (e * math.sinh(H) - H)


def arc(p, e, f1, df):
    """(r1, v1, r2, v2, tof) for the arc from f1 to f1 + df on the conic (p, e)."""
    r1, v1 = conic_state(p, e, f1)
    r2, v2 = conic_state(p, e, f1 + df)
    tof = time_since_perigee(p, e, f1 + df) - time_since_perigee(p, e, f1)
    if e < 1.0:
        tof %= 2.0 * math.pi * math.sqrt((p / (1.0 - e * e)) ** 3 / MU)
    return r1, v1, r2, v2, tof


def random_problem(rng):
    """Two positions between 6800 and 42000 km, not within 1e-3 rad of collinear."""
    while True:
        r1 = rng.uniform(6.8e6, 4.2e7) * unit(rng.normal(size=3))
        r2 = rng.uniform(6.8e6, 4.2e7) * unit(rng.normal(size=3))
        if np.linalg.norm(np.cross(unit(r1), unit(r2))) > 1e-3:
            break
    mean_radius = 0.5 * (np.linalg.norm(r1) + np.linalg.norm(r2))
    tof = rng.uniform(0.05, 0.9) * 2.0 * math.pi * math.sqrt(mean_radius**3 / MU)
    return r1, r2, tof


def unit(x):
    return x / np.linalg.norm(x)


def assert_close(actual, expected, rel):
    """Vector agreement relative to the expected vector's norm, not per component."""
    assert np.linalg.norm(np.asarray(actual) - expected) <= rel * np.linalg.norm(expected)


def transfer_angle(r1, r2, prograde):
    angle = math.atan2(np.linalg.norm(np.cross(r1, r2)), r1 @ r2)
    if (np.cross(r1, r2)[2] < 0.0) == prograde:
        angle = 2.0 * math.pi - angle
    return angle


N_500 = mean_motion(EARTH.radius + 500e3, MU)
P_500 = EARTH.radius + 500e3
P_MOLNIYA = 26600e3 * (1.0 - 0.74**2)
P_E03 = 12000e3 * (1.0 - 0.3**2)
P_HYP15 = 10000e3 * (1.5**2 - 1.0)
P_HYP3 = 8000e3 * (3.0**2 - 1.0)

# (p, e, f1, df, relative tolerance on the velocities). Within 0.01 rad of a full revolution
# y = r1 + r2 + A (z S - 1) / sqrt(C) cancels to 2.5e-5 of r1 + r2, the rounding of the residual
# moves the root by 3e-10 in z and v1 by 1e-9 relative (a reference solver reached 7e-9; the floor
# grows as the fourth power of the inverse gap), so that case gets 1e-7.
ARCS = {
    "circle 0.01 rad": (P_500, 0.0, 0.3, 0.01, 1e-9),
    "circle 1 s": (P_500, 0.0, 0.3, N_500 * 1.0, 1e-9),
    "circle 0.95 period": (P_500, 0.0, 0.3, 0.95 * 2.0 * math.pi, 1e-9),
    "molniya short arc": (P_MOLNIYA, 0.74, 0.5, 2.0, 1e-9),
    "molniya long arc": (P_MOLNIYA, 0.74, 2.5, 4.5, 1e-9),
    "ellipse pi - 1e-3": (P_E03, 0.3, 1.0, math.pi - 1e-3, 1e-9),
    "ellipse pi + 1e-3": (P_E03, 0.3, 1.0, math.pi + 1e-3, 1e-9),
    "ellipse 2 pi - 0.01": (P_E03, 0.3, 2.0, 2.0 * math.pi - 0.01, 1e-7),
    "hyperbola short arc": (P_HYP15, 1.5, -1.0, 1.5, 1e-9),
    "hyperbola long arc": (P_HYP15, 1.5, -1.8, 3.5, 1e-9),
    "hyperbola e 3": (P_HYP3, 3.0, -0.5, 0.8, 1e-9),
}


def test_montenbruck_exercise_2_6_orbit_from_two_positions():
    v1, _ = lambert(R1_BOOK, R2_BOOK, TOF_BOOK, MU_BOOK)
    a, e, i, raan, argp, M = elements_from_state(R1_BOOK, v1, MU_BOOK)
    assert a == pytest.approx(28196.776e3, abs=0.5)
    assert e == pytest.approx(0.7679436, abs=5e-8)
    expected = [20.315, 359.145, 179.425, 29.236]
    assert np.degrees([i, raan, argp, M]) == pytest.approx(expected, abs=5e-4)


def test_montenbruck_exercise_2_6_velocities_close_the_arc():
    v1, v2 = lambert(R1_BOOK, R2_BOOK, TOF_BOOK, MU_BOOK)
    r2, v2_back = propagate_kepler(R1_BOOK, v1, TOF_BOOK, MU_BOOK)
    assert_close(r2, R2_BOOK, 1e-9)
    assert_close(v2_back, v2, 1e-9)


@pytest.mark.parametrize("prograde", [True, False])
def test_random_solutions_lie_on_a_conic_through_both_positions(prograde):
    # Schaub and Junkins, Analytical Mechanics of Space Systems, Lambert section: with p from
    # the returned velocity, F = 1 - r2 (1 - cos df) / p and G = r1 r2 sin df / sqrt(mu p)
    # must carry r1 to r2. A different formula path from the universal variables, so 1e-10.
    rng = np.random.default_rng(1)
    for _ in range(200):
        r1, r2, tof = random_problem(rng)
        v1, _ = lambert(r1, r2, tof, MU, prograde=prograde)
        n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
        df = transfer_angle(r1, r2, prograde)
        p = np.linalg.norm(np.cross(r1, v1)) ** 2 / MU
        F = 1.0 - n2 * (1.0 - math.cos(df)) / p
        G = n1 * n2 * math.sin(df) / math.sqrt(MU * p)
        assert np.linalg.norm(F * r1 + G * v1 - r2) <= 1e-10 * n2


@pytest.mark.parametrize("prograde", [True, False])
def test_random_solutions_satisfy_the_lambert_velocity_locus(prograde):
    # Schaub and Junkins, Lambert section: the chord and radial components of v1 obey
    # v_c v_rho = mu c sec^2(df / 2) / (2 r1 r2) on every conic through r1 and r2.
    rng = np.random.default_rng(2)
    for _ in range(200):
        r1, r2, tof = random_problem(rng)
        v1, _ = lambert(r1, r2, tof, MU, prograde=prograde)
        chord = r2 - r1
        c = np.linalg.norm(chord)
        v_c, v_rho = np.linalg.lstsq(np.column_stack([chord / c, unit(r1)]), v1, rcond=None)[0]
        df = transfer_angle(r1, r2, prograde)
        locus = MU * c / (2.0 * np.linalg.norm(r1) * np.linalg.norm(r2) * math.cos(0.5 * df) ** 2)
        assert v_c * v_rho == pytest.approx(locus, rel=1e-10)


@pytest.mark.parametrize("prograde", [True, False])
def test_lagrange_coefficients_of_the_solution_have_unit_determinant(prograde):
    # f gdot - g fdot = 1 (Schaub and Junkins; the printed 0 is a misprint) says the returned
    # v2 conserves the angular momentum of the returned v1.
    rng = np.random.default_rng(3)
    for _ in range(200):
        r1, r2, tof = random_problem(rng)
        v1, v2 = lambert(r1, r2, tof, MU, prograde=prograde)
        basis = np.column_stack([r1, v1])
        f, g = np.linalg.lstsq(basis, r2, rcond=None)[0]
        f_dot, g_dot = np.linalg.lstsq(basis, v2, rcond=None)[0]
        assert f * g_dot - g * f_dot == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("prograde", [True, False])
def test_random_solutions_reach_r2_at_the_time_of_flight(prograde):
    rng = np.random.default_rng(4)
    for _ in range(200):
        r1, r2, tof = random_problem(rng)
        v1, v2 = lambert(r1, r2, tof, MU, prograde=prograde)
        r2_back, v2_back = propagate_kepler(r1, v1, tof, MU)
        assert_close(r2_back, r2, 1e-9)
        assert_close(v2_back, v2, 1e-9)


@pytest.mark.parametrize("name", list(ARCS))
def test_solution_reproduces_the_conic_velocities(name):
    p, e, f1, df, rel = ARCS[name]
    r1, v1_true, r2, v2_true, tof = arc(p, e, f1, df)
    v1, v2 = lambert(r1, r2, tof, MU)
    assert_close(v1, v1_true, rel)
    assert_close(v2, v2_true, rel)
    r2_back, _ = propagate_kepler(r1, v1, tof, MU)
    assert_close(r2_back, r2, rel)


def test_one_second_arc_on_a_circle_takes_one_second():
    r1, v1_true, r2, _, tof = arc(*ARCS["circle 1 s"][:4])
    assert tof == pytest.approx(1.0, rel=1e-12)
    v1, _ = lambert(r1, r2, tof, MU)
    assert_close(v1, v1_true, 1e-9)


# Arcs stop 0.1 rad short of a full revolution, where the y(z) cancellation floor is near 1e-12;
# the last 0.1 rad is covered by the explicit case above at its own tolerance.
@settings(max_examples=40, deadline=None)
@given(
    st.floats(0.0, 0.9, allow_subnormal=False),
    st.floats(0.0, 2.0 * math.pi, allow_subnormal=False),
    st.floats(0.01, math.pi - 1e-3, allow_subnormal=False)
    | st.floats(math.pi + 1e-3, 2.0 * math.pi - 0.1, allow_subnormal=False),
)
def test_any_elliptic_arc_is_recovered_from_its_end_points(e, f1, df):
    p = 70000e3 * (1.0 - e * e)
    r1, v1_true, r2, v2_true, tof = arc(p, e, f1, df)
    v1, v2 = lambert(r1, r2, tof, MU)
    assert_close(v1, v1_true, 1e-9)
    assert_close(v2, v2_true, 1e-9)


def test_prograde_and_retrograde_arcs_both_reach_r2_with_opposite_angular_momentum():
    a = EARTH.radius + 800e3
    r1 = np.array([a, 0.0, 0.0])
    r2 = a * np.array([math.cos(2.0), math.sin(2.0), 0.0])
    tof = 2.0 / mean_motion(a, MU)
    v1_pro, _ = lambert(r1, r2, tof, MU, prograde=True)
    v1_retro, _ = lambert(r1, r2, tof, MU, prograde=False)
    assert np.cross(r1, v1_pro)[2] > 0.0
    assert np.cross(r1, v1_retro)[2] < 0.0
    assert abs(v1_pro[2]) <= 1e-12 * np.linalg.norm(v1_pro)
    assert abs(v1_retro[2]) <= 1e-12 * np.linalg.norm(v1_retro)
    assert v1_pro[1] > 0.0 > v1_retro[1]
    r2_pro, _ = propagate_kepler(r1, v1_pro, tof, MU)
    r2_retro, _ = propagate_kepler(r1, v1_retro, tof, MU)
    assert_close(r2_pro, r2, 1e-9)
    assert_close(r2_retro, r2, 1e-9)


def test_retrograde_is_prograde_about_the_opposite_normal():
    v_retro = lambert(R1_BOOK, R2_BOOK, TOF_BOOK, MU_BOOK, prograde=False)
    v_flipped = lambert(R1_BOOK, R2_BOOK, TOF_BOOK, MU_BOOK, normal=(0.0, 0.0, -1.0))
    # The same conic reached through a different branch of the transfer angle.
    assert_close(v_flipped[0], v_retro[0], 1e-12)
    assert_close(v_flipped[1], v_retro[1], 1e-12)
    v_default = lambert(R1_BOOK, R2_BOOK, TOF_BOOK, MU_BOOK)
    v_pro = lambert(R1_BOOK, R2_BOOK, TOF_BOOK, MU_BOOK, prograde=True, normal=(0.0, 0.0, 1.0))
    assert np.array_equal(v_default[0], v_pro[0])
    assert np.array_equal(v_default[1], v_pro[1])
    assert np.cross(R1_BOOK, v_retro[0])[2] < 0.0 < np.cross(R1_BOOK, v_default[0])[2]


@pytest.mark.parametrize(
    "r2",
    [
        2.0 * R1_BOOK,
        -R1_BOOK,
        Rotation.from_rotvec([0.0, 0.0, 1e-14]).apply(R1_BOOK),
    ],
    ids=["same direction", "opposite direction", "1e-14 rad apart"],
)
def test_collinear_positions_are_refused(r2):
    with pytest.raises(ValueError):
        lambert(R1_BOOK, r2, TOF_BOOK, MU_BOOK)


@pytest.mark.parametrize("tof", [0.0, -1.0, math.inf, math.nan])
def test_non_positive_or_non_finite_time_of_flight_is_refused(tof):
    with pytest.raises(ValueError):
        lambert(R1_BOOK, R2_BOOK, tof, MU_BOOK)


@pytest.mark.parametrize(
    "r1, r2, mu",
    [
        (np.zeros(3), R2_BOOK, MU_BOOK),
        (R1_BOOK, np.zeros(3), MU_BOOK),
        (np.array([math.nan, 0.0, 0.0]), R2_BOOK, MU_BOOK),
        (R1_BOOK, np.array([math.inf, 0.0, 0.0]), MU_BOOK),
        (R1_BOOK, R2_BOOK, 0.0),
    ],
    ids=["zero r1", "zero r2", "nan r1", "inf r2", "zero mu"],
)
def test_zero_or_non_finite_inputs_are_refused(r1, r2, mu):
    with pytest.raises(ValueError):
        lambert(r1, r2, TOF_BOOK, mu)


def test_exhausted_iterations_raise_runtime_error():
    with pytest.raises(RuntimeError):
        lambert(R1_BOOK, R2_BOOK, TOF_BOOK, MU_BOOK, max_iter=1)


def test_lambert_matches_cw_targeting_on_the_v_bar_up_to_the_curvature_terms():
    a = EARTH.radius + 500e3
    n = mean_motion(a, MU)
    tf = 1.0 / n
    r_t0, v_t0 = circular_orbit(EARTH, 500e3)
    r_t1, v_t1 = propagate_kepler(r_t0, v_t0, tf, MU)
    C0, _ = hill_frame(r_t0, v_t0)
    C1, _ = hill_frame(r_t1, v_t1)
    # Against the exact two-body transfer, CW is off by 0.30 rho / a in the first burn and
    # 0.09 rho / a in the second for this geometry (nonlinear relative equations, DOP853 at
    # rtol 1e-13); 0.5 rho / a covers both. Lambert works on inertial vectors of size a, so its
    # rounding floor is about eps v_circ (1.2 eps v_circ measured), which is 400 times the
    # physical gap at 1 cm; 32 eps v_circ leaves margin for a different rounding path.
    floor = 32.0 * EPS * n * a
    for rho in (1e-2, 1e2, 1e3, 1e4, 1e5):
        state = np.array([0.0, -rho, 0.0, 0.0, 0.0, 0.0])
        dv1_cw, dv2_cw = two_impulse(n, state, np.zeros(3), tf)
        dr, dv = hill_to_inertial(r_t0, v_t0, state)
        v1, v2 = lambert(r_t0 + dr, r_t1, tf, MU)
        dv1 = C0.T @ (v1 - (v_t0 + dv))
        dv2 = C1.T @ (v_t1 - v2)
        gap1 = np.linalg.norm(dv1 - dv1_cw)
        gap2 = np.linalg.norm(dv2 - dv2_cw)
        assert gap1 <= 0.5 * rho / a * np.linalg.norm(dv1_cw) + floor
        assert gap2 <= 0.5 * rho / a * np.linalg.norm(dv2_cw) + floor
    assert gap1 >= 0.2 * rho / a * np.linalg.norm(dv1_cw)
