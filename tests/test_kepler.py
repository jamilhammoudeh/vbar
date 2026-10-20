import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vbar.dynamics.bodies import EARTH
from vbar.dynamics.kepler import (
    elements_from_state,
    propagate_kepler,
    solve_kepler,
    state_from_elements,
)

MU = EARTH.mu


def random_elements(rng):
    return np.array(
        [
            rng.uniform(6.8e6, 4.2e7),
            rng.uniform(0.0, 0.9),
            rng.uniform(0.05, np.pi - 0.05),
            *rng.uniform(0.0, 2 * np.pi, 3),
        ]
    )


def wrapped(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def test_kepler_equation_matches_published_solutions():
    # Schaub and Junkins, Analytical Mechanics of Space Systems, Kepler's equation table.
    assert solve_kepler(1.5, 0.8) == pytest.approx(2.163532303940202, abs=1e-15)
    # Montenbruck and Gill, Satellite Orbits, exercise 2.2.
    assert solve_kepler(math.radians(50.0), 0.72) == pytest.approx(1.59249513093, abs=5e-12)


@pytest.mark.parametrize("e", [0.0, 0.3, 0.85, 0.99])
def test_kepler_solution_keeps_the_revolution_of_the_mean_anomaly(e):
    for M in np.linspace(-20.0, 20.0, 41):
        E = solve_kepler(M, e)
        assert E - e * math.sin(E) == pytest.approx(M, abs=1e-13)


@pytest.mark.parametrize(
    "M, e", [(5.962388955709969, 0.4554446879160051), (6.178474180256207, 0.8286765614028038)]
)
def test_kepler_solution_converges_just_below_a_full_revolution(M, e):
    # Newton's correction here alternates between neighbouring floats of E near 2 pi.
    E = solve_kepler(M, e)
    assert E - e * math.sin(E) == pytest.approx(M, abs=4 * np.finfo(float).eps * M)


def test_elements_from_state_match_published_example():
    # Montenbruck and Gill, Satellite Orbits, exercise 2.3, with the book's GM.
    r = np.array([10000e3, 40000e3, -5000e3])
    v = np.array([-1.5e3, 1.0e3, -0.1e3])
    a, e, i, raan, argp, M = elements_from_state(r, v, 398600.4415e9)
    assert a == pytest.approx(25015.181e3, abs=0.5)
    assert e == pytest.approx(0.7079772, abs=5e-8)
    expected = [6.971, 173.290, 91.553, 144.225]
    assert np.degrees([i, raan, argp, M]) == pytest.approx(expected, abs=5e-4)


def test_elements_and_state_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        elements = random_elements(rng)
        back = elements_from_state(*state_from_elements(elements, MU), MU)
        assert back[0] == pytest.approx(elements[0], rel=1e-14)
        assert back[1] == pytest.approx(elements[1], abs=1e-14)
        assert np.max(np.abs(wrapped(back[2:] - elements[2:]))) < 1e-12


def test_equatorial_circular_state_gives_angles_from_the_x_axis():
    a = EARTH.radius + 500e3
    u = 0.7
    r = a * np.array([math.cos(u), math.sin(u), 0.0])
    v = math.sqrt(MU / a) * np.array([-math.sin(u), math.cos(u), 0.0])
    a_out, e, i, raan, argp, M = elements_from_state(r, v, MU)
    assert a_out == pytest.approx(a, rel=1e-14)
    assert e < 1e-15
    assert i == 0.0
    assert wrapped(raan + argp + M - u) == pytest.approx(0.0, abs=1e-14)


def test_keplerian_transition_matrix_matches_published_example():
    # Montenbruck and Gill, exercise 7.1: a = R + 650 km, e = 0.001, i = 51 deg, other angles 0,
    # one day, velocity rows and columns scaled by 1/n. Only the first row is compared; it is
    # printed to 5 decimals and central differences add under 1e-6.
    mu = 398600.4415e9
    a = 6378.137e3 + 650e3
    n = math.sqrt(mu / a**3)
    y0 = np.r_[state_from_elements([a, 0.001, math.radians(51.0), 0.0, 0.0, 0.0], mu)]
    row = np.zeros(6)
    for j, h in enumerate([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3]):
        plus, minus = y0.copy(), y0.copy()
        plus[j] += h
        minus[j] -= h
        x_plus = propagate_kepler(plus[:3], plus[3:], 86400.0, mu)[0][0]
        x_minus = propagate_kepler(minus[:3], minus[3:], 86400.0, mu)[0][0]
        row[j] = (x_plus - x_minus) / (2 * h)
    row[3:] *= n
    expected = [-279.15804, -0.68758, -0.84909, -2.08471, -176.75552, -218.27488]
    assert np.max(np.abs(row - expected)) < 6e-6


def test_universal_variables_match_mean_anomaly_propagation():
    rng = np.random.default_rng(1)
    for _ in range(20):
        elements = random_elements(rng)
        period = 2 * np.pi * math.sqrt(elements[0] ** 3 / MU)
        t = rng.uniform(-12.0, 12.0) * period
        r, v = propagate_kepler(*state_from_elements(elements, MU), t, MU)
        later = elements.copy()
        later[5] += 2 * np.pi * t / period
        r_ref, v_ref = state_from_elements(later, MU)
        assert np.linalg.norm(r - r_ref) < 1e-12 * np.linalg.norm(r_ref)
        assert np.linalg.norm(v - v_ref) < 1e-12 * np.linalg.norm(v_ref)


@pytest.mark.parametrize("t", [60.0, 3000.0, 4e4, -3000.0])
def test_hyperbolic_propagation_composes_and_conserves_energy(t):
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 12e3, 1e3])
    r1, v1 = propagate_kepler(r0, v0, t, MU)
    r_half, v_half = propagate_kepler(r0, v0, 0.5 * t, MU)
    r2, v2 = propagate_kepler(r_half, v_half, 0.5 * t, MU)
    assert np.linalg.norm(r2 - r1) < 1e-14 * np.linalg.norm(r1)
    assert np.linalg.norm(v2 - v1) < 1e-14 * np.linalg.norm(v1)
    energy = [0.5 * v @ v - MU / np.linalg.norm(r) for r, v in ((r0, v0), (r1, v1))]
    assert energy[1] == pytest.approx(energy[0], rel=1e-14)
    h0, h1 = np.cross(r0, v0), np.cross(r1, v1)
    assert np.linalg.norm(h1 - h0) < 1e-13 * np.linalg.norm(h0)


def test_propagation_runs_backwards_to_the_start():
    rng = np.random.default_rng(2)
    elements = random_elements(rng)
    r0, v0 = state_from_elements(elements, MU)
    r1, v1 = propagate_kepler(r0, v0, 5e4, MU)
    r2, v2 = propagate_kepler(r1, v1, -5e4, MU)
    assert np.linalg.norm(r2 - r0) < 1e-13 * np.linalg.norm(r0)
    assert np.linalg.norm(v2 - v0) < 1e-13 * np.linalg.norm(v0)


def test_near_radial_hyperbola_runs_back_to_its_start():
    r0 = np.array([-9500485.524854492, 4807575.325180032, 4890361.797190876])
    v0 = np.array([-11034.507532596703, -3746.9491895108595, 20270.285813437044])
    t = -920814.2466715943
    r1, v1 = propagate_kepler(r0, v0, t, MU)
    # At 2e10 m the velocity is 0.03 deg off radial and the residual's terms cancel from 9e16,
    # so the return trip ends on a bracket closed to rounding, not on a small Newton step.
    r2, _ = propagate_kepler(r1, v1, -t, MU)
    assert np.linalg.norm(r2 - r0) < 1e-12 * np.linalg.norm(r1)


def random_hyperbola(rng, radial):
    """A hyperbolic state 6700 to 50 000 km out, moving within 1 deg of radial if `radial`."""
    r0 = rng.normal(size=3)
    r0 *= rng.uniform(6.7e6, 5e7) / np.linalg.norm(r0)
    direction = rng.normal(size=3)
    if radial:
        offset = direction / np.linalg.norm(direction) * math.radians(rng.uniform(0.0, 1.0))
        direction = rng.choice([-1.0, 1.0]) * r0 / np.linalg.norm(r0) + offset
    v_escape = math.sqrt(2.0 * MU / np.linalg.norm(r0))
    v0 = direction * v_escape * rng.uniform(1.0001, 3.0) / np.linalg.norm(direction)
    return r0, v0


@pytest.mark.parametrize("radial", [False, True])
def test_random_hyperbolas_run_back_to_their_start(radial):
    # Flown back in along a near-radial arc, the residual's terms reach 1e19 and cancel, so the
    # return carries their rounding: up to 3e-10 of the far radius in 8000 trials. The problem
    # itself is far better conditioned; this is the formulation's limit.
    bound = 1e-9 if radial else 1e-10
    rng = np.random.default_rng(1)
    for _ in range(500):
        r0, v0 = random_hyperbola(rng, radial)
        t = rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(0.0, 6.0)
        r1, v1 = propagate_kepler(r0, v0, t, MU)
        r2, _ = propagate_kepler(r1, v1, -t, MU)
        assert np.linalg.norm(r2 - r0) < bound * np.linalg.norm(r1)


def two_body(t, y):
    return np.r_[y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3]


def test_near_radial_hyperbolic_arcs_match_an_integrator():
    rng = np.random.default_rng(4)
    # An outbound arc whose hyperbolic starting value has the wrong sign.
    cases = [([3676020.27, -19137178.82, 4609418.25], [1942.23, -10111.18, 2435.40], 294.0)]
    for _ in range(10):
        r0, v0 = random_hyperbola(rng, radial=True)
        cases.append((r0, v0, rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(0.0, 3.0)))
    for r0, v0, t in cases:
        r1, _ = propagate_kepler(r0, v0, t, MU)
        start = np.r_[r0, v0]
        reference = solve_ivp(two_body, (0.0, t), start, "DOP853", rtol=1e-12, atol=1e-6).y[:3, -1]
        assert np.linalg.norm(r1 - reference) < 1e-9 * np.linalg.norm(reference)


def test_a_slow_escape_flown_for_a_year_matches_an_integrator():
    # 169 m/s of excess speed from a 7000 km perigee; with the near-zero energy the Kepler
    # function's sinh terms overflow if the solver ever steps far past the root.
    r_p = 7000e3
    r0 = np.array([r_p, 0.0, 0.0])
    v0 = np.array([0.0, 1.000125 * math.sqrt(2.0 * MU / r_p), 0.0])
    t = 3e7
    r1, _ = propagate_kepler(r0, v0, t, MU)
    start = np.r_[r0, v0]
    reference = solve_ivp(two_body, (0.0, t), start, "DOP853", rtol=1e-12, atol=1e-6).y[:3, -1]
    assert np.linalg.norm(r1 - reference) < 1e-9 * np.linalg.norm(reference)


@pytest.mark.parametrize("t", [1e8, -1e8, 1e9, -1e9, 1e10, -1e10])
def test_long_near_parabolic_hyperbolic_arcs_compose_from_their_halves(t):
    r_p = 7000e3
    r0 = np.array([r_p, 0.0, 0.0])
    v0 = np.array([0.0, (1.0 + 1e-6) * math.sqrt(2.0 * MU / r_p), 0.0])
    r1, v1 = propagate_kepler(r0, v0, t, MU)
    r_half, v_half = propagate_kepler(r0, v0, 0.5 * t, MU)
    r2, v2 = propagate_kepler(r_half, v_half, 0.5 * t, MU)
    assert np.linalg.norm(r2 - r1) < 1e-11 * np.linalg.norm(r1)
    energy = [0.5 * v @ v - MU / np.linalg.norm(r) for r, v in ((r0, v0), (r1, v1))]
    assert energy[1] == pytest.approx(energy[0], rel=1e-6)


def test_an_outbound_near_parabolic_arc_flown_far_back_stays_finite():
    # Off perigee the residual's terms overflow with opposite signs (inf - inf) on a backward
    # arc; read as past the root, the solver still closes on it.
    r_norm, angle = 25176957.067439117, 0.7243643379663687
    r0 = np.array([r_norm, 0.0, 0.0])
    speed = (1.0 + 7.633e-6) * math.sqrt(2.0 * MU / r_norm)
    v0 = speed * np.array([math.sin(angle), math.cos(angle), 0.0])
    t = -3.1622776601683795e9
    r1, _ = propagate_kepler(r0, v0, t, MU)
    r_half, v_half = propagate_kepler(r0, v0, 0.5 * t, MU)
    r2, _ = propagate_kepler(r_half, v_half, 0.5 * t, MU)
    assert np.all(np.isfinite(r1))
    assert np.linalg.norm(r2 - r1) < 1e-11 * np.linalg.norm(r1)


@pytest.mark.parametrize(
    ("r0", "v0", "t", "mu"),
    [
        # 73 times escape speed toward perigee: the radius terms near 2.6e31 cancel to nothing.
        (
            [-43062279582.375, 37606472960.125, -47694531068.0],
            [-424534.578874588, 370747.86389541626, -470202.17639541626],
            -101444.8594899719,
            MU,
        ),
        # 19 times escape speed: a state 3e-3 off at the end, against an estimate of 4e-3 there.
        (
            [7748598854.9921875, -105275811173.00781, 34282050427.4375],
            [-8088.16681227088, 109889.54576922953, -35784.45046169311],
            958197.8516178454,
            MU,
        ),
        # Canonical units, far out against |a|: returned unchecked, the end velocity was reversed.
        (
            [-2144361.5576695204, -8550373.989763737, -8516332.71536541],
            [20.736314289089933, 82.68346431925966, 82.35427982968395],
            103410.93068920661,
            1.0,
        ),
        # 3.4 times past the threshold: estimated 3.4e-5 of the end radius, 2.7e-5 off.
        (
            [21601713162736.074, 10158077848490.854, -6871578556309.546],
            [-83.46701687549394, -39.249851939921626, 26.55113829280097],
            256517939840.39578,
            MU,
        ),
        # 1.4 times past it, 9.7 times escape speed: estimated 1.4e-5, 1.7e-5 off. The circular
        # speed in place of the end speed would estimate 1e-6.
        (
            [-8013715776.852051, 364342485.97387695, 2288597021.9437256],
            [96140.64190871082, -4370.781400918961, -27456.132882159203],
            83422.67419104476,
            MU,
        ),
        # An ellipse with 1 - e = 1e-8 and a parabola flown in to a 7000 km perigee from 1.6e7 and
        # 1.7e7 perigee radii: little cancels, but the end speed is high. Estimated 2.8e-5 and
        # 3.05e-5, about 2 eps (r0 / rp)^1.5.
        (
            [-101184302788927.67, -55339265096181.234, 0.0],
            [2.2099593795789376, 1.207922795071485, 0.0],
            3e13,
            MU,
        ),
        (
            [-102920742826148.11, -56291164571881.82, 0.0],
            [2.2874333255518025, 1.250356149069808, 0.0],
            3e13,
            MU,
        ),
        # An ellipse with 1 - e = 1e-7 from apogee to its 7000 km perigee: nothing cancels, but a
        # time error of eps t alone moves the perigee by eps pi sqrt(2) (1 - e)^-1.5 of its radius,
        # 3.1e-5; unchecked, it comes back 4.5e-5 off.
        (
            [-133871120621992.19, -40761022759371.125, -4130390662165.415],
            [0.00015297460609412034, -0.000486514597373636, -0.00015689783878587935],
            92156939776371.02,
            MU,
        ),
        # An ellipse with 1 - e = 1e-6 from apogee over 100.5 periods: each period taken off
        # carries the period's rounding. Estimated 3e-4; unchecked, 1.4e-4 off.
        (
            [-13999993000000.0, 0.0, 0.0],
            [0.0, -0.005335866786596964, 0.0],
            585765922087444.6,
            MU,
        ),
        # 1e6 periods from a 7000 km perigee at e = 0.98, where 2 / r0 and v0^2 / mu cancel to
        # 1 / 100 of either, so the period is known to about 300 eps. Estimated 2.1e-4;
        # unchecked, 3.5e-5 off, and one ulp of the starting speed moves it 1.6e-4.
        ([7e6, 0.0, 0.0], [0.0, 10618.238182916357, 0.0], 2060691819383.197, MU),
        # |a| = 1e-6 over 1e301: the bracket closes on an overflowed point, an infinite radius.
        ([1.0, 0.0, 0.0], [0.0, math.sqrt(1e6 + 2.0), 0.0], 1e301, 1.0),
    ],
)
def test_arcs_whose_estimated_end_error_passes_the_threshold_are_refused(r0, v0, t, mu):
    with pytest.raises(ValueError, match="1e-5 of its radius"):
        propagate_kepler(np.array(r0), np.array(v0), t, mu)


@pytest.mark.parametrize(
    ("r0", "v0", "t"),
    [
        # 4.4 times inside the threshold: estimated 2.3e-6 of the end radius, 3.3e-6 off.
        (
            [-3172277925966.012, -7330152259582.755, -9205154186468.893],
            [20.699405175995018, 47.82991754335562, 60.06447753336198],
            150064180943.9222,
        ),
        # 1.1 times inside it: estimated 8.8e-6, 2.7e-6 off in position and 3.8e-6 in velocity.
        (
            [-3106658836.328224, 7338504825.029907, 3680047357.05188],
            [-69056.50777751184, 163121.61335238256, 81799.98800402135],
            -45078.528613785245,
        ),
        # 1e4 periods from a 7000 km perigee at e = 0.98: estimated 2.1e-6, 3.5e-7 off.
        ([7e6, 0.0, 0.0], [0.0, 10618.238182916357, 0.0], 20606918193.83197),
        # Nearly at rest: vis-viva's argument can round below zero; the speed is then zero.
        (
            [5539197.509528515, 62879444.171807274, 36954234.60342952],
            [-5.805750965800457e-05, -0.0006590528557638296, -0.00038732520887766543],
            -0.010332882417905993,
        ),
    ],
)
def test_arcs_inside_the_rounding_threshold_are_returned(r0, v0, t):
    r1, v1 = propagate_kepler(np.array(r0), np.array(v0), t, MU)
    assert np.all(np.isfinite(r1)) and np.all(np.isfinite(v1))


def test_free_fall_from_rest_is_not_refused():
    # Zero speed at the start or after no time leaves nothing for the rounding check to scale.
    r0, v0 = np.array([7000e3, 0.0, 0.0]), np.zeros(3)
    assert np.array_equal(propagate_kepler(r0, v0, 0.0, MU)[0], r0)
    r1, v1 = propagate_kepler(r0, v0, 60.0, MU)
    assert r1[1:].tolist() == [0.0, 0.0] and r1[0] < 7000e3
    assert v1 @ v1 == pytest.approx(2.0 * MU * (1.0 / r1[0] - 1.0 / 7000e3), rel=1e-12)


def test_a_near_radial_hyperbola_in_canonical_units_runs_back_without_overflow():
    # mu = 1 and |a| far below one unit: a doubling floor of one unit of chi overshoots into
    # sinh overflow, which the solver must read as being past the root.
    r0 = np.array([1.3062270583062603, 0.5741915299732164, 1.4093036972845525])
    v0 = np.array([487.6565855806324, 214.30050316875523, 526.0886762042227])
    t = -586750.502222578
    r1, _ = propagate_kepler(r0, v0, t, 1.0)
    r_half, v_half = propagate_kepler(r0, v0, 0.5 * t, 1.0)
    r2, _ = propagate_kepler(r_half, v_half, 0.5 * t, 1.0)
    assert np.linalg.norm(r2 - r1) < 1e-8 * np.linalg.norm(r1)


@pytest.mark.parametrize("excess", [-1e-4, -1e-6, -1e-10, -1e-12, 1e-10, 1e-12])
def test_one_second_near_parabolic_arcs_match_a_taylor_step(excess):
    # At perigee, with speed (1 + excess) times escape speed: the period of the ellipses is
    # up to 1e18 s, so reducing a negative time by a whole period loses the time itself.
    r_p = 7000e3
    r0 = np.array([r_p, 0.0, 0.0])
    v0 = np.array([0.0, (1.0 + excess) * math.sqrt(2.0 * MU / r_p), 0.0])
    accel = -MU * r0 / r_p**3
    jerk = -MU * v0 / r_p**3
    for t in (-1.0, 1.0):
        r1, _ = propagate_kepler(r0, v0, t, MU)
        # The series through the jerk leaves out a fourth-order term of about 4e-7 m.
        taylor = r0 + v0 * t + accel * t**2 / 2.0 + jerk * t**3 / 6.0
        assert np.linalg.norm(r1 - taylor) < 1e-5


@pytest.mark.parametrize("e", [0.9, 0.9477, 0.98])
def test_universal_propagation_over_many_periods_matches_the_mean_anomaly(e):
    elements = np.array([10818400.0, e, 5.187, 2.41, 1.479, 0.089])
    r0, v0 = state_from_elements(elements, MU)
    period = 2.0 * np.pi * np.sqrt(elements[0] ** 3 / MU)
    for t in (-13.6 * period, 0.37 * period, 20.25 * period):
        r1, _ = propagate_kepler(r0, v0, t, MU)
        expected = elements.copy()
        expected[5] += 2.0 * np.pi * t / period
        r_ref, _ = state_from_elements(expected, MU)
        # Both reduce to one revolution, so the gap is the rounding of the reduced time.
        assert np.linalg.norm(r1 - r_ref) < 1e-9 * np.linalg.norm(r_ref)
