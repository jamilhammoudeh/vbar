import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vbar.conventions import hill_to_iss
from vbar.dynamics.bodies import EARTH
from vbar.dynamics.models import CWDynamics, TwoBodyDynamics
from vbar.dynamics.relative import (
    CurvilinearCWDynamics,
    YamanakaAnkersenDynamics,
    curvilinear_to_hill,
    hill_to_curvilinear,
    true_anomaly,
    ya_in_plane,
    ya_in_plane_inverse,
)

ZERO = np.zeros(3)
EPS = np.finfo(float).eps
S0 = np.array([150.0, -400.0, 30.0, 0.05, 0.1, -0.02])
ACCEL = np.array([2e-4, -1e-4, 5e-5])
A_500 = EARTH.radius + 500e3


def linearized_elliptic_rhs(t, s, a, e, mean_anomaly, n, accel):
    """Two-body motion linearized about a Keplerian target, in the target's rotating Hill frame."""
    f = true_anomaly(mean_anomaly + n * t, e)
    p = a * (1 - e * e)
    r = p / (1 + e * np.cos(f))
    f_dot = np.sqrt(EARTH.mu * p) / r**2
    r_dot = np.sqrt(EARTH.mu / p) * e * np.sin(f)
    f_ddot = -2 * r_dot * f_dot / r
    w = EARTH.mu / r**3
    x, y, z, vx, vy, vz = s
    return [
        vx,
        vy,
        vz,
        2 * f_dot * vy + f_ddot * y + f_dot**2 * x + 2 * w * x + accel[0],
        -2 * f_dot * vx - f_ddot * x + f_dot**2 * y - w * y + accel[1],
        -w * z + accel[2],
    ]


def test_ya_matrix_is_in_the_papers_frame_and_order():
    # The paper's x is along-track and z points at the central body: the ISS-style LVLH axes.
    # At e = 0 its first column is a constant along-track offset and its second the CW radial
    # oscillation z = sin f, with x' = 2 sin f and z' = cos f.
    f = np.array([0.3, 1.9])
    psi = ya_in_plane(f, 0.0, 0.0)
    assert np.allclose(psi[:, :, 0], [1.0, 0.0, 0.0, 0.0])
    expected = np.stack([-2 * np.cos(f), np.sin(f), 2 * np.sin(f), np.cos(f)], axis=-1)
    assert np.allclose(psi[:, :, 1], expected, atol=1e-15)
    hill = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    iss = hill_to_iss(hill)
    assert (iss[0], iss[2]) == (hill[1], -hill[0])


@pytest.mark.parametrize("e", [0.0, 0.3, 0.9])
@pytest.mark.parametrize("J", [0.0, 1.7, -40.0])
def test_ya_inverse_is_exact(e, J):
    f = np.array([0.3, 2.0, -3.0])
    inverse, psi = ya_in_plane_inverse(f, e, J), ya_in_plane(f, e, J)
    residual = np.max(np.abs(inverse @ psi - np.eye(4)))
    # A product of 4x4 matrices rounds to a few eps times the product of their entry sizes.
    assert residual < 4 * EPS * np.max(np.abs(inverse)) * np.max(np.abs(psi))


@pytest.mark.parametrize("e", [0.1, 0.5, 0.8])
def test_ya_matches_integrated_linear_equations_over_two_orbits(e):
    # Alfriend et al., Spacecraft Formation Flying, example 5.5: YA is indistinguishable from
    # integrating the linear equations. Here with thrust and samples inside the segment.
    a, mean_anomaly = 8000e3, 0.7
    m = YamanakaAnkersenDynamics(a, e, mean_anomaly)
    duration = 2 * 2 * np.pi / m.n
    m.reset(S0)
    final, samples = m.advance(ACCEL, duration, n_samples=9)
    reference = solve_ivp(
        linearized_elliptic_rhs,
        (0.0, duration),
        S0,
        method="DOP853",
        rtol=3e-14,
        atol=1e-14,
        args=(a, e, mean_anomaly, m.n, ACCEL),
        dense_output=True,
    )
    expected = reference.sol(np.linspace(0.0, duration, 9)).T
    scale = np.max(np.abs(expected))
    # The reference itself is good to about 2e-12 of the state at e = 0.8.
    assert np.max(np.abs(samples - expected)) < 1e-11 * scale
    assert np.max(np.abs(final - expected[-1])) < 1e-11 * scale


@pytest.mark.parametrize("duration", [1.0, 10.0, 600.0, 5677.0])
def test_circular_ya_is_cw(duration):
    m = YamanakaAnkersenDynamics(EARTH.radius + 500e3, 0.0)
    cw = CWDynamics(m.n)
    m.reset(S0)
    cw.reset(S0)
    ya_final, _ = m.advance(ACCEL, duration)
    cw_final, _ = cw.advance(ACCEL, duration)
    assert np.max(np.abs(ya_final - cw_final)) < 1e-13 * max(1.0, np.max(np.abs(cw_final)))


def one_orbit_error(model, rho, e):
    """Largest position error after one orbit against the two-body model, elliptic target."""
    a, mean_anomaly = 7500e3, 0.4
    truth = TwoBodyDynamics(elements=[a, e, 0.9, 0.3, 1.1, mean_anomaly], rtol=1e-13)
    u = np.array([0.3, -0.8, 0.5]) / np.linalg.norm([0.3, -0.8, 0.5])
    start = np.r_[rho * u, 0.4 * truth.n * rho * u[[1, 2, 0]]]
    approx = model(truth, a, e, mean_anomaly)
    errors = []
    for m in (truth, approx):
        m.reset(start)
        errors.append(m.advance(ZERO, 2 * np.pi / truth.n)[0][:3])
    return np.max(np.abs(errors[1] - errors[0]))


def ya(truth, a, e, mean_anomaly):
    return YamanakaAnkersenDynamics(a, e, mean_anomaly)


def cw(truth, a, e, mean_anomaly):
    return CWDynamics(truth.n)


def test_ya_error_against_two_body_grows_as_range_squared():
    errors = [one_orbit_error(ya, rho, 0.1) for rho in (100.0, 200.0, 400.0)]
    assert errors[1] / errors[0] == pytest.approx(4.0, rel=0.02)
    assert errors[2] / errors[1] == pytest.approx(4.0, rel=0.02)


def test_cw_error_on_an_elliptic_target_grows_with_range_and_eccentricity():
    # Linear in rho, and about linear in e: CW leaves out e rho terms.
    errors = [one_orbit_error(cw, rho, 0.1) for rho in (100.0, 200.0)]
    assert errors[1] / errors[0] == pytest.approx(2.0, rel=0.02)
    assert one_orbit_error(cw, 100.0, 0.01) < 0.15 * errors[0]
    # At 100 m YA is better than CW by three orders of magnitude.
    assert one_orbit_error(ya, 100.0, 0.1) < 1e-3 * errors[0]


@pytest.mark.parametrize("e", [0.0, 0.2, 0.6])
def test_bounded_start_repeats_every_orbit(e):
    m = YamanakaAnkersenDynamics(9000e3, e, mean_anomaly=1.3)
    m.reset(np.zeros(6))
    start = m.make_bounded([200.0, -300.0, 50.0, 0.1, 0.0, -0.05])
    if e == 0.0:
        assert start[4] == pytest.approx(-2 * m.n * start[0], rel=1e-12)
    m.reset(start)
    for _ in range(3):
        final, _ = m.advance(ZERO, 2 * np.pi / m.n)
    assert np.max(np.abs(final - start)) < 1e-9 * np.max(np.abs(start[:3]))


def test_ya_segments_compose_and_run_backward():
    m = YamanakaAnkersenDynamics(8000e3, 0.3, 0.2)
    m.reset(S0)
    whole, _ = m.advance(ACCEL, 900.0)
    phi, gamma = m.matrices(0.0, 350.0)
    assert np.allclose(m.state_in_last_segment(350.0), phi @ S0 + gamma @ ACCEL, atol=1e-12)
    m.reset(S0)
    m.advance(ACCEL, 350.0)
    split, _ = m.advance(ACCEL, 550.0)
    assert np.max(np.abs(split - whole)) < 1e-12 * np.max(np.abs(whole))
    back, _ = m.advance(ACCEL, -900.0)
    assert np.max(np.abs(back - S0)) < 1e-12 * np.max(np.abs(whole))


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 5e4])
def test_curvilinear_map_round_trips_to_rounding(scale):
    rng = np.random.default_rng(int(scale * 1000))
    for _ in range(20):
        s = np.r_[rng.normal(size=3) * scale, rng.normal(size=3) * scale * 1e-3]
        back = curvilinear_to_hill(hill_to_curvilinear(s, A_500), A_500)
        assert np.max(np.abs(back[:3] - s[:3])) < 4 * EPS * (A_500 * (scale / A_500) ** 2 + scale)
        assert np.max(np.abs(back[3:] - s[3:])) < 1e-15 * max(1.0, scale)


def test_curvilinear_coordinates_measure_arcs_and_the_radius_difference():
    s = 0.01
    on_circle = [A_500 * (np.cos(s) - 1), A_500 * np.sin(s), 0.0, 0.0, 0.0, 0.0]
    assert hill_to_curvilinear(on_circle, A_500) == pytest.approx(
        [0.0, A_500 * s, 0.0, 0.0, 0.0, 0.0], abs=1e-9
    )
    above = hill_to_curvilinear([0.0, 0.0, 100.0, 0.0, 0.0, 0.0], A_500)
    assert above[0] == pytest.approx(100.0**2 / (np.hypot(A_500, 100.0) + A_500), rel=1e-14)
    assert above[2] == pytest.approx(A_500 * np.arctan(100.0 / A_500), rel=1e-12)


def test_co_orbital_chaser_is_stationary_in_curvilinear_cw_and_in_two_body():
    # Alfriend et al., Spacecraft Formation Flying, eqs 4.77 and 5.70 to 5.71: a chaser on the
    # target's circle at angle s, at rest in the Hill frame, stays there; Cartesian CW sees it
    # drift.
    s = 5000.0 / A_500
    start = [-2 * A_500 * np.sin(s / 2) ** 2, A_500 * np.sin(s), 0.0, 0.0, 0.0, 0.0]
    curvilinear = CurvilinearCWDynamics(A_500)
    period = 2 * np.pi / curvilinear.n
    finals = {}
    for m in (curvilinear, TwoBodyDynamics(rtol=1e-13), CWDynamics(curvilinear.n)):
        m.reset(start)
        finals[m.name] = m.advance(ZERO, period)[0]
    assert np.max(np.abs(finals["cw_curvilinear"] - start)) < 1e-9
    assert np.max(np.abs(finals["twobody"][:3] - start[:3])) < 1e-6
    assert abs(finals["cw"][1] - start[1]) > 10.0


def test_curvilinear_cw_error_for_vbar_offsets_falls_as_the_fourth_power():
    truth = TwoBodyDynamics(rtol=1e-13)
    curvilinear = CurvilinearCWDynamics(A_500)
    cw = CWDynamics(truth.n)
    period = 2 * np.pi / truth.n
    errors = {}
    for d in (1000.0, 2000.0):
        out = {}
        for m in (truth, curvilinear, cw):
            m.reset([0.0, d, 0.0, 0.0, 0.0, 0.0])
            out[m.name] = m.advance(ZERO, period)[0][:3]
        errors[d] = np.max(np.abs(out["cw_curvilinear"] - out["twobody"]))
        # Cartesian CW misses the second-order drift 6 pi d^2 / a.
        assert np.max(np.abs(out["cw"] - out["twobody"])) == pytest.approx(
            6 * np.pi * d**2 / A_500, rel=1e-2
        )
    assert errors[1000.0] < 1e-6
    assert errors[2000.0] / errors[1000.0] == pytest.approx(16.0, rel=0.05)


def test_curvilinear_cw_drifts_a_cw_bounded_radial_offset():
    # Bounded in Cartesian CW is not bounded in curvilinear CW: -12 pi X^2 / a per orbit.
    m = CurvilinearCWDynamics(A_500)
    x = 1000.0
    m.reset([x, 0.0, 0.0, 0.0, -2 * m.n * x, 0.0])
    final, _ = m.advance(ZERO, 2 * np.pi / m.n)
    assert final[1] == pytest.approx(-12 * np.pi * x**2 / A_500, rel=1e-3)


def test_curvilinear_cw_burns_along_hill_axes_like_cw():
    curvilinear = CurvilinearCWDynamics(A_500)
    cw = CWDynamics(curvilinear.n)
    accel = [1e-3, -2e-3, 5e-4]
    for m in (curvilinear, cw):
        m.reset(np.zeros(6))
    final, samples = curvilinear.advance(accel, 60.0, n_samples=5)
    # A 60 s burn moves the chaser 4 m, where curvature terms are below 1e-5 m.
    assert np.allclose(final, cw.advance(accel, 60.0)[0], rtol=0, atol=1e-5)
    assert np.allclose(samples[-1], final, rtol=0, atol=1e-12)
    assert np.allclose(curvilinear.state_in_last_segment(30.0), samples[2], rtol=0, atol=1e-12)
