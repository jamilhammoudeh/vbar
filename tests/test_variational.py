import math

import numpy as np
import pytest

from vbar.conventions import hill_frame
from vbar.dynamics.bodies import EARTH, Body
from vbar.dynamics.kepler import state_from_elements
from vbar.dynamics.models import CWDynamics, TwoBodyDynamics, gravity, gravity_gradient
from vbar.dynamics.perturbations import Drag, HarrisPriester, SolarPressure
from vbar.dynamics.relative import LinearizedTwoBodyDynamics, YamanakaAnkersenDynamics

ZERO = np.zeros(3)
S0 = np.array([300.0, -800.0, 150.0, 0.1, -0.2, 0.05])
ACCEL = np.array([1e-4, -2e-4, 3e-5])


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def finite_difference_matrices(model, start, accel, duration):
    """Central differences of `advance` with respect to the start state and the acceleration."""
    phi = np.zeros((6, 6))
    for j, h in enumerate([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3]):
        step = np.zeros(6)
        step[j] = h
        model.reset(start + step)
        up, _ = model.advance(accel, duration)
        model.reset(start - step)
        down, _ = model.advance(accel, duration)
        phi[:, j] = (up - down) / (2 * h)
    gamma = np.zeros((6, 3))
    for j in range(3):
        step = np.zeros(3)
        step[j] = 1e-5
        model.reset(start)
        up, _ = model.advance(accel + step, duration)
        model.reset(start)
        down, _ = model.advance(accel - step, duration)
        gamma[:, j] = (up - down) / 2e-5
    return phi, gamma


def normalized(phi, n):
    """Velocities in units of length, n v, as in Montenbruck and Gill's exercise 7.1."""
    d = np.diag([1, 1, 1, 1 / n, 1 / n, 1 / n])
    return d @ phi @ np.linalg.inv(d)


@pytest.mark.parametrize("j2", [False, True])
def test_gravity_gradient_matches_finite_differences(j2):
    r = np.array([6.9e6, 1.2e6, 2.3e6])
    numeric = np.zeros((3, 3))
    for j in range(3):
        step = np.zeros(3)
        step[j] = 1.0
        numeric[:, j] = (gravity(r + step, EARTH, j2) - gravity(r - step, EARTH, j2)) / 2
    # Central differences at 1 m leave about 1e-10 of the gradient.
    error = np.max(np.abs(gravity_gradient(r, EARTH, j2) - numeric))
    assert error < 1e-9 * np.max(np.abs(numeric))
    assert np.allclose(gravity_gradient(np.stack([r, -r]), EARTH, j2)[1], numeric, rtol=1e-9)


FORCES = {
    "none": [],
    "drag": [Drag(HarrisPriester([0.3, 0.5, 0.8], exponent=4), 40.0, 60.0)],
    # A Sun direction the orbit never leaves; shadow crossings are tested elsewhere.
    "srp": [SolarPressure([0.0, 0.0, 1.0], 0.05, 0.01)],
}


@pytest.mark.parametrize("forces", FORCES)
def test_transition_matrices_match_finite_differences_of_the_model(forces):
    m = TwoBodyDynamics(inclination=0.9, j2=True, perturbations=FORCES[forces], rtol=1e-13)
    m.reset(S0)
    phi, gamma = m.transition_matrices(ACCEL, 1500.0)
    numeric_phi, numeric_gamma = finite_difference_matrices(m, S0, ACCEL, 1500.0)
    assert np.max(np.abs(normalized(phi - numeric_phi, m.n))) < 1e-9
    assert np.max(np.abs(gamma - numeric_gamma)) < 1e-9 * np.max(np.abs(numeric_gamma))


def test_transition_matrices_leave_the_state_alone():
    m = TwoBodyDynamics(inclination=0.9, j2=True)
    m.reset(S0)
    before = m.chaser_state()
    m.transition_matrices(ACCEL, 600.0)
    assert np.array_equal(m.chaser_state(), before)


def test_j2_transition_matrix_preserves_volume():
    # Conservative forces: det phi = 1 in the Hill frame too, whose map has determinant 1.
    m = TwoBodyDynamics(inclination=0.9, j2=True, rtol=1e-13)
    m.reset(S0)
    phi, _ = m.transition_matrices(ZERO, 2 * np.pi / m.n)
    assert np.linalg.det(phi) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize(
    "j2, row, tol",
    [
        (False, [-279.15804, -0.68758, -0.84909, -2.08471, -176.75552, -218.27488], 6e-6),
        # The book states neither its J2 value nor its integrator settings; with WGS 84's J2
        # the row agrees to 4e-5.
        (True, [-279.73142, -0.61300, -0.76831, -1.97488, -176.73944, -218.54595], 2e-4),
    ],
)
def test_transition_matrix_reproduces_published_example(j2, row, tol):
    # Montenbruck and Gill, Satellite Orbits, exercise 7.1: the inertial STM of an orbit with
    # a = R + 650 km, e = 0.001, i = 51 deg, other angles 0, after one day, velocities scaled
    # by 1/n; book GM 398600.4415 km^3/s^2 and R 6378.137 km. The chaser flies that orbit.
    body = Body("book", 398600.4415e9, 6378.137e3, j2=EARTH.j2 if j2 else 0.0)
    a = body.radius + 650e3
    r, v = state_from_elements([a, 0.001, math.radians(51.0), 0.0, 0.0, 0.0], body.mu)
    m = TwoBodyDynamics(body, altitude=650e3, inclination=math.radians(51.0), j2=j2)
    m.reset(np.zeros(6))
    target = m.target_state()
    to_hill_0 = hill_map(target, body, j2)
    m.reset(to_hill_0 @ np.r_[r - target[:3], v - target[3:]])
    phi, _ = m.transition_matrices(ZERO, 86400.0)
    m.advance(ZERO, 86400.0)
    inertial = np.linalg.inv(hill_map(m.target_state(), body, j2)) @ phi @ to_hill_0
    n = math.sqrt(body.mu / a**3)
    assert np.max(np.abs(normalized(inertial, n)[0] - row)) < tol


def hill_map(target, body, j2):
    """Linear map from the inertial offset (dr, dv) to the Hill state of the target `target`."""
    C, w = hill_frame(target[:3], target[3:], gravity(target[:3], body, j2))
    out = np.zeros((6, 6))
    out[:3, :3] = out[3:, 3:] = C.T
    out[3:, :3] = -skew(w) @ C.T
    return out


def test_linearized_two_body_is_cw_on_a_circular_orbit():
    m = LinearizedTwoBodyDynamics(j2=False, rtol=1e-13)
    cw = CWDynamics(m.n)
    period = 2 * np.pi / m.n
    for model in (m, cw):
        model.reset(S0)
    assert m.name == "linear_twobody"
    linear, _ = m.advance(ACCEL, period)
    exact, _ = cw.advance(ACCEL, period)
    assert np.max(np.abs(linear - exact)) < 1e-11 * np.max(np.abs(exact))


def test_linearized_two_body_is_yamanaka_ankersen_on_an_elliptic_orbit():
    # Two independent routes to the same linear model: integration and the closed form.
    m = LinearizedTwoBodyDynamics(j2=False, elements=[8000e3, 0.2, 0.9, 0.3, 1.1, 0.7], rtol=1e-13)
    ya = YamanakaAnkersenDynamics(8000e3, 0.2, 0.7)
    for model in (m, ya):
        model.reset(S0)
    period = 2 * np.pi / ya.n
    linear, _ = m.advance(ACCEL, period)
    closed_form, _ = ya.advance(ACCEL, period)
    assert np.max(np.abs(linear - closed_form)) < 1e-11 * np.max(np.abs(closed_form))


def pco_state(rho, alpha, n):
    s, c = np.sin(alpha), np.cos(alpha)
    return [0.5 * rho * s, rho * c, rho * s, 0.5 * rho * n * c, -rho * n * s, rho * n * c]


@pytest.mark.parametrize("alpha_deg, published", [(0.0, -11.0), (90.0, -3.5)])
def test_linear_j2_model_error_is_the_second_order_drift(alpha_deg, published):
    # Alfriend et al., Spacecraft Formation Flying, examples 7.3 to 7.6: 0.5 km projected
    # circular orbit about a mean 7100 km, 70 deg, node 45 deg target; the linear J2 model's
    # 15-orbit along-track error, read from figures, is -11 m and -3.5 m, the two-body
    # second-order drift of eq 5.64 with a small J2 part.
    mean = [7100e3, 0.0, np.radians(70.0), np.radians(45.0), 0.0, 0.0]
    truth = TwoBodyDynamics(j2=True, elements=mean)
    linear = LinearizedTwoBodyDynamics(j2=True, elements=mean)
    alpha = np.radians(alpha_deg)
    for m in (truth, linear):
        m.reset(pco_state(500.0, alpha, truth.n))
    duration = 15 * 2 * np.pi / truth.n
    error = truth.advance(ZERO, duration)[0][1] - linear.advance(ZERO, duration)[0][1]
    second_order = -15 * 9 * np.pi * 500.0**2 / (4 * 7100e3) * (2 + np.cos(2 * alpha))
    assert error == pytest.approx(second_order, rel=1e-3)
    assert error == pytest.approx(published, abs=0.5)
