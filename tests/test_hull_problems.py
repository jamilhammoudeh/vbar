"""Hull et al. two-body test problems D1 to D5 (e = 0.1 to 0.9).

As set out in Montenbruck and Gill, Satellite Orbits, sec. 4.1.6: mu = a = 1, start at
pericentre, integrate to t = 20 and score the norm of the state error.
"""

import math

import numpy as np
import pytest

from vbar.conventions import inertial_to_hill
from vbar.dynamics.bodies import EARTH
from vbar.dynamics.kepler import solve_kepler
from vbar.dynamics.models import TwoBodyDynamics


def exact_state(e, t):
    """Position and velocity of problem D(e) at time t, in units of a and sqrt(mu / a)."""
    E = solve_kepler(t, e)
    d = 1.0 - e * math.cos(E)
    q = math.sqrt(1.0 - e * e)
    return np.array([math.cos(E) - e, q * math.sin(E), -math.sin(E) / d, q * math.cos(E) / d])


def two_body(t, y):
    r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
    return np.array([y[2], y[3], -y[0] / r3, -y[1] / r3])


def rk4(y0, t_end, steps):
    h = t_end / steps
    y = np.array(y0, dtype=float)
    t = 0.0
    for _ in range(steps):
        k1 = two_body(t, y)
        k2 = two_body(t + h / 2, y + h / 2 * k1)
        k3 = two_body(t + h / 2, y + h / 2 * k2)
        k4 = two_body(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y


@pytest.mark.parametrize(
    "evaluations, error",
    [
        (200, 1.953e-1),
        (400, 6.663e-3),
        (1000, 9.051e-5),
        (2000, 4.012e-6),
        (3000, 6.847e-7),
        (4000, 1.996e-7),
        (6000, 3.608e-8),
        (8000, 1.089e-8),
    ],
)
def test_rk4_reproduces_published_d1_errors(evaluations, error):
    # Montenbruck and Gill, exercise 4.1; checks the problem set-up and error measure used below.
    e = 0.1
    y0 = [1.0 - e, 0.0, 0.0, math.sqrt((1.0 + e) / (1.0 - e))]
    measured = np.linalg.norm(rk4(y0, 20.0, evaluations // 4) - exact_state(e, 20.0))
    assert measured == pytest.approx(error, rel=5e-4)


@pytest.mark.parametrize("e", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_twobody_model_solves_hull_problems_to_nine_digits(e):
    # The chaser flies D(e) scaled to a = 70000 km, so perigee stays above the surface; the
    # target circles at the same a.
    a = 7e7
    speed = math.sqrt(EARTH.mu / a)
    m = TwoBodyDynamics(altitude=a - EARTH.radius)
    m.reset(np.zeros(6))
    target = m.target_state()
    r = np.array([(1.0 - e) * a, 0.0, 0.0])
    v = np.array([0.0, math.sqrt((1.0 + e) / (1.0 - e)) * speed, 0.0])
    m.reset(inertial_to_hill(target[:3], target[3:], r - target[:3], v - target[3:]))
    m.advance(np.zeros(3), 20.0 * math.sqrt(a**3 / EARTH.mu))
    chaser = m.chaser_state()
    scaled = np.r_[chaser[:2] / a, chaser[3:5] / speed]
    assert np.linalg.norm(scaled - exact_state(e, 20.0)) < 1e-9
