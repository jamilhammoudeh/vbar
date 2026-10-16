import math

import numpy as np
import pytest

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
