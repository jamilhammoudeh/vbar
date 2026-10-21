"""Hill-frame relative acceleration without thrust, checked against the two-body integrator."""

import numpy as np
import pytest

from vbar.dynamics import EARTH, cw_system
from vbar.dynamics.models import TwoBodyDynamics, relative_acceleration

ZERO = np.zeros(3)
INCLINATION = np.radians(51.6)


def random_states(rng, count, rho_max, speed_max):
    """Relative states with |rho| <= rho_max (m) and |v| <= speed_max (m/s)."""
    states = rng.normal(size=(count, 6))
    states[:, :3] *= (
        rho_max * rng.uniform(0.1, 1.0, count) / np.linalg.norm(states[:, :3], axis=1)
    )[:, None]
    states[:, 3:] *= (
        speed_max * rng.uniform(0.1, 1.0, count) / np.linalg.norm(states[:, 3:], axis=1)
    )[:, None]
    return states


def central_difference_acceleration(model, rel, h):
    """Hill velocity change over a coast of +-h s, divided by 2h, as the integrator sees it."""
    model.reset(rel)
    forward, _ = model.advance(ZERO, h)
    model.reset(rel)
    backward, _ = model.advance(ZERO, -h)
    return (forward[3:] - backward[3:]) / (2.0 * h)


@pytest.mark.parametrize("seed", range(20))
def test_relative_acceleration_matches_the_j2_integrator(seed):
    rel = random_states(np.random.default_rng(seed), 1, 1000.0, 1.0)[0]
    model = TwoBodyDynamics(inclination=INCLINATION, j2=True)
    model.reset(rel)
    target = model.target_state()
    f = relative_acceleration(rel, target, EARTH, j2=True)
    # The central difference over 1 ms leaves an O(h^2) term near 1e-10 of the acceleration; a
    # forward difference would leave 1e-6. The frame rate's derivative under J2 contributes
    # about 1e-3 of the acceleration at 1 km, so leaving it out fails this bound.
    fd = central_difference_acceleration(model, rel, 1e-3)
    assert np.linalg.norm(fd - f) < 1e-6 * np.linalg.norm(f)


def test_relative_acceleration_differs_from_cw_by_the_second_order_gravity_remainder():
    model = TwoBodyDynamics(inclination=INCLINATION, j2=False)
    model.reset(np.zeros(6))
    target = model.target_state()
    a_cw, _ = cw_system(model.n)
    r = np.linalg.norm(target[:3])
    rho = 1000.0
    # Second-order remainder of differential point-mass gravity at range rho, every direction.
    bound = 3.0 * EARTH.mu * rho**2 / (r - rho) ** 4
    rng = np.random.default_rng(3)
    directions = rng.normal(size=(50, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    ratios = []
    for direction in directions:
        rel = np.concatenate([rho * direction, ZERO])
        f = relative_acceleration(rel, target, EARTH)
        ratios.append(np.linalg.norm(f - (a_cw @ rel)[3:]) / bound)
    assert max(ratios) <= 1.0
    # The bound is tight: inward and outward radial offsets come close to it.
    assert max(ratios) > 0.1


def test_relative_acceleration_accepts_stacked_states():
    rng = np.random.default_rng(5)
    rels = random_states(rng, 6, 2000.0, 1.0).reshape(2, 3, 6)
    model = TwoBodyDynamics(inclination=INCLINATION, j2=True)
    model.reset(np.zeros(6))
    target = model.target_state()
    stacked = relative_acceleration(rels, target, EARTH, j2=True)
    assert stacked.shape == (2, 3, 3)
    single = np.array(
        [[relative_acceleration(rel, target, EARTH, j2=True) for rel in row] for row in rels]
    )
    # Stacked and single evaluations take different summation orders in the frame products.
    assert np.allclose(stacked, single, rtol=1e-13, atol=0.0)


def test_relative_acceleration_accepts_stacked_target_states():
    rng = np.random.default_rng(6)
    rels = random_states(rng, 4, 500.0, 0.5)
    model = TwoBodyDynamics(inclination=INCLINATION, j2=True)
    model.reset(np.zeros(6))
    targets = []
    for _ in range(4):
        model.advance(ZERO, 600.0)
        targets.append(model.target_state())
    targets = np.array(targets)
    stacked = relative_acceleration(rels, targets, EARTH, j2=True)
    assert stacked.shape == (4, 3)
    single = np.array(
        [
            relative_acceleration(rel, target, EARTH, j2=True)
            for rel, target in zip(rels, targets, strict=True)
        ]
    )
    assert np.allclose(stacked, single, rtol=1e-13, atol=0.0)
