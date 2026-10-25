"""The sampled drift clearance refines only the samples that can set its minimum.

The full computation solves every loose sample's nearest point again by bounded least squares
(`vbar.reach._nearest` on all samples at once); the clearance must not change by a bit when the
samples that cannot set the found or certified minimum skip that solve.
"""

import numpy as np
import pytest

from vbar import reach
from vbar.constraints import interval_bounds
from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.reach import Zonotope, drift_clearance

N = mean_motion(EARTH.radius + 500e3)
R = 50.0


def random_case(rng, fewest=1):
    """A zonotope of states drifting near the keep-out, its drift time (s) and sample count."""
    count = int(rng.integers(fewest, 9))
    center = np.concatenate([rng.uniform(-300.0, 300.0, 3), rng.uniform(-0.2, 0.2, 3)])
    scales = np.repeat([rng.uniform(0.1, 30.0), rng.uniform(1e-4, 2e-2)], 3)
    generators = rng.uniform(-1.0, 1.0, (6, count)) * scales[:, None]
    return Zonotope(center, generators), float(rng.uniform(500.0, 6000.0)), int(rng.integers(5, 80))


def full_clearance(z, duration, samples):
    """(found, certified, t) of `drift_clearance` with every sample refined."""
    times = np.linspace(0.0, duration, samples)
    phis, _ = cw_discrete(N, times)
    c, g = phis @ z.center, phis @ z.generators
    h = float(times[1] - times[0])
    lower, upper = reach._nearest(np.ascontiguousarray(c[:, :3]), np.ascontiguousarray(g[:, :3]))
    largest = np.zeros((samples, 6))
    largest[:, 0] = np.linalg.norm(c[:, :3], axis=1) + np.sum(np.linalg.norm(g[:, :3], axis=1), 1)
    largest[:, 3] = np.linalg.norm(c[:, 3:], axis=1) + np.sum(np.linalg.norm(g[:, 3:], axis=1), 1)
    speed, _ = interval_bounds(largest, h, 0.0, N)
    lower, upper = lower - R, upper - R
    k = int(np.argmin(upper))
    certified = float(np.min(0.5 * (lower[:-1] + lower[1:] - speed * h)))
    return float(upper[k]), certified, 0.0 + k * h


@pytest.fixture
def solves(monkeypatch):
    """Counts the bounded least-squares solves of the clearance, passing each one through."""
    calls = [0]
    solve = reach.lsq_linear

    def counted(*args, **kwargs):
        calls[0] += 1
        return solve(*args, **kwargs)

    monkeypatch.setattr(reach, "lsq_linear", counted)
    return calls


# A case takes about 60 ms; the default run takes 25 of the 1000.
CASES = (pytest.param(1000, marks=pytest.mark.slow), 25)


@pytest.mark.parametrize("cases", CASES)
def test_pruned_clearance_equals_full_clearance(cases):
    rng = np.random.default_rng(11)
    for _ in range(cases):
        z, duration, samples = random_case(rng)
        found = drift_clearance(N, z, R, duration, samples)
        assert (found.found, found.certified, found.t) == full_clearance(z, duration, samples)


@pytest.mark.parametrize("cases", CASES)
def test_pruned_clearance_solves_fewer_least_squares_problems(solves, cases):
    rng = np.random.default_rng(12)
    pruned = full = 0
    for _ in range(cases):
        # Four or more generators leave far more samples loose after the gradient search.
        z, duration, samples = random_case(rng, fewest=4)
        solves[0] = 0
        drift_clearance(N, z, R, duration, samples)
        case_pruned = solves[0]
        solves[0] = 0
        full_clearance(z, duration, samples)
        assert case_pruned <= solves[0]
        pruned += case_pruned
        full += solves[0]
    assert full > 0
    assert pruned < full
