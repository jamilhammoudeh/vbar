"""V-BAR's two-body plus J2 rung agrees with Basilisk and Orekit within stated tolerances.

Each test needs its optional extra and is skipped without it. The tolerances follow from the
integrator settings, fixed before any run: every tool integrates each spacecraft's absolute
state, about 6.9e6 m from the Earth's centre, to a relative accuracy near 1e-12 per orbit, which
is about 1e-5 m; the relative state is a difference of two such states, and V-BAR's own relative
integration is far tighter, so a coast agrees to 1e-4 m per orbit flown. A burn replay adds the
backends' inertial direction held over each substep, second order in the angle the Hill frame
turns in it: halving the substep must divide that residual by more than three.
"""

import numpy as np
import pytest

from vbar.agents import LQRAgent
from vbar.dynamics import EARTH
from vbar.dynamics.backends import Burn, hill_states
from vbar.dynamics.backends.agreement import episode_burns, vbar_arc
from vbar.metrics import record_episode
from vbar.scenarios import make, seeds

REL = np.array([100.0, -1000.0, 50.0, 0.0, 0.0, 0.1])
BURNS = [Burn(0.0, 100.0, (0.001, -0.002, 0.0005)), Burn(300.0, 50.0, (-0.001, 0.0, 0.0))]


def burn_case(name):
    """Start state (m, m/s), burns and span (s) of a replay."""
    if name == "two burns":
        return REL, BURNS, 600.0
    seed, agent_seed = seeds("approach-vbar", "validation", 1).pairs()[0]
    recorded = record_episode(make("approach-vbar"), LQRAgent(), seed=seed, agent_seed=agent_seed)
    return recorded.start, episode_burns(recorded.records), recorded.duration


def backend(name):
    if name == "basilisk":
        pytest.importorskip("Basilisk")
        from vbar.dynamics.backends import basilisk as module
    else:
        pytest.importorskip("orekit_jpype")
        from vbar.dynamics.backends import orekit as module
    return module


@pytest.mark.parametrize("name", ["basilisk", "orekit"])
def test_a_one_orbit_coast_agrees_to_a_tenth_of_a_millimetre(name):
    module = backend(name)
    times = np.arange(0.0, 5601.0, 100.0)
    ours, start_t, start_c = vbar_arc(REL, times)
    kwargs = {"substep": 10.0} if name == "basilisk" else {}
    target, chaser = module.propagate(start_t, start_c, times, EARTH, True, **kwargs)
    theirs = hill_states(target, chaser, EARTH, True)
    assert np.max(np.linalg.norm(theirs[:, :3] - ours[:, :3], axis=1)) < 1e-4
    assert np.max(np.linalg.norm(theirs[:, 3:] - ours[:, 3:], axis=1)) < 1e-7


@pytest.mark.parametrize("case", ["two burns", "lqr episode"])
@pytest.mark.parametrize("name", ["basilisk", "orekit"])
def test_a_burn_replay_agrees_and_its_residual_is_the_held_direction(name, case):
    module = backend(name)
    rel, burns, span = burn_case(case)
    times = np.arange(0.0, span + 0.5, 10.0)
    ours, start_t, start_c = vbar_arc(rel, times, burns)
    found = []
    for substep in (2.0, 1.0):
        target, chaser = module.propagate(
            start_t, start_c, times, EARTH, True, burns=burns, substep=substep
        )
        theirs = hill_states(target, chaser, EARTH, True)
        found.append(np.max(np.linalg.norm(theirs[:, :3] - ours[:, :3], axis=1)))
    assert found[1] < 1e-3
    # Second order in the substep gives four; over the long LQR episode the factor is lower
    # (3.2 at 2 s to 1 s), so the bound is three.
    assert found[0] / found[1] > 3.0


def test_samples_that_start_after_zero_match_the_slice_of_a_run_from_zero():
    module = backend("basilisk")
    start = np.concatenate([[6.878e6, 0.0, 0.0], [0.0, 7612.6, 0.0]])
    full = module.propagate(start, start, [0.0, 100.0, 200.0], EARTH, True)[0]
    late = module.propagate(start, start, [100.0, 200.0], EARTH, True)[0]
    np.testing.assert_allclose(late, full[1:], rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("name", ["basilisk", "orekit"])
def test_a_burn_edge_inside_a_substep_is_refused(name):
    module = backend(name)
    start = np.array([6.878e6, 0.0, 0.0, 0.0, 7612.6, 0.0])
    with pytest.raises(ValueError, match="whole numbers of substeps"):
        module.propagate(
            start, start, [0.0, 4.0], EARTH, True, burns=[Burn(0.5, 3.0, (0.0, 1e-3, 0.0))]
        )
