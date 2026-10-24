"""The default LQR baseline completes the 600 s R-bar hold within the scenario's episode.

Approach-rbar gives 360 steps of 10 s from starts up to 541 m from the hold point, and success
needs 600 s without a break inside 5 m and 0.05 m/s of it. A design whose slowest closed-loop
mode decays by e only every 1000 s, as the scales (10 m, 0.01 m/s) give, is still 6 to 10 m out
when the episode ends.
"""

import itertools
import math

import numpy as np
import pytest

from vbar.agents import LQRAgent
from vbar.agents.lqr import LQRConfig, regulator_model
from vbar.metrics import record_episode
from vbar.metrics.mission import completion_time, hard_violated, success
from vbar.scenarios import make
from vbar.scenarios.catalog import APPROACH_RBAR
from vbar.scenarios.seeds import SeedSplit

SID = "vbar:approach-rbar:cw:box:dv:v1"
SEEDS = (0, 1, 2)


def farthest_start_m(view):
    """Largest distance (m) from the hold point over the corners of the start-position box."""
    box = APPROACH_RBAR.dispersions["start_pos"]
    hold = np.asarray(view.goal["hold_point"], dtype=float)
    corners = np.array(list(itertools.product(*zip(box.low, box.high, strict=True))))
    return float(np.max(np.linalg.norm(corners - hold, axis=1)))


def test_default_lqr_slowest_mode_leaves_time_for_the_approach_and_the_dwell():
    view = make(SID).agent_view()
    model = regulator_model(view, LQRConfig())
    radius = np.max(np.abs(np.linalg.eigvals(model.phi - model.gamma @ model.k)))
    tau = -view.dt / math.log(radius)
    goal = view.goal
    # Decay of the slowest mode alone from the farthest start to the capture radius, then the
    # dwell; transients of the faster modes only add to this.
    needed = tau * math.log(farthest_start_m(view) / goal["capture_radius"]) + goal["hold_time"]
    assert needed <= view.max_steps * view.dt


def flight(index, **overrides):
    split = SeedSplit(SID, "validation", len(SEEDS))
    return record_episode(
        make(SID, **overrides),
        LQRAgent(),
        seed=split.seed(index),
        agent_seed=split.agent_seed(index),
    )


def test_default_lqr_completes_the_approach_rbar_hold():
    # Two samples per step instead of 20 for the soft speed limits; the goal, the dwell, the
    # 360 steps of 10 s and the keep-out check at every step boundary are unchanged.
    ep = flight(0, violation_samples=2)
    assert completion_time(ep) is not None
    assert not hard_violated(ep)
    assert success(ep)


@pytest.mark.slow
@pytest.mark.parametrize("index", SEEDS, ids=["seed0", "seed1", "seed2"])
def test_default_lqr_completes_the_approach_rbar_hold_from_each_validation_seed(index):
    assert success(flight(index))
