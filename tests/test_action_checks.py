"""Malformed actions are refused before they change anything."""

import math

import numpy as np
import pytest

from vbar.safety import PassThroughFilter
from vbar.scenarios import ApproachConfig, ApproachEnv, ApproachVectorEnv, make
from vbar.training import FilteredEnv

BAD_SHAPES = [[], [0.0, 0.0], [0.0] * 5, [[0.0], [0.0], [0.0]], [[0.0] * 3] * 2]
BAD_VALUES = [[math.nan, 0.0, 0.0], [0.0, math.inf, 0.0], [None, 0.0, 0.0]]


def twin_states(delay_s, mode="normalized"):
    """Two environments on one seed, with one valid step flown in each."""
    envs = [ApproachEnv(ApproachConfig(delay_s=delay_s, action_mode=mode)) for _ in range(2)]
    for env in envs:
        env.reset(seed=3)
        env.step(np.array([0.2, -0.1, 0.05]) * (0.02 if mode == "accel" else 1.0))
    return envs


@pytest.mark.parametrize("delay_s", [0.0, 25.0])
@pytest.mark.parametrize("action", BAD_SHAPES + BAD_VALUES, ids=repr)
def test_a_refused_action_leaves_the_environment_as_it_was(action, delay_s):
    # Zeros of the wrong shape were flown as a coast, nonzero ones raised after the spacecraft had
    # queued the command and spent propellant. Now both raise first, so the next valid step
    # matches a twin that never saw the bad action.
    tried, clean = twin_states(delay_s)
    with pytest.raises(ValueError):
        tried.step(np.array([1.0] * 5) if action == [0.0] * 5 else action)
    with pytest.raises(ValueError):
        tried.step(action)
    assert tried.t == clean.t and tried.spacecraft.mass == clean.spacecraft.mass
    command = np.array([0.0, 0.3, -0.2])
    a = tried.step(command)
    b = clean.step(command)
    assert np.array_equal(a[0], b[0]) and repr(a[4]) == repr(b[4])


@pytest.mark.parametrize("mode", ["accel", "thrust"])
def test_physical_modes_refuse_malformed_actions_too(mode):
    env = ApproachEnv(ApproachConfig(action_mode=mode))
    env.reset(seed=1)
    for action in BAD_SHAPES + BAD_VALUES:
        with pytest.raises(ValueError):
            env.step(action)


def test_impulse_coast_takes_four_finite_numbers():
    env = make("approach-vbar", action_mode="impulse_coast")
    env.reset(seed=1)
    for action in ([0.0, 0.0, 0.0], [0.0] * 5, [0.0, 0.0, 0.0, math.nan]):
        with pytest.raises(ValueError):
            env.step(action)
    env.step([0.0, 0.0, 0.0, 20.0])


def test_catalogue_vector_and_filtered_environments_refuse_nan_actions():
    env = make("docking:axis")
    env.reset(seed=1)
    with pytest.raises(ValueError, match="finite"):
        env.step([math.nan, 0.0, 0.0])
    vector = ApproachVectorEnv(ApproachConfig(), 2)
    vector.reset(seed=1)
    with pytest.raises(ValueError, match="finite"):
        vector.step(np.array([[0.0, 0.0, 0.0], [math.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="shape"):
        vector.step(np.zeros(6))
    filtered = FilteredEnv(ApproachEnv(), PassThroughFilter())
    filtered.reset(seed=1)
    with pytest.raises(ValueError, match="finite"):
        filtered.step([0.0, math.nan, 0.0])
