import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import vbar  # noqa: F401  (registers the environments)
from vbar import ApproachEnv


def test_approach_is_registered_and_passes_the_gymnasium_checker():
    env = gym.make("vbar/Approach-v0")
    check_env(env.unwrapped, skip_render_check=True)
    assert isinstance(env.unwrapped, ApproachEnv)


def test_made_and_constructed_environments_give_the_same_episode():
    made, direct = gym.make("vbar/Approach-v0"), ApproachEnv()
    obs_a, _ = made.reset(seed=3)
    obs_b, _ = direct.reset(seed=3)
    assert np.array_equal(obs_a, obs_b)
    for action in ([0.2, -0.5, 0.1], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.3]):
        assert np.array_equal(made.step(action)[0], direct.step(action)[0])
