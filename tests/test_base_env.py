import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vbar.constraints import KeepOut
from vbar.dynamics import cw_discrete
from vbar.scenarios import BaseEnv, EnvConfig
from vbar.spacecraft import GaussianNav

# Two-sided 1e-6 normal quantile: each statistical check fails a correct model with that
# probability; seeds are fixed, so the tests are deterministic.
Z = 4.89


class Coast(BaseEnv):
    """Minimal scenario: start near a point on the V-bar, keep out of 50 m."""

    scenario_name = "coast-test"

    def __init__(self, config=None, start=(0.0, 200.0, 0.0, 0.0, 0.0, 0.0)):
        self.start = np.array(start, dtype=float)
        super().__init__(config or EnvConfig(max_steps=20))

    def _sample_start(self, rng):
        return self.start + rng.uniform(-1.0, 1.0, 6) * np.repeat([10.0, 0.01], 3)

    def _constraints(self):
        return (KeepOut(50.0),)

    def _goal(self):
        return {"hold_point": (0.0, -100.0, 0.0)}


def fly(env, seed, actions):
    obs, info = env.reset(seed=seed)
    observed, truth = [obs], [info["true_state"]]
    for a in actions:
        obs, _, _, _, info = env.step(a)
        observed.append(obs)
        truth.append(info["true_state"])
    return np.array(observed), np.array(truth)


def test_base_environment_follows_the_gymnasium_api():
    check_env(Coast(EnvConfig(sensor=GaussianNav())), skip_render_check=True)


def test_same_seed_gives_the_same_episode_including_sensor_noise():
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, (10, 3))
    env = Coast(EnvConfig(sensor=GaussianNav()))
    first, second, other = fly(env, 7, actions), fly(env, 7, actions), fly(env, 8, actions)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert not np.array_equal(first[0], other[0])


def test_sensor_noise_does_not_change_the_true_trajectory():
    actions = np.random.default_rng(1).uniform(-1.0, 1.0, (10, 3))
    _, quiet = fly(Coast(EnvConfig(sensor=GaussianNav(1e-3, 1e-6))), 3, actions)
    _, noisy = fly(Coast(EnvConfig(sensor=GaussianNav(5.0, 5e-3))), 3, actions)
    assert np.array_equal(quiet, noisy)


def test_delayed_command_splits_the_step_and_propagates_each_segment():
    env = Coast(EnvConfig(delay_s=3.7), start=(0.0, 300.0, 0.0, 0.0, 0.0, 0.0))
    env.reset(seed=0)
    start = env.state.copy()
    env.step([1.0, 0.0, 0.0])
    _, _, _, _, info = env.step([0.0, -1.0, 0.0])
    trace = env.record.segment
    assert [round(d, 12) for _, d, _ in trace.segments] == [3.7, 6.3]
    assert trace.segments[1][0] == pytest.approx(13.7, abs=1e-12)
    expected = start
    pieces = [(3.7, [0, 0, 0]), (6.3, [0.02, 0, 0]), (3.7, [0.02, 0, 0]), (6.3, [0, -0.02, 0])]
    for duration, accel in pieces:
        phi, gamma = cw_discrete(env.n, duration)
        expected = phi @ expected + gamma @ np.array(accel, dtype=float)
    assert np.allclose(info["true_state"], expected, rtol=0.0, atol=1e-9)


def test_accel_actions_are_the_normalized_action_times_the_limit():
    actions = np.random.default_rng(2).uniform(-1.0, 1.0, (5, 3))
    _, normalized = fly(Coast(), 4, actions)
    _, accel = fly(Coast(EnvConfig(action_mode="accel")), 4, 0.02 * actions)
    assert np.allclose(normalized, accel, rtol=1e-14, atol=1e-12)


def test_minimum_margin_is_tracked_in_continuous_time():
    env = Coast(EnvConfig(), start=(0.0, 60.0, 5.0, 0.0, -12.0, 0.0))
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(3))
    found = env.record.segment.minima["keep_out"]
    assert info["min_margins"]["keep_out"] == found.margin < -40.0
    assert 0.0 < found.t < 10.0


def test_agent_view_is_read_only_and_describes_the_task():
    view = Coast().agent_view()
    assert view.scenario_id == "vbar:coast-test:cw:box:dv:v0"
    assert view.constraints == ({"type": "keep_out", "radius": 50.0},)
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.dt = 1.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        view.goal["hold_point"] = (0.0, 0.0, 0.0)  # type: ignore[index]


def test_observations_carry_no_true_state_information_when_noise_is_on():
    sigma = np.repeat([2.0, 2e-3], 3)
    env = Coast(EnvConfig(max_steps=5000, sensor=GaussianNav(2.0, 2e-3)))
    actions = np.random.default_rng(5).uniform(-1.0, 1.0, (4000, 3))
    observed, truth = fly(env, 11, actions)
    err = observed - truth
    count = len(err)
    # The error has the sensor's mean and covariance...
    assert np.all(np.abs(err.mean(axis=0)) < Z * sigma / np.sqrt(count))
    assert np.allclose(err.std(axis=0), sigma, rtol=Z * np.sqrt(0.5 / count))
    # ...and is uncorrelated with the true state and with the agent's past actions.
    corr = np.corrcoef(np.c_[err, truth], rowvar=False)[:6, 6:]
    assert np.all(np.abs(corr) < Z / np.sqrt(count))
    corr = np.corrcoef(np.c_[err[1:], actions], rowvar=False)[:6, 6:]
    assert np.all(np.abs(corr) < Z / np.sqrt(count))
    # Agents get only the time and the public description, none of the truth.
    assert set(env.agent_info()) == {"time_s"}
    view_fields = {f.name for f in dataclasses.fields(env.agent_view())}
    assert view_fields.isdisjoint({"state", "true_state", "record", "dynamics", "sensor"})
