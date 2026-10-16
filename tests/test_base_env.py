import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vbar.constraints import KeepOut
from vbar.dynamics import cw_discrete
from vbar.scenarios import BaseEnv, EnvConfig
from vbar.spacecraft import ExecutionError, GaussianNav, GaussMarkovNav, RangeScaledNav

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


@pytest.mark.parametrize("sensor", [GaussianNav, RangeScaledNav, GaussMarkovNav])
def test_base_environment_follows_the_gymnasium_api(sensor):
    check_env(Coast(EnvConfig(sensor=sensor())), skip_render_check=True)


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


def test_thrust_actions_are_newtons_against_the_thrust_of_a_full_command():
    actions = np.random.default_rng(6).uniform(-1.0, 1.0, (5, 3))
    base = dict(drive="thrust", propellant_kg=400.0)
    _, normalized = fly(Coast(EnvConfig(**base)), 4, actions)
    _, newtons = fly(Coast(EnvConfig(action_mode="thrust", **base)), 4, 10.0 * actions)
    assert np.allclose(normalized, newtons, rtol=1e-14, atol=1e-12)
    assert Coast(EnvConfig(action_mode="thrust")).action_space.high[0] == pytest.approx(10.0)


def test_execution_error_draws_come_from_their_own_stream():
    actions = np.random.default_rng(7).uniform(-1.0, 1.0, (10, 3))
    error = ExecutionError(scale_bias=0.02, scale_sigma=0.05, noise_sigma=0.01)
    noisy = dict(sensor=GaussianNav(), execution=error)
    obs_a, truth_a = fly(Coast(EnvConfig(**noisy)), 9, actions)
    obs_b, truth_b = fly(Coast(EnvConfig(**noisy)), 9, actions)
    obs_c, truth_c = fly(Coast(EnvConfig(sensor=GaussianNav())), 9, actions)
    assert np.array_equal(truth_a, truth_b) and np.array_equal(obs_a, obs_b)
    assert np.array_equal(truth_a[0], truth_c[0]) and not np.array_equal(truth_a[1:], truth_c[1:])
    # The sensor's draws do not depend on whether the thrusters have errors.
    assert np.allclose(obs_a - truth_a, obs_c - truth_c, rtol=0.0, atol=1e-12)


def test_observation_channels_are_off_by_default():
    env = Coast()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6,) and env.agent_view().observation == ("x", "y", "z", "vx", "vy", "vz")


def test_time_and_propellant_channels_report_what_is_left():
    channels = ("t_remaining", "propellant_fraction")
    env = Coast(EnvConfig(max_steps=20, propellant_kg=10.0, observation_channels=channels))
    obs, _ = env.reset(seed=0)
    assert env.agent_view().observation[6:] == channels
    assert obs[6:] == pytest.approx([200.0, 1.0])
    for k in range(1, 4):
        obs, _, _, _, _ = env.step([1.0, -1.0, 0.5])
        assert obs[6] == pytest.approx(200.0 - 10.0 * k)
        assert obs[7] == pytest.approx(env.spacecraft.propellant / 10.0, rel=1e-15)
    assert obs[7] < 1.0


@pytest.mark.parametrize("delay", [3.7, 25.0])
def test_pending_commands_make_the_delayed_state_markov(delay):
    # The next true state follows from the current one, the pending-command channel and the
    # new action, with no other memory of past actions.
    env = Coast(EnvConfig(delay_s=delay, observation_channels=("pending_commands",)))
    slots = int(np.ceil(delay / 10.0))
    assert env.observation_space.shape == (6 + 4 * slots,)
    obs, info = env.reset(seed=0)
    for action in np.random.default_rng(8).uniform(-1.0, 1.0, (8, 3)):
        acting = [tuple(row) for row in obs[6:].reshape(slots, 4)] + [(delay, *action)]
        expected = info["true_state"]
        for k, (start, *command) in enumerate(acting):
            end = min(acting[k + 1][0], 10.0) if k + 1 < len(acting) else 10.0
            if end > start:
                phi, gamma = cw_discrete(env.n, end - start)
                expected = phi @ expected + gamma @ (0.02 * np.array(command))
        obs, _, _, _, info = env.step(action)
        assert np.allclose(info["true_state"], expected, rtol=0.0, atol=1e-9)


def test_environment_with_every_channel_follows_the_gymnasium_api():
    cfg = EnvConfig(
        delay_s=25.0,
        observation_channels=("t_remaining", "propellant_fraction", "pending_commands"),
    )
    check_env(Coast(cfg), skip_render_check=True)
    with pytest.raises(ValueError, match="observation channels"):
        Coast(EnvConfig(observation_channels=("range",)))
    # Without a delay nothing is pending, so the channel is empty.
    env = Coast(EnvConfig(observation_channels=("pending_commands",)))
    env.reset(seed=0)
    obs, _, _, _, _ = env.step([1.0, 0.0, 0.0])
    assert obs.shape == (6,)


def test_every_thrust_change_is_a_boundary_for_the_integrator():
    # Delay 3.7 s and constant thrust that runs out partway through: each two-body integration
    # holds one acceleration and starts at a switch time.
    cfg = EnvConfig(
        dynamics="twobody_j2", inclination=0.9, delay_s=3.7, drive="thrust", propellant_kg=0.3
    )
    env = Coast(cfg)
    env.reset(seed=0)
    calls = []
    advance = env.dynamics.advance

    def logged(accel, duration, n_samples=0):
        calls.append((duration, np.array(accel)))
        return advance(accel, duration, n_samples)

    env.dynamics.advance = logged
    traced, run_out = [], None
    for k in range(12):
        env.step([1.0, -1.0, 0.5] if k % 2 else [-0.5, 1.0, 1.0])
        traced += [(d, a) for _, d, a in env.record.segment.segments]
        if env.record.actuation.empty and run_out is None:
            run_out = env.record.segment.segments[-1][0]
    assert [d for d, _ in calls] == [d for d, _ in traced]
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(calls, traced, strict=True))
    starts = np.cumsum([0.0] + [d for d, _ in calls])
    for t in [10.0 * k + 3.7 for k in range(12)]:
        assert np.min(np.abs(starts - t)) < 1e-9
    # The run-out falls inside a burn, away from every step and delay switch.
    switches = np.r_[10.0 * np.arange(13), 10.0 * np.arange(12) + 3.7]
    assert run_out is not None and np.min(np.abs(run_out - switches)) > 1e-3
    after = [a for (_, a), t in zip(calls, starts, strict=False) if t >= run_out - 1e-9]
    assert all(not np.any(a) for a in after)
