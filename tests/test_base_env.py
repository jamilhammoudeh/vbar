import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vbar.agents import DISC27
from vbar.constraints import KeepOut
from vbar.conventions import G0
from vbar.dynamics import cw_discrete
from vbar.scenarios import ApproachConfig, ApproachEnv, BaseEnv, EnvConfig
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


@pytest.mark.parametrize("sensor", [GaussianNav, GaussMarkovNav])
def test_environments_from_one_config_keep_their_own_sensor_state(sensor):
    actions = np.random.default_rng(3).uniform(-1.0, 1.0, (8, 3))
    config = EnvConfig(max_steps=20, sensor=sensor())
    alone, _ = fly(Coast(config), 1, actions)
    first, second = Coast(config), Coast(config)
    observed = [first.reset(seed=1)[0]]
    second.reset(seed=2)
    for a in actions:
        observed.append(first.step(a)[0])
        second.step(a)
    assert np.array_equal(np.array(observed), alone)


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


def test_step_record_keeps_its_own_copy_of_the_command():
    env = Coast()
    env.reset(seed=0)
    action = np.array([0.5, 0.0, 0.0])
    env.step(action)
    action[0] = -1.0  # an agent reusing its output buffer
    assert env.record.command[0] == 0.5 and env.record.filtered[0] == 0.5


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


def test_configs_take_keyword_arguments_only():
    # A positional value would land on whichever field is first, which moved when fields were
    # added; keywords keep old call sites from silently setting the wrong parameter.
    with pytest.raises(TypeError):
        EnvConfig(400e3)
    with pytest.raises(TypeError):
        ApproachConfig(400e3)


@pytest.mark.parametrize("dt", [0.0, -10.0])
def test_non_positive_step_is_rejected(dt):
    with pytest.raises(ValueError, match="dt"):
        Coast(EnvConfig(dt=dt))


def test_violation_across_a_segment_boundary_is_one_interval_per_step():
    # A 10 m/s pass through the 50 m sphere, with a 7 s delay splitting the first step at 7 s.
    start = (3.0, 100.0, -2.0, 0.0, -10.0, 0.0)
    env = ApproachEnv(ApproachConfig(start_state=start, delay_s=7.0))
    env.reset(seed=0)
    env.step(np.zeros(3))
    assert len(env.record.segment.segments) == 2
    ((entry, first_end),) = env.record.segment.violations["keep_out"]
    env.step(np.zeros(3))
    ((second_start, exit_),) = env.record.segment.violations["keep_out"]
    assert first_end == 10.0 and second_start == 10.0
    ts = np.linspace(0.0, 20.0, 2001)
    phi, _ = cw_discrete(env.n, ts)
    inside = np.flatnonzero(np.linalg.norm((phi @ np.array(start))[:, :3], axis=1) < 50.0)
    # Each crossing lies within the 0.01 s grid cell where the dense samples change sign.
    assert ts[inside[0] - 1] <= entry <= ts[inside[0]]
    assert ts[inside[-1]] <= exit_ <= ts[inside[-1] + 1]


def mixed_units_env(action_mode, drive, delay, layout="box", min_impulse=0.0):
    """Approach with execution error, a delay and a mass that falls fast enough to see."""
    cfg = ApproachConfig(
        max_steps=14,
        action_mode=action_mode,
        drive=drive,
        delay_s=delay,
        isp=220.0,
        mass_kg=500.0,
        propellant_kg=100.0,
        thrust_limit=layout,
        min_impulse=min_impulse,
        execution=ExecutionError(scale_sigma=0.2, noise_sigma=0.05),
        sensor=GaussianNav(5.0, 0.05),
        observation_channels=("pending_commands",),
    )
    return ApproachEnv(cfg)


def action_bound(view):
    """Action-space bound from the agent's view: the action scale at the initial mass."""
    return {"normalized": 1.0, "accel": view.max_accel}.get(
        view.action_mode, view.max_accel * view.mass_kg
    )


def public_echo(command, bound, layout):
    """The command limited by the thruster layout at the action-space bound."""
    u = np.asarray(command, dtype=float) / bound
    if layout == "box":
        u = np.clip(u, -1.0, 1.0)
    elif np.linalg.norm(u) > 1.0:
        u = u / np.linalg.norm(u)
    return bound * u


@pytest.mark.parametrize("drive", ["accel", "thrust"])
@pytest.mark.parametrize("action_mode", ["normalized", "accel", "thrust"])
def test_pending_commands_echo_the_issued_command_in_every_mode_and_drive(action_mode, drive):
    # A 25 s delay over 10 s steps keeps three slots: the command in effect and two queued.
    env = mixed_units_env(action_mode, drive, delay=25.0)
    env.reset(seed=3)
    commands = env.action_space.high * np.random.default_rng(1).uniform(-0.5, 0.5, (12, 3))
    commands[2::3] = 0.0
    for k, command in enumerate(commands):
        obs, _, _, _, _ = env.step(command)
        slots = obs[6:].reshape(3, 4)
        for m in range(min(k + 1, 3)):
            issued, echoed = commands[k - m], slots[-1 - m, 1:]
            moving = issued != 0.0
            assert np.all(echoed[~moving] == 0.0)
            assert np.allclose(echoed[moving] / issued[moving], 1.0, rtol=0.0, atol=1e-12)


def test_rejected_step_leaves_pending_commands_echo_aligned():
    env = mixed_units_env("accel", "thrust", delay=25.0)
    env.reset(seed=3)
    commands = env.action_space.high * np.random.default_rng(1).uniform(-0.5, 0.5, (5, 3))
    for command in commands[:3]:
        env.step(command)
    with pytest.raises(ValueError):
        env.step(commands[3], latency_s=-1.0)
    obs, _, _, _, _ = env.step(commands[4])
    slots = obs[6:].reshape(3, 4)
    issued = commands[[1, 2, 4]]
    # A misaligned slot echoes its command times one step's mass ratio, about 1 + 3e-5.
    assert np.allclose(slots[:, 1:] / issued, 1.0, rtol=0.0, atol=1e-12)


class PendingMassCheat:
    """Reads the mass history, and so each burn's delivered delta-v, from the echoed commands."""

    def __init__(self, mass, exponent, exhaust_speed):
        self.initial_mass = self.mass = mass
        self.exponent = exponent
        self.exhaust_speed = exhaust_speed

    def saturated_dv(self, echoed, bound):
        """Delivered delta-v (m/s) since the last call, from a saturated command's echo.

        A clipped command echoes the action scale at issue; over the bound, the scale at the
        initial mass, it gives the mass at issue.
        """
        ratio = float(np.max(np.abs(echoed))) / bound
        mass = self.initial_mass * ratio**self.exponent
        dv = self.exhaust_speed * np.log(self.mass / mass)
        self.mass = mass
        return dv

    def delivered_dv(self, issued, echoed):
        """Delivered delta-v 1-norm (m/s) over the last step, from the echo/issue ratio."""
        ratio = float(np.mean(echoed / issued))
        mass = self.mass * ratio**self.exponent
        dv = self.exhaust_speed * np.log(self.mass / mass)
        self.mass = mass
        return dv


@pytest.mark.parametrize(
    ("action_mode", "drive", "exponent"), [("accel", "thrust", -1), ("thrust", "accel", 1)]
)
def test_pending_commands_reveal_no_delivered_delta_v(action_mode, drive, exponent):
    env = mixed_units_env(action_mode, drive, delay=10.0)
    _, info = env.reset(seed=3)
    cheat = PendingMassCheat(500.0, exponent, 220.0 * G0)
    commands = env.action_space.high * np.random.default_rng(1).uniform(-0.5, 0.5, (12, 3))
    errors, dv_before = [], info["dv_axes_mps"]
    for command in commands:
        obs, _, _, _, info = env.step(command)
        dv_true = info["dv_axes_mps"] - dv_before
        dv_before = info["dv_axes_mps"]
        guess = cheat.delivered_dv(command, obs[7:10])
        if dv_true > 0.0:
            errors.append(abs(guess - dv_true) / dv_true)
    # Taking the commanded delta-v as delivered is off by the 20 % scale error; 0.01 is an
    # order of magnitude below that and far above the rounding a working cheat reaches.
    assert np.median(errors) > 0.01


@pytest.mark.parametrize("layout", ["box", "ball"])
@pytest.mark.parametrize("drive", ["accel", "thrust"])
@pytest.mark.parametrize("action_mode", ["normalized", "accel", "thrust"])
def test_pending_commands_echo_saturated_commands_without_the_mass(action_mode, drive, layout):
    """Each slot echoes the issued command limited by the layout at the action-space bound.

    The bound is the action scale at the initial mass (1, max_accel or max_accel * mass_kg),
    so the agent computes the echo from its own command and its view. Where the true scale is
    fixed this is the command as limited; where it moves with the mass, the true clip is the
    scale at issue and would give the mass away.
    """
    env = mixed_units_env(action_mode, drive, delay=25.0, layout=layout)
    env.reset(seed=3)
    bound = action_bound(env.agent_view())
    rng = np.random.default_rng(1)
    # Axes from half the bound to a thousand times it, so most commands clip and some do not.
    commands = bound * rng.choice([-1.0, 1.0], (12, 3)) * 10.0 ** rng.uniform(-0.3, 3.0, (12, 3))
    for k, command in enumerate(commands):
        obs, _, _, _, _ = env.step(command)
        slots = obs[6:].reshape(3, 4)
        for m in range(min(k + 1, 3)):
            expected = public_echo(commands[k - m], bound, layout)
            assert np.allclose(slots[-1 - m, 1:], expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("layout", ["box", "ball"])
@pytest.mark.parametrize(
    ("action_mode", "drive", "exponent"), [("accel", "thrust", -1), ("thrust", "accel", 1)]
)
def test_saturated_pending_commands_reveal_no_delivered_delta_v(
    action_mode, drive, exponent, layout
):
    env = mixed_units_env(action_mode, drive, delay=10.0, layout=layout)
    _, info = env.reset(seed=3)
    cheat = PendingMassCheat(500.0, exponent, 220.0 * G0)
    bound = action_bound(env.agent_view())
    # The rocket equation gives the delta-v on the norm the layout spends propellant on.
    key = {"box": "dv_axes_mps", "ball": "dv_used_mps"}[layout]
    guesses, dv_true, dv_before = [], [], info[key]
    for sign in np.random.default_rng(1).choice([-1.0, 1.0], 12):
        obs, _, _, _, info = env.step([1e3 * sign * bound, 0.0, 0.0])
        dv_true.append(info[key] - dv_before)
        dv_before = info[key]
        guesses.append(cheat.saturated_dv(obs[7:10], bound))
    # The echo after step k gives the mass at its start, so the guess after step k + 1 is the
    # delta-v delivered in step k.
    pairs = zip(guesses[1:], dv_true[:-1], strict=True)
    errors = [abs(guess - dv) / dv for guess, dv in pairs if dv > 0.0]
    assert len(errors) >= 9
    # The 0.01 margin is set as in the unsaturated cheat test above.
    assert np.median(errors) > 0.01


def thrust_per_command(view, drive, mass):
    """Thrust (N) of a command of one action unit, from the action mode's units."""
    if view.action_mode == "accel":
        return mass
    if view.action_mode == "thrust":
        return 1.0
    return view.max_accel * (mass if drive == "accel" else view.mass_kg)


@pytest.mark.parametrize("drive", ["accel", "thrust"])
@pytest.mark.parametrize("action_mode", ["normalized", "accel", "thrust"])
def test_impulse_bit_probe_reveals_nothing_in_pending_commands(action_mode, drive):
    # Two episodes take the same commands; their execution draws differ, and so do their
    # masses. Each probe asks for the 1 N s impulse bit at the geometric mean of the two masses'
    # thrusts, so a drop decided on the true mass keeps it in one episode and drops it in the
    # other. Echoes that depend only on the commands and the public limits agree.
    envs = [mixed_units_env(action_mode, drive, delay=10.0, min_impulse=1.0) for _ in range(2)]
    infos = [env.reset(seed=seed)[1] for env, seed in zip(envs, (3, 4), strict=True)]
    view = envs[0].agent_view()
    burn = 0.9 * action_bound(view) * np.ones(3)
    for k in range(12):
        if k < 5:
            command = burn
        else:
            thrusts = [thrust_per_command(view, drive, info["mass_kg"]) for info in infos]
            command = np.array([1.0 / (view.dt * np.sqrt(np.prod(thrusts))), 0.0, 0.0])
        steps = [env.step(command) for env in envs]
        infos = [info for *_, info in steps]
        echoes = [obs[7:10] for obs, *_ in steps]
        assert np.allclose(echoes[0], echoes[1], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("layout", ["box", "ball"])
@pytest.mark.parametrize("drive", ["accel", "thrust"])
def test_disc27_pending_commands_echo_the_command_of_the_issued_index(drive, layout):
    # A full command held for a step is at most a 100 N s impulse, so a 1000 N s impulse bit
    # drops every burn; the echo still names the command the agent picked.
    env = mixed_units_env("disc27", drive, delay=25.0, layout=layout, min_impulse=1e3)
    env.reset(seed=3)
    indices = np.random.default_rng(2).integers(0, len(DISC27), 12)
    for k, index in enumerate(indices):
        obs, _, _, _, _ = env.step(index)
        slots = obs[6:].reshape(3, 4)
        for m in range(min(k + 1, 3)):
            expected = public_echo(DISC27[indices[k - m]], 1.0, layout)
            assert np.allclose(slots[-1 - m, 1:], expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("layout", ["box", "ball"])
@pytest.mark.parametrize("drive", ["accel", "thrust"])
def test_impulse_coast_pending_commands_echo_the_clipped_impulse_and_zero_coasts(drive, layout):
    # Every other impulse is 0.05 of a full command per axis, at most 9 N s, which the 10 N s
    # impulse bit drops; the others are over the bound on every axis and are clipped.
    env = mixed_units_env("impulse_coast", drive, delay=25.0, layout=layout, min_impulse=10.0)
    env.reset(seed=3)
    rng = np.random.default_rng(4)
    # The zero in effect before the first command, then each echo in the order issued.
    acting = [np.zeros(3)]
    done, k = False, 0
    while not done:
        size = 0.05 if k % 2 else 10.0 ** rng.uniform(0.0, 2.0, 3)
        impulse = size * rng.choice([-1.0, 1.0], 3)
        coast = (0.0, 10.0, 20.0)[k % 3]
        obs, _, terminated, truncated, _ = env.step(np.append(impulse, coast))
        done, k = terminated or truncated, k + 1
        acting.append(public_echo(impulse, 1.0, layout))
        if coast and not done:
            acting.append(np.zeros(3))
        slots = obs[6:].reshape(3, 4)
        shown = min(int(np.sum(slots[:, 0] > 0.0)) + 1, 3)
        expected = np.array(acting[-shown:])
        assert np.allclose(slots[-shown:, 1:], expected, rtol=1e-12, atol=0.0)
    assert k >= 5
