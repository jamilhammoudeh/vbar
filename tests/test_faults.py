import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vbar.dynamics import cw_discrete
from vbar.faults import (
    After,
    AllOf,
    Always,
    InSphere,
    Intermittent,
    RangeBelow,
    RangeLoss,
    SensorBias,
    ThrusterOff,
    Until,
    Window,
)
from vbar.scenarios import (
    SCENARIOS,
    ApproachConfig,
    ApproachVectorEnv,
    BaseEnv,
    EnvConfig,
    make,
)
from vbar.spacecraft import GaussianNav

# Two-sided 1e-6 normal quantile: a correct model fails a frequency check with that
# probability; seeds are fixed, so the tests are deterministic.
Z = 4.89


class Still(BaseEnv):
    """Starts exactly at `start`, so known answers need no draw."""

    scenario_name = "fault-test"

    def __init__(self, config=None, start=(0.0, 200.0, 0.0, 0.0, 0.0, 0.0)):
        self.start = np.array(start, dtype=float)
        super().__init__(config or EnvConfig(max_steps=20))

    def _sample_start(self, rng):
        return self.start.copy()


def cw_step(env, x0, fraction):
    phi, gamma = cw_discrete(env.n, env.cfg.dt)
    return phi @ x0 + gamma @ (np.asarray(fraction) * env.cfg.max_accel)


def active_after_each_step(env, steps, seed=1):
    _, info = env.reset(seed=seed)
    flags = ["x" in info["faults_active"]]
    for _ in range(steps):
        *_, info = env.step(np.zeros(3))
        flags.append("x" in info["faults_active"])
    return flags


def test_failed_thruster_delivers_nothing_on_its_side():
    env = Still(EnvConfig(faults=(ThrusterOff(axis=1, direction=1),)))
    env.reset(seed=0)
    x0 = env.state.copy()
    _, _, _, _, info = env.step(np.array([0.0, 0.5, 0.0]))
    np.testing.assert_allclose(info["true_state"], cw_step(env, x0, [0.0, 0.0, 0.0]), rtol=1e-12)
    assert info["dv_used_mps"] == 0.0 and info["propellant_used_kg"] == 0.0


def test_failed_thruster_leaves_the_other_side_and_axes_working():
    env = Still(EnvConfig(faults=(ThrusterOff(axis=1, direction=1),)))
    env.reset(seed=0)
    x0 = env.state.copy()
    _, _, _, _, info = env.step(np.array([0.3, -0.5, -0.2]))
    expected = cw_step(env, x0, [0.3, -0.5, -0.2])
    # Same closed form on both sides; the tolerance covers rounding of 1e2 m entries.
    np.testing.assert_allclose(info["true_state"], expected, rtol=1e-12, atol=1e-12)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.3, 0.5, -0.2]))
    np.testing.assert_allclose(info["true_state"], cw_step(env, x0, [0.3, 0.0, -0.2]), rtol=1e-12)


def test_record_keeps_the_agent_command_while_the_actuation_shows_the_fault():
    env = Still(EnvConfig(faults=(ThrusterOff(axis=0, direction=-1),)))
    env.reset(seed=0)
    env.step(np.array([-1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(env.record.command, [-1.0, 0.0, 0.0])
    assert all(not np.any(accel) for _, accel in env.record.actuation.segments)


def test_sensor_bias_is_a_constant_of_the_drawn_size():
    fault = SensorBias(position_m=(2.0, 2.0), velocity_mps=(0.4, 0.8))
    env = Still(EnvConfig(faults=(fault,)), start=(10.0, 200.0, -5.0, 0.01, -0.2, 0.0))
    obs, info = env.reset(seed=3)
    offsets = [obs - info["true_state"]]
    for _ in range(5):
        obs, _, _, _, info = env.step(np.array([0.1, -0.2, 0.0]))
        offsets.append(obs - info["true_state"])
    offsets = np.array(offsets)
    # Truth near 2e2 m and the bias added once: rounding stays below 1e-12 m.
    np.testing.assert_allclose(offsets, np.broadcast_to(offsets[0], offsets.shape), atol=1e-12)
    assert np.linalg.norm(offsets[0, :3]) == pytest.approx(2.0, rel=1e-12)
    assert 0.4 <= np.linalg.norm(offsets[0, 3:]) <= 0.8


def test_sensor_bias_draws_differ_between_seeds_and_repeat_on_one():
    env = Still(EnvConfig(faults=(SensorBias(velocity_mps=(0.4, 0.8)),)))
    first, again, other = (env.reset(seed=s)[0] for s in (5, 5, 6))
    assert np.array_equal(first, again) and not np.array_equal(first, other)


def test_range_loss_keeps_the_line_of_sight_and_its_rate_only():
    fault = RangeLoss(assumed_range_m=100.0)
    fault.reset(np.random.default_rng(0))
    fault.update(0.0, np.zeros((1, 6)))
    # p = (30, 40, 0), v = (1, 0, 0): the line of sight is (0.6, 0.8, 0) and its rate
    # (v - (p . v) p / |p|^2) / |p| = ((1, 0, 0) - 30 / 2500 (30, 40, 0)) / 50
    # = (0.0128, -0.0096, 0).
    reading = fault.on_observation(0.0, np.array([30.0, 40.0, 0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(reading, [60.0, 80.0, 0.0, 1.28, -0.96, 0.0], rtol=1e-14)


def test_range_loss_reading_carries_no_range():
    fault = RangeLoss(assumed_range_m=100.0)
    env = Still(EnvConfig(faults=(fault,)), start=(5.0, 400.0, 2.0, 0.0, -1.0, 0.0))
    obs, info = env.reset(seed=0)
    for _ in range(6):
        assert np.linalg.norm(obs[:3]) == pytest.approx(100.0, rel=1e-12)
        truth = info["true_state"][:3]
        np.testing.assert_allclose(obs[:3] / 100.0, truth / np.linalg.norm(truth), rtol=1e-12)
        obs, _, _, _, info = env.step(np.zeros(3))


def test_scheduled_fault_acts_at_decisions_inside_its_window():
    fault = ThrusterOff(axis=1, direction=1, when=Window(start_s=20.0, end_s=50.0), name="x")
    env = Still(EnvConfig(faults=(fault,)))
    # Decisions at t = 0, 10, ..., 60 s: active at 20, 30 and 40 s.
    assert active_after_each_step(env, 6) == [False, False, True, True, True, False, False]
    env.reset(seed=1)
    used = []
    for _ in range(6):
        *_, info = env.step(np.array([0.0, 0.2, 0.0]))
        used.append(info["dv_used_mps"])
    gained = np.diff([0.0, *used])
    burn = 0.2 * env.cfg.max_accel * env.cfg.dt
    np.testing.assert_allclose(gained, [burn, burn, 0.0, 0.0, 0.0, burn], rtol=1e-12)


def test_triggered_fault_starts_after_the_step_in_which_its_condition_holds():
    # Closing at 2 m/s from 200 m, the range passes 150 m about 25 s in, inside the step that
    # ends at 30 s; the fault is reported active from then on.
    fault = SensorBias(velocity_mps=(0.5, 0.5), when=After(condition=RangeBelow(range_m=150.0)))
    fault.name = "x"
    env = Still(EnvConfig(faults=(fault,)), start=(0.0, 200.0, 0.0, 0.0, -2.0, 0.0))
    assert active_after_each_step(env, 6) == [False, False, False, True, True, True, True]


def test_until_ends_a_fault_when_its_condition_first_holds():
    until = Until(condition=RangeBelow(range_m=150.0))
    until.reset(np.random.default_rng(0))
    far = np.array([[0.0, 200.0, 0.0, 0.0, 0.0, 0.0]])
    near = np.array([[0.0, 100.0, 0.0, 0.0, 0.0, 0.0]])
    assert [until.update(t, s) for t, s in ((0, far), (10, near), (20, far))] == [
        True,
        False,
        False,
    ]


def test_sphere_entry_between_samples_counts():
    sphere = InSphere(center_m=(0.0, 200.0, 0.0), radius_m=6.3)
    # Two samples 20 m either side of the centre, both outside the sphere.
    through = np.array([[0.0, 220.0, 0.0, 0, 0, 0], [0.0, 180.0, 0.0, 0, 0, 0]])
    beside = through + np.array([7.0, 0.0, 0.0, 0, 0, 0])
    assert np.all(np.linalg.norm(through[:, :3] - [0.0, 200.0, 0.0], axis=1) > 6.3)
    assert sphere.met(through) and not sphere.met(beside)
    assert sphere.met(np.array([[1.0, 203.0, 0.0, 0, 0, 0]]))


def test_intermittent_fault_acts_at_its_probability():
    p, steps = 0.3, 2000
    fault = ThrusterOff(axis=0, direction=1, when=Intermittent(probability=p), name="x")
    env = Still(EnvConfig(faults=(fault,), max_steps=steps))
    flags = active_after_each_step(env, steps)[1:]
    sigma = math.sqrt(p * (1 - p) / steps)
    assert abs(np.mean(flags) - p) < Z * sigma


def test_intermittent_pattern_repeats_on_a_seed_and_survives_appending_a_fault():
    def pattern(faults, seed):
        return active_after_each_step(Still(EnvConfig(faults=faults, max_steps=50)), 50, seed)

    first = ThrusterOff(axis=0, direction=1, when=Intermittent(probability=0.5), name="x")
    other = SensorBias(velocity_mps=(0.1, 0.2), when=Intermittent(probability=0.5))
    base = pattern((first,), 11)
    assert base == pattern((first,), 11)
    assert base != pattern((first,), 12)
    assert base == pattern((first, other), 11)


def test_a_fault_that_never_acts_leaves_the_episode_unchanged():
    late = ThrusterOff(axis=1, direction=1, when=Window(start_s=1e6))
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, (10, 3))
    runs = []
    for faults in ((), (late,)):
        env = Still(EnvConfig(faults=faults, sensor=GaussianNav()))
        seen = []
        # A seeded reset, then an unseeded one that continues the same seed sequence.
        for seed in (4, None):
            obs, info = env.reset(seed=seed)
            seen.append(obs)
            for a in actions:
                obs, _, _, _, info = env.step(a)
                seen.append(obs)
            seen.append(info["true_state"])
        runs.append(np.array(seen))
    assert np.array_equal(runs[0], runs[1])


def test_unseeded_resets_draw_new_fault_parameters():
    env = Still(EnvConfig(faults=(SensorBias(velocity_mps=(0.4, 0.8)),)))
    first = env.reset(seed=3)[0]
    second = env.reset()[0]
    assert not np.array_equal(first, second)


def test_environments_from_one_config_do_not_share_fault_state():
    trigger = SensorBias(velocity_mps=(0.5, 0.5), when=After(condition=RangeBelow(range_m=150.0)))
    cfg = EnvConfig(faults=(trigger,))
    near = Still(cfg, start=(0.0, 100.0, 0.0, 0.0, 0.0, 0.0))
    far = Still(cfg, start=(0.0, 400.0, 0.0, 0.0, 0.0, 0.0))
    near.reset(seed=0)
    _, info = far.reset(seed=0)
    near.step(np.zeros(3))
    *_, info = far.step(np.zeros(3))
    assert info["faults_active"] == () and not trigger.active


def test_agents_see_no_fault_channel():
    clean = Still(EnvConfig())
    faults = (SensorBias(velocity_mps=(0.6, 0.6)), ThrusterOff(axis=1, direction=1))
    faulty = Still(EnvConfig(faults=faults))
    for env in (clean, faulty):
        env.reset(seed=0)
        env.step(np.zeros(3))
    assert clean.agent_view() == faulty.agent_view()
    assert clean.agent_info() == faulty.agent_info()


@pytest.mark.parametrize(
    "fault",
    [
        ThrusterOff(axis=2, direction=-1, when=Intermittent(probability=0.5)),
        SensorBias(position_m=(1.0, 2.0), velocity_mps=(0.1, 0.2)),
        RangeLoss(
            assumed_range_m=150.0,
            when=Until(condition=InSphere(center_m=(0, 150, 0), radius_m=6.3)),
        ),
    ],
)
def test_environment_with_faults_follows_the_gymnasium_api(fault):
    check_env(Still(EnvConfig(faults=(fault,), sensor=GaussianNav())), skip_render_check=True)


def test_vector_environment_refuses_faults():
    with pytest.raises(ValueError, match="faults"):
        ApproachVectorEnv(ApproachConfig(faults=(ThrusterOff(axis=0, direction=1),)), num_envs=2)


@pytest.mark.parametrize(
    "build",
    [
        lambda: ThrusterOff(axis=3, direction=1),
        lambda: ThrusterOff(axis=0, direction=0),
        lambda: SensorBias(velocity_mps=(0.8, 0.4)),
        lambda: SensorBias(position_m=(-1.0, 1.0)),
        lambda: SensorBias(velocity_mps=(0.1, math.inf)),
        lambda: SensorBias(velocity_mps=(math.nan, 1.0)),
        lambda: RangeLoss(assumed_range_m=0.0),
        lambda: RangeLoss(assumed_range_m=math.inf),
        lambda: Window(start_s=10.0, end_s=10.0),
        lambda: Window(start_s=math.nan),
        lambda: Intermittent(probability=1.5),
        lambda: Intermittent(probability=math.nan),
        lambda: RangeBelow(range_m=-1.0),
        lambda: InSphere(center_m=(0.0, math.nan, 0.0), radius_m=1.0),
        lambda: InSphere(center_m=(0.0, 0.0), radius_m=1.0),
        lambda: EnvConfig(faults=[ThrusterOff(axis=0, direction=1)]),
        lambda: EnvConfig(faults=(object(),)),
    ],
)
def test_bad_fault_settings_are_refused(build):
    with pytest.raises(ValueError):
        build()


def test_always_is_the_default_activation():
    fault = ThrusterOff(axis=0, direction=1)
    assert isinstance(fault.when, Always)


def test_records_name_added_faults_while_the_agent_view_leaves_them_out():
    plain = make("approach-vbar")
    late = ThrusterOff(axis=1, direction=1, when=Window(start_s=20.0))
    faulty = make("approach-vbar", faults=(late,))
    assert "faults=" in faulty.scenario_id and "faults=" not in plain.scenario_id
    assert faulty.agent_view().scenario_id == plain.agent_view().scenario_id
    assert "window" not in repr(faulty.agent_view())


@pytest.mark.parametrize(
    "name",
    [*sorted(SCENARIOS), *sorted(f"{n}:{v}" for n, s in SCENARIOS.items() for v in s.variants)],
)
def test_every_registered_scenario_applies_faults(name):
    lost = ThrusterOff(axis=1, direction=1, name="x")
    env = make(name, faults=(lost,))
    _, info = env.reset(seed=2)
    assert info["faults_active"] == ("x",)
    command = np.zeros(env.action_space.shape) if env.action_space.shape else 13
    env.step(command)
    assert env.faults[0].active and not lost.active


def test_faults_spawn_no_extra_streams():
    def spawned(faults):
        env = Still(EnvConfig(faults=faults))
        env.reset(seed=9)
        env.reset()
        return env.np_random.bit_generator.seed_seq.n_children_spawned

    # Sensor, actuator and target children at each of two resets, with or without faults.
    assert spawned(()) == 6
    assert spawned((ThrusterOff(axis=0, direction=1),)) == 6


def test_all_of_acts_only_while_every_member_does_and_updates_each_member():
    # A band that opens below 500 m and closes for good once the sphere 200 m out is entered.
    band = AllOf(
        activations=(
            After(condition=RangeBelow(range_m=500.0)),
            Until(condition=InSphere(center_m=(0.0, -200.0, 0.0), radius_m=6.3)),
        )
    )
    band.reset(np.random.default_rng(0))

    def at(y):
        return np.array([[0.0, y, 0.0, 0.0, 0.0, 0.0]])

    path = [at(-800.0), at(-400.0), at(-300.0), at(-200.0), at(-150.0), at(-600.0)]
    assert [band.update(10.0 * k, s) for k, s in enumerate(path)] == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize("activations", [(), [Always()]])
def test_all_of_refuses_an_empty_or_untupled_list(activations):
    with pytest.raises(ValueError):
        AllOf(activations=activations)


def test_all_of_updates_a_member_while_another_is_inactive():
    # The sphere is passed while the 100 m trigger has not fired; Until must still latch, so the
    # pair stays off once the trigger fires later.
    pair = AllOf(
        activations=(
            After(condition=RangeBelow(range_m=100.0)),
            Until(condition=InSphere(center_m=(0.0, -200.0, 0.0), radius_m=6.3)),
        )
    )
    pair.reset(np.random.default_rng(0))
    path = [[0.0, -200.0, 0.0], [0.0, -150.0, 0.0], [0.0, -50.0, 0.0]]
    flags = [pair.update(10.0 * k, np.array([[*p, 0.0, 0.0, 0.0]])) for k, p in enumerate(path)]
    assert flags == [False, False, False]
