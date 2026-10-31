"""The tracker's navigation filter: a Kalman filter on the exact Clohessy-Wiltshire model, fed the
tracker's own commands, with a noise model taken only from the scenario's public sensor.

The filter is checked on observations the test scripts, so the estimate can be compared with a
known truth; a leak of the environment shows as an estimate that depends on anything but the
observation, the time and the tracker's own commands. The flights of whole episodes are marked
`slow`.
"""

import copy
import dataclasses
import math

import gymnasium as gym
import numpy as np
import pytest
from scenario_references import cw_terms, reference_for
from supervision_support import Session

from vbar import run_episode
from vbar.agents import action_scale
from vbar.agents.rules import RuleSupervisor
from vbar.agents.supervision import SupervisedAgent
from vbar.agents.tracker import SetpointTracker
from vbar.dynamics import cw_discrete, cw_stm
from vbar.scenarios import make, seeds
from vbar.scenarios.suites import SUITES
from vbar.spacecraft import sensors

NAV1 = {"sigma_pos": 1.0, "sigma_vel": 1e-3}
DOCK = make("docking:axis").agent_view()
DOCK_NAV = make("docking:axis:nav1").agent_view()
N, DT = DOCK_NAV.n, DOCK_NAV.dt
PORT = np.array(DOCK_NAV.goal["port"], dtype=float)
AXIS = np.array(DOCK_NAV.goal["port_axis"], dtype=float)
# The standoff is outside every approach cone's activation range by the tracker's margin; on this
# axis (V-bar) it is an equilibrium of the Clohessy-Wiltshire model, so a chaser at rest there
# needs no thrust.
REACH = max(c["active_range"] for c in DOCK_NAV.constraints if c["type"] == "approach_cone")
STANDOFF = PORT + (REACH + 10.0) * AXIS
DESCENT = 0.3
END = DOCK_NAV.dt * DOCK_NAV.max_steps
BOUND_MANOEUVRE = 1.0


def tracker_for(view, **kwargs):
    tracker = SetpointTracker(**kwargs)
    tracker.reset(view)
    return tracker


def fix(truth, rng, sigma=(1.0, 1e-3)):
    return truth + np.repeat(sigma, 3) * rng.standard_normal(6)


# The sensor argument


def test_the_default_sensor_reads_the_noise_of_the_scenarios_variant():
    assert tracker_for(DOCK_NAV).sensor_noise == pytest.approx((1.0, 1e-3))


@pytest.mark.parametrize(
    "scenario", ["approach-vbar:nav1", "approach-rbar:nav1", "inspection:nav1"]
)
def test_the_default_sensor_reads_the_nav1_noise_of_other_scenarios(scenario):
    view = make(scenario).agent_view()
    assert tracker_for(view).sensor_noise == pytest.approx((1.0, 1e-3))


@pytest.mark.parametrize("scenario", ["docking:axis", "approach-vbar", "docking-afrl"])
def test_the_default_sensor_runs_no_filter_when_the_scenario_has_no_noisy_sensor(scenario):
    assert tracker_for(make(scenario).agent_view()).sensor_noise is None


@pytest.mark.parametrize(
    "identifier",
    [
        "vbar:my-own-task:cw:box:dv:v1",
        "vbar:docking:cw:box:dv:v1:axis:nav1",
        "vbar:docking:cw:box:dv:v2:axis:no-such-variant",
        "docking:axis:nav1",
        "not an identifier",
    ],
)
def test_the_default_sensor_runs_no_filter_for_an_identifier_outside_the_catalogue(identifier):
    view = dataclasses.replace(DOCK_NAV, scenario_id=identifier)
    assert tracker_for(view).sensor_noise is None


def test_a_sensor_configured_by_hand_is_not_read_from_the_catalogue():
    env = make("docking:axis", sensor=sensors.GaussianNav(sigma_pos=2.0))
    assert tracker_for(env.agent_view()).sensor_noise is None


def test_a_sensor_of_none_runs_no_filter_even_on_a_noisy_scenario():
    assert tracker_for(DOCK_NAV, sensor=None).sensor_noise is None


def test_a_mapping_gives_the_noise_directly_on_any_scenario():
    spec = {"sigma_pos": 0.5, "sigma_vel": 2e-3}
    assert tracker_for(DOCK, sensor=spec).sensor_noise == pytest.approx((0.5, 2e-3))
    assert tracker_for(DOCK_NAV, sensor=spec).sensor_noise == pytest.approx((0.5, 2e-3))


def test_a_mapping_may_hold_integers_and_numpy_floats():
    tracker = tracker_for(DOCK, sensor={"sigma_pos": 1, "sigma_vel": np.float64(1e-3)})
    assert tracker.sensor_noise == pytest.approx((1.0, 1e-3))
    assert all(type(x) is float for x in tracker.sensor_noise)


def test_changing_the_mapping_after_construction_does_not_change_the_noise():
    spec = {"sigma_pos": 1.0, "sigma_vel": 1e-3}
    tracker = SetpointTracker(sensor=spec)
    spec["sigma_pos"] = 50.0
    tracker.reset(DOCK)
    assert tracker.sensor_noise == pytest.approx((1.0, 1e-3))


HOSTILE = {
    "nan position": {"sigma_pos": math.nan, "sigma_vel": 1e-3},
    "nan velocity": {"sigma_pos": 1.0, "sigma_vel": math.nan},
    "inf position": {"sigma_pos": math.inf, "sigma_vel": 1e-3},
    "inf velocity": {"sigma_pos": 1.0, "sigma_vel": math.inf},
    "negative position": {"sigma_pos": -1.0, "sigma_vel": 1e-3},
    "negative velocity": {"sigma_pos": 1.0, "sigma_vel": -1e-3},
    "zero position": {"sigma_pos": 0.0, "sigma_vel": 1e-3},
    "zero velocity": {"sigma_pos": 1.0, "sigma_vel": 0.0},
    "string value": {"sigma_pos": "1.0", "sigma_vel": 1e-3},
    "bool value": {"sigma_pos": True, "sigma_vel": 1e-3},
    "none value": {"sigma_pos": 1.0, "sigma_vel": None},
    "list value": {"sigma_pos": [1.0], "sigma_vel": 1e-3},
    "unknown key": {"sigma_pos": 1.0, "sigma_vel": 1e-3, "bias_pos": 0.1},
    "misspelled key": {"sigma_pos": 1.0, "sigma_velocity": 1e-3},
    "missing velocity": {"sigma_pos": 1.0},
    "missing position": {"sigma_vel": 1e-3},
    "empty": {},
}


@pytest.mark.parametrize("spec", HOSTILE.values(), ids=HOSTILE.keys())
def test_a_hostile_mapping_raises_at_construction(spec):
    with pytest.raises(ValueError):
        SetpointTracker(sensor=spec)


@pytest.mark.parametrize("spec", ["nav1", "", 1.0, 0, [1.0, 1e-3], (1.0, 1e-3), object()])
def test_a_sensor_that_is_not_scenario_none_or_a_mapping_raises_at_construction(spec):
    with pytest.raises(ValueError):
        SetpointTracker(sensor=spec)


# The estimate


def test_the_estimate_is_none_before_the_first_act_and_after_a_reset():
    tracker = tracker_for(DOCK_NAV)
    assert tracker.estimate is None
    tracker.act(np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0]), 0.0)
    assert tracker.estimate is not None
    tracker.reset(DOCK_NAV)
    assert tracker.estimate is None


def test_the_estimate_equals_the_observation_at_the_first_fix():
    tracker = tracker_for(DOCK_NAV)
    z = np.array([0.7, -101.3, 0.4, 0.0021, -0.0013, 0.0005])
    tracker.act(z, 0.0)
    assert tracker.estimate.shape == (6,)
    assert np.allclose(tracker.estimate, z, rtol=0.0, atol=1e-12)


def test_the_estimate_is_a_read_only_copy():
    tracker = tracker_for(DOCK_NAV)
    z = np.array([0.7, -101.3, 0.4, 0.0021, -0.0013, 0.0005])
    tracker.act(z, 0.0)
    shown = tracker.estimate
    shown[:] = 1e6
    assert np.allclose(tracker.estimate, z, atol=1e-12)
    with pytest.raises(AttributeError):
        tracker.estimate = z


def test_copies_of_a_tracker_do_not_share_a_filter():
    first = SetpointTracker()
    second = copy.copy(first)
    first.reset(DOCK_NAV)
    second.reset(DOCK_NAV)
    first.act(np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0]), 0.0)
    assert first.estimate is not None and second.estimate is None


# 1 m of noise on a hold at the V-bar point, an equilibrium of the model: a filter that averages
# k fixes has a position error of 1 m / sqrt(k) per axis, sqrt(3) m / sqrt(k) in norm.
VBAR_POINT = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
TRIALS = 60


def coast_error(truth_at, fixes_at, trials=TRIALS, sigma=(1.0, 1e-3)):
    """Root mean square position error of the estimate (m) after each count in `fixes_at`."""
    last = max(fixes_at)
    squares = {k: [] for k in fixes_at}
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        tracker = tracker_for(DOCK_NAV)
        for k in range(1, last + 1):
            tracker.act(fix(truth_at(k - 1), rng, sigma), DT * (k - 1))
            if k in squares:
                miss = tracker.estimate[:3] - truth_at(k - 1)[:3]
                squares[k].append(float(miss @ miss))
    return {k: math.sqrt(float(np.mean(v))) for k, v in squares.items()}


def test_the_position_error_of_a_hold_falls_as_one_over_the_root_of_the_fixes():
    counts = (16, 64, 256)
    error = coast_error(lambda k: VBAR_POINT, counts)
    for k in counts:
        # 60 trials of 3 axes give the root mean square a relative spread of about 7%; the
        # band is four spreads either side of the ideal sqrt(3) / sqrt(k).
        ratio = error[k] * math.sqrt(k) / math.sqrt(3.0)
        assert 0.72 < ratio < 1.30, (k, ratio)


def drifting(k):
    """A free CW ellipse about the target with no along-track drift (vy0 = -2 n x0)."""
    start = np.array([30.0, -50.0, 5.0, 0.0, -2.0 * N * 30.0, 0.002])
    return cw_stm(N, DT * k) @ start


def test_the_position_error_of_a_free_coast_falls_with_the_fixes_by_the_dynamics():
    error = coast_error(drifting, (25, 100, 200), trials=40)
    # The truth moves tens of metres over these fixes, so averaging the raw positions would leave
    # an error of that size; the model carries every fix to the present. An ideal filter on this
    # arc has an error of 1.7 to 2.0 m / sqrt(k), against sqrt(3) m / sqrt(k) for a fixed point
    # (the arc adds geometry); the bound is 2.6 m / sqrt(k).
    for k, miss in error.items():
        assert miss * math.sqrt(k) < 2.6, (k, miss)
    assert error[200] < 0.5 * error[25]


def test_the_estimate_follows_a_manoeuvre_by_the_trackers_own_commands():
    scale, a_max = action_scale(DOCK_NAV), DOCK_NAV.max_accel
    phi, gamma = cw_discrete(N, DT)
    squares = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        tracker = tracker_for(DOCK_NAV)
        tracker.track_velocity(VBAR_POINT[:3], [0.0, 0.2, 0.0])
        truth = VBAR_POINT.copy()
        for k in range(60):
            command = tracker.act(fix(truth, rng), DT * k)
            if k >= 30:
                miss = tracker.estimate[:3] - truth[:3]
                squares.append(float(miss @ miss))
            truth = phi @ truth + gamma @ (np.asarray(command) / scale * a_max)
    # The velocity changes by tens of millimetres per second in a step against a velocity noise of
    # 1 mm/s, so an estimate that ignored the commands would lag the truth by metres.
    assert math.sqrt(float(np.mean(squares))) < BOUND_MANOEUVRE


# The estimate depends on the observation, the time and the tracker's own commands only


class Script:
    """A sensor that ignores the true state and reports a fixed list of fixes in order."""

    name = "script"

    def __init__(self, fixes):
        self.fixes = fixes
        self.k = 0

    def reset(self, rng):
        self.k = 0

    def observe(self, true_state, t):
        z = self.fixes[min(self.k, len(self.fixes) - 1)]
        self.k += 1
        return z.copy()


def scripted_fixes(count=80, seed=7):
    rng = np.random.default_rng(seed)
    return [fix(np.concatenate([STANDOFF, np.zeros(3)]), rng) for _ in range(count)]


def scripted_flight(seed, steps=60, sensor="scenario"):
    """Steps a tracker through an environment whose sensor reports the same fixes whatever the
    true state; returns the true starts, the estimates and the commands."""
    env = make("docking:axis", sensor=Script(scripted_fixes()))
    obs, info = env.reset(seed=seed)
    tracker = tracker_for(DOCK_NAV, sensor=sensor)
    tracker.fly_to(PORT, "contact")
    start = np.array(info["true_state"], dtype=float)
    estimates, commands = [], []
    for _ in range(steps):
        command = tracker.act(obs, env.agent_info()["time_s"])
        estimates.append(tracker.estimate)
        commands.append(np.array(command, dtype=float))
        obs, _, terminated, truncated, info = env.step(command)
        if terminated or truncated:
            break
    return start, estimates, commands, tracker


def test_two_runs_from_different_true_starts_fed_the_same_observations_agree_at_every_step():
    flights = {}
    for seed, _ in seeds("docking:axis", "validation", 8).pairs():
        flights[seed] = scripted_flight(seed)
    first = next(iter(flights))
    far = [s for s in flights if np.linalg.norm(flights[s][0][:3] - flights[first][0][:3]) > 10.0]
    assert far, "the drawn starts are all within 10 m of each other"
    _, estimates, commands, _ = flights[first]
    _, other_estimates, other_commands, _ = flights[far[0]]
    assert len(estimates) == len(other_estimates) > 30
    assert all(np.array_equal(a, b) for a, b in zip(estimates, other_estimates, strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(commands, other_commands, strict=True))


def test_the_filter_has_an_effect_in_the_scripted_flights():
    _, estimates, _, _ = scripted_flight(seeds("docking:axis", "validation", 1).pairs()[0][0])
    raw = scripted_fixes()
    assert not np.allclose(estimates[-1], raw[len(estimates) - 1], atol=1e-3)


def reachable(obj, depth=6, seen=None):
    """Every object found by following attributes, lists, tuples and mappings from `obj`."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth < 0:
        return
    seen.add(id(obj))
    yield obj
    if isinstance(obj, np.ndarray | str | bytes | int | float):
        return
    children = []
    if isinstance(obj, dict) or hasattr(obj, "items") and callable(obj.items):
        children = [*obj.keys(), *obj.values()]
    elif isinstance(obj, list | tuple | set | frozenset):
        children = list(obj)
    elif hasattr(obj, "__dict__"):
        children = list(vars(obj).values())
    for child in children:
        yield from reachable(child, depth - 1, seen)


def test_the_tracker_holds_no_reference_to_an_environment_or_a_sensor():
    _, _, _, tracker = scripted_flight(seeds("docking:axis", "validation", 1).pairs()[0][0])
    sensor_classes = tuple(
        c
        for c in vars(sensors).values()
        if isinstance(c, type)
        and c.__module__ == sensors.__name__
        and not getattr(c, "_is_protocol", False)
    )
    held = list(reachable(tracker))
    assert not [o for o in held if isinstance(o, gym.Env)]
    assert not [o for o in held if isinstance(o, sensor_classes)]
    assert not [o for o in held if type(o).__name__ in ("Script", "ScenarioEnv")]
    text = repr(vars(tracker))
    assert "Env" not in text and "Nav" not in text and "Script" not in text


def test_the_catalogue_sensor_parameters_held_are_plain_numbers_copied_at_reset():
    tracker = tracker_for(DOCK_NAV)
    assert tracker.sensor_noise is not None
    assert all(type(x) is float for x in tracker.sensor_noise)


# The tracker without a filter


def test_with_no_filter_the_commands_are_those_of_the_raw_observation():
    view = DOCK_NAV
    scale, a_max, k_v = action_scale(view), view.max_accel, 0.05
    tracker = tracker_for(view, sensor=None, k_v=k_v)
    v_cmd = np.array([0.0, 0.05, 0.0])
    tracker.track_velocity([0.0, -100.0, 0.0], v_cmd)
    rng = np.random.default_rng(3)
    for k in range(20):
        z = fix(VBAR_POINT, rng)
        wanted = (-cw_terms(z, N) + k_v * (v_cmd - z[3:6])) / a_max
        assert np.allclose(tracker.act(z, DT * k), np.clip(wanted, -1.0, 1.0) * scale, atol=1e-12)


def test_the_default_tracker_on_a_scenario_without_noise_flies_the_raw_observation():
    rng = np.random.default_rng(5)
    default, raw = tracker_for(DOCK), tracker_for(DOCK, sensor=None)
    for tracker in (default, raw):
        tracker.fly_to(PORT, "contact")
    start = np.concatenate([STANDOFF + [3.0, 0.0, -2.0], np.zeros(3)])
    for k in range(40):
        z = fix(start, rng, sigma=(0.5, 1e-3))
        assert np.array_equal(default.act(z, DT * k), raw.act(z, DT * k))


def test_a_supervisor_is_shown_the_raw_observation_not_the_estimate():
    session = Session(DOCK_NAV, tracker=SetpointTracker())
    rng = np.random.default_rng(11)
    for _ in range(6):
        z = fix(VBAR_POINT, rng)
        session.step(z)
        assert np.array_equal(np.asarray(session.sit.obs)[:6], z)


# The hold at the standoff


def descent(command):
    """Closing speed asked of the port, as a fraction of a full command along the axis."""
    return float(-np.asarray(command) @ AXIS)


def at_standoff(tracker, steps, t0=0.0):
    """Commands to hold the chaser at rest at the standoff for `steps` fixes."""
    z = np.concatenate([STANDOFF, np.zeros(3)])
    tracker.fly_to(PORT, "contact")
    return [descent(tracker.act(z, t0 + DT * k)) for k in range(steps)]


def test_the_comparison_tracker_without_a_filter_descends_from_the_standoff_at_once():
    first = at_standoff(tracker_for(DOCK_NAV, sensor=None), 1)[0]
    assert first > DESCENT


def test_the_tracker_holds_the_standoff_while_the_lateral_sigma_is_above_the_limit():
    steps = at_standoff(tracker_for(DOCK_NAV), 40)
    assert max(steps) < 0.05 * DESCENT


def test_the_tracker_descends_at_once_when_the_sensor_is_precise_enough():
    precise = {"sigma_pos": 0.001, "sigma_vel": 1e-6}
    assert at_standoff(tracker_for(DOCK_NAV, sensor=precise), 1)[0] > DESCENT


def test_the_tracker_descends_once_a_few_fixes_bring_the_lateral_sigma_below_the_limit():
    # One fix of 0.05 m leaves a lateral sigma of 0.05 m or more, above 0.045 m; a few fixes of
    # a hold bring it below. Only the release is checked: the scripted fixes stay put while the
    # tracker commands the descent, so the filter later believes the chaser moved and eases off.
    steps = at_standoff(tracker_for(DOCK_NAV, sensor={"sigma_pos": 0.05, "sigma_vel": 1e-4}), 10)
    assert steps[0] < 0.05 * DESCENT
    assert max(steps[1:]) > DESCENT


def test_the_default_lateral_sigma_is_45_millimetres():
    assert SetpointTracker().lateral_sigma == 0.045


def test_the_tracker_descends_without_waiting_when_only_the_descent_time_is_left():
    # The descent from the standoff at the window's middle speed takes about 800 s; 200 s left
    # cannot wait for the estimate.
    steps = at_standoff(tracker_for(DOCK_NAV), 1, t0=END - 200.0)
    assert steps[0] > DESCENT


def test_the_tracker_still_waits_when_far_more_time_than_the_descent_needs_is_left():
    # Half the episode left is about seven times the time of the descent.
    steps = at_standoff(tracker_for(DOCK_NAV), 5, t0=0.5 * END)
    assert max(steps) < 0.05 * DESCENT


# Whole episodes

OTHER_NAV_SCENARIOS = [
    e.scenario
    for e in SUITES["core-v1"].entries
    if e.scenario.endswith(":nav1") and e.scenario != "docking:axis:nav1"
]


def wins(name, make_agent):
    """Draws, of five validation seeds, an agent wins, and those with a keep-out violation."""
    won, violated = [], []
    for index, (seed, agent_seed) in enumerate(seeds(name, "validation", 5).pairs()):
        env = make(name)
        result = run_episode(env, make_agent(), seed=seed, agent_seed=agent_seed)
        if env._info()["success"]:
            won.append(index)
            if result["keep_out_violated"]:
                violated.append(index)
    return won, violated


def supervised():
    return SupervisedAgent(RuleSupervisor(), tracker=SetpointTracker())


@pytest.mark.slow
def test_the_supervised_tracker_docks_on_the_noisy_axis_as_often_as_the_reference_and_cleanly():
    name = "docking:axis:nav1"
    ours, violated = wins(name, supervised)
    theirs, _ = wins(name, lambda: reference_for(name))
    assert len(ours) >= len(theirs) > 0, (ours, theirs)
    assert not violated


@pytest.mark.slow
@pytest.mark.parametrize("name", OTHER_NAV_SCENARIOS)
def test_the_supervised_tracker_succeeds_on_as_many_other_noisy_runs_as_the_reference(name):
    ours, _ = wins(name, supervised)
    theirs, _ = wins(name, lambda: reference_for(name))
    assert len(ours) >= len(theirs), (ours, theirs)


@pytest.mark.parametrize(
    "setting", [{"action_mode": "accel"}, {"action_mode": "thrust"}, {"max_steps": 5000}]
)
def test_a_setting_that_leaves_the_sensor_keeps_the_catalogue_noise(setting):
    tracker = SetpointTracker()
    tracker.reset(make("docking:axis:nav1", **setting).agent_view())
    assert tracker.sensor_noise == (1.0, 1e-3)


@pytest.mark.parametrize(
    "sensor", [sensors.PerfectNav(), sensors.GaussianNav(sigma_pos=5.0)], ids=["perfect", "5 m"]
)
def test_a_sensor_changed_on_a_nav1_run_is_not_read_from_the_catalogue(sensor):
    tracker = SetpointTracker()
    tracker.reset(make("docking:axis:nav1", sensor=sensor).agent_view())
    assert tracker.sensor_noise is None


@pytest.mark.parametrize("token", ["spec=0123abcd", "kind=custom"])
def test_an_identifier_naming_its_own_spec_or_kind_runs_no_filter(token):
    view = make("docking:axis:nav1").agent_view()
    tracker = SetpointTracker()
    tracker.reset(dataclasses.replace(view, scenario_id=f"{view.scenario_id}:{token}"))
    assert tracker.sensor_noise is None
