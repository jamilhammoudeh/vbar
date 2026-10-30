"""The task briefing: every goal and constraint in words, from the public view, and the observed
state measured against them."""

import dataclasses
import math
import re

import numpy as np
import pytest

from vbar.agents.base import AgentView
from vbar.agents.briefing import TASK_KEYS, briefing, check_goal, situation, target_mode
from vbar.scenarios import (
    SCENARIOS,
    FixedOffset,
    InertialPointing,
    Maneuvering,
    Misbriefed,
    Passive,
    Tumbling,
    get,
    make,
)
from vbar.scenarios.registry import entry_class
from vbar.spacecraft import GaussianNav


def every_scenario():
    names = []
    for name in sorted(SCENARIOS):
        names.append(name)
        names.extend(f"{name}:{variant}" for variant in get(name).variants)
    return names


@pytest.mark.parametrize("name", every_scenario())
def test_every_goal_entry_and_constraint_of_the_catalogue_is_put_in_words(name):
    # The fallback lines name a raw key or a constraint type; none is needed for the catalogue.
    view = make(name).agent_view()
    text = briefing(view)
    assert "Goal entry" not in text and "Constraint " not in text and "Target:" not in text
    assert text.count("\n- ") == len(view.constraints)


def view_with(goal, constraints=()):
    return AgentView("test", 10.0, 10, 0.001, "normalized", 0.02, "box", 100.0, goal, constraints)


def test_an_unknown_goal_entry_constraint_and_task_are_still_shown():
    goal = {"task": "capture", "tumble_rate": 0.05, "target": {"mode": "passive"}}
    text = briefing(view_with(goal, ({"type": "glare", "limit": 0.4},)))
    assert "Task: capture." in text
    assert "Goal entry tumble_rate: 0.05." in text
    assert "Constraint glare: {'limit': 0.4}." in text


@pytest.mark.parametrize("task", sorted(TASK_KEYS))
def test_a_goal_without_a_key_its_task_needs_is_refused_naming_the_key(task):
    goal = {"task": task, "target": {"mode": "passive"}}
    goal.update({key: 1.0 for key in TASK_KEYS[task]})
    for key in ("hold_point", "port", "port_axis"):
        if key in goal:
            goal[key] = (0.0, -1.0, 0.0)
    if "axial_speed_window" in goal:
        goal["axial_speed_window"] = (0.05, 0.1)
    check_goal(view_with(goal))
    for key in TASK_KEYS[task]:
        partial = {k: v for k, v in goal.items() if k != key}
        with pytest.raises(ValueError, match=f"the {task} goal has no {key}"):
            check_goal(view_with(partial))


def test_an_unknown_target_mode_or_a_missing_parameter_is_refused():
    with pytest.raises(ValueError, match="unknown target mode 'spinning'"):
        target_mode({"mode": "spinning"})
    with pytest.raises(ValueError, match="the inertial target mode needs 'phase'"):
        target_mode({"mode": "inertial"})


N = 0.0011


@pytest.mark.parametrize(
    "mode",
    [
        Passive(),
        FixedOffset(math.radians(20.0)),
        FixedOffset(math.radians(20.0), amplitude=0.02, period=300.0),
        InertialPointing(1.3),
        Tumbling((0.3, -0.2, 1.0), 0.01, 0.4),
        Maneuvering(((900.0, 600.0, (0.0, 5e-4, 0.0)),)),
        Maneuvering(((10.0, 20.0, (1e-4, 0.0, 0.0)),), attitude=InertialPointing(0.7)),
        Maneuvering(((10.0, 20.0, (1e-4, 0.0, 0.0)),), attitude=FixedOffset(0.2, 0.01, 50.0)),
        Maneuvering(((10.0, 20.0, (1e-4, 0.0, 0.0)),), attitude=Tumbling((0, 0, 1), 0.02, 0.1)),
    ],
)
def test_the_target_mode_rebuilt_from_its_public_entry_turns_as_the_mode_does(mode):
    rebuilt = target_mode(mode.public(N))
    t = np.array([0.0, 37.0, 1234.5, 8000.0])
    # The same rotation formulas on the same parameters: equal to rounding.
    assert np.allclose(rebuilt.matrix(t, N), mode.matrix(t, N), rtol=0.0, atol=1e-15)
    assert rebuilt.burns() == mode.burns()


def numbers(line):
    return [float(v) for v in re.findall(r"-?\d+\.\d+", line)]


def line_of(text, start):
    return next(line for line in text.splitlines() if line.strip().startswith(start))


def test_docking_offsets_come_from_the_noisy_observation():
    env = make("docking:axis", sensor=GaussianNav(0.5, 5e-3))
    obs, info = env.reset(seed=4)
    text = situation(env.agent_view(), obs, env.agent_info())
    depth, lateral = numbers(line_of(text, "distance along the port axis"))
    axial, sideways = numbers(line_of(text, "speed along the axis toward the port"))
    axis = np.array([0.0, -1.0, 0.0])

    def offsets(state):
        offset = state[:3] - np.array([0.0, -2.0, 0.0])
        d = offset @ axis
        return d, np.linalg.norm(offset - d * axis), -(state[3:6] @ axis)

    seen, true = offsets(obs), offsets(info["true_state"])
    # Printed to 1 mm and 0.1 mm/s.
    assert (depth, lateral, axial) == pytest.approx(seen, abs=6e-4)
    assert sideways == pytest.approx(np.linalg.norm(obs[3:6] + seen[2] * axis), abs=6e-5)
    # The 0.5 m and 5 mm/s noise moves each figure far beyond its printed rounding.
    assert abs(lateral - true[1]) > 0.01 or abs(depth - true[0]) > 0.01
    assert abs(axial - true[2]) > 1e-4


def test_a_turning_port_is_reported_where_the_scenario_puts_it():
    env = make("inertial-target")
    obs, _ = env.reset(seed=2)
    view = env.agent_view()
    for t in (0.0, 600.0, 2400.0):
        text = situation(view, obs, {"time_s": t})
        values = numbers(line_of(text, "port now at"))
        assert values[:3] == pytest.approx(env.port_at(t), abs=0.006)
        assert values[3:] == pytest.approx(env.axis_at(t), abs=6e-5)


def docking_with(target):
    spec = dataclasses.replace(get("docking"), target=target)
    return entry_class(spec)(spec)


def test_a_misbriefed_target_is_described_as_its_briefing():
    # The truth turns from 1.0 rad; the briefing says 0.3 rad. A scenario honestly flying the
    # briefed mode gives the same words, so they carry nothing of the truth.
    told = InertialPointing(0.3)
    texts = []
    for env in (
        docking_with(Misbriefed(actual=InertialPointing(1.0), briefed=told)),
        docking_with(told),
    ):
        obs, _ = env.reset(seed=5)
        view = env.agent_view()
        texts.append(briefing(view) + situation(view, obs, {"time_s": 900.0}))
    assert texts[0] == texts[1]
    assert "57.3 deg" not in texts[0] and "17.19 deg" in texts[0]


def test_inspection_progress_is_the_coverage_the_agent_is_told():
    env = make("inspection")
    obs, _ = env.reset(seed=3)
    info = env.agent_info()
    assert 0.0 < info["coverage"] < 1.0
    text = situation(env.agent_view(), obs, info)
    assert f"points imaged so far = {100.0 * info['coverage']:.1f}% of all" in text


def test_a_thrust_inhibit_window_is_announced_only_while_it_lasts():
    env = make("rpm-hold")
    obs, _ = env.reset(seed=1)
    view = env.agent_view()
    assert "inhibited (commands are ignored) from 300 s to 570 s" in briefing(view)
    for t, inside in ((290.0, False), (300.0, True), (560.0, True), (570.0, False)):
        assert (
            "thrusters inhibited until t = 570 s" in situation(view, obs, {"time_s": t})
        ) is inside


def test_a_pitched_target_s_corridor_is_given_in_hill_axes_along_its_port_axis():
    # The cone's spec is in body axes, (0, -1, 0); pitched by the drawn angle, it lies along the
    # goal's port axis, and its apex behind the port.
    env = make("docking-offset")
    env.reset(seed=6)
    view = env.agent_view()
    line = line_of(briefing(view), "- Corridor")
    values = numbers(line)
    apex, axis = values[1:4], values[4:7]
    assert axis == pytest.approx(view.goal["port_axis"], abs=6e-4)
    port = np.array(view.goal["port"])
    setback = np.linalg.norm(np.array(apex) - port)
    assert np.array(apex) == pytest.approx(
        port - setback * np.array(view.goal["port_axis"]), abs=0.01
    )
    assert axis != pytest.approx([0.0, -1.0, 0.0], abs=0.05)


def test_a_corridor_on_a_turning_target_is_reported_where_it_stands_now():
    env = make("hst-rbar-grapple")
    obs, _ = env.reset(seed=2)
    view = env.agent_view()
    (cone,) = [c for c in env.constraints if c.name == "approach_cone"]
    for t in (0.0, 900.0, 3000.0):
        values = numbers(line_of(situation(view, obs, {"time_s": t}), "corridor now"))
        assert values[:3] == pytest.approx(cone.apex_at(t), abs=0.006)
        assert values[3:] == pytest.approx(cone.axis_at(t), abs=6e-5)
    assert "corridor now" not in situation(make("docking").agent_view(), obs, {"time_s": 0.0})


def test_the_comparison_tasks_entries_are_put_in_words():
    docking = briefing(make("afrl-docking").agent_view())
    assert "The episode ends as a failure if you go farther than 800 m from the target." in docking
    assert "a success only at a speed below" in docking
    assert "collision avoidance burns [0.300, 0.000, 0.000] m/s" in docking
    inspection = briefing(make("afrl-inspection").agent_view())
    assert "the planet's shadow is not modelled" in inspection
    assert "outside the planet's shadow" in briefing(make("inspection").agent_view())


def test_a_departure_is_measured_outward_along_the_port_axis():
    env = make("separation-flyaround")
    obs, _ = env.reset(seed=3)
    view = env.agent_view()
    text = situation(view, obs, env.agent_info())
    depth, lateral = numbers(line_of(text, "distance along the port axis"))
    outward, sideways = numbers(line_of(text, "speed along the axis outward"))
    (clear,) = numbers(line_of(text, "distance to the clear point"))
    port, axis = np.array(view.goal["port"]), np.array(view.goal["port_axis"])
    offset = obs[:3] - port
    # Printed to 1 mm, 0.1 mm/s and 1 cm.
    assert depth == pytest.approx(offset @ axis, abs=6e-4)
    assert lateral == pytest.approx(np.linalg.norm(offset - (offset @ axis) * axis), abs=6e-4)
    assert outward == pytest.approx(obs[3:6] @ axis, abs=6e-5) and outward > 0.0
    assert sideways == pytest.approx(np.linalg.norm(obs[3:6] - (obs[3:6] @ axis) * axis), abs=6e-5)
    assert clear == pytest.approx(
        np.linalg.norm(obs[:3] - np.array(view.goal["hold_point"])), abs=6e-3
    )


@pytest.mark.parametrize(
    "description",
    [
        {"mode": "inertial", "phase": math.nan},
        {"mode": "inertial", "phase": "abc"},
        {"mode": "inertial", "phase": True},
        {"mode": "fixed_offset", "pitch": "x"},
        {"mode": "fixed_offset", "pitch": 0.1, "limit_cycle_period": math.inf},
        {"mode": "tumbling", "axis": (0.0, 0.0, 1.0), "rate": math.inf, "phase": 0.0},
        {"mode": "tumbling", "axis": (0.0, 0.0, 0.0), "rate": 0.01, "phase": 0.0},
        {"mode": "tumbling", "axis": (0.0, 1.0), "rate": 0.01, "phase": 0.0},
        {"mode": "tumbling", "axis": (1e-300, 0.0, 0.0), "rate": 0.01, "phase": 0.0},
        {"mode": "maneuvering", "burns": ((0.0, 10.0, (0.0, math.nan, 0.0)),)},
        {"mode": "maneuvering", "burns": ((0.0, 10.0),)},
    ],
)
def test_a_target_mode_with_a_parameter_that_is_not_a_finite_number_is_refused(description):
    with pytest.raises(ValueError, match="numeric and finite|nonzero 3-vector|a burn is"):
        target_mode(description)


@pytest.mark.parametrize(
    ("name", "key", "value"),
    [
        ("docking", "lateral_limit", math.nan),
        ("docking", "axial_speed_window", (0.05, math.inf)),
        ("docking", "port", "front"),
        ("approach-vbar", "capture_speed", None),
        ("inspection", "coverage", -math.inf),
        ("separation-flyaround", "clear_range", True),
    ],
)
def test_a_goal_entry_its_task_needs_that_is_not_a_finite_number_is_refused(name, key, value):
    view = make(name).agent_view()
    check_goal(view)
    with pytest.raises(ValueError, match=f"{key} must be numeric and finite"):
        check_goal(dataclasses.replace(view, goal={**view.goal, key: value}))


@pytest.mark.parametrize("name", ["docking", "separation-flyaround"])
# A norm that underflows to zero divides by zero in the situation lines as an exact zero does.
@pytest.mark.parametrize("axis", [(0.0, 0.0, 0.0), (1e-300, 0.0, 0.0), (0.0, 1.0)])
def test_a_port_axis_that_is_not_a_nonzero_3_vector_is_refused(name, axis):
    view = make(name).agent_view()
    with pytest.raises(ValueError, match="port_axis (must be a nonzero 3-vector|must have shape)"):
        check_goal(dataclasses.replace(view, goal={**view.goal, "port_axis": axis}))


@pytest.mark.parametrize(
    ("name", "key", "value"),
    [
        ("docking", "port", (0.0, -2.0)),
        ("docking", "lateral_limit", -0.1),
        ("docking", "lateral_speed_limit", 0.0),
        ("docking", "axial_speed_window", (0.1, 0.05)),
        ("docking", "axial_speed_window", (-0.1, 0.05)),
        ("docking", "axial_speed_window", (0.05, 0.1, 0.2)),
        ("docking", "contact_range", -1.0),
        ("approach-vbar", "hold_point", (0.0, -50.0)),
        ("approach-vbar", "capture_radius", -1.0),
        ("approach-vbar", "capture_speed", (0.1, 0.1)),
        ("approach-vbar", "hold_time", -1.0),
        ("approach-vbar", "cam_impulses", ((0.3, 0.0),)),
        ("approach-vbar", "cam_impulses", 0.3),
        ("approach-vbar", "thrust_inhibit", True),
        ("inspection", "coverage", 1.5),
        ("inspection", "coverage", 0.0),
        ("inspection", "sun_exclusion", 4.0),
        ("inspection", "points", 0),
        ("separation-flyaround", "clear_range", -10.0),
    ],
)
def test_a_goal_entry_of_the_wrong_shape_or_sign_is_refused(name, key, value):
    view = make(name).agent_view()
    with pytest.raises(
        ValueError,
        match=f"{key} (must have shape|has the wrong shape|must be|is out of range)",
    ):
        check_goal(dataclasses.replace(view, goal={**view.goal, key: value}))


@pytest.mark.parametrize("key", ["hold_time", "sun_direction", "out_of_bounds", "commit_radius"])
def test_an_optional_goal_entry_the_briefing_prints_must_be_finite(key):
    view = make("approach-vbar").agent_view()
    value = (math.nan, 0.0, 0.0) if key == "sun_direction" else math.nan
    with pytest.raises(ValueError, match=f"{key} must be numeric and finite"):
        check_goal(dataclasses.replace(view, goal={**view.goal, key: value}))


def test_a_target_mode_parameter_must_be_a_single_number():
    with pytest.raises(ValueError, match="phase must be a number"):
        target_mode({"mode": "inertial", "phase": [0.1, 0.2]})


@pytest.mark.parametrize(
    ("key", "value"),
    [("plume_range", 0.0), ("cam_impulses", ()), ("hold_time", 0.0), ("commit_radius", 0.0)],
)
def test_a_goal_entry_at_the_edge_of_its_range_is_accepted(key, value):
    view = make("approach-vbar").agent_view()
    edge = dataclasses.replace(view, goal={**view.goal, key: value})
    check_goal(edge)
    assert "Goal entry" not in briefing(edge) and "burns  m/s" not in briefing(edge)
    outside = -1e-9 if np.ndim(value) == 0 else ((0.3, 0.0),)
    with pytest.raises(ValueError, match=key):
        check_goal(dataclasses.replace(view, goal={**view.goal, key: outside}))
