"""The briefing of the capture task and the departure's drift radius."""

import dataclasses
import math
import re

import numpy as np
import pytest
from test_briefing import line_of, numbers

from vbar.agents.briefing import briefing, check_goal, situation, target_mode
from vbar.scenarios import make
from vbar.scenarios.targets import GravityGradientTumble, TorqueFree, briefed_mode


def reset(name):
    env = make(name)
    obs, _ = env.reset(seed=3)
    return env, obs


def test_the_capture_sentence_names_the_capture_point_and_its_limits():
    env, _ = reset("tumbling-capture")
    view = env.agent_view()
    text = briefing(view)
    point = env.capture_point_at(0.0)[0]
    (line,) = [ln for ln in text.splitlines() if ln.startswith("Task: capture the target")]
    assert numbers(line)[:3] == pytest.approx(point, abs=6e-3)
    assert "within 0.2 m of the capture point" in line
    assert "below 0.02 m/s" in line and "for 30 s" in line
    capsule = line_of(text, "- Never come within 1.5 m of the target's body")
    assert "capsule" in capsule and "turns with the target" in capsule
    assert "torque" in text.splitlines()[1]


@pytest.mark.parametrize("name", ["tumbling-capture", "tumbling-capture:libration"])
def test_the_rebuilt_tumble_turns_as_the_scenarios_briefed_mode_does(name):
    env, _ = reset(name)
    rebuilt = target_mode(env.agent_view().goal["target"])
    truth = briefed_mode(env.target)
    assert isinstance(rebuilt, GravityGradientTumble if "libration" in name else TorqueFree)
    t = np.array([0.0, 37.0, 1234.5, 5000.0])
    assert np.allclose(rebuilt.matrix(t, env.n), truth.matrix(t, env.n), rtol=0.0, atol=1e-12)


def test_the_distance_to_the_capture_point_is_measured_at_the_time_of_the_observation():
    env, obs = reset("tumbling-capture")
    for t in (0.0, 900.0, 3000.0):
        text = situation(env.agent_view(), obs, {"time_s": t})
        point, velocity = env.capture_point_at(t)
        # Printed to 1 mm, 0.1 mm/s.
        here, speed = numbers(line_of(text, "distance to the capture point"))
        assert here == pytest.approx(np.linalg.norm(obs[:3] - point), abs=6e-4)
        assert speed == pytest.approx(np.linalg.norm(obs[3:6] - velocity), abs=6e-5)
        assert numbers(line_of(text, "capture point now at"))[:3] == pytest.approx(point, abs=6e-3)
    # The point has moved by far more than its printed rounding between t = 0 and t = 900 s.
    assert np.linalg.norm(env.capture_point_at(900.0)[0] - env.capture_point_at(0.0)[0]) > 1.0


@pytest.mark.parametrize(
    "change",
    [
        {"moments": (1000.0, math.nan, 4700.0)},
        {"moments": (1000.0, 4200.0)},
        {"moments": "abc"},
        {"effective_inertia": math.inf},
        {"rate": True},
        {"spin_angle": (0.0, 1.0)},
        {"momentum_direction": (0.0, 0.0, 0.0)},
        {"momentum_direction": (0.0, 1.0)},
        {"phase": math.nan},
    ],
)
def test_a_torque_free_mode_with_a_hostile_parameter_is_refused(change):
    description = make("tumbling-capture").agent_view().goal["target"]
    assert description["mode"] == "torque_free"
    target_mode(description)
    with pytest.raises(ValueError, match="numeric and finite|nonzero 3-vector|must be a number"):
        target_mode({**description, **change})


def test_a_torque_free_mode_missing_a_parameter_is_refused():
    description = dict(make("tumbling-capture").agent_view().goal["target"])
    del description["spin_angle"]
    with pytest.raises(ValueError, match="the torque_free target mode needs 'spin_angle'"):
        target_mode(description)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("capture_point", (1.0, 2.0)),
        ("interface", (math.nan, 0.0, 0.0)),
        ("body_ends", (-3.0, 3.0)),
        ("body_ends", ((0.0, 0.0, 0.0), (math.inf, 0.0, 0.0))),
        ("body_radius", 0.0),
        ("capture_radius", -0.1),
        ("capture_speed", 0.0),
        ("hold_time", -1.0),
    ],
)
def test_a_capture_goal_entry_of_the_wrong_shape_or_sign_is_refused(key, value):
    view = make("tumbling-capture").agent_view()
    check_goal(view)
    with pytest.raises(ValueError, match=key):
        check_goal(dataclasses.replace(view, goal={**view.goal, key: value}))


def test_the_departure_sentence_gives_the_drift_radius_the_goal_carries():
    view = make("departure").agent_view()
    radius = view.goal["drift_radius"]
    text = briefing(view)
    assert f"does not come within {radius:g} m of the target's centre for" in text
    assert "Goal entry drift_radius" not in text
    goal = {k: v for k, v in view.goal.items() if k != "drift_radius"}
    assert "does not come back into the keep-out zone for" in briefing(
        dataclasses.replace(view, goal=goal)
    )


@pytest.mark.parametrize("name", ["tumbling-capture", "tumbling-capture:libration"])
def test_the_printed_body_rate_and_axes_are_those_of_the_public_description(name):
    view = make(name).agent_view()
    text = briefing(view)
    rows = [
        [float(x) for x in group.split(",")]
        for group in re.findall(r"\[([^\]]*)\]", text[text.index("At t = 0 its angular") :])
    ]
    # Printed to four significant figures.
    assert rows[0] == pytest.approx(view.goal["target"]["omega"], rel=1e-3, abs=1e-9)
    axes = np.array(rows[1:4])
    turn = target_mode(view.goal["target"]).matrix(0.0, view.n)
    # Printed to four decimals; each printed row is a body axis in Hill axes.
    assert axes == pytest.approx(turn.T, abs=6e-5)
