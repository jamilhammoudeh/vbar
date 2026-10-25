"""The LLM agent on every task kind: its default prompt describes each goal and measures the
observation against it (`vbar.agents.briefing`, tested in test_briefing.py), and is enough to fly
every kind.

The flights at the end run whole episodes through the agent on drawn validation seeds, with a
stand-in model that reads its state from the prompt text and answers with a reference
controller's command; they take most of half an hour and are marked `slow`.
"""

import dataclasses
import json
import re

import numpy as np
import pytest
from scenario_references import reference_for
from test_briefing import every_scenario, line_of, numbers

from vbar import run_episode
from vbar.agents import LLMAgent, Reply, command_fraction
from vbar.scenarios import InertialPointing, Misbriefed, get, make, seeds
from vbar.scenarios.registry import entry_class

COAST = '{"ax": 0.0, "ay": 0.0, "az": 0.0}'
OLD_TEMPLATE = (
    "Goal: reach {hold} m (speed below {cap_speed} m/s), never within {keep_out} m. "
    "Now at x={x:.1f} y={y:.1f} z={z:.1f}, {dist:.1f} m to go. {reply_format}"
)
TASK_LINES = {
    "hold": "Task: reach the hold point",
    "berth": "Task: berthing",
    "dock": "Task: dock at the target's port",
    "inspect": "Task: inspect the target",
    "depart": "Task: depart from the target's port",
}


class Coasting:
    """A stand-in model that always coasts and keeps the prompts it was sent."""

    def __init__(self):
        self.prompts = []

    def __call__(self, messages, tools):
        self.prompts.append(messages[0]["content"])
        return Reply(COAST)


@pytest.mark.parametrize("name", every_scenario())
def test_the_llm_agent_flies_steps_of_every_catalogue_scenario(name):
    env = make(name, max_steps=2)
    model = Coasting()
    result = run_episode(env, LLMAgent(model), seed=1, agent_seed=2)
    assert len(model.prompts) == len(result["actions"]) >= 1
    assert TASK_LINES[env.agent_view().goal["task"]] in model.prompts[0]


def test_a_goal_without_a_key_its_task_needs_is_refused_at_reset():
    view = make("docking").agent_view()
    goal = {k: v for k, v in view.goal.items() if k != "axial_speed_window"}
    with pytest.raises(ValueError, match="the dock goal has no axial_speed_window"):
        LLMAgent(Coasting()).reset(dataclasses.replace(view, goal=goal))


def test_a_template_with_hold_fields_needs_a_hold_point_and_a_keep_out_radius():
    LLMAgent(Coasting(), prompt=OLD_TEMPLATE).reset(make("approach-vbar").agent_view())
    with pytest.raises(
        ValueError, match="needs a capture_speed, hold_point, which the dock goal lacks"
    ):
        LLMAgent(Coasting(), prompt=OLD_TEMPLATE).reset(make("docking").agent_view())
    # The run time assurance task has a hold point and no keep-out sphere.
    with pytest.raises(ValueError, match="needs a keep-out radius, which the hold goal lacks"):
        LLMAgent(Coasting(), prompt=OLD_TEMPLATE).reset(make("docking-afrl").agent_view())
    LLMAgent(Coasting()).reset(make("docking-afrl").agent_view())


def test_the_prompt_of_a_misbriefed_target_is_the_prompt_of_its_briefing():
    told = InertialPointing(0.3)
    prompts = []
    for target in (Misbriefed(actual=InertialPointing(1.0), briefed=told), told):
        spec = dataclasses.replace(get("docking"), target=target)
        env = entry_class(spec)(spec)
        model = Coasting()
        agent = LLMAgent(model)
        obs, _ = env.reset(seed=5)
        agent.reset(env.agent_view())
        agent.act(obs, env.agent_info())
        prompts.append(model.prompts[0])
    assert prompts[0] == prompts[1]


def test_ball_thrusters_are_described_as_a_vector_limit():
    agent = LLMAgent(Coasting())
    agent.reset(make("docking", thrust_limit="ball").agent_view())
    assert "a vector longer than 1 is shortened to length 1" in agent.thrust
    agent.reset(make("docking").agent_view())
    assert "Thrust per axis is a fraction of the maximum" in agent.thrust


class PromptPilot:
    """A stand-in model that reads the time and state from the prompt text and answers with the
    command of a reference controller given that state.

    The reference reads the goal from the public view; the state it flies on is only what the
    prompt printed, rounded as printed.
    """

    def __init__(self, reference, view):
        self.reference, self.view = reference, view
        reference.reset(view)

    def __call__(self, messages, tools):
        text = messages[0]["content"]
        t = float(re.search(r"\(t = (-?\d+) s\)", text).group(1))
        position = numbers(line_of(text, "position ="))
        velocity = numbers(line_of(text, "velocity ="))
        obs = np.array(position + velocity)
        f = command_fraction(self.view, self.reference.act(obs, {"time_s": t}))
        return Reply(json.dumps({"ax": f[0], "ay": f[1], "az": f[2]}))


# One scenario of each task kind, with the kinds' hard cases: a turning port, a pitched one, a
# lighting window, the run time assurance task (no keep-out sphere), a berthing box, a hold below
# a turning target, inspection, and a departure past a manoeuvring target.
FLIGHTS = [
    "approach-vbar",
    "docking",
    "docking:lit",
    "docking-offset",
    "inertial-target",
    "docking-afrl",
    "berthing-box",
    "hst-rbar-grapple",
    "inspection",
    "separation-flyaround:reboost",
]


@pytest.mark.slow
@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("name", FLIGHTS)
def test_a_model_flying_from_the_prompt_alone_succeeds_at_every_task_kind(name, index):
    env = make(name)
    seed, agent_seed = seeds(name, "validation", 3).pairs()[index]
    env.reset(seed=seed)
    pilot = PromptPilot(reference_for(name.split(":")[0]), env.agent_view())
    agent = LLMAgent(pilot)
    run_episode(env, agent, seed=seed, agent_seed=agent_seed)
    info = env._info()
    assert info["success"], info
    assert agent.usage()["parse_failures"] == 0
