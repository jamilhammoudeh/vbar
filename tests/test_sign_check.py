import dataclasses
import json
import re

import numpy as np
import pytest

from vbar import ApproachConfig
from vbar.agents import DISC27, DriftAgent, LLMAgent, PDAgent
from vbar.agents.llm import text_model
from vbar.agents.sign_check import DIRECTIONS, sign_check

ALL = tuple(label for label, _, _ in DIRECTIONS)


class Mapped:
    """A PD agent whose command passes through a fixed matrix before it is returned."""

    name = "mapped"

    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float)

    def reset(self, view, rng=None):
        self.pd = PDAgent()
        self.pd.reset(view)

    def act(self, obs, info):
        return self.matrix @ self.pd.act(obs, info)


class Discrete:
    """Bang-off-bang from the sign of a normalized PD command."""

    name = "discrete"

    def reset(self, view, rng=None):
        self.pd = PDAgent()
        self.pd.reset(dataclasses.replace(view, action_mode="normalized"))

    def act(self, obs, info):
        target = np.sign(np.round(self.pd.act(obs, info), 6))
        return int(np.flatnonzero(np.all(DISC27 == target, axis=1))[0])


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"action_mode": "accel"},
        {"action_mode": "thrust", "thrust_limit": "ball"},
        {"delay_s": 3.0, "observation_channels": ("t_remaining", "pending_commands")},
    ],
)
def test_pd_agent_passes_in_every_continuous_mode(fields):
    check = sign_check(PDAgent(), ApproachConfig(**fields))
    assert check.passed and set(check.commands) == set(ALL)


def test_drift_fails_every_direction():
    assert sign_check(DriftAgent()).failures == ALL


def test_iss_style_axes_and_a_flipped_radial_axis_are_caught():
    # An agent that treats Hill commands as ISS-style LVLH (x_iss = y, y_iss = -z, z_iss = -x).
    iss = Mapped([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    assert sign_check(iss).failures == ALL
    flipped = Mapped(np.diag([-1.0, 1.0, 1.0]))
    assert sign_check(flipped).failures == ("+x", "-x")


def test_discrete_actions_are_read_through_their_commands():
    assert sign_check(Discrete(), ApproachConfig(action_mode="disc27")).passed
    assert sign_check(DriftAgent(), ApproachConfig(action_mode="disc27")).failures == ALL


def model_toward_goal(prompt):
    """A stand-in model that reads its position and goal from the prompt and points at the goal."""
    position = json.loads(re.search(r"position = (\[.*?\])", prompt).group(1))
    goal = json.loads(re.search(r"hold point (\[.*?\])", prompt).group(1))
    step = np.array(goal) - np.array(position)
    ax, ay, az = step / np.max(np.abs(step))
    return json.dumps({"ax": ax, "ay": ay, "az": az})


def test_llm_agent_passes_when_its_model_reads_the_prompt_correctly():
    assert sign_check(LLMAgent(text_model(model_toward_goal))).passed
    always_up = LLMAgent(text_model(lambda prompt: '{"ax": 1, "ay": 0, "az": 0}'))
    assert sign_check(always_up).failures == ("-x", "+y", "-y", "+z", "-z")
