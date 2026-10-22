"""Random streams of the mutation wrapper: the inner agent keeps the agent's stream, and the dither
draws from a child of it that the inner agent's draws do not move."""

import numpy as np
import pytest

from vbar.agents.base import AgentView
from vbar.agents.mutants import Mutant

STEPS = 6


def view():
    return AgentView(
        scenario_id="hand-made",
        dt=10.0,
        max_steps=360,
        n=1.1e-3,
        action_mode="normalized",
        max_accel=0.02,
        thrust_limit="box",
        goal={"hold_point": (0.0, -100.0, 0.0)},
    )


class Random:
    """Draws a command from its stream at every decision and records it."""

    name = "random"

    def __init__(self, draws_per_act=1):
        self.draws_per_act = draws_per_act

    def reset(self, view, rng=None):
        self.rng = rng
        self.issued = []

    def act(self, obs, info):
        for _ in range(self.draws_per_act):
            command = self.rng.uniform(-0.1, 0.1, size=3)
        self.issued.append(command)
        return command.copy()


def commands(agent, seed):
    agent.reset(view(), np.random.default_rng(seed))
    obs = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    return np.array([agent.act(obs, {"time_s": 10.0 * k}) for k in range(STEPS)])


@pytest.mark.parametrize("kind", ["dither", "chatter", "delay", "position_bias"])
def test_random_inner_agent_draws_the_same_commands_wrapped_and_unwrapped(kind):
    inner = Random()
    commands(Mutant(inner, kind, 0.05), seed=3)
    assert np.array_equal(np.array(inner.issued), commands(Random(), seed=3))


def test_dither_is_unchanged_by_how_much_the_inner_agent_draws():
    offsets = []
    for draws in (1, 4):
        inner = Random(draws)
        wrapped = commands(Mutant(inner, "dither", 0.05), seed=3)
        offsets.append(wrapped - np.array(inner.issued))
    assert np.min(np.abs(offsets[0])) > 0.0
    # One addition and one subtraction of commands of order 0.1: rounding near 1e-17.
    assert np.max(np.abs(offsets[1] - offsets[0])) <= 1e-12
