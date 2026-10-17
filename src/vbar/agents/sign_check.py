"""Sign-convention self-check (VALIDATION V12), run on every agent before it is scored.

From rest on the V-bar, the agent is asked through its normal observation and view to reach a
goal 50 m along +x, -x, +y, -y, +z and -z in turn. Its first command must push toward the goal
on that axis. A frame or sign mistake (ISS-style axes, a flipped radial axis) fails here instead
of wasting a full evaluation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..scenarios.approach import ApproachConfig, ApproachEnv
from ..spacecraft import PerfectNav
from .base import Agent, command_fraction

DIRECTIONS = (
    ("+x", 0, 1.0),
    ("-x", 0, -1.0),
    ("+y", 1, 1.0),
    ("-y", 1, -1.0),
    ("+z", 2, 1.0),
    ("-z", 2, -1.0),
)


@dataclass(frozen=True)
class SignCheck:
    """Result of `sign_check`: `commands` maps each direction to the first command (fractions)."""

    commands: Mapping[str, tuple[float, float, float]]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def sign_check(
    agent: Agent,
    config: ApproachConfig | None = None,
    start: tuple[float, float, float] = (0.0, -300.0, 0.0),
    offset: float = 50.0,
    agent_seed: int = 0,
) -> SignCheck:
    """Ask `agent` for its first command toward a goal `offset` m along each Hill axis.

    Runs the approach scenario with `config`'s action mode, thrusters and observation channels,
    from rest at `start` (m, on the V-bar a CW equilibrium) with navigation made exact, so only
    the agent's conventions decide the sign. A direction passes when the command's component on
    that axis has the goal's sign; zero fails.
    """
    base = config or ApproachConfig()
    commands, failures = {}, []
    for label, axis, sign in DIRECTIONS:
        goal = np.array(start, dtype=float)
        goal[axis] += sign * offset
        cfg = dataclasses.replace(
            base,
            start_state=(*start, 0.0, 0.0, 0.0),
            hold_point=tuple(goal),
            sensor=PerfectNav(),
            training=False,
        )
        env = ApproachEnv(cfg)
        obs, _ = env.reset(seed=0)
        view = env.agent_view()
        agent.reset(view, np.random.default_rng(agent_seed))
        fraction = command_fraction(view, agent.act(obs, env.agent_info()))
        # Adding 0.0 turns -0.0 into 0.0 for display.
        commands[label] = (
            float(fraction[0]) + 0.0,
            float(fraction[1]) + 0.0,
            float(fraction[2]) + 0.0,
        )
        if not sign * fraction[axis] > 0.0:
            failures.append(label)
    return SignCheck(commands, tuple(failures))
