"""Agent interface. Every controller (classical, learned, or LLM) implements this."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class AgentView:
    """What an agent may know about the scenario: its public parameters, never the true state.

    `n` is the reference mean motion (rad/s), `max_accel` the acceleration (m/s^2) a command of 1
    asks for, `goal` and `constraints` the public task description, and `observation` the
    names of the observation entries in order.
    """

    scenario_id: str
    dt: float
    max_steps: int
    n: float
    action_mode: str
    max_accel: float
    thrust_limit: str
    goal: Mapping[str, Any] = field(default_factory=dict)
    constraints: tuple[Mapping[str, Any], ...] = ()
    observation: tuple[str, ...] = ("x", "y", "z", "vx", "vy", "vz")
    frame: str = "hill"

    def __post_init__(self):
        object.__setattr__(self, "goal", MappingProxyType(dict(self.goal)))
        constraints = tuple(MappingProxyType(dict(c)) for c in self.constraints)
        object.__setattr__(self, "constraints", constraints)


class Agent(Protocol):
    name: str

    def reset(self, env) -> None:
        """Called at the start of each episode. `env` gives access to its config."""

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Return an action: 3 numbers in [-1, 1] (fraction of max thrust per axis)."""


class DriftAgent:
    """Does nothing. Useful as a sanity check: the chaser just drifts."""

    name = "drift"

    def reset(self, env) -> None:
        pass

    def act(self, obs, info):
        return np.zeros(3)
