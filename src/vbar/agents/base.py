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
    asks for at the initial mass `mass_kg`, `goal` and `constraints` the public task description,
    and `observation` the names of the observation entries in order.
    """

    scenario_id: str
    dt: float
    max_steps: int
    n: float
    action_mode: str
    max_accel: float
    thrust_limit: str
    mass_kg: float | None = None
    goal: Mapping[str, Any] = field(default_factory=dict)
    constraints: tuple[Mapping[str, Any], ...] = ()
    observation: tuple[str, ...] = ("x", "y", "z", "vx", "vy", "vz")
    frame: str = "hill"

    def __post_init__(self):
        object.__setattr__(self, "goal", MappingProxyType(dict(self.goal)))
        constraints = tuple(MappingProxyType(dict(c)) for c in self.constraints)
        object.__setattr__(self, "constraints", constraints)


def action_scale(view: AgentView) -> float:
    """Action units per fraction of a full command: 1, m/s^2, or N at the initial mass."""
    if view.action_mode == "normalized":
        return 1.0
    if view.action_mode == "accel":
        return view.max_accel
    if view.action_mode == "thrust":
        if view.mass_kg is None:
            raise ValueError("thrust actions need the view's mass_kg")
        return view.max_accel * view.mass_kg
    raise ValueError(f"unknown action mode {view.action_mode!r}")


class Agent(Protocol):
    name: str

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Called at the start of each episode with the scenario's public description and an
        optional random stream for the agent's own use."""

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Return an action in the view's action mode (by default fractions of `max_accel`).

        `info` holds only agent-safe entries (the time, `time_s`).
        """


class DriftAgent:
    """Does nothing. Useful as a sanity check: the chaser just drifts."""

    name = "drift"

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        pass

    def act(self, obs, info):
        return np.zeros(3)
