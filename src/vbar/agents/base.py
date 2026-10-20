"""Agent interface. Every controller (classical, learned, or LLM) implements this."""

from __future__ import annotations

import hashlib
import itertools
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

# The "disc27" action mode: bang-off-bang fractions {-1, 0, 1}^3 per Hill axis; 13 is no thrust.
DISC27 = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=3)))
CONTINUOUS_MODES = ("normalized", "accel", "thrust")


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


def agent_generator(agent_seed: int) -> np.random.Generator:
    """The agent's random stream for `agent_seed`.

    An agent can read the seed sequence behind its generator, so the stream is seeded with a
    hash of `agent_seed`: the agent never learns `agent_seed`, nor an episode seed equal to it.
    """
    seed = operator.index(agent_seed)
    if seed < 0:
        raise ValueError("agent_seed must not be negative")
    digest = hashlib.sha256(b"vbar agent stream" + seed.to_bytes((seed.bit_length() + 7) // 8 or 1))
    return np.random.default_rng(int.from_bytes(digest.digest()[:16]))


def action_scale(view: AgentView) -> float:
    """Action units per fraction of a full command: 1, m/s^2, or N at the initial mass.

    Only the continuous modes have one; agents that produce continuous commands use this to
    reject the "disc27" and "impulse_coast" modes.
    """
    if view.action_mode == "normalized":
        return 1.0
    if view.action_mode == "accel":
        return view.max_accel
    if view.action_mode == "thrust":
        if view.mass_kg is None:
            raise ValueError("thrust actions need the view's mass_kg")
        return view.max_accel * view.mass_kg
    if view.action_mode in ("disc27", "impulse_coast"):
        raise ValueError(f"action mode {view.action_mode!r} takes no continuous command")
    raise ValueError(f"unknown action mode {view.action_mode!r}")


def command_fraction(view: AgentView, action) -> np.ndarray:
    """The thrust an action asks for, in fractions of a full command at the initial mass."""
    if view.action_mode == "disc27":
        return DISC27[int(action)].copy()
    if view.action_mode == "impulse_coast":
        return np.asarray(action, dtype=float)[:3].copy()
    return np.asarray(action, dtype=float) / action_scale(view)


def zero_action(view: AgentView):
    """The action that asks for no thrust, in the view's action mode."""
    if view.action_mode == "disc27":
        return 13
    return np.zeros(4 if view.action_mode == "impulse_coast" else 3)


class Agent(Protocol):
    name: str

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode with the scenario's public description and the agent's own stream."""

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Return an action in the view's action mode (by default fractions of `max_accel`).

        `info` holds only agent-safe entries (the time, `time_s`).
        """


class DriftAgent:
    """Does nothing. Useful as a sanity check: the chaser just drifts."""

    name = "drift"
    view: AgentView | None = None

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        self.view = view

    def act(self, obs, info):
        return np.zeros(3) if self.view is None else zero_action(self.view)
