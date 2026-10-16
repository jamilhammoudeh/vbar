"""Agent interface. Every controller (classical, learned, or LLM) implements this."""

from __future__ import annotations

from typing import Protocol

import numpy as np


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
