"""Safety filters wrap an agent and modify unsafe commands before they are applied.

Only the pass-through filter (filter off) exists so far. `FilteredAgent` counts
interventions, the decisions where the filter changed the agent's command.
"""

from __future__ import annotations

import numpy as np


class PassThroughFilter:
    """Does not change the action. The 'safety filter off' condition."""

    name = "none"

    def filter(self, obs: np.ndarray, action: np.ndarray, env) -> np.ndarray:
        return action


class FilteredAgent:
    """Agent + safety filter, presented as a single agent."""

    def __init__(self, agent, safety_filter, tol: float = 1e-6):
        self.agent = agent
        self.filter = safety_filter
        self.tol = tol
        self.name = f"{agent.name}+{safety_filter.name}"

    def reset(self, env) -> None:
        self.env = env
        self.agent.reset(env)
        self.interventions = 0
        self.decisions = 0

    def act(self, obs, info):
        raw = np.asarray(self.agent.act(obs, info), dtype=float)
        safe = np.asarray(self.filter.filter(obs, raw, self.env), dtype=float)
        self.decisions += 1
        if np.max(np.abs(safe - raw)) > self.tol:
            self.interventions += 1
        return safe
