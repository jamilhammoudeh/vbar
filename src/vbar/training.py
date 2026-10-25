"""Training wrappers: a safety filter inside the learner's loop.

When a filter changes the learner's action, the transition the environment makes follows the
filtered command, so an off-policy learner must store that command, not its own output (Sutton
and Barto, Reinforcement Learning, 2nd ed., 2018, sec. 6.5: the update uses the action
actually taken). The filter sees only what the learner saw.
"""

from __future__ import annotations

import dataclasses
from typing import cast

import gymnasium as gym
import numpy as np

from .agents.base import CONTINUOUS_MODES
from .safety import run_filter
from .scenarios.base import BaseEnv, continuous_command


class FilteredEnv(gym.Wrapper):
    """A V-BAR environment whose actions pass through `safety_filter` before they are flown.

    `info["executed_action"]` is the filtered command in the action mode's units and
    `info["intervened"]` whether it differs from the learner's action by more than `tol` on any
    axis. The filter gets `reset(view)` at every reset and `filter(obs, action)` with the
    observation the learner acted on. `env.record` keeps the learner's command and the filtered
    one. Only the continuous action modes are supported.
    """

    def __init__(self, env: gym.Env, safety_filter, tol: float = 1e-6):
        super().__init__(env)
        self.base = cast(BaseEnv, env.unwrapped)
        mode = self.base.cfg.action_mode
        if mode not in CONTINUOUS_MODES:
            raise ValueError(f"filters need a continuous action mode, not {mode!r}")
        self.safety_filter = safety_filter
        self.tol = tol

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.safety_filter.reset(self.base.agent_view())
        self.last_obs = obs
        return obs, info

    def step(self, action):
        command = continuous_command(action, self.base.cfg.action_mode)
        result, _ = run_filter(self.safety_filter, self.last_obs, command, self.tol)
        executed = np.array(result.action, dtype=float)
        obs, reward, terminated, truncated, info = self.env.step(executed)
        record = self.base.record
        self.base.record = dataclasses.replace(
            record, command=command, filtered=executed, filter_result=result
        )
        info["executed_action"] = executed.copy()
        info["intervened"] = bool(result.intervened)
        self.last_obs = obs
        return obs, reward, terminated, truncated, info
