"""Training wrappers: a safety filter inside the learner's loop.

When a filter changes the learner's action, the transition the environment makes follows the
filtered command, so an off-policy learner must store that command, not its own output (Sutton
and Barto, Reinforcement Learning, 2nd ed., 2018, sec. 6.5: the update uses the action
actually taken). The filter sees only what the learner saw.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .agents.base import CONTINUOUS_MODES
from .agents.learned_supervisor import supervisor_features
from .agents.supervision import PHASES, Command, Situation, SupervisedAgent, SupervisionStep
from .agents.tracker import SetpointTracker
from .safety import run_filter
from .scenarios.base import BaseEnv, continuous_command


class FilteredEnv(gym.Wrapper):
    """A V-BAR environment whose actions pass through `safety_filter` before they are flown.

    `info["executed_action"]` is the filtered command in the action mode's units and
    `info["intervened"]` the filter's own report in its `FilterResult`. The filter gets
    `reset(view)` at every reset and `filter(obs, action)` with the observation the learner acted
    on. `env.record` keeps the learner's command, the filtered one and the result.

    Parameters
    ----------
    env : gymnasium.Env
        A V-BAR environment in a continuous action mode.
    safety_filter : SafetyFilter
        Has `reset(view)` and `filter(obs, action)` returning a `FilterResult`.

    Raises
    ------
    ValueError
        For an action mode that is not continuous.
    TypeError
        At `step`, when the filter returns anything but a `FilterResult`, or a field of the
        result is not of its exact type.
    ValueError
        At `step`, when the filter's action is not numeric, not finite or not of the command's
        shape, or the rebuilt result fails the checks of `FilterResult`.
    """

    def __init__(self, env: gym.Env, safety_filter):
        super().__init__(env)
        self.base = cast(BaseEnv, env.unwrapped)
        mode = self.base.cfg.action_mode
        if mode not in CONTINUOUS_MODES:
            raise ValueError(f"filters need a continuous action mode, not {mode!r}")
        self.safety_filter = safety_filter

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.safety_filter.reset(self.base.agent_view())
        self.last_obs = obs
        return obs, info

    def step(self, action):
        command = continuous_command(action, self.base.cfg.action_mode)
        result, _ = run_filter(self.safety_filter, self.last_obs, command)
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


# The inner observation space is bounded by 1e7; the gap to a rung can add as much again.
_BOUND = 2e7


class _Relay:
    """The supervisor inside a `SupervisorEnv`: gives the command the learner chose."""

    name = "learner"

    def __init__(self):
        self.command = Command.CONTINUE

    def reset(self, view, ladder, rng=None):
        pass

    def decide(self, situation: Situation) -> Command:
        return self.command


def _check_penalty(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"refused_penalty must be a number, got {value!r}")
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"refused_penalty must be finite and not below zero, got {value!r}")
    return float(value)


class SupervisorEnv(gym.Wrapper):
    """A V-BAR environment whose actions are supervisory commands, one at each consult.

    The learner picks one of the `Command` values; the wrapper gives it to a `SupervisedAgent`
    and flies the inner environment until that agent next consults its supervisor, which is the
    next scheduled consult or a change of phase or rung, whichever comes first. The observation
    is `vbar.agents.learned_supervisor.supervisor_features` of the situation at the consult, so
    it holds nothing of the inner `info`. After the episode ends it is built from the final
    observation and the agent's latest phase, rung and mask.

    The step's reward is the sum of the inner rewards, each discounted by the inner discounts
    before it, so sum_i r_i prod_{j<i} d_j, and `info["discount"]` is the product of all the
    inner discounts (1 where the inner environment has none). An order the mask refuses is
    ignored by the agent and costs `refused_penalty` more. `reward_hook` is called once per step
    with the `SupervisionStep` of the consult and its result is added.

    `action_masks()` is the bool mask of the commands valid now (the sb3-contrib convention) and
    `info["action_mask"]` the same as int8 (the `Discrete.sample` convention). Only the
    continuous action modes are supported.

    Parameters
    ----------
    env : gymnasium.Env
        A V-BAR environment.
    tracker : SetpointTracker, optional
        Flies the accepted commands; copied. Default gains if None.
    period_s : float
        Time between scheduled consults (s); a whole number of the environment's steps.
    refused_penalty : float
        Reward (a magnitude, not below zero) subtracted when the command is one the mask refuses.
    reward_hook : callable, optional
        Maps the step's `SupervisionStep` to a reward added to the step's.

    Raises
    ------
    ValueError
        If the action mode is not continuous, `period_s` is not a positive whole number of steps,
        `refused_penalty` is not a finite number not below zero, or `reward_hook` is not
        callable.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        tracker: SetpointTracker | None = None,
        period_s: float = 60.0,
        refused_penalty: float = 0.0,
        reward_hook: Callable[[SupervisionStep], float] | None = None,
    ):
        super().__init__(env)
        self.base = cast(BaseEnv, env.unwrapped)
        mode = self.base.cfg.action_mode
        if mode not in CONTINUOUS_MODES:
            raise ValueError(f"supervision needs a continuous action mode, not {mode!r}")
        self.refused_penalty = _check_penalty(refused_penalty)
        if reward_hook is not None and not callable(reward_hook):
            raise ValueError("reward_hook must be callable")
        self.reward_hook = reward_hook
        self.tracker = tracker
        self.period_s = period_s
        # Built once so a bad tracker or period fails here and not at the first reset.
        self._new_agent(self.base.agent_view())
        length = len(self.base.observation_names) + 3 + len(PHASES) + 1 + len(Command)
        self.action_space = spaces.Discrete(len(Command))
        self.observation_space = spaces.Box(-_BOUND, _BOUND, shape=(length,), dtype=np.float64)
        self._agent: SupervisedAgent | None = None

    def _new_agent(self, view) -> SupervisedAgent:
        agent = SupervisedAgent(_Relay(), tracker=self.tracker, period_s=self.period_s)
        agent.reset(view)
        return agent

    def _features(self) -> np.ndarray:
        return supervisor_features(self._situation, self._view)

    def _info(self, info: dict) -> dict:
        info["action_mask"] = np.array(self._situation.mask, dtype=np.int8)
        return info

    def action_masks(self) -> np.ndarray:
        """Whether each command, indexed by `Command`, may be given now.

        Returns
        -------
        numpy.ndarray
            Shape (len(Command),), bool.

        Raises
        ------
        RuntimeError
            If the environment has not been reset.
        """
        if self._agent is None:
            raise RuntimeError("reset the environment before asking for the mask")
        return np.array(self._situation.mask, dtype=bool)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._view = self.base.agent_view()
        self._agent = self._new_agent(self._view)
        self._obs = obs
        # The first look of an episode always consults.
        self._situation = cast(Situation, self._agent.pending(obs, self.base.agent_info()))
        return self._features(), self._info(info)

    def step(self, action):
        if self._agent is None:
            raise RuntimeError("reset the environment before stepping")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"the action must be an integer command in 0 to 6, got {action!r}")
        command = Command(int(action))
        refused = not self._situation.mask[command]
        cast(_Relay, self._agent.supervisor).command = command
        flown = self._agent.act(self._obs, self.base.agent_info())
        consult = cast(SupervisionStep, self._agent.last_step)
        reward, discount = 0.0, 1.0
        while True:
            obs, inner, terminated, truncated, info = self.env.step(flown)
            reward += discount * inner
            discount *= info.get("discount", 1.0)
            self._obs = obs
            if terminated or truncated:
                last = cast(SupervisionStep, self._agent.last_step)
                self._situation = dataclasses.replace(
                    self._situation,
                    t=self.base.t,
                    obs=np.array(obs, dtype=float),
                    phase=last.phase,
                    rung=last.rung,
                    mask=last.mask,
                )
                break
            pending = self._agent.pending(obs, self.base.agent_info())
            if pending is not None:
                self._situation = pending
                break
            flown = self._agent.act(obs, self.base.agent_info())
        if refused:
            reward -= self.refused_penalty
        if self.reward_hook is not None:
            reward += float(self.reward_hook(consult))
        info["discount"] = discount
        return self._features(), reward, terminated, truncated, self._info(info)
