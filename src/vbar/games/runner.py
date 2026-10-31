"""Agents in KSPDG's interface: a base class of the same shape and a lockstep runner.

An agent written for KSPDG subclasses `KSPDGBaseAgent` and implements `get_action(observation)`,
returning a KSPDG action or None to keep the last one; KSPDG's `AgentEnvRunner(agent, env_cls,
env_kwargs, runner_timeout, debug, enable_telemetry).run()` flies it. `BaseAgent` and
`AgentEnvRunner` here have the same signatures, and the runner takes any object with that
`get_action`, so such an agent runs unchanged; only the imports differ. KSPDG flies the agent
in real time in a separate process while the game keeps running; this runner calls the agent
once per step in the same process, and simulated time waits for it.

Source: KSPDG v0.15.0 (github.com/mit-ll/spacegym-kspdg, commit 6251ca3),
`src/kspdg/agent_api/base_agent.py` and `runner.py`.
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from .vessels import _number

# What an agent's environment does before the agent's first action: coast for one shortest step.
_COAST = {"burn_vec": [0.0, 0.0, 0.0, 0.0], "ref_frame": 0}


class BaseAgent(ABC):
    """Base class with the shape of KSPDG's `KSPDGBaseAgent`.

    Parameters
    ----------
    debug : bool
        Set the agent's `logger` to DEBUG level.
    """

    def __init__(self, debug: bool = False) -> None:
        self.logger = logging.getLogger(type(self).__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)

    @abstractmethod
    def get_action(self, observation):
        """A KSPDG action for the observation, or None to keep the last action."""


def fly(
    env: Any, agent: Any, runner_timeout: float | None = None, seed: int | None = None
) -> dict[str, Any]:
    """Fly one episode of a KSPDG-format environment, calling the agent once per step.

    The environment is reset, then each step passes the latest observation to
    `agent.get_action(observation=...)` and flies the action it returns. None keeps the last
    action, which is flown again, and before the first action the vessel coasts.

    Parameters
    ----------
    env : PursuitEvasionEnv
        A KSPDG-format environment (`reset()` gives (observation, info), `step` four values).
    agent : object
        Anything with `get_action(observation)`.
    runner_timeout : float, optional
        Wall-clock seconds after which the episode is stopped where it is.
    seed : int, optional
        Passed to `env.reset`; by default the environment's own seed rule applies.

    Returns
    -------
    dict
        The last `info`, with "steps" (number of steps) and "wall_s" (wall-clock seconds).
    """
    if runner_timeout is not None:
        _number(runner_timeout, "runner_timeout")
    started = time.perf_counter()
    observation, info = env.reset() if seed is None else env.reset(seed=seed)
    action = None
    steps = 0
    done = False
    while not done:
        chosen = agent.get_action(observation=observation)
        if chosen is not None:
            action = chosen
        observation, _, done, info = env.step(_COAST if action is None else action)
        steps += 1
        if runner_timeout is not None and time.perf_counter() - started > runner_timeout:
            break
    return {**info, "steps": steps, "wall_s": time.perf_counter() - started}


class AgentEnvRunner:
    """Fly an agent in an environment, with the signature of KSPDG's `AgentEnvRunner`.

    Parameters
    ----------
    agent : object
        Anything with `get_action(observation)`; a `BaseAgent` or a KSPDG agent.
    env_cls : type
        The environment class, built as `env_cls(**env_kwargs, debug=debug)`.
    env_kwargs : dict or None
        Keyword arguments for the environment.
    runner_timeout : float, optional
        Wall-clock seconds after which the run stops.
    debug : bool
        Passed to the environment.
    enable_telemetry : bool
        Accepted for compatibility; runs are headless and have no telemetry stream, so True
        only warns.
    """

    def __init__(
        self,
        agent: Any,
        env_cls: type,
        env_kwargs: Mapping[str, Any] | None,
        runner_timeout: float | None = None,
        debug: bool = False,
        enable_telemetry: bool = False,
    ) -> None:
        if not callable(getattr(agent, "get_action", None)):
            raise TypeError("agent must have a get_action(observation) method")
        if not callable(env_cls):
            raise TypeError("env_cls must be an environment class")
        if env_kwargs is not None and not isinstance(env_kwargs, Mapping):
            raise TypeError("env_kwargs must be a mapping or None")
        if runner_timeout is not None:
            _number(runner_timeout, "runner_timeout")
        self.agent = agent
        self.env_cls = env_cls
        self.env_kwargs = dict(env_kwargs or {})
        self.runner_timeout = runner_timeout
        self.debug = debug
        if enable_telemetry:
            warnings.warn("telemetry is not available in headless runs", stacklevel=2)

    def run(self) -> dict[str, Any]:
        """Fly one episode.

        Returns
        -------
        dict
            {"agent_env_results": info} with the environment's last `info`, as KSPDG's runner
            returns, plus "steps" and "wall_s" inside it.
        """
        env = self.env_cls(**self.env_kwargs, debug=self.debug)
        try:
            info = fly(env, self.agent, self.runner_timeout)
        finally:
            env.close()
        return {"agent_env_results": info}


__all__ = ["AgentEnvRunner", "BaseAgent", "fly"]
