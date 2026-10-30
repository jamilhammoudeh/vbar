"""The pe1 scenarios: KSPDG-format classes, the Gymnasium environment and the evaluation task.

The classes carry KSPDG's names (`PE1_E3_I3_Env` is evader e3 from start I3), so a KSPDG agent
script needs only its imports changed, and are built as KSPDG's are: a parent per evader and a
class per start. They fly V-BAR's model with V-BAR's versions of KSPDG's evaders; the differences
are listed under Limits in `docs/pursuit-evasion.md`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .bots import NormalEscapeEvader, PassiveEvader
from .pursuit import PursuitEvasionEnv
from .runner import fly


class PE1_E1_ParentEnv(PursuitEvasionEnv):  # noqa: N801  KSPDG's class name
    """pe1 against V-BAR's `PassiveEvader` (KSPDG's e1)."""

    initial = "i3"

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("evader", PassiveEvader())
        super().__init__(initial=self.initial, **kwargs)


class PE1_E3_ParentEnv(PursuitEvasionEnv):  # noqa: N801  KSPDG's class name
    """pe1 against V-BAR's `NormalEscapeEvader` (KSPDG's e3).

    Parameters
    ----------
    evade_dist : float
        Evasion distance (m), KSPDG's argument; used when no `evader` is given.
    evade_throttle : float
        Forward throttle in [0, 1] while evading, KSPDG's argument; used when no `evader` is
        given.
    **kwargs
        `PursuitEvasionEnv` parameters except `initial`.
    """

    initial = "i3"

    def __init__(self, evade_dist: float = 200.0, evade_throttle: float = 1.0, **kwargs: Any):
        if "evader" not in kwargs:
            kwargs["evader"] = NormalEscapeEvader(
                evade_distance=evade_dist, throttle=evade_throttle
            )
        super().__init__(initial=self.initial, **kwargs)


class PE1_E1_I1_Env(PE1_E1_ParentEnv):  # noqa: N801  KSPDG's class name
    """pe1 from KSPDG's I1 start against the passive evader."""

    initial, scenario = "i1", "pe1-e1-i1"


class PE1_E1_I2_Env(PE1_E1_ParentEnv):  # noqa: N801  KSPDG's class name
    """pe1 from KSPDG's I2 start against the passive evader."""

    initial, scenario = "i2", "pe1-e1-i2"


class PE1_E1_I3_Env(PE1_E1_ParentEnv):  # noqa: N801  KSPDG's class name
    """pe1 from KSPDG's I3 start against the passive evader."""

    initial, scenario = "i3", "pe1-e1-i3"


class PE1_E1_I4_Env(PE1_E1_ParentEnv):  # noqa: N801  KSPDG's class name
    """pe1 from KSPDG's I4 start against the passive evader."""

    initial, scenario = "i4", "pe1-e1-i4"


class PE1_E3_I1_Env(PE1_E3_ParentEnv):  # noqa: N801  KSPDG's class name
    """pe1 from KSPDG's I1 start against the normal-escape evader."""

    initial, scenario = "i1", "pe1-e3-i1"


class PE1_E3_I2_Env(PE1_E3_ParentEnv):  # noqa: N801  KSPDG's class name
    """pe1 from KSPDG's I2 start against the normal-escape evader."""

    initial, scenario = "i2", "pe1-e3-i2"


class PE1_E3_I3_Env(PE1_E3_ParentEnv):  # noqa: N801  KSPDG's class name
    """pe1 from KSPDG's I3 start against the normal-escape evader."""

    initial, scenario = "i3", "pe1-e3-i3"


class PE1_E3_I4_Env(PE1_E3_ParentEnv):  # noqa: N801  KSPDG's class name
    """pe1 from KSPDG's I4 start against the normal-escape evader."""

    initial, scenario = "i4", "pe1-e3-i4"


SCENARIOS: dict[str, type[PursuitEvasionEnv]] = {
    cls.scenario: cls
    for cls in (
        PE1_E1_I1_Env,
        PE1_E1_I2_Env,
        PE1_E1_I3_Env,
        PE1_E1_I4_Env,
        PE1_E3_I1_Env,
        PE1_E3_I2_Env,
        PE1_E3_I3_Env,
        PE1_E3_I4_Env,
    )
}
# "pe1-e3-i3" is registered as "vbar/PursuitEvasionE3I3-v1".
GYMNASIUM_IDS = {
    key: "vbar/PursuitEvasion{}{}-v1".format(*key.upper().split("-")[1:]) for key in SCENARIOS
}


class PursuitEvasionGymEnv(gym.Env):
    """A pe1 scenario as a Gymnasium environment.

    The observation is the KSPDG observation as a float64 array of 15; the action is KSPDG's
    dictionary (`burn_vec`, `ref_frame`, `vec_type`). An episode terminates at capture and is
    truncated at the timeout. The seed of `reset` seeds the evader's random stream; without one,
    the stream's seed is drawn from `np_random`, so a seeded first reset fixes every later one.

    Parameters
    ----------
    scenario : str
        A key of `SCENARIOS`.
    **kwargs
        Passed to the scenario's class (`PursuitEvasionEnv` parameters except `initial`).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, scenario: str = "pe1-e3-i3", **kwargs: Any):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {sorted(SCENARIOS)}")
        self.game = SCENARIOS[scenario](**kwargs)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float64)
        self.action_space = gym.spaces.Dict(
            {
                "burn_vec": gym.spaces.Box(
                    np.array([-1.0, -1.0, -1.0, 0.0]),
                    np.array([1.0, 1.0, 1.0, 10.0]),
                    dtype=np.float64,
                ),
                "ref_frame": gym.spaces.Discrete(3),
                "vec_type": gym.spaces.Discrete(2),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Start an episode at the scenario's initial conditions.

        Parameters
        ----------
        seed : int, optional
            Seeds `np_random` and the evader's random stream.
        options : dict, optional
            Unused, as Gymnasium allows.

        Returns
        -------
        observation : numpy.ndarray, shape (15,)
        info : dict
        """
        super().reset(seed=seed)
        if seed is None:
            seed = int.from_bytes(self.np_random.bytes(16), "little")
        observation, info = self.game.reset(seed=seed)
        return np.array(observation), info

    def step(self, action):
        """Fly one KSPDG action (`vbar.games.parse_action`).

        Returns
        -------
        observation : numpy.ndarray, shape (15,)
        reward : float
            The negative weighted score on the last step, else 0.
        terminated : bool
            The evader was captured.
        truncated : bool
            The timeout was reached without a capture.
        info : dict
        """
        observation, reward, done, info = self.game.step(action)
        captured = self.game.captured
        return np.array(observation), reward, done and captured, done and not captured, info


@dataclass(frozen=True)
class PursuitTask:
    """The pe1 scenarios as a task of the evaluation harness (`vbar.eval.run_jobs`).

    Agents are KSPDG-format (`get_action(observation)`). A job's seed seeds the evader's random
    stream; KSPDG agents draw nothing, so its agent seed is recorded but unused. Success is a
    capture.

    Parameters
    ----------
    name : str
        The task's name in result files.
    """

    name: str = "pursuit-evasion"

    def make(self, scenario: str) -> PursuitEvasionEnv:
        """A new environment of `scenario`, a key of `SCENARIOS`."""
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {sorted(SCENARIOS)}")
        return SCENARIOS[scenario]()

    def record(self, env: Any, agent: Any, seed: int, agent_seed: int) -> dict[str, Any]:
        """One episode flown by `fly`; returns its last info."""
        return fly(env, agent, seed=seed)

    def score(self, env: Any, episode: Mapping[str, Any]) -> dict[str, Any]:
        """The KSPDG score terms of the episode, its step count and whether it captured."""
        keys = (
            "weighted_score",
            "closest_approach",
            "closest_approach_time",
            "closest_approach_speed",
            "closest_approach_pursuer_fuel_usage",
            "pursuer_fuel_usage",
            "evader_fuel_usage",
            "steps",
        )
        return {"success": env.captured, **{key: episode[key] for key in keys}}

    def identity(self, env: Any) -> dict[str, Any]:
        """The scenario, the Gymnasium identifier, the evader bot and the dynamics."""
        return {
            "scenario_id": GYMNASIUM_IDS[env.scenario],
            "evader": env.evader.name,
            "dynamics": "two-body",
        }


__all__ = [
    "GYMNASIUM_IDS",
    "PE1_E1_I1_Env",
    "PE1_E1_I2_Env",
    "PE1_E1_I3_Env",
    "PE1_E1_I4_Env",
    "PE1_E1_ParentEnv",
    "PE1_E3_I1_Env",
    "PE1_E3_I2_Env",
    "PE1_E3_I3_Env",
    "PE1_E3_I4_Env",
    "PE1_E3_ParentEnv",
    "SCENARIOS",
    "PursuitEvasionGymEnv",
    "PursuitTask",
]
