"""KSPDG-format games on two-body dynamics: the pe1 pursuit-evasion problem."""

from .bots import EvaderBot, NormalEscapeEvader, PassiveEvader
from .envs import (
    GYMNASIUM_IDS,
    SCENARIOS,
    PE1_E1_I3_Env,
    PE1_E3_I3_Env,
    PursuitEvasionGymEnv,
    PursuitTask,
)
from .frames import body_axes, ntw_axes
from .pursuit import (
    CAPTURE_DIST,
    EPISODE_TIMEOUT,
    OBSERVATION,
    PursuitEvasionEnv,
    parse_action,
    weighted_score,
)
from .runner import AgentEnvRunner, BaseAgent, fly
from .vessels import PE1_VESSEL, Vessel

__all__ = [
    "CAPTURE_DIST",
    "EPISODE_TIMEOUT",
    "GYMNASIUM_IDS",
    "OBSERVATION",
    "PE1_E1_I3_Env",
    "PE1_E3_I3_Env",
    "PE1_VESSEL",
    "SCENARIOS",
    "AgentEnvRunner",
    "BaseAgent",
    "EvaderBot",
    "NormalEscapeEvader",
    "PassiveEvader",
    "PursuitEvasionEnv",
    "PursuitEvasionGymEnv",
    "PursuitTask",
    "Vessel",
    "body_axes",
    "fly",
    "ntw_axes",
    "parse_action",
    "weighted_score",
]
