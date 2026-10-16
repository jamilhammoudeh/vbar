"""vbar: a fast, headless benchmark for AI and classical spacecraft controllers."""

from gymnasium.envs.registration import register

from .runner import run_episode, run_many
from .scenarios import ApproachConfig, ApproachEnv

__version__ = "0.2.0"
__all__ = ["ApproachConfig", "ApproachEnv", "run_episode", "run_many"]

register(id="vbar/Approach-v0", entry_point="vbar.scenarios.approach:ApproachEnv")
