"""vbar: a fast, headless benchmark for AI and classical spacecraft controllers."""

from .envs import ApproachConfig, ApproachEnv
from .runner import run_episode, run_many

__version__ = "0.0.1"
__all__ = ["ApproachConfig", "ApproachEnv", "run_episode", "run_many"]
