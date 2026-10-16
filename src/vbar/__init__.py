"""vbar: a fast, headless benchmark for AI and classical spacecraft controllers."""

from .runner import run_episode, run_many
from .scenarios import ApproachConfig, ApproachEnv

__version__ = "0.1.0"
__all__ = ["ApproachConfig", "ApproachEnv", "run_episode", "run_many"]
