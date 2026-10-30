"""vbar: a fast, headless benchmark for AI and classical spacecraft controllers."""

from gymnasium.envs.registration import register

from .runner import RealTime, ScoredRun, run_episode, run_many
from .scenarios import ApproachConfig, ApproachEnv

__version__ = "0.3.1"
__all__ = ["ApproachConfig", "ApproachEnv", "RealTime", "ScoredRun", "run_episode", "run_many"]

register(id="vbar/Approach-v0", entry_point="vbar.scenarios.approach:ApproachEnv")
for _scenario, _name in (("pe1-e1-i3", "PursuitEvasionE1I3"), ("pe1-e3-i3", "PursuitEvasionE3I3")):
    register(
        id=f"vbar/{_name}-v1",
        entry_point="vbar.games.envs:PursuitEvasionGymEnv",
        kwargs={"scenario": _scenario},
    )
