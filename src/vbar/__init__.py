"""vbar: a fast, headless benchmark for AI and classical spacecraft controllers."""

from gymnasium.envs.registration import register

from .runner import RealTime, ScoredRun, run_episode, run_many
from .scenarios import ApproachConfig, ApproachEnv

__version__ = "0.5.0"
__all__ = ["ApproachConfig", "ApproachEnv", "RealTime", "ScoredRun", "run_episode", "run_many"]

register(id="vbar/Approach-v0", entry_point="vbar.scenarios.approach:ApproachEnv")
# The pe1 games, as `vbar.games.GYMNASIUM_IDS` lists them; written out so that importing vbar
# does not import the games.
for _evader in (1, 2, 3, 4):
    for _start in (1, 2, 3, 4):
        register(
            id=f"vbar/PursuitEvasionE{_evader}I{_start}-v1",
            entry_point="vbar.games.envs:PursuitEvasionGymEnv",
            kwargs={"scenario": f"pe1-e{_evader}-i{_start}"},
        )
