"""Run episodes and collect benchmark metrics."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

import numpy as np

from .agents.base import agent_generator


@dataclass(frozen=True, kw_only=True)
class RealTime:
    """The simulation clock runs while the agent decides.

    Each command takes effect `time_scale` times the agent's wall-clock decision time (s) after
    the observation it answers, plus any actuation delay; meanwhile the previous command stays
    on. `latencies` (wall s, one per decision) replace the measured times, to repeat a recorded
    run exactly.
    """

    time_scale: float = 1.0
    latencies: tuple[float, ...] | None = None

    def __post_init__(self):
        if not (np.isfinite(self.time_scale) and self.time_scale >= 0.0):
            raise ValueError("time_scale must be finite and not negative")


class Episode:
    """One scored episode, stepped by its caller; the reward never leaves it.

    The run's agent stream is seeded by `agent_seed` alone, so it carries nothing about the
    start state or the sensor noise; without one, a seed is drawn from the operating system and
    returned in the result. Environments built for training (`training=True`) are refused.
    """

    def __init__(self, env, seed: int | None = None, agent_seed: int | None = None):
        config = getattr(getattr(env, "unwrapped", env), "cfg", None)
        if getattr(config, "training", False):
            raise ValueError("scored runs need an environment built with training=False")
        self.env = env
        self.seed = seed
        self.agent_seed = secrets.randbits(128) if agent_seed is None else agent_seed
        self.obs, self.info = env.reset(seed=seed)
        self.states = [self.info["true_state"]]
        self.actions: list[np.ndarray] = []
        self.latencies: list[float] = []
        self.done = False
        self.started = time.perf_counter()

    def agent_rng(self) -> np.random.Generator:
        return agent_generator(self.agent_seed)

    def step(self, action, wall_s: float = 0.0, latency_s: float | None = None) -> np.ndarray:
        """Fly `action` and return the next observation.

        `wall_s` is the agent's decision time; `latency_s` delays the command's effect, in
        real-time runs only.
        """
        if self.done:
            raise RuntimeError("the episode has ended")
        action = np.asarray(action, dtype=float)
        if latency_s is None:
            self.obs, _, terminated, truncated, self.info = self.env.step(action)
        else:
            self.obs, _, terminated, truncated, self.info = self.env.step(
                action, latency_s=latency_s
            )
        self.states.append(self.info["true_state"])
        self.actions.append(action)
        self.latencies.append(wall_s)
        self.done = terminated or truncated
        return self.obs

    def result(self, agent_name: str) -> dict:
        info = self.info
        return {
            "agent": agent_name,
            "seed": self.seed,
            "agent_seed": self.agent_seed,
            "captured": info["captured"],
            "keep_out_violated": info["keep_out_violated"],
            "violation_steps": info["violation_steps"],
            "min_range_to_target_m": info["min_range_to_target_m"],
            "final_dist_to_hold_m": info["dist_to_hold_m"],
            "dv_used_mps": info["dv_used_mps"],
            "time_s": info["time_s"],
            "completed": self.done,
            "wall_time_s": time.perf_counter() - self.started,
            "latencies_s": np.array(self.latencies),
            "states": np.array(self.states),
            "actions": np.array(self.actions),
        }


def run_episode(
    env,
    agent,
    seed: int | None = None,
    agent_seed: int | None = None,
    real_time: RealTime | None = None,
) -> dict:
    """Run one episode. Returns final metrics plus the true trajectory and the actions.

    The agent gets the scenario's `AgentView`, a random stream seeded by `agent_seed`, the
    observations and the agent-safe info; the true states come from `info`. The agent's stream
    is independent of `seed`, so it carries nothing about the start state or the sensor noise.
    Without an `agent_seed` one is drawn from the operating system and returned in the result.
    Environments built for training (`training=True`) are refused. `latencies_s` holds the
    agent's wall-clock time per decision; with `real_time` the clock runs while it decides.
    """
    episode = Episode(env, seed, agent_seed)
    agent.reset(env.agent_view(), episode.agent_rng())
    replay = None if real_time is None else real_time.latencies
    while not episode.done:
        t0 = time.perf_counter()
        action = agent.act(episode.obs, env.agent_info())
        wall = time.perf_counter() - t0
        latency = None
        if real_time is not None:
            k = len(episode.actions)
            if replay is not None:
                if k >= len(replay):
                    raise ValueError(f"real_time.latencies ran out at decision {k}")
                wall = replay[k]
            latency = real_time.time_scale * wall
        episode.step(action, wall, latency)
    result = episode.result(agent.name)
    if hasattr(agent, "interventions"):
        result["filter_interventions"] = agent.interventions
    return result


def run_many(env, agent, seeds, agent_seeds=None) -> list[dict]:
    """Run one episode per seed, with one `agent_seeds` entry per seed if given."""
    seeds = list(seeds)
    if agent_seeds is None:
        agent_seeds = [None] * len(seeds)
    return [run_episode(env, agent, s, a) for s, a in zip(seeds, agent_seeds, strict=True)]
