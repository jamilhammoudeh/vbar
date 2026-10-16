"""Run episodes and collect benchmark metrics."""

from __future__ import annotations

import secrets
import time

import numpy as np


def run_episode(env, agent, seed: int | None = None, agent_seed: int | None = None) -> dict:
    """Run one episode. Returns final metrics plus the true trajectory and the actions.

    The agent gets the scenario's `AgentView`, a random stream seeded by `agent_seed`, the
    observations and the agent-safe info; the true states come from `info`. The agent's stream
    is independent of `seed`, so it carries nothing about the start state or the sensor noise.
    Without an `agent_seed` one is drawn from the operating system and returned in the result.
    """
    if agent_seed is None:
        agent_seed = secrets.randbits(128)
    obs, info = env.reset(seed=seed)
    agent.reset(env.agent_view(), np.random.default_rng(agent_seed))
    states, actions = [info["true_state"]], []
    t0 = time.perf_counter()
    terminated = truncated = False
    while not (terminated or truncated):
        action = np.asarray(agent.act(obs, env.agent_info()), dtype=float)
        obs, _, terminated, truncated, info = env.step(action)
        states.append(info["true_state"])
        actions.append(action)
    result = {
        "agent": agent.name,
        "seed": seed,
        "agent_seed": agent_seed,
        "captured": info["captured"],
        "keep_out_violated": info["keep_out_violated"],
        "violation_steps": info["violation_steps"],
        "min_range_to_target_m": info["min_range_to_target_m"],
        "final_dist_to_hold_m": info["dist_to_hold_m"],
        "dv_used_mps": info["dv_used_mps"],
        "time_s": info["time_s"],
        "wall_time_s": time.perf_counter() - t0,
        "states": np.array(states),
        "actions": np.array(actions),
    }
    if hasattr(agent, "interventions"):
        result["filter_interventions"] = agent.interventions
    return result


def run_many(env, agent, seeds, agent_seeds=None) -> list[dict]:
    """Run one episode per seed, with one `agent_seeds` entry per seed if given."""
    seeds = list(seeds)
    if agent_seeds is None:
        agent_seeds = [None] * len(seeds)
    return [run_episode(env, agent, s, a) for s, a in zip(seeds, agent_seeds, strict=True)]
