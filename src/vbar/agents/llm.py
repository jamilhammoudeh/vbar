"""LLM agent: asks a language model for a thrust command at each decision step.

The model call is passed in as `complete(prompt: str) -> str`, so any hosted
API or local model works. Each call's prompt, raw reply, parsed action, parse
failure flag and wall-clock latency are stored in `self.log`.

The simulation clock is paused during the model call.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable

import numpy as np

PROMPT_TEMPLATE = """You are the guidance system of a chaser spacecraft near a target spacecraft.

Frame (centered on the target): x = radial (up, away from Earth), y = along-track
(direction of flight), z = cross-track. Units: meters and meters per second.

Goal: reach the hold point {hold} m and stop there (speed below {cap_speed} m/s).
Constraint: NEVER come within {keep_out} m of the target (the origin).
Orbital mechanics apply (Clohessy-Wiltshire dynamics, mean motion n = {n:.6f} rad/s).

Current state:
  position = [{x:.1f}, {y:.1f}, {z:.1f}] m
  velocity = [{vx:.3f}, {vy:.3f}, {vz:.3f}] m/s
  distance to hold point = {dist:.1f} m, distance to target = {rng:.1f} m

Each command is held for {dt:.0f} s. Thrust per axis is a fraction of the maximum
({max_accel} m/s^2), between -1 and 1.

Reply with ONLY a JSON object, for example: {{"ax": 0.1, "ay": -0.5, "az": 0.0}}"""


def parse_action(text: str) -> np.ndarray | None:
    """Extract {"ax":..,"ay":..,"az":..} from a model reply. Returns None on failure."""
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        a = np.array([float(data["ax"]), float(data["ay"]), float(data["az"])])
    except (ValueError, KeyError, TypeError):
        return None
    if not np.all(np.isfinite(a)):
        return None
    return np.clip(a, -1.0, 1.0)


class LLMAgent:
    def __init__(self, complete: Callable[[str], str], name: str = "llm"):
        self.complete = complete
        self.name = name
        self.log: list[dict] = []

    def reset(self, env) -> None:
        self.env = env
        self.log = []

    def act(self, obs, info):
        cfg = self.env.cfg
        x, y, z, vx, vy, vz = obs
        prompt = PROMPT_TEMPLATE.format(
            hold=list(cfg.hold_point),
            cap_speed=cfg.capture_speed,
            keep_out=cfg.keep_out_radius,
            n=self.env.n,
            x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
            dist=info["dist_to_hold_m"],
            rng=info["range_to_target_m"],
            dt=cfg.dt,
            max_accel=cfg.max_accel,
        )
        t0 = time.perf_counter()
        reply = self.complete(prompt)
        latency = time.perf_counter() - t0
        action = parse_action(reply)
        failed = action is None
        if failed:
            action = np.zeros(3)  # safe default: coast
        self.log.append(
            {
                "time_s": info["time_s"],
                "prompt": prompt,
                "reply": reply,
                "action": action.tolist(),
                "parse_failed": failed,
                "latency_s": latency,
            }
        )
        return action
