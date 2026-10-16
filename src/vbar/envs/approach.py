"""Approach-to-hold-point scenario with a keep-out zone.

Task: a chaser spacecraft must fly to a "hold point" near a target
(by default 100 m behind it on the V-bar) and stop there, without entering
a keep-out sphere around the target. This is the basic building block of
rendezvous, inspection, and docking.

The environment follows the Gymnasium API:
    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)

Action: 3 numbers in [-1, 1], scaled by `max_accel` to a constant thrust
acceleration (m/s^2) in the Hill frame, held for one time step `dt`.

Observation: the chaser's 6-element state [x, y, z, vx, vy, vz] relative to
the target (see vbar.dynamics.cw for the frame).

Benchmark metrics are reported in `info`; the reward is a simple shaped
signal for reinforcement learning and is NOT the benchmark score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..dynamics.cw import R_EARTH, mean_motion, propagate


@dataclass
class ApproachConfig:
    altitude_m: float = 500e3  # target orbit altitude (circular)
    dt: float = 10.0  # seconds per decision step
    max_steps: int = 360  # 360 x 10 s = 1 hour
    max_accel: float = 0.02  # m/s^2, max thrust acceleration per axis
    keep_out_radius: float = 50.0  # m, sphere around the target
    hold_point: tuple = (0.0, -100.0, 0.0)  # m, where the chaser must stop
    capture_radius: float = 5.0  # m, "arrived" if this close to hold point
    capture_speed: float = 0.05  # m/s, ...and slower than this
    # Start state: fixed (if given) or sampled uniformly from these ranges.
    start_state: tuple | None = None
    start_pos_range: tuple = ((-200.0, 200.0), (200.0, 600.0), (-50.0, 50.0))
    start_vel_range: tuple = ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))
    terminate_on_violation: bool = False
    violation_samples: int = 20  # sub-samples per step to catch violations
    reward_dv_weight: float = 1.0
    extra: dict = field(default_factory=dict)


class ApproachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: ApproachConfig | None = None):
        super().__init__()
        self.cfg = config or ApproachConfig()
        self.n = mean_motion(R_EARTH + self.cfg.altitude_m)
        self.hold_point = np.array(self.cfg.hold_point, dtype=float)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # Generous finite bounds (positions in m, velocities in m/s).
        self.observation_space = spaces.Box(-1e7, 1e7, shape=(6,), dtype=np.float64)
        self.state = np.zeros(6)

    # ------------------------------------------------------------------ #
    def _sample_start(self) -> np.ndarray:
        if self.cfg.start_state is not None:
            return np.array(self.cfg.start_state, dtype=float)
        pos = [self.np_random.uniform(lo, hi) for lo, hi in self.cfg.start_pos_range]
        vel = [self.np_random.uniform(lo, hi) for lo, hi in self.cfg.start_vel_range]
        return np.array(pos + vel, dtype=float)

    def _info(self) -> dict:
        pos, vel = self.state[:3], self.state[3:]
        return {
            "time_s": self.t,
            "dist_to_hold_m": float(np.linalg.norm(pos - self.hold_point)),
            "range_to_target_m": float(np.linalg.norm(pos)),
            "speed_mps": float(np.linalg.norm(vel)),
            "dv_used_mps": self.dv_used,
            "min_range_to_target_m": self.min_range,
            "keep_out_violated": self.violated,
            "violation_steps": self.violation_steps,
            "captured": self.captured,
        }

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._sample_start()
        self.t = 0.0
        self.steps = 0
        self.dv_used = 0.0
        self.min_range = float(np.linalg.norm(self.state[:3]))
        self.violated = self.min_range < self.cfg.keep_out_radius
        self.violation_steps = int(self.violated)
        self.captured = False
        return self.state.copy(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        accel = action * self.cfg.max_accel
        prev_dist = np.linalg.norm(self.state[:3] - self.hold_point)

        self.state, samples = propagate(
            self.state, self.n, self.cfg.dt, accel, n_samples=self.cfg.violation_samples
        )
        self.t += self.cfg.dt
        self.steps += 1
        # Delta-v: constant acceleration magnitude times burn duration.
        step_dv = float(np.linalg.norm(accel) * self.cfg.dt)
        self.dv_used += step_dv

        # Keep-out check over the whole step, not just its end point.
        ranges = np.linalg.norm(samples[:, :3], axis=1)
        self.min_range = min(self.min_range, float(ranges.min()))
        step_violation = bool(ranges.min() < self.cfg.keep_out_radius)
        if step_violation:
            self.violated = True
            self.violation_steps += 1

        info = self._info()
        self.captured = (
            info["dist_to_hold_m"] < self.cfg.capture_radius
            and info["speed_mps"] < self.cfg.capture_speed
        )
        info["captured"] = self.captured

        # Shaped reward for RL: progress toward hold point, minus fuel,
        # minus a penalty for being inside the keep-out zone.
        reward = (prev_dist - info["dist_to_hold_m"]) / 100.0
        reward -= self.cfg.reward_dv_weight * step_dv
        reward -= 1.0 * step_violation
        reward += 10.0 * self.captured

        terminated = self.captured or (step_violation and self.cfg.terminate_on_violation)
        truncated = self.steps >= self.cfg.max_steps
        return self.state.copy(), float(reward), terminated, truncated, info
