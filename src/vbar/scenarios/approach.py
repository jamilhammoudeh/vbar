"""Approach to a hold point with a keep-out zone.

Task: fly to a hold point near the target (by default 100 m behind it on the V-bar) and stop
there without entering a keep-out sphere around the target. The keep-out check runs in
continuous time, between and within steps.

Gymnasium API: `obs, info = env.reset(seed=...)`, then
`obs, reward, terminated, truncated, info = env.step(action)`. Actions are 3 numbers in
[-1, 1], fractions of `max_accel` per Hill axis held for `dt`; observations are the navigated
relative state [x, y, z, vx, vy, vz]. Benchmark quantities are in `info`.

The reward is the shaped signal of the first release, kept so that existing results carry over.
It is not the benchmark score, and it pays the capture bonus even after a keep-out violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constraints import Constraint, KeepOut
from .base import BaseEnv, EnvConfig, StepRecord


@dataclass
class ApproachConfig(EnvConfig):
    """Scenario parameters (m, m/s) on top of the shared `EnvConfig`.

    The start state is `start_state` if given, otherwise uniform over the position and velocity
    ranges per Hill axis.
    """

    keep_out_radius: float = 50.0
    hold_point: tuple[float, float, float] = (0.0, -100.0, 0.0)
    capture_radius: float = 5.0
    capture_speed: float = 0.05
    start_state: tuple[float, ...] | None = None
    start_pos_range: tuple = ((-200.0, 200.0), (200.0, 600.0), (-50.0, 50.0))
    start_vel_range: tuple = ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))
    terminate_on_violation: bool = False
    reward_dv_weight: float = 1.0


class ApproachEnv(BaseEnv):
    scenario_name = "approach-vbar"
    cfg: ApproachConfig

    def __init__(self, config: ApproachConfig | None = None):
        super().__init__(config or ApproachConfig())
        self.hold_point = np.array(self.cfg.hold_point, dtype=float)

    def _sample_start(self, rng: np.random.Generator) -> np.ndarray:
        if self.cfg.start_state is not None:
            return np.array(self.cfg.start_state, dtype=float)
        pos = [rng.uniform(lo, hi) for lo, hi in self.cfg.start_pos_range]
        vel = [rng.uniform(lo, hi) for lo, hi in self.cfg.start_vel_range]
        return np.array(pos + vel, dtype=float)

    def _constraints(self) -> tuple[Constraint, ...]:
        return (KeepOut(self.cfg.keep_out_radius),)

    def _goal(self) -> dict[str, Any]:
        return {
            "hold_point": tuple(self.cfg.hold_point),
            "capture_radius": self.cfg.capture_radius,
            "capture_speed": self.cfg.capture_speed,
        }

    def reset(self, *, seed=None, options=None):
        self.violated = False
        self.violation_steps = 0
        self.captured = False
        obs, info = super().reset(seed=seed, options=options)
        self.violated = self.min_margins["keep_out"] < 0.0
        self.violation_steps = int(self.violated)
        self.dist_to_hold = float(np.linalg.norm(self.state[:3] - self.hold_point))
        return obs, self._info()

    def _evaluate(self, record: StepRecord) -> tuple[float, bool, dict[str, Any]]:
        prev_dist = self.dist_to_hold
        self.dist_to_hold = float(np.linalg.norm(self.state[:3] - self.hold_point))
        step_violation = record.segment.minima["keep_out"].margin < 0.0
        if step_violation:
            self.violated = True
            self.violation_steps += 1
        self.captured = (
            self.dist_to_hold < self.cfg.capture_radius
            and float(np.linalg.norm(self.state[3:])) < self.cfg.capture_speed
        )
        reward = (prev_dist - self.dist_to_hold) / 100.0
        reward -= self.cfg.reward_dv_weight * record.actuation.dv_l2
        reward -= 1.0 * step_violation
        reward += 10.0 * self.captured
        terminated = self.captured or (step_violation and self.cfg.terminate_on_violation)
        return reward, terminated, {}

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        info.update(
            {
                "dist_to_hold_m": float(np.linalg.norm(self.state[:3] - self.hold_point)),
                "range_to_target_m": float(np.linalg.norm(self.state[:3])),
                "speed_mps": float(np.linalg.norm(self.state[3:])),
                "min_range_to_target_m": self.min_margins["keep_out"] + self.cfg.keep_out_radius,
                "keep_out_violated": self.violated,
                "violation_steps": self.violation_steps,
                "captured": self.captured,
            }
        )
        return info
