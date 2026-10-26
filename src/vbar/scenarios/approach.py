"""Approach to a hold point with a keep-out zone.

Fly to a hold point near the target (by default 100 m behind it on the V-bar) and stop there
without entering a keep-out sphere around the target, checked in continuous time within every
step. Benchmark quantities are in `info`.

Success (M1) is capture with the keep-out zone never entered; the keep-out constraint is hard.
Without a `reward` in the configuration the reward is the legacy signal of the first release,
not aligned with the metrics: progress toward the hold point in units of 100 m, minus
`reward_dv_weight` times the step's delta-v, minus 1 for a step that enters the keep-out zone,
plus 10 on capture, which it pays even after a violation, and its progress term is not
potential-based under discounting. Use a `vbar.rewards.Reward` for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constraints import Constraint, KeepOut
from .base import BaseEnv, EnvConfig, StepRecord, check_number


@dataclass(kw_only=True)
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

    def __post_init__(self):
        super().__post_init__()
        for name in ("keep_out_radius", "capture_radius", "capture_speed"):
            check_number(name, getattr(self, name), low=0.0)
        if len(self.hold_point) != 3:
            raise ValueError("hold_point has three components")
        for c in self.hold_point:
            check_number("hold_point", c)


class ApproachEnv(BaseEnv):
    scenario_name = "approach-vbar"
    cfg: ApproachConfig
    reset_options = ("start_state", "start_pos_range", "start_vel_range")

    def __init__(self, config: ApproachConfig | None = None):
        super().__init__(config or ApproachConfig())
        self.hold_point = np.array(self.cfg.hold_point, dtype=float)

    def _sample_start(self, rng: np.random.Generator) -> np.ndarray:
        options = getattr(self, "options", {})
        if self.cfg.start_state is not None and not options:
            return np.array(self.cfg.start_state, dtype=float)
        pos_range = options.get("start_pos_range", self.cfg.start_pos_range)
        vel_range = options.get("start_vel_range", self.cfg.start_vel_range)
        pos = [rng.uniform(lo, hi) for lo, hi in pos_range]
        vel = [rng.uniform(lo, hi) for lo, hi in vel_range]
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

    def _goal_distance(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(state[:3] - self.hold_point))

    def _success(self) -> bool:
        return self.captured and not self.violated

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
