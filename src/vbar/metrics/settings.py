"""Scenario-declared settings the metrics need beyond the episode (docs/metrics.md)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class MetricSettings:
    """Declared metric parameters of a scenario (SI units, angles in rad).

    `collision_radius` is the collision envelope (target extent plus chaser extent plus margin),
    by default the keep-out radius. Passive safety is evaluated only beyond `passive_exclusion`
    and abort availability only beyond `commit_radius` (both ranges from the target, m).
    `drift_horizon` is the free-drift horizon (s, default two orbits of the reference mean
    motion); `stuck_time` the time a thruster stays on before it is detected and shut off;
    `partial_burn_samples` the sub-step points per thrusting segment checked for a burn cut
    short. The abort set is `cam_impulses` (Hill delta-v vectors, m/s; by default one of size
    `cam_dv` opposite the approach), executed after `cam_delay` s and checked for `cam_horizon`
    s (default the drift horizon). Drift is sampled `drift_samples_per_orbit` times per orbit.

    Thrust within `plume_range` (m, None for the scenario's own) is scored with the Gaussian plume
    of 1-sigma angle `plume_sigma` (rad); exhaust within `plume_half_angle` of the line of sight
    to the target violates the plume constraint; forces act on `plume_target_area` (m^2).
    `composite_weights` replace the composite score's default infraction weights, and a failed run
    whose chatter exceeds `oscillation_threshold` (1/s) is labelled an oscillation. A filter holds
    the chaser stalled when it moves slower than `stall_speed` (m/s) with a safety row active.
    """

    collision_radius: float | None = None
    passive_exclusion: float = 0.0
    commit_radius: float = 0.0
    drift_horizon: float | None = None
    stuck_time: float = 30.0
    partial_burn_samples: int = 4
    cam_impulses: tuple[tuple[float, float, float], ...] | None = None
    cam_dv: float = 0.3
    cam_delay: float = 0.0
    cam_horizon: float | None = None
    drift_samples_per_orbit: int = 200
    plume_range: float | None = None
    plume_sigma: float = math.radians(13.0)
    plume_half_angle: float = math.radians(39.0)
    plume_target_area: float = 1.0
    composite_weights: Mapping[str, float] | None = None
    oscillation_threshold: float = 0.05
    stall_speed: float = 0.01

    def __post_init__(self):
        # A wrong sign passes silently otherwise: a negative collision radius scores every
        # drift as passively safe.
        for label in (
            "passive_exclusion",
            "commit_radius",
            "stuck_time",
            "cam_delay",
            "cam_dv",
            "oscillation_threshold",
            "stall_speed",
        ):
            _check(label, getattr(self, label))
        for label in ("plume_sigma", "plume_half_angle", "plume_target_area"):
            _check(label, getattr(self, label), positive=True)
        for label in ("collision_radius", "drift_horizon", "cam_horizon"):
            if getattr(self, label) is not None:
                _check(label, getattr(self, label), positive=True)
        if self.plume_range is not None:
            _check("plume_range", self.plume_range)
        if not 0.0 < self.plume_sigma < math.pi or not 0.0 < self.plume_half_angle < math.pi:
            raise ValueError("plume_sigma and plume_half_angle are angles in (0, pi) rad")
        for impulse in self.cam_impulses or ():
            if len(impulse) != 3 or not all(math.isfinite(float(c)) for c in impulse):
                raise ValueError("cam_impulses are finite 3-vectors (m/s)")
        for name, weight in (self.composite_weights or {}).items():
            _check(f"composite_weights[{name!r}]", weight)
        for label in ("partial_burn_samples", "drift_samples_per_orbit"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise ValueError(f"{label} must be an integer")
        if self.partial_burn_samples < 0:
            raise ValueError("partial_burn_samples must not be negative")
        if self.drift_samples_per_orbit < 8:
            raise ValueError("drift_samples_per_orbit must be at least 8")


def _check(label: str, value: float, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'not negative'}")
