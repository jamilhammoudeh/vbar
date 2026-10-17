"""Scenario-declared settings the metrics need beyond the episode (METRICS.md, per metric)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class MetricSettings:
    """Declared parameters of the passive-safety and abort metrics (SI units).

    `collision_radius` is the collision envelope (target extent plus chaser extent plus margin),
    by default the keep-out radius. Passive safety is evaluated only beyond `passive_exclusion`
    and abort availability only beyond `commit_radius` (both ranges from the target, m).
    `drift_horizon` is the free-drift horizon (s, default two orbits of the reference mean
    motion); `stuck_time` the time a thruster stays on before it is detected and shut off;
    `partial_burn_samples` the sub-step points per thrusting segment checked for a burn cut
    short. The abort set is `cam_impulses` (Hill delta-v vectors, m/s; by default one of size
    `cam_dv` opposite the approach), executed after `cam_delay` s and checked for `cam_horizon`
    s (default the drift horizon). Drift is sampled `drift_samples_per_orbit` times per orbit.
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

    def __post_init__(self):
        for label in ("passive_exclusion", "commit_radius", "stuck_time", "cam_delay", "cam_dv"):
            if not getattr(self, label) >= 0.0:
                raise ValueError(f"{label} must not be negative")
        if self.drift_samples_per_orbit < 8:
            raise ValueError("drift_samples_per_orbit must be at least 8")
