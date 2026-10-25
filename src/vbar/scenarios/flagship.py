"""The flagship scenario: capture of a tumbling upper stage, with its tumble presets.

The target is an upper stage (ours): a body 6 m long and 2.4 m across, principal moments
(1000, 4200, 4700) kg m^2 about its long axis x and two transverse axes, its keep-out a capsule
of radius 1.5 m about the long axis. The grapple interface is the aft ring at x = -3 m and the
capture point is 2 m beyond it, at arm's reach. The tumble is torque-free and set by its
parameters (Scheeres, Orbital Motion in Strongly Perturbed Environments, 2012, section 3.2.8):
the presets are a flat spin about the major axis, the state dissipation leaves (Schaub and
Junkins, Analytical Mechanics of Space Systems, section 4.2.1), a coning spin about the long
axis, a tumble near the separatrix, and a spin started next to the unstable intermediate axis.
The tumble is drawn per episode and briefed at t = 0 in the goal's target entry.
"""

from __future__ import annotations

import math

from .catalog import ALTITUDE, INCLINATION, LATITUDE, TWO_PI
from .registry import register
from .spec import Observation, ScenarioSpec, Shell, Uniform
from .targets import TorqueFree

MOMENTS = (1000.0, 4200.0, 4700.0)
DEG = math.radians(1.0)

TUMBLING_CAPTURE = register(
    ScenarioSpec(
        name="tumbling-capture",
        version=1,
        entry_point="vbar.scenarios.tumbling:CaptureEnv",
        summary=("Reach and hold a capture-ready state at the aft ring of a tumbling upper stage."),
        config={
            "altitude_m": ALTITUDE,
            "inclination": INCLINATION,
            "dt": 2.0,
            "max_steps": 1800,
        },
        parameters={
            "capture_point": (-5.0, 0.0, 0.0),
            "interface": (-3.0, 0.0, 0.0),
            "capture_radius": 0.2,
            "capture_speed": 0.02,
            "dwell": 30.0,
            "body_ends": ((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            "body_radius": 1.5,
            "commit_radius": 6.0,
            "passive_exclusion": 20.0,
            "plume_range": 30.0,
            "cam_impulses": (
                (0.3, 0.0, 0.0),
                (-0.3, 0.0, 0.0),
                (0.0, 0.3, 0.0),
                (0.0, -0.3, 0.0),
                (0.0, 0.0, 0.3),
                (0.0, 0.0, -0.3),
            ),
        },
        dispersions={
            "start_pos": Shell(40.0, 100.0),
            "start_vel": Uniform((-0.02, -0.02, -0.02), (0.02, 0.02, 0.02)),
            "target_effective_inertia": Uniform(4600.0, 4690.0),
            "target_rate": Uniform(0.5 * DEG, 2.0 * DEG),
            "target_phase": Uniform(0.0, 1.0),
            "target_momentum_direction": Shell(1.0, 1.0),
            "target_spin_angle": Uniform(0.0, TWO_PI),
            **LATITUDE,
        },
        target=TorqueFree(MOMENTS, 4650.0, DEG),
        tags=("capture", "tumbling", "non-cooperative", "flagship"),
        observation=Observation(
            hidden=("dwell clock",),
            note=(
                "the tumble is briefed exactly at t = 0 in the goal's target entry and is "
                "deterministic, so the state, the time and the briefing decide the next state"
            ),
        ),
        oracle={"capture_point_speed_at_2_deg_per_s_mps": 5.0 * 2.0 * DEG},
        variants={
            "coning": {
                "dispersions": {
                    "target_effective_inertia": Uniform(1050.0, 1300.0),
                    "target_rate": Uniform(1.0 * DEG, 4.0 * DEG),
                },
                "tags": ("coning",),
            },
            "separatrix": {
                "dispersions": {
                    "target_effective_inertia": Uniform(4210.0, 4290.0),
                    "target_rate": Uniform(0.5 * DEG, 1.5 * DEG),
                },
                "tags": ("separatrix",),
            },
            "intermediate": {
                "dispersions": {
                    "target_effective_inertia": Uniform(4200.04, 4200.4),
                    "target_rate": Uniform(0.5 * DEG, 1.5 * DEG),
                    "target_phase": Uniform(0.22, 0.25),
                },
                "tags": ("intermediate-axis",),
            },
        },
    )
)

__all__ = ["MOMENTS", "TUMBLING_CAPTURE"]
