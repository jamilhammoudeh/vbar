"""The flagship scenario, capture of a tumbling upper stage with its presets, and the departure.

The target is an upper stage (ours): a body 6 m long and 2.4 m across, principal moments
(1000, 4200, 4700) kg m^2 about its long axis x and two transverse axes, its keep-out a capsule
of radius 1.5 m about the long axis. The grapple interface is the aft ring at x = -3 m and the
capture point is 2 m beyond it, at arm's reach. The tumble is set by its parameters (Scheeres,
Orbital Motion in Strongly Perturbed Environments, 2012, section 3.2.8): torque-free, a flat
spin about the major axis, the state dissipation leaves (Schaub and Junkins, Analytical Mechanics
of Space Systems, section 4.2.1), a coning spin about the long axis, a tumble near the separatrix
and a spin started next to the unstable intermediate axis; and under the gravity-gradient
torque, a dead stage librating about the orbit normal. The tumble is drawn per episode and
briefed at t = 0 in the goal's target entry. The departure is the other end of a visit: from the
aft port through a departure cone onto a path that does not come back.
"""

from __future__ import annotations

import math

from .catalog import ALTITUDE, INCLINATION, ISS_ALTITUDE, LATITUDE, TWO_PI, N
from .registry import register
from .spec import Fixed, Observation, ScenarioSpec, Shell, Uniform
from .targets import GravityGradientTumble, TorqueFree

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
            # A dead stage held by the gravity gradient with its long axis radial, the stable
            # attitude for these moments (Schaub and Junkins, section 4.4.3), librating about the
            # orbit normal from a pitch and a rate a little off the orbit's.
            "libration": {
                "target": GravityGradientTumble(MOMENTS, 4700.0, N, 0.0, (0.0, 0.0, 1.0), 0.0),
                "dispersions": {
                    "target_effective_inertia": Fixed(4700.0),
                    "target_rate": Uniform(0.95 * N, 1.05 * N),
                    "target_phase": Fixed(0.0),
                    "target_momentum_direction": Fixed((0.0, 0.0, 1.0)),
                    "target_spin_angle": Uniform(math.radians(-20.0), math.radians(20.0)),
                },
                "tags": ("libration", "gravity-gradient"),
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

# A departure after Fehse (Automated Rendezvous and Docking of Spacecraft, 2003, section 2.6):
# springs push the chaser off at about 0.06 m/s, and it stays inside a 10 deg departure cone for
# the first few hundred metres on a path that does not come back, its first burns kept off the
# target. The rest is ours: the -V-bar port of a 30 m target, the cone active within 200 m, clear
# beyond 300 m, the dispersions, and the 342.62 km of the `:iss` variants (the STS-130 mean,
# Goodman, History of Space Shuttle Rendezvous, 2011, p. 227).
DEPARTURE = register(
    ScenarioSpec(
        name="departure",
        version=1,
        entry_point="vbar.scenarios.departure:DepartureEnv",
        summary=(
            "Leave the -V-bar port through a 10 deg departure cone onto a path that does not "
            "come back."
        ),
        config={
            "altitude_m": ISS_ALTITUDE,
            "inclination": INCLINATION,
            "dt": 10.0,
            "max_steps": 360,
        },
        parameters={
            "port": (0.0, -30.0, 0.0),
            "port_axis": (0.0, -1.0, 0.0),
            "body_radius": 30.0,
            "cone": (math.radians(10.0), 200.0),
            "clear_range": 300.0,
            "clear_radius": 100.0,
            "plume_range": 100.0,
        },
        dispersions={
            "separation": Uniform(0.0, 0.2),
            "separation_speed": Uniform(0.05, 0.07),
            **LATITUDE,
        },
        tags=("departure", "vbar"),
        observation=Observation(note="the state decides the next state"),
        oracle={"spring_separation_mps": 0.06},
    )
)

__all__ = ["DEPARTURE", "MOMENTS", "TUMBLING_CAPTURE"]
