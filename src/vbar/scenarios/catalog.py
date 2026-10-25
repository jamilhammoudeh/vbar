"""The v1 scenario catalogue: every built-in spec, its variants and the suites.

Shared defaults: a target on a circular 500 km Earth orbit at 51.6 deg inclination, exact
Clohessy-Wiltshire propagation, a 500 kg chaser with 0.02 m/s^2 per axis and Isp 220 s, perfect
navigation, 10 s steps. The `:iss` and Shuttle-practice variants fly at 342.62 km (185 n.mi.,
the STS-130 mean, Goodman, History of Space Shuttle Rendezvous, 2011, p. 227). Values marked
"ours" in the docs are this benchmark's choices inside published ranges; the sources are in
`docs/scenarios.md` and `docs/validation.md`.
"""

from __future__ import annotations

import math

from ..dynamics import MU_EARTH, R_EARTH, mean_motion
from ..spacecraft import GaussianNav, GaussMarkovNav
from .registry import register
from .spec import (
    AngleShell,
    Choice,
    DriftingVelocity,
    Fixed,
    LimitedSpeed,
    Observation,
    ScenarioSpec,
    Shell,
    Uniform,
)
from .suites import Suite, SuiteEntry, register_suite
from .targets import FixedOffset, InertialPointing, Maneuvering

FT = 0.3048
INCLINATION = math.radians(51.6)
ALTITUDE = 500e3
ISS_ALTITUDE = 185.0 * 1852.0
HST_ALTITUDE = 560e3
# The run time assurance docking benchmark's mean motion (Dunlap et al., 2021, eq. 14).
AFRL_N = 0.001027
AFRL_ALTITUDE = (MU_EARTH / AFRL_N**2) ** (1.0 / 3.0) - R_EARTH
N = mean_motion(R_EARTH + ALTITUDE)
N_ISS = mean_motion(R_EARTH + ISS_ALTITUDE)
N_HST = mean_motion(R_EARTH + HST_ALTITUDE)
TWO_PI = 2.0 * math.pi

CONFIG = {"altitude_m": ALTITUDE, "inclination": INCLINATION, "dt": 10.0, "max_steps": 360}
ISS_CONFIG = {"altitude_m": ISS_ALTITUDE}
# The argument of latitude only matters on the J2 rung, where the drift depends on it; it is
# drawn for every scenario so the other draws do not move when the rung changes.
LATITUDE = {"argument_of_latitude": Uniform(0.0, TWO_PI)}
NAV_VARIANTS = {
    "nav1": {"config": {"sensor": GaussianNav()}, "tags": ("nav1",)},
    "nav2": {"config": {"sensor": GaussMarkovNav()}, "tags": ("nav2",)},
}


def feet(pairs) -> tuple[tuple[float, float], ...]:
    """A gate table given in (ft, ft/s) as (m, m/s).

    Parameters
    ----------
    pairs : iterable of tuple of float
        (range ft, closing speed ft/s) gates.

    Returns
    -------
    tuple of tuple of float
        (range m, closing speed m/s) gates, in the same order.
    """
    return tuple((r * FT, v * FT) for r, v in pairs)


# Flown range-rate gates (Goodman, table 24.3 p. 209, fig. 9.3 p. 74, fig. 25.17 p. 240,
# fig. 25.22 p. 242).
GATES = {
    "shuttle-rbar-orbt": feet(
        [
            (2000, 3),
            (1700, 2.6),
            (1500, 2.3),
            (1000, 1.5),
            (600, 0.8),
            (500, 0.5),
            (400, 0.4),
            (300, 0.3),
            (200, 0.2),
            (120, 0.2),
            (100, 0.1),
        ]
    ),
    "shuttle-vbar-iss": feet([(320, 0.20), (110, 0.15), (75, 0.10), (30, 0.07)]),
    "dual-coelliptic": feet(
        [
            (9000, 45),
            (8000, 40),
            (7000, 35),
            (6000, 30),
            (5000, 25),
            (4000, 20),
            (3000, 15),
            (2000, 10),
            (1000, 5),
            (400, 4),
            (300, 3),
            (200, 2),
        ]
    ),
    "sts-130-orbt": feet(
        [
            (2000, 3.0),
            (1700, 2.4),
            (1500, 2.1),
            (1000, 1.3),
            (900, 1.1),
            (800, 0.9),
            (700, 0.6),
            (650, 0.4),
        ]
    ),
}


def hold_oracle(n: float, dy: float) -> dict[str, float]:
    """Two-impulse and forced-motion delta-v (m/s) for an along-track transfer of `dy` m.

    Tangential two-impulse n dy / (3 pi) over one orbit, radial two-impulse n dy / 2 over half
    an orbit, forced straight line at constant speed over one orbit 2 |v| + 2 n dy (Fehse eqs
    3.29 to 3.32, 3.44 to 3.48).

    Parameters
    ----------
    n : float
        Mean motion (rad/s).
    dy : float
        Along-track (V-bar) transfer distance (m).

    Returns
    -------
    dict
        "tangential_two_impulse_mps", "radial_two_impulse_mps" and
        "forced_straight_line_one_orbit_mps", each a total delta-v (m/s).
    """
    period = TWO_PI / n
    return {
        "tangential_two_impulse_mps": n * dy / (3.0 * math.pi),
        "radial_two_impulse_mps": n * dy / 2.0,
        "forced_straight_line_one_orbit_mps": 2.0 * dy / period + 2.0 * n * dy,
    }


APPROACH_VBAR = register(
    ScenarioSpec(
        name="approach-vbar",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary="Reach the hold point 100 m behind the target on the V-bar and hold it for 600 s.",
        config=CONFIG,
        parameters={
            "hold_point": (0.0, -100.0, 0.0),
            "capture_radius": 5.0,
            "capture_speed": 0.05,
            "hold_time": 600.0,
            "keep_out": 50.0,
            "closing_limit": (0.2, 4.0 * N),
            "passive_axis": "vbar",
            "plume_range": 100.0,
            "passive_exclusion": 50.0,
            "soft": ("closing_speed_limit", "passive_speed"),
        },
        dispersions={
            "start_pos": Uniform((-200.0, 200.0, -50.0), (200.0, 600.0, 50.0)),
            "start_vel": Uniform((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
            **LATITUDE,
        },
        tags=("approach", "vbar", "hold", "target-in-path"),
        observation=Observation(
            hidden=("dwell elapsed",),
            note="the dwell clock is the agent's own; the state alone decides the next state",
        ),
        oracle=hold_oracle(N, 500.0),
        variants={
            "iss": {
                "config": {**ISS_CONFIG, "max_steps": 720},
                "parameters": {
                    "hold_point": (0.0, -400.0, 0.0),
                    "keep_out": 200.0,
                    "plume_range": 300.0,
                    "passive_exclusion": 200.0,
                    "closing_limit": (0.2, 4.0 * N_ISS),
                },
                "dispersions": {
                    "start_pos": Uniform((-200.0, 600.0, -50.0), (200.0, 1200.0, 50.0))
                },
                "tags": ("iss",),
            },
            "gates-vbar-iss": {
                "parameters": {
                    "closing_limit": None,
                    "gates": GATES["shuttle-vbar-iss"],
                    "soft": ("gate_table", "passive_speed"),
                },
                "tags": ("gates",),
            },
            **NAV_VARIANTS,
        },
    )
)

APPROACH_RBAR = register(
    ScenarioSpec(
        name="approach-rbar",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary="Reach the hold point 100 m below the target on the R-bar and hold it for 600 s.",
        config=CONFIG,
        parameters={
            "hold_point": (-100.0, 0.0, 0.0),
            "capture_radius": 5.0,
            "capture_speed": 0.05,
            "hold_time": 600.0,
            "keep_out": 50.0,
            "closing_limit": (0.2, 4.0 * N),
            "passive_axis": "rbar",
            "plume_range": 100.0,
            "passive_exclusion": 50.0,
            "soft": ("closing_speed_limit", "passive_speed"),
        },
        dispersions={
            "start_pos": Uniform((-600.0, -200.0, -50.0), (-200.0, 200.0, 50.0)),
            "start_vel": Uniform((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
            **LATITUDE,
        },
        tags=("approach", "rbar", "hold"),
        observation=Observation(hidden=("dwell elapsed",), note="as approach-vbar"),
        oracle={"hold_cost_per_orbit_mps": 3.0 * N**2 * 100.0 * TWO_PI / N},
        variants={
            "iss": {
                "config": {**ISS_CONFIG, "max_steps": 720},
                "parameters": {
                    "hold_point": (-400.0, 0.0, 0.0),
                    "keep_out": 200.0,
                    "plume_range": 300.0,
                    "passive_exclusion": 200.0,
                    "closing_limit": (0.2, 4.0 * N_ISS),
                },
                "dispersions": {
                    "start_pos": Uniform((-1200.0, -200.0, -50.0), (-600.0, 200.0, 50.0))
                },
                "tags": ("iss",),
            },
            "gates-orbt": {
                "parameters": {
                    "closing_limit": None,
                    "gates": GATES["shuttle-rbar-orbt"],
                    "soft": ("gate_table", "passive_speed"),
                },
                "tags": ("gates",),
            },
            **NAV_VARIANTS,
        },
    )
)

APPROACH_HBAR = register(
    ScenarioSpec(
        name="approach-hbar",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary=(
            "Reach a hold point past the target along the orbit normal, with the target in the way."
        ),
        config=CONFIG,
        parameters={
            "hold_point": (0.0, 0.0, -100.0),
            "capture_radius": 5.0,
            "capture_speed": 0.05,
            "hold_time": 600.0,
            "keep_out": 50.0,
            "closing_limit": (0.2, 4.0 * N),
            "plume_range": 100.0,
            "passive_exclusion": 50.0,
            "cam_impulses": ((0.0, 0.0, 0.3),),
            "soft": ("closing_speed_limit",),
        },
        dispersions={
            "start_pos": Uniform((-20.0, -20.0, 300.0), (20.0, 20.0, 500.0)),
            "start_vel": Uniform((-0.02, -0.02, -0.02), (0.02, 0.02, 0.02)),
            **LATITUDE,
        },
        tags=("approach", "hbar", "hold", "target-in-path", "liveness"),
        observation=Observation(hidden=("dwell elapsed",), note="as approach-vbar"),
        oracle={"hold_cost_per_orbit_mps": N**2 * 100.0 * TWO_PI / N},
        variants=NAV_VARIANTS,
    )
)

DOCKING = register(
    ScenarioSpec(
        name="docking",
        version=2,
        entry_point="vbar.scenarios.docking:DockingEnv",
        summary="Make contact at the -V-bar port inside the reception envelope and the corridor.",
        config={**CONFIG, "dt": 5.0, "max_steps": 1200},
        parameters={
            "port": (0.0, -2.0, 0.0),
            "port_axis": (0.0, -1.0, 0.0),
            "body_radius": 2.0,
            "lateral_limit": 0.1,
            "lateral_speed_limit": 0.03,
            "preset": "probe-drogue-atv",
            "cones": ((math.radians(10.0), 50.0),),
            "commit_radius": 1.0,
            "passive_exclusion": 50.0,
            "plume_range": 100.0,
        },
        dispersions={
            "start_pos": Shell(300.0, 800.0),
            "start_vel": Uniform((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)),
            **LATITUDE,
        },
        tags=("docking", "corridor", "contact-window"),
        observation=Observation(
            hidden=("contact event",),
            note="the port and its axis are public; the state decides the next state",
        ),
        variants={
            "axis": {
                "dispersions": {
                    "start_pos": Uniform((-20.0, -502.0, -20.0), (20.0, -102.0, 20.0)),
                    "start_vel": Uniform((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
                },
                "tags": ("on-axis",),
            },
            "corridor-shuttle": {
                "config": ISS_CONFIG,
                "parameters": {
                    "cones": ((math.radians(8.0), 400.0 * FT), (math.radians(5.0), 30.0)),
                    "preset": "apds",
                    "hold": (30.0 * FT, 300.0),
                    "plume_range": 300.0,
                },
                "tags": ("shuttle", "corridor-shuttle"),
            },
            "lit": {
                "config": {"max_steps": 1500},
                "parameters": {"lighting": (math.radians(15.0), math.radians(75.0), 10.0)},
                "dispersions": {
                    "start_pos": Uniform((-20.0, -302.0, -20.0), (20.0, -102.0, 20.0)),
                    "start_vel": Uniform((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
                    "beta": Uniform(math.radians(-70.0), math.radians(70.0)),
                    "sun_phase": Uniform(0.0, TWO_PI),
                },
                "tags": ("lighting", "timing"),
            },
            **NAV_VARIANTS,
        },
    )
)

DOCKING_OFFSET = register(
    ScenarioSpec(
        name="docking-offset",
        version=1,
        entry_point="vbar.scenarios.docking:DockingEnv",
        summary=(
            "Dock at a port 20 m from the centre of a target pitched to a torque-equilibrium "
            "attitude."
        ),
        config={**CONFIG, "dt": 5.0, "max_steps": 1200},
        parameters={
            "port": (0.0, -20.0, 0.0),
            "port_axis": (0.0, -1.0, 0.0),
            "body_radius": 20.0,
            "preset": "probe-drogue-atv",
            "cones": ((math.radians(10.0), 50.0),),
            "commit_radius": 1.0,
            "passive_exclusion": 50.0,
            "plume_range": 100.0,
        },
        dispersions={
            "start_pos": Uniform((-20.0, -300.0, -20.0), (20.0, -150.0, 20.0)),
            "start_vel": Uniform((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
            "target_pitch": Uniform(math.radians(15.0), math.radians(30.0)),
            **LATITUDE,
        },
        target=FixedOffset(pitch=math.radians(20.0)),
        tags=("docking", "offset", "fly-around", "corridor"),
        observation=Observation(note="the pitch is public through the goal's target entry"),
        variants=NAV_VARIANTS,
    )
)

INERTIAL_TARGET = register(
    ScenarioSpec(
        name="inertial-target",
        version=2,
        entry_point="vbar.scenarios.docking:DockingEnv",
        summary="Dock at the port of an inertially pointing target whose axis turns at -n.",
        config={**CONFIG, "dt": 5.0, "max_steps": 1200},
        parameters={
            "port": (0.0, -2.0, 0.0),
            "port_axis": (0.0, -1.0, 0.0),
            "body_radius": 2.0,
            "preset": "probe-drogue-atv",
            "cones": ((math.radians(10.0), 50.0),),
            "commit_radius": 1.0,
            "passive_exclusion": 50.0,
            "plume_range": 100.0,
        },
        dispersions={
            "start_pos": Shell(150.0, 300.0),
            "start_vel": Uniform((-0.02, -0.02, -0.02), (0.02, 0.02, 0.02)),
            "target_phase": Uniform(0.0, TWO_PI),
            **LATITUDE,
        },
        target=InertialPointing(),
        tags=("docking", "rotating-port", "corridor"),
        observation=Observation(
            note="the phase and rate are public through the goal's target entry"
        ),
        oracle={"on_axis_lateral_speed_at_100m_mps": 100.0 * N},
        variants=NAV_VARIANTS,
    )
)

DOCKING_AFRL = register(
    ScenarioSpec(
        name="docking-afrl",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary=(
            "The run time assurance docking benchmark's task: reach the target under its speed "
            "limit."
        ),
        config={**CONFIG, "altitude_m": AFRL_ALTITUDE, "dt": 5.0, "max_steps": 1200},
        parameters={
            "hold_point": (0.0, 0.0, 0.0),
            "capture_radius": 0.5,
            "capture_speed": 0.2,
            "hold_time": 0.0,
            "keep_out": 0.0,
            "full_speed_limit": (0.2, 4.0 * AFRL_N),
            "cam_impulses": ((0.3, 0.0, 0.0),),
        },
        dispersions={
            "start_pos": Shell(300.0, 800.0),
            "start_vel": Uniform((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)),
            **LATITUDE,
        },
        tags=("docking", "afrl", "speed-limit"),
        observation=Observation(note="the state decides the next state"),
        variants=NAV_VARIANTS,
    )
)

# Comparison tasks rebuilt from published benchmarks, for results that compare directly with
# theirs; every value is the paper's unless marked ours in the docs.
# AFRL safe-autonomy docking (Hamilton, Dunlap and Hobbs, "Investigating the Impact of Choice on
# Deep Reinforcement Learning for Space Controls", 2024, arXiv 2405.12355, section III.B): a 12 kg
# deputy with 1 N per axis, docking within 10 m below 0.2 m/s, under 0.2 + 2 n (rho - 10) m/s.
AFRL_MASS = 12.0
AFRL_ACCEL = 1.0 / AFRL_MASS
AFRL_DOCKING_RADIUS = 10.0
AFRL_RL_DOCKING = register(
    ScenarioSpec(
        name="afrl-docking",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary=(
            "The AFRL deep-RL docking task: reach 10 m of the chief below 0.2 m/s, starting 100 "
            "to 150 m out."
        ),
        config={
            **CONFIG,
            "altitude_m": AFRL_ALTITUDE,
            "dt": 1.0,
            "max_steps": 2000,
            "mass_kg": AFRL_MASS,
            "max_accel": AFRL_ACCEL,
        },
        parameters={
            "hold_point": (0.0, 0.0, 0.0),
            "capture_radius": AFRL_DOCKING_RADIUS,
            "capture_speed": 0.2,
            "hold_time": 0.0,
            "keep_out": 0.0,
            "full_speed_limit": (0.2 - 2.0 * AFRL_N * AFRL_DOCKING_RADIUS, 2.0 * AFRL_N),
            "out_of_bounds": 800.0,
            "end_on_arrival": True,
            "cam_impulses": ((0.3, 0.0, 0.0),),
            "soft": ("speed_limit",),
        },
        dispersions={
            "start_pos": AngleShell(100.0, 150.0),
            "start_vel": LimitedSpeed(0.0, 0.8, 0.2, 2.0 * AFRL_N, AFRL_DOCKING_RADIUS),
            **LATITUDE,
        },
        tags=("docking", "afrl", "comparison", "speed-limit"),
        observation=Observation(note="the state decides the next state"),
        oracle={
            # Table II of the paper, PPO without run time assurance, 1 N: IQM +- std over 10
            # training seeds x 100 episodes, pooled per episode.
            "ppo_success": 0.57,
            "ppo_delta_v_l1_mps": (13.2319, 1.7153),
            "ppo_final_speed_mps": (0.0141, 0.0043),
            "ppo_length_s": (1780.944, 237.6564),
        },
    )
)
# AFRL translational inspection (van Wijk, Dunlap, Majji and Hobbs, "Deep Reinforcement Learning
# for Autonomous Spacecraft Inspection using Illumination", AAS 23-342, 2023, arXiv 2308.02743):
# every point of a 10 m chief imaged within two Sun orbits, the Sun in the orbit plane.
AFRL_INSPECTION = register(
    ScenarioSpec(
        name="afrl-inspection",
        version=1,
        entry_point="vbar.scenarios.inspection:InspectionEnv",
        summary="The AFRL illuminated inspection task: image every lit point of a 10 m chief.",
        config={
            **CONFIG,
            "altitude_m": AFRL_ALTITUDE,
            "dt": 10.0,
            "max_steps": 1224,
            "mass_kg": AFRL_MASS,
            "max_accel": AFRL_ACCEL,
        },
        parameters={
            "points": 100,
            "target_radius": 10.0,
            "max_range": 800.0,
            "sun_exclusion": 0.0,
            "keep_out": 15.0,
            "coverage": 1.0,
            "shadow": False,
            "out_of_bounds": 800.0,
        },
        dispersions={
            "start_pos": AngleShell(50.0, 100.0),
            "start_vel": Shell(0.0, 0.3),
            "beta": Fixed(0.0),
            "sun_phase": Uniform(0.0, TWO_PI),
            **LATITUDE,
        },
        tags=("inspection", "afrl", "comparison", "lighting"),
        observation=Observation(
            hidden=("coverage set",),
            note="the imaged count is in the agent's info; the full set is the camera's record",
        ),
        oracle={
            # The paper's minimum completion time, pi / n, and its IQM results (tables 2 and 3).
            "minimum_time_s": math.pi / AFRL_N,
            "ppo_points_binary_ray_pct": 99.83,
            "ppo_delta_v_binary_ray_mps": 18.08,
            "ppo_length_binary_ray_s": 3217.0,
        },
    )
)
# SpaceGym (Supreeth and Mehta, "SpaceGym: A Gymnasium-Based Benchmark for Deep Reinforcement
# Learning in Spacecraft Proximity Operations", CVPR 2026 Workshops, sections 3.1 and 3.2): a 500 kg
# chaser with 10 N per axis on a 400 km circular orbit (a = 6778 km), 1 s steps.
SPACEGYM_ALTITUDE = 6778e3 - R_EARTH
SPACEGYM_CONFIG = {
    **CONFIG,
    "altitude_m": SPACEGYM_ALTITUDE,
    "dt": 1.0,
    "mass_kg": 500.0,
    "max_accel": 10.0 / 500.0,
    "propellant_kg": 50.0,
}
SPACEGYM_DOCKING = register(
    ScenarioSpec(
        name="spacegym-docking",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary="The SpaceGym docking task: reach 1 m of the target below 0.1 m/s within 500 s.",
        config={**SPACEGYM_CONFIG, "max_steps": 500},
        parameters={
            "hold_point": (0.0, 0.0, 0.0),
            "capture_radius": 1.0,
            "capture_speed": 0.1,
            "hold_time": 0.0,
            "keep_out": 0.0,
            "out_of_bounds": 200.0,
            "cam_impulses": ((0.3, 0.0, 0.0),),
        },
        dispersions={
            "start_pos": Shell(50.0, 100.0),
            "start_vel": Shell(0.0, 0.5),
            **LATITUDE,
        },
        tags=("docking", "spacegym", "comparison"),
        observation=Observation(note="the state decides the next state"),
        oracle={"sac_success": (0.82, 0.03), "ppo_success": (0.70, 0.05)},
    )
)
SPACEGYM_RENDEZVOUS = register(
    ScenarioSpec(
        name="spacegym-rendezvous",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary=(
            "The SpaceGym rendezvous task: from 800 to 1200 m to 10 m of the target below 0.5 m/s, "
            "outside a 5 m keep-out sphere, within 1000 s."
        ),
        config={**SPACEGYM_CONFIG, "max_steps": 1000},
        parameters={
            "hold_point": (0.0, 0.0, 0.0),
            "capture_radius": 10.0,
            "capture_speed": 0.5,
            "hold_time": 0.0,
            "keep_out": 5.0,
            "terminate_on_violation": True,
            "out_of_bounds": 2000.0,
            "cam_impulses": ((0.3, 0.0, 0.0),),
        },
        dispersions={
            "start_pos": Shell(800.0, 1200.0),
            "start_vel": Shell(0.0, 1.0),
            **LATITUDE,
        },
        tags=("rendezvous", "spacegym", "comparison"),
        observation=Observation(note="the state decides the next state"),
        oracle={"sac_success": (0.62, 0.05), "ppo_success": (0.55, 0.06)},
    )
)

INSPECTION = register(
    ScenarioSpec(
        name="inspection",
        version=2,
        entry_point="vbar.scenarios.inspection:InspectionEnv",
        summary="Image 95% of the target's lit surface points from within 200 m.",
        config={**CONFIG, "max_steps": 1140},
        parameters={
            "points": 100,
            "target_radius": 5.0,
            "max_range": 200.0,
            "sun_exclusion": math.radians(30.0),
            "keep_out": 10.0,
            "coverage": 0.95,
        },
        dispersions={
            "start_pos": Shell(50.0, 150.0),
            # v2: no start on a slowly drifting orbit about the target, from which free drift
            # imaged 95% of the points on 5 of 300 validation seeds of v1.
            "start_vel": DriftingVelocity((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05), 25.0, N),
            "beta": Uniform(math.radians(-75.0), math.radians(75.0)),
            "sun_phase": Uniform(0.0, TWO_PI),
            **LATITUDE,
        },
        tags=("inspection", "lighting", "coverage"),
        observation=Observation(
            hidden=("coverage set",),
            note="the imaged count is in the agent's info; the full set is the camera's record",
        ),
        variants={
            "inertial": {"target": InertialPointing(), "tags": ("rotating-target",)},
            "offset": {"target": FixedOffset(pitch=math.radians(20.0)), "tags": ("offset",)},
            **NAV_VARIANTS,
        },
    )
)

CLOSING = register(
    ScenarioSpec(
        name="closing",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary=(
            "From the S2 hold 3 km behind an ISS-class target to the S3 hold, entering the "
            "approach ellipsoid only after the GO."
        ),
        config={**CONFIG, **ISS_CONFIG, "max_steps": 1140},
        parameters={
            "hold_point": (0.0, -400.0, 0.0),
            "capture_radius": 5.0,
            "capture_speed": 0.05,
            "hold_time": 600.0,
            "keep_out": 200.0,
            "closing_limit": (0.2, 4.0 * N_ISS),
            "passive_axis": "vbar",
            "zone": (1000.0, 2000.0, 1000.0),
            "plume_range": 300.0,
            "passive_exclusion": 200.0,
            "soft": ("closing_speed_limit", "passive_speed"),
        },
        dispersions={
            "start_pos": Uniform((-20.0, -3020.0, -20.0), (20.0, -2980.0, 20.0)),
            "start_vel": Uniform((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
            "go_time": Choice(tuple(600.0 + 10.0 * k for k in range(181))),
            **LATITUDE,
        },
        tags=("closing", "iss", "vbar", "hold", "go-event"),
        rules=(
            "always (t <= go_time implies zone_entry >= 0)",
            "eventually always[0, hold_time] (|r - hold_point| < capture_radius "
            "and |v| < capture_speed)",
        ),
        observation=Observation(
            hidden=("dwell elapsed",),
            note="the GO time is public in the zone constraint; the state decides the next state",
        ),
        oracle=hold_oracle(N_ISS, 2600.0),
        variants={
            "gates-dual-coelliptic": {
                "parameters": {
                    "closing_limit": None,
                    "gates": GATES["dual-coelliptic"],
                    "soft": ("gate_table", "passive_speed"),
                },
                "tags": ("gates",),
            },
            **NAV_VARIANTS,
        },
    )
)

RBAR_ACQUISITION = register(
    ScenarioSpec(
        name="rbar-acquisition",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary="From a V-bar hold 1 to 5 km behind an ISS-class target to a hold 500 m below it.",
        config={**CONFIG, **ISS_CONFIG, "max_steps": 1140},
        parameters={
            "hold_point": (-500.0, 0.0, 0.0),
            "capture_radius": 5.0,
            "capture_speed": 0.05,
            "hold_time": 300.0,
            "keep_out": 200.0,
            "closing_limit": (0.2, 4.0 * N_ISS),
            "passive_axis": "rbar",
            "plume_range": 300.0,
            "passive_exclusion": 200.0,
            "soft": ("closing_speed_limit", "passive_speed"),
        },
        dispersions={
            "start_pos": Uniform((-20.0, -5000.0, -20.0), (20.0, -1000.0, 20.0)),
            "start_vel": Uniform((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
            **LATITUDE,
        },
        tags=("closing", "iss", "rbar", "hold", "fly-around"),
        observation=Observation(hidden=("dwell elapsed",), note="as approach-vbar"),
        oracle={
            "tangential_fly_around_mps": 2.0 * N_ISS * 500.0,
            "radial_fly_around_mps": 3.0 * N_ISS * 500.0,
            "hold_cost_mps": 3.0 * N_ISS**2 * 500.0 * 300.0,
        },
        variants=NAV_VARIANTS,
    )
)

BERTHING_BOX = register(
    ScenarioSpec(
        name="berthing-box",
        version=1,
        entry_point="vbar.scenarios.berthing:BerthingEnv",
        summary=(
            "Forced R-bar approach to a capture box 15 m below the target and a 60 s free drift "
            "inside it."
        ),
        config=CONFIG,
        parameters={
            "box": (-15.0, 0.0, 0.0),
            "capture_radius": 0.5,
            "capture_speed": 0.01,
            "capture_time": 60.0,
            "keep_out": 10.0,
            "cone": (math.radians(10.0), 50.0),
        },
        dispersions={
            "start_pos": Uniform((-80.0, -5.0, -5.0), (-50.0, 5.0, 5.0)),
            "start_vel": Uniform((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
            **LATITUDE,
        },
        tags=("berthing", "rbar", "corridor"),
        observation=Observation(
            hidden=("capture window state",),
            note="thrust is inhibited once the box is reached; the agent's own clock tells it when",
        ),
        oracle={"drift_from_rest_120s_m": 15.0 * (3.0 - 3.0 * math.cos(120.0 * N))},
        variants=NAV_VARIANTS,
    )
)

ORBT_RBAR = register(
    ScenarioSpec(
        name="orbt-rbar",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary=(
            "The Shuttle's optimized R-bar targeted rendezvous from Ti to a hold 600 ft below "
            "the ISS."
        ),
        config={**CONFIG, **ISS_CONFIG, "max_steps": 600},
        parameters={
            "hold_point": (-600.0 * FT, 0.0, 0.0),
            "capture_radius": 5.0,
            "capture_speed": 0.05,
            "hold_time": 300.0,
            "keep_out": 100.0 * FT,
            "gates": GATES["sts-130-orbt"],
            "waypoints": ((-1823.0 * FT, -885.0 * FT, 0.0),),
            "plume_range": 1000.0 * FT,
            "passive_exclusion": 200.0,
            "soft": ("gate_table",),
        },
        dispersions={
            "start_pos": Uniform(
                (-0.2 * 1852.0 - 50.0, -8.0 * 1852.0 - 200.0, -50.0),
                (-0.2 * 1852.0 + 50.0, -8.0 * 1852.0 + 200.0, 50.0),
            ),
            "start_vel": Uniform(
                (-0.05, 1.5 * N_ISS * 0.2 * 1852.0 - 0.05, -0.05),
                (0.05, 1.5 * N_ISS * 0.2 * 1852.0 + 0.05, 0.05),
            ),
            **LATITUDE,
        },
        tags=("shuttle", "rbar", "hold", "far-range", "gates"),
        rules=("eventually (|r - waypoint_1| < 100)",),
        observation=Observation(hidden=("dwell elapsed",), note="as approach-vbar"),
        oracle={"flown_burns_mps": (6.65 + 1.2 + 1.93) * FT},
        variants=NAV_VARIANTS,
    )
)

RPM_HOLD = register(
    ScenarioSpec(
        name="rpm-hold",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary=(
            "Hold 600 ft below the ISS through a 270 s thrust inhibit standing in for the pitch "
            "manoeuvre."
        ),
        config={**CONFIG, **ISS_CONFIG, "max_steps": 180},
        parameters={
            "hold_point": (-600.0 * FT, 0.0, 0.0),
            "capture_radius": 5.0,
            "capture_speed": 0.05,
            "hold_time": 300.0,
            "keep_out": 580.0 * FT,
            "thrust_inhibit": ((300.0, 570.0),),
            "plume_range": 1000.0 * FT,
            "passive_exclusion": 200.0,
        },
        dispersions={
            "start_pos": Uniform((-600.0 * FT - 3.0, -3.0, -3.0), (-600.0 * FT + 3.0, 3.0, 3.0)),
            "start_vel": Uniform((0.3 * FT - 0.02, -0.02, -0.02), (0.3 * FT + 0.02, 0.02, 0.02)),
            **LATITUDE,
        },
        tags=("shuttle", "rbar", "hold", "inhibit"),
        rules=("always[300, 570] (thrust == 0)", "always (|r| >= 580 ft)"),
        observation=Observation(
            hidden=("dwell elapsed",),
            note="the inhibit window is public in the goal; the state decides the next state",
        ),
        oracle={"closing_speed_returning_to_start_radius_mps": 0.318 * FT},
        variants=NAV_VARIANTS,
    )
)

TORVA = register(
    ScenarioSpec(
        name="torva",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary=(
            "Twice-orbital-rate transition from 575 ft on the +R-bar to a hold 310 ft on the "
            "+V-bar."
        ),
        config={**CONFIG, **ISS_CONFIG, "max_steps": 240},
        parameters={
            "hold_point": (0.0, 310.0 * FT, 0.0),
            "capture_radius": 5.0,
            "capture_speed": 0.05,
            "hold_time": 300.0,
            "keep_out": 100.0 * FT,
            "plume_range": 1000.0 * FT,
            "passive_exclusion": 100.0,
        },
        dispersions={
            "start_pos": Uniform((-575.0 * FT - 3.0, -3.0, -3.0), (-575.0 * FT + 3.0, 3.0, 3.0)),
            "start_vel": Uniform((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
            **LATITUDE,
        },
        tags=("shuttle", "fly-around", "vbar", "hold"),
        observation=Observation(hidden=("dwell elapsed",), note="as approach-vbar"),
        oracle={
            "twice_rate_quarter_lap_mps": 5.5 * N_ISS * 575.0 * FT,
            "inertial_quarter_lap_mps": 4.422 * N_ISS * 575.0 * FT,
            "quarter_lap_duration_s": TWO_PI / N_ISS / 8.0,
        },
        variants=NAV_VARIANTS,
    )
)

VBAR_FINAL_SHUTTLE = register(
    ScenarioSpec(
        name="vbar-final-shuttle",
        version=1,
        entry_point="vbar.scenarios.docking:DockingEnv",
        summary=(
            "The Shuttle's +V-bar final approach from 320 ft to contact with its gates and "
            "corridors."
        ),
        config={**CONFIG, **ISS_CONFIG, "dt": 5.0, "max_steps": 720},
        parameters={
            "port": (0.0, 2.0, 0.0),
            "port_axis": (0.0, 1.0, 0.0),
            "body_radius": 2.0,
            "preset": "apds",
            "cones": ((math.radians(8.0), 400.0 * FT), (math.radians(5.0), 30.0)),
            # The flown gates are ranges to the port; the table is in range to the centre.
            "gates": tuple((r + 2.0, v) for r, v in GATES["shuttle-vbar-iss"]),
            "hold": (30.0 * FT, 300.0),
            "commit_radius": 1.0,
            "passive_exclusion": 50.0,
            "plume_range": 1000.0 * FT,
            "soft": ("gate_table",),
        },
        dispersions={
            "start_pos": Uniform(
                (-3.0, 2.0 + 320.0 * FT - 5.0, -3.0), (3.0, 2.0 + 320.0 * FT + 5.0, 3.0)
            ),
            "start_vel": Uniform((-0.01, -0.2 * FT - 0.01, -0.01), (0.01, -0.2 * FT + 0.01, 0.01)),
            **LATITUDE,
        },
        tags=("shuttle", "docking", "vbar", "gates", "corridor-shuttle"),
        observation=Observation(hidden=("contact event",), note="as docking"),
        oracle={"measured_contact_speed_mps": 0.113 * FT, "planned_contact_speed_mps": 0.10 * FT},
        variants=NAV_VARIANTS,
    )
)

HST_RBAR_GRAPPLE = register(
    ScenarioSpec(
        name="hst-rbar-grapple",
        version=1,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary=(
            "+R-bar approach to an inertially pointing telescope and station keeping at 35 ft "
            "on its axis."
        ),
        config={**CONFIG, "altitude_m": HST_ALTITUDE, "max_steps": 1080},
        parameters={
            "hold_point": (-35.0 * FT, 0.0, 0.0),
            "capture_radius": 1.0,
            "capture_speed": 0.01,
            "hold_time": 120.0,
            "keep_out": 8.0,
            "gates": GATES["shuttle-rbar-orbt"],
            "cone": (math.radians(10.0), 50.0 * FT),
            "waypoints": ((-150.0 * FT, 0.0, 0.0),),
            "plume_range": 1500.0 * FT,
            "passive_exclusion": 50.0,
            "soft": ("gate_table",),
        },
        dispersions={
            "start_pos": Uniform(
                (-2000.0 * FT - 10.0, -10.0, -10.0), (-2000.0 * FT + 10.0, 10.0, 10.0)
            ),
            "start_vel": Uniform((-0.02, -0.02, -0.02), (0.02, 0.02, 0.02)),
            "target_phase": Uniform(0.0, TWO_PI),
            **LATITUDE,
        },
        target=InertialPointing(),
        tags=("shuttle", "rbar", "grapple", "rotating-axis", "hold", "gates"),
        rules=(
            "eventually always[0, 120] (|r - hold_point| < 1 and |v| < 0.01 "
            "and inside the axis cone)",
        ),
        observation=Observation(
            hidden=("dwell elapsed",),
            note="the axis phase and rate are public through the goal's target entry",
        ),
        variants=NAV_VARIANTS,
    )
)

SEPARATION_FLYAROUND = register(
    ScenarioSpec(
        name="separation-flyaround",
        version=1,
        entry_point="vbar.scenarios.departure:DepartureEnv",
        summary=(
            "Undock from the +V-bar port, leave through the departure corridor and reach 2000 "
            "ft on a non-returning path."
        ),
        config={**CONFIG, **ISS_CONFIG, "max_steps": 540},
        parameters={
            "port": (0.0, 30.0, 0.0),
            "port_axis": (0.0, 1.0, 0.0),
            "body_radius": 30.0,
            "cone": (math.radians(10.0), 400.0 * FT),
            "clear_range": 2000.0 * FT,
            "clear_radius": 200.0,
            "plume_range": 1000.0 * FT,
        },
        dispersions={
            "separation": Uniform(0.0, 0.5),
            "separation_speed": Uniform(0.15 * FT, 0.2 * FT),
            **LATITUDE,
        },
        tags=("shuttle", "departure", "vbar", "fly-around"),
        observation=Observation(note="the state decides the next state"),
        oracle={"sep1_radial_mps": 1.5 * FT, "sep2_retrograde_mps": 3.0 * FT},
        variants={
            "reboost": {
                "target": Maneuvering(schedule=((900.0, 600.0, (0.0, 5e-4, 0.0)),)),
                "tags": ("maneuvering-target",),
            },
            **NAV_VARIANTS,
        },
    )
)

SUITE_V1 = register_suite(
    Suite(
        "v1",
        1,
        (
            SuiteEntry("approach-vbar"),
            SuiteEntry("approach-rbar"),
            SuiteEntry("docking"),
            SuiteEntry("inspection"),
        ),
    )
)
SUITE_V1_NAV1 = register_suite(
    Suite(
        "v1-nav1",
        1,
        (
            SuiteEntry("approach-vbar:nav1"),
            SuiteEntry("approach-rbar:nav1"),
            SuiteEntry("docking:nav1"),
            SuiteEntry("inspection:nav1"),
        ),
    )
)
SUITE_V1_OPERATIONS = register_suite(
    Suite(
        "v1-operations",
        2,
        (
            SuiteEntry("closing"),
            SuiteEntry("rbar-acquisition"),
            SuiteEntry("berthing-box"),
            SuiteEntry("docking:axis"),
            SuiteEntry("docking:lit"),
            SuiteEntry("docking-offset"),
            SuiteEntry("inertial-target"),
            SuiteEntry("approach-hbar"),
        ),
    )
)
SUITE_V1_SHUTTLE = register_suite(
    Suite(
        "v1-shuttle",
        1,
        (
            SuiteEntry("orbt-rbar"),
            SuiteEntry("rpm-hold"),
            SuiteEntry("torva"),
            SuiteEntry("vbar-final-shuttle"),
            SuiteEntry("hst-rbar-grapple"),
            SuiteEntry("separation-flyaround"),
            SuiteEntry("docking:corridor-shuttle"),
        ),
    )
)

# The headline benchmark: five scenarios, each without and with navigation noise, reported in
# full. The rest of the catalogue stays available outside the headline table.
CORE = ("approach-vbar", "approach-rbar", "docking:axis", "docking-afrl", "inspection")
SUITE_CORE = register_suite(
    Suite(
        "core",
        1,
        tuple(
            SuiteEntry(name, "validation", 100) for base in CORE for name in (base, f"{base}:nav1")
        ),
    )
)
# The same runs on the hidden test split, for scored comparisons.
SUITE_CORE_TEST = register_suite(
    Suite(
        "core-test",
        1,
        tuple(SuiteEntry(name, "test", 100) for base in CORE for name in (base, f"{base}:nav1")),
    )
)

__all__ = ["CORE", "FT", "GATES", "N", "N_HST", "N_ISS", "feet", "hold_oracle"]
