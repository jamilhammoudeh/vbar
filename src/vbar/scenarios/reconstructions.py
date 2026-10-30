"""Real rendezvous events rebuilt from public reports, as scenarios.

Each scenario is reconstructed from public reports; none uses an operator's data or claims to
match it. Values the reports give are cited in `docs/reconstructions.md`; values they do not give
are dispersions over a stated range or marked as this benchmark's assumption there. Each comes
with a variant that removes the failure (`:nominal`), so the same controller can be scored with
and without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import brentq

from ..dynamics import cw_stm
from ..faults import InSphere, Intermittent, RangeLoss, SensorBias, ThrusterOff, Until
from .hold import HoldParams, HoldPointEnv
from .registry import register
from .spec import Choice, Observation, ScenarioSpec, Uniform

__all__ = ["AimedApproachEnv", "AimedApproachParams"]

FT = 0.3048


@dataclass(frozen=True, kw_only=True)
class AimedApproachParams(HoldParams):
    """Hold-point parameters plus the point the start trajectory was aimed at.

    Parameters
    ----------
    aim_point : tuple of float
        Point (m, Hill axes) the free-drift start trajectory passes through; its z component is
        replaced by the episode's drawn out-of-plane offset.
    """

    aim_point: tuple[float, float, float] = (0.0, 0.0, 0.0)


class AimedApproachEnv(HoldPointEnv):
    """A hold-point approach that starts on a coasting arc aimed at a point near the target.

    The start position is at the drawn `start_range` (m), `start_elevation` (rad above the local
    horizontal, with the horizontal part behind the target, -y) and `out_of_plane` offset (m,
    z). The start velocity is the Clohessy-Wiltshire velocity that reaches `aim_point` (with the
    same out-of-plane offset) after a coast whose length makes the initial range rate equal to
    minus the drawn `closure_speed` (m/s); `coast_s` holds that coast's length (s).

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec; its parameters are `AimedApproachParams` and its dispersions include
        `start_range`, `start_elevation`, `out_of_plane` and `closure_speed`.
    config : mapping, optional
        `EnvConfig` field overrides (SI units).

    Raises
    ------
    ValueError
        At reset, if no coast shorter than 0.45 orbit gives the drawn closing rate (dispersions
        overridden beyond the catalogue's ranges can ask for one).
    """

    params_type: ClassVar[type] = AimedApproachParams
    params: AimedApproachParams

    def _sample_start(self, rng: np.random.Generator) -> np.ndarray:
        rho = float(self.draw["start_range"])
        elevation = float(self.draw["start_elevation"])
        z = float(self.draw["out_of_plane"])
        closure = float(self.draw["closure_speed"])
        horizontal = math.sqrt(rho * rho - z * z)
        p0 = np.array([horizontal * math.sin(elevation), -horizontal * math.cos(elevation), z])
        aim = np.asarray(self.params.aim_point, dtype=float).copy()
        aim[2] = z

        def velocity(coast: float) -> np.ndarray:
            phi = cw_stm(self.n, coast)
            return np.linalg.solve(phi[:3, 3:], aim - phi[:3, :3] @ p0)

        def range_rate_gap(coast: float) -> float:
            return float(p0 @ velocity(coast)) / rho + closure

        # The closing rate falls as the coast lengthens; past half an orbit the out-of-plane
        # block of the transition matrix is singular, so the search stops short of it.
        coast = float(brentq(range_rate_gap, 10.0, 0.45 * 2.0 * math.pi / self.n, xtol=1e-6))
        self.coast_s = coast
        return np.concatenate([p0, velocity(coast)])


INCLINATION_MIR = math.radians(51.6)

PROGRESS_M34 = register(
    ScenarioSpec(
        name="recon-progress-m34",
        version=0,
        entry_point="vbar.scenarios.reconstructions:AimedApproachEnv",
        summary=(
            "Reconstructed from public reports: Progress M-34 and Mir, 25 June 1997. A remotely "
            "piloted approach from 6 to 7 km above and behind the station with range and range "
            "rate lost; stop at a point 200 m behind it without hitting it."
        ),
        config={
            "altitude_m": 390e3,
            "inclination": INCLINATION_MIR,
            "dt": 10.0,
            "max_steps": 300,
            "mass_kg": 7000.0,
            "max_accel": 0.05,
            "faults": (RangeLoss(assumed_range_m=6500.0, name="range_lost"),),
        },
        parameters={
            "hold_point": (0.0, -200.0, 0.0),
            "aim_point": (0.0, -1000.0, 0.0),
            "capture_radius": 10.0,
            "capture_speed": 0.1,
            "hold_time": 60.0,
            "keep_out": 25.0,
            "terminate_on_violation": True,
        },
        dispersions={
            "start_range": Uniform(6000.0, 7000.0),
            "start_elevation": Uniform(math.radians(30.0), math.radians(70.0)),
            "out_of_plane": Choice((-400.0, 400.0)),
            "closure_speed": Uniform(5.0, 6.5),
            "argument_of_latitude": Uniform(0.0, 2.0 * math.pi),
        },
        tags=("reconstruction", "fault", "range-loss"),
        observation=Observation(
            hidden=("range", "range rate"),
            note="the reading is the line of sight and its rate scaled to an assumed 6.5 km",
        ),
        variants={"nominal": {"config": {"faults": ()}, "tags": ("nominal",)}},
    )
)

ETS7 = register(
    ScenarioSpec(
        name="recon-ets7",
        version=0,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary=(
            "Reconstructed from public reports: ETS-VII second docking experiment, August 1998. "
            "A V-bar approach from 520 m ahead of the target to its 2 m holding point while the "
            "thrusters that brake the approach fail intermittently."
        ),
        config={
            "altitude_m": 550e3,
            "inclination": math.radians(35.0),
            "dt": 5.0,
            "max_steps": 2880,
            "mass_kg": 2500.0,
            "max_accel": 4e-4,
            "faults": (
                ThrusterOff(
                    axis=1,
                    direction=1,
                    when=Intermittent(probability=0.3),
                    name="braking_thrusters_out",
                ),
            ),
        },
        parameters={
            "hold_point": (0.0, 2.0, 0.0),
            "capture_radius": 0.2,
            "capture_speed": 0.02,
            "hold_time": 900.0,
            "keep_out": 1.0,
            "terminate_on_violation": True,
        },
        dispersions={
            "start_pos": Uniform((-5.0, 515.0, -5.0), (5.0, 525.0, 5.0)),
            "start_vel": Uniform((-0.002, -0.002, -0.002), (0.002, 0.002, 0.002)),
            "argument_of_latitude": Uniform(0.0, 2.0 * math.pi),
        },
        tags=("reconstruction", "fault", "thruster", "vbar"),
        observation=Observation(hidden=("thruster health",), note="faults show only in the motion"),
        variants={"nominal": {"config": {"faults": ()}, "tags": ("nominal",)}},
    )
)

# DART's navigation reset to biased GPS velocity until its camera's range mode took over, which
# happened only on entering a 6.3 m sphere 200 m behind the target; DART missed the sphere.
WAYPOINT = InSphere(center_m=(0.0, -200.0, 0.0), radius_m=6.3)


DART = register(
    ScenarioSpec(
        name="recon-dart",
        version=0,
        entry_point="vbar.scenarios.hold:HoldPointEnv",
        summary=(
            "Reconstructed from public reports: DART and MUBLCOM, 15 April 2005. A V-bar approach "
            "from 1 km behind the target to station-keeping 50 ft behind it, with a biased "
            "velocity reading until a 6.3 m waypoint sphere 200 m behind the target is entered."
        ),
        config={
            "altitude_m": 750e3,
            "inclination": math.radians(90.0),
            "dt": 10.0,
            "max_steps": 720,
            "mass_kg": 360.0,
            "max_accel": 0.01,
            "isp": 60.0,
            "faults": (
                SensorBias(
                    velocity_mps=(0.4, 0.8), when=Until(condition=WAYPOINT), name="velocity_bias"
                ),
            ),
        },
        parameters={
            "hold_point": (0.0, -50.0 * FT, 0.0),
            "capture_radius": 2.0,
            "capture_speed": 0.05,
            "hold_time": 600.0,
            "keep_out": 5.0,
            "terminate_on_violation": True,
        },
        dispersions={
            "start_pos": Uniform((-10.0, -1010.0, -10.0), (10.0, -990.0, 10.0)),
            "start_vel": Uniform((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
            "argument_of_latitude": Uniform(0.0, 2.0 * math.pi),
        },
        tags=("reconstruction", "fault", "sensor-bias", "vbar"),
        observation=Observation(
            hidden=("velocity bias",),
            note="the velocity reading carries a bias drawn per episode until the waypoint",
        ),
        variants={"nominal": {"config": {"faults": ()}, "tags": ("nominal",)}},
    )
)
