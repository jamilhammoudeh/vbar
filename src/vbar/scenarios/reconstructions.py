"""Real rendezvous events rebuilt from public reports, as scenarios.

Each scenario is reconstructed from public reports; none uses an operator's data or claims to
match it. Values the reports give are cited in `docs/reconstructions.md`; values they do not give
are dispersions over a stated range or marked as this benchmark's assumption there. Each comes
with a variant that removes the failure (`:nominal`), so the same controller can be scored with
and without it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

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
    horizontal, with the horizontal part behind the target, -y) and out-of-plane offset (m, z) of
    magnitude `out_of_plane` on the side `out_of_plane_side` (+1 or -1). The aim point is
    `aim_point` moved by the drawn `aim_offset` (m, Hill axes), with the same out-of-plane offset.
    The start velocity is the Clohessy-Wiltshire velocity that reaches the aim point after a coast
    whose length makes the initial range rate equal to minus the drawn `closure_speed` (m/s),
    plus the drawn `velocity_scatter` (m/s); `coast_s` holds that coast's length (s). The spread
    keeps the start from being solved from the public spec and one line-of-sight reading: free
    drift is linear, so bearings alone do not fix the scale.

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
        If a dispersion is missing, a range or closing rate is not positive, the out-of-plane
        offset reaches the range, or no coast of 10 s to 0.45 orbit reaches the aim point at the
        drawn closing rate at some corner of the dispersions' supports.
    """

    params_type: ClassVar[type] = AimedApproachParams
    params: AimedApproachParams

    def __init__(self, spec: ScenarioSpec, config: Mapping[str, Any] | None = None):
        super().__init__(spec, config)
        _check_aimed_dispersions(self.n, spec.dispersions, self.params.aim_point)

    def _sample_start(self, rng: np.random.Generator) -> np.ndarray:
        draw = self.draw
        side = float(draw["out_of_plane_side"])
        x0, coast = _aimed_start(
            self.n,
            float(draw["start_range"]),
            float(draw["start_elevation"]),
            side * float(draw["out_of_plane"]),
            float(draw["closure_speed"]),
            np.asarray(self.params.aim_point, dtype=float) + np.asarray(draw["aim_offset"]),
        )
        self.coast_s = coast
        x0[3:] += np.asarray(draw["velocity_scatter"], dtype=float)
        return x0


def _check_aimed_dispersions(
    n: float, dispersions: Mapping[str, Any], aim_point: tuple[float, float, float]
) -> None:
    """Refuse dispersions an aimed start cannot fly, checked at the corners of their supports.

    Raises ValueError if a range is not positive, an out-of-plane offset reaches the range, a
    closing rate is not positive, or no coast reaches the aim point at some corner.
    """

    def support(key: str) -> tuple[np.ndarray, np.ndarray]:
        if key not in dispersions:
            raise ValueError(f"an aimed start needs the dispersion {key!r}")
        low, high = dispersions[key].bounds()
        return np.atleast_1d(np.asarray(low, dtype=float)), np.atleast_1d(
            np.asarray(high, dtype=float)
        )

    rho, elevation, z, closure, aim = (
        support(k)
        for k in ("start_range", "start_elevation", "out_of_plane", "closure_speed", "aim_offset")
    )
    support("out_of_plane_side")
    support("velocity_scatter")
    if not rho[0][0] > 0.0:
        raise ValueError("start_range must be positive")
    if not max(abs(z[0][0]), abs(z[1][0])) < rho[0][0]:
        raise ValueError("out_of_plane must stay below start_range")
    if not closure[0][0] > 0.0:
        raise ValueError("closure_speed must be positive")
    base = np.asarray(aim_point, dtype=float)
    for r, e, dz, c, ax, ay in itertools.product(
        *(pair for pair in zip(rho[0], rho[1], strict=True)),
        *(pair for pair in zip(elevation[0], elevation[1], strict=True)),
        *(pair for pair in zip(z[0], z[1], strict=True)),
        *(pair for pair in zip(closure[0], closure[1], strict=True)),
        (aim[0][0], aim[1][0]),
        (aim[0][1], aim[1][1]),
    ):
        _aimed_start(n, r, e, dz, c, base + np.array([ax, ay, 0.0]))


def _aimed_start(
    n: float, rho: float, elevation: float, z: float, closure: float, aim: np.ndarray
) -> tuple[np.ndarray, float]:
    """Start state on the CW arc through `aim` with initial range rate -`closure`, and its coast.

    `aim`'s z component is replaced by `z`. Raises ValueError when no coast of 10 s to 0.45 orbit
    gives that range rate.
    """
    horizontal = math.sqrt(rho * rho - z * z)
    p0 = np.array([horizontal * math.sin(elevation), -horizontal * math.cos(elevation), z])
    target = np.array([aim[0], aim[1], z])

    def velocity(coast: float) -> np.ndarray:
        phi = cw_stm(n, coast)
        return np.linalg.solve(phi[:3, 3:], target - phi[:3, :3] @ p0)

    def range_rate_gap(coast: float) -> float:
        return float(p0 @ velocity(coast)) / rho + closure

    # The closing rate falls as the coast lengthens; past half an orbit the out-of-plane block
    # of the transition matrix is singular, so the search stops short of it.
    low, high = 10.0, 0.45 * 2.0 * math.pi / n
    if range_rate_gap(low) * range_rate_gap(high) > 0.0:
        raise ValueError("no coast reaches the aim point at the drawn closing rate")
    coast = float(brentq(range_rate_gap, low, high, xtol=1e-6))
    return np.concatenate([p0, velocity(coast)]), coast


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
            "out_of_plane": Uniform(300.0, 500.0),
            "out_of_plane_side": Choice((-1.0, 1.0)),
            "aim_offset": Uniform((-100.0, -200.0, 0.0), (100.0, 200.0, 0.0)),
            "closure_speed": Uniform(5.0, 6.5),
            "velocity_scatter": Uniform((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)),
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
