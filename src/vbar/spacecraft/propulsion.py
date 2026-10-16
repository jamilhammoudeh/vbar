"""Propulsion: the acceleration a command produces and the propellant it costs.

Commands are fractions of the actuator's maximum acceleration along each Hill axis. The
actuator holds the commanded acceleration constant over each burn, so the thrust it needs falls
as propellant burns off; mass follows the rocket equation m1 = m0 exp(-dv / (Isp g0)), where dv
is measured in the norm of the thruster layout. One thruster pair per Hill axis ("box") spends
propellant for each axis separately, the 1-norm of delta-v; one steerable engine ("ball")
spends it for the 2-norm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..conventions import G0

LAYOUTS = ("box", "ball")


@dataclass(frozen=True)
class Actuation:
    """What the spacecraft did over one interval.

    `segments` are consecutive (duration s, Hill acceleration m/s^2) pieces covering the
    interval, so a thrust change inside it shows up as a segment boundary.
    """

    segments: tuple[tuple[float, np.ndarray], ...]
    dv_l2: float  # m/s
    dv_l1: float  # m/s
    propellant_kg: float
    saturated: bool = False
    below_mib: bool = False
    empty: bool = False


class Spacecraft:
    """Mass, propellant and the acceleration actually applied for a command.

    `max_accel` (m/s^2) is the acceleration a command of 1 asks for, `isp` is in seconds and
    `mass` (kg) is the initial wet mass. `propellant` (kg) defaults to the whole mass, which the
    rocket equation never exhausts.
    """

    def __init__(
        self,
        max_accel: float = 0.02,
        layout: str = "box",
        isp: float = 220.0,
        mass: float = 500.0,
        propellant: float | None = None,
    ):
        if layout not in LAYOUTS:
            raise ValueError(f"unknown thruster layout {layout!r}, expected box or ball")
        self.max_accel = max_accel
        self.layout = layout
        self.exhaust_speed = isp * G0
        self.initial_mass = mass
        self.dry_mass = mass - (mass if propellant is None else propellant)
        if not 0.0 <= self.dry_mass < mass:
            raise ValueError("propellant must be positive and no more than the mass")
        self.reset()

    def reset(self) -> None:
        self.mass = self.initial_mass

    @property
    def propellant(self) -> float:
        """Propellant left, kg."""
        return self.mass - self.dry_mass

    def _spend_rate(self, accel: np.ndarray) -> float:
        """Delta-v per second (m/s^2) in the layout's norm."""
        return float(np.sum(np.abs(accel)) if self.layout == "box" else np.linalg.norm(accel))

    def apply(self, command: ArrayLike, duration: float) -> Actuation:
        """Hold `command` (fractions of `max_accel` per Hill axis) for `duration` s."""
        accel = np.asarray(command, dtype=float) * self.max_accel
        rate = self._spend_rate(accel)
        zero = np.zeros(3)
        if rate == 0.0 or duration == 0.0:
            return Actuation(((duration, zero),), 0.0, 0.0, 0.0)
        burn = duration
        if self.mass <= self.dry_mass:
            burn = 0.0
        elif self.dry_mass > 0.0:
            # Burn time until the propellant is gone, from m0 exp(-rate t / ve) = m_dry.
            burn = min(duration, self.exhaust_speed / rate * math.log(self.mass / self.dry_mass))
        if burn < duration:
            used = self.mass - self.dry_mass
            self.mass = self.dry_mass
        else:
            used = -self.mass * math.expm1(-rate * burn / self.exhaust_speed)
            self.mass -= used
        if burn == 0.0:
            return Actuation(((duration, zero),), 0.0, 0.0, 0.0, empty=True)
        segments = [(burn, accel)]
        if burn < duration:
            segments.append((duration - burn, zero))
        return Actuation(
            tuple(segments),
            dv_l2=float(np.linalg.norm(accel)) * burn,
            dv_l1=float(np.sum(np.abs(accel))) * burn,
            propellant_kg=used,
            empty=burn < duration,
        )
