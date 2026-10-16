"""Propulsion: the acceleration a command produces and the propellant it costs.

Commands are fractions of the actuator's maximum acceleration along each Hill axis. One
thruster pair per Hill axis ("box") limits each fraction to [-1, 1] and spends propellant for
each axis separately, the 1-norm of delta-v; one steerable engine ("ball") limits the command's
length to 1, keeping its direction, and spends propellant for the 2-norm. The actuator holds the
commanded acceleration constant over each burn, so the thrust it needs falls as propellant burns
off; mass follows the rocket equation m1 = m0 exp(-dv / (Isp g0)) in the layout's norm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..conventions import G0

LAYOUTS = ("box", "ball")
# Command switch times closer than this (s) to an interval boundary fall on it.
_TIME_TOL = 1e-9


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
    rocket equation never exhausts. A thruster cannot deliver an impulse below `min_impulse`
    (N s); a command that would is dropped (per axis for the box layout). Commands take effect
    `delay` s after they are issued.
    """

    def __init__(
        self,
        max_accel: float = 0.02,
        layout: str = "box",
        isp: float = 220.0,
        mass: float = 500.0,
        propellant: float | None = None,
        min_impulse: float = 0.0,
        delay: float = 0.0,
    ):
        if layout not in LAYOUTS:
            raise ValueError(f"unknown thruster layout {layout!r}, expected box or ball")
        self.max_accel = max_accel
        self.layout = layout
        self.min_impulse = min_impulse
        self.delay = delay
        self.exhaust_speed = isp * G0
        self.initial_mass = mass
        self.dry_mass = mass - (mass if propellant is None else propellant)
        if not 0.0 <= self.dry_mass < mass:
            raise ValueError("propellant must be positive and no more than the mass")
        self.reset()

    def reset(self) -> None:
        self.mass = self.initial_mass
        self.t = 0.0
        self._queue: list[tuple[float, np.ndarray]] = []

    @property
    def propellant(self) -> float:
        """Propellant left, kg."""
        return self.mass - self.dry_mass

    def _spend_rate(self, accel: np.ndarray) -> float:
        """Delta-v per second (m/s^2) in the layout's norm."""
        return float(np.sum(np.abs(accel)) if self.layout == "box" else np.linalg.norm(accel))

    def limit(self, command: ArrayLike, duration: float) -> tuple[np.ndarray, bool, bool]:
        """Command the thrusters can deliver when held for `duration` s.

        Returns the command and whether it was saturated or had impulses below the minimum.
        """
        u = np.asarray(command, dtype=float)
        if self.layout == "box":
            limited = np.clip(u, -1.0, 1.0)
            impulse = np.abs(limited)
        else:
            length = float(np.linalg.norm(u))
            limited = u / length if length > 1.0 else u.copy()
            impulse = np.full(3, min(length, 1.0))
        saturated = bool(np.any(limited != u))
        impulse = impulse * self.max_accel * self.mass * duration
        small = (impulse > 0.0) & (impulse < self.min_impulse)
        limited[small] = 0.0
        return limited, saturated, bool(np.any(small))

    def apply(self, command: ArrayLike, duration: float) -> Actuation:
        """Issue `command` (fractions of `max_accel` per Hill axis), held for `duration` s.

        The command takes effect `delay` s after it is issued; until then the previous one
        stays on, so a delay that is not a whole number of steps splits the interval in two.
        """
        u, saturated, below_mib = self.limit(command, duration)
        self._queue.append((self.t + self.delay, u))
        end = self.t + duration
        segments: list[tuple[float, np.ndarray]] = []
        dv_l2 = dv_l1 = used = 0.0
        empty = False
        for t0, t1, piece in self._pieces(self.t, end):
            burn = self._burn(piece * self.max_accel, t1 - t0)
            segments.extend(burn[0])
            dv_l2 += burn[1]
            dv_l1 += burn[2]
            used += burn[3]
            empty = empty or burn[4]
        self.t = end
        while len(self._queue) > 1 and self._queue[1][0] <= self.t + _TIME_TOL:
            self._queue.pop(0)
        return Actuation(tuple(segments), dv_l2, dv_l1, used, saturated, below_mib, empty)

    def pending(self) -> list[tuple[float, np.ndarray]]:
        """Issued commands not yet in effect: (seconds until they take effect, command)."""
        return [(t - self.t, u) for t, u in self._queue if t > self.t + _TIME_TOL]

    def _pieces(self, start: float, end: float):
        """(t0, t1, command) for the commands in effect between `start` and `end`."""
        times = [t for t, _ in self._queue]
        t0 = start
        while t0 < end - _TIME_TOL:
            # The command in effect is the last one issued that has taken effect by t0.
            active = [k for k, t in enumerate(times) if t <= t0 + _TIME_TOL]
            command = self._queue[active[-1]][1] if active else np.zeros(3)
            later = [t for t in times if t > t0 + _TIME_TOL]
            t1 = min([end, *later])
            if end - t1 <= _TIME_TOL:
                t1 = end
            yield t0, t1, command
            t0 = t1

    def _burn(self, accel: np.ndarray, duration: float):
        """Segments, delta-v (2-norm, 1-norm), propellant used and run-out for a constant burn."""
        rate = self._spend_rate(accel)
        zero = np.zeros(3)
        if rate == 0.0 or duration == 0.0:
            return [(duration, zero)], 0.0, 0.0, 0.0, False
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
            return [(duration, zero)], 0.0, 0.0, 0.0, True
        segments = [(burn, accel)]
        if burn < duration:
            segments.append((duration - burn, zero))
        dv_l2 = float(np.linalg.norm(accel)) * burn
        dv_l1 = float(np.sum(np.abs(accel))) * burn
        return segments, dv_l2, dv_l1, used, burn < duration
