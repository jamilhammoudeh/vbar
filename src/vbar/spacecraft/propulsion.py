"""Propulsion: the acceleration a command produces and the propellant it costs.

Commands are fractions of the actuator's maximum acceleration along each Hill axis. One
thruster pair per Hill axis ("box") limits each fraction to [-1, 1] and spends propellant for
each axis separately, the 1-norm of delta-v; one steerable engine ("ball") limits the command's
length to 1, keeping its direction, and spends propellant for the 2-norm. The actuator holds the
commanded acceleration constant over each burn, so the thrust it needs falls as propellant burns
off; mass follows the rocket equation m1 = m0 exp(-dv / (Isp g0)) in the layout's norm. The
"thrust" drive holds the thrust instead, so the acceleration rises as the mass falls (Montenbruck
and Gill, Satellite Orbits, eqs 3.126 to 3.130).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from ..conventions import G0

LAYOUTS = ("box", "ball")
DRIVES = ("accel", "thrust")
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


@dataclass(frozen=True)
class ExecutionError:
    """Thrust execution error per burn: delivered = R (commanded (1 + delta) + w).

    After Alfriend et al., Spacecraft Formation Flying, eqs 11.1 to 11.3. `delta` is
    `scale_bias` plus `scale_sigma` times a standard normal draw per burn; `w` is Gaussian with
    `noise_sigma` per axis, in fractions of a full command; `R` rotates by the fixed rotation
    vector `misalignment` (rad, Hill axes). A burn is one command; coasting is never perturbed,
    and the delivered command is not limited again, since a scale bias means the thrusters
    deliver more or less than their rating.
    """

    scale_bias: float = 0.0
    scale_sigma: float = 0.0
    noise_sigma: float = 0.0
    misalignment: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def apply(self, command: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not np.any(command):
            return command
        delta = self.scale_bias + self.scale_sigma * rng.standard_normal()
        noisy = command * (1.0 + delta) + self.noise_sigma * rng.standard_normal(3)
        return Rotation.from_rotvec(self.misalignment).apply(noisy)


def _mass_flow_factors(x: float) -> tuple[float, float]:
    """H(x) = -ln(1 - x) / x and G(x) = (x + (1 - x) ln(1 - x)) / x^2 for burned mass fraction x.

    A constant-thrust burn from rest with initial acceleration a0 over tau gives delta-v
    a0 tau H and displacement a0 tau^2 G.
    """
    if x < 0.1:
        # Series; the closed forms cancel to x^2 when little mass is burned.
        h = sum(x**k / (k + 1) for k in range(20))
        g = sum(x**k / ((k + 2) * (k + 1)) for k in range(20))
        return h, g
    log = math.log1p(-x)
    return -log / x, (x + (1.0 - x) * log) / x**2


class Spacecraft:
    """Mass, propellant and the acceleration actually applied for a command.

    `max_accel` (m/s^2) is the acceleration a command of 1 asks for, `isp` is in seconds and
    `mass` (kg) is the initial wet mass. `propellant` (kg) defaults to the whole mass, which the
    rocket equation never exhausts. A thruster cannot deliver an impulse below `min_impulse`
    (N s); a command that would is dropped (per axis for the box layout). Commands take effect
    `delay` s after they are issued. With `drive="thrust"` a command of 1 asks for the thrust
    that gives `max_accel` at the initial mass, and it needs a dry mass (propellant below mass).
    `execution` perturbs each burn as delivered; `reset(rng)` gives it its random stream.
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
        drive: str = "accel",
        execution: ExecutionError | None = None,
    ):
        if layout not in LAYOUTS:
            raise ValueError(f"unknown thruster layout {layout!r}, expected box or ball")
        if drive not in DRIVES:
            raise ValueError(f"unknown drive {drive!r}, expected accel or thrust")
        self.max_accel = max_accel
        self.layout = layout
        self.min_impulse = min_impulse
        self.delay = delay
        self.drive = drive
        self.execution = execution
        self.max_thrust = max_accel * mass
        self.exhaust_speed = isp * G0
        self.initial_mass = mass
        self.dry_mass = mass - (mass if propellant is None else propellant)
        if not 0.0 <= self.dry_mass < mass:
            raise ValueError("propellant must be positive and no more than the mass")
        if drive == "thrust" and self.dry_mass == 0.0:
            raise ValueError("constant thrust needs propellant less than the mass")
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self.mass = self.initial_mass
        self.t = 0.0
        self.rng = rng if rng is not None else np.random.default_rng()
        # (time it takes effect, command as limited, command as delivered)
        self._queue: list[tuple[float, np.ndarray, np.ndarray]] = []

    @property
    def propellant(self) -> float:
        """Propellant left, kg."""
        return self.mass - self.dry_mass

    @property
    def max_force(self) -> float:
        """Thrust (N) of a command of 1 now, per axis for box and in total for ball."""
        return self.max_thrust if self.drive == "thrust" else self.max_accel * self.mass

    def _spend_rate(self, vector: np.ndarray) -> float:
        """Length of an acceleration or thrust in the layout's norm."""
        return float(np.sum(np.abs(vector)) if self.layout == "box" else np.linalg.norm(vector))

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
        impulse = impulse * self.max_force * duration
        small = (impulse > 0.0) & (impulse < self.min_impulse)
        limited[small] = 0.0
        return limited, saturated, bool(np.any(small))

    def apply(self, command: ArrayLike, duration: float) -> Actuation:
        """Issue `command` (fractions of `max_accel` per Hill axis), held for `duration` s.

        The command takes effect `delay` s after it is issued; until then the previous one
        stays on, so a delay that is not a whole number of steps splits the interval in two.
        """
        u, saturated, below_mib = self.limit(command, duration)
        delivered = u if self.execution is None else self.execution.apply(u, self.rng)
        self._queue.append((self.t + self.delay, u, delivered))
        end = self.t + duration
        segments: list[tuple[float, np.ndarray]] = []
        dv_l2 = dv_l1 = used = 0.0
        empty = False
        for t0, t1, piece in self._pieces(self.t, end):
            burn = self._burn(piece, t1 - t0)
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
        return [(t - self.t, u) for t, u, _ in self._queue if t > self.t + _TIME_TOL]

    def _pieces(self, start: float, end: float):
        """(t0, t1, command) for the commands in effect between `start` and `end`."""
        times = [t for t, _, _ in self._queue]
        t0 = start
        while t0 < end - _TIME_TOL:
            # The command in effect is the last one issued that has taken effect by t0.
            active = [k for k, t in enumerate(times) if t <= t0 + _TIME_TOL]
            command = self._queue[active[-1]][2] if active else np.zeros(3)
            later = [t for t in times if t > t0 + _TIME_TOL]
            t1 = min([end, *later])
            if end - t1 <= _TIME_TOL:
                t1 = end
            yield t0, t1, command
            t0 = t1

    def _burn(self, u: np.ndarray, duration: float):
        """Segments, delta-v (2-norm, 1-norm), propellant used and run-out for one command."""
        zero = np.zeros(3)
        if not np.any(u) or duration == 0.0:
            return [(duration, zero)], 0.0, 0.0, 0.0, False
        if self.mass <= self.dry_mass:
            return [(duration, zero)], 0.0, 0.0, 0.0, True
        ve = self.exhaust_speed
        if self.drive == "accel":
            accel = u * self.max_accel
            rate = self._spend_rate(accel)
            burn = duration
            if self.dry_mass > 0.0:
                # Burn time until the propellant is gone, from m0 exp(-rate t / ve) = m_dry.
                burn = min(duration, ve / rate * math.log(self.mass / self.dry_mass))
            used = -self.mass * math.expm1(-rate * burn / ve)
            dv = accel * burn
        else:
            thrust = u * self.max_thrust
            flow = self._spend_rate(thrust) / ve
            burn = min(duration, (self.mass - self.dry_mass) / flow)
            used = flow * burn
            h, g = _mass_flow_factors(used / self.mass)
            a0 = thrust / self.mass
            dv = a0 * burn * h
            # Two half-burn segments whose accelerations give the exact delta-v and displacement
            # of the rising acceleration; one mean-acceleration segment would leave
            # a^2 tau^3 / (12 ve), 1.5e-5 m for 0.02 m/s^2 over 10 s.
            halves = [(0.5 * burn, a0 * (4.0 * g - h)), (0.5 * burn, a0 * (3.0 * h - 4.0 * g))]
        if burn < duration:
            used = self.mass - self.dry_mass
            self.mass = self.dry_mass
        else:
            self.mass -= used
        segments = [(burn, accel)] if self.drive == "accel" else halves
        if burn < duration:
            segments.append((duration - burn, zero))
        dv_l2 = float(np.linalg.norm(dv))
        dv_l1 = float(np.sum(np.abs(dv)))
        return segments, dv_l2, dv_l1, used, burn < duration
