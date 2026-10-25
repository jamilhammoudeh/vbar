"""Target attitude modes and the time-varying geometry they give docking and inspection.

A target mode gives the rotation from the target's body axes to the Hill frame at any time
(Fehse, Automated Rendezvous and Docking of Spacecraft, 2003, section 5.3.1): LVLH-stabilised
(`Passive`), a fixed attitude offset such as a torque-equilibrium attitude with a limit cycle
(`FixedOffset`), inertially pointing (`InertialPointing`, whose axes turn at -n in the Hill
frame), spinning about a body axis fixed in the Hill frame (`Tumbling`), a rigid body tumbling
free of torques (`TorqueFree`) and a target that burns on a schedule
(`Maneuvering`). `Misbriefed` tells agents one mode while the truth follows another. Body axes
coincide with the Hill axes at zero attitude: a port on the -V-bar face has body position
(0, -R, 0) and axis (0, -1, 0).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from ..dynamics import Dynamics
from ..dynamics.bodies import EARTH, Body
from ..dynamics.rigid_body import TorqueFreeMotion


def rotation_z(angle: ArrayLike) -> np.ndarray:
    """Active rotation matrices about z for `angle` (rad), shape (..., 3, 3).

    Parameters
    ----------
    angle : array_like
        Rotation angle (rad), any shape.

    Returns
    -------
    numpy.ndarray
        Rotation matrices, shape (*angle.shape, 3, 3).
    """
    a = np.asarray(angle, dtype=float)
    c, s = np.cos(a), np.sin(a)
    out = np.zeros((*a.shape, 3, 3))
    out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1] = c, -s, s, c
    out[..., 2, 2] = 1.0
    return out


def rotation_about(axis: ArrayLike, angle: ArrayLike) -> np.ndarray:
    """Active rotation matrices about a unit `axis` by `angle` (rad), shape (..., 3, 3).

    Parameters
    ----------
    axis : array_like
        Rotation axis, shape (3,), nonzero; normalised here.
    angle : array_like
        Rotation angle (rad), any shape.

    Returns
    -------
    numpy.ndarray
        Rotation matrices (Rodrigues' formula), shape (*angle.shape, 3, 3).
    """
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    a = np.asarray(angle, dtype=float)[..., None, None]
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * cross + (1.0 - np.cos(a)) * (cross @ cross)


def _about_z(rate: np.ndarray) -> np.ndarray:
    """Angular velocities `rate` (rad/s) about the Hill z axis, shape (*rate.shape, 3)."""
    zero = np.zeros_like(rate)
    return np.stack([zero, zero, rate], axis=-1)


class TargetMode(Protocol):
    """The target's attitude and burn schedule as seen from the Hill frame.

    `matrix` returns the rotation from body to Hill axes at a time, `angular_velocity` the
    rate of the body axes in the Hill frame, `rate_bound` and `angular_acceleration_bound`
    upper bounds on its magnitude (rad/s) and on that of its derivative (rad/s^2), `public` the
    description agents get in the goal, and `burns` the target's scheduled burns, empty for a
    target that does not manoeuvre.
    """

    @property
    def name(self) -> str:
        """Short name of the mode, as it appears in the public description."""
        ...

    def matrix(self, t: ArrayLike, n: float) -> np.ndarray:
        """Rotation from body to Hill axes at time `t` (s), shape (..., 3, 3).

        Parameters
        ----------
        t : array_like
            Time since the episode start (s), any shape.
        n : float
            Mean motion of the target's orbit (rad/s).

        Returns
        -------
        numpy.ndarray
            Rotation matrices, shape (*t.shape, 3, 3).
        """
        ...

    def angular_velocity(self, t: ArrayLike, n: float) -> np.ndarray:
        """Angular velocity of the body axes relative to the Hill frame, in Hill axes.

        Parameters
        ----------
        t : array_like
            Time since the episode start (s), any shape.
        n : float
            Mean motion of the target's orbit (rad/s).

        Returns
        -------
        numpy.ndarray
            Angular velocity (rad/s), shape (*t.shape, 3).
        """
        ...

    def rate_bound(self, n: float) -> float:
        """Upper bound on the attitude rate (rad/s).

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        float
            Bound on the angular rate of the body axes in the Hill frame (rad/s).
        """
        ...

    def angular_acceleration_bound(self, n: float) -> float:
        """Upper bound on the rate of change of `angular_velocity` (rad/s^2).

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        float
            Bound on |d w / dt| for the body rate in the Hill frame, in Hill axes (rad/s^2).
        """
        ...

    def public(self, n: float) -> dict[str, Any]:
        """The mode as agents are told it.

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        dict
            "mode" and the mode's parameters (rad, rad/s, s, m/s^2).
        """
        ...

    def burns(self) -> tuple[tuple[float, float, tuple[float, float, float]], ...]:
        """Scheduled target burns as (start s, duration s, Hill acceleration m/s^2).

        Returns
        -------
        tuple
            The burns in start order; empty when the target does not manoeuvre.
        """
        ...


@dataclass(frozen=True)
class Passive:
    """LVLH-stabilised: body axes stay aligned with the Hill axes."""

    name: ClassVar[str] = "passive"

    def matrix(self, t: ArrayLike, n: float) -> np.ndarray:
        """Identity at every time.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        numpy.ndarray
            Identity matrices, shape (*t.shape, 3, 3).
        """
        shape = np.shape(np.asarray(t, dtype=float))
        return np.broadcast_to(np.eye(3), (*shape, 3, 3)).copy()

    def angular_velocity(self, t: ArrayLike, n: float) -> np.ndarray:
        """Zero at every time.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        numpy.ndarray
            Zeros (rad/s), shape (*t.shape, 3).
        """
        return np.zeros((*np.shape(np.asarray(t, dtype=float)), 3))

    def rate_bound(self, n: float) -> float:
        """Zero: the body axes do not turn in the Hill frame.

        Parameters
        ----------
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        float
            0.0 rad/s.
        """
        return 0.0

    def angular_acceleration_bound(self, n: float) -> float:
        """Zero: the rate is constant.

        Parameters
        ----------
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        float
            Bound on the derivative of `angular_velocity` (rad/s^2).
        """
        return 0.0

    def public(self, n: float) -> dict[str, Any]:
        """The mode's name.

        Parameters
        ----------
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        dict
            {"mode": "passive"}.
        """
        return {"mode": self.name}

    def burns(self) -> tuple[tuple[float, float, tuple[float, float, float]], ...]:
        """No burns.

        Returns
        -------
        tuple
            Empty.
        """
        return ()


@dataclass(frozen=True)
class FixedOffset:
    """A constant pitch about the orbit normal, with an optional attitude limit cycle.

    `pitch` (rad) turns the body about z (a torque-equilibrium attitude is 15 to 30 deg,
    Fehse 5.3.1); the limit cycle adds `amplitude` sin(2 pi t / `period`) to it, standing in for
    the deadband oscillation of an attitude controller.

    Parameters
    ----------
    pitch : float
        Constant rotation of the body about the Hill z axis (rad).
    amplitude : float
        Limit-cycle amplitude (rad), not negative; 0 for none.
    period : float
        Limit-cycle period (s), positive when `amplitude` is.

    Raises
    ------
    ValueError
        If `amplitude` or `period` is negative, or a limit cycle has no positive period.
    """

    pitch: float
    amplitude: float = 0.0
    period: float = 0.0
    name: ClassVar[str] = "fixed_offset"

    def __post_init__(self):
        if self.amplitude < 0.0 or self.period < 0.0:
            raise ValueError("amplitude and period must not be negative")
        if self.amplitude > 0.0 and not self.period > 0.0:
            raise ValueError("a limit cycle needs a positive period")

    def angle(self, t: ArrayLike) -> np.ndarray:
        """Body angle about z at `t`, pitch plus the limit cycle.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.

        Returns
        -------
        numpy.ndarray
            Angle (rad), shape of `t`.
        """
        t = np.asarray(t, dtype=float)
        if self.amplitude == 0.0:
            return np.full(t.shape, self.pitch)
        return self.pitch + self.amplitude * np.sin(2.0 * np.pi * t / self.period)

    def matrix(self, t: ArrayLike, n: float) -> np.ndarray:
        """Rotation about z by `angle(t)`.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        numpy.ndarray
            Body-to-Hill rotations, shape (*t.shape, 3, 3).
        """
        return rotation_z(self.angle(t))

    def angular_velocity(self, t: ArrayLike, n: float) -> np.ndarray:
        """The limit cycle's rate about z, 2 pi amplitude / period cos(2 pi t / period).

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        numpy.ndarray
            Angular velocity (rad/s), Hill axes, shape (*t.shape, 3).
        """
        t = np.asarray(t, dtype=float)
        if self.amplitude == 0.0:
            return _about_z(np.zeros(t.shape))
        omega = 2.0 * np.pi / self.period
        return _about_z(self.amplitude * omega * np.cos(omega * t))

    def rate_bound(self, n: float) -> float:
        """The limit cycle's peak rate, 2 pi amplitude / period.

        Parameters
        ----------
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        float
            Bound on the attitude rate (rad/s).
        """
        return 0.0 if self.amplitude == 0.0 else 2.0 * np.pi * self.amplitude / self.period

    def angular_acceleration_bound(self, n: float) -> float:
        """The limit cycle's peak acceleration, amplitude (2 pi / period)^2.

        Parameters
        ----------
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        float
            Bound on the derivative of `angular_velocity` (rad/s^2).
        """
        return 0.0 if self.amplitude == 0.0 else self.amplitude * (2.0 * np.pi / self.period) ** 2

    def public(self, n: float) -> dict[str, Any]:
        """The mode with its pitch and limit cycle.

        Parameters
        ----------
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        dict
            "mode", "pitch" (rad), "limit_cycle_amplitude" (rad), "limit_cycle_period" (s).
        """
        return {
            "mode": self.name,
            "pitch": self.pitch,
            "limit_cycle_amplitude": self.amplitude,
            "limit_cycle_period": self.period,
        }

    def burns(self) -> tuple[tuple[float, float, tuple[float, float, float]], ...]:
        """No burns.

        Returns
        -------
        tuple
            Empty.
        """
        return ()


@dataclass(frozen=True)
class InertialPointing:
    """Body axes fixed in inertial space, so they turn at -n about z in the Hill frame.

    `phase` (rad) is the body's angle about z at t = 0. On the docking axis of such a target the
    chaser needs a lateral speed r n to stay on it (Fehse 5.3.1: 0.113 m/s at 100 m).

    Parameters
    ----------
    phase : float
        Body angle about the Hill z axis at t = 0 (rad).
    """

    phase: float = 0.0
    name: ClassVar[str] = "inertial"

    def matrix(self, t: ArrayLike, n: float) -> np.ndarray:
        """Rotation about z by phase - n t.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s).

        Returns
        -------
        numpy.ndarray
            Body-to-Hill rotations, shape (*t.shape, 3, 3).
        """
        return rotation_z(self.phase - n * np.asarray(t, dtype=float))

    def angular_velocity(self, t: ArrayLike, n: float) -> np.ndarray:
        """-n about z: the Hill frame turns at n about the inertially fixed body.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s).

        Returns
        -------
        numpy.ndarray
            Angular velocity (rad/s), Hill axes, shape (*t.shape, 3).
        """
        return _about_z(np.full(np.shape(np.asarray(t, dtype=float)), -n))

    def rate_bound(self, n: float) -> float:
        """The orbital rate |n|.

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        float
            |n| (rad/s).
        """
        return abs(n)

    def angular_acceleration_bound(self, n: float) -> float:
        """Zero: the rate -n is constant.

        Parameters
        ----------
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        float
            Bound on the derivative of `angular_velocity` (rad/s^2).
        """
        return 0.0

    def public(self, n: float) -> dict[str, Any]:
        """The mode with its phase and rate.

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        dict
            "mode", "phase" (rad) and "rate" (rad/s, -n).
        """
        return {"mode": self.name, "phase": self.phase, "rate": -n}

    def burns(self) -> tuple[tuple[float, float, tuple[float, float, float]], ...]:
        """No burns.

        Returns
        -------
        tuple
            Empty.
        """
        return ()


@dataclass(frozen=True)
class Tumbling:
    """Rotation at a constant `rate` (rad/s) about a body `axis` fixed in the Hill frame.

    An interface for the Phase 9 flagship scenarios; a free tumble of a rigid body about a
    non-principal axis nutates and needs the 6-DOF target of the fidelity ladder.

    Parameters
    ----------
    axis : tuple of float
        Rotation axis, a nonzero 3-vector in body axes (equal to Hill axes at zero attitude).
    rate : float
        Rotation rate (rad/s).
    phase : float
        Rotation angle at t = 0 (rad).

    Raises
    ------
    ValueError
        If `axis` is not a nonzero 3-vector.
    """

    axis: tuple[float, float, float]
    rate: float
    phase: float = 0.0
    name: ClassVar[str] = "tumbling"

    def __post_init__(self):
        axis = tuple(float(c) for c in self.axis)
        if len(axis) != 3 or not np.linalg.norm(axis) > 0.0:
            raise ValueError("axis must be a nonzero 3-vector")
        object.__setattr__(self, "axis", axis)

    def matrix(self, t: ArrayLike, n: float) -> np.ndarray:
        """Rotation about `axis` by phase + rate t.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        numpy.ndarray
            Body-to-Hill rotations, shape (*t.shape, 3, 3).
        """
        return rotation_about(self.axis, self.phase + self.rate * np.asarray(t, dtype=float))

    def angular_velocity(self, t: ArrayLike, n: float) -> np.ndarray:
        """`rate` about the unit `axis`, which the rotation leaves fixed in the Hill frame.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        numpy.ndarray
            Angular velocity (rad/s), Hill axes, shape (*t.shape, 3).
        """
        axis = np.asarray(self.axis) / np.linalg.norm(self.axis)
        shape = np.shape(np.asarray(t, dtype=float))
        return np.broadcast_to(self.rate * axis, (*shape, 3)).copy()

    def rate_bound(self, n: float) -> float:
        """The tumble rate |rate|.

        Parameters
        ----------
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        float
            |rate| (rad/s).
        """
        return abs(self.rate)

    def angular_acceleration_bound(self, n: float) -> float:
        """Zero: the rate is constant.

        Parameters
        ----------
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        float
            Bound on the derivative of `angular_velocity` (rad/s^2).
        """
        return 0.0

    def public(self, n: float) -> dict[str, Any]:
        """The mode with its axis, rate and phase.

        Parameters
        ----------
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        dict
            "mode", "axis" (body axes), "rate" (rad/s) and "phase" (rad).
        """
        return {"mode": self.name, "axis": self.axis, "rate": self.rate, "phase": self.phase}

    def burns(self) -> tuple[tuple[float, float, tuple[float, float, float]], ...]:
        """No burns.

        Returns
        -------
        tuple
            Empty.
        """
        return ()


@dataclass(frozen=True)
class TorqueFree:
    """A rigid body tumbling free of torques, its attitude stored inertially.

    The tumble is the closed form of `vbar.dynamics.rigid_body.TorqueFreeMotion`, set by tumble
    parameters (`TorqueFreeMotion.from_tumble`): principal `moments` about the body axes, the
    effective inertia `effective_inertia` D = H^2 / (2 T), the rate `rate` w_l = H / D, the polhode
    `phase`, the inertial direction of H and the body's `spin_angle` about it. Only H is fixed,
    and only in inertial space (Schaub and Junkins, Analytical Mechanics of Space Systems, section
    4.2.1); the inertial frame coincides with the Hill frame at t = 0 and the Hill frame turns at
    n about z, so the body-to-Hill rotation is R_z(-n t) R_nb(t) and the rate relative to the
    Hill frame is R_z(-n t) R_nb w - n z.

    Parameters
    ----------
    moments : tuple of float
        Principal moments about the body x, y and z axes (kg m^2), positive.
    effective_inertia : float
        D (kg m^2), between the least and greatest moment and not the intermediate one.
    rate : float
        w_l = H / D (rad/s), positive.
    phase : float
        Fraction of the body-rate period at t = 0, in [0, 1).
    momentum_direction : tuple of float
        Direction of H at t = 0 in Hill axes (fixed inertially), nonzero; normalised here.
    spin_angle : float
        Rotation of the body about H at t = 0 (rad).

    Raises
    ------
    ValueError
        If the tumble parameters are refused by `TorqueFreeMotion.from_tumble` or the momentum
        direction is zero.
    """

    moments: tuple[float, float, float]
    effective_inertia: float
    rate: float
    phase: float = 0.0
    momentum_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    spin_angle: float = 0.0
    name: ClassVar[str] = "torque_free"
    motion: TorqueFreeMotion = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "moments", tuple(float(c) for c in self.moments))
        direction = np.asarray(self.momentum_direction, dtype=float).reshape(3)
        if not np.linalg.norm(direction) > 0.0:
            raise ValueError("momentum_direction must be nonzero")
        direction = tuple(float(c) for c in direction / np.linalg.norm(direction))
        object.__setattr__(self, "momentum_direction", direction)
        motion = TorqueFreeMotion.from_tumble(
            self.moments,
            float(self.effective_inertia),
            float(self.rate),
            float(self.phase),
            direction,
            float(self.spin_angle),
        )
        object.__setattr__(self, "motion", motion)

    def matrix(self, t: ArrayLike, n: float) -> np.ndarray:
        """R_z(-n t) R_nb(t), the inertial attitude seen from the turning Hill frame.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s).

        Returns
        -------
        numpy.ndarray
            Body-to-Hill rotations, shape (*t.shape, 3, 3).
        """
        t = np.asarray(t, dtype=float)
        return rotation_z(-n * t) @ self.motion.matrix(t)

    def angular_velocity(self, t: ArrayLike, n: float) -> np.ndarray:
        """R_z(-n t) R_nb w - n z, the body rate relative to the Hill frame.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s).

        Returns
        -------
        numpy.ndarray
            Angular velocity (rad/s), Hill axes, shape (*t.shape, 3).
        """
        t = np.asarray(t, dtype=float)
        inertial = np.einsum(
            "...ij,...j->...i", rotation_z(-n * t), self.motion.angular_velocity(t)
        )
        return inertial - _about_z(np.full(t.shape, n))

    def rate_bound(self, n: float) -> float:
        """The body rate's bound plus |n|.

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        float
            Bound on the rate relative to the Hill frame (rad/s).
        """
        return self.motion.rate_bound() + abs(n)

    def angular_acceleration_bound(self, n: float) -> float:
        """|n| times the body rate's bound plus the bound on its derivative.

        The derivative of R_z(-n t) w_n is -n z x (R_z(-n t) w_n) + R_z(-n t) dw_n/dt, and the
        inertial rate changes as fast as the body rate, dw_n/dt = R_nb dw/dt.

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        float
            Bound on |d w / dt| in the Hill frame (rad/s^2).
        """
        return abs(n) * self.motion.rate_bound() + self.motion.acceleration_bound()

    def public(self, n: float) -> dict[str, Any]:
        """The tumble parameters and the state they give at t = 0.

        Parameters
        ----------
        n : float
            Mean motion (rad/s), unused.

        Returns
        -------
        dict
            "mode", the tumble parameters (kg m^2, rad/s, rad, Hill axes at t = 0), "omega" the
            body rate at t = 0 (rad/s, body axes) and "attitude" q_h_b at t = 0 (scalar last).
        """
        return {
            "mode": self.name,
            "moments": self.moments,
            "effective_inertia": float(self.effective_inertia),
            "rate": float(self.rate),
            "phase": float(self.phase),
            "momentum_direction": self.momentum_direction,
            "spin_angle": float(self.spin_angle),
            "omega": tuple(float(c) for c in self.motion.omega0),
            "attitude": tuple(float(c) for c in Rotation.from_matrix(self.motion.r0).as_quat()),
        }

    def burns(self) -> tuple[tuple[float, float, tuple[float, float, float]], ...]:
        """No burns.

        Returns
        -------
        tuple
            Empty.
        """
        return ()


@dataclass(frozen=True)
class Maneuvering:
    """A target that burns on a schedule while keeping the attitude of `attitude`.

    `schedule` lists (start s, duration s, Hill acceleration m/s^2) of the target's burns; the
    chaser's relative state feels the opposite acceleration (a reboost after undocking, Goodman,
    History of Space Shuttle Rendezvous, 2011, STS-130). Burns may not overlap.

    Parameters
    ----------
    schedule : tuple
        Burns as (start s, duration s, acceleration m/s^2 in Hill axes); sorted here.
    attitude : TargetMode
        The attitude mode the target keeps; `Passive` by default.

    Raises
    ------
    ValueError
        If a burn starts before 0, has no positive duration or a non-3-vector acceleration,
        or two burns overlap.
    """

    schedule: tuple[tuple[float, float, tuple[float, float, float]], ...]
    attitude: TargetMode = Passive()
    name: ClassVar[str] = "maneuvering"

    def __post_init__(self):
        burns = []
        for start, duration, accel in self.schedule:
            a = tuple(float(c) for c in accel)
            if len(a) != 3 or not duration > 0.0 or start < 0.0:
                raise ValueError("a burn is (start >= 0, duration > 0, 3-vector acceleration)")
            burns.append((float(start), float(duration), (a[0], a[1], a[2])))
        burns.sort()
        for (s0, d0, _), (s1, _, _) in zip(burns, burns[1:], strict=False):
            if s1 < s0 + d0:
                raise ValueError("target burns may not overlap")
        object.__setattr__(self, "schedule", tuple(burns))

    def matrix(self, t: ArrayLike, n: float) -> np.ndarray:
        """The rotation of `attitude`.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s).

        Returns
        -------
        numpy.ndarray
            Body-to-Hill rotations, shape (*t.shape, 3, 3).
        """
        return self.attitude.matrix(t, n)

    def angular_velocity(self, t: ArrayLike, n: float) -> np.ndarray:
        """The angular velocity of `attitude`.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s).

        Returns
        -------
        numpy.ndarray
            Angular velocity (rad/s), Hill axes, shape (*t.shape, 3).
        """
        return self.attitude.angular_velocity(t, n)

    def rate_bound(self, n: float) -> float:
        """The rate bound of `attitude`.

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        float
            Bound on the attitude rate (rad/s).
        """
        return self.attitude.rate_bound(n)

    def angular_acceleration_bound(self, n: float) -> float:
        """The bound of `attitude`.

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        float
            Bound on the derivative of `angular_velocity` (rad/s^2).
        """
        return self.attitude.angular_acceleration_bound(n)

    def public(self, n: float) -> dict[str, Any]:
        """The attitude's description with this mode's name and the burns.

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        dict
            The attitude's entries, "mode" and "burns" (s, s, m/s^2 in Hill axes).
        """
        return {**self.attitude.public(n), "mode": self.name, "burns": self.schedule}

    def burns(self) -> tuple[tuple[float, float, tuple[float, float, float]], ...]:
        """The schedule.

        Returns
        -------
        tuple
            (start s, duration s, Hill acceleration m/s^2) in start order.
        """
        return self.schedule


@dataclass(frozen=True)
class Misbriefed:
    """The truth follows `actual` while agents are told `briefed` (Goodman: SPARTAN, STS-51G).

    Constraints that follow the target keep it out of their public spec, and a scenario's goal
    describes the briefed geometry (`briefed_mode`).

    Parameters
    ----------
    actual : TargetMode
        The mode the truth follows.
    briefed : TargetMode
        The mode agents are told.
    """

    actual: TargetMode
    briefed: TargetMode
    name: ClassVar[str] = "misbriefed"

    def matrix(self, t: ArrayLike, n: float) -> np.ndarray:
        """The rotation of `actual`.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s).

        Returns
        -------
        numpy.ndarray
            Body-to-Hill rotations, shape (*t.shape, 3, 3).
        """
        return self.actual.matrix(t, n)

    def angular_velocity(self, t: ArrayLike, n: float) -> np.ndarray:
        """The angular velocity of `actual`.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.
        n : float
            Mean motion (rad/s).

        Returns
        -------
        numpy.ndarray
            Angular velocity (rad/s), Hill axes, shape (*t.shape, 3).
        """
        return self.actual.angular_velocity(t, n)

    def rate_bound(self, n: float) -> float:
        """The rate bound of `actual`.

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        float
            Bound on the attitude rate (rad/s).
        """
        return self.actual.rate_bound(n)

    def angular_acceleration_bound(self, n: float) -> float:
        """The bound of `actual`.

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        float
            Bound on the derivative of `angular_velocity` (rad/s^2).
        """
        return self.actual.angular_acceleration_bound(n)

    def public(self, n: float) -> dict[str, Any]:
        """The description of `briefed`.

        Parameters
        ----------
        n : float
            Mean motion (rad/s).

        Returns
        -------
        dict
            What agents are told.
        """
        return self.briefed.public(n)

    def burns(self) -> tuple[tuple[float, float, tuple[float, float, float]], ...]:
        """The burns of `actual`.

        Returns
        -------
        tuple
            (start s, duration s, Hill acceleration m/s^2) in start order.
        """
        return self.actual.burns()


def briefed_mode(mode: TargetMode) -> TargetMode:
    """The mode agents are told: the briefing of a misbriefed target, else the mode itself.

    Parameters
    ----------
    mode : TargetMode
        The scenario's target mode.

    Returns
    -------
    TargetMode
        `mode.briefed` for a `Misbriefed` mode, else `mode`.
    """
    return mode.briefed if isinstance(mode, Misbriefed) else mode


def body_point(mode: TargetMode, point: ArrayLike, t: ArrayLike, n: float) -> np.ndarray:
    """A body-fixed point or direction expressed in Hill axes at time `t`, shape (..., 3).

    Parameters
    ----------
    mode : TargetMode
        The target's attitude mode.
    point : array_like
        Point (m) or direction in body axes, shape (3,).
    t : array_like
        Time (s), any shape.
    n : float
        Mean motion (rad/s).

    Returns
    -------
    numpy.ndarray
        The point or direction in Hill axes, shape (*t.shape, 3).
    """
    return np.einsum("...ij,j->...i", mode.matrix(t, n), np.asarray(point, dtype=float))


def body_point_velocity(mode: TargetMode, point: ArrayLike, t: ArrayLike, n: float) -> np.ndarray:
    """Hill-frame velocity of a body-fixed point, w x (R p), with the target's centre at rest.

    A capture interface on a turning target moves at this velocity (Fehse, Automated Rendezvous
    and Docking of Spacecraft, 2003, section 8.3.6), so a chaser's speeds at the interface are
    measured against it.

    Parameters
    ----------
    mode : TargetMode
        The target's attitude mode.
    point : array_like
        Point in body axes (m), shape (3,).
    t : array_like
        Time (s), any shape.
    n : float
        Mean motion (rad/s).

    Returns
    -------
    numpy.ndarray
        Velocity (m/s), Hill axes, shape (*t.shape, 3).
    """
    return np.cross(mode.angular_velocity(t, n), body_point(mode, point, t, n))


@dataclass(frozen=True)
class RotatingCone:
    """An approach cone fixed to a rotating target; margin in m.

    The apex `apex` and axis `axis` are body-fixed (Hill axes at zero attitude) and follow
    `target`; the margin is the one of `vbar.constraints.ApproachCone` evaluated against the
    cone where it stands at time `t`: max(d - R_c, s), with d the distance to the apex and s the
    signed distance to the cone. The cone's points move at most at the attitude rate times their
    distance from the target's centre, and s is the active term only within
    R_c / (1 - sin(half_angle)) of the apex, so the rate bound is
    speed + rate (|apex| + R_c / (1 - sin(half_angle))); the active range must be finite.

    Parameters
    ----------
    half_angle : float
        Cone half-angle (rad), in (0, pi/2).
    axis : tuple of float
        Cone axis, a nonzero 3-vector in body axes.
    apex : tuple of float
        Apex position in body axes (m).
    active_range : float
        R_c, the distance from the apex within which the cone applies (m), finite and positive.
    target : TargetMode
        The target's attitude mode; kept out of the public spec.
    n : float
        Mean motion (rad/s).
    name : str
        Constraint name.

    Raises
    ------
    ValueError
        If `half_angle` is outside (0, pi/2), `active_range` is not finite and positive, or
        `axis` is not a nonzero 3-vector.
    """

    half_angle: float
    axis: tuple[float, float, float]
    apex: tuple[float, float, float]
    active_range: float
    target: TargetMode
    n: float
    name: str = "approach_cone"
    resolution: ClassVar[float] = 1e-3
    # The true mode stays out of the public spec; the goal's target entry carries the briefing.
    private_fields: ClassVar[tuple[str, ...]] = ("target",)

    def __post_init__(self):
        if not 0.0 < self.half_angle < 0.5 * np.pi:
            raise ValueError("half_angle must be between 0 and pi/2")
        if not 0.0 < self.active_range < np.inf:
            raise ValueError("a rotating cone needs a finite positive active_range")
        axis = tuple(float(c) for c in self.axis)
        if len(axis) != 3 or not np.linalg.norm(axis) > 0.0:
            raise ValueError("axis must be a nonzero 3-vector")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "apex", tuple(float(c) for c in self.apex))

    def apex_at(self, t: ArrayLike) -> np.ndarray:
        """Apex position in Hill axes at `t`.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.

        Returns
        -------
        numpy.ndarray
            Position (m), shape (*t.shape, 3).
        """
        return body_point(self.target, self.apex, t, self.n)

    def axis_at(self, t: ArrayLike) -> np.ndarray:
        """Unit cone axis in Hill axes at `t`.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.

        Returns
        -------
        numpy.ndarray
            Unit vector, shape (*t.shape, 3).
        """
        a = body_point(self.target, self.axis, t, self.n)
        return a / np.linalg.norm(a, axis=-1, keepdims=True)

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        """Margin to the cone where it stands at `t`.

        Parameters
        ----------
        state : numpy.ndarray
            Relative state (m, m/s), Hill axes, shape (..., 6).
        t : float or numpy.ndarray
            Time (s), broadcast against the state's leading shape.

        Returns
        -------
        numpy.ndarray
            Margin (m), positive inside, shape (...,).
        """
        state = np.asarray(state, dtype=float)
        times = np.broadcast_to(np.asarray(t, dtype=float), state.shape[:-1])
        d = state[..., :3] - self.apex_at(times)
        rho = np.linalg.norm(d, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_phi = np.where(rho > 0.0, np.sum(d * self.axis_at(times), -1) / rho, 1.0)
        phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))
        inside = rho * np.sin(self.half_angle - phi)
        distance = np.where(phi < self.half_angle + 0.5 * np.pi, inside, -rho)
        return np.maximum(rho - self.active_range, distance)

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        """Bound on |d margin / dt|: speed + rate (|apex| + R_c / (1 - sin(half_angle))).

        Parameters
        ----------
        speed : float
            Bound on the relative speed (m/s).
        accel : float
            Bound on the relative acceleration (m/s^2), unused.
        rho_min : float
            Lower bound on the range (m), unused.

        Returns
        -------
        float
            Bound on the margin's rate (m/s).
        """
        reach = np.linalg.norm(self.apex) + self.active_range / (1.0 - np.sin(self.half_angle))
        return speed + self.target.rate_bound(self.n) * reach


@dataclass(frozen=True)
class BodyKeepOut:
    """The target's body sphere, open at the port; margin max(rho - R, min(depth, 0) + d_r), m.

    `depth` is the signed distance in front of the port plane, (r - p) . a with the axis a from
    the port toward the approach side, and `reception` d_r the depth of the mechanism's capture
    volume behind the plane, so the reception volume is not part of the keep-out while the rest
    of the sphere is, and a chaser that has just crossed the plane at the port reads a positive
    margin. The depth is clipped at zero so the second term never exceeds d_r and is the larger
    one only within R + d_r of the centre; there the turning plane moves a point at most
    rate (R + d_r + |port|), which the rate bound adds to the speed. Port and axis are
    body-fixed and follow `target`. Both terms are 1-Lipschitz in position.

    Parameters
    ----------
    radius : float
        R, the body sphere's radius (m), positive.
    port : tuple of float
        Port position in body axes (m).
    axis : tuple of float
        Port axis from the port toward the approach side, nonzero, body axes; normalised here.
    target : TargetMode
        The target's attitude mode; kept out of the public spec.
    n : float
        Mean motion (rad/s).
    reception : float
        d_r, the capture volume's depth behind the port plane (m), not negative.
    name : str
        Constraint name.

    Raises
    ------
    ValueError
        If `radius` is not positive, `reception` is negative or `axis` is zero.
    """

    radius: float
    port: tuple[float, float, float]
    axis: tuple[float, float, float]
    target: TargetMode
    n: float
    reception: float = 0.1
    name: str = "keep_out"
    resolution: ClassVar[float] = 1e-3
    private_fields: ClassVar[tuple[str, ...]] = ("target",)

    def __post_init__(self):
        if not self.radius > 0.0 or self.reception < 0.0:
            raise ValueError("radius must be positive and reception not negative")
        axis = np.asarray(self.axis, dtype=float)
        if not np.linalg.norm(axis) > 0.0:
            raise ValueError("axis must be nonzero")
        object.__setattr__(self, "axis", tuple(float(c) for c in axis / np.linalg.norm(axis)))
        object.__setattr__(self, "port", tuple(float(c) for c in self.port))

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        """Margin to the body sphere with the port open, at `t`.

        Parameters
        ----------
        state : numpy.ndarray
            Relative state (m, m/s), Hill axes, shape (..., 6).
        t : float or numpy.ndarray
            Time (s), broadcast against the state's leading shape.

        Returns
        -------
        numpy.ndarray
            Margin (m), positive outside the keep-out, shape (...,).
        """
        state = np.asarray(state, dtype=float)
        times = np.broadcast_to(np.asarray(t, dtype=float), state.shape[:-1])
        rho = np.linalg.norm(state[..., :3], axis=-1)
        offset = state[..., :3] - body_point(self.target, self.port, times, self.n)
        depth = np.sum(offset * body_point(self.target, self.axis, times, self.n), axis=-1)
        return np.maximum(rho - self.radius, np.minimum(depth, 0.0) + self.reception)

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        """Bound on |d margin / dt|: speed + rate (R + d_r + |port|).

        Parameters
        ----------
        speed : float
            Bound on the relative speed (m/s).
        accel : float
            Bound on the relative acceleration (m/s^2), unused.
        rho_min : float
            Lower bound on the range (m), unused.

        Returns
        -------
        float
            Bound on the margin's rate (m/s).
        """
        reach = self.radius + self.reception + float(np.linalg.norm(self.port))
        return speed + self.target.rate_bound(self.n) * reach


def sun_direction(beta: float, phase: float) -> np.ndarray:
    """Unit Sun direction in Hill axes at t = 0 from its elevation `beta` above the orbit plane
    (rad, positive toward +z) and its in-plane angle `phase` from +x toward +y (rad).

    Parameters
    ----------
    beta : float
        Sun elevation above the orbit plane (rad), positive toward +z.
    phase : float
        In-plane angle from +x toward +y (rad).

    Returns
    -------
    numpy.ndarray
        Unit vector in Hill axes at t = 0, shape (3,).
    """
    return np.array(
        [math.cos(beta) * math.cos(phase), math.cos(beta) * math.sin(phase), math.sin(beta)]
    )


def sun_at(sun0: ArrayLike, n: float, t: ArrayLike) -> np.ndarray:
    """Unit Sun direction in Hill axes at `t` s: the Hill frame turns at `n`, the Sun is fixed.

    Parameters
    ----------
    sun0 : array_like
        Sun direction in Hill axes at t = 0, shape (3,), nonzero; normalised here.
    n : float
        Mean motion (rad/s).
    t : array_like
        Time (s), any shape.

    Returns
    -------
    numpy.ndarray
        Unit vectors in Hill axes, shape (*t.shape, 3).
    """
    s0 = np.asarray(sun0, dtype=float)
    s0 = s0 / np.linalg.norm(s0)
    return np.einsum("...ij,j->...i", rotation_z(-n * np.asarray(t, dtype=float)), s0)


def sunlit(sun0: ArrayLike, n: float, t: ArrayLike, orbit_radius: float, body: Body = EARTH):
    """1 where the target is in sunlight at `t` s, 0 inside the body's cylindrical shadow.

    The target sits at (orbit_radius, 0, 0) in its own Hill axes; the chaser's offset of metres
    does not change the verdict.

    Parameters
    ----------
    sun0 : array_like
        Sun direction in Hill axes at t = 0, shape (3,).
    n : float
        Mean motion (rad/s).
    t : array_like
        Time (s), any shape.
    orbit_radius : float
        Target's distance from the body's centre (m).
    body : Body
        Central body whose radius sets the shadow cylinder; Earth by default.

    Returns
    -------
    numpy.ndarray
        1.0 in sunlight and 0.0 in shadow, shape of `t`.
    """
    sun = sun_at(sun0, n, t)
    # The target's position vector from the body's centre is (orbit_radius, 0, 0) in its own
    # Hill axes; it is in the shadow cylinder when it is on the far side and within one body
    # radius of the Sun-body line (Montenbruck and Gill, Satellite Orbits, sec. 3.4.1).
    along = orbit_radius * sun[..., 0]
    behind = (along < 0.0) & (orbit_radius**2 - along**2 < body.radius**2)
    return np.where(behind, 0.0, 1.0)


@dataclass(frozen=True)
class LightingWindow:
    """Near the port, the in-plane Sun angle must lie in a window and the target must be lit.

    The in-plane Sun angle is measured between the Sun's projection on the orbit plane and the
    approach axis `axis` (from the port toward the chaser); for a -V-bar port Fehse's monitoring
    window of 105 to 165 deg from +V-bar (5.4.1) is 15 to 75 deg from the axis, with at most
    about 15 min inside it per orbit at beta = 0. The margin is the robustness of "within
    `active_range` of the port implies lit inside the window": max((d - R_c) / R_c, w(t)) with
    w the least of the angular margins to the window's edges (rad) and the lit margin, +1 on the
    day side and (p - R_E) / R_E inside the shadow cylinder, p the target's distance from the
    cylinder's axis. The first term changes at most at speed / R_c, the others at about n.

    Parameters
    ----------
    window : tuple of float
        (low, high) in-plane Sun angle from the approach axis (rad), 0 <= low < high <= pi.
    axis : tuple of float
        Approach axis from the port toward the chaser, Hill axes; its in-plane part is used.
    port : tuple of float
        Port position, Hill axes (m).
    active_range : float
        R_c, the distance from the port within which the window applies (m), finite, positive.
    sun0 : tuple of float
        Sun direction in Hill axes at t = 0.
    n : float
        Mean motion (rad/s).
    orbit_radius : float
        Target's distance from the body's centre (m).
    body : Body
        Central body; Earth by default.
    name : str
        Constraint name.

    Raises
    ------
    ValueError
        If `window` is not 0 <= low < high <= pi, `active_range` is not finite and positive, or
        `axis` has no component in the orbit plane.
    """

    window: tuple[float, float]
    axis: tuple[float, float, float]
    port: tuple[float, float, float]
    active_range: float
    sun0: tuple[float, float, float]
    n: float
    orbit_radius: float
    body: Body = EARTH
    name: str = "lighting_window"
    resolution: ClassVar[float] = 1e-5

    def __post_init__(self):
        lo, hi = float(self.window[0]), float(self.window[1])
        if not 0.0 <= lo < hi <= np.pi:
            raise ValueError("window must be 0 <= low < high <= pi (rad)")
        if not 0.0 < self.active_range < np.inf:
            raise ValueError("active_range must be finite and positive")
        object.__setattr__(self, "window", (lo, hi))
        object.__setattr__(self, "axis", tuple(float(c) for c in self.axis))
        # The window is an angle in the orbit plane, undefined for an axis along the normal.
        if not len(self.axis) == 3 or not float(np.hypot(self.axis[0], self.axis[1])) > 0.0:
            raise ValueError("the approach axis needs a component in the orbit plane")
        object.__setattr__(self, "port", tuple(float(c) for c in self.port))
        object.__setattr__(self, "sun0", tuple(float(c) for c in self.sun0))

    def sun_angle(self, t: ArrayLike) -> np.ndarray:
        """In-plane angle (rad) between the Sun and the approach axis at `t`.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.

        Returns
        -------
        numpy.ndarray
            Angle in [0, pi] (rad), shape of `t`.
        """
        sun = sun_at(self.sun0, self.n, t)
        axis = np.array(self.axis[:2]) / np.linalg.norm(self.axis[:2])
        planar = np.linalg.norm(sun[..., :2], axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(planar > 0.0, sun[..., :2] @ axis / planar, 1.0)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    def lit_margin(self, t: ArrayLike) -> np.ndarray:
        """(p - R_E) / R_E with p the target's distance from the shadow cylinder's axis on the
        night side and that distance plus r s_x on the day side, continuous across the terminator
        and changing at most 2 n r / R_E per second.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.

        Returns
        -------
        numpy.ndarray
            Dimensionless margin, negative in the shadow, shape of `t`.
        """
        sun = sun_at(self.sun0, self.n, t)
        across = self.orbit_radius * np.sqrt(np.maximum(1.0 - sun[..., 0] ** 2, 0.0))
        across = across + self.orbit_radius * np.maximum(sun[..., 0], 0.0)
        return (across - self.body.radius) / self.body.radius

    def window_margin(self, t: ArrayLike) -> np.ndarray:
        """w(t), the least of the margins to the window's edges and the lit margin.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.

        Returns
        -------
        numpy.ndarray
            Margin (rad for the edges, dimensionless for the lit term), shape of `t`.
        """
        angle = self.sun_angle(t)
        edges = np.minimum(angle - self.window[0], self.window[1] - angle)
        return np.minimum(edges, self.lit_margin(t))

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        """max((d - R_c) / R_c, w(t)), d the distance from the port.

        Parameters
        ----------
        state : numpy.ndarray
            Relative state (m, m/s), Hill axes, shape (..., 6).
        t : float or numpy.ndarray
            Time (s), broadcast against the state's leading shape.

        Returns
        -------
        numpy.ndarray
            Dimensionless margin, positive when the rule holds, shape (...,).
        """
        state = np.asarray(state, dtype=float)
        times = np.broadcast_to(np.asarray(t, dtype=float), state.shape[:-1])
        d = np.linalg.norm(state[..., :3] - np.array(self.port), axis=-1)
        return np.maximum((d - self.active_range) / self.active_range, self.window_margin(times))

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        """Bound on |d margin / dt|: speed / R_c + |n| max(1, 2 r / R_E).

        Parameters
        ----------
        speed : float
            Bound on the relative speed (m/s).
        accel : float
            Bound on the relative acceleration (m/s^2), unused.
        rho_min : float
            Lower bound on the range (m), unused.

        Returns
        -------
        float
            Bound on the margin's rate (1/s).
        """
        turn = abs(self.n) * max(1.0, 2.0 * self.orbit_radius / self.body.radius)
        return speed / self.active_range + turn


class ManeuveringDynamics:
    """A dynamics model whose target burns on a schedule.

    The chaser's relative state feels the opposite of each target burn's acceleration, exact in
    the Clohessy-Wiltshire model and first order in the target's orbit change otherwise. Every
    `advance` must lie within one piece of the schedule; the environment splits its segments at
    the burns' start and end times (`burn_boundaries`).

    Parameters
    ----------
    inner : Dynamics
        The relative dynamics model that propagates the chaser.
    target : TargetMode
        The target mode whose `burns` give the schedule.
    """

    def __init__(self, inner: Dynamics, target: TargetMode):
        self.inner = inner
        self.target = target
        self.name = inner.name
        self.n = inner.n
        self.t = 0.0

    def reset(self, rel_state: ArrayLike) -> None:
        """Restart at `rel_state` and t = 0.

        Parameters
        ----------
        rel_state : array_like
            Relative state (m, m/s), Hill axes, shape (6,).
        """
        self.inner.reset(rel_state)
        self.t = 0.0

    def target_acceleration(self, t: float) -> np.ndarray:
        """The target's scheduled acceleration at `t`.

        Parameters
        ----------
        t : float
            Time (s).

        Returns
        -------
        numpy.ndarray
            Acceleration (m/s^2), Hill axes, shape (3,); zero outside the burns.
        """
        for start, duration, accel in self.target.burns():
            if start - 1e-9 <= t < start + duration - 1e-9:
                return np.array(accel)
        return np.zeros(3)

    def advance(
        self, accel_hill: ArrayLike, duration: float, n_samples: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Hold `accel_hill` for `duration` s, less the target's acceleration.

        Parameters
        ----------
        accel_hill : array_like
            Chaser's commanded acceleration (m/s^2), Hill axes, shape (3,).
        duration : float
            Segment length (s).
        n_samples : int
            Number of evenly spaced states to return, endpoints included; 0 for none.

        Returns
        -------
        tuple
            The final relative state (m, m/s), shape (6,), and the samples, shape
            (n_samples, 6), or None.

        Raises
        ------
        ValueError
            If a target burn starts or ends inside the segment.
        """
        end = self.t + duration
        for start, length, _ in self.target.burns():
            for edge in (start, start + length):
                if self.t + 1e-9 < edge < end - 1e-9:
                    raise ValueError("a target burn starts or ends inside a segment")
        relative = np.asarray(accel_hill, dtype=float) - self.target_acceleration(self.t)
        self.t = end
        return self.inner.advance(relative, duration, n_samples)

    def state_in_last_segment(self, tau: float) -> np.ndarray:
        """State `tau` seconds into the most recent segment.

        Parameters
        ----------
        tau : float
            Time into the segment (s).

        Returns
        -------
        numpy.ndarray
            Relative state (m, m/s), Hill axes, shape (6,).
        """
        return self.inner.state_in_last_segment(tau)


def burn_boundaries(mode: TargetMode) -> tuple[float, ...]:
    """Times (s) at which the target's acceleration changes.

    Parameters
    ----------
    mode : TargetMode
        The target mode.

    Returns
    -------
    tuple of float
        Sorted start and end times of the burns (s), without repeats.
    """
    edges = []
    for start, duration, _ in mode.burns():
        edges += [start, start + duration]
    return tuple(sorted(set(edges)))


def split_at(
    segments: Sequence[tuple[float, np.ndarray]], t0: float, edges: Sequence[float]
) -> tuple[tuple[float, np.ndarray], ...]:
    """Constant-acceleration segments starting at `t0`, cut at every edge time strictly inside.

    Parameters
    ----------
    segments : sequence of tuple
        (duration s, acceleration m/s^2 in Hill axes) in time order.
    t0 : float
        Start time of the first segment (s).
    edges : sequence of float
        Times (s) at which to cut.

    Returns
    -------
    tuple
        (duration s, acceleration) pieces with the same total duration.
    """
    out: list[tuple[float, np.ndarray]] = []
    t = t0
    for duration, accel in segments:
        start = t
        for edge in sorted(edges):
            if start + 1e-9 < edge < t + duration - 1e-9:
                out.append((edge - start, accel))
                start = edge
        out.append((t + duration - start, accel))
        t += duration
    return tuple(out)


__all__ = [
    "BodyKeepOut",
    "FixedOffset",
    "InertialPointing",
    "LightingWindow",
    "Maneuvering",
    "ManeuveringDynamics",
    "Misbriefed",
    "Passive",
    "RotatingCone",
    "TargetMode",
    "TorqueFree",
    "Tumbling",
    "body_point",
    "body_point_velocity",
    "briefed_mode",
    "burn_boundaries",
    "rotation_about",
    "rotation_z",
    "split_at",
    "sun_at",
    "sun_direction",
    "sunlit",
]
