"""What every backend shares: thrust segments, and the frame conversion back to Hill states."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ...conventions import hill_frame, inertial_to_hill
from ..bodies import Body
from ..models import gravity


@dataclass(frozen=True)
class Burn:
    """A constant acceleration on the chaser, held in the target's Hill axes.

    Parameters
    ----------
    start : float
        Start time (s).
    duration : float
        Duration (s), positive.
    accel_hill : tuple of float
        Acceleration (m/s^2), Hill axes of the target's orbit.

    Raises
    ------
    ValueError
        If the duration is not positive and finite or a value is not finite.
    """

    start: float
    duration: float
    accel_hill: tuple[float, float, float]

    def __post_init__(self):
        if not math.inf > self.duration > 0.0 or not math.isfinite(self.start):
            raise ValueError("a burn needs a finite start and a positive, finite duration")
        accel = tuple(float(c) for c in self.accel_hill)
        if len(accel) != 3 or not all(map(math.isfinite, accel)):
            raise ValueError("accel_hill is three finite numbers")
        object.__setattr__(self, "accel_hill", accel)


def accel_at(burns: Sequence[Burn], t: float) -> np.ndarray:
    """The Hill acceleration (m/s^2) in effect at `t`, zero between burns."""
    for burn in burns:
        if burn.start <= t < burn.start + burn.duration:
            return np.array(burn.accel_hill)
    return np.zeros(3)


def check_aligned(burns: Sequence[Burn], substep: float) -> None:
    """Raise ValueError unless every burn starts and ends on a multiple of `substep` (s).

    A backend holds a burn's direction over whole substeps, so an edge inside one would move the
    burn and the comparison would blame the wrong side.
    """
    for burn in burns:
        for edge in (burn.start, burn.start + burn.duration):
            if abs(edge / substep - round(edge / substep)) * substep > 1e-9:
                raise ValueError("burn edges must be whole numbers of substeps")


def inertial_accel(
    burns: Sequence[Burn], t: float, target: ArrayLike, body: Body, j2: bool, hold: float = 0.0
) -> np.ndarray:
    """The burn acceleration at `t` in inertial axes, to be held for `hold` seconds.

    The Hill axes turn about the orbit normal at h / r^2, so a direction held fixed in inertial
    axes for `hold` seconds is taken where the axes stand halfway through: the mean of the
    turning direction to second order in the angle turned.
    """
    target = np.asarray(target, dtype=float)
    a = accel_at(burns, t)
    if not a.any():
        return a
    C, omega = hill_frame(target[:3], target[3:], gravity(target[:3], body, j2))
    angle = 0.5 * hold * float(omega[2])
    c, s = math.cos(angle), math.sin(angle)
    turned = np.array([c * a[0] - s * a[1], s * a[0] + c * a[1], a[2]])
    return C @ turned


def relative_inertial(target: ArrayLike, chaser: ArrayLike) -> np.ndarray:
    """Chaser minus target, inertial [dr, dv] (m, m/s), for stacked states."""
    return np.asarray(chaser, dtype=float) - np.asarray(target, dtype=float)


def hill_states(target: ArrayLike, chaser: ArrayLike, body: Body, j2: bool) -> np.ndarray:
    """Hill relative states of the chaser about the target, from inertial states (m, m/s).

    The frame's rate uses the target's gravity, as V-BAR's own conversion does, so the same
    inertial states give the same Hill states whichever tool flew them.

    Parameters
    ----------
    target, chaser : array_like
        Inertial [r, v] (m, m/s), shape (..., 6).
    body : Body
        Central body.
    j2 : bool
        Whether the gravity includes J2.

    Returns
    -------
    numpy.ndarray
        [x, y, z, vx, vy, vz] (m, m/s), Hill axes, shape (..., 6).
    """
    target = np.asarray(target, dtype=float)
    rel = relative_inertial(target, chaser)
    accel = gravity(target[..., :3], body, j2)
    return inertial_to_hill(target[..., :3], target[..., 3:], rel[..., :3], rel[..., 3:], accel)


__all__ = [
    "Burn",
    "accel_at",
    "check_aligned",
    "hill_states",
    "inertial_accel",
    "relative_inertial",
]
