"""What every backend shares: thrust segments, and the frame conversion back to Hill states."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

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
    """The Hill acceleration (m/s^2) in effect at `t`, zero between burns.

    Burns must not overlap: the first one in the list that covers `t` would win. `check_burns`
    refuses burns that overlap.

    Parameters
    ----------
    burns : sequence of Burn
        Chaser accelerations, held in the target's Hill axes.
    t : float
        Time (s) from the start.

    Returns
    -------
    numpy.ndarray
        Acceleration (m/s^2), Hill axes, shape (3,).
    """
    for burn in burns:
        if burn.start <= t < burn.start + burn.duration:
            return np.array(burn.accel_hill)
    return np.zeros(3)


def check_burns(burns: Sequence[Burn]) -> None:
    """Raise ValueError unless `burns` are `Burn`s that start at t = 0 or later and never overlap.

    Burns that touch, or overlap by 1e-9 s or less (rounding of their edges), are accepted.

    Parameters
    ----------
    burns : sequence of Burn
        Chaser accelerations, in any order.

    Raises
    ------
    ValueError
        If an element is not a `Burn`, a burn starts before t = 0, or two burns overlap by more
        than 1e-9 s.
    """
    if not all(isinstance(burn, Burn) for burn in burns):
        raise ValueError("burns must be Burn instances")
    if any(burn.start < 0.0 for burn in burns):
        raise ValueError("burns must start at t = 0 or later")
    # Sorted by start, the first burn that overlaps an earlier one overlaps the one just before
    # it, so comparing neighbours is enough.
    end = -math.inf
    for burn in sorted(burns, key=lambda b: b.start):
        if end - burn.start > 1e-9:
            raise ValueError("burns must not overlap")
        end = burn.start + burn.duration


def _check_real(name: str, value: object) -> np.ndarray:
    """`value` as a float array, refusing complex, string and object entries."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numbers")
    return array.astype(float, copy=False)


def _check_state(name: str, value: object) -> np.ndarray:
    """`value` as six finite numbers."""
    state = _check_real(name, value)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError(f"{name} must be six finite numbers")
    return state


def _check_times(times: object) -> np.ndarray:
    """`times` as a non-empty 1-D array of finite, non-negative, non-decreasing numbers."""
    samples = _check_real("times", times)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("times must be a non-empty 1-D array")
    if not np.isfinite(samples).all() or samples[0] < 0.0 or np.any(np.diff(samples) < 0.0):
        raise ValueError("times must be finite, not negative and in time order")
    return samples


def _check_positive(name: str, value: object) -> None:
    """Raise ValueError unless `value` is a finite positive number."""
    # bool is an int, so True would pass as 1.
    if isinstance(value, bool) or not isinstance(value, Real) or not 0.0 < float(value) < math.inf:
        raise ValueError(f"{name} must be a finite positive number")


def check_propagation(
    target: ArrayLike,
    chaser: ArrayLike,
    times: ArrayLike,
    burns: Sequence[Burn],
    tolerance: float,
    substep: float,
    mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check the arguments both propagators take and return the states and times as arrays.

    Parameters
    ----------
    target, chaser : array_like
        Initial inertial [r, v] (m, m/s), shape (6,).
    times : array_like
        Sample times (s) from the start, 1-D.
    burns : sequence of Burn
        Chaser accelerations, held in the target's Hill axes.
    tolerance : float
        Integrator tolerance, positive.
    substep : float
        Step (s), positive.
    mass : float
        Chaser mass (kg), positive.

    Returns
    -------
    tuple of numpy.ndarray
        Target state (6,), chaser state (6,) and sample times (k,), all float.

    Raises
    ------
    ValueError
        If a state is not six finite real numbers, `times` is not a non-empty 1-D array of
        finite, non-negative, non-decreasing numbers, `tolerance`, `substep` or `mass` is not a
        finite positive number, or `burns` fails `check_burns`.
    """
    x_t = _check_state("target", target)
    x_c = _check_state("chaser", chaser)
    samples = _check_times(times)
    for name, value in (("tolerance", tolerance), ("substep", substep), ("mass", mass)):
        _check_positive(name, value)
    check_burns(burns)
    return x_t, x_c, samples


def check_aligned(burns: Sequence[Burn], substep: float) -> None:
    """Raise ValueError unless every burn starts and ends on a multiple of `substep` (s).

    A backend holds a burn's direction over whole substeps, so an edge inside one would move the
    burn and the comparison would blame the wrong side.

    Parameters
    ----------
    burns : sequence of Burn
        Chaser accelerations.
    substep : float
        Step (s), positive.

    Raises
    ------
    ValueError
        If a burn starts or ends more than 1e-9 s from a multiple of `substep`.
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

    Parameters
    ----------
    burns : sequence of Burn
        Chaser accelerations, held in the target's Hill axes.
    t : float
        Time (s) from the start.
    target : array_like
        Target inertial [r, v] (m, m/s) at `t`, shape (6,).
    body : Body
        Central body.
    j2 : bool
        Whether the gravity that sets the Hill axes' rate includes J2.
    hold : float, optional
        Time (s) the direction is held fixed, from `t`.

    Returns
    -------
    numpy.ndarray
        Acceleration (m/s^2), inertial axes, shape (3,); zero when no burn is in effect at `t`.
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
    """Chaser minus target, inertial [dr, dv] (m, m/s), for stacked states.

    Parameters
    ----------
    target, chaser : array_like
        Inertial [r, v] (m, m/s), shape (..., 6).

    Returns
    -------
    numpy.ndarray
        Inertial [dr, dv] (m, m/s), shape (..., 6).
    """
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
    "check_burns",
    "check_propagation",
    "hill_states",
    "inertial_accel",
    "relative_inertial",
]
