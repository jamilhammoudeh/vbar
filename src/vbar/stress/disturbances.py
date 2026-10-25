"""Scripted navigation and execution errors, fixed by a few numbers in the unit box.

A falsification search needs disturbances it can steer: a seed only picks a random draw, so
searching seeds is random search, and a per-step error sequence has thousands of dimensions over
an episode. These models instead fix the error by a short vector in [-1, 1]^k and scale it into a
declared error set, so every point of the box is an error the set allows. They draw nothing:
the same vector gives the same errors whatever stream the environment hands them.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from ..safety.tightening import BoxError

# Rows of navigation offsets per episode, each held for an equal share of it.
NAV_BLOCKS = 4


def _unit_box(values: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not (np.all(np.isfinite(array)) and np.all(np.abs(array) <= 1.0)):
        raise ValueError(f"{name} must lie in [-1, 1]")
    return array


class ScriptedNav:
    """Navigation error held at a scripted offset in a declared box for each quarter episode.

    The estimate is the true state plus `offsets[k]` times the box's radii, k the quarter of the
    episode the observation time falls in; observations from `duration` on keep the last row.
    It follows the sensor interface the environment calls, so the agent sees the estimate only.

    Parameters
    ----------
    offsets : array_like, shape (4, 6)
        One row per quarter episode, each entry in [-1, 1], in units of the box's radius on that
        Hill axis (order x, y, z, vx, vy, vz).
    box : BoxError
        The declared navigation box; its `r_pos` (m) and `r_vel` (m/s) scale the offsets.
    duration : float
        The episode's length, s, finite and positive.

    Raises
    ------
    ValueError
        If `offsets` is not a (4, 6) array in [-1, 1] or `duration` is not finite and positive.
    TypeError
        If `box` is not a `BoxError`.
    """

    name = "scripted"

    def __init__(self, offsets: ArrayLike, *, box: BoxError, duration: float):
        if not isinstance(box, BoxError):
            raise TypeError(f"box must be a BoxError, not {type(box).__name__}")
        duration = float(duration)
        if not (math.isfinite(duration) and duration > 0.0):
            raise ValueError("duration must be finite and positive")
        rows = _unit_box(offsets, (NAV_BLOCKS, 6), "offsets")
        self.error = rows * np.repeat([box.r_pos, box.r_vel], 3)
        self.error.flags.writeable = False
        self.duration = duration

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def observe(self, true_state: ArrayLike, t: float) -> np.ndarray:
        block = min(NAV_BLOCKS - 1, max(0, math.floor(NAV_BLOCKS * t / self.duration)))
        return np.asarray(true_state, dtype=float) + self.error[block]


class ScriptedExecution:
    """A constant acceleration error on every burn, scripted inside a declared ball.

    `direction` is mapped radially from the cube [-1, 1]^3 onto the ball of radius `da_max`,
    w = da_max max_i |d_i| d / |d|, so the cube's faces land on the sphere and its centre on
    zero. A burn delivers the command plus w / a_max; a coast is delivered unchanged, as
    `vbar.spacecraft.ExecutionError` leaves it. It follows that model's interface.

    Parameters
    ----------
    direction : array_like, shape (3,)
        The error's direction and size in the cube, each entry in [-1, 1], Hill axes.
    da_max : float
        The declared execution ball's radius, m/s^2, finite and nonnegative.
    a_max : float
        The acceleration of a full command, m/s^2, finite and positive.

    Raises
    ------
    ValueError
        If `direction` is not three numbers in [-1, 1], `da_max` is negative or not finite, or
        `a_max` is not finite and positive.
    """

    def __init__(self, direction: ArrayLike, *, da_max: float, a_max: float):
        d = _unit_box(direction, (3,), "direction")
        da_max, a_max = float(da_max), float(a_max)
        if not (math.isfinite(da_max) and da_max >= 0.0):
            raise ValueError("da_max must be finite and nonnegative")
        if not (math.isfinite(a_max) and a_max > 0.0):
            raise ValueError("a_max must be finite and positive")
        length = float(np.linalg.norm(d))
        w = np.zeros(3) if length == 0.0 else da_max * float(np.abs(d).max()) / length * d
        # In fractions of a full command, the units the spacecraft hands the model.
        self.shift = w / a_max
        self.shift.flags.writeable = False

    def apply(self, command: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not np.any(command):
            return command
        return command + self.shift


__all__ = ["NAV_BLOCKS", "ScriptedExecution", "ScriptedNav"]
