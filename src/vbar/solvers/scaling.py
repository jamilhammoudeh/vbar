"""Nondimensional units for Hill-frame problems.

Hill-frame quantities mix metres, seconds and a mean motion near 1e-3 rad/s, so a problem posed
in SI units has entries spread over many orders of magnitude. Newton-type methods are affine
invariant in exact arithmetic, but their stopping tests and factorizations are not; solving in
units of a length L and the time 1/n brings positions, speeds and accelerations to order one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HillScaling:
    """Units of length L (m), time 1/n (s), speed n L (m/s) and acceleration n^2 L (m/s^2).

    In these units the Clohessy-Wiltshire equations have unit mean motion.

    Parameters
    ----------
    length : float
        L (m), finite and positive, usually the scenario's range scale.
    n : float
        Mean motion of the target orbit (rad/s), finite and positive.

    Raises
    ------
    ValueError
        If `length` or `n` is not finite and positive.
    """

    length: float
    n: float

    def __post_init__(self):
        for name in ("length", "n"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    def _state_units(self) -> np.ndarray:
        speed = self.n * self.length
        return np.array([self.length] * 3 + [speed] * 3)

    def state(self, x) -> np.ndarray:
        """Nondimensional relative states.

        Parameters
        ----------
        x : array_like, shape (..., 6)
            Hill-frame states (m, m/s).

        Returns
        -------
        ndarray, shape (..., 6)
            Positions in units of L and velocities in units of n L.

        Raises
        ------
        ValueError
            If the last axis does not have six entries.
        """
        x = np.asarray(x, dtype=float)
        if x.shape[-1:] != (6,):
            raise ValueError("states must have six components on the last axis")
        return x / self._state_units()

    def state_si(self, x) -> np.ndarray:
        """Hill-frame states from nondimensional states.

        Parameters
        ----------
        x : array_like, shape (..., 6)
            Positions in units of L and velocities in units of n L.

        Returns
        -------
        ndarray, shape (..., 6)
            Hill-frame states (m, m/s).

        Raises
        ------
        ValueError
            If the last axis does not have six entries.
        """
        x = np.asarray(x, dtype=float)
        if x.shape[-1:] != (6,):
            raise ValueError("states must have six components on the last axis")
        return x * self._state_units()

    def accel(self, a) -> np.ndarray:
        """Nondimensional accelerations.

        Parameters
        ----------
        a : array_like
            Accelerations (m/s^2), any shape.

        Returns
        -------
        ndarray
            `a` in units of n^2 L.
        """
        return np.asarray(a, dtype=float) / (self.n * self.n * self.length)

    def accel_si(self, a) -> np.ndarray:
        """Accelerations from nondimensional accelerations.

        Parameters
        ----------
        a : array_like
            Accelerations in units of n^2 L, any shape.

        Returns
        -------
        ndarray
            `a` in m/s^2.
        """
        return np.asarray(a, dtype=float) * (self.n * self.n * self.length)

    def time(self, t: float) -> float:
        """Nondimensional duration.

        Parameters
        ----------
        t : float
            Duration (s).

        Returns
        -------
        float
            `t` in units of 1/n, the orbit angle swept (rad).
        """
        return float(t) * self.n

    def time_si(self, t: float) -> float:
        """Duration from a nondimensional duration.

        Parameters
        ----------
        t : float
            Duration in units of 1/n.

        Returns
        -------
        float
            `t` in seconds.
        """
        return float(t) / self.n
