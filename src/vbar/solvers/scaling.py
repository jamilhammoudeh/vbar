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

from .stage import StageQP


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

    def stage(self, qp: StageQP) -> StageQP:
        """The horizon problem in these units, with the same objective value.

        With D = diag(1/L x3, 1/(n L) x3) on states and 1/(n^2 L) on inputs, the scaled problem
        has A' = D A D^-1, B' = D B n^2 L, c' = D c and x0' = D x0; costs are transformed
        so that every plan has the same objective in both units, and rows keep their values.

        Parameters
        ----------
        qp : StageQP
            A problem on Hill-frame states (m, m/s) and accelerations (m/s^2).

        Returns
        -------
        StageQP
            The same problem on nondimensional states and accelerations. Its solution maps back
            through `state_si` and `accel_si`. Row multipliers equal those in SI units; the
            dynamics multipliers are y' = D^-1 y, so y = D y', which is `y' / L` for the
            position rows and `y' / (n L)` for the velocity rows.

        Raises
        ------
        ValueError
            If the states do not have six components.
        """
        if qp.A.shape[1] != 6:
            raise ValueError("states must have six components")
        nu = qp.B.shape[2]
        to_scaled = 1.0 / self._state_units()
        accel_unit = self.n * self.n * self.length
        # z_si = to_si * z_scaled on z = (x, u).
        to_si = np.concatenate([self._state_units(), np.full(nu, accel_unit)])
        x_units = self._state_units()
        return StageQP(
            x0=qp.x0 * to_scaled,
            A=to_scaled[:, None] * qp.A * x_units[None, :],
            B=to_scaled[:, None] * qp.B * accel_unit,
            c=qp.c * to_scaled,
            Q=qp.Q * np.outer(to_si, to_si),
            q=qp.q * to_si,
            P=qp.P * np.outer(x_units, x_units),
            p=qp.p * x_units,
            C=None if qp.C is None else qp.C * to_si,
            e=qp.e,
            C_N=None if qp.C_N is None else qp.C_N * x_units,
            e_N=qp.e_N,
        )
