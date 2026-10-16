"""How well one dynamics model stands in for another, independent of any controller.

The STM adequacy compares transition matrices over a horizon (Montenbruck and Gill, Satellite
Orbits, eq 7.109); the modeling-error index compares trajectories flown from a set of start
states (Alfriend et al., Spacecraft Formation Flying, eqs 8.58 to 8.60). States are Hill
states [x, y, z, vx, vy, vz]; velocities are scaled by 1/n, the reference mean motion in rad/s,
so that position and velocity count alike.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def _velocity_scale(n: float) -> np.ndarray:
    return np.array([1.0, 1.0, 1.0, 1.0 / n, 1.0 / n, 1.0 / n])


def stm_adequacy(phi_true: ArrayLike, phi_model: ArrayLike, n: float) -> np.ndarray:
    """Largest entry of I - phi_true phi_model^-1, with velocities scaled by 1/n.

    Zero when the model's transition matrix is the true one; Montenbruck and Gill take 0.1 as
    the limit of an adequate model (exercise 7.1). Stacked matrices (..., 6, 6) give one value
    each.
    """
    phi_true = np.asarray(phi_true, dtype=float)
    misfit = np.eye(6) - phi_true @ np.linalg.inv(np.asarray(phi_model, dtype=float))
    scale = _velocity_scale(n)
    return np.max(np.abs(misfit * scale[:, None] / scale[None, :]), axis=(-2, -1))


def validity_horizon(times: ArrayLike, adequacy: ArrayLike, limit: float = 0.1) -> float:
    """First of the increasing `times` (s) at which `adequacy` exceeds `limit`; inf if none."""
    over = np.flatnonzero(np.asarray(adequacy) > limit)
    return float(np.asarray(times)[over[0]]) if over.size else math.inf


def modeling_error_index(truth: ArrayLike, model: ArrayLike, n: float) -> float:
    """Largest |w(model) / w(truth) - 1| over a set of states, w the squared scaled norm.

    `truth` and `model` (k, 6) are the two models' states at one time, flown from the same k
    start states. For a small error along the state the index is about twice its fractional
    size. It stays flat as the separation grows when the model leaves out effects linear in it
    (J2, eccentricity) and grows with it when the model leaves out nonlinearity.
    """
    scale = _velocity_scale(n)
    true_norm = np.sum((np.asarray(truth, dtype=float) * scale) ** 2, axis=-1)
    model_norm = np.sum((np.asarray(model, dtype=float) * scale) ** 2, axis=-1)
    return float(np.max(np.abs(model_norm / true_norm - 1.0)))
