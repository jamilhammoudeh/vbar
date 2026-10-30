"""Frames of the KSPDG-format games: body-centred inertial, NTW and vessel body axes.

Positions and velocities are in a right-handed body-centred inertial frame with z along the
central body's north pole (KSPDG's "rhcbci"). The NTW frame of an orbit has T along the velocity,
W along the angular momentum and N = T x W, in the orbit plane and outward for a circular orbit
(KSPDG's "rhntw", components ordered N, T, W). A vessel's body axes are (forward, right, down)
(KSPDG's "rhvbody"). Each function returns a matrix whose rows are the frame's axes in inertial
coordinates, so `axes @ u_inertial` gives the components in that frame and `axes.T @ u` the
inverse.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

REF_FRAMES = ("rhvbody", "rhcbci", "rhntw")


def ntw_axes(r: ArrayLike, v: ArrayLike) -> np.ndarray:
    """Rows N, T, W of the orbit through inertial (r, v).

    Parameters
    ----------
    r, v : array_like, shape (3,)
        Position (m) and velocity (m/s), body-centred inertial.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Unit axes N, T, W as rows, in inertial coordinates.

    Raises
    ------
    ValueError
        If the velocity is zero or parallel to the position, where the frame is undefined.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    v_norm = np.linalg.norm(v)
    if not h_norm > 1e-12 * np.linalg.norm(r) * v_norm:
        raise ValueError("the NTW frame needs a velocity not parallel to the position")
    t = v / v_norm
    w = h / h_norm
    return np.stack([np.cross(t, w), t, w])


def body_axes(
    r: Sequence[float] | np.ndarray, r_target: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Rows forward, right, down of a vessel pointing at a target.

    Forward is the line of sight to the target, the vessel's pointing under target-hold. The
    roll about it is set so that down leans toward the central body: down is the part of -r
    perpendicular to forward. With the target straight above or below, down is taken from the
    inertial z axis instead.

    Parameters
    ----------
    r, r_target : array_like, shape (3,)
        Positions (m) of the vessel and its target, body-centred inertial.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Unit axes forward, right, down as rows, in inertial coordinates.

    Raises
    ------
    ValueError
        If the two positions coincide.
    """
    x, y, z = (float(c) for c in r)
    lx, ly, lz = (float(a) - b for a, b in zip(r_target, (x, y, z), strict=True))
    return axes_along((x, y, z), (lx, ly, lz))


def axes_along(
    r: Sequence[float] | np.ndarray, forward: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Rows forward, right, down of a vessel at `r` whose forward axis points along `forward`.

    Down is the part of -r perpendicular to forward, or of the inertial -z axis when forward is
    radial, as in `body_axes`.

    Parameters
    ----------
    r : array_like, shape (3,)
        Position (m), body-centred inertial.
    forward : array_like, shape (3,)
        The forward direction, any nonzero length, inertial.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Unit axes forward, right, down as rows, in inertial coordinates.

    Raises
    ------
    ValueError
        If `forward` is zero.
    """
    # Plain float arithmetic: this runs inside the integrator's derivative, where small numpy
    # calls cost more than the arithmetic.
    x, y, z = (float(c) for c in r)
    lx, ly, lz = (float(c) for c in forward)
    distance = math.sqrt(lx * lx + ly * ly + lz * lz)
    if not distance > 0.0:
        raise ValueError("the body frame needs a target away from the vessel")
    fx, fy, fz = lx / distance, ly / distance, lz / distance
    r_norm = math.sqrt(x * x + y * y + z * z)
    for ux, uy, uz in ((-x / r_norm, -y / r_norm, -z / r_norm), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)):
        along = ux * fx + uy * fy + uz * fz
        dx, dy, dz = ux - along * fx, uy - along * fy, uz - along * fz
        size = math.sqrt(dx * dx + dy * dy + dz * dz)
        if size >= 1e-6:
            break
    dx, dy, dz = dx / size, dy / size, dz / size
    right = (dy * fz - dz * fy, dz * fx - dx * fz, dx * fy - dy * fx)
    return np.array([(fx, fy, fz), right, (dx, dy, dz)])


__all__ = ["REF_FRAMES", "axes_along", "body_axes", "ntw_axes"]
