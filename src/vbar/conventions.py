"""Frames, units and constants shared across the package.

Units are SI throughout (m, m/s, m/s^2, s, kg, N) and angles are radians. Central-body
constants live in `vbar.dynamics.bodies`.

The Hill frame is centred on the target and rotates with its orbit: x radial outward (R-bar),
y along-track (V-bar), z along the orbit normal r x v. It is the frame called RIC, RSW or RTN
elsewhere. Relative velocities are rates seen in the rotating frame, not inertial velocity
differences.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

G0 = 9.80665  # m/s^2, standard gravity; converts Isp in seconds to exhaust velocity

# Rows are the ISS-style LVLH axes in Hill coordinates: x along velocity, y opposite the
# orbit normal, z toward the central body.
HILL_TO_ISS = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])


def hill_frame(
    r: ArrayLike, v: ArrayLike, accel: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Hill axes of the orbit through (r, v) and the frame's angular velocity.

    Returns `C`, whose columns are the Hill x, y, z axes in inertial coordinates
    (v_inertial = C v_hill), and `omega`, the frame's angular velocity in Hill coordinates
    (rad/s). `accel` is the reference orbit's inertial acceleration; only its orbit-normal
    component matters, which turns the orbit plane about x (zero for Keplerian motion).
    Inputs may be stacked along leading axes.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    h = np.cross(r, v)
    r_norm = np.linalg.norm(r, axis=-1)
    h_norm = np.linalg.norm(h, axis=-1)
    x = r / r_norm[..., None]
    z = h / h_norm[..., None]
    y = np.cross(z, x)
    if accel is None:
        w_x = np.zeros_like(r_norm)
    else:
        w_x = r_norm * np.sum(np.asarray(accel, dtype=float) * z, axis=-1) / h_norm
    omega = np.stack([w_x, np.zeros_like(r_norm), h_norm / r_norm**2], axis=-1)
    return np.stack([x, y, z], axis=-1), omega


def inertial_to_hill(
    r: ArrayLike, v: ArrayLike, dr: ArrayLike, dv: ArrayLike, accel: ArrayLike | None = None
) -> np.ndarray:
    """Relative state [x, y, z, vx, vy, vz] in the Hill frame of the target orbit (r, v).

    `dr` and `dv` are the chaser's inertial position and velocity minus the target's.
    """
    C, omega = hill_frame(r, v, accel)
    rho = np.einsum("...ji,...j->...i", C, dr)
    rho_dot = np.einsum("...ji,...j->...i", C, dv) - np.cross(omega, rho)
    return np.concatenate([rho, rho_dot], axis=-1)


def hill_to_inertial(
    r: ArrayLike, v: ArrayLike, rel: ArrayLike, accel: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Inertial position and velocity offsets (chaser minus target) of a Hill relative state."""
    C, omega = hill_frame(r, v, accel)
    rel = np.asarray(rel, dtype=float)
    rho, rho_dot = rel[..., :3], rel[..., 3:]
    dr = np.einsum("...ij,...j->...i", C, rho)
    dv = np.einsum("...ij,...j->...i", C, rho_dot + np.cross(omega, rho))
    return dr, dv


def hill_to_iss(x: ArrayLike) -> np.ndarray:
    """Hill vector (..., 3) or state (..., 6) expressed in ISS-style LVLH axes."""
    x = np.asarray(x, dtype=float)
    return (x.reshape(*x.shape[:-1], -1, 3) @ HILL_TO_ISS.T).reshape(x.shape)


def iss_to_hill(x: ArrayLike) -> np.ndarray:
    """ISS-style LVLH vector (..., 3) or state (..., 6) expressed in Hill axes."""
    x = np.asarray(x, dtype=float)
    return (x.reshape(*x.shape[:-1], -1, 3) @ HILL_TO_ISS).reshape(x.shape)
