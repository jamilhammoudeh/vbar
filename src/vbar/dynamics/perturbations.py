"""Earth shadow for lighting and solar radiation pressure.

Positions are body-centred inertial in m, with z along the spin axis, and may be stacked as
(..., 3). `sun` is the inertial unit vector from the body toward the Sun; over a rendezvous
episode it moves by about 1 deg per day, so models hold it fixed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .bodies import AU, EARTH, SUN_RADIUS, Body


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def _unit(v: ArrayLike) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def cylindrical_shadow(r: ArrayLike, sun: ArrayLike, body: Body = EARTH) -> np.ndarray:
    """1 in sunlight, 0 inside the body's shadow cylinder.

    For lighting, where the few seconds of penumbra do not matter (Montenbruck and Gill,
    Satellite Orbits, sec. 3.4.2).
    """
    r = np.asarray(r, dtype=float)
    along = _dot(r, _unit(sun))
    behind = (along < 0.0) & (_dot(r, r) - along**2 < body.radius**2)
    return np.where(behind, 0.0, 1.0)


def conical_shadow(
    r: ArrayLike, sun: ArrayLike, body: Body = EARTH, distance: float = AU
) -> np.ndarray:
    """Fraction of the solar disk seen from `r`, with umbra and penumbra.

    The Sun and the body are treated as disks of their apparent radii and the occulted area is
    their overlap (Montenbruck and Gill, Satellite Orbits, eqs 3.85 to 3.94). The result varies
    continuously, so solar pressure built on it needs no event handling at shadow entry.
    `distance` is the Sun's distance from the body in m.
    """
    r = np.asarray(r, dtype=float)
    to_sun = distance * _unit(sun) - r
    r_norm = np.sqrt(_dot(r, r))
    d_norm = np.sqrt(_dot(to_sun, to_sun))
    a = np.arcsin(SUN_RADIUS / d_norm)
    b = np.arcsin(body.radius / r_norm)
    c = np.arccos(np.clip(-_dot(r, to_sun) / (r_norm * d_norm), -1.0, 1.0))
    # Partial overlap of two disks of radii a and b whose centres are c apart. Concentric disks
    # fall in the other branches below.
    c_p = np.where(c > 0.0, c, 1.0)
    x = (c_p * c_p + a * a - b * b) / (2.0 * c_p)
    y = np.sqrt(np.maximum(a * a - x * x, 0.0))
    overlap = (
        a * a * np.arccos(np.clip(x / a, -1.0, 1.0))
        + b * b * np.arccos(np.clip((c_p - x) / b, -1.0, 1.0))
        - c_p * y
    )
    partial = 1.0 - overlap / (np.pi * a * a)
    # Beyond the umbra's vertex the body crosses the solar disk as a ring (annular eclipse).
    annular = 1.0 - (b / a) ** 2
    return np.select(
        [c >= a + b, c <= b - a, c <= a - b], [1.0, 0.0, annular], np.clip(partial, 0.0, 1.0)
    )
