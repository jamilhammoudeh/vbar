"""Earth shadow, upper-atmosphere density and the forces they set.

Positions are body-centred inertial in m, with z along the spin axis, and may be stacked as
(..., 3). `sun` is the inertial unit vector from the body toward the Sun; over a rendezvous
episode it moves by about 1 deg per day, so models hold it fixed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .bodies import AU, EARTH, SUN_RADIUS, Body

# Harris-Priester densities for mean solar activity: height (km), minimum and maximum (g/km^3,
# which is 1e-12 kg/m^3). Montenbruck and Gill, Satellite Orbits, table 3.8.
_HARRIS_PRIESTER = np.array(
    [
        [100, 497400.0, 497400.0],
        [120, 24900.0, 24900.0],
        [130, 8377.0, 8710.0],
        [140, 3899.0, 4059.0],
        [150, 2122.0, 2215.0],
        [160, 1263.0, 1344.0],
        [170, 800.8, 875.8],
        [180, 528.3, 601.0],
        [190, 361.7, 429.7],
        [200, 255.7, 316.2],
        [210, 183.9, 239.6],
        [220, 134.1, 185.3],
        [230, 99.49, 145.5],
        [240, 74.88, 115.7],
        [250, 57.09, 93.08],
        [260, 44.03, 75.55],
        [270, 34.30, 61.82],
        [280, 26.97, 50.95],
        [290, 21.39, 42.26],
        [300, 17.08, 35.26],
        [320, 10.99, 25.11],
        [340, 7.214, 18.19],
        [360, 4.824, 13.37],
        [380, 3.274, 9.955],
        [400, 2.249, 7.492],
        [420, 1.558, 5.684],
        [440, 1.091, 4.355],
        [460, 0.7701, 3.362],
        [480, 0.5474, 2.612],
        [500, 0.3916, 2.042],
        [520, 0.2819, 1.605],
        [540, 0.2042, 1.267],
        [560, 0.1488, 1.005],
        [580, 0.1092, 0.7997],
        [600, 0.08070, 0.6390],
        [620, 0.06012, 0.5123],
        [640, 0.04519, 0.4121],
        [660, 0.03430, 0.3325],
        [680, 0.02632, 0.2691],
        [700, 0.02043, 0.2185],
        [720, 0.01607, 0.1779],
        [740, 0.01281, 0.1452],
        [760, 0.01036, 0.1190],
        [780, 0.008496, 0.09776],
        [800, 0.007069, 0.08059],
        [840, 0.004680, 0.05741],
        [880, 0.003200, 0.04210],
        [920, 0.002210, 0.03130],
        [960, 0.001560, 0.02360],
        [1000, 0.001150, 0.01810],
    ]
)
_HP_HEIGHT = 1e3 * _HARRIS_PRIESTER[:, 0]
_HP_LOG_DENSITY = np.log(1e-12 * _HARRIS_PRIESTER[:, 1:])


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


def geodetic_height(r: ArrayLike, body: Body = EARTH) -> np.ndarray:
    """Height (m) of inertial positions `r` above the body's reference ellipsoid.

    Iterates on where the ellipsoid normal through `r` meets the spin axis (Montenbruck and
    Gill, Satellite Orbits, eqs 5.83 to 5.88). Each pass shrinks the error by about the
    eccentricity squared, 7e-3 for the Earth.
    """
    r = np.asarray(r, dtype=float)
    e2 = body.flattening * (2.0 - body.flattening)
    rho2 = r[..., 0] ** 2 + r[..., 1] ** 2
    z = r[..., 2]
    dz = e2 * z
    for _ in range(20):
        z_dz = z + dz
        normal_length = np.sqrt(rho2 + z_dz**2)
        sin_lat = z_dz / normal_length
        prime_vertical = body.radius / np.sqrt(1.0 - e2 * sin_lat**2)
        dz_next = prime_vertical * e2 * sin_lat
        converged = np.max(np.abs(dz_next - dz)) < 1e-9
        dz = dz_next
        if converged:
            return normal_length - prime_vertical
    raise RuntimeError("geodetic height did not converge")


class HarrisPriester:
    """Harris-Priester upper-atmosphere density (kg/m^3) for mean solar activity.

    Between 100 and 1000 km of geodetic height the table's minimum and maximum densities are
    interpolated exponentially and blended over the diurnal bulge, whose apex trails the Sun by
    30 deg in right ascension: rho = rho_min + (rho_max - rho_min) cos^n(psi / 2), with psi the
    angle from the apex. `exponent` n is 2 for low-inclination orbits up to 6 for polar ones
    (Montenbruck and Gill, Satellite Orbits, sec. 3.5.2). Density is zero above 1000 km.
    Density models of this kind agree with each other only to 15 to 40%.
    """

    lag = np.radians(30.0)

    def __init__(self, sun: ArrayLike, exponent: float, body: Body = EARTH):
        s = _unit(sun)
        ra = np.arctan2(s[1], s[0]) + self.lag
        cos_dec = np.hypot(s[0], s[1])
        self.apex = np.array([cos_dec * np.cos(ra), cos_dec * np.sin(ra), s[2]])
        self.exponent = exponent
        self.body = body

    def __call__(self, r: ArrayLike) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        h = geodetic_height(r, self.body)
        if np.any(h < _HP_HEIGHT[0]):
            raise ValueError("below the 100 km floor of the harris-priester table")
        rho_min = np.exp(np.interp(h, _HP_HEIGHT, _HP_LOG_DENSITY[:, 0]))
        rho_max = np.exp(np.interp(h, _HP_HEIGHT, _HP_LOG_DENSITY[:, 1]))
        cos_psi = _dot(r, self.apex) / np.sqrt(_dot(r, r))
        bulge = np.maximum(0.5 + 0.5 * cos_psi, 0.0) ** (0.5 * self.exponent)
        rho = rho_min + (rho_max - rho_min) * bulge
        return np.where(h > _HP_HEIGHT[-1], 0.0, rho)
