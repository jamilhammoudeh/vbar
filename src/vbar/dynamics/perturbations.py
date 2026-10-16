"""Earth shadow, upper-atmosphere density and the forces they set.

Positions are body-centred inertial in m, with z along the spin axis, and may be stacked as
(..., 3). `sun` is the inertial unit vector from the body toward the Sun; over a rendezvous
episode it moves by about 1 deg per day, so models hold it fixed.

Relative motion in low orbit is driven by differential gravity (J2 included), differential drag
and differential solar pressure. At 500 km a 10% mismatch in ballistic coefficient gives 2e-8
to 1e-7 m/s^2 of differential drag, metres of along-track drift per orbit. Left out, with their
differential accelerations at 1 km separation (Montenbruck and Gill, Satellite Orbits, fig. 3.1
and secs. 3.2 to 3.7): lunar and solar gravity, 1.7e-10 and 8e-11 m/s^2, under a centimetre
per orbit; harmonics beyond J2, 2e-8 to 5e-8 m/s^2 but periodic, centimetres per orbit; Earth
albedo and infrared, 10 to 35% of a solar pressure that already largely cancels between the
two spacecraft; relativity and tides, nothing measurable. They matter for the absolute orbit,
not for the chaser relative to the target.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..conventions import hill_frame
from .bodies import AU, EARTH, SOLAR_PRESSURE, SUN_RADIUS, Body

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


class Drag:
    """Cannonball drag on target and chaser in an atmosphere turning with the body.

    a = -rho |v_r| v_r / (2 B), with v_r the velocity relative to the air and B = m / (C_D A)
    each spacecraft's ballistic coefficient in kg/m^2 (Montenbruck and Gill, Satellite Orbits,
    eq 3.97). `density` maps inertial positions (..., 3) to kg/m^3, for example
    `HarrisPriester`. Drag is the largest non-gravitational relative force in low orbit;
    ballistic coefficients of similar spacecraft differ by 10 to 50%, those of a compact chaser
    and a space station by a factor of 4 to 8 (Fehse, Automated Rendezvous and Docking of
    Spacecraft, sec. 4.2.1).
    """

    name = "drag"

    def __init__(
        self,
        density: Callable[[np.ndarray], np.ndarray],
        ballistic_target: float,
        ballistic_chaser: float,
        body: Body = EARTH,
    ):
        self.density = density
        self.ballistic_target = ballistic_target
        self.ballistic_chaser = ballistic_chaser
        self.body = body

    def acceleration(self, r: ArrayLike, v: ArrayLike, ballistic: float) -> np.ndarray:
        """Drag acceleration (m/s^2) at inertial position `r` and velocity `v`."""
        r = np.asarray(r, dtype=float)
        v_air = np.asarray(v, dtype=float) - np.cross([0.0, 0.0, self.body.spin_rate], r)
        speed = np.sqrt(_dot(v_air, v_air))
        return -(0.5 * self.density(r) * speed / ballistic)[..., None] * v_air

    def accelerations(
        self, r: np.ndarray, v: np.ndarray, dr: np.ndarray, dv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Target's drag and the chaser's minus the target's, m/s^2."""
        target = self.acceleration(r, v, self.ballistic_target)
        return target, self.acceleration(r + dr, v + dv, self.ballistic_chaser) - target


class SolarPressure:
    """Cannonball solar radiation pressure on target and chaser, dimmed by the conical shadow.

    a = -nu P C_R (A / m) (AU / d)^2 d_vec / d, with d_vec the vector to the Sun, P the pressure
    on an absorbing surface at 1 AU and nu the visible fraction of the Sun (Montenbruck and Gill,
    Satellite Orbits, eq 3.75). `target` and `chaser` are C_R A / m in m^2/kg; C_R is 1 for an
    absorbing surface and 1.2 to 1.9 for real ones. Relative motion feels the difference
    (Scheeres, Orbital Motion in Strongly Perturbed Environments, sec. 2.6), which vanishes
    for equal coefficients except across the penumbra. `distance` is the Sun's in m.
    """

    name = "srp"

    def __init__(
        self,
        sun: ArrayLike,
        target: float,
        chaser: float,
        body: Body = EARTH,
        distance: float = AU,
    ):
        self.sun = _unit(sun)
        self.target = target
        self.chaser = chaser
        self.body = body
        self.distance = distance

    def acceleration(self, r: ArrayLike, coefficient: float) -> np.ndarray:
        """Acceleration (m/s^2) at inertial `r` of a body with C_R A / m = `coefficient`."""
        r = np.asarray(r, dtype=float)
        to_sun = self.distance * self.sun - r
        d = np.sqrt(_dot(to_sun, to_sun))
        lit = conical_shadow(r, self.sun, self.body, self.distance)
        return -(lit * SOLAR_PRESSURE * coefficient * AU**2 / d**3)[..., None] * to_sun

    def accelerations(
        self, r: np.ndarray, v: np.ndarray, dr: np.ndarray, dv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Target's acceleration and the chaser's minus the target's, m/s^2."""
        target = self.acceleration(r, self.target)
        return target, self.acceleration(r + dr, self.chaser) - target


@dataclass(frozen=True)
class Plate:
    """A flat surface, lit on either side, of a spacecraft holding its attitude in Hill axes.

    `area_to_mass` is the plate's area over the spacecraft's mass (m^2/kg), `normal` a unit
    vector in the target's Hill axes, and `specular` and `diffuse` the fractions of light
    reflected each way; the rest is absorbed.
    """

    area_to_mass: float
    normal: tuple[float, float, float]
    specular: float = 0.0
    diffuse: float = 0.0


class FlatPlatePressure:
    """Solar radiation pressure on spacecraft made of flat plates, dimmed by the conical shadow.

    A lit plate feels -nu P (AU / d)^2 (A / m) cos(t) [(1 - s) u + 2 (s cos(t) + f / 3) n], with
    u the unit vector to the Sun, n the normal on the lit side, t the angle between them, and s
    and f the specular and diffuse fractions (Montenbruck and Gill, Satellite Orbits, eq 3.72,
    with Lambertian diffuse reflection; Scheeres, Orbital Motion in Strongly Perturbed
    Environments, eq 2.64). Both spacecraft keep their plates fixed in the target's Hill axes
    (the chaser's own axes differ by its angular separation, 1e-4 rad at 1 km). Plates shading
    one another, and the target shading the chaser, are not modelled.
    """

    name = "srp_plate"

    def __init__(
        self,
        sun: ArrayLike,
        target: Sequence[Plate],
        chaser: Sequence[Plate],
        body: Body = EARTH,
        distance: float = AU,
    ):
        self.sun = _unit(sun)
        self.target = tuple(target)
        self.chaser = tuple(chaser)
        self.body = body
        self.distance = distance

    def acceleration(self, r: ArrayLike, axes: np.ndarray, plates: Sequence[Plate]) -> np.ndarray:
        """Acceleration (m/s^2) at inertial `r` of `plates` oriented by Hill axes `axes`.

        `axes` has the Hill x, y, z axes as columns in inertial coordinates (`hill_frame`).
        """
        r = np.asarray(r, dtype=float)
        to_sun = self.distance * self.sun - r
        d = np.sqrt(_dot(to_sun, to_sun))
        u = to_sun / d[..., None]
        lit = conical_shadow(r, self.sun, self.body, self.distance)
        scale = lit * SOLAR_PRESSURE * (AU / d) ** 2
        accel = np.zeros_like(r)
        for plate in plates:
            n = axes @ np.asarray(plate.normal, dtype=float)
            cos_t = _dot(n, u)
            n = np.where(cos_t[..., None] < 0.0, -n, n)
            cos_t = np.abs(cos_t)
            normal_part = 2.0 * (plate.specular * cos_t + plate.diffuse / 3.0)
            push = (1.0 - plate.specular) * u + normal_part[..., None] * n
            accel = accel - (scale * plate.area_to_mass * cos_t)[..., None] * push
        return accel

    def accelerations(
        self, r: np.ndarray, v: np.ndarray, dr: np.ndarray, dv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Target's acceleration and the chaser's minus the target's, m/s^2."""
        axes, _ = hill_frame(r, v)
        target = self.acceleration(r, axes, self.target)
        return target, self.acceleration(r + dr, axes, self.chaser) - target
