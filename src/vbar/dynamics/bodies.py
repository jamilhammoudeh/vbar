"""Physical constants: central bodies, the Sun, and three-body systems."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Body:
    name: str
    mu: float  # m^3/s^2
    radius: float  # m, equatorial
    j2: float = 0.0
    spin_rate: float = 0.0  # rad/s, about the inertial z axis
    flattening: float = 0.0


# One matched set, WGS 84: GM, equatorial radius and the J2 of its normal gravity field, which
# follows from those and the flattening and spin rate (NIMA TR8350.2; EGM96 report, sec. 11).
# J2 differs from EGM96's own value, which goes with GM 3.986004415e14 and a 6378136.3 m, by 3e-6.
EARTH = Body(
    "earth",
    mu=3.986004418e14,
    radius=6_378_137.0,
    j2=1.08262982131e-3,
    spin_rate=7.292115e-5,
    flattening=1 / 298.257223563,
)
# Kerbal Space Program values, as used by KSPDG. The game has no oblateness.
KERBIN = Body("kerbin", mu=3.5316e12, radius=600_000.0)

BODIES = {body.name: body for body in (EARTH, KERBIN)}

AU = 1.495978707e11  # m, IAU 2012 resolution B2
SUN_RADIUS = 6.957e8  # m, IAU 2015 nominal solar radius
# Radiation pressure on an absorbing surface at 1 AU: 1367 W/m^2 over the speed of light
# (Montenbruck and Gill, Satellite Orbits, eq 3.69).
SOLAR_PRESSURE = 4.56e-6  # N/m^2


@dataclass(frozen=True)
class ThreeBodySystem:
    """Constants of a circular restricted three-body problem, named and versioned.

    `mu` is the smaller primary's share of the total mass, `length` the primaries' distance
    (m) and `period` their orbital period (s). Nondimensional units have length 1, period
    2 pi and total mass 1. Results in metres and seconds move by about 0.15% between published
    sets of the same system (Koon, Lo, Marsden and Ross, Dynamical Systems, the Three-Body
    Problem and Space Mission Design, table 2.2.1 and ch. 6), so every result states its set.
    """

    name: str
    version: int
    mu: float
    length: float
    period: float

    @property
    def time(self) -> float:
        """Seconds per nondimensional time unit."""
        return self.period / math.tau

    @property
    def velocity(self) -> float:
        """m/s per nondimensional velocity unit."""
        return self.length / self.time

    @property
    def acceleration(self) -> float:
        """m/s^2 per nondimensional acceleration unit."""
        return self.length / self.time**2


# The mass ratio used in the near-rectilinear halo orbit literature, whose ephemeris source is
# not confirmed, the mean distance 384,400 km, and the period from Kepler's third law with
# WGS 84's GM.
_EM_MU = 0.01215058560962404
EARTH_MOON = ThreeBodySystem(
    "earth-moon",
    1,
    _EM_MU,
    3.844e8,
    math.tau * math.sqrt(3.844e8**3 * (1.0 - _EM_MU) / EARTH.mu),
)
# Koon, Lo, Marsden and Ross, table 2.2.1 and sec. 5.3.
EARTH_MOON_KLMR = ThreeBodySystem("earth-moon-klmr", 1, 0.01215, 3.85e8, 2.361e6)
SUN_JUPITER_KLMR = ThreeBodySystem("sun-jupiter-klmr", 1, 9.537e-4, 7.784e11, 3.733e8)
# Richardson's Sun-(Earth+Moon) mass ratio, which reproduces Koon et al.'s chapter 6
# constants, with the distance of their sec. 6.3 and the sidereal year of their sec. 5.3 (the
# period in their table 2.2.1, 3.147e7 s, is a misprint).
SUN_EARTH_RICHARDSON = ThreeBodySystem(
    "sun-earth-richardson", 1, 3.040357143e-6, 1.495978714e11, 3.156e7
)


@dataclass(frozen=True)
class BicircularSun:
    """The Sun as a perturber of a planet-moon three-body system, in that system's units.

    `mass` in units of the planet and moon's total mass, `distance` from their barycentre in
    units of their distance, and `rate` the Sun's angular rate in their synodic frame, where it
    turns clockwise: 1 minus the rate of their barycentre about the Sun.
    """

    name: str
    system: ThreeBodySystem
    mass: float
    distance: float
    rate: float


# Koon, Lo, Marsden and Ross, eqs 5.3.4 to 5.3.7.
SUN_FOR_EARTH_MOON_KLMR = BicircularSun(
    "sun-earth-moon-klmr", EARTH_MOON_KLMR, 328900.54, 388.81114, 0.925195985520347
)


def get_body(body: str | Body) -> Body:
    """Look up a body by name; a `Body` is returned unchanged."""
    if isinstance(body, Body):
        return body
    try:
        return BODIES[body.lower()]
    except KeyError:
        raise ValueError(f"unknown body {body!r}, expected one of {sorted(BODIES)}") from None
