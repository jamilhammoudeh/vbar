"""Central-body constants (GM, equatorial radius, J2) and the Sun's radius and distance."""

from __future__ import annotations

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


def get_body(body: str | Body) -> Body:
    """Look up a body by name; a `Body` is returned unchanged."""
    if isinstance(body, Body):
        return body
    try:
        return BODIES[body.lower()]
    except KeyError:
        raise ValueError(f"unknown body {body!r}, expected one of {sorted(BODIES)}") from None
