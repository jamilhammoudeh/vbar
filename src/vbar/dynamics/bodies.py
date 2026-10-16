"""Central bodies: gravitational parameter, equatorial radius and J2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Body:
    name: str
    mu: float  # m^3/s^2
    radius: float  # m, equatorial
    j2: float = 0.0


# WGS-84 mu and equatorial radius; J2 = -sqrt(5) C20 from EGM-96.
EARTH = Body("earth", mu=3.986004418e14, radius=6_378_137.0, j2=1.08262668e-3)
# Kerbal Space Program values, as used by KSPDG. The game has no oblateness.
KERBIN = Body("kerbin", mu=3.5316e12, radius=600_000.0)

BODIES = {body.name: body for body in (EARTH, KERBIN)}


def get_body(body: str | Body) -> Body:
    """Look up a body by name; a `Body` is returned unchanged."""
    if isinstance(body, Body):
        return body
    try:
        return BODIES[body.lower()]
    except KeyError:
        raise ValueError(f"unknown body {body!r}, expected one of {sorted(BODIES)}") from None
