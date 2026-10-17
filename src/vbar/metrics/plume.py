"""Thruster plume on the target: the plume constraint margin, exhaust time and impulse (S10).

Plume pressure follows the Gaussian model Fehse cites from Desplats (Automated Rendezvous and
Docking of Spacecraft, 2003, section 4.2.4, eqs. 4.12 and 4.13):
P(r, theta) = phi0 exp(-theta^2 / (2 theta0^2)) / r^2, with theta the angle from the exhaust axis
and phi0 set so the pressure integrated over a unit sphere equals the thrust. The force on the
target is that pressure on a declared area facing the plume.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from ..constraints import KeepOut
from .episode import Episode
from .settings import MetricSettings
from .stl import Segment, Signal, Trace, intersect, merge, segment_negative_intervals, total


def exhausts(accel: np.ndarray, layout: str) -> list[tuple[np.ndarray, float]]:
    """(unit exhaust direction, acceleration magnitude m/s^2) of each firing thruster.

    A ball layout has one engine along the acceleration; a box layout one thruster pair per Hill
    axis, so each nonzero component fires its own thruster.
    """
    accel = np.asarray(accel, dtype=float)
    if layout == "ball":
        size = float(np.linalg.norm(accel))
        return [(-accel / size, size)] if size > 0.0 else []
    return [(-np.sign(a) * np.eye(3)[i], abs(float(a))) for i, a in enumerate(accel) if a != 0.0]


def plume_normalization(sigma: float) -> float:
    """Pressure integral over a unit sphere per unit phi0; phi0 = thrust / this.

    2 pi times the integral of exp(-theta^2 / (2 sigma^2)) sin(theta) over [0, pi].
    """
    inner, _ = quad(lambda th: math.exp(-0.5 * (th / sigma) ** 2) * math.sin(th), 0.0, math.pi)
    return 2.0 * math.pi * inner


def _angle(direction: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Angle (rad) between `direction` and the line of sight from the chaser to the target, -r."""
    look = -np.asarray(state)[..., :3]
    if look.ndim == 1:
        # One state, as in the refinement searches: plain arithmetic is about 20 times faster.
        (a, b, c), (x, y, z) = direction, look
        across = math.hypot(b * z - c * y, c * x - a * z, a * y - b * x)
        return np.asarray(math.atan2(across, a * x + b * y + c * z))
    cross = np.linalg.norm(np.cross(direction, look), axis=-1)
    return np.arctan2(cross, look @ direction)


def _masses(ep: Episode) -> list[float]:
    """Mass (kg) at the start of each step."""
    if ep.view.mass_kg is None:
        raise ValueError("plume forces need the initial mass")
    used = np.cumsum([0.0, *(rec.actuation.propellant_kg for rec in ep.records)])
    return [ep.view.mass_kg - u for u in used[:-1]]


def plume_metrics(ep: Episode, settings: MetricSettings | None = None) -> dict[str, float]:
    """Plume margin (rad), time with exhaust on the target (s) and plume impulse (N s).

    Only thrust inside `plume_range` counts. The margin is the least angle between a firing
    thruster's exhaust and the line of sight to the target, minus `plume_half_angle`; the time
    is when that is negative; the impulse integrates the plume force on `plume_target_area`,
    with the thrust from the mass at the start of the step. All NaN when no range is declared.
    """
    settings = settings or MetricSettings()
    out = {"plume_margin": math.nan, "plume_time": math.nan, "plume_impulse": math.nan}
    if settings.plume_range is None:
        return out
    trace = Trace(ep)
    sigma, cone = settings.plume_sigma, settings.plume_half_angle
    scale = settings.plume_target_area / plume_normalization(sigma)
    near = KeepOut(settings.plume_range)
    margin, seconds, impulse = math.inf, 0.0, 0.0
    segments = iter(trace.segments)
    for rec, mass in zip(ep.records, _masses(ep), strict=True):
        for _ in rec.segment.segments:
            seg = next(segments)
            for direction, size in exhausts(seg.accel, ep.view.thrust_limit):
                inside = merge(segment_negative_intervals(near, trace, seg))
                if not inside:
                    continue
                low, on = _thruster(trace, seg, direction, cone, inside)
                margin = min(margin, low)
                seconds += on
                impulse += mass * size * scale * _exposure(trace, seg, direction, sigma, inside)
    out.update(plume_margin=margin, plume_time=seconds, plume_impulse=impulse)
    return out


def _thruster(trace: Trace, seg: Segment, direction, cone, inside) -> tuple[float, float]:
    """Least plume margin and time on target for one thruster over one segment."""

    def margin(state, t=0.0):
        return _angle(direction, state) - cone

    def rate(speed, accel, rho_min):
        return speed / rho_min if rho_min > 0.0 else math.inf

    signal = Signal("plume", margin, rate, 1e-6)
    on_target = intersect(segment_negative_intervals(signal, trace, seg), inside)
    low = math.inf
    for lo, hi in inside:
        found = minimize_scalar(
            lambda t: float(margin(trace.state_in(seg, t))),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-9},
        )
        ends = min(float(margin(trace.state_in(seg, lo))), float(margin(trace.state_in(seg, hi))))
        low = min(low, float(found.fun), ends)
    return low, total(on_target)


def _exposure(trace: Trace, seg: Segment, direction, sigma, inside) -> float:
    """Integral over time of exp(-theta^2 / (2 sigma^2)) / r^2 inside the plume range (s / m^2)."""

    def density(t: float) -> float:
        state = trace.state_in(seg, t)
        theta = float(_angle(direction, state))
        return math.exp(-0.5 * (theta / sigma) ** 2) / float(state[:3] @ state[:3])

    return sum(quad(density, lo, hi, epsabs=0.0, epsrel=1e-10, limit=200)[0] for lo, hi in inside)
