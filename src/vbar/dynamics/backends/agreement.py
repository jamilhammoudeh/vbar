"""The agreement table: V-BAR's two-body plus J2 rung against independent propagators.

The same inertial target and chaser states are flown by V-BAR's relative Cowell integration and
by each backend; the backends' inertial states are turned into Hill states by the same
conversion, so the residuals measure the propagation alone. Each residual is attributed by
changing one setting at a time: the integrator tolerance of both sides, and, for a controlled
episode replayed open loop, the substep over which a backend holds a burn's inertial direction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..bodies import EARTH, Body
from ..models import TwoBodyDynamics
from .common import Burn, hill_states

Propagate = Callable[..., tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Residual:
    """Largest differences over an arc between a backend and V-BAR, Hill frame.

    Parameters
    ----------
    case : str
        What was flown.
    backend : str
        The backend's name.
    position : float
        Largest position difference (m).
    velocity : float
        Largest velocity difference (m/s).
    """

    case: str
    backend: str
    position: float
    velocity: float


def vbar_arc(
    rel: np.ndarray,
    times: np.ndarray,
    burns: Sequence[Burn] = (),
    body: Body = EARTH,
    altitude: float = 500e3,
    inclination: float = math.radians(51.6),
    rtol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V-BAR's Hill states at `times` from `rel`, and the initial inertial target and chaser.

    Burns start and end on sample times or between them; the arc is split at every edge.

    Parameters
    ----------
    rel : numpy.ndarray
        Relative start state (m, m/s), Hill frame, shape (6,).
    times : numpy.ndarray
        Sample times (s), increasing, from 0.
    burns : sequence of Burn
        Burns held constant in the Hill frame.
    body : Body
        Central body.
    altitude : float
        Target's circular-orbit altitude (m).
    inclination : float
        Target's orbit inclination (rad).
    rtol : float
        Integrator relative tolerance.

    Returns
    -------
    tuple of numpy.ndarray
        Hill states (len(times), 6), target and chaser inertial states at t = 0, each (6,), in
        the body's inertial frame (m, m/s).
    """
    model = TwoBodyDynamics(body, altitude, inclination, j2=True, rtol=rtol)
    model.reset(rel)
    start_t, start_c = model.target_state(), model.chaser_state()
    edges = sorted(
        {0.0, *times.tolist(), *(b.start for b in burns), *(b.start + b.duration for b in burns)}
    )
    out = {0.0: np.asarray(rel, dtype=float)}
    for a, b in zip(edges, edges[1:], strict=False):
        accel = np.zeros(3)
        for burn in burns:
            if burn.start <= a < burn.start + burn.duration:
                accel = np.array(burn.accel_hill)
        state, _ = model.advance(accel, b - a)
        out[b] = state
    return np.array([out[t] for t in times.tolist()]), start_t, start_c


def residual(case: str, name: str, ours: np.ndarray, theirs: np.ndarray) -> Residual:
    """The largest Hill position and velocity differences between two sampled arcs.

    Parameters
    ----------
    case : str
        Case label.
    name : str
        Backend name.
    ours, theirs : numpy.ndarray
        Hill states (m, m/s) on the same sample times, shape (k, 6).

    Returns
    -------
    Residual
        Largest position difference (m) and velocity difference (m/s).
    """
    diff = np.asarray(theirs) - np.asarray(ours)
    return Residual(
        case,
        name,
        float(np.max(np.linalg.norm(diff[:, :3], axis=1))),
        float(np.max(np.linalg.norm(diff[:, 3:], axis=1))),
    )


def coast_cases(ranges: Sequence[float] = (100.0, 1e3, 1e4), orbits: Sequence[int] = (1, 10)):
    """Relative start states for the coast arcs: range r along -V-bar with small offsets.

    Parameters
    ----------
    ranges : sequence of float
        Start ranges (m).
    orbits : sequence of int
        Arc lengths in target orbits.

    Returns
    -------
    list of tuple
        (label, relative state, number of orbits).
    """
    cases = []
    for r in ranges:
        rel = np.array([0.1 * r, -r, 0.05 * r, 0.0, 0.0, 1e-4 * r])
        for k in orbits:
            cases.append((f"coast {r:g} m, {k} orbit{'s' if k > 1 else ''}", rel, k))
    return cases


def coast_table(
    backends: dict[str, Propagate],
    cases=None,
    sample: float = 100.0,
    body: Body = EARTH,
    altitude: float = 500e3,
    inclination: float = math.radians(51.6),
) -> list[Residual]:
    """Coast residuals of every backend on every case.

    Parameters
    ----------
    backends : dict
        Name to a `propagate(target, chaser, times, body, j2, ...)` function.
    cases : list, optional
        From `coast_cases`; its default by default.
    sample : float
        Sampling interval (s), a multiple of every backend's step.
    body : Body
        Central body.
    altitude : float
        Target altitude (m).
    inclination : float
        Target inclination (rad).

    Returns
    -------
    list of Residual
        One per case and backend.
    """
    out = []
    n = math.sqrt(body.mu / (body.radius + altitude) ** 3)
    for label, rel, orbits in cases or coast_cases():
        span = sample * math.floor(orbits * 2.0 * math.pi / n / sample)
        times = np.arange(0.0, span + 0.5 * sample, sample)
        ours, start_t, start_c = vbar_arc(rel, times, (), body, altitude, inclination)
        for name, propagate in backends.items():
            target, chaser = propagate(start_t, start_c, times, body, True)
            out.append(residual(label, name, ours, hill_states(target, chaser, body, True)))
    return out


def episode_burns(records) -> list[Burn]:
    """The burns a recorded episode flew, from its step records' segments.

    Parameters
    ----------
    records : sequence of StepRecord
        A recorded episode's steps.

    Returns
    -------
    list of Burn
        One per thrusting segment, in time order.
    """
    burns = []
    for record in records:
        for start, duration, accel in record.segment.segments:
            if duration > 0.0 and np.any(np.asarray(accel) != 0.0):
                ax, ay, az = (float(c) for c in accel)
                burns.append(Burn(float(start), float(duration), (ax, ay, az)))
    return burns


__all__ = ["Residual", "coast_cases", "coast_table", "episode_burns", "residual", "vbar_arc"]
