"""Flown orbits replayed through the relative-motion models.

Two spacecraft's precise orbits from flight (GRACE-FO Level-1B GNV1B, Earth-fixed) are rotated to a
quasi-inertial frame, the first sample of both starts a `TwoBodyDynamics` model, and the model's
relative state is compared with the flown one at every later sample.

The Earth-fixed states are turned inertial by the Earth's rotation alone, about the Earth-fixed z
axis at the WGS 84 rate from the Greenwich mean sidereal angle at the first sample. That frame's z
axis is the axis J2 is symmetric about, to the polar motion (about 1.5e-6 rad); precession and
nutation turn it against the stars at about 1e-11 rad/s, which over ten orbits is a fictitious
relative acceleration below 1e-8 m/s^2 at 220 km separation. The Sun direction for the density
bulge is the low-precision solar position of Montenbruck and Gill (Satellite Orbits, eqs 3.43 and
3.44, equinox of date), good to about 0.1 deg.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .conventions import inertial_to_hill
from .dynamics.bodies import EARTH, Body
from .dynamics.cw import cw_stm
from .dynamics.models import TwoBodyDynamics, gravity

__all__ = [
    "GPS_MINUS_UTC_S",
    "ReplayResult",
    "earth_fixed_to_inertial",
    "error_envelope",
    "gmst",
    "hill_states",
    "read_level1b",
    "read_orbit_excerpt",
    "replay",
    "sun_direction",
    "write_orbit_excerpt",
]

# GPS time runs ahead of UTC by the leap seconds since 1980: 18 s from 2017-01-01 on (IERS
# Bulletin C 52 onward); the value for an earlier epoch is the caller's to give.
GPS_MINUS_UTC_S = 18.0


def gmst(t_gps: ArrayLike, gps_minus_utc_s: float = GPS_MINUS_UTC_S) -> np.ndarray:
    """Greenwich mean sidereal angle (rad) at GPS seconds past 2000-01-01 12:00:00 GPS.

    The linear IAU 1982 expression 280.46061837 deg + 360.98564736629 deg per day of UT1 since
    J2000, with UT1 taken as UTC; UT1 - UTC stays below 0.9 s, a rotation below 7e-5 rad, which
    moves nothing a zonal gravity model or the density bulge resolves.

    Parameters
    ----------
    t_gps : array_like
        GPS seconds past 2000-01-01 12:00:00 GPS.
    gps_minus_utc_s : float
        Leap seconds between GPS time and UTC at the epoch, s.

    Returns
    -------
    numpy.ndarray
        Angle in [0, 2 pi), rad.
    """
    days = (np.asarray(t_gps, dtype=float) - gps_minus_utc_s) / 86400.0
    degrees = 280.46061837 + 360.98564736629 * days
    return np.radians(np.mod(degrees, 360.0))


def sun_direction(t_gps: float, gps_minus_utc_s: float = GPS_MINUS_UTC_S) -> np.ndarray:
    """Unit vector from the Earth to the Sun in the equator and equinox of date.

    Montenbruck and Gill, Satellite Orbits, eqs 3.43 and 3.44, with the ecliptic longitude moved
    from the equinox of J2000 to that of date by 1.3972 deg per Julian century; about 0.1 deg.

    Parameters
    ----------
    t_gps : float
        GPS seconds past 2000-01-01 12:00:00 GPS.
    gps_minus_utc_s : float
        Leap seconds between GPS time and UTC at the epoch, s; the formula's 0.1 deg does not
        notice the 69 s between UTC and TT.

    Returns
    -------
    numpy.ndarray
        Unit vector, shape (3,).
    """
    centuries = (t_gps - gps_minus_utc_s) / (86400.0 * 36525.0)
    mean_anomaly = math.radians(357.5256 + 35999.049 * centuries)
    longitude = math.radians(
        282.94
        + 1.3972 * centuries
        + math.degrees(mean_anomaly)
        + (6892.0 * math.sin(mean_anomaly) + 72.0 * math.sin(2.0 * mean_anomaly)) / 3600.0
    )
    obliquity = math.radians(23.43929111)
    return np.array(
        [
            math.cos(longitude),
            math.cos(obliquity) * math.sin(longitude),
            math.sin(obliquity) * math.sin(longitude),
        ]
    )


def earth_fixed_to_inertial(
    t: ArrayLike, r: ArrayLike, v: ArrayLike, theta0: float, body: Body = EARTH
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate Earth-fixed states to the frame that coincides with them at angle `theta0`.

    The inertial frame is the Earth-fixed one turned back by theta = theta0 + w t about z, with w
    the body's spin rate: r_i = Rz(theta) r and v_i = Rz(theta) (v + w x r).

    Parameters
    ----------
    t : array_like
        Times since the reference epoch, s, shape (k,).
    r, v : array_like
        Earth-fixed positions (m) and velocities (m/s), shape (k, 3).
    theta0 : float
        Earth rotation angle at t = 0, rad (`gmst` of the epoch).
    body : Body
        Central body, for its spin rate.

    Returns
    -------
    tuple of numpy.ndarray
        Inertial positions and velocities, shape (k, 3).
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    theta = theta0 + body.spin_rate * t
    c, s = np.cos(theta), np.sin(theta)
    w = np.array([0.0, 0.0, body.spin_rate])
    v_rot = v + np.cross(w, r)

    def turn(x: np.ndarray) -> np.ndarray:
        return np.stack([c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1], x[:, 2]], axis=1)

    return turn(r), turn(v_rot)


def hill_states(target: ArrayLike, chaser: ArrayLike, body: Body = EARTH) -> np.ndarray:
    """Chaser's Hill-frame state relative to the target from inertial states, shape (k, 6).

    The frame's rate about the radial axis uses point-mass plus J2 gravity at the target, the
    same for flown and modelled states, so the two are compared in one convention.
    """
    target = np.atleast_2d(np.asarray(target, dtype=float))
    chaser = np.atleast_2d(np.asarray(chaser, dtype=float))
    accel = gravity(target[:, :3], body, j2=True)
    return inertial_to_hill(
        target[:, :3],
        target[:, 3:],
        chaser[:, :3] - target[:, :3],
        chaser[:, 3:] - target[:, 3:],
        accel,
    )


@dataclass(frozen=True)
class ReplayResult:
    """Flown and modelled relative states at the samples, Hill frame, m and m/s."""

    t: np.ndarray
    flown: np.ndarray
    model: np.ndarray

    @property
    def position_error(self) -> np.ndarray:
        """Distance between modelled and flown relative positions, m."""
        return np.linalg.norm(self.model[:, :3] - self.flown[:, :3], axis=1)

    @property
    def velocity_error(self) -> np.ndarray:
        """Norm of the relative-velocity difference, m/s."""
        return np.linalg.norm(self.model[:, 3:] - self.flown[:, 3:], axis=1)


def replay(
    t: ArrayLike, target: ArrayLike, chaser: ArrayLike, dynamics: TwoBodyDynamics
) -> ReplayResult:
    """Start `dynamics` from the first flown sample and compare it with the rest.

    Parameters
    ----------
    t : array_like
        Sample times, s, increasing, shape (k,).
    target, chaser : array_like
        Flown inertial states [r, v] (m, m/s) in the dynamics model's frame, shape (k, 6).
    dynamics : TwoBodyDynamics
        The model; it is reset to the first sample and coasts between samples.

    Returns
    -------
    ReplayResult
        Flown and modelled relative states at each sample.

    Raises
    ------
    ValueError
        If the times do not increase or the shapes differ.
    """
    t = np.asarray(t, dtype=float)
    target = np.asarray(target, dtype=float)
    chaser = np.asarray(chaser, dtype=float)
    if target.shape != (len(t), 6) or chaser.shape != target.shape:
        raise ValueError("target and chaser must have shape (len(t), 6)")
    if len(t) < 2 or not np.all(np.diff(t) > 0.0):
        raise ValueError("t must hold at least two increasing times")
    dynamics.reset_inertial(target[0], chaser[0])
    model_target, model_chaser = [target[0]], [chaser[0]]
    for dt in np.diff(t):
        dynamics.advance(np.zeros(3), float(dt))
        model_target.append(dynamics.target_state())
        model_chaser.append(dynamics.chaser_state())
    body = dynamics.body
    return ReplayResult(
        t - t[0],
        hill_states(target, chaser, body),
        hill_states(np.array(model_target), np.array(model_chaser), body),
    )


def error_envelope(
    t: ArrayLike, n: float, per_orbit_m: float, accel_mps2: float, samples: int = 2001
) -> np.ndarray:
    """Upper envelope of relative position error: a linear term plus a bounded-force term.

    B(t) = per_orbit_m t n / (2 pi) + accel_mps2 int_0^t ||Phi_rv(s)|| ds, with Phi_rv the
    Clohessy-Wiltshire block mapping a velocity change to a later position (its largest singular
    value). The second term bounds the position error any acceleration no larger than
    `accel_mps2` builds up, to first order in the error; the first carries an error budget stated
    per orbit.

    Parameters
    ----------
    t : array_like
        Times since the start, s, not negative.
    n : float
        Mean motion, rad/s.
    per_orbit_m : float
        Error budget per orbit, m.
    accel_mps2 : float
        Bound on the unmodelled acceleration, m/s^2.
    samples : int
        Grid points of the trapezoid rule for the integral.

    Returns
    -------
    numpy.ndarray
        B at each time, m.
    """
    t = np.asarray(t, dtype=float)
    grid = np.linspace(0.0, float(np.max(t)), samples)
    norms = np.array([np.linalg.norm(cw_stm(n, s)[:3, 3:], 2) for s in grid])
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (norms[1:] + norms[:-1]) * np.diff(grid))])
    return per_orbit_m * t * n / (2.0 * math.pi) + accel_mps2 * np.interp(t, grid, integral)


def read_level1b(path: str | Path) -> dict[str, np.ndarray]:
    """Columns of a GRACE-FO Level-1B ASCII file, by the variable names its header lists.

    The file starts with a YAML header whose `variables` list names the columns in order and
    ends at the line "# End of YAML header"; numeric columns become float arrays and the rest
    string arrays (GRACE-FO Level-1 Data Product User Handbook, 2019, sec. 3.4).

    Parameters
    ----------
    path : str or pathlib.Path
        The file, for example `GNV1B_2019-12-03_C_04.txt`.

    Returns
    -------
    dict of str to numpy.ndarray
        One array per column.

    Raises
    ------
    ValueError
        If the header has no end line or a record has another number of columns.
    """
    names: list[str] = []
    rows: list[list[str]] = []
    in_variables = in_data = False
    with open(path, encoding="utf-8") as lines:
        for line in lines:
            if in_data:
                if line.strip():
                    rows.append(line.split())
                continue
            text = line.rstrip()
            if text.startswith("# End of YAML header"):
                in_data = True
            elif text.strip() == "variables:":
                in_variables = True
            elif in_variables and text.startswith("    - ") and text.endswith(":"):
                names.append(text.strip()[2:-1])
    if not in_data:
        raise ValueError(f"{path}: no end of the YAML header")
    if any(len(row) != len(names) for row in rows):
        raise ValueError(f"{path}: a record does not have {len(names)} columns")
    columns: dict[str, np.ndarray] = {}
    for k, name in enumerate(names):
        values = [row[k] for row in rows]
        try:
            columns[name] = np.array(values, dtype=float)
        except ValueError:
            columns[name] = np.array(values)
    return columns


def read_orbit_excerpt(path: str | Path) -> dict[str, np.ndarray]:
    """Read an excerpt written by `write_orbit_excerpt`.

    Parameters
    ----------
    path : str or pathlib.Path
        The excerpt.

    Returns
    -------
    dict of str to numpy.ndarray
        "t" (GPS seconds past 2000-01-01 12:00:00 GPS), "target" and "chaser" (Earth-fixed
        [r, v] in m and m/s, shape (k, 6)) and "range" (the corrected biased range, m).
    """
    table = np.loadtxt(path, comments="#", delimiter=",", ndmin=2)
    return {
        "t": table[:, 0],
        "target": table[:, 1:7],
        "chaser": table[:, 7:13],
        "range": table[:, 13],
    }


def write_orbit_excerpt(
    path: str | Path,
    t: ArrayLike,
    target: ArrayLike,
    chaser: ArrayLike,
    biased_range: ArrayLike,
    header: Sequence[str] = (),
) -> None:
    """Write sampled orbits and a range to a comma-separated excerpt with comment lines.

    Positions are written to 0.1 mm and velocities to 0.1 um/s, below the stated formal errors
    of GRACE-FO GNV1B (about 0.5 mm and 2 um/s).

    Parameters
    ----------
    path : str or pathlib.Path
        The file to write.
    t : array_like
        Times, shape (k,).
    target, chaser : array_like
        [r, v] in m and m/s, shape (k, 6).
    biased_range : array_like
        Range with its corrections applied, m, shape (k,).
    header : sequence of str
        Comment lines written first, without the leading "#".
    """
    t = np.asarray(t, dtype=float)
    target = np.asarray(target, dtype=float)
    chaser = np.asarray(chaser, dtype=float)
    biased_range = np.asarray(biased_range, dtype=float)
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        for line in header:
            out.write(f"# {line}\n")
        out.write("# t,tx,ty,tz,tvx,tvy,tvz,cx,cy,cz,cvx,cvy,cvz,range\n")
        for k in range(len(t)):
            fields = [f"{t[k]:.1f}"]
            for state in (target[k], chaser[k]):
                fields += [f"{x:.4f}" for x in state[:3]] + [f"{x:.7f}" for x in state[3:]]
            fields.append(f"{biased_range[k]:.5f}")
            out.write(",".join(fields) + "\n")
