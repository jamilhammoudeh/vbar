"""Minimum-fuel V-bar glideslope by linear programming (Ariba, Arzelier, Urbina, Louembet, 2016).

N impulses at equal times carry the chaser from rest at `start` to rest at `aim` along a line
parallel to the V-bar. Between impulses it coasts on Clohessy-Wiltshire arcs and must be back on
the line at each impulse, at an along-track position the program chooses. Each arc's radial
deviation from the line at its midpoint, where the deviation of a V-bar hump is largest, is
bounded, and the sum of the impulses' 1-norms is minimized (Ariba et al., IFAC-PapersOnLine 49(17),
2016, sec. 3).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from ..dynamics import cw_stm
from .planning import Impulse

__all__ = ["lp_glideslope"]

_IN_PLANE = [0, 1, 3, 4]


def lp_glideslope(
    n: float,
    start: ArrayLike,
    aim: ArrayLike,
    duration: float,
    pulses: int,
    bound: float,
) -> tuple[Impulse, ...]:
    """Impulses of the minimum-fuel V-bar glideslope from rest at `start` to rest at `aim`.

    Parameters
    ----------
    n : float
        Mean motion of the circular target orbit, rad/s, positive.
    start, aim : array_like
        Positions (m, Hill frame) on one line parallel to the V-bar: the same radial (x)
        component and z = 0. The chaser is at rest at `start` and stops at `aim`.
    duration : float
        Transfer time, s, positive.
    pulses : int
        Number of coasting arcs N, at least 1; impulses fall at k duration / N for k = 0 to N.
    bound : float
        Largest radial deviation from the line allowed at each arc's midpoint, m, positive. It
        bounds the whole arc only while the arc, duration / pulses, is shorter than one orbit,
        which is required.

    Returns
    -------
    tuple of Impulse
        N + 1 impulses (Hill frame, m/s), the last cancelling the arrival velocity.

    Raises
    ------
    ValueError
        If an input is out of range, an arc is an orbit or longer, the points are not on a V-bar
        line, or no impulses meet the bound.
    """
    start = np.asarray(start, dtype=float)
    aim = np.asarray(aim, dtype=float)
    if start.shape != (3,) or aim.shape != (3,):
        raise ValueError("start and aim must have 3 entries")
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(aim))):
        raise ValueError("start and aim must be finite")
    if start[0] != aim[0] or start[2] != 0.0 or aim[2] != 0.0:
        raise ValueError("start and aim must lie on one line parallel to the V-bar, at z = 0")
    if isinstance(n, bool):
        raise ValueError("n must be a number")
    for name, value in (("n", n), ("duration", duration), ("bound", bound)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite")
    if isinstance(pulses, bool) or not isinstance(pulses, int | np.integer) or pulses < 1:
        raise ValueError("pulses must be an integer of at least 1")
    step = duration / pulses
    # A V-bar hump that starts and ends on the line peaks at mid-arc only while the arc is
    # shorter than an orbit; beyond that the midpoint bound no longer bounds the hump.
    if not n * step < 2.0 * math.pi:
        raise ValueError("each arc, duration / pulses, must be shorter than one orbit, 2 pi / n")
    phi = cw_stm(n, step)[np.ix_(_IN_PLANE, _IN_PLANE)]
    rr, rv, vr, vv = phi[:2, :2], phi[:2, 2:], phi[2:, :2], phi[2:, 2:]
    half = cw_stm(n, 0.5 * step)[np.ix_(_IN_PLANE, _IN_PLANE)]
    mid_r, mid_v = half[0, :2], half[0, 2:]
    x0 = start[0]
    # Variables: along-track positions y_1..y_{N-1}, velocities v_0..v_{N-1} after each impulse,
    # the last impulse, then the absolute values of every impulse component.
    n_y, n_v = pulses - 1, 2 * pulses
    n_z = n_y + n_v + 2
    size = n_z + 2 * (pulses + 1)

    def position(k: int) -> tuple[np.ndarray, np.ndarray]:
        rows = np.zeros((2, size))
        fixed = np.zeros(2)
        if k == 0:
            fixed[:] = start[:2]
        elif k == pulses:
            fixed[:] = aim[:2]
        else:
            fixed[0] = x0
            rows[1, k - 1] = 1.0
        return rows, fixed

    def velocity(k: int) -> np.ndarray:
        rows = np.zeros((2, size))
        rows[0, n_y + 2 * k] = rows[1, n_y + 2 * k + 1] = 1.0
        return rows

    last = np.zeros((2, size))
    last[0, n_z - 2] = last[1, n_z - 1] = 1.0
    a_eq, b_eq, impulses = [], [], []
    for k in range(pulses):
        rows, fixed = position(k)
        rows_next, fixed_next = position(k + 1)
        a_eq.append(rows_next - rr @ rows - rv @ velocity(k))
        b_eq.append(rr @ fixed - fixed_next)
        if k == 0:
            impulses.append((velocity(0), np.zeros(2)))
        else:
            rows_before, fixed_before = position(k - 1)
            impulses.append(
                (velocity(k) - vr @ rows_before - vv @ velocity(k - 1), -vr @ fixed_before)
            )
    rows_before, fixed_before = position(pulses - 1)
    # The last impulse leaves the chaser at rest.
    a_eq.append(vr @ rows_before + vv @ velocity(pulses - 1) + last)
    b_eq.append(-vr @ fixed_before)
    impulses.append((last, np.zeros(2)))
    a_ub, b_ub = [], []
    for k, (rows, fixed) in enumerate(impulses):
        magnitude = np.zeros((2, size))
        magnitude[0, n_z + 2 * k] = magnitude[1, n_z + 2 * k + 1] = 1.0
        a_ub += [rows - magnitude, -rows - magnitude]
        b_ub += [-fixed, fixed]
    for k in range(pulses):
        rows, fixed = position(k)
        deviation = mid_r @ rows + mid_v @ velocity(k)
        offset = float(mid_r @ fixed) - x0
        a_ub += [deviation[None], -deviation[None]]
        b_ub += [np.array([bound - offset]), np.array([bound + offset])]
    cost = np.zeros(size)
    cost[n_z:] = 1.0
    solution = linprog(
        cost,
        A_ub=np.vstack(a_ub),
        b_ub=np.concatenate(b_ub),
        A_eq=np.vstack(a_eq),
        b_eq=np.concatenate(b_eq),
        bounds=[(None, None)] * size,
        method="highs",
    )
    if solution.status != 0:
        raise ValueError(f"no glideslope meets the bound: {solution.message}")
    plan = []
    for k, (rows, fixed) in enumerate(impulses):
        dv = rows @ solution.x + fixed
        plan.append(Impulse(k * step, np.array([dv[0], dv[1], 0.0])))
    return tuple(plan)
