"""Clohessy-Wiltshire (CW) relative motion.

The CW equations describe how a "chaser" spacecraft moves relative to a
"target" spacecraft that is in a circular orbit, when the two are close
together (meters to a few kilometers).

Frame (Hill / LVLH frame, centered on the target):
    x : radial, pointing away from Earth (the "R-bar")
    y : along-track, the direction the target is flying (the "V-bar")
    z : cross-track, completing a right-handed frame (orbit normal)

State vector: [x, y, z, vx, vy, vz] in meters and meters per second.

Equations of motion (n = mean motion of the target orbit, a = thrust accel):

    x'' =  3 n^2 x + 2 n y' + a_x
    y'' = -2 n x'           + a_y
    z'' = -n^2 z            + a_z

Assumptions:
    1. Target orbit is circular.
    2. Chaser is close to the target compared with the orbit radius
       (the equations come from linearizing two-body gravity).
    3. No perturbations: no J2, drag, or solar pressure.

Reference: H. Curtis, "Orbital Mechanics for Engineering Students", ch. 7.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6_378_137.0  # m


def mean_motion(semi_major_axis_m: float, mu: float = MU_EARTH) -> float:
    """Mean motion n = sqrt(mu / a^3), in rad/s."""
    return float(np.sqrt(mu / semi_major_axis_m**3))


def cw_rhs(t: float, state: np.ndarray, n: float, accel: np.ndarray) -> np.ndarray:
    """Time derivative of the state under the CW equations.

    `accel` is a constant thrust acceleration [a_x, a_y, a_z] in m/s^2.
    """
    x, y, z, vx, vy, vz = state
    ax, ay, az = accel
    return np.array(
        [
            vx,
            vy,
            vz,
            3.0 * n**2 * x + 2.0 * n * vy + ax,
            -2.0 * n * vx + ay,
            -(n**2) * z + az,
        ]
    )


def cw_stm(n: float, t: float) -> np.ndarray:
    """Exact 6x6 state transition matrix (no thrust).

    state(t) = cw_stm(n, t) @ state(0)

    This is the closed-form solution of the CW equations. The simulator
    integrates the equations numerically; this matrix is used to check
    that the numerical integration is correct.
    """
    s, c = np.sin(n * t), np.cos(n * t)
    nt = n * t
    return np.array(
        [
            [4 - 3 * c, 0, 0, s / n, 2 * (1 - c) / n, 0],
            [6 * (s - nt), 1, 0, -2 * (1 - c) / n, (4 * s - 3 * nt) / n, 0],
            [0, 0, c, 0, 0, s / n],
            [3 * n * s, 0, 0, c, 2 * s, 0],
            [-6 * n * (1 - c), 0, 0, -2 * s, 4 * c - 3, 0],
            [0, 0, -n * s, 0, 0, c],
        ]
    )


def propagate(
    state: np.ndarray,
    n: float,
    dt: float,
    accel: np.ndarray | None = None,
    n_samples: int = 0,
    rtol: float = 1e-10,
    atol: float = 1e-9,
):
    """Numerically propagate the state for `dt` seconds with constant thrust.

    Returns the final state. If `n_samples > 0`, also returns an array of
    intermediate states (n_samples, 6), used to check keep-out violations
    *between* time steps, not only at the end of each step.
    """
    if accel is None:
        accel = np.zeros(3)
    sol = solve_ivp(
        cw_rhs,
        (0.0, dt),
        np.asarray(state, dtype=float),
        args=(n, np.asarray(accel, dtype=float)),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=n_samples > 0,
    )
    if not sol.success:  # pragma: no cover - solver failures are unexpected
        raise RuntimeError(f"Integration failed: {sol.message}")
    final = sol.y[:, -1]
    if n_samples > 0:
        ts = np.linspace(0.0, dt, n_samples)
        return final, sol.sol(ts).T
    return final
