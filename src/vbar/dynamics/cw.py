"""Clohessy-Wiltshire (CW) relative motion.

The CW equations describe how a "chaser" spacecraft moves relative to a
"target" spacecraft that is in a circular orbit, when the two are close
together (meters to a few kilometers).

Frame (Hill frame, centered on the target):
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

import math

import numpy as np
from scipy.integrate import solve_ivp

from .bodies import EARTH

MU_EARTH = EARTH.mu
R_EARTH = EARTH.radius


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

    This is the textbook closed form, kept as an independent check on the propagators. Its
    1 - cos(nt) terms lose about four digits when nt is below 1e-2; `cw_discrete` does not.
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


def cw_system(n: float) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time matrices of the CW equations, state' = A state + B accel."""
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 0] = 3.0 * n**2
    A[3, 4] = 2.0 * n
    A[4, 3] = -2.0 * n
    A[5, 2] = -(n**2)
    B = np.zeros((6, 3))
    B[3:6, :] = np.eye(3)
    return A, B


# Taylor coefficients of (x - sin x) / x^3 in powers of x^2, through x^15. Below |x| = 0.5 the
# truncation error is under 1e-18 relative, where the direct difference cancels.
_X_MINUS_SIN = [(-1) ** k / math.factorial(2 * k + 3) for k in range(7)]


def _x_minus_sin(x: np.ndarray) -> np.ndarray:
    x2 = x * x
    series = x * x2 * np.polyval(_X_MINUS_SIN[::-1], x2)
    return np.where(np.abs(x) < 0.5, series, x - np.sin(x))


def cw_discrete(n: float, dt: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretization, state(t + dt) = phi state(t) + gamma accel.

    Closed forms of phi = expm(A dt) and gamma = int_0^dt expm(A s) ds B, exact for an
    acceleration held constant in the Hill frame over `dt`. Written in terms of
    k = 1 - cos(n dt) and d = n dt - sin(n dt), both evaluated without cancellation, so short
    steps keep full precision. An array of durations gives stacked matrices of shape
    (..., 6, 6) and (..., 6, 3).
    """
    t = np.asarray(dt, dtype=float)
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)
    k = 2.0 * np.sin(0.5 * nt) ** 2
    d = _x_minus_sin(nt)
    o, i = np.zeros_like(t), np.ones_like(t)
    phi = np.array(
        [
            [1 + 3 * k, o, o, s / n, 2 * k / n, o],
            [-6 * d, i, o, -2 * k / n, (nt - 4 * d) / n, o],
            [o, o, c, o, o, s / n],
            [3 * n * s, o, o, c, 2 * s, o],
            [-6 * n * k, o, o, -2 * s, 1 - 4 * k, o],
            [o, o, -n * s, o, o, c],
        ]
    )
    gamma = np.array(
        [
            [k / n**2, 2 * d / n**2, o],
            [-2 * d / n**2, 4 * k / n**2 - 1.5 * t**2, o],
            [o, o, k / n**2],
            [s / n, 2 * k / n, o],
            [-2 * k / n, (nt - 4 * d) / n, o],
            [o, o, s / n],
        ]
    )
    return np.moveaxis(phi, (0, 1), (-2, -1)), np.moveaxis(gamma, (0, 1), (-2, -1))


def cw_states(n: float, dt: np.ndarray, state: np.ndarray, accel: np.ndarray) -> np.ndarray:
    """phi(dt) state + gamma(dt) accel row by row, without forming the matrices.

    `dt` is (A,), `state` (A, 6) and `accel` (A, 3); returns (A, 6). The same closed form as
    `cw_discrete`, for many different durations at once.
    """
    t = np.asarray(dt, dtype=float)
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)
    k = 2.0 * np.sin(0.5 * nt) ** 2
    d = _x_minus_sin(nt)
    x, y, z, vx, vy, vz = np.moveaxis(state, -1, 0)
    ax, ay, az = np.moveaxis(accel, -1, 0)
    n2 = n * n
    return np.stack(
        [
            (1 + 3 * k) * x + s / n * vx + 2 * k / n * vy + k / n2 * ax + 2 * d / n2 * ay,
            -6 * d * x
            + y
            - 2 * k / n * vx
            + (nt - 4 * d) / n * vy
            - 2 * d / n2 * ax
            + (4 * k / n2 - 1.5 * t * t) * ay,
            c * z + s / n * vz + k / n2 * az,
            3 * n * s * x + c * vx + 2 * s * vy + s / n * ax + 2 * k / n * ay,
            -6 * n * k * x - 2 * s * vx + (1 - 4 * k) * vy - 2 * k / n * ax + (nt - 4 * d) / n * ay,
            -n * s * z + c * vz + s / n * az,
        ],
        axis=-1,
    )


class CWDiscretizer:
    """Caches exact CW transition matrices by duration, keyed to the nanosecond."""

    # Arbitrary durations (latencies, refinement) would otherwise grow the cache without bound.
    max_entries = 4096

    def __init__(self, n: float):
        self.n = n
        self._matrices: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._samples: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}

    def matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """(phi, gamma) for an acceleration held `dt` seconds."""
        key = round(float(dt), 9)
        if key not in self._matrices:
            if len(self._matrices) >= self.max_entries:
                self._matrices.clear()
            self._matrices[key] = self._discrete(key)
        return self._matrices[key]

    def sample_matrices(self, dt: float, m: int) -> tuple[np.ndarray, np.ndarray]:
        """Stacked (phi, gamma) at `m` evenly spaced times in [0, dt], endpoints included.

        Shapes (m, 6, 6) and (m, 6, 3), so all sub-step states come from one batched product.
        """
        key = (round(float(dt), 9), int(m))
        if key not in self._samples:
            if len(self._samples) >= self.max_entries:
                self._samples.clear()
            self._samples[key] = self._discrete(np.linspace(0.0, key[0], key[1]))
        return self._samples[key]

    def _discrete(self, dt: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cw_discrete(self.n, dt)


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
