"""Two-impulse Clohessy-Wiltshire targeting in the Hill frame.

The transfer from r0 to rf in a time tf solves the position rows of the CW transition matrix,
rf = phi_rr r0 + phi_rv v0, for the departure velocity v0 (Fehse, Automated Rendezvous and
Docking of Spacecraft, section 3.3.2, eqs 3.41 to 3.43). The matrix phi_rv is singular in plane
at n tf = 2 k pi and at the other roots of 8 cos x + 3 x sin x = 8, and across the track at
n tf = k pi (Alfriend, Vadali, Gurfil, How and Breger, Spacecraft Formation Flying, p. 92).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from ..dynamics import cw_discrete

SINGULAR_SET = "n tf = 2 k pi, the roots of 8 cos x + 3 x sin x = 8, or k pi across the track"


def _finite(values: ArrayLike, size: int, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.shape != (size,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be {size} finite numbers")
    return out


def conditioning(n: float, tf: float) -> tuple[float, float]:
    """Sensitivity measures of CW targeting that vanish on its singular set.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    tf : float
        Transfer time (s).

    Returns
    -------
    g : float
        In-plane measure (8 - 8 cos x - 3 x sin x) / x^2 with x = n tf, the in-plane
        determinant of phi_rv scaled to 1 as x goes to 0. Zero at x = 2 k pi and at the other
        roots of 8 cos x + 3 x sin x = 8 (2.81346 pi, 4.89060 pi, ...); it lies in
        [-0.283, 1.628], with its peak near x = 2.96, and is not a normalized condition number.
    h : float
        Cross-track measure sin(x) / x, in [-0.218, 1], zero at x = k pi.

    Raises
    ------
    ValueError
        If `n` or `tf` is not a positive finite number.
    """
    if not (math.isfinite(n) and n > 0.0):
        raise ValueError("n must be a positive finite number")
    if not (math.isfinite(tf) and tf > 0.0):
        raise ValueError("tf must be a positive finite number")
    x = n * tf
    # 8 (1 - cos x) = 16 sin^2(x / 2) and the quotients are formed before squaring, so neither
    # the cancellation of 1 - cos x nor the underflow of x^2 reaches g at short transfers.
    half = math.sin(0.5 * x) / x
    sinc = math.sin(x) / x
    return 16.0 * half * half - 3.0 * sinc, sinc


def cw_lambert(
    n: float, r0: ArrayLike, rf: ArrayLike, tf: float, *, tol: float = 1e-2
) -> tuple[np.ndarray, np.ndarray]:
    """Velocities that carry a coasting chaser from `r0` to `rf` in `tf` under CW motion.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    r0, rf : array_like, shape (3,)
        Departure and arrival positions in the Hill frame (m).
    tf : float
        Transfer time (s).
    tol : float
        Transfers where either `conditioning` measure is below `tol` in magnitude are refused.

    Returns
    -------
    v0 : numpy.ndarray, shape (3,)
        Required departure velocity in the Hill frame (m/s).
    vf : numpy.ndarray, shape (3,)
        Arrival velocity in the Hill frame (m/s).

    Raises
    ------
    ValueError
        If `n` or `tf` is not a positive finite number, `tf` is within `tol` of the singular
        set, or `r0` or `rf` is not three finite numbers.
    """
    g, h = conditioning(n, tf)
    if abs(g) < tol or abs(h) < tol:
        raise ValueError(f"transfer time is near the singular set ({SINGULAR_SET})")
    r0, rf = _finite(r0, 3, "r0"), _finite(rf, 3, "rf")
    # The ZOH closed form keeps 1 - cos(n tf) free of cancellation, which the textbook matrix
    # loses at short transfers.
    phi, _ = cw_discrete(n, tf)
    phi_rr, phi_rv = phi[:3, :3], phi[:3, 3:]
    phi_vr, phi_vv = phi[3:, :3], phi[3:, 3:]
    v0 = np.linalg.solve(phi_rv, rf - phi_rr @ r0)
    return v0, phi_vr @ r0 + phi_vv @ v0


def two_impulse(
    n: float, state: ArrayLike, target: ArrayLike, tf: float, *, tol: float = 1e-2
) -> tuple[np.ndarray, np.ndarray]:
    """Two impulses that bring a chaser to rest at `target` after a coast of `tf`.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    state : array_like, shape (6,)
        Chaser state [x, y, z, vx, vy, vz] in the Hill frame (m, m/s).
    target : array_like, shape (3,)
        Arrival position in the Hill frame (m).
    tf : float
        Transfer time (s).
    tol : float
        Singular-set tolerance passed to `cw_lambert`.

    Returns
    -------
    dv1 : numpy.ndarray, shape (3,)
        Departure impulse in the Hill frame (m/s).
    dv2 : numpy.ndarray, shape (3,)
        Arrival impulse in the Hill frame (m/s), the negative of the arrival velocity.

    Raises
    ------
    ValueError
        If `n` or `tf` is not a positive finite number, `tf` is within `tol` of the singular
        set, or `state` or `target` is not six or three finite numbers.
    """
    state = _finite(state, 6, "state")
    v0, vf = cw_lambert(n, state[:3], target, tf, tol=tol)
    return v0 - state[3:6], -vf
