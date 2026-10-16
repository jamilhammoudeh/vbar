"""Keplerian two-body motion: Kepler's equation, classical elements and analytic propagation.

Classical elements are [a, e, i, raan, argp, M]: semi-major axis (m), eccentricity,
inclination, right ascension of the ascending node, argument of perigee and mean anomaly (rad),
for elliptic orbits in a body-centred inertial frame with z along the spin axis. On circular or
equatorial orbits only the sums argp + M and raan + argp + M are well defined.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def solve_kepler(M: float, e: float) -> float:
    """Eccentric anomaly E (rad) with E - e sin E = M, for 0 <= e < 1.

    E is returned in the same revolution as M.
    """
    M_rev = M % math.tau
    # Starting from pi keeps Newton's method from overshooting at high eccentricity.
    E = M_rev if e < 0.8 else math.pi
    for _ in range(50):
        dE = (E - e * math.sin(E) - M_rev) / (1.0 - e * math.cos(E))
        E -= dE
        # Convergence is quadratic, so after a correction this small the error is near 1e-24;
        # demanding less stalls on one-ulp oscillations.
        if abs(dE) < 1e-12:
            return E + (M - M_rev)
    raise RuntimeError("kepler's equation did not converge")


def elements_from_state(r: ArrayLike, v: ArrayLike, mu: float) -> np.ndarray:
    """Classical elements of the elliptic orbit through inertial (r, v)."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r)
    h = np.cross(r, v)
    h_hat = h / np.linalg.norm(h)
    a = 1.0 / (2.0 / r_norm - v @ v / mu)
    e_vec = np.cross(v, h) / mu - r / r_norm
    e = np.linalg.norm(e_vec)
    i = math.acos(np.clip(h_hat[2], -1.0, 1.0))
    node = np.array([-h[1], h[0], 0.0])
    node_norm = np.linalg.norm(node)
    # An equatorial orbit has no node line; measure angles from the x axis instead.
    node = node / node_norm if node_norm > 0.0 else np.array([1.0, 0.0, 0.0])
    ahead = np.cross(h_hat, node)
    raan = math.atan2(node[1], node[0])
    u = math.atan2(r @ ahead, r @ node)
    argp = math.atan2(e_vec @ ahead, e_vec @ node)
    f = u - argp
    E = math.atan2(math.sqrt(1.0 - e * e) * math.sin(f), e + math.cos(f))
    M = E - e * math.sin(E)
    return np.array([a, e, i, raan % math.tau, argp % math.tau, M % math.tau])


def state_from_elements(elements: ArrayLike, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Inertial position (m) and velocity (m/s) from classical elements."""
    a, e, i, raan, argp, M = (float(x) for x in np.asarray(elements, dtype=float))
    E = solve_kepler(M, e)
    cos_E, sin_E = math.cos(E), math.sin(E)
    eta = math.sqrt(1.0 - e * e)
    r_norm = a * (1.0 - e * cos_E)
    speed = math.sqrt(mu * a) / r_norm
    perifocal_r = (a * (cos_E - e), a * eta * sin_E)
    perifocal_v = (-speed * sin_E, speed * eta * cos_E)
    cO, sO, ci, si, cw, sw = (
        math.cos(raan),
        math.sin(raan),
        math.cos(i),
        math.sin(i),
        math.cos(argp),
        math.sin(argp),
    )
    p_hat = np.array([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si])
    q_hat = np.array([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si])
    r = perifocal_r[0] * p_hat + perifocal_r[1] * q_hat
    v = perifocal_v[0] * p_hat + perifocal_v[1] * q_hat
    return r, v


def _stumpff(z: float) -> tuple[float, float]:
    """Stumpff functions c2(z) = (1 - cos sqrt z) / z and c3(z) = (sqrt z - sin sqrt z) / z^1.5."""
    if abs(z) < 0.1:
        # Series through z^7; the closed form of c3 cancels catastrophically near z = 0.
        c2 = sum((-z) ** k / math.factorial(2 * k + 2) for k in range(8))
        c3 = sum((-z) ** k / math.factorial(2 * k + 3) for k in range(8))
        return c2, c3
    if z > 0.0:
        s = math.sqrt(z)
        return 2.0 * math.sin(0.5 * s) ** 2 / z, (s - math.sin(s)) / (z * s)
    s = math.sqrt(-z)
    return 2.0 * math.sinh(0.5 * s) ** 2 / -z, (math.sinh(s) - s) / (-z * s)


def lagrange_coefficients(
    r0: ArrayLike, v0: ArrayLike, t: float, mu: float
) -> tuple[float, float, float, float]:
    """Lagrange coefficients (f, g, f_dot, g_dot) after `t` s, from universal variables.

    r(t) = f r0 + g v0 and v(t) = f_dot r0 + g_dot v0 on any conic, including hyperbolas.
    Universal-variable form after Bate, Mueller and White, Fundamentals of Astrodynamics, ch. 4.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    r0_norm = float(np.linalg.norm(r0))
    sqrt_mu = math.sqrt(mu)
    sigma0 = float(r0 @ v0) / sqrt_mu
    alpha = 2.0 / r0_norm - float(v0 @ v0) / mu
    if alpha >= 0.0 or t == 0.0:
        chi = sqrt_mu * t * alpha
    else:
        # Starting value for hyperbolas from Vallado, Fundamentals of Astrodynamics, alg. 8.
        a = 1.0 / alpha
        direction = math.copysign(1.0, t)
        denominator = sigma0 * sqrt_mu + direction * math.sqrt(-mu * a) * (1.0 - r0_norm * alpha)
        chi = direction * math.sqrt(-a) * math.log(-2.0 * mu * alpha * t / denominator)
    for _ in range(50):
        z = alpha * chi * chi
        c2, c3 = _stumpff(z)
        r_norm = chi * chi * c2 + sigma0 * chi * (1.0 - z * c3) + r0_norm * (1.0 - z * c2)
        residual = (
            chi**3 * c3 + sigma0 * chi * chi * c2 + r0_norm * chi * (1.0 - z * c3) - sqrt_mu * t
        )
        # dF/dchi is the radius, which is positive, so the root is unique.
        d_chi = residual / r_norm
        chi -= d_chi
        if abs(d_chi) < 1e-12 * max(1.0, abs(chi)):
            break
    else:
        raise RuntimeError("universal kepler equation did not converge")
    z = alpha * chi * chi
    c2, c3 = _stumpff(z)
    r_norm = chi * chi * c2 + sigma0 * chi * (1.0 - z * c3) + r0_norm * (1.0 - z * c2)
    f = 1.0 - chi * chi * c2 / r0_norm
    g = t - chi**3 * c3 / sqrt_mu
    f_dot = sqrt_mu * chi * (z * c3 - 1.0) / (r_norm * r0_norm)
    g_dot = 1.0 - chi * chi * c2 / r_norm
    return f, g, f_dot, g_dot


def propagate_kepler(
    r0: ArrayLike, v0: ArrayLike, t: float, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity `t` s after (r0, v0) on the two-body conic."""
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    f, g, f_dot, g_dot = lagrange_coefficients(r0, v0, t, mu)
    return f * r0 + g * v0, f_dot * r0 + g_dot * v0
