"""First-order J2 mean elements: secular rates and the mean-osculating map.

The map is Brouwer's first-order theory in Lyddane's nonsingular form, as given by Schaub and
Junkins, Analytical Mechanics of Space Systems, appendix on mean and osculating elements.
Elements are classical [a, e, i, raan, argp, M] as in `vbar.dynamics.kepler`. The map is
first order in J2: a round trip through it leaves errors of order J2^2 (metres in a in LEO),
so compare mean elements that were computed the same way.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .bodies import Body
from .kepler import solve_kepler

# Smallest |1 - 5 cos^2 i| used. The first-order eccentricity terms are resonant at the critical
# inclination (63.43 deg), where this factor vanishes; clamping it within about 0.7 deg of
# that inclination keeps them finite (Alfriend et al., Spacecraft Formation Flying, sec. 7.3).
# Every clamped term carries a factor e, so circular orbits are mapped exactly.
_CRITICAL_GUARD = 0.05


def j2_secular_rates(a: float, e: float, i: float, body: Body) -> tuple[float, float, float]:
    """First-order secular rates (rad/s) of raan, argp and M for mean elements a, e, i."""
    n = math.sqrt(body.mu / a**3)
    eta = math.sqrt(1.0 - e * e)
    k = 0.75 * n * body.j2 * (body.radius / (a * eta * eta)) ** 2
    c2 = math.cos(i) ** 2
    return -2.0 * k * math.cos(i), k * (5.0 * c2 - 1.0), n + k * eta * (3.0 * c2 - 1.0)


def mean_to_osculating(elements: ArrayLike, body: Body) -> np.ndarray:
    """Osculating elements of the orbit with first-order J2 mean elements `elements`."""
    return _brouwer_lyddane(np.asarray(elements, dtype=float), body, 1.0)


def osculating_to_mean(elements: ArrayLike, body: Body) -> np.ndarray:
    """First-order J2 mean elements of the osculating elements `elements`."""
    return _brouwer_lyddane(np.asarray(elements, dtype=float), body, -1.0)


def _brouwer_lyddane(elements: np.ndarray, body: Body, sign: float) -> np.ndarray:
    a, e, i, raan, argp, M = (float(x) for x in elements)
    gamma2 = sign * 0.5 * body.j2 * (body.radius / a) ** 2
    eta = math.sqrt(1.0 - e * e)
    gamma2p = gamma2 / eta**4
    E = solve_kepler(M, e)
    f = math.atan2(eta * math.sin(E), math.cos(E) - e)
    a_r = (1.0 + e * math.cos(f)) / eta**2
    a_r_eta = a_r * eta
    c = math.cos(i)
    s = math.sin(i)
    c2 = c * c
    s2 = s * s
    k = 1.0 - 5.0 * c2
    if abs(k) < _CRITICAL_GUARD:
        k = math.copysign(_CRITICAL_GUARD, k)
    # 1 - 11 c^2 - 40 c^4 / k, factored so that its sin^2 i zero is explicit.
    k1 = s2 * (1.0 - 15.0 * c2) / k
    cf = math.cos(f)
    w2f = 2.0 * argp + 2.0 * f
    w1f = 2.0 * argp + f
    w3f = 2.0 * argp + 3.0 * f

    da = (
        a
        * gamma2
        * ((3.0 * c2 - 1.0) * (a_r**3 - 1.0 / eta**3) + 3.0 * s2 * a_r**3 * math.cos(w2f))
    )
    de1 = gamma2p / 8.0 * e * eta**2 * k1 * math.cos(2.0 * argp)
    cf_poly = 3.0 * cf + 3.0 * e * cf**2 + e * e * cf**3
    de = de1 + 0.5 * eta**2 * (
        gamma2
        * (
            (3.0 * c2 - 1.0) / eta**6 * (e * eta + e / (1.0 + eta) + cf_poly)
            + 3.0 * s2 / eta**6 * (e + cf_poly) * math.cos(w2f)
        )
        - gamma2p * s2 * (3.0 * math.cos(w1f) + math.cos(w3f))
    )
    # The book's -e de1 / (eta^2 tan i) with the sin^2 i in k1 cancelled against tan i.
    di = -gamma2p / 8.0 * e * e * s * c * (1.0 - 15.0 * c2) / k * math.cos(2.0 * argp)
    C = 3.0 * math.cos(w2f) + 3.0 * e * math.cos(w1f) + e * math.cos(w3f)
    S = 3.0 * math.sin(w2f) + 3.0 * e * math.sin(w1f) + e * math.sin(w3f)
    di += 0.5 * gamma2p * c * s * C
    # f - M must be wrapped, or whole revolutions leak into the long-period terms.
    equation_of_center = (f - M + math.pi) % math.tau - math.pi + e * math.sin(f)
    # The book's two O(1) terms in 1/k cancel to O(e^2); summed analytically so that they
    # stay finite at the critical inclination.
    dlam_e2 = (
        -gamma2p
        / 8.0
        * e
        * e
        * (
            k1 * (1.0 + eta + eta * eta) / (1.0 + eta)
            + 0.5 * (1.0 - 33.0 * c2 - 200.0 * c2 * c2 / k - 400.0 * c2**3 / k**2)
        )
    )
    raan_shift = -gamma2p / 8.0 * e * e * c * (11.0 + 80.0 * c2 / k + 200.0 * c2 * c2 / k**2)
    raan_shift -= 0.5 * gamma2p * c * (6.0 * equation_of_center - S)
    lam = (
        M
        + argp
        + raan
        + dlam_e2
        + 0.25 * gamma2p * (-6.0 * k * equation_of_center + (3.0 - 5.0 * c2) * S)
        + raan_shift
    )
    e_dM = gamma2p / 8.0 * e * eta**3 * k1 - 0.25 * gamma2p * eta**3 * (
        2.0 * (3.0 * c2 - 1.0) * (a_r_eta**2 + a_r + 1.0) * math.sin(f)
        + 3.0
        * s2
        * (
            (-(a_r_eta**2) - a_r + 1.0) * math.sin(w1f)
            + (a_r_eta**2 + a_r + 1.0 / 3.0) * math.sin(w3f)
        )
    )

    d1 = (e + de) * math.sin(M) + e_dM * math.cos(M)
    d2 = (e + de) * math.cos(M) - e_dM * math.sin(M)
    half_i_sin = math.sin(0.5 * i) + 0.5 * math.cos(0.5 * i) * di
    d3 = half_i_sin * math.sin(raan) + math.sin(0.5 * i) * raan_shift * math.cos(raan)
    d4 = half_i_sin * math.cos(raan) - math.sin(0.5 * i) * raan_shift * math.sin(raan)
    M_new = math.atan2(d1, d2)
    raan_new = math.atan2(d3, d4)
    return np.array(
        [
            a + da,
            math.hypot(d1, d2),
            2.0 * math.asin(min(1.0, math.hypot(d3, d4))),
            raan_new % math.tau,
            (lam - M_new - raan_new) % math.tau,
            M_new % math.tau,
        ]
    )
