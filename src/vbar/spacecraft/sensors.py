"""Navigation sensors: the chaser's estimate of its state relative to the target.

Agents see only sensor output; the true state stays with the environment and the metrics. A
sensor returns the relative state [x, y, z, vx, vy, vz] in the Hill frame (m, m/s) as the
navigation system reports it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm


def bounded_normal(rng: np.random.Generator, size: int, bound: float | None) -> np.ndarray:
    """Standard normal draws, drawn again while outside `bound` sigma when it is set.

    The truncation is by rejection, so the density inside the bound keeps its shape. A bound
    makes a Gaussian error model bounded, which a certificate for bounded disturbances needs;
    derive it from an episode-level exceedance probability with
    `vbar.scenarios.per_draw_level`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of the draws.
    size : int
        Number of draws.
    bound : float or None
        Truncation level in sigma, applied to each draw; None draws without truncation.

    Returns
    -------
    numpy.ndarray
        Shape (size,), in units of sigma.

    Raises
    ------
    ValueError
        If `bound` is set and not positive.
    """
    draw = rng.standard_normal(size)
    if bound is not None:
        if not bound > 0.0:
            raise ValueError("bound must be positive, in units of sigma")
        outside = np.abs(draw) > bound
        while np.any(outside):
            draw[outside] = rng.standard_normal(int(np.sum(outside)))
            outside = np.abs(draw) > bound
    return draw


def bounded_ball(rng: np.random.Generator, size: int, bound: float | None) -> np.ndarray:
    """A standard normal vector, drawn again while its norm exceeds `bound` when it is set.

    Truncating the norm rather than each component keeps the bound under any rotation, so a
    vector projected on a line of sight stays within `bound` along it and across it.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of the draws.
    size : int
        Dimension of the vector.
    bound : float or None
        Truncation level in sigma, applied to the vector's norm; None draws without truncation.

    Returns
    -------
    numpy.ndarray
        Shape (size,), in units of sigma.

    Raises
    ------
    ValueError
        If `bound` is set and not positive.
    """
    draw = rng.standard_normal(size)
    if bound is not None:
        if not bound > 0.0:
            raise ValueError("bound must be positive, in units of sigma")
        while float(np.linalg.norm(draw)) > bound:
            draw = rng.standard_normal(size)
    return draw


class Sensor(Protocol):
    name: str

    def reset(self, rng: np.random.Generator) -> None: ...

    def observe(self, true_state: np.ndarray, t: float) -> np.ndarray: ...


class PerfectNav:
    """Reports the true relative state."""

    name = "perfect"

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def observe(self, true_state: np.ndarray, t: float) -> np.ndarray:
        return np.array(true_state, dtype=float)


class GaussianNav:
    """White Gaussian noise on every axis plus a bias held for the episode.

    `sigma_pos` (m) and `sigma_vel` (m/s) are the 1-sigma noise per Hill axis, drawn afresh at
    each observation; `bias_pos` and `bias_vel` are the 1-sigma of a bias drawn once per
    episode. The defaults are the synthetic `:nav1` track (1 m, 1 mm/s, no bias), which is not
    meant to resemble any real sensor and is not valid inside about 20 m. `bound` (in sigma)
    truncates every draw of the noise and of the bias, each per axis, so the error on an axis
    never exceeds bound (sigma + bias sigma).

    Parameters
    ----------
    sigma_pos : float
        1-sigma position noise per Hill axis (m).
    sigma_vel : float
        1-sigma velocity noise per Hill axis (m/s).
    bias_pos : float
        1-sigma of the position bias per Hill axis, drawn at reset (m).
    bias_vel : float
        1-sigma of the velocity bias per Hill axis, drawn at reset (m/s).
    bound : float or None
        Truncation level in sigma for every noise and bias draw; None leaves them unbounded.

    Raises
    ------
    ValueError
        If a noise level is negative or `bound` is set and not positive.
    """

    name = "gaussian"

    def __init__(
        self,
        sigma_pos: float = 1.0,
        sigma_vel: float = 1e-3,
        bias_pos: float = 0.0,
        bias_vel: float = 0.0,
        bound: float | None = None,
    ):
        if min(sigma_pos, sigma_vel, bias_pos, bias_vel) < 0.0:
            raise ValueError("noise levels must not be negative")
        if bound is not None and not bound > 0.0:
            raise ValueError("bound must be positive, in units of sigma")
        self.sigma = np.repeat([sigma_pos, sigma_vel], 3)
        self.bias_sigma = np.repeat([bias_pos, bias_vel], 3)
        self.bound = bound
        self.bias = np.zeros(6)
        self.rng = np.random.default_rng()

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.bias = self.bias_sigma * bounded_normal(rng, 6, self.bound)

    def observe(self, true_state: ArrayLike, t: float) -> np.ndarray:
        noise = self.sigma * bounded_normal(self.rng, 6, self.bound)
        return np.asarray(true_state, dtype=float) + self.bias + noise


class RangeScaledNav:
    """White noise that grows with range, in the sensor frame.

    The position error has 1-sigma max(floor, k_range rho, c_range rho^2) along the line of sight
    and max(floor, k_lateral rho) across it; the defaults follow Fehse's rule that relative
    navigation is good to about 1% of range (Automated Rendezvous and Docking of Spacecraft, 2.4
    and 5.3.2), and `c_range` models camera ranging from pattern size. Velocity errors are
    isotropic with max(floor_vel, k_vel rho); the velocity defaults (0.05 m/s at 500 m, 1 mm/s
    at 10 m) are synthetic, not taken from a sensor. `bound` (in sigma) truncates the norm of
    each three-dimensional draw, so the position error never exceeds bound times its sigma along
    the line of sight nor across it, and the velocity error's norm never exceeds bound times its
    sigma.

    Parameters
    ----------
    k_range : float
        Line-of-sight position sigma per metre of range (m/m).
    k_lateral : float
        Position sigma across the line of sight per metre of range (m/m).
    floor : float
        Smallest position sigma, along and across (m).
    c_range : float
        Line-of-sight position sigma per square metre of range (1/m).
    k_vel : float
        Velocity sigma per metre of range (1/s).
    floor_vel : float
        Smallest velocity sigma (m/s).
    bound : float or None
        Truncation level in sigma for the norm of each three-dimensional draw; None leaves them
        unbounded.

    Raises
    ------
    ValueError
        If a coefficient is negative or `bound` is set and not positive.
    """

    name = "range_scaled"

    def __init__(
        self,
        k_range: float = 0.01,
        k_lateral: float = 0.01,
        floor: float = 0.01,
        c_range: float = 0.0,
        k_vel: float = 1e-4,
        floor_vel: float = 1e-4,
        bound: float | None = None,
    ):
        if min(k_range, k_lateral, floor, c_range, k_vel, floor_vel) < 0.0:
            raise ValueError("noise coefficients must not be negative")
        if bound is not None and not bound > 0.0:
            raise ValueError("bound must be positive, in units of sigma")
        self.k_range, self.k_lateral, self.floor, self.c_range = k_range, k_lateral, floor, c_range
        self.k_vel, self.floor_vel = k_vel, floor_vel
        self.bound = bound
        self.rng = np.random.default_rng()

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def sigmas(self, rho: float) -> tuple[float, float, float]:
        """1-sigma range, lateral (m) and velocity (m/s) errors at range `rho` (m)."""
        along = max(self.floor, self.k_range * rho, self.c_range * rho**2)
        across = max(self.floor, self.k_lateral * rho)
        return along, across, max(self.floor_vel, self.k_vel * rho)

    def observe(self, true_state: ArrayLike, t: float) -> np.ndarray:
        state = np.asarray(true_state, dtype=float)
        rho = float(np.linalg.norm(state[:3]))
        along, across, sigma_vel = self.sigmas(rho)
        draw = bounded_ball(self.rng, 3, self.bound)
        noise = across * draw
        if rho > 0.0:
            los = state[:3] / rho
            noise += (along - across) * (draw @ los) * los
        return state + np.r_[noise, sigma_vel * bounded_ball(self.rng, 3, self.bound)]


class GaussMarkovNav:
    """Time-correlated navigation error whose velocity part is the rate of its position part.

    Per Hill axis, the error (e, de/dt) is a stationary second-order Gauss-Markov process,
    e'' = -2 zeta w e' - w^2 e + white noise, with standard deviations `sigma_pos` (m) and
    `sigma_vel` (m/s), so w = sigma_vel / sigma_pos. Integrating the observed velocity therefore
    reproduces the observed position, and averaging over steps does not remove the error, as for
    a navigation filter's output. The defaults (4 cm, 0.4 mm/s, zeta = 1) are the CDGPS relative
    navigation levels at 1 km (Alfriend et al., Spacecraft Formation Flying, ch. 12: 0.2 to 2.5 cm
    and 0.2 to 0.5 mm/s at 100 m, 5 to 40 cm and 0.5 to 3 mm/s at 10 km). This is the `:nav2`
    track. The process is propagated exactly between observations.
    """

    name = "gauss_markov"

    def __init__(self, sigma_pos: float = 0.04, sigma_vel: float = 4e-4, damping: float = 1.0):
        if not (sigma_pos > 0.0 and sigma_vel > 0.0):
            raise ValueError("sigma_pos and sigma_vel must be positive")
        # Without damping the process is not stationary and its covariance never settles.
        if not damping > 0.0:
            raise ValueError("damping must be positive")
        w = sigma_vel / sigma_pos
        self.system = np.array([[0.0, 1.0], [-(w**2), -2.0 * damping * w]])
        self.cov = np.diag([sigma_pos**2, sigma_vel**2])
        self.intensity = np.diag([0.0, 4.0 * damping * w * sigma_vel**2])
        self.rng = np.random.default_rng()
        self.error = np.zeros((2, 3))
        self.t = 0.0

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.error = np.linalg.cholesky(self.cov) @ rng.standard_normal((2, 3))
        self.t = 0.0

    def observe(self, true_state: ArrayLike, t: float) -> np.ndarray:
        if t > self.t:
            transition, added = self._discretize(t - self.t)
            # eigh rather than Cholesky: on short steps the added covariance is nearly singular.
            values, vectors = np.linalg.eigh(added)
            factor = vectors * np.sqrt(np.clip(values, 0.0, None))
            self.error = transition @ self.error + factor @ self.rng.standard_normal((2, 3))
            self.t = t
        return np.asarray(true_state, dtype=float) + self.error.ravel()

    def _discretize(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Transition matrix and added covariance over `dt` s, by Van Loan's method.

        P - F P F^T gives the same covariance in exact arithmetic but cancels on short steps,
        where the added position variance is about q^2 dt^3 / 3 against a stationary sigma_pos^2
        (Van Loan, IEEE Transactions on Automatic Control 23, 1978).
        """
        A = self.system
        block = np.zeros((4, 4))
        block[:2, :2] = -A
        block[:2, 2:] = self.intensity
        block[2:, 2:] = A.T
        exponential = expm(block * dt)
        transition = exponential[2:, 2:].T
        added = transition @ exponential[:2, 2:]
        return transition, 0.5 * (added + added.T)
