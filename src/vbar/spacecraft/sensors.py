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
    meant to resemble any real sensor and is not valid inside about 20 m.
    """

    name = "gaussian"

    def __init__(
        self,
        sigma_pos: float = 1.0,
        sigma_vel: float = 1e-3,
        bias_pos: float = 0.0,
        bias_vel: float = 0.0,
    ):
        self.sigma = np.repeat([sigma_pos, sigma_vel], 3)
        self.bias_sigma = np.repeat([bias_pos, bias_vel], 3)
        self.bias = np.zeros(6)
        self.rng = np.random.default_rng()

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.bias = self.bias_sigma * rng.standard_normal(6)

    def observe(self, true_state: ArrayLike, t: float) -> np.ndarray:
        noise = self.sigma * self.rng.standard_normal(6)
        return np.asarray(true_state, dtype=float) + self.bias + noise


class RangeScaledNav:
    """White noise that grows with range, in the sensor frame.

    The position error has 1-sigma max(floor, k_range rho, c_range rho^2) along the line of sight
    and max(floor, k_lateral rho) across it; the defaults follow Fehse's rule that relative
    navigation is good to about 1% of range (Automated Rendezvous and Docking of Spacecraft, 2.4
    and 5.3.2), and `c_range` models camera ranging from pattern size. Velocity errors are
    isotropic with max(floor_vel, k_vel rho); those defaults are ours: 0.05 m/s at 500 m, as for
    relative GPS, and 1 mm/s at 10 m.
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
    ):
        self.k_range, self.k_lateral, self.floor, self.c_range = k_range, k_lateral, floor, c_range
        self.k_vel, self.floor_vel = k_vel, floor_vel
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
        draw = self.rng.standard_normal(3)
        noise = across * draw
        if rho > 0.0:
            los = state[:3] / rho
            noise += (along - across) * (draw @ los) * los
        return state + np.r_[noise, sigma_vel * self.rng.standard_normal(3)]


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
        w = sigma_vel / sigma_pos
        self.system = np.array([[0.0, 1.0], [-(w**2), -2.0 * damping * w]])
        self.cov = np.diag([sigma_pos**2, sigma_vel**2])
        self.rng = np.random.default_rng()
        self.error = np.zeros((2, 3))
        self.t = 0.0

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.error = np.linalg.cholesky(self.cov) @ rng.standard_normal((2, 3))
        self.t = 0.0

    def observe(self, true_state: ArrayLike, t: float) -> np.ndarray:
        if t > self.t:
            transition = expm(self.system * (t - self.t))
            # Stationary process: the noise added over the interval restores the covariance.
            added = self.cov - transition @ self.cov @ transition.T
            draw = np.linalg.cholesky(added + 1e-30 * np.eye(2)) @ self.rng.standard_normal((2, 3))
            self.error = transition @ self.error + draw
            self.t = t
        return np.asarray(true_state, dtype=float) + self.error.ravel()
