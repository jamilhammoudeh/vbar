"""Navigation sensors: the chaser's estimate of its state relative to the target.

Agents see only sensor output; the true state stays with the environment and the metrics. A
sensor returns the relative state [x, y, z, vx, vy, vz] in the Hill frame (m, m/s) as the
navigation system reports it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


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
