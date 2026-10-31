"""Reference pursuers for the pe1 game, written as KSPDG-format agents.

`LineOfSightPursuer` and `VelocityMatchingPursuer` are V-BAR's. `ClosestApproachAgent` is V-BAR's
reimplementation of Jason Cheng's deterministic agent (github.com/ARCLab-MIT/kspdg, commit
a45e5c6, `arclab_mit/agents/extended_obs_agent/deterministic_agent.py` and `simulate.py`, MIT
licence); Cheng is an author of "Language Models are Spacecraft Operators" (Rodriguez-Fernandez,
Carrasco, Cheng, Scharf, Siew and Linares, arXiv 2404.00413). It predicts with V-BAR's Kepler
propagation, where the original uses poliastro.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..dynamics.bodies import KERBIN
from ..dynamics.kepler import propagate_kepler
from .runner import BaseAgent
from .vessels import _number


class LineOfSightPursuer(BaseAgent):
    """Full forward thrust toward the evader, in the body frame, for `burn` seconds at a time.

    The same strategy as KSPDG's example naive agent (`burn_vec` [1, 0, 0, 1] in the body
    frame): it closes fast and overshoots, so it is a floor, not a solution.

    Parameters
    ----------
    burn : float
        Burn duration per action (s).
    debug : bool
        Set the agent's logger to DEBUG level.
    """

    def __init__(self, burn: float = 1.0, debug: bool = False) -> None:
        super().__init__(debug=debug)
        self.burn = _number(burn, "burn", zero=True)

    def get_action(self, observation):
        return {"burn_vec": [1.0, 0.0, 0.0, self.burn], "ref_frame": 0, "vec_type": 0}


class VelocityMatchingPursuer(BaseAgent):
    """Closes on the evader along the line of sight at a commanded speed, then holds near it.

    The desired relative velocity is the line-of-sight unit vector times `speed` times
    min(1, range / `slow_range`); the pursuer thrusts in the inertial frame along the error
    between that and its relative velocity, scaled by `gain` (throttle per m/s) and clipped.

    Parameters
    ----------
    speed : float
        Closing speed (m/s) far from the evader.
    slow_range : float
        Range (m) inside which the closing speed falls in proportion to range.
    gain : float
        Throttle per m/s of velocity error.
    burn : float
        Burn duration per action (s).
    debug : bool
        Set the agent's logger to DEBUG level.
    """

    def __init__(
        self,
        speed: float = 20.0,
        slow_range: float = 200.0,
        gain: float = 0.5,
        burn: float = 0.5,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.speed = _number(speed, "speed")
        self.slow_range = _number(slow_range, "slow_range")
        self.gain = _number(gain, "gain")
        self.burn = _number(burn, "burn")

    def get_action(self, observation):
        obs = np.asarray(observation, dtype=float)
        dr = obs[9:12] - obs[3:6]
        dv = obs[12:15] - obs[6:9]
        distance = np.linalg.norm(dr)
        wanted = dr / distance * self.speed * min(1.0, distance / self.slow_range)
        # The pursuer's velocity relative to the evader is -dv; drive it to `wanted`.
        throttle = np.clip(self.gain * (wanted + dv), -1.0, 1.0)
        return {"burn_vec": [*throttle.tolist(), self.burn], "ref_frame": 1, "vec_type": 0}


def closest_approach(
    state: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], max_time: int, mu: float = KERBIN.mu
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int]:
    """The predicted closest approach of two coasting vessels, sampled each second.

    Both vessels are propagated on their two-body conics to t = 0, 1, ..., `max_time` s and the
    first sample of least range is returned, as `closest_approach` in the original's
    `simulate.py` does; the true minimum may lie up to half a second from it.

    Parameters
    ----------
    state : tuple of array_like
        Positions (m) and velocities (m/s) `(r1, v1, r2, v2)` of the two vessels, inertial
        and relative to the central body.
    max_time : int
        Last sample time (s).
    mu : float
        Gravitational parameter (m^3/s^2); Kerbin's by default, as in the original.

    Returns
    -------
    closest_state : tuple of numpy.ndarray
        `(r1, v1, r2, v2)` at the closest sample.
    closest_time : int
        Its time (s).

    Raises
    ------
    ValueError
        If `max_time` is not a non-negative integer, `state` is not four finite 3-vectors or
        `mu` is not a positive finite number.
    """
    if isinstance(max_time, bool) or not isinstance(max_time, int | np.integer) or max_time < 0:
        raise ValueError("max_time must be a non-negative integer number of seconds")
    mu = _number(mu, "mu")
    try:
        r1, v1, r2, v2 = (np.asarray(x, dtype=float) for x in state)
    except (TypeError, ValueError):
        raise ValueError("state must be four 3-vectors (r1, v1, r2, v2)") from None
    if any(x.shape != (3,) or not np.all(np.isfinite(x)) for x in (r1, v1, r2, v2)):
        raise ValueError("state must be four finite 3-vectors (r1, v1, r2, v2)")
    best = (np.inf, 0, (r1, v1, r2, v2))
    for t in range(int(max_time) + 1):
        r1_t, v1_t = propagate_kepler(r1, v1, float(t), mu)
        r2_t, v2_t = propagate_kepler(r2, v2, float(t), mu)
        distance = float(np.linalg.norm(r1_t - r2_t))
        if distance < best[0]:
            best = (distance, t, (r1_t, v1_t, r2_t, v2_t))
    return best[2], best[1]


class ClosestApproachAgent(BaseAgent):
    """Steers on the predicted closest approach; a reimplementation of Jason Cheng's agent.

    Each call predicts the closest approach of the two coasting vessels over the next 300 s
    (`closest_approach`) and fires a 5 s inertial-frame throttle burn: 0.7 toward the evader
    when the approach is now or 80 s or more away; 0.7 along the predicted miss vector (from
    pursuer to evader at the approach) when it is 25 s to 80 s away; and 0.4 along the miss
    vector minus 0.8 along the predicted relative velocity (pursuer minus evader) when it is
    under 25 s away. The constants are the original's.

    Parameters
    ----------
    debug : bool
        Set the agent's logger to DEBUG level.
    """

    horizon = 300  # s
    burn = 5.0  # s

    def get_action(self, observation):
        obs = np.asarray(observation, dtype=float)
        r_p, v_p, r_e, v_e = obs[3:6], obs[6:9], obs[9:12], obs[12:15]
        (r1, v1, r2, v2), time = closest_approach((r_p, v_p, r_e, v_e), self.horizon)
        self.logger.debug("closest approach in %d s at %s", time, (r1, v1, r2, v2))
        # The arithmetic is in the original's order, so both round alike.
        if time <= 0.1 or time >= 80:
            throttle = 0.7 * ((r_e - r_p) / np.linalg.norm(r_p - r_e))
        else:
            miss = (r2 - r1) / np.linalg.norm(r1 - r2)
            if time >= 25:
                throttle = 0.7 * miss
            else:
                throttle = 0.4 * miss - 0.8 * ((v1 - v2) / np.linalg.norm(v1 - v2))
        return {"burn_vec": [*throttle.tolist(), self.burn], "ref_frame": 1}


__all__ = [
    "ClosestApproachAgent",
    "LineOfSightPursuer",
    "VelocityMatchingPursuer",
    "closest_approach",
]
