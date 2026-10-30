"""V-BAR's own reference pursuers for the pe1 game, written as KSPDG-format agents."""

from __future__ import annotations

import numpy as np

from .runner import BaseAgent


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
        if not (np.isfinite(burn) and burn >= 0.0):
            raise ValueError("burn must not be negative")
        self.burn = float(burn)

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
        values = (speed, slow_range, gain, burn)
        if not all(np.isfinite(values)) or min(values) <= 0.0:
            raise ValueError("speed, slow_range, gain and burn must be positive and finite")
        self.speed, self.slow_range, self.gain, self.burn = (float(x) for x in values)

    def get_action(self, observation):
        obs = np.asarray(observation, dtype=float)
        dr = obs[9:12] - obs[3:6]
        dv = obs[12:15] - obs[6:9]
        distance = np.linalg.norm(dr)
        wanted = dr / distance * self.speed * min(1.0, distance / self.slow_range)
        # The pursuer's velocity relative to the evader is -dv; drive it to `wanted`.
        throttle = np.clip(self.gain * (wanted + dv), -1.0, 1.0)
        return {"burn_vec": [*throttle.tolist(), self.burn], "ref_frame": 1, "vec_type": 0}


__all__ = ["LineOfSightPursuer", "VelocityMatchingPursuer"]
