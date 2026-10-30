"""V-BAR's own evader bots for the pe1 pursuit-evasion game.

These bots are written for V-BAR from the public one-line descriptions of KSPDG's evaders
(github.com/mit-ll/spacegym-kspdg, `src/kspdg/pe1/README.md`). They are not KSPDG's bots, whose
behaviour beyond those descriptions is not reproduced, and results against them are not
comparable with results against KSPDG's.

A bot is in one of a few modes, held from one switch to the next. In each mode it thrusts a
continuous function of the two vessels' states, so the simulation integrates across a mode
without a break and stops only where the bot switches. An evader is assumed to turn its forward
axis to its thrust direction, so its thrust is limited by the forward RCS thrust and its
propellant flow is the thrust over Isp g0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .vessels import _number

Switch = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]


class EvaderBot(Protocol):
    """What the game needs from an evader bot.

    `mode` picks the mode at the start of the game; `force` is the inertial thrust (N) in a
    mode given the pursuer's and the evader's inertial positions and velocities; `switch`
    returns, for a mode, a function of the same states whose zero crossing in `direction` ends
    the mode, with the mode that follows, or None if the mode never ends.
    """

    name: str

    def mode(self, r_p: np.ndarray, v_p: np.ndarray, r_e: np.ndarray, v_e: np.ndarray) -> int: ...

    def force(
        self, mode: int, r_p: np.ndarray, v_p: np.ndarray, r_e: np.ndarray, v_e: np.ndarray
    ) -> np.ndarray: ...

    def switch(self, mode: int) -> tuple[Switch, int, int] | None: ...


@dataclass(frozen=True)
class PassiveEvader:
    """V-BAR's passive evader: it never thrusts (KSPDG's e1 is described as "passive")."""

    name: str = "vbar-passive-evader"

    def mode(self, r_p, v_p, r_e, v_e) -> int:
        return 0

    def force(self, mode, r_p, v_p, r_e, v_e) -> np.ndarray:
        return np.zeros(3)

    def switch(self, mode):
        return None


@dataclass(frozen=True, kw_only=True)
class NormalEscapeEvader:
    """V-BAR's heuristic evader: full thrust along its orbit normal while the pursuer is close.

    Written for V-BAR from KSPDG's public description of its e3 evader ("heuristic evasion
    maneuvers" that use the pursuer's relative position, with a 200 m evasion distance and full
    throttle). Below `evade_distance` the bot thrusts `thrust` along the angular momentum of its
    own orbit; at or beyond it, it coasts.

    Parameters
    ----------
    evade_distance : float
        Pursuer range (m) below which the bot thrusts.
    thrust : float
        Thrust (N) while evading.
    """

    evade_distance: float = 200.0
    thrust: float = 8000.0
    name: str = "vbar-normal-escape-evader"

    def __post_init__(self):
        _number(self.evade_distance, "evade_distance")
        _number(self.thrust, "thrust", zero=True)

    def _range(self, r_p, v_p, r_e, v_e) -> float:
        return float(np.linalg.norm(r_e - r_p)) - self.evade_distance

    def mode(self, r_p, v_p, r_e, v_e) -> int:
        return 1 if self._range(r_p, v_p, r_e, v_e) < 0.0 else 0

    def force(self, mode, r_p, v_p, r_e, v_e) -> np.ndarray:
        if mode == 0:
            return np.zeros(3)
        h = np.cross(r_e, v_e)
        return self.thrust * h / np.linalg.norm(h)

    def switch(self, mode):
        # Mode 0 ends when the range falls through the evasion distance, mode 1 when it rises.
        return (self._range, -1, 1) if mode == 0 else (self._range, 1, 0)


__all__ = ["EvaderBot", "NormalEscapeEvader", "PassiveEvader", "Switch"]
