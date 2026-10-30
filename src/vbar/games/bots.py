"""V-BAR's evader bots for the pe1 pursuit-evasion game.

Each bot is V-BAR's version of one of KSPDG's public pe1 evaders, written from their source
(github.com/mit-ll/spacegym-kspdg, v0.15.0, commit 6251ca3, `src/kspdg/pe1/e1_envs.py` to
`e4_envs.py`, MIT licence) and copying none of it. KSPDG flies its bots through the game's
autopilot in real time; these fly the same rules on V-BAR's two-body model, so their pointing is
exact and immediate and their timing is simulated time. The differences are listed under Limits
in `docs/pursuit-evasion.md`.

A bot flies a sequence of segments. In a segment its body-axis throttles are constant and its
forward axis follows a named direction; the segment ends at a mission time or where a function
of the two vessels' states crosses zero, and the bot then chooses the next one. The game
integrates across a segment without a break, so a bot's decisions are found to the
integrator's accuracy rather than at step ends.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .vessels import _number

Switch = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
POINTINGS = ("prograde", "normal")


@dataclass(frozen=True, kw_only=True)
class Segment:
    """A stretch of an evader's flight with constant body-axis throttles.

    Parameters
    ----------
    throttle : tuple of float
        Throttles along the evader's forward, right and down axes, each in [-1, 1].
    pointing : str
        Where the forward axis points: "prograde" (along the evader's inertial velocity) or
        "normal" (along its orbit's angular momentum). Down is the part of the direction to the
        central body perpendicular to forward.
    until : float
        Mission time (s) at which the segment ends; infinite for none.
    triggers : tuple of (function, int)
        Each `(g, direction)` also ends the segment where `g(r_p, v_p, r_e, v_e)` crosses zero
        rising (+1) or falling (-1); the states are the pursuer's and the evader's inertial
        positions (m) and velocities (m/s).
    """

    throttle: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pointing: str = "prograde"
    until: float = math.inf
    triggers: tuple[tuple[Switch, int], ...] = ()

    def __post_init__(self):
        throttle = np.asarray(self.throttle, dtype=float)
        if throttle.shape != (3,) or not np.all(np.abs(throttle) <= 1.0):
            raise ValueError("throttle must be three numbers in [-1, 1]")
        object.__setattr__(self, "throttle", tuple(float(x) for x in throttle))
        if self.pointing not in POINTINGS:
            raise ValueError(f"pointing must be one of {list(POINTINGS)}")
        if math.isnan(self.until):
            raise ValueError("until must be a time or infinite")
        for g, direction in self.triggers:
            if not callable(g) or direction not in (-1, 1):
                raise ValueError("a trigger is a function and a direction of -1 or 1")


class EvaderBot(Protocol):
    """What the game needs from an evader bot.

    `segment` is called at the start of the episode (with `previous` None) and each time a
    segment ends, and returns the next segment. `cause` is the index in `previous.triggers` of
    the trigger that ended it, or None at the start and when its `until` time came. A bot
    should decide a switch from `cause`, not by testing its trigger functions again: at the
    located crossing they are zero to rounding, on either side. `rng` is the episode's evader
    stream, drawn from the episode seed; a bot that draws nothing ignores it.
    """

    name: str

    def segment(
        self,
        t: float,
        r_p: np.ndarray,
        v_p: np.ndarray,
        r_e: np.ndarray,
        v_e: np.ndarray,
        previous: Segment | None,
        cause: int | None,
        rng: np.random.Generator,
    ) -> Segment: ...


def _range(r_p, v_p, r_e, v_e) -> float:
    return float(np.linalg.norm(r_e - r_p))


@dataclass(frozen=True)
class PassiveEvader:
    """V-BAR's version of KSPDG's e1 evader: it never thrusts."""

    name: str = "vbar-passive-evader"

    def segment(self, t, r_p, v_p, r_e, v_e, previous, cause, rng) -> Segment:
        return Segment()


@dataclass(frozen=True, kw_only=True)
class NormalEscapeEvader:
    """V-BAR's version of KSPDG's e3 evader: orbit-normal thrust when the pursuer is close.

    The bot points forward along its orbit's angular momentum and fires `throttle` forward
    while the pursuer's range is below `evade_distance`, else nothing.

    Parameters
    ----------
    evade_distance : float
        Pursuer range (m) below which the bot thrusts; KSPDG's `evade_dist`.
    throttle : float
        Forward throttle in [0, 1] while evading; KSPDG's `evade_throttle`.
    """

    evade_distance: float = 200.0
    throttle: float = 1.0
    name: str = "vbar-normal-escape-evader"

    def __post_init__(self):
        _number(self.evade_distance, "evade_distance")
        if not _number(self.throttle, "throttle", zero=True) <= 1.0:
            raise ValueError("throttle must be at most 1")

    def _evade(self, r_p, v_p, r_e, v_e) -> float:
        return _range(r_p, v_p, r_e, v_e) - self.evade_distance

    def segment(self, t, r_p, v_p, r_e, v_e, previous, cause, rng) -> Segment:
        if previous is None or cause is None:
            evading = self._evade(r_p, v_p, r_e, v_e) < 0.0
        else:
            # The range fell (-1) or rose (+1) through the evasion distance.
            evading = previous.triggers[cause][1] == -1
        if evading:
            return Segment(
                throttle=(self.throttle, 0.0, 0.0), pointing="normal", triggers=((self._evade, 1),)
            )
        return Segment(pointing="normal", triggers=((self._evade, -1),))


__all__ = [
    "POINTINGS",
    "EvaderBot",
    "NormalEscapeEvader",
    "PassiveEvader",
    "Segment",
    "Switch",
]
