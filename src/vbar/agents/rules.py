"""A rule-based supervisor: the staged-approach discipline a flight rule book gives.

It stops an overspeeding chaser, inhibits thrust at the commit point of a docking, avoids a
collision with a CAM, and otherwise gives the go-ahead at each rung once the chaser has settled
there (Fehse 6.5).
"""

from __future__ import annotations

import math

import numpy as np

from .base import AgentView
from .supervision import Command, Rung, Situation
from .tracker import speed_cap


class RuleSupervisor:
    """Decides by the first of these rules that applies.

    1. In the commit phase of a docking, THRUST_INHIBIT if valid.
    2. Within 1.1 collision radii of the target and closing, CAM if valid. A docking port's keep-out
       sphere is entered on purpose, so the contact rung is exempt.
    3. In transfer, closing faster than `overspeed` times the speed cap: HOLD. Stopped and
       closing slower than that: GO.
    4. Holding, with the arrival held since `settle_s` seconds before: GO.
    5. Otherwise CONTINUE.

    Parameters
    ----------
    settle_s : float
        Time (s) the arrival at a rung must have been held before the go-ahead.
    overspeed : float
        Factor on the public speed cap above which the closing speed is too high.

    Raises
    ------
    ValueError
        If a parameter is not a positive finite number.
    """

    name = "rules"

    def __init__(self, *, settle_s: float = 60.0, overspeed: float = 1.2):
        for label, value in (("settle_s", settle_s), ("overspeed", overspeed)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be a number, got {value!r}")
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{label} must be positive and finite, got {value!r}")
        self.settle_s = float(settle_s)
        self.overspeed = float(overspeed)

    def reset(
        self, view: AgentView, ladder: tuple[Rung, ...], rng: np.random.Generator | None = None
    ) -> None:
        """Start an episode: read the keep-out radius and the task from the view."""
        self._view = view
        self._dock = view.goal.get("task") == "dock"
        self._radius = max(
            (float(c["radius"]) for c in view.constraints if c["type"].startswith("keep_out")),
            default=0.0,
        )
        self._since: float | None = None

    def decide(self, situation: Situation) -> Command:
        """The command for the situation."""
        mask, phase = situation.mask, situation.phase
        r, v = situation.obs[:3], situation.obs[3:6]
        rho = float(np.linalg.norm(r))
        closing = -float(r @ v) / rho if rho > 0.0 else 0.0
        kind = situation.ladder[situation.rung].kind
        held = phase == "holding" and mask[Command.GO]
        if not held:
            self._since = None
        if phase == "commit" and self._dock and mask[Command.THRUST_INHIBIT]:
            return Command.THRUST_INHIBIT
        if (
            self._radius > 0.0
            and rho < 1.1 * self._radius
            and closing > 0.0
            and kind != "contact"
            and mask[Command.CAM]
        ):
            return Command.CAM
        fast = closing > self.overspeed * speed_cap(self._view, rho)
        if phase == "transfer" and fast and mask[Command.HOLD]:
            return Command.HOLD
        if phase == "stopped" and not fast and mask[Command.GO]:
            return Command.GO
        if held:
            if self._since is None:
                self._since = situation.t
            elif situation.t - self._since >= self.settle_s:
                return Command.GO
        return Command.CONTINUE
