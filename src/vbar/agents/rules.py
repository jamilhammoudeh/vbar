"""A rule-based supervisor: the staged-approach discipline a flight rule book gives.

It stops an overspeeding chaser, inhibits thrust at the commit point of a docking, avoids a
collision with a CAM or a retreat, and otherwise gives the go-ahead at each rung once the chaser
has settled there (Fehse 6.5).
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from vbar.dynamics import cw_stm

from .base import AgentView
from .supervision import Command, Rung, Situation
from .tracker import speed_cap

_SAMPLE_S = 5.0


class RuleSupervisor:
    """Decides by the first of these rules that applies.

    1. In the commit phase of a docking, THRUST_INHIBIT if valid.
    2. Outside the commit phase, on a rung that is not "contact" and farther out than the goal's
       commit radius, while closing: if the free drift of the observed state comes within the
       largest keep-out radius at any of the 5 s samples up to `warning_s`, and the deceleration
       closing^2 / (2 (range - radius)) needed to stop at the sphere exceeds `brake_margin` times
       `max_accel` (or the chaser is already inside the sphere): CAM if valid, else RETREAT if
       valid. The drift is propagated with the exact Clohessy-Wiltshire matrix at the view's mean
       motion.
    3. Under the same conditions, stopped or holding and closing faster than `overspeed` times the
       speed cap without having slowed since the previous consult, on `strikes` consecutive
       consults (a braking deficit): RETREAT if valid, else CAM if valid.
    4. Within 1.1 collision radii of the target and closing, CAM if valid. A docking port's keep-out
       sphere is entered on purpose, so the contact rung is exempt.
    5. In transfer, closing faster than `overspeed` times the speed cap: HOLD. Stopped and
       closing slower than that: GO.
    6. Holding, with the arrival held since `settle_s` seconds before: GO.
    7. Otherwise CONTINUE.

    Parameters
    ----------
    settle_s : float
        Time (s) the arrival at a rung must have been held before the go-ahead.
    overspeed : float
        Factor on the public speed cap above which the closing speed is too high.
    warning_s : float
        Look-ahead (s) of the predicted-collision rule.
    brake_margin : float
        Fraction of the view's `max_accel` that stopping before the keep-out sphere may need
        before a predicted collision is acted on.
    strikes : int
        Consecutive consults a braking deficit must last before the abort.

    Raises
    ------
    ValueError
        If `settle_s`, `overspeed`, `warning_s` or `brake_margin` is not a positive finite
        number, or `strikes` is not a positive integer.
    """

    name = "rules"

    def __init__(
        self,
        *,
        settle_s: float = 60.0,
        overspeed: float = 1.2,
        warning_s: float = 600.0,
        brake_margin: float = 0.5,
        strikes: int = 2,
    ):
        for label, value in (
            ("settle_s", settle_s),
            ("overspeed", overspeed),
            ("warning_s", warning_s),
            ("brake_margin", brake_margin),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be a number, got {value!r}")
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{label} must be positive and finite, got {value!r}")
        self.settle_s = float(settle_s)
        self.overspeed = float(overspeed)
        self.warning_s = float(warning_s)
        self.brake_margin = float(brake_margin)
        if isinstance(strikes, bool) or not isinstance(strikes, Integral) or strikes < 1:
            raise ValueError(f"strikes must be a positive integer, got {strikes!r}")
        self.strikes: int = int(strikes)

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
        self._commit_radius = float(view.goal.get("commit_radius", 0.0))
        # Position rows of the transition matrices at 5 s steps, built once: the slow suite
        # consults the supervisor thousands of times per episode.
        steps = int(self.warning_s / _SAMPLE_S + 1e-9)
        times = _SAMPLE_S * np.arange(1, steps + 1)
        self._drift = np.array([cw_stm(view.n, float(t))[:3] for t in times]).reshape(-1, 3, 6)
        self._strikes: int = 0
        self._last_closing: float | None = None

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
        braking = self._braking(phase, kind, closing, rho)
        if kind != "contact" and phase != "commit" and rho >= self._commit_radius:
            if (
                self._radius > 0.0
                and closing > 0.0
                and self._collides(situation.obs)
                and self._cannot_brake(rho, closing)
            ):
                for command in (Command.CAM, Command.RETREAT):
                    if mask[command]:
                        return command
            if braking:
                for command in (Command.RETREAT, Command.CAM):
                    if mask[command]:
                        return command
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

    def _collides(self, obs: np.ndarray) -> bool:
        """Whether the free drift of `obs` is strictly inside the collision radius at a sample."""
        positions = self._drift @ obs[:6]
        return bool(np.any(np.einsum("ij,ij->i", positions, positions) < self._radius**2))

    def _cannot_brake(self, rho: float, closing: float) -> bool:
        """Whether stopping before the sphere needs more than `brake_margin` of the thrust."""
        gap = rho - self._radius
        if gap <= 0.0:
            return True
        return closing**2 / (2.0 * gap) > self.brake_margin * self._view.max_accel

    def _braking(self, phase: str, kind: str, closing: float, rho: float) -> bool:
        """Count a consult of a braking deficit; true once `strikes` of them follow each other."""
        slowed = self._last_closing is not None and closing < self._last_closing
        applies = phase in ("stopped", "holding") and kind != "contact"
        fast = closing > self.overspeed * speed_cap(self._view, rho)
        self._last_closing = closing if applies else None
        # A consult where the speed fell starts the count again from itself, so a run of
        # `strikes` consults needs the speed not to fall between them.
        if not (applies and fast):
            self._strikes = 0
        elif slowed:
            self._strikes = 1
        else:
            self._strikes += 1
        return self._strikes >= self.strikes
