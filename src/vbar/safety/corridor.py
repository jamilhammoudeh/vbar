"""Corridor monitor with a phase-specific open-loop collision avoidance manoeuvre (CAM).

The switching filter of operational practice: the agent's command passes through unchanged while
the state stays inside the approach corridor, and the first decision outside the goal's commit
radius with the state outside the corridor latches an abort. The abort holds the CAM of the
approach phase whose range band holds the range, for that phase's number of decisions, then zero
thrust for the rest of the run (Fehse, *Automated Rendezvous and Docking of Spacecraft*, 2003,
sections 4.5 and 5.7). The corridor is the V-bar passive-safety speed envelope of `PassiveSpeed`
about the keep-out sphere, together with the scenario's approach cone when it declares one.

The filter makes no certified claim: the CAM is open loop and nothing checks that it clears the
keep-out. Its false-CAM, missed-CAM and CAM-induced rates are measured from paired runs by
`vbar.safety.audit.cam_rates`.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..agents.base import CONTINUOUS_MODES, AgentView, action_scale
from ..constraints import ApproachCone, PassiveSpeed
from . import FilterResult
from .cbf import _STATE_NAMES, FilterDeclaration, _state


def _phase(phase) -> tuple[float, float, np.ndarray, int]:
    try:
        lo, hi, a_c, m = phase
    except (TypeError, ValueError):
        raise ValueError("each phase is (rho_lo, rho_hi, a_c, m)") from None
    lo, hi = float(lo), float(hi)
    accel = np.array(a_c, dtype=float)
    if accel.shape != (3,) or not np.all(np.isfinite(accel)):
        raise ValueError("a phase's a_c must be three finite numbers")
    accel.flags.writeable = False
    try:
        held = operator.index(m)
    except TypeError:
        held = 0
    if isinstance(m, bool) or held < 1:
        raise ValueError("a phase's m must be a whole number of decisions, at least 1")
    if not (math.isfinite(lo) and lo < hi):
        raise ValueError("a phase's band needs a finite rho_lo below rho_hi")
    return lo, hi, accel, held


class CorridorCamFilter:
    """Pass the command through until the state leaves the corridor, then fly a latched CAM.

    Parameters
    ----------
    phases : sequence of (float, float, array_like, int)
        Approach phases (rho_lo, rho_hi, a_c, m): the band rho_lo <= rho < rho_hi of range from
        the target (m), the CAM's acceleration a_c (m/s^2, Hill frame) and the number of
        decisions m it is held, at least 1. The bands must tile [0, inf) in increasing order.

    Raises
    ------
    ValueError
        If a phase is malformed or the bands do not tile [0, inf).

    Notes
    -----
    The commit radius is the goal's `commit_radius`, measured from the target's centre, the
    convention every filter uses; on a docking goal whose port lies outside that radius the
    commit zone is empty, so the monitor never stands down.
    """

    name = "corridor-cam"

    def __init__(self, *, phases: Sequence[Any]):
        parsed = tuple(_phase(phase) for phase in phases)
        if not parsed:
            raise ValueError("phases must not be empty")
        edges = [parsed[0][0]] + [hi for _, hi, _, _ in parsed]
        starts = [lo for lo, _, _, _ in parsed]
        if edges[0] != 0.0 or edges[-1] != math.inf or starts[1:] != edges[1:-1]:
            raise ValueError("the phase bands must tile [0, inf) in increasing order")
        self.phases = parsed
        self._declaration: FilterDeclaration | None = None

    @property
    def declaration(self) -> FilterDeclaration:
        """The filter's declaration for the view of the last `reset`.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        """
        if self._declaration is None:
            raise RuntimeError("reset the filter first")
        return self._declaration

    def reset(self, view: AgentView) -> None:
        """Build the corridor for `view` and clear the abort latch.

        Parameters
        ----------
        view : AgentView
            The scenario's public description, in the Hill frame with a continuous action mode
            and a keep_out constraint.

        Raises
        ------
        ValueError
            If the action mode is not continuous, the frame is not "hill", the observation does
            not start with the Hill state, the scenario has no keep_out constraint, or a phase's
            CAM lies outside the thrust limit.
        """
        if view.action_mode not in CONTINUOUS_MODES:
            raise ValueError(f"filters need a continuous action mode, not {view.action_mode!r}")
        if view.frame != "hill":
            raise ValueError(f"filters need the hill frame, not {view.frame!r}")
        if tuple(view.observation[:6]) != _STATE_NAMES:
            raise ValueError("the observation must start with x, y, z, vx, vy, vz")
        radii = [float(spec["radius"]) for spec in view.constraints if spec["type"] == "keep_out"]
        if not radii:
            raise ValueError("the scenario has no keep_out constraint")
        a_max = float(view.max_accel)
        for _, _, a_c, _ in self.phases:
            size = np.max(np.abs(a_c)) if view.thrust_limit == "box" else np.linalg.norm(a_c)
            if size > a_max:
                raise ValueError(f"the CAM {a_c.tolist()} m/s^2 exceeds the thrust limit")
        cones = [
            ApproachCone(**{k: v for k, v in spec.items() if k != "type"})
            for spec in view.constraints
            if spec["type"] == "approach_cone"
        ]
        self._corridor: tuple[PassiveSpeed | ApproachCone, ...] = (
            PassiveSpeed("vbar", extent=max(radii), n=float(view.n)),
            *cones,
        )
        self._commit = float(view.goal.get("commit_radius", 0.0))
        # Converts m/s^2 to the view's action units at the initial mass.
        self._to_action = action_scale(view) / a_max
        self._cam: np.ndarray | None = None
        self._left = 0
        self._declaration = FilterDeclaration(
            name=self.name,
            safe_set="none: a corridor monitor makes no certified claim",
            barriers=self._corridor,
            poles=None,
            gamma=None,
            design_dt=float(view.dt),
            layout=view.thrust_limit,
            a_max=a_max,
            buffer_m=0.0,
            tightening={"navigation": None, "execution": 0.0, "model_error": None},
            backup="the phase's CAM, then zero thrust",
            certificate=None,
            counts_toward_certification=False,
        )

    def _outside(self, x: np.ndarray) -> bool:
        """Outside the commit zone with a negative corridor margin."""
        rho = float(np.linalg.norm(x[:3]))
        if rho <= self._commit:
            return False
        return min(float(c.margin(x)) for c in self._corridor) < 0.0

    def filter(self, obs: np.ndarray, action: np.ndarray) -> FilterResult:
        """Pass `action` through, or fly the latched abort.

        Parameters
        ----------
        obs : np.ndarray
            The observation, starting with the Hill state (m, m/s).
        action : np.ndarray
            The agent's command, in the view's action units.

        Returns
        -------
        FilterResult
            The command unchanged before the abort; from the first decision outside the commit
            radius and the corridor, the CAM of the phase whose band holds the range for its m
            decisions and zero thrust after them, in the view's action units, with `active` set
            and `fallback_level` 2.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        ValueError
            If the observation does not start with six finite numbers.
        """
        if self._declaration is None:
            raise RuntimeError("reset the filter first")
        x = _state(obs)
        command = np.array(action, dtype=float)
        cam = self._cam
        if cam is None:
            if not self._outside(x):
                return FilterResult(command, intervened=False)
            rho = float(np.linalg.norm(x[:3]))
            _, _, cam, self._left = next(p for p in self.phases if p[0] <= rho < p[1])
            self._cam = cam
        if self._left > 0:
            self._left -= 1
            u = cam * self._to_action
        else:
            u = np.zeros(3)
        return FilterResult(
            u, intervened=not np.array_equal(u, command), active=True, fallback_level=2
        )


__all__ = ["CorridorCamFilter"]
