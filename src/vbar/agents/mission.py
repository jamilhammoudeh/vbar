"""Mission manager: a timeline of modes, each flown by its own agent until one of its exits holds.

A mode runs until its exit gate holds (next), runs a contingency mode when its time runs out
without the gate, and leaves for an abort mode when safety demands it (Fehse, Automated
Rendezvous and Docking of Spacecraft, section 6.3: the timeline, the mode table, the transition
criteria on time and state, and the contingency mode of each phase).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dynamics import cw_discrete
from .base import Agent, AgentView, command_fraction
from .planning import predicted_margin


@dataclass(frozen=True, kw_only=True)
class Mode:
    """One mode of the timeline.

    `exit(obs, t_in_mode)` is the gate to `next`, `t_max` (s in the mode) the time after which
    the mode runs `contingency` if the gate does not hold, and `abort` the mode taken when the
    manager's abort predicate holds or a coast the agent commits is not predicted clear of the
    keep-out zone. Names refer to other modes of the same manager.
    """

    name: str
    agent: Agent
    exit: Callable[[np.ndarray, float], bool] | None = None
    t_max: float | None = None
    next: str | None = None
    contingency: str | None = None
    abort: str | None = None


class MissionManagerAgent:
    """Runs a timeline of modes, forwarding each decision to the active mode's agent.

    The first mode starts the episode. A mode's agent is reset with the view and the random
    stream each time its mode is entered, and sees only the observation and the agent-safe info.
    After the agent acts, the exits are checked in order:

    - abort, to `Mode.abort`: `abort_when(obs, time_s)` holds, or in "impulse_coast" the
      certified lower bound of `planning.predicted_margin` over the coast the action commits is
      negative.
    - contingency, to `Mode.contingency`: the time in the mode is above `t_max` and the exit
      gate does not hold.
    - next, to `Mode.next`: the exit gate holds, or the agent sets `done = True`.

    An exit whose mode is None is not taken. On any transition a coast the action commits is
    cut, so the new mode decides one step later.

    `log` holds one entry per decision: `mode`, `t_in_mode` (s), `transition` ("abort",
    "contingency", "next" or None), `to` (the next mode's name or None), and over a committed
    coast `predicted_margin` and `predicted_margin_bound` (m; None without a coast). Without an
    abort mode, a coast predicted to enter the keep-out zone is logged and flown.
    """

    def __init__(
        self,
        modes: Sequence[Mode],
        *,
        abort_when: Callable[[np.ndarray, float], bool] | None = None,
        name: str = "mission",
    ):
        """Check the timeline.

        Parameters
        ----------
        modes : sequence of Mode
            The timeline; the first mode starts the episode.
        abort_when : callable, optional
            Predicate on the observation and the episode time (s) that sends a mode with an
            abort mode there.
        name : str
            Agent name.

        Raises
        ------
        ValueError
            If there are no modes, two share a name, a mode names a missing mode, a mode has a
            contingency without `t_max`, or the last mode has a next mode.
        """
        modes = tuple(modes)
        if not modes:
            raise ValueError("a timeline needs at least one mode")
        names = [mode.name for mode in modes]
        if len(set(names)) != len(names):
            raise ValueError("mode names must be unique")
        for mode in modes:
            for exit_kind in ("next", "contingency", "abort"):
                target = getattr(mode, exit_kind)
                if target is not None and target not in names:
                    raise ValueError(f"mode {mode.name!r} names a missing {exit_kind} {target!r}")
            if mode.contingency is not None and mode.t_max is None:
                raise ValueError(f"mode {mode.name!r} has a contingency but no t_max")
        if modes[-1].next is not None:
            raise ValueError("the last mode must have no next mode")
        self.modes = {mode.name: mode for mode in modes}
        self.first = modes[0].name
        self.abort_when = abort_when
        self.name = name

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        self.view = view
        self.rng = rng
        self.one_step = cw_discrete(view.n, view.dt)
        self.log: list[dict[str, Any]] = []
        self.t_entry: float | None = None
        self._enter(self.first)

    def act(self, obs, info):
        t = info["time_s"]
        if self.t_entry is None:
            self.t_entry = t
        t_in_mode = t - self.t_entry
        mode = self.active
        action = mode.agent.act(obs, info)
        margin = bound = None
        coast_s = 0.0
        if self.view.action_mode == "impulse_coast":
            coast_s = round(float(action[3]) / self.view.dt) * self.view.dt
        if coast_s > 0.0:
            phi, gamma = self.one_step
            accel = command_fraction(self.view, action) * self.view.max_accel
            start = phi @ np.asarray(obs[:6], dtype=float) + gamma @ accel
            margin, bound = predicted_margin(self.view, start, coast_s)
        gate = mode.exit is not None and bool(mode.exit(obs, t_in_mode))
        transition, target = None, None
        unsafe = self.abort_when is not None and bool(self.abort_when(obs, t))
        if mode.abort is not None and (unsafe or (bound is not None and bound < 0.0)):
            transition, target = "abort", mode.abort
        elif mode.contingency is not None and not gate and _timed_out(t_in_mode, mode.t_max):
            transition, target = "contingency", mode.contingency
        elif mode.next is not None and (gate or getattr(mode.agent, "done", False)):
            transition, target = "next", mode.next
        self.log.append(
            {
                "mode": mode.name,
                "t_in_mode": t_in_mode,
                "transition": transition,
                "to": target,
                "predicted_margin": margin,
                "predicted_margin_bound": bound,
            }
        )
        if target is not None:
            if coast_s > 0.0:
                action = np.array(action, dtype=float)
                action[3] = 0.0
            self._enter(target)
        return action

    def _enter(self, name: str) -> None:
        """Enter mode `name`: reset its agent and restart the mode clock at the next decision."""
        self.active = self.modes[name]
        self.active.agent.reset(self.view, self.rng)
        self.t_entry = None


def _timed_out(t_in_mode: float, t_max: float | None) -> bool:
    """Whether the time in the mode (s) is strictly above its limit; never without one."""
    return t_max is not None and t_in_mode > t_max
