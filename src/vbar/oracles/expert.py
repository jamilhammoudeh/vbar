"""The fuel oracle as an agent: it plans from its first observation and flies the plan.

`OracleAgent` gets what every agent gets (the scenario's public view, its own random stream, the
observations and the agent-safe info) and nothing from the environment. With perfect navigation
its first observation is the true start, so on the CW rung it flies the verified plan behind the
`FuelOracle` upper bound.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from ..agents.base import AgentView, action_scale, hold_point
from .fuel import FuelOracle, OracleConfig

_STATE = ("x", "y", "z", "vx", "vy", "vz")


class OracleAgent:
    """Flies the plan of `FuelOracle.solve` at a free final time from its first observation.

    The plan holds one Hill acceleration (m/s^2) per step, and the agent returns it in the
    view's action mode: over `max_accel` for "normalized", as is for "accel", and times the
    initial mass `mass_kg` for "thrust", which delivers more than planned as propellant is spent.
    Open loop by default: later observations are not read. With `replan_steps`, the agent
    solves again from the observation every `replan_steps` steps until the plan's arrival step,
    for the steps left in the episode with the thrust inhibit windows shifted to the new start;
    a solve that verifies no plan keeps the current one. After the plan ends the agent asks for
    no thrust. The random stream is not used, so the actions depend only on the view and the
    observations.

    Parameters
    ----------
    config : OracleConfig, optional
        Oracle settings; the defaults when omitted.
    replan_steps : int, optional
        Steps between solves, at least 1; open loop when omitted.
    name : str
        Name used in results.

    Raises
    ------
    ValueError
        If `replan_steps` is below 1.
    """

    def __init__(
        self,
        *,
        config: OracleConfig | None = None,
        replan_steps: int | None = None,
        name: str = "oracle",
    ):
        if replan_steps is not None and operator.index(replan_steps) < 1:
            raise ValueError("replan_steps must be at least 1")
        self.config = OracleConfig() if config is None else config
        self.replan_steps = replan_steps
        self.name = name
        self.oracle = FuelOracle(config=self.config)
        self.view: AgentView | None = None

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode with the scenario's public description.

        Parameters
        ----------
        view : AgentView
            The scenario's public description.
        rng : numpy.random.Generator, optional
            The agent's random stream, unused.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, the action mode is
            "disc27", "impulse_coast" or unknown, thrust actions come without `mass_kg`, the
            briefed target manoeuvres, or the observation does not start with the Hill state.
        """
        hold_point(view, "oracle agent")
        scale = action_scale(view)
        target = view.goal.get("target")
        if isinstance(target, Mapping) and target.get("burns"):
            raise ValueError("the oracle agent cannot plan around a manoeuvring target")
        if tuple(view.observation[:6]) != _STATE:
            raise ValueError(f"the oracle agent needs the observation to start with {_STATE}")
        self.view = view
        self.scale = scale / view.max_accel
        self.plan: np.ndarray | None = None
        self.plan_start = 0
        self.arrival = 0
        self.step = 0

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """The plan's command for this step in the view's action mode.

        Parameters
        ----------
        obs : numpy.ndarray
            Observation; its first six entries are the Hill state (m, m/s).
        info : dict
            Agent-safe info, unused.

        Returns
        -------
        numpy.ndarray, shape (3,)
            The command.

        Raises
        ------
        RuntimeError
            If the agent was not reset, or the first solve verifies no plan.
        ValueError
            As `FuelOracle.solve`.
        """
        view = self.view
        if view is None:
            raise RuntimeError("reset the oracle agent before it acts")
        k = self.step
        self.step += 1
        replan = self.replan_steps is not None and k % self.replan_steps == 0 and k < self.arrival
        if self.plan is None or replan:
            self._solve(view, obs, k)
        if self.plan is None:
            raise RuntimeError("the fuel oracle verified no plan from this observation")
        row = k - self.plan_start
        if row >= len(self.plan):
            return np.zeros(3)
        return self.plan[row] * self.scale

    def _solve(self, view: AgentView, obs: np.ndarray, k: int) -> None:
        """Solve from the observation at step `k` and adopt the plan if one was verified."""
        if k > 0:
            elapsed = k * view.dt
            windows = tuple(
                (max(t_on - elapsed, 0.0), t_off - elapsed)
                for t_on, t_off in view.goal.get("thrust_inhibit", ())
                if t_off > elapsed
            )
            goal = dict(view.goal)
            if "thrust_inhibit" in goal:
                goal["thrust_inhibit"] = windows
            view = replace(view, max_steps=view.max_steps - k, goal=goal)
        found = self.oracle.solve(view, np.asarray(obs, dtype=float)[:6])
        if found.plan is not None and found.arrival_step is not None:
            self.plan, self.plan_start, self.arrival = found.plan, k, k + found.arrival_step


__all__ = ["OracleAgent"]
