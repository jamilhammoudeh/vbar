"""The supervisor study: episode outcomes, decision disagreement and chattering.

A supervisor under study flies each episode through a `StudyAgent`, which keeps every consult's
command, whether the agent accepted it, and the command the rule supervisor gives for the same
situation as a shadow that is never flown. `StudyTask` scores the episode with every metric of
`vbar.metrics.evaluate` and adds the study's own: the outcome (success, safe abort, timeout or
unsafe), the filter's intervention count, decision disagreement with the shadow over all consults
and over decision points, and decision flips per 100 environment steps.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..agents.llm import BudgetedModel
from ..agents.llm_supervisor import LLMSupervisor
from ..agents.rules import RuleSupervisor
from ..agents.supervision import Command, Order, Situation, SupervisedAgent
from ..agents.tracker import SetpointTracker
from ..safety import FilteredAgent
from .task import SpaceTask

OUTCOMES = ("success", "safe_abort", "timeout", "unsafe")
ABORTS = frozenset({Command.RETREAT, Command.CAM})
# A consult with an abort valid is a decision point only this close, in keep-out radii.
DECISION_RANGE = 3.0


class WallLimitReached(RuntimeError):
    """An episode ran longer in wall time than its limit; the study scores it as a timeout."""


def outcome(success: bool, keep_out_entered: bool, aborted: bool) -> str:
    """The outcome of a supervised episode, one of `OUTCOMES`.

    Any keep-out entry makes the episode unsafe, whatever else happened; otherwise a completed
    task is a success, an accepted RETREAT or CAM a safe abort, and anything else a timeout.

    Parameters
    ----------
    success : bool
        Whether the task was completed.
    keep_out_entered : bool
        Whether the chaser entered the keep-out sphere.
    aborted : bool
        Whether a RETREAT or CAM was accepted during the episode.

    Returns
    -------
    str
        "unsafe", "success", "safe_abort" or "timeout".
    """
    if keep_out_entered:
        return "unsafe"
    if success:
        return "success"
    return "safe_abort" if aborted else "timeout"


def decision_point(situation: Situation, radius: float) -> bool:
    """Whether a consult offers a real choice.

    It does when GO is valid while the chaser waits at a rung ("holding" or "stopped"), or when
    a RETREAT or CAM is valid while the chaser closes within `DECISION_RANGE` keep-out radii.

    Parameters
    ----------
    situation : Situation
        The consult's situation (Hill axes, m and m/s).
    radius : float
        The keep-out radius (m); 0 for a scenario without one, where no abort point exists.

    Returns
    -------
    bool
        Whether the consult is a decision point.
    """
    mask = situation.mask
    if mask[Command.GO] and situation.phase in ("holding", "stopped"):
        return True
    r = np.asarray(situation.obs[:3], dtype=float)
    v = np.asarray(situation.obs[3:6], dtype=float)
    rho = float(np.linalg.norm(r))
    closing = rho > 0.0 and float(r @ v) < 0.0
    near = radius > 0.0 and rho < DECISION_RANGE * radius
    return closing and near and (mask[Command.CAM] or mask[Command.RETREAT])


@dataclass(frozen=True)
class Consult:
    """One consult of a supervisor under study.

    Parameters
    ----------
    t : float
        Time of the consult (s).
    command : Command or None
        The command given; None for an answer that is neither a `Command` nor an `Order`.
    shadow : Command or None
        The rule supervisor's command for the same situation; None when no shadow runs.
    decision_point : bool
        Whether the consult was a decision point (`decision_point`).
    accepted : bool or None
        Whether the agent accepted the order; None until the agent has applied it.
    """

    t: float
    command: Command | None
    shadow: Command | None
    decision_point: bool
    accepted: bool | None = None


def _command(answer: Any) -> Command | None:
    if isinstance(answer, Order):
        return answer.command
    return answer if isinstance(answer, Command) else None


def disagreement(consults: Sequence[Consult]) -> dict[str, Any]:
    """Decision disagreement between a supervisor and its shadow over one episode.

    Parameters
    ----------
    consults : sequence of Consult
        The episode's consults, in time order, each with a shadow command.

    Returns
    -------
    dict
        "consults", "decision_points", "consults_differing" and "decision_points_differing"
        (counts); "disagreement" (consults differing over consults), "decision_point_share"
        (decision points over consults) and "decision_point_disagreement" (decision points
        differing over decision points), NaN where the count below is 0; "first_difference_s",
        the time of the first consult that differs (s), None if none does.

    Raises
    ------
    ValueError
        If a consult has no shadow command.
    """
    if any(c.shadow is None for c in consults):
        raise ValueError("disagreement needs a shadow command at every consult")
    differing = [c for c in consults if c.command != c.shadow]
    points = [c for c in consults if c.decision_point]
    points_differing = [c for c in points if c.command != c.shadow]
    n = len(consults)
    return {
        "consults": n,
        "decision_points": len(points),
        "consults_differing": len(differing),
        "decision_points_differing": len(points_differing),
        "disagreement": len(differing) / n if n else math.nan,
        "decision_point_share": len(points) / n if n else math.nan,
        "decision_point_disagreement": len(points_differing) / len(points) if points else math.nan,
        "first_difference_s": differing[0].t if differing else None,
    }


def chattering(consults: Sequence[Consult], steps: int) -> tuple[int, float]:
    """Decision flips and flips per 100 environment steps over one episode.

    A flip is a consult whose command is not CONTINUE and differs from the previous command of
    the episode that was not CONTINUE. Answers that are not commands are skipped.

    Parameters
    ----------
    consults : sequence of Consult
        The episode's consults, in time order.
    steps : int
        Environment steps flown, at least 1.

    Returns
    -------
    tuple
        The number of flips and the flips per 100 steps.

    Raises
    ------
    ValueError
        If `steps` is not a positive integer.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    flips = 0
    previous = None
    for consult in consults:
        if consult.command is None or consult.command == Command.CONTINUE:
            continue
        if previous is not None and consult.command != previous:
            flips += 1
        previous = consult.command
    return flips, 100.0 * flips / steps


def _keep_out_radius(view) -> float:
    return max(
        (float(c["radius"]) for c in view.constraints if c["type"].startswith("keep_out")),
        default=0.0,
    )


class _Shadowed:
    """Passes each situation to the supervisor, then to a shadow rule supervisor, and logs both."""

    def __init__(self, supervisor, shadow: bool, wall_limit_s: float | None):
        self.supervisor = supervisor
        self.shadow = RuleSupervisor() if shadow else None
        self.wall_limit_s = wall_limit_s
        self.name = getattr(supervisor, "name", type(supervisor).__name__)
        self.consults: list[Consult] = []
        self._radius = 0.0
        self._started = time.perf_counter()

    def reset(self, view, ladder, rng=None) -> None:
        self.supervisor.reset(view, ladder, rng)
        if self.shadow is not None:
            # The shadow gets no stream of its own, so it cannot change the supervisor's draws.
            self.shadow.reset(view, ladder)
        self._radius = _keep_out_radius(view)
        self.consults = []
        self._started = time.perf_counter()

    def decide(self, situation: Situation):
        if self.wall_limit_s is not None:
            elapsed = time.perf_counter() - self._started
            if elapsed > self.wall_limit_s:
                raise WallLimitReached(
                    f"the episode ran {elapsed:.0f} s, over its {self.wall_limit_s:.0f} s limit"
                )
        answer = self.supervisor.decide(situation)
        shadow = None if self.shadow is None else _command(self.shadow.decide(situation))
        self.consults.append(
            Consult(
                t=float(situation.t),
                command=_command(answer),
                shadow=shadow,
                decision_point=decision_point(situation, self._radius),
            )
        )
        return answer


class StudyAgent(SupervisedAgent):
    """A `SupervisedAgent` that keeps each consult's command, acceptance and shadow command.

    Parameters
    ----------
    supervisor : Supervisor
        The supervisor under study.
    shadow : bool
        Whether a `RuleSupervisor` is shown every situation as a shadow, never flown.
    wall_limit_s : float, optional
        Wall time (s) after which the next consult raises `WallLimitReached`.
    provenance : mapping, optional
        Further settings the provenance hash covers (the model, the prompt), JSON-safe.
    tracker : SetpointTracker, optional
        The tracker; default gains if None.
    period_s : float
        Time between scheduled consults (s).
    mode : str
        "ladder", "waypoint" or "velocity".
    name : str, optional
        The agent's name.

    Raises
    ------
    ValueError
        If `shadow` is not a bool, `wall_limit_s` is not positive and finite, or the arguments of
        `SupervisedAgent` are invalid.
    """

    def __init__(
        self,
        supervisor,
        *,
        shadow: bool = False,
        wall_limit_s: float | None = None,
        provenance: Mapping[str, Any] | None = None,
        tracker: SetpointTracker | None = None,
        period_s: float = 60.0,
        mode: str = "ladder",
        name: str | None = None,
    ):
        if not isinstance(shadow, bool):
            raise ValueError(f"shadow must be a bool, got {shadow!r}")
        if wall_limit_s is not None and (
            isinstance(wall_limit_s, bool)
            or not isinstance(wall_limit_s, (int, float))
            or not (math.isfinite(wall_limit_s) and wall_limit_s > 0.0)
        ):
            raise ValueError(f"wall_limit_s must be positive and finite, got {wall_limit_s!r}")
        self._shadowed = _Shadowed(supervisor, shadow, wall_limit_s)
        super().__init__(
            self._shadowed,
            tracker=tracker,
            period_s=period_s,
            mode=mode,
            name=name,
        )
        self.config = {
            "supervisor": self._shadowed.name,
            "shadow": shadow,
            "wall_limit_s": wall_limit_s,
            "period_s": self.period_s,
            "mode": self.mode,
            **dict(provenance or {}),
        }

    @property
    def flown(self):
        """The supervisor under study."""
        return self._shadowed.supervisor

    @property
    def consults(self) -> tuple[Consult, ...]:
        """The consults of the current episode, in time order."""
        return tuple(self._shadowed.consults)

    def act(self, obs, info) -> np.ndarray:
        """The tracker's command, as `SupervisedAgent.act`, keeping the consult's acceptance."""
        action = super().act(obs, info)
        step = self.last_step
        if step is not None and step.consulted:
            log = self._shadowed.consults
            log[-1] = replace(log[-1], accepted=bool(step.accepted))
        return action


def _ledger(supervisor) -> float | None:
    """Dollars spent so far by the budget an `LLMSupervisor` calls through, if it has one."""
    budget = getattr(supervisor, "complete", None) if type(supervisor) is LLMSupervisor else None
    return budget.spent_usd if type(budget) is BudgetedModel else None


def _usage(supervisor, spent_before: float) -> dict[str, Any]:
    """Model calls, tokens, dollars and parse failures of the supervisor's episode."""
    if type(supervisor) is not LLMSupervisor:
        return {
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "parse_failures": 0,
            "invalid_actions": 0,
        }
    usage = supervisor.usage()
    spent = _ledger(supervisor)
    # The budget's ledger is the harness's own count; the supervisor's price is the fallback.
    cost = usage["cost_usd"] if spent is None else spent - spent_before
    return {
        "model_calls": usage["model_calls"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cost_usd": cost,
        "parse_failures": usage["parse_failures"],
        "invalid_actions": usage["invalid_actions"],
    }


def study_metrics(
    metrics: Mapping[str, Any],
    consults: Sequence[Consult],
    steps: int,
    supervisor=None,
    spent_before: float = 0.0,
) -> dict[str, Any]:
    """The study's measures of one episode, from its metrics and its consults.

    Parameters
    ----------
    metrics : mapping
        `vbar.metrics.evaluate` of the episode ("success", "min_margin", "intervention_rate").
    consults : sequence of Consult
        The episode's consults.
    steps : int
        Environment steps flown.
    supervisor : Supervisor, optional
        The supervisor flown; an `LLMSupervisor` adds its model usage.
    spent_before : float
        What the supervisor's `BudgetedModel` had spent before the episode (USD), so a budget
        shared by several episodes is charged to each by its own calls.

    Returns
    -------
    dict
        "outcome", "aborted", "keep_out_entered", "intervention_count" (decisions the filter
        changed, F1n; NaN with the filter off), "steps", "flips", "chattering" (flips per 100
        steps), the `disagreement` fields when the consults carry shadow commands (None
        otherwise), "model_calls", "input_tokens", "output_tokens", "cost_usd" (USD; None for a
        model without a price), "parse_failures", "invalid_actions", and "invalid_action_rate"
        (parse failures over consults, NaN without consults).
    """
    entered = float(metrics["min_margin"].get("keep_out", math.inf)) < 0.0
    aborted = any(c.accepted and c.command in ABORTS for c in consults)
    flips, rate = chattering(consults, steps)
    out: dict[str, Any] = {
        "outcome": outcome(bool(metrics["success"]), entered, aborted),
        "aborted": aborted,
        "keep_out_entered": entered,
        "intervention_count": math.nan,
        "steps": steps,
        "flips": flips,
        "chattering": rate,
    }
    decisions = metrics.get("filter_decisions", 0)
    if decisions:
        # F1 is a ratio of counts, so the count is recovered exactly by rounding.
        out["intervention_count"] = round(metrics["intervention_rate"] * decisions)
    shadowed = bool(consults) and all(c.shadow is not None for c in consults)
    keys = (
        "consults_differing",
        "decision_points_differing",
        "disagreement",
        "decision_point_share",
        "decision_point_disagreement",
        "first_difference_s",
    )
    found = disagreement(consults) if shadowed else dict.fromkeys(keys)
    found["consults"] = len(consults)
    found["decision_points"] = sum(c.decision_point for c in consults)
    out.update(found)
    out.update(_usage(supervisor, spent_before))
    out["invalid_action_rate"] = out["parse_failures"] / len(consults) if consults else math.nan
    return out


@dataclass(frozen=True)
class StudyTask(SpaceTask):
    """The spacecraft task for `StudyAgent` runs, scored with the study's measures.

    Parameters
    ----------
    name : str
        The task's name in result files.
    """

    def record(self, env: Any, agent: Any, seed: int, agent_seed: int) -> Any:
        """One episode, with the study agent's consults.

        Parameters
        ----------
        env : ScenarioEnv
            The environment.
        agent : StudyAgent or FilteredAgent
            A `StudyAgent`, wrapped in a `FilteredAgent` when a filter is on.
        seed, agent_seed : int
            Episode and agent seeds.

        Returns
        -------
        tuple
            The `Episode`, the consults, the supervisor flown and what its budget had spent
            before the episode (USD).

        Raises
        ------
        ValueError
            If the agent is not a `StudyAgent` or a `FilteredAgent` around one.
        """
        inner = agent.agent if type(agent) is FilteredAgent else agent
        if type(inner) is not StudyAgent:
            raise ValueError("the study flies a StudyAgent")
        spent_before = _ledger(inner.flown) or 0.0
        episode = super().record(env, agent, seed, agent_seed)
        return episode, inner.consults, inner.flown, spent_before

    def score(self, env: Any, episode: Any) -> Mapping[str, Any]:
        """Every metric of `vbar.metrics.evaluate` and the study's measures (`study_metrics`).

        Parameters
        ----------
        env : ScenarioEnv
            The environment the episode was flown in.
        episode : tuple
            What `record` returned.

        Returns
        -------
        dict
            The metrics by name.
        """
        flown, consults, supervisor, spent_before = episode
        metrics = dict(super().score(env, flown))
        metrics["filter_decisions"] = sum(rec.filter_result is not None for rec in flown.records)
        metrics.update(
            study_metrics(metrics, consults, len(flown.records), supervisor, spent_before)
        )
        return metrics


__all__ = [
    "ABORTS",
    "DECISION_RANGE",
    "OUTCOMES",
    "Consult",
    "StudyAgent",
    "StudyTask",
    "WallLimitReached",
    "chattering",
    "decision_point",
    "disagreement",
    "outcome",
    "study_metrics",
]
