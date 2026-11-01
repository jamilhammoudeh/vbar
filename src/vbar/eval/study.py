"""The supervisor study: episode outcomes, decision disagreement, chattering, and its runner.

A supervisor under study flies each episode through a `StudyAgent`, which keeps every consult's
command, whether the agent accepted it, and the command the rule supervisor gives for the same
situation as a shadow that is never flown. `StudyTask` scores the episode with every metric of
`vbar.metrics.evaluate` and adds the study's own: the outcome (success, safe abort, timeout or
unsafe), the filter's intervention count, decision disagreement with the shadow over all consults
and over decision points, and decision flips per 100 environment steps.

A `Row` flies one model over its `Cell`s, each paired seed by seed with the rule supervisor on the
same validation seeds. `run_cell` gives every episode its own model budget, starts episodes only
while the cell's share of the row's cap holds them, and compares the pair by two one-sided tests
on a paired bootstrap interval, at each look of a sequential row.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..agents.llm import BudgetedModel, BudgetExceeded, Model, Price
from ..agents.llm_supervisor import SUPERVISOR_PROMPT, LLMSupervisor
from ..agents.rules import RuleSupervisor
from ..agents.supervision import Command, Order, Situation, SupervisedAgent
from ..agents.tracker import SetpointTracker
from ..faults import After, AllOf, Intermittent, RangeBelow, ThrusterOff
from ..metrics.summary import clopper_pearson
from ..safety import FilteredAgent
from ..scenarios import make, seeds
from .runner import Factory, Job, run_jobs
from .stats import _percentile
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
    fault : str, optional
        A key of `FAULTS` injected into every environment the task builds.

    Raises
    ------
    ValueError
        If `fault` is not a key of `FAULTS`.
    """

    fault: str | None = None

    def __post_init__(self):
        if self.fault is not None and self.fault not in FAULTS:
            raise ValueError(f"unknown fault {self.fault!r}, expected one of {sorted(FAULTS)}")

    def make(self, scenario: str) -> Any:
        """The catalogue environment of `scenario`, with fresh objects of the task's fault.

        Parameters
        ----------
        scenario : str
            Scenario name with variants.

        Returns
        -------
        ScenarioEnv
            A new environment.
        """
        if self.fault is None:
            return make(scenario)
        return make(scenario, faults=FAULTS[self.fault][1]())

    def identity(self, env: Any) -> Mapping[str, Any]:
        """The identifier flown, its dynamics model and the fault's name.

        Parameters
        ----------
        env : ScenarioEnv
            The environment.

        Returns
        -------
        dict
            "scenario_id", "dynamics" and "fault".
        """
        return {**super().identity(env), "fault": self.fault}

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


def _braking_loss() -> tuple:
    # The along-track thrusters that brake an approach from +y fail, at random decisions with
    # probability 0.93, once the chaser is inside 300 m: tuned so the rule supervisor fails
    # about half of the validation seeds.
    within = After(condition=RangeBelow(range_m=300.0))
    return (
        ThrusterOff(
            axis=1,
            direction=+1,
            when=AllOf(activations=(within, Intermittent(probability=0.93))),
            name="braking_intermittent",
        ),
    )


# Fault name to the scenario it is tuned for and a factory of fresh fault objects (they keep
# state, so every environment gets its own).
FAULTS: dict[str, tuple[str, Callable[[], tuple]]] = {
    "braking_loss": ("approach-vbar", _braking_loss),
}


@dataclass(frozen=True)
class Cell:
    """A scenario flown with the filter on or off, and optionally a fault from `FAULTS`.

    Parameters
    ----------
    scenario : str
        Scenario name with variants.
    filter : bool
        Whether the safety filter is on.
    fault : str, optional
        A key of `FAULTS`; the scenario must be the one the fault is tuned for.

    Raises
    ------
    ValueError
        If `scenario` is not a name, `filter` is not a bool, or `fault` is unknown or does not
        belong to the scenario.
    """

    scenario: str
    filter: bool
    fault: str | None = None

    def __post_init__(self):
        if not isinstance(self.scenario, str) or not self.scenario:
            raise ValueError("scenario must be a name")
        if not isinstance(self.filter, bool):
            raise ValueError(f"filter must be a bool, got {self.filter!r}")
        if self.fault is not None:
            if self.fault not in FAULTS:
                raise ValueError(f"unknown fault {self.fault!r}, expected one of {sorted(FAULTS)}")
            if FAULTS[self.fault][0] != self.scenario:
                raise ValueError(f"the {self.fault} fault is tuned for {FAULTS[self.fault][0]}")

    @property
    def label(self) -> str:
        """The cell's name, such as "approach-vbar:nav1 filter on"."""
        fault = "" if self.fault is None else f" {self.fault}"
        return f"{self.scenario} filter {'on' if self.filter else 'off'}{fault}"


def _positive(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not (math.isfinite(value) and value > 0.0)
    ):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return float(value)


def _count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True, kw_only=True)
class Row:
    """One supervisor's runs over a set of cells, each paired with the rule supervisor.

    Each cell flies the first `seeds` seeds of its scenario's validation draw, the same for every
    row, so a row with fewer seeds is a slice of one with more. A sequential row stops a cell at
    the first of its `looks` where every measure is decided; a row without looks flies every seed.

    Parameters
    ----------
    name : str
        The row's name, used for its result files.
    cells : tuple of Cell
        The cells, each once.
    seeds : int
        Most seeds per cell.
    looks : tuple of int
        Seed counts at which a sequential row judges each cell, increasing, the last `seeds`;
        empty for a fixed count.
    cap_usd : float, optional
        Most dollars the row may spend (USD), split evenly over its cells; None for a model
        without a price.
    episode_usd : float, optional
        Most dollars one episode may spend (USD); given exactly when `cap_usd` is.
    wall_limit_s : float, optional
        Wall time (s) after which an episode stops and is scored as a timeout.
    own_prompt : bool
        Whether the row is flown with a declared prompt of its own, not the default.

    Raises
    ------
    ValueError
        If a field has the wrong type, sign or order, a cell repeats, or `episode_usd` is given
        without `cap_usd`, is missing with it, or is more than a cell's share.
    """

    name: str
    cells: tuple[Cell, ...]
    seeds: int
    looks: tuple[int, ...] = ()
    cap_usd: float | None = None
    episode_usd: float | None = None
    wall_limit_s: float | None = None
    own_prompt: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a name")
        if (
            not isinstance(self.cells, tuple)
            or not self.cells
            or not all(isinstance(c, Cell) for c in self.cells)
        ):
            raise ValueError("cells must be a non-empty tuple of Cell")
        if len(set(self.cells)) != len(self.cells):
            raise ValueError("a cell appears twice")
        _count("seeds", self.seeds)
        if not isinstance(self.looks, tuple):
            raise ValueError("looks must be a tuple")
        for look in self.looks:
            _count("a look", look)
        if self.looks and (
            any(b <= a for a, b in zip(self.looks, self.looks[1:], strict=False))
            or self.looks[-1] != self.seeds
        ):
            raise ValueError("looks must increase and end at seeds")
        if (self.cap_usd is None) != (self.episode_usd is None):
            raise ValueError("cap_usd and episode_usd go together")
        if self.cap_usd is not None:
            _positive("cap_usd", self.cap_usd)
            _positive("episode_usd", self.episode_usd)
            if self.episode_usd > self.share_usd:
                raise ValueError("episode_usd is more than a cell's share of cap_usd")
        if self.wall_limit_s is not None:
            _positive("wall_limit_s", self.wall_limit_s)
        if not isinstance(self.own_prompt, bool):
            raise ValueError(f"own_prompt must be a bool, got {self.own_prompt!r}")

    @property
    def sequential(self) -> bool:
        """Whether cells stop at a look once decided."""
        return bool(self.looks)

    @property
    def share_usd(self) -> float | None:
        """Each cell's share of the cap (USD); None without a cap."""
        return None if self.cap_usd is None else self.cap_usd / len(self.cells)

    @property
    def confidence(self) -> float:
        """Confidence of the paired intervals: 97.5% at each of 4 looks (Bonferroni over the
        looks of a two one-sided test at 0.05), 90% for a fixed count (the test's interval)."""
        if not self.sequential:
            return 1.0 - 2.0 * ALPHA
        return 1.0 - 2.0 * ALPHA / len(self.looks)


# One-sided level of each of the two one-sided tests (Schuirmann 1987).
ALPHA = 0.05
# Equivalence margins: success rate within 5 percentage points; filter interventions within 0.5
# per episode, judged with the filter on only.
MARGINS = {"success": 0.05, "intervention_count": 0.5}

CORE_SCENARIOS = ("approach-vbar", "approach-rbar", "docking:axis", "docking-afrl", "inspection")
CORE_CELLS = tuple(
    Cell(scenario + variant, on)
    for scenario in CORE_SCENARIOS
    for variant in ("", ":nav1")
    for on in (True, False)
)
FAULT_CELL = Cell("approach-vbar", False, "braking_loss")
BASE_FILTER_ON = tuple(Cell(s, True) for s in CORE_SCENARIOS)
NAV_FILTER_OFF = tuple(Cell(s + ":nav1", False) for s in CORE_SCENARIOS)

# The rows of the study. Each per-episode ceiling sits above the dearest measured episode of its
# model class, so an ordinary episode is not cut short by its budget.
ROWS: dict[str, Row] = {
    row.name: row
    for row in (
        Row(
            name="main",
            cells=(*CORE_CELLS, FAULT_CELL),
            seeds=270,
            looks=(50, 100, 200, 270),
            cap_usd=110.0,
            episode_usd=0.06,
        ),
        Row(
            name="ablation-1",
            cells=NAV_FILTER_OFF,
            seeds=30,
            cap_usd=4.0,
            episode_usd=0.06,
            own_prompt=True,
        ),
        Row(
            name="ablation-2",
            cells=NAV_FILTER_OFF,
            seeds=30,
            cap_usd=4.0,
            episode_usd=0.06,
            own_prompt=True,
        ),
        Row(name="frontier", cells=BASE_FILTER_ON, seeds=30, cap_usd=40.0, episode_usd=0.5),
        Row(
            name="second-frontier-probe",
            cells=BASE_FILTER_ON[:3],
            seeds=1,
            cap_usd=1.0,
            episode_usd=0.33,
        ),
        Row(
            name="second-frontier",
            cells=BASE_FILTER_ON[:3],
            seeds=20,
            cap_usd=20.0,
            episode_usd=0.6,
        ),
        Row(name="local", cells=BASE_FILTER_ON, seeds=5, wall_limit_s=3600.0),
    )
}


# Bound on one request to the model (bytes): the supervisor's prompt is about 4 kB and the
# runner flies without repair turns. With the reply limit it prices the one call that may pass a
# budget, which each episode's budget is kept below.
REQUEST_BYTES = 16384


@dataclass(frozen=True)
class StudyModel:
    """The model a row flies as the supervisor.

    Parameters
    ----------
    label : str
        Name recorded in the provenance of every run.
    complete : callable
        Zero-argument factory of the model, `complete() -> Model`, called once per episode; with
        several workers it must be importable.
    price : Price, optional
        Declared price (USD per million tokens); needed for a row with a cap.
    max_output_tokens : int, optional
        The reply limit the model is called with; given exactly when `price` is.

    Raises
    ------
    ValueError
        If `label` is not a name, `complete` is not callable, `price` is not a `Price`, or
        `max_output_tokens` is not a positive integer or comes without `price`.
    """

    label: str
    complete: Callable[[], Model]
    price: Price | None = None
    max_output_tokens: int | None = None

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a name")
        if not callable(self.complete):
            raise ValueError("complete must be a factory of the model")
        if self.price is not None and not isinstance(self.price, Price):
            raise ValueError(f"price must be a Price, got {self.price!r}")
        if (self.price is None) != (self.max_output_tokens is None):
            raise ValueError("price and max_output_tokens go together")
        if self.max_output_tokens is not None:
            _count("max_output_tokens", self.max_output_tokens)


@dataclass(frozen=True)
class _StudiedAgent:
    """Builds the agent of one episode: a fresh model, its budget, the supervisor and the shadow."""

    model: StudyModel
    prompt: str
    job_usd: float | None
    wall_limit_s: float | None

    def __call__(self) -> StudyAgent:
        complete = self.model.complete()
        if self.job_usd is not None:
            complete = BudgetedModel(
                complete,
                price=self.model.price,
                max_usd=self.job_usd,
                max_output_tokens=self.model.max_output_tokens,
            )
        supervisor = LLMSupervisor(complete, prompt=self.prompt, price=self.model.price)
        return StudyAgent(
            supervisor,
            shadow=True,
            wall_limit_s=self.wall_limit_s,
            provenance={
                "model": self.model.label,
                "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
                "job_usd": self.job_usd,
            },
        )


def rule_agent() -> StudyAgent:
    """The rule supervisor every row is paired with, flown through a `StudyAgent`."""
    return StudyAgent(RuleSupervisor())


def episode_outcome(result: Mapping[str, Any]) -> str | None:
    """A result's outcome: one of `OUTCOMES`, "error", or None for an unfinished episode.

    An episode stopped by its model budget is unfinished; one stopped by its wall limit is a
    timeout; one stopped by any other error is "error".

    Parameters
    ----------
    result : mapping
        A result of `run_jobs` flown with a `StudyTask`.

    Returns
    -------
    str or None
        The outcome.
    """
    error = result.get("error")
    if error is None:
        return str(result["metrics"]["outcome"])
    if error.startswith(f"{BudgetExceeded.__name__}:"):
        return None
    if error.startswith(f"{WallLimitReached.__name__}:"):
        return "timeout"
    return "error"


def compare(
    differences: Sequence[float],
    margin: float,
    confidence: float,
    reps: int = 10_000,
    rng: np.random.Generator | int = 0,
) -> dict[str, Any]:
    """Mean paired difference with its percentile bootstrap interval and the equivalence verdict.

    "verdict" is "equivalent" when the interval lies inside (-margin, margin), "beyond_margin"
    when it lies wholly beyond one margin, else "undecided"; a measure is decided unless
    undecided. "tost" is the two one-sided tests' result: "equivalent", "different" when the
    interval excludes 0 without lying inside the margins, else "not_equivalent" (Schuirmann 1987).

    Parameters
    ----------
    differences : sequence of float
        Per-seed differences, studied minus rule.
    margin : float
        Equivalence margin, positive.
    confidence : float
        Confidence level of the interval, in (0, 1).
    reps : int
        Bootstrap resamples, at least 100.
    rng : numpy.random.Generator or int
        Generator or seed of the resampling.

    Returns
    -------
    dict
        "n", "estimate", "lower", "upper", "margin", "confidence", "verdict", "tost", "decided";
        with no differences the numbers are NaN, "tost" None and the verdict "undecided".

    Raises
    ------
    ValueError
        If a difference is not finite, `margin` is not positive, `confidence` is outside (0, 1),
        or `reps` is below 100.
    """
    d = np.asarray(differences, dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("differences must be finite")
    _positive("margin", margin)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 100:
        raise ValueError(f"reps must be an integer of at least 100, got {reps!r}")
    out: dict[str, Any] = {"n": len(d), "margin": margin, "confidence": confidence}
    if not len(d):
        nan = math.nan
        return {
            **out,
            "estimate": nan,
            "lower": nan,
            "upper": nan,
            "verdict": "undecided",
            "tost": None,
            "decided": False,
        }
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    means = d[gen.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    lower, upper = _percentile(means, confidence)
    inside = -margin < lower and upper < margin
    if inside:
        verdict = "equivalent"
    elif lower > margin or upper < -margin:
        verdict = "beyond_margin"
    else:
        verdict = "undecided"
    if inside:
        tost = "equivalent"
    elif lower > 0.0 or upper < 0.0:
        tost = "different"
    else:
        tost = "not_equivalent"
    return {
        **out,
        "estimate": float(d.mean()),
        "lower": lower,
        "upper": upper,
        "verdict": verdict,
        "tost": tost,
        "decided": verdict != "undecided",
    }


def _rate(k: int, n: int) -> dict[str, Any]:
    if n == 0:
        return {"k": 0, "n": 0, "rate": math.nan, "lower": math.nan, "upper": math.nan}
    lower, upper = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": k / n, "lower": lower, "upper": upper}


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _spend(result: Mapping[str, Any], episode_usd: float) -> float:
    """Dollars an episode spent; an episode stopped by an error is charged its ceiling."""
    if result.get("error") is not None:
        return episode_usd
    cost = result["metrics"].get("cost_usd")
    return 0.0 if cost is None else float(cost)


def side_summary(results: Sequence[Mapping[str, Any]], faulted: bool) -> dict[str, Any]:
    """Outcomes, rates and per-episode means of one supervisor's runs of a cell.

    Parameters
    ----------
    results : sequence of mapping
        The runs, as `run_jobs` gives them with a `StudyTask`.
    faulted : bool
        Whether the cell injects a fault, which adds the abort-when-faulted rate.

    Returns
    -------
    dict
        "episodes", "unfinished", "outcomes" (counts of each outcome and of "error");
        "mission_success" (success over finished episodes, a timeout or error a failure),
        "unsafe" (unsafe over episodes with an outcome, a timeout safe) and, in a faulted cell,
        "abort_when_faulted" (safe aborts over finished episodes), each with its Clopper-Pearson
        95% interval; per-episode means of "chattering", "consults", "input_tokens",
        "output_tokens", "cost_usd" (USD) and "wall_s" (s) over scored episodes;
        "invalid_action_rate" (parse failures over consults, pooled); "spent_usd", the cost of
        every run, each run stopped by an error at no recorded cost counted as 0.
    """
    outcomes = [episode_outcome(r) for r in results]
    finished = [o for o in outcomes if o is not None]
    counts = {o: finished.count(o) for o in (*OUTCOMES, "error")}
    known = len(finished) - counts["error"]
    scored = [r for r in results if r.get("error") is None]
    metrics = [r["metrics"] for r in scored]
    consults = sum(int(m["consults"]) for m in metrics)
    out: dict[str, Any] = {
        "episodes": len(results),
        "unfinished": len(results) - len(finished),
        "outcomes": counts,
        "mission_success": _rate(counts["success"], len(finished)),
        "unsafe": _rate(counts["unsafe"], known),
    }
    if faulted:
        out["abort_when_faulted"] = _rate(counts["safe_abort"], len(finished))
    out.update(
        {
            key: _mean([float(m[key] or 0.0) for m in metrics])
            for key in ("chattering", "consults", "input_tokens", "output_tokens", "cost_usd")
        }
    )
    out["wall_s"] = _mean([float(r["wall_s"]) for r in scored])
    out["invalid_action_rate"] = (
        sum(int(m["parse_failures"]) for m in metrics) / consults if consults else math.nan
    )
    out["spent_usd"] = sum(float(m["cost_usd"] or 0.0) for m in metrics)
    return out


def summarize_cell(
    row: Row,
    cell: Cell,
    studied: Sequence[Mapping[str, Any]],
    rule: Sequence[Mapping[str, Any]],
    reps: int = 10_000,
    rng: np.random.Generator | int = 0,
) -> dict[str, Any]:
    """The comparison of a cell's runs with the rule supervisor's on the same seeds.

    Runs are paired by seed index; a pair is used when both episodes finished. The success
    measure is judged in every cell, the filter's intervention count only with the filter on,
    each against `MARGINS` at the row's confidence.

    Parameters
    ----------
    row : Row
        The row.
    cell : Cell
        The cell.
    studied, rule : sequence of mapping
        The runs of the supervisor under study and of the rule supervisor, in seed order.
    reps : int
        Bootstrap resamples.
    rng : numpy.random.Generator or int
        Generator or seed of the resampling.

    Returns
    -------
    dict
        "row", "cell", "seeds", "pairs", "measures" (name to `compare`), "decided" (every
        measure decided), "studied" and "rule" (`side_summary`), and "disagreement": the
        studied runs' consults, decision points and the counts and shares that differ from the
        shadow, pooled over scored episodes.
    """
    pairs = [
        (s, r)
        for s, r in zip(studied, rule, strict=False)
        if episode_outcome(s) is not None and episode_outcome(r) is not None
    ]
    success = [
        float(episode_outcome(s) == "success") - float(episode_outcome(r) == "success")
        for s, r in pairs
    ]
    measures = {"success": compare(success, MARGINS["success"], row.confidence, reps, rng)}
    if cell.filter:
        counts = [
            float(s["metrics"]["intervention_count"]) - float(r["metrics"]["intervention_count"])
            for s, r in pairs
            if s.get("error") is None and r.get("error") is None
        ]
        measures["intervention_count"] = compare(
            counts, MARGINS["intervention_count"], row.confidence, reps, rng
        )
    shown = [s["metrics"] for s in studied if s.get("error") is None]
    totals = {
        key: sum(int(m[key] or 0) for m in shown)
        for key in (
            "consults",
            "decision_points",
            "consults_differing",
            "decision_points_differing",
        )
    }

    def share(k: str, n: str) -> float:
        return totals[k] / totals[n] if totals[n] else math.nan

    return {
        "row": row.name,
        "cell": cell.label,
        "seeds": len(studied),
        "pairs": len(pairs),
        "measures": measures,
        "decided": all(m["decided"] for m in measures.values()),
        "studied": side_summary(studied, cell.fault is not None),
        "rule": side_summary(rule, cell.fault is not None),
        "disagreement": {
            **totals,
            "disagreement": share("consults_differing", "consults"),
            "decision_point_share": share("decision_points", "consults"),
            "decision_point_disagreement": share("decision_points_differing", "decision_points"),
        },
    }


def _file_name(cell: Cell) -> str:
    return cell.label.replace(":", "_").replace(" ", "_") + ".jsonl"


def run_cell(
    row: Row,
    cell: Cell,
    model: StudyModel,
    *,
    out_dir: str | Path,
    prompt: str = SUPERVISOR_PROMPT,
    safety_filter: Factory | None = None,
    workers: int = 1,
    reps: int = 10_000,
) -> dict[str, Any]:
    """Fly one cell of a row and its paired rule runs, and compare them.

    The supervisor under study flies the cell's seeds in order, `workers` episodes at a time,
    each through its own `BudgetedModel` capped at the row's `episode_usd` less one worst-case
    call (`REQUEST_BYTES` of input and the reply limit), so an episode cannot spend more than
    `episode_usd`. A batch starts only while the cell's spend plus `episode_usd` for each of its
    episodes stays within the cell's share of the cap; a run stopped by an error is charged
    `episode_usd`. At each look (or after every seed, for a row without looks) the rule
    supervisor flies the same seeds and the cell is summarized; a sequential row stops at the
    first look where every measure is decided. Results go to `<out_dir>/<row>/` and
    `<out_dir>/rule/`, one JSON-lines file per cell, and a rerun resumes from them.

    Parameters
    ----------
    row : Row
        The row.
    cell : Cell
        One of the row's cells.
    model : StudyModel
        The model flown as the supervisor.
    out_dir : str or Path
        Folder of the result files.
    prompt : str
        The supervisor's prompt template; a row with `own_prompt` needs one of its own.
    safety_filter : callable or str, optional
        Factory of the filter, needed in a cell with the filter on and unused otherwise.
    workers : int
        Episodes flown at once.
    reps : int
        Bootstrap resamples of each comparison.

    Returns
    -------
    dict
        `summarize_cell` at the last look flown, with "model", "status" ("decided",
        "undecided" after the last look, "fixed" for a row without looks, or "budget" when the
        cell's share ran out first) and "look" (seeds flown).

    Raises
    ------
    ValueError
        If the cell is not in the row, a filter-on cell has no filter, an `own_prompt` row gets
        the default prompt, a capped row's model has no price, the reply limit leaves no budget
        for an episode, or `workers` is not a positive integer.
    """
    if cell not in row.cells:
        raise ValueError(f"{cell.label} is not a cell of the {row.name} row")
    if cell.filter and safety_filter is None:
        raise ValueError("a cell with the filter on needs safety_filter")
    if row.own_prompt and prompt == SUPERVISOR_PROMPT:
        raise ValueError(f"the {row.name} row is flown with a prompt of its own")
    _count("workers", workers)
    job_usd = None
    if row.cap_usd is not None:
        if model.price is None or model.max_output_tokens is None:
            raise ValueError(f"the {row.name} row has a cap, so its model needs a price")
        worst_call = model.price.cost(REQUEST_BYTES, model.max_output_tokens)
        job_usd = cast(float, row.episode_usd) - worst_call
        if job_usd <= 0.0:
            raise ValueError("one worst-case call costs more than episode_usd")
    task = StudyTask(fault=cell.fault)
    guard = safety_filter if cell.filter else None
    jobs = [
        Job(cell.scenario, "validation", index, seed, agent_seed)
        for index, (seed, agent_seed) in enumerate(
            seeds(cell.scenario, "validation", row.seeds).pairs()
        )
    ]
    studied_out = Path(out_dir) / row.name / _file_name(cell)
    rule_out = Path(out_dir) / "rule" / _file_name(cell)
    studied_out.parent.mkdir(parents=True, exist_ok=True)
    rule_out.parent.mkdir(parents=True, exist_ok=True)
    factory = _StudiedAgent(model, prompt, job_usd, row.wall_limit_s)
    studied: list[dict] = []
    status = "undecided" if row.sequential else "fixed"
    report: dict[str, Any] = {}
    for look in row.looks or (row.seeds,):
        while len(studied) < look:
            room = min(look - len(studied), workers)
            if row.share_usd is not None:
                ceiling = cast(float, row.episode_usd)
                spent = sum(_spend(r, ceiling) for r in studied)
                # The rounding allowance keeps an exact share from losing its last episode.
                room = min(room, math.floor((row.share_usd - spent) / ceiling + 1e-9))
            if room < 1:
                status = "budget"
                break
            batch = jobs[len(studied) : len(studied) + room]
            studied.extend(
                run_jobs(
                    batch, factory, task=task, safety_filter=guard, workers=room, out=studied_out
                )
            )
        rule = run_jobs(
            jobs[: len(studied)],
            rule_agent,
            task=task,
            safety_filter=guard,
            workers=workers,
            out=rule_out,
        )
        report = summarize_cell(row, cell, studied, rule, reps)
        if status == "budget":
            break
        if row.sequential and report["decided"]:
            status = "decided"
            break
    return {**report, "model": model.label, "status": status, "look": len(studied)}


def run_row(row: Row, model: StudyModel, **options: Any) -> list[dict[str, Any]]:
    """`run_cell` for every cell of a row, in order.

    Parameters
    ----------
    row : Row
        The row.
    model : StudyModel
        The model flown as the supervisor.
    **options
        Keyword arguments of `run_cell` (`out_dir` is required).

    Returns
    -------
    list of dict
        One report per cell.
    """
    return [run_cell(row, cell, model, **options) for cell in row.cells]


__all__ = [
    "ABORTS",
    "ALPHA",
    "BASE_FILTER_ON",
    "CORE_CELLS",
    "CORE_SCENARIOS",
    "FAULTS",
    "FAULT_CELL",
    "MARGINS",
    "NAV_FILTER_OFF",
    "ROWS",
    "Cell",
    "REQUEST_BYTES",
    "Row",
    "StudyModel",
    "compare",
    "episode_outcome",
    "rule_agent",
    "run_cell",
    "run_row",
    "side_summary",
    "summarize_cell",
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
