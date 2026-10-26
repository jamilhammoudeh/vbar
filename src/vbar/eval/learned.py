"""Evaluation rules for learned agents: training runs are the unit, and every run counts.

A learned entry reports every training run it launched, at least five (ten recommended).
Diverged, cap-saturated and non-terminating runs stay in N as failed runs; they are never
replaced by reruns (Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical
Precipice", NeurIPS 2021; Sutton and Barto, Reinforcement Learning, 2nd ed., 2018). An entry
also declares what shapes its result beyond the weights: the scored action mode, the learner's
discount and trace parameters, the training reward, stabilisers, how its hyperparameters were
tuned and how many configurations were tried, and, offline, where its data came from.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

RUN_STATUSES = ("completed", "diverged", "cap_saturated", "non_terminating")
CHECKPOINTS = ("final", "best_on_validation")
ACTION_MODES = ("deterministic", "sampled")
TUNING = ("shared", "per_scenario")
MIN_RUNS = 5


@dataclass(frozen=True, kw_only=True)
class TrainingRun:
    """One launched training run and what it cost.

    Parameters
    ----------
    seed : int
        The run's training seed.
    status : str
        One of `RUN_STATUSES`; any but "completed" is a failed run.
    checkpoint : str
        "final" or "best_on_validation", the weights scored.
    env_steps : int
        Environment steps in training.
    simulated_time_s : float
        Simulated time in training (s).
    wall_clock_s : float
        Wall-clock training time (s).
    violations : int
        Hard-constraint violations during training.
    propellant_kg : float
        Propellant used during training (kg).

    Raises
    ------
    ValueError
        If the status or checkpoint is unknown or a cost is negative or not finite.
    """

    seed: int
    status: str
    checkpoint: str = "final"
    env_steps: int
    simulated_time_s: float
    wall_clock_s: float
    violations: int
    propellant_kg: float

    def __post_init__(self):
        if self.status not in RUN_STATUSES:
            raise ValueError(f"unknown run status {self.status!r}, expected one of {RUN_STATUSES}")
        if self.checkpoint not in CHECKPOINTS:
            raise ValueError(f"unknown checkpoint {self.checkpoint!r}, expected {CHECKPOINTS}")
        for name in (
            "env_steps",
            "simulated_time_s",
            "wall_clock_s",
            "violations",
            "propellant_kg",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be finite and not negative")

    @property
    def failed(self) -> bool:
        """Whether the run diverged, saturated its cap or did not terminate."""
        return self.status != "completed"


@dataclass(frozen=True, kw_only=True)
class LearnedEntry:
    """A learned agent as it must be declared for scoring.

    Parameters
    ----------
    name : str
        The entry's name.
    algorithm : str
        The learning algorithm (for example "PPO").
    runs : sequence of TrainingRun
        Every training run launched, at least `MIN_RUNS`.
    action_mode : str
        "deterministic" (the default scored mode) or "sampled".
    action_seeds : sequence of int, optional
        Fixed action-sampling seeds, required when `action_mode` is "sampled".
    tau_s : float, optional
        The learner's discount as a time constant (s): gamma = exp(-dt / tau); None for no
        discount.
    trace_lambda : float, optional
        Lambda or GAE lambda, in [0, 1].
    reward_components : mapping
        Training-reward component to weight.
    shaping : str, optional
        The shaping potential, described.
    stabilizers : mapping
        Target networks, replay ratio, n-step targets and the like, name to setting.
    tuning : str
        "shared" (one set of hyperparameters for every scenario in `tuned_on`) or
        "per_scenario".
    tuned_on : sequence of str
        The scenarios the hyperparameters were tuned on.
    configurations_tried : int
        Hyperparameter configurations tried, each training-reward variant counting as one.
    offline : bool
        Whether the agent was trained from a fixed data set.
    data_source : str, optional
        Where offline data came from; required when `offline`.

    Raises
    ------
    ValueError
        If fewer than `MIN_RUNS` runs are declared, two runs share a seed, a sampled mode has no
        seeds, a field is out of range, or an offline entry names no data source.
    """

    name: str
    algorithm: str
    runs: tuple[TrainingRun, ...]
    action_mode: str = "deterministic"
    action_seeds: tuple[int, ...] | None = None
    tau_s: float | None = None
    trace_lambda: float | None = None
    reward_components: Mapping[str, float] = field(default_factory=dict)
    shaping: str | None = None
    stabilizers: Mapping[str, Any] = field(default_factory=dict)
    tuning: str = "shared"
    tuned_on: tuple[str, ...] = ()
    configurations_tried: int = 1
    offline: bool = False
    data_source: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "runs", tuple(self.runs))
        object.__setattr__(self, "tuned_on", tuple(self.tuned_on))
        if len(self.runs) < MIN_RUNS:
            raise ValueError(
                f"a learned entry reports at least {MIN_RUNS} training runs, not {len(self.runs)}"
            )
        if len({run.seed for run in self.runs}) != len(self.runs):
            raise ValueError("training runs need distinct seeds")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"unknown action mode {self.action_mode!r}, expected {ACTION_MODES}")
        if self.action_mode == "sampled" and not self.action_seeds:
            raise ValueError("a sampled action mode is scored with fixed action seeds")
        if self.tau_s is not None and not (math.isfinite(self.tau_s) and self.tau_s > 0.0):
            raise ValueError("tau_s must be finite and positive")
        if self.trace_lambda is not None and not 0.0 <= self.trace_lambda <= 1.0:
            raise ValueError("trace_lambda lies in [0, 1]")
        if self.tuning not in TUNING:
            raise ValueError(f"unknown tuning {self.tuning!r}, expected {TUNING}")
        if self.tuning == "shared" and not self.tuned_on:
            raise ValueError("shared hyperparameters name the scenarios they were tuned on")
        if isinstance(self.configurations_tried, bool) or self.configurations_tried < 1:
            raise ValueError("configurations_tried must be at least 1")
        if self.offline and not self.data_source:
            raise ValueError("an offline entry names its data source")

    @property
    def failed_runs(self) -> int:
        """Number of runs that diverged, saturated their cap or did not terminate."""
        return sum(run.failed for run in self.runs)

    def gamma(self, dt: float) -> float:
        """The discount per decision of `dt` seconds, exp(-dt / tau_s), or 1 without one.

        Parameters
        ----------
        dt : float
            Decision interval (s).

        Returns
        -------
        float
            The per-step discount.
        """
        return 1.0 if self.tau_s is None else math.exp(-dt / self.tau_s)


def run_scores(
    entry: LearnedEntry, scores: Mapping[int, float | None], failed_score: float = 0.0
) -> np.ndarray:
    """One score per training run, in the entry's order; failed runs score `failed_score`.

    Parameters
    ----------
    entry : LearnedEntry
        The entry.
    scores : mapping
        Training seed to the run's evaluation score (the success rate of its weights over the
        suite, say); a failed run may have None.
    failed_score : float
        The score a failed run counts with (0 for a rate).

    Returns
    -------
    numpy.ndarray
        Scores, shape (len(entry.runs),).

    Raises
    ------
    ValueError
        If a completed run has no finite score, or `scores` names a run the entry does not
        declare (a run cannot be swapped for a rerun).
    """
    known = {run.seed for run in entry.runs}
    unknown = set(scores) - known
    if unknown:
        raise ValueError(f"scores for undeclared training runs {sorted(unknown)}")
    out = []
    for run in entry.runs:
        value = scores.get(run.seed)
        if run.failed:
            out.append(failed_score)
            continue
        if value is None or not math.isfinite(value):
            raise ValueError(f"completed training run {run.seed} has no score")
        out.append(float(value))
    return np.array(out)


def normalized(
    values: ArrayLike, reference_low: ArrayLike, reference_high: ArrayLike
) -> np.ndarray:
    """Scores normalised per start state: (value - low) / (high - low).

    `reference_low` and `reference_high` are the scores of two references on the same start
    states (free drift and an oracle, say), so a cross-scenario IQM averages comparable
    numbers.

    Parameters
    ----------
    values, reference_low, reference_high : array_like
        Scores per start state, same shape.

    Returns
    -------
    numpy.ndarray
        The normalised scores.

    Raises
    ------
    ValueError
        If the shapes differ or a reference pair is equal or not finite.
    """
    x, low, high = (np.asarray(a, dtype=float) for a in (values, reference_low, reference_high))
    if not x.shape == low.shape == high.shape:
        raise ValueError("values and references need the same shape")
    span = high - low
    if not np.all(np.isfinite(span) & (span != 0.0)):
        raise ValueError("each reference pair must be finite and distinct")
    return (x - low) / span


def by_tuning(entries: Sequence[LearnedEntry]) -> dict[str, list[LearnedEntry]]:
    """Entries split by how their hyperparameters were tuned, for separate aggregates.

    Parameters
    ----------
    entries : sequence of LearnedEntry
        The entries.

    Returns
    -------
    dict
        "shared" and "per_scenario" to their entries.
    """
    return {kind: [e for e in entries if e.tuning == kind] for kind in TUNING}


__all__ = [
    "ACTION_MODES",
    "CHECKPOINTS",
    "MIN_RUNS",
    "RUN_STATUSES",
    "TUNING",
    "LearnedEntry",
    "TrainingRun",
    "by_tuning",
    "normalized",
    "run_scores",
]
