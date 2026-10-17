"""Agent metrics A1 to A9: latency, compute, tokens, parse failures, faithfulness and solvers.

Latency comes from the step records; LLM quantities from the agent's own per-decision log
(`Episode.agent_log`), whose entries may carry `parse_failed`, `reply`, `input_tokens` and
`output_tokens`, and a model-based agent's solver fields `solver`, `tolerance`, `status`,
`iterations`, `warm_start`, `fallback`, `prep_s` and `feedback_s`. The sign check and training
figures are reported by the harness that ran them.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .episode import Episode

Posterior = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

TRAINING_COST_KEYS = ("env_steps", "simulated_time_s", "wall_clock_s", "compute")
LEARNING_COST_KEYS = ("constraint_violations", "propellant_kg", "learning_curve_area")


@dataclass(frozen=True, kw_only=True)
class AgentReport:
    """Facts about the agent from outside the episode.

    `sign_check_pass` is the result of the convention self-check, `training_cost` and
    `cost_of_learning` a learned agent's training figures, and `token_prices` the declared price
    in USD per million input and output tokens.
    """

    sign_check_pass: bool | None = None
    training_cost: Mapping[str, Any] | None = None
    cost_of_learning: Mapping[str, Any] | None = None
    token_prices: tuple[float, float] | None = None

    def __post_init__(self):
        for name, keys in (
            ("training_cost", TRAINING_COST_KEYS),
            ("cost_of_learning", LEARNING_COST_KEYS),
        ):
            given = getattr(self, name)
            if given is not None and set(keys) - set(given):
                raise ValueError(f"{name} needs {sorted(set(keys) - set(given))}")


def parse_belief(text: str) -> dict[str, Any] | None:
    """The first JSON object in a reply with a numeric `range_m` and a boolean `closing`."""
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            data = json.loads(match.group(0))
        except ValueError:
            continue
        range_m, closing = data.get("range_m"), data.get("closing")
        if isinstance(range_m, int | float) and isinstance(closing, bool):
            return {"range_m": float(range_m), "closing": closing}
    return None


def _decision_observations(ep: Episode) -> list[np.ndarray]:
    """The observation each decision was made on: the first one, then each step's."""
    if ep.first_obs is None:
        raise ValueError("faithfulness needs the episode's first observation")
    return [ep.first_obs, *(rec.obs for rec in ep.records[:-1])]


def faithfulness(ep: Episode) -> float:
    """A5: share of decisions whose stated belief matches the observation the agent received.

    The log holds one entry per decision, in order. A belief matches when its range is within 10%
    of the observed range and its closing flag agrees with the sign of r . v in the observation.
    A decision without a belief does not match; NaN without an agent log.
    """
    if not ep.agent_log:
        return math.nan
    hits = 0
    for entry, obs in zip(ep.agent_log, _decision_observations(ep), strict=True):
        belief = parse_belief(str(entry.get("reply", "")))
        if belief is None:
            continue
        rho = float(np.linalg.norm(obs[:3]))
        closing = float(obs[:3] @ obs[3:6]) < 0.0
        hits += abs(belief["range_m"] - rho) <= 0.1 * rho and belief["closing"] == closing
    return hits / len(ep.agent_log)


def belief_consistency(ep: Episode, posterior: Posterior | None) -> float:
    """A5b: root mean square of the stated range's error from a reference posterior, in its sigmas.

    `posterior` maps the observations up to and including a decision (rows, in order) to the mean
    (m, Hill) and covariance (m^2) of the position, from those observations alone. The range's
    standard deviation is the first-order sqrt(u' P u), u the unit mean. Over the decisions with a
    stated belief; 0 when every stated range is the posterior's; NaN without a posterior, a log or
    a belief. An agent that restates each noisy fix scores A5 = 1 and a large A5b; one that states
    few beliefs is judged on those few, so A5b is read with A5, which counts the missing ones.
    """
    if posterior is None or not ep.agent_log:
        return math.nan
    observations = np.array(_decision_observations(ep))
    errors = []
    for k, (entry, _) in enumerate(zip(ep.agent_log, observations, strict=True)):
        belief = parse_belief(str(entry.get("reply", "")))
        if belief is None:
            continue
        mean, cov = posterior(observations[: k + 1])
        mean = np.asarray(mean, dtype=float)[:3]
        rho = float(np.linalg.norm(mean))
        if rho == 0.0:
            raise ValueError("the posterior mean is at the target, so it has no range direction")
        u = mean / rho
        variance = float(u @ np.asarray(cov, dtype=float)[:3, :3] @ u)
        if not variance > 0.0:
            raise ValueError("the posterior range variance must be positive")
        errors.append((belief["range_m"] - rho) / math.sqrt(variance))
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else math.nan


def _reported(log: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    """Values of `key` over the log entries that report it."""
    return [e[key] for e in log if e.get(key) is not None]


def _amounts(log: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    """Reported values of `key`, each a finite number at least 0, since the agent writes them."""
    values = _reported(log, key)
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int | float) or not 0.0 <= v < math.inf:
            raise ValueError(f"log field {key} must be a finite number >= 0, got {v!r}")
    return [float(v) for v in values]


def _flags(log: Sequence[Mapping[str, Any]], key: str) -> list[bool]:
    """Reported values of `key`, each a bool."""
    values = _reported(log, key)
    for v in values:
        if not isinstance(v, bool | np.bool_):
            raise ValueError(f"log field {key} must be true or false, got {v!r}")
    return [bool(v) for v in values]


def solver_metrics(ep: Episode) -> dict[str, Any]:
    """A1r, A1f, A1fp and A9 to A9f from the solver fields of the agent's log.

    `prep_s` is the work done before the observation arrives and `feedback_s` the time from the
    observation to the command (s), the split of real-time iteration schemes (Rawlings, Mayne and
    Diehl, Model Predictive Control, 2nd ed., 2017, section 8.9). Each metric is over the entries
    that report its field; NaN, or empty, where none does. Times, tolerances and iteration counts
    must be finite numbers at least 0 and the warm-start and fallback flags bools.
    """
    log = ep.agent_log
    prep, feedback = _amounts(log, "prep_s"), _amounts(log, "feedback_s")
    iterations = _amounts(log, "iterations")
    warm, fallback = _flags(log, "warm_start"), _flags(log, "fallback")
    statuses: dict[str, int] = {}
    for status in _reported(log, "status"):
        statuses[str(status)] = statuses.get(str(status), 0) + 1
    return {
        "preparation_median": float(np.median(prep)) if prep else math.nan,
        "feedback_median": float(np.median(feedback)) if feedback else math.nan,
        "feedback_p95": float(np.percentile(feedback, 95)) if feedback else math.nan,
        "solver_metadata": {
            "solver": tuple(sorted({str(s) for s in _reported(log, "solver")})),
            "tolerance": tuple(sorted(set(_amounts(log, "tolerance")))),
            "status": dict(sorted(statuses.items())),
        },
        "solver_iterations_median": float(np.median(iterations)) if iterations else math.nan,
        "warm_start_rate": float(np.mean(warm)) if warm else math.nan,
        "agent_fallback_count": float(sum(fallback)) if fallback else math.nan,
    }


def agent_metrics(
    ep: Episode, report: AgentReport | None = None, posterior: Posterior | None = None
) -> dict[str, Any]:
    """A1 to A9 for one episode; A5b needs the reference `posterior` (see `belief_consistency`)."""
    report = report or AgentReport()
    latency = np.array([rec.agent_latency_s for rec in ep.records])
    log = ep.agent_log
    out: dict[str, Any] = {
        "latency_median": float(np.median(latency)) if len(latency) else math.nan,
        "latency_p95": float(np.percentile(latency, 95)) if len(latency) else math.nan,
        "compute": float(latency.sum()),
        "input_tokens": math.nan,
        "output_tokens": math.nan,
        "token_cost": math.nan,
        "parse_failure_rate": math.nan,
        "reasoning_faithfulness": faithfulness(ep),
        "belief_consistency": belief_consistency(ep, posterior),
        "sign_check_pass": math.nan if report.sign_check_pass is None else report.sign_check_pass,
        "training_cost": dict(report.training_cost) if report.training_cost else None,
        "cost_of_learning": dict(report.cost_of_learning) if report.cost_of_learning else None,
        **solver_metrics(ep),
    }
    if log:
        out["parse_failure_rate"] = float(np.mean([bool(e.get("parse_failed")) for e in log]))
    if log and all("input_tokens" in e and "output_tokens" in e for e in log):
        tokens_in = float(sum(e["input_tokens"] for e in log))
        tokens_out = float(sum(e["output_tokens"] for e in log))
        out.update(input_tokens=tokens_in, output_tokens=tokens_out)
        if report.token_prices is not None:
            price_in, price_out = report.token_prices
            out["token_cost"] = (tokens_in * price_in + tokens_out * price_out) / 1e6
    return out
