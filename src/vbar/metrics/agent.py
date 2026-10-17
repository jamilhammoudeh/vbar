"""Agent metrics A1 to A8: latency, compute, tokens, parse failures and faithfulness.

Latency comes from the step records; LLM quantities from the agent's own per-decision log
(`Episode.agent_log`), whose entries may carry `parse_failed`, `reply`, `input_tokens` and
`output_tokens`. The sign check and training figures are reported by the harness that ran them.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .episode import Episode

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


def agent_metrics(ep: Episode, report: AgentReport | None = None) -> dict[str, Any]:
    """A1 to A8 for one episode."""
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
        "sign_check_pass": math.nan if report.sign_check_pass is None else report.sign_check_pass,
        "training_cost": dict(report.training_cost) if report.training_cost else None,
        "cost_of_learning": dict(report.cost_of_learning) if report.cost_of_learning else None,
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
