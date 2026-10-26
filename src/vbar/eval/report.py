"""The report of a suite: complete or refused, every metric with its interval.

Per scenario, success is a rate with its Clopper-Pearson interval, performance metrics are
interquartile means with bootstrap intervals, and safety metrics are their unsafe tails with
distribution-free intervals, never an interquartile mean. A suite is reported in full or not at
all, and the report carries one hash over the provenance of every run in it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..metrics import clopper_pearson
from .provenance import combined_hash
from .runner import by_scenario
from .stats import summary_interval
from .tails import tail_summary


def default_summaries() -> dict[str, str]:
    """How each V-BAR metric ranks across runs, from the metric catalogue.

    Returns
    -------
    dict
        Metric name to "iqm", "lower_tail", "upper_tail" or "rate"; metrics summarised by none
        are left out.
    """
    from ..metrics import METRICS

    return {name: info.summary for name, info in METRICS.items() if info.summary != "none"}


def _column(results: Sequence[Mapping[str, Any]], name: str) -> list[Any]:
    return [r["metrics"].get(name, math.nan) for r in results]


def _numbers(values: Sequence[Any]) -> np.ndarray:
    return np.array(
        [
            float(v) if isinstance(v, int | float) and not isinstance(v, bool) else math.nan
            for v in values
        ]
    )


def _summarise(values: Sequence[Any], kind: str, reps: int, rng: int) -> dict[str, Any] | None:
    if kind == "rate":
        flags = [v for v in values if isinstance(v, bool)]
        if not flags:
            return None
        count = sum(flags)
        lower, upper = clopper_pearson(count, len(flags))
        return {
            "rate": count / len(flags),
            "count": count,
            "n": len(flags),
            "lower": lower,
            "upper": upper,
        }
    x = _numbers(values)
    kept = x[~np.isnan(x)]
    if len(kept) == 0:
        return None
    if kind == "iqm":
        found = summary_interval(kept, reps=reps, rng=rng)
        return {
            "iqm": found.estimate,
            "lower": found.lower,
            "upper": found.upper,
            "n": len(kept),
            "missing": len(x) - len(kept),
        }
    tail = tail_summary(x, "low" if kind == "lower_tail" else "high", reps=reps, rng=rng)
    return tail


def scenario_summary(
    results: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, str] | None = None,
    reps: int = 10_000,
    rng: int = 0,
) -> dict[str, Any]:
    """Every metric of one scenario's runs, summarised as its catalogue entry says.

    Parameters
    ----------
    results : sequence of mapping
        The scenario's results.
    summaries : mapping, optional
        Metric name to summary kind; the V-BAR catalogue by default.
    reps : int
        Bootstrap resamples.
    rng : int
        Seed of the bootstrap, so a report is reproducible.

    Returns
    -------
    dict
        "runs", "errors", "failure_labels" (label to count) and "metrics" (name to summary; a
        per-constraint metric to one summary per constraint).
    """
    summaries = default_summaries() if summaries is None else summaries
    labels: dict[str, int] = {}
    for result in results:
        label = result.get("failure_label")
        if label is not None:
            labels[label] = labels.get(label, 0) + 1
    metrics: dict[str, Any] = {}
    for name, kind in summaries.items():
        values = _column(results, name)
        if any(isinstance(v, Mapping) for v in values):
            names = sorted({c for v in values if isinstance(v, Mapping) for c in v})
            per = {}
            for c in names:
                found = _summarise(
                    [v.get(c, math.nan) if isinstance(v, Mapping) else math.nan for v in values],
                    kind,
                    reps,
                    rng,
                )
                if found is not None:
                    per[c] = found
            metrics[name] = per
            continue
        found = _summarise(values, kind, reps, rng)
        if found is not None:
            metrics[name] = found
    return {
        "runs": len(results),
        "errors": sum(r.get("error") is not None for r in results),
        "failure_labels": dict(sorted(labels.items())),
        "metrics": metrics,
    }


def suite_report(
    suite,
    results: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, str] | None = None,
    reps: int = 10_000,
    rng: int = 0,
    task: Any = None,
) -> dict[str, Any]:
    """The report of one agent and filter on a whole suite.

    Parameters
    ----------
    suite : Suite
        The suite; results missing any of its runs are refused.
    results : sequence of mapping
        Results with "scenario", "scenario_id", "seed", "agent", "filter", "metrics",
        "failure_label", "error" and "provenance".
    summaries : mapping, optional
        Metric name to summary kind; the V-BAR catalogue by default.
    reps : int
        Bootstrap resamples per interval.
    rng : int
        Seed of every bootstrap.
    task : Task, optional
        The task the runs were flown with, whose environments name the identifiers the suite's
        runs must carry; the catalogue's by default.

    Returns
    -------
    dict
        "suite", "agent", "filter", "runs", "provenance" (hash over the runs' hashes) and
        "scenarios" (entry name to `scenario_summary`), entries in suite order.

    Raises
    ------
    ValueError
        If a run of the suite is missing or repeated, the results mix agents or filters, or a run
        has no provenance.
    """
    identifiers = None
    if task is not None:
        identifiers = {
            e.scenario: task.identity(task.make(e.scenario))["scenario_id"] for e in suite.entries
        }
    suite.require_complete(results, identifiers)
    wanted = {(sid, seed) for sid, seed, _ in suite.runs(identifiers)}
    kept = [r for r in results if (r["scenario_id"], int(r["seed"])) in wanted]
    if len(kept) != len(wanted):
        raise ValueError("a report takes one result per run; the results hold repeated runs")
    agents = {(r["agent"], r["filter"]) for r in kept}
    if len(agents) != 1:
        raise ValueError(f"a report covers one agent and filter, not {sorted(map(str, agents))}")
    if any(r.get("provenance") is None for r in kept):
        raise ValueError("every run in a report needs its provenance")
    groups = by_scenario(kept)
    agent, safety_filter = next(iter(agents))
    return {
        "suite": suite.id,
        "agent": agent,
        "filter": safety_filter,
        "runs": len(kept),
        "provenance": combined_hash(r["provenance"] for r in kept),
        "scenarios": {
            entry.scenario: scenario_summary(groups[entry.scenario], summaries, reps, rng)
            for entry in suite.entries
        },
    }


def _number(value: float) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.4g}"


def to_markdown(report: Mapping[str, Any], metrics: Sequence[str] | None = None) -> str:
    """A report as Markdown tables: success, then each chosen metric per scenario.

    Parameters
    ----------
    report : mapping
        A `suite_report`.
    metrics : sequence of str, optional
        Metrics to tabulate besides success; by default delta-v, time to complete and the
        keep-out margin, as available.

    Returns
    -------
    str
        The Markdown text.
    """
    chosen = list(metrics or ("delta_v", "time_to_complete", "min_margin"))
    lines = [
        f"# {report['suite']}: {report['agent']}"
        + ("" if report["filter"] is None else f" with {report['filter']}"),
        "",
        f"{report['runs']} runs; provenance `{report['provenance']}`.",
        "",
        "| Scenario | Success (95% CI) | " + " | ".join(chosen) + " |",
        "|---|---|" + "---|" * len(chosen),
    ]
    for name, row in report["scenarios"].items():
        found = row["metrics"]
        rate = found.get("success")
        cells = [
            "n/a"
            if rate is None
            else f"{rate['count']}/{rate['n']} ({rate['lower']:.3f} to {rate['upper']:.3f})"
        ]
        for metric in chosen:
            cells.append(_cell(found.get(metric)))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _cell(summary: Any) -> str:
    if summary is None:
        return "n/a"
    if "iqm" in summary:
        low, high = _number(summary["lower"]), _number(summary["upper"])
        return f"{_number(summary['iqm'])} ({low} to {high})"
    if "worst" in summary:
        low, high = (_number(v) for v in summary["quantile_interval"])
        worst, quantile = _number(summary["worst"]), _number(summary["quantile"])
        return f"worst {worst}, 5% {quantile} ({low} to {high})"
    if all(isinstance(v, Mapping) for v in summary.values()):
        return "; ".join(f"{k}: {_cell(v)}" for k, v in summary.items())
    return "n/a"


__all__ = ["default_summaries", "scenario_summary", "suite_report", "to_markdown"]
