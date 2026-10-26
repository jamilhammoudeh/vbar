"""Summaries of one metric across episodes: interquartile mean, unsafe tail, rate.

Each metric's catalogue entry says which one ranks it. Performance metrics use the interquartile
mean; safety metrics use the unsafe tail (the worst value, the 5th percentile on the unsafe side and
the mean beyond it), because the interquartile mean drops the quarter of episodes where near misses
are (Kochenderfer et al., Algorithms for Decision Making, 2022, section 14.1). Confidence intervals
are left to the statistics of the evaluation harness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import beta

from .filter import iqm
from .registry import METRICS

# Metrics every episode has: a NaN is a broken run, which must count as one, not drop out.
_DEFINED = ("composite_score", "progress", "success")


def unsafe_tail(values: ArrayLike, worse: str, level: float = 0.05) -> dict[str, float]:
    """Worst value, the `level` quantile on the unsafe side, and the mean beyond it.

    `worse` is "low" when lower values are less safe (margins) and "high" when higher are (ratios,
    durations). The quantile is an episode value (the inverse of the empirical distribution, no
    interpolation), so infinite values, which S5, S6 and S4t give by design, are not mixed with
    finite ones. The tail mean averages the values at or beyond it, a conditional value at risk;
    it always holds at least the worst value. NaN values are left out and counted: `n` is the
    number of values summarised and `missing` the number of NaN values left out.
    """
    if worse not in ("low", "high"):
        raise ValueError(f"unknown side {worse!r}, expected low or high")
    x = np.asarray(values, dtype=float)
    counts = {"n": int(np.sum(~np.isnan(x))), "missing": int(np.sum(np.isnan(x)))}
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"worst": np.nan, "quantile": np.nan, "tail_mean": np.nan, **counts}
    if worse == "low":
        q = float(np.quantile(x, level, method="inverted_cdf"))
        tail = {"worst": float(x.min()), "quantile": q, "tail_mean": float(x[x <= q].mean())}
        return {**tail, **counts}
    # Mirrored, so the upper tail holds as many episodes as the lower one would.
    q = -float(np.quantile(-x, level, method="inverted_cdf"))
    tail = {"worst": float(x.max()), "quantile": q, "tail_mean": float(x[x >= q].mean())}
    return {**tail, **counts}


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for a success rate from `successes` of `trials`."""
    if successes != int(successes) or trials != int(trials):
        raise ValueError("successes and trials must be whole numbers")
    if not 0 <= successes <= trials or trials == 0:
        raise ValueError("need 0 <= successes <= trials and trials > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between 0 and 1")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else beta.ppf(0.5 * alpha, successes, trials - successes + 1)
    upper = (
        1.0 if successes == trials else beta.ppf(1 - 0.5 * alpha, successes + 1, trials - successes)
    )
    return float(lower), float(upper)


def summarize(name: str, values: ArrayLike | Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """The ranking summary of metric `name` over episode values, as its catalogue entry says.

    Per-constraint metrics (S1 to S3) come as one mapping per episode; each constraint is then
    summarised on its own, a constraint missing from an episode, or an episode given as a number
    (NaN) instead of a mapping, counting as NaN. NaN values are left out of a summary and
    counted in its `missing` entry, beside `n`, the number summarised; the success, progress and
    composite score of every episode are defined, so a NaN among them raises `ValueError`.
    """
    if isinstance(values, Sequence) and any(isinstance(v, Mapping) for v in values):
        per_episode = [v if isinstance(v, Mapping) else {} for v in values]
        names = sorted({c for v in per_episode for c in v})
        return {c: summarize(name, [v.get(c, np.nan) for v in per_episode]) for c in names}
    kind = METRICS[name].summary
    x = np.asarray(values, dtype=float)
    missing = int(np.sum(np.isnan(x)))
    if missing and name in _DEFINED:
        raise ValueError(f"{name} is NaN in {missing} episodes")
    x = x[~np.isnan(x)]
    if kind == "lower_tail":
        return {**unsafe_tail(x, "low"), "missing": missing}
    if kind == "upper_tail":
        return {**unsafe_tail(x, "high"), "missing": missing}
    if kind == "rate":
        if len(x) == 0:
            return {
                "rate": np.nan,
                "count": 0.0,
                "n": 0,
                "lower": np.nan,
                "upper": np.nan,
                "missing": missing,
            }
        if not np.all((x == 0.0) | (x == 1.0)):
            raise ValueError(f"{name} is a rate, so its values must be 0 or 1")
        count = int(x.sum())
        lower, upper = clopper_pearson(count, len(x))
        return {
            "rate": float(x.mean()),
            "count": float(count),
            "n": len(x),
            "lower": lower,
            "upper": upper,
            "missing": missing,
        }
    if kind == "iqm":
        return {"iqm": iqm(x) if len(x) else np.nan, "n": len(x), "missing": missing}
    raise ValueError(f"{name} is not summarised across episodes")
