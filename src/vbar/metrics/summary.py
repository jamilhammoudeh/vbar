"""Summaries of one metric across episodes: interquartile mean, unsafe tail, rate.

Each metric's catalogue entry says which one ranks it. Performance metrics use the interquartile
mean; safety metrics use the unsafe tail (the worst value, the 5th percentile on the unsafe side and
the mean beyond it), because the interquartile mean drops the quarter of episodes where near misses
are (Kochenderfer et al., Algorithms for Decision Making, 2022, section 14.1). Confidence intervals
are left to the statistics of the evaluation harness.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import beta

from .filter import iqm
from .registry import METRICS


def unsafe_tail(values: ArrayLike, worse: str, level: float = 0.05) -> dict[str, float]:
    """Worst value, the `level` quantile on the unsafe side, and the mean beyond it.

    `worse` is "low" when lower values are less safe (margins) and "high" when higher are (ratios,
    durations). The tail mean averages the values at or beyond the quantile, a conditional value
    at risk; it always holds at least the worst value. NaN values are left out.
    """
    if worse not in ("low", "high"):
        raise ValueError(f"unknown side {worse!r}, expected low or high")
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"worst": np.nan, "quantile": np.nan, "tail_mean": np.nan}
    if worse == "low":
        q = float(np.quantile(x, level))
        return {"worst": float(x.min()), "quantile": q, "tail_mean": float(x[x <= q].mean())}
    q = float(np.quantile(x, 1.0 - level))
    return {"worst": float(x.max()), "quantile": q, "tail_mean": float(x[x >= q].mean())}


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for a success rate from `successes` of `trials`."""
    if not 0 <= successes <= trials or trials == 0:
        raise ValueError("need 0 <= successes <= trials and trials > 0")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else beta.ppf(0.5 * alpha, successes, trials - successes + 1)
    upper = (
        1.0 if successes == trials else beta.ppf(1 - 0.5 * alpha, successes + 1, trials - successes)
    )
    return float(lower), float(upper)


def summarize(name: str, values: ArrayLike) -> dict[str, float]:
    """The ranking summary of metric `name` over episode values, as its catalogue entry says."""
    kind = METRICS[name].summary
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if kind == "lower_tail":
        return unsafe_tail(x, "low")
    if kind == "upper_tail":
        return unsafe_tail(x, "high")
    if kind == "rate":
        if len(x) == 0:
            return {"rate": np.nan, "count": 0.0, "n": 0, "lower": np.nan, "upper": np.nan}
        count = int(round(float(x.sum())))
        lower, upper = clopper_pearson(count, len(x))
        return {
            "rate": float(x.mean()),
            "count": float(count),
            "n": len(x),
            "lower": lower,
            "upper": upper,
        }
    if kind == "iqm":
        return {"iqm": iqm(x) if len(x) else np.nan}
    raise ValueError(f"{name} is not summarised across episodes")
