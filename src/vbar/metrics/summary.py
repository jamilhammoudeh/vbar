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
        return {"rate": float(x.mean()) if len(x) else np.nan, "count": float(x.sum()), "n": len(x)}
    if kind == "iqm":
        return {"iqm": iqm(x) if len(x) else np.nan}
    raise ValueError(f"{name} is not summarised across episodes")
