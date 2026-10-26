"""Safety tails with intervals, tolerance bounds, and success-rate comparisons.

A safety metric is reported by its unsafe tail (`vbar.metrics.unsafe_tail`): the worst run, the
5th percentile on the unsafe side and the mean beyond it. Their intervals are distribution free.
The percentile's is a pair of order statistics whose coverage is a binomial sum in n alone; the
worst run is bounded by a one-sided tolerance bound, the order statistic that lies beyond the
5th percentile with the stated confidence; the tail mean takes a bootstrap. Split-conformal
bounds are not used: their coverage holds on average over calibration sets, and given one set it
falls short almost half the time (Kochenderfer, Katz, Corso and Moss, Algorithms for Validation,
2026, sections 7.1 and 12.2). Success rates carry Clopper-Pearson intervals, and an unpaired claim
that two rates differ by less than their interval width is not resolved.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binom

from ..metrics import clopper_pearson, unsafe_tail
from .stats import _check, _generator, _percentile


def tolerance_order(n: int, coverage: float, confidence: float) -> int | None:
    """The order statistic giving a one-sided lower (coverage, confidence) tolerance bound.

    The k-th smallest of n runs lies at or below the (1 - coverage) quantile with probability
    P(Binomial(n, 1 - coverage) >= k); this is the largest k for which that is at least
    `confidence`.

    Parameters
    ----------
    n : int
        Number of runs.
    coverage : float
        Fraction of the population the bound must lie under, in (0, 1).
    confidence : float
        Confidence level, in (0, 1).

    Returns
    -------
    int or None
        The 1-based rank k, or None when even the smallest run is not a bound with this
        confidence (fewer runs than the zero-failure count).

    Raises
    ------
    ValueError
        If `coverage` or `confidence` is outside (0, 1).
    """
    if not (0.0 < coverage < 1.0 and 0.0 < confidence < 1.0):
        raise ValueError("coverage and confidence must lie between 0 and 1")
    if n < 1:
        return None
    ranks = np.arange(1, n + 1)
    holds = binom.sf(ranks - 1, n, 1.0 - coverage) >= confidence
    return int(ranks[holds][-1]) if np.any(holds) else None


def conformal_coverage_risk(n: int, coverage: float) -> float:
    """Probability that a split-conformal lower bound covers less than `coverage`.

    Split conformal takes the order statistic floor((1 - coverage)(n + 1)); given the
    calibration runs its coverage falls short when fewer runs than that lie below the
    (1 - coverage) quantile.

    Parameters
    ----------
    n : int
        Number of calibration runs.
    coverage : float
        Target coverage, in (0, 1).

    Returns
    -------
    float
        P(Binomial(n, 1 - coverage) <= rank - 1).
    """
    rank = math.floor((1.0 - coverage) * (n + 1))
    return float(binom.cdf(rank - 1, n, 1.0 - coverage))


def _quantile_ranks(n: int, level: float, confidence: float) -> tuple[int | None, int | None]:
    """1-based ranks (l, u) with P(X_(l) <= q_level <= X_(u)) >= confidence, equal tails."""
    alpha = 0.5 * (1.0 - confidence)
    ranks = np.arange(1, n + 1)
    # P(X_(l) > q) = P(B <= l - 1) and P(X_(u) < q) = P(B >= u), B ~ Binomial(n, level).
    low_ok = binom.cdf(ranks - 1, n, level) <= alpha
    high_ok = binom.sf(ranks - 1, n, level) <= alpha
    lower = int(ranks[low_ok][-1]) if np.any(low_ok) else None
    upper = int(ranks[high_ok][0]) if np.any(high_ok) else None
    return lower, upper


def _lower_tail_means(rows: np.ndarray, level: float) -> np.ndarray:
    """The lower tail mean of each sorted row, as `unsafe_tail` computes it for one sample."""
    n = rows.shape[1]
    # The inverse empirical distribution at `level`: the ceil(level n)-th smallest value.
    k = max(math.ceil(level * n) - 1, 0)
    inside = rows <= rows[:, k : k + 1]
    return np.where(inside, rows, 0.0).sum(axis=1) / inside.sum(axis=1)


def tail_summary(
    values: ArrayLike,
    worse: str,
    level: float = 0.05,
    confidence: float = 0.95,
    reps: int = 10_000,
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """The unsafe tail of a safety metric with distribution-free intervals.

    Parameters
    ----------
    values : array_like
        One value per run; NaN values are left out and counted.
    worse : {"low", "high"}
        "low" for margins (lower is less safe), "high" for violations and ratios.
    level : float
        Tail probability (0.05: the 5th percentile on the unsafe side).
    confidence : float
        Confidence level of every interval and bound.
    reps : int
        Bootstrap resamples of the tail mean, at least 100.
    rng : numpy.random.Generator or int, optional
        Generator or seed of the bootstrap.

    Returns
    -------
    dict
        "worst", "quantile", "tail_mean", "n" and "missing" as `vbar.metrics.unsafe_tail`;
        "quantile_interval" (lower, upper), infinite on a side where too few runs bound it;
        "tail_mean_interval" (lower, upper), the bootstrap percentile interval; "worst_bound",
        the one-sided tolerance bound on the unsafe `level` quantile at `confidence`, NaN with too
        few runs.

    Raises
    ------
    ValueError
        If `worse` is unknown or no value is finite or infinite (all NaN, or none).
    """
    if worse not in ("low", "high"):
        raise ValueError(f"unknown side {worse!r}, expected low or high")
    _check(reps, confidence)
    x = np.asarray(values, dtype=float).ravel()
    tail = unsafe_tail(x, worse, level)
    kept = x[~np.isnan(x)]
    if len(kept) == 0:
        raise ValueError("a tail needs at least one value")
    # The upper tail is the lower tail of the negated values.
    sign = 1.0 if worse == "low" else -1.0
    ordered = np.sort(sign * kept)
    n = len(ordered)
    lo, hi = _quantile_ranks(n, level, confidence)
    bounds = (
        -math.inf if lo is None else ordered[lo - 1],
        math.inf if hi is None else ordered[hi - 1],
    )
    if sign < 0:
        bounds = (-bounds[1], -bounds[0])
    rank = tolerance_order(n, 1.0 - level, confidence)
    worst_bound = math.nan if rank is None else float(sign * ordered[rank - 1])
    gen = _generator(rng)
    draws = gen.integers(0, n, size=(reps, n))
    means = sign * _lower_tail_means(np.sort(sign * kept[draws], axis=1), level)
    return {
        **tail,
        "quantile_interval": (float(bounds[0]), float(bounds[1])),
        "tail_mean_interval": _percentile(means, confidence),
        "worst_bound": worst_bound,
    }


def compare_rates(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    confidence: float = 0.95,
    require_resolved: bool = False,
) -> dict[str, Any]:
    """Two success rates with their Clopper-Pearson intervals, and whether they are told apart.

    An unpaired difference smaller than the wider of the two interval widths is not resolved:
    the runs cannot support a claim that one rate is higher.

    Parameters
    ----------
    successes_a, n_a, successes_b, n_b : int
        Successes and runs of each.
    confidence : float
        Confidence level of the intervals.
    require_resolved : bool
        Raise when the difference is not resolved.

    Returns
    -------
    dict
        "interval_a" and "interval_b" (lower, upper), "difference" (rate a minus rate b) and
        "resolved".

    Raises
    ------
    ValueError
        If a count is impossible, or `require_resolved` and the difference is smaller than the
        interval width.
    """
    interval_a = clopper_pearson(successes_a, n_a, confidence)
    interval_b = clopper_pearson(successes_b, n_b, confidence)
    difference = successes_a / n_a - successes_b / n_b
    width = max(interval_a[1] - interval_a[0], interval_b[1] - interval_b[0])
    resolved = abs(difference) > width
    if require_resolved and not resolved:
        raise ValueError(
            f"the rates differ by {abs(difference):.3g}, smaller than the interval width "
            f"{width:.3g}; more runs are needed for the claim"
        )
    return {
        "interval_a": interval_a,
        "interval_b": interval_b,
        "difference": difference,
        "resolved": bool(resolved),
    }


__all__ = ["compare_rates", "conformal_coverage_risk", "tail_summary", "tolerance_order"]
