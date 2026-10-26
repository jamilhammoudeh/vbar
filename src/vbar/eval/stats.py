"""Scores across runs and scenarios: IQM, stratified bootstrap, profiles and comparisons.

The interquartile mean (the mean of the middle half of the runs) is barely moved by the few runs
that fail badly, and its interval comes from a bootstrap that resamples runs within each scenario,
so the spread between scenarios does not widen it (Agarwal et al., "Deep Reinforcement Learning at
the Edge of the Statistical Precipice", NeurIPS 2021). The normal interval is never used: for
rare events it goes below zero (Kochenderfer, Wheeler and Wray, Algorithms for Decision Making,
2022, section 14.1). Scenarios are taken in sorted order, so the order of a mapping never changes
a result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import rankdata, trim_mean

Samples = Mapping[str, ArrayLike]


@dataclass(frozen=True)
class Interval:
    """A point estimate and its confidence interval.

    Parameters
    ----------
    estimate : float
        The statistic on the data.
    lower, upper : float
        The interval's ends.
    """

    estimate: float
    lower: float
    upper: float

    @property
    def width(self) -> float:
        """upper - lower."""
        return self.upper - self.lower


def iqm(values: ArrayLike) -> float:
    """Interquartile mean: the mean of the middle half of the values.

    Parameters
    ----------
    values : array_like
        Scores, one per run.

    Returns
    -------
    float
        The mean after a quarter is trimmed from each end.

    Raises
    ------
    ValueError
        If there are no values.
    """
    x = np.asarray(values, dtype=float).ravel()
    if len(x) == 0:
        raise ValueError("the interquartile mean needs at least one value")
    return float(trim_mean(x, 0.25))


def _strata(samples: Samples, name: str = "samples") -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError(f"{name} hold no scenarios")
    out = {}
    for key in sorted(samples):
        values = np.asarray(samples[key], dtype=float).ravel()
        if len(values) == 0:
            raise ValueError(f"scenario {key!r} has no runs")
        out[key] = values
    return out


def aggregate_iqm(samples: Samples) -> float:
    """The interquartile mean of every run of every scenario, pooled.

    Parameters
    ----------
    samples : mapping
        Scenario to its runs' scores.

    Returns
    -------
    float
        The IQM of the pooled runs.

    Raises
    ------
    ValueError
        If there is no scenario or a scenario has no runs.
    """
    strata = _strata(samples)
    return iqm(np.concatenate(list(strata.values())))


def _generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _check(reps: int, confidence: float) -> None:
    if isinstance(reps, bool) or not isinstance(reps, int | np.integer) or reps < 100:
        raise ValueError("the bootstrap needs at least 100 resamples")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between 0 and 1")


def _percentile(values: np.ndarray, confidence: float) -> tuple[float, float]:
    # Order statistics of the resampled values, so infinite statistics stay values.
    alpha = 1.0 - confidence
    lower = np.quantile(values, 0.5 * alpha, method="inverted_cdf")
    upper = np.quantile(values, 1.0 - 0.5 * alpha, method="inverted_cdf")
    return float(lower), float(upper)


def stratified_bootstrap(
    samples: Samples,
    statistic: Callable[[Mapping[str, np.ndarray]], float],
    reps: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> Interval:
    """A statistic of per-scenario runs with its stratified percentile bootstrap interval.

    Each resample draws, with replacement, as many runs from each scenario as it has, scenario by
    scenario, and applies `statistic` to the resampled mapping.

    Parameters
    ----------
    samples : mapping
        Scenario to its runs' scores.
    statistic : callable
        Mapping of scenario to an array of scores, to a number.
    reps : int
        Number of resamples, at least 100.
    confidence : float
        Confidence level of the interval, in (0, 1).
    rng : numpy.random.Generator or int, optional
        Generator or seed of the resampling.

    Returns
    -------
    Interval
        The statistic on the data and the interval.

    Raises
    ------
    ValueError
        If a scenario has no runs, there is none, `reps` is below 100 or `confidence` is outside
        (0, 1).
    """
    strata = _strata(samples)
    _check(reps, confidence)
    gen = _generator(rng)
    draws = {k: gen.integers(0, len(v), size=(reps, len(v))) for k, v in strata.items()}
    estimate = float(statistic(strata))
    if statistic is aggregate_iqm:
        # The same draws, applied to every resample at once.
        pooled = np.concatenate([strata[k][draws[k]] for k in strata], axis=1)
        values = np.asarray(trim_mean(pooled, 0.25, axis=1), dtype=float)
    else:
        values = np.array(
            [statistic({k: strata[k][draws[k][r]] for k in strata}) for r in range(reps)],
            dtype=float,
        )
    lower, upper = _percentile(values, confidence)
    return Interval(estimate, lower, upper)


def performance_profile(samples: Samples, taus: ArrayLike) -> np.ndarray:
    """Fraction of runs scoring above each threshold, each scenario weighted equally.

    Parameters
    ----------
    samples : mapping
        Scenario to its runs' scores.
    taus : array_like
        Thresholds.

    Returns
    -------
    numpy.ndarray
        One fraction per threshold, the mean over scenarios of the fraction of their runs with a
        score strictly above it.
    """
    strata = _strata(samples)
    thresholds = np.asarray(taus, dtype=float).ravel()
    fractions = [np.mean(v[None, :] > thresholds[:, None], axis=1) for v in strata.values()]
    return np.mean(fractions, axis=0)


def _paired_strata(x: Samples, y: Samples) -> tuple[dict, dict]:
    a, b = _strata(x, "x"), _strata(y, "y")
    if set(a) != set(b):
        raise ValueError("both sets of runs need the same scenarios")
    return a, b


def _improvement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """P(X > Y) + P(X = Y) / 2 per row, from the Mann-Whitney rank sum of each row."""
    n, m = x.shape[-1], y.shape[-1]
    ranks = rankdata(np.concatenate([x, y], axis=-1), axis=-1)
    u = ranks[..., :n].sum(axis=-1) - n * (n + 1) / 2.0
    return u / (n * m)


def probability_of_improvement(
    x: Samples,
    y: Samples,
    reps: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> Interval:
    """Probability that a run of x scores above a run of y, averaged over scenarios.

    Within a scenario it is P(X > Y) + P(X = Y) / 2 over every pair of runs, the Mann-Whitney U
    statistic over n m; the interval resamples x and y independently within each scenario.

    Parameters
    ----------
    x, y : mapping
        Scenario to the runs' scores of each agent; the same scenarios.
    reps : int
        Number of resamples, at least 100.
    confidence : float
        Confidence level, in (0, 1).
    rng : numpy.random.Generator or int, optional
        Generator or seed of the resampling.

    Returns
    -------
    Interval
        The probability and its interval.

    Raises
    ------
    ValueError
        If the scenarios differ or one has no runs.
    """
    a, b = _paired_strata(x, y)
    _check(reps, confidence)
    gen = _generator(rng)
    estimate = float(np.mean([_improvement(a[k], b[k]) for k in a]))
    per_stratum = []
    for k in a:
        xs = a[k][gen.integers(0, len(a[k]), size=(reps, len(a[k])))]
        ys = b[k][gen.integers(0, len(b[k]), size=(reps, len(b[k])))]
        per_stratum.append(_improvement(xs, ys))
    lower, upper = _percentile(np.mean(per_stratum, axis=0), confidence)
    return Interval(estimate, lower, upper)


def paired_difference(
    x: Samples,
    y: Samples,
    reps: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> Interval:
    """The IQM of per-seed differences x - y, runs paired by index within each scenario.

    Run i of x and of y used the same seeds (common random numbers), so their difference keeps
    out the variation the two share; the interval resamples pairs within each scenario.

    Parameters
    ----------
    x, y : mapping
        Scenario to the runs' scores, in the same seed order.
    reps : int
        Number of resamples, at least 100.
    confidence : float
        Confidence level, in (0, 1).
    rng : numpy.random.Generator or int, optional
        Generator or seed of the resampling.

    Returns
    -------
    Interval
        The IQM of the pooled differences and its interval.

    Raises
    ------
    ValueError
        If the scenarios differ or a scenario's runs are not paired one to one.
    """
    a, b = _paired_strata(x, y)
    if any(len(a[k]) != len(b[k]) for k in a):
        raise ValueError("paired runs need the same number of runs per scenario")
    differences = {k: a[k] - b[k] for k in a}
    return stratified_bootstrap(differences, aggregate_iqm, reps, confidence, rng)


def summary_interval(
    values: ArrayLike, reps: int = 10_000, confidence: float = 0.95, rng: Any = None
) -> Interval:
    """The IQM of one scenario's runs with its bootstrap interval.

    Parameters
    ----------
    values : array_like
        The runs' scores.
    reps : int
        Number of resamples, at least 100.
    confidence : float
        Confidence level, in (0, 1).
    rng : numpy.random.Generator or int, optional
        Generator or seed of the resampling.

    Returns
    -------
    Interval
        The IQM and its interval.
    """
    return stratified_bootstrap({"runs": values}, aggregate_iqm, reps, confidence, rng)


__all__ = [
    "Interval",
    "aggregate_iqm",
    "iqm",
    "paired_difference",
    "performance_profile",
    "probability_of_improvement",
    "stratified_bootstrap",
    "summary_interval",
]
