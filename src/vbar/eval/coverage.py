"""Coverage of the scoring intervals on synthetic outcomes whose truth is known.

An interval stated at 95% must cover the true value in 95% of repeated experiments. Each study
draws `repeats` synthetic experiments of `n` runs from a distribution with a known value, builds
the interval the report would, and counts how often it covers that value; with 1000 repeats the
count has a binomial standard error of 0.7 points at 95%. The same studies give McNemar's
false-positive rate for two equal controllers flown on the same seeds.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, trim_mean

from ..metrics import clopper_pearson
from .stats import Interval, _check, _generator, _percentile, _unit, _whole


@dataclass(frozen=True)
class Coverage:
    """The share of repeats whose interval covered the truth.

    Parameters
    ----------
    covered : int
        Repeats whose interval held the true value.
    repeats : int
        Repeats run.
    confidence : float
        The intervals' stated level, in (0, 1).

    Raises
    ------
    ValueError
        If `repeats` is not a whole number of at least 1, `covered` is not a whole number between
        0 and `repeats`, or `confidence` is outside (0, 1).
    """

    covered: int
    repeats: int
    confidence: float

    def __post_init__(self):
        _whole("repeats", self.repeats, 1)
        _whole("covered", self.covered, 0)
        if self.covered > self.repeats:
            raise ValueError("covered cannot exceed repeats")
        _unit("confidence", self.confidence)

    @property
    def rate(self) -> float:
        """covered / repeats."""
        return self.covered / self.repeats

    @property
    def holds(self) -> bool:
        """Whether the rate is within three binomial standard errors below the stated level."""
        error = math.sqrt(self.confidence * (1.0 - self.confidence) / self.repeats)
        return self.rate >= self.confidence - 3.0 * error


def mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value for paired successes and failures.

    Parameters
    ----------
    only_a : int
        Seeds on which controller a succeeded and b failed.
    only_b : int
        Seeds on which b succeeded and a failed.

    Returns
    -------
    float
        The probability, under equal success rates, of a split at least this uneven: twice the
        smaller binomial tail of Binomial(only_a + only_b, 1/2), at most 1.

    Raises
    ------
    ValueError
        If a count is negative or not a whole number.
    """
    only_a = _whole("only_a", only_a, 0)
    only_b = _whole("only_b", only_b, 0)
    total = only_a + only_b
    if total == 0:
        return 1.0
    low = min(only_a, only_b)
    return float(min(1.0, 2.0 * binom.cdf(low, total, 0.5)))


def _bootstrap_iqm_rows(samples: np.ndarray, reps: int, confidence: float, gen) -> np.ndarray:
    """Percentile bootstrap interval of the IQM for each row of `samples`, shape (rows, 2)."""
    rows, n = samples.shape
    out = np.empty((rows, 2))
    for k in range(rows):
        draws = samples[k][gen.integers(0, n, size=(reps, n))]
        out[k] = _percentile(np.asarray(trim_mean(draws, 0.25, axis=1)), confidence)
    return out


def rate_coverage(
    p: float, n: int, repeats: int = 1000, confidence: float = 0.95, rng=None
) -> Coverage:
    """Coverage of the Clopper-Pearson interval for a success rate `p` from `n` runs.

    Parameters
    ----------
    p : float
        True success rate, in [0, 1].
    n : int
        Runs per experiment, at least 1.
    repeats : int
        Experiments, at least 1.
    confidence : float
        The interval's level, in (0, 1).
    rng : numpy.random.Generator or int, optional
        Generator or seed.

    Returns
    -------
    Coverage
        How often the interval held `p`.

    Raises
    ------
    ValueError
        If `p` is outside [0, 1], `n` or `repeats` is not a whole number of at least 1, or
        `confidence` is outside (0, 1).
    """
    _unit("p", p, "[]")
    _whole("n", n, 1)
    _whole("repeats", repeats, 1)
    _unit("confidence", confidence)
    gen = _generator(rng)
    counts = gen.binomial(n, p, size=repeats)
    # One interval per possible count, n + 1 of them, rather than one per repeat.
    holds = np.array(
        [lo <= p <= hi for lo, hi in (clopper_pearson(k, n, confidence) for k in range(n + 1))]
    )
    return Coverage(int(np.sum(holds[counts])), repeats, confidence)


def iqm_coverage(
    n: int,
    repeats: int = 1000,
    reps: int = 1000,
    confidence: float = 0.95,
    rng=None,
    widen: float = 1.0,
) -> Coverage:
    """Coverage of the bootstrap interval of the IQM for a continuous metric.

    The runs are normal with mean 2 and standard deviation 1 (a delta-v in m/s, say), whose
    population IQM is the mean, 2, by symmetry.

    Parameters
    ----------
    n : int
        Runs per experiment.
    repeats : int
        Experiments.
    reps : int
        Bootstrap resamples per interval.
    confidence : float
        The interval's level.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    widen : float
        Factor applied to the interval's half-width about its centre; 1 is the report's
        interval, and values below 1 give a deliberately narrow one.

    Returns
    -------
    Coverage
        How often the interval held 2.

    Raises
    ------
    ValueError
        If `n` or `repeats` is not a whole number of at least 1, `reps` is below 100,
        `confidence` is outside (0, 1), or `widen` is not finite and positive.
    """
    _check_study(n, repeats, reps, confidence, widen)
    gen = _generator(rng)
    samples = 2.0 + gen.standard_normal((repeats, n))
    return _covered(_bootstrap_iqm_rows(samples, reps, confidence, gen), 2.0, widen, confidence)


def paired_coverage(
    n: int,
    repeats: int = 1000,
    reps: int = 1000,
    confidence: float = 0.95,
    rng=None,
    widen: float = 1.0,
    shared: float = 0.9,
) -> Coverage:
    """Coverage of the paired-difference interval for two controllers on the same seeds.

    Each seed's runs share a part of their outcome (the seed's difficulty, with weight `shared`
    of the unit variance), so the per-seed difference has a smaller spread than either run. The
    true difference of the IQMs is 0.3.

    Parameters
    ----------
    n : int
        Seeds per experiment.
    repeats : int
        Experiments.
    reps : int
        Bootstrap resamples per interval.
    confidence : float
        The interval's level.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    widen : float
        Half-width factor, as in `iqm_coverage`.
    shared : float
        Share of each run's variance that comes from its seed, in [0, 1).

    Returns
    -------
    Coverage
        How often the interval held 0.3.

    Raises
    ------
    ValueError
        As `iqm_coverage`, and if `shared` is outside [0, 1).
    """
    _check_study(n, repeats, reps, confidence, widen)
    _unit("shared", shared, "[)")
    gen = _generator(rng)
    seed_part = math.sqrt(shared) * gen.standard_normal((repeats, n))
    own = math.sqrt(1.0 - shared)
    a = 2.3 + seed_part + own * gen.standard_normal((repeats, n))
    b = 2.0 + seed_part + own * gen.standard_normal((repeats, n))
    # The difference is symmetric about 0.3, so its IQM is 0.3.
    return _covered(_bootstrap_iqm_rows(a - b, reps, confidence, gen), 0.3, widen, confidence)


def _check_study(n: int, repeats: int, reps: int, confidence: float, widen: float) -> None:
    """Refuse the settings the two bootstrap studies share, before any draw."""
    _whole("n", n, 1)
    _whole("repeats", repeats, 1)
    _check(reps, confidence)
    if not (math.isfinite(widen) and widen > 0.0):
        raise ValueError("widen must be finite and positive")


def _covered(bounds: np.ndarray, truth: float, widen: float, confidence: float) -> Coverage:
    centre = 0.5 * (bounds[:, 0] + bounds[:, 1])
    half = 0.5 * (bounds[:, 1] - bounds[:, 0]) * widen
    hits = (centre - half <= truth) & (truth <= centre + half)
    return Coverage(int(np.sum(hits)), len(bounds), confidence)


def mcnemar_false_positives(
    p: float,
    n: int,
    repeats: int = 1000,
    alpha: float = 0.05,
    rng=None,
    shared: float = 0.5,
    test: Callable[[int, int], float] = mcnemar,
) -> Interval:
    """Rate at which McNemar's test calls two equal controllers different, with its interval.

    Both controllers succeed with probability `p` on every seed; with weight `shared` their
    outcomes come from the same draw (a seed hard for one is hard for the other), else from
    their own.

    Parameters
    ----------
    p : float
        The common success rate.
    n : int
        Seeds per experiment.
    repeats : int
        Experiments.
    alpha : float
        The test's level.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    shared : float
        Probability that a seed's two outcomes are one draw, in [0, 1].
    test : callable
        The p-value of a paired comparison from the discordant counts.

    Returns
    -------
    Interval
        The false-positive rate and its Clopper-Pearson interval.

    Raises
    ------
    ValueError
        If `p` or `shared` is outside [0, 1], `n` or `repeats` is not a whole number of at least
        1, or `alpha` is outside (0, 1).
    """
    _unit("p", p, "[]")
    _whole("n", n, 1)
    _whole("repeats", repeats, 1)
    _unit("alpha", alpha)
    _unit("shared", shared, "[]")
    gen = _generator(rng)
    common = gen.random((repeats, n)) < p
    same = gen.random((repeats, n)) < shared
    a = np.where(same, common, gen.random((repeats, n)) < p)
    b = np.where(same, common, gen.random((repeats, n)) < p)
    only_a = np.sum(a & ~b, axis=1)
    only_b = np.sum(b & ~a, axis=1)
    rejected = sum(test(int(x), int(y)) < alpha for x, y in zip(only_a, only_b, strict=True))
    lower, upper = clopper_pearson(rejected, repeats)
    return Interval(rejected / repeats, lower, upper)


__all__ = [
    "Coverage",
    "iqm_coverage",
    "mcnemar",
    "mcnemar_false_positives",
    "paired_coverage",
    "rate_coverage",
]
