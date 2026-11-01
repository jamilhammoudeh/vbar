"""Multiple-comparison corrections and a paired sign-flip test.

`holm` is the step-down procedure of Holm (1979), which controls the family-wise error rate
without any assumption on the dependence between the tests. `benjamini_hochberg` is the step-up
procedure of Benjamini and Hochberg (1995), which controls the false discovery rate for
independent tests and for tests with positive dependence. `sign_flip_test` is a randomisation test
(Fisher 1935) of two agents flown on the same seeds: under the null the paired differences are
symmetric about zero, so flipping the sign of each difference at random gives the null
distribution of their mean.

Holm, S., A simple sequentially rejective multiple test procedure, Scandinavian Journal of
Statistics 6, 1979. Benjamini, Y. and Hochberg, Y., Controlling the false discovery rate: a
practical and powerful approach to multiple testing, Journal of the Royal Statistical Society B
57, 1995. Fisher, R. A., The Design of Experiments, 1935.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from vbar.eval.stats import Samples, _generator, _paired_strata, _unit, _whole

# Entries of the sign matrix drawn at once, which bounds the memory of a large test.
_CHUNK = 2_000_000


@dataclass(frozen=True)
class Correction:
    """The outcome of a multiple-comparison procedure, in the order of the input p-values.

    Parameters
    ----------
    adjusted : numpy.ndarray
        Adjusted p-values in [0, 1]; a hypothesis is rejected when its adjusted value is at most
        the level.
    reject : numpy.ndarray
        Boolean array, True where the hypothesis is rejected.
    """

    adjusted: np.ndarray
    reject: np.ndarray


@dataclass(frozen=True)
class PairedTest:
    """A paired sign-flip test.

    Parameters
    ----------
    statistic : float
        Mean of the pooled differences x - y, in the units of the scores.
    p_value : float
        Two-sided randomisation p-value, at least 1 / (reps + 1).
    pairs : int
        Number of pairs of runs, over all scenarios.
    """

    statistic: float
    p_value: float
    pairs: int


def _p_values(pvalues: Any) -> np.ndarray:
    p = np.asarray(pvalues)
    if p.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    if p.size and p.dtype.kind not in "iuf":
        raise ValueError("pvalues must be numbers")
    p = p.astype(float)
    if np.isnan(p).any():
        raise ValueError("pvalues hold NaN")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("pvalues must lie in [0, 1]")
    return p


def holm(pvalues: ArrayLike, alpha: float = 0.05) -> Correction:
    """Holm's step-down correction, which controls the family-wise error rate at `alpha`.

    With the p-values sorted ascending and m of them, the adjusted value of the i-th is
    min(1, max over j <= i of (m - j + 1) p_(j)). No assumption is made on the dependence between
    the tests.

    Parameters
    ----------
    pvalues : array_like
        One p-value per hypothesis, each in [0, 1].
    alpha : float
        Family-wise level, in (0, 1).

    Returns
    -------
    Correction
        Adjusted p-values and rejections, in the order of `pvalues`.

    Raises
    ------
    ValueError
        If `pvalues` is not a one-dimensional array of numbers in [0, 1] (NaN, booleans and
        strings are refused) or `alpha` is not a number in (0, 1).
    """
    _unit("alpha", alpha)
    p = _p_values(pvalues)
    m = len(p)
    order = np.argsort(p, kind="stable")
    steps = (m - np.arange(m)) * p[order]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(steps))
    return Correction(adjusted, adjusted <= alpha)


def benjamini_hochberg(pvalues: ArrayLike, q: float = 0.05) -> Correction:
    """Benjamini and Hochberg's step-up correction, which controls the false discovery rate at `q`.

    With the p-values sorted ascending and m of them, the adjusted value of the i-th is
    min(1, min over j >= i of m p_(j) / j). The control holds for independent tests and for tests
    with positive dependence.

    Parameters
    ----------
    pvalues : array_like
        One p-value per hypothesis, each in [0, 1].
    q : float
        False discovery rate, in (0, 1).

    Returns
    -------
    Correction
        Adjusted p-values and rejections, in the order of `pvalues`.

    Raises
    ------
    ValueError
        If `pvalues` is not a one-dimensional array of numbers in [0, 1] (NaN, booleans and
        strings are refused) or `q` is not a number in (0, 1).
    """
    _unit("q", q)
    p = _p_values(pvalues)
    m = len(p)
    order = np.argsort(p, kind="stable")
    steps = m * p[order] / np.arange(1, m + 1)
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, np.minimum.accumulate(steps[::-1])[::-1])
    return Correction(adjusted, adjusted <= q)


def sign_flip_test(
    x: Samples,
    y: Samples,
    reps: int = 10_000,
    rng: np.random.Generator | int | None = None,
) -> PairedTest:
    """Paired two-sided randomisation test that the mean difference of two agents is zero.

    Runs are paired by index within each scenario. The statistic is the mean of all differences
    x - y pooled. Each of `reps` random assignments flips the sign of every difference
    independently; the p-value counts, with the observed assignment, those whose absolute mean
    reaches the observed one. It assumes the differences are symmetric about zero under the null.

    Parameters
    ----------
    x, y : mapping
        Scenario to its runs' scores, the same scenarios and as many runs in each.
    reps : int
        Number of random sign assignments, at least 100.
    rng : numpy.random.Generator or int, optional
        Generator or seed of the sign assignments.

    Returns
    -------
    PairedTest
        The mean difference, the p-value and the number of pairs.

    Raises
    ------
    ValueError
        If there is no scenario or a scenario has no runs, the scenarios differ, the counts of
        runs differ, a score is NaN, a difference is not finite, or `reps` is not a whole number
        of at least 100.
    """
    a, b = _paired_strata(x, y)
    _whole("reps", reps, 100)
    for key in a:
        if len(a[key]) != len(b[key]):
            raise ValueError(f"scenario {key!r} has unequal numbers of runs")
    with np.errstate(invalid="ignore"):
        d = np.concatenate([a[key] - b[key] for key in a])
    if not np.isfinite(d).all():
        raise ValueError("a difference is not finite")
    gen = _generator(rng)
    n = len(d)
    observed = abs(float(d.mean()))
    # The tolerance counts equal-magnitude statistics that differ only in the last bits.
    threshold = observed * (1.0 - 1e-12)
    rows = max(1, _CHUNK // n)
    hits = 0
    left = reps
    while left:
        size = min(rows, left)
        signs = gen.integers(0, 2, size=(size, n), dtype=np.int8) * 2.0 - 1.0
        hits += int(np.count_nonzero(np.abs(signs @ d) / n >= threshold))
        left -= size
    return PairedTest(float(d.mean()), (1 + hits) / (reps + 1), n)
