"""The scoring intervals hold their stated level on synthetic outcomes with a known truth.

The bootstrap studies fly 1000 repeats each and take tens of seconds, so they are marked `slow`;
the rest run in the default suite.
"""

import numpy as np
import pytest
from scipy.stats import binomtest

from vbar.eval.coverage import (
    Coverage,
    iqm_coverage,
    mcnemar,
    mcnemar_false_positives,
    paired_coverage,
    rate_coverage,
)


def test_mcnemar_is_the_exact_binomial_test_on_the_discordant_seeds():
    for only_a, only_b in ((7, 1), (3, 3), (0, 9), (12, 4)):
        expected = binomtest(only_a, only_a + only_b, 0.5).pvalue
        assert mcnemar(only_a, only_b) == pytest.approx(expected, rel=1e-12)
    assert mcnemar(0, 0) == 1.0
    with pytest.raises(ValueError):
        mcnemar(-1, 3)


def test_a_coverage_holds_within_three_binomial_errors():
    # At 1000 repeats the standard error at 95% is 0.0069, so 0.930 holds and 0.925 does not.
    assert Coverage(930, 1000, 0.95).holds
    assert not Coverage(925, 1000, 0.95).holds


@pytest.mark.parametrize("p", [0.05, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("n", [10, 100])
def test_the_rate_interval_covers_the_true_rate(p, n):
    # Clopper-Pearson is exact, so it covers at least 95% at every n and rate.
    assert rate_coverage(p, n, 1000, rng=7).holds


def test_equal_controllers_are_called_different_at_most_5_percent_of_the_time():
    found = mcnemar_false_positives(0.8, 100, 1000, rng=5)
    assert found.estimate <= 0.05


def test_a_test_that_ignores_the_pairing_is_caught():
    # Rejecting whenever the discordant seeds split unevenly at all is far above the level.
    def eager(only_a, only_b):
        return 0.0 if only_a != only_b else 1.0

    assert mcnemar_false_positives(0.8, 100, 300, rng=6, test=eager).lower > 0.05


@pytest.mark.slow
@pytest.mark.parametrize("n", [50, 100])
def test_the_iqm_and_paired_intervals_cover_the_truth_at_the_papers_seed_counts(n):
    assert iqm_coverage(n, 1000, 1000, rng=3).holds
    assert paired_coverage(n, 1000, 1000, rng=4).holds


@pytest.mark.slow
def test_a_deliberately_narrow_interval_fails_the_coverage_test():
    assert not iqm_coverage(100, 1000, 1000, rng=3, widen=0.7).holds
    assert not paired_coverage(100, 1000, 1000, rng=4, widen=0.7).holds


# Sizes are small, so a refusal that is missing runs the study in milliseconds, not minutes.
RATE = {"p": 0.5, "n": 10, "repeats": 20, "rng": 0}
IQM = {"n": 10, "repeats": 5, "reps": 100, "rng": 0}
FALSE_POSITIVES = {"p": 0.8, "n": 10, "repeats": 20, "rng": 0}
COUNTS = [("0", 0), ("negative", -1), ("fractional", 2.5)]
PROBABILITY = [("nan", np.nan), ("negative", -0.1), ("above-1", 1.1)]
OPEN_UNIT = [("nan", np.nan), ("0", 0.0), ("1", 1.0)]
WIDEN = [("0", 0.0), ("negative", -1.0), ("nan", np.nan), ("inf", np.inf)]
# Numpy's own "cannot convert float NaN to integer" does not say what was wrong with the input.
NOT_WHOLE = "whole|finite|integers"


def hostile(function, label, match=None, **arguments):
    return pytest.param(function, arguments, match, id=f"{function.__name__}-{label}")


def vary(function, base, argument, cases, match=None):
    return [
        hostile(function, f"{argument}-{label}", match, **{**base, argument: value})
        for label, value in cases
    ]


@pytest.mark.parametrize(
    "function, arguments, match",
    [
        *[
            hostile(mcnemar, f"{name}-{label}", match, **{"only_a": 3, "only_b": 1, name: value})
            for name in ("only_a", "only_b")
            for label, value, match in (
                ("inf", np.inf, NOT_WHOLE),
                ("minus-inf", -np.inf, NOT_WHOLE),
                ("nan", np.nan, NOT_WHOLE),
                ("fractional", 1.5, None),
                ("negative", -1, None),
            )
        ],
        hostile(Coverage, "repeats-0", covered=0, repeats=0, confidence=0.95),
        hostile(Coverage, "repeats-negative", covered=0, repeats=-1, confidence=0.95),
        hostile(Coverage, "repeats-fractional", covered=1, repeats=2.5, confidence=0.95),
        hostile(Coverage, "repeats-nan", covered=1, repeats=np.nan, confidence=0.95),
        hostile(Coverage, "covered-negative", covered=-1, repeats=10, confidence=0.95),
        hostile(Coverage, "covered-above-repeats", covered=11, repeats=10, confidence=0.95),
        hostile(Coverage, "covered-fractional", covered=2.5, repeats=10, confidence=0.95),
        *vary(Coverage, {"covered": 5, "repeats": 10}, "confidence", OPEN_UNIT),
        *vary(rate_coverage, RATE, "p", PROBABILITY),
        *vary(rate_coverage, RATE, "n", COUNTS),
        *vary(rate_coverage, RATE, "repeats", COUNTS),
        *vary(rate_coverage, RATE, "confidence", OPEN_UNIT),
        *vary(iqm_coverage, IQM, "n", [("0", 0), ("fractional", 2.5)]),
        *vary(iqm_coverage, IQM, "repeats", [("0", 0)]),
        *vary(iqm_coverage, IQM, "reps", [("99", 99)]),
        *vary(iqm_coverage, IQM, "confidence", OPEN_UNIT),
        *vary(iqm_coverage, IQM, "widen", WIDEN),
        *vary(paired_coverage, IQM, "n", [("0", 0), ("fractional", 2.5)]),
        *vary(paired_coverage, IQM, "repeats", [("0", 0)]),
        *vary(paired_coverage, IQM, "reps", [("99", 99)]),
        *vary(paired_coverage, IQM, "confidence", OPEN_UNIT),
        *vary(paired_coverage, IQM, "widen", WIDEN),
        *vary(
            paired_coverage,
            IQM,
            "shared",
            [("1", 1.0), ("negative", -0.1), ("nan", np.nan), ("2", 2.0)],
        ),
        *vary(mcnemar_false_positives, FALSE_POSITIVES, "p", [*PROBABILITY[:1], ("above-1", 1.5)]),
        *vary(mcnemar_false_positives, FALSE_POSITIVES, "p", PROBABILITY[1:2]),
        *vary(mcnemar_false_positives, FALSE_POSITIVES, "n", [("0", 0)]),
        *vary(mcnemar_false_positives, FALSE_POSITIVES, "repeats", [("0", 0)]),
        *vary(mcnemar_false_positives, FALSE_POSITIVES, "alpha", OPEN_UNIT),
        *vary(
            mcnemar_false_positives,
            FALSE_POSITIVES,
            "shared",
            [("nan", np.nan), ("negative", -0.1), ("above-1", 1.5)],
        ),
    ],
)
def test_hostile_coverage_inputs_are_refused(function, arguments, match):
    with pytest.raises(ValueError, match=match):
        function(**arguments)


def test_numpy_integers_count_experiments_runs_and_discordant_seeds():
    assert mcnemar(np.int64(3), np.int64(1)) == mcnemar(3, 1)
    assert Coverage(np.int64(5), np.int64(10), 0.95).rate == 0.5
    count = np.int64
    assert rate_coverage(0.5, count(10), count(20), rng=0) == rate_coverage(0.5, 10, 20, rng=0)
    assert iqm_coverage(count(10), count(5), count(100), rng=0) == iqm_coverage(10, 5, 100, rng=0)
    assert paired_coverage(count(10), count(5), count(100), rng=0) == paired_coverage(
        10, 5, 100, rng=0
    )
    assert mcnemar_false_positives(0.8, count(10), count(20), rng=0) == mcnemar_false_positives(
        0.8, 10, 20, rng=0
    )


def test_probabilities_at_the_ends_of_their_ranges_are_accepted():
    for p in (0.0, 1.0):
        assert rate_coverage(p, 10, 20, rng=0).covered == 20
        assert mcnemar_false_positives(p, 10, 20, rng=0).estimate == 0.0
    for shared in (0.0, 1.0):
        assert mcnemar_false_positives(0.8, 10, 20, shared=shared, rng=0).estimate <= 1.0
    assert paired_coverage(**{**IQM, "shared": 0.0}).repeats == 5
