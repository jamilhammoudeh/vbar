"""The scoring intervals hold their stated level on synthetic outcomes with a known truth.

The bootstrap studies fly 1000 repeats each and take tens of seconds, so they are marked `slow`;
the rest run in the default suite.
"""

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
