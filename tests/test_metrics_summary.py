import re
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta

from vbar.metrics import METRICS, clopper_pearson, summarize, unsafe_tail

CARDS = Path(__file__).parents[1] / "src" / "vbar" / "metrics" / "cards"

# 40 margins: 38 comfortable ones and two near misses.
MARGINS = [20.0 + k for k in range(38)] + [0.5, -2.0]


def test_the_unsafe_tail_of_margins_keeps_the_near_misses_the_iqm_drops():
    tail = unsafe_tail(MARGINS, "low")
    assert tail["worst"] == -2.0
    # The 5th percentile of 40 values is the second lowest, so the tail averages both near misses.
    assert tail["quantile"] == 0.5
    assert tail["tail_mean"] == -0.75
    assert summarize("min_margin", MARGINS) == tail
    # The interquartile mean of the same values says nothing about the two near misses.
    iqm = summarize("delta_v", MARGINS)["iqm"]
    assert iqm > 25.0


def test_ratios_where_higher_is_worse_take_the_upper_tail():
    ratios = [0.2, 0.3, 0.25, 1.4, 0.1]
    tail = unsafe_tail(ratios, "high")
    assert tail["worst"] == 1.4 and tail["tail_mean"] == pytest.approx(1.4)
    assert summarize("closing_speed_ratio", ratios) == tail


def test_per_constraint_margins_are_summarised_constraint_by_constraint():
    episodes = [{"keep_out": m, "approach_cone": 5.0} for m in MARGINS]
    found = summarize("min_margin", episodes)
    assert found["keep_out"] == summarize("min_margin", MARGINS)
    assert found["approach_cone"] == {
        "worst": 5.0,
        "quantile": 5.0,
        "tail_mean": 5.0,
        "n": 40,
        "missing": 0,
    }
    # A constraint an episode lacks, or an episode without the metric, is left out, not zero.
    sparse = summarize("min_margin", [{"keep_out": 3.0, "zone": 2.0}, {"keep_out": 4.0}, np.nan])
    assert sparse["zone"]["worst"] == 2.0 and sparse["keep_out"]["worst"] == 3.0
    assert (sparse["zone"]["n"], sparse["zone"]["missing"]) == (1, 2)


def test_both_tails_hold_the_same_number_of_episodes():
    x = np.arange(100.0)
    low, high = unsafe_tail(x, "low"), unsafe_tail(x, "high")
    # Five of 100 on each side: 0 to 4 and 95 to 99.
    assert (low["quantile"], low["tail_mean"]) == (4.0, 2.0)
    assert (high["quantile"], high["tail_mean"]) == (95.0, 97.0)


def test_infinite_values_stay_episode_values_in_the_tail():
    # S5 is infinite through a hold gate, S6 when never closing, S4t when every drift is certified.
    ratios = summarize("closing_speed_ratio", [0.5] * 20 + [np.inf] * 2)
    assert ratios == {
        "worst": np.inf,
        "quantile": np.inf,
        "tail_mean": np.inf,
        "n": 22,
        "missing": 0,
    }
    assert summarize("min_time_to_keep_out", [np.inf] * 5)["quantile"] == np.inf
    threat = summarize("passive_threat_time", [10.0, np.inf, np.inf, np.inf])
    assert threat == {"worst": 10.0, "quantile": 10.0, "tail_mean": 10.0, "n": 4, "missing": 0}


def test_nan_is_left_out_and_counted_but_never_for_a_metric_every_episode_has():
    found = summarize("sign_check_pass", [1.0, 0.0, 1.0, np.nan, 1.0])
    assert (found["rate"], found["count"], found["n"], found["missing"]) == (0.75, 3.0, 4, 1)
    dv = summarize("delta_v", [1.0, np.nan, 2.0])
    assert (dv["n"], dv["missing"]) == (2, 1)
    # A run whose score is NaN failed; leaving it out would raise the IQM over the others.
    for name in ("success", "progress", "composite_score"):
        with pytest.raises(ValueError, match="NaN in 1"):
            summarize(name, [1.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="0 or 1"):
        summarize("success", [0.4, 0.4])
    assert np.isnan(unsafe_tail([np.nan], "low")["worst"])


def test_every_safety_metric_ranks_by_its_tail_and_performance_by_the_iqm():
    for name in (
        "min_margin",
        "passive_safety_margin",
        "passive_safe_fraction",
        "stuck_thruster_margin",
        "min_time_to_keep_out",
        "cam_margin",
    ):
        assert METRICS[name].summary == "lower_tail", name
    assert METRICS["closing_speed_ratio"].summary == "upper_tail"
    for name in (
        "delta_v",
        "delta_v_axes",
        "terminal_pos_error",
        "terminal_vel_error",
        "time_to_complete",
    ):
        assert METRICS[name].summary == "iqm", name
    with pytest.raises(ValueError, match="not summarised"):
        summarize("failure_label", [1.0])


def test_clopper_pearson_gives_the_exact_intervals_of_the_book_case_and_ours():
    # 3 failures in 1e4 is the case of Kochenderfer et al., Algorithms for Decision Making, 2022,
    # section 14.1, whose normal interval (eq. 14.5) goes below 0; 90 and 95 of 100 are ours, the
    # rates a 100-episode cell resolves. All are exact intervals, recomputed.
    assert clopper_pearson(90, 100) == pytest.approx((0.8238, 0.9510), abs=1e-4)
    assert clopper_pearson(95, 100) == pytest.approx((0.887, 0.984), abs=1e-3)
    assert clopper_pearson(3, 10000) == pytest.approx((6.187e-5, 8.765e-4), rel=1e-3)
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0
    for successes, trials, confidence in ((3, 2, 0.95), (2.5, 10, 0.95), (3, 10, 1.5)):
        with pytest.raises(ValueError):
            clopper_pearson(successes, trials, confidence)


def test_the_resolution_table_on_the_cards_matches_the_intervals():
    text = (CARDS / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$", text, re.M)
    assert len(rows) == 5
    for n, *widths in rows:
        for p, width in zip((0.5, 0.9, 0.95, 0.99), widths, strict=True):
            # At k = p n exactly; rounding k would score 49.5 of 50 as 50 of 50.
            k, n = p * int(n), int(n)
            lower, upper = beta.ppf(0.025, k, n - k + 1), beta.ppf(0.975, k + 1, n - k)
            assert round((upper - lower) / 2, 3) == float(width)
            if k == int(k):
                assert (lower, upper) == pytest.approx(clopper_pearson(int(k), n))


def test_a_rate_summary_carries_its_interval():
    found = summarize("success", [1.0] * 90 + [0.0] * 10)
    assert found["rate"] == 0.9
    assert (found["lower"], found["upper"]) == pytest.approx((0.8238, 0.9510), abs=1e-4)


# Numpy's own "cannot convert float NaN to integer" does not say what was wrong with the input.
NOT_WHOLE = "whole|finite|integers"
COUNTS = {"successes": 3, "trials": 10}
BAD_COUNTS = [
    ("inf", np.inf, NOT_WHOLE),
    ("minus-inf", -np.inf, NOT_WHOLE),
    ("nan", np.nan, NOT_WHOLE),
    ("fractional", 2.5, None),
    ("negative", -1, None),
]


def hostile(function, label, match=None, **arguments):
    return pytest.param(function, arguments, match, id=f"{function.__name__}-{label}")


@pytest.mark.parametrize(
    "function, arguments, match",
    [
        *[
            hostile(clopper_pearson, f"{name}-{label}", match, **{**COUNTS, name: value})
            for name in COUNTS
            for label, value, match in BAD_COUNTS
        ],
        hostile(clopper_pearson, "no-trials", successes=0, trials=0),
        hostile(clopper_pearson, "more-successes-than-trials", successes=11, trials=10),
        *[
            hostile(clopper_pearson, f"confidence-{label}", **COUNTS, confidence=value)
            for label, value in (("nan", np.nan), ("0", 0.0), ("1", 1.0))
        ],
        hostile(unsafe_tail, "unknown-side", values=MARGINS, worse="worst"),
        # A level is refused whatever the values hold, also when none is left to take a quantile of.
        *[
            hostile(
                unsafe_tail, f"level-{label}-{kind}", "level", values=values, worse="low", level=v
            )
            for label, v in (("nan", np.nan), ("negative", -0.1), ("above-1", 1.5))
            for kind, values in (("margins", MARGINS), ("all-nan", [np.nan]), ("empty", []))
        ],
    ],
)
def test_hostile_summary_inputs_are_refused(function, arguments, match):
    with pytest.raises(ValueError, match=match):
        function(**arguments)


def test_numpy_integers_count_successes_and_trials():
    found = clopper_pearson(np.int64(90), np.int64(100))
    assert found == clopper_pearson(90, 100)
