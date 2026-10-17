import re
from pathlib import Path

import numpy as np
import pytest

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


def test_infinite_values_stay_episode_values_in_the_tail():
    # S5 is infinite through a hold gate, S6 when never closing, S4t when every drift is certified.
    ratios = summarize("closing_speed_ratio", [0.5] * 20 + [np.inf] * 2)
    assert ratios == {"worst": np.inf, "quantile": np.inf, "tail_mean": np.inf}
    assert summarize("min_time_to_keep_out", [np.inf] * 5)["quantile"] == np.inf
    threat = summarize("passive_threat_time", [10.0, np.inf, np.inf, np.inf])
    assert threat == {"worst": 10.0, "quantile": 10.0, "tail_mean": 10.0}


def test_rates_count_successes_and_nan_is_left_out():
    found = summarize("success", [1.0, 0.0, 1.0, np.nan, 1.0])
    assert (found["rate"], found["count"], found["n"]) == (0.75, 3.0, 4)
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


def test_clopper_pearson_matches_the_published_validation_intervals():
    # The cases of Kochenderfer et al., Algorithms for Decision Making, 2022, section 14.1 and
    # example 14.5: 90 of 100, 95 of 100, 3 failures in 1e4.
    assert clopper_pearson(90, 100) == pytest.approx((0.8238, 0.9510), abs=1e-4)
    assert clopper_pearson(95, 100) == pytest.approx((0.887, 0.984), abs=1e-3)
    assert clopper_pearson(3, 10000) == pytest.approx((6.187e-5, 8.765e-4), rel=1e-3)
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0
    with pytest.raises(ValueError):
        clopper_pearson(3, 2)


def test_the_resolution_table_on_the_cards_matches_the_intervals():
    text = (CARDS / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$", text, re.M)
    assert len(rows) == 5
    for n, *widths in rows:
        for p, width in zip((0.5, 0.9, 0.95, 0.99), widths, strict=True):
            lower, upper = clopper_pearson(round(p * int(n)), int(n))
            assert round((upper - lower) / 2, 3) == float(width)


def test_a_rate_summary_carries_its_interval():
    found = summarize("success", [1.0] * 90 + [0.0] * 10)
    assert found["rate"] == 0.9
    assert (found["lower"], found["upper"]) == pytest.approx((0.8238, 0.9510), abs=1e-4)
