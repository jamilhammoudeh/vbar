import numpy as np
import pytest

from vbar.metrics import METRICS, summarize, unsafe_tail

# 40 margins: 38 comfortable ones and two near misses.
MARGINS = [20.0 + k for k in range(38)] + [0.5, -2.0]


def test_the_unsafe_tail_of_margins_keeps_the_near_misses_the_iqm_drops():
    tail = unsafe_tail(MARGINS, "low")
    assert tail["worst"] == -2.0
    # The 5th percentile of 40 values falls between the two lowest, so the tail mean is the worst.
    assert tail["quantile"] == pytest.approx(np.quantile(MARGINS, 0.05))
    assert tail["tail_mean"] == pytest.approx(
        np.mean([v for v in MARGINS if v <= tail["quantile"]])
    )
    assert summarize("min_margin", MARGINS) == tail
    # The interquartile mean of the same values says nothing about the two near misses.
    iqm = summarize("delta_v", MARGINS)["iqm"]
    assert iqm > 25.0


def test_ratios_where_higher_is_worse_take_the_upper_tail():
    ratios = [0.2, 0.3, 0.25, 1.4, 0.1]
    tail = unsafe_tail(ratios, "high")
    assert tail["worst"] == 1.4 and tail["tail_mean"] == pytest.approx(1.4)
    assert summarize("closing_speed_ratio", ratios) == tail


def test_rates_count_successes_and_nan_is_left_out():
    found = summarize("success", [1.0, 0.0, 1.0, np.nan, 1.0])
    assert found == {"rate": 0.75, "count": 3.0, "n": 4}
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
