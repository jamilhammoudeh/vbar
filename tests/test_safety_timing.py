import time

import numpy as np
import pytest
from metric_cases import episode

from vbar.agents import PDAgent
from vbar.safety import FilteredAgent, FilterResult, run_filter

OBS, COMMAND = np.zeros(6), np.zeros(3)
# Sleeps stay short so each test runs well under a second; only lower bounds are asserted on them.
PREP_SLEEP_S = 0.05
FEEDBACK_SLEEP_S = 0.02


class PreparedFilter:
    """Sleeps in `prepare` and in `filter`, logs each call with its clock, and claims `claims`."""

    name = "prepared"

    def __init__(self, prep_sleep_s=0.0, feedback_sleep_s=0.0, **claims):
        self.prep_sleep_s, self.feedback_sleep_s, self.claims = (
            prep_sleep_s,
            feedback_sleep_s,
            claims,
        )
        self.calls = []

    def reset(self, view):
        self.calls = []

    def prepare(self):
        start = time.perf_counter()
        if self.prep_sleep_s:
            time.sleep(self.prep_sleep_s)
        self.calls.append(("prepare", start, time.perf_counter()))

    def filter(self, obs, action):
        start = time.perf_counter()
        if self.feedback_sleep_s:
            time.sleep(self.feedback_sleep_s)
        self.calls.append(("filter", start, time.perf_counter()))
        return FilterResult(action, False, **self.claims)


class UnpreparedFilter:
    """Has no `prepare`; sleeps in `filter` and claims `claims`."""

    name = "unprepared"

    def __init__(self, feedback_sleep_s=0.0, **claims):
        self.feedback_sleep_s, self.claims = feedback_sleep_s, claims

    def reset(self, view):
        pass

    def filter(self, obs, action):
        if self.feedback_sleep_s:
            time.sleep(self.feedback_sleep_s)
        return FilterResult(action, False, **self.claims)


class FailingPrepareFilter(PreparedFilter):
    """Raises from `prepare`."""

    def prepare(self):
        self.calls.append(("prepare", time.perf_counter(), time.perf_counter()))
        raise RuntimeError("warm start lost")


def test_filter_result_has_prep_and_feedback_fields_defaulting_to_none():
    result = FilterResult(COMMAND, False)
    assert result.prep_s is None and result.feedback_s is None


def test_prepare_that_sleeps_50_ms_and_reports_zero_is_recorded_at_50_ms_or_more():
    result, _ = run_filter(PreparedFilter(prep_sleep_s=PREP_SLEEP_S, prep_s=0.0), OBS, COMMAND)
    assert result.prep_s >= PREP_SLEEP_S


def test_feedback_time_is_measured_around_filter_and_excludes_prepare():
    f = PreparedFilter(prep_sleep_s=PREP_SLEEP_S, feedback_sleep_s=FEEDBACK_SLEEP_S)
    before = time.perf_counter()
    result, _ = run_filter(f, OBS, COMMAND)
    after = time.perf_counter()
    (_, prep_start, prep_end), (_, filt_start, filt_end) = f.calls
    assert result.feedback_s >= filt_end - filt_start >= FEEDBACK_SLEEP_S
    assert result.prep_s >= prep_end - prep_start >= PREP_SLEEP_S
    # The clocks bracket each phase from outside, so these bounds hold under any load.
    assert result.feedback_s <= after - prep_end
    assert result.prep_s <= filt_start - before


def test_compute_time_is_the_sum_of_prep_and_feedback_times():
    f = PreparedFilter(prep_sleep_s=PREP_SLEEP_S, feedback_sleep_s=FEEDBACK_SLEEP_S)
    result, _ = run_filter(f, OBS, COMMAND)
    assert result.compute_s == pytest.approx(result.prep_s + result.feedback_s, rel=1e-12)
    assert result.compute_s >= PREP_SLEEP_S + FEEDBACK_SLEEP_S


def test_filter_without_prepare_records_no_prep_time_and_compute_equal_to_feedback():
    result, _ = run_filter(UnpreparedFilter(feedback_sleep_s=FEEDBACK_SLEEP_S), OBS, COMMAND)
    assert result.prep_s is None
    assert result.feedback_s >= FEEDBACK_SLEEP_S
    assert result.compute_s == result.feedback_s


def test_prepared_filter_claims_are_overwritten_by_the_measured_times():
    f = PreparedFilter(
        prep_sleep_s=PREP_SLEEP_S,
        feedback_sleep_s=FEEDBACK_SLEEP_S,
        prep_s=0.0,
        feedback_s=0.0,
        compute_s=0.0,
    )
    result, _ = run_filter(f, OBS, COMMAND)
    assert result.prep_s >= PREP_SLEEP_S
    assert result.feedback_s >= FEEDBACK_SLEEP_S
    assert result.compute_s >= PREP_SLEEP_S + FEEDBACK_SLEEP_S


def test_prepared_filter_cannot_inflate_its_times():
    f = PreparedFilter(prep_s=100.0, feedback_s=100.0, compute_s=100.0)
    before = time.perf_counter()
    result, _ = run_filter(f, OBS, COMMAND)
    wall = time.perf_counter() - before
    assert result.prep_s <= wall and result.feedback_s <= wall and result.compute_s <= wall


def test_unprepared_filter_claimed_prep_time_is_replaced_by_none():
    f = UnpreparedFilter(prep_s=5.0, feedback_s=0.0, compute_s=0.0)
    result, _ = run_filter(f, OBS, COMMAND)
    assert result.prep_s is None
    assert result.feedback_s > 0.0
    assert result.compute_s == result.feedback_s


def test_prepare_is_called_once_per_decision_before_filter():
    f = PreparedFilter()
    for _ in range(3):
        run_filter(f, OBS, COMMAND)
    assert [name for name, _, _ in f.calls] == ["prepare", "filter"] * 3
    starts = [start for _, start, _ in f.calls]
    assert starts == sorted(starts)


def test_filtered_agent_prepares_once_per_decision_and_records_the_split():
    f = PreparedFilter(prep_sleep_s=0.005, prep_s=0.0)
    agent = FilteredAgent(PDAgent(), f)
    episode(agent, start_state=(0.0, -400.0, 30.0, 0.0, 0.0, 0.0), max_steps=5)
    names = [name for name, _, _ in f.calls]
    assert agent.decisions > 0
    assert names == ["prepare", "filter"] * agent.decisions
    assert agent.last_result.prep_s >= 0.005
    assert agent.last_result.compute_s == pytest.approx(
        agent.last_result.prep_s + agent.last_result.feedback_s, rel=1e-12
    )


def test_exception_in_prepare_propagates_and_filter_is_not_called():
    f = FailingPrepareFilter()
    with pytest.raises(RuntimeError, match="warm start lost"):
        run_filter(f, OBS, COMMAND)
    assert [name for name, _, _ in f.calls] == ["prepare"]


class ZeroingResult(FilterResult):
    """Zeroes its own times whenever it is built, `dataclasses.replace` included."""

    def __post_init__(self):
        super().__post_init__()
        for field in ("prep_s", "feedback_s", "compute_s"):
            object.__setattr__(self, field, 0.0)


class ZeroingResultFilter(PreparedFilter):
    """Sleeps in both phases and returns a `ZeroingResult`."""

    def filter(self, obs, action):
        super().filter(obs, action)
        return ZeroingResult(action, False)


def test_result_subclass_cannot_zero_the_measured_times():
    f = ZeroingResultFilter(prep_sleep_s=PREP_SLEEP_S, feedback_sleep_s=FEEDBACK_SLEEP_S)
    try:
        result, elapsed = run_filter(f, OBS, COMMAND)
    except TypeError:
        # Rejecting the subclass outright also keeps a false time out of the record.
        return
    assert result.prep_s >= PREP_SLEEP_S
    assert result.feedback_s >= FEEDBACK_SLEEP_S
    assert result.compute_s >= PREP_SLEEP_S + FEEDBACK_SLEEP_S
    assert elapsed >= PREP_SLEEP_S + FEEDBACK_SLEEP_S


class SlowPrepareLookupFilter(UnpreparedFilter):
    """Spends `PREP_SLEEP_S` resolving its `prepare` attribute, then returns a no-op."""

    @property
    def prepare(self):
        time.sleep(PREP_SLEEP_S)
        return lambda: None


class SlowMissingPrepareFilter(UnpreparedFilter):
    """Spends `PREP_SLEEP_S` in `__getattr__` before reporting it has no `prepare`."""

    def __getattr__(self, name):
        if name == "prepare":
            time.sleep(PREP_SLEEP_S)
        raise AttributeError(name)


@pytest.mark.parametrize("make", [SlowPrepareLookupFilter, SlowMissingPrepareFilter])
def test_time_spent_looking_up_prepare_is_counted(make):
    result, elapsed = run_filter(make(), OBS, COMMAND)
    assert result.compute_s >= PREP_SLEEP_S
    assert elapsed >= PREP_SLEEP_S


def test_returned_wall_time_is_compute_time_with_prepare_included():
    f = PreparedFilter(prep_sleep_s=PREP_SLEEP_S, feedback_sleep_s=FEEDBACK_SLEEP_S)
    result, elapsed = run_filter(f, OBS, COMMAND)
    assert elapsed == result.compute_s
    assert elapsed >= PREP_SLEEP_S + FEEDBACK_SLEEP_S
