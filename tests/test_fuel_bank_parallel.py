"""A bank built by several processes equals the serial bank: the LPs of a start depend only on
that start, so the entries concatenated in the order of the starts are the serial entries."""

import numpy as np
import pytest
from test_fuel_bank_undecided import no_window_feasible, small_view, starts, undecided_linprog

from vbar.oracles import DualBank, OracleConfig, fuel
from vbar.oracles.fuel import NO_FEASIBLE_WINDOW

FEW_WINDOWS = OracleConfig(bank_stride=10)


class InProcessPool:
    """Stands in for the process pool so monkeypatched solvers reach the chunks."""

    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, *columns):
        return [fn(*args) for args in zip(*columns, strict=False)]


def same_entries(a, b):
    assert len(a._entries) == len(b._entries)
    for left, right in zip(a._entries, b._entries, strict=True):
        for x, y in zip(left, right, strict=True):
            assert np.array_equal(x, y)


def same_bank(parallel, serial, points):
    same_entries(parallel, serial)
    assert parallel.uncertified == serial.uncertified
    assert parallel.infeasible == serial.infeasible
    assert parallel._infeasible_starts == serial._infeasible_starts
    for point in points:
        assert parallel.bounds(point) == serial.bounds(point)
        assert np.array_equal(parallel.window_lowers(point), serial.window_lowers(point))


def test_two_workers_build_the_serial_bank():
    view, points = small_view(), starts(5)
    serial = DualBank.build(view, points, config=FEW_WINDOWS)
    parallel = DualBank.build(view, points, config=FEW_WINDOWS, workers=2)
    assert len(serial._entries) > 5
    same_bank(parallel, serial, points)


def test_chunks_keep_the_rows_of_undecided_and_infeasible_starts(monkeypatch):
    view, points = small_view(), starts(6)
    serial_points = points[[0, 2, 3, 5]]
    reference = DualBank.build(view, serial_points, config=FEW_WINDOWS)
    # Start 1 is undecided and start 4 has no feasible window; they fall in different chunks.
    poisoned, _ = undecided_linprog(monkeypatch)
    poisoned["row"] = points[1]
    no_window_feasible(monkeypatch, points[4])
    monkeypatch.setattr(fuel, "ProcessPoolExecutor", InProcessPool)
    serial = DualBank.build(view, points, config=FEW_WINDOWS)
    parallel = DualBank.build(view, points, config=FEW_WINDOWS, workers=3)
    assert serial.uncertified == (1,) and serial.infeasible == {4: NO_FEASIBLE_WINDOW}
    same_bank(parallel, serial, points)
    same_entries(parallel, reference)


def test_chunks_of_several_starts_keep_rows_at_their_offsets(monkeypatch):
    view, points = small_view(), starts(9)
    # Nine starts on two workers make chunks of two, so rows 3 and 7 sit second in theirs.
    poisoned, _ = undecided_linprog(monkeypatch)
    poisoned["row"] = points[3]
    no_window_feasible(monkeypatch, points[7])
    monkeypatch.setattr(fuel, "ProcessPoolExecutor", InProcessPool)
    serial = DualBank.build(view, points, config=FEW_WINDOWS)
    parallel = DualBank.build(view, points, config=FEW_WINDOWS, workers=2)
    assert serial.uncertified == (3,) and serial.infeasible == {7: NO_FEASIBLE_WINDOW}
    same_bank(parallel, serial, points)


def test_a_worker_error_reaches_the_caller():
    points = starts(3)
    points[2, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        DualBank.build(small_view(), points, config=FEW_WINDOWS, workers=2)


def test_a_solver_failure_in_a_chunk_is_not_swallowed(monkeypatch):
    def failing(*args, **options):
        raise RuntimeError("HiGHS failed")

    monkeypatch.setattr(fuel, "linprog", failing)
    monkeypatch.setattr(fuel, "ProcessPoolExecutor", InProcessPool)
    with pytest.raises(RuntimeError, match="HiGHS failed"):
        DualBank.build(small_view(), starts(3), config=FEW_WINDOWS, workers=2)


def test_numpy_integer_workers_are_accepted():
    DualBank.build(small_view(), starts(2), config=FEW_WINDOWS, workers=np.int64(1))


@pytest.mark.parametrize("workers", [0, -1, 1.5, True, "2"])
def test_workers_below_one_or_not_an_integer_are_refused(workers):
    with pytest.raises(ValueError, match="workers"):
        DualBank.build(small_view(), starts(2), workers=workers)
