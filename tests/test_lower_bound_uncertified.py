"""The lower-bound check keeps a scenario whose bank left some starts uncertified, holds the runs
from those starts to no bound and reports how many there were and which."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

SID = "vbar:approach-vbar:cw:box:dv:v1"
SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "lower_bound_check.py"


def load_script(monkeypatch):
    """The script as a module, run in this process; no bytecode is written next to it."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("lower_bound_check_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PartlyUncertifiedBank:
    """A bank on the given view that left start 1 of 3 uncertified, with a positive bound."""

    uncertified = (1,)

    def __init__(self, view):
        self.view = view

    def bounds(self, start):
        from vbar.metrics.efficiency import FuelBounds

        return FuelBounds(
            lower=1.0,
            upper=math.inf,
            norm="l1",
            rung="cw",
            accel_limit=self.view.max_accel,
            impulsive_lower=0.5,
        )


def test_lower_bound_script_keeps_a_scenario_with_uncertified_starts_and_reports_them(
    capsys, caplog, monkeypatch
):
    import vbar.metrics
    from vbar.agents import DriftAgent
    from vbar.oracles import DualBank

    script = load_script(monkeypatch)
    monkeypatch.setattr(script, "baselines", lambda: {"drift": DriftAgent})
    monkeypatch.setattr(
        DualBank, "build", staticmethod(lambda view, starts, **options: PartlyUncertifiedBank(view))
    )
    gap = vbar.metrics.optimality_gap
    held = []

    def recorded_gap(ep, bounds):
        held.append(bounds)
        return gap(ep, bounds)

    monkeypatch.setattr(vbar.metrics, "optimality_gap", recorded_gap)
    argv = ["--scenario", SID, "--agent", "drift", "--count", "3", "--max-steps", "3"]
    status = script.main(argv + ["--violation-samples", "2"])
    captured = capsys.readouterr()
    # Three steps leave no time for the dwell, so no run succeeds and nothing is certified.
    assert status == 3
    rows = [line.split() for line in captured.out.splitlines() if line.split()[:1] == [SID]]
    assert len(rows) == 1 and int(rows[0][2]) == 3
    # Runs are flown in seed order; the uncertified start's run is held to no bound.
    assert [bounds is None for bounds in held] == [False, True, False]
    assert all(np.isfinite(bounds.lower) for bounds in held if bounds is not None)
    message = f"{SID}: 1 of 3 starts uncertified, validation indices 1"
    assert message in captured.err or message in caplog.text
