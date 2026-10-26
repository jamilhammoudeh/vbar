"""The harness runs a domain other than spacecraft rendezvous through the task interface alone."""

import pytest
from eval_toy import SUMMARIES, Brake, PlanarSuite, PlanarTask

from vbar.eval import run_suite
from vbar.eval.report import suite_report


def test_a_planar_double_integrator_runs_through_the_runner_and_the_report():
    suite = PlanarSuite()
    results = run_suite(suite, Brake, task=PlanarTask(), workers=2)
    assert len(results) == 40 and all(r["error"] is None for r in results)
    assert {r["task"] for r in results} == {"planar"}
    report = suite_report(suite, results, SUMMARIES, reps=200, task=PlanarTask())
    assert list(report["scenarios"]) == ["planar-weak", "planar-strong"]
    for row in report["scenarios"].values():
        assert set(row["metrics"]) == set(SUMMARIES)
        assert "worst_bound" in row["metrics"]["min_clearance"]
    # Some starts send the brake straight through the obstacle, so the domain's own success
    # predicate fails there; the report counts it the same way as for a spacecraft.
    rates = [row["metrics"]["success"]["rate"] for row in report["scenarios"].values()]
    assert all(0.0 < rate < 1.0 for rate in rates)
    with pytest.raises(ValueError, match="missing"):
        suite_report(suite, results[1:], SUMMARIES, reps=200, task=PlanarTask())
