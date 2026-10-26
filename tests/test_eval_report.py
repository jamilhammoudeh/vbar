"""The suite report and the command line that runs and rebuilds it."""

import json

import pytest
from eval_cases import ShortTask

from vbar.agents import DriftAgent, PDAgent
from vbar.eval import run_suite
from vbar.eval.__main__ import main, reproduce
from vbar.eval.report import scenario_summary, suite_report, to_markdown
from vbar.eval.results import read_results
from vbar.scenarios import Suite, SuiteEntry

SMALL = Suite(
    "small", 1, (SuiteEntry("approach-vbar", count=3), SuiteEntry("docking-afrl", count=2))
)


@pytest.fixture(scope="module")
def pd_results():
    return run_suite(SMALL, PDAgent, task=ShortTask())


def test_a_complete_suite_is_reported_per_scenario_with_intervals(pd_results):
    report = suite_report(SMALL, pd_results, reps=200, task=ShortTask())
    assert report["suite"] == "small-v1" and report["runs"] == 5
    assert list(report["scenarios"]) == ["approach-vbar", "docking-afrl"]
    row = report["scenarios"]["approach-vbar"]
    assert row["runs"] == 3 and row["errors"] == 0
    success = row["metrics"]["success"]
    assert success["n"] == 3 and 0.0 <= success["lower"] <= success["upper"] <= 1.0
    delta_v = row["metrics"]["delta_v"]
    assert delta_v["lower"] <= delta_v["iqm"] <= delta_v["upper"]
    # Safety metrics take their unsafe tail, never an interquartile mean.
    margin = row["metrics"]["min_margin"]["keep_out"]
    assert "iqm" not in margin and {"worst", "quantile_interval", "worst_bound"} <= set(margin)
    assert sum(row["failure_labels"].values()) == 3 - success["count"]


def test_a_report_missing_a_run_or_mixing_agents_is_refused(pd_results):
    with pytest.raises(ValueError, match="1 of 5 runs"):
        suite_report(SMALL, pd_results[:-1], reps=200, task=ShortTask())
    drift = run_suite(SMALL, DriftAgent, task=ShortTask())
    with pytest.raises(ValueError, match="one agent"):
        suite_report(SMALL, pd_results[:-1] + drift[-1:], reps=200, task=ShortTask())


def test_a_report_refuses_two_results_for_one_run(pd_results):
    again = [{**r, "provenance": "other"} for r in pd_results[:1]]
    with pytest.raises(ValueError, match="one result per run"):
        suite_report(SMALL, pd_results + again, reps=200, task=ShortTask())


def test_the_report_hash_covers_every_run_and_not_their_order(pd_results):
    forward = suite_report(SMALL, pd_results, reps=200, task=ShortTask())
    backward = suite_report(SMALL, list(reversed(pd_results)), reps=200, task=ShortTask())
    assert forward["provenance"] == backward["provenance"]
    assert forward == backward
    drift = suite_report(
        SMALL, run_suite(SMALL, DriftAgent, task=ShortTask()), reps=200, task=ShortTask()
    )
    assert drift["provenance"] != forward["provenance"]


def test_an_errored_run_counts_in_the_report():
    from eval_cases import Crashes

    results = run_suite(SMALL, Crashes, task=ShortTask())
    report = suite_report(SMALL, results, reps=200, task=ShortTask())
    row = report["scenarios"]["docking-afrl"]
    assert row["errors"] == 2 and row["failure_labels"] == {"error": 2}
    assert row["metrics"]["success"]["count"] == 0 and row["metrics"]["success"]["n"] == 2


def test_the_markdown_table_has_a_row_per_scenario(pd_results):
    text = to_markdown(suite_report(SMALL, pd_results, reps=200, task=ShortTask()))
    lines = text.splitlines()
    assert lines[0] == "# small-v1: pd"
    assert sum(line.startswith("| approach-vbar |") for line in lines) == 1
    assert sum(line.startswith("| docking-afrl |") for line in lines) == 1


def test_a_scenario_summary_leaves_out_metrics_no_run_has():
    found = scenario_summary([{"metrics": {"success": True}, "failure_label": None}], reps=100)
    assert set(found["metrics"]) == {"success"}


def test_the_command_line_runs_and_reports_a_suite(tmp_path, capsys, monkeypatch):
    from vbar.scenarios import SUITES

    monkeypatch.setitem(SUITES, SMALL.id, SMALL)
    out = tmp_path / "pd.jsonl"
    # The command line flies the catalogue scenarios in full; a short agent factory keeps it quick.
    assert (
        main(["run", "--suite", SMALL.id, "--agent", "eval_cases:Crashes", "--out", str(out)]) == 0
    )
    assert len(read_results(out)) == 5
    markdown = tmp_path / "pd.md"
    assert (
        main(["report", str(out), "--markdown", str(markdown), "--json", str(tmp_path / "r.json")])
        == 0
    )
    assert markdown.read_text(encoding="utf-8").startswith("# small-v1: pd")
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["runs"] == 5
    assert "5 runs" in capsys.readouterr().out


def test_a_manifest_rebuilds_its_tables_without_flying_a_run_twice(tmp_path, monkeypatch):
    from vbar.scenarios import SUITES

    monkeypatch.setitem(SUITES, SMALL.id, SMALL)
    manifest = tmp_path / "reproduce.toml"
    manifest.write_text(
        "[[table]]\n"
        'name = "crash"\n'
        f'suite = "{SMALL.id}"\n'
        'agent = "eval_cases:Crashes"\n'
        'results = "crash.jsonl"\n'
        'markdown = "crash.md"\n',
        encoding="utf-8",
    )
    written = reproduce(manifest)
    assert written == [tmp_path / "crash.md"] and written[0].exists()
    reproduce(manifest)
    assert len(read_results(tmp_path / "crash.jsonl")) == 5
