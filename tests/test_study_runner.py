"""The supervisor study's runner: paired runs, per-episode budgets, looks and comparisons.

Every flight is approach-vbar cut to 12 steps (the study's environments are patched to a short
horizon), so no episode succeeds and a run takes about a second; the stand-in model is scripted
and priced, and no paid model is called.
"""

import json
import math

import numpy as np
import pytest

from vbar.agents import SUPERVISOR_PROMPT, Price, Reply
from vbar.eval import read_results
from vbar.eval import study as study_module
from vbar.eval.study import (
    REQUEST_BYTES,
    Cell,
    Row,
    StudyModel,
    compare,
    episode_outcome,
    run_cell,
    run_row,
    side_summary,
    summarize_cell,
)
from vbar.safety import PassThroughFilter
from vbar.scenarios import make as catalogue_make

PRICE = Price(input_per_mtok=1.0, output_per_mtok=10.0)
OFF = Cell("approach-vbar", False)
ON = Cell("approach-vbar", True)
CALLS = []


@pytest.fixture(autouse=True)
def short_flights(monkeypatch):
    monkeypatch.setattr(
        study_module, "make", lambda name, **kw: catalogue_make(name, max_steps=12, **kw)
    )
    CALLS.clear()


def go_when_valid():
    """A stand-in model: GO when the prompt lists it as valid, else CONTINUE."""

    def complete(messages, tools):
        CALLS.append(1)
        prompt = messages[-1]["content"]
        valid = prompt.split("Commands you may give now:", 1)[-1].split("\n", 1)[0]
        command = "GO" if "GO" in [c.strip(" .") for c in valid.split(",")] else "CONTINUE"
        return Reply(json.dumps({"command": command}), 1000, 20)

    return complete


MODEL = StudyModel("stand-in", go_when_valid, PRICE, max_output_tokens=100)
FREE = StudyModel("stand-in-free", go_when_valid)
# The one call the runner keeps each episode below, and a ceiling well above an episode's spend
# (a few consults of 1000 and 20 tokens, about 0.0012 dollars each).
WORST = PRICE.cost(REQUEST_BYTES, 100)
CEILING = WORST + 0.02


def test_a_fixed_row_flies_every_seed_paired_with_the_rule_supervisor(tmp_path):
    row = Row(name="slice", cells=(OFF,), seeds=3)
    report = run_cell(row, OFF, FREE, out_dir=tmp_path, reps=200)
    assert report["status"] == "fixed" and report["look"] == 3 and report["pairs"] == 3
    studied = read_results(tmp_path / "slice" / "approach-vbar_filter_off.jsonl")
    rule = read_results(tmp_path / "rule" / "approach-vbar_filter_off.jsonl")
    assert [r["seed"] for r in studied] == [r["seed"] for r in rule]
    assert [r["agent_seed"] for r in studied] == [r["agent_seed"] for r in rule]
    assert {r["agent"] for r in studied} == {"supervised-llm"}
    assert {r["agent"] for r in rule} == {"supervised-rules"}
    assert report["studied"]["outcomes"]["timeout"] == 3
    assert report["measures"]["success"]["verdict"] == "equivalent"
    assert "intervention_count" not in report["measures"]
    assert report["studied"]["chattering"] == 0.0
    assert report["disagreement"]["consults"] == 6


def test_a_sequential_row_stops_at_the_first_look_where_every_measure_is_decided(tmp_path):
    row = Row(name="seq", cells=(OFF,), seeds=4, looks=(2, 4))
    report = run_cell(row, OFF, FREE, out_dir=tmp_path, reps=200)
    assert report["status"] == "decided" and report["look"] == 2
    assert len(read_results(tmp_path / "seq" / "approach-vbar_filter_off.jsonl")) == 2
    assert len(CALLS) == 2 * 2


def test_a_cell_with_the_filter_on_judges_the_intervention_count(tmp_path):
    row = Row(name="on", cells=(ON,), seeds=2)
    report = run_cell(row, ON, FREE, out_dir=tmp_path, safety_filter=PassThroughFilter, reps=200)
    count = report["measures"]["intervention_count"]
    assert count["n"] == 2 and count["estimate"] == 0.0 and count["verdict"] == "equivalent"


def test_a_rerun_resumes_from_the_files_without_calling_the_model(tmp_path):
    row = Row(name="slice", cells=(OFF,), seeds=2)
    first = run_cell(row, OFF, FREE, out_dir=tmp_path, reps=200)
    calls = len(CALLS)
    again = run_cell(row, OFF, FREE, out_dir=tmp_path, reps=200)
    assert len(CALLS) == calls
    assert again["measures"] == first["measures"]


def admitted_within_share(report, row):
    """Every episode started fit the share, and one more ceiling would not have."""
    spent = report["studied"]["spent_usd"]
    assert report["status"] == "budget"
    assert spent <= row.share_usd < spent + row.episode_usd


def test_the_cell_share_bounds_how_many_episodes_start(tmp_path):
    # Room for one ceiling and a few episodes' spend: a few start, the sixth never does.
    row = Row(name="capped", cells=(OFF,), seeds=6, cap_usd=CEILING + 0.005, episode_usd=CEILING)
    report = run_cell(row, OFF, MODEL, out_dir=tmp_path, reps=200)
    admitted_within_share(report, row)
    assert 1 <= report["look"] < 6
    assert report["studied"]["unfinished"] == 0


def test_a_batch_of_workers_starts_only_as_many_episodes_as_the_share_holds(tmp_path, monkeypatch):
    # Each episode reports 0.4 ceilings against a share of 2.5: two start (2.5 left), then one
    # (1.7 left), then one (1.3 left), then none (0.9 left, less than a ceiling).
    batches = []

    def fake_run_jobs(jobs, agent, **options):
        if agent is study_module.rule_agent:
            return [result("timeout") for _ in jobs]
        batches.append(len(jobs))
        return [result("timeout", cost_usd=0.4 * CEILING) for _ in jobs]

    monkeypatch.setattr(study_module, "run_jobs", fake_run_jobs)
    row = Row(name="pair", cells=(OFF,), seeds=6, cap_usd=2.5 * CEILING, episode_usd=CEILING)
    report = run_cell(row, OFF, MODEL, out_dir=tmp_path, workers=2, reps=200)
    assert batches == [2, 1, 1]
    assert report["status"] == "budget" and report["look"] == 4


def test_an_episode_its_budget_stops_is_unfinished_and_charged_its_ceiling(tmp_path):
    # The ceiling less one worst-case call leaves less than the first call's worst case.
    ceiling = WORST + 0.001
    row = Row(name="cut", cells=(OFF,), seeds=2, cap_usd=1.5 * ceiling, episode_usd=ceiling)
    report = run_cell(row, OFF, MODEL, out_dir=tmp_path, reps=200)
    studied = read_results(tmp_path / "cut" / "approach-vbar_filter_off.jsonl")
    assert studied[0]["error"].startswith("BudgetExceeded:")
    assert episode_outcome(studied[0]) is None and not CALLS
    assert report["studied"]["unfinished"] == 1 and report["pairs"] == 0
    # The unfinished run is charged a whole ceiling, so the second never starts.
    assert report["status"] == "budget" and report["look"] == 1


def test_a_run_past_its_wall_limit_is_a_timeout(tmp_path, monkeypatch):
    row = Row(name="slow", cells=(OFF,), seeds=1, wall_limit_s=1e-9)
    report = run_cell(row, OFF, FREE, out_dir=tmp_path, reps=200)
    studied = read_results(tmp_path / "slow" / "approach-vbar_filter_off.jsonl")
    assert studied[0]["error"].startswith("WallLimitReached:")
    assert report["studied"]["outcomes"]["timeout"] == 1 and report["pairs"] == 1


def test_run_row_flies_every_cell(tmp_path):
    row = Row(name="both", cells=(OFF, Cell("approach-rbar", False)), seeds=1)
    reports = run_row(row, FREE, out_dir=tmp_path, reps=200)
    assert [r["cell"] for r in reports] == [c.label for c in row.cells]


def one(cell, **fields):
    return Row(name="r", cells=(cell,), seeds=1, **fields)


@pytest.mark.parametrize(
    ("row", "cell", "model", "options", "match"),
    [
        (Row(name="r", cells=(OFF,), seeds=1), ON, FREE, {}, "not a cell"),
        (Row(name="r", cells=(ON,), seeds=1), ON, FREE, {}, "safety_filter"),
        (Row(name="r", cells=(OFF,), seeds=1, own_prompt=True), OFF, FREE, {}, "prompt"),
        (
            Row(name="r", cells=(OFF,), seeds=1, cap_usd=1.0, episode_usd=1.0),
            OFF,
            FREE,
            {},
            "price",
        ),
        (
            Row(name="r", cells=(OFF,), seeds=1, cap_usd=1e-3, episode_usd=1e-3),
            OFF,
            MODEL,
            {},
            "worst",
        ),
        (Row(name="r", cells=(OFF,), seeds=1), OFF, FREE, {"workers": 0}, "workers"),
        (Row(name="r", cells=(OFF,), seeds=1), OFF, FREE, {"workers": 1.0}, "workers"),
    ],
)
def test_run_cell_refuses_what_it_cannot_fly(tmp_path, row, cell, model, options, match):
    with pytest.raises(ValueError, match=match):
        run_cell(row, cell, model, out_dir=tmp_path, **options)


@pytest.mark.parametrize(
    "fields",
    [
        {"label": ""},
        {"complete": "go"},
        {"price": (1.0, 10.0), "max_output_tokens": 100},
        {"price": PRICE},
        {"max_output_tokens": 100},
        {"price": PRICE, "max_output_tokens": 0},
        {"price": PRICE, "max_output_tokens": True},
    ],
)
def test_a_study_model_refuses_bad_fields(fields):
    with pytest.raises(ValueError):
        StudyModel(**{"label": "m", "complete": go_when_valid, **fields})


def test_equal_paired_outcomes_are_equivalent():
    found = compare([0.0] * 50, 0.05, 0.975, reps=500)
    assert found["verdict"] == "equivalent" and found["tost"] == "equivalent" and found["decided"]
    assert found["lower"] == found["upper"] == 0.0


def test_a_difference_wholly_past_a_margin_is_decided_and_different():
    found = compare([1.0] * 40 + [0.0] * 10, 0.05, 0.975, reps=500)
    assert found["verdict"] == "beyond_margin" and found["tost"] == "different"
    assert found["decided"] and found["lower"] > 0.05


def test_a_wide_interval_around_zero_is_undecided_and_not_equivalent():
    found = compare([1.0, -1.0] * 10, 0.05, 0.975, reps=500)
    assert found["verdict"] == "undecided" and found["tost"] == "not_equivalent"
    assert not found["decided"]


def test_an_interval_off_zero_that_reaches_inside_the_margin_is_different_but_undecided():
    found = compare([1.0] * 3 + [0.0] * 57, 0.05, 0.5, reps=2000, rng=1)
    assert found["lower"] > 0.0 and found["lower"] < 0.05 < found["upper"]
    assert found["tost"] == "different" and found["verdict"] == "undecided"


def test_the_interval_is_the_percentile_bootstrap_of_the_mean():
    d = np.random.default_rng(3).normal(0.0, 1.0, 60)
    found = compare(d, 0.5, 0.9, reps=4000, rng=7)
    means = d[np.random.default_rng(7).integers(0, 60, size=(4000, 60))].mean(axis=1)
    assert found["lower"] == np.quantile(means, 0.05, method="inverted_cdf")
    assert found["upper"] == np.quantile(means, 0.95, method="inverted_cdf")
    wider = compare(d, 0.5, 0.975, reps=4000, rng=7)
    assert wider["lower"] < found["lower"] and wider["upper"] > found["upper"]


def test_no_pairs_leave_a_measure_undecided():
    found = compare([], 0.05, 0.9)
    assert found["n"] == 0 and math.isnan(found["estimate"]) and not found["decided"]


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (([math.nan], 0.05, 0.9), "finite"),
        (([0.0], 0.0, 0.9), "margin"),
        (([0.0], 0.05, 1.0), "confidence"),
        (([0.0], 0.05, 0.9, 99), "reps"),
    ],
)
def test_compare_refuses_bad_input(args, match):
    with pytest.raises(ValueError, match=match):
        compare(*args)


def result(outcome=None, error=None, **metrics):
    base = {
        "outcome": outcome,
        "consults": 4,
        "decision_points": 1,
        "consults_differing": 1,
        "decision_points_differing": 0,
        "chattering": 0.5,
        "input_tokens": 100,
        "output_tokens": 10,
        "cost_usd": 0.01,
        "parse_failures": 1,
        "intervention_count": 2,
    }
    return {"error": error, "wall_s": 1.0, "metrics": {**base, **metrics}}


def test_the_rates_count_timeouts_as_failures_and_errors_as_neither_safe_nor_unsafe():
    runs = [
        result("success"),
        result("safe_abort"),
        result("timeout"),
        result("unsafe"),
        result(error="RuntimeError: provider down"),
        result(error="BudgetExceeded: spent"),
        result(error="WallLimitReached: slow"),
    ]
    side = side_summary(runs, faulted=True)
    assert side["unfinished"] == 1
    assert side["outcomes"] == {
        "success": 1,
        "safe_abort": 1,
        "timeout": 2,
        "unsafe": 1,
        "error": 1,
    }
    assert side["mission_success"]["k"] == 1 and side["mission_success"]["n"] == 6
    assert side["unsafe"]["k"] == 1 and side["unsafe"]["n"] == 5
    assert side["abort_when_faulted"]["k"] == 1 and side["abort_when_faulted"]["n"] == 6
    assert 0.0 < side["mission_success"]["lower"] < 1 / 6 < side["mission_success"]["upper"]
    assert side["invalid_action_rate"] == pytest.approx(4 / 16)
    assert "abort_when_faulted" not in side_summary(runs, faulted=False)


def test_pairs_need_both_episodes_finished_and_measure_success_as_studied_minus_rule():
    row = Row(name="r", cells=(ON,), seeds=4)
    studied = [
        result("success"),
        result("timeout"),
        result(error="BudgetExceeded: spent"),
        result("success", intervention_count=5),
    ]
    rule = [result("timeout"), result("timeout"), result("success"), result("success")]
    report = summarize_cell(row, ON, studied, rule, reps=200)
    assert report["pairs"] == 3
    assert report["measures"]["success"]["estimate"] == pytest.approx(1 / 3)
    assert report["measures"]["intervention_count"]["estimate"] == pytest.approx(1.0)
    assert report["disagreement"]["consults"] == 12
    assert report["disagreement"]["disagreement"] == pytest.approx(3 / 12)


def test_runs_of_another_model_or_prompt_are_not_resumed(tmp_path):
    row = Row(name="slice", cells=(OFF,), seeds=1)
    run_cell(row, OFF, FREE, out_dir=tmp_path, reps=200)
    calls = len(CALLS)
    run_cell(row, OFF, StudyModel("other", go_when_valid), out_dir=tmp_path, reps=200)
    assert len(CALLS) > calls
    calls = len(CALLS)
    run_cell(row, OFF, FREE, out_dir=tmp_path, prompt=SUPERVISOR_PROMPT + " ", reps=200)
    assert len(CALLS) > calls


class BudgetExceeded(RuntimeError):
    """Not the harness's budget refusal, only its name."""


def raising(error):
    def factory():
        def complete(messages, tools):
            raise error

        return complete

    return factory


@pytest.mark.parametrize(
    "error",
    [
        BudgetExceeded("spent"),
        study_module.BudgetExceeded("spent"),
        study_module.WallLimitReached("slow"),
        ValueError("bad"),
    ],
)
def test_an_error_the_model_raises_is_an_error_whatever_its_name(tmp_path, error):
    model = StudyModel("raising", raising(error), PRICE, max_output_tokens=100)
    row = Row(name="raise", cells=(OFF,), seeds=1, cap_usd=CEILING, episode_usd=CEILING)
    report = run_cell(row, OFF, model, out_dir=tmp_path, reps=200)
    studied = read_results(tmp_path / "raise" / "approach-vbar_filter_off.jsonl")
    assert studied[0]["error"].startswith("ModelError:")
    assert episode_outcome(studied[0]) == "error"
    assert report["pairs"] == 1 and report["studied"]["unfinished"] == 0


def miscounting():
    def complete(messages, tools):
        return Reply(json.dumps({"command": "CONTINUE"}), -5, math.nan)

    return complete


def test_token_counts_that_are_not_whole_and_positive_are_recorded_as_none_reported(tmp_path):
    model = StudyModel("miscounting", miscounting, PRICE, max_output_tokens=100)
    row = Row(name="count", cells=(OFF,), seeds=1, cap_usd=CEILING, episode_usd=CEILING)
    report = run_cell(row, OFF, model, out_dir=tmp_path, reps=200)
    assert report["studied"]["input_tokens"] == 0 and report["studied"]["output_tokens"] == 0
    # Without counts every call is charged its worst case, so the spend is not zero.
    assert report["studied"]["cost_usd"] > 0.0


class Trapped(str):
    """Text whose methods raise the budget's refusal once the parser reads it."""

    def __str__(self):
        return self

    def find(self, *args, **kwargs):
        raise study_module.BudgetExceeded("spent")

    def __getitem__(self, key):
        raise study_module.BudgetExceeded("spent")


def trapping():
    def complete(messages, tools):
        return Reply(Trapped(json.dumps({"command": "CONTINUE"})), 1000, 20)

    return complete


def test_reply_text_is_read_as_plain_text_so_its_methods_cannot_stop_the_run(tmp_path):
    model = StudyModel("trapping", trapping, PRICE, max_output_tokens=100)
    row = Row(name="trap", cells=(OFF,), seeds=1, cap_usd=CEILING, episode_usd=CEILING)
    report = run_cell(row, OFF, model, out_dir=tmp_path, reps=200)
    studied = read_results(tmp_path / "trap" / "approach-vbar_filter_off.jsonl")
    assert studied[0]["error"] is None and episode_outcome(studied[0]) == "timeout"
    assert report["pairs"] == 1
