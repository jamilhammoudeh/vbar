"""The suite runner, the result files and the provenance hash."""

import json
import math

import numpy as np
import pytest
from eval_cases import Crashes, ShortTask

from vbar.agents import DriftAgent, PDAgent
from vbar.eval import Job, read_results, run_jobs, suite_jobs
from vbar.eval.provenance import provenance_fields, provenance_hash, settings_of
from vbar.eval.results import (
    decode,
    encode,
    load_trajectories,
    reproducible,
    save_trajectories,
)
from vbar.metrics import record_episode
from vbar.safety import PassThroughFilter
from vbar.scenarios import SUITES, make, seeds


def jobs(name="approach-vbar", count=3):
    pairs = seeds(name, "validation", count).pairs()
    return [Job(name, "validation", k, s, a, "test-suite") for k, (s, a) in enumerate(pairs)]


def test_results_do_not_depend_on_the_number_of_workers():
    work = jobs() + jobs("approach-rbar", 2)
    serial = run_jobs(work, PDAgent, task=ShortTask())
    parallel = run_jobs(work, PDAgent, task=ShortTask(), workers=2)
    assert [reproducible(r) for r in serial] == [reproducible(r) for r in parallel]
    assert [r["scenario"] for r in serial] == [j.scenario for j in work]
    assert len({r["provenance"] for r in serial}) == len(work)


def test_an_interrupted_run_resumes_without_flying_a_run_twice(tmp_path):
    out = tmp_path / "results.jsonl"
    work = jobs(count=4)
    first = run_jobs(work[:2], PDAgent, task=ShortTask(), out=out)
    assert len(read_results(out)) == 2
    resumed = run_jobs(work, PDAgent, task=ShortTask(), out=out)
    stored = read_results(out)
    assert len(stored) == 4 and [r["index"] for r in stored] == [0, 1, 2, 3]
    assert [reproducible(r) for r in resumed[:2]] == [reproducible(r) for r in first]
    again = run_jobs(work, PDAgent, task=ShortTask(), out=out)
    assert len(read_results(out)) == 4
    assert [r["provenance"] for r in again] == [r["provenance"] for r in resumed]


def test_a_crashing_agent_is_recorded_as_a_failure_not_dropped(tmp_path):
    out = tmp_path / "results.jsonl"
    results = run_jobs(jobs(count=2), Crashes, task=ShortTask(), out=out)
    assert len(results) == 2 and len(read_results(out)) == 2
    for result in results:
        assert result["error"] == "RuntimeError: controller fault"
        assert result["metrics"]["success"] is False and result["failure_label"] == "error"
        assert result["provenance"] is not None


def test_a_filter_and_a_factory_path_are_recorded():
    result = run_jobs(
        jobs(count=1),
        "vbar.agents:PDAgent",
        task=ShortTask(),
        safety_filter=PassThroughFilter,
    )[0]
    assert (result["agent"], result["filter"]) == ("pd", "none")
    plain = run_jobs(jobs(count=1), PDAgent, task=ShortTask())[0]
    assert plain["filter"] is None and plain["provenance"] != result["provenance"]
    with pytest.raises(ValueError, match="module:attribute"):
        run_jobs(jobs(count=1), "vbar.agents.PDAgent", task=ShortTask())


def test_a_suite_lists_one_job_per_run_in_suite_order():
    suite = SUITES["core-v1"]
    work = suite_jobs(suite)
    assert [(j.seed, j.agent_seed) for j in work] == [(s, a) for _, s, a in suite.runs()]
    assert {j.suite for j in work} == {"core-v1"} and work[100].scenario == "approach-vbar:nav1"


def base_fields(**changes):
    agent = changes.pop("agent", PDAgent())
    fields = {
        "task": "space",
        "identity": {"scenario_id": "vbar:approach-vbar:cw:box:dv:v1", "dynamics": "cw"},
        "seed": 2**127 + 3,
        "agent_seed": 2**126 + 5,
        "agent": agent,
        "safety_filter": None,
        **changes,
    }
    return provenance_hash(provenance_fields(**fields))


def test_the_provenance_hash_changes_with_every_field_it_covers():
    reference = base_fields()
    assert base_fields() == reference
    changed = [
        base_fields(seed=2**127 + 4),
        base_fields(agent_seed=2**126 + 6),
        base_fields(agent=PDAgent(wn=0.02)),
        base_fields(agent=DriftAgent()),
        base_fields(safety_filter=PassThroughFilter()),
        base_fields(
            identity={"scenario_id": "vbar:approach-vbar:cw:box:dv:v1:nav1", "dynamics": "cw"}
        ),
        base_fields(
            identity={"scenario_id": "vbar:approach-vbar:cw:box:dv:v1", "dynamics": "twobody"}
        ),
        base_fields(task="other"),
    ]
    assert len({reference, *changed}) == len(changed) + 1
    assert len(reference) == 64 and int(reference, 16) >= 0


def test_agent_settings_are_its_declared_config_or_its_plain_attributes():
    assert settings_of(PDAgent(wn=0.02)) == {"name": "pd", "wn": 0.02, "zeta": 1.0}

    class Declared(PDAgent):
        config = {"gains": (1.0, 2.0), "limit": math.inf}

    assert settings_of(Declared()) == {"gains": [1.0, 2.0], "limit": "inf"}


def test_result_lines_keep_128_bit_seeds_and_non_finite_metrics_exactly():
    seed = 2**128 - 1
    result = {
        "schema": 1,
        "task": "space",
        "suite": None,
        "scenario": "inspection",
        "scenario_id": "vbar:inspection:cw:box:dv:v2",
        "split": "test",
        "index": 7,
        "seed": seed,
        "agent_seed": seed - 2,
        "agent": "nan",
        "filter": None,
        "metrics": {
            "success": True,
            "terminal_pos_error": math.nan,
            "min_time_to_keep_out": math.inf,
            "min_margin": {"keep_out": -math.inf, "cone": np.float64(2.5)},
            "infractions": {"keep_out": np.int64(1)},
            "labels": ("a", "b"),
        },
        "failure_label": None,
        "error": None,
        "provenance": "ab" * 32,
        "environment": {"vbar": "0.3.1"},
        "wall_s": 1.5,
    }
    line = encode(result)
    json.loads(line)  # strict JSON: no NaN or Infinity tokens
    assert "NaN" not in line and "Infinity" not in line
    back = decode(line)
    assert back["seed"] == seed and back["agent_seed"] == seed - 2
    assert math.isnan(back["metrics"]["terminal_pos_error"])
    assert back["metrics"]["min_time_to_keep_out"] == math.inf
    assert back["metrics"]["min_margin"] == {"keep_out": -math.inf, "cone": 2.5}
    # Only numeric fields are read back from text.
    assert back["agent"] == "nan"
    with pytest.raises(ValueError, match="schema"):
        encode({**result, "extra": 1})
    with pytest.raises(ValueError, match="schema"):
        decode(json.dumps({"schema": 2}))


def test_trajectories_round_trip_by_provenance(tmp_path):
    env = make("approach-vbar", max_steps=3)
    episode = record_episode(env, PDAgent(), seed=11, agent_seed=12)
    path = tmp_path / "runs.npz"
    save_trajectories(path, {"abc": episode})
    loaded = load_trajectories(path)["abc"]
    assert np.array_equal(loaded["states"], episode.states)
    assert np.array_equal(loaded["times"], episode.times)
    assert loaded["commands"].shape == (3, 3) and loaded["obs"].shape == (3, 6)


def test_the_workers_count_is_checked():
    for workers in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="workers"):
            run_jobs(jobs(count=1), PDAgent, task=ShortTask(), workers=workers)


RESULT = {
    "schema": 1,
    "task": "space",
    "suite": None,
    "scenario": "inspection",
    "scenario_id": "vbar:inspection:cw:box:dv:v2",
    "split": "test",
    "index": 7,
    "seed": 11,
    "agent_seed": 12,
    "agent": "pd",
    "filter": None,
    "metrics": {"success": True},
    "failure_label": None,
    "error": None,
    "provenance": "ab" * 32,
    "environment": {"vbar": "0.3.1"},
    "wall_s": 1.5,
}
JOB = {"scenario": "approach-vbar", "split": "validation", "index": 0, "seed": 1, "agent_seed": 2}
NOT_A_SEED = [
    ("fractional", 1.5),
    ("nan", math.nan),
    ("true", True),
    ("negative", -1),
]


def line(**changes):
    return json.dumps({**json.loads(encode(RESULT)), **changes})


def hostile(function, label, match=None, **arguments):
    return pytest.param(function, arguments, match, id=f"{function.__name__}-{label}")


@pytest.mark.parametrize(
    "function, arguments, match",
    [
        hostile(decode, "a-list", line="[1, 2]"),
        hostile(decode, "no-fields", line="{}"),
        hostile(
            decode,
            "missing-fields",
            r"(?=.*agent_seed)(?=.*metrics)(?=.*provenance)",
            line='{"schema": 1}',
        ),
        hostile(decode, "seed-text", r"\bseed\b", line=line(seed="abc")),
        hostile(decode, "seed-fractional", r"\bseed\b", line=line(seed=1.5)),
        hostile(decode, "seed-null", r"\bseed\b", line=line(seed=None)),
        hostile(decode, "seed-true", r"\bseed\b", line=line(seed=True)),
        hostile(decode, "seed-negative", r"\bseed\b", line=line(seed=-1)),
        hostile(decode, "agent_seed-negative-text", "agent_seed", line=line(agent_seed="-3")),
        hostile(decode, "metrics-a-list", "metrics", line=line(metrics=[])),
        *[
            hostile(
                encode,
                f"{name}-{label}",
                name if name == "agent_seed" else r"\bseed\b",
                result={**RESULT, name: value},
            )
            for name in ("seed", "agent_seed")
            for label, value in (*NOT_A_SEED, ("text", "7"))
        ],
        *[
            hostile(Job, f"{name}-{label}", **{**JOB, name: value})
            for name in ("seed", "agent_seed")
            for label, value in (*NOT_A_SEED, ("none", None))
        ],
        *[
            hostile(Job, f"index-{label}", **{**JOB, "index": value})
            for label, value in (("negative", -1), ("fractional", 1.5), ("true", True))
        ],
        hostile(Job, "scenario-empty", **{**JOB, "scenario": ""}),
        hostile(Job, "scenario-number", **{**JOB, "scenario": 3}),
        hostile(Job, "split-number", **{**JOB, "split": 3}),
    ],
)
def test_hostile_result_and_job_inputs_are_refused(function, arguments, match):
    with pytest.raises(ValueError, match=match):
        function(**arguments)


def test_a_decimal_string_and_a_json_integer_are_both_read_as_a_seed():
    for seed in ("7", 7):
        found = decode(line(seed=seed, agent_seed=seed))
        assert (found["seed"], found["agent_seed"]) == (7, 7)


def test_numpy_integers_are_written_and_read_as_seeds_and_indices():
    written = {**RESULT, "seed": np.int64(5), "agent_seed": np.uint32(6)}
    assert decode(encode(written)) == decode(encode({**RESULT, "seed": 5, "agent_seed": 6}))
    job = Job("approach-vbar", "validation", np.int64(2), np.int64(5), np.uint32(6))
    assert (job.index, job.seed, job.agent_seed) == (2, 5, 6)
    # Gymnasium refuses a numpy integer as a seed, so a job holds Python integers.
    assert all(type(v) is int for v in (job.index, job.seed, job.agent_seed))


def test_a_job_takes_any_split_name_the_task_defines():
    for split in ("validation", "calibration", "planar-train"):
        assert Job("planar-strong", split, 0, 1, 2).split == split
