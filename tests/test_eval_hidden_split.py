"""The hidden test split's result files hold keyed digests of the episode seeds, never the seeds."""

import hashlib
import hmac
import importlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from eval_cases import ShortTask
from hypothesis import given, settings
from hypothesis import strategies as st

from vbar.agents import PDAgent
from vbar.eval import Job, read_results, run_job, run_jobs, run_suite, suite_jobs
from vbar.eval.report import latest_results, suite_report
from vbar.eval.results import FIELDS, decode, encode, reproducible, write_results
from vbar.scenarios import SUITES, Suite, SuiteEntry, seeds
from vbar.scenarios.seeds import HIDDEN_KEY_VARIABLE

KEY = "0123456789abcdef" * 4
OTHER_KEY = "fedcba9876543210" * 4
LOG_VARIABLE = "VBAR_HIDDEN_SEED_LOG"
TEST_KEY_TEXT = "vbar test seeds " + KEY
VALIDATION_KEY_TEXT = "vbar validation seeds 1"
SEED = 2**127 + 12345
AGENT_SEED = 2**126 + 678
TINY = Suite("tiny-hidden", 1, (SuiteEntry("approach-vbar", "test", 3),))


@pytest.fixture(scope="module", autouse=True)
def throwaway_key():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HIDDEN_KEY_VARIABLE, KEY)
        patch.delenv(LOG_VARIABLE, raising=False)
        yield


def reference_digest(key_text, seed, index, stream="episode"):
    message = f"vbar seed digest|{stream}|{index}|{seed}"
    return hmac.new(key_text.encode(), message.encode(), hashlib.sha256).hexdigest()


def spellings(number):
    """Every way a seed is spelled in text: decimal, and hex with or without leading zeros."""
    return {str(number), f"{number:x}", f"{number:032x}", f"{number:X}", f"{number:032X}"}


def assert_holds_no_seed(text, pairs):
    for number in (n for pair in pairs for n in pair):
        for form in spellings(number):
            assert form not in text
    assert not re.search(r'"(?:agent_)?seed"\s*:\s*"\d', text)


def digests(split, seed, agent_seed, index):
    from vbar.scenarios.seeds import seed_digest

    try:
        return seed_digest(split, seed, index), seed_digest(split, agent_seed, index, "agent")
    except ValueError:
        return None, None


def built(split="test", index=0, seed=SEED, agent_seed=AGENT_SEED):
    """A result of the schema with digests computed from the given seeds."""
    seed_hash, agent_hash = digests(split, seed, agent_seed, index)
    return {
        "schema": 1,
        "task": "space",
        "suite": "tiny-hidden-v1",
        "scenario": "approach-vbar",
        "scenario_id": "vbar:approach-vbar:cw:box:dv:v1",
        "split": split,
        "index": index,
        "seed": seed,
        "agent_seed": agent_seed,
        "seed_digest": seed_hash,
        "agent_seed_digest": agent_hash,
        "agent": "pd",
        "filter": None,
        "metrics": {"success": True},
        "failure_label": None,
        "error": None,
        "provenance": "ab" * 32,
        "environment": {"vbar": "0.3.1"},
        "wall_s": 1.5,
    }


def log_lines(path):
    return [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines()]


def stubs(raw=False):
    """Results of TINY as a published file holds them: scenario, no seed, the seed's digest."""
    from vbar.scenarios.seeds import seed_digest

    entry = TINY.entries[0]
    return [
        {
            "scenario_id": entry.scenario_id,
            "seed": seed if raw else None,
            "seed_digest": seed_digest("test", seed, k),
        }
        for k, (seed, _) in enumerate(entry.seeds().pairs())
    ]


def identifiers(results):
    return {"approach-vbar": results[0]["scenario_id"]}


def assert_published(results):
    assert len(results) == 3 and all(r["seed"] is None for r in results)
    assert all(re.fullmatch(r"[0-9a-f]{64}", r["seed_digest"]) for r in results)


@pytest.fixture(scope="module")
def tiny_run(throwaway_key, tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny") / "tiny.jsonl"
    flown = run_suite(TINY, PDAgent, task=ShortTask(), out=out)
    return flown, read_results(out)


def test_the_digest_and_log_names_are_part_of_the_seeds_module():
    module = importlib.import_module("vbar.scenarios.seeds")
    assert module.HIDDEN_LOG_VARIABLE == LOG_VARIABLE
    assert "seed_digest" in module.__all__


def test_a_core_test_result_line_holds_no_seed_only_digests(tmp_path):
    from vbar.scenarios.seeds import seed_digest

    work = suite_jobs(SUITES["core-test-v1"])[:2]
    assert [(j.scenario, j.split, j.index) for j in work] == [
        ("approach-vbar", "test", 0),
        ("approach-vbar", "test", 1),
    ]
    out = tmp_path / "core-test.jsonl"
    run_jobs(work, PDAgent, task=ShortTask(), out=out)
    text = out.read_text(encoding="utf-8")
    assert_holds_no_seed(text, [(j.seed, j.agent_seed) for j in work])
    assert KEY not in text
    raw_lines = [json.loads(line) for line in text.splitlines()]
    assert all(raw["seed"] is None and raw["agent_seed"] is None for raw in raw_lines)
    stored = read_results(out)
    assert len(stored) == 2
    for job, result in zip(work, stored, strict=True):
        assert result["error"] is None and result["suite"] == "core-test-v1"
        assert result["seed"] is None and result["agent_seed"] is None
        assert re.fullmatch(r"[0-9a-f]{64}", result["seed_digest"])
        assert re.fullmatch(r"[0-9a-f]{64}", result["agent_seed_digest"])
        assert result["seed_digest"] == seed_digest("test", job.seed, job.index)
        assert result["agent_seed_digest"] == seed_digest(
            "test", job.agent_seed, job.index, "agent"
        )
        assert result["seed_digest"] != result["agent_seed_digest"]
    assert stored[0]["seed_digest"] != stored[1]["seed_digest"]


@pytest.mark.parametrize("split", ["train", "validation"])
def test_a_public_split_file_keeps_the_decimal_seeds_and_holds_their_digests(tmp_path, split):
    from vbar.scenarios.seeds import seed_digest

    pairs = seeds("approach-vbar", split, 2).pairs()
    work = [Job("approach-vbar", split, k, s, a, "tiny") for k, (s, a) in enumerate(pairs)]
    out = tmp_path / "public.jsonl"
    run_jobs(work, PDAgent, task=ShortTask(), out=out)
    raw_lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    stored = read_results(out)
    for job, raw, result in zip(work, raw_lines, stored, strict=True):
        assert raw["seed"] == str(job.seed) and raw["agent_seed"] == str(job.agent_seed)
        assert (result["seed"], result["agent_seed"]) == (job.seed, job.agent_seed)
        assert raw["seed_digest"] == seed_digest(split, job.seed, job.index)
        assert raw["agent_seed_digest"] == seed_digest(split, job.agent_seed, job.index, "agent")


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_a_flown_job_records_both_digests_in_every_split(split):
    from vbar.scenarios.seeds import seed_digest

    seed, agent_seed = seeds("approach-vbar", split, 4).pairs()[3]
    job = Job("approach-vbar", split, 3, seed, agent_seed, "tiny")
    result = run_job(job, ShortTask(), PDAgent)
    assert result["error"] is None
    assert result["seed_digest"] == seed_digest(split, seed, 3)
    assert result["agent_seed_digest"] == seed_digest(split, agent_seed, 3, "agent")


def test_a_job_on_a_split_without_a_key_leaves_the_digests_empty():
    result = run_job(Job("approach-vbar", "calibration", 0, 1, 2), ShortTask(), PDAgent)
    assert result["error"] is None
    assert result["seed_digest"] is None and result["agent_seed_digest"] is None
    assert (result["seed"], result["agent_seed"]) == (1, 2)


def test_the_result_schema_lists_the_digest_fields():
    assert {"seed_digest", "agent_seed_digest"} <= set(FIELDS)
    doc = importlib.import_module("vbar.eval.results").__doc__
    assert re.search(r"(?<!agent_)seed_digest", doc) and "agent_seed_digest" in doc


def test_the_digest_is_the_keyed_hash_of_the_stream_index_and_seed(monkeypatch):
    from vbar.scenarios.seeds import seed_digest

    # Vectors computed once with hmac and sha256 from the standard library, so a change of the
    # message layout or the key is caught without the test repeating the code under test.
    seed = 2**127 + 5
    assert seed_digest("validation", seed, 3) == (
        "fd9a2706d70763760c5344571152f080adb1026fda7a70a49d5a0f3d775ab3da"
    )
    assert seed_digest("validation", seed, 3, "agent") == (
        "8c930204e102ad257b0215cb735385f3b6a1ec4273e6dca788c99f86a17ceae2"
    )
    monkeypatch.setenv(HIDDEN_KEY_VARIABLE, "ab" * 32)
    assert seed_digest("test", 7, 0) == (
        "d047609b510c289c77e92973b186834a719190cbd2de4674c222da9d722c1f17"
    )
    for split, key_text in (("train", "vbar train seeds 1"), ("validation", VALIDATION_KEY_TEXT)):
        assert seed_digest(split, SEED, 2) == reference_digest(key_text, SEED, 2)
        assert seed_digest(split, SEED, 2, "agent") == reference_digest(key_text, SEED, 2, "agent")


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 2**128 - 1),
    index=st.integers(0, 10**6),
    stream=st.sampled_from(["episode", "agent"]),
)
def test_the_digest_follows_the_reference_and_separates_neighbouring_inputs(seed, index, stream):
    from vbar.scenarios.seeds import seed_digest

    found = seed_digest("test", seed, index, stream)
    assert re.fullmatch(r"[0-9a-f]{64}", found)
    assert found == reference_digest(TEST_KEY_TEXT, seed, index, stream)
    assert found != seed_digest("test", seed + 1, index, stream)
    assert found != seed_digest("test", seed, index + 1, stream)


def test_the_digest_is_deterministic_and_changes_with_each_input(monkeypatch):
    from vbar.scenarios.seeds import seed_digest

    reference = seed_digest("test", SEED, 4)
    assert seed_digest("test", SEED, 4) == reference
    others = [
        seed_digest("test", SEED, 5),
        seed_digest("test", SEED + 1, 4),
        seed_digest("test", SEED, 4, "agent"),
        seed_digest("validation", SEED, 4),
        seed_digest("train", SEED, 4),
    ]
    assert len({reference, *others}) == len(others) + 1
    monkeypatch.setenv(HIDDEN_KEY_VARIABLE, OTHER_KEY)
    assert seed_digest("test", SEED, 4) != reference


def test_the_digest_is_not_a_plain_hash_of_the_seed():
    from vbar.scenarios.seeds import seed_digest

    message = f"vbar seed digest|episode|0|{SEED}"
    plain = {
        hashlib.sha256(text.encode()).hexdigest()
        for text in (str(SEED), f"{SEED:032x}", message, TEST_KEY_TEXT + message)
    }
    assert seed_digest("test", SEED, 0) not in plain


def test_the_digest_ignores_the_case_and_white_space_of_the_key(monkeypatch):
    from vbar.scenarios.seeds import seed_digest

    reference = seed_digest("test", SEED, 1)
    monkeypatch.setenv(HIDDEN_KEY_VARIABLE, f"  {KEY.upper()}\n")
    assert seed_digest("test", SEED, 1) == reference


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        pytest.param(("calibration", 5, 0), "split", id="unknown-split"),
        pytest.param(("validation", 5, -1), "index", id="negative-index"),
        pytest.param(("validation", -5, 0), "seed", id="negative-seed"),
        pytest.param(("validation", 1.5, 0), "seed", id="fractional-seed"),
        pytest.param(("validation", "5", 0), "seed", id="text-seed"),
        pytest.param(("validation", None, 0), "seed", id="no-seed"),
    ],
)
def test_the_digest_refuses_a_bad_split_index_or_seed(arguments, match):
    from vbar.scenarios.seeds import seed_digest

    with pytest.raises(ValueError, match=match):
        seed_digest(*arguments)
    seed_digest("validation", 0, 0)
    seed_digest("validation", 2**128 - 1, 0)


@pytest.mark.parametrize("key", [None, "", "ab" * 8])
def test_the_digest_of_the_test_split_needs_the_hidden_key(monkeypatch, key):
    from vbar.scenarios.seeds import seed_digest

    if key is None:
        monkeypatch.delenv(HIDDEN_KEY_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(HIDDEN_KEY_VARIABLE, key)
    with pytest.raises(ValueError, match=HIDDEN_KEY_VARIABLE):
        seed_digest("test", SEED, 0)
    seed_digest("validation", SEED, 0)


def test_a_test_result_is_written_with_null_seeds_whatever_the_dict_holds():
    for held in (SEED, np.int64(5), None):
        result = {**built("test"), "seed": held, "agent_seed": held}
        before = dict(result)
        line = encode(result)
        raw = json.loads(line)
        assert raw["seed"] is None and raw["agent_seed"] is None
        assert result == before
    line = encode(built("test"))
    assert_holds_no_seed(line, [(SEED, AGENT_SEED)])
    back = decode(line)
    assert back["seed"] is None and back["agent_seed"] is None
    assert back["seed_digest"] == built("test")["seed_digest"]
    assert back["agent_seed_digest"] == built("test")["agent_seed_digest"]


def test_only_decimal_and_hexadecimal_spellings_of_a_seed_are_searched_for():
    # The limit the docs state: a seed written another way is not found, so the line is written.
    for other in (f"{SEED:o}", f"{SEED:b}", f"{SEED:_}", f"{SEED:,}"):
        written = encode({**built("test"), "metrics": {"success": True, "note": other}})
        assert other in written


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"seed_digest": None}, id="no-seed-digest"),
        pytest.param({"agent_seed_digest": None}, id="no-agent-seed-digest"),
        pytest.param({"seed_digest": None, "agent_seed_digest": None}, id="neither"),
    ],
)
def test_a_test_result_without_digests_is_refused(changes):
    good = built("test")
    encode(good)
    with pytest.raises(ValueError, match="digest"):
        encode({**good, **changes})


def test_digests_are_required_only_on_the_test_split():
    result = {**built("validation"), "split": "calibration"}
    result["seed_digest"] = result["agent_seed_digest"] = None
    line = encode(result)
    raw = json.loads(line)
    assert raw["seed"] == str(SEED) and raw["agent_seed"] == str(AGENT_SEED)
    back = decode(line)
    assert (back["seed"], back["agent_seed"]) == (SEED, AGENT_SEED)
    assert back["seed_digest"] is None and back["agent_seed_digest"] is None


def line_of(split, **changes):
    return json.dumps({**json.loads(encode(built(split))), **changes})


@pytest.mark.parametrize("split", ["train", "validation", "calibration"])
def test_a_null_seed_is_refused_outside_the_test_split(split):
    base = "validation" if split == "calibration" else split
    result = {**built(base), "split": split}

    def line(**changes):
        return json.dumps({**json.loads(encode(result)), **changes})

    assert decode(line())["seed"] == SEED
    with pytest.raises(ValueError, match=r"\bseed\b"):
        decode(line(seed=None))
    with pytest.raises(ValueError, match="agent_seed"):
        decode(line(agent_seed=None))


def test_null_seeds_are_read_on_the_test_split_and_other_bad_seeds_are_not():
    both = decode(line_of("test", seed=None, agent_seed=None))
    assert both["seed"] is None and both["agent_seed"] is None
    for bad in ("abc", 1.5, True, -1):
        with pytest.raises(ValueError, match=r"\bseed\b"):
            decode(line_of("test", seed=bad))


def test_reproducible_text_of_a_published_result_holds_digests_and_no_seed(tiny_run):
    _, stored = tiny_run
    kept = json.loads(reproducible(stored[0]))
    assert kept["seed"] is None and kept["agent_seed"] is None
    assert kept["seed_digest"] == stored[0]["seed_digest"]
    assert reproducible(stored[0]) != reproducible(stored[1])


def test_a_flown_test_split_run_appends_its_seeds_to_the_private_log(tmp_path, monkeypatch):
    from vbar.scenarios.seeds import seed_digest

    log = tmp_path / "private-seeds.jsonl"
    monkeypatch.setenv(LOG_VARIABLE, str(log))
    work = suite_jobs(SUITES["core-test-v1"])[:2]
    out = tmp_path / "public.jsonl"
    run_jobs(work, PDAgent, task=ShortTask(), out=out)
    pairs = seeds("approach-vbar", "test", 100).pairs()
    stored = read_results(out)
    entries = log_lines(log)
    assert len(entries) == 2
    for index, entry in enumerate(entries):
        seed, agent_seed = pairs[index]
        assert entry == {
            "suite": "core-test-v1",
            "scenario_id": stored[index]["scenario_id"],
            "split": "test",
            "index": index,
            "seed": str(seed),
            "agent_seed": str(agent_seed),
            "seed_digest": seed_digest("test", seed, index),
            "agent_seed_digest": seed_digest("test", agent_seed, index, "agent"),
        }
    assert_holds_no_seed(out.read_text(encoding="utf-8"), pairs[:2])


def test_the_private_log_is_append_only_and_takes_test_split_results_only(tmp_path, monkeypatch):
    log = tmp_path / "private-seeds.jsonl"
    monkeypatch.setenv(LOG_VARIABLE, str(log))
    out = tmp_path / "public.jsonl"
    first = [built("validation", 0), built("test", 1, SEED + 1, AGENT_SEED + 1), built("train", 2)]
    write_results(out, [*first, built("test", 2, SEED + 2, AGENT_SEED + 2)])
    write_results(out, [built("test", 3, SEED + 3, AGENT_SEED + 3)], append=False)
    entries = log_lines(log)
    assert [entry["index"] for entry in entries] == [1, 2, 3]
    assert {entry["split"] for entry in entries} == {"test"}
    assert [entry["seed"] for entry in entries] == [str(SEED + k) for k in (1, 2, 3)]
    assert [entry["agent_seed"] for entry in entries] == [str(AGENT_SEED + k) for k in (1, 2, 3)]
    assert len(read_results(out)) == 1


@pytest.mark.parametrize("value", [None, ""])
def test_nothing_is_logged_while_the_log_variable_is_unset_or_empty(tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    if value is None:
        monkeypatch.delenv(LOG_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(LOG_VARIABLE, value)
    out = tmp_path / "public" / "results.jsonl"
    write_results(out, [built("test")])
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["results.jsonl"]


def test_runs_on_public_splits_write_nothing_to_the_private_log(tmp_path, monkeypatch):
    log = tmp_path / "private-seeds.jsonl"
    monkeypatch.setenv(LOG_VARIABLE, str(log))
    calibration = {**built("validation"), "split": "calibration"}
    calibration["seed_digest"] = calibration["agent_seed_digest"] = None
    write_results(tmp_path / "written.jsonl", [built("train"), built("validation"), calibration])
    seed, agent_seed = seeds("approach-vbar", "validation", 1).pairs()[0]
    job = Job("approach-vbar", "validation", 0, seed, agent_seed, "tiny")
    run_jobs([job], PDAgent, task=ShortTask(), out=tmp_path / "flown.jsonl")
    assert not log.exists() or log.read_text(encoding="utf-8") == ""


def test_an_unusable_private_log_raises_before_the_public_line_is_written(tmp_path, monkeypatch):
    # A directory cannot be opened for appending.
    monkeypatch.setenv(LOG_VARIABLE, str(tmp_path))
    out = tmp_path / "public.jsonl"
    with pytest.raises(OSError):
        write_results(out, [built("test")])
    assert not out.exists() or out.read_text(encoding="utf-8") == ""


# Small enough that their 32-digit hex forms start with zeros, and long enough that a chance
# match inside a digest is out of the question.
CRASH_SEEDS = (2**100 + 12345, 2**90 + 678)
SPELLED_AS = {
    "decimal": str,
    "hex": lambda number: f"{number:x}",
    "padded-upper-hex": lambda number: f"{number:032X}",
}


def fly_a_crashing_job(split, spell, out):
    """Fly one job whose task raises an exception that names both seeds, and write its line."""

    class NamesTheSeeds(ShortTask):
        def record(self, env, agent, seed, agent_seed):
            raise RuntimeError(f"episode {spell(seed)} and agent {spell(agent_seed)} failed")

    job = Job("approach-vbar", split, 0, *CRASH_SEEDS, "tiny")
    run_jobs([job], PDAgent, task=NamesTheSeeds(), out=out)


@pytest.mark.parametrize("spell", SPELLED_AS.values(), ids=list(SPELLED_AS))
def test_the_error_text_of_a_crashed_test_run_carries_no_seed(tmp_path, monkeypatch, spell):
    monkeypatch.setenv(HIDDEN_KEY_VARIABLE, OTHER_KEY)
    out = tmp_path / "crashed.jsonl"
    fly_a_crashing_job("test", spell, out)
    text = out.read_text(encoding="utf-8")
    assert_holds_no_seed(text, [CRASH_SEEDS])
    error = json.loads(text)["error"]
    assert error.startswith("RuntimeError: episode ") and error.endswith(" failed")
    assert "and agent" in error
    stored = read_results(out)
    assert len(stored) == 1 and stored[0]["failure_label"] == "error"
    assert stored[0]["error"] == error


@pytest.mark.parametrize("spell", SPELLED_AS.values(), ids=list(SPELLED_AS))
@pytest.mark.parametrize("split", ["train", "validation"])
def test_the_error_text_of_a_crashed_public_split_run_keeps_the_seeds(tmp_path, split, spell):
    out = tmp_path / "crashed.jsonl"
    fly_a_crashing_job(split, spell, out)
    seed, agent_seed = CRASH_SEEDS
    expected = f"RuntimeError: episode {spell(seed)} and agent {spell(agent_seed)} failed"
    assert json.loads(out.read_text(encoding="utf-8"))["error"] == expected
    assert read_results(out)[0]["error"] == expected


def test_resuming_a_test_split_file_flies_nothing_a_second_time(tmp_path, monkeypatch):
    flown = []

    class Counting(ShortTask):
        def record(self, env, agent, seed, agent_seed):
            flown.append(seed)
            return super().record(env, agent, seed, agent_seed)

    log = tmp_path / "private-seeds.jsonl"
    monkeypatch.setenv(LOG_VARIABLE, str(log))
    work = suite_jobs(SUITES["core-test-v1"])[:2]
    out = tmp_path / "public.jsonl"
    first = run_jobs(work, PDAgent, task=Counting(), out=out)
    assert len(flown) == 2
    again = run_jobs(work, PDAgent, task=Counting(), out=out)
    assert len(flown) == 2
    assert len(read_results(out)) == 2 and len(log_lines(log)) == 2
    assert [r["provenance"] for r in again] == [r["provenance"] for r in first]
    assert [r["seed_digest"] for r in again] == [r["seed_digest"] for r in first]


def test_a_test_split_suite_is_complete_when_its_results_hold_only_digests():
    published = stubs()
    assert all(r["seed"] is None for r in published)
    for ordered in (published, published[::-1]):
        assert TINY.missing(ordered) == [] and TINY.complete(ordered)
        TINY.require_complete(ordered)


def test_hand_built_results_with_only_a_scenario_and_a_seed_still_match_a_test_split_suite():
    by_seed = [{"scenario_id": sid, "seed": seed} for sid, seed, _ in TINY.runs()]
    assert TINY.complete(by_seed) and not TINY.complete(by_seed[:-1])


def test_results_holding_both_a_seed_and_its_digest_match_a_test_split_suite():
    with_both = stubs(raw=True)
    assert TINY.complete(with_both) and not TINY.complete(with_both[:-1])


def test_a_missing_test_split_run_is_named_without_its_seed():
    published = stubs()
    pairs = seeds("approach-vbar", "test", 3).pairs()
    assert len(TINY.missing(published[:2])) == 1
    with pytest.raises(ValueError, match="reported in full or not at all") as caught:
        TINY.require_complete(published[:2])
    message = str(caught.value)
    assert "1 of 3 runs" in message
    assert not any(form in message for pair in pairs for n in pair for form in spellings(n))
    assert "approach-vbar" in message and re.search(r"\b2\b", message)


def wrong_index_digest(results):
    from vbar.scenarios.seeds import seed_digest

    seed = TINY.entries[0].seeds().seed(0)
    return [{**results[0], "seed_digest": seed_digest("test", seed, 5)}, *results[1:]]


@pytest.mark.parametrize(
    "break_it",
    [
        pytest.param(
            lambda r: [{**r[0], "seed_digest": r[1]["seed_digest"]}, *r[1:]], id="swapped"
        ),
        pytest.param(lambda r: [{**r[0], "seed_digest": "0" * 64}, *r[1:]], id="unknown-digest"),
        pytest.param(wrong_index_digest, id="digest-of-another-index"),
        pytest.param(
            lambda r: [{**r[0], "scenario_id": "vbar:other"}, *r[1:]], id="other-scenario"
        ),
    ],
)
def test_a_result_that_matches_no_run_leaves_that_run_missing(break_it):
    broken = break_it(stubs())
    assert len(TINY.missing(broken)) == 1 and not TINY.complete(broken)
    with pytest.raises(ValueError, match="1 of 3 runs"):
        TINY.require_complete(broken)


def test_a_suite_on_a_split_without_a_key_matches_its_results_by_seed():
    from vbar.scenarios.suites import match_runs

    class Entry:
        scenario, split, count = "planar", "calibration", 2

    class Own:
        entries = (Entry(),)

        def runs(self, identifiers=None):
            return [("planar-id", 11, 12), ("planar-id", 21, 22)]

    results = [
        {"scenario_id": "planar-id", "seed": 21},
        {"scenario_id": "planar-id", "seed": None, "seed_digest": "ab" * 32},
    ]
    assert match_runs(Own(), results) == [1, None]


def test_a_result_of_no_run_of_the_suite_is_ignored():
    foreign = {"scenario_id": stubs()[0]["scenario_id"], "seed": None, "seed_digest": "0" * 64}
    assert TINY.complete([*stubs(), foreign])


def test_a_complete_test_split_suite_is_reported_from_its_published_lines(tiny_run):
    flown, stored = tiny_run
    assert_published(stored)
    assert TINY.missing(stored, identifiers(stored)) == []
    report = suite_report(TINY, stored, reps=200, task=ShortTask())
    assert report["suite"] == "tiny-hidden-v1" and report["runs"] == 3
    assert report["scenarios"]["approach-vbar"]["runs"] == 3
    in_memory = suite_report(TINY, flown, reps=200, task=ShortTask())
    assert in_memory["provenance"] == report["provenance"] and in_memory["runs"] == 3


def test_a_report_missing_a_test_split_run_is_refused_without_the_seed(tiny_run):
    _, stored = tiny_run
    assert_published(stored)
    pairs = seeds("approach-vbar", "test", 3).pairs()
    with pytest.raises(ValueError, match="reported in full or not at all") as caught:
        suite_report(TINY, stored[:2], reps=200, task=ShortTask())
    message = str(caught.value)
    assert "1 of 3 runs" in message
    assert not any(form in message for pair in pairs for n in pair for form in spellings(n))
    assert "approach-vbar" in message and re.search(r"\b2\b", message)


def test_a_report_whose_digest_was_swapped_for_another_runs_is_refused(tiny_run):
    _, stored = tiny_run
    assert_published(stored)
    swapped = [{**stored[0], "seed_digest": stored[1]["seed_digest"]}, *stored[1:]]
    assert len(TINY.missing(swapped, identifiers(stored))) == 1
    with pytest.raises(ValueError, match="reported in full or not at all"):
        suite_report(TINY, swapped, reps=200, task=ShortTask())


def test_a_report_refuses_two_test_split_results_for_one_run(tiny_run):
    _, stored = tiny_run
    assert_published(stored)
    again = [{**stored[0], "provenance": "other"}]
    with pytest.raises(ValueError, match="one result per run"):
        suite_report(TINY, stored + again, reps=200, task=ShortTask())


def test_a_report_ignores_a_result_that_belongs_to_no_run_of_the_suite(tiny_run):
    _, stored = tiny_run
    assert_published(stored)
    foreign = {**stored[0], "seed_digest": "0" * 64, "provenance": "foreign"}
    report = suite_report(TINY, [*stored, foreign], reps=200, task=ShortTask())
    assert report["runs"] == 3


def test_a_report_refuses_test_split_results_of_two_agents(tiny_run):
    _, stored = tiny_run
    assert_published(stored)
    mixed = [*stored[:2], {**stored[2], "agent": "drift"}]
    with pytest.raises(ValueError, match="one agent"):
        suite_report(TINY, mixed, reps=200, task=ShortTask())


def test_the_latest_of_two_results_of_one_test_split_run_is_kept(tiny_run):
    _, stored = tiny_run
    assert_published(stored)
    later = {**stored[0], "provenance": "later", "wall_s": 9.0}
    kept = latest_results([*stored, later])
    assert [r["provenance"] for r in kept] == [
        "later",
        stored[1]["provenance"],
        stored[2]["provenance"],
    ]
    drift = {**stored[0], "agent": "drift"}
    assert len(latest_results([*stored, drift])) == 4


# The ways a secret key shows up in text: as the environment holds it, in capitals, with some
# letters in capitals, and inside a longer text.
KEY_FORMS = {
    "as-is": lambda key: key,
    "upper": str.upper,
    "mixed-case": lambda key: key[:20].upper() + key[20:],
    "inside-text": lambda key: f"pd-{key}-v2",
}
PAIR = (SEED, AGENT_SEED)


def agent_named(form):
    """An in-process agent factory whose name is the hidden key as the environment holds it."""

    class NamedAfterTheKey(PDAgent):
        def __init__(self):
            super().__init__()
            self.name = form(os.environ[HIDDEN_KEY_VARIABLE])

    return NamedAfterTheKey


def assert_file_holds_no_secret(path, pairs):
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    assert KEY not in text.lower()
    assert_holds_no_seed(text, pairs)


def assert_message_holds_no_secret(error, pairs):
    message = str(error)
    assert KEY not in message.lower()
    assert_holds_no_seed(message, pairs)
    assert_holds_no_seed(message.lower(), pairs)


@pytest.mark.parametrize("form", KEY_FORMS.values(), ids=list(KEY_FORMS))
def test_an_agent_named_after_the_hidden_key_is_refused_and_the_file_holds_no_key(tmp_path, form):
    pairs = seeds("approach-vbar", "test", 1).pairs()
    job = Job("approach-vbar", "test", 0, *pairs[0], "tiny")
    out = tmp_path / "named.jsonl"
    with pytest.raises(ValueError) as caught:
        run_jobs([job], agent_named(form), task=ShortTask(), out=out)
    assert_message_holds_no_secret(caught.value, pairs)
    assert_file_holds_no_secret(out, pairs)


def test_a_validation_run_of_an_agent_named_after_the_hidden_key_is_written_as_it_is(tmp_path):
    seed, agent_seed = seeds("approach-vbar", "validation", 1).pairs()[0]
    job = Job("approach-vbar", "validation", 0, seed, agent_seed, "tiny")
    out = tmp_path / "named.jsonl"
    run_jobs([job], agent_named(KEY_FORMS["upper"]), task=ShortTask(), out=out)
    stored = read_results(out)
    assert stored[0]["error"] is None and stored[0]["agent"] == KEY.upper()


@pytest.mark.parametrize("form", KEY_FORMS.values(), ids=list(KEY_FORMS))
def test_the_error_text_of_a_crashed_test_run_carries_no_key(tmp_path, form):
    class NamesTheKey(ShortTask):
        def record(self, env, agent, seed, agent_seed):
            raise RuntimeError(form(os.environ[HIDDEN_KEY_VARIABLE]))

    pairs = seeds("approach-vbar", "test", 1).pairs()
    job = Job("approach-vbar", "test", 0, *pairs[0], "tiny")
    out = tmp_path / "crashed.jsonl"
    run_jobs([job], PDAgent, task=NamesTheKey(), out=out)
    text = out.read_text(encoding="utf-8")
    assert_file_holds_no_secret(out, pairs)
    error = json.loads(text)["error"]
    assert error.startswith("RuntimeError: ")
    stored = read_results(out)
    assert len(stored) == 1 and stored[0]["failure_label"] == "error"
    assert stored[0]["error"] == error


def stored_error_of_a_raising_job(message_of, out):
    """The `error` stored for a test split job whose task raises `message_of(seed, agent_seed)`."""

    class Raises(ShortTask):
        def record(self, env, agent, seed, agent_seed):
            raise RuntimeError(message_of(seed, agent_seed))

    job = Job("approach-vbar", "test", 0, *CRASH_SEEDS, "tiny")
    run_jobs([job], PDAgent, task=Raises(), out=out)
    error = json.loads(out.read_text(encoding="utf-8"))["error"]
    assert read_results(out)[0]["error"] == error
    return error


# The zeros in front of a padded seed may stay in the text; the seed itself may not.
SCRUBBED_SEEDS = {
    "decimal": (str, r"RuntimeError: episode <hidden> and agent <hidden> failed"),
    "hex": (lambda n: f"{n:x}", r"RuntimeError: episode <hidden> and agent <hidden> failed"),
    "padded-upper-hex": (
        lambda n: f"{n:032X}",
        r"RuntimeError: episode 0*<hidden> and agent 0*<hidden> failed",
    ),
}
SCRUBBED_KEYS = {
    "as-is": "<hidden>",
    "upper": "<hidden>",
    "mixed-case": "<hidden>",
    "inside-text": "pd-<hidden>-v2",
}


@pytest.mark.parametrize("name", SCRUBBED_SEEDS)
def test_a_crashed_test_run_stores_its_message_with_each_seed_replaced_by_a_marker(tmp_path, name):
    spell, expected = SCRUBBED_SEEDS[name]
    error = stored_error_of_a_raising_job(
        lambda seed, agent_seed: f"episode {spell(seed)} and agent {spell(agent_seed)} failed",
        tmp_path / "crashed.jsonl",
    )
    assert re.fullmatch(expected, error)


@pytest.mark.parametrize("name", SCRUBBED_KEYS)
def test_a_crashed_test_run_stores_its_message_with_the_key_replaced_by_a_marker(tmp_path, name):
    shown = KEY_FORMS[name](KEY)
    error = stored_error_of_a_raising_job(
        lambda seed, agent_seed: f"key {shown} given", tmp_path / "crashed.jsonl"
    )
    assert error == f"RuntimeError: key {SCRUBBED_KEYS[name]} given"


def test_a_crashed_test_run_replaces_every_occurrence_of_every_secret_in_its_message(tmp_path):
    error = stored_error_of_a_raising_job(
        lambda seed, agent_seed: (
            f"episode {seed}, again {seed}, agent {agent_seed:x}, key {KEY.upper()}"
        ),
        tmp_path / "crashed.jsonl",
    )
    assert error == "RuntimeError: episode <hidden>, again <hidden>, agent <hidden>, key <hidden>"


@pytest.mark.parametrize("form", KEY_FORMS.values(), ids=list(KEY_FORMS))
def test_encode_replaces_the_key_in_the_error_text_and_keeps_the_rest(form):
    error = f"ValueError: bad {form(KEY)} and {SEED} given"
    line = encode({**built("test"), "error": error, "failure_label": "error"})
    assert KEY not in line.lower()
    assert_holds_no_seed(line, [PAIR])
    kept = json.loads(line)["error"]
    assert kept.startswith("ValueError: bad ") and kept.endswith(" given")
    assert " and " in kept


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"metrics": {"success": True, "echo": SEED}}, id="seed-as-a-metric"),
        pytest.param(
            {"metrics": {"success": True, "echo": AGENT_SEED}}, id="agent-seed-as-a-metric"
        ),
        pytest.param({"metrics": {"success": True, "echo": f"{SEED:x}"}}, id="seed-in-hex"),
        pytest.param(
            {"metrics": {"success": True, "echo": f"{AGENT_SEED:X}"}}, id="agent-seed-in-upper-hex"
        ),
        pytest.param(
            {"metrics": {"success": True, "trace": [{"deep": f"seed={SEED:x}"}]}},
            id="nested-metric",
        ),
        pytest.param({"metrics": {f"{SEED:x}": 1.0}}, id="seed-as-a-metric-name"),
        pytest.param({"filter": KEY}, id="key-as-the-filter"),
        pytest.param({"filter": KEY.upper()}, id="key-in-capitals-as-the-filter"),
        pytest.param({"scenario_id": f"vbar:{KEY}:v1"}, id="key-in-the-scenario-id"),
        pytest.param({"scenario": KEY}, id="key-as-the-scenario"),
        pytest.param({"agent": KEY}, id="key-as-the-agent"),
        pytest.param({"suite": KEY}, id="key-as-the-suite"),
        pytest.param({"failure_label": KEY}, id="key-as-the-failure-label"),
        pytest.param({"environment": {"vbar": "0.3.1", "host": KEY}}, id="key-in-the-environment"),
        pytest.param({"provenance": KEY}, id="key-as-the-provenance"),
        pytest.param({"seed_digest": KEY}, id="key-as-a-digest"),
        pytest.param({"seed_digest": f"{SEED:064x}"}, id="seed-padded-to-a-digest"),
        pytest.param(
            {"agent_seed_digest": f"{AGENT_SEED:064x}"}, id="agent-seed-padded-to-a-digest"
        ),
    ],
)
def test_a_test_result_holding_the_key_or_a_seed_anywhere_in_its_line_is_refused(changes):
    encode(built("test"))
    with pytest.raises(ValueError) as caught:
        encode({**built("test"), **changes})
    assert_message_holds_no_secret(caught.value, [PAIR])


def test_only_seed_forms_of_eight_characters_or_more_are_searched_for():
    short = built("test", 0, 9_999_999, 12)
    line = encode({**short, "metrics": {"success": True, "steps": 9_999_999, "n": 12}})
    assert '"steps": 9999999' in line
    long = built("test", 0, 10_000_000, 12)
    with pytest.raises(ValueError):
        encode({**long, "metrics": {"success": True, "steps": 10_000_000}})


def test_the_key_is_searched_for_as_the_variable_holds_it_now(monkeypatch):
    monkeypatch.setenv(HIDDEN_KEY_VARIABLE, f"  {OTHER_KEY.upper()}\n")
    result = built("test")
    encode({**result, "filter": KEY})
    with pytest.raises(ValueError):
        encode({**result, "filter": OTHER_KEY})
    with pytest.raises(ValueError):
        encode({**result, "agent": OTHER_KEY.upper()})


def test_a_test_result_is_still_checked_for_seeds_while_no_hidden_key_is_set(monkeypatch):
    monkeypatch.delenv(HIDDEN_KEY_VARIABLE, raising=False)
    result = {**built("test"), "seed_digest": "ab" * 32, "agent_seed_digest": "cd" * 32}
    line = encode({**result, "filter": "ab" * 8})
    assert json.loads(line)["filter"] == "ab" * 8
    with pytest.raises(ValueError):
        encode({**result, "metrics": {"success": True, "echo": SEED}})


# The search for seeds in a line looks for whole numbers only, so a seed held in any other form
# would go unseen; the result is refused instead.
NOT_A_WHOLE_SEED = {
    "decimal-text": str(SEED + 7),
    "hex-text": f"{SEED + 7:x}",
    "float": 2.0**100,
    "fraction": 1.5,
    "numpy-float": np.float64(2.0**100),
    "negative-int": -1,
    "negative-large-int": -(SEED + 7),
    "true": True,
    "false": False,
}


@pytest.mark.parametrize("field", ["seed", "agent_seed"])
@pytest.mark.parametrize("bad", NOT_A_WHOLE_SEED.values(), ids=list(NOT_A_WHOLE_SEED))
def test_a_test_result_whose_seed_is_not_a_whole_number_is_refused_without_naming_it(field, bad):
    encode(built("test"))
    with pytest.raises(ValueError, match="seed") as caught:
        encode({**built("test"), field: bad})
    message = str(caught.value)
    assert str(bad) not in message and repr(bad) not in message
    assert_message_holds_no_secret(caught.value, [(SEED + 7, SEED + 7)])


@pytest.mark.parametrize("field", ["seed", "agent_seed"])
@pytest.mark.parametrize("bad", ["decimal-text", "hex-text"])
def test_a_text_seed_cannot_reach_the_line_through_the_error_text(field, bad):
    seed = NOT_A_WHOLE_SEED[bad]
    result = {
        **built("test"),
        field: seed,
        "error": f"RuntimeError: seed {seed} failed",
        "failure_label": "error",
    }
    with pytest.raises(ValueError):
        encode(result)


@pytest.mark.parametrize("logged", [False, True])
def test_a_test_result_with_a_text_seed_is_not_written_with_or_without_a_private_log(
    tmp_path, monkeypatch, logged
):
    log = tmp_path / "private-seeds.jsonl"
    if logged:
        monkeypatch.setenv(LOG_VARIABLE, str(log))
    out = tmp_path / "public.jsonl"
    result = {**built("test"), "seed": str(SEED), "error": f"RuntimeError: {SEED}"}
    with pytest.raises(ValueError):
        write_results(out, [result])
    assert_file_holds_no_secret(out, [PAIR])
    assert not log.exists() or log.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("make", [np.int64, np.uint64])
def test_numpy_integer_seeds_are_accepted_and_removed_from_the_error_text(make):
    number = 2**40 + 3
    result = {
        **built("test"),
        "seed": make(number),
        "agent_seed": make(number + 1),
        "error": f"RuntimeError: {number} and {number + 1:x} failed",
        "failure_label": "error",
    }
    raw = json.loads(encode(result))
    assert raw["seed"] is None and raw["agent_seed"] is None
    assert raw["error"] == "RuntimeError: <hidden> and <hidden> failed"


@pytest.mark.parametrize("split", ["train", "validation", "calibration"])
def test_a_public_split_line_may_hold_a_seed_or_a_key_anywhere(split):
    base = {**built("validation" if split == "calibration" else split), "split": split}
    echoed = [SEED, f"{AGENT_SEED:x}", f"{SEED:X}"]
    result = {
        **base,
        "metrics": {"success": True, "echo": echoed},
        "filter": str(SEED),
        "agent": KEY,
        "scenario_id": KEY.upper(),
        "error": f"RuntimeError: {KEY} {SEED} {AGENT_SEED:x}",
    }
    line = encode(result)
    raw = json.loads(line)
    assert raw["metrics"]["echo"] == echoed and raw["filter"] == str(SEED)
    assert raw["agent"] == KEY and raw["scenario_id"] == KEY.upper()
    assert raw["error"] == result["error"]
    assert decode(line)["agent"] == KEY
    assert reproducible(result) == reproducible(decode(line))


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(16**7, 2**128 - 1),
    agent_seed=st.integers(16**7, 2**128 - 1),
    which=st.integers(0, 1),
    spelling=st.sampled_from(sorted(SPELLED_AS)),
    field=st.sampled_from(
        ["filter", "agent", "scenario", "scenario_id", "suite", "metrics", "error"]
    ),
    pad=st.text(alphabet="ab ._-", max_size=6),
)
def test_a_seed_in_any_field_of_a_test_line_is_refused_and_in_the_error_is_removed(
    seed, agent_seed, which, spelling, field, pad
):
    # Both forms are at least 8 characters from 16**7 up, the length the search starts at.
    text = f"{pad}{SPELLED_AS[spelling]((seed, agent_seed)[which])}{pad}"
    result = built("test", 3, seed, agent_seed)
    if field == "error":
        line = encode({**result, "error": "RuntimeError: " + text, "failure_label": "error"})
        assert_holds_no_seed(line, [(seed, agent_seed)])
        assert json.loads(line)["error"].startswith("RuntimeError: ")
        return
    held = {"note": text} if field == "metrics" else text
    with pytest.raises(ValueError):
        encode({**result, field: held})


@settings(max_examples=60, deadline=None)
@given(
    capitals=st.lists(st.booleans(), min_size=64, max_size=64),
    field=st.sampled_from(["filter", "agent", "scenario_id", "metrics", "error"]),
    pad=st.text(alphabet="xyz ._-", max_size=6),
)
def test_the_key_in_any_case_in_any_field_of_a_test_line_is_refused_and_in_the_error_is_removed(
    capitals, field, pad
):
    spelled = "".join(c.upper() if up else c for c, up in zip(KEY, capitals, strict=True))
    text = f"{pad}{spelled}{pad}"
    result = built("test")
    if field == "error":
        line = encode({**result, "error": "RuntimeError: " + text, "failure_label": "error"})
        assert KEY not in line.lower()
        assert json.loads(line)["error"].startswith("RuntimeError: ")
        return
    held = {"note": text} if field == "metrics" else text
    with pytest.raises(ValueError):
        encode({**result, field: held})


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("spelling", ["as-given", "log-dot-dot", "out-dot-dot", "relative"])
def test_a_private_log_that_is_the_results_file_is_refused_before_anything_is_written(
    tmp_path, monkeypatch, spelling, append, existing
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "other").mkdir()
    out = tmp_path / "public.jsonl"
    same_file = {
        "as-given": (out, str(out)),
        "log-dot-dot": (out, str(tmp_path / "other" / ".." / "public.jsonl")),
        "out-dot-dot": (tmp_path / "other" / ".." / "public.jsonl", str(out)),
        "relative": (Path("other") / ".." / "public.jsonl", "public.jsonl"),
    }
    given_out, log = same_file[spelling]
    if existing:
        write_results(out, [built("validation")])
    before = out.read_bytes() if existing else None
    monkeypatch.setenv(LOG_VARIABLE, log)
    with pytest.raises(ValueError) as caught:
        write_results(given_out, [built("test", 1, SEED + 1, AGENT_SEED + 1)], append=append)
    assert_message_holds_no_secret(caught.value, [(SEED + 1, AGENT_SEED + 1)])
    assert out.exists() == existing
    assert not existing or out.read_bytes() == before


@pytest.mark.skipif(
    os.name != "nt", reason="file names differ in case only on case-insensitive disks"
)
def test_a_private_log_spelled_in_another_case_is_the_results_file_on_windows(
    tmp_path, monkeypatch
):
    out = tmp_path / "public.jsonl"
    monkeypatch.setenv(LOG_VARIABLE, str(tmp_path / "PUBLIC.JSONL"))
    with pytest.raises(ValueError):
        write_results(out, [built("test")])
    assert not out.exists()


def test_a_run_whose_private_log_is_its_results_file_leaves_no_seed_in_the_file(
    tmp_path, monkeypatch
):
    out = tmp_path / "results.jsonl"
    monkeypatch.setenv(LOG_VARIABLE, str(out))
    work = suite_jobs(SUITES["core-test-v1"])[:1]
    with pytest.raises(ValueError):
        run_jobs(work, PDAgent, task=ShortTask(), out=out)
    assert_file_holds_no_secret(out, [(work[0].seed, work[0].agent_seed)])


def test_a_private_log_in_another_file_is_accepted_whatever_its_name(tmp_path, monkeypatch):
    out = tmp_path / "public.jsonl"
    (tmp_path / "private").mkdir()
    for k, log in enumerate(
        [tmp_path / "public.jsonl.seeds", tmp_path / "private" / "public.jsonl"], start=1
    ):
        monkeypatch.setenv(LOG_VARIABLE, str(log))
        write_results(out, [built("test", k, SEED + k, AGENT_SEED + k)])
        assert [entry["index"] for entry in log_lines(log)] == [k]
    assert len(read_results(out)) == 2
    assert_holds_no_seed(out.read_text(encoding="utf-8"), [PAIR, (SEED + 1, AGENT_SEED + 1)])


def hard_link_or_skip(source, link):
    try:
        os.link(source, link)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"no hard links here: {error}")


@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("named_by", ["the-log", "the-results-path"])
def test_a_private_log_that_is_a_hard_link_to_the_results_file_is_refused_before_a_seed_is_written(
    tmp_path, monkeypatch, named_by, append
):
    out = tmp_path / "public.jsonl"
    alias = tmp_path / "alias.jsonl"
    # A published line holds no seed, so a seed in the file afterwards was written by the call.
    write_results(out, [built("test", 0)])
    before = out.read_bytes()
    hard_link_or_skip(out, alias)
    given_out, log = (out, alias) if named_by == "the-log" else (alias, out)
    pair = (SEED + 1, AGENT_SEED + 1)
    monkeypatch.setenv(LOG_VARIABLE, str(log))
    with pytest.raises(ValueError) as caught:
        write_results(given_out, [built("test", 1, *pair)], append=append)
    assert_message_holds_no_secret(caught.value, [pair])
    assert_file_holds_no_secret(out, [pair])
    assert not append or out.read_bytes() == before


@pytest.mark.skipif(
    sys.platform != "win32", reason="only Windows drops a trailing space or dot from a file name"
)
@pytest.mark.parametrize("trailer", [" ", ".", ". "])
@pytest.mark.parametrize("named_by", ["the-log", "the-results-path"])
def test_a_private_log_spelled_with_a_trailing_space_or_dot_is_the_results_file_on_windows(
    tmp_path, monkeypatch, named_by, trailer
):
    out = tmp_path / "public.jsonl"
    spelled = str(out) + trailer
    given_out, log = (out, spelled) if named_by == "the-log" else (spelled, str(out))
    pair = (SEED + 1, AGENT_SEED + 1)
    monkeypatch.setenv(LOG_VARIABLE, log)
    with pytest.raises(ValueError) as caught:
        write_results(given_out, [built("test", 1, *pair)])
    assert_message_holds_no_secret(caught.value, [pair])
    assert_file_holds_no_secret(out, [pair])


@pytest.mark.parametrize("alias", ["hard-link", "trailing-space"])
def test_a_run_whose_private_log_is_an_alias_of_its_results_file_leaves_no_seed_in_the_file(
    tmp_path, monkeypatch, alias
):
    out = tmp_path / "results.jsonl"
    work = suite_jobs(SUITES["core-test-v1"])[:1]
    if alias == "hard-link":
        write_results(out, [built("test", 5)])
        log = tmp_path / "alias.jsonl"
        hard_link_or_skip(out, log)
    elif sys.platform == "win32":
        log = str(out) + " "
    else:
        pytest.skip("only Windows drops a trailing space from a file name")
    monkeypatch.setenv(LOG_VARIABLE, str(log))
    with pytest.raises(ValueError):
        run_jobs(work, PDAgent, task=ShortTask(), out=out)
    assert_file_holds_no_secret(out, [(work[0].seed, work[0].agent_seed)])


BAD_DIGESTS = {
    "the-seed-in-decimal": str(SEED),
    "the-seed-in-hex": f"{SEED:x}",
    "empty": "",
    "63-digits": "ab" * 31 + "a",
    "65-digits": "ab" * 32 + "a",
    "capitals": "AB" * 32,
    "not-hex": "ab" * 31 + "gg",
    "prefix": "0x" + "ab" * 31,
    "trailing-newline": "ab" * 32 + "\n",
    "leading-space": " " + "ab" * 31 + "a",
    "arabic-indic-digits": "\u0660" * 64,
    "a-number": 12345,
}


@pytest.mark.parametrize("field", ["seed_digest", "agent_seed_digest"])
@pytest.mark.parametrize("bad", BAD_DIGESTS.values(), ids=list(BAD_DIGESTS))
def test_a_test_result_whose_digest_is_not_64_lowercase_hex_digits_is_refused(field, bad):
    with pytest.raises(ValueError, match="digest") as caught:
        encode({**built("test"), field: bad})
    assert_holds_no_seed(str(caught.value), [PAIR])
    encode({**built("test"), field: OTHER_KEY})
    encode({**built("test"), field: "f" * 64})


@pytest.mark.parametrize("field", ["seed_digest", "agent_seed_digest"])
@pytest.mark.parametrize("bad", [None, *BAD_DIGESTS.values()], ids=["none", *BAD_DIGESTS])
def test_a_test_split_line_with_a_malformed_digest_is_not_read(field, bad):
    decode(line_of("test"))
    with pytest.raises(ValueError, match="digest"):
        decode(line_of("test", **{field: bad}))


@pytest.mark.parametrize("split", ["train", "validation"])
def test_a_public_split_result_without_digests_is_still_written_and_read(split):
    result = {**built(split), "seed_digest": None, "agent_seed_digest": None}
    back = decode(encode(result))
    assert (back["seed"], back["agent_seed"]) == (SEED, AGENT_SEED)
    assert back["seed_digest"] is None and back["agent_seed_digest"] is None


def test_reproducible_text_of_a_flown_test_result_holds_no_seed_and_matches_its_published_line(
    tiny_run,
):
    flown, stored = tiny_run
    pairs = seeds("approach-vbar", "test", 3).pairs()
    for fresh, published in zip(flown, stored, strict=True):
        text = reproducible(fresh)
        assert_holds_no_seed(text, pairs)
        assert text == reproducible(published)
        # The seeds stay in memory: the private log is written from them.
        assert fresh["seed"] is not None and fresh["agent_seed"] is not None


def test_reproducible_text_of_a_crashed_test_result_holds_no_seed_and_matches_its_line():
    seed, agent_seed = CRASH_SEEDS
    error = f"RuntimeError: episode {seed} and agent {agent_seed:x} failed"
    fresh = {**built("test", 0, seed, agent_seed), "error": error, "failure_label": "error"}
    text = reproducible(fresh)
    assert_holds_no_seed(text, [CRASH_SEEDS])
    assert text == reproducible(decode(encode(fresh)))
    assert json.loads(text)["error"].startswith("RuntimeError: episode ")


def test_reproducible_text_of_a_public_split_result_keeps_its_seed():
    for split in ("train", "validation"):
        result = built(split)
        text = reproducible(result)
        assert str(SEED) in text and str(AGENT_SEED) in text
        assert text == reproducible(decode(encode(result)))


def test_results_read_back_from_a_published_test_file_add_nothing_to_the_private_log(
    tmp_path, monkeypatch
):
    published = tmp_path / "published.jsonl"
    write_results(published, [built("test", k, SEED + k, AGENT_SEED + k) for k in range(2)])
    stored = read_results(published)
    assert all(r["seed"] is None and r["agent_seed"] is None for r in stored)
    log = tmp_path / "private-seeds.jsonl"
    monkeypatch.setenv(LOG_VARIABLE, str(log))
    copy = tmp_path / "copy.jsonl"
    write_results(copy, stored)
    assert not log.exists() or log.read_text(encoding="utf-8") == ""
    assert read_results(copy) == stored
