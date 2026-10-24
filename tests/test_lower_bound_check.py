"""No baseline run beats a lower bound certified for the rung it flew.

A `DualBank` built on the CW view gives, for any start, a lower bound on the delta-v of every
successful run of that view on the CW rung whose delivered acceleration stays within the view's
limit (Boyd and Vandenberghe, Convex Optimization, 2004, section 5.1.3). A successful baseline run
below it would be a simulator or metric bug, which `optimality_gap` reports as E3f. The bound is a
model bound for a run on another rung: E3c is False there, E3f is never set and the ratio is E3m.

Episodes of approach-rbar are shortened from 360 steps to 120 steps of 10 s, 210 for LQR, and
check the soft speed constraints at 2 samples per step instead of 20; the goal, the 600 s dwell,
the step length and the CW dynamics are unchanged, and each run's bank is built on the same
shortened view, so its bound holds for these runs exactly as at full length. The two-body
control flies 20 steps of 60 s, since two-body steps cost about ten times a CW step. The full
check over every baseline, scenario and 20 validation seeds is `benchmarks/lower_bound_check.py`.
"""

import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vbar.agents import LQRAgent, PDAgent, TwoImpulseAgent
from vbar.metrics import optimality_gap, record_episode
from vbar.metrics.efficiency import delta_v_axes
from vbar.metrics.mission import success
from vbar.oracles import DualBank, FuelOracle, OracleAgent, OracleConfig, bounds_for
from vbar.scenarios import make
from vbar.scenarios.seeds import SeedSplit

SID = "vbar:approach-rbar:cw:box:dv:v1"
TWO_BODY = "vbar:approach-rbar:twobody:box:dv:v1"
SEEDS = (0, 1)
# 1200 s: every agent below but LQR completes the 600 s dwell from validation seeds 0 and 1 (PD
# by step 98, two-impulse by 107, the oracle at 120).
SHORT = {"max_steps": 120, "violation_samples": 2}
# 2100 s: LQR at its default weights completes the dwell by step 198 (seed 0) and 190 (seed 1).
LQR_SHORT = {"max_steps": 210, "violation_samples": 2}
# 20 steps of 60 s: the oracle agent's open-loop plan completes the dwell on the two-body rung.
COARSE = {"dt": 60.0, "max_steps": 20, "violation_samples": 2}
# Fewer arrival candidates and LPs keep each solve short; any choice gives valid bounds.
FAST = OracleConfig(arrival_candidates=4, bank_stride=20)
BOUND_TOL = 1e-9
# Observed oracle-agent ratios over the bank bound: 1.163 (seed 0) and 1.261 (seed 1). A bank
# that lost its entries would give 0 and pass every soundness check; this keeps it tight.
ORACLE_GAP_MAX = 1.35
SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "lower_bound_check.py"


def make_agent(name):
    """The baselines under check, set to finish the dwell within the shortened episode."""
    if name == "pd":
        # The default 0.01 rad/s completes the dwell at step 121, past the 120 kept here.
        return PDAgent(wn=0.02)
    if name == "lqr":
        return LQRAgent()
    if name == "two-impulse":
        # A 0.5 rad transfer, about 450 s at 500 km, instead of the default 1 rad (about 900 s).
        return TwoImpulseAgent(n_tf=0.5, mode="retarget")
    return OracleAgent(config=FAST)


AGENTS = ("pd", "lqr", "two-impulse", "oracle")


def episode_options(name):
    """The shortened episode the agent flies."""
    return LQR_SHORT if name == "lqr" else SHORT


@pytest.fixture(scope="module")
def banks():
    """The bank of each agent's shortened view, built once on first use from the true starts
    of both validation seeds: every agent flies those starts, and a bank bound holds for any
    start on its view, so one build per episode length serves every run and the controls."""
    built = {}

    def bank(name):
        options = episode_options(name)
        key = options["max_steps"]
        if key not in built:
            env = make(SID, **options)
            split = SeedSplit(SID, "validation", len(SEEDS))
            starts = [env.reset(seed=split.seed(i))[1]["true_state"] for i in SEEDS]
            built[key] = DualBank.build(env.agent_view(), np.array(starts), config=FAST)
        return built[key]

    return bank


@pytest.fixture(scope="module")
def flights():
    """Recorded runs keyed by (agent, seed index), each flown once on first use; several tests
    read the same runs and a run costs one to two seconds."""
    split = SeedSplit(SID, "validation", len(SEEDS))
    runs = {}

    def flight(name, index):
        if (name, index) not in runs:
            runs[name, index] = record_episode(
                make(SID, **episode_options(name)),
                make_agent(name),
                seed=split.seed(index),
                agent_seed=split.agent_seed(index),
            )
        return runs[name, index]

    return flight


def halved(ep):
    """The episode with every executed acceleration and delta-v halved and the truth kept."""
    records = []
    for rec in ep.records:
        act = rec.actuation
        segments = tuple(
            (duration, 0.5 * np.asarray(a, dtype=float)) for duration, a in act.segments
        )
        actuation = replace(
            act,
            segments=segments,
            dv_l1=0.5 * act.dv_l1,
            dv_l2=0.5 * act.dv_l2,
            propellant_kg=0.5 * act.propellant_kg,
        )
        records.append(replace(rec, actuation=actuation))
    return replace(ep, records=tuple(records))


@pytest.mark.parametrize("index", SEEDS, ids=["seed0", "seed1"])
@pytest.mark.parametrize("name", AGENTS)
def test_successful_baseline_runs_stay_above_the_certified_lower_bound(banks, flights, name, index):
    """The lqr cases build a bank of their own, about 2 s of the lqr-seed0 case: LQR needs 210
    steps to complete the dwell, and a bank's bound holds only on the view it was built on."""
    ep = flights(name, index)
    if not success(ep):
        pytest.skip(f"{name} did not complete the dwell from validation seed {index}")
    bounds = banks(name).bounds(ep.start)
    gap = optimality_gap(ep, bounds)
    assert ep.dynamics == "cw"
    assert bounds.rung == "cw" and bounds.accel_limit == ep.view.max_accel
    assert bounds.lower > 0.0
    # Certified, so E3f False below is the theorem's claim and not the uncertified default.
    assert gap["bound_certified"] is True
    assert gap["below_lower_bound"] is False
    assert delta_v_axes(ep) >= bounds.lower * (1.0 - BOUND_TOL)
    assert gap["optimality_gap"] >= 1.0 - BOUND_TOL


@pytest.mark.parametrize("name", AGENTS)
def test_every_listed_baseline_completes_the_dwell_at_least_once(flights, name):
    assert any(success(flights(name, index)) for index in SEEDS)


@pytest.mark.parametrize("index", SEEDS, ids=["seed0", "seed1"])
def test_oracle_agent_run_is_within_its_stated_ratio_of_the_bank_bound(banks, flights, index):
    ep = flights("oracle", index)
    assert success(ep)
    gap = optimality_gap(ep, banks("oracle").bounds(ep.start))
    assert 1.0 <= gap["optimality_gap"] <= ORACLE_GAP_MAX


def test_halved_actuation_is_flagged_below_the_certified_bound(banks, flights):
    ep = flights("oracle", 0)
    doctored = halved(ep)
    bounds = banks("oracle").bounds(ep.start)
    # The truth is untouched, so the doctored run still succeeds; only its delta-v is impossible.
    assert success(doctored)
    assert delta_v_axes(doctored) == pytest.approx(0.5 * delta_v_axes(ep), rel=1e-12)
    assert delta_v_axes(doctored) < bounds.lower
    gap = optimality_gap(doctored, bounds)
    assert gap["bound_certified"] is True
    assert gap["below_lower_bound"] is True
    assert gap["optimality_gap"] < 1.0


def test_two_body_run_gets_a_model_ratio_and_no_flag():
    env = make(TWO_BODY, **COARSE)
    split = SeedSplit(SID, "validation", len(SEEDS))
    ep = record_episode(
        env, OracleAgent(config=FAST), seed=split.seed(0), agent_seed=split.agent_seed(0)
    )
    assert ep.dynamics == "twobody"
    assert success(ep)
    bounds = bounds_for(ep, oracle=FuelOracle(config=FAST))
    assert bounds.rung == "cw" and bounds.lower > 0.0
    gap = optimality_gap(ep, bounds)
    assert gap["bound_certified"] is False
    assert gap["bound_rung"] == "cw"
    assert gap["below_lower_bound"] is False
    assert math.isfinite(gap["optimality_gap_model"])
    assert gap["optimality_gap_model"] == pytest.approx(delta_v_axes(ep) / bounds.lower, rel=1e-12)
    assert math.isnan(gap["optimality_gap"])


def load_script(monkeypatch):
    """The script as a module, run in this process so the test skips the import of the package;
    no bytecode is written next to it."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("lower_bound_check_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lower_bound_script_exists():
    assert SCRIPT.is_file()


def exit_status(script, argv):
    """The status `main` returns or exits with."""
    try:
        return script.main(argv)
    except SystemExit as exit:
        return exit.code


def test_lower_bound_script_refuses_the_test_split(capsys, monkeypatch):
    script = load_script(monkeypatch)
    assert exit_status(script, ["--split", "test"]) == 2
    captured = capsys.readouterr()
    assert "test" in captured.err
    assert captured.out.strip() == ""


def test_lower_bound_script_rejects_an_unknown_agent_as_a_usage_error_before_flying(
    capsys, monkeypatch
):
    import vbar.metrics
    from vbar.oracles import DualBank

    script = load_script(monkeypatch)
    flown = []
    monkeypatch.setattr(vbar.metrics, "record_episode", lambda *args, **seeds: flown.append(args))
    monkeypatch.setattr(
        DualBank, "build", staticmethod(lambda view, starts, **options: flown.append(view))
    )
    status = exit_status(script, ["--scenario", SID, "--agent", "no-such-agent", "--count", "1"])
    captured = capsys.readouterr()
    assert status == 2
    assert flown == []
    assert "no-such-agent" in captured.err
    assert captured.out.strip() == ""


def test_lower_bound_script_prints_counts_for_one_scenario_seed_and_agent(capsys, monkeypatch):
    script = load_script(monkeypatch)
    # 70 steps leave too little time for the dwell, so the row counts a failed run; the table
    # and its counts are what is checked here, the bound itself above.
    status = script.main(
        [
            "--scenario",
            SID,
            "--agent",
            "pd",
            "--count",
            "1",
            "--max-steps",
            "70",
            "--violation-samples",
            "2",
        ]
    )
    out = capsys.readouterr().out
    lines = [line.split() for line in out.splitlines() if line.strip()]
    header = next(tokens for tokens in lines if "agent" in tokens)
    assert header[:6] == ["scenario", "agent", "runs", "successes", "certified", "e3f"]
    rows = [tokens for tokens in lines if tokens[:2] == [SID, "pd"]]
    assert len(rows) == 1
    runs, successes, certified, flagged = (int(v) for v in rows[0][2:6])
    assert runs == 1
    assert successes == 0
    # Only successful runs count as certified, so this sweep certified nothing.
    assert (certified, flagged) == (0, 0)
    assert status == 3


class PlanlessAgent:
    """Coasts one step and then finds no plan, as the oracle agent can mid-episode."""

    name = "planless"
    reason = "planless agent found no plan"

    def reset(self, view, rng=None):
        self.view = view
        self.steps = 0

    def act(self, obs, info):
        self.steps += 1
        if self.steps > 1:
            raise RuntimeError(self.reason)
        return np.zeros(3)


def test_lower_bound_script_counts_an_agent_that_raises_as_a_failed_run(
    capsys, caplog, monkeypatch
):
    from vbar.agents import DriftAgent

    script = load_script(monkeypatch)
    monkeypatch.setattr(
        script, "baselines", lambda: {"planless": PlanlessAgent, "drift": DriftAgent}
    )
    status = script.main(
        [
            "--scenario",
            SID,
            "--agent",
            "planless",
            "--agent",
            "drift",
            "--count",
            "2",
            # Three steps keep the test fast; the bank then fails, so no run is certified here.
            "--max-steps",
            "3",
            "--violation-samples",
            "2",
        ]
    )
    captured = capsys.readouterr()
    # A raised run is a failure, not a flag; with no bank, no run is certified either.
    assert status == 3
    lines = [line.split() for line in captured.out.splitlines() if line.strip()]
    header = next(tokens for tokens in lines if "agent" in tokens)
    assert header[:6] == ["scenario", "agent", "runs", "successes", "certified", "e3f"]
    rows = {tokens[1]: tokens for tokens in lines if tokens[:1] == [SID]}
    assert [int(v) for v in rows["planless"][2:6]] == [2, 0, 0, 0]
    # The sweep goes on to the next agent after the one that raised.
    assert int(rows["drift"][2]) == 2
    # Logged or printed, the reason reaches standard error once for the scenario and agent.
    reported = (captured.err.count(PlanlessAgent.reason), caplog.text.count(PlanlessAgent.reason))
    assert max(reported) == 1


SECOND = "vbar:approach-vbar:cw:box:dv:v1"


def failing_build(error, scenario_id, reason):
    """A `DualBank.build` that raises `error` on `scenario_id` and ValueError elsewhere."""

    def build(view, starts, **options):
        # The view's identifier carries the episode overrides after the scenario's.
        if view.scenario_id.startswith(scenario_id):
            raise error(reason)
        raise ValueError("no bank in this test")

    return staticmethod(build)


def run_drift(script, scenarios):
    """The script's status for one three-step drift run per scenario."""
    argv = ["--agent", "drift", "--count", "1", "--max-steps", "3", "--violation-samples", "2"]
    for scenario_id in scenarios:
        argv += ["--scenario", scenario_id]
    return script.main(argv)


def table_rows(out):
    """The table's rows keyed by (scenario, agent), as counts in column order."""
    lines = [line.split() for line in out.splitlines() if line.strip()]
    return {
        (tokens[0], tokens[1]): [int(v) for v in tokens[2:6]]
        for tokens in lines
        if tokens[0] != "scenario"
    }


def test_lower_bound_script_skips_a_scenario_whose_bank_solver_fails_and_scores_the_next(
    capsys, caplog, monkeypatch
):
    from vbar.agents import DriftAgent
    from vbar.oracles import DualBank

    script = load_script(monkeypatch)
    monkeypatch.setattr(script, "baselines", lambda: {"drift": DriftAgent})
    reason = "dual LP solver failed"
    monkeypatch.setattr(DualBank, "build", failing_build(RuntimeError, SID, reason))
    status = run_drift(script, [SID, SECOND])
    captured = capsys.readouterr()
    # SECOND has no bank in this test, so neither scenario certified a run.
    assert status == 3
    rows = table_rows(captured.out)
    assert (SID, "drift") not in rows
    assert rows[SECOND, "drift"] == [1, 0, 0, 0]
    message = f"not scored {SID}: {reason}"
    assert message in captured.err or message in caplog.text


def test_lower_bound_script_flies_a_scenario_without_a_bank_uncertified(
    capsys, caplog, monkeypatch
):
    from vbar.agents import DriftAgent
    from vbar.oracles import DualBank

    script = load_script(monkeypatch)
    monkeypatch.setattr(script, "baselines", lambda: {"drift": DriftAgent})
    reason = "the oracle cannot plan this goal"
    monkeypatch.setattr(DualBank, "build", failing_build(ValueError, SID, reason))
    status = run_drift(script, [SID])
    captured = capsys.readouterr()
    assert status == 3
    runs, _, certified, flagged = table_rows(captured.out)[SID, "drift"]
    assert (runs, certified, flagged) == (1, 0, 0)
    assert reason in captured.err or reason in caplog.text


def test_lower_bound_script_reports_no_certified_run_when_every_bank_solver_fails(
    capsys, caplog, monkeypatch
):
    from vbar.agents import DriftAgent
    from vbar.oracles import DualBank

    script = load_script(monkeypatch)
    monkeypatch.setattr(script, "baselines", lambda: {"drift": DriftAgent})
    monkeypatch.setattr(DualBank, "build", failing_build(RuntimeError, "vbar:", "solver failed"))
    status = run_drift(script, [SID, SECOND])
    captured = capsys.readouterr()
    assert table_rows(captured.out) == {}
    # An empty sweep proves nothing about the bound, so it must not read as a pass.
    assert status == 3


def replay_script(script, monkeypatch, ep, bank):
    """The script's status for one run on SID of the episode's length that replays `ep` and is
    scored against `bank`, whatever view the script builds its bank on."""
    import vbar.metrics
    from vbar.agents import DriftAgent
    from vbar.oracles import DualBank

    monkeypatch.setattr(script, "baselines", lambda: {"drift": DriftAgent})
    monkeypatch.setattr(DualBank, "build", staticmethod(lambda view, starts, **options: bank))
    monkeypatch.setattr(vbar.metrics, "record_episode", lambda env, agent, **seeds: ep)
    argv = ["--scenario", SID, "--agent", "drift", "--count", "1"]
    argv += ["--max-steps", str(ep.view.max_steps), "--violation-samples", "2"]
    return script.main(argv)


def test_lower_bound_script_passes_a_certified_run_above_the_bound(
    capsys, banks, flights, monkeypatch
):
    script = load_script(monkeypatch)
    ep = flights("oracle", 0)
    bank = banks("oracle")
    assert success(ep)
    assert bank.view == ep.view
    status = replay_script(script, monkeypatch, ep, bank)
    assert table_rows(capsys.readouterr().out)[SID, "drift"] == [1, 1, 1, 0]
    assert status == 0


def test_lower_bound_script_fails_a_run_below_the_certified_bound(
    capsys, banks, flights, monkeypatch
):
    script = load_script(monkeypatch)
    bank = banks("oracle")
    assert bank.view == flights("oracle", 0).view
    status = replay_script(script, monkeypatch, halved(flights("oracle", 0)), bank)
    assert table_rows(capsys.readouterr().out)[SID, "drift"] == [1, 1, 1, 1]
    assert status == 1


def test_lower_bound_script_refuses_a_run_scored_against_a_bank_built_on_another_view(
    flights, monkeypatch
):
    script = load_script(monkeypatch)
    ep = flights("oracle", 0)
    # An empty bank bounds every start by 0, so only the view check can stop the run passing.
    bank = DualBank(make(SID, **LQR_SHORT).agent_view(), config=FAST)
    assert ep.view == make(SID, **SHORT).agent_view()
    assert bank.view != ep.view
    with pytest.raises(RuntimeError, match="view"):
        replay_script(script, monkeypatch, ep, bank)


class OffRungBank:
    """The bank's bounds stated for the two-body rung, so none is certified for a CW run."""

    def __init__(self, bank):
        self.view = bank.view
        self.bank = bank

    def bounds(self, start):
        return replace(self.bank.bounds(start), rung="twobody")


def test_lower_bound_script_neither_certifies_nor_flags_a_run_below_an_uncertified_bound(
    capsys, banks, flights, monkeypatch
):
    script = load_script(monkeypatch)
    doctored = halved(flights("oracle", 0))
    bank = OffRungBank(banks("oracle"))
    # Below the bank's lower bound and successful, so only E3c keeps it from being flagged.
    assert success(doctored)
    assert delta_v_axes(doctored) < bank.bank.bounds(doctored.start).lower
    status = replay_script(script, monkeypatch, doctored, bank)
    assert table_rows(capsys.readouterr().out)[SID, "drift"] == [1, 1, 0, 0]
    assert status == 3
