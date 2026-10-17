import dataclasses
import json
import math
import re

import numpy as np
import pytest
from metric_cases import episode, pd_episode

from vbar.agents import LLMAgent
from vbar.metrics import (
    AgentReport,
    agent_metrics,
    belief_consistency,
    faithfulness,
    parse_belief,
    solver_metrics,
)
from vbar.spacecraft import GaussianNav

ACTION = {"ax": 0.0, "ay": 0.1, "az": 0.0}


def numbers(prompt, label):
    found = re.search(label + r" = \[([^\]]*)\]", prompt).group(1)
    return np.array([float(v) for v in found.split(",")])


def believer(lie=False):
    """A model that states its range and whether it is closing, from the prompt it was given."""

    def complete(prompt):
        r, v = numbers(prompt, "position"), numbers(prompt, "velocity")
        belief = {
            "range_m": float(np.linalg.norm(r)) * (2.0 if lie else 1.0),
            "closing": bool(r @ v < 0),
        }
        return json.dumps(ACTION) + " " + json.dumps(belief)

    return complete


def test_latency_median_p95_and_total_from_the_records():
    ep = pd_episode(max_steps=5)
    latencies = [0.1, 0.2, 0.3, 0.4, 2.0]
    ep = dataclasses.replace(
        ep,
        records=tuple(
            dataclasses.replace(r, agent_latency_s=s)
            for r, s in zip(ep.records, latencies, strict=True)
        ),
    )
    m = agent_metrics(ep)
    assert m["latency_median"] == pytest.approx(0.3)
    assert m["latency_p95"] == pytest.approx(np.percentile(latencies, 95))
    assert m["compute"] == pytest.approx(3.0)


def test_parse_failures_are_the_share_of_unreadable_replies():
    replies = iter(["no idea", json.dumps(ACTION)] * 5)
    ep = episode(
        LLMAgent(lambda prompt: next(replies)), start_state=(0, -300, 0, 0, 0, 0), max_steps=10
    )
    assert agent_metrics(ep)["parse_failure_rate"] == 0.5


def test_tokens_and_cost_from_the_log_and_a_declared_price_table():
    ep = pd_episode(max_steps=4)
    log = tuple({"input_tokens": 1000, "output_tokens": 50} for _ in ep.records)
    ep = dataclasses.replace(ep, agent_log=log)
    m = agent_metrics(ep, AgentReport(token_prices=(3.0, 15.0)))
    assert m["input_tokens"] == 4000 and m["output_tokens"] == 200
    assert m["token_cost"] == pytest.approx((4000 * 3.0 + 200 * 15.0) / 1e6)
    assert math.isnan(agent_metrics(ep)["token_cost"])


def test_a_belief_that_matches_the_observation_is_faithful_and_a_wrong_one_is_not():
    start = (0.0, -300.0, 10.0, 0.0, 0.2, 0.0)
    honest = episode(LLMAgent(believer()), start_state=start, max_steps=6)
    liar = episode(LLMAgent(believer(lie=True)), start_state=start, max_steps=6)
    assert faithfulness(honest) == 1.0
    assert faithfulness(liar) == 0.0


def test_faithfulness_is_judged_against_the_observation_not_the_truth():
    # With 50 m of navigation noise an honest agent restates what it saw, which is off the truth.
    start = (0.0, -300.0, 10.0, 0.0, 0.2, 0.0)
    ep = episode(
        LLMAgent(believer()), start_state=start, max_steps=6, sensor=GaussianNav(50.0, 0.01)
    )
    assert faithfulness(ep) == 1.0
    truth = [np.linalg.norm(s[:3]) for s in ep.states[:-1]]
    seen = [np.linalg.norm(o[:3]) for o in (ep.first_obs, *(r.obs for r in ep.records[:-1]))]
    assert max(abs(a - b) for a, b in zip(truth, seen, strict=True)) > 1.0


def stated_ranges(ep, offsets):
    """The episode with a log whose beliefs state each observed range plus an offset (m)."""
    observed = [ep.first_obs, *(r.obs for r in ep.records[:-1])]
    log = tuple(
        {"reply": json.dumps({"range_m": float(np.linalg.norm(o[:3])) + d, "closing": True})}
        for o, d in zip(observed, offsets, strict=True)
    )
    return dataclasses.replace(ep, agent_log=log)


def latest_fix(observations):
    """A posterior that trusts the latest fix with a 2 m standard deviation per axis."""
    return observations[-1][:3], 4.0 * np.eye(3)


def test_belief_consistency_is_the_rms_error_in_posterior_sigmas():
    ep = stated_ranges(pd_episode(max_steps=4), [2.0, -2.0, 6.0, -6.0])
    assert belief_consistency(ep, latest_fix) == pytest.approx(np.sqrt(5.0), abs=1e-9)
    assert math.isnan(belief_consistency(ep, None))
    assert math.isnan(agent_metrics(ep)["belief_consistency"])


def test_restating_noisy_fixes_is_faithful_but_not_consistent_with_the_posterior():
    # At rest on the V-bar the truth stays put, so with 50 m fixes the posterior is their mean
    # with covariance 50^2 / k per axis.
    coast = json.dumps({"ax": 0.0, "ay": 0.0, "az": 0.0})
    fixes = []

    def parrot(prompt):
        r, v = numbers(prompt, "position"), numbers(prompt, "velocity")
        belief = {"range_m": float(np.linalg.norm(r)), "closing": bool(r @ v < 0)}
        return coast + " " + json.dumps(belief)

    def averager(prompt):
        fixes.append(numbers(prompt, "position"))
        mean = np.mean(fixes, axis=0)
        return coast + " " + json.dumps({"range_m": float(np.linalg.norm(mean)), "closing": False})

    def mean_of_fixes(observations):
        return observations[:, :3].mean(axis=0), 50.0**2 / len(observations) * np.eye(3)

    start, nav = (0.0, -300.0, 0.0, 0.0, 0.0, 0.0), GaussianNav(50.0, 0.01)
    restated = episode(LLMAgent(parrot), start_state=start, max_steps=8, sensor=nav)
    averaged = episode(LLMAgent(averager), start_state=start, max_steps=8, sensor=nav)
    assert faithfulness(restated) == 1.0
    assert belief_consistency(restated, mean_of_fixes) > 1.0
    # The prompt prints the fixes rounded, so the averaged belief is off by far less than a sigma.
    assert belief_consistency(averaged, mean_of_fixes) < 1e-3


def test_solver_fields_and_the_latency_split_come_from_the_entries_that_report_them():
    ep = pd_episode(max_steps=4)
    log = (
        {
            "solver": "osqp",
            "tolerance": 1e-6,
            "status": "solved",
            "iterations": 12,
            "warm_start": False,
            "fallback": False,
            "prep_s": 0.02,
            "feedback_s": 0.001,
        },
        {
            "solver": "osqp",
            "tolerance": 1e-6,
            "status": "solved",
            "iterations": 5,
            "warm_start": True,
            "fallback": False,
            "prep_s": 0.03,
            "feedback_s": 0.002,
        },
        {
            "solver": "osqp",
            "tolerance": 1e-6,
            "status": "max_iter",
            "iterations": 40,
            "warm_start": True,
            "fallback": True,
            "prep_s": 0.01,
            "feedback_s": 0.004,
        },
        {
            "solver": "ipopt",
            "tolerance": 1e-8,
            "status": "solved",
            "iterations": 8,
            "warm_start": True,
            "fallback": False,
        },
    )
    m = solver_metrics(dataclasses.replace(ep, agent_log=log))
    assert m["preparation_median"] == pytest.approx(0.02)
    assert m["feedback_median"] == pytest.approx(0.002)
    assert m["feedback_p95"] == pytest.approx(np.percentile([0.001, 0.002, 0.004], 95))
    assert m["solver_metadata"] == {
        "solver": ("ipopt", "osqp"),
        "tolerance": (1e-8, 1e-6),
        "status": {"max_iter": 1, "solved": 3},
    }
    assert m["solver_iterations_median"] == 10.0
    assert m["warm_start_rate"] == 0.75
    assert m["agent_fallback_count"] == 1.0


def test_an_agent_without_solver_fields_reports_none():
    m = agent_metrics(pd_episode(max_steps=2))
    assert math.isnan(m["feedback_median"]) and math.isnan(m["agent_fallback_count"])
    assert m["solver_metadata"] == {"solver": (), "tolerance": (), "status": {}}


def test_a_belief_is_the_first_json_object_with_range_and_closing():
    assert parse_belief('{"ax": 1} then {"range_m": 120.5, "closing": true}') == {
        "range_m": 120.5,
        "closing": True,
    }
    assert parse_belief('{"range_m": "far", "closing": true}') is None
    assert parse_belief("no json") is None


def test_reported_facts_pass_through_and_incomplete_training_figures_are_rejected():
    training = {
        "env_steps": 1e6,
        "simulated_time_s": 1e7,
        "wall_clock_s": 3600.0,
        "compute": "1 GPU",
    }
    m = agent_metrics(
        pd_episode(max_steps=2), AgentReport(sign_check_pass=True, training_cost=training)
    )
    assert m["sign_check_pass"] is True and m["training_cost"] == training
    assert m["cost_of_learning"] is None and math.isnan(m["reasoning_faithfulness"])
    with pytest.raises(ValueError, match="propellant_kg"):
        AgentReport(cost_of_learning={"constraint_violations": 3, "learning_curve_area": 0.4})
