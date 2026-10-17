import dataclasses
import json
import math
import re

import numpy as np
import pytest
from metric_cases import episode, pd_episode

from vbar.agents import LLMAgent
from vbar.metrics import AgentReport, agent_metrics, faithfulness, parse_belief
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
