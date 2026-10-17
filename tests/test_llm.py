import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import LLMAgent, Price, Reply, text_model

START = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)


def env(steps=4, **fields):
    return ApproachEnv(ApproachConfig(start_state=START, max_steps=steps, **fields))


class Counting:
    """A stand-in model that answers with fixed text and token counts and keeps its requests."""

    def __init__(self, text='{"ax": 0.1, "ay": -0.2, "az": 0.0}', tokens=(120, 30)):
        self.text, self.tokens, self.requests = text, tokens, []

    def __call__(self, messages, tools):
        self.requests.append((messages, tools))
        return Reply(self.text, *self.tokens)


def test_model_gets_one_user_message_built_from_the_observation():
    model = Counting()
    run_episode(env(steps=2), LLMAgent(model))
    messages, tools = model.requests[0]
    assert tools == [] and [m["role"] for m in messages] == ["user"]
    assert "position = [0.0, 400.0, 0.0] m" in messages[0]["content"]
    assert "(t = 10 s)" in model.requests[1][0][0]["content"]


def test_tokens_and_cost_follow_the_declared_price():
    agent = LLMAgent(Counting(tokens=(120, 30)), price=Price(3.0, 15.0))
    run_episode(env(steps=4), agent)
    per_decision = (120 * 3.0 + 30 * 15.0) / 1e6
    assert all(e["cost_usd"] == pytest.approx(per_decision) for e in agent.log)
    usage = agent.usage()
    assert (usage["input_tokens"], usage["output_tokens"], usage["model_calls"]) == (480, 120, 4)
    assert usage["cost_usd"] == pytest.approx(4 * per_decision)
    assert LLMAgent(Counting()).usage()["cost_usd"] is None


def test_prompt_template_is_configurable_and_checked():
    template = "Hold {hold}, now at x={x:.0f} y={y:.0f} z={z:.0f}. {reply_format}"
    model = Counting()
    run_episode(env(steps=1), LLMAgent(model, prompt=template))
    content = model.requests[0][0][0]["content"]
    assert content.startswith("Hold [0.0, -100.0, 0.0], now at x=0 y=400 z=0. Reply with ONLY")
    with pytest.raises(ValueError, match="unknown prompt fields"):
        LLMAgent(model, prompt="Range {range_to_target}")


def test_unparseable_replies_coast_and_are_counted():
    agent = LLMAgent(text_model(lambda prompt: "thrusting toward the target"))
    result = run_episode(env(steps=3), agent)
    assert result["dv_used_mps"] == 0.0
    assert agent.usage()["parse_failures"] == 3
    assert np.array_equal(result["actions"], np.zeros((3, 3)))
