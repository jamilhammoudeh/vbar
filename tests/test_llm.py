import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import LLMAgent, Price, Reply, ToolCall, text_model
from vbar.agents.tools import CW_TOOLS
from vbar.dynamics import cw_discrete
from vbar.spacecraft import GaussianNav

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


ANSWER = Reply('{"ax": 0.0, "ay": 0.0, "az": 0.0}')


class Scripted:
    """Replies from a list in turn (the last one repeats) and keeps copies of its requests."""

    def __init__(self, *replies):
        self.replies, self.requests = replies, []

    def __call__(self, messages, tools):
        self.requests.append(([dict(m) for m in messages], list(tools)))
        return self.replies[min(len(self.requests), len(self.replies)) - 1]


def first_decision(model, sensor=None, **agent_fields):
    environment = env(sensor=sensor) if sensor else env()
    obs, info = environment.reset(seed=1)
    agent = LLMAgent(model, tools=CW_TOOLS, **agent_fields)
    agent.reset(environment.agent_view())
    agent.act(obs, environment.agent_info())
    return agent.log[0], obs, info, environment.n


def call(name, **arguments):
    return Reply(tool_calls=(ToolCall(name, arguments, id=f"call-{name}"),))


def test_tool_results_come_from_the_noisy_observation_not_the_truth():
    model = Scripted(call("predict_coast", seconds=600.0), ANSWER)
    entry, obs, info, n = first_decision(model, sensor=GaussianNav(5.0, 5e-3))
    phi, _ = cw_discrete(n, 600.0)
    result = entry["tool_calls"][0]["result"]
    # Rounded to 0.01 m; the truth's prediction is metres away through the 5 m noise.
    assert np.allclose(result["position_m"], (phi @ obs[:6])[:3], atol=0.006)
    assert np.max(np.abs(np.array(result["position_m"]) - (phi @ info["true_state"])[:3])) > 1.0
    (first, offered), (second, _) = model.requests
    assert [t["name"] for t in offered] == ["predict_coast", "transfer_burn"]
    assert [m["role"] for m in second] == ["user", "assistant", "tool"]
    assert second[1]["tool_calls"][0]["id"] == second[2]["tool_call_id"] == "call-predict_coast"
    assert not entry["parse_failed"] and entry["model_calls"] == 2


def test_tool_rounds_are_capped_and_the_last_call_offers_no_tools():
    model = Scripted(call("predict_coast", seconds=60.0))
    entry, *_ = first_decision(model, max_tool_rounds=2)
    assert [len(tools) for _, tools in model.requests] == [2, 2, 0]
    assert entry["model_calls"] == 3 and len(entry["tool_calls"]) == 2 and entry["parse_failed"]


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (call("thrust_now"), "unknown tool"),
        (call("predict_coast"), "bad arguments"),
        (call("predict_coast", seconds=-5.0), "seconds must be in"),
        (call("transfer_burn", x=0.0, y=-100.0, z=0.0, seconds=float("nan")), "finite"),
    ],
)
def test_bad_tool_calls_come_back_as_errors(reply, error):
    entry, *_ = first_decision(Scripted(reply, ANSWER))
    assert error in entry["tool_calls"][0]["result"]["error"]


def test_transfer_burn_arrives_at_rest_at_the_target():
    model = Scripted(call("transfer_burn", x=0.0, y=-100.0, z=0.0, seconds=1500.0), ANSWER)
    entry, obs, _, n = first_decision(model)
    result = entry["tool_calls"][0]["result"]
    phi, _ = cw_discrete(n, 1500.0)
    arrival = phi @ np.r_[obs[:3], obs[3:6] + result["first_burn_mps"]]
    # Burns are rounded to 1e-4 m/s, which moves the arrival by up to 0.15 m over 1500 s.
    assert np.allclose(arrival[:3], [0.0, -100.0, 0.0], atol=0.3)
    assert np.allclose(arrival[3:] + result["second_burn_mps"], 0.0, atol=1e-3)
    half_orbit = np.pi / n
    model = Scripted(call("transfer_burn", x=0.0, y=-100.0, z=0.0, seconds=half_orbit), ANSWER)
    entry, *_ = first_decision(model)
    assert "singular" in entry["tool_calls"][0]["result"]["error"]
