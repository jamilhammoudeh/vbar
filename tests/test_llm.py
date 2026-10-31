import json
import re
import time

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, RealTime, run_episode
from vbar.agents import (
    LLMAgent,
    Price,
    RecordedModel,
    ReplayError,
    ReplayModel,
    Reply,
    ToolCall,
    belief_matches,
    parse_action,
    parse_belief,
    text_model,
)
from vbar.agents.llm import first_json, json_depth
from vbar.agents.tools import CW_TOOLS
from vbar.dynamics import cw_discrete
from vbar.spacecraft import GaussianNav

START = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)


def env(steps=4, **fields):
    fields.setdefault("start_state", START)
    return ApproachEnv(ApproachConfig(max_steps=steps, **fields))


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
    assert "position = [0.00, 400.00, 0.00] m" in messages[0]["content"]
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
        (call("predict_coast", seconds="300"), "seconds must be a number, got '300'"),
        (call("predict_coast", seconds=True), "seconds must be a number, got True"),
        (call("transfer_burn", x=0.0, y="-100", z=0.0, seconds=900.0), "y must be a number"),
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


class Stochastic:
    """A stand-in model with its own random stream: random fractions, sometimes a tool call."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, messages, tools):
        if tools and self.rng.random() < 0.3:
            return call("predict_coast", seconds=float(self.rng.integers(10, 900)))
        ax, ay, az = self.rng.uniform(-0.3, 0.3, 3)
        tokens = (int(self.rng.integers(500, 900)), int(self.rng.integers(10, 40)))
        return Reply(json.dumps({"ax": ax, "ay": ay, "az": az}), *tokens)


def noisy_run(model, seed=4, **run_fields):
    environment = ApproachEnv(ApproachConfig(max_steps=25, sensor=GaussianNav(2.0, 2e-3)))
    agent = LLMAgent(model, tools=CW_TOOLS)
    return run_episode(environment, agent, seed=seed, agent_seed=0, **run_fields), agent


def test_runs_replay_identically_from_cached_replies(tmp_path):
    recorder = RecordedModel(Stochastic(7))
    recorded, first = noisy_run(recorder)
    assert any(e["tool_calls"] for e in first.log)
    recorder.save(tmp_path / "transcript.json")
    replayed, second = noisy_run(ReplayModel.load(tmp_path / "transcript.json"))
    assert np.array_equal(recorded["states"], replayed["states"])
    assert np.array_equal(recorded["actions"], replayed["actions"])
    assert [e["messages"] for e in first.log] == [e["messages"] for e in second.log]
    assert first.usage()["input_tokens"] == second.usage()["input_tokens"]


def test_replay_refuses_a_run_that_diverges_or_outlasts_the_transcript():
    recorder = RecordedModel(Stochastic(7))
    noisy_run(recorder)
    with pytest.raises(ReplayError, match="request 0 differs"):
        noisy_run(ReplayModel(recorder.entries), seed=5)
    with pytest.raises(ReplayError, match="more than the 3 recorded"):
        noisy_run(ReplayModel(recorder.entries[:3]))


def test_real_time_runs_replay_with_their_recorded_latencies():
    recorder = RecordedModel(Stochastic(2))
    recorded, _ = noisy_run(recorder, real_time=RealTime(time_scale=500.0))
    repeat = RealTime(time_scale=500.0, latencies=tuple(recorded["latencies_s"]))
    replayed, _ = noisy_run(ReplayModel(recorder.entries), real_time=repeat)
    assert np.array_equal(recorded["states"], replayed["states"])


def reader(range_scale=1.0):
    """A stand-in model that reads its observed state from the prompt and reports a belief."""

    def complete(prompt):
        position = np.array(json.loads(re.search(r"position = (\[.*?\])", prompt).group(1)))
        velocity = np.array(json.loads(re.search(r"velocity = (\[.*?\])", prompt).group(1)))
        belief = {
            "range_m": range_scale * float(np.linalg.norm(position)),
            "closing": bool(position @ velocity < 0.0),
        }
        return json.dumps({"ax": 0.0, "ay": -0.3, "az": 0.0, **belief})

    return text_model(complete)


def test_a_model_that_reads_its_inputs_is_faithful_through_navigation_noise():
    # 20 m of noise on a 400 m range: a model that reads its inputs is faithful to them.
    cfg = dict(start_state=(30.0, 400.0, 0.0, 0.0, -0.5, 0.0), sensor=GaussianNav(20.0, 0.02))
    agent = LLMAgent(reader(), belief=True)
    run_episode(env(steps=20, **cfg), agent, seed=3)
    assert agent.usage()["faithful_fraction"] == 1.0
    assert "closing" in agent.log[0]["prompt"]
    off = LLMAgent(reader(range_scale=1.5), belief=True)
    run_episode(env(steps=20, **cfg), off, seed=3)
    assert off.usage()["faithful_fraction"] == 0.0


def test_beliefs_need_both_fields_and_the_right_closing_sign():
    obs = np.array([0.0, 400.0, 0.0, 0.0, -0.5, 0.0])
    assert belief_matches({"range_m": 380.0, "closing": True}, obs)
    assert not belief_matches({"range_m": 380.0, "closing": False}, obs)
    assert not belief_matches({"range_m": 350.0, "closing": True}, obs)
    assert not belief_matches(None, obs)
    assert parse_belief('{"ax": 0, "ay": 0, "az": 0, "range_m": 10, "closing": "yes"}') is None
    assert parse_belief('{"ax": 0, "ay": 0, "az": 0}') is None
    plain = LLMAgent(text_model(lambda prompt: '{"ax": 0, "ay": 0, "az": 0}'))
    run_episode(env(steps=2), plain)
    assert plain.log[0]["faithful"] is None and plain.usage()["faithful_fraction"] is None
    assert "range_m" not in plain.log[0]["prompt"]


@pytest.mark.filterwarnings("ignore:overflow", "ignore:invalid value")
def test_tools_refuse_to_return_non_finite_numbers():
    environment = env()
    obs, _ = environment.reset(seed=0)
    view = environment.agent_view()
    from vbar.agents.tools import predict_coast, transfer_burn

    found = transfer_burn(obs, view, {"x": 1e308, "y": -1e308, "z": 1e308, "seconds": 600.0})
    assert "not finite" in found["error"]
    assert (
        "not finite"
        in predict_coast(np.r_[1e308, 0, 0, 1e308, 0, 0], view, {"seconds": 60.0})["error"]
    )


def test_numbers_too_large_for_a_float_are_errors_not_crashes():
    huge = int("9" * 400)
    entry, *_ = first_decision(Scripted(call("predict_coast", seconds=huge), ANSWER))
    assert "bad arguments" in entry["tool_calls"][0]["result"]["error"]
    reply = (
        '{"ax": ' + "9" * 400 + ', "ay": 0, "az": 0, "range_m": ' + "9" * 400 + ', "closing": true}'
    )
    assert parse_action(reply) is None and parse_belief(reply) is None


def test_deeply_nested_replies_are_parse_failures():
    reply = (
        '{"ax": ' + "[" * 3000 + "]" * 3000 + ', "ay": 0, "az": 0, "range_m": 1, "closing": true}'
    )
    assert parse_action(reply) is None and parse_belief(reply) is None
    agent = LLMAgent(text_model(lambda prompt: reply), belief=True)
    assert run_episode(env(steps=2), agent)["dv_used_mps"] == 0.0
    assert agent.usage()["parse_failures"] == 2


@pytest.mark.parametrize(
    ("name", "arguments"),
    [("predict_coast", [1, 2]), ("predict_coast", "seconds"), ("predict_coast", None), (["x"], {})],
)
def test_malformed_tool_calls_come_back_as_errors(name, arguments):
    entry, *_ = first_decision(
        Scripted(Reply(tool_calls=(ToolCall(name, arguments, "c"),)), ANSWER)
    )
    assert "error" in entry["tool_calls"][0]["result"]
    assert not entry["parse_failed"]


def test_tool_calls_with_odd_arguments_record_and_replay(tmp_path):
    odd = [[1, 2], "seconds", None, 5]
    replies = [
        Reply(tool_calls=(ToolCall("predict_coast", a, f"c{k}"),)) for k, a in enumerate(odd)
    ]
    recorder = RecordedModel(Scripted(*replies, ANSWER))
    environment = ApproachEnv(ApproachConfig(start_state=START, max_steps=1))
    first = LLMAgent(recorder, tools=CW_TOOLS, max_tool_rounds=4)
    recorded = run_episode(environment, first, seed=0, agent_seed=1)
    assert all("error" in call["result"] for call in first.log[0]["tool_calls"])
    recorder.save(tmp_path / "odd.json")
    second = LLMAgent(ReplayModel.load(tmp_path / "odd.json"), tools=CW_TOOLS, max_tool_rounds=4)
    replayed = run_episode(environment, second, seed=0, agent_seed=1)
    assert np.array_equal(recorded["states"], replayed["states"])
    assert first.log[0]["messages"] == second.log[0]["messages"]


def test_a_reply_without_text_is_a_parse_failure():
    # Some providers send content None with tool calls; an adapter may pass it on.
    agent = LLMAgent(lambda messages, tools: Reply(text=None))
    result = run_episode(env(steps=2), agent)
    assert result["dv_used_mps"] == 0.0 and agent.usage()["parse_failures"] == 2


def nested_belief(depth):
    """A belief reply whose outer object and arrays nest `depth` levels."""
    inner = depth - 1
    return '{"range_m": 1, "closing": true, "x": ' + "[" * inner + "]" * inner + "}"


def test_replies_nested_32_levels_parse():
    reply = nested_belief(32)
    assert json_depth(reply) == 32
    assert first_json(reply)["closing"] is True
    assert parse_belief(reply) == {"range_m": 1.0, "closing": True}


@pytest.mark.parametrize("depth", [33, 3000, 50000])
def test_replies_nested_past_the_limit_are_refused_on_every_python(depth):
    # 50000 levels make json.loads raise on every supported Python; 3000 parse on some builds.
    reply = nested_belief(depth)
    assert first_json(reply) is None and parse_belief(reply) is None


def test_brackets_inside_strings_do_not_count_as_nesting():
    assert json_depth('{"note": "[[[[{{{{", "v": [1, [2]]}') == 3
    assert json_depth(r'{"q": "an escaped \" [ quote"}') == 1


@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_llm_agent_refuses_modes_without_a_continuous_command(mode):
    environment = ApproachEnv(ApproachConfig(action_mode=mode, max_steps=2))
    with pytest.raises(ValueError, match=f"'{mode}' takes no continuous command"):
        run_episode(environment, LLMAgent(text_model(lambda prompt: ANSWER.text)))


def test_an_action_object_holding_a_nested_value_is_read():
    reply = '{"ax": 0.5, "ay": -0.25, "az": 0, "note": {"why": "brake {gently}"}}'
    assert np.array_equal(parse_action(reply), [0.5, -0.25, 0.0])


def test_an_action_after_prose_braces_or_before_deep_brackets_is_read():
    action = '{"ax": 0.1, "ay": 0, "az": 0}'
    assert parse_action("{ " * 7 + action) is not None
    assert parse_action("{ " * 8 + action) is None
    assert parse_action(action + " " + "[" * 100) is not None


def test_a_hostile_reply_is_parsed_in_linear_time():
    # The old lazy regex took time quadratic in the number of opening braces (about 100 s for
    # 65,000); each of the eight tries now stops where its value closes or nests too deep.
    start = time.perf_counter()
    for reply in ("{" * 200_000, '{"a": "' + "x" * 2_000_000, "[" * 200_000):
        assert parse_action(reply) is None
    assert time.perf_counter() - start < 60.0


def test_the_depth_check_stops_as_soon_as_a_value_nests_too_deep():
    from vbar.agents.llm import MAX_JSON_DEPTH, _depth_from

    assert _depth_from("{" * 1000, 0) == MAX_JSON_DEPTH + 1
    assert _depth_from('{"a": [[1]]} ' + "[" * 1000, 0) == 3
