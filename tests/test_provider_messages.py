"""The messages-format adapter against a fake provider on 127.0.0.1."""

import itertools
import json
import logging
import socket
import ssl
from pathlib import Path

import numpy as np
import pytest
from provider_fakes import (
    FAKE_KEY,
    PAD,
    FakeProvider,
    Response,
    Sleeps,
    closed_port_url,
    messages_error,
    messages_reply,
    redacted_block,
    thinking_block,
)

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import InfrastructureError, LLMAgent, RecordedModel, ReplayModel, Reply, ToolCall
from vbar.agents.tools import CW_TOOLS
from vbar.providers import MESSAGES, ProviderError
from vbar.providers.messages import MessagesModel
from vbar.spacecraft import GaussianNav

MODEL = "test-model"
# A neutral stand-in for the version header the catalogue supplies.
VERSION_HEADER = {"x-test-version": "2026-10-21"}
GOLDEN = Path(__file__).parent / "data" / "messages_request.json"
SPECS = [tool.spec() for tool in CW_TOOLS]
PROMPT = [{"role": "user", "content": "Hold at the V-bar point."}]
CALL_1 = {"seconds": 600.0}
CALL_2 = {"x": 0.0, "y": -100.0, "z": 0.0, "seconds": 1500.0}
THINKING = thinking_block("Coast first, then plan the transfer.", "c2lnbmF0dXJlLTE=")
REDACTED = redacted_block("cmVkYWN0ZWQtMQ==")


def assistant(*calls, text=""):
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [{"id": i, "name": n, "arguments": a} for i, n, a in calls],
    }


def tool(call_id, name, content):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


CONVERSATION = [
    *PROMPT,
    assistant(("call-1", "predict_coast", CALL_1), ("call-2", "transfer_burn", CALL_2)),
    tool("call-1", "predict_coast", json.dumps({"position_m": [1.0, 2.0, 0.0]})),
    tool("call-2", "transfer_burn", json.dumps({"error": "singular transfer"})),
]
FIRST_REPLY = messages_reply(
    tool_calls=[("call-1", "predict_coast", CALL_1), ("call-2", "transfer_burn", CALL_2)],
    thinking=[THINKING, REDACTED],
)
ANSWER = '{"ax": 0.0, "ay": 0.0, "az": 0.0}'
INFRASTRUCTURE = {"rate", "network", "server"}


def messages_model(url, *, sleep=None, max_output_tokens=2048, key=FAKE_KEY, **fields):
    return MessagesModel(
        base_url=url,
        model=MODEL,
        key=key,
        max_output_tokens=max_output_tokens,
        headers=VERSION_HEADER,
        sleep=Sleeps() if sleep is None else sleep,
        **fields,
    )


def sentence(kind, status=None):
    if kind == "refused":
        return MESSAGES["refused"].format(status=status)
    return MESSAGES[kind].format(m=MODEL) if kind == "model" else MESSAGES[kind]


def lower(headers):
    return {k.lower(): v for k, v in headers.items()}


# Requests (criteria 6 and 8)


def test_the_request_with_echoed_thinking_and_merged_results_equals_the_golden_body():
    with FakeProvider([Response(200, FIRST_REPLY), Response(200, messages_reply(ANSWER))]) as fake:
        model = messages_model(fake.url)
        model(PROMPT, SPECS)
        model(CONVERSATION, SPECS)
    request = fake.requests[1]
    assert request.method == "POST" and request.path == "/v1/messages"
    assert request.json() == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_key_goes_in_a_bearer_header_with_the_callers_headers():
    with FakeProvider([Response(200, messages_reply(ANSWER))]) as fake:
        messages_model(fake.url)(PROMPT, [])
    headers = lower(fake.requests[0].headers)
    assert headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert headers["x-test-version"] == "2026-10-21"
    assert headers["content-type"].startswith("application/json")
    assert FAKE_KEY not in fake.requests[0].raw.decode("utf-8")


def test_a_call_without_tools_sends_no_tools_field():
    with FakeProvider([Response(200, messages_reply(ANSWER))]) as fake:
        messages_model(fake.url)(PROMPT, [])
    assert "tools" not in fake.requests[0].json()


@pytest.mark.parametrize("limit", [64, 2048, 32000])
def test_the_reply_limit_sent_is_max_output_tokens(limit):
    with FakeProvider([Response(200, messages_reply(ANSWER))]) as fake:
        messages_model(fake.url, max_output_tokens=limit)(PROMPT, SPECS)
    body = fake.requests[0].json()
    assert body["max_tokens"] == limit
    assert "max_completion_tokens" not in body and "temperature" not in body


def test_tool_results_come_first_in_the_merged_user_message():
    conversation = [*CONVERSATION, {"role": "user", "content": "Go on."}]
    with FakeProvider([Response(200, messages_reply(ANSWER))]) as fake:
        messages_model(fake.url)(conversation, SPECS)
    sent = fake.requests[0].json()["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert [block["type"] for block in sent[2]["content"]] == [
        "tool_result",
        "tool_result",
        "text",
    ]
    assert sent[2]["content"][2]["text"] == "Go on."


def test_thinking_is_sent_back_with_the_tool_calls_it_came_with():
    first = messages_reply(
        tool_calls=[("call-a", "predict_coast", CALL_1)],
        thinking=[thinking_block("plan a", "sig-a")],
    )
    second = messages_reply(
        tool_calls=[("call-b", "predict_coast", CALL_1)],
        thinking=[thinking_block("plan b", "sig-b"), redacted_block("opaque-b")],
    )
    replies = [Response(200, first), Response(200, second), Response(200, messages_reply(ANSWER))]
    with FakeProvider(replies) as fake:
        model = messages_model(fake.url)
        model(PROMPT, SPECS)
        model([{"role": "user", "content": "Another step."}], SPECS)
        model(
            [*PROMPT, assistant(("call-a", "predict_coast", CALL_1)), tool("call-a", "x", "{}")],
            SPECS,
        )
    content = fake.requests[2].json()["messages"][1]["content"]
    assert content[0] == thinking_block("plan a", "sig-a")
    assert [block["type"] for block in content] == ["thinking", "tool_use"]


def test_assistant_text_is_kept_between_thinking_and_tool_use():
    first = messages_reply("Checking the coast.", [("call-1", "predict_coast", CALL_1)], [THINKING])
    with FakeProvider([Response(200, first), Response(200, messages_reply(ANSWER))]) as fake:
        model = messages_model(fake.url)
        model(PROMPT, SPECS)
        model(
            [
                *PROMPT,
                assistant(("call-1", "predict_coast", CALL_1), text="Checking the coast."),
                tool("call-1", "predict_coast", "{}"),
            ],
            SPECS,
        )
    content = fake.requests[1].json()["messages"][1]["content"]
    assert content == [
        THINKING,
        {"type": "text", "text": "Checking the coast."},
        {"type": "tool_use", "id": "call-1", "name": "predict_coast", "input": CALL_1},
    ]


def test_a_new_decision_sends_no_thinking():
    with FakeProvider([Response(200, FIRST_REPLY)]) as fake:
        model = messages_model(fake.url)
        model(PROMPT, SPECS)
        model([{"role": "user", "content": "Next step."}], SPECS)
    assert fake.requests[1].json()["messages"] == [{"role": "user", "content": "Next step."}]


# Replies (criterion 7)


def test_a_text_reply_parses_with_its_token_counts():
    body = messages_reply(ANSWER, input_tokens=812, output_tokens=45)
    with FakeProvider([Response(200, body)]) as fake:
        reply = messages_model(fake.url)(PROMPT, [])
    assert reply == Reply(ANSWER, 812, 45, ())


def test_tool_calls_parse_and_thinking_is_not_in_the_text():
    with FakeProvider([Response(200, messages_reply(ANSWER, [], [THINKING]))]) as fake:
        assert messages_model(fake.url)(PROMPT, SPECS).text == ANSWER
    body = messages_reply(
        tool_calls=[("call_a", "predict_coast", {"seconds": 600})],
        thinking=[THINKING],
        input_tokens=900,
        output_tokens=60,
    )
    with FakeProvider([Response(200, body)]) as fake:
        reply = messages_model(fake.url)(PROMPT, SPECS)
    assert reply == Reply("", 900, 60, (ToolCall("predict_coast", {"seconds": 600}, "call_a"),))


@pytest.mark.parametrize("body", [{}, {"content": None}, {"type": "message"}])
def test_a_success_answer_without_content_is_a_server_error(body):
    with FakeProvider([Response(200, body)]) as fake:
        with pytest.raises(ProviderError) as caught:
            messages_model(fake.url)(PROMPT, [])
    assert caught.value.kind == "server" and str(caught.value) == sentence("server")


# A whole episode (criterion 9) and what the agent's requests carry (criterion 17)


def scripted_provider():
    """Calls a tool with fresh thinking after each prompt, then answers; answers vary."""
    counter = itertools.count()
    issued = []

    def respond(request):
        body = request.json()
        if "tools" in body and len(body["messages"]) == 1:
            n = next(counter)
            block = thinking_block(f"step {n}", f"sig-{n}")
            issued.append(block)
            calls = [(f"toolu-{n}", "predict_coast", {"seconds": 300})]
            return Response(200, messages_reply(tool_calls=calls, thinking=[block]))
        content = body["messages"][0]["content"]
        ax = (len(content) % 7 - 3) / 10
        answer = json.dumps({"ax": ax, "ay": -0.1, "az": 0.0})
        return Response(200, messages_reply(answer, input_tokens=len(content)))

    return respond, issued


def episode(model, steps=25):
    environment = ApproachEnv(ApproachConfig(max_steps=steps, sensor=GaussianNav(2.0, 2e-3)))
    agent = LLMAgent(model, tools=CW_TOOLS)
    return run_episode(environment, agent, seed=4, agent_seed=0), agent


def test_a_25_step_episode_with_tools_records_and_replays_exactly(tmp_path):
    respond, issued = scripted_provider()
    with FakeProvider(respond) as fake:
        recorder = RecordedModel(messages_model(fake.url))
        recorded, first = episode(recorder)
    assert len(fake.requests) == 50
    assert all(len(e["tool_calls"]) == 1 for e in first.log)
    for block, request in zip(issued, fake.requests[1::2], strict=True):
        assert request.json()["messages"][1]["content"][0] == block
    recorder.save(tmp_path / "transcript.json")
    assert FAKE_KEY not in (tmp_path / "transcript.json").read_text(encoding="utf-8")
    replayed, second = episode(ReplayModel.load(tmp_path / "transcript.json"))
    assert np.array_equal(recorded["states"], replayed["states"])
    assert np.array_equal(recorded["actions"], replayed["actions"])
    assert [e["messages"] for e in first.log] == [e["messages"] for e in second.log]
    assert first.usage() == second.usage()


def user_texts(message):
    """The texts of a sent user message: its string content, text blocks and tool results."""
    if isinstance(message["content"], str):
        return [message["content"]]
    return [b.get("text", b.get("content")) for b in message["content"] if b["type"] != "thinking"]


def test_requests_carry_only_the_logged_prompts_and_results_never_the_true_position():
    respond, _ = scripted_provider()
    with FakeProvider(respond) as fake:
        result, agent = episode(messages_model(fake.url), steps=6)
    assert len(fake.requests) == 2 * len(agent.log)
    for k, entry in enumerate(agent.log):
        logged = entry["messages"]
        for request, held in zip(
            fake.requests[2 * k : 2 * k + 2], (logged[:1], logged), strict=True
        ):
            body = request.json()
            assert "system" not in body
            sent = [t for m in body["messages"] if m["role"] == "user" for t in user_texts(m)]
            assert sent == [m["content"] for m in held if m["role"] in ("user", "tool")]
        assert body["messages"][0]["content"] == entry["prompt"]
    raw = b"".join(r.raw for r in fake.requests).decode("utf-8")
    for x, y, z in result["states"][:, :3]:
        assert f"[{x:.2f}, {y:.2f}, {z:.2f}]" not in raw


# Errors (criteria 2 and 10)

SPEND_LIMIT = "You have reached your specified API usage limits. You will regain access later."
ERROR_TABLE = [
    (401, {}, messages_error("authentication_error", "invalid x-api-key"), "key"),
    (402, {}, messages_error("billing_error", "Your credit balance is too low."), "credit"),
    (
        429,
        {},
        messages_error("rate_limit_error", "Spend limit.", "enforced_spend_limit_reached"),
        "credit",
    ),
    (
        429,
        {"Retry-After": "30"},
        messages_error("rate_limit_error", "Spend limit.", "enforced_spend_limit_reached"),
        "credit",
    ),
    (400, {}, messages_error("invalid_request_error", SPEND_LIMIT), "credit"),
    (429, {"Retry-After": "1"}, messages_error("rate_limit_error", "Slow down."), "rate"),
    (429, {}, "Too many requests", "rate"),
    (404, {}, messages_error("not_found_error", "model: test-model"), "model"),
    (400, {}, messages_error("invalid_request_error", "messages: field required"), "refused"),
    (400, {}, "Bad request", "refused"),
    (403, {}, messages_error("permission_error", "Not allowed."), "refused"),
    (413, {}, messages_error("request_too_large", "Request too large."), "refused"),
    (500, {}, messages_error("api_error", "Internal error."), "server"),
    (529, {}, messages_error("overloaded_error", "Overloaded."), "server"),
    (502, {}, "Bad gateway", "server"),
]


@pytest.mark.parametrize(("status", "headers", "body", "kind"), ERROR_TABLE)
def test_error_responses_map_to_a_kind_and_its_sentence(status, headers, body, kind):
    with FakeProvider([Response(status, body, headers)]) as fake:
        with pytest.raises(ProviderError) as caught:
            messages_model(fake.url)(PROMPT, [])
    error = caught.value
    assert error.kind == kind
    assert str(error) == sentence(kind, status)
    assert isinstance(error, InfrastructureError) == (kind in INFRASTRUCTURE)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 429, 500, 529])
def test_errors_never_carry_the_key_or_the_providers_text(status):
    echoed = f"Incorrect API key provided: {FAKE_KEY}. Visit the console."
    with FakeProvider([Response(status, messages_error("api_error", echoed))]) as fake:
        with pytest.raises(ProviderError) as caught:
            messages_model(fake.url)(PROMPT, [])
    for text in (str(caught.value), repr(caught.value), str(caught.value.args)):
        assert FAKE_KEY not in text and "Visit the console" not in text


def test_the_model_repr_and_debug_logs_omit_the_key(caplog):
    caplog.set_level(logging.DEBUG)
    model = messages_model("http://127.0.0.1:1")
    assert FAKE_KEY not in repr(model) and FAKE_KEY not in str(model)
    with FakeProvider([Response(401, messages_error("authentication_error", FAKE_KEY))]) as fake:
        with pytest.raises(ProviderError):
            messages_model(fake.url)(PROMPT, [])
    with FakeProvider([Response(200, messages_reply(ANSWER))]) as fake:
        messages_model(fake.url)(PROMPT, [])
    assert all(FAKE_KEY not in r.getMessage() for r in caplog.records)


def test_a_refused_connection_and_a_timeout_are_network_errors():
    with pytest.raises(ProviderError) as caught:
        messages_model(closed_port_url())(PROMPT, [])
    assert caught.value.kind == "network" and isinstance(caught.value, InfrastructureError)
    with FakeProvider([Response(200, messages_reply(ANSWER), delay_s=2.0)]) as fake:
        with pytest.raises(ProviderError) as caught:
            messages_model(fake.url, timeout=0.3)(PROMPT, [])
    assert caught.value.kind == "network" and len(fake.requests) == 1


def test_an_unverified_certificate_is_a_tls_error(monkeypatch):
    def refuse(self):
        raise ssl.SSLCertVerificationError(1, "certificate verify failed")

    monkeypatch.setattr("http.client.HTTPSConnection.connect", refuse)
    with pytest.raises(ProviderError) as caught:
        messages_model("https://127.0.0.1:9")(PROMPT, [])
    assert caught.value.kind == "tls" and not isinstance(caught.value, InfrastructureError)


# Redirects, plain HTTP, credentials in the address and the key's characters (criterion 11)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_refused_and_the_second_host_gets_nothing(status):
    with FakeProvider([Response(200, messages_reply(ANSWER))]) as second:
        location = {"Location": second.url + "/v1/messages"}
        with FakeProvider([Response(status, None, location)]) as first:
            with pytest.raises(ProviderError) as caught:
                messages_model(first.url)(PROMPT, [])
    assert caught.value.kind == "network"
    assert second.requests == []


@pytest.fixture
def no_connections(monkeypatch):
    opened = []

    def record(*args, **kwargs):
        opened.append(args)
        raise OSError("connection attempted")

    monkeypatch.setattr(socket, "create_connection", record)
    monkeypatch.setattr(socket, "getaddrinfo", record)
    return opened


@pytest.mark.parametrize("url", ["http://10.0.0.5:8080", "http://192.168.1.20"])
def test_plain_http_to_another_host_is_refused_before_connecting(url, no_connections):
    with pytest.raises((ValueError, ProviderError)):
        messages_model(url)(PROMPT, [])
    assert no_connections == []


SECRET = "pw-5d1c8e2f9a7b"


@pytest.mark.parametrize(
    "url", [f"https://user:{SECRET}@api.example.com", f"http://user:{SECRET}@127.0.0.1:9"]
)
def test_a_base_url_with_credentials_is_refused_without_quoting_them(url, no_connections):
    with pytest.raises(ValueError) as caught:
        messages_model(url)(PROMPT, [])
    error = caught.value
    while error is not None:
        for text in (str(error), repr(error), str(error.args)):
            assert SECRET not in text and "user:" not in text
        error = error.__cause__ or error.__context__
    assert no_connections == []


@pytest.mark.parametrize(
    "key",
    [
        f"{FAKE_KEY}\r",
        f"{FAKE_KEY}\nX-Injected: 1",
        f"{FAKE_KEY}\x00",
        f"{FAKE_KEY[:8]}\u00e9{FAKE_KEY[8:]}",
        f"{FAKE_KEY[:8]}\u200b{FAKE_KEY[8:]}",
        f"{FAKE_KEY[:8]}\u0430{FAKE_KEY[8:]}",
    ],
)
def test_a_key_with_a_control_or_non_ascii_character_is_refused(key, no_connections):
    with pytest.raises(ValueError) as caught:
        messages_model("http://127.0.0.1:9", key=key)(PROMPT, [])
    for text in (str(caught.value), repr(caught.value), str(caught.value.args)):
        assert FAKE_KEY[8:] not in text
    assert no_connections == []


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_a_reply_limit_that_is_not_a_positive_integer_is_refused(limit):
    with pytest.raises(ValueError):
        messages_model("http://127.0.0.1:9", max_output_tokens=limit)


# Retries (criterion 12)


def test_a_rate_limit_is_retried_twice_after_the_given_delay():
    sleep = Sleeps()
    rate = Response(429, messages_error("rate_limit_error", "Slow down."), {"Retry-After": "3"})
    with FakeProvider([rate]) as fake:
        with pytest.raises(ProviderError) as caught:
            messages_model(fake.url, sleep=sleep)(PROMPT, [])
    assert caught.value.kind == "rate"
    assert len(fake.requests) == 3 and sleep.delays == [3.0, 3.0]


def test_an_overloaded_answer_is_retried_and_can_clear():
    sleep = Sleeps()
    overloaded = Response(529, messages_error("overloaded_error", "Overloaded."))
    with FakeProvider([overloaded, Response(200, messages_reply(ANSWER))]) as fake:
        reply = messages_model(fake.url, sleep=sleep)(PROMPT, [])
    assert reply.text == ANSWER and len(fake.requests) == 2 and len(sleep.delays) == 1


@pytest.mark.parametrize(
    "response",
    [
        Response(
            429,
            messages_error("rate_limit_error", "Spend limit.", "enforced_spend_limit_reached"),
        ),
        Response(
            429,
            messages_error("rate_limit_error", "Spend limit.", "enforced_spend_limit_reached"),
            {"Retry-After": "30"},
        ),
        Response(402, messages_error("billing_error", "Your credit balance is too low.")),
        Response(400, messages_error("invalid_request_error", SPEND_LIMIT)),
        Response(401, messages_error("authentication_error", "invalid x-api-key")),
        Response(404, messages_error("not_found_error", "model: test-model")),
        Response(403, messages_error("permission_error", "Not allowed.")),
    ],
)
def test_credit_key_model_and_refused_errors_are_not_retried(response):
    sleep = Sleeps()
    with FakeProvider([response]) as fake:
        with pytest.raises(ProviderError):
            messages_model(fake.url, sleep=sleep)(PROMPT, [])
    assert len(fake.requests) == 1 and sleep.delays == []


# Size of a reply

LIMIT = 8 * 1024 * 1024


def padded_reply(total):
    body = messages_reply(PAD)
    return body, total - (len(json.dumps(body).encode("utf-8")) - len(PAD))


def test_a_reply_one_byte_over_8_mib_is_a_server_error():
    body, pad = padded_reply(LIMIT + 1)
    with FakeProvider([Response(200, body, pad=pad)]) as fake:
        with pytest.raises(ProviderError) as caught:
            messages_model(fake.url)(PROMPT, [])
    assert caught.value.kind == "server"


def test_a_reply_of_exactly_8_mib_is_accepted():
    body, pad = padded_reply(LIMIT)
    with FakeProvider([Response(200, body, pad=pad)]) as fake:
        reply = messages_model(fake.url)(PROMPT, [])
    assert len(reply.text) == pad and set(reply.text) == {"a"}
