"""The chat-format adapter and the HTTP layer under it, against a fake provider on 127.0.0.1."""

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
    chat_error,
    chat_reply,
    closed_port_url,
)

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import (
    InfrastructureError,
    LLMAgent,
    RecordedModel,
    ReplayModel,
    Reply,
    ToolCall,
)
from vbar.agents.tools import CW_TOOLS
from vbar.providers import MESSAGES, ProviderError
from vbar.providers.chat import ChatModel
from vbar.providers.http import check_url, post_json
from vbar.spacecraft import GaussianNav

MODEL = "test-model"
GOLDEN = Path(__file__).parent / "data" / "chat_request.json"
SPECS = [tool.spec() for tool in CW_TOOLS]
PROMPT = [{"role": "user", "content": "Hold at the V-bar point."}]
CONVERSATION = [
    *PROMPT,
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call-1", "name": "predict_coast", "arguments": {"seconds": 600.0}},
            {
                "id": "call-2",
                "name": "transfer_burn",
                "arguments": {"x": 0.0, "y": -100.0, "z": 0.0, "seconds": 1500.0},
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "predict_coast",
        "content": json.dumps({"position_m": [1.0, 2.0, 0.0]}),
    },
    {
        "role": "tool",
        "tool_call_id": "call-2",
        "name": "transfer_burn",
        "content": json.dumps({"error": "singular transfer"}),
    },
]
ANSWER = '{"ax": 0.0, "ay": 0.0, "az": 0.0}'
INFRASTRUCTURE = {"rate", "network", "server"}


def chat_model(url, *, sleep=None, max_output_tokens=2048, **fields):
    return ChatModel(
        base_url=url + "/v1",
        model=MODEL,
        key=FAKE_KEY,
        max_output_tokens=max_output_tokens,
        sleep=Sleeps() if sleep is None else sleep,
        **fields,
    )


def with_parsed_arguments(body):
    """The request body with each tool call's arguments string parsed, so spacing is free."""
    body = json.loads(json.dumps(body))
    for message in body["messages"]:
        for call in message.get("tool_calls", ()):
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return body


def sentence(kind, status=None):
    if kind == "refused":
        return (
            f"The provider refused the request (status {status}); run `vbar setup --show` to "
            "check the model, and report it if the problem stays."
        )
    return MESSAGES[kind].format(m=MODEL) if kind == "model" else MESSAGES[kind]


# Requests (criteria 6 and 8)


def test_the_request_with_tool_results_equals_the_golden_body():
    with FakeProvider([Response(200, chat_reply(ANSWER))]) as fake:
        chat_model(fake.url)(CONVERSATION, SPECS)
    (request,) = fake.requests
    assert request.method == "POST" and request.path == "/v1/chat/completions"
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert with_parsed_arguments(request.json()) == with_parsed_arguments(golden)


def test_the_key_goes_in_a_bearer_header_and_the_body_is_json():
    with FakeProvider([Response(200, chat_reply(ANSWER))]) as fake:
        chat_model(fake.url)(PROMPT, [])
    headers = {k.lower(): v for k, v in fake.requests[0].headers.items()}
    assert headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert headers["content-type"].startswith("application/json")
    assert FAKE_KEY not in fake.requests[0].raw.decode("utf-8")


def test_a_call_without_tools_sends_no_tools_field():
    with FakeProvider([Response(200, chat_reply(ANSWER))]) as fake:
        chat_model(fake.url)(PROMPT, [])
    assert "tools" not in fake.requests[0].json()


@pytest.mark.parametrize("limit", [64, 2048, 32000])
def test_the_reply_limit_sent_is_max_output_tokens(limit):
    with FakeProvider([Response(200, chat_reply(ANSWER))]) as fake:
        chat_model(fake.url, max_output_tokens=limit)(PROMPT, SPECS)
    body = fake.requests[0].json()
    assert body["max_completion_tokens"] == limit
    assert "max_tokens" not in body and "temperature" not in body


def test_raw_string_arguments_are_sent_back_unchanged():
    raw = '{"seconds": 600'
    conversation = [
        *PROMPT,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-9", "name": "predict_coast", "arguments": raw}],
        },
        {"role": "tool", "tool_call_id": "call-9", "name": "predict_coast", "content": "{}"},
    ]
    with FakeProvider([Response(200, chat_reply(ANSWER))]) as fake:
        chat_model(fake.url)(conversation, SPECS)
    sent = fake.requests[0].json()["messages"][1]["tool_calls"][0]
    assert sent["function"] == {"name": "predict_coast", "arguments": raw}


# Replies (criterion 7)


def test_a_text_reply_parses_with_its_token_counts():
    body = chat_reply(ANSWER, prompt_tokens=812, completion_tokens=45)
    with FakeProvider([Response(200, body)]) as fake:
        reply = chat_model(fake.url)(PROMPT, [])
    assert reply == Reply(ANSWER, 812, 45, ())


def test_tool_calls_parse_with_their_ids_and_arguments():
    calls = [
        ("call_a", "predict_coast", '{"seconds": 600}'),
        ("call_b", "transfer_burn", '{"x": 0, "y": -100, "z": 0, "seconds": 1500}'),
    ]
    body = chat_reply(tool_calls=calls, prompt_tokens=900, completion_tokens=60)
    with FakeProvider([Response(200, body)]) as fake:
        reply = chat_model(fake.url)(PROMPT, SPECS)
    assert reply.text == ""
    assert (reply.input_tokens, reply.output_tokens) == (900, 60)
    assert reply.tool_calls == (
        ToolCall("predict_coast", {"seconds": 600}, "call_a"),
        ToolCall("transfer_burn", {"x": 0, "y": -100, "z": 0, "seconds": 1500}, "call_b"),
    )


def test_malformed_arguments_reach_the_agent_and_come_back_as_an_error():
    raw = '{"seconds": 600'
    replies = [
        Response(200, chat_reply(tool_calls=[("call_a", "predict_coast", raw)])),
        Response(200, chat_reply(ANSWER)),
    ]
    environment = ApproachEnv(ApproachConfig(max_steps=1, start_state=(0, 400, 0, 0, 0, 0)))
    with FakeProvider(replies) as fake:
        agent = LLMAgent(chat_model(fake.url), tools=CW_TOOLS)
        run_episode(environment, agent, seed=0, agent_seed=1)
    (call,) = agent.log[0]["tool_calls"]
    assert call["arguments"] == raw and "error" in call["result"]
    second = fake.requests[1].json()["messages"]
    assert second[1]["tool_calls"][0]["function"]["arguments"] == raw
    assert second[2]["role"] == "tool" and "error" in json.loads(second[2]["content"])


# A whole episode (criterion 9)


def scripted_provider(request):
    """Calls a tool when tools are offered after the prompt, then answers; answers vary."""
    body = request.json()
    turn = len(body["messages"])
    if "tools" in body and body["messages"][-1]["role"] == "user":
        return Response(
            200, chat_reply(tool_calls=[(f"call-{turn}", "predict_coast", '{"seconds": 300}')])
        )
    content = body["messages"][0]["content"]
    ax = (len(content) % 7 - 3) / 10
    return Response(
        200, chat_reply(json.dumps({"ax": ax, "ay": -0.1, "az": 0.0}), prompt_tokens=len(content))
    )


def episode(model):
    environment = ApproachEnv(ApproachConfig(max_steps=25, sensor=GaussianNav(2.0, 2e-3)))
    agent = LLMAgent(model, tools=CW_TOOLS)
    return run_episode(environment, agent, seed=4, agent_seed=0), agent


def test_a_25_step_episode_with_tools_records_and_replays_exactly(tmp_path):
    with FakeProvider(scripted_provider) as fake:
        recorder = RecordedModel(chat_model(fake.url))
        recorded, first = episode(recorder)
    assert len(fake.requests) == 50
    assert all(len(e["tool_calls"]) == 1 for e in first.log)
    recorder.save(tmp_path / "transcript.json")
    assert FAKE_KEY not in (tmp_path / "transcript.json").read_text(encoding="utf-8")
    replayed, second = episode(ReplayModel.load(tmp_path / "transcript.json"))
    assert np.array_equal(recorded["states"], replayed["states"])
    assert np.array_equal(recorded["actions"], replayed["actions"])
    assert [e["messages"] for e in first.log] == [e["messages"] for e in second.log]
    assert first.usage() == second.usage()


# Errors (criteria 2 and 10)


def credit_429(type_, code):
    return 429, {}, chat_error("Quota reached.", type_, code), "credit"


# Credit is a 429 whose error type is insufficient_quota or whose code names a spent balance or a
# spend limit; any other 429, rate_limit_exceeded among them, is a rate limit and is retried.
ERROR_TABLE = [
    (401, {}, chat_error("Incorrect API key.", "invalid_request_error", "invalid_api_key"), "key"),
    credit_429("insufficient_quota", "insufficient_quota"),
    credit_429("insufficient_quota", None),
    credit_429("requests", "credit_balance_exhausted"),
    credit_429("requests", "organization_spend_limit_exceeded"),
    credit_429("requests", "project_spend_limit_exceeded"),
    credit_429("requests", "organization_usage_limit_exceeded"),
    (429, {"Retry-After": "1"}, chat_error("Limit.", "requests", "rate_limit_exceeded"), "rate"),
    (429, {"Retry-After": "1"}, chat_error("Slow down.", "requests", None), "rate"),
    (429, {}, "Too many requests", "rate"),
    # Assumed, not confirmed in the format's reference: a 404 means the model is unknown.
    (404, {}, chat_error("No such model.", "invalid_request_error", "model_not_found"), "model"),
    # Any other 4xx is a request the provider will never accept, so "try again" would mislead.
    (400, {}, chat_error("Bad request.", "invalid_request_error", None), "refused"),
    (
        400,
        {},
        chat_error("Unknown parameter.", "invalid_request_error", "unknown_parameter"),
        "refused",
    ),
    (400, {}, "Bad request", "refused"),
    (403, {}, chat_error("Not allowed.", "permission_error", None), "refused"),
    (403, {}, "Forbidden", "refused"),
    (405, {}, "Method not allowed", "refused"),
    (413, {}, chat_error("Request too large.", "invalid_request_error", None), "refused"),
    (422, {}, chat_error("Unprocessable.", "invalid_request_error", None), "refused"),
    (500, {}, chat_error("Internal error.", "server_error", None), "server"),
    (502, {}, "Bad gateway", "server"),
    (503, {}, chat_error("Overloaded.", "server_error", None), "server"),
]


@pytest.mark.parametrize(("status", "headers", "body", "kind"), ERROR_TABLE)
def test_error_responses_map_to_a_kind_and_its_sentence(status, headers, body, kind):
    with FakeProvider([Response(status, body, headers)]) as fake:
        with pytest.raises(ProviderError) as caught:
            chat_model(fake.url)(PROMPT, [])
    error = caught.value
    assert error.kind == kind
    assert str(error) == sentence(kind, status)
    assert isinstance(error, InfrastructureError) == (kind in INFRASTRUCTURE)


def test_the_messages_table_has_a_sentence_for_every_kind():
    kinds = {"key", "credit", "rate", "network", "model", "server", "tls", "cap", "refused"}
    assert set(MESSAGES) == kinds
    assert all(MESSAGES[k].endswith(".") for k in kinds)
    assert "{m}" in MESSAGES["model"]


@pytest.mark.parametrize("body", [{}, {"choices": []}])
def test_a_success_answer_without_a_choice_is_a_server_error(body):
    with FakeProvider([Response(200, body)]) as fake:
        with pytest.raises(ProviderError) as caught:
            chat_model(fake.url)(PROMPT, [])
    assert caught.value.kind == "server" and str(caught.value) == sentence("server")


def test_the_cap_is_not_a_provider_error_kind():
    # The cap is reported by BudgetExceeded, not by a provider failure.
    with pytest.raises(ValueError):
        ProviderError("cap")


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
def test_errors_never_carry_the_key_or_the_providers_text(status):
    echoed = f"Incorrect API key provided: {FAKE_KEY}. Visit the dashboard."
    with FakeProvider(
        [Response(status, chat_error(echoed, "invalid_request_error", None))]
    ) as fake:
        with pytest.raises(ProviderError) as caught:
            chat_model(fake.url)(PROMPT, [])
    for text in (str(caught.value), repr(caught.value), str(caught.value.args)):
        assert FAKE_KEY not in text and "Visit the dashboard" not in text


def test_the_model_repr_and_debug_logs_omit_the_key(caplog):
    caplog.set_level(logging.DEBUG)
    model = chat_model("http://127.0.0.1:1")
    assert FAKE_KEY not in repr(model) and FAKE_KEY not in str(model)
    with FakeProvider([Response(401, chat_error(FAKE_KEY, "invalid_request_error", None))]) as fake:
        with pytest.raises(ProviderError):
            chat_model(fake.url)(PROMPT, [])
    with FakeProvider([Response(200, chat_reply(ANSWER))]) as fake:
        chat_model(fake.url)(PROMPT, [])
    assert all(FAKE_KEY not in r.getMessage() for r in caplog.records)


def test_a_refused_connection_is_a_network_error():
    sleep = Sleeps()
    with pytest.raises(ProviderError) as caught:
        chat_model(closed_port_url(), sleep=sleep)(PROMPT, [])
    assert caught.value.kind == "network" and str(caught.value) == sentence("network")
    assert isinstance(caught.value, InfrastructureError)


def test_a_timeout_is_a_network_error_and_is_not_retried():
    with FakeProvider([Response(200, chat_reply(ANSWER), delay_s=2.0)]) as fake:
        with pytest.raises(ProviderError) as caught:
            chat_model(fake.url, timeout=0.3)(PROMPT, [])
    assert caught.value.kind == "network" and isinstance(caught.value, InfrastructureError)
    assert len(fake.requests) == 1


def test_an_unverified_certificate_is_a_tls_error(monkeypatch):
    def refuse(self):
        raise ssl.SSLCertVerificationError(1, "certificate verify failed")

    monkeypatch.setattr("http.client.HTTPSConnection.connect", refuse)
    with pytest.raises(ProviderError) as caught:
        ChatModel(
            base_url="https://127.0.0.1:9/v1",
            model=MODEL,
            key=FAKE_KEY,
            max_output_tokens=64,
            sleep=Sleeps(),
        )(PROMPT, [])
    assert caught.value.kind == "tls" and str(caught.value) == sentence("tls")
    assert not isinstance(caught.value, InfrastructureError)


# Redirects and plain HTTP (criterion 11)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_refused_and_the_second_host_gets_nothing(status):
    with FakeProvider([Response(200, chat_reply(ANSWER))]) as second:
        location = {"Location": second.url + "/v1/chat/completions"}
        with FakeProvider([Response(status, None, location)]) as first:
            with pytest.raises(ProviderError):
                chat_model(first.url)(PROMPT, [])
            with pytest.raises(ProviderError):
                post_json(
                    url=first.url + "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {FAKE_KEY}"},
                    body={"model": MODEL},
                    timeout=5.0,
                )
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


@pytest.mark.parametrize("url", ["http://10.0.0.5:8080/v1", "http://192.168.1.20/v1"])
def test_plain_http_to_another_host_is_refused_before_connecting(url, no_connections):
    with pytest.raises((ValueError, ProviderError)):
        ChatModel(base_url=url, model=MODEL, key=FAKE_KEY, max_output_tokens=64, sleep=Sleeps())(
            PROMPT, []
        )
    with pytest.raises((ValueError, ProviderError)):
        post_json(
            url=url + "/chat/completions",
            headers={"Authorization": f"Bearer {FAKE_KEY}"},
            body={"model": MODEL},
            timeout=5.0,
        )
    assert no_connections == []


@pytest.mark.parametrize(
    "key",
    [
        f"{FAKE_KEY}\r",
        f"{FAKE_KEY}\nX-Injected: 1",
        f"{FAKE_KEY}\x00",
        f"{FAKE_KEY}\x1b",
        f"\x7f{FAKE_KEY}",
    ],
)
def test_a_key_with_a_control_character_is_refused_before_connecting(key, no_connections):
    # http.client would refuse such a header itself, with an error that quotes the key.
    with pytest.raises((ValueError, ProviderError)) as caught:
        ChatModel(
            base_url="http://127.0.0.1:9/v1",
            model=MODEL,
            key=key,
            max_output_tokens=64,
            sleep=Sleeps(),
        )(PROMPT, [])
    for text in (str(caught.value), repr(caught.value), str(caught.value.args)):
        assert FAKE_KEY not in text
    assert no_connections == []


@pytest.mark.parametrize("variable", ["HTTP_PROXY", "http_proxy"])
def test_plain_http_to_loopback_ignores_environment_proxies(variable, monkeypatch):
    with FakeProvider([Response(200, chat_reply(ANSWER))]) as proxy:
        for name in ("HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(variable, proxy.url)
        with FakeProvider([Response(200, chat_reply(ANSWER))]) as fake:
            assert chat_model(fake.url)(PROMPT, []).text == ANSWER
            post_json(
                url=fake.url + "/v1/chat/completions",
                headers={"Authorization": f"Bearer {FAKE_KEY}"},
                body={"model": MODEL},
                timeout=5.0,
            )
    assert len(fake.requests) == 2
    assert proxy.requests == []


def test_plain_http_to_localhost_is_allowed():
    with FakeProvider([Response(200, chat_reply(ANSWER))]) as fake:
        local = fake.url.replace("127.0.0.1", "localhost")
        assert chat_model(local)(PROMPT, []).text == ANSWER


# Retries (criterion 12)

RATE = Response(
    429, chat_error("Rate limit reached.", "requests", "rate_limit_exceeded"), {"Retry-After": "3"}
)
CREDIT = Response(
    429, chat_error("Quota.", "insufficient_quota", "insufficient_quota"), {"Retry-After": "3"}
)


def test_a_rate_limit_is_retried_twice_after_the_given_delay():
    sleep = Sleeps()
    with FakeProvider([RATE]) as fake:
        with pytest.raises(ProviderError) as caught:
            chat_model(fake.url, sleep=sleep)(PROMPT, [])
    assert caught.value.kind == "rate"
    assert len(fake.requests) == 3
    assert sleep.delays == [3.0, 3.0]


def test_a_rate_limit_that_clears_returns_the_reply():
    sleep = Sleeps()
    with FakeProvider([RATE, Response(200, chat_reply(ANSWER))]) as fake:
        reply = chat_model(fake.url, sleep=sleep)(PROMPT, [])
    assert reply.text == ANSWER and len(fake.requests) == 2 and sleep.delays == [3.0]


def test_retry_waits_are_capped_at_30_seconds():
    sleep = Sleeps()
    slow = Response(429, RATE.body, {"Retry-After": "120"})
    with FakeProvider([slow]) as fake:
        with pytest.raises(ProviderError):
            chat_model(fake.url, sleep=sleep)(PROMPT, [])
    assert len(fake.requests) <= 3
    assert all(0 <= s <= 30 for s in sleep.delays)


def test_a_rate_limit_without_retry_after_waits_a_bounded_time():
    sleep = Sleeps()
    with FakeProvider([Response(429, RATE.body)]) as fake:
        with pytest.raises(ProviderError):
            chat_model(fake.url, sleep=sleep)(PROMPT, [])
    assert len(fake.requests) == 3 and len(sleep.delays) == 2
    assert all(0 <= s <= 30 for s in sleep.delays)


@pytest.mark.parametrize("status", [500, 503])
def test_server_errors_are_retried_at_most_twice(status):
    sleep = Sleeps()
    with FakeProvider([Response(status, chat_error("Internal.", "server_error", None))]) as fake:
        with pytest.raises(ProviderError) as caught:
            chat_model(fake.url, sleep=sleep)(PROMPT, [])
    assert caught.value.kind == "server" and len(fake.requests) == 3 and len(sleep.delays) == 2


@pytest.mark.parametrize(
    "response",
    [
        CREDIT,
        Response(401, chat_error("Bad key.", "invalid_request_error", "invalid_api_key")),
        Response(404, chat_error("No model.", "invalid_request_error", "model_not_found")),
        Response(400, chat_error("Bad request.", "invalid_request_error", None)),
        Response(403, chat_error("Not allowed.", "permission_error", None)),
    ],
)
def test_credit_key_model_and_refused_errors_are_not_retried(response):
    sleep = Sleeps()
    with FakeProvider([response]) as fake:
        with pytest.raises(ProviderError):
            chat_model(fake.url, sleep=sleep)(PROMPT, [])
    assert len(fake.requests) == 1 and sleep.delays == []


# HTTP layer: credentials in the address and the size of a reply

SECRET = "pw-5d1c8e2f9a7b"
USERINFO_URLS = [
    f"https://user:{SECRET}@api.example.com/v1",
    f"https://{SECRET}@api.example.com/v1",
    f"http://user:{SECRET}@127.0.0.1:9/v1",
    f"http://user:{SECRET}@10.0.0.5:8080/v1",
]


def exception_texts(error):
    """str, repr and args of an exception and of every exception chained to it."""
    texts = []
    while error is not None:
        texts += [str(error), repr(error), str(error.args)]
        error = error.__cause__ or error.__context__
    return texts


@pytest.mark.parametrize("url", USERINFO_URLS)
def test_a_base_url_with_credentials_is_refused_without_quoting_them(url, no_connections, caplog):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ValueError) as caught:
        ChatModel(base_url=url, model=MODEL, key=FAKE_KEY, max_output_tokens=64, sleep=Sleeps())(
            PROMPT, []
        )
    for text in exception_texts(caught.value):
        assert SECRET not in text and "user:" not in text
    assert all(SECRET not in r.getMessage() for r in caplog.records)
    assert no_connections == []


@pytest.mark.parametrize("url", USERINFO_URLS)
def test_post_json_and_check_url_refuse_credentials_without_quoting_them(
    url, no_connections, caplog
):
    caplog.set_level(logging.DEBUG)
    full = url + "/chat/completions"
    with pytest.raises(ValueError) as checked:
        check_url(full)
    with pytest.raises(ValueError) as posted:
        post_json(
            url=full,
            headers={"Authorization": f"Bearer {FAKE_KEY}"},
            body={"model": MODEL},
            timeout=5.0,
        )
    for text in exception_texts(checked.value) + exception_texts(posted.value):
        assert SECRET not in text and "user:" not in text
    assert all(SECRET not in r.getMessage() for r in caplog.records)
    assert no_connections == []


MiB = 1024 * 1024
LIMIT = 8 * MiB


def padded_reply(total):
    """A valid chat reply of exactly `total` bytes, its text padded with "a"."""
    body = chat_reply(PAD)
    return body, total - (len(json.dumps(body).encode("utf-8")) - len(PAD))


def padded_error(total):
    body = chat_error(PAD, "invalid_request_error", "invalid_api_key")
    return body, total - (len(json.dumps(body).encode("utf-8")) - len(PAD))


@pytest.mark.parametrize("length", [True, False])
@pytest.mark.parametrize(("status", "padded"), [(200, padded_reply), (401, padded_error)])
def test_a_reply_one_byte_over_8_mib_is_a_server_error(status, padded, length):
    body, pad = padded(LIMIT + 1)
    with FakeProvider([Response(status, body, pad=pad, length=length)]) as fake:
        with pytest.raises(ProviderError) as caught:
            chat_model(fake.url)(PROMPT, [])
    assert caught.value.kind == "server"


@pytest.mark.parametrize("length", [True, False])
@pytest.mark.parametrize(("status", "padded"), [(200, padded_reply), (401, padded_error)])
def test_a_reply_far_over_8_mib_is_not_read_past_the_limit(status, padded, length):
    # 64 MiB of valid JSON: a client that reads it all accepts it; socket buffers on loopback
    # hold far less than the 24 MiB margin, so a client that stops near 8 MiB stays under it.
    body, pad = padded(64 * MiB)
    with FakeProvider([Response(status, body, pad=pad, length=length)]) as fake:
        with pytest.raises(ProviderError) as caught:
            chat_model(fake.url)(PROMPT, [])
    assert caught.value.kind == "server"
    assert fake.sent and max(fake.sent) < LIMIT + 24 * MiB


def test_a_reply_of_exactly_8_mib_is_accepted():
    body, pad = padded_reply(LIMIT)
    with FakeProvider([Response(200, body, pad=pad)]) as fake:
        reply = chat_model(fake.url)(PROMPT, [])
    assert len(reply.text) == pad and set(reply.text) == {"a"}


# Input checks: the key's characters and the reply's content field

NON_ASCII_KEYS = [
    f"{FAKE_KEY[:8]}é{FAKE_KEY[8:]}",
    f"{FAKE_KEY[:8]}\u200b{FAKE_KEY[8:]}",
    f"{FAKE_KEY[:8]}\u00a0{FAKE_KEY[8:]}",
    f"{FAKE_KEY[:8]}а{FAKE_KEY[8:]}",
]


@pytest.mark.parametrize("key", NON_ASCII_KEYS)
def test_a_key_with_a_non_ascii_character_is_refused_before_connecting(key, no_connections):
    with pytest.raises(ValueError) as caught:
        ChatModel(
            base_url="http://127.0.0.1:9/v1",
            model=MODEL,
            key=key,
            max_output_tokens=64,
            sleep=Sleeps(),
        )(PROMPT, [])
    for text in exception_texts(caught.value):
        assert key not in text and FAKE_KEY[8:] not in text
    assert no_connections == []


@pytest.mark.parametrize(
    "content", [[{"type": "text", "text": ANSWER}], 42, 0, True, {"text": ANSWER}]
)
def test_a_reply_content_that_is_not_text_or_null_is_a_server_error(content):
    body = chat_reply(ANSWER)
    body["choices"][0]["message"]["content"] = content
    with FakeProvider([Response(200, body)]) as fake:
        with pytest.raises(ProviderError) as caught:
            chat_model(fake.url)(PROMPT, [])
    assert caught.value.kind == "server" and str(caught.value) == sentence("server")


def test_a_null_reply_content_is_empty_text():
    body = chat_reply("", tool_calls=[("call-1", "predict_coast", '{"seconds": 60}')])
    assert body["choices"][0]["message"]["content"] is None
    with FakeProvider([Response(200, body)]) as fake:
        assert chat_model(fake.url)(PROMPT, []).text == ""
