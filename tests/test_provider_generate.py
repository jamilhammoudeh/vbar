"""The generateContent adapter against a fake provider on 127.0.0.1."""

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
    generate_error,
    generate_reply,
)

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import InfrastructureError, LLMAgent, RecordedModel, ReplayModel, Reply, ToolCall
from vbar.agents.tools import CW_TOOLS
from vbar.providers import MESSAGES, ProviderError
from vbar.providers.generate import GenerateModel
from vbar.spacecraft import GaussianNav

MODEL = "test-model"
PATH = "/v1beta/models/test-model:generateContent"
# A neutral stand-in for the key header the catalogue supplies.
KEY_HEADER = "x-api-key-header"
GOLDEN = Path(__file__).parent / "data" / "generate_request.json"
SPECS = [tool.spec() for tool in CW_TOOLS]
PROMPT = [{"role": "user", "content": "Hold at the V-bar point."}]
CALL_1 = {"seconds": 600.0}
CALL_2 = {"x": 0.0, "y": -100.0, "z": 0.0, "seconds": 1500.0}
SIGNATURE = "c2lnbmF0dXJlLTE="


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
FIRST_REPLY = generate_reply(
    calls=[("call-1", "predict_coast", CALL_1), ("call-2", "transfer_burn", CALL_2)],
    signature=SIGNATURE,
)
ANSWER = '{"ax": 0.0, "ay": 0.0, "az": 0.0}'
INFRASTRUCTURE = {"rate", "network", "server"}


def generate_model(url, *, sleep=None, max_output_tokens=2048, key=FAKE_KEY, **fields):
    return GenerateModel(
        base_url=url,
        model=MODEL,
        key=key,
        max_output_tokens=max_output_tokens,
        auth_header=KEY_HEADER,
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


def test_the_request_with_echoed_signatures_and_function_responses_equals_the_golden_body():
    # UNVERIFIED: whether the format needs the signature echoed inside the functionCall part is
    # open (G3 and the spec's status-mapping note); the golden body assumes it does.
    with FakeProvider([Response(200, FIRST_REPLY), Response(200, generate_reply(ANSWER))]) as fake:
        model = generate_model(fake.url)
        model(PROMPT, SPECS)
        model(CONVERSATION, SPECS)
    request = fake.requests[1]
    assert request.method == "POST" and request.path == PATH
    assert request.json() == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_key_goes_in_its_header_and_never_in_the_address_or_body():
    with FakeProvider([Response(200, generate_reply(ANSWER))]) as fake:
        generate_model(fake.url)(PROMPT, [])
    request = fake.requests[0]
    headers = lower(request.headers)
    assert headers[KEY_HEADER] == FAKE_KEY
    assert "authorization" not in headers
    assert request.path == PATH and FAKE_KEY not in request.path
    assert FAKE_KEY not in request.raw.decode("utf-8")


def test_a_call_without_tools_sends_no_tools_field():
    with FakeProvider([Response(200, generate_reply(ANSWER))]) as fake:
        generate_model(fake.url)(PROMPT, [])
    assert "tools" not in fake.requests[0].json()


@pytest.mark.parametrize("limit", [64, 2048, 32000])
def test_the_reply_limit_sent_is_max_output_tokens(limit):
    with FakeProvider([Response(200, generate_reply(ANSWER))]) as fake:
        generate_model(fake.url, max_output_tokens=limit)(PROMPT, SPECS)
    config = fake.requests[0].json()["generationConfig"]
    assert config["maxOutputTokens"] == limit
    assert "temperature" not in config


def test_signatures_are_sent_back_with_the_calls_they_came_with():
    # UNVERIFIED echo of thought signatures inside functionCall parts (spec section 2).
    first = generate_reply(calls=[("call-a", "predict_coast", CALL_1)], signature="sig-a")
    second = generate_reply(calls=[("call-b", "predict_coast", CALL_1)], signature="sig-b")
    replies = [Response(200, first), Response(200, second), Response(200, generate_reply(ANSWER))]
    with FakeProvider(replies) as fake:
        model = generate_model(fake.url)
        model(PROMPT, SPECS)
        model([{"role": "user", "content": "Another step."}], SPECS)
        model(
            [*PROMPT, assistant(("call-a", "predict_coast", CALL_1)), tool("call-a", "x", "{}")],
            SPECS,
        )
    (part,) = fake.requests[2].json()["contents"][1]["parts"]
    assert part["thoughtSignature"] == "sig-a"
    assert part["functionCall"]["name"] == "predict_coast"
    assert part["functionCall"]["args"] == CALL_1


def test_calls_without_ids_get_distinct_ids_and_go_back_unchanged():
    body = generate_reply(
        calls=[(None, "predict_coast", CALL_1), (None, "predict_coast", {"seconds": 60.0})],
        signature="sig-n",
    )
    with FakeProvider([Response(200, body), Response(200, generate_reply(ANSWER))]) as fake:
        model = generate_model(fake.url)
        reply = model(PROMPT, SPECS)
        ids = [c.id for c in reply.tool_calls]
        assert all(ids) and len(set(ids)) == 2
        model(
            [
                *PROMPT,
                assistant(*[(c.id, c.name, c.arguments) for c in reply.tool_calls]),
                *[tool(c.id, c.name, "{}") for c in reply.tool_calls],
            ],
            SPECS,
        )
    contents = fake.requests[1].json()["contents"]
    assert contents[1]["parts"] == body["candidates"][0]["content"]["parts"]
    responses = [p["functionResponse"] for p in contents[2]["parts"]]
    assert [r["name"] for r in responses] == ["predict_coast", "predict_coast"]


def test_a_new_decision_sends_only_its_prompt():
    with FakeProvider([Response(200, FIRST_REPLY)]) as fake:
        model = generate_model(fake.url)
        model(PROMPT, SPECS)
        model([{"role": "user", "content": "Next step."}], SPECS)
    assert fake.requests[1].json()["contents"] == [
        {"role": "user", "parts": [{"text": "Next step."}]}
    ]


# Replies (criterion 7)


def test_output_tokens_count_thoughts_and_input_counts_tool_use_prompts():
    body = generate_reply(
        ANSWER,
        prompt_tokens=812,
        candidates_tokens=40,
        thoughts_tokens=300,
        tool_use_prompt_tokens=25,
    )
    with FakeProvider([Response(200, body)]) as fake:
        reply = generate_model(fake.url)(PROMPT, [])
    assert reply == Reply(ANSWER, 837, 340, ())


def test_missing_token_fields_count_as_zero():
    body = generate_reply(ANSWER, prompt_tokens=812, candidates_tokens=40)
    del body["usageMetadata"]["thoughtsTokenCount"]
    with FakeProvider([Response(200, body)]) as fake:
        reply = generate_model(fake.url)(PROMPT, [])
    assert (reply.input_tokens, reply.output_tokens) == (812, 40)


def test_tool_calls_parse_and_thought_parts_are_not_in_the_text():
    with FakeProvider([Response(200, generate_reply(ANSWER, thought_text="hmm"))]) as fake:
        assert generate_model(fake.url)(PROMPT, SPECS).text == ANSWER
    body = generate_reply(
        calls=[("call_a", "predict_coast", {"seconds": 600})],
        prompt_tokens=900,
        candidates_tokens=50,
        thoughts_tokens=10,
    )
    with FakeProvider([Response(200, body)]) as fake:
        reply = generate_model(fake.url)(PROMPT, SPECS)
    assert reply == Reply("", 900, 60, (ToolCall("predict_coast", {"seconds": 600}, "call_a"),))


@pytest.mark.parametrize("body", [{}, {"candidates": []}, {"candidates": [{}]}])
def test_a_success_answer_without_a_candidate_is_a_server_error(body):
    with FakeProvider([Response(200, body)]) as fake:
        with pytest.raises(ProviderError) as caught:
            generate_model(fake.url)(PROMPT, [])
    assert caught.value.kind == "server" and str(caught.value) == sentence("server")


# A whole episode (criterion 9) and what the agent's requests carry (criterion 17)


def scripted_provider():
    """Calls a tool with a fresh signature after each prompt, then answers; answers vary."""
    counter = itertools.count()
    issued = []

    def respond(request):
        body = request.json()
        if "tools" in body and len(body["contents"]) == 1:
            n = next(counter)
            issued.append(f"sig-{n}")
            calls = [(f"fc-{n}", "predict_coast", {"seconds": 300})]
            return Response(200, generate_reply(calls=calls, signature=f"sig-{n}"))
        text = body["contents"][0]["parts"][0]["text"]
        ax = (len(text) % 7 - 3) / 10
        answer = json.dumps({"ax": ax, "ay": -0.1, "az": 0.0})
        return Response(200, generate_reply(answer, prompt_tokens=len(text)))

    return respond, issued


def episode(model, steps=25):
    environment = ApproachEnv(ApproachConfig(max_steps=steps, sensor=GaussianNav(2.0, 2e-3)))
    agent = LLMAgent(model, tools=CW_TOOLS)
    return run_episode(environment, agent, seed=4, agent_seed=0), agent


def test_a_25_step_episode_with_tools_records_and_replays_exactly(tmp_path):
    respond, issued = scripted_provider()
    with FakeProvider(respond) as fake:
        recorder = RecordedModel(generate_model(fake.url))
        recorded, first = episode(recorder)
    assert len(fake.requests) == 50
    assert all(len(e["tool_calls"]) == 1 for e in first.log)
    for signature, request in zip(issued, fake.requests[1::2], strict=True):
        assert request.json()["contents"][1]["parts"][0]["thoughtSignature"] == signature
    recorder.save(tmp_path / "transcript.json")
    assert FAKE_KEY not in (tmp_path / "transcript.json").read_text(encoding="utf-8")
    replayed, second = episode(ReplayModel.load(tmp_path / "transcript.json"))
    assert np.array_equal(recorded["states"], replayed["states"])
    assert np.array_equal(recorded["actions"], replayed["actions"])
    assert [e["messages"] for e in first.log] == [e["messages"] for e in second.log]
    assert first.usage() == second.usage()


def user_values(content):
    """What a sent user turn carries: its texts and its function responses."""
    return [
        p["text"] if "text" in p else p["functionResponse"]["response"] for p in content["parts"]
    ]


def test_requests_carry_only_the_logged_prompts_and_results_never_the_true_position():
    respond, _ = scripted_provider()
    with FakeProvider(respond) as fake:
        result, agent = episode(generate_model(fake.url), steps=6)
    assert len(fake.requests) == 2 * len(agent.log)
    for k, entry in enumerate(agent.log):
        logged = entry["messages"]
        for request, held in zip(
            fake.requests[2 * k : 2 * k + 2], (logged[:1], logged), strict=True
        ):
            body = request.json()
            assert "systemInstruction" not in body
            sent = [v for c in body["contents"] if c["role"] == "user" for v in user_values(c)]
            expected = [
                m["content"] if m["role"] == "user" else json.loads(m["content"])
                for m in held
                if m["role"] in ("user", "tool")
            ]
            assert sent == expected
        assert body["contents"][0]["parts"][0]["text"] == entry["prompt"]
    raw = b"".join(r.raw for r in fake.requests).decode("utf-8")
    for x, y, z in result["states"][:, :3]:
        assert f"[{x:.2f}, {y:.2f}, {z:.2f}]" not in raw


# Errors (criteria 2 and 10)

# Assumed mapping: the format's documented status codes are for another API and G3 is open.
# A 400 whose reason is API_KEY_INVALID is a bad key; other 4xx except 404 and 429 are refused.
BAD_KEY = generate_error(
    400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT", "API_KEY_INVALID"
)
ERROR_TABLE = [
    (400, {}, BAD_KEY, "key"),
    # Assumed (G3): a 401 also means the key was refused.
    (401, {}, generate_error(401, "Request had invalid credentials.", "UNAUTHENTICATED"), "key"),
    (429, {}, generate_error(429, "Resource exhausted.", "RESOURCE_EXHAUSTED"), "rate"),
    (429, {"Retry-After": "1"}, "Too many requests", "rate"),
    (404, {}, generate_error(404, "models/test-model is not found.", "NOT_FOUND"), "model"),
    (400, {}, generate_error(400, "Invalid argument.", "INVALID_ARGUMENT"), "refused"),
    (400, {}, "Bad request", "refused"),
    (403, {}, generate_error(403, "Permission denied.", "PERMISSION_DENIED"), "refused"),
    (500, {}, generate_error(500, "Internal error.", "INTERNAL"), "server"),
    (503, {}, generate_error(503, "Overloaded.", "UNAVAILABLE"), "server"),
    (504, {}, "Gateway timeout", "server"),
]


@pytest.mark.parametrize(("status", "headers", "body", "kind"), ERROR_TABLE)
def test_error_responses_map_to_a_kind_and_its_sentence(status, headers, body, kind):
    with FakeProvider([Response(status, body, headers)]) as fake:
        with pytest.raises(ProviderError) as caught:
            generate_model(fake.url)(PROMPT, [])
    error = caught.value
    assert error.kind == kind
    assert str(error) == sentence(kind, status)
    assert isinstance(error, InfrastructureError) == (kind in INFRASTRUCTURE)


@pytest.mark.parametrize("status", [400, 403, 404, 429, 500])
def test_errors_never_carry_the_key_or_the_providers_text(status):
    echoed = f"API key {FAKE_KEY} not valid. Visit the console."
    with FakeProvider([Response(status, generate_error(status, echoed, "X"))]) as fake:
        with pytest.raises(ProviderError) as caught:
            generate_model(fake.url)(PROMPT, [])
    for text in (str(caught.value), repr(caught.value), str(caught.value.args)):
        assert FAKE_KEY not in text and "Visit the console" not in text


def test_the_model_repr_and_debug_logs_omit_the_key(caplog):
    caplog.set_level(logging.DEBUG)
    model = generate_model("http://127.0.0.1:1")
    assert FAKE_KEY not in repr(model) and FAKE_KEY not in str(model)
    with FakeProvider([Response(400, BAD_KEY)]) as fake:
        with pytest.raises(ProviderError):
            generate_model(fake.url)(PROMPT, [])
    with FakeProvider([Response(200, generate_reply(ANSWER))]) as fake:
        generate_model(fake.url)(PROMPT, [])
    assert all(FAKE_KEY not in r.getMessage() for r in caplog.records)


def test_a_refused_connection_and_a_timeout_are_network_errors():
    with pytest.raises(ProviderError) as caught:
        generate_model(closed_port_url())(PROMPT, [])
    assert caught.value.kind == "network" and isinstance(caught.value, InfrastructureError)
    with FakeProvider([Response(200, generate_reply(ANSWER), delay_s=2.0)]) as fake:
        with pytest.raises(ProviderError) as caught:
            generate_model(fake.url, timeout=0.3)(PROMPT, [])
    assert caught.value.kind == "network" and len(fake.requests) == 1


def test_an_unverified_certificate_is_a_tls_error(monkeypatch):
    def refuse(self):
        raise ssl.SSLCertVerificationError(1, "certificate verify failed")

    monkeypatch.setattr("http.client.HTTPSConnection.connect", refuse)
    with pytest.raises(ProviderError) as caught:
        generate_model("https://127.0.0.1:9")(PROMPT, [])
    assert caught.value.kind == "tls" and not isinstance(caught.value, InfrastructureError)


# Redirects, plain HTTP, credentials in the address and the key's characters (criterion 11)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_refused_and_the_second_host_gets_nothing(status):
    with FakeProvider([Response(200, generate_reply(ANSWER))]) as second:
        location = {"Location": second.url + PATH}
        with FakeProvider([Response(status, None, location)]) as first:
            with pytest.raises(ProviderError) as caught:
                generate_model(first.url)(PROMPT, [])
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
        generate_model(url)(PROMPT, [])
    assert no_connections == []


SECRET = "pw-5d1c8e2f9a7b"


@pytest.mark.parametrize(
    "url", [f"https://user:{SECRET}@api.example.com", f"http://user:{SECRET}@127.0.0.1:9"]
)
def test_a_base_url_with_credentials_is_refused_without_quoting_them(url, no_connections):
    with pytest.raises(ValueError) as caught:
        generate_model(url)(PROMPT, [])
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
        generate_model("http://127.0.0.1:9", key=key)(PROMPT, [])
    for text in (str(caught.value), repr(caught.value), str(caught.value.args)):
        assert FAKE_KEY[8:] not in text
    assert no_connections == []


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_a_reply_limit_that_is_not_a_positive_integer_is_refused(limit):
    with pytest.raises(ValueError):
        generate_model("http://127.0.0.1:9", max_output_tokens=limit)


@pytest.mark.parametrize("name", ["../other", "a/b", "m?x=1", "m#f"])
def test_a_model_name_that_would_change_the_path_is_refused(name):
    with pytest.raises(ValueError):
        GenerateModel(
            base_url="http://127.0.0.1:9",
            model=name,
            key=FAKE_KEY,
            max_output_tokens=64,
            auth_header=KEY_HEADER,
            sleep=Sleeps(),
        )


# Retries (criterion 12)


def test_a_rate_limit_is_retried_twice_after_the_given_delay():
    sleep = Sleeps()
    rate = Response(
        429, generate_error(429, "Resource exhausted.", "RESOURCE_EXHAUSTED"), {"Retry-After": "3"}
    )
    with FakeProvider([rate]) as fake:
        with pytest.raises(ProviderError) as caught:
            generate_model(fake.url, sleep=sleep)(PROMPT, [])
    assert caught.value.kind == "rate"
    assert len(fake.requests) == 3 and sleep.delays == [3.0, 3.0]


def test_a_server_error_is_retried_and_can_clear():
    sleep = Sleeps()
    unavailable = Response(503, generate_error(503, "Overloaded.", "UNAVAILABLE"))
    with FakeProvider([unavailable, Response(200, generate_reply(ANSWER))]) as fake:
        reply = generate_model(fake.url, sleep=sleep)(PROMPT, [])
    assert reply.text == ANSWER and len(fake.requests) == 2 and len(sleep.delays) == 1


@pytest.mark.parametrize(
    "response",
    [
        Response(400, BAD_KEY),
        Response(401, generate_error(401, "Request had invalid credentials.", "UNAUTHENTICATED")),
        Response(404, generate_error(404, "models/test-model is not found.", "NOT_FOUND")),
        Response(403, generate_error(403, "Permission denied.", "PERMISSION_DENIED")),
    ],
)
def test_key_model_and_refused_errors_are_not_retried(response):
    sleep = Sleeps()
    with FakeProvider([response]) as fake:
        with pytest.raises(ProviderError):
            generate_model(fake.url, sleep=sleep)(PROMPT, [])
    assert len(fake.requests) == 1 and sleep.delays == []


# Size of a reply

LIMIT = 8 * 1024 * 1024


def padded_reply(total):
    body = generate_reply(PAD)
    return body, total - (len(json.dumps(body).encode("utf-8")) - len(PAD))


def test_a_reply_one_byte_over_8_mib_is_a_server_error():
    body, pad = padded_reply(LIMIT + 1)
    with FakeProvider([Response(200, body, pad=pad)]) as fake:
        with pytest.raises(ProviderError) as caught:
            generate_model(fake.url)(PROMPT, [])
    assert caught.value.kind == "server"


def test_a_reply_of_exactly_8_mib_is_accepted():
    body, pad = padded_reply(LIMIT)
    with FakeProvider([Response(200, body, pad=pad)]) as fake:
        reply = generate_model(fake.url)(PROMPT, [])
    assert len(reply.text) == pad and set(reply.text) == {"a"}


def test_the_key_header_name_is_required():
    with pytest.raises(TypeError):
        GenerateModel(
            base_url="http://127.0.0.1:9",
            model=MODEL,
            key=FAKE_KEY,
            max_output_tokens=64,
            sleep=Sleeps(),
        )


def test_a_function_call_without_args_is_a_call_with_no_arguments():
    # The format marks `args` optional, so a call to a tool without parameters may omit it.
    body = generate_reply(calls=[("call-0", "predict_coast", {})])
    del body["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]
    with FakeProvider([Response(200, body)]) as fake:
        reply = generate_model(fake.url)(PROMPT, SPECS)
    assert reply.tool_calls == (ToolCall("predict_coast", {}, "call-0"),)
