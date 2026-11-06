"""The provider catalogue, and `connect`, which turns an entry, model and key into a model."""

import datetime
import json
import math
from collections.abc import Mapping
from urllib.parse import urlsplit

import pytest
from provider_fakes import FAKE_KEY, FakeProvider, Response, chat_reply

from vbar.agents.llm import Price, Reply
from vbar.providers import Provider, connect
from vbar.providers.catalog import PROVIDERS, ListedModel
from vbar.providers.http import check_url

WIRES = {"chat", "messages", "generate"}
ENTRIES = list(PROVIDERS.values())
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def test_the_catalogue_is_keyed_by_each_entry_key():
    assert ENTRIES
    for key, provider in PROVIDERS.items():
        assert isinstance(provider, Provider)
        assert provider.key == key


def test_any_chat_address_is_the_last_menu_entry():
    assert list(PROVIDERS)[-1] == "other"


def test_a_local_chat_server_is_listed():
    assert "local" in PROVIDERS


@pytest.mark.parametrize("provider", ENTRIES, ids=lambda p: p.key)
def test_every_entry_has_the_documented_shape(provider):
    assert provider.label.strip()
    assert provider.wire in WIRES
    assert provider.auth_header is None or provider.auth_header.strip()
    assert isinstance(provider.extra_headers, Mapping)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in provider.extra_headers.items())
    assert isinstance(provider.extra_body, Mapping)
    if provider.base_url is not None:
        check_url(provider.base_url)


@pytest.mark.parametrize("provider", ENTRIES, ids=lambda p: p.key)
def test_every_listed_model_has_a_dated_price_and_its_source(provider):
    names = [m.name for m in provider.models]
    assert len(names) == len(set(names))
    for listed in provider.models:
        assert listed.name.strip()
        assert isinstance(listed.price, Price)
        rates = (listed.price.input_per_mtok, listed.price.output_per_mtok)
        assert all(math.isfinite(rate) and rate >= 0.0 for rate in rates)
        datetime.date.fromisoformat(listed.as_of)
        assert listed.source.startswith("https://")


def test_the_other_entry_takes_its_address_and_model_from_the_user():
    other = PROVIDERS["other"]
    assert other.wire == "chat"
    assert other.base_url is None
    assert other.models == ()


def test_the_local_entry_is_a_chat_server_on_this_computer():
    local = PROVIDERS["local"]
    assert local.wire == "chat"
    parts = urlsplit(local.base_url)
    assert parts.scheme == "http"
    assert parts.hostname in LOOPBACK


def test_connect_sends_the_key_and_reply_limit_to_the_given_address():
    with FakeProvider(
        [Response(200, chat_reply("ready", prompt_tokens=9, completion_tokens=2))]
    ) as fake:
        model = connect(
            PROVIDERS["other"], "test-model", FAKE_KEY, max_output_tokens=64, base_url=fake.url
        )
        reply = model([{"role": "user", "content": "Reply with the word ready"}], [])
    assert isinstance(reply, Reply)
    assert reply.text == "ready"
    assert (reply.input_tokens, reply.output_tokens) == (9, 2)
    (request,) = fake.requests
    assert request.path == "/chat/completions"
    headers = {k.lower(): v for k, v in request.headers.items()}
    assert headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = request.json()
    assert body["model"] == "test-model"
    assert body["max_completion_tokens"] == 64


def test_the_connected_model_does_not_show_its_key():
    model = connect(
        PROVIDERS["other"],
        "test-model",
        FAKE_KEY,
        max_output_tokens=64,
        base_url="https://api.example.com/v1",
    )
    assert FAKE_KEY not in repr(model)
    assert FAKE_KEY not in str(model)


def test_the_other_entry_needs_an_address():
    with pytest.raises(ValueError):
        connect(PROVIDERS["other"], "test-model", FAKE_KEY, max_output_tokens=64)


def test_a_local_server_is_sent_a_placeholder_key_instead_of_a_stored_one(monkeypatch):
    # The local entry sends a placeholder key, which its server ignores (spec section 11).
    monkeypatch.setenv("VBAR_LLM_API_KEY", FAKE_KEY)
    with FakeProvider([Response(200, chat_reply("ready"))]) as fake:
        model = connect(
            PROVIDERS["local"], "test-model", None, max_output_tokens=64, base_url=fake.url
        )
        model([{"role": "user", "content": "Reply with the word ready"}], [])
    (request,) = fake.requests
    headers = {k.lower(): v for k, v in request.headers.items()}
    assert headers["authorization"].startswith("Bearer ")
    assert headers["authorization"].removeprefix("Bearer ").strip()
    assert FAKE_KEY not in request.raw.decode("utf-8") + json.dumps(headers)


def test_an_address_to_another_host_over_plain_http_is_refused():
    with pytest.raises(ValueError):
        connect(
            PROVIDERS["other"],
            "test-model",
            FAKE_KEY,
            max_output_tokens=64,
            base_url="http://api.example.com/v1",
        )


# Rules for every catalogue entry (spec section 11); the local entry is the one on loopback.
LISTED = [(p, m) for p in ENTRIES for m in p.models]
REMOTE = [p for p in ENTRIES if p.base_url is None or urlsplit(p.base_url).hostname not in LOOPBACK]
LOCAL_BASE = "http://127.0.0.1:11434/v1"


def listed_id(pair):
    return f"{pair[0].key}/{pair[1].name}"


@pytest.mark.parametrize("pair", LISTED, ids=listed_id)
def test_every_listed_model_says_whether_it_takes_tools(pair):
    assert isinstance(pair[1].tools, bool)


@pytest.mark.parametrize("pair", LISTED, ids=listed_id)
def test_a_model_carries_extra_body_fields_only_on_the_chat_wire(pair):
    provider, listed = pair
    assert isinstance(listed.extra_body, Mapping)
    if listed.extra_body:
        assert provider.wire == "chat"


@pytest.mark.parametrize("pair", [(p, m) for p in REMOTE for m in p.models], ids=listed_id)
def test_every_remote_model_has_a_positive_price_read_on_the_catalogue_date(pair):
    listed = pair[1]
    assert listed.price.input_per_mtok > 0.0
    assert listed.price.output_per_mtok > 0.0
    assert listed.as_of == "2026-11-05"
    assert urlsplit(listed.source).scheme == "https"
    assert urlsplit(listed.source).hostname


@pytest.mark.parametrize("provider", REMOTE, ids=lambda p: p.key)
def test_every_remote_provider_is_reached_over_https(provider):
    if provider.base_url is not None:
        assert urlsplit(provider.base_url).scheme == "https"


def test_exactly_one_entry_is_a_local_server_at_the_default_port():
    local = [p for p in ENTRIES if p not in REMOTE]
    assert [p.base_url for p in local] == [LOCAL_BASE]
    assert local[0].wire == "chat"


def test_the_menu_ends_with_the_entry_that_takes_any_address():
    assert ENTRIES[-1].base_url is None
    assert all(p.base_url is not None for p in ENTRIES[:-1])


def test_a_placeholder_key_may_go_to_the_local_default_address():
    # Building the model checks the address and the key; nothing is sent.
    local = next(p for p in ENTRIES if p not in REMOTE)
    connect(local, "test-model", None, max_output_tokens=64)


def chat_entry(*models):
    return Provider(
        key="test-entry",
        label="Test entry",
        summary="a chat server for tests",
        wire="chat",
        base_url=None,
        models=models,
    )


def test_a_model_extra_body_is_sent_in_every_request():
    listed = ListedModel(
        "test-model",
        Price(1.0, 2.0),
        "2026-11-05",
        "https://example.com/prices",
        tools=True,
        extra_body={"reasoning_effort": "none"},
    )
    with FakeProvider([Response(200, chat_reply("ready"))]) as fake:
        model = connect(
            chat_entry(listed), "test-model", FAKE_KEY, max_output_tokens=64, base_url=fake.url
        )
        model([{"role": "user", "content": "first"}], [])
        model([{"role": "user", "content": "second"}], [])
    assert [r.json()["reasoning_effort"] for r in fake.requests] == ["none", "none"]


def test_a_model_extra_body_off_the_chat_wire_is_refused():
    with pytest.raises(ValueError):
        Provider(
            key="test-entry",
            label="Test entry",
            summary="a messages server for tests",
            wire="messages",
            base_url="https://api.example.com",
            models=(
                ListedModel(
                    "test-model",
                    Price(1.0, 2.0),
                    "2026-11-05",
                    "https://example.com/prices",
                    tools=True,
                    extra_body={"reasoning_effort": "none"},
                ),
            ),
        )


def test_an_entry_fixed_headers_go_with_every_request_but_never_replace_the_key():
    entry = Provider(
        key="test-entry",
        label="Test entry",
        summary="a chat server for tests",
        wire="chat",
        base_url=None,
        extra_headers={"X-Api-Version": "2026-10-21", "Authorization": "Bearer replaced"},
    )
    with FakeProvider([Response(200, chat_reply("ready"))]) as fake:
        model = connect(entry, "test-model", FAKE_KEY, max_output_tokens=64, base_url=fake.url)
        model([{"role": "user", "content": "first"}], [])
        model([{"role": "user", "content": "second"}], [])
    assert len(fake.requests) == 2
    for request in fake.requests:
        headers = {k.lower(): v for k, v in request.headers.items()}
        assert headers["x-api-version"] == "2026-10-21"
        assert headers["authorization"] == f"Bearer {FAKE_KEY}"
