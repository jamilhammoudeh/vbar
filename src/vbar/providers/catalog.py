"""The providers `vbar setup` offers, in menu order, and the models each lists with a dated price.

A listed price is the provider's published standard rate on the day it was read, with the page it
was read from; the provider's bill is final. Where a model has a lower rate for short prompts or
until a date, the higher rate is listed, so a dollar cap stops early rather than late. A model
that is not listed needs a price given by hand before a run can hold a dollar cap.
"""

from __future__ import annotations

import datetime
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..agents.llm import Price
from .http import check_url

WIRES = frozenset({"chat", "messages", "generate"})
# Sent to a server that takes no key; local servers ignore it but some clients require one.
PLACEHOLDER_KEY = "vbar-local"
AS_OF = "2026-11-05"
MESSAGES_PRICES = "https://platform.claude.com/docs/en/about-claude/pricing"
CHAT_PRICES = "https://developers.openai.com/api/docs/pricing"
GENERATE_PRICES = "https://aistudio.google.com/docs/pricing"


@dataclass(frozen=True)
class ListedModel:
    """A model a provider offers, with its published price.

    Parameters
    ----------
    name : str
        The name the provider's API takes.
    price : Price
        US dollars per million input and output tokens.
    as_of : str
        The ISO date the price was read.
    source : str
        The https page the price was read from.
    tools : bool
        Whether the model takes tool calls in the provider's wire format.
    extra_body : Mapping, optional
        Further fields of every request body for this model; chat format only.

    Raises
    ------
    ValueError
        If the name is empty, a rate is negative or not finite, the date is not an ISO date or
        the source is not an https address.
    """

    name: str
    price: Price
    as_of: str
    source: str
    tools: bool
    extra_body: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("a listed model needs a name")
        rates = (self.price.input_per_mtok, self.price.output_per_mtok)
        if not all(math.isfinite(rate) and rate >= 0.0 for rate in rates):
            raise ValueError(f"{self.name}: price rates must be finite and not negative")
        datetime.date.fromisoformat(self.as_of)
        if not self.source.startswith("https://"):
            raise ValueError(f"{self.name}: the price source must be an https address")


@dataclass(frozen=True)
class Provider:
    """A provider entry of the setup menu.

    Parameters
    ----------
    key : str
        The name stored in the settings and given to `vbar setup --provider`.
    label : str
        The name shown in the menu.
    summary : str
        What the entry offers, in a few words.
    wire : str
        The request format: "chat", "messages" or "generate".
    base_url : str or None
        The API root; None when the user gives it.
    auth_header : str or None, optional
        The header that carries the bare key, which the generate format needs; None sends
        `Authorization: Bearer <key>`.
    extra_headers : Mapping of str to str, optional
        Further headers of every request, such as the API version the server requires.
    extra_body : Mapping, optional
        Further fields of every request body; chat format only.
    models : tuple of ListedModel, optional
        Models with a listed price, the default first.
    needs_key : bool, optional
        Whether the provider takes an API key; an entry that does not is sent `PLACEHOLDER_KEY`.

    Raises
    ------
    ValueError
        If the wire format is unknown, the address is refused by `check_url`, a model is listed
        twice, or a key setting, header or body field is one the wire format's adapter cannot
        take.
    """

    key: str
    label: str
    summary: str
    wire: str
    base_url: str | None
    auth_header: str | None = None
    extra_headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    extra_body: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    models: tuple[ListedModel, ...] = ()
    needs_key: bool = True

    def __post_init__(self):
        if self.wire not in WIRES:
            raise ValueError(f"{self.key}: unknown wire format {self.wire!r}")
        if self.base_url is not None:
            check_url(self.base_url)
        if (self.wire == "generate") != (self.auth_header is not None):
            raise ValueError(f"{self.key}: the generate format, and only it, needs auth_header")
        if self.wire == "generate" and self.extra_headers:
            raise ValueError(f"{self.key}: the generate adapter sends no extra headers")
        if self.wire != "chat" and not self.needs_key:
            raise ValueError(f"{self.key}: only the chat format is called without a key")
        if self.wire != "chat" and (self.extra_body or any(m.extra_body for m in self.models)):
            raise ValueError(f"{self.key}: only the chat adapter sends extra body fields")
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.key}: a model is listed twice")

    def listed(self, name: str) -> ListedModel | None:
        """The listed model called `name`, or None if the entry does not list it."""
        return next((m for m in self.models if m.name == name), None)


PROVIDERS: Mapping[str, Provider] = MappingProxyType(
    {
        provider.key: provider
        for provider in (
            Provider(
                key="anthropic",
                label="Anthropic",
                summary="Claude models",
                wire="messages",
                base_url="https://api.anthropic.com",
                extra_headers=MappingProxyType({"anthropic-version": "2026-10-21"}),
                models=(
                    ListedModel(
                        "claude-sonnet-5-5", Price(2.00, 10.00), AS_OF, MESSAGES_PRICES, True
                    ),
                    ListedModel(
                        "claude-opus-5-5", Price(4.00, 20.00), AS_OF, MESSAGES_PRICES, True
                    ),
                    # 0.10 and 0.50 for prompts up to 100k tokens.
                    ListedModel(
                        "claude-haiku-5-5", Price(0.50, 2.50), AS_OF, MESSAGES_PRICES, True
                    ),
                ),
            ),
            Provider(
                key="openai",
                label="OpenAI",
                summary="GPT models",
                wire="chat",
                base_url="https://api.openai.com/v1",
                models=(
                    # 0.10 and 0.50 up to 272K input tokens. It takes tools in the chat format
                    # only with reasoning effort "none".
                    ListedModel(
                        "gpt-6-luna",
                        Price(0.20, 0.75),
                        AS_OF,
                        CHAT_PRICES,
                        True,
                        MappingProxyType({"reasoning_effort": "none"}),
                    ),
                    # The long-context rates. Their tool calls need the Responses API, which V-BAR
                    # does not speak.
                    ListedModel("gpt-6.1-terra", Price(2.00, 7.50), AS_OF, CHAT_PRICES, False),
                    ListedModel("gpt-6.1-sol", Price(4.00, 15.00), AS_OF, CHAT_PRICES, False),
                ),
            ),
            Provider(
                key="google",
                label="Google",
                summary="Gemini models",
                wire="generate",
                base_url="https://generativelanguage.googleapis.com",
                auth_header="x-goog-api-key",
                models=(
                    # 2 and 10 through Nov 30 2026.
                    ListedModel("gemini-4-argon", Price(4.00, 20.00), AS_OF, GENERATE_PRICES, True),
                    # 0.75 and 3.75 through Dec 31 2026.
                    ListedModel(
                        "gemini-3.8-flash", Price(1.50, 7.50), AS_OF, GENERATE_PRICES, True
                    ),
                    ListedModel(
                        "gemini-3.5-flash-lite", Price(0.30, 2.50), AS_OF, GENERATE_PRICES, True
                    ),
                ),
            ),
            Provider(
                key="local",
                label="Local server",
                summary="models running on this computer, no key needed",
                wire="chat",
                # The default port of a widely used local model server; --base-url changes it.
                base_url="http://127.0.0.1:11434/v1",
                needs_key=False,
            ),
            Provider(
                key="other",
                label="Other address",
                summary="any server that takes chat-format requests",
                wire="chat",
                base_url=None,
            ),
        )
    }
)
