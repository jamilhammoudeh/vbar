"""The messages format: `{base}/v1/messages` with a bearer key and content blocks.

Tools go as `{name, description, input_schema}`, consecutive tool results as `tool_result` blocks
at the start of one user message, and the reply limit as `max_tokens`. The thinking and
redacted thinking blocks of a reply that called tools are sent back unchanged with those calls,
since the format refuses a tool round whose thinking was dropped. No temperature is sent.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

from ..agents.llm import Reply, ToolCall
from . import provider_error
from .http import (
    Answer,
    check_key,
    check_reply_limit,
    check_url,
    error_field,
    post_with_retries,
    token_count,
)

# The 400 message the format sends when the account's own usage limit is reached.
SPEND_LIMIT_PREFIX = "You have reached your specified API usage limits"
THINKING_TYPES = frozenset({"thinking", "redacted_thinking"})
# Tool calls whose thinking is kept; a decision needs only its own few.
MAX_KEPT_CALLS = 1024


def classify(answer: Answer) -> str | None:
    """The failure kind of a messages-format answer, or None for a 2xx answer."""
    status = answer.status
    if 200 <= status < 300:
        return None
    if status == 401:
        return "key"
    if status == 402:
        return "credit"
    if status == 429:
        details = error_field(answer.body, "details")
        code = details.get("error_code") if isinstance(details, dict) else None
        return "credit" if code == "enforced_spend_limit_reached" else "rate"
    if status == 404:
        return "model"
    if status == 400:
        message = error_field(answer.body, "message")
        if isinstance(message, str) and message.startswith(SPEND_LIMIT_PREFIX):
            return "credit"
    if 400 <= status < 500:
        return "refused"
    return "server"


def _results_message(converted: list[dict[str, Any]]) -> bool:
    """Whether the last converted message is a user message opened by tool results."""
    if not converted or converted[-1]["role"] != "user":
        return False
    content = converted[-1]["content"]
    return isinstance(content, list) and content[0]["type"] == "tool_result"


class MessagesModel:
    """A `Model` that calls a messages-format server: `complete(messages, tools) -> Reply`.

    Parameters
    ----------
    base_url : str
        The API root, such as `https://host`; `/v1/messages` is appended. Plain http is allowed
        only to localhost or a loopback address.
    model : str
        The model name the server knows.
    key : str
        The API key, sent as `Authorization: Bearer`.
    max_output_tokens : int
        The reply limit, sent as `max_tokens`.
    headers : Mapping of str to str, optional
        Further request headers, such as the API version the server requires.
    timeout : float, optional
        Seconds to wait for the connection and for each read.
    sleep : callable, optional
        Waits the given seconds before a retry.

    Raises
    ------
    ValueError
        If the address is refused, the key is empty or holds a control or non-ASCII character, or
        `max_output_tokens` is not a positive integer.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        key: str,
        max_output_tokens: int,
        headers: Mapping[str, str] | None = None,
        timeout: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.url = base_url.rstrip("/") + "/v1/messages"
        check_url(self.url)
        check_key(key)
        check_reply_limit(max_output_tokens)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.sleep = sleep
        self._key = key
        # Thinking blocks of each reply that called tools, keyed by each of its call ids.
        self._thinking: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def __repr__(self) -> str:
        return f"MessagesModel(url={self.url!r}, model={self.model!r})"

    def _assistant(self, message: Mapping[str, Any]) -> dict[str, Any]:
        calls = message.get("tool_calls") or []
        content: list[dict[str, Any]] = []
        if calls:
            content += self._thinking.get(calls[0]["id"], [])
        if message["content"]:
            content.append({"type": "text", "text": message["content"]})
        content += [
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                "input": dict(call["arguments"])
                if isinstance(call["arguments"], Mapping)
                else call["arguments"],
            }
            for call in calls
        ]
        return {"role": "assistant", "content": content}

    def request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """The request body for a neutral conversation and tool list."""
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message["content"],
                }
                if _results_message(converted):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
            elif role == "user" and _results_message(converted):
                # The format wants one user turn: the results first, then the text.
                converted[-1]["content"].append({"type": "text", "text": message["content"]})
            elif role == "assistant":
                converted.append(self._assistant(message))
            else:
                converted.append({"role": role, "content": message["content"]})
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": converted,
        }
        if tools:
            body["tools"] = [
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "input_schema": spec["parameters"],
                }
                for spec in tools
            ]
        return body

    def _reply(self, body: Any) -> Reply:
        texts, calls, thinking = [], [], []
        for block in body["content"]:
            kind = block["type"]
            if kind == "text":
                texts.append(block["text"])
            elif kind == "tool_use":
                calls.append(ToolCall(block["name"], block["input"], block["id"]))
            elif kind in THINKING_TYPES:
                thinking.append(dict(block))
        if not all(isinstance(text, str) for text in texts):
            raise provider_error("server")
        for call in calls:
            self._thinking[call.id] = thinking
            self._thinking.move_to_end(call.id)
        while len(self._thinking) > MAX_KEPT_CALLS:
            self._thinking.popitem(last=False)
        usage = body.get("usage")
        return Reply(
            text="".join(texts),
            input_tokens=token_count(usage, "input_tokens"),
            output_tokens=token_count(usage, "output_tokens"),
            tool_calls=tuple(calls),
        )

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        """The model's reply to a provider-neutral conversation.

        Parameters
        ----------
        messages : list of dict
            The conversation, as `LLMAgent` builds it.
        tools : list of dict
            Tool specs with `name`, `description` and `parameters`.

        Returns
        -------
        Reply
            The text without thinking, the tool calls and the token counts the server reports
            (0 if missing).

        Raises
        ------
        ProviderError
            For a refused or failed request; rate, network and server failures are also
            `InfrastructureError`. A malformed success answer is kind "server".
        """
        answer = post_with_retries(
            url=self.url,
            headers={**self.headers, "Authorization": f"Bearer {self._key}"},
            body=self.request(messages, tools),
            timeout=self.timeout,
            classify=classify,
            sleep=self.sleep,
            model=self.model,
        )
        try:
            return self._reply(answer.body)
        except (KeyError, IndexError, TypeError, AttributeError):
            raise provider_error("server") from None
