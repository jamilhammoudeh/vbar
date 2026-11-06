"""The chat format: `{base}/chat/completions` with a bearer key, served by hosts and local servers.

Tools go as `{"type": "function", "function": spec}`, tool results as `role: "tool"` messages with
`tool_call_id`, and the reply limit as `max_completion_tokens`, which counts reasoning tokens too.
No temperature is sent.
"""

from __future__ import annotations

import json
import time
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

# The error codes and type of a 429 that means the balance or a spend limit is used up.
CREDIT_CODES = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    }
)


def classify(answer: Answer) -> str | None:
    """The failure kind of a chat-format answer, or None for a 2xx answer."""
    status = answer.status
    if 200 <= status < 300:
        return None
    if status == 401:
        return "key"
    if status == 429:
        error_type = error_field(answer.body, "type")
        code = error_field(answer.body, "code")
        if error_type == "insufficient_quota" or code in CREDIT_CODES:
            return "credit"
        return "rate"
    # The format's reference does not document 404; servers send it for an unknown model.
    if status == 404:
        return "model"
    # The chat format documents no credit or spend-limit case under the other 4xx statuses.
    if 400 <= status < 500:
        return "refused"
    return "server"


def _message(message: Mapping[str, Any]) -> dict[str, Any]:
    """A provider-neutral message in the chat format."""
    if message["role"] == "tool":
        return {
            "role": "tool",
            "tool_call_id": message["tool_call_id"],
            "content": message["content"],
        }
    converted: dict[str, Any] = {"role": message["role"], "content": message["content"]}
    if message.get("tool_calls"):
        converted["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    # Arguments the model sent as an unparseable string go back as it sent them.
                    "arguments": call["arguments"]
                    if isinstance(call["arguments"], str)
                    else json.dumps(call["arguments"]),
                },
            }
            for call in message["tool_calls"]
        ]
    return converted


def _tool_call(call: Mapping[str, Any]) -> ToolCall:
    raw = call["function"]["arguments"]
    arguments: Any = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, RecursionError):
            parsed = None
        # Anything but an object stays the raw string, which the agent answers with an error.
        if isinstance(parsed, dict):
            arguments = parsed
    return ToolCall(call["function"]["name"], arguments, call.get("id") or "")


def _reply(body: Any) -> Reply:
    message = body["choices"][0]["message"]
    content = message.get("content")
    # Content parts and other shapes are not part of the format's text reply.
    if content is not None and not isinstance(content, str):
        raise provider_error("server")
    return Reply(
        text=content or "",
        input_tokens=token_count(body.get("usage"), "prompt_tokens"),
        output_tokens=token_count(body.get("usage"), "completion_tokens"),
        tool_calls=tuple(_tool_call(call) for call in message.get("tool_calls") or ()),
    )


class ChatModel:
    """A `Model` that calls a chat-format server: `complete(messages, tools) -> Reply`.

    Parameters
    ----------
    base_url : str
        The API root, such as `https://host/v1`; `/chat/completions` is appended. Plain http is
        allowed only to localhost or a loopback address.
    model : str
        The model name the server knows.
    key : str
        The API key, sent as `Authorization: Bearer`.
    max_output_tokens : int
        The reply limit, sent as `max_completion_tokens`.
    headers : Mapping of str to str, optional
        Further request headers.
    extra_body : Mapping, optional
        Further fields of every request body; the fields V-BAR sets take precedence.
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
        extra_body: Mapping[str, Any] | None = None,
        timeout: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        check_url(self.url)
        check_key(key)
        check_reply_limit(max_output_tokens)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.headers = dict(headers or {})
        self.extra_body = dict(extra_body or {})
        self.timeout = timeout
        self.sleep = sleep
        self._key = key

    def __repr__(self) -> str:
        return f"ChatModel(url={self.url!r}, model={self.model!r})"

    def request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """The request body for a neutral conversation and tool list."""
        body: dict[str, Any] = {
            **self.extra_body,
            "model": self.model,
            "messages": [_message(m) for m in messages],
            "max_completion_tokens": self.max_output_tokens,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": dict(spec)} for spec in tools]
        return body

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
            The text, the tool calls and the token counts the server reports (0 if missing).

        Raises
        ------
        ProviderError
            For a refused or failed request; rate, network and server failures are also
            `InfrastructureError`. A malformed success answer is kind "server".
        """
        headers = {**self.headers, "Authorization": f"Bearer {self._key}"}
        answer = post_with_retries(
            url=self.url,
            headers=headers,
            body=self.request(messages, tools),
            timeout=self.timeout,
            classify=classify,
            sleep=self.sleep,
            model=self.model,
        )
        try:
            return _reply(answer.body)
        except (KeyError, IndexError, TypeError, AttributeError):
            raise provider_error("server") from None
