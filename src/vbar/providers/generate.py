"""The generate format: `{base}/v1beta/models/{model}:generateContent` with parts and turns.

Tools go as one `functionDeclarations` list, the assistant's turn has the role "model", tool
results are `functionResponse` parts of one user turn, and the reply limit is
`generationConfig.maxOutputTokens`. A reply that called tools is sent back with its parts
unchanged, thought signatures included. Output tokens count the thoughts as well as the reply.
No temperature is sent.
"""

from __future__ import annotations

import json
import re
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

# Model names that cannot leave their path segment or reach the query.
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Tool calls whose reply parts are kept; a decision needs only its own few.
MAX_KEPT_CALLS = 1024


def classify(answer: Answer) -> str | None:
    """The failure kind of a generate-format answer, or None for a 2xx answer.

    The format's documented codes are for another API; this mapping is assumed.
    """
    status = answer.status
    if 200 <= status < 300:
        return None
    details = error_field(answer.body, "details")
    reasons = (
        {d.get("reason") for d in details if isinstance(d, dict)}
        if isinstance(details, list)
        else set()
    )
    if status == 401 or (status == 400 and "API_KEY_INVALID" in reasons):
        return "key"
    if status == 429:
        return "rate"
    if status == 404:
        return "model"
    if 400 <= status < 500:
        return "refused"
    return "server"


def _response(content: str) -> dict[str, Any]:
    """A tool result as the object a `functionResponse` carries."""
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, RecursionError):
        value = None
    return value if isinstance(value, dict) else {"content": content}


def _results_turn(contents: list[dict[str, Any]]) -> bool:
    """Whether the last converted turn is a user turn opened by function responses."""
    if not contents or contents[-1]["role"] != "user":
        return False
    return "functionResponse" in contents[-1]["parts"][0]


class GenerateModel:
    """A `Model` that calls a generate-format server: `complete(messages, tools) -> Reply`.

    Parameters
    ----------
    base_url : str
        The API root, such as `https://host`; `/v1beta/models/{model}:generateContent` is
        appended. Plain http is allowed only to localhost or a loopback address.
    model : str
        The model name the server knows: letters, digits, ".", "_" and "-".
    key : str
        The API key, sent in the `auth_header` header.
    max_output_tokens : int
        The reply limit, sent as `generationConfig.maxOutputTokens`.
    auth_header : str
        The name of the provider's key header.
    timeout : float, optional
        Seconds to wait for the connection and for each read.
    sleep : callable, optional
        Waits the given seconds before a retry.

    Raises
    ------
    ValueError
        If the model name would change the address, the address is refused, the key is empty or
        holds a control or non-ASCII character, or `max_output_tokens` is not a positive integer.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        key: str,
        max_output_tokens: int,
        auth_header: str,
        timeout: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not isinstance(model, str) or not MODEL_NAME.fullmatch(model):
            raise ValueError(f"model name {model!r} holds characters an address cannot carry")
        self.url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
        check_url(self.url)
        check_key(key)
        check_reply_limit(max_output_tokens)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.sleep = sleep
        self.auth_header = auth_header
        self._key = key
        # The parts of each reply that called tools, keyed by each of its call ids.
        self._parts: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        # Ids V-BAR gave to calls the server sent without one; they are not sent back.
        self._given_ids: set[str] = set()
        self._next_id = 0

    def __repr__(self) -> str:
        return f"GenerateModel(url={self.url!r}, model={self.model!r})"

    def _model_turn(self, message: Mapping[str, Any]) -> dict[str, Any]:
        calls = message.get("tool_calls") or []
        if calls and calls[0]["id"] in self._parts:
            return {"role": "model", "parts": self._parts[calls[0]["id"]]}
        parts: list[dict[str, Any]] = []
        if message["content"]:
            parts.append({"text": message["content"]})
        for call in calls:
            function: dict[str, Any] = {"name": call["name"], "args": call["arguments"]}
            if call["id"] and call["id"] not in self._given_ids:
                function = {"id": call["id"], **function}
            parts.append({"functionCall": function})
        return {"role": "model", "parts": parts}

    def request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """The request body for a neutral conversation and tool list."""
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "tool":
                function: dict[str, Any] = {
                    "name": message["name"],
                    "response": _response(message["content"]),
                }
                call_id = message["tool_call_id"]
                if call_id and call_id not in self._given_ids:
                    function = {"id": call_id, **function}
                part = {"functionResponse": function}
                if _results_turn(contents):
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
            elif role == "assistant":
                contents.append(self._model_turn(message))
            elif _results_turn(contents):
                contents[-1]["parts"].append({"text": message["content"]})
            else:
                contents.append({"role": "user", "parts": [{"text": message["content"]}]})
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": self.max_output_tokens},
        }
        if tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": spec["name"],
                            "description": spec["description"],
                            "parameters": spec["parameters"],
                        }
                        for spec in tools
                    ]
                }
            ]
        return body

    def _call_id(self, function: Mapping[str, Any]) -> str:
        given = function.get("id")
        if isinstance(given, str) and given:
            return given
        self._next_id += 1
        call_id = f"vbar-call-{self._next_id}"
        self._given_ids.add(call_id)
        return call_id

    def _reply(self, body: Any) -> Reply:
        parts = body["candidates"][0]["content"]["parts"]
        texts, calls = [], []
        for part in parts:
            if "functionCall" in part:
                function = part["functionCall"]
                calls.append(
                    ToolCall(function["name"], function.get("args", {}), self._call_id(function))
                )
            elif "text" in part and not part.get("thought"):
                texts.append(part["text"])
        if not all(isinstance(text, str) for text in texts):
            raise provider_error("server")
        kept = [dict(part) for part in parts]
        for call in calls:
            self._parts[call.id] = kept
            self._parts.move_to_end(call.id)
        while len(self._parts) > MAX_KEPT_CALLS:
            dropped, _ = self._parts.popitem(last=False)
            self._given_ids.discard(dropped)
        usage = body.get("usageMetadata")
        return Reply(
            text="".join(texts),
            input_tokens=token_count(usage, "promptTokenCount")
            + token_count(usage, "toolUsePromptTokenCount"),
            output_tokens=token_count(usage, "candidatesTokenCount")
            + token_count(usage, "thoughtsTokenCount"),
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
            The text without thought parts, the tool calls and the token counts the server
            reports: input is the prompt and tool-use prompt, output the reply and thoughts
            (0 for a missing field).

        Raises
        ------
        ProviderError
            For a refused or failed request; rate, network and server failures are also
            `InfrastructureError`. A malformed success answer is kind "server".
        """
        answer = post_with_retries(
            url=self.url,
            headers={self.auth_header: self._key},
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
