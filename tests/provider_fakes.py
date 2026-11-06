"""A fake model provider on 127.0.0.1 that records requests and sends scripted responses."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

FAKE_KEY = "test-key-7f3a9c1e5b2d8f4a6c0e"
# Stands in a JSON body for `Response.pad` bytes of filler streamed in its place.
PAD = "@@pad@@"


@dataclass(frozen=True)
class Response:
    """What the fake answers: status, headers and a body (a dict is sent as JSON).

    With `pad`, the body's `PAD` marker is replaced by that many bytes of "a", streamed in
    chunks, with a Content-Length header only if `length` is set.
    """

    status: int = 200
    body: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)
    delay_s: float = 0.0
    pad: int = 0
    length: bool = True


@dataclass(frozen=True)
class Request:
    """One request the fake received."""

    method: str
    path: str
    headers: Mapping[str, str]
    raw: bytes

    def json(self) -> Any:
        return json.loads(self.raw)


Responder = Callable[[Request], Response]


class FakeProvider:
    """A provider on 127.0.0.1 answering with `responses` in turn, the last repeating.

    `responses` may instead be a function of the request. Use as a context manager; `url` is the
    root address and `requests` holds what arrived.
    """

    def __init__(self, responses: Sequence[Response] | Responder = ()):
        self.responses = responses
        self.requests: list[Request] = []
        # Bytes of each response body written so far, in request order.
        self.sent: list[int] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                fake._answer(self)

            def do_GET(self):
                fake._answer(self)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # A handler still sleeping for a client that timed out must not hold up shutdown.
        self.server.daemon_threads = True
        self.server.block_on_close = False
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _respond(self, request: Request) -> Response:
        if callable(self.responses):
            return self.responses(request)
        if not self.responses:
            return Response(500, {"error": {"message": "no scripted response"}})
        return self.responses[min(len(self.requests), len(self.responses)) - 1]

    def _answer(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        request = Request(
            handler.command, handler.path, dict(handler.headers.items()), handler.rfile.read(length)
        )
        self.requests.append(request)
        response = self._respond(request)
        if response.pad:
            self._stream(handler, response)
            return
        if response.delay_s:
            time.sleep(response.delay_s)
        body = response.body
        if isinstance(body, dict | list):
            payload = json.dumps(body).encode("utf-8")
            content_type = "application/json"
        else:
            payload = (body or "").encode("utf-8")
            content_type = "text/plain"
        try:
            handler.send_response(response.status)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(payload)))
            for name, value in response.headers.items():
                handler.send_header(name, value)
            handler.end_headers()
            handler.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # A client that timed out has closed the connection.
            pass

    def _stream(self, handler: BaseHTTPRequestHandler, response: Response) -> None:
        head, tail = json.dumps(response.body).encode("utf-8").split(PAD.encode("utf-8"))
        index = len(self.sent)
        self.sent.append(0)
        chunk = b"a" * 65536
        try:
            handler.send_response(response.status)
            handler.send_header("Content-Type", "application/json")
            if response.length:
                handler.send_header("Content-Length", str(len(head) + response.pad + len(tail)))
            handler.end_headers()
            handler.wfile.write(head)
            self.sent[index] += len(head)
            left = response.pad
            while left:
                part = chunk[: min(left, len(chunk))]
                handler.wfile.write(part)
                self.sent[index] += len(part)
                left -= len(part)
            handler.wfile.write(tail)
            self.sent[index] += len(tail)
        except OSError:
            # The client stopped reading and closed the connection.
            pass

    def __enter__(self) -> FakeProvider:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def closed_port_url() -> str:
    """An address on 127.0.0.1 where nothing listens."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


class Sleeps:
    """An injected sleep that records the delays it was asked for and returns at once."""

    def __init__(self):
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def chat_reply(text="", tool_calls=(), prompt_tokens=812, completion_tokens=45):
    """A chat-format response body: one choice, optional tool calls, usage counts.

    `tool_calls` holds (id, name, arguments string) triples.
    """
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = [
            {"id": id_, "type": "function", "function": {"name": name, "arguments": arguments}}
            for id_, name, arguments in tool_calls
        ]
    return {
        "id": "chatcmpl-0001",
        "object": "chat.completion",
        "created": 1762300000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


def chat_error(message, type_, code):
    """A chat-format error body."""
    return {"error": {"message": message, "type": type_, "param": None, "code": code}}


def messages_reply(text="", tool_calls=(), thinking=(), input_tokens=812, output_tokens=45):
    """A messages-format response body: thinking blocks, then text, then tool_use blocks.

    `tool_calls` holds (id, name, input) triples; `thinking` holds content blocks as sent.
    """
    content: list[dict[str, Any]] = [dict(block) for block in thinking]
    if text:
        content.append({"type": "text", "text": text})
    content += [
        {"type": "tool_use", "id": id_, "name": name, "input": input_}
        for id_, name, input_ in tool_calls
    ]
    return {
        "id": "msg-0001",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": content,
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def thinking_block(thinking, signature):
    return {"type": "thinking", "thinking": thinking, "signature": signature}


def redacted_block(data):
    return {"type": "redacted_thinking", "data": data}


def messages_error(type_, message, error_code=None):
    """A messages-format error body; `error_code` goes in `error.details`."""
    error: dict[str, Any] = {"type": type_, "message": message}
    if error_code is not None:
        error["details"] = {"error_code": error_code}
    return {"type": "error", "error": error, "request_id": "req-0001"}


def generate_reply(
    text="",
    calls=(),
    signature=None,
    prompt_tokens=812,
    candidates_tokens=40,
    thoughts_tokens=5,
    tool_use_prompt_tokens=None,
    thought_text=None,
):
    """A generateContent response body with one candidate.

    `calls` holds (id or None, name, args) triples; `signature` goes on the first call's part,
    or on the text part when there is no call. `thought_text` adds a thought part first.
    """
    parts: list[dict[str, Any]] = []
    if thought_text is not None:
        parts.append({"text": thought_text, "thought": True})
    if text:
        parts.append({"text": text})
    for id_, name, args in calls:
        call: dict[str, Any] = {"name": name, "args": args}
        if id_ is not None:
            call["id"] = id_
        parts.append({"functionCall": call})
    if signature is not None:
        parts[-len(calls) if calls else -1]["thoughtSignature"] = signature
    usage: dict[str, Any] = {
        "promptTokenCount": prompt_tokens,
        "candidatesTokenCount": candidates_tokens,
        "thoughtsTokenCount": thoughts_tokens,
        "totalTokenCount": prompt_tokens + candidates_tokens + thoughts_tokens,
    }
    if tool_use_prompt_tokens is not None:
        usage["toolUsePromptTokenCount"] = tool_use_prompt_tokens
        usage["totalTokenCount"] += tool_use_prompt_tokens
    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": usage,
        "modelVersion": "test-model",
        "responseId": "resp-0001",
    }


def generate_error(code, message, status, reason=None):
    """A generateContent error body; `reason` goes in an ErrorInfo detail."""
    error: dict[str, Any] = {"code": code, "message": message, "status": status}
    if reason is not None:
        error["details"] = [{"reason": reason, "domain": "api.example.com"}]
    return {"error": error}
