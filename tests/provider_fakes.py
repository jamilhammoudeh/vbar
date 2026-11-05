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


@dataclass(frozen=True)
class Response:
    """What the fake answers: status, headers and a body (a dict is sent as JSON)."""

    status: int = 200
    body: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)
    delay_s: float = 0.0


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
