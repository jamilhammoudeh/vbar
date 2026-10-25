"""A Model Context Protocol tool server: an LLM harness flies scored episodes through tool calls.

JSON-RPC 2.0 messages, one per line on stdin and stdout, with the methods `initialize`, `ping`,
`tools/list` and `tools/call`. The tools describe the task, start the next episode from a fixed
list of seeds, return the observation, apply a command, and offer the coast and transfer
predictions of `vbar.agents.tools`. Like any scored agent, the client sees only the public
view, the observations and the time: never the true state, the reward or the result, which the
server keeps (`ToolServer.results`, or a JSON-lines file from the command line). When the
input ends, an episode still running and every seed not yet flown are recorded as unfinished,
so a client cannot drop an episode by disconnecting.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from collections.abc import Callable, Iterable
from functools import partial
from typing import IO, Any

import numpy as np

from .. import __version__
from ..runner import ScoredRun
from .base import action_scale
from .remote import view_to_dict
from .tools import CW_TOOLS, call_tool

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
# JSON-RPC error codes.
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND = -32700, -32600, -32601
INVALID_PARAMS, INTERNAL_ERROR = -32602, -32603

FRAME = (
    "Hill frame centered on the target: x radial (up, away from Earth), y along-track (direction "
    "of flight), z cross-track. Positions in m, velocities in m/s. Commands are fractions of the "
    "maximum acceleration per axis, between -1 and 1, each held for one step."
)
NO_ARGUMENTS: dict[str, Any] = {"type": "object", "properties": {}}


class ToolError(Exception):
    """A tool call that cannot be carried out; reported to the client as a tool error."""


class ToolServer:
    """Serves episodes of `env`, one per seed in `seeds`, in order.

    Each finished episode's `run_episode`-style result is appended to `results` and passed to
    `on_result`. `agent_name` labels the results.
    """

    def __init__(
        self,
        env,
        seeds: Iterable[int],
        agent_name: str = "mcp-client",
        on_result: Callable[[dict], None] | None = None,
    ):
        self.env = env
        self.view = env.agent_view()
        self.scale = action_scale(self.view)
        self.seeds = list(seeds)
        self.agent_name = agent_name
        self.on_result = on_result
        self.results: list[dict] = []
        self.episode: ScoredRun | None = None
        self.next_seed = 0
        self.since = time.perf_counter()
        self.tools: dict[str, tuple[str, dict[str, Any], Callable[[dict], dict]]] = {
            "describe_task": (
                "The task: frame, goal, constraints, step length and limits.",
                NO_ARGUMENTS,
                self._describe,
            ),
            "start_episode": (
                "Start the next episode and return its first observation.",
                NO_ARGUMENTS,
                self._start,
            ),
            "observe": ("The current observation and time.", NO_ARGUMENTS, self._observe),
            "command": (
                "Hold a command for one step and return the next observation. Arguments are "
                "fractions of the maximum acceleration along x, y and z, between -1 and 1.",
                {
                    "type": "object",
                    "properties": {k: {"type": "number"} for k in ("ax", "ay", "az")},
                    "required": ["ax", "ay", "az"],
                },
                self._command,
            ),
        }
        for tool in CW_TOOLS:
            self.tools[tool.name] = (
                tool.description,
                dict(tool.parameters),
                partial(self._predict, tool.name),
            )

    def _describe(self, args: dict) -> dict:
        return {"frame": FRAME, "view": view_to_dict(self.view), "episodes": len(self.seeds)}

    def _snapshot(self) -> dict:
        episode = self._running()
        names = self.view.observation
        return {
            "observation": dict(
                zip(names, np.asarray(episode.obs, dtype=float).tolist(), strict=True)
            ),
            "time_s": float(episode.info["time_s"]),
            "steps_left": self.view.max_steps - len(episode.actions),
            "episode": self.next_seed,
        }

    def _running(self) -> ScoredRun:
        if self.episode is None:
            raise ToolError("no episode is running; call start_episode")
        return self.episode

    def _start(self, args: dict) -> dict:
        if self.episode is not None:
            raise ToolError("the current episode has not ended")
        if self.next_seed >= len(self.seeds):
            raise ToolError("no episodes left")
        self.episode = ScoredRun(self.env, self.seeds[self.next_seed])
        # A scenario may draw its goal at each reset (a port's attitude, the Sun).
        self.view = self.env.agent_view()
        self.next_seed += 1
        self.since = time.perf_counter()
        return self._snapshot()

    def _observe(self, args: dict) -> dict:
        return self._snapshot()

    def _command(self, args: dict) -> dict:
        episode = self._running()
        try:
            fraction = np.array([float(args[k]) for k in ("ax", "ay", "az")])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ToolError(f"command needs numbers ax, ay and az: {error}") from error
        if not np.all(np.isfinite(fraction)):
            raise ToolError("command values must be finite")
        wall = time.perf_counter() - self.since
        episode.step(np.clip(fraction, -1.0, 1.0) * self.scale, wall)
        if episode.done:
            self._record(episode)
            self.episode = None
            return {"episode_over": True, "episodes_left": len(self.seeds) - self.next_seed}
        self.since = time.perf_counter()
        return {"episode_over": False, **self._snapshot()}

    def _record(self, episode: ScoredRun) -> None:
        result = episode.result(self.agent_name)
        self.results.append(result)
        if self.on_result is not None:
            self.on_result(result)

    def finish(self) -> None:
        """Record the running episode and every seed not yet flown as unfinished."""
        if self.episode is not None:
            self._record(self.episode)
            self.episode = None
        while self.next_seed < len(self.seeds):
            self._record(ScoredRun(self.env, self.seeds[self.next_seed]))
            self.next_seed += 1

    def _predict(self, name: str, args: dict) -> dict:
        found = call_tool({t.name: t for t in CW_TOOLS}, name, self._running().obs, self.view, args)
        if "error" in found:
            raise ToolError(found["error"])
        return found

    def handle(self, message: Any) -> dict | None:
        """The response to one JSON-RPC message, or None for a notification."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        if "id" not in message:
            return None
        ident = message["id"]
        if not _valid_id(ident):
            return _error(None, INVALID_REQUEST, "id must be a string, a finite number or null")
        method, params = message.get("method"), message.get("params", {})
        if not isinstance(params, dict):
            return _error(ident, INVALID_PARAMS, "params must be an object")
        if method == "initialize":
            asked = params.get("protocolVersion")
            version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return _result(
                ident,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "vbar", "version": __version__},
                },
            )
        if method == "ping":
            return _result(ident, {})
        if method == "tools/list":
            tools = [
                {"name": name, "description": text, "inputSchema": schema}
                for name, (text, schema, _) in self.tools.items()
            ]
            return _result(ident, {"tools": tools})
        if method == "tools/call":
            name, arguments = params.get("name"), params.get("arguments", {})
            if not isinstance(name, str) or name not in self.tools:
                return _error(ident, INVALID_PARAMS, f"unknown tool {name!r}")
            if not isinstance(arguments, dict):
                return _error(ident, INVALID_PARAMS, "arguments must be an object")
            try:
                payload = self.tools[name][2](arguments)
            except ToolError as error:
                return _result(ident, {"content": [_text(str(error))], "isError": True})
            return _result(
                ident,
                {
                    "content": [_text(json.dumps(payload))],
                    "structuredContent": payload,
                    "isError": False,
                },
            )
        return _error(ident, METHOD_NOT_FOUND, f"unknown method {method!r}")

    def serve(self, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
        """Answer messages from `stdin` on `stdout`, one per line, until the input ends.

        A message that fails inside the server gets an internal error, and the loop goes on.
        The session ends when the input ends or the output can no longer be written; then
        unfinished episodes are recorded (`finish`), however the session ended.
        """
        source, sink = stdin or sys.stdin, stdout or sys.stdout
        try:
            for line in source:
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except (ValueError, RecursionError):
                    response: dict | None = _error(None, PARSE_ERROR, "invalid JSON")
                else:
                    try:
                        response = self.handle(message)
                    except Exception as error:  # one bad request must not end the scored session
                        ident = message.get("id") if isinstance(message, dict) else None
                        reason = f"{type(error).__name__}: {error}"
                        response = _error(ident, INTERNAL_ERROR, reason)
                if response is not None:
                    try:
                        text = json.dumps(response, allow_nan=False)
                    except ValueError:
                        # Ids are checked when the request arrives, so only a result can be
                        # non-finite here; strict JSON has no NaN or infinity.
                        fallback = _error(
                            response.get("id"), INTERNAL_ERROR, "the result is not finite"
                        )
                        text = json.dumps(fallback, allow_nan=False)
                    try:
                        sink.write(text + "\n")
                        sink.flush()
                    except OSError:
                        # The client stopped reading: the session is over.
                        break
        finally:
            self.finish()


def _valid_id(ident: Any) -> bool:
    """JSON-RPC ids are strings, numbers or null; numbers must be finite for strict JSON."""
    if ident is None or isinstance(ident, str):
        return True
    if isinstance(ident, bool) or not isinstance(ident, int | float):
        return False
    return isinstance(ident, int) or bool(np.isfinite(ident))


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _result(ident, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def _error(ident, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


def main(argv: list[str] | None = None) -> None:
    """Serve the approach scenario over stdio; results go to a JSON-lines file."""
    from ..scenarios import ApproachConfig, ApproachEnv

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="seeds to repeat; by default drawn and recorded"
    )
    parser.add_argument("--episodes", type=int, default=1, help="episodes with drawn seeds")
    parser.add_argument("--max-steps", type=int, default=ApproachConfig().max_steps)
    parser.add_argument("--out", default="results.jsonl", help="JSON lines, one per episode")
    parser.add_argument("--name", default="mcp-client", help="agent name in the results")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    # A small seed can be found by matching the first observation, so seeds are drawn.
    seeds = args.seeds or [secrets.randbits(128) for _ in range(args.episodes)]
    env = ApproachEnv(ApproachConfig(max_steps=args.max_steps))
    with open(args.out, "w", encoding="utf-8") as out:

        def write(result: dict) -> None:
            plain = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
            out.write(json.dumps(plain) + "\n")
            out.flush()

        ToolServer(env, seeds, args.name, on_result=write).serve()


if __name__ == "__main__":
    main()
