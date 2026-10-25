"""Agents in another process or language, over TCP with one JSON message per line.

`AgentServer` waits for a remote agent to connect and presents it as a local `Agent`, so
`run_episode` scores it like any other; `serve_agent` is the Python client that runs a local
agent for a server. The protocol, version 1, is a lockstep exchange of UTF-8 JSON objects, each
on one line:

    client -> server  {"type": "hello", "name": <agent name>, "protocol": 1}
    server -> client  {"type": "reset", "view": <AgentView>, "seed": <int>, "spawn_key": [...]}
    client -> server  {"type": "ready"}
    server -> client  {"type": "act", "obs": [...], "info": {"time_s": <float>}}
    client -> server  {"type": "action", "action": [...] or <int>}
    server -> client  {"type": "close"}

A client may answer any message with {"type": "error", "message": ...}. The server sends only
what an in-process agent receives: the public view, the agent's own seed (independent of the
episode seed) and the observations with the agent-safe info. Floats are written with Python's
shortest round-trip form, so a Python client acts on exactly the numbers an in-process agent
would.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import secrets
import socket
import time
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import numpy as np

from .base import Agent, AgentView

PROTOCOL = 1
# Longest message line accepted, in bytes; a view or an observation is a few kilobytes.
MAX_LINE = 1 << 20


class RemoteAgentError(RuntimeError):
    """The remote agent broke the protocol, reported an error or did not answer in time."""


def view_to_dict(view: AgentView) -> dict[str, Any]:
    """The view as JSON-ready data; tuples become lists."""

    def plain(value):
        if isinstance(value, Mapping):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, tuple | list | np.ndarray):
            return [plain(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return {f.name: plain(getattr(view, f.name)) for f in fields(view)}


def view_from_dict(data: Mapping[str, Any]) -> AgentView:
    """Rebuild a view; sequences in the goal and the observation names become tuples."""
    goal = {k: tuple(v) if isinstance(v, list) else v for k, v in data["goal"].items()}
    return AgentView(
        **{
            **data,
            "goal": goal,
            "constraints": tuple(data["constraints"]),
            "observation": tuple(data["observation"]),
        }
    )


class _Line:
    """One JSON object per line over a socket.

    The socket's timeout bounds each whole message, not each read, so a peer that sends a byte at
    a time cannot hold the session open; lines longer than `MAX_LINE` bytes are refused.
    """

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.timeout = sock.gettimeout()
        self.buffer = b""

    def send(self, message: Mapping[str, Any]) -> None:
        text = json.dumps(message, allow_nan=False, separators=(",", ":"))
        self.sock.settimeout(self.timeout)
        self.sock.sendall(text.encode("utf-8") + b"\n")

    def _line(self) -> bytes:
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        while b"\n" not in self.buffer:
            if len(self.buffer) > MAX_LINE:
                raise RemoteAgentError(f"the remote agent sent a line longer than {MAX_LINE} bytes")
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0.0:
                    raise RemoteAgentError("the remote agent did not answer in time")
                self.sock.settimeout(left)
            try:
                chunk = self.sock.recv(65536)
            except TimeoutError as error:
                raise RemoteAgentError("the remote agent did not answer in time") from error
            except OSError as error:
                raise RemoteAgentError(
                    f"the connection to the remote agent failed: {error}"
                ) from error
            if not chunk:
                raise RemoteAgentError("the remote agent closed the connection")
            self.buffer += chunk
        line, _, self.buffer = self.buffer.partition(b"\n")
        if len(line) > MAX_LINE:
            raise RemoteAgentError(f"the remote agent sent a line longer than {MAX_LINE} bytes")
        return line

    def receive(self) -> dict[str, Any]:
        line = self._line()
        try:
            message = json.loads(line)
        except (ValueError, RecursionError) as error:
            # Covers malformed JSON, bytes that are not UTF-8, and nesting too deep to parse.
            raise RemoteAgentError("the remote agent sent a line that is not JSON") from error
        if not isinstance(message, dict):
            raise RemoteAgentError("the remote agent sent JSON that is not an object")
        if message.get("type") == "error":
            raise RemoteAgentError(f"remote agent error: {message.get('message', '')}")
        return message

    def close(self) -> None:
        self.sock.close()


class RemoteAgent:
    """A connected remote agent, used like any local agent."""

    def __init__(self, line: _Line, name: str):
        self.line = line
        self.name = name

    def _expect(self, kind: str) -> dict[str, Any]:
        message = self.line.receive()
        if message.get("type") != kind:
            raise RemoteAgentError(f"expected a {kind!r} message, got {message.get('type')!r}")
        return message

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        self.mode = view.action_mode
        seq = rng.bit_generator.seed_seq if rng is not None else np.random.SeedSequence()
        self.line.send(
            {
                "type": "reset",
                "view": view_to_dict(view),
                "seed": int(seq.entropy),  # type: ignore[attr-defined]
                "spawn_key": [int(k) for k in seq.spawn_key],  # type: ignore[attr-defined]
            }
        )
        self._expect("ready")

    def act(self, obs, info):
        self.line.send(
            {"type": "act", "obs": np.asarray(obs, dtype=float).tolist(), "info": dict(info)}
        )
        action = self._expect("action").get("action")
        if self.mode == "disc27":
            # Whole numbers of any type are indices, as `BaseEnv.step` reads them.
            whole = isinstance(action, int | float) and not isinstance(action, bool)
            # Range first: float() overflows on integers past 1e308.
            if not (whole and 0 <= action < 27 and float(action).is_integer()):
                raise RemoteAgentError(f"expected an integer action from 0 to 26, got {action!r}")
            return int(action)
        size = 4 if self.mode == "impulse_coast" else 3
        try:
            command = np.array(action, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise RemoteAgentError(f"expected {size} numbers, got {action!r}") from error
        if command.shape != (size,) or not np.all(np.isfinite(command)):
            raise RemoteAgentError(f"expected {size} finite numbers, got {action!r}")
        return command

    def close(self) -> None:
        # A client that already disconnected cannot be told to close.
        with contextlib.suppress(OSError):
            self.line.send({"type": "close"})
        self.line.close()


class AgentServer:
    """Listens on `host`:`port` (0 picks a free port) for remote agents.

    `timeout` (s) bounds every wait for a client's message.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0, timeout: float = 60.0):
        self.timeout = timeout
        self.sock = socket.create_server((host, port))

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.sock.getsockname()[:2]
        return host, port

    def accept(self) -> RemoteAgent:
        """Wait for one agent to connect and say hello."""
        self.sock.settimeout(self.timeout)
        try:
            conn, _ = self.sock.accept()
        except TimeoutError as error:
            raise RemoteAgentError("no agent connected in time") from error
        conn.settimeout(self.timeout)
        line = _Line(conn)
        try:
            hello = line.receive()
        except RemoteAgentError:
            line.close()
            raise
        version = hello.get("protocol")
        if hello.get("type") != "hello" or type(version) is not int or version != PROTOCOL:
            line.close()
            raise RemoteAgentError(f"expected a protocol {PROTOCOL} hello, got {hello}")
        return RemoteAgent(line, str(hello.get("name", "remote")))

    def close(self) -> None:
        self.sock.close()


def serve_agent(agent: Agent, host: str, port: int, timeout: float | None = None) -> None:
    """Connect to an agent server and run `agent` until the server closes the session."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        line = _Line(sock)
        line.send({"type": "hello", "name": agent.name, "protocol": PROTOCOL})
        while True:
            message = line.receive()
            kind = message.get("type")
            if kind == "reset":
                seq = np.random.SeedSequence(message["seed"], spawn_key=message["spawn_key"])
                agent.reset(view_from_dict(message["view"]), np.random.default_rng(seq))
                line.send({"type": "ready"})
            elif kind == "act":
                action = np.asarray(
                    agent.act(np.array(message["obs"], dtype=float), message["info"])
                )
                # Converted as `BaseEnv.step` converts it; one number (a discrete index in any type
                # or shape) goes as a number.
                try:
                    numbers = np.asarray(action, dtype=float)
                except (TypeError, ValueError, OverflowError):
                    line.send({"type": "error", "message": f"unusable action {action!r}"})
                    return
                reply: Any = numbers.item() if numbers.size == 1 else numbers.tolist()
                try:
                    line.send({"type": "action", "action": reply})
                except ValueError:
                    line.send(
                        {"type": "error", "message": "the agent returned a non-finite action"}
                    )
                    return
            elif kind == "close":
                return
            else:
                line.send({"type": "error", "message": f"unknown message type {kind!r}"})
                return


def main(argv: list[str] | None = None) -> None:
    """Host a scenario, the approach scenario by default, for one remote agent and write its
    results.

    However the session fails (at hello, during an episode, by a disconnect or a message the
    server cannot use), the episode being flown is written unfinished with the error, and so is
    every seed not yet flown, so a session cannot drop a bad episode.
    """

    from ..runner import ScoredRun
    from ..scenarios import ApproachConfig, ApproachEnv, make

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--scenario", help="catalogue name, name:variant or identifier; the approach by default"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="seeds to repeat; by default drawn and recorded"
    )
    parser.add_argument("--episodes", type=int, default=1, help="episodes with drawn seeds")
    parser.add_argument("--out", default="results.jsonl", help="JSON lines, one per episode")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    # A small seed can be found by matching the first observation, so seeds are drawn.
    seeds = args.seeds or [secrets.randbits(128) for _ in range(args.episodes)]
    try:
        env = make(args.scenario) if args.scenario else ApproachEnv(ApproachConfig())
    except (KeyError, ValueError) as error:
        parser.error(str(error.args[0]))
    server = AgentServer(args.host, args.port, args.timeout)
    print(f"waiting for an agent on {args.host}:{server.address[1]}", flush=True)

    def write(out, result: dict) -> None:
        plain = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
        plain["scenario_id"] = env.scenario_id
        out.write(json.dumps(plain) + "\n")
        out.flush()

    agent: RemoteAgent | None = None
    episode: ScoredRun | None = None
    written, reason = 0, "the session ended"
    with open(args.out, "w", encoding="utf-8") as out:
        try:
            agent = server.accept()
            for seed in seeds:
                episode = ScoredRun(env, seed)
                agent.reset(env.agent_view(), episode.agent_rng())
                while not episode.done:
                    t0 = time.perf_counter()
                    action = agent.act(episode.obs, env.agent_info())
                    episode.step(action, time.perf_counter() - t0)
                result = episode.result(agent.name)
                write(out, result)
                written, episode = written + 1, None
                print(f"seed {seed}: captured {result['captured']}", flush=True)
        except Exception as error:  # whatever the client did, the session is recorded below
            reason = str(error)
            print(f"session ended: {ascii(reason)}", flush=True)
        except BaseException as error:
            reason = f"interrupted ({type(error).__name__})"
            raise
        finally:
            # Also on an interrupt: the episode being flown and every seed left are written.
            name = agent.name if agent is not None else "unknown"
            for k, seed in enumerate(seeds[written:]):
                flown = episode if k == 0 and episode is not None else ScoredRun(env, seed)
                write(out, {**flown.result(name), "error": reason})
            if agent is not None:
                agent.close()
            server.close()


if __name__ == "__main__":
    main()
