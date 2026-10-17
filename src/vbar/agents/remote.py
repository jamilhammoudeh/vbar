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
import socket
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import numpy as np

from .base import Agent, AgentView

PROTOCOL = 1


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
    """One JSON object per line over a socket."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.file = sock.makefile("rwb")

    def send(self, message: Mapping[str, Any]) -> None:
        text = json.dumps(message, allow_nan=False, separators=(",", ":"))
        self.file.write(text.encode("utf-8") + b"\n")
        self.file.flush()

    def receive(self) -> dict[str, Any]:
        try:
            line = self.file.readline()
        except TimeoutError as error:
            raise RemoteAgentError("the remote agent did not answer in time") from error
        except OSError as error:
            raise RemoteAgentError(f"the connection to the remote agent failed: {error}") from error
        if not line:
            raise RemoteAgentError("the remote agent closed the connection")
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
        # Closing flushes; a peer that reset the connection cannot take the last bytes.
        with contextlib.suppress(OSError):
            self.file.close()
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
            if not (whole and float(action).is_integer() and 0 <= action < 27):
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
        hello = line.receive()
        if hello.get("type") != "hello" or hello.get("protocol") != PROTOCOL:
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
                action = agent.act(np.array(message["obs"], dtype=float), message["info"])
                if isinstance(action, int | np.integer):
                    reply: Any = int(action)
                else:
                    reply = np.asarray(action, dtype=float).tolist()
                line.send({"type": "action", "action": reply})
            elif kind == "close":
                return
            else:
                line.send({"type": "error", "message": f"unknown message type {kind!r}"})
                return


def main(argv: list[str] | None = None) -> None:
    """Host the default approach scenario for one remote agent and write its results.

    However the session fails (at hello, during an episode, by a disconnect or a message the
    server cannot use), the episode being flown is written unfinished with the error, and so is
    every seed not yet flown, so a session cannot drop a bad episode.
    """
    import time

    from ..runner import Episode
    from ..scenarios import ApproachConfig, ApproachEnv

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out", default="results.jsonl", help="JSON lines, one per episode")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    server = AgentServer(args.host, args.port, args.timeout)
    print(f"waiting for an agent on {args.host}:{server.address[1]}", flush=True)
    env = ApproachEnv(ApproachConfig())

    def write(out, result: dict) -> None:
        plain = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
        out.write(json.dumps(plain) + "\n")
        out.flush()

    agent: RemoteAgent | None = None
    with open(args.out, "w", encoding="utf-8") as out:
        try:
            agent = server.accept()
        except Exception as error:  # nothing was flown; every seed is recorded unflown
            for seed in args.seeds:
                write(out, {**Episode(env, seed).result("unknown"), "error": str(error)})
            print(f"no session: {error}", flush=True)
            server.close()
            return
        try:
            for k, seed in enumerate(args.seeds):
                episode = Episode(env, seed)
                try:
                    agent.reset(env.agent_view(), episode.agent_rng())
                    while not episode.done:
                        t0 = time.perf_counter()
                        action = agent.act(episode.obs, env.agent_info())
                        episode.step(action, time.perf_counter() - t0)
                except Exception as error:  # whatever the client did, the episode is recorded
                    write(out, {**episode.result(agent.name), "error": str(error)})
                    for left in args.seeds[k + 1 :]:
                        write(out, Episode(env, left).result(agent.name))
                    print(f"seed {seed}: {error}", flush=True)
                    return
                write(out, episode.result(agent.name))
                print(f"seed {seed}: captured {episode.info['captured']}", flush=True)
        finally:
            agent.close()
            server.close()


if __name__ == "__main__":
    main()
