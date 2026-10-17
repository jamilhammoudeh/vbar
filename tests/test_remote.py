import dataclasses
import json
import os
import shutil
import socket
import struct
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import DISC27, PDAgent, remote
from vbar.agents.base import AgentView
from vbar.agents.remote import AgentServer, RemoteAgentError, main, serve_agent
from vbar.spacecraft import GaussianNav

CLEAR = (-150.0, -400.0, 0.0, 0.0, 0.0, 0.0)
THROUGH = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)


class Wander:
    """Random commands from the agent's own stream, nudged by the observation."""

    name = "wander"

    def reset(self, view, rng=None):
        self.rng = rng

    def act(self, obs, info):
        return np.clip(self.rng.normal(0.0, 0.3, 3) - 1e-4 * obs[:3], -1.0, 1.0)


class Bang:
    """Bang-off-bang toward the hold point, in the discrete mode."""

    name = "bang"

    def reset(self, view, rng=None):
        self.hold = np.array(view.goal["hold_point"])

    def act(self, obs, info):
        direction = np.sign(np.round(self.hold - obs[:3], 0))
        return int(np.flatnonzero(np.all(DISC27 == direction, axis=1))[0])


def remote_run(agent, cfg, seed=0, agent_seed=7):
    server = AgentServer(timeout=20.0)
    host, port = server.address
    client = threading.Thread(target=serve_agent, args=(agent, host, port), daemon=True)
    client.start()
    remote = server.accept()
    try:
        return run_episode(ApproachEnv(cfg), remote, seed=seed, agent_seed=agent_seed), remote
    finally:
        remote.close()
        server.close()
        client.join(5.0)


@pytest.mark.parametrize(
    ("make", "fields"),
    [
        (PDAgent, dict(start_state=CLEAR)),
        (PDAgent, dict(start_state=THROUGH, action_mode="accel")),
        (Wander, dict(max_steps=60, sensor=GaussianNav(3.0, 3e-3))),
        (Bang, dict(start_state=CLEAR, max_steps=60, action_mode="disc27")),
    ],
)
def test_remote_runs_match_in_process_runs_exactly(make, fields):
    cfg = ApproachConfig(**fields)
    local = run_episode(ApproachEnv(cfg), make(), seed=3, agent_seed=7)
    remote, agent = remote_run(make(), cfg, seed=3, agent_seed=7)
    assert agent.name == make().name
    # JSON carries floats in their shortest round-trip form, so the runs are identical.
    assert np.array_equal(local["states"], remote["states"])
    assert np.array_equal(local["actions"], remote["actions"])
    assert local["captured"] == remote["captured"]


def raw_session(handle, timeout=20.0):
    """Run `handle(file)` as a hand-written client against a one-step episode."""
    server = AgentServer(timeout=timeout)
    seen = {}

    def client():
        with socket.create_connection(server.address) as sock:
            stream = sock.makefile("rwb")
            handle(stream, seen)

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    try:
        remote = server.accept()
        try:
            run_episode(ApproachEnv(ApproachConfig(max_steps=1)), remote, seed=0, agent_seed=5)
        finally:
            remote.close()
    finally:
        server.close()
        thread.join(5.0)
    return seen


def send(stream, message):
    stream.write(json.dumps(message).encode() + b"\n")
    stream.flush()


def receive(stream):
    return json.loads(stream.readline())


def test_the_server_sends_only_what_an_in_process_agent_receives():
    def handle(stream, seen):
        send(stream, {"type": "hello", "name": "spy", "protocol": 1})
        seen["reset"] = receive(stream)
        send(stream, {"type": "ready"})
        seen["act"] = receive(stream)
        send(stream, {"type": "action", "action": [0.0, 0.0, 0.0]})
        seen["close"] = receive(stream)

    seen = raw_session(handle)
    assert set(seen["reset"]) == {"type", "view", "seed", "spawn_key"}
    assert set(seen["reset"]["view"]) == {f.name for f in dataclasses.fields(AgentView)}
    assert seen["reset"]["seed"] == 5 and seen["reset"]["spawn_key"] == []
    assert set(seen["act"]) == {"type", "obs", "info"} and set(seen["act"]["info"]) == {"time_s"}
    assert seen["close"] == {"type": "close"}


def test_protocol_errors_and_silence_are_reported():
    def refuses(stream, seen):
        send(stream, {"type": "hello", "name": "x", "protocol": 1})
        receive(stream)
        send(stream, {"type": "error", "message": "cannot fly this"})

    with pytest.raises(RemoteAgentError, match="cannot fly this"):
        raw_session(refuses)

    def wrong_version(stream, seen):
        send(stream, {"type": "hello", "name": "x", "protocol": 2})

    with pytest.raises(RemoteAgentError, match="protocol 1 hello"):
        raw_session(wrong_version)

    def silent(stream, seen):
        send(stream, {"type": "hello", "name": "x", "protocol": 1})
        receive(stream)
        stream.readline()

    with pytest.raises(RemoteAgentError, match="did not answer in time"):
        raw_session(silent, timeout=0.3)


def test_command_line_host_writes_one_result_per_seed(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = tmp_path / "results.jsonl"
    host = threading.Thread(
        target=main, args=(["--port", str(port), "--seeds", "1", "2", "--out", str(out)],)
    )
    host.start()
    for _ in range(100):
        try:
            serve_agent(PDAgent(), "127.0.0.1", port)
            break
        except ConnectionRefusedError:
            threading.Event().wait(0.05)
    host.join(30.0)
    results = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["seed"] for r in results] == [1, 2] and all(r["agent"] == "pd" for r in results)


@pytest.mark.skipif(
    not (os.environ.get("VBAR_TEST_MATLAB") and shutil.which("matlab")),
    reason="needs MATLAB and VBAR_TEST_MATLAB=1 (about a minute to start MATLAB)",
)
def test_matlab_pd_client_flies_the_python_pd_trajectories(tmp_path):
    server = AgentServer(timeout=300.0)
    port = server.address[1]
    clients = Path(__file__).parents[1] / "clients" / "matlab"
    command = f"addpath('{clients.as_posix()}'); vbar_agent_client('127.0.0.1', {port}, pd_agent())"
    matlab = subprocess.Popen(["matlab", "-batch", command])
    remote = server.accept()
    try:
        for seed in (0, 1):
            flown = run_episode(ApproachEnv(), remote, seed=seed)
            local = run_episode(ApproachEnv(), PDAgent(), seed=seed)
            assert np.array_equal(flown["states"], local["states"])
    finally:
        remote.close()
        server.close()
        matlab.wait(120)


def test_command_line_host_records_episodes_a_client_abandons(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = tmp_path / "results.jsonl"
    host = threading.Thread(
        target=main, args=(["--port", str(port), "--seeds", "1", "2", "--out", str(out)],)
    )
    host.start()
    for _ in range(100):
        try:
            sock = socket.create_connection(("127.0.0.1", port))
            break
        except ConnectionRefusedError:
            threading.Event().wait(0.05)
    with sock:
        stream = sock.makefile("rwb")
        send(stream, {"type": "hello", "name": "quitter", "protocol": 1})
        receive(stream)
        send(stream, {"type": "ready"})
        for _ in range(2):
            receive(stream)
            send(stream, {"type": "action", "action": [0.0, 0.0, 0.0]})
        receive(stream)
        stream.write(b"this is not json\n")
        stream.flush()
    host.join(30.0)
    results = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["seed"], r["completed"], len(r["actions"])) for r in results] == [
        (1, False, 2),
        (2, False, 0),
    ]
    assert "error" in results[0]


def host_with(client, tmp_path):
    """Run the command-line host for seeds 1 and 2 against `client(stream, sock)`."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = tmp_path / "results.jsonl"
    argv = ["--port", str(port), "--seeds", "1", "2", "--out", str(out), "--timeout", "10"]
    host = threading.Thread(target=main, args=(argv,))
    host.start()
    for _ in range(100):
        try:
            sock = socket.create_connection(("127.0.0.1", port))
            break
        except ConnectionRefusedError:
            threading.Event().wait(0.05)
    stream = sock.makefile("rwb")
    send(stream, {"type": "hello", "name": "faulty", "protocol": 1})
    receive(stream)
    send(stream, {"type": "ready"})
    receive(stream)
    send(stream, {"type": "action", "action": [0.0, 0.0, 0.0]})
    receive(stream)
    client(stream, sock)
    host.join(30.0)
    return [json.loads(line) for line in out.read_text().splitlines()]


@pytest.mark.parametrize(
    "line",
    [
        b'{"type": "action", "action": [1.0, 2.0]}',
        b'{"type": "action", "action": "abc"}',
        b'{"type": "action"}',
        b'{"type": "action", "action": [NaN, 0.0, 0.0]}',
    ],
)
def test_a_malformed_action_ends_the_session_with_every_episode_recorded(line, tmp_path):
    def client(stream, sock):
        stream.write(line + b"\n")
        stream.flush()
        stream.close()
        sock.close()

    results = host_with(client, tmp_path)
    assert [(r["seed"], r["completed"], len(r["actions"])) for r in results] == [
        (1, False, 1),
        (2, False, 0),
    ]
    assert np.all(np.isfinite(results[0]["states"]))


# The host closes into the reset connection; that must not raise in its thread.
@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_reset_connection_still_leaves_every_episode_recorded(tmp_path):
    def client(stream, sock):
        # An abortive close, as when the client process is killed.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        # The stream holds the socket open; both must close for the reset to be sent.
        stream.close()
        sock.close()

    results = host_with(client, tmp_path)
    assert [(r["seed"], r["completed"]) for r in results] == [(1, False), (2, False)]


class Hops:
    """Burn-and-coast decisions: a short burn toward the hold point, then a minute of coast."""

    name = "hops"

    def reset(self, view, rng=None):
        self.hold = np.array(view.goal["hold_point"])

    def act(self, obs, info):
        return np.r_[np.clip(1e-3 * (self.hold - obs[:3]), -1.0, 1.0), 60.0]


class Wrong:
    """Returns the action of another mode."""

    name = "wrong"

    def __init__(self, action):
        self.action = action

    def reset(self, view, rng=None):
        pass

    def act(self, obs, info):
        return self.action


def test_every_action_mode_crosses_the_server_and_wrong_shapes_are_refused():
    cfg = ApproachConfig(start_state=CLEAR, max_steps=40, action_mode="impulse_coast")
    local = run_episode(ApproachEnv(cfg), Hops(), seed=0, agent_seed=1)
    remote, _ = remote_run(Hops(), cfg, seed=0, agent_seed=1)
    assert np.array_equal(local["states"], remote["states"])
    for mode, action in (
        ("impulse_coast", [0.0, 0.0, 0.0]),
        ("normalized", [0.0] * 4),
        ("disc27", 27),
    ):
        with pytest.raises(RemoteAgentError, match="expected"):
            remote_run(Wrong(action), ApproachConfig(max_steps=2, action_mode=mode))


def test_a_send_that_fails_mid_episode_still_leaves_every_episode_recorded(tmp_path, monkeypatch):
    # The client resets just after answering, so the host's next message cannot be sent.
    original = remote._Line.send
    sent = []

    def send_until_reset(self, message):
        if message.get("type") == "act":
            sent.append(message)
            if len(sent) == 2:
                raise ConnectionResetError("connection reset by the client")
        return original(self, message)

    monkeypatch.setattr(remote._Line, "send", send_until_reset)

    def client(stream, sock):
        send(stream, {"type": "action", "action": [0.0, 0.0, 0.0]})
        stream.close()
        sock.close()

    results = host_with(client, tmp_path)
    assert [(r["seed"], r["completed"], len(r["actions"])) for r in results] == [
        (1, False, 1),
        (2, False, 0),
    ]
    assert "reset" in results[0]["error"]
