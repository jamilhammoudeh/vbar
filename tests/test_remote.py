import dataclasses
import json
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import DISC27, PDAgent, remote
from vbar.agents.base import AgentView, agent_generator
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
    served, agent = remote_run(make(), cfg, seed=3, agent_seed=7)
    assert agent.name == make().name
    # JSON carries floats in their shortest round-trip form, so the runs are identical.
    assert np.array_equal(local["states"], served["states"])
    assert np.array_equal(local["actions"], served["actions"])
    assert local["captured"] == served["captured"]


def raw_session(handle, timeout=20.0, config=None):
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
            env = ApproachEnv(config or ApproachConfig(max_steps=1))
            run_episode(env, remote, seed=0, agent_seed=5)
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
    # The seed is the hashed one behind an in-process agent's generator, not agent_seed 5.
    assert seen["reset"]["seed"] == agent_generator(5).bit_generator.seed_seq.entropy
    assert seen["reset"]["spawn_key"] == []
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


def host_with(client, tmp_path, timeout=10.0, episodes=("--seeds", "1", "2")):
    """Run the command-line host for seeds 1 and 2 against `client(stream, sock)`."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = tmp_path / "results.jsonl"
    argv = ["--port", str(port), *episodes, "--out", str(out), "--timeout", str(timeout)]
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


def test_the_host_draws_unguessable_seeds_unless_given_some(tmp_path):
    def client(stream, sock):
        stream.close()
        sock.close()

    results = host_with(client, tmp_path, episodes=("--episodes", "2"))
    seeds = [r["seed"] for r in results]
    # 128-bit draws: below 2**64 with probability 2**-64 each.
    assert len(set(seeds)) == 2 and min(seeds) >= 2**64


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


def test_burn_and_coast_actions_cross_the_server_and_wrong_shapes_are_refused():
    cfg = ApproachConfig(start_state=CLEAR, max_steps=40, action_mode="impulse_coast")
    local = run_episode(ApproachEnv(cfg), Hops(), seed=0, agent_seed=1)
    served, _ = remote_run(Hops(), cfg, seed=0, agent_seed=1)
    assert np.array_equal(local["states"], served["states"])
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


@pytest.mark.parametrize(
    "line",
    [
        b'{"type": "action", "action": [1' + b"0" * 400 + b", 0, 0]}",
        b'{"type": "action", "action": ' + b"[" * 200_000 + b"]" * 200_000 + b"}",
        b'{"type": "action", "action": [0, 0, 0], "x": "\xff\xfe"}',
    ],
    ids=["huge integer", "deep nesting", "invalid utf-8"],
)
def test_hostile_action_lines_still_leave_every_episode_recorded(line, tmp_path):
    def client(stream, sock):
        stream.write(line + b"\n")
        stream.flush()
        stream.close()
        sock.close()

    results = host_with(client, tmp_path)
    assert [(r["seed"], r["completed"]) for r in results] == [(1, False), (2, False)]


def test_a_failed_hello_records_every_seed_unflown(tmp_path):
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
    with sock:
        stream = sock.makefile("rwb")
        send(stream, {"type": "hello", "name": "old", "protocol": 0})
        stream.close()
    host.join(30.0)
    results = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["seed"], r["completed"], len(r["actions"])) for r in results] == [
        (1, False, 0),
        (2, False, 0),
    ]


def test_remote_discrete_indices_follow_the_environment_rule():
    # Whole numbers in any type are indices, as in process; fractions and -1 are refused.
    cfg = ApproachConfig(max_steps=2, action_mode="disc27")
    remote_run(Wrong(13.0), cfg)
    for action in (13.5, -1):
        with pytest.raises(RemoteAgentError, match="expected"):
            remote_run(Wrong(action), cfg)


def test_a_message_sent_byte_by_byte_still_times_out_as_one_message(tmp_path):
    # The timeout bounds the whole message, not each byte, so trickling cannot stall the host.
    took = []

    def client(stream, sock):
        start = time.perf_counter()
        try:
            for byte in b'{"type": "action", "action": [0.0, 0.0, 0.0]}':
                sock.sendall(bytes([byte]))
                time.sleep(0.25)
        except OSError:
            pass
        took.append(time.perf_counter() - start)
        stream.close()
        sock.close()

    results = host_with(client, tmp_path, timeout=1.0)
    assert [(r["seed"], r["completed"]) for r in results] == [(1, False), (2, False)]
    assert "in time" in results[0]["error"]


def test_a_line_longer_than_the_limit_is_refused(tmp_path):
    def client(stream, sock):
        try:
            stream.write(
                b'{"type": "action", "action": "' + b"x" * (remote.MAX_LINE + 10) + b'"}\n'
            )
            stream.flush()
        except OSError:
            pass
        stream.close()
        sock.close()

    results = host_with(client, tmp_path)
    assert [(r["seed"], r["completed"]) for r in results] == [(1, False), (2, False)]
    assert "longer than" in results[0]["error"]


def test_an_interrupt_mid_episode_still_leaves_every_episode_recorded(tmp_path, monkeypatch):
    # An operator's Ctrl+C is not an Exception; the session is recorded on the way out anyway.
    original = remote.RemoteAgent.act
    calls = []

    def interrupted(self, obs, info):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(self, obs, info)

    monkeypatch.setattr(remote.RemoteAgent, "act", interrupted)
    raised = []

    def run_main(argv):
        try:
            main(argv)
        except KeyboardInterrupt:
            raised.append(True)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = tmp_path / "results.jsonl"
    argv = ["--port", str(port), "--seeds", "1", "2", "--out", str(out)]
    host = threading.Thread(target=run_main, args=(argv,))
    host.start()
    for _ in range(100):
        try:
            serve_agent(PDAgent(), "127.0.0.1", port)
            break
        except ConnectionRefusedError:
            threading.Event().wait(0.05)
        except RemoteAgentError:
            break
    host.join(30.0)
    results = [json.loads(line) for line in out.read_text().splitlines()]
    assert raised == [True]
    assert [(r["seed"], r["completed"], len(r["actions"])) for r in results] == [
        (1, False, 1),
        (2, False, 0),
    ]


def test_one_number_actions_cross_the_server_in_any_shape():
    # A policy batched to shape (1,) returns array([13]); in process that is index 13.
    cfg = ApproachConfig(start_state=CLEAR, max_steps=3, action_mode="disc27")
    local = run_episode(ApproachEnv(cfg), Wrong(np.array([13])), seed=0, agent_seed=1)
    served, _ = remote_run(Wrong(np.array([13])), cfg, seed=0, agent_seed=1)
    assert np.array_equal(local["states"], served["states"])
    with pytest.raises(RemoteAgentError, match="non-finite"):
        remote_run(Wrong(np.array([np.nan, 0.0, 0.0])), ApproachConfig(max_steps=2))


def raw_host_session(lines_after_hello, timeout=5.0):
    """Send raw bytes to `AgentServer.accept` and return the agent or the error."""
    server = AgentServer(timeout=timeout)
    outcome = {}

    def client():
        with socket.create_connection(server.address) as sock:
            sock.sendall(lines_after_hello)
            outcome["closed"] = sock.recv(1) == b""

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    try:
        outcome["agent"] = server.accept()
    except RemoteAgentError as error:
        outcome["error"] = str(error)
    finally:
        server.close()
    if "agent" in outcome:
        outcome["agent"].close()
    thread.join(5.0)
    return outcome


def test_messages_sent_together_are_read_one_at_a_time():
    # The hello and an early "ready" in one packet: the second stays buffered for the reset.
    server = AgentServer(timeout=5.0)
    received = {}

    def client():
        with socket.create_connection(server.address) as sock:
            hello = {"type": "hello", "name": "eager", "protocol": 1}
            sock.sendall((json.dumps(hello) + "\n" + json.dumps({"type": "ready"}) + "\n").encode())
            stream = sock.makefile("rwb")
            received["reset"] = receive(stream)
            send(stream, {"type": "action", "action": [0.0, 0.0, 0.0]})
            receive(stream)

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    agent = server.accept()
    try:
        agent.reset(ApproachEnv().agent_view(), np.random.default_rng(0))
        assert np.array_equal(agent.act(np.zeros(6), {"time_s": 0.0}), np.zeros(3))
    finally:
        agent.close()
        server.close()
    thread.join(5.0)
    assert received["reset"]["type"] == "reset"


def test_hellos_that_fail_close_the_connection():
    long = b'{"type": "hello", "name": "' + b"x" * (remote.MAX_LINE + 10)
    for raw in (b'{"type": "hello", "name": "x", "protocol": true}\n', long):
        outcome = raw_host_session(raw)
        assert "error" in outcome and outcome["closed"]
    assert "longer than" in raw_host_session(long)["error"]


def test_the_client_converts_actions_as_the_environment_does():
    # In process, True and "13" are read as numbers; through the client they must be too.
    cfg = ApproachConfig(start_state=CLEAR, max_steps=3, action_mode="disc27")
    for action in ("13", 13):
        local = run_episode(ApproachEnv(cfg), Wrong(action), seed=0, agent_seed=1)
        served, _ = remote_run(Wrong(action), cfg, seed=0, agent_seed=1)
        assert np.array_equal(local["states"], served["states"])
    with pytest.raises(RemoteAgentError, match="unusable action"):
        remote_run(Wrong([int("9" * 400), 0, 0]), ApproachConfig(max_steps=2))


def test_a_huge_discrete_index_is_a_protocol_error():
    # Sent raw, past the Python client's own conversion: the server must refuse it itself.

    def handle(stream, seen):
        send(stream, {"type": "hello", "name": "huge", "protocol": 1})
        receive(stream)
        send(stream, {"type": "ready"})
        receive(stream)
        send(stream, {"type": "action", "action": int("9" * 400)})

    with pytest.raises(RemoteAgentError, match="expected an integer action"):
        raw_session(handle, config=ApproachConfig(max_steps=1, action_mode="disc27"))
