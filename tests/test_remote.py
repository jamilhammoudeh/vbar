import dataclasses
import json
import os
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import DISC27, PDAgent
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
