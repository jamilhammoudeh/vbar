import io
import json
import subprocess
import sys

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent
from vbar.agents.mcp import METHOD_NOT_FOUND, PARSE_ERROR, ToolServer
from vbar.agents.tools import predict_coast


def request(server, method, ident=1, **params):
    return server.handle({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})


def call(server, name, **arguments):
    result = request(server, "tools/call", name=name, arguments=arguments)["result"]
    return result, result.get("structuredContent")


def test_handshake_lists_the_tools_and_answers_errors():
    server = ToolServer(ApproachEnv(), [0])
    init = request(server, "initialize", protocolVersion="2024-11-05")["result"]
    assert init["protocolVersion"] == "2024-11-05" and init["capabilities"] == {"tools": {}}
    assert (
        request(server, "initialize", protocolVersion="1999-01-01")["result"]["protocolVersion"]
        == "2025-06-18"
    )
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = [t["name"] for t in request(server, "tools/list")["result"]["tools"]]
    assert names == [
        "describe_task",
        "start_episode",
        "observe",
        "command",
        "predict_coast",
        "transfer_burn",
    ]
    assert request(server, "resources/list")["error"]["code"] == METHOD_NOT_FOUND
    out = io.StringIO()
    server.serve(io.StringIO("not json\n"), out)
    assert json.loads(out.getvalue())["error"]["code"] == PARSE_ERROR


def test_an_episode_flown_through_tool_calls_is_the_in_process_episode():
    env = ApproachEnv(ApproachConfig(start_state=(-150.0, -400.0, 0.0, 0.0, 0.0, 0.0)))
    server = ToolServer(env, [3], agent_name="pd-over-mcp")
    pd = PDAgent()
    pd.reset(env.agent_view())
    _, snapshot = call(server, "start_episode")
    while True:
        obs = np.array(list(snapshot["observation"].values()))
        ax, ay, az = pd.act(obs, {"time_s": snapshot["time_s"]})
        _, snapshot = call(server, "command", ax=ax, ay=ay, az=az)
        if snapshot["episode_over"]:
            break
        assert set(snapshot) == {"episode_over", "observation", "time_s", "steps_left", "episode"}
    local = run_episode(ApproachEnv(env.cfg), PDAgent(), seed=3)
    (served,) = server.results
    assert served["agent"] == "pd-over-mcp" and served["captured"]
    assert np.array_equal(served["states"], local["states"])
    assert np.array_equal(served["actions"], local["actions"])
    assert snapshot == {"episode_over": True, "episodes_left": 0}


def test_episodes_run_in_order_and_cannot_be_restarted_midway():
    server = ToolServer(ApproachEnv(ApproachConfig(max_steps=2)), [0])
    result, _ = call(server, "observe")
    assert result["isError"] and "start_episode" in result["content"][0]["text"]
    call(server, "start_episode")
    result, _ = call(server, "start_episode")
    assert result["isError"] and "has not ended" in result["content"][0]["text"]
    result, _ = call(server, "command", ax="fast", ay=0, az=0)
    assert result["isError"]
    call(server, "command", ax=0, ay=0, az=0)
    call(server, "command", ax=0, ay=0, az=0)
    result, _ = call(server, "start_episode")
    assert result["isError"] and "no episodes left" in result["content"][0]["text"]
    assert len(server.results) == 1


def test_predictions_come_from_the_current_observation():
    env = ApproachEnv()
    server = ToolServer(env, [5])
    _, snapshot = call(server, "start_episode")
    _, found = call(server, "predict_coast", seconds=300.0)
    obs = np.array(list(snapshot["observation"].values()))
    assert found == predict_coast(obs, env.agent_view(), {"seconds": 300.0})
    result, _ = call(server, "predict_coast", seconds=-1.0)
    assert result["isError"]
    _, task = call(server, "describe_task")
    assert task["view"]["goal"]["hold_point"] == [0.0, -100.0, 0.0] and task["episodes"] == 1


def test_command_line_server_speaks_json_rpc_on_stdio(tmp_path):
    # A real subprocess, about 3 s, nearly all of it importing numpy and scipy.
    out = tmp_path / "results.jsonl"
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "start_episode"}},
        *[
            {
                "jsonrpc": "2.0",
                "id": 3 + k,
                "method": "tools/call",
                "params": {"name": "command", "arguments": {"ax": 0, "ay": -0.1, "az": 0}},
            }
            for k in range(3)
        ],
    ]
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "vbar.agents.mcp",
            "--seeds",
            "4",
            "--max-steps",
            "3",
            "--out",
            str(out),
        ],
        input="\n".join(json.dumps(m) for m in lines) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    responses = [json.loads(line) for line in done.stdout.splitlines()]
    assert [r["id"] for r in responses] == [1, 2, 3, 4, 5]
    assert responses[-1]["result"]["structuredContent"]["episode_over"]
    (result,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert result["seed"] == 4 and len(result["actions"]) == 3
    assert result["dv_used_mps"] == pytest.approx(0.06, rel=1e-3)
