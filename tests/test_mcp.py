import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent
from vbar.agents.mcp import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    ToolServer,
    main,
)
from vbar.agents.remote import view_to_dict
from vbar.agents.tools import predict_coast

# Only a hang should reach this; a loaded machine took 143 s for what takes 3 s idle.
HANG = 600.0


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
        expected = {"episode_over", "observation", "time_s", "situation", "steps_left", "episode"}
        assert set(snapshot) == expected
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
        timeout=HANG,
    )
    responses = [json.loads(line) for line in done.stdout.splitlines()]
    assert [r["id"] for r in responses] == [1, 2, 3, 4, 5]
    assert responses[-1]["result"]["structuredContent"]["episode_over"]
    (result,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert result["seed"] == 4 and len(result["actions"]) == 3
    assert result["dv_used_mps"] == pytest.approx(0.06, rel=1e-3)


@pytest.mark.parametrize("count", ["0", "-2"])
def test_the_command_line_refuses_fewer_than_one_episode(count, capsys):
    with pytest.raises(SystemExit):
        main(["--episodes", count])
    assert "--episodes must be at least 1" in capsys.readouterr().err


def test_the_command_line_draws_unguessable_seeds_unless_given_some(tmp_path):
    out = tmp_path / "results.jsonl"
    argv = ["--episodes", "2", "--max-steps", "1", "--out", str(out)]
    command = [sys.executable, "-m", "vbar.agents.mcp", *argv]
    subprocess.run(command, input="", capture_output=True, text=True, timeout=HANG, check=True)
    seeds = [json.loads(line)["seed"] for line in out.read_text().splitlines()]
    # Both episodes are recorded unfinished; 128-bit draws fall below 2**64 with probability 2**-64.
    assert len(set(seeds)) == 2 and min(seeds) >= 2**64


@pytest.mark.parametrize(
    "message",
    [
        {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": ["command"]},
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "command", "arguments": [1]},
        },
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "command", "arguments": "abc"},
        },
        {"jsonrpc": "2.0", "id": 9, "method": "initialize", "params": [1, 2]},
    ],
)
def test_malformed_requests_get_errors_and_the_server_keeps_answering(message):
    server = ToolServer(ApproachEnv(), [0])
    call(server, "start_episode")
    out = io.StringIO()
    ping = {"jsonrpc": "2.0", "id": 10, "method": "ping"}
    server.serve(io.StringIO(json.dumps(message) + "\n" + json.dumps(ping) + "\n"), out)
    first, second = (json.loads(line) for line in out.getvalue().splitlines())
    assert first["error"]["code"] == INVALID_PARAMS and second == {
        "jsonrpc": "2.0",
        "id": 10,
        "result": {},
    }


def test_an_abandoned_session_still_records_every_episode():
    # A client cannot drop an episode that is going badly by closing the connection.
    results = []
    server = ToolServer(ApproachEnv(), [0, 1, 2], on_result=results.append)
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "start_episode"}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "command", "arguments": {"ax": 0, "ay": 0, "az": 0}},
        },
    ]
    server.serve(io.StringIO("".join(json.dumps(m) + "\n" for m in lines)), io.StringIO())
    assert [r["seed"] for r in results] == [0, 1, 2]
    assert [r["completed"] for r in results] == [False, False, False]
    assert [len(r["actions"]) for r in results] == [1, 0, 0]
    assert not any(r["captured"] for r in results)


class Hangup(io.StringIO):
    """Output whose reader has gone after the first reply."""

    def write(self, text):
        if self.getvalue():
            raise BrokenPipeError("reader closed")
        return super().write(text)


def test_a_client_that_stops_reading_still_leaves_every_episode_recorded():
    results = []
    server = ToolServer(ApproachEnv(), [0, 1], on_result=results.append)
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "start_episode"}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "command", "arguments": {"ax": 0, "ay": 0, "az": 0}},
        },
    ]
    server.serve(io.StringIO("".join(json.dumps(m) + "\n" for m in lines)), Hangup())
    assert [(r["seed"], r["completed"], len(r["actions"])) for r in results] == [
        (0, False, 1),
        (1, False, 0),
    ]


def test_a_tool_name_that_is_not_a_string_and_a_failing_tool_get_errors():
    server = ToolServer(ApproachEnv(), [0])
    assert request(server, "tools/call", name=["command"])["error"]["code"] == INVALID_PARAMS

    def broken(arguments):
        raise RuntimeError("tool failed")

    server.tools["observe"] = ("broken", {"type": "object"}, broken)
    out = io.StringIO()
    message = {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "observe"}}
    server.serve(io.StringIO(json.dumps(message) + "\n"), out)
    error = json.loads(out.getvalue())["error"]
    assert error["code"] == INTERNAL_ERROR and "tool failed" in error["message"]


def test_deeply_nested_input_is_a_parse_error_and_the_server_goes_on():
    server = ToolServer(ApproachEnv(), [0])
    out = io.StringIO()
    ping = json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping"})
    server.serve(io.StringIO("[" * 200_000 + "]" * 200_000 + "\n" + ping + "\n"), out)
    first, second = (json.loads(line) for line in out.getvalue().splitlines())
    assert first["error"]["code"] == PARSE_ERROR and second["id"] == 9


def test_a_non_finite_request_id_gets_a_strict_json_error():
    server = ToolServer(ApproachEnv(), [0])
    out = io.StringIO()
    server.serve(io.StringIO('{"jsonrpc": "2.0", "id": NaN, "method": "ping"}\n'), out)
    reply = json.loads(out.getvalue(), parse_constant=lambda name: pytest.fail(name))
    assert reply["error"]["code"] == -32600 and reply["id"] is None


def test_a_non_finite_prediction_is_a_tool_error_that_keeps_its_id():
    server = ToolServer(ApproachEnv(), [0])
    call(server, "start_episode")
    out = io.StringIO()
    huge = {"x": 1e308, "y": -1e308, "z": 1e308, "seconds": 600.0}
    message = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "transfer_burn", "arguments": huge},
    }
    server.serve(io.StringIO(json.dumps(message) + "\n"), out)
    reply = json.loads(out.getvalue())
    assert reply["id"] == 2 and reply["result"]["isError"]
    assert "not finite" in reply["result"]["content"][0]["text"]


@pytest.mark.parametrize("ident", ["[NaN]", '{"a": Infinity}', "true"])
def test_ids_that_are_not_strings_numbers_or_null_are_refused_in_strict_json(ident):
    server = ToolServer(ApproachEnv(), [0])
    out = io.StringIO()
    server.serve(io.StringIO('{"jsonrpc": "2.0", "id": ' + ident + ', "method": "ping"}\n'), out)
    reply = json.loads(out.getvalue(), parse_constant=lambda name: pytest.fail(name))
    assert reply["id"] is None and reply["error"]["code"] == -32600


def test_a_result_that_cannot_be_strict_json_keeps_its_request_id():
    server = ToolServer(ApproachEnv(), [0])
    server.tools["observe"] = ("broken", {"type": "object"}, lambda arguments: {"x": float("inf")})
    out = io.StringIO()
    message = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "observe"}}
    server.serve(io.StringIO(json.dumps(message) + "\n"), out)
    reply = json.loads(out.getvalue())
    assert reply["id"] == 7 and reply["error"]["code"] == INTERNAL_ERROR


def test_a_command_too_large_for_a_float_is_a_tool_error():
    server = ToolServer(ApproachEnv(), [0])
    call(server, "start_episode")
    result, _ = call(server, "command", ax=int("9" * 400), ay=0, az=0)
    assert result["isError"]


@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_the_tool_server_refuses_modes_without_a_continuous_command(mode):
    with pytest.raises(ValueError, match=f"'{mode}' takes no continuous command"):
        ToolServer(ApproachEnv(ApproachConfig(action_mode=mode)), [0])


@pytest.mark.parametrize("name", ["inspection", "docking-offset"])
def test_the_task_described_is_the_episode_being_flown(name):
    # Both scenarios draw their goal at each reset: the Sun direction, the port's attitude.
    from vbar.scenarios import make

    env = make(name)
    built = env.agent_view()
    server = ToolServer(env, [5])
    call(server, "start_episode")
    _, task = call(server, "describe_task")
    assert task["view"]["goal"] == view_to_dict(env.agent_view())["goal"]
    assert task["view"]["goal"] != view_to_dict(built)["goal"]


def test_the_task_and_each_observation_come_with_the_episode_s_briefing_and_situation():
    # docking-offset draws its target's pitch, and so its port, at each reset.
    from vbar.agents.briefing import briefing, situation
    from vbar.scenarios import make

    env = make("docking-offset")
    server = ToolServer(env, [5])
    _, before = call(server, "describe_task")
    _, snapshot = call(server, "start_episode")
    assert snapshot["briefing"] == briefing(env.agent_view()) != before["briefing"]
    _, task = call(server, "describe_task")
    assert task["briefing"] == briefing(env.agent_view())
    assert task["briefing"].startswith("Task: dock at the target's port")
    obs = np.array(list(snapshot["observation"].values()))
    assert snapshot["situation"] == situation(env.agent_view(), obs, env.agent_info())
    assert "lateral offset from the axis" in snapshot["situation"]
    _, snapshot = call(server, "command", ax=0.0, ay=0.0, az=0.0)
    obs = np.array(list(snapshot["observation"].values()))
    assert snapshot["situation"] == situation(env.agent_view(), obs, env.agent_info())


def test_the_command_line_serves_a_catalogue_scenario_by_name(tmp_path):
    # A real subprocess: a few seconds idle, nearly all of it importing numpy and scipy.
    out = tmp_path / "results.jsonl"
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "describe_task"}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "start_episode"}},
        *[
            {
                "jsonrpc": "2.0",
                "id": 3 + k,
                "method": "tools/call",
                "params": {"name": "command", "arguments": {"ax": 0, "ay": 0, "az": 0}},
            }
            for k in range(2)
        ],
    ]
    argv = ["--scenario", "docking-afrl", "--seeds", "9", "--max-steps", "2", "--out", str(out)]
    done = subprocess.run(
        [sys.executable, "-m", "vbar.agents.mcp", *argv],
        input="\n".join(json.dumps(m) for m in lines) + "\n",
        capture_output=True,
        text=True,
        timeout=HANG,
    )
    task = json.loads(done.stdout.splitlines()[0])["result"]["structuredContent"]
    assert task["view"]["scenario_id"].startswith("vbar:docking-afrl:")
    (result,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert result["scenario_id"] == task["view"]["scenario_id"] and len(result["actions"]) == 2


def test_the_command_line_refuses_an_unknown_scenario(capsys):
    with pytest.raises(SystemExit):
        main(["--scenario", "docking-sideways"])
    assert "unknown scenario 'docking-sideways'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--seeds", "-1", "5"], "seeds must not be negative, got -1"),
        (["--seeds", "5", "-1"], "seeds must not be negative, got -1"),
        (["--max-steps", "0"], "--max-steps must be at least 1"),
        (["--max-steps", "-3"], "--max-steps must be at least 1"),
    ],
)
def test_the_command_line_refuses_bad_options_before_it_serves(argv, message, capsys):
    with pytest.raises(SystemExit):
        main(argv)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        {"ax": "0.5", "ay": 0, "az": 0},
        {"ax": True, "ay": False, "az": True},
        {"ax": 0, "ay": 0},
        {"ax": 0, "ay": None, "az": 0},
    ],
)
def test_a_command_takes_only_json_numbers_and_flies_nothing_otherwise(arguments):
    server = ToolServer(ApproachEnv(ApproachConfig(max_steps=2)), [0])
    call(server, "start_episode")
    result, _ = call(server, "command", **arguments)
    assert (
        result["isError"] and "command needs numbers ax, ay and az" in result["content"][0]["text"]
    )
    assert len(server.episode.actions) == 0
    result, _ = call(server, "predict_coast", seconds=True)
    assert result["isError"] and "seconds must be a number" in result["content"][0]["text"]


def test_the_command_line_reads_utf8_whatever_the_locale(tmp_path):
    # A real subprocess, a few seconds idle. Windows consoles decode text input in the ANSI code
    # page, which garbled non-ASCII ids before the server read its input as UTF-8 bytes.
    ident = "é中\U0001f600"
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": ident, "method": "ping"}, ensure_ascii=False).encode(),
        b'{"jsonrpc": "2.0", "id": 2, "method": "ping", "x": "\xff\xfe"}',
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}).encode(),
    ]
    out = tmp_path / "results.jsonl"
    done = subprocess.run(
        [sys.executable, "-m", "vbar.agents.mcp", "--max-steps", "1", "--out", str(out)],
        input=b"\n".join(lines) + b"\n",
        capture_output=True,
        timeout=HANG,
        env={**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"},
    )
    responses = [json.loads(line) for line in done.stdout.splitlines()]
    assert responses[0] == {"jsonrpc": "2.0", "id": ident, "result": {}}
    assert responses[1]["error"]["code"] == PARSE_ERROR and responses[1]["id"] is None
    assert responses[2]["id"] == 3


def test_a_lit_docking_task_is_described_in_strict_json():
    from vbar.scenarios import make

    server = ToolServer(make("docking:lit"), [3])
    response = request(server, "tools/call", name="describe_task")
    assert not response["result"]["isError"]
    json.dumps(response, allow_nan=False)


def test_the_command_line_reads_its_seeds_from_a_file(tmp_path):
    # Seeds on the command line are visible to every process that lists the others.
    seeds_file = tmp_path / "seeds.txt"
    # With a byte-order mark, as some Windows editors save UTF-8.
    seeds_file.write_text(f"\ufeff{2**100 + 7}\n\n  12\n", encoding="utf-8")
    out = tmp_path / "results.jsonl"
    argv = ["--seeds-file", str(seeds_file), "--max-steps", "1", "--out", str(out)]
    command = [sys.executable, "-m", "vbar.agents.mcp", *argv]
    subprocess.run(command, input="", capture_output=True, text=True, timeout=HANG, check=True)
    assert [json.loads(line)["seed"] for line in out.read_text().splitlines()] == [2**100 + 7, 12]


@pytest.mark.parametrize(
    "lines, message",
    [
        ("5\n-1\n", "--seeds-file: seeds must not be negative, got -1"),
        ("5\nseven\n", "--seeds-file: invalid seed"),
        ("\n\n", "--seeds-file holds no seeds"),
        # UTF-16, as Windows PowerShell 5.1 writes with ">".
        ("5\n".encode("utf-16"), "--seeds-file: 'utf-8' codec can't decode"),
    ],
)
def test_a_bad_seeds_file_is_refused_before_the_session(lines, message, tmp_path, capsys):
    seeds_file = tmp_path / "seeds.txt"
    if isinstance(lines, bytes):
        seeds_file.write_bytes(lines)
    else:
        seeds_file.write_text(lines, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--seeds-file", str(seeds_file)])
    assert message in capsys.readouterr().err


def test_seeds_come_from_the_command_line_or_a_file_not_both(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--seeds", "1", "--seeds-file", str(tmp_path / "missing.txt")])
    assert "not allowed with argument" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--seeds-file", str(tmp_path / "missing.txt")])
    assert "--seeds-file:" in capsys.readouterr().err
    # An empty name, as an unset shell variable gives, must not fall back to drawn seeds.
    with pytest.raises(SystemExit):
        main(["--seeds-file", ""])
    assert "--seeds-file:" in capsys.readouterr().err
