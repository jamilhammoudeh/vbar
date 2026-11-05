"""The language-model example flies offline until a model client is filled in, with no extras."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import LLMAgent
from vbar.agents.llm import as_reply

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "llm_controller.py"
NAMES = ("VBAR_LLM_API_KEY", "VBAR_LLM_MODEL")
TIMEOUT_S = 600


def load_example(path=EXAMPLE):
    spec = importlib.util.spec_from_file_location("llm_controller_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(folder, **variables):
    # A copy, so a .env at the repository root cannot reach the run through the folder search.
    script = folder / EXAMPLE.name
    shutil.copy(EXAMPLE, script)
    env = {k: v for k, v in os.environ.items() if k not in NAMES}
    env.update(variables)
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=folder,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )


@pytest.fixture
def no_dotenv(monkeypatch):
    monkeypatch.setitem(sys.modules, "dotenv", None)


@pytest.fixture
def unset_names(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)


def test_script_without_a_key_flies_an_episode_and_prints_the_usage(tmp_path):
    run = run_script(tmp_path)
    assert run.returncode == 0, run.stderr
    assert "decisions" in run.stdout


def test_script_with_a_key_calls_the_unfilled_complete(tmp_path):
    run = run_script(tmp_path, VBAR_LLM_API_KEY="test-key")
    assert run.returncode != 0
    assert "NotImplementedError" in run.stderr
    assert "complete()" in run.stderr


def test_script_with_an_empty_key_flies_offline(tmp_path):
    run = run_script(tmp_path, VBAR_LLM_API_KEY="")
    assert run.returncode == 0, run.stderr
    assert "decisions" in run.stdout


def test_importing_the_script_runs_no_episode(capsys, no_dotenv):
    module = load_example()
    assert capsys.readouterr().out == ""
    assert all(callable(getattr(module, name)) for name in ("settings", "complete", "stand_in"))
    assert callable(module.main)


def test_complete_as_shipped_asks_for_a_model_client(no_dotenv):
    module = load_example()
    with pytest.raises(NotImplementedError, match=re.escape("complete()")):
        module.complete([], [])


def test_stand_in_replies_with_a_zero_thrust_command(no_dotenv):
    module = load_example()
    command = json.loads(as_reply(module.stand_in([], [])).text)
    assert command == {"ax": 0.0, "ay": 0.0, "az": 0.0}


def test_stand_in_flies_an_episode_without_parse_failures(no_dotenv):
    module = load_example()
    env = ApproachEnv(ApproachConfig(start_state=(0, -400, 0, 0, 0, 0)))
    agent = LLMAgent(module.stand_in)
    result = run_episode(env, agent)
    assert agent.usage()["parse_failures"] == 0
    assert not np.any(result["actions"])


def test_main_without_a_key_does_not_call_complete(monkeypatch, capsys, no_dotenv, unset_names):
    module = load_example()

    def fail(messages, tools):
        raise AssertionError("complete() was called without a key")

    monkeypatch.setattr(module, "complete", fail)
    module.main()
    assert "decisions" in capsys.readouterr().out


def test_main_with_a_key_asks_complete_for_every_decision(
    monkeypatch, capsys, no_dotenv, unset_names
):
    module = load_example()
    calls = []

    def recording(messages, tools):
        calls.append(len(messages))
        return module.stand_in(messages, tools)

    monkeypatch.setattr(module, "complete", recording)
    monkeypatch.setenv("VBAR_LLM_API_KEY", "test-key")
    module.main()
    assert calls
    assert "decisions" in capsys.readouterr().out


def test_settings_without_dotenv_reads_the_environment(monkeypatch, no_dotenv):
    module = load_example()
    monkeypatch.setenv("VBAR_LLM_API_KEY", "os-key")
    monkeypatch.setenv("VBAR_LLM_MODEL", "os-model")
    assert module.settings() == ("os-key", "os-model")


def test_settings_without_dotenv_ignores_an_env_file(monkeypatch, tmp_path, no_dotenv, unset_names):
    (tmp_path / ".env").write_text("VBAR_LLM_API_KEY=file-key\nVBAR_LLM_MODEL=file-model\n")
    monkeypatch.chdir(tmp_path)
    module = load_example()
    assert module.settings() == (None, None)


@pytest.mark.parametrize(
    ("key", "model", "expected"),
    [("", "", (None, None)), ("", "m", (None, "m")), ("k", "", ("k", None))],
)
def test_settings_gives_none_for_an_empty_value(monkeypatch, no_dotenv, key, model, expected):
    module = load_example()
    monkeypatch.setenv("VBAR_LLM_API_KEY", key)
    monkeypatch.setenv("VBAR_LLM_MODEL", model)
    assert module.settings() == expected


def test_settings_without_dotenv_names_the_extra_when_an_env_file_is_present(
    monkeypatch, tmp_path, capsys, no_dotenv, unset_names
):
    (tmp_path / ".env").write_text("VBAR_LLM_API_KEY=file-key\n")
    monkeypatch.chdir(tmp_path)
    module = load_example()
    assert module.settings() == (None, None)
    lines = capsys.readouterr().err.splitlines()
    assert any('pip install "vbar[llm]"' in line for line in lines)


def test_settings_without_dotenv_is_silent_when_there_is_no_env_file(
    monkeypatch, tmp_path, capsys, no_dotenv, unset_names
):
    monkeypatch.chdir(tmp_path)
    module = load_example()
    assert module.settings() == (None, None)
    assert capsys.readouterr().err == ""


def test_settings_gives_none_for_unset_variables(no_dotenv, unset_names):
    module = load_example()
    assert module.settings() == (None, None)
