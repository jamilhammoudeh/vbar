"""With the llm extra, the language-model example reads its settings from a .env file."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("dotenv")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "llm_controller.py"
NAMES = ("VBAR_LLM_API_KEY", "VBAR_LLM_MODEL")
TIMEOUT_S = 600


def load_copy(folder):
    # The folder search starts at the script, so the copy sits where the .env is meant to be found.
    script = folder / EXAMPLE.name
    shutil.copy(EXAMPLE, script)
    spec = importlib.util.spec_from_file_location("llm_controller_example", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def unset_names(monkeypatch):
    # load_dotenv writes os.environ directly; setting before deleting lets monkeypatch undo it.
    for name in NAMES:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_env_file_in_the_working_folder_is_read(monkeypatch, tmp_path, unset_names):
    (tmp_path / ".env").write_text("VBAR_LLM_API_KEY=file-key\nVBAR_LLM_MODEL=file-model\n")
    monkeypatch.chdir(tmp_path)
    assert load_copy(tmp_path).settings() == ("file-key", "file-model")


def test_env_file_in_a_parent_folder_is_read(monkeypatch, tmp_path, unset_names):
    (tmp_path / ".env").write_text("VBAR_LLM_API_KEY=file-key\nVBAR_LLM_MODEL=file-model\n")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert load_copy(work).settings() == ("file-key", "file-model")


def test_environment_variable_wins_over_the_env_file(monkeypatch, tmp_path, unset_names):
    (tmp_path / ".env").write_text("VBAR_LLM_API_KEY=file-key\nVBAR_LLM_MODEL=file-model\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VBAR_LLM_API_KEY", "os-key")
    assert load_copy(tmp_path).settings() == ("os-key", "file-model")


def test_blank_entries_copied_from_the_template_give_none(monkeypatch, tmp_path, unset_names):
    (tmp_path / ".env").write_text("# model client\nVBAR_LLM_API_KEY=\nVBAR_LLM_MODEL=\n")
    monkeypatch.chdir(tmp_path)
    assert load_copy(tmp_path).settings() == (None, None)


def test_script_with_a_key_in_the_env_file_calls_the_unfilled_complete(tmp_path):
    (tmp_path / ".env").write_text("VBAR_LLM_API_KEY=file-key\n")
    shutil.copy(EXAMPLE, tmp_path / EXAMPLE.name)
    env = {k: v for k, v in os.environ.items() if k not in NAMES}
    run = subprocess.run(
        [sys.executable, str(tmp_path / EXAMPLE.name)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert run.returncode != 0
    assert "complete()" in run.stderr


def test_env_file_is_read_from_the_working_folder_not_the_script_folder(
    monkeypatch, tmp_path, unset_names
):
    run, code = tmp_path / "run", tmp_path / "code"
    run.mkdir()
    code.mkdir()
    (run / ".env").write_text("VBAR_LLM_API_KEY=file-key\nVBAR_LLM_MODEL=file-model\n")
    monkeypatch.chdir(run)
    assert load_copy(code).settings() == ("file-key", "file-model")
