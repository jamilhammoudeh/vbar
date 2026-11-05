"""The stored API key and settings: where they live, who can read them, what overrides them."""

import ast
import json
import logging
import os
import stat
import sys
import warnings
from pathlib import Path

import pytest

from vbar.providers import store
from vbar.providers.store import forget_key, load_key, load_settings, save_key, save_settings

KEY = "test-key-7f3a9c1e5b2d8f4a6c0e"
POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")


@pytest.fixture(autouse=True)
def config_dir(tmp_path, monkeypatch):
    folder = tmp_path / "config"
    monkeypatch.setenv("VBAR_CONFIG_DIR", str(folder))
    monkeypatch.delenv("VBAR_LLM_API_KEY", raising=False)
    return folder


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_a_saved_key_loads_back_from_credentials_json(config_dir):
    save_key(KEY)
    assert load_key() == KEY
    assert json.loads((config_dir / "credentials.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("pasted", [KEY + "\r\n", KEY + "\n", "  " + KEY, "\t " + KEY + " \r\n"])
def test_a_pasted_key_is_saved_without_surrounding_whitespace(pasted):
    save_key(pasted)
    assert load_key() == KEY


def test_no_key_is_loaded_before_one_is_saved():
    assert load_key() is None


def test_saving_a_key_again_replaces_the_first():
    save_key("first-key-000000")
    save_key(KEY)
    assert load_key() == KEY


def test_forget_removes_the_stored_key_and_keeps_the_settings(config_dir):
    save_settings({"provider": "other", "model": "m-1"})
    save_key(KEY)
    forget_key()
    assert load_key() is None
    assert KEY not in "".join(p.read_text(encoding="utf-8") for p in config_dir.iterdir())
    assert load_settings() == {"provider": "other", "model": "m-1"}


def test_forget_without_a_stored_key_does_nothing():
    forget_key()
    assert load_key() is None


def test_settings_round_trip_and_start_empty():
    assert load_settings() == {}
    settings = {"provider": "other", "model": "m-1", "base_url": "http://127.0.0.1:11434/v1"}
    save_settings(settings)
    assert load_settings() == settings


def test_the_key_never_reaches_config_json(config_dir):
    save_key(KEY)
    save_settings({"provider": "other", "model": "m-1"})
    assert KEY not in (config_dir / "config.json").read_text(encoding="utf-8")


def test_the_environment_variable_overrides_the_stored_key(monkeypatch):
    save_key(KEY)
    monkeypatch.setenv("VBAR_LLM_API_KEY", "env-key-123456789")
    assert load_key() == "env-key-123456789"


def test_the_environment_variable_works_with_nothing_stored(monkeypatch, config_dir):
    monkeypatch.setenv("VBAR_LLM_API_KEY", "env-key-123456789")
    assert load_key() == "env-key-123456789"
    assert not (config_dir / "credentials.json").exists()


def test_an_empty_environment_variable_does_not_hide_the_stored_key(monkeypatch):
    save_key(KEY)
    monkeypatch.setenv("VBAR_LLM_API_KEY", "")
    assert load_key() == KEY


@pytest.mark.parametrize("bad", ["", "   ", "abc\ndef", "abc\r\nX-Injected: 1"])
def test_empty_keys_and_keys_with_line_breaks_are_refused(bad, config_dir):
    # A line break in a key would split the Authorization header it is sent in.
    with pytest.raises(ValueError):
        save_key(bad)
    assert not (config_dir / "credentials.json").exists()


def test_saving_leaves_no_temporary_file_behind(config_dir):
    save_key(KEY)
    save_settings({"provider": "other", "model": "m-1"})
    assert sorted(p.name for p in config_dir.iterdir()) == ["config.json", "credentials.json"]


def test_storage_logs_and_warnings_never_contain_the_key(caplog):
    caplog.set_level(logging.DEBUG)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_key(KEY)
        load_key()
        forget_key()
    assert all(KEY not in r.getMessage() for r in caplog.records)
    assert all(KEY not in str(w.message) for w in caught)


@POSIX
def test_the_key_file_is_0600_in_a_0700_folder(config_dir):
    save_key(KEY)
    assert mode(config_dir) == 0o700
    assert mode(config_dir / "credentials.json") == 0o600


@POSIX
def test_a_key_file_readable_by_others_is_warned_about_and_tightened(config_dir):
    save_key(KEY)
    path = config_dir / "credentials.json"
    path.chmod(0o644)
    with pytest.warns(UserWarning) as caught:
        assert load_key() == KEY
    assert mode(path) == 0o600
    assert all(KEY not in str(w.message) for w in caught)


@pytest.mark.skipif(os.name != "nt", reason="Windows profile folder")
def test_the_default_folder_on_windows_is_appdata_vbar(tmp_path, monkeypatch):
    monkeypatch.delenv("VBAR_CONFIG_DIR")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    save_key(KEY)
    assert (tmp_path / "vbar" / "credentials.json").exists()


@POSIX
def test_the_default_folder_follows_xdg_config_home(tmp_path, monkeypatch):
    monkeypatch.delenv("VBAR_CONFIG_DIR")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    save_key(KEY)
    assert (tmp_path / "xdg" / "vbar" / "credentials.json").exists()


@POSIX
def test_without_xdg_config_home_the_folder_is_dot_config_vbar(tmp_path, monkeypatch):
    monkeypatch.delenv("VBAR_CONFIG_DIR")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    save_key(KEY)
    assert (tmp_path / ".config" / "vbar" / "credentials.json").exists()


def test_the_store_imports_only_the_standard_library():
    tree = ast.parse(Path(store.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= set(sys.stdlib_module_names) | {"vbar"}
