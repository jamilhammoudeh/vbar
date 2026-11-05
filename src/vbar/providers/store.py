"""The stored API key and provider settings.

`credentials.json` holds the key and `config.json` the provider settings, in `%APPDATA%\\vbar`
on Windows and `$XDG_CONFIG_HOME/vbar`, else `~/.config/vbar`, elsewhere; `VBAR_CONFIG_DIR`
overrides the folder and a non-empty `VBAR_LLM_API_KEY` overrides the stored key. On POSIX the
folder is 0700 and the files 0600. On Windows the files rely on the profile folder's access list.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CREDENTIALS = "credentials.json"
SETTINGS = "config.json"


def config_dir() -> Path:
    """The folder that holds `credentials.json` and `config.json`.

    Returns
    -------
    Path
        `VBAR_CONFIG_DIR` if set, else the per-user configuration folder followed by `vbar`.
    """
    override = os.environ.get("VBAR_CONFIG_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    else:
        # The XDG specification says a relative value is invalid and must be ignored.
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".config"
    return base / "vbar"


def _write_private(name: str, text: str) -> None:
    """Replace `name` in the config folder with `text`, through a 0600 temporary file."""
    folder = config_dir()
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(folder, 0o700)
    # mkstemp opens with O_EXCL and mode 0600, so the key is never in a file others can read.
    handle, temporary = tempfile.mkstemp(dir=folder, prefix=f".{name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, folder / name)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read_object(path: Path) -> dict[str, Any] | None:
    """The JSON object in `path`, or None if the file does not exist."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        content = json.loads(text)
    except json.JSONDecodeError:
        # The decoder keeps the text on the exception; a chained traceback would show the key.
        raise ValueError(f"{path} is not valid JSON; run `vbar setup` again") from None
    if not isinstance(content, dict):
        raise ValueError(f"{path} does not hold a JSON object; run `vbar setup` again")
    return content


def save_key(key: str) -> None:
    """Store the API key in `credentials.json`, replacing any stored key.

    Parameters
    ----------
    key : str
        The provider's API key. Surrounding whitespace is removed.

    Raises
    ------
    ValueError
        If the key is empty or holds a control character such as a line break, which would split
        the HTTP header it is sent in.
    """
    if not isinstance(key, str):
        raise ValueError("the API key must be a string")
    key = key.strip()
    if not key:
        raise ValueError("the API key is empty")
    if any(ord(c) < 32 or ord(c) == 127 for c in key):
        raise ValueError("the API key holds a control character")
    _write_private(CREDENTIALS, json.dumps({"api_key": key}))


def load_key() -> str | None:
    """The API key: `VBAR_LLM_API_KEY` if set and not empty, else the stored key.

    On POSIX a key file that other users can read is set to 0600 with a warning.

    Returns
    -------
    str or None
        The key, or None if neither the variable nor a stored key is present.

    Raises
    ------
    ValueError
        If `credentials.json` is not a JSON object with a non-empty `api_key` string.
    """
    override = os.environ.get("VBAR_LLM_API_KEY")
    if override:
        return override
    path = config_dir() / CREDENTIALS
    content = _read_object(path)
    if content is None:
        return None
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        os.chmod(path, 0o600)
        warnings.warn(
            f"{path} was readable by other users; its permissions are now 0600",
            UserWarning,
            stacklevel=2,
        )
    key = content.get("api_key")
    if not isinstance(key, str) or not key:
        raise ValueError(f"{path} holds no API key; run `vbar setup` again")
    return key


def forget_key() -> None:
    """Delete the stored API key; the settings stay. Does nothing if no key is stored."""
    (config_dir() / CREDENTIALS).unlink(missing_ok=True)


def load_settings() -> dict[str, Any]:
    """The stored provider settings, or an empty dict if none are stored.

    Raises
    ------
    ValueError
        If `config.json` is not a JSON object.
    """
    return _read_object(config_dir() / SETTINGS) or {}


def save_settings(settings: Mapping[str, Any]) -> None:
    """Store the provider settings in `config.json`, replacing the stored ones.

    Parameters
    ----------
    settings : Mapping
        JSON-serializable settings, such as `provider`, `model` and `base_url`.
    """
    _write_private(SETTINGS, json.dumps(dict(settings), indent=1))
