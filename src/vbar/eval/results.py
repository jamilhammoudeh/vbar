"""Result files: one JSON object per episode, and the trajectories beside them.

Schema 1 fields: `schema`, `task`, `suite`, `scenario` (the name with variants), `scenario_id`
(the identifier flown), `split`, `index`, `seed` and `agent_seed` (decimal strings, since 128-bit
integers do not survive readers that parse numbers as doubles), `seed_digest` and
`agent_seed_digest` (the keyed digests of the two seeds, `vbar.scenarios.seeds.seed_digest`),
`agent`, `filter`, `metrics`, `failure_label`, `error`, `provenance`, `environment` and
`wall_s`. NaN and infinite floats are written as the strings "nan", "inf" and "-inf" (strict JSON
has no such numbers) and the metrics and wall time are read back as floats.

The seeds of the test split are never written: those lines hold `"seed": null` and
`"agent_seed": null` and are matched to their runs by the digests. When the environment variable
`VBAR_HIDDEN_SEED_LOG` names a file, `write_results` appends the seeds of every test split result
to it, so the maintainers keep them privately.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..scenarios.seeds import HIDDEN_LOG_VARIABLE, split_key
from .stats import _whole

SCHEMA_VERSION = 1
_DIGEST = re.compile(r"[0-9a-f]{64}")
FIELDS = (
    "schema",
    "task",
    "suite",
    "scenario",
    "scenario_id",
    "split",
    "index",
    "seed",
    "agent_seed",
    "seed_digest",
    "agent_seed_digest",
    "agent",
    "filter",
    "metrics",
    "failure_label",
    "error",
    "provenance",
    "environment",
    "wall_s",
)
_SPECIAL = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
# Metrics measured on the wall clock: they differ between repeats of the same run, so checks of
# reproducibility leave them out.
WALL_CLOCK_METRICS = (
    "latency_median",
    "latency_p95",
    "preparation_median",
    "feedback_median",
    "feedback_p95",
    "compute",
    "filter_compute_median",
    "filter_compute_p95",
)


def to_json(value: Any) -> Any:
    """`value` in JSON-safe form: numpy scalars and arrays as Python, non-finite floats as text.

    Parameters
    ----------
    value : Any
        A metric value or a result field.

    Returns
    -------
    Any
        Built from dict, list, str, int, float, bool and None only.
    """
    if isinstance(value, np.ndarray):
        return [to_json(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Mapping):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [to_json(v) for v in value]
    return value


def from_json(value: Any) -> Any:
    """The inverse of `to_json` for floats: "nan", "inf" and "-inf" become floats again.

    Parameters
    ----------
    value : Any
        A value parsed from a result file.

    Returns
    -------
    Any
        The value with the non-finite floats restored.
    """
    if isinstance(value, str):
        return _SPECIAL.get(value, value)
    if isinstance(value, dict):
        return {k: from_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_json(v) for v in value]
    return value


def encode(result: Mapping[str, Any]) -> str:
    """One result as a line of strict JSON.

    Parameters
    ----------
    result : mapping
        A result with the schema's fields.

    Returns
    -------
    str
        The JSON text, without a newline.

    Raises
    ------
    ValueError
        If a schema field is missing or an unknown one is present, a seed is not a whole number
        of at least 0 (numpy integers are accepted, text is not; a result of the test split may
        hold None), a result of the test split lacks a digest of 64 lowercase hex digits for either
        seed, or its line would hold the hidden key or one of its seeds in any field.

    Notes
    -----
    In a result of the test split the seeds and the key are removed from the `error` text, in
    decimal and in hexadecimal, since an exception message can carry them; anywhere else in the
    line they are refused. The key is the one in `VBAR_HIDDEN_SEED_KEY`, when it is set.
    """
    missing = [f for f in FIELDS if f not in result]
    extra = [f for f in result if f not in FIELDS]
    if missing or extra:
        raise ValueError(f"result fields differ from schema {SCHEMA_VERSION}: {missing or extra}")
    if result["split"] == "test":
        for name in ("seed_digest", "agent_seed_digest"):
            _check_digest(name, result[name])
        for name in ("seed", "agent_seed"):
            # The search for a seed only knows whole numbers, so any other spelling is refused.
            if result[name] is not None:
                _whole(name, result[name], 0)
        text = json.dumps(to_json(_published(result)), sort_keys=True, allow_nan=False)
        _check_unpublished(text, result["seed"], result["agent_seed"])
        return text
    fields = dict(result)
    for name in ("seed", "agent_seed"):
        fields[name] = str(_whole(name, result[name], 0))
    return json.dumps(to_json(fields), sort_keys=True, allow_nan=False)


def _check_digest(name: str, value: Any) -> None:
    """Raise ValueError unless `value` is a SHA-256 digest in 64 lowercase hex digits."""
    if not (isinstance(value, str) and _DIGEST.fullmatch(value)):
        raise ValueError(f"{name} must be 64 lowercase hex digits")


def _key() -> str | None:
    """The hidden key as `split_key` normalises it, or None when the variable holds none."""
    try:
        return split_key("test").removeprefix("vbar test seeds ")
    except ValueError:
        return None


def _forms(seed: Any) -> list[str]:
    """The decimal and hexadecimal spellings of a seed that are long enough to be searched for."""
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0:
        return []
    number = int(seed)
    # A drawn seed is 128 bits; a short number would also damage ordinary text.
    return [form for form in (str(number), f"{number:x}") if len(form) >= 8]


def _published(result: Mapping[str, Any]) -> dict[str, Any]:
    """A test split result as it is published: no seeds, and none of them or the key in `error`."""
    error = result["error"]
    if isinstance(error, str):
        secrets = [*_forms(result["seed"]), *_forms(result["agent_seed"])]
        key = _key()
        if key:
            secrets.append(key)
        for secret in sorted(secrets, key=len, reverse=True):
            error = re.sub(re.escape(secret), "<hidden>", error, flags=re.IGNORECASE)
    return {**result, "seed": None, "agent_seed": None, "error": error}


def _check_unpublished(text: str, *seeds: Any) -> None:
    """Raise ValueError if a line of the test split holds the key or one of the seeds."""
    lowered = text.lower()
    key = _key()
    if key and key in lowered:
        raise ValueError("a test split result would publish the hidden key")
    if any(form in lowered for seed in seeds for form in _forms(seed)):
        raise ValueError("a test split result would publish a seed")


def decode(line: str) -> dict[str, Any]:
    """One result from its JSON line, with integer seeds and non-finite floats restored.

    The seeds of a result of the test split are None, as they are written.

    Parameters
    ----------
    line : str
        A line written by `encode`.

    Returns
    -------
    dict
        The result.

    Raises
    ------
    ValueError
        If the line is not a result of a schema this version reads, a field is missing, `metrics`
        is not an object, a seed is neither decimal text nor a JSON integer of at least 0
        (null is read only in the test split), or a line of the test split lacks a digest of 64
        lowercase hex digits for either seed.
    """
    raw = json.loads(line)
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"not a schema {SCHEMA_VERSION} result")
    missing = [f for f in FIELDS if f not in raw]
    if missing:
        raise ValueError(f"result lacks fields {missing}")
    if not isinstance(raw["metrics"], dict):
        raise ValueError("metrics must be an object")
    # Only numbers are written as text, so only the numeric fields are read back from it.
    result = {k: from_json(v) if k in ("metrics", "wall_s") else v for k, v in raw.items()}
    for name in ("seed", "agent_seed"):
        withheld = raw[name] is None and raw["split"] == "test"
        result[name] = None if withheld else _seed(name, raw[name])
    if raw["split"] == "test":
        for name in ("seed_digest", "agent_seed_digest"):
            _check_digest(name, raw[name])
    return result


def _seed(name: str, value: Any) -> int:
    """A seed read from a result line: decimal text or a JSON integer, at least 0."""
    # Digits only: int() would also take signs, spaces and underscores.
    if isinstance(value, str) and value.isascii() and value.isdigit():
        value = int(value)
    return _whole(name, value, 0)


def write_results(
    path: str | Path, results: Iterable[Mapping[str, Any]], append: bool = True
) -> None:
    """Write results as JSON lines.

    The line of a test split result holds no seed (`encode`). If the environment variable
    `VBAR_HIDDEN_SEED_LOG` names a file, the result's seeds are first appended to it as one JSON
    line, with the suite, scenario identifier, split, index, both seeds as decimal strings and
    both digests; results of other splits add nothing, nor do test results that hold no seeds
    because they were read from a published file.

    Parameters
    ----------
    path : str or Path
        The file.
    results : iterable of mapping
        Results with the schema's fields.
    append : bool
        Append to the file (True) or replace it.

    Raises
    ------
    ValueError
        If `encode` refuses a result, or the private seed log is the results file (under any
        name, a hard link included), whose lines it would mix with the seeds.
    OSError
        If the private seed log is set and cannot be opened for appending; the result's line is
        not written then.
    """
    path = Path(path)
    seed_log = os.environ.get(HIDDEN_LOG_VARIABLE, "")
    if seed_log and Path(seed_log).resolve() == path.resolve():
        raise ValueError("the private seed log must not be the results file")
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists() and path.stat().st_size:
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            # A run killed mid-write leaves half a line; the run is flown again, so cut it off.
            path.write_bytes(data[: data.rfind(b"\n") + 1])
    with open(path, "a" if append else "w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            line = encode(result)
            if seed_log and result["split"] == "test" and result["seed"] is not None:
                _log_seeds(seed_log, path, result)
            handle.write(line + "\n")


def _log_seeds(log: str, results_path: Path, result: Mapping[str, Any]) -> None:
    """Append one test split result's seeds to the private log; the log is not the results file."""
    entry = {
        name: result[name]
        for name in ("suite", "scenario_id", "split", "index", "seed_digest", "agent_seed_digest")
    }
    for name in ("seed", "agent_seed"):
        entry[name] = str(_whole(name, result[name], 0))
    with open(log, "a", encoding="utf-8", newline="\n") as handle:
        # Path.resolve does not see a hard link, nor a trailing space or dot that Windows drops.
        if os.path.samefile(log, results_path):
            raise ValueError("the private seed log must not be the results file")
        handle.write(json.dumps(to_json(entry), sort_keys=True, allow_nan=False) + "\n")


def read_results(path: str | Path) -> list[dict[str, Any]]:
    """Every result in a JSON-lines file, in file order; blank lines are skipped.

    Parameters
    ----------
    path : str or Path
        The file.

    Returns
    -------
    list of dict
        The results.

    Raises
    ------
    ValueError
        If a line is not a result, except a last line cut off before its newline, which is
        dropped with a warning; the message names the line.
    """
    out = []
    with open(path, encoding="utf-8") as handle:
        lines = [(n, line) for n, line in enumerate(handle, start=1) if line.strip()]
    for k, (number, line) in enumerate(lines):
        try:
            out.append(decode(line))
        except ValueError as error:
            cut = k == len(lines) - 1 and not line.endswith("\n")
            if cut and isinstance(error, json.JSONDecodeError):
                # A write killed midway leaves a last line with no newline; that run is flown again.
                warnings.warn(f"{path}, line {number}: dropped a truncated last line", stacklevel=2)
                break
            raise ValueError(f"{path}, line {number}: {error}") from error
    return out


def save_trajectories(path: str | Path, episodes: Mapping[str, Any]) -> None:
    """Save the true states, times, observations and commands of episodes, keyed by provenance.

    Parameters
    ----------
    path : str or Path
        The `.npz` file.
    episodes : mapping
        Provenance hash to a recorded `Episode`.
    """
    arrays = {}
    for key, ep in episodes.items():
        arrays[f"{key}/times"] = ep.times
        arrays[f"{key}/states"] = ep.states
        arrays[f"{key}/obs"] = np.array([r.obs for r in ep.records]).reshape(len(ep.records), -1)
        arrays[f"{key}/commands"] = np.array([r.command for r in ep.records]).reshape(
            len(ep.records), -1
        )
    np.savez_compressed(path, **arrays)


def load_trajectories(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """The trajectories saved by `save_trajectories`.

    Parameters
    ----------
    path : str or Path
        The `.npz` file.

    Returns
    -------
    dict
        Provenance hash to {"times", "states", "obs", "commands"}.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path) as data:
        for name in data.files:
            key, field = name.rsplit("/", 1)
            out.setdefault(key, {})[field] = data[name]
    return out


def reproducible(result: Mapping[str, Any]) -> str:
    """The part of a result a repeat of the run must reproduce, as canonical JSON text.

    Parameters
    ----------
    result : mapping
        A result.

    Returns
    -------
    str
        The result without its wall time, environment record and wall-clock metrics, and, for the
        test split, as it is published: without the seeds, which stay out of every comparison text.
    """
    if result["split"] == "test":
        result = _published(result)
    metrics = {k: v for k, v in result["metrics"].items() if k not in WALL_CLOCK_METRICS}
    kept = {k: v for k, v in result.items() if k not in ("wall_s", "environment", "metrics")}
    return json.dumps(to_json({**kept, "metrics": metrics}), sort_keys=True)


__all__ = [
    "FIELDS",
    "SCHEMA_VERSION",
    "WALL_CLOCK_METRICS",
    "decode",
    "encode",
    "from_json",
    "load_trajectories",
    "read_results",
    "reproducible",
    "save_trajectories",
    "to_json",
    "write_results",
]
