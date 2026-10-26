"""Result files: one JSON object per episode, and the trajectories beside them.

Schema 1 fields: `schema`, `task`, `suite`, `scenario` (the name with variants), `scenario_id`
(the identifier flown), `split`, `index`, `seed` and `agent_seed` (decimal strings, since 128-bit
integers do not survive readers that parse numbers as doubles), `agent`, `filter`, `metrics`,
`failure_label`, `error`, `provenance`, `environment` and `wall_s`. NaN and infinite floats are
written as the strings "nan", "inf" and "-inf" (strict JSON has no such numbers) and the metrics
and wall time are read back as floats.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
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
        If a schema field is missing or an unknown one is present.
    """
    missing = [f for f in FIELDS if f not in result]
    extra = [f for f in result if f not in FIELDS]
    if missing or extra:
        raise ValueError(f"result fields differ from schema {SCHEMA_VERSION}: {missing or extra}")
    seeds = {"seed": str(int(result["seed"])), "agent_seed": str(int(result["agent_seed"]))}
    fields = {**result, **seeds}
    return json.dumps(to_json(fields), sort_keys=True, allow_nan=False)


def decode(line: str) -> dict[str, Any]:
    """One result from its JSON line, with integer seeds and non-finite floats restored.

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
        If the line is not a result of a schema this version reads.
    """
    raw = json.loads(line)
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"not a schema {SCHEMA_VERSION} result")
    # Only numbers are written as text, so only the numeric fields are read back from it.
    result = {k: from_json(v) if k in ("metrics", "wall_s") else v for k, v in raw.items()}
    result["seed"] = int(raw["seed"])
    result["agent_seed"] = int(raw["agent_seed"])
    return result


def write_results(
    path: str | Path, results: Iterable[Mapping[str, Any]], append: bool = True
) -> None:
    """Write results as JSON lines.

    Parameters
    ----------
    path : str or Path
        The file.
    results : iterable of mapping
        Results with the schema's fields.
    append : bool
        Append to the file (True) or replace it.
    """
    with open(path, "a" if append else "w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(encode(result) + "\n")


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
        If a line is not a result; the message names the line.
    """
    out = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                out.append(decode(line))
            except ValueError as error:
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
        The result without its wall time, environment record and wall-clock metrics.
    """
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
