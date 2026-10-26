"""Provenance: one hash per run of everything that decides its result, and the machine it ran on.

The hash covers the package version, the source tree's git commit (and whether it had local
changes), the task, the identifier flown and its dynamics model, both seeds, and the agent's and
the filter's names and settings. It is the SHA-256 of their canonical JSON, so the same run gives
the same hash on every platform. The interpreter and library versions are recorded beside it but
not hashed: results are reproducible across them only within the stated tolerances.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy

_PLAIN = (bool, int, float, str, type(None))


def settings_of(component: Any) -> dict[str, Any] | None:
    """The settings of an agent or filter that the provenance hash covers.

    A component that declares a `config` mapping gives that; otherwise its public attributes of
    plain types (numbers, strings, booleans, None and tuples or lists of them) at the time of the
    call, which for the built-in agents are their constructor arguments.

    Parameters
    ----------
    component : Any
        An agent or a safety filter, or None.

    Returns
    -------
    dict or None
        Name to value, sorted by name; None for no component.
    """
    if component is None:
        return None
    declared = getattr(component, "config", None)
    if isinstance(declared, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(declared.items())}
    found = {}
    for key, value in sorted(vars(component).items()) if hasattr(component, "__dict__") else ():
        if key.startswith("_"):
            continue
        if isinstance(value, _PLAIN) or (
            isinstance(value, tuple | list) and all(isinstance(v, _PLAIN) for v in value)
        ):
            found[key] = _canonical(value)
    return found


def _canonical(value: Any) -> Any:
    """JSON-safe form: NaN and infinities as strings, tuples as lists, numpy scalars as Python."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_canonical(v) for v in value]
    return value


@functools.cache
def source_revision() -> tuple[str | None, bool]:
    """The git commit of the installed source tree and whether it has local changes.

    Returns
    -------
    tuple
        (commit SHA, dirty); (None, False) when the package is not a git checkout or git is
        not available.
    """
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return None, False
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, False
    return sha or None, bool(status.strip())


def environment() -> dict[str, Any]:
    """The software and machine a run used, recorded with the results.

    Returns
    -------
    dict
        "vbar", "git", "dirty", "python", "numpy", "scipy" and "platform".
    """
    from .. import __version__

    sha, dirty = source_revision()
    return {
        "vbar": __version__,
        "git": sha,
        "dirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": f"{sys.platform}-{platform.machine()}",
    }


def provenance_fields(
    *,
    task: str,
    identity: Mapping[str, Any],
    seed: int,
    agent_seed: int,
    agent: Any,
    safety_filter: Any = None,
) -> dict[str, Any]:
    """The fields one run's provenance hash covers.

    Parameters
    ----------
    task : str
        The task's name.
    identity : mapping
        The task's identity of the run (for spacecraft tasks the identifier and dynamics).
    seed, agent_seed : int
        Episode and agent seeds.
    agent : Agent
        The agent as built for the run, before its first reset.
    safety_filter : SafetyFilter, optional
        The filter, or None for none.

    Returns
    -------
    dict
        The fields, JSON-safe; the seeds as decimal strings.
    """
    from .. import __version__

    sha, dirty = source_revision()
    return {
        "vbar": __version__,
        "git": sha,
        "dirty": dirty,
        "task": task,
        **{str(k): _canonical(v) for k, v in identity.items()},
        "seed": str(int(seed)),
        "agent_seed": str(int(agent_seed)),
        "agent": {
            "name": getattr(agent, "name", type(agent).__name__),
            "config": settings_of(agent),
        },
        "filter": None
        if safety_filter is None
        else {
            "name": getattr(safety_filter, "name", type(safety_filter).__name__),
            "config": settings_of(safety_filter),
        },
    }


def provenance_hash(fields: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of the canonical JSON of `fields`.

    Parameters
    ----------
    fields : mapping
        JSON-safe fields (`provenance_fields`).

    Returns
    -------
    str
        64 hex digits.
    """
    text = json.dumps(_canonical(dict(fields)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def combined_hash(hashes: Iterable[str]) -> str:
    """One hash for a set of runs, independent of their order.

    Parameters
    ----------
    hashes : iterable of str
        Run provenance hashes.

    Returns
    -------
    str
        SHA-256 hex digest of the sorted hashes, one per line.
    """
    text = "\n".join(sorted(hashes))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


__all__ = [
    "combined_hash",
    "environment",
    "provenance_fields",
    "provenance_hash",
    "settings_of",
    "source_revision",
]
