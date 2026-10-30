"""Command line: run a suite, report it, or rebuild every table and script a manifest lists.

python -m vbar.eval run --suite core-v1 --agent vbar.agents:PDAgent --out pd.jsonl
python -m vbar.eval report pd.jsonl --markdown pd.md
python -m vbar.eval reproduce reproduce.toml
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path, PurePath

from ..scenarios import SUITES
from .report import latest_results, suite_report, to_markdown
from .results import read_results, to_json
from .runner import run_suite

logger = logging.getLogger(__name__)


def _suite(suite_id: str):
    if suite_id not in SUITES:
        raise SystemExit(f"unknown suite {suite_id!r}; known: {', '.join(sorted(SUITES))}")
    return SUITES[suite_id]


def _report(results_path: Path, suite_id: str | None, markdown: Path | None, out: Path | None):
    results = latest_results(read_results(results_path))
    if suite_id is None:
        suites = {r["suite"] for r in results}
        if len(suites) != 1 or None in suites:
            raise SystemExit("the results name no single suite; give --suite")
        (suite_id,) = suites
    report = suite_report(_suite(str(suite_id)), results)
    text = to_markdown(report)
    if markdown is not None:
        markdown.write_text(text, encoding="utf-8")
    if out is not None:
        out.write_text(json.dumps(to_json(report), indent=1, sort_keys=True), encoding="utf-8")
    return text


_TABLE_KEYS = {"name", "suite", "agent", "results", "markdown", "filter"}
_TABLE_REQUIRED = ("name", "suite", "agent", "results", "markdown")
_SCRIPT_KEYS = {"name", "script", "args", "outputs", "stdout", "requires", "extra"}
_SCRIPT_REQUIRED = ("name", "script", "outputs")


def _check_entry(
    kind: str,
    entry: dict,
    allowed: set[str],
    required: Sequence[str],
    paths: Sequence[str],
    lists: Sequence[str],
) -> list[str]:
    label = f"[[{kind}]] {entry.get('name', '?')!r}"
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise ValueError(f"{label}: unknown key {', '.join(unknown)}")
    for key in required:
        if key not in entry:
            raise ValueError(f"{label}: missing key {key}")
    for key, value in entry.items():
        if key in lists:
            ok = isinstance(value, list) and all(isinstance(v, str) and v for v in value)
        else:
            ok = isinstance(value, str) and bool(value)
        if not ok:
            what = "a list of non-empty text" if key in lists else "non-empty text"
            raise ValueError(f"{label}: {key} must be {what}")
    seen = []
    for key in paths:
        value = entry.get(key, [])
        for text in value if isinstance(value, list) else [value]:
            path = PurePath(text)
            if not path.parts or path.is_absolute() or path.anchor or ".." in path.parts:
                raise ValueError(f"{label}: {key} {text!r} must be inside the manifest folder")
            seen.append(path.as_posix())
    return seen


def _validate(spec: dict) -> None:
    unknown = sorted(set(spec) - {"table", "script"})
    if unknown:
        raise ValueError(f"unknown manifest key {', '.join(unknown)}")
    for key in ("table", "script"):
        entries = spec.get(key, [])
        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
            raise ValueError(f"{key} must be an array of tables")
    produced = []
    for table in spec.get("table", []):
        produced += _check_entry(
            "table", table, _TABLE_KEYS, _TABLE_REQUIRED, ("results", "markdown"), ()
        )
    for entry in spec.get("script", []):
        files = _check_entry(
            "script",
            entry,
            _SCRIPT_KEYS,
            _SCRIPT_REQUIRED,
            ("script", "outputs", "stdout"),
            ("args", "outputs", "requires"),
        )
        if not entry["outputs"] and "stdout" not in entry:
            raise ValueError(f"[[script]] {entry['name']!r}: lists no output and no stdout file")
        produced += [f for f in files if f != PurePath(entry["script"]).as_posix()]
    for path in produced:
        if produced.count(path) > 1:
            raise ValueError(f"{path} is produced more than once")


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _run_script(entry: dict, root: Path, force: bool, skip_missing: bool) -> list[Path]:
    name = entry["name"]
    outputs = [root / o for o in entry["outputs"]]
    stdout = root / entry["stdout"] if "stdout" in entry else None
    produced = outputs + ([stdout] if stdout is not None else [])
    if not force and all(p.exists() for p in produced):
        return []
    missing = [m for m in entry.get("requires", []) if not _importable(m)]
    if missing:
        extra = entry.get("extra")
        hint = f'; install it with: pip install "vbar[{extra}]"' if extra else ""
        message = f"script {name!r} needs {', '.join(missing)}{hint}"
        if not skip_missing:
            raise ImportError(message)
        logger.warning("skipped %s", message)
        return []
    done = subprocess.run(
        [sys.executable, entry["script"], *entry.get("args", [])],
        cwd=root,
        stdout=subprocess.PIPE if stdout is not None else None,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"script {name!r} exited with status {done.returncode}")
    if stdout is not None:
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_bytes(done.stdout)
    for path in outputs:
        if not path.exists():
            raise RuntimeError(f"script {name!r} did not write {path.relative_to(root).as_posix()}")
    return produced


def reproduce(
    manifest: Path, workers: int = 1, *, force: bool = False, skip_missing: bool = False
) -> list[Path]:
    """Rebuild whatever a manifest lists: its tables first, then its scripts in order.

    The manifest is TOML with one `[[table]]` per published table: `name`, `suite`, `agent`
    (a factory path), optional `filter`, and `results` and `markdown` paths relative to the
    manifest. Runs already in a results file with the same provenance are not flown again.
    A `[[script]]` makes files outside the suite runner: `name`, `script` (a Python file),
    `outputs` (files it writes), optional `args`, `stdout` (a file that receives its standard
    output), `requires` (importable module names) and `extra` (the vbar extra that installs
    them). Scripts run with the current interpreter in the manifest's folder, and one whose
    outputs and stdout file all exist is not run again unless `force` is set. The whole manifest
    is validated before anything runs: every path must lie inside the manifest's folder and be
    produced once, and keys and types must match.

    Parameters
    ----------
    manifest : Path
        The manifest file.
    workers : int
        Processes per suite run.
    force : bool
        Run scripts whose outputs already exist.
    skip_missing : bool
        Log a warning and skip a script whose required modules are not installed, instead of
        raising.

    Returns
    -------
    list of Path
        The Markdown tables written, then the files each script that ran produced.

    Raises
    ------
    ValueError
        An entry has an unknown or missing key, or a path outside the manifest's folder.
    ImportError
        A script's required module is missing and `skip_missing` is false.
    RuntimeError
        A script exits with a non-zero status or does not write a listed output.
    """
    spec = tomllib.loads(manifest.read_text(encoding="utf-8"))
    _validate(spec)
    root = manifest.parent
    written = []
    for table in spec.get("table", []):
        results = root / table["results"]
        run_suite(
            _suite(table["suite"]),
            table["agent"],
            safety_filter=table.get("filter"),
            workers=workers,
            out=results,
        )
        markdown = root / table["markdown"]
        _report(results, table["suite"], markdown, None)
        written.append(markdown)
    for entry in spec.get("script", []):
        written += _run_script(entry, root, force, skip_missing)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of `python -m vbar.eval`.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; `sys.argv[1:]` by default.

    Returns
    -------
    int
        The exit status.
    """
    parser = argparse.ArgumentParser(prog="python -m vbar.eval")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run every episode of a suite")
    run.add_argument("--suite", required=True)
    run.add_argument("--agent", required=True, help="agent factory, module:attribute")
    run.add_argument("--filter", default=None, help="safety filter factory, module:attribute")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--out", type=Path, required=True, help="JSON-lines results file")
    report = commands.add_parser("report", help="report a complete results file")
    report.add_argument("results", type=Path)
    report.add_argument("--suite", default=None)
    report.add_argument("--markdown", type=Path, default=None)
    report.add_argument("--json", type=Path, default=None)
    again = commands.add_parser("reproduce", help="rebuild every table of a manifest")
    again.add_argument("manifest", type=Path)
    again.add_argument("--workers", type=int, default=1)
    again.add_argument("--force", action="store_true", help="run scripts whose outputs exist")
    again.add_argument(
        "--skip-missing", action="store_true", help="skip scripts whose extras are not installed"
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        results = run_suite(
            _suite(args.suite),
            args.agent,
            safety_filter=args.filter,
            workers=args.workers,
            out=args.out,
        )
        errors = sum(r["error"] is not None for r in results)
        sys.stdout.write(f"{len(results)} runs in {args.out}, {errors} with errors\n")
    elif args.command == "report":
        sys.stdout.write(_report(args.results, args.suite, args.markdown, args.json))
    else:
        for path in reproduce(
            args.manifest, args.workers, force=args.force, skip_missing=args.skip_missing
        ):
            sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
