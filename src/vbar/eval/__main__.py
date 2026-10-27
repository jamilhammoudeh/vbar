"""Command line: run a suite, report it, or rebuild every table a manifest lists.

python -m vbar.eval run --suite core-v1 --agent vbar.agents:PDAgent --out pd.jsonl
python -m vbar.eval report pd.jsonl --markdown pd.md
python -m vbar.eval reproduce reproduce.toml
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from ..scenarios import SUITES
from .report import latest_results, suite_report, to_markdown
from .results import read_results, to_json
from .runner import run_suite


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


def reproduce(manifest: Path, workers: int = 1) -> list[Path]:
    """Run whatever a manifest's tables still lack and write each table.

    The manifest is TOML with one `[[table]]` per published table: `name`, `suite`, `agent`
    (a factory path), optional `filter`, and `results` and `markdown` paths relative to the
    manifest. Runs already in a results file with the same provenance are not flown again.

    Parameters
    ----------
    manifest : Path
        The manifest file.
    workers : int
        Processes per suite run.

    Returns
    -------
    list of Path
        The Markdown files written.
    """
    spec = tomllib.loads(manifest.read_text(encoding="utf-8"))
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
        for path in reproduce(args.manifest, args.workers):
            sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
