"""Generate expert datasets: the oracle agent on public seeds, one NPZ file and card per scenario.

Run with `python benchmarks/make_datasets.py OUTPUT_DIR [--scenario ID ...] [--split validation]
[--start 0] [--count 50]`. Files go only to OUTPUT_DIR, which must already exist, as
`<scenario identifier>-<split>-<first>-<last>.npz`, colons as hyphens, with the card beside it.
The files are hosted outside the repository.
"""

import argparse
import logging
import sys
from pathlib import Path

SCENARIOS = (
    "vbar:approach-vbar:cw:box:dv:v1",
    "vbar:approach-rbar:cw:box:dv:v1",
)


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="existing directory the files are written to")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help=f"scenario identifier, repeatable (default: {', '.join(SCENARIOS)})",
    )
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--start", type=int, default=0, help="first episode index")
    parser.add_argument("--count", type=int, default=50, help="number of episodes")
    parser.add_argument("--replan-steps", type=int, default=None, help="steps between solves")
    args = parser.parse_args(argv)
    if not args.output.is_dir():
        parser.error(f"{args.output} is not an existing directory")
    if args.start < 0 or args.count < 1:
        parser.error("--start must not be negative and --count must be at least 1")
    return args


def main(argv: list[str]) -> None:
    args = parse(argv)
    # Loaded after the arguments are checked, so a bad command line fails without the import.
    from vbar.oracles.datasets import expert_dataset
    from vbar.oracles.expert import OracleAgent

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = OracleAgent(replan_steps=args.replan_steps)
    indices = range(args.start, args.start + args.count)
    for scenario_id in args.scenarios or SCENARIOS:
        dataset = expert_dataset(scenario_id, args.split, indices, agent=agent)
        name = scenario_id.removeprefix("vbar:").replace(":", "-")
        stem = f"{name}-{args.split}-{indices[0]}-{indices[-1]}"
        path = args.output / f"{stem}.npz"
        dataset.save(path)
        logging.info("wrote %s (%s)", path, dataset.card["e1_summary"])


if __name__ == "__main__":
    main(sys.argv[1:])
