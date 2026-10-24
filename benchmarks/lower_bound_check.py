"""Check that no successful baseline run spends less than a lower bound certified for its rung.

Run with `python benchmarks/lower_bound_check.py [--scenario ID ...] [--agent NAME ...]
[--count 20] [--split validation] [--max-steps N] [--violation-samples N]`. Each scenario gets one
`DualBank` built from the true starts of its seeds, and every run is scored against the bank's
bound for its start with `optimality_gap`. The table on standard output has one row per scenario
and agent: runs, successes, successful runs held to a certified bound, and runs flagged E3f.
A run the agent ends by raising `RuntimeError` counts as a failed run. Skipped agents, scenarios
whose bank fails to solve, the reasons of raised runs, flagged runs and progress go to standard
error. The exit status is 1 when any run is flagged, 3 when no successful run anywhere was held
to a certified bound, so nothing was checked, 2 on a usage error, and 0 otherwise.
"""

import argparse
import logging
import math
import sys
import time
from collections.abc import Callable
from typing import Any

SCENARIOS = (
    "vbar:approach-vbar:cw:box:dv:v1",
    "vbar:approach-vbar:cw:ball:dv:v1",
    "vbar:approach-rbar:cw:box:dv:v1",
    "vbar:approach-hbar:cw:box:dv:v1",
    "vbar:rbar-acquisition:cw:box:dv:v1",
    "vbar:docking-afrl:cw:box:dv:v1",
)
# One severity per kind, the middle level of each ordered row of docs/mutants.md.
MUTANT_SEVERITY = {
    "position_bias": 10.0,
    "velocity_bias": 0.03,
    "delay": 3.0,
    "sign_flip": 1.0,
    "dither": 0.15,
    "chatter": 0.1,
    "radial_sign": 1.0,
    "no_coriolis": 1.0,
    "bang_bang": 0.3,
}
COLUMNS = ("runs", "successes", "certified", "e3f")
EXIT_STATUS = (
    "exit status: 0 when certified runs were checked and none was flagged, 1 when any run is "
    "flagged E3f, 2 on a usage error, 3 when no run was held to a certified bound"
)


def parse(argv: list[str]) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=EXIT_STATUS)
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help=f"scenario identifier, repeatable (default: {', '.join(SCENARIOS)})",
    )
    parser.add_argument(
        "--agent",
        action="append",
        dest="agents",
        help="agent name, repeatable (default: every baseline and mutant)",
    )
    parser.add_argument("--count", type=int, default=20, help="number of seeds per scenario")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--max-steps", type=int, default=None, help="episode length in steps")
    parser.add_argument(
        "--violation-samples", type=int, default=None, help="constraint samples per step"
    )
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be at least 1")
    for option in ("max_steps", "violation_samples"):
        value = getattr(args, option)
        if value is not None and value < 1:
            parser.error(f"--{option.replace('_', '-')} must be at least 1")
    return parser, args


def baselines() -> dict[str, Callable[[], Any]]:
    """Factories of every exported baseline and one mutant of `PDAgent` per kind, by name."""
    from vbar.agents import (
        MUTANT_KINDS,
        CWTrackingAgent,
        DriftAgent,
        ElevationTimedAgent,
        FehseSequenceAgent,
        GlideslopeAgent,
        LambertAgent,
        LQGAgent,
        LQRAgent,
        LyapunovAgent,
        Mutant,
        NominalTrackingAgent,
        PDAgent,
        PilotAgent,
        TwoImpulseAgent,
    )
    from vbar.oracles import OracleAgent

    factories: list[Callable[[], Any]] = [
        PDAgent,
        LQRAgent,
        LQGAgent,
        CWTrackingAgent,
        LyapunovAgent,
        TwoImpulseAgent,
        lambda: TwoImpulseAgent(mode="open_loop", name="two-impulse-open-loop"),
        lambda: TwoImpulseAgent(mode="track", name="two-impulse-track"),
        GlideslopeAgent,
        lambda: GlideslopeAgent(mode="open_loop", name="glideslope-open-loop"),
        LambertAgent,
        lambda: LambertAgent(mode="open_loop", name="lambert-open-loop"),
        FehseSequenceAgent,
        lambda: NominalTrackingAgent(TwoImpulseAgent(mode="open_loop")),
        # The STORRM terminal-phase burn's 42 deg (docs/shuttle-baselines.md).
        lambda: ElevationTimedAgent(trigger=math.radians(42.0)),
        PilotAgent,
        DriftAgent,
        OracleAgent,
    ]
    for kind in MUTANT_KINDS:
        severity = MUTANT_SEVERITY[kind]
        factories.append(lambda kind=kind, severity=severity: Mutant(PDAgent(), kind, severity))
    return {factory().name: factory for factory in factories}


def main(argv: list[str]) -> int:
    parser, args = parse(argv)
    # Loaded after the arguments are checked, so a bad command line fails without the import.
    import numpy as np

    from vbar.agents.base import agent_generator
    from vbar.metrics import optimality_gap, record_episode
    from vbar.metrics.efficiency import layout_delta_v
    from vbar.metrics.mission import success
    from vbar.oracles import DualBank
    from vbar.scenarios import make
    from vbar.scenarios.seeds import SeedSplit

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    known = baselines()
    names = args.agents or list(known)
    unknown = [name for name in names if name not in known]
    if unknown:
        parser.error(f"unknown agent {unknown[0]!r}; known: {', '.join(known)}")
    scenarios = args.scenarios or list(SCENARIOS)
    overrides = {
        key: value
        for key, value in (
            ("max_steps", args.max_steps),
            ("violation_samples", args.violation_samples),
        )
        if value is not None
    }

    width = (max(map(len, scenarios)), max(map(len, names)))
    print(
        f"{'scenario':<{width[0]}}  {'agent':<{width[1]}}  "
        + "  ".join(f"{column:>9}" for column in COLUMNS),
        flush=True,
    )
    flagged = 0
    certified = 0
    for scenario_id in scenarios:
        env = make(scenario_id, **overrides)
        view = env.agent_view()
        split = SeedSplit(scenario_id, args.split, args.count)
        start_time = time.perf_counter()
        starts = [env.reset(seed=split.seed(i))[1]["true_state"] for i in range(args.count)]
        try:
            bank = DualBank.build(view, np.array(starts))
        except ValueError as err:
            bank = None
            logging.warning("no certified bound on %s: %s", scenario_id, err)
        except RuntimeError as err:
            # A solver failure says nothing about the runs, so the scenario has no rows.
            logging.warning("not scored %s: %s", scenario_id, err)
            continue
        logging.info("%s: bank built in %.0f s", scenario_id, time.perf_counter() - start_time)
        for name in names:
            try:
                known[name]().reset(view, agent_generator(split.agent_seed(0)))
            except ValueError as err:
                logging.warning("skipped %s on %s: %s", name, scenario_id, err)
                continue
            start_time = time.perf_counter()
            counts = dict.fromkeys(COLUMNS, 0)
            raised: dict[str, list[int]] = {}
            for i in range(args.count):
                counts["runs"] += 1
                # An agent that finds no plan mid-run (the oracle agent) raises RuntimeError; the
                # run did not reach the goal, so it counts as a failure.
                try:
                    ep = record_episode(
                        env, known[name](), seed=split.seed(i), agent_seed=split.agent_seed(i)
                    )
                except RuntimeError as err:
                    raised.setdefault(str(err), []).append(i)
                    continue
                # A bank's bound holds only on the view it was built on.
                if bank is not None and ep.view != bank.view:
                    raise RuntimeError(
                        f"{scenario_id} seed index {i} ran on a view other than the bank's"
                    )
                bounds = None if bank is None else bank.bounds(ep.start)
                gap = optimality_gap(ep, bounds)
                if success(ep):
                    counts["successes"] += 1
                    counts["certified"] += bool(gap["bound_certified"])
                if gap["below_lower_bound"]:
                    counts["e3f"] += 1
                    logging.warning(
                        "E3f: %s %s %s index %d: delta-v %.9g m/s, lower %.9g, impulsive %.9g",
                        scenario_id,
                        name,
                        args.split,
                        i,
                        layout_delta_v(ep),
                        bounds.lower,
                        bounds.impulsive_lower,
                    )
            for reason, indices in raised.items():
                logging.warning(
                    "%s %s raised at %s indices %s: %s",
                    scenario_id,
                    name,
                    args.split,
                    ",".join(map(str, indices)),
                    reason,
                )
            flagged += counts["e3f"]
            certified += counts["certified"]
            logging.info(
                "%s %s: %d runs in %.0f s",
                scenario_id,
                name,
                counts["runs"],
                time.perf_counter() - start_time,
            )
            print(
                f"{scenario_id:<{width[0]}}  {name:<{width[1]}}  "
                + "  ".join(f"{counts[column]:>9}" for column in COLUMNS),
                flush=True,
            )
    if flagged:
        return 1
    if not certified:
        logging.warning("no run was held to a certified bound")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
