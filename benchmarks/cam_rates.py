"""False-CAM, missed-CAM and CAM-induced rates of the corridor CAM filter on approach-vbar.

Run with `python benchmarks/cam_rates.py [--seeds 1000] [--workers N] [--seed S]`. Each seed is
flown on approach-vbar (500 km, keep-out 50 m, box 0.02 m/s^2, dt 10 s, 360 steps) from the
scenario's own start ranges, once with `vbar.safety.corridor.CorridorCamFilter` and once
without it, by `vbar.metrics.episode.record_pair`, and scored by `vbar.safety.audit.cam_rates`.
The filter has one phase over every range: the CAM along +V-bar, away from the target for the
scenario's starts ahead of it (Fehse, *Automated Rendezvous and Docking of Spacecraft*, 2003,
section 4.5), at full thrust for ceil(0.3 m/s / (a_max dt)) = 2 decisions, 0.4 m/s.

Nominals: the PD law to the scenario's hold point; the PD law to a hold point 150 m ahead of
the target on the V-bar, whose off runs stay clear of the keep-out from most starts, so
false-CAM and CAM-induced get trials; full command straight at the target (`Inward` of
filter_audit.py); and the PD law switching to the inward one at a step drawn per seed uniformly
from 0 to 179. Each nominal flies `--seeds` seeds (1000 by default), each with an episode seed
and a different agent seed (128 bits each) and the switch step, drawn from one 128-bit root seed
that the script prints; `--seed` gives it. A rate whose denominator is empty is printed as such.

Every rate is printed with its 95% Clopper-Pearson interval; the rates characterise the filter
and nothing is asserted. Pairs are flown in `--workers` processes (all cores by default). Each
pair is scored where it is flown, since episodes do not pickle, and the counts and trials are
summed, which gives the counts `cam_rates` gives over all pairs.
"""

import argparse
import functools
import math
import os
import secrets
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from filter_audit import Inward

from vbar import ApproachEnv
from vbar.agents.pd import PDAgent
from vbar.metrics import clopper_pearson
from vbar.metrics.episode import record_pair
from vbar.safety.audit import cam_rates
from vbar.safety.corridor import CorridorCamFilter
from vbar.scenarios.approach import ApproachConfig

CFG = ApproachConfig()
CAM_DV = 0.3
PHASES = ((0.0, math.inf, (0.0, CFG.max_accel, 0.0), math.ceil(CAM_DV / (CFG.max_accel * CFG.dt))),)
# The switch step is drawn from the first half of the 360-step run.
SWITCH_STEPS = 180
HOLD_AHEAD = (0.0, 150.0, 0.0)
NOMINALS = ("pd", "pd-ahead", "inward", "pd-inward")
RATES = (("false_cam", "false-CAM"), ("missed_cam", "missed-CAM"), ("cam_induced", "CAM-induced"))


class PdAhead(PDAgent):
    """The PD law to `HOLD_AHEAD` instead of the scenario's hold point."""

    def __init__(self):
        super().__init__(name="pd-ahead")

    def reset(self, view, rng=None):
        super().reset(view, rng)
        self.hold = np.array(HOLD_AHEAD)


class PdThenInward:
    """The PD law until decision `switch`, then full command straight at the target."""

    name = "pd-inward"

    def __init__(self, switch):
        self.switch = switch

    def reset(self, view, rng=None):
        self.pd, self.inward = PDAgent(), Inward()
        self.pd.reset(view, rng)
        self.inward.reset(view, rng)
        self.k = 0

    def act(self, obs, info):
        agent = self.pd if self.k < self.switch else self.inward
        self.k += 1
        return agent.act(obs, info)


def nominal_maker(name, switch):
    """A callable returning a new agent of nominal `name` on every call."""
    if name == "pd":
        return PDAgent
    if name == "pd-ahead":
        return PdAhead
    if name == "inward":
        return Inward
    return functools.partial(PdThenInward, switch)


def draw_seed(rng):
    return int.from_bytes(rng.bytes(16), "little")


def jobs(name, count, stream):
    """(nominal maker, episode seed, agent seed) of each seed of one nominal."""
    rng = np.random.default_rng(stream)
    drawn = []
    for _ in range(count):
        seed = draw_seed(rng)
        agent_seed = draw_seed(rng)
        while agent_seed == seed:
            agent_seed = draw_seed(rng)
        switch = int(rng.integers(0, SWITCH_STEPS))
        drawn.append((nominal_maker(name, switch), seed, agent_seed))
    return drawn


def fly(job):
    make, seed, agent_seed = job
    pair = record_pair(ApproachEnv(), make, CorridorCamFilter(phases=PHASES), seed, agent_seed)
    return cam_rates([pair])


def total(rates, field, label):
    count = sum(getattr(r, field).count for r in rates)
    trials = sum(getattr(r, field).trials for r in rates)
    if trials == 0:
        return f"{label} no trials"
    lo, hi = clopper_pearson(count, trials)
    return f"{label} {count}/{trials} {count / trials:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=None)
    args = parser.parse_args()
    root = secrets.randbits(128) if args.seed is None else args.seed
    print(f"root seed {root:#034x}, phases {PHASES}", flush=True)
    streams = np.random.SeedSequence(root).spawn(len(NOMINALS))
    t_all = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for name, stream in zip(NOMINALS, streams, strict=True):
            t0 = time.perf_counter()
            per_seed = jobs(name, args.seeds, stream)
            rates = list(
                pool.map(fly, per_seed, chunksize=max(1, len(per_seed) // (4 * args.workers)))
            )
            needed = sum(r.missed_cam.trials for r in rates)
            columns = " ".join(total(rates, field, label) for field, label in RATES)
            print(
                f"{name:9s} pairs {len(rates)} needed {needed} {columns} "
                f"{time.perf_counter() - t0:.0f} s",
                flush=True,
            )
    print(f"total {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
