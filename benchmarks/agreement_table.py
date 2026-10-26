"""The agreement table of V-BAR's two-body plus J2 rung against Basilisk and Orekit.

python benchmarks/agreement_table.py    (needs the basilisk and orekit extras; about half an hour)

Coast arcs at three ranges over one and ten orbits, and a controlled episode (the PD baseline on
the V-bar approach) replayed open loop. Each residual is attributed by changing one setting at a
time: V-BAR's integrator tolerance, the backend's, and the substep over which a backend holds a
burn's inertial direction.
"""

import math
from functools import partial

import numpy as np

from vbar.agents import PDAgent
from vbar.dynamics import EARTH
from vbar.dynamics.backends import basilisk, hill_states, orekit
from vbar.dynamics.backends.agreement import coast_cases, episode_burns, residual, vbar_arc
from vbar.metrics import record_episode
from vbar.scenarios import ApproachConfig, ApproachEnv

ALTITUDE, INCLINATION = 500e3, math.radians(51.6)
BACKENDS = {
    "Basilisk": (
        partial(basilisk.propagate, substep=10.0, tolerance=1e-12),
        partial(basilisk.propagate, substep=10.0, tolerance=1e-10),
    ),
    "Orekit": (
        partial(orekit.propagate, tolerance=1e-6),
        partial(orekit.propagate, tolerance=1e-4),
    ),
}


def coasts():
    n = math.sqrt(EARTH.mu / (EARTH.radius + ALTITUDE) ** 3)
    rows = []
    for label, rel, orbits in coast_cases():
        span = 100.0 * math.floor(orbits * 2.0 * math.pi / n / 100.0)
        times = np.arange(0.0, span + 50.0, 100.0)
        ours, start_t, start_c = vbar_arc(rel, times, rtol=1e-12)
        looser, _, _ = vbar_arc(rel, times, rtol=1e-10)
        for name, (tight, loose) in BACKENDS.items():
            found = residual(
                label,
                name,
                ours,
                hill_states(*tight(start_t, start_c, times, EARTH, True), EARTH, True),
            )
            backend_loose = residual(
                label,
                name,
                ours,
                hill_states(*loose(start_t, start_c, times, EARTH, True), EARTH, True),
            )
            ours_loose = residual(
                label,
                name,
                looser,
                hill_states(*tight(start_t, start_c, times, EARTH, True), EARTH, True),
            )
            rows.append((found, backend_loose, ours_loose))
    return rows


def episode():
    env = ApproachEnv(ApproachConfig(start_state=(30.0, -400.0, 10.0, 0.0, 0.0, 0.0)))
    recorded = record_episode(env, PDAgent(), seed=1, agent_seed=2)
    burns = episode_burns(recorded.records)
    times = np.arange(0.0, recorded.duration + 0.5, 10.0)
    ours, start_t, start_c = vbar_arc(recorded.start, times, burns)
    rows = []
    for name, module in (("Basilisk", basilisk), ("Orekit", orekit)):
        found = []
        for substep in (1.0, 0.5):
            target, chaser = module.propagate(
                start_t, start_c, times, EARTH, True, burns=burns, substep=substep
            )
            found.append(
                residual(
                    f"PD episode, {len(burns)} burns",
                    name,
                    ours,
                    hill_states(target, chaser, EARTH, True),
                )
            )
        rows.append(found)
    return rows


def main():
    head = ("Position (m)", "Velocity (m/s)", "Loosened backend (m)", "Loosened V-BAR (m)")
    print("| Case | Backend | " + " | ".join(head) + " |")
    print("|---|---|" + "---|" * len(head))
    for found, backend_loose, ours_loose in coasts():
        print(
            f"| {found.case} | {found.backend} | {found.position:.2e} | {found.velocity:.2e} | "
            f"{backend_loose.position:.2e} | {ours_loose.position:.2e} |"
        )
    print()
    head = ("Position, 1 s substep (m)", "Position, 0.5 s substep (m)", "Velocity, 1 s (m/s)")
    print("| Case | Backend | " + " | ".join(head) + " |")
    print("|---|---|" + "---|" * len(head))
    for one, half in episode():
        cells = (one.position, half.position, one.velocity)
        print(f"| {one.case} | {one.backend} | " + " | ".join(f"{c:.2e}" for c in cells) + " |")


if __name__ == "__main__":
    main()
