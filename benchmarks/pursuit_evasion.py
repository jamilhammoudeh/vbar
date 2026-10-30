"""Speed of the pe1 pursuit-evasion game against real time, in one process and in parallel.

Run with `python benchmarks/pursuit_evasion.py`. The ratio is mission time over wall-clock time;
neither reference pursuer captures V-BAR's escaping evader, so each episode lasts the full 240 s.
The numbers depend on the machine.
"""

import os
import time

from vbar.eval import Job, run_jobs
from vbar.games import PE1_E3_I3_Env, PursuitTask, fly
from vbar.games.pursuers import LineOfSightPursuer, VelocityMatchingPursuer

if __name__ == "__main__":
    for agent in (LineOfSightPursuer(), VelocityMatchingPursuer()):
        env = PE1_E3_I3_Env()
        flown = fly(env, agent)
        print(
            f"{type(agent).__name__:24s} {flown['steps']:4d} steps {env.t:5.0f} s mission "
            f"{flown['wall_s']:7.3f} s wall, {env.t / flown['wall_s']:7.1f} x real time"
        )
    workers = os.cpu_count() or 1
    jobs = [Job("pe1-e3-i3", "validation", i, 2**100 + i, 2**101 + i) for i in range(2 * workers)]
    start = time.perf_counter()
    results = run_jobs(
        jobs, "vbar.games.pursuers:VelocityMatchingPursuer", task=PursuitTask(), workers=workers
    )
    wall = time.perf_counter() - start
    assert all(result["error"] is None and not result["metrics"]["success"] for result in results)
    print(
        f"{len(jobs)} episodes on {workers} workers: {wall:.1f} s wall, "
        f"{len(jobs) * 240.0 / wall:.1f} x real time in total"
    )
