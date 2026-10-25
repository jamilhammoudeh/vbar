"""v0 demo: why spacecraft controllers need safety filters.

Runs the classical PD baseline from two starting points:
  1. A clear path: it reaches the hold point safely.
  2. The target is in the way: it flies straight through the keep-out zone.

Then runs 30 random starts and reports the violation rate.
Saves a plot to outputs/v0_trajectories.png.
"""

from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(f'the demo needs {err.name}: pip install "vbar[plot]"') from err

matplotlib.use("Agg")
import numpy as np

from vbar import ApproachConfig, ApproachEnv, run_episode, run_many
from vbar.agents import DriftAgent, PDAgent
from vbar.viz import plot_trajectories

out = Path("outputs")
out.mkdir(exist_ok=True)

clear = ApproachEnv(ApproachConfig(start_state=(-150, -400, 0, 0, 0, 0)))
blocked = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0)))

r_clear = run_episode(clear, PDAgent())
r_blocked = run_episode(blocked, PDAgent())
r_drift = run_episode(blocked, DriftAgent())

for name, r in [("clear path", r_clear), ("target in the way", r_blocked), ("drift", r_drift)]:
    print(
        f"{name:18s} captured={r['captured']!s:5s} violated={r['keep_out_violated']!s:5s} "
        f"min_range={r['min_range_to_target_m']:7.1f} m  dv={r['dv_used_mps']:.2f} m/s  "
        f"sim={r['time_s']:.0f} s in {r['wall_time_s'] * 1000:.0f} ms"
    )

# Batch run: random starts ahead of the target (the hard case).
env = ApproachEnv()
results = run_many(env, PDAgent(), seeds=range(30))
viol = np.mean([r["keep_out_violated"] for r in results])
cap = np.mean([r["captured"] for r in results])
dv = np.median([r["dv_used_mps"] for r in results])
print(
    f"\n30 random starts: capture rate {cap:.0%}, keep-out violation rate {viol:.0%}, "
    f"median dv {dv:.2f} m/s"
)

r_clear["agent"] = "PD, clear path"
r_blocked["agent"] = "PD, target in the way"
r_drift["agent"] = "drift (no thrust)"
ax = plot_trajectories(
    [r_clear, r_blocked, r_drift],
    blocked,
    title="Approach to hold point: a naive controller flies through the keep-out zone",
)
ax.figure.savefig(out / "v0_trajectories.png", dpi=170)
print(f"\nSaved {out / 'v0_trajectories.png'}")
