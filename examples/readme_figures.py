"""Regenerate the README figure and animation from fixed seeds.

    python examples/readme_figures.py

Flies the PD baseline from a start with a clear path to the hold point and from a seeded
default start with the target in the way, then writes `docs/images/pd_approach.png` and
`docs/images/pd_approach.gif`. Needs the plot extra: pip install "vbar[plot]".
"""

from pathlib import Path

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent
from vbar.viz import SERIES_COLORS, approach_animation, trajectory_figure

IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"


def main(out: Path = IMAGES) -> tuple[Path, Path]:
    clear_env = ApproachEnv(ApproachConfig(start_state=(0.0, -400.0, 0.0, 0.0, 0.0, 0.0)))
    env = ApproachEnv()
    clear = run_episode(clear_env, PDAgent(), seed=1, agent_seed=2)
    # Seed 5 starts within 2 m of the orbit plane, so the plane view shows its pass truly.
    blocked = run_episode(env, PDAgent(), seed=5, agent_seed=6)

    out.mkdir(parents=True, exist_ok=True)
    png = out / "pd_approach.png"
    fig = trajectory_figure(
        [clear, blocked], env, labels=["PD, clear path", "PD, target in the way"]
    )
    fig.savefig(png, dpi=170)
    # The second slot, so the run keeps the colour it has in the figure.
    gif = approach_animation(
        blocked,
        env,
        out / "pd_approach.gif",
        title="PD baseline, target in the way (seed 5)",
        color=SERIES_COLORS[1],
        # 800 px wide matches the README column and keeps the file near 1 MB.
        dpi=100,
    )
    return png, gif


if __name__ == "__main__":
    for path in main():
        print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
