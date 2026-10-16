"""Plotting helpers (requires matplotlib: pip install vbar[plot])."""

from __future__ import annotations

import numpy as np


def plot_trajectories(results: list[dict], env, ax=None, title: str | None = None):
    """Top-down view: along-track (y, V-bar) horizontal, radial (x, R-bar) vertical.

    Draws the keep-out zone, the target, and the hold point. One line per result.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    styles = ["-", "--", "-.", ":"]
    for i, r in enumerate(results):
        s = r["states"]
        status = "violated keep-out" if r["keep_out_violated"] else "safe"
        label = f'{r["agent"]} ({status}, dv={r["dv_used_mps"]:.2f} m/s)'
        ax.plot(s[:, 1], s[:, 0], styles[i % len(styles)], lw=1.8, label=label)
        ax.plot(s[0, 1], s[0, 0], "o", color=ax.lines[-1].get_color(), ms=5)

    ko = env.cfg.keep_out_radius
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(ko * np.cos(th), ko * np.sin(th), color="0.85", label=f"keep-out ({ko:.0f} m)")
    ax.plot(0, 0, "k+", ms=10, label="target")
    hp = env.hold_point
    ax.plot(hp[1], hp[0], "k*", ms=12, label="hold point")
    ax.set_xlabel("along-track y, V-bar (m)")
    ax.set_ylabel("radial x, R-bar (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    if title:
        ax.set_title(title)
    return ax
