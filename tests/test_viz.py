"""Figures render headless from seeded runs, draw the record's values, and keep the screen style.

The pixel statistics in `tests/golden/viz_figure.json` catch unintended style changes without a
byte-exact image. After an intended change to the style, run `python tests/test_viz.py` to
regenerate them and look at the figure before committing.
"""

import json
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.colors import to_rgb  # noqa: E402
from PIL import Image  # noqa: E402

from vbar import ApproachConfig, ApproachEnv, run_episode  # noqa: E402
from vbar.agents import PDAgent  # noqa: E402
from vbar.viz import (  # noqa: E402
    GRID,
    KEEP_OUT_FILL,
    SERIES_COLORS,
    SURFACE,
    _ApproachScene,
    approach_animation,
    plot_trajectories,
    screen_style,
    trajectory_figure,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden" / "viz_figure.json"
LABELS = ["PD, clear path", "PD, target in the way"]


def fly():
    env = ApproachEnv()
    clear_env = ApproachEnv(ApproachConfig(start_state=(0.0, -400.0, 0.0, 0.0, 0.0, 0.0)))
    clear = run_episode(clear_env, PDAgent(), seed=1, agent_seed=2)
    blocked = run_episode(env, PDAgent(), seed=5, agent_seed=6)
    return env, clear, blocked


@pytest.fixture(scope="module")
def runs():
    return fly()


@pytest.fixture(scope="module")
def figure(runs):
    env, clear, blocked = runs
    fig = trajectory_figure([clear, blocked], env, labels=LABELS)
    fig.canvas.draw()
    return fig


def line(fig, gid):
    return next(ln for ax in fig.axes for ln in ax.lines if ln.get_gid() == gid)


def pixel_stats(fig) -> dict:
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].reshape(-1, 3).astype(int)
    colours, counts = np.unique(rgb, axis=0, return_counts=True)

    def share(colour, tol):
        target = np.round(np.array(to_rgb(colour)) * 255)
        return float(np.mean(np.all(np.abs(rgb - target) <= tol, axis=1)))

    return {
        "size_px": [int(v) for v in fig.canvas.get_width_height()],
        "surface": "#{:02x}{:02x}{:02x}".format(*colours[np.argmax(counts)]),
        "series": [share(c, 12) for c in SERIES_COLORS[:2]],
        "grid": share(GRID, 3),
        "keep_out": share(KEEP_OUT_FILL, 3),
        "ink": float(np.mean(rgb.sum(axis=1) < 3 * 90)),
    }


def test_screen_style_sets_its_values_and_restores_the_settings():
    before = dict(matplotlib.rcParams)
    with screen_style():
        rc = matplotlib.rcParams
        assert rc["figure.facecolor"] == rc["axes.facecolor"] == "#fcfcfb"
        assert rc["text.color"] == "#0b0b0b"
        assert (rc["grid.color"], rc["grid.linewidth"]) == ("#e6e5e1", 0.6)
        assert not rc["axes.spines.top"] and not rc["axes.spines.right"]
        assert (rc["axes.titlelocation"], rc["axes.titleweight"]) == ("left", "bold")
        assert rc["xtick.direction"] == rc["ytick.direction"] == "out"
        assert rc["savefig.dpi"] >= 170
    assert dict(matplotlib.rcParams) == before


def test_a_controller_keeps_its_colour_slot_across_figures():
    assert SERIES_COLORS == ("#2a78d6", "#eb6834", "#1baf7a")


def test_figure_draws_the_recorded_states(runs, figure):
    env, clear, blocked = runs
    for i, r in enumerate([clear, blocked]):
        s = r["states"]
        t_min = np.arange(len(s)) * env.cfg.dt / 60.0
        path = line(figure, f"path{i}")
        np.testing.assert_array_equal(path.get_xdata(), s[:, 1])
        np.testing.assert_array_equal(path.get_ydata(), s[:, 0])
        rng = line(figure, f"range{i}")
        np.testing.assert_array_equal(rng.get_xdata(), t_min)
        np.testing.assert_array_equal(rng.get_ydata(), np.linalg.norm(s[:, :3], axis=1))
        speed = line(figure, f"speed{i}")
        np.testing.assert_array_equal(speed.get_ydata(), np.linalg.norm(s[:, 3:6], axis=1))
    notes = [t.get_text() for t in figure.axes[0].texts if t.get_gid() == "closest1"]
    assert notes == [f"closest {blocked['min_range_to_target_m']:.1f} m"]
    assert not clear["keep_out_violated"] and blocked["keep_out_violated"]


def test_labels_do_not_overlap_or_leave_their_panel(figure):
    from matplotlib.text import Text

    renderer = figure.canvas.get_renderer()
    for ax in figure.axes:
        frame = ax.get_window_extent(renderer)
        # The text alone: a leader line may cross another label's margin.
        boxes = [Text.get_window_extent(t, renderer) for t in ax.texts if t.get_text()]
        for k, box in enumerate(boxes):
            assert frame.x0 - 1 <= box.x0 and box.x1 <= frame.x1 + 1
            assert frame.y0 - 1 <= box.y0 and box.y1 <= frame.y1 + 1
            assert not any(box.overlaps(other) for other in boxes[k + 1 :])


def test_figure_pixels_match_the_reference_statistics(figure):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    stats = pixel_stats(figure)
    assert stats["surface"] == golden["surface"] == SURFACE
    assert stats["size_px"] == pytest.approx(golden["size_px"], abs=2)
    # Shares of the image in each colour; text rendering differs slightly between platforms.
    assert stats["series"] == pytest.approx(golden["series"], rel=0.15)
    for key in ("grid", "keep_out", "ink"):
        assert stats[key] == pytest.approx(golden[key], rel=0.15), key


def test_plot_trajectories_draws_into_given_axes_and_limits_the_series(runs):
    from matplotlib.figure import Figure

    env, clear, blocked = runs
    ax = Figure().add_subplot()
    assert plot_trajectories([clear, blocked], env, ax=ax) is ax
    np.testing.assert_array_equal(line(ax.figure, "path0").get_xdata(), clear["states"][:, 1])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["pd", "pd"]
    with pytest.raises(ValueError, match="at most 3 runs"):
        plot_trajectories([clear] * 4, env, ax=Figure().add_subplot())
    with pytest.raises(ValueError, match="one entry per run"):
        trajectory_figure([clear], env, labels=["a", "b"])


def test_animation_marker_and_readout_follow_the_record(runs):
    env, _, blocked = runs
    s = blocked["states"]
    scene = _ApproachScene(blocked, env, "seed 5", trail_s=60.0, width_in=4.0, dpi=50)
    for k in (0, scene.k_closest, len(s) - 1):
        scene.frame(k)
        np.testing.assert_array_equal(scene.marker.get_xydata(), s[k : k + 1, [1, 0]])
        trail = scene.fade.get_segments()
        if k:
            np.testing.assert_array_equal(trail[-1][-1], s[k, [1, 0]])
        readout = [t.get_text() for t in scene.readout]
        assert readout[1] == f"{np.linalg.norm(s[k, :3]):.0f} m"
        assert ("inside keep-out" in readout) == (np.linalg.norm(s[k, :3]) < 50.0)
        assert scene.closest.get_visible() == (k >= scene.k_closest)


def test_animation_is_a_gif_with_one_frame_per_state(runs, tmp_path):
    env, _, blocked = runs
    path = approach_animation(blocked, env, tmp_path / "a.gif", width_in=3.0, dpi=40)
    with Image.open(path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == len(blocked["states"])
        corner = gif.convert("RGB").getpixel((1, 1))
    assert corner == tuple(round(255 * c) for c in to_rgb(SURFACE))


def test_readme_shows_the_committed_assets_within_size():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name, limit in (("pd_approach.png", 1e6), ("pd_approach.gif", 3e6)):
        assert f"docs/images/{name}" in readme
        assert (ROOT / "docs" / "images" / name).stat().st_size < limit


if __name__ == "__main__":
    env, clear, blocked = fly()
    stats = pixel_stats(trajectory_figure([clear, blocked], env, labels=LABELS))
    GOLDEN.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN}")
