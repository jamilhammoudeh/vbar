"""The figures show the keep-out violation, which way each craft flew, and the whole path."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from PIL import Image  # noqa: E402

from vbar import ApproachConfig, ApproachEnv, run_episode  # noqa: E402
from vbar.agents import PDAgent  # noqa: E402
from vbar.viz import (  # noqa: E402
    PATH_WIDTH,
    SERIES_COLORS,
    _ApproachScene,
    approach_animation,
    trajectory_figure,
)


@pytest.fixture(scope="module")
def runs():
    env = ApproachEnv()
    clear_env = ApproachEnv(ApproachConfig(start_state=(0.0, -400.0, 0.0, 0.0, 0.0, 0.0)))
    clear = run_episode(clear_env, PDAgent(), seed=1, agent_seed=2)
    blocked = run_episode(env, PDAgent(), seed=5, agent_seed=6)
    return env, clear, blocked


@pytest.fixture(scope="module")
def figure(runs):
    env, clear, blocked = runs
    return trajectory_figure([clear, blocked], env)


def by_gid(ax, gid):
    return [a for a in [*ax.lines, *ax.texts] if a.get_gid() == gid]


def test_only_the_violating_run_is_outlined_inside_the_keep_out_zone(runs, figure):
    hill, rng = figure.axes[0], figure.axes[1]
    assert by_gid(hill, "inside0") == [] and by_gid(rng, "inside_range0") == []
    (inside,) = by_gid(hill, "inside1")
    np.testing.assert_array_equal(inside.get_xydata(), runs[2]["states"][:, [1, 0]])
    assert inside.get_clip_path() is not None
    assert inside.get_color() == SERIES_COLORS[1]
    (note,) = by_gid(hill, "violated1")
    assert note.get_text() == "keep-out violated"
    assert by_gid(rng, "inside_range1")[0].get_clip_path() is not None


def test_paths_are_the_heaviest_lines_and_carry_an_arrowhead_each(figure):
    hill = figure.axes[0]
    for i in (0, 1):
        assert by_gid(hill, f"path{i}")[0].get_linewidth() == PATH_WIDTH
        assert by_gid(hill, f"heading{i}") != []
    others = [ln.get_linewidth() for ln in hill.lines if ln.get_gid() is None]
    assert max(others, default=0.0) < PATH_WIDTH
    assert max(p.get_linewidth() for p in hill.patches) < PATH_WIDTH
    assert hill.xaxis.get_gridlines()[0].get_linewidth() < PATH_WIDTH


def test_speed_panel_has_no_legend(figure):
    assert figure.axes[2].get_legend() is None


def test_animation_keeps_the_whole_path_solid_and_in_view(runs):
    env, _, blocked = runs
    s = blocked["states"]
    scene = _ApproachScene(blocked, env, "seed 5", 60.0, 4.0, 50, SERIES_COLORS[1])
    ax = scene.fig.axes[0]
    (y_lo, y_hi), (x_lo, x_hi) = ax.get_xlim(), ax.get_ylim()
    assert y_lo < s[:, 1].min() and s[:, 1].max() < y_hi
    assert x_lo < s[:, 0].min() and s[:, 0].max() < x_hi
    inside = np.linalg.norm(s[:, :3], axis=1) < env.cfg.keep_out_radius
    entry = int(np.argmax(inside))
    for k in (entry - 1, entry, len(s) - 1):
        scene.frame(k)
        np.testing.assert_array_equal(scene.past.get_xydata(), s[: k + 1, [1, 0]])
        assert scene.past.get_linewidth() == PATH_WIDTH
        assert scene.past.get_alpha() in (None, 1.0)
        assert scene.violated.get_visible() == (k >= entry)


def test_every_gif_frame_is_stored_whole(runs, tmp_path):
    env, _, blocked = runs
    path = approach_animation(blocked, env, tmp_path / "a.gif", width_in=3.0, dpi=40)
    with Image.open(path) as gif:
        for k in range(gif.n_frames):
            gif.seek(k)
            assert gif.tile[0][1] == (0, 0, *gif.size), k


def test_violation_note_sits_inside_the_hill_axes_on_every_frame_it_shows(runs):
    env, _, blocked = runs
    scene = _ApproachScene(blocked, env, "seed 5", 60.0, 4.0, 50, SERIES_COLORS[1])
    renderer = scene.canvas.get_renderer()
    hill = scene.fig.axes[0].get_window_extent(renderer)
    assert scene.k_entry is not None and scene.k_entry < scene.k_closest
    for k in range(scene.k_entry, scene.k_closest + 1):
        scene.frame(k)
        box = scene.violated.get_window_extent(renderer)
        assert hill.x0 <= box.x0 and box.x1 <= hill.x1, k
        assert hill.y0 <= box.y0 and box.y1 <= hill.y1, k


def test_violation_between_recorded_states_is_noted_once_the_closest_approach_is_passed(runs):
    env, _, blocked = runs
    s = blocked["states"]
    # The record still says the zone was entered, but no recorded state lies inside it.
    outside = np.linalg.norm(s[:, :3], axis=1) >= env.cfg.keep_out_radius
    record = {**blocked, "states": s[outside]}
    scene = _ApproachScene(record, env, "seed 5", 60.0, 4.0, 50, SERIES_COLORS[1])
    assert scene.k_entry is None
    shown = min(scene.k_closest + 1, len(record["states"]) - 1)
    for k in range(len(record["states"])):
        scene.frame(k)
        assert scene.violated.get_visible() == (k >= shown), k


def test_hill_panel_labels_the_orbital_velocity_arrow(figure):
    texts = [t.get_text() for t in figure.axes[0].texts]
    assert "orbital velocity" in texts
