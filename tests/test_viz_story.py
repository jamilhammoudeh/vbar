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
    env = runs[0]
    r_ko, tol = env.cfg.keep_out_radius, 1e-6 * env.cfg.keep_out_radius
    hill, rng = figure.axes[0], figure.axes[1]
    assert by_gid(hill, "inside0") == [] and by_gid(rng, "inside_range0") == []
    (inside,) = by_gid(hill, "inside1")
    assert inside.get_color() == SERIES_COLORS[1]
    assert inside.get_clip_path() is None
    uv = inside.get_xydata()
    uv = uv[np.isfinite(uv).all(axis=1)]
    assert np.hypot(uv[:, 0], uv[:, 1]).max() <= r_ko + tol
    assert abs(np.hypot(*uv[0]) - r_ko) < tol and abs(np.hypot(*uv[-1]) - r_ko) < tol
    (note,) = by_gid(hill, "violated1")
    assert note.get_text() == "keep-out violated"
    (inside_range,) = by_gid(rng, "inside_range1")
    assert inside_range.get_clip_path() is None
    rg = inside_range.get_ydata()
    rg = rg[np.isfinite(rg)]
    assert rg.max() <= r_ko + tol
    assert abs(rg[0] - r_ko) < tol and abs(rg[-1] - r_ko) < tol


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


def test_animation_outlines_only_the_part_inside_the_zone_up_to_the_marker(runs):
    env, _, blocked = runs
    r_ko = env.cfg.keep_out_radius
    tol = 1e-6 * r_ko
    s = blocked["states"]
    scene = _ApproachScene(blocked, env, "seed 5", 60.0, 4.0, 50, SERIES_COLORS[1])
    inside = np.linalg.norm(s[:, :3], axis=1) < r_ko
    entry = int(np.argmax(inside))
    exit_ = entry + int(np.argmax(~inside[entry:]))
    k = (entry + exit_) // 2
    assert entry < k < exit_
    scene.frame(k)
    uv = scene.inside.get_xydata()
    uv = uv[np.isfinite(uv).all(axis=1)]
    assert abs(np.hypot(*uv[0]) - r_ko) < tol
    np.testing.assert_allclose(uv[-1], [s[k, 1], s[k, 0]])
    assert np.hypot(uv[:, 0], uv[:, 1]).max() <= r_ko + tol


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


def _range_line(states, dt):
    """Range (m) against time (min), as the range panel plots it."""
    return np.column_stack(
        [np.arange(len(states)) * dt / 60.0, np.linalg.norm(states[:, :3], axis=1)]
    )


def _range_crossings(states, dt, r_ko):
    """Points (t, r) where range, linear between recorded states, crosses `r_ko`, with the step."""
    t, rng = _range_line(states, dt).T
    out = []
    for j in np.flatnonzero(np.diff((rng < r_ko).astype(int))):
        f = (r_ko - rng[j]) / (rng[j + 1] - rng[j])
        out.append((np.array([t[j] + f * (t[j + 1] - t[j]), r_ko]), j))
    return out


def _circle_crossings(states, r_ko):
    """Points (y, x) where the in-plane path, straight between states, meets the circle."""
    p = states[:, [1, 0]]
    out = []
    for j in range(len(p) - 1):
        a, d = p[j], p[j + 1] - p[j]
        qa, qb, qc = d @ d, 2 * a @ d, a @ a - r_ko**2
        disc = qb**2 - 4 * qa * qc
        if disc <= 0:
            continue
        for s in ((-qb - np.sqrt(disc)) / (2 * qa), (-qb + np.sqrt(disc)) / (2 * qa)):
            if 0 < s < 1:
                out.append((a + s * d, j))
    return out


def _ends(line):
    uv = line.get_xydata()
    uv = uv[np.isfinite(uv).all(axis=1)]
    return uv[0], uv[-1]


def test_keep_out_emphasis_ends_at_the_interpolated_crossings(runs, figure):
    env, _, blocked = runs
    s, r_ko = blocked["states"], env.cfg.keep_out_radius
    crossings = [c for c, _ in _range_crossings(s, env.cfg.dt, r_ko)]
    (rng_line,) = by_gid(figure.axes[1], "inside_range1")
    for end, crossing in zip(_ends(rng_line), crossings, strict=True):
        np.testing.assert_allclose(end, crossing, atol=1e-9)
    p_in, p_out = (c for c, _ in _circle_crossings(s, r_ko))
    (hill_line,) = by_gid(figure.axes[0], "inside1")
    for end, crossing in zip(_ends(hill_line), (p_in, p_out), strict=True):
        np.testing.assert_allclose(end, crossing, atol=1e-9)
        assert np.hypot(*end) == pytest.approx(r_ko)
    step = np.max(np.hypot(*np.diff(s[:, [1, 0]], axis=0).T))
    inside = s[np.hypot(s[:, 0], s[:, 1]) < r_ko][:, [1, 0]]
    assert np.hypot(*(p_in - inside[0])) < step and np.hypot(*(p_out - inside[-1])) < step


def test_keep_out_outline_pixels_stop_at_the_crossings(runs, figure):
    env, _, blocked = runs
    s, r_ko = blocked["states"], env.cfg.keep_out_radius
    dpi = 170
    figure.set_dpi(dpi)
    (note,) = by_gid(figure.axes[0], "violated1")
    note.set_visible(False)
    try:
        figure.canvas.draw()
        rgb = np.asarray(figure.canvas.buffer_rgba())[..., :3].astype(int)
    finally:
        note.set_visible(True)
    red = np.abs(rgb - [0xA3, 0x20, 0x1A]).sum(axis=2) < 40
    height = rgb.shape[0]
    # Half the outline stroke, in px; a butt end spreads it sideways by its slope at the cut.
    half = (PATH_WIDTH + 4.2) / 2 * dpi / 72
    cases = (
        (figure.axes[0], s[:, [1, 0]], _circle_crossings(s, r_ko)),
        (figure.axes[1], _range_line(s, env.cfg.dt), _range_crossings(s, env.cfg.dt, r_ko)),
    )
    for ax, uv, crossings in cases:
        box = ax.get_window_extent()
        rows = slice(height - int(box.y1), height - int(box.y0))
        cols = np.flatnonzero(red[rows, int(box.x0) : int(box.x1)].any(axis=0)) + int(box.x0)
        assert cols.size, ax.get_title()
        ends = []
        for c, j in crossings:
            d = np.diff(ax.transData.transform(uv[j : j + 2]), axis=0)[0]
            ends.append((ax.transData.transform(c)[0], half * abs(d[1]) / np.hypot(*d)))
        (x_lo, spread_lo), (x_hi, spread_hi) = sorted(ends)
        assert x_lo - spread_lo - 2 <= cols[0] <= x_lo + 2, ax.get_title()
        assert x_hi - 2 <= cols[-1] + 1 <= x_hi + spread_hi + 2, ax.get_title()


def test_each_pass_through_the_zone_is_cut_at_its_own_crossings():
    from vbar.viz import _inside

    # Radius 5: in at u = -3 and out at u = 3 along v = 4 with a state inside, then a single
    # step along u = 4 from two outside states whose chord crosses at v = -3 and v = 3.
    uv = np.array([[-10.0, 4.0], [0.0, 4.0], [10.0, 4.0], [10.0, 0.0], [4.0, -10.0], [4.0, 10.0]])
    parts = _inside(uv, 5.0, slice(0, 2))
    breaks = np.flatnonzero(np.isnan(parts[:, 0]))
    first, second = np.split(parts, breaks)
    np.testing.assert_allclose(first, [[-3.0, 4.0], [0.0, 4.0], [3.0, 4.0]])
    np.testing.assert_allclose(second[1:], [[4.0, -3.0], [4.0, 3.0]])
    assert _inside(uv[:1], 5.0, slice(0, 2)).shape == (0, 2)
    assert _inside(uv[2:4], 5.0, slice(0, 2)).shape == (0, 2)
