"""Hill-frame figures and animations of approach runs (needs pip install "vbar[plot]").

Positions are drawn in the orbit plane with the along-track axis (y, V-bar) horizontal and the
radial axis (x, R-bar) up, at equal aspect, so the keep-out sphere's cross-section is a circle.
Every plotted value is read from a run record (`run_episode`) or the scenario configuration.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

_INSTALL = 'pip install "vbar[plot]"'

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
KEEP_OUT_FILL = "#ebe9e3"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")

SCREEN_STYLE = MappingProxyType(
    {
        # Bundled with matplotlib, so a figure renders the same on every platform.
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "figure.dpi": 100.0,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 170.0,
        "pdf.fonttype": 42,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 8.0,
        "legend.handlelength": 1.6,
    }
)

_NOTE = {"fontsize": 7.5, "color": INK_SECONDARY, "linespacing": 1.2}


def _matplotlib():
    try:
        import matplotlib
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(f"plotting needs {err.name}: {_INSTALL}") from err
    return matplotlib


@contextmanager
def screen_style() -> Iterator[None]:
    """Apply the screen style to every figure made inside a ``with`` block.

    The style is used for the README, the documentation and reports: a warm off-white surface,
    near-black text, a light grid, no top or right spines and left-aligned bold titles. Series
    colours come from `SERIES_COLORS` in a fixed order. The previous matplotlib settings are
    restored on exit.

    Yields
    ------
    None

    Raises
    ------
    ModuleNotFoundError
        If matplotlib is not installed.
    """
    with _matplotlib().rc_context(dict(SCREEN_STYLE)):
        yield


def _times(result: dict, env) -> np.ndarray:
    # States are recorded once per decision, `dt` apart, from t = 0.
    return np.arange(len(result["states"])) * env.cfg.dt


def _check_count(results: Sequence[dict]) -> None:
    if len(results) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} runs per figure, got {len(results)}")


def _labels(results: Sequence[dict], labels: Sequence[str] | None) -> list[str]:
    if labels is None:
        return [str(r["agent"]) for r in results]
    if len(labels) != len(results):
        raise ValueError("labels must have one entry per run")
    return list(labels)


def _hold_ring():
    """A dashed ring marker, so the hold point keeps its size at any scale."""
    from matplotlib.path import Path as MplPath

    arcs = [MplPath.arc(a, a + 25.0) for a in np.arange(0.0, 360.0, 45.0)]
    return MplPath.make_compound_path(*arcs)


def _dense(line: np.ndarray, spacing: float) -> np.ndarray:
    """Points along a polyline at most `spacing` apart, for label placement only."""
    if len(line) < 2:
        return line
    seg = np.hypot(*np.diff(line, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.linspace(0.0, s[-1], max(2, int(s[-1] / spacing) + 2))
    return np.column_stack([np.interp(u, s, line[:, 0]), np.interp(u, s, line[:, 1])])


class _Labels:
    """Puts each label on the side of its anchor that covers the fewest plotted points.

    Positions are worked out in pixels from one layout pass; a label never overlaps one placed
    before it or leaves the axes unless no side is free.
    """

    def __init__(self, ax, lines: Sequence[np.ndarray]):
        from matplotlib.backends.backend_agg import RendererAgg
        from matplotlib.text import Text

        fig = ax.figure
        fig.draw_without_rendering()
        self.ax = ax
        # Agg measures text the same as the vector backends, for canvases without a renderer.
        if hasattr(fig.canvas, "get_renderer"):
            self.renderer = fig.canvas.get_renderer()
        else:
            width, height = fig.bbox.size
            self.renderer = RendererAgg(int(width), int(height), fig.dpi)
        self.px = fig.dpi / 72.0
        self.points = np.zeros((0, 2))
        for line in lines:
            self.avoid(line)
        self.frame = ax.get_window_extent(self.renderer)
        self.taken: list[tuple[float, float, float, float]] = []
        self._size = lambda t: Text.get_window_extent(t, self.renderer).size

    def avoid(self, line: np.ndarray) -> None:
        """Keep labels off a polyline given in data coordinates."""
        on_screen = self.ax.transData.transform(np.asarray(line, dtype=float).reshape(-1, 2))
        self.points = np.concatenate([self.points, _dense(on_screen, 1.5)])

    def _score(self, box) -> float:
        x0, y0, x1, y1 = box
        p = self.points
        covered = np.count_nonzero(
            (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
        )
        f = self.frame
        # The padding may reach past the axes edge; the text itself may not.
        outside = x0 + 3 < f.x0 or y0 + 3 < f.y0 or x1 - 3 > f.x1 or y1 - 3 > f.y1
        clash = any(x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1 for b in self.taken)
        return covered + 1000.0 * (outside + clash)

    def place(self, text: str, candidates, gaps_pt: Sequence[float], **kw):
        """Annotate `text` at the best of `candidates`, each (data xy, (sx, sy)) side of it.

        Nearer gaps (pt) are tried first; a farther one wins only if it covers fewer points.
        """
        ax = self.ax
        ann = ax.annotate(
            text,
            candidates[0][0],
            xytext=(0.0, 0.0),
            textcoords="offset points",
            ha="center",
            va="center",
            **kw,
        )
        w, h = self._size(ann)
        scored = []
        options = [(g * self.px, xy, side) for g in gaps_pt for xy, side in candidates]
        for rank, (gap, xy, (sx, sy)) in enumerate(options):
            dx, dy = sx * (w / 2 + gap), sy * (h / 2 + gap)
            cx, cy = ax.transData.transform(xy) + (dx, dy)
            box = (cx - w / 2 - 3, cy - h / 2 - 3, cx + w / 2 + 3, cy + h / 2 + 3)
            scored.append((self._score(box) + 0.1 * rank, xy, box, (dx, dy)))
        _, xy, box, offset = min(scored, key=lambda option: option[0])
        ann.xy = xy
        ann.xyann = (offset[0] / self.px, offset[1] / self.px)
        self.taken.append(box)
        return ann

    def corner(self, width: float, height: float, order: Sequence[str]) -> tuple[float, ...]:
        """The emptiest corner box of `width` by `height` px inside the axes, in pixels."""
        f, inset = self.frame, 6 * self.px
        boxes = []
        for name in order:
            x0 = f.x1 - inset - width if "right" in name else f.x0 + inset
            y0 = f.y1 - inset - height if "top" in name else f.y0 + inset
            boxes.append((x0, y0, x0 + width, y0 + height))
        rank = {box: k for k, box in enumerate(boxes)}
        box = min(boxes, key=lambda b: self._score(b) + 0.1 * rank[b])
        self.taken.append(box)
        return box

    def to_axes(self, x: float, y: float) -> tuple[float, float]:
        f = self.frame
        return (x - f.x0) / f.width, (y - f.y0) / f.height

    def flight_key(self, note: str | None) -> None:
        """A "direction of flight" arrow along +y, with `note` under it, in an empty corner."""
        ax = self.ax
        head = ax.text(
            0, 0, "direction of flight", transform=ax.transAxes, ha="right", va="center", **_NOTE
        )
        sub = None
        if note:
            sub = ax.text(
                0,
                0,
                note,
                transform=ax.transAxes,
                ha="right",
                va="center",
                **{**_NOTE, "color": MUTED},
                gid="tick_key",
            )
        arrow, gap = 24 * self.px, 4 * self.px
        w1, h1 = self._size(head)
        w2, h2 = self._size(sub) if sub else (0.0, 0.0)
        line = 3 * self.px if sub else 0.0
        x0, y0, x1, y1 = self.corner(
            max(w1 + gap + arrow, w2), h1 + line + h2, ("top right", "bottom right", "top left")
        )
        y_head = y1 - h1 / 2
        head.set_position(self.to_axes(x1 - arrow - gap, y_head))
        ax.annotate(
            "",
            xy=self.to_axes(x1, y_head),
            xytext=self.to_axes(x1 - arrow, y_head),
            xycoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "color": INK_SECONDARY,
                "lw": 0.9,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        if sub:
            sub.set_position(self.to_axes(x1, y0 + h2 / 2))


_AROUND = ((0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1))


def _hill_limits(results: Sequence[dict], env) -> tuple[tuple[float, float], tuple[float, float]]:
    r_ko = env.cfg.keep_out_radius
    hold = np.asarray(env.hold_point, dtype=float)
    y = np.concatenate([r["states"][:, 1] for r in results] + [[-r_ko, r_ko, hold[1]]])
    x = np.concatenate([r["states"][:, 0] for r in results] + [[-r_ko, r_ko, hold[0]]])
    # Room around the data for the start labels and the direction-of-flight key.
    pad_y = 0.05 * (y.max() - y.min())
    pad_x = max(0.2 * (x.max() - x.min()), 0.07 * (y.max() - y.min()))
    return (y.min() - pad_y, y.max() + pad_y), (x.min() - pad_x, x.max() + pad_x)


def _draw_hill(
    ax, results, env, names, tick_s, *, legend: bool, closest_note: bool, colors=SERIES_COLORS
):
    from matplotlib.patches import Circle

    cfg = env.cfg
    r_ko = cfg.keep_out_radius
    hold = np.asarray(env.hold_point, dtype=float)
    step = max(1, round(tick_s / cfg.dt))
    (y_lo, y_hi), (x_lo, x_hi) = _hill_limits(results, env)

    ax.add_patch(Circle((0.0, 0.0), r_ko, facecolor=KEEP_OUT_FILL, edgecolor=MUTED, lw=0.7))
    ax.plot(0.0, 0.0, "s", ms=4.5, color=INK, zorder=4)
    ax.plot(hold[1], hold[0], marker=_hold_ring(), ms=13, mew=1.0, mfc="none", mec=INK, zorder=8)
    for i, r in enumerate(results):
        color = colors[i]
        y, x = r["states"][:, 1], r["states"][:, 0]
        ax.plot(y, x, color=color, zorder=5, label=names[i], gid=f"path{i}")
        ax.plot(
            y[step::step],
            x[step::step],
            "o",
            ms=2.8,
            color=color,
            mec=SURFACE,
            mew=0.6,
            zorder=6,
            gid=f"ticks{i}",
        )
        ax.plot(y[0], x[0], "o", ms=7, mfc=SURFACE, mec=color, mew=1.8, zorder=7)
    ax.set_xlim(y_lo, y_hi)
    ax.set_ylim(x_lo, x_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("along-track y (V-bar), m")
    ax.set_ylabel("radial x (R-bar), m")

    angle = np.linspace(0.0, 2.0 * np.pi, 73)
    ring = 0.06 * (y_hi - y_lo) * np.column_stack([np.cos(angle), np.sin(angle)])
    obstacles = [r_ko * ring / ring[0, 0], np.zeros((1, 2)), hold[[1, 0]] + 0.15 * ring]
    obstacles += [r["states"][:, [1, 0]] for r in results]
    labels = _Labels(ax, obstacles)

    for i, r in enumerate(results):
        start = (r["states"][0, 1], r["states"][0, 0])
        labels.place(
            names[i],
            [(start, side) for side in _AROUND],
            (6,),
            fontsize=8,
            weight="bold",
            color=INK,
            gid=f"name{i}",
        )
    labels.place("hold point", [((hold[1], hold[0]), s) for s in _AROUND[:4]], (8,), **_NOTE)
    labels.place(
        "target",
        [((0.0, 0.0), s) for s in ((0, -1), (0, 1), (1, 0), (-1, 0))],
        (4,),
        **{**_NOTE, "color": INK},
    )
    diagonals = [(1, -1), (-1, -1), (1, 1), (-1, 1)]
    labels.place(
        f"keep-out\n{r_ko:g} m radius",
        [((r_ko * sx / np.sqrt(2), r_ko * sy / np.sqrt(2)), (sx, sy)) for sx, sy in diagonals],
        (4,),
        **_NOTE,
    )
    if closest_note:
        for i, r in enumerate(results):
            if not r["keep_out_violated"]:
                continue
            k = int(np.argmin(np.linalg.norm(r["states"][:, :3], axis=1)))
            at = (r["states"][k, 1], r["states"][k, 0])
            labels.place(
                f"closest {r['min_range_to_target_m']:.1f} m",
                [(at, side) for side in _AROUND],
                (22, 40),
                **{**_NOTE, "color": INK},
                arrowprops={
                    "arrowstyle": "-",
                    "color": INK_SECONDARY,
                    "lw": 0.7,
                    "shrinkA": 1,
                    "shrinkB": 3,
                },
                zorder=9,
                gid=f"closest{i}",
            )
    labels.flight_key(f"dots every {tick_s:g} s" if results else None)
    if legend and len(results) >= 2:
        ax.legend(loc="best")
    return labels


def plot_trajectories(
    results: list[dict],
    env,
    ax=None,
    title: str | None = None,
    *,
    labels: Sequence[str] | None = None,
    tick_s: float = 60.0,
):
    """Draw approach runs in the orbit plane of the target's Hill frame.

    Along-track y (V-bar) is horizontal and radial x (R-bar) up, at equal aspect. The target is
    a square at the origin, the keep-out zone a filled disk with its radius, the hold point a
    dashed ring. Each run is a line with dots every `tick_s` and an open circle at its start,
    labelled directly; a run that entered the keep-out zone is annotated with its closest
    approach from the record. Cross-track motion (z) is not drawn.

    Parameters
    ----------
    results : list of dict
        At most three run records from `run_episode`, drawn in `SERIES_COLORS` order.
    env : ApproachEnv
        The environment the runs were flown in; supplies ``cfg.dt`` (s),
        ``cfg.keep_out_radius`` (m) and ``hold_point`` (Hill frame, m).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure in the screen style is made if None.
    title : str, optional
        Axes title.
    labels : sequence of str, optional
        One name per run for the direct labels and legend; defaults to each record's ``agent``.
    tick_s : float, optional
        Interval between time dots along each path, s.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    ModuleNotFoundError
        If matplotlib is not installed.
    ValueError
        If there are more than three runs or `labels` does not match them.
    """
    _matplotlib()
    _check_count(results)
    names = _labels(results, labels)
    with screen_style():
        if ax is None:
            import matplotlib.pyplot as plt

            height = 3.6
            if results:
                (y_lo, y_hi), (x_lo, x_hi) = _hill_limits(results, env)
                height = float(np.clip(6.45 * (x_hi - x_lo) / (y_hi - y_lo), 1.4, 6.0)) + 0.9
            _, ax = plt.subplots(figsize=(7.2, height), layout="constrained")
        if title:
            ax.set_title(title)
        _draw_hill(ax, results, env, names, tick_s, legend=True, closest_note=True)
    return ax


def _draw_series(ax, results, env, names, key: str) -> list[np.ndarray]:
    lines = []
    for i, r in enumerate(results):
        t = _times(r, env) / 60.0
        cols = slice(0, 3) if key == "range" else slice(3, 6)
        v = np.linalg.norm(r["states"][:, cols], axis=1)
        ax.plot(t, v, color=SERIES_COLORS[i], label=names[i], gid=f"{key}{i}")
        lines.append(np.column_stack([t, v]))
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=0.0)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("time, min")
    return lines


def _limit_line(labels: _Labels, value: float, text: str) -> None:
    ax = labels.ax
    ax.axhline(value, color=INK_SECONDARY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    t_lo, t_hi = ax.get_xlim()
    labels.avoid(np.array([[t_lo, value], [t_hi, value]]))
    candidates = [((t_hi, value), (-1, 1)), ((t_lo, value), (1, 1))]
    candidates += [((t, value), (0, 1)) for t in np.linspace(t_lo, t_hi, 13)[2:-2]]
    labels.place(text, candidates, (1.5,), **_NOTE)


def trajectory_figure(
    results: list[dict],
    env,
    *,
    labels: Sequence[str] | None = None,
    tick_s: float = 60.0,
    width_in: float = 7.2,
):
    """Hill-frame view of approach runs over range and speed panels sharing the time axis.

    The top panel draws the runs as `plot_trajectories` does. Below it, range to target (m) with
    the keep-out radius, and relative speed in the Hill frame (m/s) with the capture speed, both
    against time (min). Every value is read from
    the run records and the scenario configuration. The screen style is applied whatever the
    caller's settings; save at 170 dpi or more for the README.

    Parameters
    ----------
    results : list of dict
        One to three run records from `run_episode`, drawn in `SERIES_COLORS` order.
    env : ApproachEnv
        The environment the runs were flown in; supplies ``cfg.dt`` (s),
        ``cfg.keep_out_radius`` (m), ``cfg.capture_speed`` (m/s) and ``hold_point`` (m).
    labels : sequence of str, optional
        One name per run; defaults to each record's ``agent``.
    tick_s : float, optional
        Interval between time dots along each path, s.
    width_in : float, optional
        Figure width, in; the height follows from the Hill panel's aspect.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, not attached to pyplot.

    Raises
    ------
    ModuleNotFoundError
        If matplotlib is not installed.
    ValueError
        If there are no runs, more than three, or `labels` does not match them.
    """
    _matplotlib()
    if not results:
        raise ValueError("trajectory_figure needs at least one run")
    _check_count(results)
    names = _labels(results, labels)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    (y_lo, y_hi), (x_lo, x_hi) = _hill_limits(results, env)
    # Height of the equal-aspect panel at the width left after its tick labels.
    hill_h = float(np.clip((width_in - 0.75) * (x_hi - x_lo) / (y_hi - y_lo), 1.4, 4.5)) + 0.8
    with screen_style():
        layout = ConstrainedLayoutEngine(h_pad=0.05, w_pad=0.05, hspace=0.08, wspace=0.06)
        fig = Figure(figsize=(width_in, hill_h + 2.3), layout=layout)
        FigureCanvasAgg(fig)
        grid = fig.add_gridspec(2, 2, height_ratios=[hill_h, 2.3])
        ax = fig.add_subplot(grid[0, :])
        ax_range = fig.add_subplot(grid[1, 0])
        ax_speed = fig.add_subplot(grid[1, 1], sharex=ax_range)
        ax.set_title("Position in the target's Hill frame, orbit plane")
        ax_range.set_title("Range to target, m")
        ax_speed.set_title("Relative speed, m/s")
        range_points = _draw_series(ax_range, results, env, names, "range")
        speed_points = _draw_series(ax_speed, results, env, names, "speed")
        _draw_hill(ax, results, env, names, tick_s, legend=False, closest_note=True)
        range_labels = _Labels(ax_range, range_points)
        _limit_line(
            range_labels, env.cfg.keep_out_radius, f"keep-out {env.cfg.keep_out_radius:g} m"
        )
        speed_labels = _Labels(ax_speed, speed_points)
        _limit_line(speed_labels, env.cfg.capture_speed, f"capture {env.cfg.capture_speed:g} m/s")
        if len(results) >= 2:
            ax_speed.legend(loc="upper right")
    return fig


class _ApproachScene:
    """One run in the Hill frame, redrawn at each recorded state for an animation."""

    def __init__(
        self,
        result: dict,
        env,
        title: str,
        trail_s: float,
        width_in: float,
        dpi,
        color: str = SERIES_COLORS[0],
    ):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import LineCollection
        from matplotlib.figure import Figure

        self.states = np.asarray(result["states"], dtype=float)
        self.t = _times(result, env)
        self.r_ko = env.cfg.keep_out_radius
        self.trail = max(2, round(trail_s / env.cfg.dt))
        (y_lo, y_hi), (x_lo, x_hi) = _hill_limits([result], env)
        height = float(np.clip((width_in - 0.75) * (x_hi - x_lo) / (y_hi - y_lo), 1.6, 4.4))
        with screen_style():
            self.fig = Figure(figsize=(width_in, height + 0.85), dpi=dpi, layout="constrained")
            self.canvas = FigureCanvasAgg(self.fig)
            ax = self.fig.add_subplot()
            ax.set_title(title)
            labels = _draw_hill(
                ax, [result], env, ["start"], 60.0, legend=False, closest_note=True, colors=[color]
            )
            # The path is drawn frame by frame, and the closest approach shown once reached.
            for artist in [*ax.lines, *ax.texts]:
                if artist.get_gid() in ("path0", "ticks0", "tick_key", "name0"):
                    artist.remove()
            self.closest = next((a for a in ax.texts if a.get_gid() == "closest0"), None)
            self.k_closest = int(np.argmin(np.linalg.norm(self.states[:, :3], axis=1)))
            self.past = ax.plot([], [], color=color, lw=1.0, alpha=0.3, zorder=4)[0]
            # Butt caps: round caps overlap at the joints and bead the faded trail.
            self.fade = LineCollection([], linewidths=2.2, capstyle="butt", color=color, zorder=5)
            ax.add_collection(self.fade)
            self.marker = ax.plot(
                [], [], "o", ms=7.5, color=color, mec=SURFACE, mew=1.2, zorder=9, gid="marker"
            )[0]
            # One text per cell on shared baselines, so the two columns line up.
            text: dict[str, Any] = {
                "transform": ax.transAxes,
                "ha": "left",
                "va": "baseline",
                "fontsize": 8,
            }
            names = [
                ax.text(0, 0, n, color=INK_SECONDARY, **text) for n in ("time", "range", "speed")
            ]
            self.readout = [ax.text(0, 0, "", color=INK, **text) for _ in range(4)]
            self.readout[3].set_text("inside keep-out")
            w1 = max(labels._size(t)[0] for t in names)
            w2 = labels._size(self.readout[3])[0]
            row, gap = 11.5 * labels.px, 8 * labels.px
            x0, _, _, y1 = labels.corner(w1 + gap + w2, 4 * row, ("top left", "bottom left"))
            for column, x in ((names, x0), (self.readout, x0 + w1 + gap)):
                for r, t in enumerate(column):
                    t.set_position(labels.to_axes(x, y1 - (r + 0.75) * row))
            self.canvas.draw()
            # The layout is fixed from here, so frames line up and render faster.
            self.fig.set_layout_engine("none")

    def frame(self, k: int) -> np.ndarray:
        """Draw the scene at recorded state `k` and return the image as RGB."""
        s = self.states
        lo = max(0, k - self.trail)
        self.past.set_data(s[: lo + 1, 1], s[: lo + 1, 0])
        pts = s[lo : k + 1][:, [1, 0]]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        # The trail fades from full colour at the marker into the faint path behind it.
        alpha = np.linspace(0.3, 1.0, self.trail)[self.trail - len(segs) :]
        self.fade.set_segments(list(segs))
        self.fade.set_alpha(alpha.tolist() if len(segs) else None)
        self.marker.set_data([s[k, 1]], [s[k, 0]])
        if self.closest is not None:
            self.closest.set_visible(k >= self.k_closest)
        rng = float(np.linalg.norm(s[k, :3]))
        minutes, seconds = divmod(round(self.t[k]), 60)
        lines = [
            f"{minutes:d} min {seconds:02d} s",
            f"{rng:.0f} m",
            f"{np.linalg.norm(s[k, 3:6]):.2f} m/s",
        ]
        lines.append("inside keep-out" if rng < self.r_ko else "")
        for t, value in zip(self.readout, lines, strict=True):
            t.set_text(value)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()


def approach_animation(
    result: dict,
    env,
    path: str | Path,
    *,
    title: str | None = None,
    frame_s: float = 0.08,
    trail_s: float = 120.0,
    width_in: float = 8.0,
    dpi: float = 150.0,
    color: str = SERIES_COLORS[0],
) -> Path:
    """Write a GIF of one run in the Hill frame, one frame per recorded state.

    A marker moves along the path in the orbit plane with a trail that fades over `trail_s` of
    flight, over the keep-out disk, the target and the hold point. A readout gives the time,
    the range and the relative speed of the state shown, all from the record. The first frame
    is held for 0.6 s and the last for 2 s. Every frame shares one palette, so colours stay
    steady and the file stays small enough to show inline on a repository page.

    Parameters
    ----------
    result : dict
        One run record from `run_episode`.
    env : ApproachEnv
        The environment the run was flown in; supplies ``cfg.dt`` (s), ``cfg.keep_out_radius``
        (m) and ``hold_point`` (m).
    path : str or pathlib.Path
        Output file, written as GIF.
    title : str, optional
        Title above the plot; defaults to the record's ``agent``.
    frame_s : float, optional
        Display time of each frame, s of wall clock.
    trail_s : float, optional
        Simulated time over which the trail fades, s.
    width_in : float, optional
        Image width, in.
    dpi : float, optional
        Pixels per inch of the image.
    color : str, optional
        Colour of the run; pass the run's `SERIES_COLORS` slot from its other figures, so it
        looks the same everywhere.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    ModuleNotFoundError
        If matplotlib is not installed (Pillow comes with it).
    """
    _matplotlib()
    from PIL import Image

    scene = _ApproachScene(
        result, env, title or str(result["agent"]), trail_s, width_in, dpi, color
    )
    images = [scene.frame(k) for k in range(len(scene.states))]
    # One palette, from frames with the trail at its longest, keeps colours steady.
    sheet = np.concatenate([images[len(images) // 2], images[scene.k_closest], images[-1]])
    reduced = Image.fromarray(sheet).quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    palette = np.array(reduced.getpalette() or [], dtype=int).reshape(-1, 3)[:128]
    # Median cut averages each box, which shifts the surface; put it back exactly.
    surface = np.round(np.array(_matplotlib().colors.to_rgb(SURFACE)) * 255).astype(int)
    palette[np.argmin(np.abs(palette - surface).sum(axis=1))] = surface
    frames = []
    for rgb in images:
        # Exact nearest entries: Pillow's own mapping rounds colours and misses the surface.
        colours, index = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
        distance = ((colours[:, None, :].astype(int) - palette[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(distance, axis=1).astype(np.uint8)
        frame = Image.frombytes("P", rgb.shape[1::-1], nearest[index.ravel()].tobytes())
        frame.putpalette(palette.ravel().tolist())
        frames.append(frame)
    durations = [round(1000 * frame_s)] * len(frames)
    durations[0] = 600
    durations[-1] = 2000
    path = Path(path)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return path


__all__ = [
    "SCREEN_STYLE",
    "SERIES_COLORS",
    "approach_animation",
    "plot_trajectories",
    "screen_style",
    "trajectory_figure",
]
