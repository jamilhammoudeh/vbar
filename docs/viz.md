# Figures

Code: `vbar.viz` (`screen_style`, `SCREEN_STYLE`, `SERIES_COLORS`, `VIOLATION`, `PATH_WIDTH`,
`plot_trajectories`, `trajectory_figure`, `approach_animation`). Plotting needs the `plot` extra:
`pip install "vbar[plot]"`; without it each function raises `ModuleNotFoundError` naming that
command.

## Screen style

`screen_style()` is a context manager that applies `SCREEN_STYLE`, a fixed set of matplotlib
settings, to every figure made inside it, and restores the previous settings on exit. It is the
style for the README, these pages and reports:

| Element | Setting |
|---|---|
| surface (figure and axes) | #fcfcfb |
| primary text, titles | #0b0b0b; titles bold, left-aligned |
| axis labels, tick labels, notes | #52514e |
| axes lines and ticks | #8a8984, 0.8 pt; ticks outward, 3 pt |
| grid | #e6e5e1, 0.6 pt, under the data |
| spines | left and bottom only |
| text | DejaVu Sans, 9 pt (8 pt ticks, 7.5 pt notes) |
| PNG resolution | 170 dpi |

Runs take the colours of `SERIES_COLORS` in order: #2a78d6, #eb6834, #1baf7a. The three are
distinguishable with the common colour-vision deficiencies on the light surface; they are not
cycled past three, so a figure takes at most three runs, and a controller keeps its colour from
figure to figure when the runs are passed in the same order. Colour never identifies a run on its
own: each run's path is labelled directly, `plot_trajectories` adds a legend for two or more
runs, and the panels of `trajectory_figure` rely on the labelled paths above them. A keep-out
violation is marked with a status colour, #a3201a, used only for its outline and note and never
for a run.

DejaVu Sans is bundled with matplotlib, so a figure renders the same on every platform, which the
style check in the tests relies on.

## Hill-frame view

`plot_trajectories(results, env)` draws runs from `run_episode` in the orbit plane of the
target's Hill frame ([frames.md](frames.md)): along-track $y$ (V-bar) horizontal, radial $x$
(R-bar) up, at equal aspect so the keep-out sphere's cross-section stays a circle. It draws:

- the target as a small square at the origin;
- the keep-out zone as a light disk of radius `cfg.keep_out_radius`, labelled with its radius;
- the hold point as a dashed ring of fixed screen size (it marks a point, not a tolerance);
- each run as the heaviest line in the panel, with dots every `tick_s` seconds (60 by default),
  an open circle at its start labelled with its name, and an arrowhead partway along showing
  which way the craft moved;
- for a run that entered the keep-out zone, the part of its path inside the disk drawn heavier
  with a #a3201a outline, and a note "keep-out violated" over its closest approach;
- an arrow for the orbital velocity of the target (+$y$), which is not the craft's heading, and
  the dot interval.

Cross-track motion ($z$) is not drawn: a run that starts off the orbit plane looks closer to the
target than it is. The range panels below and the closest-approach note use the full 3D position;
the outline inside the disk is drawn only for a run whose record says it entered the zone.

## Range and speed

`trajectory_figure(results, env)` puts the Hill-frame view over two panels that share the time
axis (minutes): range to target $\lVert \mathbf r \rVert$ (m) with the keep-out radius as a dashed
line, and relative speed $\lVert \mathbf v \rVert$ in the Hill frame (m/s) with the capture speed
`cfg.capture_speed` as a dashed line. Each quantity has its own axis. The part of a range curve
below the keep-out radius gets the same heavier stroke and outline as in the Hill panel. The
figure is sized from the Hill panel's aspect at a fixed width (7.2 in by default); at 170 dpi it
is about 1220 px wide.

## Animation

`approach_animation(result, env, path)` writes a GIF of one run in the Hill-frame view: one frame
per recorded state, a marker at the state shown, the path flown so far as a solid line with dots
every 60 s, its newest `trail_s` seconds (60 by default) widening toward the marker, and a readout
of the time, range and speed of that state, with "inside keep-out" while the range is below the
keep-out radius. Once the run enters the zone the part inside the disk gets its outline and the
"keep-out violated" note shows; a run that entered only between recorded states gets the note
from the state after its recorded closest. The closest-approach number appears once the run has
passed it. The axes are fixed from the first frame and hold the whole path. Frames show for
`frame_s` seconds (0.08 by default), the first for 0.6 s and the last for 2 s. All frames share
one 128-colour palette, so colours do not flicker, and each frame is stored whole, so a viewer
that does not composite GIF frames still shows the full plot (about 2.4 MB for an 8 in wide,
75-frame run at 150 dpi).

## Values and placement

Every plotted value comes from the run record or the scenario configuration, never re-estimated:
positions and velocities from `states`, times as step index times `cfg.dt` (the record holds one
state per decision from $t = 0$), and the closest approach from `min_range_to_target_m`. That
minimum includes the refinement inside steps ([environment.md](environment.md)), so it can be
below the range of every recorded state; its note points at the recorded state nearest the
target. Delta-v over time is not plotted, because the record holds only the total.

Labels and notes are placed after one layout pass. Each one tries a short list of sides of its
anchor and takes the one whose box covers the fewest points of the plotted lines (sampled every
1.5 px), does not overlap a label already placed, and stays inside the axes. The orbital-velocity
arrow and the animation readout take the emptiest corner the same way.

## Usage

```python
from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent
from vbar.viz import SERIES_COLORS, approach_animation, trajectory_figure

env = ApproachEnv()
clear_env = ApproachEnv(ApproachConfig(start_state=(0.0, -400.0, 0.0, 0.0, 0.0, 0.0)))
clear = run_episode(clear_env, PDAgent(), seed=1, agent_seed=2)
blocked = run_episode(env, PDAgent(), seed=5, agent_seed=6)

fig = trajectory_figure([clear, blocked], env, labels=["PD, clear path", "PD, target in the way"])
fig.savefig("pd_approach.png", dpi=170)
approach_animation(
    blocked,
    env,
    "pd_approach.gif",
    title="PD, target in the way",
    width_in=5.0,
    color=SERIES_COLORS[1],
)
```

Both runs share the keep-out radius, hold point, step and capture speed of `env`; only the start
differs. The animation takes `color=SERIES_COLORS[1]` so the run keeps the colour it has in the
figure; the default is the first slot. `trajectory_figure` returns a `Figure` not attached to
pyplot. For a figure of your own in the same style, make it inside `screen_style()`:

```python
from matplotlib.figure import Figure

from vbar.viz import SERIES_COLORS, screen_style

with screen_style():
    fig = Figure(figsize=(4.0, 2.4), layout="constrained")
    ax = fig.add_subplot()
    ax.plot([0, 1, 2], [3.4, 2.4, 2.9], color=SERIES_COLORS[0])
    ax.set_title("Delta-v by seed, m/s")
fig.savefig("dv.png")
```

The README figure and animation come from `python examples/readme_figures.py`, which flies the
two runs above and writes `docs/images/pd_approach.png` and `docs/images/pd_approach.gif`.

## Validation

`tests/test_viz.py` renders each figure headless (Agg) from seeded runs and checks that:

- the plotted lines hold exactly the recorded positions, ranges and speeds, at step times, and
  the closest-approach note quotes the record;
- no two labels in a panel overlap and none leaves its panel;
- inside `screen_style()` the settings match the table above, and leaving it restores them;
- the rendered README figure keeps its pixel size, its surface colour, and the shares of the
  image drawn in each run colour, the grid colour, the keep-out fill and dark ink, within 15% of
  the reference in `tests/golden/viz_figure.json` (exact for the size and surface); this catches
  a changed colour, line width or font size without depending on exact pixels;
- the animation's marker, trail and readout show the recorded state of each frame, and the GIF
  has one frame per state;
- the README shows both committed assets, the animation under 3 MB.

`tests/test_viz_story.py` checks what the figures must say: only a run whose record has a
keep-out violation is outlined, clipped to the zone in the Hill and range panels and labelled;
each path is the heaviest line in its panel and carries an arrowhead; the speed panel has no
legend; in the animation the flown path stays solid at full width, the axes hold the whole path
from the first frame, the violation note shows from the frame the run enters the zone (or the
state after its recorded closest when no recorded state is inside) and stays inside the Hill
panel on every frame, the Hill panel labels the orbital-velocity arrow, and every GIF frame is
stored whole.

After an intended change to the style, `python tests/test_viz.py` rewrites the reference.
