# Replay

Code: `vbar.replay` (`export_rrd`). A recorded episode (`vbar.metrics.record_episode`) is written
as a [Rerun](https://rerun.io) recording, a `.rrd` file that the Rerun viewer opens in 3D with a
time slider. The export reads the step records and the scenario's public description and does not
simulate anything again. It needs the `replay` extra, `pip install "vbar[replay]"` (Rerun is
licensed MIT or Apache-2.0); without it `export_rrd` raises `ModuleNotFoundError`, an
`ImportError`, that names this command. Importing `vbar.replay` does not need Rerun.

Rerun's first-run notice says that it collects anonymous usage data about the features used in
its viewer; `rerun analytics details` explains what is sent and how to opt out. V-BAR collects
nothing.

The file holds the true trajectory. It is for people reviewing a run, never for the agent that
flew it, and a replay of an episode from the hidden test split should stay as private as that
split.

## Use

```python
from pathlib import Path

from vbar import ApproachConfig, ApproachEnv
from vbar.agents import PDAgent
from vbar.metrics import record_episode
from vbar.replay import export_rrd

env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0)))
episode = record_episode(env, PDAgent(), seed=1, agent_seed=2)
path = export_rrd(episode, Path("pd_approach.rrd"), card={})
print(path)
```

`examples/replay_demo.py` does the same with the full metric card and writes
`outputs/pd_approach.rrd`; open it with `rerun outputs/pd_approach.rrd`. `card` is the episode's
metrics as `vbar.metrics.evaluate` returns them; without it `export_rrd` evaluates the episode
itself, which is the slow part of an export (35 s for a 333-step episode on a loaded machine,
against about 2 s for 70 steps); pass the card when it is already computed. `settings` passes the scenario's declared
`MetricSettings` to that evaluation. `arrow_m` is the length in
metres at which a full command is drawn.

## Frame, units and time

Positions are in the target's Hill frame ([frames.md](frames.md)): x radial, y along-track, z
orbit normal, right-handed. The viewer is told that x is right, y forward and z up, so the orbit
plane is horizontal. Lengths are metres, speeds m/s, accelerations m/s^2, times seconds.

Two timelines are set. `sim_time` is the episode time as a duration, stored as whole nanoseconds,
so a time read back differs from the record by at most 1 ns. `step` counts decisions: step $k$
is the state after $k$ steps, and a command or event of decision $k$ is at the time of state $k$.

The viewer draws positions as 32-bit floats, whose spacing at 400 m is 0.03 mm. The same numbers are also stored as 64-bit components on the same
entities (`position_m`, `velocity_mps`, `accel_mps2`, `dv_mps`, `start_s`, `end_s`, `radius_m`),
and Scalars are 64-bit already, so the stored values read back equal to the record's.

## What the file holds

| Entity | Content |
|---|---|
| `hill` | the axes of the frame |
| `hill/target` | the target at the origin |
| `hill/chaser` | the chaser's true position at every decision, with `position_m` and `velocity_mps` |
| `hill/chaser/trail` | the path through the recorded states up to now |
| `hill/chaser/command` | the agent's command as an arrow at the chaser, with `accel_mps2` (command times the view's `max_accel`) |
| `hill/chaser/filtered` | the command after the safety filter, only in a run with a filter |
| `hill/chaser/flown` | the acceleration the spacecraft flew in the step, with `accel_mps2` (mean over the step) and `dv_mps` (its delta-v vector) |
| `hill/constraints/<name>` | each keep-out sphere (wireframe, with `radius_m`) and each approach corridor (rings and rulings, with `apex_m`, `axis`, `half_angle_rad`, `active_range_m`) |
| `hill/goal/hold_point`, `hill/goal/port` | the goal's hold point or port, when the scenario has one |
| `state/range_m`, `state/speed_mps` | range and speed of the true state |
| `events/violation/<name>` | one event per interval in which a constraint's margin is negative, at its start, with `start_s` and `end_s`; ERROR for a hard constraint, WARN for a soft one |
| `events/filter/intervention` | a decision at which the filter changed the flown acceleration |
| `events/filter/status` | a decision at which the filter's status was not "feasible" |
| `events/outcome` | the verdict, at the end of the episode |
| `card/<metric>` | metric values over the first $k$ steps, at the end of step $k$ |
| `card/final/<metric>` | the final value of each finite numeric metric of the card, dictionaries by key, at the end of the episode |
| `meta/episode` | scenario identifier, dynamics model, step length, action mode, package version |

Series carry their metric identifier and unit in their name, for example `E1 delta_v (m/s)`.

An arrow's drawn length is the command as a fraction of a full command times `arrow_m`, by
default a tenth of the chaser's greatest range, so the arrows are visible beside the path. The
stored `accel_mps2` does not depend on it.

## How the events are found

Violation intervals are the ones the environment recorded for each step
(`SegmentTrace.violations`, found in continuous time inside the step, [constraints.md](constraints.md)),
with intervals that touch across a step boundary joined, as the metrics join them. A violation
that lasts two steps is therefore one event.

A filter intervention is read from the record, not from the filter's report: the decision counts
when the filter says a safety row was active, or when the acceleration the actuator flies
differs between the agent's command and the filtered command, the rule of the intervention-rate
metric ([metrics.md](metrics.md)). A filter that reports no intervention while halving every
command still gets its events.

## The metric card over time

Each running series is a metric of the card evaluated on the first $k$ steps of the record and
plotted at the end of step $k$: delta-v (E1, E1b), propellant mass (E2), burn count (E4), duty
cycle (E5) and chatter (E6), and in a run with a filter the intervention rate (F1) and size (F2).
They are accumulated step by step, without evaluating every prefix of the record again, and the
test compares them with the metric functions on prefixes of the record.
The safety metrics are accumulated from the environment's per-step results: the lowest margin so
far of each constraint (S1), its depth (S2) and the time spent below zero (S3). The values at the
last step equal the full card's, except the violation duration, which the card rebuilds from the
trajectory and which agrees to a relative 1e-6. Metrics that exist only for a whole episode
(success, contact, passive safety, the composite) and the filter metrics without a running
definition here appear in `card/final`.

## Assumptions and limits

- Only `KeepOut` spheres and `ApproachCone` corridors are drawn. Other constraints, and cones
  that turn with a rotating target, appear through their violation events and their margins in the
  card but not as geometry. A corridor with no `active_range` is drawn out to the greater of the
  chaser's greatest distances from its apex and from the target.
- The goal's hold point and port are drawn where the briefing puts them at $t = 0$.
- The trail is written in full at every step, so a file grows with the square of the number of
  steps (measured: 1.2 MB at 333 steps).
- The trail joins the states at decisions with straight lines; the curve inside a step is not
  drawn, though the violation intervals inside it are found on it.
- The command arrow uses the initial mass (`max_accel`), as the metrics do, so under a
  constant-thrust drive the flown acceleration can exceed a full command late in a run.
- Rerun's Python API changes between 0.x releases. The extra pins the minor version it was tested
  with (0.41).

## Validation

`tests/test_replay.py` reads the files back with Rerun's own reader and asserts:

- positions, velocities, commanded and flown accelerations equal the record's values, and the
  drawn single-precision positions agree to one part in 1e6;
- times agree with the record to 1 ns;
- violation events start and end at the intervals the record holds, computed independently in the
  test, and a run that never enters the zone has none;
- the corridor's vertices lie on the declared cone to the single-precision accuracy;
- filter events fall on the decisions where the record shows the command changed, whatever the
  filter reports;
- the series end at the full card's values;
- exporting changes neither the episode nor its metrics;
- no seed is written to the file.

`tests/test_extra_replay.py` runs without Rerun and checks the install message.
