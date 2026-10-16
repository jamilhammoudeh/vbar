# Metrics

Code: `vbar.metrics`. Metrics are computed from a recorded episode after it ends: the scenario's
public description, the true start state and every step record. They read the truth; agents never
see a record. Every metric is a function of the episode alone (plus declared settings and, for the
optimality gap, oracle bounds), so it can be recomputed from a saved run.

## Recording an episode

`record_episode(env, agent, seed, agent_seed)` runs one episode the way `vbar.run_episode` does
(the agent gets the view, its own random stream, the observations and the time) and returns an
`Episode`:

| Field | Meaning |
|---|---|
| `view` | the scenario's `AgentView` (goal, constraints, action mode, `n`, `dt`, mass) |
| `start` | true state at t = 0 (m, m/s, Hill) |
| `records` | one `StepRecord` per step: true state, observation, the agent's own `command`, the `filtered` command applied, the spacecraft's `Actuation`, the step's segment trace, the `FilterResult` and the agent's wall time |
| `constraints`, `hard` | the scenario's constraint objects and the names of those whose violation fails the run (all but the scenario's `soft_constraints`) |
| `terminated`, `truncated` | how the episode ended |
| `isp`, `dynamics`, `body`, `altitude_m`, `inclination` | actuator and propagation settings |
| `agent_log` | the agent's own per-decision log, when it keeps one (`LLMAgent.log`) |

When the agent is a `FilteredAgent`, the record's `command` is the agent's raw command, `filtered`
the filter's output, and the agent's latency excludes the filter's compute time.

## The trajectory between samples

Each step's segment trace holds the constant-acceleration segments and evenly spaced samples of
each (20 per full step by default). `Trace(episode)` rebuilds the true state at any time: the
sample at or before $t$, propagated over $\tau = t - t_s$ with the exact Clohessy-Wiltshire
zero-order-hold solution under the segment's acceleration,

$$
\mathbf x(t) = \Phi(\tau)\,\mathbf x(t_s) + \Gamma(\tau)\,\mathbf a.
$$

This is exact on the Clohessy-Wiltshire model. On the two-body models it drops the difference
between their natural acceleration and Clohessy-Wiltshire's over at most one sample spacing:
measured below 1e-8 m at 150 m without J2 (below 2e-7 m at 2 km), and about 1e-6 m with J2, whose
Hill frame turns faster than $n$ by about $\tfrac32 J_2 (R/a)^2 n$. Both are far below the 1 mm
resolution of the range constraints, and the reconstruction needs only the records and $n$.

## Signals in continuous time

A signal is anything with a margin $m(\mathbf x, t)$, a rate bound and a resolution: every
constraint ([constraints.md](constraints.md)), or a `Signal(name, margin_fn, rate_fn, resolution)`
built for a metric (distance to the goal, the hold predicate, the contact plane).

- `minimum(signal, trace)`: the lowest value over the episode, per segment by `segment_minimum`
  (sampling, Brent refinement and the Lipschitz certificate), returned with its time and certified
  lower bound.
- `negative_intervals(signal, trace)`: the time intervals where $m < 0$. On each sample interval
  with end values $m_i, m_{i+1}$ and rate bound $L$: a sign change is bracketed by Brent's root
  finder (tolerance 1e-9 s); two non-negative ends are searched for a dip only when
  $\tfrac12(m_i + m_{i+1} - Lh) < 0$, and two negative ends for a rise above zero only when
  $\tfrac12(m_i + m_{i+1} + Lh) > 0$; a dip or rise found gives two roots. Intervals that touch
  across samples, segments and steps are merged. Like the in-step minimum, this assumes at most
  one dip per sample interval.
- `merge`, `complement(intervals, start, end)` and `total` combine interval lists.

## Mission (M1 to M5)

The goal is the scenario's public goal: `hold_point` $\mathbf r_g$, `capture_radius` $R_{\text{cap}}$,
`capture_speed` $v_{\text{cap}}$ and optionally `goal_velocity` $\mathbf v_g$ (zero by default).
The capture predicate is $|\mathbf r - \mathbf r_g| < R_{\text{cap}}$ and
$|\mathbf v - \mathbf v_g| < v_{\text{cap}}$, checked at step boundaries after t = 0.

| ID | Name | Units | Definition |
|---|---|---|---|
| M1 | `success` | bool | the capture predicate holds at some step boundary and no hard constraint's margin fell below zero, at the start or anywhere inside a step (the environment's certified in-step minima) |
| M2 | `terminal_pos_error` | m | $\lvert\mathbf r - \mathbf r_g\rvert$ at the end |
| M3 | `terminal_vel_error` | m/s | $\lvert\mathbf v - \mathbf v_g\rvert$ at the end |
| M4 | `time_to_complete` | s | first step boundary at which the capture predicate holds; the episode length $T$ if never |
| M5 | `progress` | fraction | $\operatorname{clip}(1 - d_{\text{end}}/d_0, 0, 1)$, $d$ the distance to the goal |

`mission_metrics(episode)` returns all five; `evaluate(episode)` returns every metric, keyed by the
names in `METRICS`, whose entries give each metric's ID, units and version.

## How it is tested

`tests/test_metrics_episode.py` (cases in `tests/metric_cases.py`):

- A recorded episode has exactly the true states `run_episode` returns for the same seeds; a
  filtered agent's raw command, filtered command and result are in every record; an LLM agent's log
  is kept.
- The trace returns the samples at sample times and matches the closed-form solution from the
  step start to 1e-9 of the state's size between them; it follows the two-body model's dense
  output to 1e-8 m without J2 and 5e-6 m with J2.
- The keep-out intervals of a free-drift flyby through the 50 m sphere match the closed-form entry
  and exit times (Brent on the exact transition matrix, 1e-13 s) to 1e-6 s, and do not change when
  the same trajectory is stepped at 3.7, 10 or 25 s.
- A 12 m/s pass that dips 5 m into the sphere between two samples 10 s apart is found, and a signal
  negative at both samples but positive in between gives two intervals.
- Mission (`tests/test_metrics_mission.py`): the PD agent from 400 m behind succeeds inside the
  capture ball; resting on the V-bar hold point, an exact equilibrium, succeeds at the first step
  with zero terminal errors; capturing after flying through the keep-out zone is not a success;
  free drift through the target fails with progress $1 - d_{\text{end}}/250$; moving away gives
  zero progress. Terminal errors are the same at 5, 10 and 20 s steps (1e-9 relative), and all five
  are unchanged by mirroring the start across the orbit plane.

## Example

```python
from vbar import ApproachConfig, ApproachEnv
from vbar.agents import DriftAgent
from vbar.constraints import KeepOut
from vbar.metrics import Trace, negative_intervals, record_episode, total

# Free drift that passes about 16 m from the target.
config = ApproachConfig(start_state=(0.0, 150.0, 0.0, 0.0, -1.5, 0.0), max_steps=20)
episode = record_episode(ApproachEnv(config), DriftAgent(), seed=0, agent_seed=0)
inside = negative_intervals(KeepOut(50.0), Trace(episode))
print([(round(a, 2), round(b, 2)) for a, b in inside])  # [(67.29, 129.48)]
print(round(total(inside), 2))  # 62.19 s inside the keep-out sphere
```
