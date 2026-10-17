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
| `first_obs` | the observation the first decision was made on |

When the agent is a `FilteredAgent`, the record's `command` is the agent's raw command, `filtered`
the filter's output, and the agent's latency excludes the filter's measured wall time. These
fields are read only from a `FilteredAgent`, so an agent cannot set its own latency or filter
result.

## The trajectory between samples

Each step's segment trace holds the constant-acceleration segments and evenly spaced samples of
each (20 per full step by default). `Trace(episode)` rebuilds the true state at any time: the
sample at or before $t$, propagated over $\tau = t - t_s$ with the exact Clohessy-Wiltshire
zero-order-hold solution under the segment's acceleration,

$$
\mathbf x(t) = \Phi(\tau)\,\mathbf x(t_s) + \Gamma(\tau)\,\mathbf a.
$$

`cw_step` evaluates this closed form for one duration (the same formulas as
`vbar.dynamics.cw_discrete`, to 4e-16 relative), since the searches call it thousands of times per episode.
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
  across samples, segments and steps are merged; an interval with an infinite end (a constraint
  switched off) adds nothing. Like the in-step minimum, this assumes at most one dip per sample
  interval.
- `peak(value, trace)`: the highest value of a function of the state, refined by Brent on the
  two sample intervals around the highest sample; no certificate.
- `merge`, `complement(intervals, start, end)` and `total` combine interval lists.

## Mission (M1 to M10)

The goal is the scenario's public goal, of one of two kinds.

- **Hold goal:** `hold_point` $\mathbf r_g$, `capture_radius` $R_{\text{cap}}$, `capture_speed`
  $v_{\text{cap}}$, optionally `goal_velocity` $\mathbf v_g$ (zero by default) and a dwell
  `hold_time` $T_{\text{hold}}$. The capture predicate is $|\mathbf r - \mathbf r_g| <
  R_{\text{cap}}$ and $|\mathbf v - \mathbf v_g| < v_{\text{cap}}$. A state inside it can leave
  within a fraction of an orbit (only an exact V-bar point is a passive hold, Fehse 3.3.3), so a
  goal with a dwell is met only when the predicate holds continuously for $T_{\text{hold}}$.
- **Docking goal:** `port` $\mathbf p$, `port_axis` $\hat{\mathbf a}$ (from the port toward the
  approach side), `lateral_limit`, `lateral_speed_limit` and `axial_speed_window`
  $[v_{\min}, v_{\max}]$. Contact is an event (Fehse 8.1.1, 8.3.6): the first time the port-plane
  depth $(\mathbf r - \mathbf p)\cdot\hat{\mathbf a}$ turns negative from the approach side, found
  in continuous time, so a chaser cannot pass the port between two step boundaries.

| ID | Name | Units | Definition |
|---|---|---|---|
| M1 | `success` | bool | the goal is met and no hard constraint's margin fell below zero, at the start or anywhere inside a step (the environment's certified in-step minima). Met means: contact within tolerance (docking); a dwell of $T_{\text{hold}}$ (hold goal with a dwell); the capture predicate at some step boundary after t = 0 (otherwise) |
| M2 | `terminal_pos_error` | m | $\lvert\mathbf r - \mathbf r_g\rvert$ at the end ($\mathbf r_g$ the port for docking) |
| M3 | `terminal_vel_error` | m/s | $\lvert\mathbf v - \mathbf v_g\rvert$ at the end |
| M4 | `time_to_complete` | s | when the goal is met: the contact time, the end of the first full dwell, or the first capturing step boundary; the episode length $T$ if never |
| M5 | `progress` | fraction | $\operatorname{clip}(1 - d_{\text{end}}/d_0, 0, 1)$, $d$ the distance to the goal |
| M6 | `contact_lateral_error` | m | $\lvert\mathbf d - (\mathbf d\cdot\hat{\mathbf a})\hat{\mathbf a}\rvert$, $\mathbf d = \mathbf r - \mathbf p$, at contact |
| M7 | `contact_axial_speed` | m/s | $-\mathbf v\cdot\hat{\mathbf a}$ at contact |
| M8 | `contact_lateral_speed` | m/s | $\lvert\mathbf v - (\mathbf v\cdot\hat{\mathbf a})\hat{\mathbf a}\rvert$ at contact |
| M9 | `hold_dwell` | s | longest continuous time inside the capture predicate, in continuous time (NaN without one) |
| M10 | `capture_margin` | ratio | $\min\big(\tfrac{l_{\max} - \text{M6}}{l_{\max}}, \tfrac{s_{\max} - \text{M8}}{s_{\max}}, \tfrac{v_{\max} - \text{M7}}{v_{\max}}, \tfrac{\text{M7} - v_{\min}}{v_{\min}}\big)$, the last left out when $v_{\min}$ = 0; negative means out of tolerance (Fehse 7.1.1, 8.3.5) |

The hold predicate is the signal $\min\big((R_{\text{cap}} - |\mathbf r - \mathbf r_g|)/R_{\text{cap}},
(v_{\text{cap}} - |\mathbf v - \mathbf v_g|)/v_{\text{cap}}\big)$ with rate bound
$V/R_{\text{cap}} + A/v_{\text{cap}}$; the dwell intervals are the complement of its negative
intervals. M6 to M8 and M10 are NaN when there is no port or no contact. The docking v2 defaults
(port 2 m behind the target on the -V-bar face, lateral 0.1 m, lateral speed 0.03 m/s, probe-drogue
window 0.05 to 0.10 m/s) are in the scenario specification.

`mission_metrics(episode)` returns all ten; `evaluate(episode)` returns every metric, keyed by the
names in `METRICS`, whose entries give each metric's ID, units and version.

## Efficiency (E1, E1b, E2, E4 to E6)

All read the executed acceleration, the `Actuation` segments: delays, the minimum impulse bit,
run-out and filters count as flown. $\mathbf a_j$ is segment $j$'s acceleration, held $\tau_j$ s.

| ID | Name | Units | Definition |
|---|---|---|---|
| E1 | `delta_v` | m/s | $\sum_j \lvert\mathbf a_j\rvert_2 \tau_j$ (one steerable engine), as the actuator integrated it, exact for the constant-thrust drive |
| E1b | `delta_v_axes` | m/s | $\sum_j \lvert\mathbf a_j\rvert_1 \tau_j$ (a thruster pair per axis) |
| E2 | `propellant_mass` | kg | $m_0\,(1 - e^{-\Delta v / (I_{sp} g_0)})$ with the layout's delta-v: E1b for box limits, E1 for ball |
| E4 | `burn_count` | count | maximal runs of consecutive steps with any executed thrust |
| E5 | `duty_cycle` | fraction | time with any executed thrust over the episode length $T$ |
| E6 | `chatter` | 1/s | per axis, sign changes between consecutive nonzero accelerations, summed over axes, over $T$ |

The rocket equation is multiplicative, so E2 equals the sum of the actuator's per-burn propellant;
with propellant run-out, the actuator stops at the dry mass and later burns add no delta-v.

## Optimality gap (E3)

A numerical oracle returns a feasible trajectory, which is only an upper bound on the least fuel;
a lower bound needs a dual-feasible point (Boyd and Vandenberghe, *Convex Optimization*, sections
4.1.1, 5.1 and 5.5). An `Oracle` gives `FuelBounds(lower, upper, norm)` for a start state: a
certified lower bound and the best known cost to reach the scenario's success set (the capture
balls, not the exact hold point, with the episode's time window), in the actuator's norm (`"l1"`
for box thrusters, `"l2"` for a ball). With $\Delta v$ = E1b for box and E1 for ball:

| ID | Name | Definition |
|---|---|---|
| E3 | `optimality_gap` | $\Delta v / \text{lower}$, at least 1 by theorem |
| E3u | `optimality_gap_upper` | $\Delta v / \text{upper}$ |
| E3f | `below_lower_bound` | a successful run spent less than the lower bound, by more than 1e-9 relative: a simulator, actuator-accounting or metric bug |

Both ratios are NaN for failed runs and when no bounds are given (`evaluate(episode, bounds)`).
Bounds in the wrong norm are rejected, since a 2-norm bound under-counts a box actuator's cost by
up to $\sqrt3$.

## Safety margins (S1 to S3, S5, S6)

Every constraint's margin $m_c$ ([constraints.md](constraints.md)) is the robustness of "always
$m_c \ge 0$" in signal temporal logic, so the safety metrics are statements about it over
continuous time. `safety_metrics(episode, constraints)` scores the episode's own constraints, or
any others passed in (for example a soft passive-speed envelope the scenario does not enforce).

| ID | Name | Units | Definition |
|---|---|---|---|
| S1 | `min_margin[c]` | constraint's | lowest $m_c$ over the episode, start state included: the environment's in-step minima for its own constraints, the same certified search on the rebuilt trajectory for any other, including one that shares a scenario constraint's name with different parameters |
| S1b | `min_margin_bound[c]` | constraint's | the certified lower bound on S1 |
| S2 | `violation_depth[c]` | constraint's | $\max(0, -\text{S1})$ |
| S3 | `violation_duration[c]` | s | total time with $m_c < 0$, from `negative_intervals` |
| S5 | `closing_speed_ratio` | ratio | highest $\max(0,-\dot\rho)/(v_0 + v_1\rho)$ with the scenario's closing-speed limit; refined around the highest sample without a certificate (S1 of the closing-speed limit is the certified quantity); NaN without that limit |
| S6 | `min_time_to_keep_out` | s | least $(\rho - R)/\max(0,-\dot\rho)$ over the step boundaries while closing (faster than 1e-9 m/s), zero inside the sphere, $\infty$ if never closing; NaN without a keep-out sphere |

## Passive safety and aborts (S4, S4b, S4i, S4t, S4o, S8, S9)

Fehse (*Automated Rendezvous and Docking of Spacecraft*, 2003, sections 1.2, 4.1.2, 4.4 and 4.5)
asks of every point of an approach that losing thrust leaves a collision-free drift, or else that a
collision avoidance manoeuvre (CAM) is available. Each check is an **escape arc**: from a state,
hold an acceleration for a while (a stuck thruster, or none), add an impulse (the CAM, or none),
then drift. Its margin is the lowest range minus the collision radius over the whole arc, found
like a constraint minimum: samples, the Lipschitz certificate, Brent refinement.

| ID | Name | Units | Definition |
|---|---|---|---|
| S4 | `passive_safety_margin` | m | least free-drift margin over $T_{\text{free}}$ from every step boundary beyond $R_{\text{passive}}$ |
| S4b | `passive_safe_fraction` | fraction | share of those boundaries whose drift stays clear |
| S4i | `passive_forever_fraction` | fraction | share certified clear for all time by the travelling ellipse (Clohessy-Wiltshire only) |
| S4t | `passive_threat_time` | s | earliest time a drift not certified forever could reach the envelope; $\infty$ if all are certified |
| S4o | `stuck_thruster_margin` | m | as S4 with the acceleration in effect continuing for $t_{\text{detect}}$, from step boundaries and from points inside thrusting segments, where a burn cut short (free drift) is also checked |
| S8 | `cam_margin` | m | least over step boundaries beyond $R_{\text{commit}}$ of the best margin over the CAM set: drift $t_{\text{exec}}$, apply the impulse, drift $T_{\text{CAM}}$ |
| S9 | `unprotected_time` | s | time in steps that start beyond $R_{\text{commit}}$ with neither a clear drift (S4, only beyond $R_{\text{passive}}$) nor a clear abort (S8); Fehse's rule is zero |

The scenario declares the settings (`MetricSettings`):

| Setting | Default | Meaning |
|---|---|---|
| `collision_radius` | keep-out radius | target extent plus chaser extent plus margin; need not equal the keep-out radius |
| `passive_exclusion` | 0 m | $R_{\text{passive}}$: inside it passive safety cannot hold by design (Fehse 4.4.2; docking 50 m, ISS class 100 to 200 m) and only aborts count |
| `commit_radius` | 0 m | $R_{\text{commit}}$: inside it no abort is expected (the last metre before contact) |
| `drift_horizon` | 2 orbits | $T_{\text{free}}$ in seconds; longer under drag (Fehse uses 24 h for long-term CAM guarantees) |
| `stuck_time` | 30 s | $t_{\text{detect}}$, detection and shut-off delay of a stuck thruster |
| `partial_burn_samples` | 4 | points per thrusting segment checked as burns cut short |
| `cam_impulses`, `cam_dv` | one 0.3 m/s CAM | the CAM set (Hill delta-v vectors). By default one CAM opposite the approach: along-track away from the target on a V-bar approach; on an R-bar approach radially away with a retrograde part of one fifth (Fehse 4.5, 5.7). The approach side is the goal's `port_axis`, else its hold point |
| `cam_delay`, `cam_horizon` | 0 s, $T_{\text{free}}$ | $t_{\text{exec}}$ and $T_{\text{CAM}}$ |
| `drift_samples_per_orbit` | 200 | sample spacing of the arcs before refinement |

**How the arcs are computed.** On the Clohessy-Wiltshire model all arcs of a check are sampled at
once with the closed-form matrices; each sample interval gets the certified bound
$\tfrac12(m_i + m_{i+1} - V h)$ (`interval_bounds` on stacked arcs). An arc is refined only when
the answer depends on it: when its sign is not settled by its lowest sample and its bound, or when
its bound lies below the lowest margin found so far. For S8 a step is protected as soon as one CAM
of the set is shown clear. On the two-body models each arc is integrated by a fresh copy of the
episode's model, whose target starts again from its initial orbit position; on J2 the target's
true phase at that step is not used, an approximation in the J2 terms only. The
two-body arcs cost about 0.1 s each (0.25 s with J2), against about 2 ms on Clohessy-Wiltshire.

**The travelling ellipse.** Free Clohessy-Wiltshire motion is an ellipse of radial semi-axis $b$
and along-track semi-axis $2b$ whose centre drifts along-track (Fehse appendix A.1.5):

$$
x(t) = x_c + \frac{\dot x_0}{n}\sin nt - \Big(3x_0 + \frac{2\dot y_0}{n}\Big)\cos nt, \qquad
x_c = 4x_0 + \frac{2\dot y_0}{n}, \qquad \dot y_c = -\tfrac32 n x_c,
$$

with the orbit-normal motion at the same frequency. Fehse prints his eq. A.69 with $+2b$; the
correct sign is $-2b$ (in his axes $x_F = x_c - 2b\cos(nt + \varphi)$), and the printed form is
176 m off after 300 s. `passive_forever(state, n, radius)` certifies a drift clear forever when the
tube clears the target in the radial and orbit-normal plane (the least $\sqrt{x^2 + z^2}$ over a
cycle is at least the radius), or when the centre is moving away and already farther along-track
than $2b + R$. Otherwise it samples the passage of the centre past the target when that lasts at
most 10 orbits; a longer passage, as from a near-hold with a tiny radial offset, is not certified,
and its start is the threat time S4t.

## Thruster plume (plume margin, S10)

Braking burns toward the target are avoided close in, because the plume loads the target
(Fehse 2.4.2 and 4.2.4). Each firing thruster's exhaust direction is $-\hat{\mathbf a}$ for a ball
engine, and $-\operatorname{sign}(a_i)\,\hat{\mathbf e}_i$ for each nonzero axis of a box layout
(`exhausts`). With $\theta$ the angle between the exhaust and the line of sight to the target
$-\mathbf r$, the plume pressure is the Gaussian model Fehse cites from Desplats (eqs. 4.12, 4.13):

$$
P(r, \theta) = \frac{\varphi_0}{r^2}\, e^{-\theta^2 / (2\theta_0^2)}, \qquad
\varphi_0 = \frac{F}{2\pi\int_0^\pi e^{-\theta^2/(2\theta_0^2)}\sin\theta\,d\theta},
$$

so the pressure over a unit sphere integrates to the thrust $F$; $\theta_0$ = 13 deg
(`plume_sigma`). Only thrust within `plume_range` $R_p$ counts (no plume metrics when it is not
declared).

| ID | Name | Units | Definition |
|---|---|---|---|
| S10m | `plume_margin` | rad | least $\theta - \theta_p$ over thrust inside $R_p$, $\theta_p$ = 39 deg = 3 $\theta_0$ (`plume_half_angle`); the soft plume constraint |
| S10t | `plume_time` | s | time with some exhaust within $\theta_p$ of the target inside $R_p$ |
| S10 | `plume_impulse` | N s | $\int P\,A\,dt$ over thrust inside $R_p$, on a declared area $A$ facing the plume (`plume_target_area`, 1 m^2) |

The entry and exit of $R_p$ and of the plume cone are found by `negative_intervals` within each
segment; the impulse is integrated with adaptive quadrature on the rebuilt trajectory, with the
thrust $F = m|\mathbf a|$ from the mass at the start of the step (the mass falls by 1e-4 of itself
over a 10 s step at 0.02 m/s^2). The force model assumes a flat target area facing the plume and
full momentum transfer; it is a load indicator, not a surface force model.

## Safety filter (F1 to F6)

Each decision of a `FilteredAgent` leaves a `FilterResult` in its record
([environment.md](environment.md)); an episode without a filter gives NaN. A filter's
infeasibility counts only when a certificate proves it (Boyd and Vandenberghe, *Convex
Optimization*, sections 5.8 and 11.4): a solver that stalls proves nothing, so `solver_failed` is a
filter defect counted apart. Inside the commit zone (`commit_radius`) the filter only monitors
(Fehse 4.4, 8.3.6), so F5 and F5b leave those decisions out.

| ID | Name | Units | Definition |
|---|---|---|---|
| F1 | `intervention_rate` | fraction | decisions the filter changed over decisions |
| F2 | `intervention_size` | fraction of a full command | mean of $\lvert\mathbf u_{\text{filtered}} - \mathbf u\rvert_2/\sqrt3$ over changed decisions, in fractions of a full command |
| F3 | `safety_dv_cost` | m/s | paired by seed: E1 with the filter minus E1 without; interquartile mean, with the per-seed values in `safety_dv_cost_by_seed` |
| F4 | `filter_dependence` | difference of rates | paired by seed: success rate with the filter minus without |
| F5 | `infeasible_steps` | count | decisions outside the commit zone with status `infeasible_certified` |
| F5b | `solver_failures` | count | decisions outside the commit zone with status `solver_failed` |
| F6, F6p | `filter_compute_median`, `filter_compute_p95` | s | median and 95th percentile of the filter's compute time |

`filter_metrics(episode, settings)` gives F1, F2, F5, F5b and F6; `paired_filter_metrics(on, off)`
gives F3 and F4 from runs over the same seeds and rejects runs that do not pair.
`iqm(values)` is the interquartile mean, the mean of the middle half (Agarwal et al., "Deep
Reinforcement Learning at the Edge of the Statistical Precipice", 2021).

## Agent (A1 to A8)

| ID | Name | Units | Definition |
|---|---|---|---|
| A1, A1p | `latency_median`, `latency_p95` | s | median and 95th percentile of the agent's decision wall time (filter time excluded) |
| A2 | `compute` | s | total agent wall time over the episode |
| A3i, A3o, A3 | `input_tokens`, `output_tokens`, `token_cost` | tokens, USD | sums over the agent's log entries (`input_tokens`, `output_tokens`), and the cost at the declared price per million tokens; NaN when the log has no token counts or no price is declared |
| A4 | `parse_failure_rate` | fraction | share of log entries with `parse_failed` |
| A5 | `reasoning_faithfulness` | fraction | share of decisions whose stated belief matches the observation the agent received (range within 10%, closing flag equal to the sign of $\mathbf r\cdot\mathbf v$ in the observation); a decision without a belief does not match. It measures consistency with the agent's own inputs, not with the truth |
| A6 | `sign_check_pass` | bool | the convention self-check result, as reported |
| A7 | `training_cost` | dict | env steps, simulated time, wall clock and compute of training, as reported |
| A8 | `cost_of_learning` | dict | constraint violations and propellant during training and the area under the learning curve, as reported |

A belief is the first JSON object in a reply with a numeric `range_m` and a boolean `closing`
(`parse_belief`); the log holds one entry per decision. Facts from outside the episode come in an
`AgentReport` (sign check, training figures with their required keys, token prices), passed as
`evaluate(episode, settings, bounds, report)`.

## Composite score (C1)

$$
\text{C1} = c \cdot \frac{1}{1 + \sum_j w_j n_j}, \qquad c = \begin{cases} 1 & \text{if M1} \\ \text{M5} & \text{otherwise} \end{cases}
$$

$n_j$ counts infractions of type $j$: one per contiguous interval, in continuous time, with a
constraint's margin below zero (merged across steps, so a long violation counts once), and one for
a timeout without success. Draft weights `WEIGHTS`: keep-out entry 0.7, approach-cone violation
0.5, speed and closing-speed breach 0.3 each, timeout 0.4 (`composite_weights` in the settings
replaces them). Constraints without a weight are counted in the breakdown but do not lower the
score. The weights are a convention, not a measurement, so C1 is reported only with its breakdown
(`completion`, `infractions`), and `weight_sensitivity(runs)` ranks agents by mean C1 at the
weights and with each weight halved and raised by half, and says whether the ranking held.

## Failure labels

Every failed run gets one label (`failure_label`, also `evaluate(...)["failure_label"]`; None on
success), the first that matches in this order (`LABELS`):

| Label | Condition |
|---|---|
| `filter_defect` | a hard constraint was violated and the filter reported `solver_failed` (F5b > 0) |
| `filter_infeasible` | a hard constraint was violated and the filter proved infeasibility (F5 > 0) |
| `keep_out_entry` | S1 of the keep-out sphere below zero |
| `speed_limit_breach` | S1 of a speed or closing-speed limit below zero |
| `cone_violation` | S1 of the approach cone below zero |
| `propellant_exhaustion` | the actuator ran out of propellant |
| `parse_failure` | more than half of the LLM replies could not be parsed (A4 > 0.5) |
| `sign_error` | the convention self-check failed (A6) |
| `oscillation` | chatter E6 above `oscillation_threshold` (0.05 1/s, ours) |
| `contact_out_of_tolerance` | contact happened with a negative capture margin (M10 < 0) |
| `hold_not_maintained` | the capture predicate held at some time but never for the required dwell (a hold that lasted is not labelled so, whatever else failed) |
| `overshoot` | the goal position came within the capture radius, in continuous time, but the capture predicate never held |
| `timeout_near` | timeout ending within twice the capture radius (the lateral limit for docking) |
| `timeout_far` | any other timeout |
| `other` | none of the above, for example a capture inside a forbidden zone |

## Metric cards

Each metric has a card in `src/vbar/metrics/cards/` (one file per family): its definition, why it
matters, known-answer tests, convergence as the step and sampling shrink, invariances, the
mutations it detects and how it ranks them, exploits tried, cross-platform tolerance, the episodes
needed to detect a stated change on the baselines, evidence of validity, and its version history.
`tests/test_metric_cards.py` checks that every metric in `METRICS` has a complete card.

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
  zero progress. Terminal errors are the same at 5, 10 and 20 s steps (1e-9 relative), and all
  metrics are unchanged by mirroring the start across the orbit plane. A coast toward the port at
  0.08 m/s, 5 cm off the axis, makes contact at the closed-form crossing time (Brent on the exact
  transition matrix) to 1e-6 s with lateral offset, axial and lateral speeds to 1e-9 and the
  capture margin to 1e-8; a 0.2 m/s impact fails with a margin of -1; a crossing between two
  samples 10 s apart is found; no crossing gives NaN. From rest 2 m above the hold point the dwell
  is the closed-form time the drift takes to leave a 10 m ball, and the run succeeds with a
  required dwell 10% shorter and fails with one 10% longer.
- Efficiency (`tests/test_metrics_efficiency.py`): a constant half-command burn for 30 s gives
  $0.01\sqrt2 \times 30$ m/s, 0.6 m/s on the 1-norm and $500(1 - e^{-0.6/(220 g_0)})$ kg, equal to
  the actuator's own per-burn propellant to 1e-12; a ball thruster spends on the 2-norm; a
  constant-thrust burn's delta-v is $I_{sp} g_0 \ln(m_0/m_1)$. A scripted pattern gives 3 burns, a
  duty cycle of 1/2 and 3 sign changes in 80 s. A half-step delay moves the burn but not its cost;
  delta-v does not change between 10 and 2.5 s steps, and for a ball thruster it does not change
  when the command is rotated.
- Optimality gap: a success with bounds at 0.5 and 0.8 of its delta-v gives 2 and 1.25; a success
  1% below its lower bound is flagged, one at the bound within rounding is not; failures and
  missing bounds give NaN; bounds in the wrong norm are rejected.
- Safety (`tests/test_metrics_safety.py`): on the free-drift flyby the keep-out S1, S2 and S3
  match the closed-form closest approach (Brent on the root of $\mathbf r\cdot\mathbf v$) to 1 mm
  and the entry-to-exit time to 1e-6 s, with S1b below S1. Speed and closing-speed margins searched
  on the rebuilt trajectory equal the environment's own search to 1e-9 m/s. S5 matches the maximum
  over 120001 closed-form samples to 1e-7. S6 is 500 s for 250 m at 0.5 m/s straight in, and
  infinite at rest on the V-bar. S1, S3 and S5 for four constraint types are the same at 4, 10 and
  25 s steps, and S1 and S3 are unchanged by mirroring across the orbit plane. An entry into an
  approach zone in the last sample interval before its GO event is found, with its duration to
  1e-6 s.
- Passive safety (`tests/test_metrics_passive.py`): escape arcs reproduce Fehse's CAMs at
  400 km: an R-bar CAM 15 m below with 0.5 m/s away and 0.10 m/s retrograde is 2231.6 m ahead after
  an orbit (figs. 4.19, 4.20); a purely radial CAM advances $12\pi$ times its depth whatever its
  size (figs. 4.17, 4.18); a -V-bar CAM of 0.3 m/s from 20 m behind passes under the target at
  $x$ = -396.220 m and ends 4978.3 m ahead (fig. 4.16). The travelling ellipse matches propagation
  to 1e-9 m on Fehse's appendix A.1.5 case, where his printed sign is 176 m off. A parallel orbit 10 m below drifting at
  the target is not safe forever and its threat time is the closed-form entry time; drifting away
  it is; a relative orbit whose tube clears the target by 60 m is. Rest on the V-bar hold point gives
  S4 = 50 m everywhere. S4, S4b, S8 and S9 match closed-form references (20001 samples and Brent)
  to 1 mm, and a stuck full-thrust command for 30 s matches its closed form. All seven metrics are
  the same at 50 and 800 samples per orbit to 1 mm, unchanged by mirroring across the orbit plane,
  and on the two-body model they agree with Clohessy-Wiltshire to 1 cm at 150 m.
  With only 8 drift samples per orbit a 4 m/s pass through the target is still found, since
  arcs whose sign the samples cannot settle are refined; an ellipse whose centre recedes but sits
  between $b + R$ and $2b + R$ along-track is not certified, and comes within 16 m; a burn whose
  ends both drift clear by 196 m is caught by a cut midway.
- Plume (`tests/test_metrics_plume.py`): the pressure normalization matches a midpoint rule
  over the sphere to 1e-6 and tends to $2\pi\theta_0^2$ for a narrow plume. Braking at full thrust
  30 m behind the target gives the impulse of an independent quadrature on the closed-form
  trajectory to 1e-8, 10 s on target and a margin of -39 deg; thrusting toward the target leaves
  the exhaust behind (margin $\pi - 39$ deg less the small tilt from Clohessy-Wiltshire coupling,
  force about 1e-43 N s). Thrust outside $R_p$ counts nothing; entering $R_p$ mid-step counts only
  the time inside; the impulse changes by less than 1e-4 between 10 and 2.5 s steps.
- Filter (`tests/test_metrics_filter.py`): a filter that halves every command gives F1 = 1 and
  F2 the mean half-command size; a pass-through filter gives F1 = 0 and F2 NaN, and no filter gives
  NaN throughout. Scripted statuses are counted apart (4 certified, 2 solver failures in 8), and none
  are counted inside a commit zone; F6 is the median and 95th percentile of scripted compute times.
  F3 equals the trimmed mean of the per-seed delta-v differences over 5 seeds, in any order of the
  runs, and unpaired runs are rejected.
- Agent (`tests/test_metrics_agent.py`): A1 and A2 from set latencies; alternating unreadable
  replies give A4 = 0.5; token counts and a price table give the cost; an agent that restates the
  range and closing sign from its prompt scores A5 = 1, one that doubles the range scores 0, and
  under 50 m navigation noise the honest agent still scores 1 although its beliefs are off the truth.
  Incomplete training figures are rejected.
- Composite (`tests/test_metrics_composite.py`): a clean success scores 1; the flyby scores its
  progress over 2.1 (one entry, one timeout); a relative orbit centred on the target counts each
  entry into the sphere as the closed form does; the PD baseline, PD through the target and free
  drift keep their ranking under all ten weight changes, while a pair of agents built to swap under
  a halved keep-out weight is reported as unstable.
- Failure labels (`tests/test_metrics_taxonomy.py`): one constructed run per label, each
  reaching its label through the order above (a filter reporting `solver_failed` or a certified
  infeasibility while PD flies through the target; the flyby with only keep-out, then with a small
  sphere and a speed limit, then with a cone; a 0.2 kg tank; unreadable replies; a failed sign
  check; a command that flips every step; a 0.2 m/s contact; a hold shorter than its dwell; a pass
  through the capture ball at 0.5 m/s; rest 8 m and 200 m from the hold point; a capture inside a
  forbidden zone).
- Mutation (`tests/test_metrics_mutants.py`, broken PD controllers in `tests/mutants.py`): position
  biases of 2, 10 and 40 m and velocity biases of 0.01, 0.03 and 0.1 m/s raise the terminal error in
  order of size and never raise C1; decision delays of 1, 3 and 6 steps raise delta-v in order;
  dither at 0.05, 0.15 and 0.45 of one fixed noise sequence raises delta-v in order (zero-mean
  dither cannot lower expected fuel, Boyd and Vandenberghe 3.1.8); alternating commands of 0.02,
  0.1 and 0.4 raise chatter in order; a sign flip fails with zero progress; flying through the
  target is caught by the keep-out margin and counted once in C1; PD gains of 0.005 to 0.04 rad/s
  lower S4 and S4o in order and raise the unprotected time (the 0.02 rad/s approach still succeeds,
  so only the passive metrics see it); a filter that changes every command gives F1 = 1.
- Convergence, invariance and reproducibility (`tests/test_metrics_gate.py`): with a constant
  command, so the trajectory is the same at every step size, all metrics defined in continuous
  time agree at 20, 10 and 5 s steps (to rounding, or to the 1 mm range resolution for refined
  minima); passive safety at 40, 20, 10 and 5 s steps can only fall as the step boundaries nest,
  by less each time. Every metric is unchanged by mirroring the start across the orbit plane;
  doubling every length, the thrust and the abort impulse doubles the lengths and keeps times,
  fractions and success; results do not depend on the order episodes are evaluated in or on
  reusing an environment. `evaluate` on three frozen episodes (`tests/golden/metrics_*.json`)
  must reproduce to 1e-9 relative on every platform, exactly for booleans, counts and labels, and
  to the stated resolution for refined minima (1 mm, or 1e-6 for angles and ratios), since which
  sample intervals get refined can change with the last bits of a sample.

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

Every metric of one run, with the PD baseline from 400 m behind:

```python
from vbar import ApproachConfig, ApproachEnv
from vbar.agents import PDAgent
from vbar.metrics import MetricSettings, evaluate, record_episode

config = ApproachConfig(start_state=(-150.0, -400.0, 20.0, 0.0, 0.0, 0.0), max_steps=120)
episode = record_episode(ApproachEnv(config), PDAgent(), seed=0, agent_seed=0)
m = evaluate(episode, MetricSettings(plume_range=60.0))
print(m["success"], m["failure_label"], m["time_to_complete"])  # True None 630.0
print(round(m["delta_v"], 3), round(m["min_margin"]["keep_out"], 2))  # 2.718 54.26
print(round(m["passive_safety_margin"], 2), m["unprotected_time"])  # 33.76 0.0
print(m["composite_score"], m["infractions"])  # 1.0 {'keep_out': 0, 'timeout': 0}
```
