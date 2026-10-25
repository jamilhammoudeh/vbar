# Changelog

Notable changes to V-BAR. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0, a minor
release may change the public API, and every such change is listed under Changed or Removed.

## [Unreleased]

### Added

- `vbar.viz.screen_style`, a context manager that applies the screen style (`SCREEN_STYLE`) for
  README, documentation and report figures, and `SERIES_COLORS`, the three run colours in order.
- `vbar.viz.trajectory_figure`: the Hill-frame view over range and speed panels sharing the time
  axis, with the keep-out radius and capture speed as reference lines.
- `vbar.viz.approach_animation`: a GIF of one run in the Hill frame with a moving marker, the
  path flown so far, a keep-out violation marked from the frame it starts, and a readout of time,
  range and speed.
- `examples/readme_figures.py` regenerates the README figure and animation from fixed seeds;
  `docs/viz.md` documents the figures.
- `vbar.scenarios.targets.body_point_velocity`: the Hill-frame velocity w x (R p) of a point fixed
  to a turning target.
- `Episode.scenario_id`: the identifier of what was flown.
- `BaseEnv.task_outcome()` and `Episode.outcome`: the environment's verdict on a goal the metrics
  cannot rebuild from the trajectory (the inspection's coverage).

### Changed

- The README and `CITATION.cff` give the Zenodo DOI, which resolves to the latest release.
- `vbar.viz.plot_trajectories` draws in the screen style: the keep-out disk with its radius, the
  hold point as a dashed ring, time dots every `tick_s` (60 s), open start circles, a direct label
  per run, an arrowhead showing each run's direction, the part of a run inside the keep-out zone
  outlined and labelled with its closest approach, and an orbital-velocity arrow. Legend entries
  are the run names (`labels`, or each record's `agent`) without the status and delta-v. It raises
  `ValueError` for more than three runs; draw more runs over several figures.
- The `TargetMode` protocol has `angular_velocity(t, n)`, the rate of the body axes relative to
  the Hill frame in Hill axes (rad/s); every built-in mode implements it, and a mode without it
  fails where a port's velocity is needed.

### Fixed

- The `pending_commands` channel echoed a command clipped at the true action scale, which in `accel`
  mode with the thrust drive, or `thrust` mode with the acceleration drive, moves with the mass, so
  a saturated command gave away the mass and each burn's delivered delta-v; a command dropped by the
  minimum impulse bit echoed zero, one bit on the mass per step with the acceleration drive or in
  `accel` mode. Each slot now echoes the command as issued, limited at the action-space bound, and
  commands zeroed by the minimum impulse bit or a thrust inhibit echo as issued.
- A misbriefed target (`Misbriefed`) no longer shows through the agent's view: the constraint
  specs are built from the briefed mode, so a cone on a target briefed as LVLH-stabilised no
  longer carries the rate `n` of the true mode, and the view's identifier hashes the briefed
  spec rather than the true one.
- The metrics judge an inspection by its coverage. The inspection goal no longer carries a hold
  point at the target with a 200 m capture radius, under which every inspection run, free drift
  included, was a metrics success with a composite score of 1.

## [0.3.1] - 2026-10-24

### Changed

- `CITATION.cff` gives the author's ORCID iD.

## [0.3.0] - 2026-10-24

### Changed

- `PDAgent` and `LLMAgent` raise `ValueError` at `reset` for a goal whose task is not a hold or a
  berthing, or without a hold point (and, for the LLM agent, without a keep-out radius), instead
  of a `KeyError` at reset or a `KeyError` or `StopIteration` at the first decision.
- `propagate_kepler` and `lagrange_coefficients` raise `ValueError` for arcs whose estimated end
  error from rounding exceeds 1e-5 of the end radius: near-radial and near-parabolic arcs toward
  perigee from far out, near-parabolic ellipses flown from apogee to perigee, and eccentric
  ellipses flown to perigee over very many periods. 0.2.0 returned most of the near-parabolic
  ellipses flown from apogee to perigee, off by up to 7.5e-2 of the perigee radius at
  1 - e = 1e-9 and up to twice that at 1e-10.
- `Constraint.rate_bound(speed, accel, rho_min=0.0)` takes a lower bound on the range;
  `segment_minimum` passes it by keyword, so a constraint written as `rate_bound(speed, accel)`
  raises `TypeError`. Add the `rho_min` parameter (it may be ignored).
- `Constraint.resolution` is a read-only property in the protocol rather than a class variable;
  a class attribute still satisfies it.
- `LLMAgent` calls `complete(messages, tools)` and accepts a `Reply` or text; wrap a
  `prompt -> text` function with `vbar.agents.text_model`. Options after `complete` are
  keyword-only.
- `run_episode` without a `seed` draws a 128-bit one and returns it in `seed` (it was None), so
  every run can be repeated and an agent cannot guess its episode; `run_episode` and `run_many`
  raise `ValueError` for `agent_seed` equal to `seed`.

### Added

- A `slow` test marker: the default test run leaves out the whole-episode scenario audits; run
  them with `python -m pytest -m slow`.
- The scenario catalogue (`vbar.scenarios`): `ScenarioSpec` with typed dispersions, target
  attitude modes (`Passive`, `FixedOffset`, `InertialPointing`, `Tumbling`, `Maneuvering`,
  `Misbriefed`), the kinds `HoldPointEnv`, `DockingEnv`, `InspectionEnv`, `BerthingEnv` and
  `DepartureEnv`, identifiers that name every result-changing option
  (`vbar:{scenario}:{dynamics}:{actuator}:{cost}:v{n}` plus variants and `field=value` tokens),
  the registry `SCENARIOS` with `make`, `register`, `parse_id`, `get`, `scenario_id` and
  `gymnasium_id`, tags (`tags_of`, `by_tag`, `group_by_tag`), entry-point plugins
  (`vbar.scenarios`, `vbar.agents`, `vbar.filters`; `plugins`, `load_plugins`), seed splits with
  a hidden test key (`seeds`, `SeedSplit`, `SPLITS`, `VBAR_HIDDEN_SEED_KEY`) and fixed suites
  (`SUITES`, `Suite`, `SuiteEntry`, `register_suite`). The kinds derive from `ScenarioEnv`, with
  their parameters in `ScenarioParams`, `HoldParams`, `DockingParams`, `InspectionParams`,
  `BerthingParams` and `DepartureParams`; specs take the dispersions `Fixed`, `Uniform`,
  `Normal`, `Choice` and `Shell`, an `Observation` description and a `TargetMode`, and docking
  names its contact speed window from `PRESETS`. Seventeen scenarios with their `:nav1` and
  `:nav2` variants: `approach-vbar`, `approach-rbar`, `approach-hbar`, `docking` v2 (contact as
  an event in the step, mechanism presets, corridor and lighting variants), `docking-offset`,
  `inertial-target`, `docking-afrl`, `inspection`, `closing`, `rbar-acquisition`,
  `berthing-box`, `orbt-rbar`, `rpm-hold`, `torva`, `vbar-final-shuttle`, `hst-rbar-grapple` and
  `separation-flyaround`, all registered with Gymnasium as `vbar/<Name>-v<n>`.
- `BaseEnv` hooks for scenarios: an episode draw before the start state, a command limit (thrust
  inhibit windows), segment splitting at a target's burns and an event time that ends a step
  early with the unflown propellant given back.
- `bound` on `GaussianNav`, `RangeScaledNav` and `ExecutionError`: Gaussian draws truncated by
  rejection at a level in sigma, per axis for the first and last and on the norm of each
  three-dimensional draw for `RangeScaledNav`; `vbar.scenarios.per_draw_level` derives the level
  from an episode-level exceedance probability, and `vbar.spacecraft.bounded_normal` and
  `bounded_ball` make the truncated draws.
- `vbar.solvers.HillScaling`: nondimensional units for Hill-frame problems (length L, time 1/n,
  acceleration n^2 L) with conversions of states, accelerations and durations.
- `vbar.solvers.project`: the exact projection of a three-axis command onto up to 32 linear rows
  (`Rows`) under a box or ball thrust limit (`ActuatorSet`), in a diagonal weighted norm, by
  enumeration of active sets. `Projection` gives the multipliers, the active rows and faces, and a
  status of "feasible", "infeasible_certified" (with a verified Farkas certificate from the
  sum-of-infeasibilities phase I and the conflicting rows named) or "solver_failed"; when
  infeasible, the least-violation command in the actuator set (none or a box), or under a ball
  limit the unlimited one scaled onto the ball.
- `vbar.solvers.StageQP` and `riccati_solve`: stage-structured horizon QPs with the dynamics as
  equality rows, solved without their rows by a backward Riccati recursion (one-triangle Cholesky
  of each stage's input block, the cost to go kept symmetric) at a cost linear in the horizon;
  returns states, inputs and dynamics multipliers.
- `vbar.solvers.solve_horizon`, `IPMSettings` and `HorizonSolution`: a primal-dual interior-point
  method with Mehrotra's predictor-corrector step for horizon QPs with stage and terminal rows,
  each Newton step solved by the Riccati recursion. Statuses "converged", "iteration_cap" (no
  plan), "infeasible_certified" (a Farkas certificate from the sum-of-infeasibilities LP, verified
  in numpy, with the conflicting (stage, row) pairs) and "solver_failed".
- `HillScaling.stage`: a horizon problem in Hill-frame units with the same objective value.
- A second-order in-step certificate for the keep-out sphere: `KeepOut.curvature_bound` and
  `KeepOut.second_order_bounds`, a chord bound on the squared range, and `segment_minimum` and
  `segment_minima` report the larger of it and the Lipschitz bound. A constraint may define
  `second_order_bounds(states, h, speed, accel)` to get the same treatment. On 200 coasting flybys
  sampled 20 times per 10 s the certified gap falls from 1.76 m to 4.9e-5 m at the median.
  `min_margin_bound` (S1b, metric version 2) uses it, and a refined minimum may now be reported
  from its sample when the bound shows the sample is within the constraint's resolution of the
  true minimum.
- `vbar/Approach-v0` registered with Gymnasium on import, so `gymnasium.make` builds the approach
  scenario; it passes Gymnasium's environment checker.
- `py.typed` marker, so type checkers use the package's annotations.
- A test that runs every python example in README.md and `docs/`, coverage reporting in CI, a
  throughput benchmark (`benchmarks/steps.py`) and weekly mutation testing of the core modules.
- SECURITY.md.
- Constraint types: speed and closing-speed limits (`SpeedLimit`, `ClosingSpeedLimit`,
  `closing_speed`), range-rate gate tables (`GateTable`), an approach cone with its apex at the
  port (`ApproachCone`), Sun exclusion (`SunExclusion`), passive-speed envelopes (`PassiveSpeed`)
  and an approach-zone ellipsoid (`ZoneEntry`). Vector parameters are held as tuples, and the
  agent's view gets copies of every parameter. `vbar.constraints.range_lower_bounds` bounds the
  range between samples, and `sampled_negative_intervals` finds where a margin is negative from
  evenly spaced samples and rate bounds.
- `FilterResult` with a status (one of `FILTER_STATUSES`), slack, conflicting constraints and
  compute time; `vbar.safety.run_filter` gets a filter's decision on copies of the command and
  observation and times it; `FilteredAgent` keeps the last command, result and measured filter
  time.
- `vbar.metrics`: `record_episode` returns an `Episode` with every step record (drawing and
  recording seeds not given, and refusing `agent_seed` equal to `seed`, as `run_episode` does), and
  `evaluate(episode, settings, bounds, report, posterior)` returns every metric keyed by the names
  in `METRICS`: mission metrics with docking contact and hold dwell, efficiency, the optimality gap
  against oracle bounds, constraint margins with violation depth and duration in continuous time,
  passive safety, stuck thrusters and collision-avoidance availability, thruster plume, safety
  filter and agent metrics, a composite score with its breakdown and a failure label. Each group
  has its own function: `mission_metrics`, `efficiency_metrics`, `optimality_gap`,
  `safety_metrics`, `passive_metrics`, `plume_metrics`, `filter_metrics`, `filter_behaviour`,
  `agent_metrics` and `composite_metrics`, with `success`, `captured`, `completion_time`,
  `contact` (`Contact`), `composite_score`, `infractions`, `WEIGHTS`, `weight_sensitivity`,
  `failure_label` and `LABELS`. `Trace` rebuilds the true trajectory between samples; `minimum` and
  `negative_intervals` evaluate a constraint or a `Signal` in continuous time, and `merge`,
  `complement` and `total` combine intervals. Passive safety uses `TravellingEllipse`,
  `passive_forever`, `escape_arc` and `cam_impulses`; plume metrics `exhausts` and
  `plume_normalization`; agent metrics take an `AgentReport` and use `faithfulness` and
  `parse_belief`. A card per metric is in `src/vbar/metrics/cards/`. Scenarios name their
  `soft_constraints`.
- Filter metrics F7 conservatism (`filter_conservatism`), F8 unnecessary interventions, F9 time
  to re-enter the safe set, F10 stall time, the fallback count and the certificate columns, read
  from new optional `FilterResult` fields (`active`, `recovering`, `fallback_level`,
  `certificate`, `certificate_horizon`, `iterations`); F3 and F4 from paired runs with the filter
  on and off (`paired_filter_metrics`), and the interquartile mean `iqm`.
- Agent solver metrics from optional log fields (`solver_metrics`): the preparation and feedback
  split of the latency, solver labels, tolerances and status counts, iterations, warm starts and
  fallbacks.
- Agent metric A5b (`belief_consistency`), the stated range's error from a reference navigation
  posterior (a `Posterior`), when one is passed to `evaluate`.
- `FuelBounds(lower, upper, norm, rung)`, given by an `Oracle` for a start state, names the
  dynamics rung (the `Episode.dynamics` name, for example "cw") its bounds are certified for; on
  other rungs the ratio is reported as `optimality_gap_model`, with `bound_certified` and
  `bound_rung`, and the bug flag is not set.
- Summaries across episodes: `MetricInfo.summary` says how each metric ranks, `summarize` applies
  it (per constraint for S1 to S3), `unsafe_tail` gives the worst value, quantile and tail mean of
  safety metrics, and `clopper_pearson` the exact interval of a rate. The card template has a
  Summary field and a table of rate resolution against episode count.
- `allowed_speed(rho)` on `ClosingSpeedLimit` and `GateTable`, and `MetricSettings.stall_speed`.
- `vbar.rewards`: training rewards (`Reward`) with per-second discounting, terms tied to the
  metrics (time, fuel, time in violation, violation entries, a success bonus that needs success
  without a hard violation), potential-based progress shaping and an exactly integrated quadratic
  cost (`QuadraticCost`).
- `vbar.constraints.violation_intervals` and `SegmentTrace.violations`: the intervals inside each
  step where a constraint's margin is negative.
- `EnvConfig.reward` selects a library reward and adds the step's `discount` to `info`;
  `BaseEnv.transition` builds the step's `Transition`; `SegmentTrace.starts` holds the state at
  each segment's start. Without it the approach scenario keeps its legacy reward.
- `action_mode="disc27"`: 27 bang-off-bang commands (the rows of `vbar.agents.DISC27`) as a
  `Discrete(27)` action, with its own scenario ID suffix; `vbar.agents.command_fraction` and
  `zero_action` for every mode. The drift agent handles every mode; the PD and LLM agents reject
  the discrete one.
- `action_mode="impulse_coast"`: a one-step burn and a coast of chosen length as one decision,
  discounted by its duration.
- `EnvConfig.training` and `reset(options=...)` with `"start_state"`, `"start_pos_range"` and
  `"start_vel_range"` for training; `run_episode` refuses such environments.
- `vbar.training.FilteredEnv`: a safety filter inside the training loop, reporting the executed
  action and whether the filter intervened.
- `vbar.constraints.segment_minima`: the certified in-step minimum for a batch of segments;
  `interval_bounds` accepts stacked samples.
- `vbar.dynamics.cw_states`: Clohessy-Wiltshire states for many durations at once.
- `vbar.scenarios.ApproachVectorEnv`: many approach episodes stepped as one batch on the exact
  Clohessy-Wiltshire path, with Gymnasium's vector API and same-step autoreset.
- `vbar.agents.sign_check.sign_check`: the sign-convention self-check run before scoring, with
  its result in a `SignCheck`.
- Real-time runs: `run_episode(..., real_time=RealTime(...))` applies the agent's decision time as
  latency; every result has `latencies_s` and the clock setting in `real_time`; `vbar.ScoredRun`
  steps a scored episode, with the same seed rules as `run_episode`; `BaseEnv.step` and
  `Spacecraft.apply` take a per-command latency.
- `LLMAgent` prompt templates (`prompt=`), token counts and cost from a declared `Price`, and
  `usage()` totals.
- LLM tool calling (`LLMAgent(tools=...)`, `ToolCall`) with Clohessy-Wiltshire coast and
  transfer tools computed from the agent's own observation (`vbar.agents.tools`: `CW_TOOLS`,
  `predict_coast`, `transfer_burn`, `Tool`, `call_tool`). Tools return an error for results that
  are not finite and for arguments too large for a float.
- `RecordedModel` and `ReplayModel`: record an LLM run's requests and replies and replay it
  exactly without the model; a replay that diverges from the recording raises `ReplayError`.
- `LLMAgent(belief=True)`: the model reports its range and closing belief, checked against the
  observation with the parser and match rule of metric A5 (`parse_belief`, `belief_matches`, and
  `faithful_fraction` in `usage()`).
- `vbar.agents.plan`: plan submission with an external CW verifier (`PlanVerifier`, `Verdict`),
  `PlanAgent` with revision from the verdict, and `LLMPlanner`. `parse_plan` refuses steps that
  are not whole numbers of at least zero and plans longer than `max_rows` (`MAX_PLAN_ROWS`, a
  million by default); the verifier predicts at most the steps left, and refuses at `reset` a
  goal whose task is not a hold or a berthing, or without a hold point.
- `vbar.agents.remote`: an agent server (`AgentServer`, each connection a `RemoteAgent`) with a
  JSON-lines TCP protocol, a Python client
  (`serve_agent`), a command-line host, and a MATLAB client in `clients/matlab/`. The server
  bounds each message by its timeout and each line by 1 MiB, accepts only an integer protocol and
  records episodes a client leaves unfinished; the host records every episode even when
  interrupted; `RemoteAgent.act` refuses actions of the wrong form with `RemoteAgentError`; the
  Python client converts actions as the environment does. The host draws each episode's seed and
  records it unless `--seeds` is given.
- `vbar.agents.mcp`: a Model Context Protocol tool server over stdio for LLM harnesses
  (`python -m vbar.agents.mcp`, or `ToolServer` in process; a tool call that cannot be carried
  out raises `ToolError` and reaches the client as a tool error). It records episodes a client
  leaves unfinished and answers `params` or `arguments` that are not objects, tool names that are
  not strings and request ids that are not strings, finite numbers or null with JSON-RPC errors.
  Its command line draws seeds the same way.
- `vbar.agents.policy`: MLP policies with observation normalization and action squashing
  (`MLPPolicy`, flown in process by `PolicyAgent`), ONNX export with both built in
  (`export_onnx`), an ONNX agent (`OnnxAgent`) and latency profiles (`profile_latency`); the ONNX
  parts need the optional `onnx` extra and without it raise `ModuleNotFoundError` naming the install
  command.
- Every `run_episode` result has `completed`.

### Fixed

- An agent could read `agent_seed` back from its own generator and, when it equalled the
  episode seed, rebuild the start state and the sensor noise. The agent's stream is now seeded
  from a hash of `agent_seed` (`vbar.agents.agent_generator`), so agents that draw random numbers
  get different draws than in 0.2.0 for the same `agent_seed`, and `agent_seed` may not equal
  `seed`. A small seed can still be guessed, so seeds not given are drawn and recorded (see
  Changed).
- LLM replies with a number too large for a float, or nested more than 32 levels deep, are parse
  failures on every Python version. In 0.2.0 the number raised `OverflowError`, and nesting deep
  enough for `json.loads` to give up raised `RecursionError`; either ended the run.
- `run_episode` recorded the agent's own array as each action, so an agent that reused one array
  for every command rewrote its recorded history; each action is now copied.
- `propagate_kepler` and `lagrange_coefficients` converge on ellipses, parabolas and hyperbolas
  alike: the universal anomaly is found by safeguarded Newton inside a bracket, and an elliptic time
  is reduced with an exact remainder. In 0.2.0 a few in a thousand random orbits with e above 0.89
  propagated over several periods, about one in five random hyperbolas flown back from far out, and
  short arcs on hyperbolas within 1e-8 of escape speed raised "did not converge"; some
  near-radial arcs raised `OverflowError` or `ZeroDivisionError`, or a math domain error from the
  hyperbolic starting value. Arcs that rounding would move too far now raise `ValueError` (see
  Changed); 0.2.0 raised `RuntimeError`, `OverflowError`, `ZeroDivisionError` or a math domain
  error on most of the hyperbolas among them.
- The `pending_commands` channel echoed each queued command as the stored thrust fraction scaled
  at the current mass, so in `accel` mode with the thrust drive, or `thrust` mode with the
  acceleration drive, the slot showed the mass change since the command was issued and hence the
  delivered delta-v of each burn. Slots now hold the command as issued, in action units at the
  time it was issued; matching mode and drive pairs are unchanged.
- `run_episode` copied an `interventions` attribute from any agent into `filter_interventions`;
  only `FilteredAgent`'s own count is reported now.
- `vbar.viz.plot_trajectories` and `examples/v0_demo.py` without matplotlib raised a bare
  `ModuleNotFoundError`; its message now names the `plot` extra's install command.

## [0.2.0] - 2026-10-16

### Changed

- `vbar.envs` is now `vbar.scenarios`. `from vbar import ApproachConfig, ApproachEnv` still works.
- Agents implement `reset(view, rng)` instead of `reset(env)`. `view` is a read-only `AgentView`
  with the scenario's public description; `rng` is the agent's own random stream.
- The `info` an agent receives holds only the time. The Gymnasium `info` from `step` keeps the
  metrics and the true state for scoring.
- Safety filters implement `reset(view)` and `filter(obs, action)`.
- Config dataclasses take keyword arguments only, and `ApproachConfig.extra` was removed.
- `run_episode` takes `agent_seed` and returns it; `run_many` takes `agent_seeds`.

### Added

- Spacecraft model: mass, propellant and the rocket equation, run-out, box or ball thrust limits,
  minimum impulse bit, actuation delay with piecewise segments, constant-acceleration or
  constant-thrust drive, and a thrust execution error model.
- Navigation sensors: white noise with an episode bias, range-scaled errors in sensor axes, and
  time-correlated errors.
- Continuous constraint checking with local refinement and a certified lower bound on the
  minimum margin within each step.
- Optional observation channels: time left, propellant left and pending commands.
- Action modes in normalized units, m/s^2 or newtons.
- Dynamics: differential drag and solar radiation pressure with Earth shadow, elliptic target
  orbits, Yamanaka-Ankersen, curvilinear Clohessy-Wiltshire, linearized two-body motion with J2,
  transition matrices from the variational equations, model-error measures, Earth-Moon three-body
  models and halo-orbit tools.
- Documentation pages in `docs/` for every part.

## [0.1.0] - 2026-10-16

### Added

First release with the validated dynamics layer: exact Clohessy-Wiltshire, nonlinear two-body
relative motion with J2, Kepler propagation and J2 mean elements.

[Unreleased]: https://github.com/jamilhammoudeh/vbar/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/jamilhammoudeh/vbar/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/jamilhammoudeh/vbar/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jamilhammoudeh/vbar/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jamilhammoudeh/vbar/releases/tag/v0.1.0
