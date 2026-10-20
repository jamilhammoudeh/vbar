# Changelog

Notable changes to V-BAR. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0, a minor
release may change the public API, and every such change is listed under Changed or Removed.

## [Unreleased]

### Changed

- `Constraint.rate_bound(speed, accel, rho_min=0.0)` takes a lower bound on the range;
  `segment_minimum` passes it by keyword, so a constraint written as `rate_bound(speed, accel)`
  raises `TypeError`. Add the `rho_min` parameter (it may be ignored).
- `Constraint.resolution` is a read-only property in the protocol rather than a class variable;
  a class attribute still satisfies it.
- Filter metric F1 counts decisions with an active safety row or a change in the flown
  acceleration, not every change; clipping to the thrust limit is counted apart (F1c). F2 is the
  change in flown acceleration in m/s^2 rather than fractions of a full command over sqrt(3), and
  F2n normalises it by the actuator's widest change.
- S5 divides by the speed the scenario's closing-rate law allows, so it also scores range-rate
  tables, and no longer raises for a limit with v0 <= 0; the plume range is the scenario's unless
  the settings give one.
- `LLMAgent` calls `complete(messages, tools)` and accepts a `Reply` or text; wrap a
  `prompt -> text` function with `vbar.agents.text_model`. Options after `complete` are
  keyword-only.

### Added

- `vbar/Approach-v0` registered with Gymnasium on import, so `gymnasium.make` builds the approach
  scenario; it passes Gymnasium's environment checker.
- `py.typed` marker, so type checkers use the package's annotations.
- A test that runs every python example in README.md and `docs/`, coverage reporting in CI, a
  throughput benchmark (`benchmarks/steps.py`) and weekly mutation testing of the core modules.
- SECURITY.md.
- Constraint types: speed and closing-speed limits, range-rate gate tables, an approach cone with
  its apex at the port, Sun exclusion, passive-speed envelopes and an approach-zone ellipsoid.
- `FilterResult` with a status, slack, conflicting constraints and compute time; `FilteredAgent`
  keeps the last command, result and measured filter time.
- `vbar.metrics`: `record_episode` returns an `Episode` with every step record, and
  `evaluate(episode, settings, bounds, report, posterior)` returns every metric keyed by the names
  in `METRICS`: mission metrics with docking contact and hold dwell, efficiency, the optimality gap
  against oracle bounds, constraint margins with violation depth and duration in continuous time,
  passive safety, stuck thrusters and collision-avoidance availability, thruster plume, safety
  filter and agent metrics, a composite score with its breakdown and a failure label. `Trace`
  rebuilds the true trajectory between samples; `minimum` and `negative_intervals` evaluate signals
  in continuous time. A card per metric is in `src/vbar/metrics/cards/`. Scenarios name their
  `soft_constraints`.
- Filter metrics F7 conservatism, F8 unnecessary interventions, F9 time to re-enter the safe set,
  F10 stall time, the fallback count and the certificate columns, read from new optional
  `FilterResult` fields (`active`, `recovering`, `fallback_level`, `certificate`,
  `certificate_horizon`, `iterations`).
- Agent solver metrics from optional log fields: the preparation and feedback split of the
  latency, solver labels, tolerances and status counts, iterations, warm starts and fallbacks.
- Agent metric A5b, the stated range's error from a reference navigation posterior, when one
  is passed to `evaluate`.
- `FuelBounds(lower, upper, norm, rung)` names the dynamics rung (the `Episode.dynamics` name,
  for example "cw") its bounds are certified for; on other rungs the ratio is reported as
  `optimality_gap_model`, with `bound_certified` and `bound_rung`, and the bug flag is not set.
- Summaries across episodes: `MetricInfo.summary` says how each metric ranks, `summarize` applies
  it (per constraint for S1 to S3), `unsafe_tail` gives the worst value, quantile and tail mean of
  safety metrics, and `clopper_pearson` the exact interval of a rate. The card template has a
  Summary field and a table of rate resolution against episode count.
- `allowed_speed(rho)` on `ClosingSpeedLimit` and `GateTable`, and `MetricSettings.stall_speed`.
- `vbar.rewards`: training rewards with per-second discounting, terms tied to the metrics (time,
  fuel, time in violation, violation entries, a success bonus that needs success without a hard
  violation), potential-based progress shaping and an exactly integrated quadratic cost.
- `vbar.constraints.violation_intervals` and `SegmentTrace.violations`: the intervals inside each
  step where a constraint's margin is negative.
- `EnvConfig.reward` selects a library reward and adds the step's `discount` to `info`;
  `BaseEnv.transition` builds the step's `Transition`; `SegmentTrace.starts` holds the state at
  each segment's start. Without it the approach scenario keeps its legacy reward.
- `action_mode="disc27"`: 27 bang-off-bang commands as a `Discrete(27)` action, with its own
  scenario ID suffix; `vbar.agents.command_fraction` and `zero_action` for every mode. The drift
  agent handles every mode; the PD and LLM agents reject the discrete one.
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
- `vbar.agents.sign_check.sign_check`: the sign-convention self-check run before scoring.
- Real-time runs: `run_episode(..., real_time=RealTime(...))` applies the agent's decision
  time as latency; every result has `latencies_s`; `vbar.Episode` steps a scored episode;
  `BaseEnv.step` and `Spacecraft.apply` take a per-command latency.
- `LLMAgent` prompt templates (`prompt=`), token counts and cost from a declared `Price`, and
  `usage()` totals.
- LLM tool calling (`LLMAgent(tools=...)`, `ToolCall`) with Clohessy-Wiltshire coast and
  transfer tools computed from the agent's own observation (`vbar.agents.tools`).
- `RecordedModel` and `ReplayModel`: record an LLM run's requests and replies and replay it
  exactly without the model.
- `LLMAgent(belief=True)`: the model reports its range and closing belief, checked against the
  observation (`belief_matches`, `faithful_fraction`).
- `vbar.agents.plan`: plan submission with an external CW verifier (`PlanVerifier`, `Verdict`),
  `PlanAgent` with revision from the verdict, and `LLMPlanner`.
- `vbar.agents.remote`: an agent server with a JSON-lines TCP protocol, a Python client
  (`serve_agent`), a command-line host, and a MATLAB client in `clients/matlab/`.
- `vbar.agents.mcp`: a Model Context Protocol tool server over stdio for LLM harnesses
  (`python -m vbar.agents.mcp`).
- `vbar.agents.policy`: MLP policies with observation normalization and action squashing, ONNX
  export with both built in, an ONNX agent and latency profiles; optional `onnx` extra.
- Every `run_episode` result has `completed`; the agent and tool servers record episodes a
  client leaves unfinished, and the tool server answers `params` or `arguments` that are not
  objects, null included, with a JSON-RPC error.
- `RemoteAgent.act` refuses actions of the wrong form with `RemoteAgentError`; the tool server
  answers a tool name that is not a string with -32602.
- The agent server bounds each whole message by its timeout and lines by 1 MiB, accepts only an
  integer protocol, and the command-line host records every episode even when interrupted.
- LLM tools return an error for results that are not finite and for numbers too large for a
  float; such a reply counts as a parse failure. The Python agent client converts actions as
  the environment does. The tool server refuses request ids that are not strings, finite
  numbers or null.
- `parse_plan` takes `max_rows` (a million by default) and refuses steps that are not whole numbers
  of at least zero, where it used to truncate or skip them; the verifier predicts at most the
  steps left; replies too deeply nested are parse failures.

### Fixed

- An agent given an `agent_seed` equal to the episode seed could read it back from its own
  generator and rebuild the start state and the sensor noise. The agent's stream is now seeded
  from a hash of `agent_seed` (`vbar.agents.agent_generator`), so agents that draw random numbers
  get different draws than in 0.2.0 for the same `agent_seed`.
- `propagate_kepler` and `lagrange_coefficients` converge at high eccentricity over many periods:
  an elliptic time is reduced to one period and the universal anomaly is solved inside a bracket.
  Before, a few in a thousand random orbits with e above 0.89, propagated over several periods,
  raised "did not converge".
- LLM replies nested more than 32 levels deep are parse failures on every Python version; on 3.13
  and later, deeper replies used to parse and the belief parser accepted them.

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

[Unreleased]: https://github.com/jamilhammoudeh/vbar/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jamilhammoudeh/vbar/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jamilhammoudeh/vbar/releases/tag/v0.1.0
