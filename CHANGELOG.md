# Changelog

Notable changes to V-BAR. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0, a minor
release may change the public API, and every such change is listed under Changed or Removed.

## [Unreleased]

### Added

- `vbar.replay.export_rrd` and the optional extra `replay`: writes a recorded episode as a Rerun
  `.rrd` file in the Hill frame with SI units, with the chaser and target, a trail, the keep-out
  spheres and approach corridors, the hold point or port, the command, filtered command and flown
  acceleration, violation and filter events, and the metric card as scalars over time.
  `examples/replay_demo.py` writes one for the PD baseline; `docs/replay.md` documents it.
- `vbar.games`: KSPDG's pe1 pursuit-evasion problem on two-body dynamics around Kerbin, in
  KSPDG's observation list, action dictionary and four-value step: `PursuitEvasionEnv`, the
  scenarios `PE1_E1_I3_Env` and `PE1_E3_I3_Env` (KSPDG's I3 start against V-BAR's own
  `PassiveEvader` and `NormalEscapeEvader`), registered as `vbar/PursuitEvasionE1I3-v1` and
  `vbar/PursuitEvasionE3I3-v1` (`PursuitEvasionGymEnv`); `BaseAgent` and `AgentEnvRunner` with
  the signatures of KSPDG's agent interface, so KSPDG agents run unchanged; `PursuitTask` for
  `vbar.eval.run_jobs`; the reference pursuers `LineOfSightPursuer` and
  `VelocityMatchingPursuer`; `docs/pursuit-evasion.md`.
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
- Every scenario goal carries `cam_impulses`, the abort set the abort metrics check.
- The core suite `core-v1` (the V-bar and R-bar approaches, `docking:axis`, `docking-afrl` and the
  inspection, each with and without `:nav1`, 100 validation seeds each) and `core-test-v1` (the
  same on the hidden test split); `Suite.require_complete` refuses a partial report.
- `DriftingVelocity`, a start-velocity dispersion that excludes slowly drifting orbits.
- Comparison tasks rebuilt from their papers: `afrl-docking` and `afrl-inspection` (the AFRL
  safe-autonomy deep-RL docking and illuminated inspection tasks) and `spacegym-docking` and
  `spacegym-rendezvous`; the dispersions `AngleShell` and `LimitedSpeed`; `HoldParams.out_of_bounds`
  and `end_on_arrival`; `InspectionParams.shadow` and `out_of_bounds`.
- `vbar.eval`, the evaluation harness: the suite runner (`run_suite`, `run_jobs`) with resumable
  JSON-lines results, a provenance hash per run and the `Task` interface.
- `vbar.eval.stats` and `vbar.eval.tails`: the IQM, stratified bootstrap, performance profiles,
  probability of improvement and paired differences; safety tails with order-statistic intervals
  and tolerance bounds, and rate comparisons that refuse unresolved claims.
- `vbar.eval.coverage`: coverage of the scoring intervals on synthetic outcomes, and McNemar's
  exact test.
- `vbar.eval.learned`: learned-agent entries (`LearnedEntry`, `TrainingRun`), training runs as the
  unit, failed runs never replaced.
- `vbar.eval.report` and the command line `python -m vbar.eval run | report | reproduce`: suite
  reports that refuse a partial suite, and `reproduce.toml`, which rebuilds every published table.
- `Episode.reset_s` and the runner's `reset_s`: the agent's time in `reset`.
- `Episode.target_misbriefed`: whether the target moved otherwise than the briefing said, set by
  `record_episode` from the environment's true target mode; never part of what an agent sees.
- `vbar.dynamics.backends`: Basilisk and Orekit as independent propagators of the two-body plus J2
  rung (the optional extras `basilisk` and `orekit`), and the agreement table against them
  (`benchmarks/agreement_table.py`).
- `relative_acceleration` in `vbar.dynamics`: the Hill-frame acceleration of a coasting chaser
  from the target's inertial state, with or without J2, for model-based controllers.
- `vbar.agents.lqr` design functions: discrete LQR by the discrete Riccati equation, the exact
  sampled-data stage cost of a continuous quadratic cost (Van Loan), delay and input-rate
  augmentation, and the steady-state input of a hold point.
- `vbar.agents.planning`: `Impulse`, `impulse_action` (an impulse as a one-step burn with a coast,
  or a multi-step burn when it exceeds one step of thrust) and `predicted_margin` (the keep-out
  margin over a planned coast at the environment's sub-step samples, with its certified bound).
- `vbar.agents.cw_targeting` functions: `cw_lambert` and `two_impulse` (Clohessy-Wiltshire
  two-impulse targeting by solving the transition-matrix relation) and `conditioning` (a sensitivity
  measure that vanishes on the singular transfer times).
- `TwoImpulseAgent` (open-loop, re-targeting and tracking modes) and `TwoImpulseBound`, an
  interim fuel oracle whose upper bound is the best two-impulse cost over a grid of transfer
  angles, in `vbar.agents.cw_targeting`; the Curtis section 7.5 rendezvous examples are reproduced
  in the tests. `TwoImpulseBound` raises `TypeError` for a `grid_points` that is not an integer and
  `ValueError` below 2.
- `CWTrackingAgent` (the Clohessy-Wiltshire tracking law of Alfriend et al. with a reference
  hook) and `LyapunovAgent` with `DeclaredModel` (the Cartesian Lyapunov law of Schaub and Junkins
  on a declared two-body or J2 model) in `vbar.agents.lyapunov`. `DeclaredModel` raises
  `ValueError` for an unknown dynamics or body, a negative or non-finite altitude and a
  non-finite inclination.
- Fehse trajectory elements in `vbar.agents.elements`: tangential and radial V-bar transfers, the
  CW Hohmann, fly-arounds, forced straight-line motion, the twice-orbital-rate circle and
  finite-burn durations, with the agents that fly them (`ElementAgent`, `TorvaAgent`,
  `FehseSequenceAgent`, `NominalTrackingAgent`).
- `GlideslopeAgent` in `vbar.agents.glideslope`: the exponential range-rate glideslope of Hablani,
  Tapper and Dana-Bashian with pulses computed from the CW transition matrices between waypoints
  on the line of sight.
- `lambert` in `vbar.dynamics.lambert`: a universal-variable Lambert solver for the two-body
  problem (single revolution, prograde or retrograde), checked against Montenbruck and Gill
  Exercise 2.6 and the Lagrange-coefficient identities, refusing transfers too short for its
  rounding; and `LambertAgent`, which targets the hold point on two-body truth from the mean motion
  and a declared gravitational parameter.
- `MissionManagerAgent` and `Mode` in `vbar.agents.mission`: a timeline of modes, each flown by its
  own agent, with exit conditions, time limits, contingency and abort transitions, after Fehse
  section 6.3.
- `LQGAgent` and `LQGConfig` in `vbar.agents.lqg`: a steady-state Kalman filter on the CW model
  with a declared sensor law and optional bias states, feeding the LQR hold-point regulator; a
  baseline for noisy navigation, never an oracle. `LQGConfig` raises `ValueError` for a sigma that
  is not positive and finite and for a process noise or bias walk that is negative or not finite.
- `Mutant` and `MUTANT_KINDS` in `vbar.agents.mutants`: deliberately broken controllers (biases,
  delay, sign errors, dither, chatter, a dropped Coriolis term, bang-bang) wrapped around any agent,
  for checking that the metrics detect each fault and rank its severity.
- Shuttle-practice baselines in `vbar.agents.shuttle`: the coelliptic (NSR) and football aim
  points, an elevation-timed transfer burn, and a gate-following pilot model with fixed pulses
  on the range rate and the line-of-sight rate, after Goodman, History of Space Shuttle Rendezvous.
- The classical baselines in this section and `MUTANT_KINDS` are exported from `vbar.agents`; the
  MPC and CLF-CBF baselines are imported from their modules. `docs/validation.md` lists the
  published examples the baseline tests reproduce.
- Every classical baseline that flies to the hold point refuses a goal whose task is not a hold or
  a berthing, and a goal without a hold point, at reset, as the other built-in agents do;
  `ElementAgent` and `TorvaAgent`, which never read the goal, accept any task.
- `admissible_set` in `vbar.agents.mpc`: the maximal output admissible set of a stable closed loop
  under symmetric row constraints, by the Gilbert and Tan iteration on unit-scaled rows, the terminal
  set of the MPC baselines; every returned set is certified invariant by exact weak-duality
  multipliers, and a set that cannot be certified raises.
- `MPCConfig`, `mpc_problem` and `mpc_solve` in `vbar.agents.mpc`: the hold-point MPC horizon QP on
  the zero-order-hold CW model from the shared regulator design, with keep-out rows sampled within
  each step, a slack under an exact penalty, an optional terminal set, and the solver's status,
  path and conflicting rows passed through from `vbar.solvers`.
- `MPCAgent` in `vbar.agents.mpc`: the hold-point MPC baseline, one interior-point solve per
  decision from a warm start, a passing side chosen with hysteresis, the in-step keep-out
  certificate of each applied step logged, and documented fallback levels when the solve fails.
- `NMPCAgent` and `shooting_linearization` in `vbar.agents.nmpc`: nonlinear MPC by multiple
  shooting on the declared dynamics model, with a real-time iteration split and a cold retry when
  a warm-started solve fails.
- `regulator_model` and `RegulatorModel` in `vbar.agents.lqr`: the LQR design (augmented model,
  gain, Riccati solution, steady state and stage costs, with read-only arrays) shared by `LQRAgent`
  and the MPC baselines, so an MPC equals the LQR wherever no constraint binds.
- `PiecewiseAffineMPCAgent` in `vbar.agents.mpc_pwa`: the box-constrained regulator to a hold point
  solved once at reset as a multiparametric quadratic program over a box of states, with each
  decision a lookup in its table of polyhedral regions and one affine law; `mpqp_regions` and
  `lookup` build and search the table.
- `ClfCbfAgent` and `clf_cbf_qp` in `vbar.agents.clf_cbf`: a hold-point controller that solves one
  small QP per decision, a control Lyapunov row at the LQR's guaranteed decay rate relaxed at a
  cost and a keep-out barrier row with the thrust limit kept exactly, falling back to the
  least-violation command when the rows conflict.
- `certify_impulsive`, `PrimerCertificate`, `indirect_optimum` and `hamiltonian` in
  `vbar.oracles`: Lawden's primer-vector conditions checked over the whole transfer for an
  impulsive plan on CW motion in the actuator's norm, so a certified plan's cost is a known
  optimum, and the finite-thrust optimum by the primer's switching control.
- `minimum_time`, `MinimumTime` and `verify_infeasible` in `vbar.oracles`: the fewest held steps
  into the capture ball on CW motion with box thrusters, with a plan verified by exact propagation
  and, for each smaller step, a multiplier that proves no acceleration history within the limit
  reaches the ball, or the step listed as uncertified.
- `OracleAgent`, `expert_dataset` and `ExpertDataset` in `vbar.oracles`: an agent that flies the
  verified fuel plan of the oracle open loop from its own view, and expert datasets of its episodes
  on validation seeds, saved as NPZ files with a dataset card and regenerated byte for byte;
  `benchmarks/make_datasets.py` writes them to a directory the caller names.
- `PassingSideOracle`, `PassingSolution` and `PassingSide` in `vbar.oracles`: fuel bounds when the
  target lies between the chaser and its goal, with one keep-out halfspace per side of approach pushed
  until the plan clears the keep-out sphere between samples, and the lower bound of the fuel oracle
  without the sphere.
- Free final time and dwells in `FuelOracle`: without `arrival_step`, `solve` and `bounds` bound
  the least delta-v of a success at any step of the episode, a hold with `hold_time` being in the
  capture ball at floor(hold_time / dt) consecutive step boundaries, with a verified plan that
  holds the dwell; `OracleConfig.arrival_candidates`, `bank_stride` and `max_refinements`.
  `DualBank` in `vbar.oracles` reuses the multipliers of window programs across starts and
  windows by a time shift, each bound evaluated through the primer bound, and `bounds_for(ep)`
  gives the bounds of a recorded episode for the optimality-gap metrics. Docking goals and briefed
  targets with burns raise `ValueError`.
- `benchmarks/lower_bound_check.py`: flies every classical baseline and one mutant per kind on the
  CW approach, acquisition and docking-afrl scenarios over public seeds, scores each run against a
  `DualBank` bound for its start, prints runs, successes, certified runs and E3f flags per scenario
  and agent, and exits with 1 when any run is flagged and 3 when no run was held to a certified
  bound.
- `FuelBounds.accel_limit` and `FuelBounds.impulsive_lower`: a lower bound counts as certified for
  an episode only when its acceleration limit is at least what the run delivered (per axis for a
  box, in 2-norm for a ball, at the final mass under the constant-thrust drive) as well as matching
  its dynamics rung; otherwise it is reported as a model bound, and the certified figure falls back
  to the impulsive bound when one is given. `Episode.drive` records the thrust drive from the
  environment configuration.
- `vbar.oracles`: the CW primer vector and a rigorous finite-thrust dual lower bound on the fuel of
  any admissible manoeuvre, with the dual norm of the thruster layout and a certificate that holds
  between grid points (`vbar.oracles.primer`); and the exact constrained infinite-horizon LQ value
  with its lower bound and a validity flag, for CW only (`vbar.oracles.lq`).
- `FuelMPCAgent` in `vbar.agents.mpc_fuel`: fuel-optimal model predictive control by linear
  programming on the CW model with a declared receding or shrinking horizon, the arrival time
  reported, a unique applied action through a strictly convex tie-break, and a certified bound on
  each plan's excess fuel over the linear-programming optimum; box thrust limits only.
  `FuelMPCConfig` and `fuel_plan` raise `TypeError` for a `max_iter` that is not an integer and
  `ValueError` below 1.
- `FuelOracle` in `vbar.oracles`: certified lower and upper bounds on the fuel of a success on the
  CW model, at a free final time by default and at a fixed arrival step with `arrival_step`, by an
  exact linear program for a box thrust layout and a bracket of linear programs for a ball,
  returned as `FuelBounds` for the optimality-gap metrics.
- `LQRAgent` and `LQRConfig` in `vbar.agents.lqr`: hold-point regulation on the discrete CW model
  with the steady-state input as feedforward, a sampled-data or discrete stage cost, delay
  augmentation from the pending-commands channel and an optional input-rate weight. The default
  scales, 10 m, 0.03 m/s and the view's `max_accel`, give the slowest closed-loop mode a 333 s
  time constant, so the agent reaches and holds the `approach-rbar` point within its episode.
  `LQRConfig` raises `ValueError` for a scale that is not positive and finite and for a
  `rate_weight` that is negative or not finite.
- `vbar.agents.lqg` functions: the steady-state Kalman gain of the discrete CW model and the
  detectability test for sensor bias states (a constant along-track position bias is not
  detectable).
- `Episode.target`: the scenario's true target attitude mode, recorded for the metrics.

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
- `BaseEnv.step`, `ApproachVectorEnv.step` and `FilteredEnv.step` raise `ValueError` for an
  action of the wrong shape or with a non-finite entry before anything is flown. Zeros of the
  wrong shape were flown as a coast, and a nonzero action of the wrong length had already been
  queued and paid for when it failed. The vector environment no longer takes the actions as one
  flat array.
- `EnvConfig`, `ApproachConfig`, `MetricSettings`, `KeepOut`, `SpeedLimit` and
  `ClosingSpeedLimit` raise `ValueError` for NaN, infinite or non-numeric values, and for negative
  ones where the quantity cannot be negative (a NaN keep-out radius reported no violation, a
  negative collision radius scored every drift safe).
- `summarize` and `unsafe_tail` report `n` and `missing` (the NaN values left out), and
  `summarize` raises for a NaN success, progress or composite score.
- A2 `compute` (version 2) includes the agent's time in `reset`.
- E3, E3m, E3u, E3f and E3c are at version 2. E3u is NaN for an infinite upper bound (0 in
  version 1); with `FuelBounds.accel_limit` at its default, infinite, and a target that moved as
  briefed, every other value is the version 1 value. A run whose target moved otherwise than
  briefed (`Episode.target_misbriefed`, set by `record_episode` from the environment's true target
  mode) is certified for no lower bound, so its ratio is reported as E3m.
- Identifiers name a sensor subclass by its class, never by a built-in track name such as `nav1`;
  a custom `label` that takes a built-in name raises; an environment of another class than its
  spec names adds a `kind=` token.
- `inspection` is version 2: its start velocity is drawn again until the start drifts along
  track (`DriftingVelocity`), since free drift completed version 1 on about 1 seed in 60.
- Docking contact speeds are measured against the port's own velocity, which a turning target
  gives: `inertial-target` is v2 (its 2 m port moves at 2.2 mm/s) and the `v1-operations` suite
  is v2.
- The contact metrics (M1, M4, M6 to M8, M10, now v2) count a crossing of the port plane only
  at a point within the goal's `contact_range` of the port, carry the port with the target's true
  attitude mode and measure speeds against the port's velocity, applying the environment's rule
  to the recorded samples; a crossing far beside the target is no longer a contact, and on a
  turning port they now agree with the environment's verdict. M2, M3 and M5 (v2) take a docking
  goal's port where it stands, with its velocity.
- `DockingEnv` judges the contact range at the crossing point rather than at the samples either
  side of it, a difference only for crossings within a sample's travel of the range's edge.

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
- The hidden test key is read without surrounding white space and in either case and must be at
  least 32 hex digits, so a key read from its file with the newline gives the same split as the
  bare key.
- `PDAgent` clipped each axis to `max_accel` even on a ball thrust layout, so a diagonal command
  exceeded the limit in 2-norm and the actuator saturated it; it now scales such a command back
  to the ball with its direction kept, and refuses an unknown thrust layout at reset.

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
