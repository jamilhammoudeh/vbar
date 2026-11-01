# Changelog

Notable changes to V-BAR. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0, a minor
release may change the public API, and every such change is listed under Changed or Removed.

## [Unreleased]

### Added

- `vbar.eval.inference`: `holm` and `benjamini_hochberg` corrections for a family of p-values and
  `sign_flip_test`, a paired randomisation test of two agents flown on the same seeds.

## [0.5.0] - 2026-10-31

### Added

- `vbar.scenarios.seeds.seed_digest`, the keyed digest a published result of the test split holds
  in place of its seed; `vbar.scenarios.suites.match_runs`, which matches results to a suite's runs by
  seed or by that digest; and `vbar.scenarios.seeds.HIDDEN_LOG_VARIABLE`, the environment variable
  naming the private file the test split's seeds are written to.
- `[[script]]` entries in `reproduce.toml` for files the suite runner does not make, with the README
  figure, the agreement table and the coverage table listed, and `--force` and `--skip-missing` on
  `python -m vbar.eval reproduce`. A manifest with unknown keys, duplicated output paths or wrong
  types is refused with `ValueError`.
- `vbar.agents.glideslope_lp.lp_glideslope`: the minimum-fuel V-bar glideslope by linear programming of
  Ariba, Arzelier, Urbina and Louembet (2016); `vbar.reproductions` rebuilds their case 1 and prints
  the match table (`python -m vbar.reproductions`); `docs/reproductions.md` documents it.
- `vbar.games`: `Segment`, the stretch of an evader bot's flight with constant body-axis throttles,
  a pointing (prograde or orbit normal), an end time and end triggers; `axes_along`; a `seed`
  for `PursuitEvasionEnv` and its `reset`, feeding the evader's random stream (a fresh 128-bit
  seed when none is given, kept in `seed`), passed on by `fly(..., seed=...)`, `PursuitTask` and
  the Gymnasium environments' `reset(seed=...)`.
- `vbar.flightdata`: flown orbits replayed through the relative-motion models (`replay`, `hill_states`,
  `earth_fixed_to_inertial`, `gmst`, `sun_direction`, `error_envelope`, a GRACE-FO Level-1B reader and
  an excerpt format), and `TwoBodyDynamics.reset_inertial`. A GRACE-FO excerpt of 2019-12-03
  (doi:10.5067/GFL1B-ASJ04, CC0) checks the two-body, J2 and drag rungs over ten orbits against a bound
  stated before the run; `docs/flight-data.md` documents it.
- The suite `comparison-v1` (`afrl-docking`, `afrl-inspection`, `spacegym-docking` and
  `spacegym-rendezvous`, 100 validation seeds each) and a row of `reproduce.toml` for the LQR
  baseline on it.
- `examples/quickstart.ipynb`, a notebook that installs vbar from PyPI, flies the PD baseline, prints
  its metrics and rebuilds a table of two episodes through the entry point of
  `python -m vbar.eval reproduce`.
- `vbar.safety.projections`: halfspace, box, ball and second-order cone projections, and an exact
  projection out of a keep-out sphere with the hard case handled.
- `vbar.safety.tightening`: box, ball and ellipsoid error sets with halfspace, chance and keep-out
  margins and the execution-bias offsets of exponential and first-order barriers.
- `vbar.safety.barriers`: control barrier functions for the keep-out sphere (`KeepOutEcbf`, an
  exponential barrier with real poles and its start set), a range-dependent speed limit
  (`SpeedCbf`), a closing-speed limit (`ClosingSpeedCbf`) and an approach cone (`ConeEcbf`, with
  `infeasible_radius` and an `active_range` that may be `inf`, the `ApproachCone` default), each
  giving its condition as one linear row on the command; and two evading-maneuver barriers
  without rows: `PassiveBarrier`, the certified free-drift keep-out margin over a horizon set by
  the secular drift, with its envelope-theorem gradient (with `terminal=True`, drift clear for
  all time through a terminal set reached within the cap), whose `value` is lowered by 3 eps times
  the magnitudes the margin is computed from, so it never exceeds the exact margin where the
  drift's minimum falls at a sample and is negative for a state on the sphere whose drift dips
  inside (`CamBarrier` inherits this); and `BrakingBarrier(radius, a_b, rho_cap, v_cap, n, dt,
  a_max)`, the braking barrier h_b and the outer barrier `outer_value`, h_o = rho_cap - rho -
  max(0, rho_dot)^2 / (2 a_b), with the speed cap |v| <= v_cap. Its backup cancels the CW
  acceleration and brakes along and across the line of sight, and every held step of the backup
  keeps the set. `braking_deceleration` gives the braking acceleration the backup holds under the
  thrust limit (0.00758 m/s^2 for the approach box, and 0.0 where the backup keeps no set, above
  33.37 s there); `BrakingBarrier` refuses an `a_b` above it.
- `vbar.safety.sampled`: `held_segment_bound`, a certified lower bound on the keep-out margin over
  a held step on the exact CW solution, tightened for navigation error sets and execution error,
  and `line_search`, the bisection from a candidate command toward a backup, returning a
  `LineSearch`.
- `vbar.safety.cbf`: `CbfFilter`, an exponential keep-out barrier with speed rows projected under
  the thrust limit, a pole rule max(p) dt <= 0.6 checked at reset, a declared buffer and safe
  set, and an exact held-step check (end-of-step speed cap by cutting planes, certified keep-out
  bound); with declared `execution` and `model_error` every speed row enforces v0 lowered by
  `first_order_bias(execution + model_error, gamma)`, reset refuses an error that uses up a
  speed limit, and the declared safe set adds `execution + model_error` to the thrust budget;
  `BackupFilter`, free-drift or braking backup filters that check each held step on the
  exact CW solution and search toward the backup, with "nominal" or "bounded-W" certificates
  (the braking filter under error sets still tightens its checks but makes no certified claim:
  its declaration and steps carry no certificate, and it still counts toward certification;
  without error sets the free-drift set is invariant under drift, so every step inside it is
  feasible; the braking filter accepts a step only when it ends in the braking set, or no
  further out of it when it starts outside, and refuses an explicit `a_b` above
  `braking_deceleration`; a backup step that leaves the set from inside is "solver_failed", and
  a "solver_failed" step carries no certificate);
  `SoftCbfFilter`, the exact-penalty form of the barrier rows, which never counts toward
  certification; and `FilterDeclaration`, what each filter enforces and assumes.
- `vbar.safety.audit`: `audit_filter`, which flies a certified safety filter around a nominal
  agent from starts drawn inside or outside its declared safe set, with one recorded root seed
  of 64 to 128 bits, and judges each step on the certified in-step bound, re-flying uncertain
  steps on the exact CW solution with up to 64 times the samples; it reports violations, uncertified
  steps, infeasible and failed decisions with the check of each infeasibility proof, and the
  time to re-enter the set, in a `FilterAudit` with a `passed` verdict. A filter that does not
  count toward certification (a soft filter, for one) is refused with its declared safe set
  named in the message; so are dynamics other than CW, imperfect navigation, execution error,
  actuation delay, a minimum impulse bit and the thrust drive.
- `vbar.reach`: zonotope and polytope sets, affine loops on the exact CW step built with the
  closed-loop matrix (`cw_loop`), exact support functions of reachable sets by backward
  propagation of the direction, certified keep-out clearance of a set of states in free drift
  (`drift_clearance`) and in a closed loop that is refused if it saturates (`loop_clearance`),
  each lower bound lowered by its rounding so that none sits above the exact margin, set
  propagation with optional order reduction, exact zonotope containment and an invariance
  test, minimum time under a thrust box with a separating-direction certificate
  (`min_time_steps`), and a bound on the differential gravity a CW model leaves out.
- `vbar.metrics.record_pair` and `vbar.runner.run_pairs`: one seed flown with a safety filter on
  and off, with the same episode and agent seeds, a new agent per run, the unfiltered run through
  the pass-through filter and the order drawn at random, returned as a `FilterPair`; and F8p
  `unnecessary_intervention_rate_paired`, the share of active decisions in episodes whose
  unfiltered run broke no hard constraint. `run_pairs` takes `count` as an integer of at least 1.
- `vbar.safety.InfeasibilityProof` and `FilterResult.proof`: an "infeasible_certified" step of
  `CbfFilter` carries the Farkas certificate of its rows under the thrust limit, each row in its
  own SI units, and
  `verify()` checks it from its own arrays; the step's action is the least-violation command.
- `vbar.safety.rta`: the four run time assurance filters of Dunlap et al. (2022), explicit and
  implicit switching and optimization, on the paper's Euler plant, with `AfrlSetup` (the paper's
  and the authors' code's speed-limit slopes), `AfrlConstraints` and the natural-motion backup
  `NmtBackup`. Implicit optimization uses the paper's eq. 11 with its bracket corrected. None of
  the four counts toward certification. `AfrlSetup.paper().view()` and
  `AfrlSetup.authors_code().view()` name the scenarios that fly them,
  `vbar:docking-afrl-paper:euler1:box:dv:v1` and its `:code` variant, where other setups use
  `dunlap2021-euler`, and the four filters accept the `euler1` dynamics part when the setup's dt
  is that rung's 1 s step.
- `vbar.stress.cross_entropy`, a seeded cross-entropy search over a box with a mean shift, a
  floored spread, elites kept across generations, independent chains, rejection of points
  outside an acceptance test, and a dispersion report; and `vbar.safety.audit.falsify_filter`,
  which searches a certified filter's declared safe set for the start whose run comes closest to
  failing and judges that run as the audit does, with the verdict "none_found" at best.
- `vbar.reach.drift_cover` and `DriftCover`: a box of start states split into parts whose free
  drift is certified clear of the keep-out, parts that provably enter it, and undecided parts,
  with counterexamples checked against a declared set.
- `vbar.safety.rta.reproduce`, `start_state` and `Reproduction`: the Dunlap et al. Euler docking
  run with the paper's LQR primary, stepping `EulerCWDynamics`, and
  `benchmarks/afrl_reproduction.py` over 100 seeded starts at 2.15 n and 4 n plus the authors'
  start cases. The script prints the scenario IDs and reports each of the paper's claims as met
  or not met; it exits 0 when the reproduction runs and fails only on a non-finite result. The
  AFRL backup gain is the position and velocity block of the LQR on the state augmented with
  integrators, as in the authors' design.
- `benchmarks/filter_audit.py --parts cover`: a sampling-free check of the free-drift filter's
  declared set by `drift_cover` over the audited run. It fails only on a counterexample; the
  share of drawn in-set starts that lie in certified boxes is reported, and marked when below
  99%, but never gated.
- `FilterResult.prep_s` and `feedback_s`: `run_filter` calls a filter's optional `prepare()` before
  `filter`, times both itself and sets `compute_s` to their sum, so a filter cannot report its own
  latency split.
- `vbar.safety.tightening`: `box_hull`, `tube_margin` (closed-loop tube with a bounded tail),
  `delay_margin` (open loop over a delay window), `per_draw_sigma`, `scenario_sample_count` and
  `scenario_margin`, the margins shared by the filters for execution and navigation errors.
- `vbar.safety.predictive.PredictiveFilter`: plans `horizon` held commands toward the agent's
  command and applies the first only when the plan is certified on the exact held steps and the
  free-drift terminal set; falls back to the shifted plan, then zero thrust, so inside its
  declared set it is never stuck; predicts through pending commands.
- `vbar.safety.corridor.CorridorCamFilter`: passes commands through until the state leaves the
  V-bar passive-safety corridor (and a declared approach cone) outside the commit radius, then
  latches a phase-specific open-loop CAM followed by zero thrust; it makes no certified claim and
  the audit refuses it. `vbar.safety.audit.cam_rates` gives the false-CAM, missed-CAM and
  CAM-induced rates of paired runs with 95% Clopper-Pearson intervals, read from the recorded
  commands.
- `vbar.safety.barriers.CamBarrier` and `BackupFilter(barrier="cam")` ("cbf-cam"): a backup
  filter whose set holds the states that a CAM held for up to m whole steps, then free drift,
  keeps outside the keep-out for all time; the backup is zero thrust where free drift is
  certified and the CAM elsewhere, the set is kept by it, and the certificate is "nominal"
  (error sets are refused). `vbar.metrics.passive.cam_direction` gives the unit CAM direction of
  a goal, which `cam_impulses` and the filter share. `CamBarrier` raises `ValueError` for `steps`
  or `samples` that is not a whole number in range, infinity and NaN included.
- `drift_cover` takes `accel` and `burn_time`: a box of starts carried through a held collision
  avoidance burn and then free drift, with a certified in-step bound during the burn.
- `benchmarks/filter_audit.py` audits, searches and covers the CAM and predictive filters too
  (the CAM set level by level, each level's burn then drift), and takes `--filters` to run a
  subset.
- `vbar.safety.CERTIFICATE_KINDS` ("nominal", "bounded-W", "chance-epsilon",
  "finite-horizon"): `FilterResult` and `FilterDeclaration` raise `ValueError` for a
  certificate outside it or for a kind and a horizon not given together, and `run_filter`
  checks the result again after the filter returns it. A free-drift backup filter whose declared
  error set has a positive exceedance certifies "chance-epsilon".
- `PredictiveFilter` modes "finite-horizon" and "soft" (exact-penalty slacks, never counted toward
  certification), the braking backup as the recovery outside the declared set, `backup="cam"`
  ("predictive-cam-n10"), and a check of the pending command's wait against `delay_s`.
- `CbfFilter` enforces `approach_cone` constraints with one `ConeEcbf` row each, named
  "approach_cone", "approach_cone#1", ...; with a cone its declaration says "cone activation
  untreated", does not count toward certification, and `audit_filter` refuses it.
- `vbar.safety.tightening.error_zonotope`: a zonotope centered at zero that contains a box,
  two-ball or ellipsoid error set, exact for the box.
- `BackupFilter(barrier="passive", navigation=...)` decides its set and the end of each held step
  on the free-drift clearance of the whole navigation error set (`vbar.reach.drift_clearance`)
  instead of the point estimate.
- `PredictiveFilter` takes `navigation`, `execution` and `grid`: declared error sets tighten its
  program and its checks, open loop, and certify "bounded-W" (or "chance-epsilon") up to two
  orbits.
- `vbar.safety.tightening.declared_errors`: the navigation box of a truncated `GaussianNav` and
  the execution bound da_max of a truncated `ExecutionError`, from the model parameters and the
  truncation level alone; `episode_draws` counts the Gaussian draws of an episode that the level
  covers, for `vbar.scenarios.per_draw_level`.
- `CbfFilter`, `SoftCbfFilter`, `BackupFilter` and `PredictiveFilter` take `exceedance` and
  `per_draw_level`, the truncation their error sets come from, and report them in
  `FilterDeclaration` fields of the same names (None by default); `contains_start(x)` tells
  whether every estimate within the navigation set of a true state lies in the declared set.
- `benchmarks/filter_audit_errors.py`: the filter audit, the falsification search and the
  Gaussian exceedance rates under truncated `:nav1` navigation and execution errors, one line
  per filter and part with p, the per-draw level and the draw count; the braking and cam filters
  make no certified claim under errors and are listed, never audited.
- `benchmarks/predictive_comparison.py`, comparing the predictive, barrier and free-drift filters
  on the filter metrics and on passes where the barrier filter's keep-out row is infeasible.
- The `euler1` dynamics rung, `vbar.dynamics.EulerCWDynamics`: Clohessy-Wiltshire
  stepped by forward Euler in 1 s steps, defined only at its nodes. Environments on it check
  constraints at the nodes (`vbar.constraints.node_minimum`, `node_violations`); `Trace`,
  `passive_metrics` and `evaluate` raise `ValueError` for its episodes. The plant refuses a mean
  motion that is not positive and finite, an infinite or NaN duration and a nonzero duration of
  no whole step, and an environment refuses a `dt` under one step.
- The `docking-afrl-paper` scenario (v1) and its `:code` variant, registered as
  `vbar/DockingAfrlPaper-v1` and `vbar/DockingAfrlPaperCode-v1`: the run time assurance docking
  benchmark as published, on the `euler1` rung.
- `audit_filter(..., errors=)`: "bounded" flies `GaussianNav` and `ExecutionError` models
  truncated at the filter's declared `per_draw_level` and refuses declared sets that do not hold
  those `declared_errors` derives; "gaussian" flies them untruncated, sets `passed` to None and
  reports `FilterAudit.exceedance_rates`, per covered constraint and "any", an `ExceedanceRate`
  with the count of violating runs, its 95% Clopper-Pearson interval and the declared level.
  `FilterAudit` reports `errors`, `exceedance`, `per_draw_level`, `draws` and `covered`, and
  `RunAudit` and `FilterAudit` report `uncovered`, the uncertified decisions that are not a coast
  inside the last certificate's horizon; under "bounded" one fails the audit. "none", the
  default, audits as before.
- `exceedance` and `per_draw_level` are given together and only with an error set, in a filter
  and in `FilterDeclaration`; otherwise they raise `ValueError`.
- `vbar.stress.ScriptedNav` and `vbar.stress.ScriptedExecution`: a navigation error held at a
  scripted offset in a declared `BoxError` for each quarter of an episode, and a constant burn
  error mapped radially from the cube onto a declared execution ball, for a search to steer. Both
  follow the interfaces the environment calls and draw nothing.
- `falsify_filter(..., errors="bounded")` searches 33 numbers, the start, a navigation offset per
  quarter episode and one burn error, under the truncated models the bounded audit accepts, each
  run flown with the scripted errors scaled by the filter's declared sets; it refuses
  "gaussian". Under "bounded" an uncovered decision gives the verdict "uncovered".
  `FilterFalsification` records `errors`.
- `vbar.spacecraft.ExecutionModel`, the protocol of a burn's execution error;
  `EnvConfig.execution` and `Spacecraft(execution=)` take any model that follows it.
- `LLMAgent` flies every task kind: hold, berthing, docking, inspection and departure. The default
  prompt describes the goal, the target's attitude mode and every public constraint in sentences
  (`vbar.agents.briefing.briefing`) and measures the observation against them each step
  (`situation`: offsets and speeds relative to a docking port's axis where the briefed mode has
  turned it, coverage so far, distance to the hold or clear point). New prompt fields `briefing`,
  `situation` and `thrust`; `target_mode` rebuilds a target mode from its public description.
- The LLM briefing describes the capture task (the body-fixed capture point, the tumble of a
  torque-free or gravity-gradient target, the capsule keep-out) and a departure's `drift_radius`.
- `python -m vbar.agents.mcp` and `python -m vbar.agents.remote` take `--scenario` (a catalogue
  name, variant or identifier) and write the scenario identifier into each result; the MCP
  `describe_task` tool and the `start_episode` reply return the briefing, and each observation the
  situation lines.
- `--seeds-file` for both servers: seeds to repeat, one per line, kept out of the process list.
- The supervisory command channel ([docs/supervision.md](docs/supervision.md)): `SupervisedAgent`
  turns a supervisor's CONTINUE, GO, HOLD, RETREAT, CAM and THRUST_INHIBIT into set points for a
  `SetpointTracker`, refusing and recording an order the validity mask does not allow, over the
  ladder of rungs `ladder` cuts from a hold, berthing, docking or inspection goal. Supervisors:
  `RuleSupervisor`, `LLMSupervisor` (with `parse_order` and `SUPERVISOR_PROMPT`) and
  `PolicySupervisor` on `supervisor_features`. `vbar.training.SupervisorEnv` lets a learner pick
  the commands. `SetpointTracker` flies a Kalman estimate on a noisy sensor (`sensor`,
  `estimate`, `sensor_noise`) and holds the docking standoff until the lateral error is small
  (`lateral_sigma`).
- `vbar.games`: KSPDG's pe1 starts I1, I2 and I4 from its mission saves beside I3, and the scenario
  classes `PE1_E1_I1_Env` to `PE1_E1_I4_Env` and `PE1_E3_I1_Env` to `PE1_E3_I4_Env` with the
  parents `PE1_E1_ParentEnv` and `PE1_E3_ParentEnv` (the e3 classes take KSPDG's `evade_dist` and
  `evade_throttle`), registered as `vbar/PursuitEvasionE1I1-v1` to `vbar/PursuitEvasionE3I4-v1`.
- `vbar.games.RandomEvader`, V-BAR's version of KSPDG's e2 evader (random throttles held for random
  times inside the control range, drawn from the episode seed), and its scenarios `PE1_E2_I1_Env`
  to `PE1_E2_I4_Env` (`PE1_E2_ParentEnv`), registered as `vbar/PursuitEvasionE2I1-v1` to
  `vbar/PursuitEvasionE2I4-v1`.
- `vbar.games.ProgradeEvader`, V-BAR's version of KSPDG's e4 evader (a fifth of its forward thrust
  prograde from 0.5 s on), and its scenarios `PE1_E4_I1_Env` to `PE1_E4_I4_Env`
  (`PE1_E4_ParentEnv`), registered as `vbar/PursuitEvasionE4I1-v1` to `vbar/PursuitEvasionE4I4-v1`.
- `vbar.games.pursuers.ClosestApproachAgent` and `closest_approach`: a reimplementation of Jason
  Cheng's deterministic pe1 agent (github.com/ARCLab-MIT/kspdg) on V-BAR's Kepler propagation,
  with the original's predictor signature.
- `vbar.games.compat`: KSPDG's module paths and names (`kspdg.agent_api.base_agent`,
  `kspdg.pe1.e3_envs`, `kspdg.PE1_E3_I3_V1` and the rest of pe1) for V-BAR's classes once
  `install()` runs, with `NotImplementedError` and the reason for lbg1, sb1 and LG3 to LG7, and
  `python -m vbar.games.compat script.py` to run a KSPDG agent file as it is.

### Changed

- A safety filter's `filter(obs, action)` must return a `FilterResult`; `run_filter`,
  `FilteredAgent` and `vbar.training.FilteredEnv` raise `TypeError` for a bare action, and the
  two wrappers no longer take `tol` or decide `intervened` from the change in the command. Wrap
  the action in `FilterResult(action, intervened)` and drop `tol`. `run_filter` takes
  `(safety_filter, obs, command)` and sets the result's `compute_s` to the measured wall time,
  whatever the filter reported, so F6 and F6p are version 2. `PassThroughFilter.filter` returns
  `FilterResult(action, intervened=False)`.
- `run_filter`, `FilteredAgent` and `FilteredEnv` check a filter's result inside the timed region:
  every field must have its exact type (`TypeError` otherwise; a numpy scalar, a subclass, or an
  int where a float is expected, such as `slack=0`, are refused), and the action must be numeric,
  finite and of the command's shape (`ValueError` otherwise). The result is rebuilt from those
  plain values, so the recorded action is a copy the filter cannot write to later. Return plain
  `bool`, `str`, `float` and `int` values and float arrays, for example `slack=0.0`.
- A constraint's `second_order_bounds(states, h, speed, accel, jerk=None)` takes a bound on the
  jerk; the speed, closing-speed and gate-table limits define it, so their refinement nearly
  stops. `min_margin_bound` (S1b) is version 3 for these bounds, and the golden metric fixtures
  were regenerated.
- F3 `safety_dv_cost` and F4 `filter_dependence` are version 2: `paired_filter_metrics` raises
  `ValueError` for a pair whose agent seeds or scenarios differ, which version 1 accepted.
- The default LLM prompt is new, so transcripts recorded with 0.3 do not replay under it; pass
  0.3's `PROMPT_TEMPLATE` text as `prompt=` to replay them. Positions are printed to 1 cm and
  velocities to 0.1 mm/s, as docking's 0.1 m lateral limit needs.
- `LLMAgent.reset` no longer refuses goals that are not holds or that have no keep-out radius; it
  refuses a goal without a key its task needs, and a template whose `hold`, `dist`, `cap_speed`
  or `keep_out` field the goal cannot fill.
- `vbar.games.EvaderBot` is one method, `segment(t, r_p, v_p, r_e, v_e, previous, cause, rng)`,
  returning a `Segment`, in place of `mode`, `force` and `switch`; the game applies the vessel's
  thrust limits and per-axis propellant flow to the evader as to the pursuer.
  `NormalEscapeEvader(thrust=...)` is now `throttle`, in [0, 1] (1 is the former 8000 N).
- The result lines of the test split (`core-test-v1`) hold the keyed digest `seed_digest` of each
  seed instead of the seed, matched to their runs by `Suite.missing` and `match_runs`. A line that
  would hold a test seed or the hidden key is refused, and `VBAR_HIDDEN_SEED_LOG` names the private
  file the seeds are appended to instead.
- Every result line carries `seed_digest` and `agent_seed_digest`, so `read_results` refuses a result
  file written by 0.4.0, which lacks them; the suite has to be flown again.
- The provenance hash of a run covers a digest of local edits to tracked source files, so a resumed
  run is not served results of other edits. `vbar.eval.provenance.source_revision` returns
  (commit, dirty, digest) where it returned (commit, dirty), and `environment()` gains the key
  `diff`.
- `vbar.eval` refuses a result line with a missing field or a bad seed, and a job with a bad seed
  or index.
- `vbar.eval.stats` and `vbar.eval.tails` refuse NaN scores, fractional counts and probabilities
  outside their range; the studies of `vbar.eval.coverage` refuse impossible sizes and
  probabilities before drawing; the learned-agent entries of `vbar.eval.learned` need whole-number
  counts and seeds.
- The Basilisk and Orekit propagators of `vbar.dynamics.backends` refuse malformed states, sample
  times, tolerances and overlapping burns before the extra is imported.
- `Episode.reset_s` also appears in the result lines of the agent host
  (`python -m vbar.agents.remote`).
- The MATLAB PD client `clients/matlab/pd_agent.m` raises the error `vbar:pd_agent` at reset for a
  goal whose task is not a hold or a berthing, or that has no hold point, as the Python agents do.

### Fixed

- `DualBank.build` raised `RuntimeError` when HiGHS ended one window LP with no verdict (model
  status 15, scipy status 4), as on window 301 of approach-vbar box validation start 19 of 100, so
  `benchmarks/lower_bound_check.py` left the whole scenario unscored. Every fuel-oracle LP so ended
  is now solved again by the interior point method and then by dual simplex with presolve
  switched, which decides that window; a start still undecided adds no multipliers and is listed
  in the new `DualBank.uncertified`, the other starts keep their bounds, and the check holds that
  start's runs to no bound and logs the count and indices of uncertified starts.
- `LLMAgent` read a reply's action with a pattern that stopped at the first closing brace, so an
  action object holding a nested value was a parse failure, and the search took time quadratic in
  the reply's opening braces. `parse_action` now decodes the first object that opens at one of the
  reply's first eight braces (`vbar.agents.llm.first_object`), nested values included; a brace in
  prose or a malformed object before the action is skipped, so parse failures (A4) can be fewer
  than an earlier version counted on the same replies.
- `ToolServer`'s `describe_task` and predictions used the view of the environment as built, not of
  the episode being flown, so on scenarios that draw their goal at each reset (a docking port's
  attitude, the Sun direction) a client was told another episode's goal. They now use the view of
  the episode running, as `run_episode` does; before the first episode `describe_task` gives the
  environment's view as built.
- A negative seed crashed `python -m vbar.agents.mcp` and `python -m vbar.agents.remote` before
  any result was written; both now refuse it with a usage error. The remote host also refuses a
  timeout that is not a positive number of seconds and a port outside 0 to 65535, and the MCP
  server a step limit below 1.
- The agent server and the MCP `command` tool flew quoted numbers and booleans as action numbers
  (`["0.5", "0", "0"]`, `true`), though the protocol and the tool schema ask for numbers; both now
  refuse them.
- `python -m vbar.agents.mcp` decoded its input in the locale's code page, so on Windows a non-ASCII
  request (an id, a name) sent as UTF-8 came back garbled; it now reads UTF-8 whatever the locale,
  and a line that is not UTF-8 gets a parse error.
- `AgentServer.accept` served the first connection only, so one that never said hello held the host
  until its timeout and the session ended with every seed unflown; it now drops a connection that
  sends no valid hello within 10 s (`HELLO_WAIT_S`) and waits for the next.
- `view_to_dict` left the lighting window's central body as a `Body`, so the agent server and the
  MCP server could not send the view of a scenario with a lighting window (`docking:lit`); it now
  writes dataclasses as the mapping of their fields.
- A `ZoneEntry` whose `go_time` fell inside a step, as with 15 s steps in the closing scenario,
  had a `lower_bound` above the true minimum (-0.053 m against -0.400 m with GO 1.4 s into a
  10 s step) and a violation that ended at the last sample before GO. `segment_minimum`,
  `segment_minima` and `violation_intervals` now check such a segment up to GO, resampled there.
- `violation_intervals` and the STL metrics raised `ValueError` when a sample on the boundary
  rounded to the other side of zero once evaluated again; the end nearer zero is now taken as the
  crossing.
- `CWDiscretizer` rounded every duration to the nanosecond, so a step of 5e-10 s or less did not
  move the state and 3e5 steps of 1/3 s drifted 1.2 mm; it now keys its cache to the exact
  duration.
- On a coarse interval with two dips the refined minimum could settle in the shallower one, so a
  violation went unreported while `lower_bound` stayed negative. Such intervals are now cut up and
  searched until their bound rules a violation out, one is found, or 16 cuts are spent, and
  `violation_intervals` searches the reported minimum's time as a sample.
- `segment_minimum` could report a `KeepOut` `lower_bound` above the exact minimum where that
  minimum fell at a sample, by up to 8e-14 m on a radial hop, from rounding in the sample's
  margin. The bound is now lowered by 3 eps times the magnitudes the margin is computed from;
  `segment_minima` and the speed limits added in this release do the same.
- `project` without an actuator set accepted rows to 1e-10 of |u|_inf, so a nominal command of
  1e10 returned u_x = 0 against u_x >= 1 as feasible; rows now hold to 1e-10 of the problem's
  scale plus the rounding of each row's value. Its certificate without an actuator set accepted
  A^T lam up to 1e-9 |lam|_1 and so declared rows such as x >= 1, -x + 1e-9 y >= -0.5 infeasible;
  it is now checked in exact rational arithmetic. Nearly parallel active rows gave a wrong
  optimum, 60% above the true objective for rows 1.4e-7 off parallel, and rows within 1e-8 of
  parallel gave "solver_failed" or a false certificate; each subset is now solved through a QR
  factorization of its rows instead of the KKT matrix. Phase I ran in raw units against HiGHS's
  absolute tolerances, so a 1% conflict in units of 1e-5 gave "solver_failed"; it now runs in
  units of the problem's scale.
- `solve_horizon` ended 14 of 300 feasible CW proximity problems in "solver_failed": one target
  for every row drove lam / s of strongly active rows to 1e21 while a weakly active row lagged,
  and the Riccati sweep broke down. Rows already 1e-3 tol_gap inside the complementarity test
  are now held there; 0 of the 300 fail. An inactive row's multiplier was judged against terms
  that include the dynamics multipliers, so a strongly active row hid a wrong one (1.34 where the
  exact value is 0, the plan 1.2e-4 off); it is now judged against the cost terms. A violated
  row started with slack 1 in its own units, so the solve depended on the rows' units; it now
  starts with its violation.
- The coast and transfer tools refuse arguments that are not numbers: a quoted number or a boolean
  (`"seconds": "300"`, `"seconds": true`) comes back to the model as an error instead of being
  read as a number.
- `PDAgent` raises `ValueError` for a `wn` that is not positive and finite or a `zeta` that is
  negative or not finite, gains that used to give NaN commands.
- `vbar.games.NormalEscapeEvader` acts only while the pursuer is inside KSPDG's 2200 m evader
  control range (`control_range`, `CONTROL_RANGE`) and keeps its last command outside it, as
  KSPDG's e3 does; it burned at any range below its evasion distance. Episodes with the default
  200 m evasion distance are unchanged.
- `vbar.games.PursuitEvasionEnv` captures a fast pass whose range dips under the capture distance
  and out again within one integrator step; it was missed, and the episode flew on.
- `vbar.metrics.clopper_pearson` raised `OverflowError` for an infinite count and an unclear
  `ValueError` for NaN, and `unsafe_tail` took a `level` outside [0, 1] when every value was NaN
  or there was none; both now raise `ValueError` with a clear message.

## [0.4.0] - 2026-10-30

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
- Mission reconstructions from public reports, `recon-progress-m34`, `recon-ets7` and `recon-dart`
  (Gymnasium `vbar/ReconProgressM34-v0`, `vbar/ReconEts7-v0`, `vbar/ReconDart-v0`), each with a
  `:nominal` variant without its failure; `docs/reconstructions.md` documents them.
  `vbar.scenarios.reconstructions.AimedApproachEnv` and `AimedApproachParams`: a hold-point approach
  that starts on a coasting arc aimed at a point near the target at a drawn range and closing rate.
- `vbar.faults` and `EnvConfig.faults`: faults injected mid-episode, scheduled (`Window`),
  triggered by the true trajectory (`After`, `Until` with `RangeBelow` or `InSphere`),
  intermittent, or combined with `AllOf`; effects `ThrusterOff`, `SensorBias` and `RangeLoss`; `info["faults_active"]`
  names the faults acting. Each fault draws from its own stream. `docs/faults.md` documents them.
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
- `vbar.dynamics.rigid_body`: `principal_axes` with a right-handed axis set, and
  `TorqueFreeMotion`, the closed-form attitude and body rate of a rigid body free of torques
  (Jacobi elliptic rates, the precession angle through Carlson's integrals), built from a rate or
  from tumble parameters, with rate and acceleration bounds.
- `TorqueFree`, a target attitude mode for a rigid body tumbling free of torques, set by tumble
  parameters, its attitude stored inertially and seen from the turning Hill frame; its public
  description rebuilds the tumble.
- `vbar.dynamics.rigid_body.GravityGradientMotion` and the target mode `GravityGradientTumble`: a
  tumble under the gravity-gradient torque of a circular orbit, integrated numerically from the
  briefed state, with rate bounds from the orbit-frame Jacobi integral; `rate_in_n` sets its rate
  as a multiple of the mean motion.
- The preset `tumbling-capture:libration` (`vbar/TumblingCaptureLibration-v1`): a dead stage held
  by the gravity gradient with its long axis radial, librating about the orbit normal, its spin
  drawn in units of the episode's mean motion.
- The scenario kind `CaptureEnv` (`vbar.scenarios.tumbling`), with the turning capsule keep-out
  `RotatingCapsule`: reach a capture-ready state at a capture point fixed to a turning target,
  within a ball and a speed relative to the moving point, and hold it for a dwell.
- The flagship scenario `tumbling-capture` (`vbar/TumblingCapture-v1`) and its tumble presets
  `:coning`, `:separatrix` and `:intermediate`: a tumbling upper stage, its tumble drawn per
  episode and briefed at t = 0.
- Capture goals in the mission metrics: `goal_state`, `goal_errors`, `captured` and `hold_signal`
  take the time, the target mode and the mean motion, `goal_bounds` gives the goal point's speed
  and acceleration bounds, and M1 to M5 and M9 (M9 now v2) follow the moving capture point.
- The scenario `departure` (`vbar/Departure-v1`): leave the -V-bar port of an ISS-class target
  through a 10 deg departure cone onto a path that does not come back (Fehse, section 2.6).

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
- The `TargetMode` protocol has `angular_acceleration_bound(n)`, a bound on the rate of change of
  `angular_velocity` (rad/s^2) that certifies predicates on points moving with the target.
- The failure label T1 (version 2) has `returning_drift`, for a departure that reached its clear
  range only on free drifts that come back to the target, labelled a timeout or `other` before.

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
- The mission metrics took a departure goal, which names the port it leaves from, for a docking
  goal: M1 and M4 waited for a contact that never comes, and a crossing of the port plane raised
  `KeyError`. A docking goal is now one with a contact window, and a departure goal (`task`
  "depart") carries `drift_radius` and is met as the departure judges it: at the first step
  boundary beyond `clear_range` from which the free drift over `drift_horizon` stays outside
  `drift_radius`.
- F10 `stall_time` (version 2) counted a whole step for a decision whose step a contact cut short,
  so a slow docking with an active safety row read up to one step too long.

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

[Unreleased]: https://github.com/jamilhammoudeh/vbar/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/jamilhammoudeh/vbar/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jamilhammoudeh/vbar/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/jamilhammoudeh/vbar/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/jamilhammoudeh/vbar/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jamilhammoudeh/vbar/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jamilhammoudeh/vbar/releases/tag/v0.1.0
