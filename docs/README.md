# V-BAR documentation

How each part of V-BAR is built: the equations, the numerical methods, the assumptions and limits,
and how it is tested.

| Page | Covers |
|---|---|
| [frames.md](frames.md) | units, the Hill frame, relative states, the ISS-style LVLH frame, rotation conventions |
| [dynamics.md](dynamics.md) | the model interface and the list of models, Clohessy-Wiltshire, nonlinear two-body motion with J2, the relative acceleration without thrust, Kepler propagation, J2 mean elements, central bodies |
| [perturbations.md](perturbations.md) | Harris-Priester density and geodetic height, differential drag, cannonball and flat-plate solar pressure, cylindrical and conical Earth shadow, the forces left out |
| [flight-data.md](flight-data.md) | GRACE-FO orbits of 3 December 2019 replayed through the two-body, J2 and drag rungs: the data and its licence, the Earth-fixed to inertial rotation, the error bound stated before the run, the result and the K-band check |
| [relative-motion.md](relative-motion.md) | elliptic targets in the two-body model, Yamanaka-Ankersen, curvilinear Clohessy-Wiltshire, linearized two-body motion with J2, their error scalings |
| [transition-matrices.md](transition-matrices.md) | transition and input matrices from the variational equations, STM adequacy, the modeling-error index |
| [three-body.md](three-body.md) | the CR3BP and its constant sets, the libration-point, periodic linear and bicircular models, Richardson seeds, correctors, continuation and Floquet analysis |
| [rigid-body.md](rigid-body.md) | principal axes, torque-free attitude in closed form (Jacobi elliptic rates, the precession angle as a third-kind integral), tumble parameters, rate bounds, the gravity-gradient torque and its Jacobi integral |
| [spacecraft.md](spacecraft.md) | thruster layouts and limits, the minimum impulse bit, mass and the rocket equation, constant-acceleration and constant-thrust drives, actuation delay and segments, execution error |
| [sensors.md](sensors.md) | perfect navigation, white noise with an episode bias (`:nav1`), range-scaled errors in the sensor frame, time-correlated Gauss-Markov errors (`:nav2`) |
| [faults.md](faults.md) | fault injection: scheduled, triggered and intermittent activations, failed thrusters, sensor bias, lost range, their streams and what agents see |
| [constraints.md](constraints.md) | constraint margins, the keep-out sphere, speed and closing-speed limits, range-rate gate tables, the approach cone, Sun exclusion, passive-speed envelopes and the approach-zone ellipsoid, sampling and refinement inside a step, the Lipschitz certificate and its assumptions, violation intervals |
| [environment.md](environment.md) | the base environment and the approach scenario: actions, observation channels, random streams, propagation, keep-out checking, what agents and metrics see, reward, agents, configuration |
| [scenarios.md](scenarios.md) | scenario specs and dispersions, identifiers, seed splits, suites, tags and plugins, target attitude modes, the hold, docking, inspection, berthing and departure kinds, the v1 catalogue with its sources and audit |
| [reconstructions.md](reconstructions.md) | Progress M-34, ETS-VII and DART rebuilt from public reports: the facts and their sources, the dispersions and assumptions, the faults, what reference controllers do on them |
| [metrics.md](metrics.md) | recording an episode, the trajectory between samples, signals and their violation intervals in continuous time, mission metrics with contact and hold, efficiency and safety metrics, passive safety and aborts, thruster plume, the optimality gap, safety-filter and agent metrics, the composite score, failure labels |
| [rewards.md](rewards.md) | training rewards: per-second discounting, terms aligned with the metrics, potential-based progress shaping, the integrated quadratic cost |
| [training.md](training.md) | interfaces for reinforcement learning: the discrete bang-off-bang action mode, burn-and-coast decisions, training-only starts, a safety filter in the training loop |
| [vector-env.md](vector-env.md) | the vectorized approach environment: batched propulsion, propagation and keep-out checks, same-step autoreset, supported options, speed |
| [agents.md](agents.md) | tools around agents: the sign-convention check, real-time runs, plan submission with an external verifier |
| [llm-agents.md](llm-agents.md) | LLM agents: the model function, the prompt and its fields, parsing, tool calling and the built-in tools, tokens and cost, reasoning faithfulness, transcript replay |
| [remote-agents.md](remote-agents.md) | agents in another process or language: the agent server, its JSON-lines protocol, the Python and MATLAB clients, the tool server for LLM harnesses |
| [policy-export.md](policy-export.md) | neural policies with normalization and squashing, ONNX export, the ONNX agent, latency profiles |
| [solvers.md](solvers.md) | nondimensional Hill-frame units, the exact projection of a command onto linear rows under a box or ball thrust limit, its multipliers, the phase I certificate of infeasibility, stage-structured horizon problems and their Riccati solve, the primal-dual interior-point method and its statuses |
| [viz.md](viz.md) | the screen style and its colours, the Hill-frame view, range and speed panels, the approach animation, label placement, regenerating the README figures |
| [replay.md](replay.md) | replays of a recorded episode for the Rerun viewer: the file's entities and units, the violation and filter events, the metric card over time, limits |
| [backends.md](backends.md) | Basilisk and Orekit as independent propagators of the two-body plus J2 rung, burns held in the Hill frame, the agreement table and how its residuals are attributed |
| [pursuit-evasion.md](pursuit-evasion.md) | KSPDG's pe1 pursuit-evasion game on two-body dynamics: the problem and score, KSPDG's observation and action formats, frames and RCS model, time and event handling, the agent interface that runs KSPDG agents unchanged, V-BAR's evader bots, parallel runs, speed, limits |
| [evaluation.md](evaluation.md) | the evaluation harness: the suite runner, result files, the provenance hash, IQM and stratified bootstrap statistics, safety tails with order-statistic intervals and tolerance bounds, rate comparisons, learned-agent rules, reports and one-command reproduction |
| [lqr.md](lqr.md) | the LQR baseline: the exact zero-order-hold design model, discrete and sampled-data stage costs, the hold-point input, actuation delay and the command-change weight |
| [lqg.md](lqg.md) | the LQG baseline: the steady-state Kalman filter on the Clohessy-Wiltshire model, declared measurement biases and their detectability, the regulator on the estimate |
| [mpc.md](mpc.md) | MPC: the hold-point MPC agent with warm starts, side choice and fallback levels, the horizon QP on the zero-order-hold CW model with keep-out rows sampled within each step and an exact slack penalty, its solver statuses, and the terminal set by the Gilbert and Tan iteration with its exact-arithmetic invariance certificate to a 1e-9 relative margin |
| [nonlinear-mpc.md](nonlinear-mpc.md) | nonlinear MPC: multiple shooting on a declared dynamics model, the shooting linearization, the real-time iteration split, the horizon QP through the solvers and its fallbacks |
| [mpc-fuel.md](mpc-fuel.md) | fuel-optimal MPC: the least 1-norm delta-v plan to rest at the hold point on the zero-order-hold CW model, the strictly convex tie-break and the certified fuel excess bound |
| [piecewise-affine-mpc.md](piecewise-affine-mpc.md) | piecewise-affine MPC: the box-constrained regulator solved at reset as a multiparametric quadratic program, its table of polyhedral regions with affine laws, and the lookup |
| [clf-cbf.md](clf-cbf.md) | CLF-CBF QP controller: a Lyapunov decrease row at the LQR's guaranteed rate, a keep-out barrier row with an in-step certificate, the dualized relaxation solved by exact projections, and its fallbacks |
| [lyapunov.md](lyapunov.md) | Lyapunov baselines: Clohessy-Wiltshire reference tracking, the Cartesian law on a declared two-body model |
| [two-impulse.md](two-impulse.md) | two-impulse Clohessy-Wiltshire targeting, its singular transfer times, the two-impulse agent and the two-impulse fuel bound |
| [glideslope.md](glideslope.md) | the pulsed glideslope approach: the exponential range profile, its pulses and the agent that flies them |
| [trajectory-elements.md](trajectory-elements.md) | transfers, fly-arounds, straight-line forced motion and the twice-orbital-rate circle, and the agents that fly them |
| [shuttle-baselines.md](shuttle-baselines.md) | Space Shuttle practice: NSR and football aim points, elevation-timed burns, the pilot pulse model |
| [lambert.md](lambert.md) | Lambert's problem by universal variables and the Lambert targeting agent |
| [mission-manager.md](mission-manager.md) | modes, gates, contingency and abort modes, and the mission manager agent |
| [mutants.md](mutants.md) | mutant agents: deliberately broken controllers that check which metric catches which defect |
| [oracles.md](oracles.md) | fuel oracles: the primer vector and the finite-thrust dual lower bound, the fuel oracle over a fixed window or a free final time with its dual bank, and the exact constrained LQ value |
| [oracle-certificates.md](oracle-certificates.md) | primer-vector certificates: Lawden's conditions checked over the whole transfer for box thrusters and one engine, the certified primer peak and impulsive lower bound, the indirect finite-thrust optimum by the switching control, and the Hamiltonian along it |
| [minimum-time-oracle.md](minimum-time-oracle.md) | minimum-time oracle: the fewest held steps into the capture ball for box thrusters, a verified plan, and a primer-vector proof that each smaller step is out of reach |
| [passing-side-oracle.md](passing-side-oracle.md) | passing-side oracle: fuel bounds when the target lies in the way, keep-out halfspaces per side of approach, plans verified against the keep-out sphere between samples, and the side-free lower bound |
| [expert-datasets.md](expert-datasets.md) | the oracle expert agent and expert trajectory datasets: open-loop flight of the verified fuel plan, NPZ files with a dataset card, byte-for-byte regeneration from validation seeds, and the generation script |
| [reproductions.md](reproductions.md) | published results rebuilt from their papers: Ariba et al.'s LP glideslope and Hablani's standard glideslope on their V-bar case, the axis mapping, what matches and what does not |
| [safety-filters.md](safety-filters.md) | closed-form projections, the exact projection out of a keep-out sphere |
| [validation.md](validation.md) | what the tests check, the published examples they reproduce, and the sources of the methods |
