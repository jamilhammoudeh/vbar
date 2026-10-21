# V-BAR documentation

How each part of V-BAR is built: the equations, the numerical methods, the assumptions and limits,
and how it is tested.

| Page | Covers |
|---|---|
| [frames.md](frames.md) | units, the Hill frame, relative states, the ISS-style LVLH frame, rotation conventions |
| [dynamics.md](dynamics.md) | the model interface and the list of models, Clohessy-Wiltshire, nonlinear two-body motion with J2, Kepler propagation, J2 mean elements, central bodies |
| [perturbations.md](perturbations.md) | Harris-Priester density and geodetic height, differential drag, cannonball and flat-plate solar pressure, cylindrical and conical Earth shadow, the forces left out |
| [relative-motion.md](relative-motion.md) | elliptic targets in the two-body model, Yamanaka-Ankersen, curvilinear Clohessy-Wiltshire, linearized two-body motion with J2, their error scalings |
| [transition-matrices.md](transition-matrices.md) | transition and input matrices from the variational equations, STM adequacy, the modeling-error index |
| [three-body.md](three-body.md) | the CR3BP and its constant sets, the libration-point, periodic linear and bicircular models, Richardson seeds, correctors, continuation and Floquet analysis |
| [spacecraft.md](spacecraft.md) | thruster layouts and limits, the minimum impulse bit, mass and the rocket equation, constant-acceleration and constant-thrust drives, actuation delay and segments, execution error |
| [sensors.md](sensors.md) | perfect navigation, white noise with an episode bias (`:nav1`), range-scaled errors in the sensor frame, time-correlated Gauss-Markov errors (`:nav2`) |
| [constraints.md](constraints.md) | constraint margins, the keep-out sphere, speed and closing-speed limits, range-rate gate tables, the approach cone, Sun exclusion, passive-speed envelopes and the approach-zone ellipsoid, sampling and refinement inside a step, the Lipschitz certificate and its assumptions, violation intervals |
| [environment.md](environment.md) | the base environment and the approach scenario: actions, observation channels, random streams, propagation, keep-out checking, what agents and metrics see, reward, agents, configuration |
| [scenarios.md](scenarios.md) | scenario specs and dispersions, identifiers, seed splits, suites, tags and plugins, target attitude modes, the hold, docking, inspection, berthing and departure kinds, the v1 catalogue with its sources and audit |
| [metrics.md](metrics.md) | recording an episode, the trajectory between samples, signals and their violation intervals in continuous time, mission metrics with contact and hold, efficiency and safety metrics, passive safety and aborts, thruster plume, the optimality gap, safety-filter and agent metrics, the composite score, failure labels |
| [rewards.md](rewards.md) | training rewards: per-second discounting, terms aligned with the metrics, potential-based progress shaping, the integrated quadratic cost |
| [training.md](training.md) | interfaces for reinforcement learning: the discrete bang-off-bang action mode, burn-and-coast decisions, training-only starts, a safety filter in the training loop |
| [vector-env.md](vector-env.md) | the vectorized approach environment: batched propulsion, propagation and keep-out checks, same-step autoreset, supported options, speed |
| [agents.md](agents.md) | tools around agents: the sign-convention check, real-time runs, plan submission with an external verifier |
| [llm-agents.md](llm-agents.md) | LLM agents: the model function, the prompt and its fields, parsing, tool calling and the built-in tools, tokens and cost, reasoning faithfulness, transcript replay |
| [remote-agents.md](remote-agents.md) | agents in another process or language: the agent server, its JSON-lines protocol, the Python and MATLAB clients, the tool server for LLM harnesses |
| [policy-export.md](policy-export.md) | neural policies with normalization and squashing, ONNX export, the ONNX agent, latency profiles |
| [validation.md](validation.md) | what the tests check, the published examples they reproduce, and the sources of the methods |
