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
| [constraints.md](constraints.md) | constraint margins, the keep-out sphere, sampling and refinement inside a step, the Lipschitz certificate and its assumptions |
| [environment.md](environment.md) | the base environment and the approach scenario: actions, observation channels, random streams, propagation, keep-out checking, what agents and metrics see, reward, agents, configuration |
| [metrics.md](metrics.md) | recording an episode, the trajectory between samples, signals and their violation intervals in continuous time, mission metrics with contact and hold, efficiency and safety metrics, passive safety and aborts, thruster plume, the optimality gap, safety-filter and agent metrics, the composite score, failure labels |
| [training.md](training.md) | interfaces for reinforcement learning: the discrete bang-off-bang action mode |
| [training.md](training.md) | interfaces for reinforcement learning: the discrete bang-off-bang action mode, burn-and-coast decisions |
| [training.md](training.md) | interfaces for reinforcement learning: the discrete bang-off-bang action mode, burn-and-coast decisions, training-only starts |
| [agents.md](agents.md) | tools around agents: the sign-convention check |
| [vector-env.md](vector-env.md) | the vectorized approach environment: batched propulsion, propagation and keep-out checks, same-step autoreset, supported options, speed |
| [training.md](training.md) | interfaces for reinforcement learning: the discrete bang-off-bang action mode, burn-and-coast decisions, training-only starts, a safety filter in the training loop |
| [rewards.md](rewards.md) | training rewards: per-second discounting, terms aligned with the metrics, potential-based progress shaping, the integrated quadratic cost |
| [validation.md](validation.md) | what the tests check, the published examples they reproduce, and the sources of the methods |
