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
| [environment.md](environment.md) | the approach scenario: actions, observations, start states, keep-out checking, metrics, reward, agents |
| [validation.md](validation.md) | what the tests check, the published examples they reproduce, and the sources of the methods |
