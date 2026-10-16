# Changelog

Notable changes to V-BAR. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0, a minor
release may change the public API, and every such change is listed under Changed or Removed.

## [Unreleased]

### Added

- `vbar/Approach-v0` registered with Gymnasium on import, so `gymnasium.make` builds the approach
  scenario; it passes Gymnasium's environment checker.
- `py.typed` marker, so type checkers use the package's annotations.
- A test that runs every python example in README.md and `docs/`, coverage reporting in CI, a
  throughput benchmark (`benchmarks/steps.py`) and weekly mutation testing of the core modules.
- SECURITY.md.

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
