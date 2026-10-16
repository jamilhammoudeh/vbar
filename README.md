# V-BAR: Verified Benchmark for Autonomous Rendezvous

[![ci](https://github.com/jamilhammoudeh/vbar/actions/workflows/ci.yml/badge.svg)](https://github.com/jamilhammoudeh/vbar/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vbar)](https://pypi.org/project/vbar/)
[![Python](https://img.shields.io/pypi/pyversions/vbar)](https://pypi.org/project/vbar/)

V-BAR is a headless Python benchmark for spacecraft rendezvous and proximity-operations
controllers. Classical guidance laws, reinforcement learning agents and LLM agents fly the same
scenarios and are scored with the same metrics.

**Status: early development (0.1).** The dynamics layer is validated (see
[Tests](#tests)). The environment has one scenario, the safety-filter interface has
only its pass-through ("filter off") implementation, and interfaces may change before 1.0.

## Why

KSPDG, the main existing environment for this kind of work, runs on Kerbal Space Program in real
time and needs a copy of the game. V-BAR is a plain Python package that runs much faster than
real time on Linux, macOS and Windows, so experiments with hundreds of seeds and many controllers
are practical. Its physics is checked against closed-form solutions and published textbook
examples, so a result reflects the controller and not a simulator error.

## Install

```bash
pip install vbar
```

Python 3.12 or later. The core dependencies are numpy, scipy and gymnasium. Plotting needs the
`plot` extra: `pip install "vbar[plot]"`. To work on V-BAR itself, see
[CONTRIBUTING.md](https://github.com/jamilhammoudeh/vbar/blob/main/CONTRIBUTING.md).

## Quickstart

```python
from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent

env = ApproachEnv(ApproachConfig(start_state=(0, -400, 0, 0, 0, 0)))
result = run_episode(env, PDAgent())
print(result["captured"], result["keep_out_violated"], result["dv_used_mps"])
```

This flies the PD baseline from 400 m behind the target to the hold point 100 m behind it, about
620 s of flight, and prints `True False 2.41...`: it arrived, it never entered the keep-out zone,
and it used 2.4 m/s of delta-v.

Benchmarks need many runs, not one. With the default random starts, which put the chaser 200 to
600 m ahead of the target so the target sits between it and the hold point:

```python
import numpy as np
from vbar import ApproachEnv, run_many
from vbar.agents import PDAgent

results = run_many(ApproachEnv(), PDAgent(), seeds=range(100))
print(np.mean([r["captured"] for r in results]), np.mean([r["keep_out_violated"] for r in results]))
```

The PD baseline reaches the hold point in every run and flies through the keep-out zone in every
run, because it ignores the target. That gap is what safety filters and better controllers are
measured against. In a clone of the repository, `python examples/v0_demo.py` shows both cases
and saves a plot to `outputs/` (needs the `plot` extra).

## Frame and units

Everything is in the target's Hill frame (also called RIC, RSW or RTN), centred on the target and
rotating with its orbit:

- x: radial, pointing away from Earth (the R-bar)
- y: along-track, the direction the target is moving (the V-bar); "behind the target" is -y
- z: orbit normal, completing a right-handed frame

States are `[x, y, z, vx, vy, vz]` in metres and metres per second, with velocities measured in
the rotating frame. All units are SI and angles are in radians. The ISS-style LVLH frame used in
much of the operations literature orders and signs the axes differently;
`vbar.conventions.hill_to_iss` and `iss_to_hill` convert between the two.

## The scenario: approach to a hold point

The chaser must reach a hold point 100 m behind the target and stop there without entering a 50 m
keep-out sphere around the target. The target is on a circular 500 km orbit. Defaults, all set in
`ApproachConfig`:

| Item | Default |
|---|---|
| Start | position uniform in x in [-200, 200] m, y in [200, 600] m, z in [-50, 50] m; each velocity component in [-0.1, 0.1] m/s; or a fixed `start_state` |
| Action | 3 numbers in [-1, 1], scaled by `max_accel` = 0.02 m/s^2 per axis, held for one step |
| Observation | the chaser's relative state `[x, y, z, vx, vy, vz]` (no sensor noise yet) |
| Step and length | `dt` = 10 s, at most 360 steps (one hour) |
| Success | within 5 m of the hold point and slower than 0.05 m/s; the episode ends on success |
| Keep-out | checked at 20 points inside every step, so passes through the zone between steps are caught; entering it does not end the episode unless `terminate_on_violation` is set |

Each run reports whether it captured, whether and in how many steps it entered the keep-out zone,
the minimum range to the target, the final distance to the hold point, delta-v used (the sum of
|a| dt over steps, the usual proxy for propellant), flight time and wall-clock time, plus the
intervention count when a safety filter wraps the agent. `run_episode` also returns the full state
and action history.

The environment currently propagates with Clohessy-Wiltshire dynamics. The nonlinear models below
are available on their own through `vbar.dynamics`.

## Writing a controller

Any object with a `name`, a `reset(env)` method and an `act(obs, info)` method that returns three
numbers in [-1, 1] is an agent:

```python
import numpy as np


class BrakeAgent:
    name = "brake"

    def reset(self, env):
        self.max_accel = env.cfg.max_accel

    def act(self, obs, info):
        return np.clip(-obs[3:] / (10.0 * self.max_accel), -1.0, 1.0)
```

`reset` receives the environment, so a controller can read its configuration (`env.cfg`), mean
motion (`env.n`) and hold point (`env.hold_point`).

### LLM agents

`LLMAgent` wraps any model you can call as a function from a prompt string to a reply string:

```python
from vbar.agents import LLMAgent


def complete(prompt: str) -> str:
    """Call any hosted or local model and return its text reply."""
    ...


agent = LLMAgent(complete, name="my-model")
result = run_episode(env, agent)
agent.log  # every prompt, reply, parsed action, parse failure and latency
```

Each step's prompt gives the frame, the goal, the keep-out radius, the mean motion, the current
state and the thrust limit, and asks for a JSON reply such as `{"ax": 0.1, "ay": -0.5, "az": 0.0}`.
A reply that cannot be parsed counts as a parse failure and the spacecraft coasts for that step.
The simulation clock is paused while the model answers.

### Reinforcement learning

`ApproachEnv` follows the Gymnasium API (`reset(seed=...)`, `step(action)`), with a `Box(-1, 1,
(3,))` action space and a 6-element observation, so it works with standard RL libraries. Its
reward is a shaped training signal (progress toward the hold point, minus delta-v, minus a penalty
for each step in the keep-out zone, plus a bonus on capture) and is not the benchmark score; score
agents on the metrics in `info`.

## Dynamics

The dynamics can be used without the environment:

```python
import numpy as np
from vbar.dynamics import make_dynamics

dyn = make_dynamics("twobody_j2", altitude=500e3, inclination=np.radians(51.6))
dyn.reset([0.0, -200.0, 0.0, 0.0, 0.0, 0.0])  # Hill-frame state, m and m/s
x, _ = dyn.advance([0.0, 0.0, 0.0], 600.0)  # hold zero acceleration for 600 s
```

| Model | What it is |
|---|---|
| `"cw"` | Clohessy-Wiltshire: relative motion linearized about a circular orbit, with the exact solution for an acceleration held constant over a step. Accurate while the separation is small compared with the orbit radius. |
| `"twobody"` | Nonlinear relative motion under point-mass gravity, integrated numerically in a form that keeps the relative state accurate far below a millimetre. |
| `"twobody_j2"` | The same with Earth's oblateness (J2). The target starts on a circular orbit in J2 mean elements, and the Hill frame includes the extra rotation J2 causes. |

Also included: Kepler propagation for any conic (`vbar.dynamics.kepler`), conversion between
states and classical orbital elements for elliptic orbits, and J2 secular rates with a first-order mean-osculating
element map (`vbar.dynamics.mean_elements`). Earth uses the WGS 84 constants; a Kerbin body with
KSP's values is available for comparison with KSPDG.

## Documentation

How each part is built, with its equations, methods, assumptions and limits, is in
[docs/](https://github.com/jamilhammoudeh/vbar/tree/main/docs):
[frames](https://github.com/jamilhammoudeh/vbar/blob/main/docs/frames.md),
[dynamics](https://github.com/jamilhammoudeh/vbar/blob/main/docs/dynamics.md),
[environment](https://github.com/jamilhammoudeh/vbar/blob/main/docs/environment.md) and
[validation](https://github.com/jamilhammoudeh/vbar/blob/main/docs/validation.md).

## Tests

```bash
python -m pytest -q
```

The models are tested against closed-form solutions, an independent integrator and published
textbook examples (listed in the validation page). CI runs the tests on Linux, macOS and Windows.

## Contributing

See [CONTRIBUTING.md](https://github.com/jamilhammoudeh/vbar/blob/main/CONTRIBUTING.md).

## Citation

If you use V-BAR in your work, please cite it using the metadata in
[CITATION.cff](https://github.com/jamilhammoudeh/vbar/blob/main/CITATION.cff) (GitHub's "Cite
this repository" button gives BibTeX and APA).

## License

[MIT](https://github.com/jamilhammoudeh/vbar/blob/main/LICENSE)
