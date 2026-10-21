# V-BAR: Verified Benchmark for Autonomous Rendezvous

[![ci](https://github.com/jamilhammoudeh/vbar/actions/workflows/ci.yml/badge.svg)](https://github.com/jamilhammoudeh/vbar/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vbar)](https://pypi.org/project/vbar/)
[![Python](https://img.shields.io/pypi/pyversions/vbar)](https://pypi.org/project/vbar/)

V-BAR is a headless Python benchmark for spacecraft rendezvous and proximity-operations
controllers. Classical guidance laws, reinforcement learning agents and LLM agents fly the same
scenarios and are scored with the same metrics.

**Status: early development (0.2).** The dynamics, spacecraft, sensor and constraint layers are
validated (see [Tests](#tests)). The environment has one scenario, the safety-filter interface has
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
| Action | 3 numbers in [-1, 1], scaled by `max_accel` = 0.02 m/s^2 per axis, held for one step; `action_mode="accel"` or `"thrust"` takes m/s^2 or newtons instead |
| Observation | the chaser's relative state `[x, y, z, vx, vy, vz]` from the navigation sensor, exact by default (`PerfectNav`); `GaussianNav`, `RangeScaledNav` and `GaussMarkovNav` add errors, and `observation_channels` appends time left, propellant left and pending commands |
| Spacecraft | 500 kg, Isp 220 s, propellant following the rocket equation; per-axis (`"box"`) or total (`"ball"`) thrust limit, minimum impulse bit, actuation delay, constant-acceleration or constant-thrust drive and thrust execution error are options |
| Step and length | `dt` = 10 s, at most 360 steps (one hour) |
| Success | within 5 m of the hold point and slower than 0.05 m/s; the episode ends on success |
| Keep-out | checked in continuous time: 20 samples per step, refined between them, with a certified lower bound; entering it does not end the episode unless `terminate_on_violation` is set |

Each run reports whether it captured, whether and in how many steps it entered the keep-out zone,
the minimum range to the target, the final distance to the hold point, delta-v used (each step's
delta-v summed, the usual proxy for propellant), flight time and wall-clock time, plus the
intervention count when a safety filter wraps the agent. `run_episode` also returns the full true
state and action history and the seed of the agent's random stream. Propellant used, mass and the
certified bound on the keep-out margin are in `info`.

The environment propagates with Clohessy-Wiltshire dynamics by default;
`ApproachConfig(dynamics="twobody")` or `"twobody_j2"` selects one of the nonlinear models below.

## Scoring

`vbar.metrics.record_episode(env, agent, seed)` records a run and `vbar.metrics.evaluate(episode)`
scores it from the true trajectory in continuous time: mission (success, terminal errors, time,
progress, docking contact, hold dwell), efficiency (delta-v in both norms, propellant, burns, duty
cycle, chatter, the optimality gap against oracle bounds), constraint margins with violation depth
and duration, passive safety and collision-avoidance availability, thruster plume, safety-filter
and agent metrics, a composite score with its breakdown, and a failure label for failed runs.
Definitions, tests and one card per metric are in
[docs/metrics.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/metrics.md).

## Writing a controller

Any object with a `name`, a `reset(view, rng)` method and an `act(obs, info)` method that returns
three numbers in [-1, 1] is an agent:

```python
import numpy as np


class BrakeAgent:
    name = "brake"

    def reset(self, view, rng=None):
        self.max_accel = view.max_accel

    def act(self, obs, info):
        return np.clip(-obs[3:6] / (10.0 * self.max_accel), -1.0, 1.0)
```

`reset` receives a read-only `AgentView` with the scenario's public description (time step, mean
motion, actuator limits, goal and constraints) and a random stream for the agent's own use. `obs`
comes from the navigation sensor and `info` holds only the time, so a controller never sees the
true state. The agent's stream is seeded by `run_episode(env, agent, seed, agent_seed)`
independently of the episode seed. Seeds not given are drawn (128 bits) and returned in the
result, so the run can be repeated; scored comparisons keep them drawn, since an agent can find a
small seed by matching its first observation against the public scenario
([docs/environment.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/environment.md)).

### LLM agents

`LLMAgent` wraps any model through a function that takes the decision's messages and the tools
offered and returns the reply; `text_model` adapts a plain prompt-to-text function:

```python
from vbar.agents import LLMAgent, Price, Reply


def complete(messages, tools):
    """Send the conversation to any hosted or local model and return its reply."""
    return Reply('{"ax": 0.0, "ay": 0.0, "az": 0.0}', input_tokens=900, output_tokens=20)


agent = LLMAgent(complete, name="my-model", price=Price(3.0, 15.0))
result = run_episode(env, agent)
print(agent.usage())  # decisions, model calls, tokens, cost and parse failures
```

Each step's prompt gives the frame, the goal, the keep-out radius, the mean motion, the current
state and the thrust limit, and asks for a JSON reply such as `{"ax": 0.1, "ay": -0.5, "az": 0.0}`.
A reply that cannot be parsed counts as a parse failure and the spacecraft coasts for that step.
The prompt is configurable, the agent can offer tools (coast and transfer predictions from its
own observation), every run can be recorded and replayed without the model, and an optional
belief check measures whether the model's stated reasoning matches its inputs
([docs/llm-agents.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/llm-agents.md)). The simulation clock is paused while the model
answers, unless the run is in real time.

Agents can also run in another process or language through the agent server (with Python and
MATLAB clients), LLM harnesses can fly episodes through a Model Context Protocol tool server
([docs/remote-agents.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/remote-agents.md)), and planners can submit whole plans to an
external verifier ([docs/agents.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/agents.md)). Every agent should pass the
sign-convention check, `vbar.agents.sign_check.sign_check(agent)`, before it is scored.

### Reinforcement learning

`ApproachEnv` follows the Gymnasium API (`reset(seed=...)`, `step(action)`), by default with a
`Box(-1, 1, (3,))` action space and a 6-element observation, and passes Gymnasium's environment
checker, so it works with standard RL libraries. Importing `vbar` registers it:

```python
import gymnasium as gym

import vbar  # noqa: F401

env = gym.make("vbar/Approach-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Train with a library reward, whose terms follow the benchmark metrics and whose discount is
declared per second, so the task does not change with the step length
([docs/rewards.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/rewards.md)):

```python
from vbar import ApproachConfig, ApproachEnv
from vbar.rewards import Reward

reward = Reward(tau=3600.0, success_bonus=10.0, fuel_weight=1.0, entry_weights={"keep_out": 5.0})
env = ApproachEnv(ApproachConfig(reward=reward, training=True))
obs, info = env.reset(seed=0)
obs, r, terminated, truncated, info = env.step([0.0, -0.1, 0.0])
print(round(info["discount"], 6))  # discount this transition by exp(-dt / tau)
```

Without `reward` the scenario returns its legacy shaped reward (progress toward the hold point,
minus delta-v, minus a penalty for each step in the keep-out zone, plus a bonus on capture), which
is not aligned with the metrics and is not the benchmark score; score agents with `vbar.metrics`.
`ApproachVectorEnv` steps many copies as one batch ([docs/vector-env.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/vector-env.md));
the discrete bang-off-bang and burn-and-coast action modes, training-only starts and a safety
filter in the training loop are on [docs/training.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/training.md), and trained networks
export to ONNX with their normalization built in ([docs/policy-export.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/policy-export.md)).
Rewards and `info` come from the true state, so fly trained policies for scoring through
`run_episode` or `vbar.metrics.record_episode`, which pass them only the observation and the time
and refuse environments built for training.

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

Also included, each documented in [docs/](https://github.com/jamilhammoudeh/vbar/blob/main/docs/README.md):

- differential drag and solar radiation pressure with Earth shadow for the two-body model
  (`vbar.dynamics.perturbations`), and elliptic or phased target orbits;
- the Yamanaka-Ankersen elliptic model, curvilinear Clohessy-Wiltshire and linearized two-body
  motion with J2 (`vbar.dynamics.relative`); transition matrices from the variational
  equations; model-error measures between models (`vbar.dynamics.fidelity`);
- Earth-Moon three-body models: the circular restricted problem, linear models about libration
  points and periodic orbits, and the bicircular Sun perturbation (`vbar.dynamics.cr3bp`), with
  halo-orbit seeds, correctors, continuation and Floquet analysis
  (`vbar.dynamics.periodic_orbits`);
- Kepler propagation for any conic (`vbar.dynamics.kepler`), conversion between states and
  classical elements for elliptic orbits, and J2 secular rates with a first-order mean-osculating
  element map (`vbar.dynamics.mean_elements`).

Earth uses the WGS 84 constants; a Kerbin body with KSP's values is available for comparison with
KSPDG.

## Scenario catalogue

Beyond the approach above, `vbar.scenarios` holds seventeen scenarios built from specs: the
V-bar, R-bar and orbit-normal approaches with a dwell at the hold point, docking as a contact
event with mechanism windows and corridors (with on-axis, Shuttle-corridor and Sun-window
variants), an offset port, an inertially pointing target, the run time assurance docking task,
inspection under lighting and shadow, the ISS-class closing and R-bar acquisition, a berthing
box, and the Shuttle's flown profiles (ORBT, the pitch-manoeuvre hold, the twice-orbital-rate
transition, the V-bar final approach, the HST grapple approach and the separation fly-around).
Each has `:nav1` and `:nav2` variants and a Gymnasium ID:

```python
import gymnasium as gym

from vbar.scenarios import SCENARIOS, make, seeds

print(sorted(SCENARIOS))
env = make("docking:axis")
print(env.scenario_id)
env.reset(seed=seeds("docking:axis", "validation", 10).seed(0))
env = gym.make("vbar/ApproachRbar-v1")
```

Identifiers name every option that changes a result, seeds come in train, validation and hidden
test splits, and suites list the runs a report must cover in full
([docs/scenarios.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/scenarios.md)).

## Documentation

How each part is built, with its equations, methods, assumptions and limits, is in
[docs/](https://github.com/jamilhammoudeh/vbar/blob/main/docs/README.md), one page per part:
frames, dynamics, perturbations, relative-motion models, transition matrices, three-body models,
spacecraft, sensors, constraints, the environment, scenarios, metrics, rewards, training
interfaces, the vectorized environment, agent tools, LLM agents, remote agents, policy export,
and validation.

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
