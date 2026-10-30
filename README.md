# V-BAR: Verified Benchmark for Autonomous Rendezvous

[![ci](https://github.com/jamilhammoudeh/vbar/actions/workflows/ci.yml/badge.svg)](https://github.com/jamilhammoudeh/vbar/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vbar)](https://pypi.org/project/vbar/)
[![Python](https://img.shields.io/pypi/pyversions/vbar)](https://pypi.org/project/vbar/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23632751.svg)](https://doi.org/10.5281/zenodo.23632751)

V-BAR is a headless Python benchmark for spacecraft rendezvous and proximity-operations
controllers. Classical guidance laws, reinforcement learning agents and LLM agents fly the same
scenarios and are scored with the same metrics.

![The PD baseline in the target's Hill frame. From 400 m behind the target it flies straight to the hold point 100 m behind. From a seeded start 523 m ahead it flies through the 50 m keep-out zone, 14.4 m from the target at closest, before stopping at the hold point. Range and speed against time are shown below.](https://raw.githubusercontent.com/jamilhammoudeh/vbar/main/docs/images/pd_approach.png)

The PD baseline reaches the hold point from behind the target (blue) but flies through the keep-out zone from ahead of it (orange), closing to 14.4 m.

**Status: early development (0.4).** The dynamics, spacecraft, sensor and constraint layers are
validated (see [Tests](#tests)). Version 0.3 added the metrics, agent tools (remote and ONNX
policy agents, the sign check, reward and training interfaces), a scenario catalogue and
`vbar.solvers` (command projection and horizon quadratic programs). Version 0.4 adds the classical
baselines and fuel oracles, the evaluation harness, comparison tasks rebuilt from published
benchmarks, agreement checks against Basilisk and Orekit, pursuit-evasion in KSPDG's format,
replay export to Rerun, fault injection, mission reconstructions and a tumbling-capture scenario.
Safety filters beyond the pass-through ("filter off") and the LLM supervisor study are in
progress, and interfaces may change before 1.0.

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
`plot` extra: `pip install "vbar[plot]"`; ONNX policy export and inference need the `onnx` extra:
`pip install "vbar[onnx]"`; replays for the Rerun viewer need the `replay` extra:
`pip install "vbar[replay]"`. The agreement checks against Basilisk and Orekit need the `basilisk`
extra, `pip install "vbar[basilisk]"`, and the `orekit` extra, `pip install "vbar[orekit]"` with a
Java runtime (11 or later) on `JAVA_HOME`. To work on V-BAR itself, see
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

The PD baseline reaches the hold point and flies through the keep-out zone on every default seed,
because it ignores the target. That gap is what safety filters and better controllers are measured
against. In a clone of the repository, `python examples/v0_demo.py` shows both cases and saves a
plot to `outputs/` (needs the `plot` extra).

![Animation of the PD baseline from seed 5 in the Hill frame. A marker leaves its start 523 m ahead of the target, crosses the 50 m keep-out zone with a readout showing time, range and speed, passes 14.4 m from the target, and stops at the hold point 100 m behind it.](https://raw.githubusercontent.com/jamilhammoudeh/vbar/main/docs/images/pd_approach.gif)

The orange run animated: it spends about 2.3 minutes inside the keep-out zone.

`python examples/readme_figures.py` regenerates this animation and the figure above
([docs/viz.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/viz.md)).

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
These are the benchmark's scores; the keys `run_episode` returns are a quick check of one run.

```python
from vbar import ApproachConfig, ApproachEnv
from vbar.agents import PDAgent
from vbar.metrics import evaluate, record_episode

env = ApproachEnv(ApproachConfig(start_state=(0, -400, 0, 0, 0, 0)))
m = evaluate(record_episode(env, PDAgent()))
print(m["success"], m["composite_score"], m["failure_label"])  # True 1.0 None
```

Definitions, tests and one card per metric are in
[docs/metrics.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/metrics.md).

A whole suite is run and reported with `vbar.eval`: every (scenario, seed) of the suite in
parallel, one JSON-lines result per run with a provenance hash, and a report that refuses a
partial suite and gives success rates with exact intervals, interquartile means with stratified
bootstrap intervals and safety metrics by their unsafe tails. The core suite `core-v1` holds the
headline benchmark: the V-bar and R-bar approaches, the on-axis docking, the run time assurance
docking and the inspection, each with and without navigation noise.

```
python -m vbar.eval run --suite core-v1 --agent vbar.agents:DriftAgent --workers 8 --out drift.jsonl
python -m vbar.eval report drift.jsonl --markdown drift.md
```

See [docs/evaluation.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/evaluation.md).

## Writing a controller

Any object with a `name`, a `reset(view, rng)` method and an `act(obs, info)` method that returns
three numbers in [-1, 1] is an agent:

```python
import numpy as np

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.sign_check import sign_check


class SpringAgent:
    name = "spring"

    def reset(self, view, rng=None):
        self.max_accel = view.max_accel
        self.hold_point = np.array(view.goal["hold_point"])

    def act(self, obs, info):
        # Critically damped at 0.01 rad/s, nine times the orbital rate, so the CW terms are small.
        a = 1e-4 * (self.hold_point - obs[:3]) - 0.02 * obs[3:6]
        return np.clip(a / self.max_accel, -1.0, 1.0)


print(sign_check(SpringAgent()).passed)  # True
env = ApproachEnv(ApproachConfig(start_state=(0, -400, 0, 0, 0, 0)))
print(run_episode(env, SpringAgent())["captured"])  # True
```

`reset` receives a read-only `AgentView` with the scenario's public description (time step, mean
motion, actuator limits, goal and constraints) and a random stream for the agent's own use. `obs`
comes from the navigation sensor and `info` holds only the time, so a controller never sees the
true state. The agent's stream is seeded by `run_episode(env, agent, seed, agent_seed)`
independently of the episode seed. Seeds not given are drawn (128 bits) and returned in the
result, so the run can be repeated; scored comparisons keep them drawn, since an agent can find a
small seed by matching its first observation against the public scenario
([docs/environment.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/environment.md)).

### Classical baselines

`vbar.agents` includes classical controllers to compare against. Each page gives the method, its
limits and how it is tested.

| Agents | Method | Page |
|---|---|---|
| `PDAgent` | CW-cancelling proportional-derivative law to the hold point | [environment.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/environment.md) |
| `LQRAgent`, `LQGAgent` | discrete LQR to the hold point, and the same law on a Kalman filter estimate | [lqr.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/lqr.md), [lqg.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/lqg.md) |
| `CWTrackingAgent`, `LyapunovAgent` | Lyapunov tracking on the CW model and on a declared two-body model | [lyapunov.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/lyapunov.md) |
| `TwoImpulseAgent` | two-impulse CW targeting | [two-impulse.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/two-impulse.md) |
| `GlideslopeAgent` | pulsed glideslope approach | [glideslope.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/glideslope.md) |
| `ElementAgent`, `TorvaAgent`, `FehseSequenceAgent`, `NominalTrackingAgent` | transfers, fly-arounds, forced motion and nominal tracking | [trajectory-elements.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/trajectory-elements.md) |
| `ElevationTimedAgent`, `PilotAgent` | Space Shuttle elevation-timed burns and manual pulses | [shuttle-baselines.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/shuttle-baselines.md) |
| `LambertAgent` | two-impulse targeting by Lambert's problem (`vbar.dynamics.lambert.lambert`) | [lambert.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/lambert.md) |
| `MissionManagerAgent` | a timeline of modes with gates, contingencies and an abort | [mission-manager.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/mission-manager.md) |
| `MPCAgent` (`vbar.agents.mpc`) | hold-point MPC with keep-out rows sampled within each step and an optional certified terminal set | [mpc.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/mpc.md) |
| `NMPCAgent` (`vbar.agents.nmpc`) | nonlinear MPC by multiple shooting on a declared dynamics model | [nonlinear-mpc.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/nonlinear-mpc.md) |
| `PiecewiseAffineMPCAgent` (`vbar.agents.mpc_pwa`) | box-constrained regulator solved once as a multiparametric QP, then a table lookup | [piecewise-affine-mpc.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/piecewise-affine-mpc.md) |
| `FuelMPCAgent` (`vbar.agents.mpc_fuel`) | least 1-norm delta-v plan to rest at the hold point, planned again at every decision | [mpc-fuel.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/mpc-fuel.md) |
| `ClfCbfAgent` (`vbar.agents.clf_cbf`) | one QP per decision with a Lyapunov decrease row and a keep-out barrier row | [clf-cbf.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/clf-cbf.md) |

`Mutant(agent, kind, severity)` wraps any of them with one deliberate defect, to check that the
metrics catch it ([mutants.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/mutants.md)).

`vbar.oracles` gives what the baselines are scored against: certified fuel bounds from the primer
vector and a finite-thrust dual bound (`FuelOracle`, `bounds_for`), primer-vector certificates, the
minimum-time and passing-side oracles, and `OracleAgent` with `expert_dataset` for expert
trajectories ([oracles.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/oracles.md),
[oracle-certificates.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/oracle-certificates.md),
[minimum-time-oracle.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/minimum-time-oracle.md),
[passing-side-oracle.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/passing-side-oracle.md),
[expert-datasets.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/expert-datasets.md)).

### LLM agents

`LLMAgent` wraps any model through a function that takes the decision's messages and the tools
offered and returns the reply; `text_model` adapts a plain prompt-to-text function:

```python
from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import LLMAgent, Price, Reply, text_model


def complete(messages, tools):
    """Send the conversation to any hosted or local model and return its reply."""
    return Reply('{"ax": 0.0, "ay": 0.0, "az": 0.0}', input_tokens=900, output_tokens=20)


env = ApproachEnv(ApproachConfig(start_state=(0, -400, 0, 0, 0, 0)))
agent = LLMAgent(complete, name="my-model", price=Price(3.0, 15.0))
result = run_episode(env, agent)
print(agent.usage())  # decisions, model calls, tokens, cost and parse failures

plain = LLMAgent(text_model(lambda prompt: '{"ax": 0.0, "ay": 0.0, "az": 0.0}'))
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
  element map (`vbar.dynamics.mean_elements`);
- a replay of flown orbits through these models (`vbar.flightdata`): GRACE-FO's two satellites on
  3 December 2019, 182 km apart, stay inside an error bound stated before the run over ten orbits
  on the J2 and drag rungs, a check of the nonlinear models, not of Clohessy-Wiltshire.
- published results rebuilt from their papers (`vbar.reproductions`): Ariba et al.'s minimum-fuel
  V-bar glideslope by linear programming (`vbar.agents.glideslope_lp`) on Hablani's case, compared
  number by number with `python -m vbar.reproductions`.

Earth uses the WGS 84 constants; a Kerbin body with KSP's values is available for comparison with
KSPDG.

## Pursuit-evasion in KSPDG's format

`vbar.games` rebuilds KSPDG's pe1 pursuit-evasion problem on two-body dynamics, with KSPDG's
observation list, action dictionary and agent interface, so an agent written for KSPDG runs
unchanged, headless, faster than real time and in parallel. The evaders are V-BAR's own bots.

```python
from vbar.games import PE1_E3_I3_Env, AgentEnvRunner
from vbar.games.pursuers import VelocityMatchingPursuer

results = AgentEnvRunner(VelocityMatchingPursuer(), PE1_E3_I3_Env, {"episode_timeout": 30.0}).run()
print(results["agent_env_results"]["weighted_score"])
```

The scenarios are also Gymnasium environments (`vbar/PursuitEvasionE3I3-v1`); see
[docs/pursuit-evasion.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/pursuit-evasion.md).

## Scenario catalogue

Beyond the approach above, `vbar.scenarios` holds twenty-three scenarios built from specs: the
V-bar, R-bar and orbit-normal approaches with a dwell at the hold point, docking as a contact
event with mechanism windows and corridors (with on-axis, Shuttle-corridor and Sun-window
variants), an offset port, an inertially pointing target, the run time assurance docking task,
inspection under lighting and shadow, the ISS-class closing and R-bar acquisition, a berthing
box, and the Shuttle's flown profiles (ORBT, the pitch-manoeuvre hold, the twice-orbital-rate
transition, the V-bar final approach, the HST grapple approach and the separation fly-around),
four comparison tasks rebuilt from their papers (the AFRL docking and illuminated inspection, the
SpaceGym docking and rendezvous), a departure from an aft port, and the capture of a tumbling
upper stage with five tumble presets (flat spin, coning, near the separatrix, next to the
intermediate axis, a dead stage librating in the gravity gradient). Each has a Gymnasium ID, and
all but the comparison tasks, the departure and the capture have `:nav1` and `:nav2` variants:

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

Three real rendezvous events are rebuilt from public reports with their failures as fault models:
the Progress M-34 collision with Mir, the ETS-VII thruster anomaly and the DART collision, each
with a `:nominal` variant without the failure
([docs/reconstructions.md](https://github.com/jamilhammoudeh/vbar/blob/main/docs/reconstructions.md)).

## Documentation

How each part is built, with its equations, methods, assumptions and limits, is in
[docs/](https://github.com/jamilhammoudeh/vbar/blob/main/docs/README.md), one page per part: frames,
dynamics, perturbations, relative-motion models, flight-data replay, transition matrices, three-body
models, spacecraft, sensors, faults, constraints, the environment, scenarios, reconstructions,
reproduced results, metrics, rewards, training interfaces, the vectorized environment, agent tools,
LLM agents, remote agents, policy export, solvers, the evaluation harness, the external propagators,
the classical baselines, pursuit-evasion in KSPDG's format and validation.

## Tests

```bash
python -m pytest -q          # the default run, without the whole-episode audits
python -m pytest -q -m slow  # the whole-episode scenario audits
```

The models are tested against closed-form solutions, an independent integrator and published
textbook examples (listed in the validation page). CI runs the tests on Linux, macOS and Windows.

## Contributing

See [CONTRIBUTING.md](https://github.com/jamilhammoudeh/vbar/blob/main/CONTRIBUTING.md).

## Citation

If you use V-BAR in your work, please cite it using the metadata in
[CITATION.cff](https://github.com/jamilhammoudeh/vbar/blob/main/CITATION.cff) (GitHub's "Cite
this repository" button gives BibTeX and APA). Each release is archived on Zenodo; the DOI
[10.5281/zenodo.23632751](https://doi.org/10.5281/zenodo.23632751) always resolves to the latest
version, and each version has its own DOI on that page.

## License

[MIT](https://github.com/jamilhammoudeh/vbar/blob/main/LICENSE)
