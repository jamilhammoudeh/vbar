# V-BAR: Verified Benchmark for Autonomous Rendezvous

V-BAR is a headless Python benchmark for spacecraft rendezvous and proximity-operations
controllers. Classical guidance, reinforcement learning agents and LLM agents run on the same
scenarios and are scored with the same metrics.

**Status: early development.** The dynamics layer is validated; the environment has one
scenario. Safety filters, more scenarios and the scoring harness are in development, and
interfaces may change before 1.0.

## Why

KSPDG, the main existing environment for this kind of work, runs on Kerbal Space Program in real
time and needs a copy of the game. V-BAR is a plain Python package that runs much faster than real
time on Linux, macOS and Windows, so experiments with many seeds and many controllers are
practical. The physics is checked against closed-form solutions and published textbook examples.

## Contents

- Relative motion in the target's Hill frame (x radial out, y along-track, z orbit normal; SI
  units):
  - Clohessy-Wiltshire, with the exact zero-order-hold solution.
  - Nonlinear two-body relative motion, optionally with J2, with the target started from J2 mean
    elements.
- Kepler propagation, classical orbital elements and a first-order J2 mean-osculating map.
- An approach-to-hold-point environment (Gymnasium API) with a keep-out zone checked between time
  steps, not only at them.
- A PD baseline and an adapter for any LLM that takes text and returns text.

## Install

```bash
pip install vbar
```

For development:

```bash
git clone https://github.com/jamilhammoudeh/vbar
cd vbar
pip install -e ".[dev]"
```

Requires Python 3.12 or later. Core dependencies are numpy, scipy and gymnasium.

## Quickstart

```python
from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent

env = ApproachEnv(ApproachConfig(start_state=(0, -400, 0, 0, 0, 0)))
result = run_episode(env, PDAgent())
print(result["captured"], result["keep_out_violated"], result["dv_used_mps"])
```

This flies from 400 m behind the target to the hold point 100 m behind it, about 620 s of flight,
in well under a second. To save a trajectory plot to `outputs/`:

```bash
python examples/v0_demo.py
```

### LLM agents

```python
from vbar.agents import LLMAgent


def complete(prompt: str) -> str:
    """Call any hosted or local model and return its text reply."""
    ...


agent = LLMAgent(complete, name="my-model")
result = run_episode(env, agent)
agent.log  # every prompt, reply, parsed action, parse failure and latency
```

### Dynamics on their own

```python
import numpy as np
from vbar.dynamics import make_dynamics

dyn = make_dynamics("twobody_j2", altitude=500e3, inclination=np.radians(51.6))
dyn.reset([0.0, -200.0, 0.0, 0.0, 0.0, 0.0])  # Hill-frame position (m) and velocity (m/s)
x, _ = dyn.advance([0.0, 0.0, 0.0], 600.0)  # hold zero acceleration for 600 s
```

`make_dynamics` also accepts `"cw"` and `"twobody"`.

## Scenario: approach to hold point

The chaser must reach a hold point 100 m behind the target on the V-bar and stop there without
entering a 50 m keep-out sphere around the target. The target is on a circular 500 km orbit.

Reported in `info` and the result dict: capture, keep-out violation, minimum range to the target,
delta-v used and elapsed time, plus the intervention count when a safety filter wraps the agent.

## Validation

```bash
python -m pytest -q
```

The tests compare the models with closed-form solutions, an independent high-order integrator,
universal-variable Kepler propagation, worked examples from standard astrodynamics texts, and
conserved quantities (energy, angular momentum, the Jacobi integral). CI runs them on Linux,
macOS and Windows with the newest and the oldest supported dependencies.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT
