# V-BAR: Verified Benchmark for Autonomous Rendezvous

`vbar` is a fast, headless, cross-platform benchmark for testing AI and classical spacecraft controllers, with and without safety filters.

> **Status: v0 (early development).** One scenario, validated close-range dynamics, classical and LLM agent interfaces.

## Why

Testing AI controllers for spacecraft rendezvous today usually means running a full game engine (KSPDG, built on Kerbal Space Program): a paid game, GUI-only, macOS/Linux, real-time. That makes large, statistically meaningful experiments slow. V-BAR runs anywhere Python runs, with no game, and simulates an hour of flight in a fraction of a second.

Goals:

- **Fair comparisons:** the same scenarios, metrics, and number of runs for every controller: classical, reinforcement learning, or LLM.
- **Safety as a first-class metric:** keep-out violations are checked continuously (including between time steps), and safety filters can be switched on and off.
- **Reproducible:** seeds, logged prompts and replies, and validated physics.

## Install

```bash
pip install -e ".[dev]"   # from a clone of this repo
```

## Quickstart

```python
from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent

env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0)))
result = run_episode(env, PDAgent())
print(result["captured"], result["keep_out_violated"], result["dv_used_mps"])
```

Run the demo (saves a trajectory plot to `outputs/`):

```bash
python examples/v0_demo.py
```

### Plug in any LLM

```python
from vbar.agents import LLMAgent


def complete(prompt: str) -> str:
    """Call any hosted or local model and return its text reply."""
    ...


agent = LLMAgent(complete, name="my-model")
result = run_episode(env, agent)
agent.log  # every prompt, reply, parsed action, parse failure, latency
```

## Scenario: approach to hold point

The chaser must reach a hold point 100 m behind the target on the V-bar and stop, without entering a 50 m keep-out sphere around the target.

**Dynamics:** Clohessy-Wiltshire relative motion (circular target orbit at 500 km, close range, no perturbations). Integrated numerically (DOP853) and **validated against the exact closed-form solution** to sub-millimeter accuracy over a full orbit (`tests/test_cw.py`).

**Metrics (in `info` and the result dict):** capture, keep-out violation (checked between steps), minimum range to target, delta-v used, time, filter interventions.

## Validation

```bash
pytest -q
```

Tests check the dynamics against known exact results: the closed-form solution, V-bar equilibrium, closed relative ellipses, and the 12π·|x0| along-track drift per orbit.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT
