# Vectorized environment

Code: `vbar.scenarios.vector` (`ApproachVectorEnv`). Many copies of the approach scenario
([environment.md](environment.md)) stepped as one batch, for reinforcement learning that needs
many environment steps. It follows Gymnasium's vector API (`reset`, `step`, batched spaces) and
works with Gymnasium 0.29 and 1.x.

## How it is built

Each copy keeps its own `ApproachEnv` for what happens once per episode or needs its own random
stream: reset, the start state, the sensor and its noise. Everything done every step is an array
operation over the copies:

- **Commands and propulsion.** Actions are converted to fractions of a full command at each
  copy's current mass, limited to the box or ball, cleared below the minimum impulse bit, and
  burned at constant acceleration with the rocket equation, as `Spacecraft` does
  ([spacecraft.md](spacecraft.md)).
- **Propagation.** One step of the exact Clohessy-Wiltshire solution,
  $\mathbf x_{k+1} = \Phi\mathbf x_k + \Gamma\mathbf a_k$, for all copies at once, and the same for
  the `violation_samples` samples inside the step ([dynamics.md](dynamics.md)).
- **Keep-out check.** `segment_minima` finds every copy's minimum margin over the step with the
  same certified bound and refinement rule as the single environment, searching every candidate
  interval of the batch in one golden-section pass ([constraints.md](constraints.md)).
- **Propellant run-out inside a step.** A copy whose tank runs dry part way through a step has
  two segments; that step is flown by the copy's own environment, exactly as in a single run.
  Run-out happens at most once per episode.
- **Rewards and termination.** The legacy reward, a library reward
  ([rewards.md](rewards.md)) built from a batched `Transition`, capture, keep-out entries and the
  time limit are computed per row.

## Interface

- `ApproachVectorEnv(config, num_envs)` takes an `ApproachConfig`. Every copy uses the same
  configuration.
- `reset(seed=s)` seeds copy $i$ with $s + i$; `reset(seed=[s_0, s_1, ...])` gives each copy its
  own seed; `options` go to every copy (training environments only, [training.md](training.md)).
- `step(actions)` takes an array with one action per copy, `(N, 3)` or `(N,)` for `"disc27"`,
  and returns observations `(N, d)`, rewards, `terminated` and `truncated` `(N,)` and `info`, a
  dict of arrays with the single environment's keys (`min_margins` and `min_margin_bounds` hold
  one array per constraint), plus `discount` with a library reward.
- `agent_view()` and `scenario_id` are those of the single environment.

**Endings.** A copy that ends resets in the same step (Gymnasium's same-step autoreset): its row
of the returned observation is the new episode's first observation. `info["final_obs"]` holds
the true last observation of every copy that ended (rows of the others are NaN),
`info["final_info"]` the info of the step before the reset, and `info["_final_obs"]` and
`info["_final_info"]` mark the copies that ended. `terminated` is true for capture and for a
keep-out entry with `terminate_on_violation`, `truncated` at the time limit; on capture at the
last step both are true, and a learner should treat the step as terminal. A learner must
bootstrap from `info["final_obs"]` at a truncation, not from the new episode's observation.

## Supported options

The `"cw"` dynamics with the constant-acceleration drive, both thruster layouts, the minimum
impulse bit, propellant limits, every sensor, the `"normalized"`, `"accel"`, `"thrust"` and
`"disc27"` action modes, the `t_remaining` and `propellant_fraction` observation channels and
library rewards. Actuation delay, the constant-thrust drive, execution errors, the
`"impulse_coast"` mode and the nonlinear dynamics models are refused with a `ValueError`; for
those, Gymnasium's `SyncVectorEnv` over single environments works.

## Speed

On one desktop core, where a single environment takes about 1 ms per step with the default
configuration and small random commands, a batch takes about 6 ms per step for 8 copies, 8 ms
for 64 and 14 ms for 256: about 8 times faster per copy at 64 and 19 times at 256. Most of the
time per step goes to the keep-out refinement, whose cost hardly grows with the batch. The
numbers depend on the machine.

## Assumptions and limits

- Each copy's trajectory, observations, flags and metrics are those of a single environment with
  the same seed and actions, up to rounding: the batched products sum in a different order, and
  the golden-section refinement lands within about 1e-7 m of the Brent search's minimum. Booleans
  and counts are equal on the tested runs; a capture or keep-out decision exactly at its threshold
  could in principle differ.
- The sensors run one copy at a time, so a noisy sensor costs a few microseconds per copy and
  step.

## How it is tested

`tests/test_vector.py`, against single `ApproachEnv`s given the same seeds and actions:

- Through captures (terminated) and time limits (truncated) with same-step autoreset: the
  `terminated` and `truncated` flags are identical, observations, true states and the true final
  observation of every ending agree to 1e-9, and time, delta-v, propellant, mass, violation and
  capture flags, violation counts and minimum margins (1e-6 m) match.
- The same with keep-out entries that end the episode, a 0.05 kg tank that runs dry inside steps
  (15 times in the test), a library reward with every term including violation costs and the
  quadratic cost (rewards to 1e-9), and the time and propellant channels.
- The same in acceleration mode with ball thrusters, a minimum impulse bit and white-noise
  navigation; in thrust mode with time-correlated navigation; and with discrete actions.
- Integer and list seeds give each copy the seed of its single environment; unsupported options
  are refused.
- The batched keep-out minimum agrees with the scalar one over 40 random flybys
  (`tests/test_constraints.py`).

## Example

```python
import numpy as np

from vbar import ApproachConfig
from vbar.rewards import Reward
from vbar.scenarios import ApproachVectorEnv

reward = Reward(tau=3600.0, success_bonus=10.0, fuel_weight=1.0, progress_weight=0.01)
envs = ApproachVectorEnv(ApproachConfig(reward=reward), num_envs=16)
obs, info = envs.reset(seed=0)
rng = np.random.default_rng(0)
for _ in range(5):
    obs, rewards, terminated, truncated, info = envs.step(rng.uniform(-0.1, 0.1, (16, 3)))
print(obs.shape, rewards.shape, info["min_margins"]["keep_out"].shape)  # (16, 6) (16,) (16,)
```
