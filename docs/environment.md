# Approach environment

Code: `vbar.scenarios.base` (`BaseEnv`, `EnvConfig`, `StepRecord`), `vbar.scenarios.approach`
(`ApproachEnv`, `ApproachConfig`) and `vbar.runner`. Every scenario is built on `BaseEnv`, which
composes a dynamics model ([dynamics.md](dynamics.md)), a spacecraft
([spacecraft.md](spacecraft.md)), a navigation sensor ([sensors.md](sensors.md)) and the
scenario's constraints ([constraints.md](constraints.md)). The approach is the first scenario.
Both configurations take keyword arguments only.

## Task

Fly the chaser to a hold point and stop there without entering a keep-out sphere around the
target. Defaults: hold point $(0, -100, 0)$ m, 100 m behind the target on the V-bar; keep-out
radius 50 m; target on a circular 500 km orbit.

## Interface

Gymnasium API: `reset(seed=...)` returns `(obs, info)`; `step(action)` returns
`(obs, reward, terminated, truncated, info)`. Importing `vbar` registers the scenario as
`vbar/Approach-v0`, so `gymnasium.make("vbar/Approach-v0", config=ApproachConfig(...))` builds it,
and it passes `gymnasium.utils.env_checker.check_env` (tested in `tests/test_registration.py`).

- **Action:** three numbers held for one step of `dt` = 10 s, in the units of `action_mode`:
  `"normalized"` (default, `Box(-1, 1, (3,))`) takes fractions of `max_accel` = 0.02 m/s^2 per
  Hill axis, `"accel"` takes m/s^2 and `"thrust"` newtons, with the box bounds at a full command at
  the initial mass. The environment converts the action to fractions of a full command at the
  current mass, and the spacecraft applies its thrust limits, minimum impulse bit and delay.
  `"disc27"` takes an index into 27 bang-off-bang commands, and `"impulse_coast"` a one-step
  burn followed by a coast of chosen length in one decision ([training.md](training.md)).
- **Observation:** the sensor's navigated state $[x, y, z, \dot x, \dot y, \dot z]$, exact by
  default, followed by the channels listed in `observation_channels`, always in this order:
  - `t_remaining`: time left in the episode (s);
  - `propellant_fraction`: propellant left over propellant at the start;
  - `pending_commands`: $\lceil \text{delay} / \Delta t\rceil$ slots of (s until the command takes
    effect, command in action units): the command in effect first, then those not yet in effect,
    zero-filled at the front. With these the next state follows from the observation and the new
    action alone.

  `AgentView.observation` names every entry.
- **Episode:** at most `max_steps` = 360 steps (one hour). It ends early on capture, or on entering
  the keep-out zone if `terminate_on_violation` is set.

## Start states and random streams

With `start_state` unset, each reset draws position components uniformly from x in [-200, 200] m,
y in [200, 600] m and z in [-50, 50] m, and each velocity component from [-0.1, 0.1] m/s. These
starts are ahead of the target, so the target lies between the chaser and the hold point. A fixed
`start_state` overrides the draw.

`reset(seed=...)` seeds Gymnasium's generator, which draws the start state. Its seed sequence
spawns independent sensor, actuator and target streams at every reset, so the same seed gives the
same episode, and adding sensor noise or execution error never moves the start state or another
model's draws. The agent's stream is seeded separately by `run_episode` (below), so nothing an
agent holds is derived from the episode seed. Each environment works on its own copy of the
configured sensor, so environments built from one configuration do not share noise.

## Propagation

Each step the spacecraft turns the command into segments of constant Hill-frame acceleration,
split at the actuation delay's switch time, at propellant run-out and into the two halves of a
constant-thrust burn ([spacecraft.md](spacecraft.md)). The dynamics model advances each segment
with its own call, so no integrator steps across a thrust change. `dynamics` selects the model:
`"cw"` (default, the exact Clohessy-Wiltshire solution for constant acceleration), `"twobody"` or
`"twobody_j2"`, about a circular orbit set by `body`, `altitude_m` and `inclination`.

## Keep-out check

Each segment is sampled at `violation_samples` = 20 points per full step, and the lowest margin is
refined between samples with a certified lower bound ([constraints.md](constraints.md)). The
environment tracks the minimum range to the target, whether the zone was ever entered, the number
of steps that entered it, and the certified lower bound on the margin. A scenario lists in
`soft_constraints` the constraints it scores without failing the run; every other constraint is
hard (the approach scenario's keep-out sphere is hard).

## Capture

After each step: distance to the hold point below `capture_radius` = 5 m and speed below
`capture_speed` = 0.05 m/s, both from the true state.

## What the agent and the metrics see

Agents get only the observation, `env.agent_view()` and `env.agent_info()`, which holds the time.
`AgentView` is frozen and holds the scenario's public description: `scenario_id`, `dt`,
`max_steps`, the mean motion `n`, `action_mode`, `max_accel`, `thrust_limit`, `mass_kg`, `goal`
and `constraints` (read-only mappings), `observation` and `frame`.

The Gymnasium `info` dict is for metrics and logging and holds the truth:

| Key | Meaning |
|---|---|
| `time_s` | simulated time (s) |
| `true_state` | the true relative state (m, m/s) |
| `dist_to_hold_m` | distance to the hold point (m) |
| `range_to_target_m` | distance to the target (m) |
| `speed_mps` | relative speed (m/s) |
| `dv_used_mps` | delta-v used, the 2-norm of each step's delta-v summed (m/s) |
| `dv_axes_mps` | the same in the 1-norm, what box thrusters spend propellant on (m/s) |
| `propellant_used_kg` | propellant used (kg) |
| `mass_kg` | current mass (kg) |
| `min_range_to_target_m` | minimum range so far, including inside steps (m) |
| `min_margins` | lowest margin so far of each constraint |
| `min_margin_bounds` | certified lower bound on each of those |
| `keep_out_violated` | whether the keep-out zone was entered |
| `violation_steps` | number of steps that entered it |
| `captured` | whether the capture condition holds |

`env.record` is the last step's `StepRecord`: time, true state, observation, command, filtered
command, the spacecraft's `Actuation` and the step's segments, segment start states, samples,
constraint minima and violation intervals.

`run_episode(env, agent, seed, agent_seed)` resets the environment, calls
`agent.reset(view, rng)` with a generator seeded by `agent_seed` alone, and returns the final
metrics above plus `agent`, `seed`, `agent_seed`, `wall_time_s`, the true state history `states`,
the actions `actions`, and `filter_interventions` when the agent is wrapped in a safety filter.
Without an `agent_seed` one is drawn from the operating system and returned, so any run can be
repeated. `run_many(env, agent, seeds, agent_seeds)` runs one episode per seed.

## Reward

A training signal for reinforcement learning, not the benchmark score. With
`reward=Reward(...)` in the configuration the reward comes from the reward library and `info`
holds the step's `discount` ([rewards.md](rewards.md)); success is capture with the keep-out zone
never entered. Without it the scenario returns its legacy reward, the first release's shaped
signal, which is not aligned with the metrics. For step $k$, with $d_k$ the distance to the hold
point and $\Delta v_k$ the step's delta-v (2-norm):

$$
r_k = \frac{d_{k-1} - d_k}{100} - w\,\Delta v_k - \mathbb 1[\text{keep-out entered in step } k] + 10\,\mathbb 1[\text{captured}]
$$

with $w$ = `reward_dv_weight` = 1. It pays the capture bonus even after a keep-out violation. It
is computed from the true state, so scored runs do not pass it to the agent.

## Agents

An agent has a `name`, a `reset(view, rng)` method and an `act(obs, info)` method that returns an
action in the view's action mode. Provided:

- `PDAgent`: cancels the Clohessy-Wiltshire terms and applies a proportional-derivative law toward
  the hold point, $\mathbf a = -\mathbf a_{\text{CW}} - \omega_n^2(\boldsymbol\rho - \boldsymbol\rho_{\text{hold}}) - 2\zeta\omega_n\dot{\boldsymbol\rho}$,
  with $\omega_n$ = 0.01 rad/s and $\zeta$ = 1 by default, clipped to the thrust limit per axis
  and converted to the action mode (in thrust mode, the thrust at the initial mass). It ignores the
  keep-out zone, so it is the unfiltered baseline.
- `DriftAgent`: no thrust.
- `LLMAgent(complete, name)`: formats a prompt with the frame, goal, keep-out radius, mean motion,
  observed state, step length and thrust limit, calls `complete(prompt)`, and parses a JSON reply
  `{"ax": ..., "ay": ..., "az": ...}` of fractions, which it converts to the action mode. An
  unparseable reply is logged as a parse failure and the spacecraft coasts. Every call's prompt,
  reply, action, failure flag and latency go to `log`.
- `FilteredAgent(agent, safety_filter)` (`vbar.safety`): wraps an agent with a filter that has
  `reset(view)` and `filter(obs, action)`, and counts the steps where the filter changed the
  command. The filter returns the action or a `FilterResult`: the action, whether it intervened,
  a status (`feasible`, `infeasible_certified` when a certificate proves no safe action exists,
  `solver_failed` when it found neither), the accepted slack, the conflicting constraints and its
  wall time. A bare action is wrapped as `feasible`, intervened when a component moved by more
  than `tol` (1e-6), with the measured time. After each decision `last_command` and `last_result`
  hold the agent's command and the result, and `last_filter_s` the filter's measured time, for
  the step record ([metrics.md](metrics.md)). Only the pass-through filter exists so far.

`vbar.agents.action_scale(view)` gives the action units per fraction of a full command.

## Configuration

`EnvConfig` (all scenarios):

| Field | Default | Meaning |
|---|---|---|
| `dt` | 10 s | decision step |
| `max_steps` | 360 | episode length |
| `dynamics` | `"cw"` | `"cw"`, `"twobody"` or `"twobody_j2"` |
| `body`, `altitude_m`, `inclination` | `"earth"`, 500e3 m, 0 rad | the target's circular orbit |
| `action_mode` | `"normalized"` | `"normalized"`, `"accel"`, `"thrust"`, `"disc27"` or `"impulse_coast"` |
| `max_accel` | 0.02 m/s^2 | acceleration of a full command at the initial mass |
| `thrust_limit` | `"box"` | thruster layout, `"box"` or `"ball"` |
| `drive` | `"accel"` | hold the acceleration or the thrust over a burn |
| `isp`, `mass_kg`, `propellant_kg` | 220 s, 500 kg, whole mass | propulsion |
| `min_impulse` | 0 N s | minimum impulse bit |
| `delay_s` | 0 s | actuation delay |
| `execution` | none | thrust execution error |
| `sensor` | `PerfectNav()` | navigation sensor |
| `violation_samples` | 20 | constraint samples per full step |
| `observation_channels` | none | optional observation channels |
| `reward` | none | a `vbar.rewards.Reward`; none gives the scenario's legacy reward |
| `training` | off | allow training-only reset options; `run_episode` refuses it ([training.md](training.md)) |

`ApproachConfig` adds `keep_out_radius` (50 m), `hold_point` ((0, -100, 0) m), `capture_radius`
(5 m), `capture_speed` (0.05 m/s), `start_state`, `start_pos_range`, `start_vel_range`,
`terminate_on_violation` (off) and `reward_dv_weight` (1). `dt` must be positive.

## Assumptions and limits

- `info` carries the true state. A training loop or evaluator that hands `info` to an agent breaks
  the separation; `run_episode` passes only `agent_info()`.
- `scenario_id` names the scenario, the dynamics model and the thruster layout, not the sensor,
  delay or other options.
- Pending commands are reported in action units at the current mass, so in thrust mode with the
  constant-acceleration drive, or acceleration mode with the constant-thrust drive, they drift
  from the issued values as propellant burns.

## How it is tested

`tests/test_base_env.py` and `tests/test_agents.py`:

- Gymnasium's environment checker passes with each sensor and with every observation channel.
- The same seed gives the same episode, sensor noise included; noise never changes the true
  trajectory, and execution errors draw from their own stream.
- A delayed command splits the step, and the state matches the Clohessy-Wiltshire transition
  matrices applied piece by piece to 1e-9 m; with the pending-command channel the next state
  follows from the observation and the action alone, for delays of 3.7 and 25 s.
- Under the two-body model with J2, every thrust change (delay, run-out) starts a new
  integration.
- With noise on, observation minus truth has the sensor's mean and covariance and no correlation
  with the true state or past actions over 4000 steps.
- Agents receive only the view, a random stream and the time; an agent that rebuilds the episode's
  streams from its own generator recovers neither the sensor noise nor the start state, while the
  same attack on a stream spawned from the episode seed recovers both to rounding; the same
  `agent_seed` repeats a run.
- The PD agent flies the same trajectory in normalized and acceleration modes, in thrust mode
  under the constant-thrust drive, and with observation channels on; the LLM agent's fractions are
  converted to each mode.

## Example

```python
from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent
from vbar.spacecraft import GaussianNav

config = ApproachConfig(
    start_state=(-150.0, -400.0, 0.0, 0.0, 0.0, 0.0),
    sensor=GaussianNav(sigma_pos=0.5, sigma_vel=5e-4),
    delay_s=2.0,
    observation_channels=("t_remaining", "pending_commands"),
)
env = ApproachEnv(config)
print(env.agent_view().observation)  # x .. vz, t_remaining, then one pending-command slot
result = run_episode(env, PDAgent(), seed=0, agent_seed=1)
print(result["captured"], result["keep_out_violated"], round(result["dv_used_mps"], 2))
```
