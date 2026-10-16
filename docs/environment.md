# Approach environment

Code: `vbar.envs.approach` (`ApproachEnv`, `ApproachConfig`) and `vbar.runner`. This page
describes the current environment; it is being rebuilt on a shared base environment, and this
page changes with it.

## Task

Fly the chaser to a hold point and stop there without entering a keep-out sphere around the
target. Defaults: hold point $(0, -100, 0)$ m, 100 m behind the target on the V-bar; keep-out
radius 50 m; target on a circular 500 km orbit.

## Interface

Gymnasium API: `reset(seed=...)` returns `(obs, info)`; `step(action)` returns
`(obs, reward, terminated, truncated, info)`.

- **Action:** `Box(-1, 1, (3,))`. The action is clipped to [-1, 1] and multiplied by `max_accel`
  (0.02 m/s^2) per axis, giving a Hill-frame acceleration held for one step of `dt` = 10 s.
- **Observation:** the chaser's relative state $[x, y, z, \dot x, \dot y, \dot z]$, exact (there
  is no sensor model yet).
- **Episode:** at most `max_steps` = 360 steps (one hour). It ends early on capture, or on entering
  the keep-out zone if `terminate_on_violation` is set.

## Start states

With `start_state` unset, each reset draws position components uniformly from x in [-200, 200] m,
y in [200, 600] m and z in [-50, 50] m, and each velocity component from [-0.1, 0.1] m/s, using the
environment's seeded generator. These starts are ahead of the target, so the target lies between
the chaser and the hold point. A fixed `start_state` overrides the draw.

## Propagation

Each step integrates the Clohessy-Wiltshire equations numerically (DOP853, relative tolerance
1e-10, absolute 1e-9) with the mean motion of the 500 km orbit. The exact propagator and the
nonlinear models in [dynamics.md](dynamics.md) are not used by the environment yet.

## Keep-out check

Each step is sampled at `violation_samples` = 20 evenly spaced times, endpoints included, so a
pass through the keep-out sphere between step boundaries is caught. The environment tracks the
minimum range to the target, whether the zone was ever entered, and the number of steps in which
it was.

## Capture

After each step: distance to the hold point below `capture_radius` = 5 m and speed below
`capture_speed` = 0.05 m/s.

## Metrics

Returned in `info` after every step:

| Key | Meaning |
|---|---|
| `time_s` | simulated time (s) |
| `dist_to_hold_m` | distance to the hold point (m) |
| `range_to_target_m` | distance to the target (m) |
| `speed_mps` | relative speed (m/s) |
| `dv_used_mps` | delta-v used, $\sum_k \lvert\mathbf a_k\rvert\,\Delta t$ (m/s) |
| `min_range_to_target_m` | minimum range so far, including inside steps (m) |
| `keep_out_violated` | whether the keep-out zone was entered |
| `violation_steps` | number of steps that entered it |
| `captured` | whether the capture condition holds |

`run_episode(env, agent, seed)` returns these final values plus the agent's name, the seed, the
wall-clock time, and the full state and action histories, and `filter_interventions` when the
agent is wrapped in a safety filter. `run_many(env, agent, seeds)` runs one episode per seed.

## Reward

A shaped training signal for reinforcement learning, not the benchmark score. For step $k$, with
$d_k$ the distance to the hold point:

$$
r_k = \frac{d_{k-1} - d_k}{100} - w\,\lvert\mathbf a_k\rvert\,\Delta t - \mathbb 1[\text{keep-out entered in step } k] + 10\,\mathbb 1[\text{captured}]
$$

with $w$ = `reward_dv_weight` = 1.

## Agents

An agent has a `name`, a `reset(env)` method and an `act(obs, info)` method returning three numbers
in [-1, 1]. Provided:

- `PDAgent`: cancels the Clohessy-Wiltshire terms and applies a proportional-derivative law toward
  the hold point, $\mathbf a = -\mathbf a_{\text{CW}} - \omega_n^2(\boldsymbol\rho - \boldsymbol\rho_{\text{hold}}) - 2\zeta\omega_n\dot{\boldsymbol\rho}$,
  with $\omega_n$ = 0.01 rad/s and $\zeta$ = 1 by default, clipped to the thrust limit. It ignores
  the keep-out zone, so it is the unfiltered baseline.
- `DriftAgent`: no thrust.
- `LLMAgent(complete, name)`: formats a prompt with the frame, goal, keep-out radius, mean motion,
  current state, step length and thrust limit, calls `complete(prompt)`, and parses a JSON reply
  `{"ax": ..., "ay": ..., "az": ...}`. An unparseable reply is logged as a parse failure and the
  spacecraft coasts. Every call's prompt, reply, action, failure flag and latency go to `log`.
- `FilteredAgent(agent, safety_filter)` (`vbar.safety`): wraps an agent with a filter and counts
  the steps where the filter changed the command. Only the pass-through filter exists so far.
