# Lyapunov tracking baselines

Code: `CWTrackingAgent`, `LyapunovAgent`, `LyapunovConfig` and `DeclaredModel` in
`vbar.agents.lyapunov`. Hill frame ([frames.md](frames.md)), SI units.

Two classical baselines built on control-Lyapunov functions. Each cancels the natural relative
acceleration of a model, feeds forward the acceleration of the desired motion and closes a
spring and damper on the error, so on its own model the error $\mathbf e = \mathbf r -
\mathbf r_d$ obeys

$$
\ddot{\mathbf e} + K_2\,\dot{\mathbf e} + K_1\,\mathbf e = 0 .
$$

`CWTrackingAgent` cancels the Clohessy-Wiltshire terms and tracks any reference trajectory;
`LyapunovAgent` cancels the acceleration of a declared two-body model and holds the scenario's
hold point. Neither plans a path or handles the keep-out zone or any other constraint.

## How it is built

**CW tracking law.** Alfriend et al., *Spacecraft Formation Flying*, sec. 10.1.1, eq 10.8, gives
$\mathbf u = -(n^2 I + F_1)\,\mathbf e - N - k\,n\,\dot{\mathbf e}$. With the CW acceleration
of a state written $F_1\mathbf r + N(\mathbf v)$,

$$
F_1 = \operatorname{diag}(3n^2, 0, -n^2), \qquad
N(\mathbf v) = (2n\,\dot y,\; -2n\,\dot x,\; 0),
$$

terms derived here from the CW equations, the agent commands, for a reference
$(\mathbf r_d, \mathbf v_d, \mathbf a_d)$,

$$
\mathbf u = \mathbf a_d - F_1\mathbf r - N(\mathbf v) - n^2\,\mathbf e - k\,n\,\dot{\mathbf e},
\qquad \dot{\mathbf e} = \mathbf v - \mathbf v_d .
$$

On CW the error then obeys $\ddot{\mathbf e} + k n\,\dot{\mathbf e} + n^2\mathbf e = 0$, so
$K_1 = n^2$, $K_2 = k n$, and $k = 2$ is critically damped. For a natural CW reference
$\mathbf a_d = F_1\mathbf r_d + N(\mathbf v_d)$ and this is the book's form; for any other
reference the $\mathbf a_d$ term feeds its acceleration forward. `reference(t)` returns
$(\mathbf r_d, \mathbf v_d, \mathbf a_d)$ in m, m/s and m/s^2 at `t` = `info["time_s"]`; with
`reference=None` the target is the hold point at rest and the law equals
`PDAgent(wn=n, zeta=k/2)`. The same law is the hybrid Hill-frame Lyapunov law of Schaub and
Junkins, *Analytical Mechanics of Space Systems*, eq 13.266, on a circular target. $n$ is the
view's `n`.

**Cartesian Lyapunov law.** Schaub and Junkins, sec. 13.7, eq 13.229, with the hold point at
rest as the desired motion:

$$
\mathbf u = -\mathbf f(\mathbf x, t) - K_1(\mathbf r - \mathbf r_{\text{hold}}) - K_2\,\mathbf v,
$$

where $\mathbf f$ is the Hill-frame acceleration of the observed state with no thrust on the
declared model: the CW terms $F_1\mathbf r + N(\mathbf v)$ for `"cw"`, and
`vbar.dynamics.relative_acceleration` ([dynamics.md](dynamics.md)) for `"twobody"` and
`"twobody_j2"`, which needs the target's inertial state at time $t$. On the declared `"cw"` model
with $K_1 = n^2$ the law is `CWTrackingAgent(k=K2/n)`.

**Declared model knowledge.** The view gives the orbit rate but not the dynamics rung,
altitude, inclination or body. `LyapunovAgent` takes them from `LyapunovConfig.model`, a
`DeclaredModel` whose fields mirror the scenario's public `EnvConfig.dynamics`, `altitude_m`,
`inclination` and `body`, with the same defaults; the harness passes the scenario's values. At
`reset` the agent builds its own model with `make_dynamics` from those fields, starts its target
at the ascending node of the declared orbit (circular, or circular in mean elements under J2), as
the scenario's target starts, and at each act advances that target to `info["time_s"]`. It never
receives an environment object. A declared model that differs from the flown one, such as `"cw"`
declared on `"twobody_j2"`, is a model mismatch, and the offset it causes is part of the score.

**Command.** The acceleration is clipped per axis to `max_accel` (box layout) or scaled onto the
ball of radius `max_accel` keeping its direction (ball layout), then issued in the view's action
mode as `PDAgent` does: fractions, m/s^2, or newtons at the initial mass.

## Parameters and defaults

| Parameter | Default | Why |
|---|---|---|
| `CWTrackingAgent.k` | 2.0 | Critically damped error, no overshoot toward the target. |
| `CWTrackingAgent.reference` | None | The hold point at rest, the common baseline task. |
| `LyapunovConfig.k1` | 1e-4 1/s^2 | `PDAgent`'s $\omega_n^2$ at $\omega_n$ = 0.01 rad/s. |
| `LyapunovConfig.k2` | 0.02 1/s | `PDAgent`'s $2\zeta\omega_n$ at $\zeta = 1$. |
| `DeclaredModel.dynamics` | `"cw"` | Same as `EnvConfig`. |
| `DeclaredModel.altitude_m` | 500e3 m | Same as `EnvConfig`. |
| `DeclaredModel.inclination` | 0.0 rad | Same as `EnvConfig`. |
| `DeclaredModel.body` | `"earth"` | Same as `EnvConfig`. |

With the default gains `LyapunovAgent` on `"cw"` gives the same command as the default
`PDAgent`, so offsets and mutation tests compare like with like. The configs are frozen and
keyword-only.

## Assumptions and limits

- The CW tracking law assumes a circular target. The eccentric hybrid law, with time-varying
  stiffness and Coriolis matrices, is not implemented.
- The law is continuous; the environment holds each command for a step. On the matched two-body
  model the differential gravity is cancelled exactly, so the R-bar hold at (-100, 0, 0) m
  stays within 1e-6 m. Under J2 the cancelled acceleration turns while it is held, and the
  matched hold stays off by a zero-order-hold offset: below 2e-4 m in the tests, and in a
  separate computation at dt = 10 s the matched J2 offset is 6.5e-5 m.
- Declared `"cw"` on `"twobody"` drops the second-order gravity remainder, $-3n^2x^2/r$ on the
  radial axis, which the position gain balances at $3n^2x^2/(r K_1)$ = 5.343e-5 m for the R-bar
  hold at 500 km. Declared `"cw"` on `"twobody_j2"` leaves J2 uncancelled and holds several
  millimetres off.
- Saturation is handled by clipping, not by the design; the error equation holds only while the
  command is inside the limit.
- On the two-body models the agent integrates its own target orbit each step, which roughly
  doubles the cost of a step.
- Only the continuous action modes are supported: `reset` raises `ValueError` from
  `action_scale` for `"disc27"` and `"impulse_coast"`. An unknown declared dynamics raises
  `ValueError` at `reset`, and a gain that is not positive and finite raises `ValueError` at
  construction.
- The observation is used as the state; navigation noise passes straight into the command.
- `info` is read only for `time_s`.

## How it is tested

`tests/test_lyapunov.py`, at 500 km on the WGS 84 Earth with a 10 s step and a 0.02 m/s^2 box
limit unless stated:

- **Tracking law** (Alfriend et al., eq 10.8): with no reference the command equals
  `PDAgent(wn=n, zeta=k/2)` to 1e-12 for k = 2 and 0.7
  (`test_tracking_law_at_rest_on_the_hold_point_is_the_cw_cancelling_pd_law`); on a natural CW
  trajectory from zero error the command is below 1e-12 $n^2|\mathbf r|$, which a wrong sign on
  the Coriolis terms breaks
  (`test_tracking_a_natural_cw_trajectory_with_zero_error_needs_no_thrust`); the command equals
  the feedforward, cancellation and error terms summed independently
  (`test_tracking_command_feeds_forward_the_reference_and_damps_the_velocity_error`).
- **Critical damping**: from an error (10, -10, 5) m at rest, flown on the exact CW step at
  1 s, the error never grows and is 0.2032639 m after one orbit, against
  $15(1 + 2\pi)e^{-2\pi}$ = 0.20401 m for the continuous law
  (`test_tracking_error_from_rest_decays_critically_damped_over_one_orbit`).
- **Cartesian law** (Schaub and Junkins, eq 13.229): on the declared `"cw"` model it equals the
  tracking law and at defaults the PD law
  (`test_lyapunov_law_on_the_declared_cw_model_is_the_tracking_law`,
  `test_default_lyapunov_law_is_the_default_pd_law`); on `"twobody"` and `"twobody_j2"` at
  51.6 deg the command plus $\mathbf f$ from an independently propagated target equals
  $-K_1\mathbf e - K_2\mathbf v$ to 1e-9 of $|\mathbf f|$ at times up to 5000 s
  (`test_closed_loop_acceleration_is_the_linear_error_law_on_the_declared_model`); the declared
  altitude, not the view, sets the model
  (`test_declared_altitude_sets_the_model_whatever_the_view_says`).
- **Hold offsets** at (-100, 0, 0) m: matched `"twobody"` below 1e-6 m
  (`test_matched_two_body_model_holds_the_r_bar_point_to_rounding`); declared `"cw"` on
  `"twobody"` at $3n^2x^2/(r K_1)$ to 1%
  (`test_cw_model_on_two_body_motion_holds_off_by_the_second_order_gravity_remainder`); matched
  `"twobody_j2"` below 2e-4 m and declared `"cw"` on `"twobody_j2"` above 2e-3 m
  (`test_matched_j2_model_holds_the_r_bar_point_a_hundred_times_closer_than_the_cw_model`).
- **Approach**: `LyapunovAgent` with the matched J2 model completes the approach from
  (0, -400, 30) m on `"twobody_j2"` (`test_lyapunov_agent_completes_the_approach_on_the_j2_rung`).
- **Gains**: `CWTrackingAgent` and `LyapunovConfig` refuse a gain that is zero, negative,
  infinite or NaN (`tests/test_lyapunov_config.py`).
- **Agent**: commands agree in fractions across the three continuous modes; discrete and
  impulsive modes are refused; box clipping and ball scaling; frozen keyword-only configs whose
  `DeclaredModel` fields and defaults match `EnvConfig`; an unknown declared dynamics is refused;
  the sign check passes; reset wipes the declared target orbit; two agents from one config on
  interleaved environments match separate runs; both agents run on a hand-made view reading no
  `info` key but `time_s`; and `LyapunovAgent` holds no environment object and a fresh agent
  replays every command from a copy of the view, the observations and the times.

## Example

```python
import numpy as np

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.lyapunov import CWTrackingAgent, DeclaredModel, LyapunovAgent, LyapunovConfig

behind = (0.0, -400.0, 30.0, 0.0, 0.0, 0.0)
env = ApproachEnv(ApproachConfig(start_state=behind))
out = run_episode(env, CWTrackingAgent(), seed=1, agent_seed=2)
print(out["captured"], round(out["final_dist_to_hold_m"], 2))

# The J2 rung, with the scenario's public fields declared to the agent.
fields = {"dynamics": "twobody_j2", "inclination": np.radians(51.6)}
env = ApproachEnv(ApproachConfig(start_state=behind, max_steps=200, **fields))
agent = LyapunovAgent(LyapunovConfig(model=DeclaredModel(**fields)))
out = run_episode(env, agent, seed=1, agent_seed=2)
print(out["captured"], round(out["final_dist_to_hold_m"], 2))
```
