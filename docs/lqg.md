# LQG baseline

Code: `LQGAgent`, `LQGConfig` and the design functions `kalman_gain` and `bias_detectable` in
`vbar.agents.lqg`. Hill frame ([frames.md](frames.md)), SI units.

`LQGAgent` is a steady-state Kalman filter on the Clohessy-Wiltshire model feeding the LQR
hold-point law of [lqr.md](lqr.md): the regulator acts on the filtered state instead of the raw
measurement. It is a classical baseline for noisy navigation, never an oracle: it knows the sensor
law only as declared in its configuration, and nothing it estimates is scored.

## How it is built

**Model.** The filter state is the Hill state $\mathbf x = [x, y, z, \dot x, \dot y, \dot z]$
followed by $n_d$ measurement biases $\mathbf b$, with

$$
\mathbf x_{k+1} = \Phi\,\mathbf x_k + \Gamma(\mathbf u_k + \mathbf w_k), \qquad
\mathbf b_{k+1} = \mathbf b_k + \mathbf e_k, \qquad
\mathbf y_k = \mathbf x_k + C_d\,\mathbf b_k + \mathbf v_k,
$$

$(\Phi, \Gamma)$ the zero-order-hold CW step from `cw_discrete(view.n, view.dt)`
([transition-matrices.md](transition-matrices.md)), $\mathbf u_k$ the commanded acceleration and
$C_d$ a unit column per declared bias on its channel.

**Declared sensor law.** The measurement noise is $\mathbf v_k \sim N(0, R_v)$ with
$R_v = \operatorname{diag}(\sigma_p^2 I_3, \sigma_v^2 I_3)$, from `LQGConfig.sigma_pos_m` and
`sigma_vel_mps`. The defaults, 1 m and 1 mm/s, are those of `GaussianNav`, the `:nav1` track
([sensors.md](sensors.md)). The sigmas are declared model knowledge: whoever runs the agent sets
them from the public scenario parameters, and the agent never reads the environment's sensor. A
filter told the wrong sigmas is mistuned, and that mismatch is part of what the episode scores.

**Process noise.** $\mathbf w_k \sim N(0, \sigma_a^2 I_3)$ is a white acceleration held over each
step, $\sigma_a$ = `process_sigma_mps2` (default 1e-6 m/s^2), so the process covariance per step is
$Q_d = \sigma_a^2\,\Gamma\Gamma^T$. It must be positive: every CW mode lies on the unit circle and
does not decay, and with no process noise on a mode the steady-state gain on it is zero, so the
filter would stop correcting it. Each bias walks with per-step sigma `bias_walk` times its
channel's sigma (default 1e-3), for the same reason. `reset` raises
`ValueError("process noise must be positive: with none the filter gain collapses")` for a zero
process noise, or a zero walk with bias states, and `kalman_gain` checks by the PBH test that the
process noise reaches every mode on or outside the unit circle and that the measurements see each
one.

**Bias states.** `bias_states` lists the measurement channels with a constant bias to estimate,
from `"x"`, `"z"`, `"vx"`, `"vy"` and `"vz"`. An along-track position bias `"y"` is refused when
the config is built: a constant $y$ offset is itself an equilibrium of the CW equations, so a
biased measurement of a chaser at rest cannot be told from an unbiased one of a chaser elsewhere
on the V-bar. By Lemma 1.8 of Rawlings, Mayne and Diehl, *Model Predictive Control*, 2nd ed.,
the augmented model is detectable if and only if $(\Phi, C)$ is and

$$
\operatorname{rank}\begin{bmatrix} I - \Phi & 0 \\ C & C_d \end{bmatrix} = 6 + n_d,
$$

which a $y$ bias leaves at 6 of 7 (`bias_detectable`).

**Gain.** The steady-state predicted covariance $P^-$ solves the filter Riccati equation, the
dual of the LQR one, by `scipy.linalg.solve_discrete_are` in coordinates scaled by the
measurement sigmas (the covariances span many decades in SI units). Then

$$
L = P^- C^T (C P^- C^T + R_v)^{-1}, \qquad
P^+ = (I - LC)\,P^-(I - LC)^T + L R_v L^T,
$$

the updated covariance in Joseph form, which stays symmetric and positive semidefinite in
rounding (Crassidis and Junkins, *Optimal Estimation of Dynamic Systems*, 2nd ed., sections 3.3.2
and 3.3.4). `kalman_gain` raises `ValueError` when the Riccati residual exceeds 1e-10 of $P$.

**Filter.** At each decision the agent updates and then predicts:

$$
\hat{\mathbf x}_k^+ = \hat{\mathbf x}_k^- + L(\mathbf y_k - C\hat{\mathbf x}_k^-), \qquad
\hat{\mathbf x}_{k+1}^- = \Phi\hat{\mathbf x}_k^+ + \Gamma\,\mathbf u_k^{\text{eff}},
$$

with the gain times the innovation added. The first prediction is the first measurement with zero
biases. $\mathbf u_k^{\text{eff}}$ is the command that acts over the coming step: the command just
issued, or with an actuation delay and the pending-commands channel
([environment.md](environment.md)) the oldest pending slot, converted from action units with
`max_accel / action_scale(view)`. Predicting with the command just issued under a delay puts the
estimate one command ahead of the plant.

**Regulator.** The command is the `LQRAgent` law with the weights of `LQGConfig.regulator`
applied to the state estimate $\hat{\mathbf x}_k^+$ (the first six entries) and the pending slots
as observed: $\mathbf u = \mathbf u_s - K(\mathbf z - \mathbf z_s)$, delay-augmented when the
channel is present, clipped to `max_accel` and issued in the view's action mode. The bias
estimates correct the measurement, not the state the regulator drives.

`estimate` (the state and biases after the last update) and `covariance` ($P^+$) are attributes
for tests only; the recorder never reads them.

## Assumptions and limits

- A baseline, never an oracle. Its estimate is not a navigation reference, and its normalized
  estimation error is reported by the tests only, not scored.
- The filter is linear and steady state: no transient covariance from a prior, no
  range-dependent noise, no correlated (Gauss-Markov) errors. On sensors whose noise grows with
  range, or with time-correlated errors, it is mistuned by construction.
- The design model is CW about a circular target; on the two-body rungs the unmodelled terms
  enter as process noise the filter does not account for.
- No certified constraint satisfaction under Gaussian navigation error: the regulator ignores the
  keep-out zone, and the estimate error is unbounded in the tails.
- The delay must be a whole number of steps, as for `LQRAgent`.
- `LQGConfig` raises `ValueError` at construction for a sigma that is not positive and finite and
  for a process noise or bias walk that is negative or not finite (`tests/test_baseline_inputs.py`);
  a zero process noise is accepted there and refused at `reset`, and so is a zero walk when there
  are bias states. Without bias states the walk is not used.
- The regulator refuses at reset goals other than a hold or a berthing, and goals without a
  hold point, with a `ValueError`.
- Only the continuous action modes are supported: `reset` raises `ValueError` from
  `action_scale` for `"disc27"` and `"impulse_coast"`.
- `info` is passed to the regulator, which does not read it; the agent needs no environment
  object.

## How it is tested

`tests/test_lqg.py` (design functions) and `tests/test_lqg_agent.py` (agent), at 500 km on the
WGS 84 Earth with a 10 s step and a 0.02 m/s^2 box limit unless stated:

- **Bias detectability** (Rawlings, Mayne and Diehl, chapter 1, Lemma 1.8): with the full state
  measured, a bias on any one channel but $y$ is detectable, a $y$ bias is not, biases on all six
  channels together are not, and a constant acceleration on any axis is
  (`test_constant_bias_on_one_channel_is_detectable_except_along_track_position` and the tests
  after it). The agent refuses a $y$ bias state
  (`test_along_track_position_bias_is_refused`).
- **Gain and covariance** (Crassidis and Junkins, sections 3.3.2 and 3.3.4): the steady-state
  gain is the limit of the Joseph-form covariance recursion, with and without a bias state; the
  steady state is a fixed point of one filter cycle; both covariances are symmetric and positive
  definite at the `:nav1` sigmas, at cheap measurements and at loud process noise; zero process noise and process noise that misses an
  oscillating axis are refused
  (`test_steady_state_gain_is_the_limit_of_the_joseph_recursion` and the tests after it).
- **Innovation sign** (Alfriend et al., *Spacecraft Formation Flying*; the printed innovation
  sign is a misprint): after two acts the estimate equals the prediction plus the gain times the
  innovation, from the design functions, to 1e-9 per channel sigma
  (`test_update_adds_the_gain_times_the_innovation`).
- **Estimation on `GaussianNav(1.0, 1e-3)`**: over three orbits from (0, -400, 30) m the RMS
  position error of the estimate over the last orbit is below 1 m, against an expected
  $\sqrt{3} \times 1$ m = 1.73 m for the raw fix (three axes of 1 m noise), and the mean normalized estimation error over the last two orbits is at most
  $6 + 3\sqrt{12/N}$, the chi-square mean of six states plus three standard deviations
  (`test_estimate_error_over_the_last_orbit_is_well_inside_the_sensor_noise`,
  `test_normalized_estimation_error_is_at_most_its_chi_square_mean`). With a radial bias state at
  a walk of 1e-2, a drawn 5 m-sigma radial position bias is estimated within 0.5 m after three
  orbits (`test_radial_measurement_bias_is_estimated_within_half_a_metre`).
- **Regulator**: with sigmas of 1e-9 on exact navigation the commands equal the `LQRAgent`'s to
  1e-9 relative for both stage costs
  (`test_filter_that_trusts_exact_measurements_flies_the_lqr_commands`); with a 10 s delay and
  the pending channel, predicting with the command in effect captures on exact and noisy
  navigation (`test_predicting_with_the_command_in_effect_captures_under_a_one_step_delay`), and
  a 15 s delay is refused.
- **Agent**: commands agree in fractions across the three continuous modes to 1e-12; the
  discrete and impulsive modes are refused; the sign check passes with and without delay; reset
  wipes the previous episode; two agents from one config on interleaved environments match
  separate runs; and the agent runs on a hand-made view and observation reading no `info` key
  but `time_s`.

## Example

```python
from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.lqg import LQGAgent, LQGConfig
from vbar.spacecraft import GaussianNav

behind = (0.0, -400.0, 30.0, 0.0, 0.0, 0.0)
noisy = ApproachConfig(start_state=behind, max_steps=600, sensor=GaussianNav(1.0, 1e-3))
out = run_episode(ApproachEnv(noisy), LQGAgent(), seed=1, agent_seed=2)
print(out["captured"], round(out["final_dist_to_hold_m"], 2), round(out["dv_used_mps"], 3))

# A radial position bias, declared and estimated as a filter state. The script reads the drawn
# bias from the environment to compare; the agent never sees it.
biased = ApproachConfig(start_state=behind, sensor=GaussianNav(1.0, 1e-3, bias_pos=2.0))
env = ApproachEnv(biased)
agent = LQGAgent(LQGConfig(bias_states=("x",), bias_walk=1e-2))
run_episode(env, agent, seed=1, agent_seed=2)
print(round(float(agent.estimate[6]), 2), round(float(env.sensor.bias[0]), 2))
```
