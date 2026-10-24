# LQR baseline

Code: `LQRAgent`, `LQRConfig` and the design functions `dlqr`, `sampled_data_cost`,
`delay_augmented`, `rate_augmented` and `steady_state_input` in `vbar.agents.lqr`. Hill frame
([frames.md](frames.md)), SI units.

`LQRAgent` is a discrete linear-quadratic regulator that drives the chaser to the scenario's hold
point and keeps it there. It is a classical baseline: the gain is designed once at `reset` from
the public scenario description (`AgentView`) and the law is a fixed matrix product at each step.
It plans no path and ignores the keep-out zone and every other constraint.

## How it is built

**Model.** The design model is the exact zero-order-hold step of the Clohessy-Wiltshire
equations, $\mathbf x_{k+1} = \Phi\,\mathbf x_k + \Gamma\,\mathbf u_k$, with
$\mathbf x = [x, y, z, \dot x, \dot y, \dot z]$ and $\mathbf u$ the commanded acceleration held
over one step, from `cw_discrete(view.n, view.dt)`
([transition-matrices.md](transition-matrices.md)).

**Cost.** `LQRConfig` sets three scales: a position error `pos_scale_m` ($p$), a velocity error
`vel_scale_mps` ($v$) and an acceleration `accel_scale_mps2` ($a$, the view's `max_accel` when
None) that each cost one unit. With $Q_c = \operatorname{diag}(p^{-2} I_3, v^{-2} I_3)$ and
$R_c = a^{-2} I_3$ there are two stage costs:

- `stage_cost="sampled"` (default) minimizes the continuous cost
  $\int (\mathbf x^T Q_c \mathbf x + \mathbf u^T R_c \mathbf u)\,dt$ exactly over held steps.
  One step contributes $\mathbf x_k^T Q_d \mathbf x_k + \mathbf u_k^T R_d \mathbf u_k +
  2\mathbf x_k^T S_d \mathbf u_k$, computed by Van Loan's method (Rawlings, Mayne and Diehl,
  *Model Predictive Control*, 2nd ed., sec. 8.5.2): with
  $A_z = \begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix}$ and $Q_z = \operatorname{diag}(Q_c, R_c)$,

  $$
  E = \exp\left(\begin{bmatrix} -A_z^T & Q_z \\ 0 & A_z \end{bmatrix}\Delta t\right), \qquad
  \begin{bmatrix} Q_d & S_d \\ S_d^T & R_d \end{bmatrix} = E_{22}^T E_{12},
  $$

  symmetrized, in coordinates scaled by the cost weights so the exponential mixes entries of one
  size. The gain then does not depend on `dt` beyond the sampling itself. At `reset` the agent
  checks that the step blocks of $\exp(A_z \Delta t)$ equal $(\Phi, \Gamma)$ from the closed form
  to 1e-12 relative, so the cost is built for the model that is flown.
- `stage_cost="discrete"` sums $\mathbf x_k^T Q_c \mathbf x_k + \mathbf u_k^T R_c \mathbf u_k$
  once per step, the textbook discrete LQR. Its gain changes with `dt`.

**Gain.** $P$ is the stabilizing solution of the discrete algebraic Riccati equation with cross
term $S$, solved by `scipy.linalg.solve_discrete_are` in coordinates that give the diagonals of
$Q$ and $R$ unit size, and

$$
K = (R + \Gamma^T P \Gamma)^{-1}(\Gamma^T P \Phi + S^T), \qquad \mathbf u = -K\mathbf x.
$$

This is the form of Rawlings, Mayne and Diehl, sec. 1.5.1; the discrete gain printed by Alfriend
et al., *Spacecraft Formation Flying*, eqs 3.46 and 3.47, lacks the $(R + \Gamma^T P \Gamma)^{-1}$
factor. `dlqr` raises `ValueError` when the Riccati residual exceeds 1e-10 of $P$ or the closed
loop $\Phi - \Gamma K$ has spectral radius 1 or more.

**Hold point.** The target is $\mathbf x_s = (\mathbf r_{\text{hold}}, 0)$ from
`view.goal["hold_point"]`, held by the constant input $\mathbf u_s$ that solves
$\Gamma\mathbf u_s = (I - \Phi)\mathbf x_s$ (Rawlings, Mayne and Diehl, sec. 1.5.1, eq 1.41).
On CW that is $-3n^2 x$ on the radial axis and $n^2 z$ on the normal axis; a V-bar point needs
none. The command is $\mathbf u = \mathbf u_s - K(\mathbf z - \mathbf z_s)$. Regulating the
deviation without $\mathbf u_s$ leaves a steady offset at off-V-bar points. `reset` raises
`ValueError` when $\mathbf u_s$ exceeds `max_accel` (per axis for the box layout, in norm for
the ball).

**Delay.** With `observation_channels=("pending_commands",)` and an actuation delay of $d$ whole
steps, the observation carries $d$ slots `pending_j_t, pending_j_x, pending_j_y, pending_j_z`
([environment.md](environment.md)), and slot $j$ holds $\mathbf u_{k-d+j}$, the oldest first.
The agent counts the slots from `view.observation` and designs on the augmented state
$\mathbf z_k = (\mathbf x_k, \mathbf u_{k-d}, \dots, \mathbf u_{k-1})$ (`delay_augmented`;
Rawlings, Mayne and Diehl, sec. 1.2.5 and exercise 1.61):

$$
\Phi_a = \begin{bmatrix} \Phi & \Gamma & 0 & \cdots \\ 0 & 0 & I & \\ & & & \ddots \\
0 & \cdots & & 0 \end{bmatrix}, \qquad
\Gamma_a = \begin{bmatrix} 0 \\ \vdots \\ I \end{bmatrix}.
$$

The state weight pairs $\mathbf x_k$ with the input acting on it, $\mathbf u_{k-d}$ (the cross
term $S$ of the sampled cost sits there), and $R$ weights the decision $\mathbf u_k$; shifting
the input weight by $d$ steps changes the infinite sum only by the commands already issued. The
channel echoes each command in the action units of the mode at the time it was issued, which
are the units the agent sent it in, so the agent multiplies by `max_accel / action_scale(view)`
to get the acceleration back. Each act checks that every nonzero wait is a whole number of
steps to 1e-6 and raises `ValueError("pending commands are not aligned to the decision step")`
otherwise. Without the channel the agent assumes no delay.

**Rate weight.** `rate_weight` $w$ adds $(\mathbf u_k - \mathbf u_{k-1})^T R_\Delta
(\mathbf u_k - \mathbf u_{k-1})$ per step with $R_\Delta = w\,a^{-2} I_3$ (Rawlings, Mayne and
Diehl, exercise 1.25). Without delay, `rate_augmented` appends $\mathbf u_{k-1}$ to the state and
the agent keeps its own last command, zero at the start of an episode; with delay,
$\mathbf u_{k-1}$ is already the newest slot and the weight is added there. The default 0 keeps
the plain design.

**Command.** The acceleration is clipped per axis to `max_accel` (box layout) or scaled onto the
ball of radius `max_accel` keeping its direction (ball layout), then issued in the view's action
mode as `PDAgent` does: fractions, m/s^2, or newtons at the initial mass.

`LQRConfig.alfriend(n)` returns the scales 1 m, $n$ m/s and $n^2$ m/s^2, the weights
$Q = \operatorname{diag}(1, 1, 1, n^{-2}, n^{-2}, n^{-2})$, $R = I/n^4$ of Alfriend et al.,
sec. 10.1.2, under which the gains scale with the orbit and carry over between bodies.

**Shared design.** `regulator_model(view, config)` builds everything above except the command
and returns it as a frozen `RegulatorModel`: the augmented $\Phi$ and $\Gamma$, the stage
weights $Q$, $R$ and $S$ on the deviation from $(\mathbf z_s, \mathbf u_s)$, the gain $K$, the
Riccati solution $P$, $\mathbf x_s$, $\mathbf u_s$, $\mathbf z_s$, the pending-command slot
indices and whether the agent keeps its own last command. `LQRAgent.reset` calls it, and the MPC
baselines reuse it for their stage and terminal cost, so an MPC and `LQRAgent` with the same
`LQRConfig` give the same command wherever no constraint binds. It refuses the same views as
`reset`, with the same messages.

## Parameters and defaults

`LQRConfig` is a frozen, keyword-only dataclass.

| Parameter | Default | Reason |
|---|---|---|
| `pos_scale_m` | 10 m | The position weight $10^{-2}$ m$^{-2}$ of the known-answer tests below |
| `vel_scale_mps` | 0.03 m/s | Sets the slowest closed-loop mode: when $p/v$ is long next to the step it is that mode's time constant, 333 s here, so the R-bar hold of `approach-rbar` is reached and held within its 3600 s; at 0.01 m/s (1000 s) its validation seeds end 6 to 10 m out |
| `accel_scale_mps2` | `None`: the view's `max_accel` | The weights follow the scenario's thrust |
| `stage_cost` | `"sampled"` | The gain does not depend on `dt` beyond the sampling |
| `rate_weight` | 0 | The plain design |

## Assumptions and limits

- The design model is CW about a circular target. On the two-body rungs the unmodelled
  differential gravity and J2 terms act as a disturbance the regulator rejects only in part.
- Constraints are not part of the design. With aggressive weights the regulator flies close to
  a straight line and can pass through the keep-out zone: at scales of 1 m and 1 m/s the
  approach from 400 m behind the target, used in the delay test, captures without a mission
  success for that reason.
- Saturation is handled by clipping the unconstrained law, which is not the constrained optimum
  and gives no stability guarantee from far away; the design assumes the linear region near the
  hold point.
- The delay must be a whole number of steps; the split-step input model of a fractional delay is
  not implemented, and the agent refuses one at the first act whose waits show it.
- Only the continuous action modes are supported: `reset` raises `ValueError` from
  `action_scale` for `"disc27"` and `"impulse_coast"`.
- The agent refuses at reset goals other than a hold or a berthing, and goals without a hold
  point, with a `ValueError`.
- The observation is used as the state. Under navigation noise the noise passes straight into
  the command.
- `info` is not read; the agent needs no environment object.

## How it is tested

`tests/test_lqr.py` (design functions) and `tests/test_lqr_agent.py` (agent), at 500 km on the
WGS 84 Earth with a 10 s step and a 0.02 m/s^2 box limit unless stated:

- **Hold-point input** (Rawlings, Mayne and Diehl, sec. 1.5.1, eq 1.41): the R-bar hold at
  (-100, 0, 0) m needs $3n^2 \cdot 100$ = 3.6749087912e-4 m/s^2 at steps of 1, 10 and 60 s, to
  1e-10 relative, and a V-bar point none
  (`test_r_bar_hold_needs_three_n_squared_times_the_offset`, `test_v_bar_hold_needs_no_input`).
  With the discrete cost at scales (10 m, 0.01 m/s), the loop without $\mathbf u_s$ settles at
  (-3.702, 0.041, 0) m from the hold point and with it within 1e-9 m
  (`test_regulating_to_the_hold_point_without_feedforward_leaves_an_offset`,
  `test_feedforward_removes_the_hold_point_offset`). Flown in the environment, the agent holds
  the R-bar point to below 1e-6 m over 360 steps with either stage cost, and with $\mathbf u_s$
  forced to zero at scales (10 m, 0.01 m/s) it ends 3.702 m away after 1080 steps
  (`test_target_input_holds_the_r_bar_point_to_rounding`,
  `test_regulating_without_target_input_settles_at_the_known_offset`).
- **Delay** (sec. 1.2.5, exercise 1.61): designing for no delay and flying with a one-step delay
  gives closed-loop spectral radii 1.3921, 1.0057 and 0.9012 at scales (1 m, 1 m/s),
  (100 m, 0.01 m/s) and (100 m, 1 m/s), and the augmented design recovers the undelayed radius
  to 1e-4 (`test_one_step_delay_ignored_by_the_design_changes_the_closed_loop_radius`,
  `test_delay_augmented_design_recovers_the_undelayed_radius`). In the environment with a 10 s
  delay from (0, -400, 30) m the agent captures within 300 steps where the same gains designed
  without the delay do not
  (`test_delay_augmented_design_captures_where_the_undelayed_design_does_not`); a 15 s delay is
  refused (`test_delay_that_is_not_a_whole_number_of_steps_is_refused`). The command with one
  pending slot equals the augmented law in every continuous mode
  (`test_one_step_delay_command_is_the_augmented_lqr_law_in_every_mode`).
- **Sampled cost** (sec. 8.5.2): with $Q_c = \operatorname{diag}(10^{-4} I_3, I_3)$,
  $R_c = 10^3 I$ and $\mathbf x_0$ = (50, -80, 10 m, 0.02, -0.01, 0.005 m/s),
  $\mathbf x_0^T P \mathbf x_0$ = 116.65796914 at 10 s and 122.99249126 at 60 s, to 1e-10
  relative (`test_sampled_data_cost_gives_the_exact_zoh_optimum`). The Van Loan blocks equal a
  Gauss-Legendre quadrature of the closed forms to 1e-9 at steps from 1e-6 s to 1e4 s, compose
  over two half steps, stay positive semidefinite, and give a stable design at every step
  (`test_van_loan_cost_matches_gauss_legendre_quadrature` and the tests after it).
- **Disturbance offset** (sec. 5.5): a constant unmodelled acceleration (2e-7, 1e-6, -3e-7)
  m/s^2 leaves the offset (1.288762e-4, 5.842058e-4, -1.759851e-4) m
  (`test_constant_disturbance_leaves_the_known_steady_offset`).
- **Gain form** (Alfriend et al., eqs 3.46 and 3.47 as corrected above, and sec. 10.1.2): the
  Riccati solution satisfies the equation to 1e-10 of $P$, the gain closes a stable loop with
  $\mathbf u = -K\mathbf x$, and with the weights of `LQRConfig.alfriend` the gains for two
  orbits whose steps cover the same angle differ by $(n_2/n_1)^2$ on position and $n_2/n_1$ on
  velocity to 1e-9 (`test_gains_scale_with_mean_motion_when_the_step_scales_with_the_period`).
- **Agent**: the command equals $\mathbf u_s - K(\mathbf x - \mathbf x_s)$ from the design
  functions to 1e-9 for both stage costs; commands agree in fractions across the three
  continuous modes to 1e-12; box clipping and ball scaling; a rate weight makes the command
  depend on the previous one and lowers the summed squared command change on the approach;
  the sign check passes for each stage cost, with a rate weight and with delay; reset wipes the
  previous episode; two agents from one config on interleaved environments match separate runs;
  and the agent runs on a hand-made view and observation reading no `info` key but `time_s`.

## Example

```python
from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.lqr import LQRAgent, LQRConfig

behind = (0.0, -400.0, 30.0, 0.0, 0.0, 0.0)
env = ApproachEnv(ApproachConfig(start_state=behind))
out = run_episode(env, LQRAgent(), seed=1, agent_seed=2)
print(out["captured"], round(out["final_dist_to_hold_m"], 2), round(out["dv_used_mps"], 3))

# A 10 s actuation delay with the pending-commands channel: the agent designs on the
# delay-augmented model.
delayed = ApproachConfig(
    start_state=behind, delay_s=10.0, observation_channels=("pending_commands",)
)
agent = LQRAgent(LQRConfig(stage_cost="discrete", pos_scale_m=5.0))
out = run_episode(ApproachEnv(delayed), agent, seed=1, agent_seed=2)
print(out["captured"], round(out["final_dist_to_hold_m"], 2))
```
