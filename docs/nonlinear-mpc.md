# Nonlinear MPC

Code: `shooting_linearization`, `NMPCConfig` and `NMPCAgent` in `vbar.agents.nmpc`. Hill frame
([frames.md](frames.md)), SI units.

A hold-point MPC whose prediction model is the declared nonlinear relative motion, two-body with
or without J2, rather than the CW step. The horizon is cut into one segment per step with a free
node state at the start of each segment (direct multiple shooting, Rawlings, Mayne and Diehl,
*Model Predictive Control*, 2nd ed., sections 8.5.2 and 8.6.3). Each decision linearizes the
declared model along the previous plan shifted by one step and solves one horizon QP: one
Gauss-Newton step per decision, the real-time iteration of section 8.9.2, eq. 8.60. Stage cost,
terminal cost, thrust box and keep-out rows are those of the linear MPC ([mpc.md](mpc.md)), and
the QP is solved by `solve_horizon` of `vbar.solvers` ([solvers.md](solvers.md)).

## How it is built

**Declared model.** The agent never reads the environment's dynamics. `NMPCConfig.model` is a
`DeclaredModel` ([lyapunov.md](lyapunov.md)) with the public scenario fields: dynamics ("cw",
"twobody" or "twobody_j2"), altitude, inclination and body. The target's argument of latitude
at the episode start is not published by the scenarios, so it is a declared assumption,
`NMPCConfig.argument_of_latitude`. Circular two-body relative motion does not depend on it; with
J2 it does, and the log carries the declared value as `declared_phase` on "twobody_j2".

**Shooting linearization.** `shooting_linearization(model, t0, nodes, controls, dt,
argument_of_latitude=0.0)` takes nodes $\mathbf s_0, \dots, \mathbf s_N$ and controls
$\mathbf q_0, \dots, \mathbf q_{N-1}$; segment $k$ starts at $\mathbf s_k$ at time
$t_0 + k\,\Delta t$ and holds the Hill acceleration $\mathbf q_k$ for $\Delta t$. With
$\mathbf F_k$ the state the declared model reaches at the segment's end,

$$
A_k = \frac{\partial \mathbf F_k}{\partial \mathbf s_k},\qquad
B_k = \frac{\partial \mathbf F_k}{\partial \mathbf q_k},\qquad
\mathbf c_k = \mathbf F_k - A_k \mathbf s_k - B_k \mathbf q_k ,
$$

so the linearized segment $\mathbf s_{k+1} = A_k \mathbf s_k + B_k \mathbf q_k + \mathbf c_k$ is
exact at the nodes. On "cw" $A_k$ and $B_k$ are the exact zero-order-hold step and
$\mathbf c_k = 0$. On "twobody" and "twobody_j2" the target starts on the declared circular orbit
(mean elements with J2) at the declared argument of latitude and coasts alone to each node time,
so the node times do not depend on the plan; the chaser is placed at $\mathbf s_k$ beside it,
$A_k$ and $B_k$ come from the variational equations of the same model
(`TwoBodyDynamics.transition_matrices`, Montenbruck and Gill, *Satellite Orbits*, section 7.2)
and $\mathbf F_k$ from `advance`. No finite differences of the adaptive integrator are taken:
they are noisy at the square root of its tolerance (Rawlings, Mayne and Diehl, section 8.4.6).

**Real-time iteration.** Each decision at time $t$ has two phases.

1. *Preparation*, which needs no new observation once a plan exists: the linearization above at
   the shifted plan from $t_0 = t$, and the QP of `mpc_problem` with its keep-out rows linearized
   at the same plan and its CW dynamics replaced by $A_k$, $B_k$, $\mathbf c_k$. The continuity
   residual of that plan, $\max_k |\mathbf F_k - \mathbf s_{k+1}|_\infty$ (m and m/s), is logged
   as `continuity`.
2. *Feedback*: the QP from the observed state $\mathbf x_0$, warm-started at the shifted plan.
   When that start gives "iteration_cap" or "solver_failed" the same QP is solved again from the
   solver's own start, and `cold_retry` is logged: a warm start with an input on its bound can
   fail where a cold start converges. "infeasible_certified" is a property of the problem and is
   not retried.

The controls applied or planned are rolled out from $\mathbf x_0$ through the linearized
segments. The second state of that rollout is the predicted next state (`predicted`) and the
first node of the next plan; the plan is shifted by one step and its last move is the LQR law on
CW, clipped to the box, with the CW step to its last node. The shifted plan is kept for time
$t + \Delta t$ and dropped when the next decision comes at another time (by more than $10^{-9}$
of a step).

At the first decision, and after a decision with no plan, the linearization point and warm start
come from the observed state: the plan of the linear MPC on CW (`mpc_solve` on `mpc_problem` with
the keep-out rows at the current position), or the clipped LQR rollout on CW when that plan
fails. The preparation then waits for the observation like the rest. `reference` in the log is
"shifted", "linear_mpc" or "lqr".

**Fallbacks.** A converged QP gives level 0 and its first move, clipped to the box. Otherwise the
shifted plan's first move is applied (level 1) when a plan exists, and the LQR law clipped to the
thrust limit (level 2, the declared backup) when none does; after level 2 the next decision
starts cold. An unconverged iterate is never applied: `solve_horizon` returns none.
`fallbacks[level]` counts every decision.

**Log.** One entry per decision: `time_s`; `status` of the solve applied and `warm_status` of the
warm-started one; `cold_retry`, `warm_start`, `iterations` (both solves), `solver` ("rti") and
`tolerance` (the larger interior-point tolerance); `level` and `fallback`; `reference`;
`max_slack_m`, the largest keep-out slack of a converged plan, a positive value being a predicted
violation; `continuity`; `predicted`; `declared_phase`; `prep_s` and `feedback_s` (s), the
preparation and feedback times of the split above, which the solver metrics report
([metrics.md](metrics.md)). The log and `fallbacks` are the agent's own bookkeeping; the recorder
and metrics never read them as scores.

## Parameters and defaults

`NMPCConfig` is a frozen, keyword-only dataclass; a non-finite `argument_of_latitude` raises
`ValueError`.

| Parameter | Default | Meaning |
|---|---|---|
| `mpc` | `MPCConfig()` | Weights, horizon (20 steps), terminal ingredients, keep-out rows, penalty and solver settings of [mpc.md](mpc.md); `side` and `side_hysteresis` are not read |
| `model` | `DeclaredModel()` | Declared model: "cw", 500 km, inclination 0, Earth; the catalogue scenarios fly 51.6 deg, so pass the scenario's values |
| `argument_of_latitude` | 0.0 | Declared argument of latitude of the target at the episode start (rad) |

Fixed in the module: the retry statuses ("iteration_cap", "solver_failed") and the schedule
tolerance of $10^{-9}$ of a step. `NMPCAgent(config=None, name="nmpc")`.

An unknown declared dynamics is refused when the `DeclaredModel` is built. `reset` raises
`ValueError` when the goal is not a hold or a berthing, the action mode takes no continuous command
("disc27", "impulse_coast"), the thrust layout is unknown, the regulator has a rate weight, the
observation carries pending commands, or the horizon problem cannot be built (the terminal set and
in-step bound errors of `mpc_problem`, checked at reset rather than mid-episode).

## Assumptions and limits

- One Gauss-Newton step per decision: the QP solution is not the optimum of the nonlinear
  problem, and the iteration converges over decisions only while the shifted plan stays close to
  it. Shifting perturbs the contraction of the real-time iteration (Rawlings, Mayne and Diehl,
  section 8.9.2).
- The keep-out rows and the positions sampled within each step are those of the CW step about
  each node, linearized at the shifted plan; on two-body the in-step positions differ by the
  second-order gravity terms CW drops: 400 m behind the target, a declared CW model mispredicts
  the two-body motion by more than 5e-7 m per 10 s step (asserted below). The rows are a
  prediction only.
- The held step is not certified against the keep-out sphere: no bound on the range between
  samples of the step actually flown is computed, and `max_slack_m` shows a predicted violation
  only.
- The terminal cost is the CW DARE cost and the appended last move is the CW LQR law: an
  approximation on two-body with no stability claim. The appended segment is stepped on CW, so
  the continuity residual of a shifted plan includes the CW error of its last segment.
- No passing side is chosen. The first plan comes from the linear MPC with the keep-out rows at
  the current position, so a straight path through the sphere is not routed around it.
- The prediction is as good as the declared model. Declared CW on two-body motion mispredicts
  by the gravity remainder; on J2 a misdeclared argument of latitude mispredicts by more than
  1e-5 m per step at a quarter orbit off (asserted below).
- A ball limit is replaced in the plan by its inscribed box $a_{\max}/\sqrt 3$ per axis; the
  level 2 backup scales onto the ball as the LQR agent does.
- A command delay (pending commands in the observation) and a rate weight are refused at reset:
  the shooting model is the six-element Hill state.
- Cost per decision is one variational and one segment integration per step of the horizon,
  plus the target's coast from the episode start, which is replayed at every call, and one or
  two interior-point solves.

## How it is validated

`tests/test_nmpc.py`, on a 500 km circular orbit of the WGS 84 Earth at 51.6 deg with 10 s steps
and a 0.02 m/s^2 box limit per axis unless stated. Jacobians are compared nondimensionally
(velocities in metres per radian of orbit). Independent computations are a separate shooting
code that places a fresh model at each node, a dense KKT solve in the test, or central
differences of `TwoBodyDynamics.advance`.

- **Linearization on CW**: for random nodes from 120 s, $A_k$ and $B_k$ equal the exact
  zero-order-hold step to $10^{-12}$ and $\mathbf c_k$ is below $10^{-12}$ of the 300 m states.
- **Variational Jacobians** (independent check): against central differences of `advance` with
  steps of 1e-2 m, 1e-3 m/s and 1e-4 m/s^2, $A_k$ and $B_k$ agree to $10^{-5}$ relative, on
  two-body at the second node from 300 s and on J2 at the episode start, at an argument of
  latitude of 0.7 rad; the affine term reproduces the flown segment end to $10^{-8}$.
- **Near the target**: the two-body step is within $10^{-7}$ of CW at 1 m and departs from it by
  more than $5 \times 10^{-7}$ at 400 m; on circular two-body the linearization does not depend on
  the declared argument of latitude (0 against 2 rad, to $10^{-9}$).
- **Shapes**: inconsistent, empty or non-finite plans are refused.
- **One Gauss-Newton step** (multiple shooting, Rawlings, Mayne and Diehl, sections 8.5.2 and
  8.6.3; a check of ours, not a book example): minimum-energy rendezvous to the origin at rest from
  (100, -2000, 50) m, (0.05, 0.2, -0.02) m/s on two-body. Four 60 s segments: the CW plan flown
  open loop misses by 0.017893 m, and one step from `shooting_linearization` at the CW plan,
  solved by a dense KKT system, leaves continuity below 1e-10 and an open-loop miss below 1e-9 m.
  Twenty segments over half an orbit (slow): the CW plan misses by 3.9909 m, and one step leaves
  continuity below 3e-10, a miss below 1e-7 m and 0.07806% more energy than the CW plan.
- **Agent on CW**: with no keep-out sphere every decision equals the linear MPC's first move to
  $10^{-6} a_{\max}$ over three decisions from 300 m behind the hold point, two of them on the
  along-track limit, and near the hold point the command is the LQR law.
- **Fallbacks**, with `solve_horizon` replaced by a stub: "iteration_cap" and "solver_failed"
  give level 2 and the clipped LQR law without a plan, and level 1 and the linear MPC's second
  move with one.
- **Real-time iteration** (section 8.9.2): on `approach-vbar` flown on two-body from 300 m
  behind the hold point, the matched declared model predicts the next observation to 1e-8 m and
  1e-9 m/s over three decisions of a two-step horizon, all at level 0, with continuity between 0
  and 1e-4 (five decisions of a ten-step horizon, slow, to 1e-8 m); declared CW mispredicts by
  more than 5e-7 m. The second linearization is called with the first plan's prediction as its
  first node and at the next decision time, and its nodes and controls do not change when the
  observation is moved by a metre; `prep_s` and `feedback_s` are positive.
- **Argument of latitude**: on two-body the commands do not change with the declared phase (to
  $10^{-8} a_{\max}$) or with the drawn one; on J2 the drawn phase declared predicts to 1e-8 m and
  a quarter orbit off misses by more than 1e-5 m, and both are logged.
- **Agent contract**: "normalized", "accel" and "thrust" give equal fractions to $10^{-12}$; reset
  wipes the previous episode; two agents from one config stepped on interleaved environments
  equal separate runs bitwise; the agent runs on a hand-made view and reads only `time_s` from
  `info`, holds no environment object and replays from the view alone; a recorded episode has
  one log entry per decision with valid solver fields; the recorder and metrics never read
  `fallbacks`, `continuity` or `declared_phase`.

## Example

```python
import numpy as np

from vbar.agents.base import agent_generator
from vbar.agents.lyapunov import DeclaredModel
from vbar.agents.mpc import MPCConfig
from vbar.agents.nmpc import NMPCAgent, NMPCConfig, shooting_linearization
from vbar.scenarios import make

# The catalogue's target orbit: 500 km circular at 51.6 deg.
two_body = DeclaredModel(dynamics="twobody", inclination=np.radians(51.6))

# One coasting 10 s segment 400 m behind the target: the two-body step departs from CW.
nodes = np.array([[0.0, -400.0, 0.0, 0.0, 0.0, 0.0]] * 2)
a, b, c = shooting_linearization(two_body, 0.0, nodes, np.zeros((1, 3)), 10.0)
a_cw, _, _ = shooting_linearization(DeclaredModel(), 0.0, nodes, np.zeros((1, 3)), 10.0)
print(np.max(np.abs(a[0] - a_cw[0])), c[0])

# Three decisions of a two-step horizon on the two-body rung, from 300 m behind the hold point.
env = make("approach-vbar", dynamics="twobody", training=True)
obs, _ = env.reset(seed=3, options={"start_state": (0.0, -400.0, 0.0, 0.0, 0.0, 0.0)})
agent = NMPCAgent(NMPCConfig(mpc=MPCConfig(horizon=2), model=two_body))
agent.reset(env.agent_view(), agent_generator(4))
for _ in range(3):
    obs, *_ = env.step(agent.act(obs, env.agent_info()))
    entry = agent.log[-1]
    miss = np.linalg.norm(entry["predicted"][:3] - obs[:3])
    print(entry["level"], entry["status"], entry["reference"], f"{miss:.1e} m")
print(agent.fallbacks)
```
