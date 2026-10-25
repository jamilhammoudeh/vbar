# CLF-CBF QP controller

Code: `ClfCbfConfig`, `ClfDesign`, `clf_design`, `clf_row`, `ClfCbfSolution`, `clf_cbf_qp` and
`ClfCbfAgent` in `vbar.agents.clf_cbf`. Hill frame ([frames.md](frames.md)), SI units.

A controller to a hold point that solves one small quadratic program per decision: stay close to
the hold input, decrease a control Lyapunov function (CLF) toward the hold point as fast as the
continuous LQR would, and keep a control barrier function (CBF) on the range to the target
nonnegative. The Lyapunov row may be relaxed at a cost; the barrier row and the thrust limit may
not (Ames, Coogan, Egerstedt, Notomista, Sreenath and Tabuada, "Control barrier functions: theory
and applications", European Control Conference 2019, sections II.C and IV). The relaxation is
dualized, so each decision is a scalar root search over exact projections onto the barrier row
and the thrust limit ([solvers.md](solvers.md), "Exact projection onto linear rows").

## How it is built

**Lyapunov function.** With $\mathbf e = \mathbf x - \mathbf x_s$ the deviation from the hold
point at rest and $\mathbf u_s$ the constant input that holds it ($-3n^2 x_s$ radially, $n^2
z_s$ normally; the hold point at rest is an equilibrium of both the continuous and the
zero-order-hold model, so the input is the same), $V = \mathbf e^T P\mathbf e$ with $P$ the
stabilizing solution of the continuous Riccati equation on the CW model,

$$
A^T P + P A - P B R_c^{-1} B^T P + Q_c = 0,\qquad
Q_c = \mathrm{diag}(s_r^{-2} I, s_v^{-2} I),\quad R_c = s_a^{-2} I,
$$

with the position, velocity and acceleration scales $s_r, s_v, s_a$ of `LQRConfig` (the
acceleration scale is `max_accel` unless set), the weights of [lqr.md](lqr.md) in continuous
time. `clf_design` solves it in units of the scales, where $Q_c$ and $R_c$ are identities
(`scipy.linalg.solve_continuous_are`), and refuses a solution whose residual exceeds $10^{-10}$
of $P$ there.

**Decay rate.** Under the continuous LQR $\mathbf u = \mathbf u_s - R_c^{-1}B^TP\mathbf e$,
$\dot V = -\mathbf e^T M\mathbf e$ with $M = Q_c + PBR_c^{-1}B^TP$. The largest rate $\gamma$
with $\dot V \le -\gamma V$ at every deviation is the smallest $\gamma$ over $\mathbf e$ of
$\mathbf e^T M\mathbf e / \mathbf e^T P\mathbf e$, the smallest eigenvalue of the pencil
$M\mathbf e = \gamma P\mathbf e$ (`scipy.linalg.eigh`). At the default weights on a 500 km orbit
it is $4.486407652117\times10^{-3}$ 1/s, a time constant of 223 s. The bound
$\lambda_{\min}(M)/\lambda_{\max}(P) = 8.94\times10^{-6}$ 1/s also holds but is 500 times looser,
so a CLF row built on it would accept commands far slower than the LQR.

**Lyapunov row.** Since $A\mathbf x_s + B\mathbf u_s = 0$, $\dot V = 2\mathbf e^TPA\mathbf e +
\mathbf g\cdot(\mathbf u - \mathbf u_s)$ with $\mathbf g = 2(P\mathbf e)_v$, the velocity rows
of $P\mathbf e$. `clf_row` returns $\mathbf g$ and the bound of

$$
\mathbf g\cdot\mathbf u \le \mathbf g\cdot\mathbf u_s - \gamma V - 2\mathbf e^TPA\mathbf e +
\delta ,
$$

that is $\dot V \le -\gamma V + \delta$; both are zero at the hold point at rest.

**Barrier row.** $h = \rho - R_f$, with $\rho = |\mathbf r|$ and $R_f$ the keep-out radius plus
`buffer_m`. $h$ has relative degree two: $\dot\rho = \hat{\mathbf r}\cdot\mathbf v$ and
$\ddot\rho = (|\mathbf v|^2 - \dot\rho^2)/\rho + \hat{\mathbf r}\cdot(\mathbf a_{cw}(\mathbf x)
+ \mathbf u)$, with $\mathbf a_{cw} = (3n^2x + 2n v_y,\ -2n v_x,\ -n^2 z)$ the CW acceleration of
the state. The exponential barrier of Ames et al. section IV asks for $\ddot h + (p_1 + p_2)\dot
h + p_1p_2 h \ge 0$, whose characteristic roots are $-p_1$ and $-p_2$ (the paper prints them as
$p_1, \dots, p_r$; with positive $p_i$ the roots are their negatives). As a row on the command:

$$
\hat{\mathbf r}\cdot\mathbf u \ge -\frac{|\mathbf v|^2 - \dot\rho^2}{\rho} -
\hat{\mathbf r}\cdot\mathbf a_{cw}(\mathbf x) - (p_1 + p_2)\dot\rho - p_1p_2(\rho - R_f).
$$

The row normal has unit length away from the target; at $\rho = 0$ there is no direction and no
row.

**QP.** Each decision solves

$$
\min_{\mathbf u,\ \delta}\ \tfrac12|\mathbf u - \mathbf u_s|^2 + \tfrac w2\delta^2
\quad\text{s.t.}\quad \mathbf g\cdot\mathbf u \le b_V + \delta,\quad
\text{the barrier row},\quad \mathbf u \in \mathcal U,
$$

with $\mathcal U$ the view's box ($|u_i| \le a_{\max}$) or ball ($|\mathbf u| \le a_{\max}$) and
$w$ the `clf_weight`. `clf_cbf_qp` takes any reference command, any relaxed row and up to 32
hard rows.

**Dualized solve.** With a multiplier $\lambda \ge 0$ on the relaxed row, stationarity in
$\delta$ gives $\delta = \lambda / w$, and the minimizer over $\mathbf u$ for fixed $\lambda$ is
the projection $\mathbf u(\lambda)$ of $\mathbf u_s - \lambda\mathbf g$ onto the hard rows and
$\mathcal U$. Complementarity leaves one equation,

$$
\lambda = w\,\max\bigl(0,\ \mathbf g\cdot\mathbf u(\lambda) - b_V\bigr).
$$

A projection onto a convex set is monotone, so $\mathbf g\cdot\mathbf u(\lambda)$ does not
increase with $\lambda$ and the residual $\lambda - w\max(0, \cdot)$ is strictly increasing: the
root is unique. If the projection at $\lambda = 0$ meets the relaxed row, it is the solution with
$\delta = 0$. Otherwise the root lies in $[0,\ w(\mathbf g\cdot\mathbf u(0) - b_V)]$, where
`scipy.optimize.brentq` finds it to four ulps of $\lambda$. Under a box the residual is piecewise
linear, so the search ends after a few projections; each is computed once and kept for the rest
of the search.

**Fallbacks.** The barrier row can conflict with the thrust limit: a state closing too fast to
stop inside the remaining distance asks for more outward thrust than the limit allows along
$\hat{\mathbf r}$ (see the limits below). `project` then
certifies the conflict, and the command is its least-violation point from $\mathbf u_s$, with
status `"infeasible_certified"`; the relaxed row is dropped, its multiplier is 0 and $\delta$ is
reported as the row's violation at the command. If a projection fails without a certificate
(`"solver_failed"`), the command is the one that projection returned, with that status.

**Agent.** `ClfCbfAgent.reset` builds the design for the view's orbit, thrust and hold point
and takes the largest keep-out radius among the view's constraints. Each decision builds both
rows at the observed state, solves the QP from $\mathbf u_s$ and issues the command in the
view's action mode. The command is held for the step, where neither row's continuous-time
guarantee applies, so the agent certifies the keep-out margin of the held step on the CW model:
the step is sampled at 20 points and `vbar.constraints.segment_minimum` refines the minimum and
bounds it between samples (the larger of the first- and second-order bounds,
[constraints.md](constraints.md)). The bound is logged, not enforced.

**Log.** `agent.log` holds one entry per decision: `time_s`; `status`, the projection's status;
`level`, 0 for a feasible QP, 1 for the least-violation command and 2 for the command of a failed
projection; `cbf_active`, whether the barrier row binds; `delta` and `multiplier` of the relaxed
row; `held_bound`, the certified lower bound (m) on the keep-out margin over the held step, or
`None` without a keep-out constraint; and `elapsed_s`, the decision's wall time.
`agent.fallbacks` counts decisions by level. Both are the agent's own bookkeeping, which no
metric reads as a score.

## Parameters and defaults

`ClfCbfConfig` is a frozen, keyword-only dataclass.

| Parameter | Default | Reason |
|---|---|---|
| `regulator` | `LQRConfig()` | The weights `LQRAgent` uses, in continuous time; its `rate_weight` must be 0 and its `stage_cost` is not used |
| `poles` | `(0.02, 0.05)` 1/s | Ours. $p_{\max}\,dt = 0.5$ at the 10 s step stays below 0.618, the value of $p\,dt$ at which a double-pole barrier under a held command first loses invariance on a double integrator, $(\sqrt5 - 1)/2$ (our computation from Ames et al. definition 7) |
| `buffer_m` | 1.0 m | Ours. Added to the keep-out radius, so the chaser rides the barrier 1 m outside the keep-out sphere |
| `clf_weight` | `1e-2` | Ours, from a closed-loop sweep on `approach-hbar` (measured once): with this weight 10 of 10 dispersed starts (seeds 1 to 10) were captured, with $10^{-8}$ none of 3; a stiffer row trades barrier riding for progress toward the hold point |

Fixed in the module: 20 samples over the held step, the environment's rate; the root search
tolerance of four ulps of $\lambda$.

## Assumptions and limits

- CW motion on a circular target orbit at the view's mean motion, for the design, both rows and
  the held-step certificate. On two-body or J2 truth they are those of the CW model.
- All guarantees of both rows are for continuous time (Ames et al.); under the scenario's
  sample-and-hold they do not hold. The held-step bound reports what one step of the commanded
  input does on the CW model from the observed state; it does not include navigation or
  execution error, and nothing adjusts the command when it is negative.
- The exponential barrier keeps $h \ge 0$ in continuous time only from states where it already
  holds and where $\dot h + \max(p_1, p_2)\,h \ge 0$ (Ames et al. theorem 8, read with the roots
  $-p_i$). Breaking that condition does not by itself make the QP infeasible: at $(0, 0, 52)$ m
  closing at 0.1 m/s ($\dot h + p_{\max} h = -0.05$ m/s) the row asks for $6.06\times10^{-3}$
  m/s$^2$ outward and the solve is feasible with the barrier active. The least-violation fallback
  happens exactly when the row's right side exceeds the largest $\hat{\mathbf r}\cdot\mathbf u$
  the limit allows, $a_{\max}|\hat{\mathbf r}|_1$ under a box and $a_{\max}$ under a ball: at the
  same point closing at 0.3 m/s it asks for $2.006\times10^{-2}$ m/s$^2$ against 0.02 and falls
  back (measured once, box and ball).
- The barrier is a sufficient condition for staying outside the sphere, not the least
  restrictive one: the outside of a sphere is not compact, so the necessity result of Ames et al.
  (theorem 3) does not apply.
- On the line through the target and the hold point, with the target in between, the barrier
  correction is collinear with the Lyapunov command and CW gives no lateral push, so the chaser
  rides the barrier and does not pass the target: from $(0, 0, 300)$ m on `approach-hbar` it
  stays 1.0 m outside the keep-out sphere for the whole episode.
- One keep-out sphere, centred on the target; the other constraint types are not modelled.
- No command delay and no rate weight: `reset` refuses a view whose observation carries pending
  commands and a `regulator` with a positive `rate_weight`, each with a `ValueError`.
- Holds and berthings only: other tasks and goals without a hold point are refused, as are the
  `"disc27"` and `"impulse_coast"` action modes.
- Each decision costs a few projections and the certificate. Over 90 decisions of the
  `approach-hbar` head-on and pass runs (seeds 0 and 1) the median took 4.5 ms and the slowest
  19.5 ms, measured once on one machine at 90% processor load from other programs.

## Validation

Tests: `tests/test_clf_cbf.py`. References marked independent come from a continuous Riccati
solve in the test, from SLSQP (`scipy.optimize.minimize`) on the four variables
$(\mathbf u, \delta)$, or from the barrier row written out in the test, not from the module. CW
cases are for a 500 km circular orbit of the WGS 84 Earth, a 10 s step and $a_{\max} = 0.02$
m/s$^2$.

- Design: $P$ equals an independent continuous Riccati solve to $10^{-9}$ of $P$; the hold
  point is the goal's at rest and $\mathbf u_s$ the continuous hold input to $10^{-15}$ m/s$^2$.
- Decay rate: equal to the smallest eigenvalue of the independent pencil $(M, P)$ to $10^{-9}$
  relative and to $4.486407652117\times10^{-3}$ 1/s. The row normal is $2(P\mathbf e)_v$ at 20
  deviations. At 100 random deviations (normal, 200, 200 and 50 m and 0.1 m/s per axis) the
  continuous LQR command meets the row with $\delta = 0$ to $10^{-12}$ relative; at 1.5 times
  the rate at least one draw fails, so the rate is tight. The row is empty at the hold point at
  rest.
- Dualized solve: on 100 random problems per layout (box and ball), each with 0 to 3 hard rows,
  the solution equals the independent four-variable QP to $10^{-6}\,a_{\max}$ in $\mathbf u$ and
  $\delta$. On the same draws the hard rows hold to $10^{-10}\,a_{\max}$, the relaxed row holds
  with its $\delta \ge 0$, the multiplier equals $w\delta$, the command is inside the limit, 0,
  1, 2 and 3 hard rows bind, and the relaxed row binds in at least 50.
- An inactive relaxed row with no hard rows returns the reference command with $\delta = 0$.
- Conflict: a row $u_z \ge 0.03$ m/s$^2$ under the 0.02 m/s$^2$ box or ball gives the
  least-violation command of `project` to $10^{-12}\,a_{\max}$ and status
  `"infeasible_certified"`, with the row among the conflicting ones and $\delta$ the relaxed
  row's violation. A state 5 m outside the barrier radius closing at 1 m/s along the orbit
  normal needs $6.606737\times10^{-2}$ m/s$^2$ outward (independent row); the agent commands
  full thrust outward and logs the status. A stub that fails every projection shows its command
  applied and `"solver_failed"` logged.
- Agent at the V-bar, R-bar and H-bar holds at rest: the command is the hold input to $10^{-12}$
  of it, with the barrier inactive and $\delta = 0$. Under a ball limit the command at
  $(80, 300, -60)$ m at rest is inside the ball and at least half of $a_{\max}$.
- Closed loop on `approach-hbar` (training) from $(0, 0, 300)$ m at rest, seeds 0 and 1, 30
  decisions: every projection feasible, the barrier active from decision 20 to 29 and the lowest
  continuous keep-out margin 11.341355447 m (to $10^{-6}$ m). For seed 0 every command meets the
  independent barrier row to $10^{-9}\,a_{\max}$, with equality where it is active. Over the whole
  360 decisions of seed 0 (slow test) the lowest margin is 1.0 m to $10^{-3}$ m, the buffer.
- A pass from $(30, 0, 45)$ m at $(0, 0.2, -0.3)$ m/s with the barrier active at all 15
  decisions: every command meets the independent barrier row with equality to
  $10^{-9}\,a_{\max}$, the lowest continuous margin is 0.48936 m, and on some step the minimum
  falls more than $10^{-6}$ m below the lowest of the environment's 20 samples, while every
  logged held-step bound is nonnegative. On this pass and the two head-on runs the logged bound
  is never above the environment's continuous minimum (to $10^{-9}$ m) and at most 2 mm below
  it.
- A dispersed `approach-hbar` start (seed 1, slow test) is captured without entering the
  keep-out sphere.
- Refusals: tasks other than a hold, a hold goal without a hold point, the discrete and
  impulsive modes, poles that are not two positive finite rates, a negative or undefined buffer
  and a weight that is not positive.
- Agent contract: equal commands in fractions in the three continuous action modes, reset
  clears the previous episode, two agents from one configuration on interleaved environments
  match separate runs, the agent runs on a hand-made view and reads only `time_s`, the recorder
  keeps one log entry per decision, and the recorder and metrics never read the log fields.

## Usage

```python
import numpy as np

from vbar.agents.base import agent_generator
from vbar.agents.clf_cbf import ClfCbfAgent, ClfCbfConfig, clf_cbf_qp
from vbar.scenarios import make
from vbar.solvers import ActuatorSet, Rows

# One QP: stay near u_ref, keep g.u <= -1e-3 (relaxed) and u_z >= 5e-3 (hard), box 0.02 m/s^2.
rows = Rows(np.array([[0.0, 0.0, 1.0]]), np.array([5e-3]), ("keep_out",))
solution = clf_cbf_qp(
    np.zeros(3), np.array([1.0, 0.0, 0.0]), -1e-3, rows, ActuatorSet(0.02, "box"), clf_weight=1.0
)
print(solution.status, solution.u, solution.delta, solution.projection.active)

env = make("approach-hbar", training=True)
agent = ClfCbfAgent(ClfCbfConfig())
obs, _ = env.reset(seed=0, options={"start_state": (30.0, 0.0, 45.0, 0.0, 0.2, -0.3)})
agent.reset(env.agent_view(), agent_generator(1))
for _ in range(15):
    obs, _, terminated, truncated, _ = env.step(agent.act(obs, env.agent_info()))
    if terminated or truncated:
        break
print(agent.fallbacks, min(entry["held_bound"] for entry in agent.log))
```
