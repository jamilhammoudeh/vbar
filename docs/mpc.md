# Model predictive control

Code: `admissible_set`, `MPCConfig`, `mpc_problem`, `mpc_solve`, `MPCPlan` and `MPCAgent` in
`vbar.agents.mpc`. Hill frame ([frames.md](frames.md)), SI units.

## Terminal set

An MPC that ends its horizon inside a set where the LQR law keeps every constraint, and stays
there, inherits the stability of the LQR (Rawlings, Mayne and Diehl, *Model Predictive Control*,
2nd ed., sec. 2.5.4). The largest such set for the closed loop $\mathbf x^+ = F\mathbf x$,
$F = \Phi - \Gamma K$, under output rows $|h_i \mathbf x| \le b_i$ is the maximal output
admissible set

$$
O_\infty = \{\mathbf x : |h_i F^t \mathbf x| \le b_i \text{ for every row } i \text{ and every }
t \ge 0\}.
$$

For the hold-point regulator $\mathbf x$ is the deviation $\mathbf z - \mathbf z_s$ from the hold
point, and the rows are the thrust limit on $\mathbf u = \mathbf u_s - K(\mathbf z - \mathbf z_s)$,
$|K_i(\mathbf z - \mathbf z_s)| \le a_{\max} - |u_{s,i}|$ per axis, together with any state rows.

### How it is built

`admissible_set(f, h, bound, t_max=200)` follows Gilbert and Tan, "Linear systems with state and
control constraints: the theory and application of maximal output admissible sets", *IEEE
Transactions on Automatic Control* 36(9), 1991, algorithm 3.2. With
$O_t = \{\mathbf x : |h F^s \mathbf x| \le b,\ s = 0, \dots, t\}$, it adds the rows $h F^t$ for
$t = 0, 1, \dots$ and stops at the first $t^*$ for which every row of $h F^{t^*+1}$ is redundant
over $O_{t^*}$. Then $O_{t^*+1} = O_{t^*}$, so $F O_{t^*} \subseteq O_{t^*}$, no later row cuts
the set, and $O_\infty = O_{t^*}$.

Redundancy of a candidate row $c$ is one linear program, $\max c\,\mathbf x$ over $O_t$, by
`scipy.optimize.linprog` (HiGHS dual simplex, presolve off; on these nearly parallel rows the
presolve has reported an unbounded program as infeasible and stopped short of the optimum of
another). The origin is always feasible, so a reported infeasibility is a numerical failure and
the program is solved again by interior point. $O_t$ is symmetric about the origin, so the
minimum is minus the maximum and one program covers both signs. The row is redundant when the
maximum is at most its bound times $1 + 10^{-9}$. The margin absorbs the rounding of a row whose
maximum equals its bound exactly, which would otherwise add rows the set does not need. An
unbounded program (the set is still open along $c$) keeps the row.

The programs run in scaled coordinates: each row is divided by its bound, each state column by
its largest entry, and each row, the candidate included, by its norm, with the right-hand sides
scaled to match. Gains in SI units span many decades between position and velocity columns and
rows decay as $F^t$, which unscaled would put entries below the solver's $10^{-9}$ threshold. The
rows and $t^*$ then do not depend on the units of $h$ or of the bounds.

The function returns the raw rows $h, hF, \dots, hF^{t^*}$ stacked by time, so row $j$ carries
the bound `bound[j % p]` for $p$ output rows, together with $t^*$. The set is
$\{\mathbf x : |\text{rows}\,\mathbf x| \le b\}$ row by row.

**Certificate.** Before returning, each row $c$ of $hF^{t^*+1}$ is proved redundant over the
returned rows $r_j$ with bounds $b_j$ without relying on the solver's tolerances. For any
multipliers $\lambda$ with $\sum_j \lambda_j r_j = c$, every $\mathbf x$ in the set has
$|c\,\mathbf x| \le \sum_j |\lambda_j|\, b_j$ (weak duality on the symmetric set). The function
takes a maximal set of independent rows in order of their activity at the LP maximizer, solves
for $\lambda$ in exact rational arithmetic on the floating-point entries, and requires the sum to
be at most the bound of $c$ times $1 + 10^{-9}$. When the rows span only the observable subspace,
a row $c$ in their span is certified the same way; a row with no exact solution lies outside the
span, where the set is unbounded, so it is kept and the iteration goes on, as after an unbounded
program. Where the set is so long and thin that the LP's optimum is
not accurate to that margin, the proof fails and the function refuses the set rather than return
one that may not be invariant. On CW at 500 km, steps of 10 and 60 s and a 0.02 m/s^2 box, over
discrete LQR scales from (0.1 m, 1e-4 m/s, 0.002 m/s^2) to (1e4 m, 10 m/s, 0.2 m/s^2), it refused
7 of 658 sets, all with position weights 1e13 or more times below the velocity weights and sets
reaching 1e11 m or more; in 6 of them a feasible point puts a next row above its bound by 2e-6
to 1.1e-4.

It raises `ValueError` when the spectral radius of $F$ is $1 - 10^{-12}$ or more ("terminal set
needs a stable closed loop") and when the rows at $t_{\max} + 1$ are still not redundant
("terminal set is not finitely determined within t_max steps"), and `RuntimeError` when a program
ends without an optimum or a proof of unboundedness, or when the certificate fails ("terminal
set could not be certified").

### Assumptions and limits

- $F$ must be asymptotically stable. With a mode on the unit circle, such as the along-track
  drift of the CW step without feedback, the set has no finite description (Rawlings, Mayne and
  Diehl, exercise 2.5), and close to the circle $t^*$ grows without limit: 4, 12 and 30 for a
  slow pole at 0.5, 0.9 and 0.99 in the exercise's system.
- Every bound must be positive, so the origin is in the interior. If $(F, h)$ is not observable
  the set contains the unobservable subspace and is unbounded along it. The certificate then
  needs each row of $hF^{t^*+1}$ exactly in the span of the returned rows, so a pair that is
  unobservable only up to rounding, as after a change of coordinates that mixes the subspaces,
  is refused at the cap with `ValueError`; write it in exactly unobservable coordinates instead.
- Rows are symmetric, $|h_i \mathbf x| \le b_i$. A one-sided row is written about the hold point
  by the caller. The keep-out halfspace $\mathbf n_h \cdot \mathbf r \ge R + b_h$, with
  $\mathbf n_h$ the unit vector to the hold point, becomes
  $|\mathbf n_h \cdot \delta\mathbf r| \le |\mathbf r_{\text{hold}}| - R - b_h$; its mirror image
  also caps the range beyond the hold point, which only shrinks the set. $b_h$ covers the
  chord-to-arc gap between samples: the rows hold at the samples only.
- The symmetric thrust row $|K_i \delta\mathbf z| \le a_{\max} - |u_{s,i}|$ lies inside the exact
  interval $u_{s,i} - a_{\max} \le K_i \delta\mathbf z \le u_{s,i} + a_{\max}$ and equals it only
  where the hold needs no steady input.
- The cost is at most $p(t^* + 1)$ linear programs over at most $2p(t^* + 1)$ rows; a step stops
  at its first row that is not redundant.

### How it is tested

`tests/test_mpc_terminal_set.py`:

- **Exercise 2.5** (Rawlings, Mayne and Diehl, p. 173): $F = [[0.9, 1], [0, 0.09]]$,
  $h = (1, 1)$, $|y| \le 1$ gives the rows $(1, 1)$, $(0.9, 1.09)$, $(0.81, 0.9981)$ and
  $t^* = 2$ to 1e-12, and the slow poles 0.5, 0.9 and 0.99 give $t^* = 4, 12, 30$.
- The set is positively invariant (no row of $F O_\infty$ exceeds its bound by more than 1e-9),
  rows whose maximum equals the bound are redundant, scaling the bound by 1e-12 or 1e12 leaves
  $t^*$ and the rows unchanged, the rows follow a change of state coordinates, and each output
  row keeps its own bound.
- A pole at 1, at $1 - 10^{-13}$ or at 1.05, and the free-drift CW step, are refused; $t^*$
  above `t_max` is refused.
- On CW at 500 km with the discrete LQR of $Q = \operatorname{diag}(10^{-2} I_3, 10^4 I_3)$,
  $R = 2500 I$ and a 0.02 m/s^2 box, $t^* = 5, 1, 1$ at steps of 1, 10 and 60 s, and the
  largest along-track offset at rest inside the set is 24.154232, 201.509569 and 1237.851968 m.
- On `approach-vbar` with the hold row ($b_h$ = 2.25 m), 200 states drawn in the set and flown
  three orbits under the LQR loop keep the thrust box and a certified keep-out margin at every
  step; without the hold row the set contains states inside the 50 m sphere.

### Example

```python
import numpy as np

from vbar.agents.lqr import dlqr
from vbar.agents.mpc import admissible_set
from vbar.dynamics import EARTH, cw_discrete, mean_motion

# Rawlings, Mayne and Diehl, exercise 2.5.
f = np.array([[0.9, 1.0], [0.0, 0.09]])
rows, t_star = admissible_set(f, np.array([[1.0, 1.0]]), np.array([1.0]))
print(t_star, rows.tolist())

# The thrust box of an LQR hold on V-bar, 500 km, 10 s steps.
n = mean_motion(EARTH.radius + 500e3)
phi, gamma = cw_discrete(n, 10.0)
k, _ = dlqr(phi, gamma, np.diag([1e-2] * 3 + [1e4] * 3), 2500.0 * np.eye(3))
rows, t_star = admissible_set(phi - gamma @ k, k, np.full(3, 0.02))
along = np.abs(rows[:, 1])
reach = np.min(0.02 / along[along > 0.0])
print(t_star, rows.shape, round(reach, 1))
```

## Horizon problem

`mpc_problem(model, view, z0, reference, config)` builds the QP the hold-point MPC solves at each
decision, as a `StageQP` of `vbar.solvers` ([solvers.md](solvers.md)); `mpc_solve(qp)` solves
it. The model is `regulator_model(view, config.regulator)` ([lqr.md](lqr.md)), so the MPC shares
the LQR baseline's stage cost, delay and rate augmentation and hold point $(\mathbf z_s,
\mathbf u_s)$, and equals the LQR law wherever no row binds.

### How it is built

The states are the augmented $\mathbf z$ of the regulator model (Hill state, then pending or
last commands in m/s^2) and the inputs are the three Hill accelerations $\mathbf u_k$ and a
keep-out slack $s_k$ (m). The dynamics are the exact zero-order-hold CW step,
$\mathbf z_{k+1} = \Phi \mathbf z_k + \Gamma \mathbf u_k$; the slack has a zero column. With
$H = \begin{bmatrix} Q & S \\ S^T & R \end{bmatrix}$ the regulator's stage weight and $P$ its DARE
solution,

$$
V_N = \sum_{k=0}^{N-1} \Big[ \tfrac12
\begin{bmatrix} \mathbf z_k - \mathbf z_s \\ \mathbf u_k - \mathbf u_s \end{bmatrix}^T H
\begin{bmatrix} \mathbf z_k - \mathbf z_s \\ \mathbf u_k - \mathbf u_s \end{bmatrix}
+ \rho\, s_k + \tfrac12 \rho_q s_k^2 \Big]
+ \tfrac12 \beta\, (\mathbf z_N - \mathbf z_s)^T P (\mathbf z_N - \mathbf z_s).
$$

This is half the LQR agent's cost, the factor $\tfrac12$ of `StageQP`. With $\beta = 1$ and no
binding row, dynamic programming makes $\tfrac12 \delta\mathbf z^T P\, \delta\mathbf z$ the cost to
go at every stage, so the first move is $\mathbf u_s - K(\mathbf z_0 - \mathbf z_s)$ for every
horizon (Rawlings, Mayne and Diehl, *Model Predictive Control*, 2nd ed., sec. 2.5.1). $\beta > 1$
inflates the terminal cost (theorem 2.41) and has no such equality.

Each stage carries $7 + m$ rows, $m$ = `row_samples`, since `StageQP` takes one row count per
stage:

- six thrust rows $-a_{\text{box}} \le u_{k,i} \le a_{\text{box}}$, with $a_{\text{box}} =
  a_{\max}$ on a box limit and $a_{\max}/\sqrt 3$, the box inscribed in the ball, on a ball limit
  (the solver takes linear rows only);
- the slack bound, written $\sigma_k \ge 1$ m for the input $\sigma_k = s_k + 1$ m. On the plain
  row $s_k \ge 0$ the interior-point solver stops with "solver_failed" even at the optimum, since
  the row's only term is $|s_k|$, which vanishes there, and its complementarity test has nothing
  to measure against; the offset gives the row a scale and changes nothing else;
- $m$ keep-out rows at the times $\tau_j = j\,\Delta t/(m-1)$ within step $k$.

**Keep-out rows.** With the keep-out radius $R$ (the largest "keep_out" constraint of the view),
the position at $\tau_j$ is affine in the stage, $\mathbf r_{kj} = \Phi_r(\tau_j)\,\mathbf x_k +
\Gamma_r(\tau_j)\,\mathbf u_{\text{act},k}$, where $\mathbf u_{\text{act},k}$ is the command
acting during step $k$: the oldest pending slot of $\mathbf z_k$ when commands wait, else
$\mathbf u_k$. The row is the tangent plane at the reference plan's sample
$\mathbf r^{\text{ref}}_{kj}$ (`reference`, states and commands over the horizon),

$$
\mathbf n_{kj} \cdot \mathbf r_{kj} + s_k \ge \min\big(R + b_{kj},\ |\mathbf r^{\text{ref}}_{kj}|\big),
\qquad \mathbf n_{kj} = \mathbf r^{\text{ref}}_{kj} / |\mathbf r^{\text{ref}}_{kj}|.
$$

Since $|\mathbf r| \ge \mathbf n \cdot \mathbf r$, a row met with zero slack puts the sample at
range $R + b$ or more. The minimum with $|\mathbf r^{\text{ref}}|$ means a row never cuts its own
reference, so a reference that is a plan of the next problem keeps it feasible. The tightening
$b$ covers range lost between samples $h = \Delta t/(m-1)$ apart: `instep_bound="first_order"`
takes $b = V h / 2$, `"second_order"` takes $b = (V h)^2/(8R) + A h^2/8$, with $V$ and $A$ the
speed and acceleration bounds of `interval_bounds` (`vbar.constraints`) on the reference samples
at $\sqrt 3\, a_{\text{box}}$ of thrust, each sample taking the larger value of its two
intervals. When those bounds are infinite (their gain $g = c\,h\,(1.5 n^2 h + 2n)/2$ is 1 or
more) the problem is refused: "row_samples too few for the in-step bound at this step". The first
sample of the first step is the current state and a sample at the origin has no normal; both get
the inert row $0 \ge -1$, which an interior point meets strictly.

**Exact penalty.** The slack costs $\rho s + \rho_q s^2/2$ with $\rho_q = 10^{-6} \rho / R$, which
keeps the input block positive definite. A linear penalty is exact (Boyd and Vandenberghe,
*Convex Optimization*, exercise 5.16): one slack serves the $m$ keep-out rows of its stage, so at
the hard solution with $s_k = 0$ stationarity in $s_k$ reads $\rho = \sum_j \lambda_{kj} + \nu_k$,
with $\lambda_{kj}$ the stage's keep-out multipliers and $\nu_k$ that of the slack bound. When
$\rho$ is at least $\sum_j \lambda_{kj}$ for every stage, $\nu_k \ge 0$, the hard solution meets
the optimality conditions of the soft problem, and the soft problem returns it with zero slack;
the quadratic term has zero slope at $s = 0$ and does not change this. `penalty=None` sets $\rho$ to 10 times the largest such sum found over twenty states on
`approach-vbar` with the default weights and $N = 20$, 205.958 per m, expressed as
$2059.49\ \text{m} \times \bar q_r$ with $\bar q_r$ the model's mean position weight per step, so
it follows the scale of the weights; on `approach-vbar` it is 2059.58 per m. A converged plan with
positive slack is a predicted keep-out violation.

**Terminal set.** `terminal="cost"` adds no terminal rows. `terminal="set"` adds the maximal
admissible set of the LQR loop $F = \Phi - \Gamma K$ (section above) as hard rows on
$\mathbf z_N$: the thrust rows $|K_i \delta\mathbf z| \le a_{\text{box}} - |u_{s,i}|$ and, with a
keep-out sphere, the hold-point halfspace $|\mathbf n_h \cdot \delta\mathbf r| \le |\mathbf
r_{\text{hold}}| - R - b_h$. The set's rows hold at samples $\Delta t$ apart, and $b_h = A\,
\Delta t^2 / 8$ covers what a linear row can lose between them, with $A$ from `interval_bounds`
at the set's largest range $|\mathbf r_{\text{hold}}| + |(e_x, e_y, e_z)|$ and speed
$|(e_{v_x}, e_{v_y}, e_{v_z})|$, $e_i$ the set's extent along each Hill axis (one LP each,
enlarged by $10^{-6}$ for the LP tolerance). Since the set depends on $b_h$, $b_h$ is computed on
the set built with the allowance at the hold point at rest; the final set lies inside that one,
so its own allowance is no larger. On `approach-vbar` with the default weights $b_h$ = 1.4598 m and
$t^* = 1$. The rows are kept for the last eight models and steps, since a controller builds one
problem per decision from one model.

**Solve.** `mpc_solve(qp, settings=None, warm=None)` first solves the problem without rows and
with zero slack by the Riccati recursion. If that plan meets every row up to $10^{-12}$ of the
row's terms it is the optimum of the full problem, since the slack's cost is lowest at zero and
couples to nothing, and the result has path "riccati" and zero iterations. Otherwise
`solve_horizon(qp, settings=settings, warm=warm)` solves the full problem, path "ipm". The
solver's status passes through unchanged: "converged" gives the states, the commands and the
slack in metres; "iteration_cap", "infeasible_certified" (possible only with terminal rows) and
"solver_failed" give no plan, since an unconverged iterate is not a plan, and
"infeasible_certified" carries the certificate's (stage, row) pairs, stage $N$ for terminal rows.
`elapsed_s` is the wall-clock time of the whole call.

### Parameters and defaults

`MPCConfig` is a frozen, keyword-only dataclass; values outside their range raise `ValueError`.

| Parameter | Default | Meaning |
|---|---|---|
| `regulator` | `LQRConfig()` | Weights, stage-cost convention and rate weight of the regulator model |
| `horizon` | 20 | Steps $N$, at least 1; 200 s at 10 s steps |
| `terminal` | `"cost"` | `"cost"`: $\beta P$ only; `"set"`: also the terminal set as hard rows |
| `beta` | 1.0 | Terminal cost inflation, at least 1 |
| `t_max` | 200 | Largest determination index of the terminal set |
| `row_samples` | 5 | Keep-out samples per step, both ends included, at least 2 |
| `instep_bound` | `"first_order"` | Tightening $V h/2$, or `"second_order"` $(Vh)^2/(8R) + A h^2/8$ |
| `penalty` | `None` | Slack price $\rho$ per m; `None` takes the rule above |
| `side` | `"auto"` | Passing side of the keep-out sphere: `"auto"`, `"+x"`, `"-x"`, `"+z"`, `"-z"`; not read by `mpc_problem` |
| `side_hysteresis` | 3 | Failed decisions in a row before an `"auto"` side is chosen again; not read by `mpc_problem` |
| `settings` | `None` | `IPMSettings` of the solves; `None` takes the defaults |

Fixed in the module: the slack offset 1 m, $\rho_q = 10^{-6}\rho/R$, the Riccati acceptance
$10^{-12}$ of each row's terms, and the LP margin $10^{-6}$ on the terminal set's extents.

### Assumptions and limits

- The keep-out rows are linearized at the reference plan, and their tightening uses speed bounds
  of the reference samples, not of the plan the solver returns. A reference that passes through
  the sphere gives rows that ask only to stay as far out as the reference, so a plan near it can
  enter the sphere; the reference must go around. The rows are a prediction; the certificate of
  the step actually flown is the controller's.
- Only the "keep_out" constraint is modelled, with the largest radius when there are several;
  approach corridors, speed limits and other constraint types are not rows of this problem.
- A ball thrust limit is replaced by its inscribed box, which forgoes up to $\sqrt 3$ of the
  thrust along the diagonals.
- The default penalty is shown exact on the twenty states of the tests. Other weights or
  geometries can need a larger $\rho$; a converged plan then shows positive slack.
- The hold-point allowance $b_h$ grows with the set's range and speed. With the discrete weights
  below and 60 s steps the set reaches 8.3 km radially and 9.8 km across track, $b_h$ would be
  103 m, more than the 50 m between the sphere and the hold point, and terminal "set" is refused.
- $\beta > 1$ and terminal "cost" carry no exact-LQR or stability claim; with terminal "cost" the
  plan need not end in the terminal set.

### How it is tested

`tests/test_mpc_problem.py`, on a 500 km circular orbit of the WGS 84 Earth with a 0.02 m/s^2
limit per axis and 10 s steps unless stated. "Discrete weights" are
$Q = \operatorname{diag}(10^{-2} I_3, 10^4 I_3)$, $R = 2500 I$ summed once per step.

- **Layout.** The slack is a fourth input with a zero dynamics column and no cross cost terms;
  every stage has $7 + m$ rows for $m$ = 2 and 5, rows with no coefficients have negative bounds,
  the first stage skips its first sample, and terminal "cost" has no terminal rows. The terminal
  cost is $\beta P$ for $\beta$ = 1 and 2, lowest at the hold point. Configurations outside their
  range and references of the wrong length are refused.
- **Unconstrained first move** (Rawlings, Mayne and Diehl, sec. 2.5.1): over 50 random states and
  $N$ = 1, 2, 5, 20, 60, the Riccati first move equals the LQR law to $10^{-10}$ relative for the
  discrete weights, the sampled cost, a one-step delay and a rate weight of 0.1; it equals
  `LQRAgent.act` for the sampled and delayed cases, and `mpc_solve` takes the Riccati path.
- **Box rows** (sec. 2.5.4): from rest at 0.5 and 0.99 of the along-track reach of the terminal
  set (201.509569 m for the box, 116.341604 m for the inscribed box of the ball), the first move
  is the LQR law to $10^{-10} a_{\max}$ on the Riccati path and to $10^{-6} a_{\max}$ by the
  interior-point solver. At 3 and 20 times the reach, $N = 30$, the box cuts the LQR move by
  0.040 and 0.380 m/s^2 (box) and 0.023 and 0.219 m/s^2 (ball), each to $10^{-6}$ relative of a
  bounded-variable least-squares solve of the condensed problem.
- **Constrained cost** (proposition 2.18): 604.528706 m behind the hold point at rest, the
  optimal cost is $1.855711116 \times 10^5$ for $N$ = 5, 10, 20 and 40 to $10^{-8}$ relative,
  equal to the condensed least-squares value, nonincreasing in $N$ and above the unconstrained
  $\tfrac12 \delta\mathbf z^T P\,\delta\mathbf z = 1.845593979 \times 10^5$.
- **Terminal cost and stability**: the two-step first-move loop has spectral radius 0.99005 with
  the DARE terminal cost (the LQR loop) and 0.99995 with none, to $10^{-5}$.
- **Sampled cost** (Van Loan): with $Q_c = \operatorname{diag}(10^{-4} I_3, I_3)$, $R_c = 10^3
  I$ and $N = 5$ from $(50, -80, 10)$ m, $(0.02, -0.01, 0.005)$ m/s, twice the cost is
  116.65796914 at 10 s steps and 122.99249126 at 60 s, to $10^{-10}$ relative.
- **Exact penalty** (Boyd and Vandenberghe, exercise 5.16): on `approach-vbar`, for ten approach
  starts and ten passes 55 to 90 m from the target, the soft solution equals the hard one to
  $10^{-6} a_{\max}$ with slack below $10^{-9}$ m; at a hundredth of the default penalty a pass
  takes positive slack; the default is $10 \times 205.958$ per m to 5%.
- **Between samples** (sec. 8.5.1): zero-thrust coasts with both ends on the sphere dip inside
  it between them by 1.783e-2 m (0.2 m/s, 10 s, $R$ = 50 m), 2.806e-1 m (1 m/s) and 6.491e-2 m
  (0.2 m/s, 5 s, $R$ = 2 m). With the reference 10% farther out, the rows at two samples per step
  fall short by at least the dip under both bounds, and at five samples by more than half of it.
  With a one-step delay the rows depend on the pending command and not on the one being issued.
- **Shifted plan** (secs. 2.4.2 and 2.7): for 20 converged plans with the terminal set on
  `approach-vbar`, 55 to 120 m from the target, the plan shifted by one step with the LQR move
  appended meets every row of the next problem to $10^{-7}$ of the row's terms.
- **Terminal set**: on `approach-vbar` the terminal rows are invariant under the LQR loop, keep
  the LQR command in the box, and keep the hold-point component of the position at 51.4593 m or
  more.
- **Extreme steps**: from $10^{-6}$ s to $10^4$ s with the discrete weights and $N = 5$, the
  problem either builds and gives the LQR first move to $10^{-9}$, or is refused: "row_samples
  too few" at $10^4$ s, and with terminal "set" "terminal set" at $10^{-6}$ and $10^{-3}$ s
  ($t^* > 20$), at 60 s (an allowance of about 103 m on the hold row, above the 50 m between the
  hold point and the sphere) and at 600 s (no bound between samples on the hold row).
- **Solver outcomes**: "iteration_cap", "infeasible_certified" and "solver_failed" pass through
  with no plan, and the arguments reach `solve_horizon` unchanged; a converged solve is split
  into commands and slack in metres; one iteration allowed gives "iteration_cap"; from 500 m
  behind the hold point the terminal set is out of reach in 10 steps and the solve is
  "infeasible_certified" with a terminal row in the certificate.

### Example

```python
import numpy as np

from vbar.agents.lqr import regulator_model
from vbar.agents.mpc import MPCConfig, mpc_problem, mpc_solve
from vbar.scenarios import make

view = make("approach-vbar").agent_view()
config = MPCConfig(horizon=20)
model = regulator_model(view, config.regulator)


def lqr_rollout(z0, steps):
    """Clipped LQR rollout from z0, the reference the keep-out rows are linearized at."""
    x = np.empty((steps + 1, len(z0)))
    u = np.empty((steps, 3))
    x[0] = z0
    for k in range(steps):
        u[k] = np.clip(model.us - model.k @ (x[k] - model.zs), -view.max_accel, view.max_accel)
        x[k + 1] = model.phi @ x[k] + model.gamma @ u[k]
    return x, u


# 30 m behind the hold point at rest: no row binds and the plan is the LQR one.
z0 = np.array([0.0, -130.0, 0.0, 0.0, 0.0, 0.0])
plan = mpc_solve(mpc_problem(model, view, z0, lqr_rollout(z0, 20), config))
print(plan.status, plan.path, plan.u[0])

# 300 m on the far side of the target, closing at 0.1 m/s: the thrust box binds.
z0 = np.array([0.0, 300.0, 0.0, 0.0, -0.1, 0.0])
plan = mpc_solve(mpc_problem(model, view, z0, lqr_rollout(z0, 20), config))
print(plan.status, plan.path, plan.iterations, plan.u[0], plan.slack.max())
```

## Controller

`MPCAgent(config=None, name="mpc")` is the hold-point MPC baseline. At every decision it builds
the horizon problem above at the observed state, solves it with `mpc_solve` and applies the first
move (Rawlings, Mayne and Diehl, sec. 2.5); when the solver gives no plan it falls back to the
previous plan or to the LQR law. It takes the continuous action modes and hold or berthing goals,
as `LQRAgent` does ([lqr.md](lqr.md)).

### How it is built

**Decision.** Each `act(obs, info)` reads only `info["time_s"]` and does the following.

1. **State.** $\mathbf z_0$ is built as `LQRAgent` builds it: the observed Hill state, then the
   pending commands converted from action units to m/s^2 (a wait that is not a whole number of
   steps raises "pending commands are not aligned to the decision step"), or the agent's own last
   command when a rate weight is set without delay.
2. **Reference and warm start.** With a plan $(\mathbf z_{0..N}, \mathbf u_{0..N-1})$ from the
   previous decision, the shifted plan is $\mathbf u_1, \dots, \mathbf u_{N-1}$ followed by the LQR
   move at its last state clipped to the box, $\operatorname{clip}(\mathbf u_s - K(\mathbf z_N -
   \mathbf z_s), \pm a_{\text{box}})$, flown on the model from the new $\mathbf z_0$. The keep-out
   rows are linearized at it, and the interior-point solve starts from it with every slack at
   1 mm ($\sigma = 1.001$ m). The shifted plan is its own reference, so it meets the rows that ask
   only for its own range to rounding; with slacks of $10^{-14}$ m there the solver's first Newton
   system does not factor and the solve ends "solver_failed" at iteration 0, which the millimetre
   of slack avoids. It moves only the starting point; no row and no slack bound is relaxed, so
   the solution is that of the same problem. With no plan the reference is the clipped LQR
   rollout from $\mathbf z_0$, or the passing-side rollout below, and the solver starts from its
   own default.
3. **Lifting.** A keep-out row asks only for the range of its reference sample when that is
   below $R + b$, so a reference inside the sphere keeps later plans inside it. In every
   reference, states after the first that are nearer than $1.1R$ to the target are moved
   radially out to $1.1R$, which leaves room for what a reference loses within a step at closing
   speeds up to $0.1R/\Delta t$; the rows of a reference outside $1.1R$ then ask for $R + b$.
   The shifted plan is lifted too. Its last step comes from the LQR law, which heads straight
   for the hold point, so the tail of a plan can pass through the sphere; unlifted, its rows ask
   only for that tail's own range and every later plan inherits it. In one run of the default
   agent from the seed-0 starts without this lift, the chaser entered the sphere to 21.5 m from
   the target on `approach-vbar` and 6.2 m on `approach-hbar`; with it, both runs complete the
   dwell (see the whole-episode tests below). Where the shifted plan is outside $1.1R$ it is
   unchanged and stays feasible for its rows.
4. **Solve.** `mpc_solve` takes the Riccati path when the plan without rows meets every row,
   else runs `solve_horizon`; the config's `settings` reach every solve.
5. **Command.** A converged plan gives level 0 and its first move, clipped to the box (which moves
   an interior-point solution by at most the solver's feasibility tolerance). Otherwise the
   shifted plan gives level 1 and its first move. With no plan either, level 2 applies the LQR
   law $\mathbf u_s - K(\mathbf z_0 - \mathbf z_s)$ clipped per axis on a box limit or scaled onto
   the ball on a ball limit, the `LQRAgent` command. An unconverged iterate is never applied; the
   solver returns none. A plan, solved or shifted, is kept for the next decision; level 2 leaves
   none. The command is issued in the view's action mode.
6. **Certificate.** The applied step is held for $\Delta t$ on the CW model from the observed
   state, with the command acting during it (the oldest pending command under a delay), and
   `segment_minimum(KeepOut(R), ...)` over 20 samples of the step gives a certified lower bound
   on the keep-out margin (the larger of the first- and second-order bounds of `vbar.constraints`
   on each interval). A level-0 plan with a negative bound is solved once more with the keep-out
   rows of its first step raised by the shortfall plus the keep-out resolution, 1 mm, from the
   solver's own start, and the new plan is applied if it converges, the first one otherwise.

**Passing side.** The side of the target is contested when the straight segment from the
position to the hold point passes within $1.2R$ of the target. Each side has a waypoint
$\pm 1.2R\,\mathbf e$, with $\mathbf e$ the unit component of $\hat{\mathbf x}$ orthogonal to the
segment, or of $\hat{\mathbf z}$ when that of $\hat{\mathbf x}$ is shorter than 0.1 (the radial
axis within about 6 degrees of the segment, where navigation noise would set its direction).
Labels are "+x", "-x", "+z" and "-z". A side's reference is the clipped LQR rollout to the
waypoint at rest, with its holding input from `steady_state_input`, until it comes within $0.2R$
of the waypoint, and to the hold point after that.

The waypoints sit at $1.2R$ rather than at $R + 2b$. The tightening $b$ is computed per row
inside `mpc_problem` from the reference's own speed bounds and is a fraction of a metre at the
catalogue's speeds, so waypoints at $R + 2b$ sit about a metre outside the sphere and a
rollout toward them can pass inside it. From $(3, 80, 0)$ m closing at 0.25 m/s on
`approach-vbar`, the 20-step reference toward the +x waypoint comes within 49.84 m of the
target with the waypoint at $R + 2b$ for $b$ = 0.4 m, and within 51.88 m at $1.2R$ (60 m for the
50 m sphere). The contested distance uses the same $1.2R$, so a side is contested exactly
when the straight segment passes nearer than the waypoints. The leg ends within $0.2R$ of the
waypoint, and the x side gives way to z below an orthogonal component of 0.1.

With `side="auto"`, a contested decision with no side yet solves both sides' problems from the
solver's own start, since the shifted plan belongs to one side at most, and commits for the
episode to the lower converged objective (the first side on a tie). The plan then carries the
side, and later decisions use the shifted plan. The side is chosen again only after
`side_hysteresis` decisions in a row at level 1 or 2 while it is contested, so solver failures
can change it and navigation noise cannot. A fixed side solves one problem; when its axis lies
along the segment it has no waypoint and the reference is the plain rollout. Each side is a
convex problem; the nonconvex choice between them is made once.

**Log.** `log` has one entry per decision and `fallbacks` counts decisions by level, so it sums
to the number of decisions; `plan` is the (states, commands) plan followed, None after level 2.
They are the agent's own bookkeeping; the recorder and metrics never read them as scores.

| Key | Meaning |
|---|---|
| `time_s` | `info["time_s"]` of the decision (s) |
| `status`, `path`, `conflicting` | Of the solve whose plan is applied, or of the first side's when no side converged |
| `iterations` | Interior-point iterations of every solve of the decision, 0 on the Riccati path |
| `elapsed_s` | Wall-clock time of the decision (s) |
| `solver`, `tolerance` | "mpc_solve", and the larger of the settings' feasibility and gap tolerances |
| `level`, `fallback` | 0, 1 or 2, and whether the level is above 0 |
| `warm_start` | Whether an interior-point solve started from the shifted plan |
| `resolved` | Whether the certificate asked for the second solve |
| `certified_bound` | Certified lower bound on the keep-out margin over the applied step (m) |
| `certified` | `certified_bound` $\ge 0$ and no plan slack above 1 mm |
| `predicted_margin` | Least keep-out margin of the plan at its row samples (m); None at level 2 |
| `max_slack` | Largest keep-out slack of a plan solved now (m); None at levels 1 and 2 |
| `side` | The configured fixed side, or the committed "auto" side; None until one is chosen |
| `in_xf` | Whether the plan ends in the terminal set, to the solver's feasibility tolerance; None at level 2 or when the set could not be built |

Without a keep-out constraint the keep-out entries are None and `certified` is True.

**Reset.** `reset(view, rng)` builds the regulator model, builds the problem once at the hold
point, so "row_samples too few" and, with terminal "set", the terminal set's errors raise there
rather than mid-episode, and builds the terminal set for the `in_xf` entry. With terminal "cost" a
set that cannot be built gives a `UserWarning` starting with the set's error, and `in_xf` is None.
It clears the plan, the side, the failure count, the last command, the log and the counts; the
controller has no random choices and does not use `rng`.

### Parameters and defaults

The agent reads every field of `MPCConfig` (table above); `side` and `side_hysteresis` are read
only here. Fixed in the module:

| Constant | Value | Meaning |
|---|---|---|
| `WAYPOINT_RADIUS` | 1.2 | Waypoint radius and contested-segment distance, in units of $R$ |
| `WAYPOINT_REACHED` | 0.2 | Distance to the waypoint, in units of $R$, where its leg of the reference ends |
| `REFERENCE_RADIUS` | 1.1 | Radius, in units of $R$, that reference states are lifted to |
| `PARALLEL_TOL` | 0.1 | Shortest orthogonal component of $\hat{\mathbf x}$ for an x side |
| `WARM_SLACK_M` | 1e-3 m | Slack of the warm start |
| `VIOLATION_SAMPLES` | 20 | Samples of the held step in the certificate, the environment's default |

### Assumptions and limits

- The model is the exact zero-order-hold CW step. On other dynamics, or with navigation or
  execution errors, the plan and the certificate are of the model from the observed state, not
  of the flown step; the certificate is logged, not enforced.
- Only the "keep_out" constraint is modelled; approach corridors, speed limits and other
  constraint types are neither rows nor certified.
- The committed side makes the law depend on the episode's history, not only on the state.
- A contested decision with no side costs two interior-point solves. Each decision makes one more
  solve only when its certificate fails. A solve that does not converge is not retried: it falls
  back.
- With terminal "cost" there is no stability claim; `in_xf` shows whether the plan ends where the
  LQR law keeps the constraints. With terminal "set", starts from which the set is out of reach
  in $N$ steps fly the LQR law at level 2 until it comes within reach.
- On a ball limit the plans use the inscribed box and level 2 uses the ball.
- No wall-clock budget: the iteration cap stops a solve, so runs reproduce across machines. On one
  machine an interior-point decision at $N$ = 10 to 20 takes 0.1 to 0.3 s and a Riccati decision
  about 10 ms, so episodes that keep a row binding are about a hundred times slower than the LQR
  agent's.

### How it is tested

`tests/test_mpc_agent.py`, on a 500 km circular orbit of the WGS 84 Earth with 10 s steps, a
0.02 m/s^2 limit per axis and the 50 m sphere, default weights unless stated.

- **Hold point**: on `approach-rbar` from the hold point at rest, ten commands equal the hold
  input $3n^2 \cdot 100$ m $= 3.6749087912 \times 10^{-4}$ m/s^2 radially to $10^{-10}$ relative,
  all on the Riccati path with the interior-point solver never called; the plan stays at the hold
  point to $10^{-9}$ m, 50 m outside the sphere, certified and in the terminal set.
- **LQR equivalence**: from $(-120, 25, 10)$ m at rest on `approach-rbar`, twenty decisions equal
  the `LQRAgent` actions to $10^{-10}$ relative. With a 10 s delay and the pending-command
  channel, the discrete weights and `t_max=0`, twenty decisions from 150 m behind the V-bar hold
  point equal the delay-augmented `LQRAgent`; a 15 s delay is refused, "not aligned".
- **Thrust limit**: from 400 m below the R-bar hold point the box binds and the commands stay in
  it; on a ball limit they stay in the inscribed box.
- **Keep-out**: from $(3, 80, 0)$ m closing at 0.25 m/s, the plan's row samples are at the radius
  or beyond to $10^{-6}$ m, the logged predicted margin matches them, and the certified bound is
  nonnegative and below a 2001-sample scan of the applied step; the clipped LQR plan from the
  same state enters the sphere.
- **Terminal set**: `in_xf` is True at the hold point and False from 400 m below it with
  $N = 10$; at 60 s steps with the discrete weights the set fails, which warns with terminal
  "cost" and raises with terminal "set"; at $10^4$ s steps "row_samples too few" raises at reset.
- **Warm start**: each interior-point solve after the first starts from the previous plan
  shifted, with the clipped LQR move at its end, slacks at 1 m or more, and states equal to the
  previous plan's to $10^{-6}$ m.
- **Fallbacks**, with `solve_horizon` replaced by a stub: the outcomes "iteration_cap",
  converged, "iteration_cap", "solver_failed", converged, "solver_failed" give the levels
  2, 0, 1, 1, 0, 1 and `fallbacks` [2, 3, 1]; level 2 is the clipped LQR command and level 1 the
  previous plan's next moves, also when the stub returns an iterate. One iteration allowed gives
  "iteration_cap" and the LQR law at every decision. With terminal "set" and $N = 10$ from
  $(0, 400, 0)$ m, 500 m from the V-bar hold point beyond the target, the solve is
  "infeasible_certified" with a terminal row in `conflicting`, at level 2.
- **Passing side**: on `approach-hbar:nav1` from $(0, 0, 150)$ m at rest, symmetric under
  $(x, y) \to (-x, -y)$, the side is committed for four decisions on two seeds, while fresh agents
  at the same observations choose both sides; mirrored states choose mirrored sides; fixed sides
  at mirrored states give mirrored commands to $10^{-6} a_{\max}$ with one solve each; with
  failures at decisions 1, 2 and 4, 5, 6 only decisions 0 and 7 compare both sides.
- **Log and records**: every entry has the keys above, `fallbacks` sums to the decisions, a
  recorded episode keeps one entry per decision and its solver metrics read them, and the
  recorder and metrics never read `fallbacks`, `certified_bound`, `max_slack`,
  `predicted_margin` or `in_xf`.
- **Sign check** (`vbar.agents.sign_check`): passed, with first commands equal to the LQR
  agent's to $10^{-10}$ relative.
- **Agent rules**: "disc27", "impulse_coast" and tasks other than a hold are refused at reset;
  "normalized", "accel" and "thrust" give equal fractions to $10^{-12}$; a reset agent repeats a
  fresh one bitwise; two agents from one config stepped interleaved repeat separate runs bitwise;
  on a hand-made view the agent reads only `info["time_s"]`.
- **Whole episodes** (marked slow): on `approach-rbar` (210 steps), `approach-vbar` and
  `approach-hbar` (360 steps) the agent completes the dwell without entering the sphere, where
  on the last two the LQR agent from the same start enters it; past the target from the closing
  state with 20 constraint samples per step, the flown margin stays nonnegative in continuous
  time and is at or above each step's certified bound and within 2 mm of it.

### Example

```python
import numpy as np

from vbar.agents.mpc import MPCAgent, MPCConfig
from vbar.scenarios import make

# 150 m beyond the target on V-bar: the target is in the way, so a side is chosen.
env = make("approach-vbar", training=True)
obs, _ = env.reset(seed=0, options={"start_state": (0.5, 150.0, 0.0, 0.0, 0.0, 0.0)})
agent = MPCAgent(MPCConfig(horizon=10))
agent.reset(env.agent_view(), np.random.default_rng(1))
for _ in range(3):
    obs, *_ = env.step(agent.act(obs, env.agent_info()))
for entry in agent.log:
    print(entry["level"], entry["path"], entry["side"], entry["warm_start"], entry["certified"])
print(agent.fallbacks)
```
