# Model predictive control

Code: `admissible_set` in `vbar.agents.mpc`. Hill frame ([frames.md](frames.md)), SI units.

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
