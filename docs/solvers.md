# Solvers

Code: `vbar.solvers` (`HillScaling`, `Rows`, `ActuatorSet`, `project`, `Projection`, `StageQP`,
`riccati_solve`). The
optimization problems that safety filters and horizon methods solve, built on numpy and scipy only,
deterministic and with a defined status for every outcome.

## Hill-frame units

`HillScaling(length, n)` gives nondimensional units for Hill-frame problems: length $L$ (m), time
$1/n$ (s), speed $nL$ and acceleration $n^2L$. Hill-frame quantities mix metres, seconds and a mean
motion near 1e-3 rad/s, so a problem posed in SI units has entries spread over many orders of
magnitude. Newton-type methods are affine invariant in exact arithmetic, but their stopping tests
and factorizations are not, and first-order methods stall on such conditioning (Boyd and
Vandenberghe, *Convex Optimization*, secs. 9.3 and 9.5). In these units positions, speeds and
accelerations of a proximity operation are of order one and the Clohessy-Wiltshire equations have
unit mean motion:

$$
\ddot x' - 2\dot y' - 3x' = a_x', \qquad \ddot y' + 2\dot x' = a_y', \qquad \ddot z' + z' = a_z',
$$

with $x' = x/L$, $t' = nt$ and $a' = a/(n^2L)$. `state` and `state_si` convert states of shape
`(..., 6)` (m, m/s), `accel` and `accel_si` accelerations of any shape (m/s^2), `time` and
`time_si` durations (s). $L$ and $n$ must be finite and positive; $L$ is usually the scenario's
range scale.

The tests check the units of each quantity, the round trips, that `cw_rhs` with $n = 1$ on scaled
states equals the scaled SI derivative, and that the exact transition over 1 ms, 10 s, 600 s and
5400 s in scaled units equals the SI propagation to 1e-12 relative.

`stage(qp)` returns a horizon problem (below) in these units. With
$D = \operatorname{diag}(1/L, 1/L, 1/L, 1/(nL), 1/(nL), 1/(nL))$ on states and $1/(n^2L)$ on
inputs, it maps $A \to DAD^{-1}$, $B \to n^2L\,DB$, $c \to Dc$ and $x_0 \to Dx_0$, and transforms
the costs and rows so that every plan has the same objective value and the same row values in both
units. The scaled solution maps back through `state_si` and `accel_si`, and the row multipliers
are unchanged and the dynamics multipliers map by $D^{-1}$. The tests solve a horizon problem
in both units and compare the SI plans to 1e-9 relative (measured: 2e-15).

```python
from vbar.dynamics import EARTH, mean_motion
from vbar.solvers import HillScaling

units = HillScaling(length=100.0, n=mean_motion(EARTH.radius + 400e3))
print(units.state([-100.0, -500.0, 0.0, 0.0, 0.2, 0.0]).round(4))  # 0.2 m/s is 1.7678 n L
print(round(float(units.accel(2e-3)), 3))  # 2 mm/s^2 is 15.625 n^2 L
print(round(units.time(600.0), 4))  # 600 s is 0.6788 rad of orbit
```

## Exact projection onto linear rows

`project(u_nom, rows, actuator=None, weight=None)` solves

$$
\min_u \tfrac12 (u - u_{\text{nom}})^T H (u - u_{\text{nom}}) \quad \text{subject to} \quad
a_i \cdot u \ge b_i,\ i = 1..k, \qquad u \in U,
$$

for a three-axis command $u$, with $H = \operatorname{diag}(\text{weight})$ positive (the identity
by default), `Rows(a, b, names)` holding at most 32 rows, and `ActuatorSet(a_max, layout)` the box
$|u_i| \le a_{\max}$ or the ball $|u|_2 \le a_{\max}$ (none by default). This is the problem of a
safety filter with control barrier function, speed or closing-speed rows. Rows are scaled to unit
normals internally, so rows in different units mix freely.

### Method

At the solution of a QP in three variables at most three linearly independent rows are active.
Every subset $S$ of at most three rows (box faces count as rows) whose unit normals have smallest
singular value above 1e-12 gives the equality-constrained problem with $a_S u = b_S$, solved from
its KKT system

$$
\begin{bmatrix} H & -A_S^T \\ A_S & 0 \end{bmatrix}
\begin{bmatrix} u \\ \lambda_S \end{bmatrix} =
\begin{bmatrix} H u_{\text{nom}} \\ b_S \end{bmatrix},
$$

all subsets at once as stacked 6x6 systems. A candidate is optimal when every row holds and every
multiplier is nonnegative, the KKT conditions (Boyd and Vandenberghe, sec. 5.5.3, eq. 5.49).
With linear rows these conditions are necessary and sufficient whenever the rows can be met, so a
feasible problem always has a winning subset; the lowest objective among winners is returned.
With 32 rows and a box that is 9178 subsets, and there is no iteration count or starting point.

A ball limit is dualized. For a fixed multiplier $\mu \ge 0$ on $|u|^2 \le a_{\max}^2$ the
problem is the rows-only projection with $H + \mu I$, and $|u(\mu)|$ is nonincreasing in $\mu$.
When the rows-only solution lies outside the ball, $\mu$ is found by Brent's method, bisection
safeguarded by secant steps, on $a_{\max}/|u(\mu)| - 1$, which is linear in $\mu$ when $H$ is a
multiple of $I$ and no row is active; the radius is met to 1e-12 $a_{\max}$.

### Infeasibility and the certificate

When no subset wins, the phase I problem minimizes the sum of infeasibilities (Boyd and
Vandenberghe, sec. 11.4.1, eq. 11.20),

$$
\min_{u, s} \textstyle\sum_i s_i \quad \text{subject to} \quad
\hat a_i \cdot u + s_i \ge \hat b_i,\ s_i \ge 0, \qquad u \in \text{box},
$$

with unit rows $\hat a_i$, by `scipy.optimize.linprog` (HiGHS); a ball is replaced here by its
bounding box. The LP's dual gives $\lambda \ge 0$, which is checked in numpy as a certificate of
infeasibility (a theorem of alternatives, sec. 5.8.3): $b^T\lambda$ must exceed the largest
$\lambda^T A u$ over the actuator set, that is $a_{\max}|A^T\lambda|_1$ for a box,
$a_{\max}|A^T\lambda|_2$ for a ball and 0 with no limit (where $A^T\lambda$ must also vanish to
1e-9 of $|\lambda|_1$). Then for every $u$ in the set $\lambda^T(Au - b) < 0$, so some row fails.
The status is "infeasible_certified" only when the check passes, and "solver_failed" otherwise.
With a ball and rows that can be met, the nearest row-feasible point to the origin, $u_m$, decides:
if $|u_m| > a_{\max}$ its multipliers $\lambda_m$ satisfy $u_m = A^T\lambda_m$ and
$b^T\lambda_m = |u_m|^2$, so $b^T\lambda_m > a_{\max}|A^T\lambda_m|_2$.

`certificate` holds $\lambda$ over the rows, followed for a box by its six faces in the order
"box+x", "box-x", "box+y", "box-y", "box+z", "box-z" ("box+x" is the face $u_x = a_{\max}$),
chosen so that $A^T\lambda = 0$ and $b^T\lambda > 0$ over rows and faces together. `conflicting`
names the rows, faces and "ball" in its support (entries above 1e-9 of the largest), the
constraints that cannot hold together. The returned command is the one nearest
$u_{\text{nom}}$ in the actuator set among those that violate no row more than the phase I point
does; with a ball it is then scaled onto the ball. `phase1_margin` is the sum of the remaining
violations $(b_i - a_i \cdot u)/|a_i|$ in command units.

### Statuses and outputs

- "feasible": `u` is the optimum, `multipliers` the row multipliers ($\ge 0$, in objective units
  per row unit), `active` the rows with a positive multiplier followed by the active faces or
  "ball", and `phase1_margin` the largest row violation at `u`, $\le 0$ up to rounding and $-\infty$
  without rows.
- "infeasible_certified": proven by `certificate`; `multipliers` are zero.
- "solver_failed": no answer and no proof, a defect of the solver rather than of the command;
  `u` is the best command found.

Rows are accepted to 1e-10 of the problem's scale, the largest of $a_{\max}$, $|b_i|/|a_i|$ and
$|u|_\infty$, so the answer does not depend on the units of the command. The nominal command is
left out of that scale on purpose: the multipliers carry $H(u - u_{\text{nom}})$ and get its
rounding allowance, but a large nominal command must not loosen a safety row. Equal inputs give
bitwise equal outputs.

### Assumptions and limits

- Three axes, a diagonal weight, at most 32 rows; row names must be unique and differ from the
  actuator names.
- A zero row $0 \cdot u \ge b$ holds when $b \le 0$ and conflicts on its own otherwise.
- When the rows touch the ball at a single point the feasible set is that point, Slater's
  condition fails and finite multipliers do not exist in general, as in Boyd and Vandenberghe,
  exercise 5.26; the point is returned as feasible with infinite multipliers on its active rows.
- Rows within 1e-12 of parallel count as dependent; nearly parallel rows active together give
  large multipliers. A subset of such rows whose KKT or correction system rounds to an exactly
  singular matrix is solved again on its own by least squares, and its candidate is checked
  like any other.

### Validation

The tests in `tests/test_solvers_qp.py` check:

- Boyd and Vandenberghe, exercise 4.3, the box-constrained QP with
  $P = [[13, 12, -2], [12, 17, 6], [-2, 6, 12]]$ and $q = (-22, -14.5, 13)$, posed as a projection
  through the Cholesky factor of $P$: $x^* = (1, 0.5, -1)$, $p^* = -21.625$ and multipliers 1 on
  $x_1 \le 1$ and 2 on $x_3 \ge -1$, to 1e-12.
- The halfspace projection $u = u_0 + \max(0, b - a\cdot u_0)\,a/|a|^2$ (sec. 8.1.1) and its
  weighted form, on 200 random instances each, to 1e-12; box clipping and radial scaling onto the
  ball without rows.
- One row under a box or ball on 500 random instances each: feasible exactly when
  $a_{\max}|a|_1 \ge b$ (box) or $a_{\max}|a|_2 \ge b$ (ball), and the solution equal to a
  one-dimensional search on the row's multiplier, $u(\lambda) = P_U(u_{\text{nom}} + \lambda a)$,
  to 1e-9 $a_{\max}$.
- Two to six random rows under a box or a ball, 200 feasible instances each, against
  `scipy.optimize.minimize` (SLSQP) to 1e-6 relative in the objective and 1e-6 $a_{\max}$ in $u$,
  with every reported active row tight; one instance with 32 rows; the KKT conditions on 200
  instances without a limit, and at a vertex where four rows meet.
- Certified infeasibility with a certificate verified independently: two opposite rows, a row the
  box cannot reach (named with the two faces it needs), a ball too small for a row, a zero row,
  and 300 random conflicts, each naming the two conflicting rows.
- The single-point feasible set of two rows meeting the box at a vertex, and a row tangent to the
  ball.
- Bitwise determinism, the same answer in SI and in `HillScaling` units to 1e-12 $a_{\max}$, and
  the refusal of 33 rows.

```python
import numpy as np

from vbar.solvers import ActuatorSet, Rows, project

thrust = ActuatorSet(2e-3, "box")  # m/s^2 per axis
rows = Rows(np.array([[0.0, 1.0, 0.0]]), np.array([-1e-3]), ("closing",))
result = project(np.array([0.0, -2e-3, 1e-3]), rows, thrust)
print(result.status, result.u, result.active)  # feasible [ 0.    -0.001  0.001] ('closing',)

reach = Rows(np.array([[1.0, 1.0, 0.0]]), np.array([5e-3]), ("reach",))
result = project(np.zeros(3), reach, thrust)
print(result.status, result.conflicting)  # infeasible_certified ('reach', 'box+x', 'box+y')
print(result.u)  # [0.002 0.002 0.   ], the least-violation command
```

## Horizon problems and the Riccati solve

`StageQP(x0, A, B, c, Q, q, P, p, C, e, C_N, e_N)` holds a convex QP over $N \ge 1$ stages,

$$
\min \sum_{k=0}^{N-1} \tfrac12 z_k^T Q_k z_k + q_k^T z_k + \tfrac12 x_N^T P x_N + p^T x_N,
\qquad z_k = (x_k, u_k),
$$

subject to $x_0 = x_{\text{init}}$, $x_{k+1} = A_k x_k + B_k u_k + c_k$, stage rows
$C_k z_k \ge e_k$ and terminal rows $C_N x_N \ge e_N$. Shapes are `A` (N, nx, nx), `B`
(N, nx, nu), `c` (N, nx), `Q` (N, nx+nu, nx+nu), `q` (N, nx+nu), `P` (nx, nx), `p` (nx,), `C`
(N, m, nx+nu), `e` (N, m), `C_N` (mN, nx), `e_N` (mN,); `C`, `e` and `C_N`, `e_N` are None
together for no rows. Construction refuses wrong shapes, values that are not finite, an empty
horizon and Hessians whose asymmetry exceeds 1e-12 of their largest entry, with `ValueError`; the
Hessians are stored symmetrized. The states stay variables and the dynamics stay equality rows, so
the KKT system is banded with one block per stage.

`riccati_solve(qp)` solves the problem without its rows by the backward Riccati recursion of
Rawlings, Mayne and Diehl, *Model Predictive Control*, 2nd ed., sec. 8.8.3, eqs. (8.51) to (8.54).
From $P_N = P$, $p_N = p$, each stage forms

$$
H_{uu} = Q_{uu} + B^T P_{k+1} B, \quad H_{ux} = Q_{ux} + B^T P_{k+1} A, \quad
h_u = q_u + B^T (P_{k+1} c + p_{k+1}),
$$

factors $H_{uu}$ by Cholesky (`scipy.linalg.cho_factor`), and sets $K_k = -H_{uu}^{-1} H_{ux}$,
$k_k = -H_{uu}^{-1} h_u$,

$$
P_k = Q_{xx} + A^T P_{k+1} A + H_{ux}^T K_k, \qquad
p_k = q_x + A^T (P_{k+1} c + p_{k+1}) + K_k^T h_u,
$$

then replaces $P_k$ by $(P_k + P_k^T)/2$. A forward rollout gives $u_k = K_k x_k + k_k$, the states,
and the dynamics multipliers $y_k = P_{k+1} x_{k+1} + p_{k+1}$, the gradient of the cost to go, in
the Lagrangian $f + \sum_k y_k^T (A_k x_k + B_k u_k + c_k - x_{k+1})$. The cost is
$O(N (n_x + n_u)^3)$. The optional `extra = (dQ, dq, dP, dp)` adds terms to the costs and
`rhs = (c, x0)` replaces the affine terms and initial state for one solve, which lets a method
with rows pass its barrier terms and residuals through the same sweep.

The symmetrization is cheap and keeps $P_k$ exactly symmetric. Its effect depends on how the stage
system is solved: rounding leaves $P_k$ slightly asymmetric, and on cheap-control problems in SI
units a recursion that solves the asymmetric $H_{uu}$ by LU is off by 33% in $u$ on our 40-stage
CW test problem, while this implementation, which factors one triangle of $H_{uu}$ by Cholesky,
matches the dense KKT solve to about 1e-15 with or without the step.

### Assumptions and limits

- $Q_k$ and $P$ are symmetric positive semidefinite and every $H_{uu}$ of the recursion is
  positive definite; a Cholesky failure raises `ValueError` naming the stage. Definiteness of $Q_k$
  is not checked beyond that.
- The rows are ignored; a method that enforces them calls this solve for each of its steps.
- The inputs are never condensed. On CW with 60 s steps the condensed Hessian of 200 stages has a
  condition number above 1e10, while the stage-wise solve is unaffected.

### Validation

The tests in `tests/test_solvers_stage.py` compare against the KKT system of the whole horizon
assembled and solved by a sparse LU, which eliminates nothing:

- Exact CW with 40 stages of 10, 15 and 20 s, affine terms, small input-state cross terms,
  $Q = \operatorname{diag}(10^{-2} \times 3, 1 \times 3)$ and $R = 10^3 I$: states and inputs to
  1e-10 relative, dynamics multipliers to 1e-9.
- The same recursion solving the unsymmetrized stage system by LU, written in the test, is off by
  more than 10% in $u$ on that problem (33% measured), so the agreement above is not automatic;
  the module's $P_k$ stays symmetric to rounding at every stage.
- `HillScaling.stage`: the problem solved in SI and in Hill units gives the same states, inputs,
  objective and row values to 1e-9 relative, with the dynamics multipliers mapped by $D^{-1}$.
- `extra` equals the problem rebuilt with those terms added, and `rhs` the problem rebuilt with
  the given affine terms and start, to 1e-10 relative.
- With the DARE solution as terminal cost, the first move equals the LQR gain's to 1e-10.
- One stage, and 200 stages of 60 s where the condensed Hessian's condition number exceeds 1e10,
  each to 1e-9 against the sparse solve.
- Refusal of mismatched shapes, asymmetric Hessians, an empty horizon, inconsistent rows and an
  input block that is not positive definite.

```python
import numpy as np

from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.solvers import StageQP, riccati_solve

n = 40
a, b = cw_discrete(mean_motion(EARTH.radius + 500e3), np.full(n, 20.0))
cost = np.zeros((9, 9))
cost[:6, :6] = np.diag([1e-2] * 3 + [1.0] * 3)
cost[6:, 6:] = 1e2 * np.eye(3)
qp = StageQP(
    x0=np.array([0.0, -150.0, 0.0, 0.0, 0.0, 0.0]),
    A=a,
    B=b,
    c=np.zeros((n, 6)),
    Q=np.repeat(cost[None], n, axis=0),
    q=np.zeros((n, 9)),
    P=np.diag([1.0] * 3 + [100.0] * 3),
    p=np.zeros(6),
    C=None,
    e=None,
    C_N=None,
    e_N=None,
)
x, u, y = riccati_solve(qp)
print(x.shape, u.shape, y.shape)  # (41, 6) (40, 3) (40, 6)
print(np.abs(u).max().round(4))  # 0.3252, the largest input (m/s^2)
```
