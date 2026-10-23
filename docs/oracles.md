# Fuel oracles

Code: `primer` and `dual_bound` in `vbar.oracles.primer`, exported from `vbar.oracles`. Hill
frame, metres, seconds.

An oracle answers how little delta-v a scenario needs. A plan that reaches the goal gives an
upper bound; only a dual point gives a lower bound. `dual_bound` turns any set of Lagrange
multipliers into a lower bound that is a theorem for exact Clohessy-Wiltshire (CW) motion, so a
run that spends less than it has hit a simulator, actuator-accounting or metric bug. Its
validity never depends on how the multipliers were found: a solver only makes them good, not
valid.

## How it is built

**Primer vector.** With $\tau = t_j - t$ the time left to a constraint time $t_j$, $B = [0;\,I]$
and $\Phi$ the CW transition matrix, the primer of a multiplier $\nu_j \in \mathbb R^6$ is

$$
\mathbf p_j(t) = B^T \Phi(\tau)^T \nu_j = \Phi_{rv}(\tau)^T \nu_{j,r} + \Phi_{vv}(\tau)^T \nu_{j,v},
$$

with $s = \sin n\tau$, $c = \cos n\tau$:

$$
\begin{aligned}
p_x &= \tfrac{s}{n}\nu_x - \tfrac{2(1-c)}{n}\nu_y + c\,\nu_{vx} - 2s\,\nu_{vy}, \\
p_y &= \tfrac{2(1-c)}{n}\nu_x + \tfrac{4s - 3n\tau}{n}\nu_y + 2s\,\nu_{vx} + (4c - 3)\,\nu_{vy}, \\
p_z &= \tfrac{s}{n}\nu_z + c\,\nu_{vz}.
\end{aligned}
$$

`primer(n, nu, tau)` evaluates it from the velocity columns of `cw_discrete`, whose closed form
writes $1 - \cos n\tau$ as $2\sin^2(n\tau/2)$ and $n\tau - \sin n\tau$ without cancellation, so
short times keep full precision. At $\tau = 0$ the primer is the velocity part of $\nu$. This is
Lawden's primer vector (D. F. Lawden, *Optimal Trajectories for Space Navigation*, 1963;
Longuski, Guzman and Prussing, *Optimal Control with Aerospace Applications*, 2014, section
10.5). With several constraint times $t_1 < \dots < t_J$ the primer is
$\mathbf p(t) = \sum_{t_j > t} \mathbf p_j(t)$, which jumps at each $t_j$.

**Dual bound.** The goal set is the closed capture ball
$S = \{|\mathbf r - \mathbf r_g| \le \rho_r,\ |\mathbf v - \mathbf v_g| \le \rho_v\}$ around the
hold point and goal velocity, with $\rho_r$ the `capture_radius` and $\rho_v$ the
`capture_speed`. For any acceleration history with the state in $S$ at every $t_j$,

$$
\nu_j^T\bigl(\mathbf x(t_j) - \Phi(t_j)\mathbf x_0\bigr) = \int_0^{t_j} \mathbf p_j(t)^T \mathbf a(t)\,dt
\quad\text{and}\quad
\min_{\mathbf x \in S} \nu_j^T \mathbf x = \nu_j^T \mathbf x_g - \rho_r|\nu_{j,r}|_2 - \rho_v|\nu_{j,v}|_2,
$$

the second being the support function of the ball. Summed over $j$,
$\int_0^{t_J} \mathbf p^T \mathbf a\,dt \ge D(\nu)$ with

$$
D(\nu) = \sum_j \Bigl[\nu_j^T\bigl(\mathbf x_g - \Phi(t_j)\mathbf x_0\bigr) - \rho_r|\nu_{j,r}|_2 - \rho_v|\nu_{j,v}|_2\Bigr].
$$

Hölder's inequality gives $\mathbf p^T\mathbf a \le |\mathbf p|_*|\mathbf a| \le |\mathbf a| +
a\max(0, |\mathbf p|_* - 1)$ when $|\mathbf a| \le a$, with $|\cdot|_*$ the dual of the
actuator norm. Hence every admissible history spends at least

$$
LB(\nu) = D(\nu) - a\,E(\nu), \qquad E(\nu) = \int_0^{t_J} \max\bigl(0, |\mathbf p(t)|_* - 1\bigr)\,dt
$$

(Boyd and Vandenberghe, *Convex Optimization*, 2004, section 5.1.3 for weak duality and example
3.26 for the dual norm). For box thrusters, $|a_i| \le a$ per axis and delta-v in the 1-norm
(`norm="l1"`), the dual norm is $\ell_\infty$ applied per axis, so $E$ sums
$\max(0, |p_i| - 1)$ over the three axes. For a ball, $|\mathbf a|_2 \le a$ and delta-v in the
2-norm (`norm="l2"`), it is $\max(0, |\mathbf p|_2 - 1)$. The bound holds for every $\nu$ and is
concave in $\nu$; maximising it over $\nu$ approaches the true optimum.

**Impulsive bound.** With no acceleration limit (`accel=numpy.inf`) the same argument gives
$LB = D(\nu) / \sup_t |\mathbf p(t)|_*$, invariant to scaling $\nu$, the bound behind Lawden's
condition $|\mathbf p|_* \le 1$. It holds for every acceleration history, impulsive or not.

**Cell certificate.** $E$ and the supremum must cover the whole interval, not sample points: a
primer can exceed 1 between samples. The interval $[0, t_J]$ is cut into cells aligned with the
$t_j$, shared among the intervals $[t_{j-1}, t_j]$ by length with at least one each. On a cell of
width $h$ the primer is bounded by the larger endpoint value plus $L h / 2$, where $L$ bounds
$|d\mathbf p/dt|$ on all of $t$, from the derivatives of the closed form above:

$$
\begin{aligned}
L_x &= |\nu_x| + 2|\nu_y| + n|\nu_{vx}| + 2n|\nu_{vy}|, \\
L_y &= 2|\nu_x| + 7|\nu_y| + 2n|\nu_{vx}| + 4n|\nu_{vy}|, \\
L_z &= |\nu_z| + n|\nu_{vz}|,
\end{aligned}
$$

summed over the multipliers active on the interval; per axis for the box and
$|(L_x, L_y, L_z)|_2$ for the ball, since $\bigl||\mathbf p(s)|_2 - |\mathbf p(t)|_2\bigr| \le
|\mathbf p(s) - \mathbf p(t)|_2$. At the right edge of an interval the primer of $t_j$ is taken at
$\tau = 0$, its left limit. Each cell contributes $h\max(0, \text{bound} - 1)$ to $E$, and the
supremum is the largest cell bound.

**Rounding.** $E$ and the supremum are multiplied by $1 + 10^{-12}$, and the bound is lowered by
$64\varepsilon$ times the sum of the absolute values of its terms, so that it stays below the
exact value of the formula.

## Parameters and defaults

| Parameter | Default | Reason |
|---|---|---|
| `norm` | required | `"l1"` for `thrust_limit="box"`, `"l2"` for `"ball"`; any other pairing is refused, since the dual norm must belong to the actuator |
| `accel` | `view.max_accel` | The limit the thrusters can deliver; `numpy.inf` for the impulsive bound |
| `cells` | 8 per `view.dt` over $[0, t_J]$ | The constants $L$ are a few times $n$ times the primer's size, so the slack $L h / 2$ is a few times $n h / 2$ of it |

## Assumptions and limits

- Exact CW motion with no other forcing: a circular target orbit and no target manoeuvre. The
  bound is a theorem on the `"cw"` rung only and certifies nothing on two-body or J2 dynamics.
- The flown acceleration must stay within `accel` in the actuator norm over $[0, t_J]$. With the
  thrust drive the acceleration rises as mass falls, and an execution-error scale bias is not
  limited again; either can carry a run above `view.max_accel`, and then only the impulsive bound
  applies.
- The goal must have `hold_point`, `capture_radius` and `capture_speed` (a zero radius or speed
  is allowed); other goals raise `ValueError`. The success test is strict ($<$), the ball here
  closed, which only lowers the bound.
- Thrust-inhibit windows in the goal are not used: thrust is assumed available throughout, which
  keeps the bound valid but looser.
- Constraint times must be positive and strictly increasing, with one multiplier each, and `n`
  must equal `view.n` to $10^{-12}$ relative.
- `dual_bound` returns the dual function's value, which is negative for poor multipliers. Choosing
  the multipliers and the constraint times (for a dwell or a free arrival time) is left to the
  caller.

## Validation

Tests: `tests/test_oracle_primer.py`.

- The primer matches $B^T\Phi^T\nu$ from `cw_stm` to $10^{-12}$ of its summed terms at
  $n\tau = 0.3$ and 4, and the nine-term power series of $e^{A\tau}B$ at $n\tau = 10^{-6}$, where
  the textbook form loses four digits; at $\tau = 0$ it equals the velocity multiplier exactly.
- Boyd and Vandenberghe, sections 4.3 and 5.1.3, the half-orbit transfer from
  $(-200, -1500, 50)$ m to $(0, -100, 0)$ m at rest on a 6778 km orbit with $|a_i| \le 2 \times
  10^{-3}$ m/s^2, posed as a zero-order-hold LP solved with HiGHS: optima 0.8977056 m/s on 40
  steps and 0.8949323 m/s on 160. The bound from the 40-step dual at 20000 cells lies in
  $[0.892140, 0.8930615]$, below the 160-step optimum, and the 160-step dual gives a higher bound
  in $[0.893775, 0.8946304]$; the upper ends are the continuous-time values of these duals
  (independent midpoint sums), so a larger result would be unsound for some $\nu$.
- Soundness: 200 duals perturbed by 30% and 10 near the optimum never exceed the 640-step plan's
  cost 0.8947293 m/s; the bound is concave along five random rays; on 100 random duals with one or
  two constraint times, box and ball, at 1 to 2000 cells, it never exceeds the 20000-point
  midpoint value of $D - aE$; with two constraint times on the V-bar hold point 100 duals stay below the
  same plan's cost.
- A closed-form case of ours: from 500 m out of plane at rest to the origin at rest in half an
  orbit at 500 km with $a = 10^{-3}$ m/s^2, one burn centred on the node with half-width
  $\delta$, $2a\sin\delta = n^2 z_0$, costs $2a\delta/n$, and $\nu_z = n/\cos\delta$ certifies
  it; the bound reaches it to $2 \times 10^{-4}$ for box and ball. The impulsive bound reaches the
  impulsive optimum $n z_0$ to $2 \times 10^{-4}$. On three cells, where the primer peaks mid-cell above
  every edge value, both bounds stay below their optima.
- The impulsive bound is invariant to scaling $\nu$ from $10^{-3}$ to $10^4$; both bounds scale
  with start, goal, capture ball and acceleration limit; the capture ball lowers the bound by its
  support function to $10^{-12}$.
- Dual norms: a $\nu$ whose primer stays below 1 on every axis but reaches 1.31 in the 2-norm
  gives a box bound equal to $D(\nu)$ and a ball bound at least the midpoint
  excess below it. The ball problem's 40-step outer LP dual gives a
  bound in $[0.826710, 0.8271803]$, below the 0.8324776 m/s cost of a plan checked against the
  ball on every step, and 200 perturbed duals stay below that plan.
- Unknown norms, goals without a capture ball, non-increasing times and a multiplier count other
  than the time count raise `ValueError`.

## Example

The out-of-plane transfer above, certified finite-thrust and impulsive:

```python
import math

import numpy as np

from vbar.agents.base import AgentView
from vbar.dynamics import R_EARTH, mean_motion
from vbar.oracles import dual_bound

n = mean_motion(R_EARTH + 500e3)
tf = math.pi / n
z0, accel = 500.0, 1e-3
view = AgentView(
    scenario_id="out-of-plane",
    dt=10.0,
    max_steps=math.ceil(tf / 10.0),
    n=n,
    action_mode="accel",
    max_accel=accel,
    thrust_limit="box",
    goal={"hold_point": (0.0, 0.0, 0.0), "capture_radius": 0.0, "capture_speed": 0.0},
)
start = np.array([0.0, 0.0, z0, 0.0, 0.0, 0.0])
delta = math.asin(n**2 * z0 / (2 * accel))
nu = np.array([[0.0, 0.0, n / math.cos(delta), 0.0, 0.0, 0.0]])

finite = dual_bound(n, view, start, nu, [tf], norm="l1", cells=20_000)
impulsive = dual_bound(n, view, start, nu, [tf], norm="l1", accel=np.inf, cells=20_000)
assert 2 * accel * delta / n * (1 - 2e-4) <= finite <= 2 * accel * delta / n
assert n * z0 * (1 - 2e-4) <= impulsive <= n * z0
```

## Exact constrained LQ value

Code: `LQOracle` and `LQBound` in `vbar.oracles.lq`, exported from `vbar.oracles`. Hill frame,
SI units.

For a quadratic cost to a hold point, `LQOracle` returns a lower bound on the cost of every
controller and, when a run-time check passes, the exact least cost under the thrust box. It is
the reference for regret on a quadratic reward: on exact CW motion a controller's cost minus the
value is never negative.

### How it is built

**Deviation coordinates.** With the hold point $\mathbf x_s = (\mathbf r_h, \mathbf 0)$ and the
constant input $\mathbf u_s$ that holds it under one step, $\Gamma \mathbf u_s = (I - \Phi)\mathbf
x_s$ (`steady_state_input`; Rawlings, Mayne and Diehl, *Model Predictive Control*, 2nd ed.,
section 1.5.1), the oracle works on $\mathbf z = \mathbf x - \mathbf x_s$ and $\mathbf u =
\mathbf a - \mathbf u_s$. The step model is the zero-order-hold CW step
$\mathbf z_{k+1} = \Phi\mathbf z_k + \Gamma\mathbf u_k$ (`cw_discrete`), the stage cost is

$$
\ell(\mathbf z, \mathbf u) = \mathbf z^T Q \mathbf z + \mathbf u^T R \mathbf u + 2\mathbf z^T S \mathbf u,
$$

with no factor 1/2 (the book carries one; fold it into the weights to compare), and the thrust
box $|a_i| \le a_{max}$ becomes $-a_{max} - u_{s,i} \le u_i \le a_{max} - u_{s,i}$.

**Lower bound.** With $P$ and $K$ from `dlqr`, $\mathbf z_0^T P \mathbf z_0$ is the least
infinite sum of $\ell$ over all inputs held over each step, with or without the box, so it bounds
every controller from below.

**Exact value.** For a horizon $N$, let $V_N$ be the least value of
$\sum_{k<N} \ell(\mathbf z_k, \mathbf u_k) + \mathbf z_N^T P \mathbf z_N$ over inputs in the box.
If the optimal predicted $\mathbf z_N$ lies in the maximal admissible set $O_\infty$ of the loop
$\Phi - \Gamma K$, from which the LQR never meets the box, then $V_N$ equals the constrained
infinite-horizon optimum $V_\infty$: the $N$-step plan followed by the LQR is feasible and costs
$V_N$, so $V_\infty \le V_N$; and $\mathbf z^T P \mathbf z$ is below the cost of any tail, so
$V_N \le V_\infty$ (Rawlings, Mayne and Diehl, section 2.5.4, proposition 2.18 and the notes of
chapter 2). The rows of $O_\infty$ come from `admissible_set` (Gilbert and Tan, 1991; see
[mpc.md](mpc.md)) for the symmetric box $|u_i| \le a_{max} - |u_{s,i}|$. That box lies inside
the shifted one, so its set lies inside the exact set and the check never accepts a terminal
state the exact set would refuse.

**Solve.** The inputs are condensed: $\mathbf z_i = \Phi^i \mathbf z_0 + \sum_{j<i}
\Phi^{i-1-j}\Gamma\mathbf u_j$, with $\Phi^i$ from the closed form at $i\,\Delta t$ so that long
horizons carry no product rounding. The weights are factored, $[Q, S; S^T, R] = L L^T$ and $P =
L_P L_P^T$, by symmetric eigendecomposition after Jacobi scaling, which turns the cost into the
squared norm of stacked residual rows. The box-constrained problem is solved by a primal
active-set method with Cholesky solves on the condensed Hessian, and the solution is accepted
when its KKT conditions hold to $10^{-9}$ of the gradient's terms. $V_N$ is the squared norm of
the residual at the solution.

**Horizon search.** The horizons are tried in order and the first whose terminal state passes
the check is reported with its value. If none passes, `valid` is False, `value` is NaN and
`horizon` is None; the lower bound is reported in every case.

**From a reward.** `LQOracle.from_cost(cost, dt=..., a_max=...)` takes $Q$, $R$ and $S$ from the
blocks of `QuadraticCost.matrix(dt, None)`, the cost rate integrated exactly over a step with the
acceleration held (Van Loan, 1978), and the cost's reference as the hold point.

### Parameters and defaults

| Parameter | Default | Reason |
|---|---|---|
| `horizons` | (5, 10, 20, 40, 80, 120) | Short horizons are cheap and suffice near the set; 120 steps reach starts whose first LQR command is many times the limit |
| `hold` | (0, 0, 0) | The target's position; any hold point at rest that needs less than `a_max` of steady thrust on each axis |
| `s` | zero | The cross weight of a sampled-data cost; `from_cost` sets it |
| `layout` | `"box"` | The only layout supported |
| `delay` | 0 | Actuation delay (s); a whole number of steps |

### Assumptions and limits

- Exact for CW motion only. `LQBound.rung` is `"cw"`; on another dynamics rung both numbers are
  model values, not bounds.
- Inputs held from each step start to the next. A delay that is not a whole number of steps
  splits each held input across two steps and is refused. With a whole-step delay the inputs as
  applied are still step-held, so both numbers hold for the applied history; a delayed
  controller cannot choose the first inputs, so it may not reach the value.
- Box thrusters only; `layout="ball"` is refused. No state constraints: keep-out zones, corridors
  and the capture tolerances are not part of the value.
- Undiscounted costs only, summed over an infinite horizon from the start. A cost on
  $\mathbf a$ rather than $\mathbf a - \mathbf u_s$ has no finite sum at a hold that needs steady
  thrust, so `from_cost` accepts only a reference at rest that needs none (on the V-bar) and
  refuses a moving reference.
- With a hold off the V-bar the admissible-set check uses the smaller symmetric box, so a start
  can report `valid` False at a horizon where the exact set would accept it, never the reverse.
- A hold point that needs `a_max` or more of steady thrust on an axis is refused.

### Validation

Tests: `tests/test_lq_oracle.py`. Unless stated, every case is on a 500 km circular orbit of the
WGS 84 Earth with $\Delta t$ = 10 s, a 0.02 m/s^2 box,
$Q = \mathrm{diag}(1/10^2 \times 3,\ 1/0.01^2 \times 3)/2$ and $R = I/0.02^2/2$. The reference QP
in the tests is solved independently, by bounded-variable least squares on a Cholesky factor of
the condensed Hessian built from matrix powers.

- Rawlings, Mayne and Diehl, section 2.5.4 and proposition 2.18, on a case of ours: from rest at
  three times the along-track offset at which the LQR first saturates ($3 \times 201.5$ m, the
  offset taken from the Gilbert and Tan rows), the value is 1.855711116e5 and the bound
  1.845593979e5 at every horizon of 5, 10, 20, 40, 80 and 120 steps, to half a unit in the last
  digit. The same start moved by a V-bar hold point gives the same numbers.
- From that start one step ends with the terminal state about twice outside the set: `valid` is
  False, `value` NaN, `horizon` None and the bound unchanged. With horizons (1, 2, 3) the reported
  horizon is 2.
- On eight starts at 1.5 to 8 times the saturation offset with horizons of 1 to 5 steps, each at
  least 2% from the set boundary, `valid` is True exactly when the reference QP's terminal state
  lies in the set, and the value matches the reference to $10^{-9}$ relative.
- From 10 random starts (seed 20261022, uniform within 600, 600 and 200 m and 0.3 m/s per axis),
  each saturating the first LQR command, the bound is below the value, the value matches the
  10-step reference to $10^{-9}$, and the LQR clipped to the box costs at least the value; it is
  strictly dearer from at least two of them.
- From two far starts the 10-step terminal state is outside the set and the 20-step one inside;
  the oracle reports horizon 20 with the reference's value. From 100 km along-track no horizon of
  5, 10 or 20 steps passes, and the bound stays below the 20-step value.
- Inside the set (0.5 and 0.99 of the 201.5 m offset and a mixed start) the value equals the
  bound to $10^{-9}$; at the hold point both are zero.
- A hold 100 m below the target shifts the box by the holding input $3n^2 \cdot 100$ m: from 300 m
  above and below it the values are 45508.69042384245 and 45516.80517143117 (a 40-step QP with
  the shifted box), against 45512.67924967781 for both with an unshifted box.
- `from_cost` with the sampled weights of `QuadraticCost` and a hold 100 m behind the target: the
  LQR flown 2000 steps from a start inside the set costs the value and the bound to $10^{-9}$
  under `QuadraticCost`, and the bound would move by more than $10^{-3}$ without the cross weight.
  From three times its saturation offset the value is 1152387.583084355, below the clipped LQR's
  1152388.003498732.
- Refused: a `from_cost` reference that needs steady thrust, a moving reference, the ball layout,
  delays of 2.5, 5 and 15 s at $\Delta t$ = 10 s, a hold point beyond the thrust limit, starts of
  the wrong length or with NaN, and thrust limits of 0, -0.02 and infinity. Zero delay gives the
  default result, and every result carries rung `"cw"`.

### Example

The case above, from three times the saturation offset, with a V-bar hold 100 m behind the
target:

```python
import numpy as np

from vbar.dynamics import EARTH, mean_motion
from vbar.oracles import LQOracle

n = mean_motion(EARTH.radius + 500e3)
a_max = 0.02
q = np.diag([1 / 10.0**2] * 3 + [1 / 0.01**2] * 3) / 2
r = np.eye(3) / a_max**2 / 2
hold = (0.0, -100.0, 0.0)
x0 = np.array([0.0, -100.0 + 604.528705795314, 0.0, 0.0, 0.0, 0.0])

bound = LQOracle(n=n, dt=10.0, q=q, r=r, a_max=a_max, hold=hold)(x0)
assert bound.valid and bound.horizon == 5
assert abs(bound.value - 1.855711116e5) < 5e-4
assert abs(bound.lower - 1.845593979e5) < 5e-4

# One step ends outside the admissible set: only the bound is reported.
short = LQOracle(n=n, dt=10.0, q=q, r=r, a_max=a_max, hold=hold, horizons=(1,))(x0)
assert not short.valid and np.isnan(short.value)
assert short.lower == bound.lower
```
