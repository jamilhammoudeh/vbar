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
- A computed case, not a book example, with the LP and its duality from Boyd and Vandenberghe,
  sections 4.3 and 5.1.3: the half-orbit transfer from
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

## Fixed-window fuel oracle

Code: `FuelOracle`, `OracleConfig` and `OracleSolution` in `vbar.oracles.fuel`, exported from
`vbar.oracles`. Hill frame, SI units.

`FuelOracle.solve(view, start, arrival_step=k)` bounds the least delta-v, in the norm of the
thruster layout, that puts the chaser in the capture ball at the end of step $k$. The lower
bounds are certificates from `dual_bound` and hold in continuous time; the upper bound is the
cost of a plan flown by exact propagation. Without `arrival_step` the final time is free; that
case, the dwell of a hold goal, `FuelOracle.bounds` (the `Oracle` protocol of the optimality gap
metric) and `DualBank` are described in
[Free final time and the dual bank](#free-final-time-and-the-dual-bank).

### How it is built

**Step grid.** With inputs $\mathbf u_j$ held over $[j\Delta t, (j+1)\Delta t)$, the state at the
end of step $k$ is

$$
\mathbf x_k = \Phi(k\Delta t)\,\mathbf x_0 + \sum_{j<k} \Phi\bigl((k-1-j)\Delta t\bigr)\,\Gamma\,\mathbf u_j,
$$

with $\Phi$ and $\Gamma$ from `cw_discrete` at each duration, so long windows carry no product
rounding. Positions are scaled by
$L = \max(|\mathbf r_0|, |\mathbf r_g|, \rho_r, |\mathbf v_0|/n, |\mathbf v_g|/n, \rho_v/n)$,
velocities and the cost by $Ln$, and inputs by the thrust limit $a$; scaling the start, the goal,
the capture ball and $a$ together leaves the scaled program unchanged.

**Box LP.** For box thrusters, $|u_{j,i}| \le a$ and delta-v $\Delta t\sum_j |\mathbf u_j|_1$,
the least cost is a linear program in $\mathbf u = \mathbf u^+ - \mathbf u^-$ (Boyd and
Vandenberghe, *Convex Optimization*, 2004, section 4.3). The capture ball enters through its
bounding box, $|x_{k,i} - x_{g,i}| \le \rho$ per component, $\rho_r$ for positions and $\rho_v$
for velocities. This outer LP has no path constraints and no thrust inhibit, so it is a
relaxation: its optimum, `finite_optimum`, is below the cost of every input history held over
steps that reaches the ball within the limit.

**Ball bracket.** For a ball, $|\mathbf u_j|_2 \le a$ and delta-v
$\Delta t\sum_j|\mathbf u_j|_2$, the 2-norm is bracketed by one shared set $D$ of unit
directions. From outside, $s_j \ge \mathbf d^T\mathbf u_j/a$ for every $\mathbf d \in D$ and
$s_j \le 1$, at cost $a\Delta t\sum_j s_j$; since $\mathbf d^T\mathbf u \le |\mathbf u|_2$, this
is again a relaxation. $D$ starts with the 6 axes and 8 diagonals, and after each solve gains
$\mathbf u_j/|\mathbf u_j|$ for every step where $|\mathbf u_j|/a - s_j$ is at least `cut_tol`
times $|\mathbf u_j|/a$ and above HiGHS's feasibility tolerance $10^{-7}$ (J. E. Kelley, *The
cutting-plane method for solving convex programs*, J. SIAM 8, 1960). The loop stops when no
step is short or after `cut_rounds` solves. A cut that binds at no step for three solves in a
row leaves the set, and Kelley's rule adds it back if a step needs it again. Each solve goes
through the LP's dual, which has one row per variable instead of one per cut and step. From
inside, the plan draws each $\mathbf u_j$ from the convex hull of the final $D$,
`inner_directions` Fibonacci directions and the origin:
$\mathbf u_j = a\sum_d \lambda_{jd}\mathbf d$ with $\lambda \ge 0$ and
$\sum_d\lambda_{jd} \le 1$, at cost $a\Delta t\sum_{j,d}\lambda_{jd}$. Then
$|\mathbf u_j|_2 \le a\sum_d\lambda_{jd} \le a$, so every solution lies within the thrust ball
and costs at most the LP value.

**Lower bounds.** The terminal multipliers $\nu$ of the outer LP, unscaled to the Hill state, go
into `dual_bound` with the one constraint time $k\Delta t$ and `cells_per_step` cells per step:
with the acceleration limit `max_accel` for `finite_lower`, without one for `impulsive_lower`.
Dividing $\nu$ by its certified primer peak removes the excess integral and turns the impulsive
value into a finite-thrust bound, so `finite_lower` is the larger of the two; both are at least
0. The certificate does not depend on the solver: a poor $\nu$ gives a weaker bound, not a wrong
one. If the outer LP is infeasible within the limit, $\nu$ comes from the LP without the limit
and `finite_optimum` is inf; if that is infeasible too, both lower bounds are 0. For a ball,
HiGHS can report a solve error instead of an unbounded dual on a window with no plan within the
limit; $\nu$ then also comes from the LP without the limit, `finite_optimum` is NaN, and the plan
is still tried. That LP is a relaxation, its optimum no larger, and its $\nu$ goes through
`dual_bound` like any other, so the bounds stay valid. A solve error without the limit raises
`RuntimeError`.

**Plan and upper bound.** The inner LP aims inside the box of half-width
$\text{aim}\cdot\rho/\sqrt3$, which lies in aim times the ball, and holds the inputs of steps that
overlap a `thrust_inhibit` window at zero. Its inputs, which meet the thrust limit to HiGHS's
tolerance, are clipped exactly into it, propagated step by step with `cw_discrete`, and accepted
when the end state lies within $(1 + \text{aim})/2$ of the capture radius and of the capture
speed and, if keep-out is hard, when the certified lower bound of `segment_minima` on the range
margin is nonnegative on every step, sampled 20 times per step as the environment samples it.
`upper` is the plan's cost $\Delta t\sum_j|\mathbf u_j|$ in the layout's norm. It is inf when no
plan passed, when the capture radius or speed is zero, when the dwell of a hold goal does not
fit after step $k$, or when a hard constraint other than keep-out is listed; that constraint is
then named in `unmodelled`. With `hard=None` every listed constraint
counts as hard. The `hard` argument of `solve` decides which listed constraints the plan is checked
against: marking keep-out soft gives a finite upper bound that was not checked against it.
`bounds` treats every listed constraint as hard.

**Bounds.** `bounds` is a `FuelBounds` with lower `finite_lower`, `upper`, norm `"l1"` (box) or
`"l2"` (ball), rung `"cw"`, `accel_limit` equal to `max_accel` and `impulsive_lower`, so the
optimality gap metric falls back to the impulsive bound for a run that exceeded the limit.

### Parameters and defaults

| Parameter | Default | Reason |
|---|---|---|
| `cells_per_step` | 8 | The default of `dual_bound`; more cells tighten the certified bound at linear cost (the tests use 500) |
| `aim` | 0.9 | The plan aims inside 90% of the capture ball and must land inside 95%, which absorbs HiGHS's tolerance and the clipping into the limit and keeps the rest as margin |
| `cut_rounds` | 20 | The bracket on the half-orbit ball transfer below closes within 0.5% |
| `cut_tol` | $10^{-4}$ | Relative gap between a step's cutting-plane value and its 2-norm below which no cut is added |
| `inner_directions` | 64 | Fibonacci directions that let the plan thrust where the cuts have not reached |
| `arrival_step` | None | `solve` only; 1 to `max_steps`, a free final time when None |
| `hard` | every listed constraint | `solve` only; names of the hard constraints |

### Assumptions and limits

- Exact CW motion with no other forcing. Every bound carries rung `"cw"` and is no theorem on
  another rung.
- One arrival step. The bounds are for a chaser in the capture ball at the end of step $k$, and
  with a dwell at the boundaries after it that the dwell covers; a run captured at another step
  is not covered by them. The free final time is in the next section.
- The lower bounds hold in continuous time for every acceleration history within `max_accel` in
  the actuator norm (`finite_lower`) or for any acceleration (`impulsive_lower`).
  `finite_optimum` and `upper` are step-grid values, for inputs held over each step, and differ
  from the continuous figures at first order in $n\Delta t$: on the Fehse transfers below with
  720 steps the step-grid optimum lies up to 0.79% (one orbit) and 0.33% (half an orbit) above
  the impulsive optimum, and the impulsive bound from its multipliers up to 0.60% and 0.25% below
  it.
- The lower bounds relax every path constraint and the thrust inhibit windows, which keeps them
  valid and makes them looser. The plan is checked against keep-out and thrust inhibit only; any
  other hard constraint, such as `speed_limit`, `approach_cone` or `zone_entry`, voids the upper
  bound. Goal waypoints are ignored.
- The goal must be a hold or berthing with `hold_point`, `capture_radius` and `capture_speed`.
  Other goals, an unknown thruster layout, a start that is not 6 finite numbers and an arrival
  step outside 1 to `max_steps` raise `ValueError`; a HiGHS failure other than infeasibility
  raises `RuntimeError`. A zero capture radius or speed is allowed and gives an infinite upper
  bound.
- For a ball the outer LP is a relaxation at every round, so the bracket is as wide as the Kelley
  loop leaves it after `cut_rounds` solves.

### Validation

Tests: `tests/test_fuel_oracle.py`. Values marked as independent were computed with
`cw_discrete`, HiGHS and a separate implementation of the LPs and of the cell bound.

- A computed case, not a book example, with the LP and its duality from Boyd and Vandenberghe,
  sections 4.3 and 5.1.3: the half-orbit transfer from
  $(-200, -1500, 50)$ m to $(0, -100, 0)$ m at rest on a 6778 km orbit with a
  $2 \times 10^{-3}$ m/s^2 limit on 40 steps, at 500 cells per step. Box, point target:
  `finite_optimum` is 0.8977056 m/s to $10^{-6}$, and `finite_lower` lies in
  $[0.892140, 0.8947293]$, the upper end being the cost of a 640-step plan; the step-grid
  optimum, the bound without its excess term or with half of it, and the 2-norm dual norm would
  each leave that window (independent). Box with a capture ball of 1 m and $10^{-3}$ m/s: the
  outer optimum 0.8901371 m/s and the plan's cost 0.8937729 m/s match an independent LP to
  $10^{-6}$, and the bound lies in $[0.886695, 0.8904004]$. With 20 m and 0.02 m/s the plan costs
  0.8242298 m/s and the bound lies in $[0.80105, 0.8163351]$, a window the bound without the
  capture ball's support terms would exceed (independent). Ball with 0.01 m and $10^{-5}$ m/s:
  `finite_optimum` is at least 0.8280, the plan costs at most 0.8330 and within 0.5% of it, and
  the certified bound lies in $[0.8265, 0.8324776]$ and within 0.8% of the plan; 0.8324776 m/s
  is the cost of an independent plan checked against the thrust ball on every step.
- Fehse, *Automated Rendezvous and Docking of Spacecraft*, 2003, eqs 3.29 to 3.32 and 3.37:
  500 m along the V-bar at 500 km, one orbit with tangential impulses ($n\,\Delta y/(3\pi)$) and
  half an orbit with radial ones ($n\,\Delta y/2$), with a 1 m/s^2 box limit on 720 steps.
  `finite_optimum` lies between the impulsive optimum and 0.79% (0.33%) above it, the impulsive
  bound between 0.60% (0.25%) below it and the optimum, and `finite_lower` between the impulsive
  bound and `finite_optimum`. A plan into a 0.01 m, $10^{-5}$ m/s ball costs at most 0.79%
  (0.33%) above the impulsive optimum. With a ball layout on 40 steps the impulsive bound is
  positive and at most the optimum.
- Curtis, *Orbital Mechanics for Engineering Students*, example 7.5: 2 km behind the target on a
  300 km orbit, rendezvous in 1.49 h, capture ball 1 m and $10^{-3}$ m/s. The two-impulse
  solution costs 0.245230 m/s in the 2-norm (the book's 0.2452) and 0.263461 m/s in the 1-norm.
  The impulsive bound stays below the two-impulse cost in the layout's norm and below
  `TwoImpulseBound`'s upper bound, and is at least 0.249 (box, 120 steps) and 0.218 (ball, 40
  steps), independent. A plan with burns on the first and last steps costs at least
  `finite_lower`, and for box thrusters at least `upper`, since it is a feasible point of the
  inner LP.
- Alfriend, Vadali, Gurfil, How and Breger, *Spacecraft Formation Flying*, example 5.2, with
  $n = 1$ over $nT = 5$ on 500 steps and a limit far above need: the plan costs less than 85% of
  the best two-impulse 1-norm cost on a 400-point grid (0.0457373) and no less than the
  impulsive optimum 0.0375601 (independent); the impulsive bound lies in $[0.0370, 0.0375601]$.
- A PD run of the approach scenario, saturated per axis at `max_accel`, spends at least
  `finite_lower` and `impulsive_lower` up to the step at which it first enters the capture ball.
- Properties on 10 to 16 random starts within 200 m and 0.2 m/s per axis at 500 km, with
  $\Delta t$ = 20 s, 30 steps and a 0.02 m/s^2 box. Scaling the start, the capture ball and the
  limit by 0.01 and 3.7 scales both lower bounds and the upper bound to $10^{-9}$. The bound for
  $\mathbf x_1 + \mathbf x_2$ stays below the sum of the plans for $\mathbf x_1$ and
  $\mathbf x_2$ with half the limit and half the ball. Reversing the start and mirroring it
  across the orbit plane leave both lower bounds and `finite_optimum` unchanged to $10^{-9}$.
  The bound at the midpoint of two starts stays below the mean of their plans' costs, and
  `finite_optimum` is convex in the start. Every plan, flown step by step, lands in the ball
  within the aim margin, stays within the limit and costs `upper` to $10^{-12}$, for arrival at
  steps 10 and 30.
- A hard `speed_limit` the plan was not checked against makes `upper` inf and is named in
  `unmodelled`; with no hard constraints the plan counts, and the lower bound does not change.
  A goal without a capture ball and arrival steps 0 and 31 of 30 raise `ValueError`.

### Example

The box transfer above with a capture ball of 1 m and $10^{-3}$ m/s:

```python
import math

import numpy as np

from vbar.agents.base import AgentView
from vbar.dynamics import mean_motion
from vbar.oracles import FuelOracle, OracleConfig

n = mean_motion(6778e3)
view = AgentView(
    scenario_id="half-orbit",
    dt=math.pi / n / 40,
    max_steps=40,
    n=n,
    action_mode="accel",
    max_accel=2e-3,
    thrust_limit="box",
    goal={"hold_point": (0.0, -100.0, 0.0), "capture_radius": 1.0, "capture_speed": 1e-3},
)
start = np.array([-200.0, -1500.0, 50.0, 0.0, 0.0, 0.0])

sol = FuelOracle(config=OracleConfig(cells_per_step=500)).solve(view, start, arrival_step=40)
assert 0.886695 <= sol.finite_lower <= sol.finite_optimum <= sol.upper
assert abs(sol.finite_optimum - 0.8901371) < 1e-6
assert abs(sol.upper - 0.8937729) < 1e-6
assert sol.plan.shape == (40, 3) and np.abs(sol.plan).max() <= view.max_accel
assert sol.bounds.rung == "cw" and sol.bounds.accel_limit == view.max_accel
```

## Free final time and the dual bank

Code: `FuelOracle.solve` without `arrival_step`, `FuelOracle.bounds`, `DualBank` and
`bounds_for` in `vbar.oracles.fuel`, exported from `vbar.oracles`. Hill frame, SI units.

A run may reach the goal at any step, and a hold goal with a dwell must then stay in the capture
ball for `hold_time`. `FuelOracle.solve(view, start)` bounds the least delta-v of a success at any
time within the episode, and `FuelOracle.bounds(view, start)` returns those bounds with every
listed constraint hard. `bounds_for(ep, oracle=None)` is the call a harness makes before
`optimality_gap` or `evaluate`: the bounds from the episode's start with its hard constraints.
`DualBank` gives lower bounds for many starts from one set of linear programs.

### How it is built

**Windows.** The capture predicate is strict, so a dwell is an open time interval of length at
least `hold_time` in the open ball, and its closure, of the same length, lies in the closed ball.
It contains at least $m = \lfloor \text{hold\_time}/\Delta t \rfloor$ step boundaries; without a
dwell, $m = 1$. A success therefore puts the chaser in the closed ball at the boundaries
$k, \dots, k + m - 1$ of some window $k$ from 1 to $N - m + 1$, $N$ = `max_steps`. With
$0 < \text{hold\_time} < \Delta t$, $m = 0$: the dwell can fall between two boundaries, and both
lower bounds are 0. A dwell longer than the episode raises `ValueError`.

**Window bound.** For window $k$ and multipliers $\nu_j$ at $t_j = (k + j)\Delta t$,
$j = 0, \dots, m - 1$, `dual_bound` gives

$$
LB_k(\nu) = \sum_j \bigl[\nu_j^T(\mathbf x_g - \Phi(t_j)\mathbf x_0) - \rho_r|\nu_{j,r}|_2
- \rho_v|\nu_{j,v}|_2\bigr] - a\,E_k(\nu),
$$

with $E_k$ the certified excess integral over $[0, t_{m-1}]$, and the impulsive bound
$D_k(\nu)/M_k(\nu)$ with $M_k$ the certified primer peak. The window's own multipliers come from
the outer LP of the previous section with the box rows at all $m$ boundaries. For box thrusters
the rows after the first boundary are written in state form,
$\mathbf e_{i+1} = A\,\mathbf e_i + B\,\mathbf w_{k+i} + \mathbf d$, with $\mathbf e$ the scaled
error from the goal state and $\mathbf d$ the goal's free drift over a step, so a long dwell keeps
the program sparse; the multipliers are the reduced costs of the error variables. The free-final-
time lower bound is $\max(0, \min_k \max_\nu LB_k(\nu))$ over the multipliers at hand, each a
theorem whatever the solver returned.

**Time shift.** The primer of multipliers at $t_j = (k + j)\Delta t$ depends on $t$ only through
$\sigma = t_{m-1} - t$:

$$
\mathbf p(\sigma) = \sum_{j:\ \sigma \ge (m-1-j)\Delta t} B^T\Phi\bigl(\sigma - (m-1-j)\Delta t\bigr)^T\nu_j .
$$

Moving the same multipliers to window $k'$ moves only the upper end of the excess integral and of
the peak, from $\sigma = (k + m - 1)\Delta t$ to $(k' + m - 1)\Delta t$. One profile per set of
multipliers, accumulated step by step over $\sigma$ on the cells of `dual_bound`, therefore gives
every window, and the displacement term is affine in the start,
$C - G_{k'}\mathbf x_0$ with $G_{k'} = \sum_j \Phi(t'_j)^T\nu_j$. The rounding floor is
$64\varepsilon$ times the sum of the absolute values of every product in the displacement term,
at least the floor `dual_bound` takes, and the excess and peak carry the same $1 + 10^{-12}$.

**Bank.** `DualBank.build(view, starts)` solves the LPs of windows $1, 1 + s, 1 + 2s, \dots$ and
the last, $s$ = `bank_stride`, from every start and keeps their multipliers. `window_lowers(start)`
is each window's largest bound over the kept multipliers, which can be negative, inf for windows
that end after the episode; `bounds(start)` is a `FuelBounds` with `lower` the larger of 0 and the
least window bound, `impulsive_lower` the same for the impulsive bounds, `upper` inf, rung
`"cw"` and `accel_limit` = `max_accel`. A start, in the bank or not, costs one matrix product per
kept set of multipliers.

**Free final time in `solve`.** A one-start bank at `bank_stride` comes first; then the LP of the
window whose bound is least is solved, and again, until that window has its own multipliers or
`max_refinements` LPs are added; with 0 the bound is the bank's. The
upper bound tries `arrival_candidates` arrival steps spread from $N - h$ down to 1, and the window
whose lower bound is least, with $h = \lceil \text{hold\_time}/\Delta t \rceil$: the environment
counts a dwell in whole steps from a step boundary, so a plan holds it for $h$ steps after
arriving. Each inner LP aims inside the box of half-width
$\text{aim}\cdot\rho/\sqrt3$ at the arrival boundary and the $h$ boundaries after it. Its plan,
flown exactly, is accepted when it lies within $(1 + \text{aim})/2$ of the capture radius and
speed at the arrival boundary and throughout the $h$ steps after it, and clears keep-out if that
is hard. Between two of the 20 samples per step, $\delta$ apart, the distance from the goal
grows by at most $M\delta^2/8$ over the larger of the two sampled ones, with $M$ the acceleration
bound of `interval_bounds` for the position and $3n^2|\mathbf v| + 2n|\mathbf a|$ for the
velocity, the thrust being constant over a step. `upper` is the cheapest accepted plan,
`arrival_step` its arrival step (None without a plan) and `finite_optimum` NaN: the least outer
LP optimum over the windows would need every window's LP, and the bank solves only some of them.

**Fixed arrival with a dwell.** `solve(view, start, arrival_step=k)` with a dwell bounds membership
at the boundaries $k$ to $\min(k + m - 1, N)$ and plans the dwell from step $k$ when
$k + h \le N$; otherwise `upper` is inf.

**Refusals.** A docking goal (contact is an event at a continuous time, not a hold at step
boundaries) and a briefed target whose `goal["target"]["burns"]` is not empty (a relative
acceleration the CW bound does not model) raise `ValueError` in `solve`, `bounds` and `DualBank`;
a manoeuvring mode with an empty schedule adds no acceleration and is accepted.

### Parameters and defaults

| Parameter | Default | Reason |
|---|---|---|
| `arrival_candidates` | 16 | At most 17 inner LPs for the upper bound; any choice gives a valid upper bound |
| `bank_stride` | 10 | A 360-step episode needs 37 window LPs per start; 1 solves every window |
| `max_refinements` | 8 | Extra window LPs a free-final-time `solve` adds where the bound is least; caps the cost of a ball layout, whose window LPs are Kelley loops; 0 keeps the bank's bound |
| `cells_per_step`, `aim`, `cut_rounds`, `cut_tol`, `inner_directions` | as above | Shared with the fixed window |
| `hard` | every listed constraint | `solve` only; `bounds_for` passes the episode's hard constraints |
| `oracle` | `FuelOracle()` | `bounds_for` only |

### Assumptions and limits

- The fixed window's assumptions hold: exact CW motion with no other forcing, rung `"cw"`, lower
  bounds that relax path constraints and thrust inhibit, plans checked against keep-out and
  thrust inhibit only.
- A window whose LP is infeasible within the thrust limit, or on which HiGHS fails there for a
  ball, takes the multipliers of its LP without the limit, as in the fixed window; its bound is
  finite, never inf, though no plan within the limit may reach it. Proving such a window
  unreachable would hold for one start only, while the bank's multipliers serve every start.
- A window whose LP is infeasible even without the thrust limit has no multipliers of its own; its
  bound comes from multipliers shifted from other windows, which are valid there and may be weak.
- The upper bound is the cheapest plan over the candidates, not over every window. The bracket is
  wide when the dwell is long, since the plans aim inside a box of half-width
  $\text{aim}\cdot\rho/\sqrt3$ through the whole dwell; the tests assert at most 8% on
  approach-rbar and 17% on rbar-acquisition at shortened lengths.
- The view carries the briefing only: a target that burns while the briefing says it does not is
  not refused, and the bound is then a model bound.
- `bounds_for` reads the true start `ep.start` and returns rung `"cw"` whatever the episode's
  dynamics, so `optimality_gap` certifies it only on a CW run.
- A ball layout solves one Kelley loop per window it visits, each of up to `cut_rounds` LPs, so
  a free-final-time ball solve costs many times a fixed-window one; a fixed `arrival_step` or a
  larger `bank_stride` keeps it short.

### Validation

Tests: `tests/test_fuel_oracle_free_time.py`. Values marked as independent were computed with a
separate implementation of the window LPs (HiGHS on the state-space form) and of the cell bound;
an independent every-window bound is the largest bound over the multipliers of every window's own
LP shifted to it.

- approach-rbar shortened to 100 steps, from its nominal start $(-400, 0, 0)$ m at rest, with its
  600 s dwell, keep-out hard, 4 arrival candidates and a stride of 20: the plan keeps the strict
  capture predicate at 50 samples per step through the dwell, stays more than 50 m from the
  target and costs at most 8% above `finite_lower`; `finite_lower` is at most an independent
  verified plan's cost, 2.4505454253 m/s, and `upper` at least the independent every-window
  bound, 2.2669232883 m/s, both to $10^{-9}$. Without the dwell the bound is no higher, and with it
  the bound is more than 0.5 m/s above the independent dwell-free bound 1.2434996264 m/s. A
  halved lower bound would leave a bracket above 50%.
- A one-start bank of the same case with the default stride: its lower bound is at most the
  plan's cost and at least 95% of the solve's, `window_lowers` is finite for windows 1 to 41 and
  inf after, and `lower` is the larger of 0 and their least.
- rbar-acquisition shortened to 200 steps, from $(0, -3000, 0)$ m at rest to 500 m below the
  target with a 300 s dwell: the bracket is at most 17%, the plan holds the dwell and stays more
  than 200 m from the target, `finite_lower` is at most the independent verified plan's
  4.9221014980 m/s and `upper` at least the independent every-window bound 4.1303658977 m/s; the
  dwell lifts the bound more than 0.5 m/s above the independent
  dwell-free 3.1400200881 m/s.
- A bank of 20 starts drawn from the approach-vbar dispersion, 60 steps without the dwell, a
  stride of 20: for every start its lower bound is at most the cost of the start's verified plan,
  keep-out not checked, and at least 95% of the start's own free-time lower bound.
- A start that drifts onto $(-100, 0, 0)$ m at rest at the end of step 17 of 20 s, at 500 km with a
  ball of 0.01 m and $10^{-4}$ m/s, and misses it at every other boundary: with every window's own
  LP, window 17's bound is at most 0 and every other at least 0.0072 m/s (independent 0.0072022);
  with a stride of 10, window 17 is reached only by shifted multipliers and is still at most 0
  while every other window is positive; both banks and `solve` report 0. A bank that skipped
  window 17 would report a positive bound.
- 30 steps of 20 s, a 1 m and 0.01 m/s ball at the origin, random starts within 200 m and
  0.2 m/s per axis: the shifted bounds of a two-start bank stay below the verified fixed-arrival
  plan at every odd window that has one, at least 10 of them, for a start in the bank and one
  outside it, with at least 20 windows positive; for two starts the free-time lower bound is
  positive and below every fixed-arrival plan at steps 3 to 30 by 3, and its plan lands in the
  ball.
- `hold_time` 5 s and 9.999 s at $\Delta t$ = 10 s give 0 from `solve` and from a bank; 10 s and
  0 give positive bounds.
- A briefed `Maneuvering` target raises in `solve` (free and fixed), `bounds` and
  `DualBank.build`; the docking scenario raises; `bounds` equals the free-time solve's bounds;
  `arrival_candidates` and `bank_stride` default to 16 and 10 and refuse 0.
- approach-rbar with a 100 s dwell and keep-out as its only constraint, 50 steps, flown open loop
  by an agent that solves the oracle from its first observation: the run succeeds, E3 is
  certified, not below the bound and between 1 and upper over lower, E3u is 1 to $10^{-6}$, and
  `bounds_for`'s upper bound equals the flown plan's cost to $10^{-12}$. A speed limit listed but
  not hard for the run leaves `bounds_for` unchanged, while `bounds`, with every listed
  constraint hard, has an infinite upper bound.

### Example

approach-rbar shortened to 100 steps from its nominal start, then a recorded run that flies the
oracle's plan:

```python
from dataclasses import replace

import numpy as np

from vbar.metrics import optimality_gap, record_episode
from vbar.oracles import DualBank, FuelOracle, OracleConfig, bounds_for
from vbar.scenarios import HoldPointEnv, get, make

view = replace(make("vbar:approach-rbar:cw:box:dv:v1").agent_view(), max_steps=100)
draw = get("approach-rbar").nominal()
start = np.r_[draw["start_pos"], draw["start_vel"]]

config = OracleConfig(arrival_candidates=4, bank_stride=20)
sol = FuelOracle(config=config).solve(view, start, hard={"keep_out"})
# The 600 s dwell is 60 steps of 10 s after the arrival step.
assert sol.plan.shape == (sol.arrival_step + 60, 3)
assert (sol.upper - sol.finite_lower) / sol.upper <= 0.08

bank = DualBank.build(view, [start], config=config)
lows = bank.window_lowers(start)
assert np.all(np.isinf(lows[41:])) and bank.bounds(start).lower == max(0.0, lows.min())
assert bank.bounds(start).lower <= sol.upper


class OpenLoopOracle:
    name = "open-loop-oracle"

    def reset(self, view, rng=None):
        self.view, self.plan, self.step = view, None, 0

    def act(self, obs, info):
        if self.plan is None:
            self.plan = FuelOracle(config=config).solve(self.view, obs).plan
        u = self.plan[self.step] if self.step < len(self.plan) else np.zeros(3)
        self.step += 1
        return u / self.view.max_accel


spec = get("approach-rbar")
short = {**spec.parameters, "hold_time": 100.0, "closing_limit": None, "passive_axis": None}
env = HoldPointEnv(replace(spec, parameters={**short, "soft": ()}), {"max_steps": 50})
ep = record_episode(env, OpenLoopOracle(), seed=7, agent_seed=8)
gap = optimality_gap(ep, bounds_for(ep, oracle=FuelOracle(config=config)))
assert gap["bound_certified"] and not gap["below_lower_bound"]
assert gap["optimality_gap"] >= 1.0 and abs(gap["optimality_gap_upper"] - 1.0) < 1e-6
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
