# Passing-side oracle

Code: `PassingSideOracle`, `PassingSolution` and `PassingSide` in `vbar.oracles.passing`. Hill
frame, metres, seconds.

When the keep-out sphere lies across the transfer, the cheapest plan of the
[fuel oracle](oracles.md) passes through it and the fuel oracle reports no upper bound.
`PassingSideOracle` finds plans that go around the sphere, on a set of sides, and reports the
cheapest one that passes verification as the upper bound. The lower bound stays the fuel
oracle's, which ignores the sphere and so holds for every side.

## How it is built

**Sides.** Outside a ball in three dimensions any two paths with the same ends can be deformed
into each other, so there are no homotopy classes to enumerate; the choice is which side to pass
on. With $\mathbf r_0$ the start, $\mathbf r_g$ the hold point,
$\mathbf d = (\mathbf r_g - \mathbf r_0)/|\mathbf r_g - \mathbf r_0|$ the line of approach and
$\hat{\mathbf z}$ the orbit normal,

$$
\mathbf w = \frac{\hat{\mathbf z} - (\hat{\mathbf z}\cdot\mathbf d)\,\mathbf d}
{|\hat{\mathbf z} - (\hat{\mathbf z}\cdot\mathbf d)\,\mathbf d|}, \qquad
\mathbf u = \mathbf w \times \mathbf d, \qquad
\mathbf e_j = \cos\theta_j\,\mathbf u + \sin\theta_j\,\mathbf w, \quad
\theta_j = \frac{2\pi j}{N},\ j = 0, \dots, N - 1,
$$

with $N$ = `sides`. For an approach in the orbit plane $\mathbf w = \hat{\mathbf z}$, so 4 or 8
sides include $\pm\hat{\mathbf z}$. A line along the orbit normal takes the radial axis in place
of $\hat{\mathbf z}$, and a start on the hold point takes V-bar as its line.

**Halfspaces.** The keep-out constraint $|\mathbf r_j| \ge R$ at step boundary $j$ is not
convex. It is replaced by the halfspace

$$
\mathbf q_j^T \mathbf r_j \ge R + m, \qquad |\mathbf q_j| = 1,
$$

with $m$ the `margin`. By Cauchy-Schwarz $|\mathbf r| \ge \mathbf q^T\mathbf r$, so every point of
the halfspace lies outside the sphere of radius $R + m$. With
$\mathbf q_j = \bar{\mathbf r}_j / |\bar{\mathbf r}_j|$ it is the first-order condition of the
convex function $|\mathbf r|$ at $\bar{\mathbf r}_j$,
$|\mathbf r| \ge |\bar{\mathbf r}| + \mathbf q^T(\mathbf r - \bar{\mathbf r}) = \mathbf q^T\mathbf r$
(Boyd and Vandenberghe, *Convex Optimization*, 2004, eq. 3.2).

**Iterations.** Each side starts from the keep-out-free plan of `FuelOracle` and its arrival step
$k$. Its positions inside $1.5R$ are moved along $\mathbf e$ to $1.5R$,

$$
\tilde{\mathbf r}_j = \mathbf r_j + t\,\mathbf e, \qquad
t = -\mathbf r_j^T\mathbf e + \sqrt{(\mathbf r_j^T\mathbf e)^2 - |\mathbf r_j|^2 + (1.5R)^2},
$$

and $\mathbf q_j = \tilde{\mathbf r}_j / |\tilde{\mathbf r}_j|$ gives the first LP. Each later LP
takes $\mathbf q_j$ from the previous iterate's positions. The previous iterate
$\bar{\mathbf r}$ met $\bar{\mathbf q}_j^T\bar{\mathbf r}_j \ge R + m$, so the new row holds at it:
$\mathbf q_j^T\bar{\mathbf r}_j = |\bar{\mathbf r}_j| \ge \bar{\mathbf q}_j^T\bar{\mathbf r}_j \ge
R + m$. The previous iterate is feasible for the next LP, so the LP value never rises. A side
runs `iterations` LPs, or stops at the first infeasible one.

**Side LP.** It is the inner LP of `FuelOracle` at arrival step $k$ with the halfspaces added, in
state-space form so that each halfspace touches only the three position variables of its
boundary. With $K = k + h$, $h = \lceil$`hold_time`$/\Delta t\rceil$ the steps of the dwell,
$\Phi$ and $\Gamma$ the CW zero-order-hold matrices of one step from `cw_discrete` and $a$ =
`max_accel`, box thrusters solve

$$
\begin{aligned}
\min\ & \Delta t\,a \sum_{j=1}^{K} |\mathbf w_j|_1 \\
\text{s.t.}\ & \mathbf x_j = \Phi\,\mathbf x_{j-1} + a\,\Gamma\,\mathbf w_j,\quad j = 1, \dots, K, \\
& |w_{j,i}| \le 1, \\
& |x_{j,i} - x_{g,i}| \le \text{aim}\,\rho_i/\sqrt3,\quad j = k, \dots, K, \\
& \mathbf q_j^T\mathbf r_j \ge R + m,\quad j = 1, \dots, K,
\end{aligned}
$$

with $\mathbf w_j = 0$ on steps that overlap a `thrust_inhibit` window, $\rho_i$ the capture radius
for positions and the capture speed for velocities, and the state scaled as in the fuel oracle
(positions by its length $L$, velocities by $Ln$). For a ball,
$\mathbf w_j = \sum_d \lambda_{jd}\,\mathbf d$ with $\lambda \ge 0$ and
$\sum_d \lambda_{jd} \le 1$ over the 6 axes, the 8 diagonals and `inner_directions` Fibonacci
directions, at cost $\Delta t\,a\sum\lambda$; then $|\mathbf w_j|_2 \le 1$, so the plan is within
the thrust ball and costs at most the LP value. The constraint matrix, bounds and cost are built
once per solve; only the halfspace rows change between sides and iterations.

**Verification.** The LP enforces the halfspaces at step boundaries only, so a plan that meets
them can still cut into the sphere between boundaries. Each iterate's inputs are clipped exactly
into the thrust limit, propagated step by step and checked as `FuelOracle` checks its plans:
within $(1 + \text{aim})/2$ of the capture radius and speed from the arrival boundary to the end
of the dwell, between samples included, and the certified lower bound of `segment_minima` on the
keep-out margin nonnegative on every step, sampled 20 times per step. An iterate that fails is
listed with `verified` False and never counts.

**Bounds.** Every success that stays outside the sphere is also a success of the problem without
it, whichever side it passes on and whether or not that side is among those searched. The
lower bounds of `FuelOracle` at a free final time certify that keep-out-free problem, so they hold
for every passing plan; `finite_lower` and `impulsive_lower` are those values to the bit. The
upper bound is the cost of the cheapest verified iterate over all sides, in the layout's norm,
and `plan` is that iterate.

**Shortcuts.** The sides are searched only when they can help:

- keep-out soft (not in `hard`) or not listed: the fuel oracle's plan is the answer, unchecked
  against the sphere;
- the keep-out-free plan passes verification with the sphere: it is the answer, the same plan
  and arrival step as `FuelOracle.solve` with keep-out hard;
- a hard constraint other than keep-out: `upper` is inf, `plan` None and `unmodelled` names it;
- no keep-out-free plan: `upper` is inf.

In each case `sides` is empty.

## Parameters and defaults

| Parameter | Default | Reason |
|---|---|---|
| `config` | `OracleConfig()` | The fuel oracle's settings; `aim` sets the side LP's target box and `inner_directions` its ball directions |
| `sides` | 8 | Directions every 45 degrees; for an approach in the orbit plane they include the orbit normal both ways |
| `iterations` | 5 | LPs per side; each can only lower the side's LP value, at the price of one LP and one verification |
| `margin` | $0.02R$ | Height of the halfspaces over the keep-out radius (1 m on a 50 m sphere, 1.6 m on an 80 m one), room for the path between boundaries that scales with the sphere; any finite value is accepted; a negative one lets the halfspaces cut into the sphere, and only the plans that still clear it in verification are reported |
| `hard` | every listed constraint | `solve` only; names of the hard constraints |

The first halfspaces point at the keep-out-free positions moved out to $1.5R$, a fixed choice
that starts each side well clear of the sphere on its own side.

## Assumptions and limits

- Exact CW motion with no other forcing, as for the fuel oracle. Every bound carries rung `"cw"`.
- A finite set of sides is searched. The cheapest plan may pass on a side between them, and then
  the upper bound is higher than it would be with more sides. The lower bound does not depend on
  the sides, so the bracket stays valid and only widens.
- Every side keeps the arrival step of the keep-out-free plan; a detour that arrives earlier or
  later and costs less is not searched.
- Iterations converge to a plan that is locally best for its side, given the halfspaces; nothing
  certifies that it is the best plan on that side.
- The margin is fixed. If the path dips more than $m$ between boundaries, the iterate fails
  verification and the side may end with no verified plan.
- For box thrusters the plan's cost equals the LP value to HiGHS's tolerance, so the listed
  costs never rise.
- For a ball the side LP draws each input from the convex hull of the 6 axes, the 8 diagonals
  and `inner_directions` Fibonacci directions, not from the fuel oracle's refined cut set, so a
  side plan can cost a little more than the same path would with more directions. Every plan is
  within the thrust ball and its cost is in the 2-norm. That cost is at most the LP value, which
  never rises, but the listed costs can rise slightly between iterates, and the direction set is
  not symmetric under $z \to -z$, so mirror sides can differ. The fuel oracle's ball LPs with a
  long dwell are slow, and so is this oracle there.
- Degenerate lines of approach: a start on the hold point has no line, and V-bar is taken as
  its line; a line along the orbit normal takes the radial axis in place of $\hat{\mathbf z}$
  when building $\mathbf w$.
- A start from which no plan can stay outside the sphere, or none that the side LPs find,
  reports `upper` inf, `plan` None and `arrival_step` None; a side whose first LP is infeasible
  lists no costs. The lower bound is still the fuel oracle's keep-out-free bound, which can be
  as weak as 0, so the oracle proves nothing about such a start.
- The plan is checked against keep-out and thrust inhibit only; any other hard constraint voids
  the upper bound, as for `FuelOracle`.
- Goals, layouts and starts that `FuelOracle` refuses raise the same `ValueError`; `sides` or
  `iterations` below 1 and a margin that is not finite raise `ValueError`.

## Validation

Tests: `tests/test_passing_side_oracle.py`. Values marked as independent were computed with a
separate state-space LP solved with HiGHS, with the halfspaces at the step boundaries and a 1 m
margin, not with this module.

The scenario is approach-vbar (keep-out radius 50 m, hold point 100 m behind the target on
V-bar, 600 s dwell, 10 s steps, box thrusters at 0.02 m/s^2) shortened to 120 steps, from
$(0, 400, 0)$ m at rest, with keep-out the only hard constraint. On this transfer the target lies
on the line of approach. The fuel oracle runs with 2 arrival candidates, one bank LP per start
and no refinements, and the shared solve uses 4 sides and 3 LPs per side.

- The fuel oracle with keep-out hard finds no plan. Its keep-out-free plan arrives at step 60
  for 2.5257353762 m/s (independent, to $10^{-6}$) and passes within 50 m of the target.
- The passing-side lower bounds equal the fuel oracle's exactly. The upper bound is finite,
  above the keep-out-free cost by more than 0.1%, at most 2.60 m/s (independent, third iterates:
  2.5466298566 m/s on $+x$, 2.5495420949 on $\pm z$, 3.5826066652 on $-x$), within 8% of the
  lower bound, and arrives at step 60. Its plan stays within the thrust limit, costs `upper` to
  $10^{-12}$, holds the capture ball through the dwell at 50 samples per step, and stays at
  least 50 m from the target at 500 samples per step.
- Flown open loop in `HoldPointEnv`, the plan succeeds with a nonnegative keep-out margin and a
  delta-v equal to `upper` to $10^{-9}$; the keep-out-free plan flown the same way fails and
  enters the sphere.
- The four directions are unit vectors normal to the line of approach at quarter turns and
  include $\pm\hat{\mathbf z}$. The $+z$ and $-z$ sides have the same iterate costs to $10^{-6}$,
  as the CW out-of-plane motion is symmetric under $z \to -z$.
- No iterate cost exceeds the one before it by more than $10^{-9}$ relative, and the $+z$ side
  falls by more than 1% from its first iterate (independent: from 2.97 m/s to 2.550 by the third).
  At least three sides verify, each side's `upper` is the least verified iterate cost, and every
  verified side plan holds the capture ball and stays at least 50 m from the target at 1000
  samples per step.
- Mutation: with halfspaces 2 m inside the sphere ($m = -2$), every $+x$ iterate is the
  keep-out-free plan, which meets every row and fails verification, and the last iterate of each
  side fails (independent: the $-x$ iterates come within 49.57, 52.47 and 48.17 m); a verifier
  that trusted the halfspaces would report the keep-out-free cost. With $m = 0$ the third $+x$
  iterate meets the rows at the boundaries but is rejected because the path dips below 50 m
  between them (49.99 m, independent); any verified plan still stays outside the sphere.
- Default margin: on the 50 m sphere every side's first iterate is the same as with an explicit
  1 m margin (with no margin the $+x$ first iterate would cost 2.5467835 m/s instead of
  2.5616174). With the keep-out radius set to 80 m the default gives the same first iterate and
  verdict on the $+x$ side as an explicit 1.6 m, which costs 3.3930340 m/s against 3.3750074 at a
  fixed 1 m.
- From $(150, 200, 0)$ m at rest the keep-out-free plan costs 1.4303029426 m/s (independent) and
  passes more than 70 m from the target; the passing-side oracle returns the fuel oracle's upper
  bound, lower bounds and arrival step. With keep-out soft from the nominal start it returns the
  keep-out-free plan's cost.
- With every listed constraint hard, the closing-speed and passive-speed constraints are not
  modelled: `upper` is inf, `plan` None, `unmodelled` names them, and `bounds` equals
  `solve(...).bounds`.
- Ball thrusters, on the same transfer shortened to 60 steps without a dwell, with 4 cut rounds
  and 16 inner directions: the keep-out-free plan passes within 50 m of the target; with 2 sides
  and 2 iterations the lower bounds equal the fuel oracle's, the bounds are in the 2-norm, and
  the finite upper bound is the least side upper bound. Every verified side plan is within the
  thrust ball to $10^{-12}$, costs its 2-norm delta-v to $10^{-12}$, ends in the capture ball and
  stays at least 50 m from the target at 1000 samples per step. The test's note gives 1.8463 m/s
  for the keep-out-free plan, 45.3 m from the target, and 2.0458 ($+x$) and 3.4393 m/s ($-x$)
  for the sides.
- A start on the hold point moving at 3 m/s toward the target, with 4 sides: the sides lie in
  the plane normal to V-bar and include $\pm\hat{\mathbf z}$. Stopping takes 225 m at
  0.02 m/s^2 and the chaser crosses the V-bar origin within 37 s, by when it can move at most
  13.5 m sideways, so no plan clears the sphere: none is reported, `arrival_step` is None and the
  lower bound is nonnegative. The same start at rest costs 0, with lower bound 0 and no sides.
- A line along the orbit normal, from $(0, -100, 300)$ m to the hold point: the 4 directions
  are unit vectors in the orbit plane and include $\pm\hat{\mathbf x}$. The line passes 100 m
  from the target, so the solve needs no detour and returns finite bounds and a plan at least
  50 m from the target.
- Defaults are 8 sides and 5 iterations; bad settings, malformed starts, a docking goal and a
  briefed manoeuvring target raise `ValueError`.

## Example

The V-bar case above:

```python
import math
from dataclasses import replace

import numpy as np

from vbar.oracles.fuel import FuelOracle, OracleConfig
from vbar.oracles.passing import PassingSideOracle
from vbar.scenarios import make

view = replace(make("vbar:approach-vbar:cw:box:dv:v1").agent_view(), max_steps=120)
start = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])
config = OracleConfig(arrival_candidates=2, bank_stride=100, max_refinements=0)
keep_out = {"keep_out"}

# The cheapest keep-out-free plan crosses the sphere, so the fuel oracle has no upper bound.
assert math.isinf(FuelOracle(config=config).solve(view, start, hard=keep_out).upper)

oracle = PassingSideOracle(config=config, sides=4, iterations=3)
passing = oracle.solve(view, start, hard=keep_out)
assert 0.0 < passing.finite_lower <= passing.upper < 2.60
assert passing.arrival_step == 60
assert passing.upper == min(side.upper for side in passing.sides)
```
