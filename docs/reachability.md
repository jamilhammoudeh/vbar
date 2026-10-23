# Reachability

Code: `vbar.reach` (`Zonotope`, `Polytope`, `LinearLoop`, `CwLoop`, `cw_loop`, `support`,
`ball_distance`, `drift_clearance`, `drift_cover`, `DriftCover`, `loop_clearance`, `Clearance`,
`propagate`, `contains`,
`invariant_step`, `min_time_steps`, `MinTime`, `gravity_remainder_bound`).

With the command held for each step, Clohessy-Wiltshire motion is linear, so every state
reachable from a set of start states under a set of disturbances or commands forms a set that
can be described exactly. This module computes those sets and their support functions on the
exact CW step and uses them for four checks: a certified keep-out clearance for a whole set of
states, an invariance test for closed loops, the fewest steps to a target with a proof that
fewer are impossible, and a bound on the gravity a CW model leaves out. The methods are those of
Kochenderfer, Katz, Corso and Moss, *Algorithms for Validation*, chapters 8 and 9.

## Sets

- `Zonotope(center, generators)` is $\{c + G\beta : |\beta_j| \le 1\}$, with the generators as
  the columns of $G$. `Zonotope.box(center, half_widths)` builds an axis-aligned box and
  `Zonotope.point(x)` a single state. A linear map and a Minkowski sum of zonotopes are
  zonotopes: `z.map(M, b)` is $\{Mx + b\}$ and `z1 + z2` concatenates the generators (sec.
  8.3.2), so the number of generators grows by the disturbance's at every step, but never
  exponentially as a polytope's vertex count does.
- The support function is closed form (sec. 8.4.1):
  $h(d) = \max_{x} d \cdot x = d \cdot c + \sum_j |d \cdot g_j|$, attained at
  $c + G\,\mathrm{sign}(G^T d)$ (`support_point`).
- `reduce(order)` returns an outer zonotope with at most `order` times $n$ generators: it keeps
  the generators furthest from axis-aligned, ranked by $|g|_1 - |g|_\infty$, and replaces the
  rest by their interval hull (Girard's reduction, HSCC 2005, as stated in arXiv:2209.09321).
  It is used only where an explicit set must stay small.
- `vertices()` enumerates the $2^m$ corners and keeps the convex hull's vertices, for full-rank
  zonotopes with at most 16 generators.
- `Polytope(H, g)` is $\{x : Hx \le g\}$; its support is a linear program (`scipy.optimize.linprog`,
  HiGHS) and raises `ValueError` for an empty or unbounded set.
- A thrust limit enters as `vbar.solvers.ActuatorSet`: its support is $a_{\max}|d|_1$ for a box
  and $a_{\max}|d|_2$ for a ball.

## Loops

`LinearLoop(A, B, c)` is $x_{k+1} = A x_k + B w_k + c$ with $w_k$ any value in a disturbance
set, independently at each step. `cw_loop(n, dt, gain=None, hold=None)` builds it from the
exact CW matrices $\Phi, \Gamma$ of `cw_discrete(n, dt)`:

- without a gain, free drift with the thrust as disturbance: $A = \Phi$, $B = \Gamma$, $c = 0$;
- with a gain $K$ (3 by 6), the feedback $u = -K(\hat x - x_{\text{hold}})$ on an estimate
  $\hat x = x + e$: $A = \Phi - \Gamma K$, $B = -\Gamma K$ for the navigation error $e$, and
  $c = \Gamma K x_{\text{hold}}$.

The closed-loop matrix is formed before any set operation. Propagating the open loop and the
feedback as separate sets would treat the state in $\Phi x$ and in $-\Gamma K x$ as two
independent points of the set and inflate it (the dependency effect, sec. 8.2, example 8.2). The
`CwLoop` also gives the state inside a step, $\Phi(s) - \Gamma(s) K$, for clearance between
decisions.

## Support functions

`support(loop, x0, w, d, steps)` returns $\max d \cdot x_N$ over every start in `x0` and every
disturbance sequence in `w`:

$$
h_{R_N}(d) = h_{X_0}\!\left((A^T)^N d\right)
  + \sum_{j=0}^{N-1} \left[ h_W\!\left(B^T (A^T)^j d\right) + (A^T)^j d \cdot c \right],
$$

the support of the Minkowski sum of eq. 8.20. The direction is propagated backward, so no set is
formed and the value is exact at any horizon. When `x0` or `w` is a `Polytope` the sum becomes one
linear program over the start and every disturbance (eq. 8.27); for zonotopes and thrust limits
both forms agree, which the tests use as a cross-check.

`steps` counts steps after the start: `steps=0` is the start set. The book numbers the start set
$R_1$, so its $R_{1:20}$ is `steps` 0 to 19 here.

## Keep-out clearance of a set

`ball_distance(z, radius)` bounds the keep-out margin $\min_{p \in Z} |p| - R$ of a zonotope of
positions from both sides. A point $r$ of the set near the origin, from bounded least squares on
$\min_\beta |c + G\beta|$, gives the upper bound $|r| - R$. Its direction
$\lambda = r/|r|$ is a separating hyperplane: every point of the set has
$|p| \ge \lambda \cdot p \ge \lambda \cdot c - \sum_j |\lambda \cdot g_j|$, which gives the lower
bound. Both hold whatever the solver's accuracy; at the solver's optimum they meet. The search
runs on all samples at once by an accelerated projected gradient (300 iterations), and samples
whose bounds still differ by more than $10^{-6}$ of the distance are solved again by bounded
least squares (`scipy.optimize.lsq_linear`, BVLS).

`drift_clearance(n, x0, radius, duration, samples)` applies this to the free drift of every state
in `x0`, the convex avoid set of eq. 8.29, at `samples` evenly spaced times. It returns a
`Clearance`:

- `found`: the lowest upper bound over the samples, the margin of a state of the set (m), at time
  `t` (s).
- `certified`: a lower bound over all times. On each sample interval of length $h$,
  $\frac12 (l_i + l_{i+1} - v h)$, with $l_i$ the certified sample margins and $v$ a bound on the
  speed of every trajectory of the set over the interval, since the nearest distance cannot change
  faster than the fastest trajectory. $v$ comes from `interval_bounds` applied to the largest
  range and speed over each sample set, $|c| + \sum_j |g_j|$, so the speed added within the
  interval by the CW accelerations is included. Intervals too long for that bound give `-inf`.
- `clear`: whether `certified` is at least zero. A result says "clear"; the printed alg. 8.4
  returns the opposite flag (true when the set meets the avoid set).

`loop_clearance(loop, x0, w, radius, steps, samples_per_step, actuator=None, order=None)` does the
same for a `CwLoop` over `steps` held steps, sampling each step from its start to its end. With a
gain the command is checked on every step's set: if $|K(\hat x - x_{\text{hold}})|$ can exceed the
thrust limit anywhere on it, it raises `ValueError` ("the loop saturates on the set"), since the
spacecraft would then clip the command and the linear loop, and its sets, would be wrong. The
command bound also enters the speed bound. `order` reduces each step's set before the next.

## Cover of a box of start states

`drift_cover(n, lo, hi, radius, duration, samples, contains, max_boxes, depth)` checks the free
drift of every state of the box $[lo, hi]$ against a keep-out sphere, and looks for states that a
declared safe set accepts although their drift enters. `contains` is any function of a state that
says whether the declared set holds it, such as the `contains` of a free-drift backup filter; it
is called on box centers only.

Boxes are checked largest first (fewest halvings). With $c_k + G_k\beta$ the positions of a box's
drift at sample $k$ and $R$ the radius, a box is

1. certified if the certified margin of `drift_clearance` over its drift is at least zero, so
   every state of it drifts clear at every time, between samples included;
2. excluded if $|c_k| + \sum_j |g_{kj}| \le R - 1$ mm at some sample. By the triangle inequality
   every drift from it then enters by more than the keep-out's 1 mm resolution. If `contains`
   accepts the center, the center is a counterexample;
3. undecided, with its center as a counterexample, if `contains` accepts the center and the
   center's lowest sampled margin is below $-1$ mm. That margin is exact for that state at that
   time. The box is not split further;
4. otherwise halved along the axis whose generator moves the position most,
   $|\Phi_r(t^*) g_j|$ largest, at the sample $t^*$ where $|c_k| - \sum_j |g_{kj}|$ is lowest,
   the sample at which the box can come closest to the keep-out.

A center whose sampled margin is already negative rules out step 1, so the nearest-point search
of `ball_distance` runs only for the other boxes.

The cover stops splitting an axis once it is halved `depth` times; a box whose axes are all at
that depth is undecided. After `max_boxes` checks the boxes still queued are undecided.

It returns a `DriftCover`:

- `certified`, `excluded`, `undecided`: fractions of the box's volume, counted over the axes on
  which the box has width, so a box halved $s$ times holds $2^{-s}$ and the three add up to 1.
  A box that stopped at a counterexample counts as undecided;
- `counterexamples`: `(x, margin, t)` for each counterexample, the state (m, m/s), its lowest
  sampled margin (m) and that sample's time (s);
- `certified_boxes`, `undecided_boxes`: the corners of those boxes, arrays of shape (k, 2, 6);
- `boxes`: the number of boxes checked.

Limits:

- Only free drift is covered. A state the declared set accepts is checked for its own drift;
  whether the drift from it stays in the declared set, and anything a filter does with other
  commands, is not.
- Exclusion is proven for every state of the box, but whether the declared set holds any of them
  is asked only at the center.
- Each check of a box that is neither excluded nor entering at its center runs the nearest-point
  search on every sample, so its cost grows with `samples`. Coverage in six dimensions per box
  checked is not bounded in advance.

```python
import math

import numpy as np

from vbar.dynamics import mean_motion
from vbar.reach import drift_cover
from vbar.safety.barriers import PassiveBarrier

n = mean_motion(6378137.0 + 500e3)  # rad/s, 500 km circular
hold = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])  # V-bar hold point, m and m/s
half = np.array([0.0, 0.0, 0.0] + [1.4140e-3] * 3)  # velocity box, m/s
cover = drift_cover(n, hold - half, hold + half, 50.0, 4.0 * math.pi / n, 4001,
                    lambda x: True, 8, 4)
print(cover.boxes, cover.certified)  # 1 1.0

# A declared set built for a 45 m keep-out accepts a drift that passes the 50 m one at 47 m.
x = np.array([47.0, 100.0, 0.0, 0.0, -1.5 * n * 47.0, 0.0])
tiny = np.array([1e-3] * 3 + [1e-6] * 3)
declared = PassiveBarrier(radius=45.0, n=n).contains
cover = drift_cover(n, x - tiny, x + tiny, 50.0, 3610.0, 3611, declared, 16, 4)
state, margin, t = cover.counterexamples[0]
print(round(margin, 3), t)  # -3.0 1282.0
```

`benchmarks/filter_audit.py --parts cover` runs the cover on the free-drift backup filter's
declared set over the audited approach-vbar run; see the Audit section of
[safety-filters.md](safety-filters.md).

## Propagation, containment and invariance

- `propagate(loop, x0, w, steps, order=None)` returns the explicit sets $R_0$ to $R_N$, each the
  image of the previous output, $A R_k \oplus B W \oplus c$. The printed alg. 9.4 re-reads its own
  input each step instead; the outputs must be fed back. Without `order` the sets are exact; with
  it they are outer sets.
- `contains(outer, inner)` is exact. A full-dimensional zonotope is the intersection of its facet
  halfspaces, whose normals are orthogonal to every $n - 1$ generators of full rank. `inner` lies
  inside iff its support along each normal and its negative is at most the outer one, compared to
  $10^{-12}$ of the outer extent along the normal. It raises `ValueError` for an outer zonotope that
  is not full-dimensional or has more than 4096 facets ($2\binom{m}{n-1}$; at most 14 generators in
  six dimensions).
- `invariant_step(loop, x0, w, max_steps, order=2.0)` returns the first $d$ whose image
  $A R_{d-1} \oplus B W \oplus c$ lies inside $R_{d-1}$, or None. Here $R_{d-1}$ is the carried
  set, the previous image reduced to `order` times $n$ generators, an outer set of the exact one.
  When it contains its own image every later state stays in it (sec. 8.2), so the loop is safe for
  all time only if that carried set is clear of the avoid set; clearing the exact sets up to $d$ is
  not enough. The image itself is tested exactly. Free drift never
  passes (its sets drift apart), so the test is for closed loops such as a hold-point regulator.

## Minimum time

`min_time_steps(phi, gamma, x0, xf, a_max, n_max)` finds the fewest held steps $N$ from `x0` to
`xf` under a per-axis thrust box. Each horizon is a feasibility linear program on the stacked
commands, $\sum_k \Phi^{N-1-k}\Gamma u_k = x_f - \Phi^N x_0$, $|u_{k,i}| \le a_{\max}$, tried from
$N = 1$. The result `MinTime` gives:

- `steps`, the first feasible $N$, its `plan` (N by 3, m/s^2) and the plan's `residual` on the
  exact steps;
- `nu`, a direction proving $N - 1$ steps infeasible. The target lies outside the reachable set
  $R_{N-1}$ iff some $\nu$ has $\nu \cdot (x_f - \Phi^{N-1} x_0) > h_{R_{N-1}}(\nu)$ (eq. 8.24 and
  8.27). $\nu$ is found by a second linear program over $[-1, 1]^6$;
- `gap`, that margin recomputed in closed form from $\nu$. The certificate is kept only when the
  gap exceeds $10^{-9}$ of its terms' size; otherwise `nu` is None. $\nu$ weights metres and m/s
  together, so only the sign of `gap` is meaningful.

A ball limit raises `ValueError`: its certificate needs a cone program.

## Gravity left out of the CW model

`gravity_remainder_bound(mu, r, rho)` bounds point-mass gravity minus its linearization at a
separation $\rho$ from a reference point at radius $r$:

$$
|g(r + \rho) - g(r) - G\rho| \le \frac{3\mu\rho^2}{(r - \rho)^4}, \qquad 0 \le \rho < r.
$$

The second derivative of $-\mu s/|s|^3$ along any unit direction is at most $6\mu/|s|^4$, its
value along the radius, and $|s| \ge r - \rho$ on the segment; Taylor's theorem with remainder
(sec. 9.3) gives half of it times $\rho^2$. The form with $r^4$ in the denominator is the leading
term only and is exceeded inward of the reference point.

## Assumptions and limits

- Exact only for the CW model with the command held over each step. On two-body truth the
  remainder above is the model error per unit mass, which these sets do not include.
- Disturbances act independently at each step, anywhere in their set. Gaussian errors have no
  bounded set: truncate them, and the result then certifies only the truncated model.
- Closed loops must stay linear on the set: `loop_clearance` refuses a loop whose command
  saturates. Filters with projections, such as `CbfFilter`, are piecewise affine and are not
  covered.
- Invariance with order reduction is sufficient, not necessary: reduction adds volume, and a
  slowly contracting loop may never pass, although its exact sets are invariant.
- Clearance is certified between samples by the speed bound above; with long sample intervals
  the bound is weak or `-inf`.

```python
import math

import numpy as np

from vbar.dynamics import cw_discrete, mean_motion
from vbar.reach import Zonotope, cw_loop, drift_clearance, min_time_steps, support
from vbar.solvers import ActuatorSet

n = mean_motion(6378137.0 + 500e3)  # rad/s, 500 km circular
start = Zonotope.box([0.0, -1000.0, 0.0, 0.0, 0.0, 0.0], [10.0] * 3 + [0.01] * 3)  # m, m/s
thrust = ActuatorSet(1e-3, "box")  # m/s^2 per axis
print(round(support(cw_loop(n, 60.0), start, thrust, np.eye(6)[0], 10), 6))  # 277.215354

hold = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])  # V-bar hold point, at rest
nav = Zonotope.box(hold, [0.6] * 3 + [3e-4] * 3)  # 3-sigma box, m and m/s
two_orbits = 4.0 * math.pi / n
found = drift_clearance(n, nav, 50.0, two_orbits, 4001)
print(round(found.found, 3), found.certified <= found.found, found.clear)  # -6.157 True False

phi, gamma = cw_discrete(n, 60.0)
best = min_time_steps(phi, gamma, start.center, hold, 1e-3, 40)
print(best.steps, best.gap > 0.0, best.residual < 1e-6)  # 28 True True
```

## Validation

The tests in `tests/test_reach.py` check:

- Exercises 8.3, 8.6 and 8.8: the images $(-2, 2)$, $(3, -3)$, $(8, -8)$; the zonotope with center
  $(1, 2)$ and generators $(0, 1)$, $(1, 1)$ has vertices $(0, 0)$, $(0, 2)$, $(2, 2)$,
  $(2, 4)$; the support point along $(-4, 1)$ is $(1, 1)$. Exact.
- Examples 8.3 to 8.5 (mass-spring-damper of app. A.4, our values): the largest $|p|$ over the
  start set and 19 steps is 0.2, 0.2 and 0.3504 for observation-noise half-widths 0.1, 1.0, 2.5
  (0.3947 over 39 steps), to 1e-4.
- Eq. 8.20 against eq. 8.27 on CW (our case: 500 km, 60 s steps, 10 steps, box 1e-3 m/s^2,
  start box $\pm$(10 m, 0.01 m/s) about (0, -1000, 0) m): closed form and linear program agree to
  1e-6 m in six directions, 277.215354 m along +x.
- Eq. 8.29 on CW free drift from the V-bar hold at 100 m, keep-out 50 m, two orbits (our cases):
  margin 50.000 m for the point; for a box of (3 m, 3 mm/s) found $-50.000$ m with a certified
  bound below it and above $-50.35$ m; (0.6 m, 0.3 mm/s) gives $-6.157$ m; a velocity box of
  1.4140 mm/s is certified clear and 1.4160 mm/s has a found violation (the closed form
  $50n/\max_t[2(1 - \cos t) + 3t - 4\sin t]$ gives 1.4147 mm/s).
- `ball_distance` brackets the true distance on 100 drawn zonotopes.
- `drift_clearance` from the stacked sample sets equals the same computation with every sample set
  built as its own zonotope, to 1e-12 of the 50 m scale. `tests/test_reach_cover.py` runs the
  point, the (0.6 m, 0.3 mm/s) box and the 1.4160 mm/s box; `benchmarks/drift_cover_checks.py`
  runs all five cases above, 2001 and 4001 samples included, and also compares them with stored
  values of the per-sample computation.
- `drift_cover` (`tests/test_reach_cover.py`; our cases, 500 km, keep-out 50 m): the 1.4140 mm/s
  velocity box at the hold is certified as one box over two orbits with 4001 samples, and the
  1.4160 mm/s box is split. With $x_0 = \delta$ and $\dot y_0 = -1.5 n \delta$ the drift keeps
  $x = \delta$ and $y = y_0 - 1.5 n \delta t$. For $\delta$ = 5 m entering at 3602 s, the passive
  barrier with its drift horizon capped at 3600 s declares the state safe, and the cover of a box
  of $\pm$(1 mm, 1e-6 m/s) about it over 3610 s returns a counterexample after 3602 s. For
  $\delta$ = 47 m a declared keep-out of 45 m gives a counterexample with margin between $-3.0$
  and $-2.9$ m, and the declared 50 m keep-out excludes the box and gives none. A drift reaching
  1.5 mm inside the keep-out is excluded and one reaching 0.5 mm inside is not, the 1 mm
  resolution. A box 0.1 m wide radially and 2 mm/s along-track is halved along-track, the axis
  that moves its closest sample most.
- Covers over two orbits of seeded boxes of in-plane closing velocities at the hold that straddle
  the 1.4147 mm/s passive limit, declared safe everywhere: every drawn state of a certified box
  drifts clear at 1 s steps, and every counterexample flown again enters by more than 1 mm at its
  reported time. The tests fly 16 states of one certified box; `benchmarks/drift_cover_checks.py`
  flies 64 states of each of 50.
- Eq. 8.24 and 8.27 (our case): from (0, -1000, 0) to (0, -100, 0) at rest under a 1e-3 m/s^2 box
  with 60 s steps the first feasible horizon is 28 steps, the plan's residual is below 1e-6, and
  the direction found for 27 steps has a positive gap recomputed in closed form.
- Sec. 8.2: a hold-point regulator is invariant, free drift is not within 50 steps, and an outer
  zonotope scaled by $1 - 10^{-6}$ no longer contains the set.
- Sec. 9.3 (our case, $r$ = 6878.137 km): the largest remainder over directions divided by
  $\rho^2$ tends to $3.0000\,\mu/r^4$, and $3\mu\rho^2/(r - \rho)^4$ holds over 181 by 121
  directions at 100 m, 1 km and 10 km.
