# Safety filters

Code: `vbar.safety.projections` (`halfspace`, `box`, `ball`, `second_order_cone`,
`keep_out_projection`, `KeepOutProjection`) and `vbar.safety.tightening` (`BoxError`, `BallError`,
`EllipsoidError`, `halfspace_margin`, `chance_margin`, `keep_out_margin`, `ecbf_bias`,
`first_order_bias`), `vbar.safety.barriers` (`KeepOutEcbf`, `SpeedCbf`, `ClosingSpeedCbf`,
`ConeEcbf`, `PassiveBarrier`, `BrakingBarrier`, `braking_deceleration`) and
`vbar.safety.sampled` (`held_segment_bound`, `line_search`, `LineSearch`).

A safety filter sits between an agent and the spacecraft and changes a command only when it would
lead out of a safe set. This page starts with the building blocks the filters use: closed-form
projections, the exact projection out of a keep-out sphere, and the margins that account for
navigation and execution error. The control barrier function filters themselves follow in later
sections.

## Closed-form projections

Each function returns the Euclidean projection of a point onto a convex set (Boyd and Vandenberghe,
*Convex Optimization*, sec. 8.1.1 and exercise 8.3(c)):

- `halfspace(u, a, b)` onto $\{u : a \cdot u \ge b\}$:
  $u + \max(0, b - a \cdot u)\, a / |a|^2$. A zero normal raises `ValueError`.
- `box(u, a_max)` onto $|u_i| \le a_{\max}$: clipping on every axis.
- `ball(u, a_max)` onto $|u|_2 \le a_{\max}$: radial scaling when $|u| > a_{\max}$.
- `second_order_cone(x, t)` onto $\{(x, t) : |x|_2 \le t\}$, in three regimes:

$$
P(x, t) =
\begin{cases}
(0, 0) & |x| \le -t, \\
(x, t) & |x| \le t, \\
\tfrac12 \left(1 + \tfrac{t}{|x|}\right) (x, |x|) & \text{otherwise.}
\end{cases}
$$

Membership in the cone needs $t \ge 0$ as well as $t^2 \ge |x|^2$; a test of the squares alone
also accepts the negative cone $|x| \le -t$, which the first regime sends to the apex.

## Exact projection out of a keep-out sphere

`keep_out_projection(u_nom, G, h, radius)` solves

$$
\min_u |u - u_{\text{nom}}|^2 \quad \text{subject to} \quad |G u + h|_2 \ge R,
$$

the nearest command whose resulting position $G u + h$ (for example the position after one held
step, linear in the command) lies outside the keep-out sphere of radius $R$. The constraint is
nonconvex, but it is a single quadratic inequality, so the problem has zero duality gap (Boyd and
Vandenberghe, appendix B.1). It returns `KeepOutProjection(u, lam, hard_case)`.

### Method

With the multiplier $\lambda \ge 0$, stationarity gives

$$
u(\lambda) = (I - \lambda G^T G)^{-1} (u_{\text{nom}} + \lambda G^T h),
\qquad 0 \le \lambda < 1/\sigma_{\max}(G)^2,
$$

where the upper limit keeps the Lagrangian convex in $u$. With the singular value decomposition
$G = U S V^T$ and $z = U^T (G u_{\text{nom}} + h)$, the image is $w_i(\lambda) = z_i / (1 -
\lambda s_i^2)$, so $|G u(\lambda) + h|$ increases with $\lambda$ and crosses $R$ once. The search
runs on $\mu = 1 - \lambda s_1^2 \in (0, 1]$, in which the top term is $z_1 / \mu$ without
cancellation, by Brent's method to 4 ulp of $\mu$. The command follows from the image as
$u = u_{\text{nom}} + V\, \mathrm{diag}(1/s_i)(w - z)$, which equals $u(\lambda)$ without a linear
solve. A command already outside is returned unchanged with $\lambda = 0$.

When $z$ has no component along the top left singular vector (below $10^{-12} R$) and the other
directions cannot reach $R$ even at $\lambda = 1/s_1^2$, there is no root: this is the hard case of
the trust-region problem. Then $\lambda = 1/s_1^2$, where $I - \lambda G^T G$ is singular, and the
missing range is added along the top right singular vector, which spans its null space;
`hard_case` is True. The optimum is then not unique, since the opposite sign of that component is
optimal too. Near the hard case the regular branch gives the same command to the size of the top
component.

### Assumptions and limits

- One constraint and no thrust limit. With a thrust bound the problem has two quadratic
  constraints and the zero-gap result no longer holds; the projection is a reference to compare
  linearized barrier filters against, not a filter under the thrust limit.
- $G$ may be rectangular. A command component in the null space of $G$ does not move.
- $G = 0$ with $u_{\text{nom}}$ inside raises `ValueError`, since no command leaves the sphere.

## Error sets and margins

An error set $E$ bounds $e = x - \hat x$, the difference between the true Hill-frame state and the
state the filter acts on (m and m/s, order $x, y, z, v_x, v_y, v_z$). A linear row
$a \cdot x \ge b$ holds for every $x \in \hat x + E$ exactly when
$a \cdot \hat x \ge b + h_E(a)$, with the support function $h_E(a) = \max_{e \in E} a \cdot e$
(Boyd and Vandenberghe, sec. 4.3.1). All three sets are symmetric about zero.

| Set | Definition | `halfspace_margin(a, error)` |
|---|---|---|
| `BoxError(r_pos, r_vel)` | $\lvert e_i\rvert \le r_{\text{pos}}$ on position, $\le r_{\text{vel}}$ on velocity | $r_{\text{pos}} \lvert a_{\text{pos}}\rvert_1 + r_{\text{vel}} \lvert a_{\text{vel}}\rvert_1$ |
| `BallError(r_pos, r_vel)` | $\lvert e_{\text{pos}}\rvert_2 \le r_{\text{pos}}$ and $\lvert e_{\text{vel}}\rvert_2 \le r_{\text{vel}}$ | $r_{\text{pos}} \lvert a_{\text{pos}}\rvert_2 + r_{\text{vel}} \lvert a_{\text{vel}}\rvert_2$ |
| `EllipsoidError(P)` | $\{P z : \lvert z\rvert_2 \le 1\}$, $P$ 6x6 | $\lvert P^T a\rvert_2$ (eq. 4.37) |

`EllipsoidError.from_gaussian(cov, k)` builds the $k$-sigma ellipsoid of a zero-mean Gaussian error
with covariance $S$: $P = k S^{1/2}$, the symmetric square root from an eigendecomposition with
eigenvalues clipped at zero. Its `exceedance`, the probability per draw that the error lies
outside, is $P(\chi^2_6 > k^2)$ (`scipy.stats.chi2.sf(k**2, 6)`), since $e^T S^{-1} e$ is
chi-square with 6 degrees of freedom. A singular $S$ has fewer degrees of freedom and a smaller
tail, so the stated value is then an upper bound. An `EllipsoidError` built from $P$ directly is a
hard bound with `exceedance` 0. The covariance must be symmetric and positive semidefinite to
1e-12 of its largest entry.

`chance_margin(a, cov, eta)` is the margin for a row that must hold with probability $\eta$ under
Gaussian error, $\Phi^{-1}(\eta) |S^{1/2} a|_2$ (Boyd and Vandenberghe, eq. 4.38). The constraint
is convex only for $\eta \ge 1/2$, and $\eta = 1$ has no finite margin, so `eta` outside
$[0.5, 1)$ raises `ValueError`. The probability is per row and per decision; joint satisfaction
over time or over several rows needs a union bound.

`keep_out_margin(phi_r, error)` is the largest change of position under the error,
$\max_{e \in E} |\Phi_r e|_2$, with $\Phi_r$ the position rows of the state transition matrix.
The range $|r|$ is 1-Lipschitz in $r$ and the Clohessy-Wiltshire map is linear, so the true range
is at least the range predicted from $\hat x$ minus this margin. It is exact for an ellipsoid,
$\sigma_{\max}(\Phi_r P)$, attained at $P v_1$ with $v_1$ the top right singular vector, and for a
box, the largest value over its 64 vertices, where a convex function on a polytope peaks. For two
balls it is the bound $r_{\text{pos}} \sigma_{\max}(\Phi_{r,\text{pos}}) + r_{\text{vel}}
\sigma_{\max}(\Phi_{r,\text{vel}})$, exact when both blocks peak along the same output direction
and above the true maximum otherwise.

### Execution bias

A persistent unmodelled acceleration $\delta a$ along the barrier gradient moves the equilibrium a
barrier filter rides at. For an exponential control barrier function with real poles $p_1, p_2$,
the barrier value on the boundary follows

$$
\ddot h + (p_1 + p_2) \dot h + p_1 p_2 h = -\delta a,
$$

which settles at $h = -\delta a / (p_1 p_2)$. `ecbf_bias(da_max, poles)` returns the depth
$\delta a_{\max} / (p_1 p_2)$ in m, the amount by which the barrier must be moved out. For a
first-order barrier $\dot h = -\gamma h - \delta a$, `first_order_bias(da_max, gamma)` returns
$\delta a_{\max} / \gamma$ in the barrier's units. Poles and gains must be positive.

## Control barrier functions

A control barrier function $h(x)$ is nonnegative on a safe set of Hill-frame states
$x = (r, v)$. A filter keeps the state in the set by requiring a condition on the commanded
acceleration $u$ that limits how fast $h$ may fall (Ames et al., *Control Barrier Functions:
Theory and Applications*, ECC 2019). Every barrier below with a `row(x, a_cw)` method returns that
condition as one linear row $a \cdot u \ge b$, with $a_{cw}(x)$ the natural acceleration of the
dynamics at $x$ (for Clohessy-Wiltshire, `cw_rhs(0.0, x, n, np.zeros(3))[3:]`). The rows go to
`vbar.solvers.project`, which finds the nearest command under the thrust limit.

With $\rho = |r|$, $\hat r = r / \rho$ and $\dot\rho = \hat r \cdot v$, the range has
$\ddot\rho = (|v|^2 - \dot\rho^2)/\rho + \hat r \cdot (a_{cw} + u)$.

### Keep-out sphere

`KeepOutEcbf(radius, poles)` is the exponential control barrier function (sec. IV, Definition 7)
for $h = \rho - R$, which has relative degree 2. With real poles $p_1, p_2 > 0$ the condition
$\ddot h + (p_1 + p_2) \dot h + p_1 p_2 h \ge 0$ is the row

$$
\hat r \cdot u \ge -\frac{|v|^2 - \dot\rho^2}{\rho} - \hat r \cdot a_{cw}
  - (p_1 + p_2)\dot\rho - p_1 p_2 (\rho - R).
$$

`gains` returns $(p_1 p_2, p_1 + p_2)$, for which the closed loop $\ddot h = -p_1 p_2 h -
(p_1 + p_2)\dot h$ has eigenvalues $-p_1, -p_2$ (sec. IV.B prints them as $p_i$).
`contains(x)` is the set from which the continuous-time condition keeps $h \ge 0$:
$h \ge 0$ and $\dot\rho + \max(p_1, p_2)\, h \ge 0$. Theorem 8 prints the eigenvalue condition
with the wrong sign; as printed, a closing start would need a positive eigenvalue. The pole order
along the chain is free, so the faster pole decides: from $h_0 = 10$ m closing at 0.5 m/s, the
double pole 0.04 1/s crosses zero at 100 s (lowest $-0.0168$ m at 125 s), while the poles
(0.04, 0.5) stay safe.

Poles must be real and positive, otherwise `ValueError`: the comparison step after eq. 18 holds
only for real closed-loop poles.

### Speed limits

Speed constraints have relative degree 1, so they use the first-order condition
$\dot s \ge -\gamma s$ (sec. II, Definition 2):

- `SpeedCbf(v0, v1, gamma)`: $s = v_0 + v_1 \rho - |v|$, row
  $-\hat v \cdot u \ge \hat v \cdot a_{cw} - v_1 \dot\rho - \gamma s$. `row` returns None below
  $|v| = 10^{-9}$ m/s, where $\hat v$ is undefined and $s \ge v_0 > 0$.
- `ClosingSpeedCbf(v0, v1, gamma)`: $s = v_0 + v_1 \rho - c$ with the closing speed
  $c = \max(0, -\dot\rho)$, row $\hat r \cdot u \ge -(|v|^2 - \dot\rho^2)/\rho - \hat r \cdot
  a_{cw} - v_1 \dot\rho - \gamma s$ while closing; None when $\dot\rho \ge 0$.

$v_0$ and $\gamma$ must be positive and $v_1$ nonnegative.

### Approach cone

`ConeEcbf(half_angle, axis, apex, active_range, poles)` is the exponential barrier for
$h_c = a \cdot (r - p) - \cos\theta\, |r - p|$, with unit axis $a$, apex $p$ and half-angle
$\theta$. With $d = r - p$ and $g = a - \cos\theta\, \hat d$ the gradient, the row is

$$
g \cdot u \ge \cos\theta \frac{|v_\perp|^2}{|d|} - g \cdot a_{cw} - (p_1 + p_2)\dot h_c
  - p_1 p_2 h_c,
$$

where $v_\perp$ is the part of $v$ normal to $d$; the first term is the curvature of $|d|$. On the
cone surface $|g| = \sin\theta$ (0.173648 at 10 degrees), so $h_c$ is the signed distance to the
surface scaled by $\sin\theta$. `row` returns the condition within `active_range` of the apex,
including outside the cone, where it drives $h_c$ back up, and None beyond it or at the apex.

On the surface with $h_c = \dot h_c = 0$ and a lateral speed $v_{lat}$ the row needs
$g \cdot u \ge \cos\theta\, v_{lat}^2 / |d|$, and a ball of radius $a_{\max}$ offers at most
$a_{\max} \sin\theta$. `infeasible_radius(v_lat, a_max)` returns the distance inside which that
fails, $\cot\theta\, v_{lat}^2 / a_{\max}$: 0.028, 0.255 and 0.709 m for 0.01, 0.03 and 0.05 m/s
at 10 degrees and 0.02 m/s^2. A box contains the ball, so the radius holds for both thrust
limits.

### Assumptions and limits

- The conditions are continuous-time. Under a command held for a step $\Delta t$ they can fail.
  For the double pole $p$ on a double integrator, the set $\dot h + p h \ge 0$ stops being
  invariant at $p \Delta t = (\sqrt5 - 1)/2 = 0.618$, and $h$ first dips below zero between samples
  at $p \Delta t = 0.786578$, the root of $2s^4 + 4s^3 - 6s^2 + 1$. On Clohessy-Wiltshire head-on
  the first failing step is $0.78658/p$ within 0.3% for $p$ = 0.05 and 0.1 1/s. So under sample
  and hold the rows need $\max(p_1, p_2)\, \Delta t$ below about 0.6.
- When the nominal command pushes into the barrier the state rides $h = 0$, and the held command
  misses the change of the CW acceleration over the step. The state can then dip below $h = 0$ by
  an amount that grows with the tangential speed and the step and falls with the poles, at any
  gain, so a command that rides the boundary needs a buffer on $R$ or a check of the held step.
- A row says nothing about the thrust limit. Under the limit it can be infeasible: one row
  $a \cdot u \ge b$ is feasible exactly when $b \le a_{\max} |a|_1$ (box) or
  $b \le a_{\max} |a|_2$ (ball).
- All guarantees assume the dynamics that produced $a_{cw}$.

## Evading-maneuver barriers

A barrier can also be built from a backup command $\beta$ that is known to be safe: $h(x)$ is
the margin that $\beta$ keeps from $x$, and $\beta$ is then a safe command everywhere on
$h \ge 0$ (sec. III, eq. 12 and Theorem 4). These barriers have no row. A filter checks a
candidate command on the exact held segment and searches toward $\beta$. Both
classes give `value(x)`, `contains(x)`, `horizon(x)` and `backup(x, actuator)`.

### Free drift

`PassiveBarrier(radius, n, grid=5.0, t_cap=None)` uses $\beta = 0$, which every thrust limit
contains. $h_p(x)$ is the lowest keep-out margin of the unforced CW motion from $x$. In the CW
solution the along-track position is

$$
y(\tau) = y_0 - \frac{2 v_{x0}}{n} + \left(6 x_0 + \frac{4 v_{y0}}{n}\right)\sin n\tau
  + \frac{2 v_{x0}}{n} \cos n\tau - (6 n x_0 + 3 v_{y0})\,\tau,
$$

whose periodic part is at most $B = 6|x_0| + |y_0| + 4|v_{x0}|/n + 4|v_{y0}|/n$. Past
$(R + B)/|6 n x_0 + 3 v_{y0}|$ the range exceeds $R$ for good, so the drift is evaluated on the
horizon

$$
T(x) = \min\!\left(t_{\text{cap}},\ \frac{R + B}{|6 n x_0 + 3 v_{y0}|} + \text{grid}\right),
$$

with $t_{\text{cap}}$ ten orbits by default. `horizon(x)` returns $T(x)$ and whether $T <
t_{\text{cap}}$, in which case the sign of $h_p$ holds for all time. On a closed relative orbit ($6
n x_0 + 3 v_{y0} = 0$, for example a hold point on V-bar) the minimum repeats every orbit, the
horizon is capped, and the sign holds for $t_{\text{cap}}$ only.

The drift is sampled every `grid` seconds with one product of the stored transition matrices, and
each sample interval gets the larger of the first-order bound and the second-order bound on
$|r|^2$ that `vbar.constraints.segment_minimum` uses, at zero thrust. `value(x)` is the lowest of
these, so $h_p \ge 0$ certifies the drift up to the horizon; a negative value bounds the deepest
incursion from below. `grid` must be short enough for the in-step bounds to exist (below about
350 s at 500 km; at 5 s their gain is 0.011). `value` also takes a stack of states, shape (k, 6).

`gradient(x)` is the gradient of the exact drift minimum by the envelope theorem,
$\nabla h_p = \Phi_r(\tau^*)^T \hat r(\tau^*)$, with $\tau^*$ the minimiser found from the samples
and refined by a bounded scalar search. It raises `ValueError` on a closed relative orbit
($|6 n x_0 + 3 v_{y0}| < 10^{-9}$ m/s), where the minimiser is not unique and $h_p$ is not smooth,
and when $\tau^*$ is at 0 or at the end of the horizon, where the minimum is not stationary.


### Braking

`BrakingBarrier(radius, a_b, rho_cap, v_cap)` is the keep-out analogue of the survey's
lane-keeping barrier (sec. V.B):

$$
h_b = \rho - R - \frac{\max(0, -\dot\rho)^2}{2 a_b}.
$$

Its backup is full outward braking along $\hat r$, $\beta = a_{\max} \hat r / \max_i |\hat r_i|$
for a box and $a_{\max} \hat r$ for a ball, so $\hat r \cdot \beta \ge a_{\max}$. While closing at
$c = -\dot\rho$, $\dot h_b = \dot\rho + (c / a_b)\ddot\rho \ge \dot\rho + c = 0$ as long as
$\ddot\rho \ge a_b$. `braking_deceleration(a_max, layout, v_cap, dt, radius, rho_cap, n)` returns
the $a_b$ for which that holds under a command held for `dt` in the region $\rho \le
\rho_{\text{cap}}$, $|v| \le v_{\text{cap}}$:

$$
a_b = a_{\max}\left(1 - k \frac{v_{\text{cap}}\, \Delta t}{R}\right)
  - \left(3 n^2 \rho_{\text{cap}} + 2 n v_{\text{cap}}\right),
\qquad k = \sqrt3 \text{ (box)},\ 1 \text{ (ball)}.
$$

Outside the keep-out $\hat r$ turns by at most $v_{\text{cap}} t / R$ within a step, which costs at
most $|\beta|\, v_{\text{cap}} \Delta t / R$ of the braking, and the CW terms take at most
$3 n^2 \rho + 2 n |v|$. For a box of 0.02 m/s^2, $v_{\text{cap}} = 0.3$ m/s, $\Delta t = 10$ s,
$\rho_{\text{cap}} = 2$ km and 500 km, $a_b = 0.00991$ m/s^2. The value may be zero or negative
(at $\rho_{\text{cap}} = 20$ km the CW terms alone exceed 0.02 m/s^2); `BrakingBarrier` refuses
$a_b \le 0$. `contains(x)` requires $h_b \ge 0$ inside the region.

$L_g h_b$ vanishes at $\dot\rho = 0$, so a first-order row on $h_b$ would put no condition on $u$
there, and a whole held step of inward thrust leaves the set. The class therefore has no `row`.

## Held-segment check and line search

### Certified margin over a held step

`held_segment_bound(x, u, dt, m, radius, n, error=None, da_max=0.0)` returns a lower bound on
$|r(t)| - R$ over a step of length `dt` with the command `u` held. The `m` samples are
$\Phi(t_i) x + \Gamma(t_i) u$ from `CWDiscretizer.sample_matrices`, exact for CW, and each sample
interval gets the larger of the first-order and the second-order bounds of
`vbar.constraints.segment_minimum`, without its refinement. Without error sets the result equals
the `lower_bound` that `segment_minimum` reports on the same samples. It returns `-inf` when the
step is too long for the in-step bounds.

With an error set $E$ on $x$ (`BoxError`, `BallError`, `EllipsoidError`) and an execution error
$\delta a$ with $|\delta a| \le$ `da_max`, constant over the step, the true trajectory differs
from the estimated one by a CW trajectory from $e \in E$ under $\delta a$. Its position at a sample
is at most

$$
g_i = \texttt{keep\_out\_margin}(\Phi_r(t_i), E) + \delta a_{\max}\, \sigma_{\max}(\Gamma_r(t_i)),
$$

and its velocity at most $g^v_i$, the same with the velocity rows. Between samples the deviation's
position grows at most at its speed $W$, and its speed at most at
$3 n^2 |r| + 2 n |v| + \delta a_{\max}$, so with $h$ the sample spacing

$$
W \le \frac{\max_i g^v_i + \tfrac{h}{2}\left(3 n^2 \max_i g_i + \delta a_{\max}\right)}
  {1 - n h - \tfrac34 n^2 h^2},
$$

and each interval's bound is lowered by $(g_i + g_{i+1} + W h)/2$. The tightening depends only on
the step, the error set and `da_max`, and is cached for an error set object that is used again.

### Line search toward the backup

`line_search(accept, u_c, beta, iterations=12)` tries the candidate `u_c` first and, if `accept`
rejects it, bisects on $\theta \in [0, 1]$ with $u = \theta u_c + (1 - \theta)\beta$, keeping the
largest accepted $\theta$. It returns `LineSearch(u, theta, checks)`: $\theta = 1$ after one
check, otherwise a multiple of $2^{-12}$ after 13 checks, and $\theta = 0$, the backup itself,
when nothing else passes. The backup is never checked here: its safety is the barrier's theorem.
Bisection finds the largest accepted $\theta$ only when the accepted values form an interval
$[0, \theta^*]$; otherwise it returns an accepted value that need not be the largest.

## Validation

The tests in `tests/test_safety_projections.py` check:

- The halfspace projection on hand cases (sec. 8.1.1), box clipping and radial scaling onto the
  ball.
- All three cone regimes of exercise 8.3(c) on hand cases, including $(3, 4)$ with $t = 0$ mapping
  to $(1.5, 2, 2.5)$; on 200 random points of the third regime the result lies on the boundary and
  the residual is orthogonal to it; a point of the negative cone maps to the apex.
- The keep-out projection on 200 random cases (3x3 $G$, $R$ in [2, 3], $u_{\text{nom}}$ deep
  inside) against an independent global search over step directions from $u_{\text{nom}}$, a 4000
  point grid refined by the simplex method: objectives agree to 1e-9 relative (measured 5e-14),
  the result lies on the sphere to 1e-12, $\lambda$ satisfies the stationarity condition and lies
  in $(0, 1/\sigma_{\max}^2)$, and the objective never exceeds that of the projection onto the
  tangent halfspace at $u_{\text{nom}}$, which lies inside the feasible region because $|Gu + h|$
  is convex.
- A constructed hard case, axis-aligned and rotated, returns `hard_case` True with
  $\lambda = 1/s_1^2$, a command outside the sphere and the objective of the global search; a
  nearby regular case gives the same command magnitudes to 1e-7.
- A rectangular $G$ moves the command only in its row space; refusals of a zero $G$ and bad inputs.

The tests in `tests/test_safety_tightening.py` check:

- Each support function equals the maximum over the set's extreme points to 1e-12 relative: the
  64 box vertices; for the balls and the ellipsoid the maximizing point attains it and no sampled
  boundary point exceeds it.
- `from_gaussian` gives $P P^T = k^2 S$, and its exceedance equals the closed form
  $e^{-x/2}(1 + x/2 + x^2/8)$ of the 6-degree-of-freedom tail at $x = k^2$ to 1e-12 and the
  fraction of 200 000 Gaussian draws outside the ellipsoid within five binomial deviations.
- $\Phi^{-1}(1 - 10^{-6}) = 4.753$ to 1e-3 (sec. 7.4.2); the chance margin at $\eta = 0.9$ is
  exceeded by 10% of 200 000 draws within five binomial deviations; it equals the support of the
  $k$-sigma ellipsoid at $\eta = \Phi(k)$; $\eta = 0.5$ gives 0, and $\eta = 0.4$, just below 0.5
  and $\eta = 1$ raise.
- The ellipsoid keep-out margin is attained by the top singular vector, and no sampled boundary
  point exceeds it; the box margin bounds interior samples; the two-ball bound bounds samples and
  is attained when both blocks share their singular vectors.
- `ecbf_bias` gives 0.08, 0.02 and 0.008 m for $(p, \delta a) = (0.05, 2\cdot10^{-4})$,
  $(0.1, 2\cdot10^{-4})$ and $(0.05, 2\cdot10^{-5})$ with a double pole, to 1e-5 m, and equals minus
  the steady state of the barrier dynamics integrated numerically for the poles (0.05, 0.05),
  (0.03, 0.08) and (0.02, 0.5); `first_order_bias` likewise.

The tests in `tests/test_safety_barriers.py` check:

- The closed loop with `gains` has eigenvalues $-p_1, -p_2$ for the poles (0.05, 0.05),
  (0.02, 0.1) and (1, 3), to 1e-9 (Ames et al., sec. IV.B, with the sign corrected).
- The corrected start condition of Theorem 8: from $h_0 = 10$ m, $\dot h_0 = -0.5$ m/s the poles
  (0.06, 0.06), (0.05, 0.08), (0.04, 0.5) and (0.5, 0.04) are accepted by `contains` and keep
  $h \ge 0$ for 600 s; the double pole 0.04 is rejected, and its closed form crosses zero at
  100.0 s with minimum $-0.0168$ m at 125 s, to 1e-6 m. A property test compares `contains`
  with the condition on drawn starts and poles; complex, zero, negative and NaN poles raise.
- Each row equals $\ddot h + k_1 \dot h + k_0 h$ (or $\dot s + \gamma s$) along the held
  segment by fourth-order differences of the exact CW solution, to 1e-6 relative, and matches a
  hand computation at one state to 1e-12; the speed rows are dropped below 1e-9 m/s and while
  not closing, tested just inside and outside each limit.
- The cone: on the surface $|g| = \sin 10^\circ = 0.173648$ and the free $\ddot h_c =
  -\cos\theta\, v^2/\rho = -1.77265 \cdot 10^{-4}$ m/s^2 at 5 m and 0.03 m/s, to 1e-5; the row
  matches the differences on 20 states inside the cone; the infeasible radii are 0.028, 0.255 and
  0.709 m.
- The free-drift value lies below the drift margin sampled every second over the horizon, and
  within $5\,(|v|_{\max} + 0.05)$ m of it, on 50 approach-vbar starts; past the horizon no
  violation is missed. The horizon formula is checked on a decided and a capped state.
- The envelope gradient matches central differences of the refined drift minimum to 1e-6
  relative on 5 states (measured 1.1e-7); a closed relative orbit, a minimum at $\tau = 0$ and a
  minimum at the horizon end raise.
- The periodic bound $B$ holds on 1000 random states within 2 km and 1 m/s and on drawn states
  within 5 km and 2 m/s.
- `braking_deceleration` equals the formula for both layouts and 0.00991 m/s^2 for the approach
  box; zero, negative and NaN $a_b$ raise, and so does the value at $\rho_{\text{cap}} = 20$ km.
  Under the backup held for 10 s, $h_b$ does not fall on 100 closing states inside the region,
  for box and ball.
- The sample-and-hold thresholds of the double integrator, found by bisection over starts on the
  boundary of the set, are $(\sqrt5 - 1)/2 = 0.618034$ for the set and 0.786578, the root of
  $2s^4 + 4s^3 - 6s^2 + 1$, for $h$, to 1e-4; both lie above 0.6. On CW
  head-on with a reckless inward command, the row through `project` holds at $0.5$ and $0.997$
  times $0.78658/p$ and fails at $1.003$ times it, for $p$ = 0.05 and 0.1 1/s.

The tests in `tests/test_safety_sampled.py` check:

- `held_segment_bound` equals `segment_minimum`'s certified bound on the same samples exactly, on
  four steps (passing, grazing, closing at the end and already inside); it lies below the
  minimum over 2001 exact samples and within one sample interval's first-order loss of it; a
  property test over positions within 300 m, speeds to 0.5 m/s, thrust to 0.02 m/s^2 and steps
  of 1 to 20 s finds no case above the true minimum.
- A navigation error lowers the bound by at least the range error at the start of the lowest
  interval and at most the largest range error plus its rate over one interval, for a box and
  two balls; the tightened bound holds at all 64 vertices of the error box. The execution error
  lowers it within the same limits from $\sigma_{\max}(\Gamma_r)$, and the bound holds for 20
  drawn errors of the full size. Larger error sets give strictly lower bounds, and a zero set
  gives the nominal bound.
- `line_search` keeps an accepted candidate after one check, returns the backup after 13 checks
  without asking about it, ends on $\lfloor 0.3 \cdot 4096 \rfloor / 4096$ for an acceptance
  threshold of 0.3, follows `iterations`, and on an acceptance set that is not an interval stops
  below the upper window.

```python
import numpy as np

from vbar.safety.projections import keep_out_projection, second_order_cone
from vbar.safety.tightening import EllipsoidError, chance_margin, ecbf_bias, halfspace_margin

x, t = second_order_cone([3.0, 4.0], 0.0)
print(x, t)  # [1.5 2. ] 2.5

G = np.diag([50.0, 50.0, 50.0])  # m per m/s^2 over one held step
h = np.array([-30.0, 0.0, 0.0])  # m, inside a 50 m keep-out with zero command
result = keep_out_projection(np.array([0.1, 0.0, 0.0]), G, h, 50.0)
print(result.u.round(6), result.hard_case)  # [-0.4  0.   0. ] False

cov = np.diag([1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4])  # m^2 and m^2/s^2
a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # a row on the radial position
error = EllipsoidError.from_gaussian(cov, 3.0)
print(round(halfspace_margin(a, error), 6), f"{error.exceedance:.4f}")  # 3.0 0.1736
print(round(chance_margin(a, cov, 1.0 - 1e-6), 3))  # 4.753
print(round(ecbf_bias(2e-4, (0.05, 0.05)), 6))  # 0.08
```

```python
import numpy as np

from vbar.dynamics import cw_rhs, mean_motion
from vbar.safety.barriers import KeepOutEcbf, PassiveBarrier, braking_deceleration
from vbar.solvers import ActuatorSet, Rows, project

n = mean_motion(6378137.0 + 500e3)  # rad/s, 500 km circular
box = ActuatorSet(0.02, "box")  # m/s^2 per axis
u_nom = np.array([0.0, -0.02, 0.0])  # m/s^2, straight at the target

x = np.array([0.0, 60.0, 0.0, 0.0, -0.3, 0.0])  # m and m/s, closing along V-bar
ecbf = KeepOutEcbf(radius=50.0, poles=(0.05, 0.05))
a, b = ecbf.row(x, cw_rhs(0.0, x, n, np.zeros(3))[3:])
u = project(u_nom, Rows(a[None], np.array([b]), ("keep_out",)), box).u
print(ecbf.contains(x), u.round(6))  # True [0.    0.005 0.   ]

x = np.array([0.0, 80.0, 0.0, 0.0, 0.0, 0.0])  # holding on V-bar, 80 m out
passive = PassiveBarrier(radius=50.0, n=n)
print(round(passive.value(x), 3), passive.horizon(x)[1])  # 29.998 False

a_b = braking_deceleration(
    a_max=0.02, layout="box", v_cap=0.3, dt=10.0, radius=50.0, rho_cap=2000.0, n=n
)
print(round(a_b, 5))  # 0.00991
```

```python
from vbar.dynamics import cw_discrete
from vbar.safety.sampled import held_segment_bound, line_search
from vbar.safety.tightening import BoxError

phi, gamma = cw_discrete(n, 10.0)
error = BoxError(r_pos=0.5, r_vel=0.005)


def accept(u):
    held = held_segment_bound(x, u, 10.0, 20, 50.0, n, error=error)
    return held >= 0.0 and passive.value(phi @ x + gamma @ u) >= 0.0


found = line_search(accept, u_nom, passive.backup(x, box))
print(found.theta, found.checks)  # 0.45166015625 13
```
