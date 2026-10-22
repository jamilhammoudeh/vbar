# Safety filters

Code: `vbar.safety.projections` (`halfspace`, `box`, `ball`, `second_order_cone`,
`keep_out_projection`, `KeepOutProjection`) and `vbar.safety.tightening` (`BoxError`, `BallError`,
`EllipsoidError`, `halfspace_margin`, `chance_margin`, `keep_out_margin`, `ecbf_bias`,
`first_order_bias`), `vbar.safety.barriers` (`KeepOutEcbf`, `SpeedCbf`, `ClosingSpeedCbf`,
`ConeEcbf`, `PassiveBarrier`, `BrakingBarrier`, `braking_deceleration`), `vbar.safety.sampled`
(`held_segment_bound`, `line_search`, `LineSearch`), `vbar.safety.cbf` (`CbfFilter`,
`BackupFilter`, `SoftCbfFilter`, `FilterDeclaration`) and `vbar.safety.audit` (`audit_filter`,
`FilterAudit`, `RunAudit`, `Refinement`, `FlaggedDecision`).

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

## Filters

`vbar.safety.cbf` holds three filters built from the pieces above. Each one has a keyword-only
constructor, `reset(view)` and `filter(obs, action)` returning a `FilterResult`, and can be wrapped
around any agent with `FilteredAgent`.

### What every filter does

- **Commands.** The action is converted to an acceleration in m/s^2 at the initial mass
  (`command_fraction` times `max_accel`), clipped onto the thrust limit (per axis for a box,
  radially for a ball), filtered, and returned in the view's action mode. Clipping is what the
  spacecraft does anyway, so a clipped command is not an intervention. `intervened` is True when
  the filtered acceleration differs from the clipped one by more than $10^{-6} a_{\max}$.
- **What a filter reads.** The first six observation entries, the Hill-frame state estimate, and
  the `AgentView`. Barriers, safe sets and `recovering` are evaluated on the estimate; error sets
  are declared in the constructor, never read from the environment.
- **Refusals at reset** (`ValueError`, short message): an action mode that is not continuous; a
  frame other than "hill" or an observation that does not start with x, y, z, vx, vy, vz; a
  pending-commands channel (the filter cannot predict actuation delay); a dynamics model other
  than "cw" without a declared `model_error`; a constraint type the filter does not enforce,
  unless it is listed in `ignore=`; a scenario without a keep-out constraint. `keep_out` itself
  cannot be ignored.
- **Commit radius.** Within the goal's `commit_radius` of the target (m, 0 when the goal has
  none) every filter only watches: it returns the clipped command with `active` False and
  `recovering` and `fallback_level` None.
- **Declaration.** After `reset`, `declaration` is a `FilterDeclaration`: name, safe set, barrier
  objects, poles, gain, design dt, thrust layout and limit, keep-out buffer, declared error sets,
  backup command, certificate kind, whether the filter counts toward certification, and the
  ignored constraint types. `contains(x)` tests membership of the declared safe set. Names are
  fixed at construction: "cbf-ecbf-p0.05", "cbf-passive", "cbf-brake", "cbf-ecbf-soft-p0.05".

### Exponential barrier filter

`CbfFilter(poles=(0.03, 0.03), gamma=0.03, v_cap=0.2, execution=0.0, model_error=None,
ignore=())` builds, at every decision, the keep-out row of `KeepOutEcbf` with the radius
$R_f = R + b$, a speed-cap row `SpeedCbf(v_cap, 0, gamma)`, and a `SpeedCbf` or `ClosingSpeedCbf`
row for each declared `speed_limit` and `closing_speed_limit`, and projects the clipped command
onto them under the thrust limit with `vbar.solvers.project`. The result is the minimal correction
for these rows, not the least restrictive safe command: the rows describe a strict subset of the
states from which the keep-out can be kept.

The rows are continuous-time conditions, so the filter is designed for the held step:

- **Pole rule.** `reset` refuses $\max(p_1, p_2)\,\Delta t > 0.6$ and $\gamma\,\Delta t > 0.6$ and
  names the largest allowed value ($0.6/\Delta t$, for example 0.06 1/s at 10 s). On the double
  integrator the start set stops being invariant under hold at $p\,\Delta t = 0.618$ and the
  barrier dips between samples at 0.787 (section above).
- **Buffer.** $b = 1.2 \cdot 3 n^2 v_{\text{cap}} \Delta t / (p_1 p_2) + \delta a_{\max}/(p_1 p_2)$,
  the boundary-riding depth with 20% to spare plus the execution bias, with
  $\delta a_{\max}$ = `execution` + `model_error`. For the double pole 0.03 1/s,
  $v_{\text{cap}} = 0.2$ m/s and $\Delta t = 10$ s at 500 km, $b = 9.8$ mm.
- **Declared safe set** $S_E$: `KeepOutEcbf.contains` at $R_f$, $|v| \le v_{\text{cap}}$, the
  declared speed limits met, and $|a_{cw}(x)| + 1.1\,(2 p_{\max} + v_1)\, v_{\text{cap}} \le
  a_{\max}$ with $v_1$ the largest declared speed slope. On $S_E$ the command $u^* = -a_{cw} + 2
  p_{\max} c\, \hat r - v_1 c\, \hat v$, $c = \max(0, -\dot\rho)$, meets every row and lies in the
  ball of radius $a_{\max}$, so the projection is feasible under box and ball limits. With the
  example values the acceleration condition holds for $|x| \lesssim 1.7$ km.
- **Held-step checks.** A held command can break the rows between samples: thrust across the
  velocity raises the speed at second order in $\Delta t$, and $\hat r$ turns within the step, so
  $\hat r \cdot u$ changes. Each decision therefore caps the speed at the end of the step,
  $|\Phi_v x + \Gamma_v u| \le v_{\text{cap}}$ on the exact CW solution, by cutting planes: each
  projection that ends too fast adds the tangent of the convex condition
  $|\Phi_v x + \Gamma_v u| \le 0.999\, v_{\text{cap}}$ at its command, which every command meeting
  that tightened cap also meets, so an infeasible set of cuts proves the cap tightened by 0.1%
  infeasible, not the cap itself. After 16 cuts a command that still ends above the cap is
  returned as "solver_failed" with its excess speed as `slack` (m/s). The keep-out is then
  checked on the held step: `held_segment_bound` at least
  $\min(0, \rho - R)$ and the end range at least $\min(0, \rho - R_f)$, for every execution error
  up to $\delta a_{\max}$. A command that fails is replaced by the projection with the keep-out
  row raised by the least amount that passes, found by bisection below the largest feasible
  raise (the bisection assumes the check passes on an interval). Without such a raise the most
  outward feasible command is returned as "solver_failed" with its predicted shortfall in m.
- **No certificate.** Recursive feasibility under hold is not proven, so `certificate` is None.
  Navigation error sets are refused: the rows are nonlinear in the state.

| Outcome | `status` | `active` | `slack` | `conflicting` | `fallback_level` |
|---|---|---|---|---|---|
| No row active | feasible | False | 0 | () | 0 |
| A row active (positive multiplier) | feasible | True | 0 | () | 0 |
| Rows beyond the thrust limit | infeasible_certified, with `proof` | True | phase I sum of infeasibilities, m/s^2 | rows and limits of the Farkas certificate | 0 |
| Projection failed | solver_failed | True | largest row violation of the action, m/s^2 | () | 2 |
| Speed cap not met after 16 cuts, inside $S_E$ | solver_failed | True | end-of-step speed above the cap, m/s | () | 0 |
| The same, recovering | feasible | True | as above | () | 0 |
| No raise of the keep-out row passes the held-step check, inside $S_E$, short by more than 1 mm | solver_failed | True | predicted keep-out shortfall, m | () | 2 |
| The same, recovering or short by at most 1 mm | feasible | True | as above | () | 2 |

`recovering` is True outside $S_E$, judged on the estimate. When the rows cannot all be met the
action is the projection's least-violation command (next section); when the projection fails it
is the command that does most for the keep-out row, $a_{\max}\,\mathrm{sign}(\hat r)$ (box) or
$a_{\max}\hat r$ (ball). Outside $S_E$ a step that cannot keep the held check, for example inside
the keep-out while closing, is the predicted shortfall the step accepts, not a defect, so from
outside the set the filter returns only "feasible" or "infeasible_certified". Inside $S_E$ a
shortfall of at most the keep-out resolution, 1 mm, is the slack of the certified bound at the
boundary and is also "feasible".

### Backup filters

`BackupFilter(barrier="passive" | "braking", candidate="clipped", ...)` checks each held step on
the exact CW solution and searches toward the barrier's backup with `line_search`. A command $u$
is accepted when

1. `held_segment_bound(x, u, dt, 20, R, n, error, da_max)` $\ge f_s$, and
2. the barrier value at the end of the step, $x_1 = \Phi(\Delta t) x + \Gamma(\Delta t) u$, is
   $\ge f_b$, and, for the braking barrier,
3. $x_1$ stays in the region, $|r_1| \le \max(\rho_{\text{cap}}, \rho)$ and
   $|v_1| \le \max(v_{\text{cap}}, |v|)$, since the braking argument holds only there,

with the floors $f_s = \min(0, \rho - R)$ and $f_b = \min(0, h(x))$: 0 inside the set, and "no
worse than now" when recovering. The candidate is the clipped command; with
`candidate="projection"` and the free-drift barrier it is first projected onto the first-order
row $\nabla h_p \cdot (v, a_{cw} + u) \ge -\gamma h_p$, which corrects less.

The backup is accepted without the check when the state is inside the barrier's set; its safety
is the barrier's theorem (sec. III, Theorem 4). For free drift the drift from $x_1$ is part of
the certified drift from $x$. Under full outward braking $h_b$ does not fall while
$\rho \le \rho_{\text{cap}}$ and $|v| \le v_{\text{cap}}$. The backup lies in both thrust
limits, so without error sets the filter is recursively feasible at any dt and never reports an
infeasible step. A backup that fails the check inside the set by more than the keep-out
resolution, 1 mm, is a design defect and is reported as "solver_failed"; below that it is the
slack of the certified bound at the boundary.

The braking backup is radial in both layouts: $a_{\max} \hat r / \max_i |\hat r_i|$ for a box,
the longest radial command inside it, and $a_{\max} \hat r$ for a ball. Both give
$\hat r \cdot \beta \ge a_{\max}$ with $|\beta| \le \sqrt3\, a_{\max}$ (box), the bound that
`braking_deceleration` allows for the turn of $\hat r$ within a step. A box corner
$a_{\max}\,\mathrm{sign}(\hat r)$ also brakes, but its part across the line of sight raises the
speed and can carry the state out of the region in which the braking argument holds.

- `a_b` defaults to `braking_deceleration` at the view's thrust limit and dt minus
  $\delta a_{\max}$; reset refuses a value that is not positive. `rho_cap` (2000 m) and `v_cap`
  (0.3 m/s) bound the region.
- With error sets (`navigation`, `execution`, `model_error`) check 1 is tightened as in
  `held_segment_bound`, and check 2 by the error carried to the end state: for free drift the
  largest range error of the drift from it over the horizon, for braking the range and closing
  speed errors at the end state. The drift horizon `t_cap` is then two orbits by default, and
  recursive feasibility is not claimed: the backup's safety is no longer a theorem. A step whose
  returned command passes the tightened checks carries the certificate "bounded-W" with that
  horizon; a backup that fails them is still returned, as the most outward command, but with
  `certificate` and `certificate_horizon` None, `active` True, `fallback_level` 2 and its
  predicted shortfall in m as `slack`.
- The filter enforces the keep-out only and refuses other constraint types: free drift or
  braking may leave them.

| Outcome | `active` | `fallback_level` | `certificate`, `certificate_horizon` | `iterations` |
|---|---|---|---|---|
| Candidate accepted, $\theta = 1$ | False | 0 | "nominal"; inf, or `t_cap` on a closed relative orbit | 1 |
| $0 < \theta < 1$ | True | 0 | as above | 13 |
| Backup, $\theta = 0$ | True | 2 | as above | 13 |
| Recovering | True if $\theta < 1$ | 0 or 2 | None, None | 1 or 13 |
| Backup failing the tightened checks (error sets) | True | 2 | None, None | 13 |

The status is "feasible" in every row. When recovering, `slack` is the predicted violation of the
returned action, $\max(0, f_s - \text{bound}, f_b - h(x_1))$ in m. For free drift the horizon is
inf when the secular drift proves the minimum at $x_1$ and `t_cap` otherwise; for braking it is
inf.

### Soft filter

`SoftCbfFilter(alpha=1.0, ...)` takes the rows of `CbfFilter` and solves the exact penalty
problem

$$
\min_u \tfrac12 |u - u_{\text{nom}}|^2 + \alpha \sum_i \max(0, b_i - a_i \cdot u)
$$

over the thrust limit (Boyd and Vandenberghe, exercise 5.16). It is strongly convex, so the
solution is unique; at it the rows split into a set $V$ whose penalty is at its slope $\alpha$
and the rest, held as hard rows, so the solution is the projection of
$u_{\text{nom}} + \alpha \sum_{V} a_i$ onto the rows outside $V$. All $2^k$ sets of the
$k \le 4$ rows are tried and the candidate with the lowest penalized objective is returned. When
$\alpha$ exceeds every multiplier of the hard projection the result equals that projection; the
soft filter has no held-step check, so it differs from `CbfFilter` wherever that check acts. Below
that, or where the hard rows are infeasible, rows are violated, and `slack` (their summed
violation, m/s^2) and `conflicting` report them. The status is "feasible", since the penalized
problem always has a solution. A soft barrier has no invariance guarantee: the filter never counts
toward certification (`counts_toward_certification` is False) and its certificate is None.

### Infeasibility, proofs and least-violation actions

Outside its declared set a filter's rows can ask for more than the thrust limit gives. One row
$a \cdot u \ge b$ is feasible exactly when $b \le a_{\max}|a|_1$ (box) or $b \le a_{\max}|a|_2$
(ball); several rows can conflict although each alone is feasible, for example the keep-out row
pushing out and the speed cap holding back at 0.15 m/s 10 m inside the sphere.

`vbar.solvers.project` decides this exactly. When no candidate active set wins, its
sum-of-infeasibilities phase I gives a dual vector $\lambda \ge 0$, checked before infeasibility
is reported. A `CbfFilter` step with status "infeasible_certified" carries it in `proof`, an
`InfeasibilityProof(normals, bounds, multipliers, layout, a_max, names)` on the rows the filter
built, in SI, each row in its own units: the barrier and speed rows have unitless normals and
bounds in m/s^2, and the end-of-step speed cuts have normals $-\Gamma_v^T \hat v$ in s and bounds
in m/s. The certificate holds in any units, so nothing is rescaled between the filter and the
proof. Every $u$ in the actuator set $U$ has
$\lambda \cdot (A u) \le \max_{u \in U} (A^T \lambda) \cdot u$, the support $a_{\max}|A^T \lambda|_1$
of a box or $a_{\max}|A^T \lambda|_2$ of a ball, so

$$
b \cdot \lambda > \text{support}, \qquad \lambda \ge 0
$$

shows that no command meets every row (Boyd and Vandenberghe, sec. 5.8.3). `verify()` checks
exactly this from the proof's own arrays: $\lambda \ge 0$ to $10^{-12}$ of its largest entry, and
the support exceeded by $10^{-10} \max(a_{\max}, \max_i |b_i|/|a_i|) \sum_i \lambda_i |a_i|$, the
margin `project` certifies with. The proof lists the rows `project` names as conflicting; when
dropping the others' negligible multipliers breaks that margin, it lists every row, the
end-of-step speed cuts included. The arrays are read-only copies, and only an
"infeasible_certified" result may carry a proof (`FilterResult` raises otherwise), so an audit can
re-verify every infeasible step without the filter.

The action of an infeasible step is `project`'s least-violation command: under a box, the command
nearest the clipped nominal among those in the box that violate no row by more than the phase I
point; under a ball, the same search without the limit, scaled onto the ball, which can violate
the rows more than the phase I point ([solvers.md](solvers.md)). `slack` is the phase I sum of
the rows' violations at it, each scaled by its normal's length, $\sum_i (b_i - a_i \cdot u)/|a_i|$
in m/s^2 of command, and `conflicting` names the rows and limits of the certificate.

```python
import numpy as np

from vbar import ApproachConfig, ApproachEnv
from vbar.safety.cbf import CbfFilter

view = ApproachEnv(ApproachConfig(action_mode="accel")).agent_view()  # commands in m/s^2
ecbf = CbfFilter(poles=(0.05, 0.05), v_cap=0.1, gamma=0.02)
ecbf.reset(view)

x = np.array([0.0, 40.0, 0.0, 0.0, 0.15, 0.0])  # m and m/s, 10 m inside the keep-out, leaving
result = ecbf.filter(x, np.zeros(3))
proof = result.proof
print(result.status, result.recovering)  # infeasible_certified True
print(proof.names, proof.verify())  # ('keep_out', 'speed_cap') True
print(result.action.round(4), round(result.slack, 5))  # [0.   0.01 0.  ] 0.011
```

### Assumptions and limits

- All filters assume the CW model; other dynamics need a declared `model_error`, which enters as
  an acceleration bound.
- Actuation delay is refused. Thrust drive (acceleration rising as mass falls), the minimum
  impulse bit and execution misalignment change the delivered command without the filter
  knowing; on such scenarios its predictions are wrong.
- `CbfFilter`'s speed cap restricts the agent beyond the scenario: from rest, a full command
  held for 10 s reaches 0.2 m/s, so with a 0.1 m/s cap the filter halves it. Declared
  `speed_limit` and `closing_speed_limit` rows are first-order conditions at the start of the
  step only; only the filter's own cap is checked at the end of the step.
- The braking barrier's region (`rho_cap`, `v_cap`) is declared, not enforced: a nominal that
  leaves it puts the filter in recovery.
- The free-drift barrier evaluates up to 11 355 drift samples per check (ten orbits at 5 s); a
  step that needs the line search makes 13 checks.

## Audit

`audit_filter(safety_filter, env, *, nominal=None, starts=1000, mode="inside", box=None,
seed=None, max_steps=None, workers=1)` flies a filter around a nominal agent from seeded starts
and judges every step on the true trajectory. It returns a `FilterAudit` with one `RunAudit`
per start and the totals.

### How the starts are drawn

- Starts are drawn uniformly from `box`, the (low, high) corners of a box of Hill states (m,
  m/s), by default the scenario's start position and velocity ranges, and kept when the filter's
  own `contains(x)` method says they lie on the asked side of its declared safe set (the
  `FilterDeclaration` has no membership test; the same method judges flagged decisions and
  re-entry): inside for `mode="inside"`, outside for `mode="outside"`. `drawn` and `rejected`
  count the draws. After
  1000 draws per start asked for, the audit stops with `ValueError`.
- `seed` is a root seed in $[2^{64}, 2^{128})$, drawn from the operating system (128 bits) when
  not given and recorded in the report. The starts come from one stream of it; each start's episode
  seed and agent seed, 128 bits each, from another, and the agent seed never equals the episode
  seed. A seed below $2^{64}$ is refused: it could be found by trying candidates.
- Each start is flown through `record_episode` in a fresh environment built from the scenario's
  class and config with `start_state` set, at the scenario's dt and for `max_steps` steps (the
  scenario's own by default), with the filter wrapped around `nominal` (`PDAgent()` by default)
  by `FilteredAgent`.

### How a step is judged

For every hard constraint, each step's certified in-step lower bound from the environment, the
larger of the first-order and second-order bounds of `segment_minimum` on its samples, is
compared with minus the constraint's `resolution` (1 mm for the keep-out):

- **Certified**: the bound is at least $-\text{resolution}$.
- **Violation**: the bound is below it and so is the refined minimum, a point on the
  trajectory.
- Otherwise the bound alone cannot decide, and the step is **re-flown**: every segment of the
  step is rebuilt from its recorded start state and acceleration with the closed-form CW
  solution (`cw_states`) on a grid with 4, 16 and then 64 times the environment's sample
  intervals, nesting the original samples, and `segment_minimum` is run again. The first grid
  whose bound reaches $-\text{resolution}$ certifies the step; a refined minimum below it makes
  the step a violation; a step that neither happens to by 64 times is **uncertified**. Each
  re-flown step is listed as a `Refinement` with both bounds, the last factor and its outcome.

Every decision whose status is not "feasible" is listed as a `FlaggedDecision`: its time, status,
conflicting rows, the filter's `recovering`, and whether the true state was inside the declared
set. For "infeasible_certified" decisions the result's `proof` is checked with its own
`verify()`; `proof_verified` is None when the result carries no proof.

### Pass rules

- Inside the set (`mode="inside"`): `passed` needs no violation, no uncertified step, no
  infeasible and no "solver_failed" decision in any run.
- Outside it (`mode="outside"`): `passed` needs no "solver_failed" decision and a proof that
  verifies on every "infeasible_certified" one (`proofless` and `proofs_rejected` both 0). Each
  run reports `reentry_s`, the first decision time at which the true state is back in the set,
  inf if it never is; `reentered` and `longest_reentry_s` summarise them.

### Refusals

`ValueError` for dynamics other than "cw", a sensor other than `PerfectNav`, a configured
execution error, actuation delay, a nonzero minimum impulse bit or the "thrust" drive (each
changes the delivered command without the filter knowing), a scenario config without
`start_state`, a filter whose declaration does not count toward certification
(`SoftCbfFilter`), a root seed outside $[2^{64}, 2^{128})$, a non-positive `starts`, `max_steps`
or `workers`, and an unknown mode; `TypeError` for a filter without `declaration` and
`contains`.

### Parallel runs

With `workers` above 1 the runs are flown in a process pool. Every start and seed is drawn
before the pool starts, so the report does not depend on `workers`. The filter must not have
been reset yet: a reset filter holds its view, which does not pickle.

### Assumptions and limits

- The audit is only as strong as its starts: sampled starts miss rare failures, so a clean audit
  is not a proof of invariance. A falsification search over the declared set is a separate check.
- Re-flying uses the CW closed form, so it is exact only for "cw" dynamics, the only dynamics
  the audit accepts.
- Re-entry is judged at decision times, every dt, on the true state; a state that passes through
  the set between decisions is not counted.
- Inside the set, an infeasible or "solver_failed" decision fails the audit wherever it
  happens, also after the state has left the set.

```python
from vbar import ApproachEnv
from vbar.safety.audit import audit_filter
from vbar.safety.cbf import BackupFilter

near = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))  # m, m/s
found = audit_filter(
    BackupFilter(barrier="braking"),
    ApproachEnv(),
    starts=5,
    box=near,
    seed=0xC3A5C85C97CB3127B4A3E1A53C1F2D0B,
    max_steps=30,
)
print(found.violations, found.uncertified, found.infeasible, found.passed)  # 0 0 0 True
print(found.drawn, found.rejected, found.runs[0].seed != found.runs[0].agent_seed)  # 5 0 True
```

## Falsification

The audit flies starts drawn uniformly from the declared safe set. A falsification search
chooses its starts instead: it looks for the start whose run comes closest to failing, by
minimizing the run's certified robustness over the set (Kochenderfer, Katz, Corso and Moss,
*Algorithms for Validation*, ch. 4). A failure it finds is a counterexample. Finding none shows
nothing about the starts it did not fly, so the best verdict it gives is "none_found", never
"safe".

### Search kernel

`vbar.stress.cross_entropy(score, lo, hi, *, accept=None, budget=1000, population=50,
elites=6, smoothing=0.7, sigma_floor=0.02, chains=2, seed=None, workers=1)` minimizes
`score(x)` over the points of the box $[lo, hi]$ that `accept(x)` passes. It is the
cross-entropy method (sec. 7.3) on the unit box $z \in [0, 1]^d$, $x = lo + z\,(hi - lo)$:

- Each chain's first generation is uniform on the box, and its elites' mean and spread set the
  first proposal. Later candidates are drawn from independent normal distributions per coordinate.
- The elites of a chain are the `elites` lowest-scoring points among its new generation and its
  previous elites, so a narrow well that gave only a few draws is not dropped for a broad one.
- The update moves the mean toward the elites and keeps the spread above a floor:

$$
\mu \leftarrow (1 - a)\,\mu + a\,\bar z_{\text{elite}}, \qquad
\sigma \leftarrow \max\!\big(\sigma_{\text{floor}},\ (1 - a)\,\sigma + a\,s_{\text{elite}}\big),
$$

  with $a$ = `smoothing`. The book's algorithm 7.5 updates a full covariance without smoothing or
  a floor, and collapses onto the first elites before it reaches a rare failure.
- Two chains (by default) search independently and share the budget, so a second failure region
  is not abandoned for the first (example 4.8).
- Draws outside the box or rejected by `accept` are drawn again, up to 1000 times per candidate,
  and counted in `rejected`; they are never scored.
- `dispersion` is the mean distance from 4096 uniform probe points of the unit box to the nearest
  evaluated point, an estimate of the average dispersion of sec. 5.3.2: how much of the box the
  search left unvisited.
- The root seed must lie in $[2^{64}, 2^{128})$, or is drawn with 128 bits, and is recorded; each
  chain's stream and the probes are spawned from it. All draws are made in the calling process
  and scores are kept in order, so `workers` does not change the result.

The default of 6 elites out of 50 was chosen on the two test problems below, over 200 seeds
of the test's seed family: 6 elites reached the narrow one-dimensional well within 1e-3 on 200
of 200 seeds, 7 on 196 and 10 on 185; 5, 6 and 7 elites each found the six-dimensional ball
on 100 of 100 seeds.

### Filter falsifier

`vbar.safety.audit.falsify_filter(safety_filter, env, *, nominal=None, budget=1000, box=None,
seed=None, max_steps=None, workers=1, population=50, chains=2)` searches a filter's declared safe
set:

- It refuses exactly what `audit_filter` refuses (dynamics other than CW, imperfect navigation,
  execution error, delay, minimum impulse bit, the thrust drive, a filter that does not count
  toward certification), and flies each start with the audit's own run, at the scenario's dt.
- `accept` is the filter's `contains`, so every flown start lies in its declared set.
- The score of a start is its run's certified robustness: the lowest certified in-step bound of
  any hard constraint, after the audit's re-flying, divided by the constraint's resolution. A
  score below -1 is a bound deeper than the resolution.
- Every run shares one episode seed and one different agent seed drawn from the root seed, so the
  score depends on the start alone.
- The worst start is flown again and judged as the audit judges any run. The `verdict` comes from
  that run, never from the score: "violation", "uncertified", "infeasible", "solver_failed" or
  "none_found".

The result, a `FilterFalsification`, holds the verdict, the worst start and its re-flown true
`states`, its `RunAudit`, the score, the evaluations and rejections, the dispersion, the box and
the root seed, and the search's own record. No field claims safety.

### Assumptions and limits

- The search samples starts only; disturbances, navigation and execution errors are not searched.
- Within its budget the search can miss a failure region smaller than its spread floor allows it
  to resolve, or one far from every elite. Report the budget and dispersion with every
  "none_found".
- The score is per start, the minimum over a whole episode; two failures in one episode count
  once.

```python
import numpy as np

from vbar import ApproachEnv
from vbar.safety.audit import falsify_filter
from vbar.safety.cbf import CbfFilter
from vbar.stress import cross_entropy


def two_wells(x):
    """A broad shallow well at 0.15 and the global minimum, a narrow one, at 0.83."""
    return min((x[0] - 0.15) ** 2 + 0.05, 20.0 * (x[0] - 0.83) ** 2)


seed = (1 << 64) + 0x5DEECE66D  # a fixed root seed; scored runs draw one
found = cross_entropy(two_wells, [0.0], [1.0], budget=500, seed=seed)
print(round(float(found.x[0]), 3), found.evaluations)  # 0.83 500

near = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))  # m, m/s
ecbf = CbfFilter(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1)
report = falsify_filter(ecbf, ApproachEnv(), box=near, budget=30, seed=seed, max_steps=3)
print(report.verdict, report.score >= -1.0, report.evaluations)  # none_found True 30
print(np.asarray(report.states).shape)  # (4, 6)
```

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

The tests in `tests/test_safety_infeasible.py` check:

- `verify()` against an independent Farkas check on drawn rows, multipliers and both layouts, and
  its sign tolerance just inside and just outside $10^{-12}$; a proof on any status but
  "infeasible_certified" raises.
- One row under a box or ball limit on 300 drawn cases per layout (seed 7, those within 1e-9 of
  the boundary dropped): `project` is infeasible exactly when
  $b > a_{\max}|a|_1$ (box) or $a_{\max}|a|_2$ (ball); feasible answers match a 200-step
  bisection on the dual, and infeasible ones equal the support maximiser.
- From starts inside the keep-out closing, beyond the speed cap and inside the keep-out leaving,
  with a PD and a reckless nominal, `CbfFilter` returns only "feasible" or "infeasible_certified";
  every infeasible step's proof verifies, matches the filter's own rows at the estimate and names
  every conflicting row; two rows that each fit the box but not together are both named.
- The least-violation action equals `project`'s phase I command on the proof's rows and no command
  in the box violates them less in total (an independent linear program).
- From starts inside the declared sets, at dt 1, 2, 5, 10 and 20 s, `CbfFilter` and the braking
  backup filter have no infeasible step and no proof.

The tests in `tests/test_safety_cbf.py` and `tests/test_safety_soft.py` check:

- Keyword-only constructors, fixed names, and every reset refusal just inside and just outside
  its limit: the pole and gain rule at 0.0599 and 0.0601 1/s for dt 10 s (and 0.03 1/s at 20 s),
  the action modes, frame, observation order, pending commands, dynamics without a model error,
  an unenforced constraint with and without `ignore=`, a missing keep-out, and a default braking
  deceleration that is not positive.
- The declaration: buffer $1.2 \cdot 3 n^2 v_{\text{cap}} \Delta t/p^2$ plus $\delta a/p^2$ to 1e-9
  relative, certificate kinds "nominal" and "bounded-W", and the soft filter never counting.
- Clipping in every continuous mode and both layouts is not an intervention, and each row of the
  outcome tables gives its fields; inside the commit radius every filter only watches.
- Closed loops on the exact CW environment at dt 1, 2, 5, 10 and 20 s from starts inside each
  declared set: no certified in-step bound below -1 mm, no infeasible step, the speed under 1.1
  $v_{\text{cap}}$; riding the boundary with 0.2 m/s across the line of sight under a command
  pushing inward; a persistent inward 2e-4 m/s^2 with `execution=2e-4`.
- The bare barrier row with $p\,\Delta t = 1$ fails from every one of 100 starts, and a
  first-order row on the braking barrier fails under hold (Ames et al., sec. IV and V.B).
- Free drift on approach-vbar starts reaches the hold point with no violation; the projection
  candidate corrects less than the clipped one; certificate horizons follow the secular drift.
- Two true starts seen through one scripted sensor give identical decisions, and recovery is
  judged on the estimate.
- The soft filter's action minimises the penalised objective against 500 random commands, and
  equals the hard projection when $\alpha$ exceeds the largest multiplier.

The tests in `tests/test_safety_audit.py` check:

- Each certified filter (`CbfFilter` at poles 0.06 1/s with v_cap 0.1 m/s and at the default
  0.03 1/s, `BackupFilter` with free drift and with braking), around the PD nominal and around a
  nominal pushing at full thrust straight at the target, on 20 starts near the keep-out (52 to
  150 m, up to 0.1 m/s per axis) for 3 steps: no violation, no uncertified step, no infeasible
  decision, and every run's worst certified bound at least -1 mm.
- Every start lies inside the declared set and the box, `drawn` is the starts plus `rejected`,
  and the same root seed gives the same starts and seeds for both nominals.
- Seeds: the root seed is recorded; the 40 episode and agent seeds are distinct, below
  $2^{128}$ and the largest has at least 124 bits; a root seed drawn when none is given has more
  than 64 bits; -1, $2^{128}$ and True are refused.
- A filter that declares the `CbfFilter` set but passes every command through is caught: around
  the inward nominal it has violations deeper than 1 mm and does not pass.
- With 2 samples per step, the braking filter riding the boundary at full thrust leaves steps
  whose bound is below -1 mm while the refined minimum is not; re-flying certifies all of them,
  some only at 16 times the samples, and the audit passes. With only the 4 times grid allowed,
  some stay uncertified and the audit fails.
- Re-flying a step with the environment's own samples reproduces its certified bound and
  refined minimum to 1e-9 relative, for three commands.
- Refusals: the soft filter, "twobody" and "twobody_j2" dynamics (even with a declared model
  error), Gaussian navigation, an execution error, a filter without a declared set, a box with no
  start inside the set, and non-positive `starts`, `max_steps` and `workers` and an unknown mode.
- Outside the set, `CbfFilter` at poles 0.06 1/s from starts closing at 0.3 to 0.5 m/s at 55 to
  70 m: every run re-enters the set at a whole number of steps, there are infeasible decisions,
  each flagged decision is recovering and outside the set with named conflicting rows, and
  `passed` follows the outside rule. On stand-in decisions, a proof that verifies, one that
  fails and a missing one are recorded as True, False and None, and only verified proofs with
  no "solver_failed" decision pass.
- Two workers give the same starts, seeds, step counts and worst bounds as one.
- A "solver_failed" decision inside the set fails `passed` and is counted; actuation delay, a
  minimum impulse bit and the "thrust" drive are refused just off zero and "accel" and accepted
  at them; a root seed of $2^{64} - 1$ is refused and $2^{64}$ accepted; with no refinement grid
  left, a step the bound cannot decide is reported uncertified.

`benchmarks/filter_known_answers.py` reproduces, with seed 1903 (measured, see its output):
0 of 100 starts violate with poles 0.05 1/s at dt 1, 5 and 10 s and 100 of 100 with poles
0.1 1/s at dt 10 s (worst -0.313 m), on the bare rows without a thrust limit; under a box of
0.02 m/s^2 the barrier row with poles 0.05 1/s violates in 0 of 40 runs at dt 1, 5, 10 s and 40
of 40 at 20 s, the braking first-order row in 40 of 40 at every dt, and the braking backup filter
in none at 1, 5, 10, 20 s; at least 393 of 400 approach-vbar starts are certified passively safe
and the free-drift backup filter flies 40 of them with no violation, all reaching the hold point.

`benchmarks/filter_audit.py` runs 1000 starts for each of the four filters: inside the declared
set from the scenario's start ranges around the PD and the inward nominal, inside it near the
keep-out around the inward nominal, and outside it near the keep-out around the PD nominal, each
for the scenario's 360 steps, and asserts that every audit passes; it prints each audit's counts,
worst bound, re-entry times and root seed.

The tests in `tests/test_falsify.py` check:

- On a broad shallow well at 0.15 and a narrow global well at 0.83, the best point lies within
  1e-3 of 0.83 after 500 evaluations, for each of 20 seeds.
- In six dimensions, a ball of radius 0.03 at the bottom of a smooth bowl is found within 1000
  evaluations for 5 of 5 seeds, while 1000 uniform draws miss it (it fills about 4e-6 of the cube).
- The same seed gives the same points and scores with one or two workers and repeats exactly; the
  spread never falls below its floor; rejected draws are counted and never scored; points map from
  the unit cube onto the box; a drawn seed has 128 bits; seeds below $2^{64}$ or out of range are
  refused and $2^{64}$ itself is accepted.

The tests in `tests/test_safety_audit.py` check:

- A `CbfFilter` and a braking `BackupFilter`, each planted to pass every command through from
  starts in a slab 0.32 m wide (2% of a start box up to 1 m outside the keep-out), are caught on
  one-step episodes: verdict "violation" within 200 evaluations, with the worst start in the slab
  and in the declared set and its re-flown states.
- Short searches on a correct `CbfFilter` and a correct braking `BackupFilter` give "none_found"
  with a finite score of at least -1 and no field naming safety; the same root seed repeats the
  search.
- The verdict comes from the re-flown run: a run that scores below -1 with no violation but no
  refinement left is "uncertified", a certified infeasible decision gives "infeasible" and a
  failed solve "solver_failed".
- Draws outside the declared set are counted and never flown.
- Small seeds, a soft filter, two-body dynamics, imperfect navigation and execution error are
  refused, as in the audit.

`benchmarks/filter_audit.py` searches each certified filter with the PD and the inward nominal
near the keep-out, `--budget` runs each (2000 by default), and asserts "none_found" with a score
of at least -1; it prints each search's verdict, worst score, evaluations, dispersion and root
seed.

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

```python
import numpy as np

from vbar import ApproachConfig, ApproachEnv
from vbar.safety.cbf import BackupFilter, CbfFilter, SoftCbfFilter

env = ApproachEnv(ApproachConfig(action_mode="accel"))  # commands in m/s^2
view = env.agent_view()
x = np.array([0.0, 55.0, 0.0, 0.0, -0.05, 0.0])  # m and m/s, closing along V-bar
push = np.array([0.0, -0.05, 0.0])  # m/s^2, toward the target and beyond the box

ecbf = CbfFilter(poles=(0.05, 0.05), v_cap=0.1, gamma=0.02)
ecbf.reset(view)
result = ecbf.filter(x, push)
print(result.action.round(4), result.status, result.active)  # [ 0.    -0.001  0.   ] feasible True
print(round(ecbf.declaration.buffer_m, 4), ecbf.declaration.certificate)  # 0.0018 None

braking = BackupFilter(barrier="braking", rho_cap=2000.0, v_cap=0.3)
braking.reset(view)
result = braking.filter(x, push)
print(result.action.round(4), result.iterations, result.certificate)  # [ 0.   -0.02  0.  ] 1 nominal

soft = SoftCbfFilter(alpha=1e-4, poles=(0.05, 0.05), v_cap=0.1, gamma=0.02)
soft.reset(view)
result = soft.filter(x, push)
print(result.slack > 0.0, soft.declaration.counts_toward_certification)  # True False
```
