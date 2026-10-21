# Safety filters

Code: `vbar.safety.projections` (`halfspace`, `box`, `ball`, `second_order_cone`,
`keep_out_projection`, `KeepOutProjection`) and `vbar.safety.tightening` (`BoxError`, `BallError`,
`EllipsoidError`, `halfspace_margin`, `chance_margin`, `keep_out_margin`, `ecbf_bias`,
`first_order_bias`).

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
