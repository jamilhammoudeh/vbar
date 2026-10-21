# Safety filters

Code: `vbar.safety.projections` (`halfspace`, `box`, `ball`, `second_order_cone`,
`keep_out_projection`, `KeepOutProjection`).

A safety filter sits between an agent and the spacecraft and changes a command only when it would
lead out of a safe set. This page starts with the building blocks the filters use: closed-form
projections and the exact projection out of a keep-out sphere. The control barrier function
filters themselves follow in later sections.

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

```python
import numpy as np

from vbar.safety.projections import keep_out_projection, second_order_cone

x, t = second_order_cone([3.0, 4.0], 0.0)
print(x, t)  # [1.5 2. ] 2.5

G = np.diag([50.0, 50.0, 50.0])  # m per m/s^2 over one held step
h = np.array([-30.0, 0.0, 0.0])  # m, inside a 50 m keep-out with zero command
result = keep_out_projection(np.array([0.1, 0.0, 0.0]), G, h, 50.0)
print(result.u.round(6), result.hard_case)  # [-0.4  0.   0. ] False
```
