# Primer-vector certificates

Code: `certify_impulsive`, `PrimerCertificate`, `indirect_optimum` and `hamiltonian` in
`vbar.oracles.certificate`, exported from `vbar.oracles`. Hill frame, SI units.

`certify_impulsive` decides whether an impulsive plan is fuel-optimal on Clohessy-Wiltshire
(CW) motion, in the norm of the actuator, by checking Lawden's primer-vector conditions over the
whole transfer, not only at sample points. A certified plan's cost equals a lower bound that
holds for any acceleration history, so it can serve as a known optimum. `indirect_optimum` solves
the finite-thrust problem by the switching control of the primer, which cross-checks the
step-grid LPs of the [fuel oracle](oracles.md), and `hamiltonian` evaluates the Hamiltonian that
must stay constant along its solution.

## How it is built

**Primer and the impulsive bound.** For a terminal multiplier $\nu \in \mathbb R^6$ (position
part in 1/s, velocity part dimensionless) the primer is

$$
\mathbf p(t) = B^T \Phi(t_f - t)^T \nu, \qquad B = [0;\, I],
$$

with $\Phi$ the CW transition matrix (closed form in the [fuel oracles](oracles.md) page). For a
plan with velocity changes $\Delta\mathbf v_j$ at times $t_j$ that moves the start $\mathbf x_0$
to the target $\mathbf x_g$ at $t_f$,

$$
D(\nu) = \nu^T\big(\mathbf x_g - \Phi(t_f)\,\mathbf x_0\big)
= \sum_j \mathbf p(t_j)^T \Delta\mathbf v_j
\le \sup_t |\mathbf p(t)|_* \sum_j |\Delta\mathbf v_j|,
$$

by Hölder's inequality, where $|\cdot|$ is the actuator norm and $|\cdot|_*$ its dual: the 1-norm
cost and largest component $\max_i |p_i|$ for box thrusters (`norm="l1"`), the 2-norm cost and
the 2-norm of $\mathbf p$ for one engine (`norm="l2"`). So every plan costs at least
$D(\nu) / \sup_t |\mathbf p|_*$, for any $\nu$ (D. F. Lawden, *Optimal Trajectories for Space
Navigation*, 1963). Equality holds when $\mathbf p(t_j)$ is the burn direction at every burn
($\operatorname{sign}\Delta v_{j,i}$ on each burning axis for a box, $\Delta\mathbf v_j /
|\Delta\mathbf v_j|_2$ for one engine) and $|\mathbf p|_* \le 1$ throughout. These are Lawden's
conditions; on linear dynamics they are sufficient for optimality (Longuski, Guzman and Prussing,
*Optimal Control with Aerospace Applications*, 2014, section 10.5).

**Fitting the multiplier.** The burn directions give linear equations in $\nu$, solved by least
squares; a largest residual above $10^{-9}$ means no multiplier puts the primer along every burn
(`reason="direction"`). When the equations leave $\nu$ free, the choice decides the peak, and
the least-squares point need not keep it at 1 even for an optimal plan. If $|\mathbf p|_* \le 1$
with equality at an interior burn, $|\mathbf p|_*$ has a maximum there, so the rate of the
burning components (box) or of $\mathbf u^T \mathbf p$ (one engine, $\mathbf u$ the burn
direction) is zero; at a burn on $t = 0$ or $t_f$ it must point into the interval. These rows
are added, and a linear program over the remaining freedom maximizes the room $\varepsilon$ in

$$
\mathbf d^T \mathbf p(t_m) + \varepsilon\, w(t_m) \le 1, \qquad
w(t) = \min\big(1, (n \operatorname{dist}(t, \{t_j\}))^2\big),
$$

over 257 grid times and the dual directions $\mathbf d$ ($\pm\mathbf e_i$ for a box, 26 unit
directions for one engine), solved by HiGHS at feasibility tolerance $10^{-10}$. Local maxima of
the solution's primer above 1, found on 2049 samples and refined by Newton steps, join the rows
until none is left, at most 20 rounds (an exchange method for the semi-infinite program: Hettich
and Kortanek, *Semi-infinite programming*, SIAM Review, 1993). The room keeps the primer away
from 1 between grid times, where a plain minimum-peak LP returns a vertex that touches 1 at
grid times and exceeds it between them. If no certifying multiplier exists,
the certificate keeps the multiplier of least peak among those that fit the burns: the
least-squares point or the solution of the LP that minimizes $z$ subject to
$\mathbf d^T\mathbf p(t_m) \le z$ with the same exchange, whichever peaks lower.

**Certified supremum.** On each cell $[a, b]$ of width $h$ the functions $g = \pm p_i$ (box) or
$g = |\mathbf p|_2^2$ (one engine) are bounded by Taylor's theorem of second order. The third
derivatives of the primer are sinusoids in $n\tau$ with amplitudes

$$
|p_x'''| \le n^2 A, \quad |p_y'''| \le 2n^2 A, \quad |p_z'''| \le n^2 \sqrt{\nu_z^2 + n^2\nu_{vz}^2},
\qquad A = \sqrt{(\nu_x - 2n\nu_{vy})^2 + (2\nu_y + n\nu_{vx})^2},
$$

and $|(|\mathbf p|^2)'''| \le 2(3|\mathbf p'||\mathbf p''| + |\mathbf p||\mathbf p'''|)$ with each
factor bounded on the cell. With $|g'''| \le K_3$, $g'' \le K = \tfrac12(g''(a) + g''(b)) +
\tfrac12 K_3 h$ on the cell, so

$$
g(a + s) \le g(a) + g'(a)\,s + \tfrac12 K s^2, \qquad g(b - s) \le g(b) - g'(b)\,s + \tfrac12 K s^2,
$$

and the cell's top is the smaller of the two maxima over $s \in [0, h]$. The value, slope and
curvature come from $\mathbf p^{(k)}(t) = (-1)^k (A^k B)^T \Phi(t_f - t)^T \nu$, and the rounding
of each, 64 machine epsilons of the summed magnitudes, is added. Cells start at 256 plus the burn
times and are bisected while their top exceeds the largest sampled value by more than
$10^{-10}$ of the larger of 1 and that value. A first-order bound leaves a slack $L h / 2$ at a
burn, where the primer is 1, and would need cells near $10^{-9}/L$ to certify; the second-order
bound returns 1 to rounding wherever the primer is stationary with negative curvature or slopes
away from the burn. The plan is certified when it reaches its target, the burns are aligned and
the bound, `primer_max`, is at most $1 + 10^{-9}$; then `lower` $= D(\nu)/$`primer_max` matches
the cost to about $10^{-9}$.

**Coast Hamiltonian.** On a coast $H = -\nu^T \Phi(t_f - t) A\, \mathbf x(t)$ is constant, and at
a burn it steps by $\mathbf p'(t_j)^T \Delta\mathbf v_j$, which vanishes when the primer is
stationary along the burn. `hamiltonian_spread` is its spread over the coasts relative to the
largest product $|\nu_i \Phi_{ij} A_{jk} x_k|$ summed over the indices.

**Indirect finite-thrust optimum.** With $|a_i| \le a$ per axis (box) or $|\mathbf a|_2 \le a$
(one engine), the fuel-optimal control thrusts $a\operatorname{sign}(p_i)$ on each axis where
$|p_i| > 1$, or $a\,\mathbf p/|\mathbf p|_2$ where $|\mathbf p|_2 > 1$, and coasts elsewhere
(Longuski, Guzman and Prussing, chapters 4 and 10). The dual function

$$
\varphi(\nu) = D(\nu) - a \int_0^{t_f} \max\big(0, |\mathbf p(t)|_* - 1\big)\,dt
$$

of `dual_bound` is concave, and its gradient is $\mathbf d$ minus the final-state reach of that
control, so its maximizer lands on the target and then the control's cost equals
$\varphi(\nu)$. `indirect_optimum` maximizes it by Levenberg-Marquardt steps from `nu0`, in
multipliers scaled by $(n, n, n, 1, 1, 1)$, with the analytic Hessian: the moving switching
times for a box, plus 24-point Gauss-Legendre quadrature on pieces of at most $n\,\Delta t =
0.25$ for one engine. Switching times are located by bisecting cells until each function is
certified below 1, above 1 or monotone on each, then by safeguarded Newton steps. The box reach
over an arc is exact, $\Phi(t_f - t_1)\,\Gamma(t_1 - t_0)\,\mathbf e_i$ with the zero-order-hold
input matrix $\Gamma$. The solve stops when the miss, velocities over $n$, is below $10^{-12}$ of
$|\mathbf d|$, and raises `RuntimeError` if it is still above $10^{-9}$.

**Hamiltonian.** `hamiltonian(n, nu, tf, t, x, a, norm=...)` returns
$H = |\mathbf a| - \nu^T \Phi(t_f - t)(A\mathbf x + B\mathbf a)$, the costate being
$-\Phi(t_f - t)^T \nu$. Between switches the time derivatives of the two terms cancel, and at a
switch the step in $|\mathbf a|$ matches the step in $\nu^T\Phi B\mathbf a$ because
$|\mathbf p|_* = 1$ there, so $H$ is constant along the optimum.

## Parameters and defaults

| Parameter | Default | Reason |
|---|---|---|
| `norm` | required | `"l1"` for box thrusters, `"l2"` for one engine; the dual norm must belong to the actuator |
| `tf` | required | Final time; burn times must lie in $[0, t_f]$ and increase strictly |
| `accel` | required | Acceleration limit of `indirect_optimum` in the actuator norm |
| `nu0` | required | Starting multiplier of `indirect_optimum`, such as the equality multipliers of a step-grid LP |
| Certificate tolerance | $10^{-9}$ | On the primer peak above 1, the direction residual and the miss relative to the size of the final-state terms (velocities over $n$) |
| Refinement gap | $10^{-10}$ | A tenth of the tolerance, so a tight bound of an optimal primer stays within it |
| Starting cells | 256 plus burn times | Cheap to evaluate; bisection refines only the cells whose bound is loose |
| Exchange | 2049 samples, 20 rounds | Eight times finer than the LP grid, so maxima between grid times are found |

## Assumptions and limits

- Exact CW motion: a circular target orbit and no other forcing. Sufficiency rests on linear
  dynamics, so a certificate is a theorem on the `"cw"` rung only.
- Impulsive model: velocity changes are instantaneous. `lower` holds for every acceleration
  history, so a finite-thrust run below it has a bug; a finite-thrust plan is not certified by
  `certify_impulsive`.
- Fixed final time and a point target: the plan must reach `target` at `tf`. Capture balls,
  dwell, free arrival times and path constraints such as keep-out are not part of the check.
- `reason="primer_max"` proves the plan is not optimal when the burns fix the multiplier (six
  independent direction equations). When they leave it free, it means that no certifying
  multiplier was found by the LP and its exchange rounds.
- `primer_max` is a certified upper bound of $\sup |\mathbf p|_*$ for the reported `nu`. It
  exceeds the largest sampled value by at most $10^{-10}$ of the larger of 1 and that value
  (of the square, for one engine), unless 80 refinement rounds or cells of 16 machine epsilons
  of $t_f$ are reached first; it is then still an upper bound, only looser.
- `indirect_optimum` assumes no singular arc, where $|\mathbf p|_*$ stays at 1 over an interval;
  there the switching control is not defined. When the target is out of reach at the given
  acceleration the dual function is unbounded. In either case the miss does not converge and the
  solve raises `RuntimeError` instead of returning a control that misses.
- `hamiltonian` is constant only along an optimal switching control.

## Validation

Tests: `tests/test_oracle_certificate.py`. Values marked as independent were computed with
`cw_discrete`, HiGHS, Brent root finding and quadrature in a separate implementation.

- Every certificate in the tests is checked against a 20001-point sample of the primer refined by
  a bounded search at each local maximum: `primer_max` is never below it and, when the peak is at
  most $1 + 10^{-6}$, within $10^{-9}$ above it; `lower` equals $\max(0, D(\nu)/$`primer_max`$)$
  to $10^{-10}$, and `cost` is the sum of 1-norms or 2-norms of the impulses to $10^{-12}$.
- Fehse, *Automated Rendezvous and Docking of Spacecraft*, 2003, eqs 3.29 to 3.32 and 3.37: from
  500 m behind the target to the target point on a 500 km circular orbit (WGS 84), one orbit
  with two tangential burns of $n\,\Delta y/(6\pi)$ or half an orbit with two radial burns of
  $n\,\Delta y/4$. Both certify as box-optimal, with `primer_max` in $[1 - 10^{-12}, 1 + 10^{-9}]$
  and cost $n\,\Delta y/(3\pi)$ and $n\,\Delta y/2$ to $10^{-12}$; the radial plan also certifies
  for one engine. A second radial burn 1% too large gives `reason="miss"`.
- Alfriend, Vadali, Gurfil, How and Breger, *Spacecraft Formation Flying*, 2010, example 5.2, in
  units with $n = 1$: the two-impulse plan at $n t_f = 4.6404$ costs 0.0464255 in the 1-norm and
  0.0357202 in the 2-norm (independent). Its six burning components fix the multiplier, and the
  primer peaks at 5.5929 (box) and 1.9217 (one engine) inside the transfer, so it fails with
  `reason="primer_max"` and a `lower` below 0.0377879, the cost of a 2000-interval grid plan
  (independent). The four-impulse 1-norm optimum over $n T = 5$ (independent) certifies at cost
  0.0375601 to $10^{-5}$, with `hamiltonian_spread` at most $10^{-8}$. A 40-step fuel-oracle plan
  for one engine costs at most 0.96 of the two-impulse plan.
- Dual norm: burns along $(\cos 30°, 0, \sin 30°)$ and $(\cos 150°, 0, \sin 150°)$ at 0 and 1.8
  from rest ($n = 1$) give a unique primer whose 2-norm reaches 1.1606 while no component exceeds
  0.8661 (independent); the one-engine certificate rejects the plan, and its primer equals both
  burn directions to $10^{-9}$.
- A peak between samples: two $+z$ burns around $t_m = 0.618\ldots$ whose primer
  $A\cos(t - t_m)$, $A = 1 + 10^{-6}$, exceeds 1 only on the $2.8\times10^{-3}$ between them,
  where the largest of 1001 uniform samples is $1 + 9.994\times10^{-7}$ (independent). Both
  norms report `primer_max` within $[A - 10^{-12}, A + 10^{-9}]$ and `reason="primer_max"`.
- A burn against the primer: from $v_z = -10^{-3}$ through the origin, burns of $+2\times10^{-3}$
  at 0 and $+10^{-3}$ at $\pi$ ($n = 1$) cost three times the one-burn optimum; no primer is $+1$
  at both, so both norms report `reason="direction"`, with `lower` in $[0, 10^{-3}]$.
- A computed case, not a book example: half an orbit of a 6778 km circle from
  $(-200, -1500, 50)$ m to $(0, -100, 0)$ m at rest with a $2\times10^{-3}$ m/s^2 limit, whose
  step-grid LP multipliers give lower bounds by LP duality (Boyd and Vandenberghe, *Convex
  Optimization*, 2004, sections 4.3 and 5.1.3). From the 160-step LP multipliers, the box
  indirect optimum is 0.8947125 m/s to $10^{-7}$ (independent), below the 160-step and 640-step
  LP optima 0.8949323 and 0.8947293 by $2.5\times10^{-4}$ and $1.9\times10^{-5}$ of them
  (independent); its switching control, flown arc by arc,
  lands within $10^{-6}$ of the distance to cover, its cost is $a$ times the summed burn time to
  $10^{-9}$, and `dual_bound` at 20000 cells with its multiplier lies within $2\times10^{-3}$
  below it. For one engine the optimum is 0.8289721 m/s to $10^{-6}$ (independent), below the
  0.8324776 m/s of a 40-step plan checked against the thrust ball, with the same miss and cost
  checks by adaptive quadrature and `dual_bound` within $3\times10^{-3}$.
- Along the box optimum, `hamiltonian` matches its three terms summed independently and spreads
  by at most $10^{-8}$ of the largest term; with the costate's sign flipped the spread is 1.77
  times it (independent).
- Unknown norms, burn times outside $[0, t_f]$ and a zero acceleration limit raise `ValueError`.
- A plan of finite numbers whose impulse norm, cost, propagated final state or terminal miss
  overflows raises `ValueError` rather than comparing an infinite miss with an infinite size, as
  does `hamiltonian` with $n \le 0$ or a non-finite $t_f$, `nu`, `t`, `x` or `a`.

## Example

The Fehse one-orbit tangential transfer certified as box-optimal, a plan that stops short, and the
finite-thrust out-of-plane transfer of the [fuel oracles](oracles.md) page solved indirectly:

```python
import math

import numpy as np

from vbar.dynamics import R_EARTH, mean_motion
from vbar.oracles import certify_impulsive, indirect_optimum

n = mean_motion(R_EARTH + 500e3)
tf = 2 * math.pi / n
dv = n * 500.0 / (6 * math.pi)
start = np.array([0.0, -500.0, 0.0, 0.0, 0.0, 0.0])
plan = [(0.0, np.array([0.0, -dv, 0.0])), (tf, np.array([0.0, dv, 0.0]))]

cert = certify_impulsive(n, start, plan, np.zeros(6), tf=tf, norm="l1")
assert cert.optimal and cert.primer_max <= 1 + 1e-9
assert math.isclose(cert.cost, n * 500.0 / (3 * math.pi), rel_tol=1e-12)
assert cert.cost / (1 + 1e-9) <= cert.lower <= cert.cost

short = [plan[0], (tf, plan[1][1] / 2)]
assert certify_impulsive(n, start, short, np.zeros(6), tf=tf, norm="l1").reason == "miss"

half, z0, accel = math.pi / n, 500.0, 1e-3
out_of_plane = np.array([0.0, 0.0, z0, 0.0, 0.0, 0.0])
cost, nu = indirect_optimum(
    n, out_of_plane, np.zeros(6), half, accel=accel, norm="l1", nu0=[0.0, 0.0, n, 0.0, 0.0, 0.0]
)
delta = math.asin(n**2 * z0 / (2 * accel))
assert math.isclose(cost, 2 * accel * delta / n, rel_tol=1e-9)
assert math.isclose(nu[2], n / math.cos(delta), rel_tol=1e-9)
```
