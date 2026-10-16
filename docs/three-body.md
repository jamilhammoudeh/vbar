# Three-body models

Code: `vbar.dynamics.cr3bp`, `vbar.dynamics.periodic_orbits` and the constant sets in
`vbar.dynamics.bodies`. Methods follow Koon, Lo, Marsden and Ross, *Dynamical Systems, the
Three-Body Problem and Space Mission Design* (cited below as Koon et al.), unless stated.

## Circular restricted three-body problem

**Frame and units.** The synodic frame turns with the two primaries: origin at their barycentre,
x from the larger primary to the smaller, y along the smaller primary's motion, z along their
angular momentum. Inside the models everything is nondimensional: the primaries' distance, their
total mass and their angular rate are 1, the primaries sit at $(-\mu, 0, 0)$ and
$(1 - \mu, 0, 0)$, and $\mu$ is the smaller primary's share of the mass. The axes have the
orientation and Coriolis sign of the Hill frame of the smaller primary's orbit, and at $\mu = 0$
the linearization about $(1, 0, 0)$ is Clohessy-Wiltshire with $n = 1$.

**Equations** (Koon et al., eq 2.3.8):

$$
\ddot x - 2\dot y = \frac{\partial\Omega}{\partial x}, \qquad
\ddot y + 2\dot x = \frac{\partial\Omega}{\partial y}, \qquad
\ddot z = \frac{\partial\Omega}{\partial z}, \qquad
\Omega = \frac{x^2 + y^2}{2} + \frac{1 - \mu}{r_1} + \frac{\mu}{r_2}
$$

with $r_1$ and $r_2$ the distances to the primaries. `acceleration`, `rhs`, `potential`,
`potential_hessian` and `jacobian` (the matrix of the variational equations,
$\begin{bmatrix} 0 & I \\ \nabla^2\Omega & 2W^T \end{bmatrix}$ with $W$ the cross-product matrix
of $\hat{\mathbf z}$) evaluate them on stacked states.

**Jacobi constant.** `jacobi_constant(state, mu)` is $C = 2\Omega - v^2$ with $\Omega$ as above,
which has no constant term. Koon et al.'s convention adds $\mu(1 - \mu)$ (their eq 2.3.14), which
puts L4 and L5 at exactly 3; with $\mu$ = 0.01215 the two differ by 0.012002. The Coriolis terms do
no work, so $C$ is conserved whatever their sign: conservation cannot detect a sign error, and the
tests check the frame against an integration in inertial axes instead.

**Lagrange points.** `lagrange_points(mu)` returns L1 to L5, shape (5, 3): L1 between the
primaries, L2 beyond the smaller, L3 beyond the larger, L4 leading the smaller by 60 deg. The
collinear points are the roots of $\partial\Omega/\partial x$ on the x axis, found by bracketed
root search; the slope runs from $-\infty$ to $+\infty$ between neighbouring singularities, so
each interval holds exactly one root (Koon et al., sec. 2.5).

**Collinear-point constants.** `collinear_coefficient(mu, point, n)` is Richardson's Legendre
coefficient $c_n$ at L1 or L2 (Koon et al., eq 6.3.1), in units of $\gamma$, the point's distance
from the smaller primary:

$$
c_n = \frac{1}{\gamma^3}\left[(\pm 1)^n \mu + (-1)^n (1 - \mu)\frac{\gamma^{n+1}}{(1 \mp \gamma)^{n+1}}\right]
$$

upper signs for L1, lower for L2. `libration_frequencies(c2)` returns the saddle rate $\lambda$,
the in-plane and out-of-plane frequencies $\omega_p$ and $\omega_v$, and the in-plane amplitude
ratio $\kappa$ (Koon et al., eqs 6.4.1 to 6.4.3):

$$
\lambda^2, \omega_p^2 = \frac{\pm(c_2 - 2) + \sqrt{9c_2^2 - 8c_2}}{2}, \qquad
\omega_v^2 = c_2, \qquad \kappa = \frac{\omega_p^2 + 1 + 2c_2}{2\omega_p}
$$

## Constant sets

`ThreeBodySystem(name, version, mu, length, period)` holds one named, versioned set: the mass
ratio, the primaries' distance (m) and their period (s). Its properties `time`, `velocity` and
`acceleration` are the SI sizes of one nondimensional unit: $T/2\pi$, $L/(T/2\pi)$ and
$L/(T/2\pi)^2$. Results in metres and seconds move by about 0.15% between published sets of the
same system, so every result should name its set.

| Set | $\mu$ | Distance (m) | Period (s) | Source |
|---|---|---|---|---|
| `EARTH_MOON` (`earth-moon` v1) | 0.01215058560962404 | 3.844e8 | 2357389.9 (27.2846 days) | mass ratio of the near-rectilinear halo orbit literature; mean distance; period from Kepler's third law with WGS 84's GM |
| `EARTH_MOON_KLMR` | 0.01215 | 3.85e8 | 2.361e6 | Koon et al., table 2.2.1 |
| `SUN_JUPITER_KLMR` | 9.537e-4 | 7.784e11 | 3.733e8 | Koon et al., table 2.2.1 |
| `SUN_EARTH_RICHARDSON` | 3.040357143e-6 | 1.495978714e11 | 3.156e7 | Richardson's Sun-(Earth+Moon) ratio, which reproduces Koon et al.'s ch. 6 constants; their sec. 6.3 distance and sec. 5.3 sidereal year (table 2.2.1's 3.147e7 s is a misprint) |

Scenarios use `EARTH_MOON`; the tests use the book's sets so that its numbers can be reproduced.

## Relative motion in the CR3BP

`CR3BPDynamics(system, target_state, rtol=1e-12)`: the target follows its own CR3BP trajectory
from the nondimensional `target_state`, and the chaser is carried as an offset from it driven by
the difference of the primaries' gravity, the linear Coriolis and centrifugal terms, and the held
acceleration:

$$
\delta\ddot{\mathbf r} = \begin{bmatrix} \delta x + 2\delta\dot y \\ \delta y - 2\delta\dot x \\ 0 \end{bmatrix}
+ \sum_{i=1,2}\Bigl(\mathbf g_i(\mathbf r + \delta\mathbf r) - \mathbf g_i(\mathbf r)\Bigr)
+ \mathbf a
$$

The gravity differences use the same cancellation-free form as the two-body model
([dynamics.md](dynamics.md)), and the offset has its own absolute tolerances (the two-body model's
1e-9 m and 1e-12 m/s, made nondimensional). Integrating two absolute states and differencing them
would close a halo orbit only to centimetres.

At the interface the model is in SI: the relative state is chaser minus target in synodic axes,
in m and m/s, with the velocity as seen in the synodic frame; `advance(accel_hill, ...)` holds the
acceleration constant in synodic axes. `n` is the synodic frame's rate, $2\pi$ over the
primaries' period: a target in the CR3BP has no mean motion of its own. With a massless secondary
and the target at rest on the unit circle the model is two-body relative motion about a circular
target, with the synodic axes as Hill axes. `target_state()` and `chaser_state()` return the
nondimensional synodic states.

Integration is DOP853 throughout, at `rtol` = 1e-12 by default for `CR3BPDynamics` and
`BicircularDynamics` and 1e-13 for `PeriodicLinearDynamics` and the periodic-orbit tools.

## Libration-point model

`LibrationPointDynamics(n, c2)`, or `LibrationPointDynamics.at_point(system, point)` for L1 or L2:
the constant-coefficient linearization at a collinear point in synodic axes and SI units (Koon et
al., eq 6.4.1):

$$
\ddot x - 2n\dot y - (1 + 2c_2)\,n^2 x = a_x, \qquad
\ddot y + 2n\dot x + (c_2 - 1)\,n^2 y = a_y, \qquad
\ddot z + c_2 n^2 z = a_z
$$

With $c_2 = 1$ it is Clohessy-Wiltshire. `libration_discrete(n, c2, dt)` gives the exact
zero-order-hold matrices in closed form. The in-plane matrix $A$ satisfies
$(A^2 - \lambda^2)(A^2 + \omega_p^2) = 0$, so

$$
e^{At} = C(A^2) + A\,S(A^2), \qquad \int_0^t e^{A\tau}d\tau = S(A^2) + A\,K(A^2)
$$

with $C(m) = \cosh(\sqrt m\,t)$, $S$ its time integral and $K$ the integral of $S$, evaluated at
the matrix $A^2$ by two-point interpolation between its eigenvalues $m = \lambda^2$ and
$m = -\omega_p^2$. Their power series are used below $|m t^2| = 1$, where the closed forms cancel.
This keeps the matrices exact on short steps and at $c_2 = 1$, where Clohessy-Wiltshire's zero
eigenvalue is defective. The motion has a saddle, e-folding in 1.5 days at Earth-Moon L1 and
2 days at L2, and two oscillations.

Limits: left out against `CR3BPDynamics` with its target at the point are the terms of second
order in the separation.

## Periodic linear model

`PeriodicLinearDynamics(system, target_state, period, rtol=1e-13, pieces=32)`: relative motion
linearized about a periodic CR3BP orbit (Koon et al., eqs 4.2.4 and 6.7.1). The offset obeys
$\delta\dot{\mathbf x} = A(t)\,\delta\mathbf x + B\,\mathbf a$ with $A(t)$ from the potential's
Hessian along the target's orbit, which repeats with the orbit's period.

**Construction.** One period is integrated once, in `pieces` stretches each started from the
identity. Each stretch stores the target, its transition matrix $\phi(t)$ and
$G(t) = \int \phi^{-1}B\,d\tau$. Within a stretch

$$
\phi(t_2, t_1) = \phi(t_2)\,\phi(t_1)^{-1}, \qquad \Gamma = \phi(t_2)\bigl(G(t_2) - G(t_1)\bigr)
$$

and a segment across stretches is the product of these; periodicity carries them to every later
period. `matrices(t0, duration)` returns the segment's matrices in SI; `monodromy` is the
transition matrix over one period. Short stretches keep the matrices small: one taken from the
start of the period grows past 1e3 near the perilune of a near-rectilinear halo orbit, and its
products would lose as many digits.

**Inverse without a solve.** The flow is Hamiltonian in the canonical momenta
$\mathbf p = \mathbf v + \hat{\mathbf z}\times\mathbf r$, so in $[\mathbf r, \mathbf v]$
coordinates $\phi^T J \phi = J$ with
$J = \begin{bmatrix} 2W & I \\ -I & 0 \end{bmatrix}$ and $\phi^{-1} = J^{-1}\phi^T J$
(`transition_inverse`). The standard symplectic form does not hold in these coordinates.

**Accuracy.** On the 9:2 near-rectilinear halo orbit, segments match a direct integration of the
linear equations to 1e-13 of the state, and to about 1e-8 through perilune. `target_state` must
start a periodic orbit of nondimensional `period` to that precision; integrated at
`rtol` = 1e-13 that orbit closes to 1e-10 per period, about 5 cm.

Limits: left out against `CR3BPDynamics` are the terms of second order in the separation.

## Bicircular model

`BicircularDynamics(sun, target_state, sun_angle=0.0, rtol=1e-12)` adds the Sun to the CR3BP
(Koon et al., eqs 5.3.1 to 5.3.7). The Sun moves on a circle about the Earth-Moon barycentre in the
primaries' plane, turning clockwise in the synodic frame from `sun_angle` (rad) at reset. The
barycentre's own acceleration toward the Sun enters as a uniform term:

$$
\ddot{\mathbf r} = \mathbf a_{\text{CR3BP}} - m_S\frac{\mathbf r - \mathbf r_S}{|\mathbf r - \mathbf r_S|^3}
- m_S\frac{\mathbf r_S}{a_S^3}, \qquad
\mathbf r_S = a_S\,(\cos\theta_S, \sin\theta_S, 0), \qquad \theta_S = \theta_0 - \omega_S t
$$

`BicircularSun(name, system, mass, distance, rate)` holds the Sun's constants in the system's
units. `SUN_FOR_EARTH_MOON_KLMR` has $m_S$ = 328900.54, $a_S$ = 388.81114 and
$\omega_S$ = 0.925195985520347, which is $1 - \sqrt{(m_S + 1)/a_S^3}$ (Koon et al.,
eqs 5.3.4 to 5.3.7), with `EARTH_MOON_KLMR`. The chaser feels the difference of the Sun's pull
between it and the target, its tidal acceleration.

The model is periodic in time, so it has no Jacobi integral: with the Sun's potential
$\Omega_S$ added, $2(\Omega + \Omega_S) - v^2$ changes along a path by twice the integral of
$\partial\Omega_S/\partial t$ at fixed position, minus twice the work of the held acceleration.

Limits: the primaries keep their circles, so the model does not solve Newton's equations for four
bodies; the Sun stays in the Earth-Moon plane, leaving out the 5.1 deg tilt of the Moon's orbit to
the ecliptic. It is a fidelity step, not an ephemeris model.

## Periodic orbits

Code: `vbar.dynamics.periodic_orbits`, on nondimensional synodic states.

**Richardson seed.** `richardson_coefficients(mu, point)` computes Richardson's third-order halo
coefficients at L1 or L2 (Richardson, "Analytic Construction of Periodic Orbits about the
Collinear Points", *Celestial Mechanics* 22, 1980; Koon et al., sec. 6.6). Two of them are
misprinted in Koon et al.'s sec. 6.6: $b_{22} = +3c_3\omega_p/d_1$, and $a_{32}$ contains
$4c_3(\kappa a_{24} - b_{22})$. With the printed sign of $b_{22}$ the book's own Sun-Earth values
are not reproduced. `richardson_halo(mu, point, az, branch=1)` returns the seed state at the x-z
plane crossing and the third-order period. The in-plane amplitude follows from
$l_1 A_x^2 + l_2 A_z^2 + \Delta = 0$, so `az` (nondimensional) must be large enough for a halo to
exist. `branch` is Richardson's $m$, 1 or 3, the two mirror-image families.

**Correctors.** `half_period_crossing(state, mu)` integrates the state and its transition matrix
to the next crossing of the x-z plane, located to rounding. The correctors drive the velocity
components normal to that plane to zero by Newton's method, with the crossing-time terms that
Koon et al.'s sec. 6.7 leaves out. The crossing time moves with the start,
$\delta t = -\phi_{2\cdot}\,\delta\mathbf x_0 / \dot y$, so

$$
\delta\dot x = \Bigl(\phi_{4\cdot} - \frac{\ddot x}{\dot y}\,\phi_{2\cdot}\Bigr)\delta\mathbf x_0, \qquad
\delta\dot z = \Bigl(\phi_{6\cdot} - \frac{\ddot z}{\dot y}\,\phi_{2\cdot}\Bigr)\delta\mathbf x_0
$$

`correct_lyapunov(state, mu, tol=1e-12)` corrects $\dot y_0$ of a planar Lyapunov orbit with $x_0$
held (Koon et al., sec. 4.4). `correct_halo(state, mu, fixed="z", tol=1e-12)` corrects a halo
orbit holding $z_0$ or $x_0$. Both return a `PeriodicOrbit(state, period, jacobi, iterations)`.
Holding $z_0$ fails where the family turns back in $z_0$, as it does on the way from small
Earth-Moon L2 halos to the near-rectilinear ones: past the family's largest $z_0$, about 0.075,
the corrector lands on an unrelated orbit of about 27 days instead of failing.

**Continuation.** `continue_halo(start, mu, step, count, direction=(0, 1, 0))` follows a halo
family by pseudo-arclength continuation. The unknowns $(x_0, z_0, \dot y_0)$ satisfy two crossing
conditions, so the family is a curve; each step moves `step` along its tangent (the null vector of
the crossing Jacobian) and corrects on the plane normal to it, which passes turning points in
$x_0$ or $z_0$. The first step goes to the side of `direction`, by default growing $z_0$. A step
whose corrector moves more than half a step off the prediction is rejected as too long for the
family's curvature. `family_jump(previous, current, step, slope=None)` flags a converged orbit
whose start state moved more than twice the step, or whose period changed by more than three
times `slope` (the period's rate of change on the previous step) times the step plus 1% of the
period, or by a fifth without a slope; `continue_halo` raises `FamilyJump` on it.
`halo_by_period(start, mu, period, step=0.02, max_step=0.1, direction=(0, 1, 0))` continues the
same way, doubling the step up to `max_step` while the corrector needs four iterations or fewer
and halving it when a step fails or lands off the family, until the period is passed. It then
finds the step that lands on the period, to 1e-12, by false position (Illinois variant) between
the two bracketing members.

**Stability.** `monodromy(state, period, mu)` is the transition matrix over one period.
`stability_indices(M)` returns $(\lambda + 1/\lambda)/2$ of its eigenvalues, which come in pairs
$\lambda, 1/\lambda$ plus a pair at 1 (Koon et al., sec. 4.3); $|\nu| > 1$ is unstable.
`floquet_modes(state, period, mu, times)` returns the eigenvalues and the periodic Floquet modes
$\bar{\mathbf e}_i(t) = e^{-(t/T)\log\lambda_i}\,\phi(t, 0)\,\mathbf e_i(0)$ at `times`, as columns
(Koon et al., sec. 8.7). The pair at 1 is a Jordan block, which rounding splits by about the
square root of the machine epsilon, so its modes are poorly determined.

**Names.** Halo orbits are named here by libration point and by the sign of $z$ where they are
farthest from the smaller primary. "Northern" and "southern" swap meaning between Sun-Earth L1
and Earth-Moon L2, where $c_3$ changes sign: at Earth-Moon L2 the $m = 1$ branch reaches farthest
below the plane.

**Reference orbit.** The 9:2 synodic-resonant L2 near-rectilinear halo orbit with its apolune below
the plane, in Koon et al.'s Earth-Moon set, crosses the x-z plane on the perilune side at

$$
\mathbf x_0 = (0.9873849199042153,\ 0,\ 0.008369210996156513,\ 0,\ 1.6744665364075588,\ 0)
$$

with nondimensional period 1.5088887454509277 (6.5624 days, 2/9 of the 29.530589-day synodic
month), perilune radius 3,227 km, apolune radius 71,274 km and Jacobi constant 3.0466675644. Its
monodromy eigenvalues are -2.176942, -0.45936, $0.684704 \pm 0.728821i$ and a pair at 1, a
stability index of -1.318: mildly unstable, with a sign flip each revolution. Closure needs the
full double-precision state.

## Validation

Tests: `tests/test_cr3bp.py`, `tests/test_periodic_orbits.py` and `tests/test_bicircular.py`;
published values in [validation.md](validation.md). Values described as computed come from a
separate implementation of Koon et al.'s methods with their constants.

- Constant sets reproduce Koon et al.'s table 2.2.1 velocity units, and the Earth-Moon set's period
  is its Kepler value.
- Earth-Moon and Sun-Jupiter Lagrange points, Jacobi constants in both conventions, L1 to L3
  energies (Koon et al., secs. 2.5 and 4.4), the collinear constants at Earth-Moon L1 and L2 and
  Richardson's Sun-Earth L1 constants (secs. 6.3 and 6.4) match their printed or computed digits;
  L4 loses linear stability at Routh's mass ratio; at $\mu = 0$ the Jacobian equals
  Clohessy-Wiltshire's exactly.
- The synodic equations match an integration in inertial axes with the primaries on circles to
  1e-11; a model with the Coriolis sign flipped conserves the Jacobi constant equally well and
  is caught by this check; mirrored states run backward onto mirrored states.
- The relative model matches differenced absolute propagations over a day to 2e-4 m and
  1e-9 m/s, the accuracy of the absolute reference; it conserves the Jacobi constant to 1e-12 over
  two days, and a held acceleration changes it by $-2\,\mathbf a\cdot\Delta\mathbf r$ to 1e-9 of
  the change;
  with a massless secondary it is two-body relative motion to 1e-6 m.
- The libration-point model at $c_2 = 1$ equals Clohessy-Wiltshire to rounding from 1 ms to
  28 days, matches the matrix exponential of the block system to 1e-14, its error against the
  CR3BP grows as the square of the separation with and without thrust, and its saddle e-folds in
  1.5 and 2 days at Earth-Moon L1 and L2.
- The periodic linear model matches integration of the linear equations to 1e-12, and to 2e-8
  through perilune; its matrices over two periods are the monodromy matrix squared; segments run
  backward; its error against the CR3BP grows as the square of the separation with and without
  thrust; CR3BP transition matrices keep the rotating symplectic form $J$ and fail the standard
  one.
- Richardson's coefficients reproduce Koon et al.'s printed Sun-Earth L1 values to 1e-6 with the
  corrected $b_{22}$; the ISEE-3 halo's amplitudes and 177.72-day period (sec. 6.6) are
  reproduced; the planar Lyapunov corrector and the halo corrector from Richardson's seed reach the
  computed orbits; holding $z_0$ jumps family and `family_jump` detects it; pseudo-arclength
  continuation passes the $z_0$ turning point; `halo_by_period` reaches the 9:2 near-rectilinear
  halo orbit to 1e-10; its monodromy eigenvalues, stability index and periodic Floquet modes match.
- The bicircular Sun rate is consistent with its mass and distance; a massless Sun leaves the
  CR3BP bit for bit; the Jacobi constant changes by the Sun's time dependence and the thrust work
  to 1e-10; the equations match an inertial-frame integration to 1e-10; relative motion matches
  differenced absolute propagations.

## Examples

A chaser 1 km from a target on the near-rectilinear halo orbit, one day of coasting in the
nonlinear and the periodic linear model:

```python
import numpy as np

from vbar.dynamics.bodies import EARTH_MOON_KLMR
from vbar.dynamics.cr3bp import CR3BPDynamics, PeriodicLinearDynamics

nrho = [0.9873849199042153, 0.0, 0.008369210996156513, 0.0, 1.6744665364075588, 0.0]
nonlinear = CR3BPDynamics(EARTH_MOON_KLMR, nrho)
linear = PeriodicLinearDynamics(EARTH_MOON_KLMR, nrho, period=1.5088887454509277)
for model in (nonlinear, linear):
    model.reset([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
exact, _ = nonlinear.advance(np.zeros(3), 86400.0)
approx, _ = linear.advance(np.zeros(3), 86400.0)
print("separation after a day (m):", np.linalg.norm(exact[:3]))
print("linear model error (m):", np.linalg.norm(approx[:3] - exact[:3]))
```

A small Earth-Moon L2 halo orbit from Richardson's seed, corrected, and its stability:

```python
import numpy as np

from vbar.dynamics import periodic_orbits as po
from vbar.dynamics.bodies import EARTH_MOON_KLMR

mu = EARTH_MOON_KLMR.mu
seed, _ = po.richardson_halo(mu, point=2, az=8000e3 / EARTH_MOON_KLMR.length)
orbit = po.correct_halo(seed, mu, fixed="z")
days = orbit.period * EARTH_MOON_KLMR.time / 86400
print(f"period {days:.2f} days, Jacobi constant {orbit.jacobi:.6f}")
print(
    "stability indices:",
    np.round(po.stability_indices(po.monodromy(orbit.state, orbit.period, mu)).real, 3),
)
```
