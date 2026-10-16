# Relative-motion models

Code: `vbar.dynamics.relative`, and `TwoBodyDynamics` in `vbar.dynamics.models`. All models here
hold the chaser's state relative to the target in the target's Hill frame
([frames.md](frames.md)) and satisfy the `Dynamics` interface ([dynamics.md](dynamics.md)).

They sit between Clohessy-Wiltshire and the nonlinear two-body model. What each leaves out sets
how its error grows with the separation $\rho$ (Alfriend et al., *Spacecraft Formation Flying*,
sec. 8.3):

| Left out | Error grows as | Models that leave it out |
|---|---|---|
| Two-body nonlinearity | $\rho^2$ | `cw`, `ya`, `cw_curvilinear`, `linear_twobody`, `linear_j2` |
| J2 | $\rho$ | `cw`, `ya`, `cw_curvilinear`, `linear_twobody` |
| The target's eccentricity $e$ | $e\rho$ | `cw`, `cw_curvilinear` |

## Elliptic and phased targets in the two-body model

`TwoBodyDynamics(elements=[a, e, i, raan, argp, M])` (m, rad) starts the target from these
classical elements instead of the ascending node of a circular orbit; they replace `altitude`
(which becomes $a - R$) and `inclination`. With `j2=True` they are first-order mean elements and
the target starts from the osculating elements of the mean-osculating map
([dynamics.md](dynamics.md)). `n` is the Keplerian mean motion of $a$. The output is in the
target's Hill frame with its exact rotation rate, which on an elliptic orbit is
$h/r^2$ and varies around the orbit.

## Yamanaka-Ankersen

`YamanakaAnkersenDynamics(semi_major_axis, eccentricity, mean_anomaly=0.0, body="earth")`:
relative motion linearized about an elliptic Keplerian target, exact for the linearized problem
at any $e < 1$ and equal to Clohessy-Wiltshire at $e = 0$ (Yamanaka and Ankersen, "New State
Transition Matrix for Relative Motion on an Arbitrary Elliptical Orbit", *Journal of Guidance,
Control, and Dynamics* 25(1), 2002). `mean_anomaly` is the target's at reset; the orbit's
orientation does not enter. `n` is the mean motion.

**Coordinates.** The paper works in $[x, z, x', z']$ with $x$ along-track and $z$ toward the
central body (the ISS-style LVLH axes of [frames.md](frames.md)), each coordinate scaled by
$\rho_f = 1 + e\cos f$ and differentiated with respect to the true anomaly $f$:

$$
\tilde x = \rho_f\,x, \qquad \tilde x' = -e\sin f\,x + \frac{\dot x}{k^2\rho_f}, \qquad
k^2 = \frac{\mu^2}{h^3}, \qquad \dot f = k^2\rho_f^2
$$

**Free motion.** With $s = \rho_f\sin f$, $c = \rho_f\cos f$, $s' = \cos f + e\cos 2f$,
$c' = -(\sin f + e\sin 2f)$ and $J = k^2 (t - t_0)$, the in-plane fundamental matrix is

$$
\Psi(f, J) = \begin{bmatrix}
1 & -c\,(1 + 1/\rho_f) & s\,(1 + 1/\rho_f) & 3\rho_f^2 J \\
0 & s & c & 2 - 3esJ \\
0 & 2s & 2c - e & 3\,(1 - 2esJ) \\
0 & s' & c' & -3e\,(s'J + s/\rho_f^2)
\end{bmatrix}
$$

and the scaled out-of-plane coordinate obeys $\tilde y'' + \tilde y = 0$. The inverse of $\Psi$ is
in closed form: at $J = 0$ it is the paper's pseudo-initial-value matrix, and a shift in $J$ is a
nilpotent update whose inverse is exact. The transition matrix from $t_0$ is

$$
\Phi = T(f)^{-1}\,\Psi\bigl(f, k^2\Delta t\bigr)\,\Psi(f_0, 0)^{-1}\,T(f_0)
$$

with $T$ the map from Hill states to scaled states.

**Forced motion.** The response to an acceleration held in Hill axes has no closed form:

$$
\Gamma = T(f)^{-1}\,\Psi\bigl(f, k^2\Delta t\bigr)\int_0^{\Delta t}
\Psi\bigl(f(\tau), k^2\tau\bigr)^{-1}\,T\bigl(f(\tau)\bigr)\,B\,d\tau
$$

is evaluated by 12-point Gauss-Legendre quadrature on pieces of at most 1/32 orbit, shortened on
eccentric orbits by the peak rate of the true anomaly over its mean,
$(1 + e)^2/(1 - e^2)^{3/2}$. All sample times of a segment share one pass of the quadrature.

**Bounded starts.** `make_bounded(rel_state)` changes the along-track velocity so that the
coefficient of the secular ($J$) solution vanishes and the motion repeats every orbit; at $e = 0$
this is $v_y = -2nx$.

Limits: a Keplerian target (no J2 or drag); linear, so the error against the two-body model grows
as $\rho^2$, about $\rho^2/a$ per orbit.

## Curvilinear Clohessy-Wiltshire

`CurvilinearCWDynamics(radius, body="earth")`: the Clohessy-Wiltshire equations in curvilinear
coordinates about a circular target orbit of `radius` (Alfriend et al., eqs 5.79 to 5.88):

$$
\bigl[\,r - r_0,\ r_0\theta,\ r_0\phi,\ \dot r,\ r_0\dot\theta,\ r_0\dot\phi\,\bigr]
$$

with $\theta$ the chaser's angle from the target in the target's orbit plane and $\phi$ its
latitude above that plane. The equations hold to first order in these as they do in Cartesian Hill
coordinates, but the coordinates follow the orbit's curvature. States in and out are Cartesian
Hill states; `hill_to_curvilinear` and `curvilinear_to_hill` map them exactly and without
cancellation (for example $r - r_0 = (2r_0 x + x^2 + y^2 + z^2)/(r + r_0)$). Propagation uses the
exact Clohessy-Wiltshire zero-order hold. The acceleration is resolved along the chaser's local
radial, along-track and normal axes at the start of each segment and held there; those axes turn
against the Hill axes by the change of the chaser's angle from the target, 1e-6 rad over 10 m.

Accuracy depends on the start geometry:

- A chaser at rest on the target's circle stays there, as in the two-body model; Cartesian
  Clohessy-Wiltshire sees it drift (Alfriend et al., eqs 4.77, 5.70 and 5.71).
- A V-bar offset $D$, which Cartesian Clohessy-Wiltshire gets wrong by $6\pi D^2/a$ per orbit
  (2.74 m at 1 km and 500 km), is right to $O(D^4)$: 1.4e-7 m at 1 km.
- A radial offset $X$ started with Cartesian Clohessy-Wiltshire's bounded velocity
  $v_y = -2nX$ drifts $-12\pi X^2/a$ per orbit here, against $+3\pi X^2/a$ in the two-body model
  and zero in Cartesian Clohessy-Wiltshire. Starts for this model are best designed in its own
  coordinates.

## Linearized two-body motion with J2

`LinearizedTwoBodyDynamics(body="earth", altitude=500e3, inclination=0.0, j2=True, rtol=1e-11,
elements=None)` integrates the target's own orbit, with J2 when `j2` is set, and the chaser's
inertial offset under the gravity gradient along it:

$$
\delta\ddot{\mathbf r} = G(\mathbf r)\,\delta\mathbf r + C\,\mathbf a_{\text{Hill}}, \qquad
G = \frac{\partial\mathbf g}{\partial\mathbf r}
= -\frac{\mu}{r^3}\left(I - 3\,\hat{\mathbf r}\hat{\mathbf r}^T\right) + G_{J_2}
$$

where $G_{J_2}$ is the analytic derivative of the J2 acceleration of [dynamics.md](dynamics.md)
(`gravity_gradient`). The offset is mapped to the Hill frame exactly, as in the two-body model.
This is the exact first-order model for any target the two-body model can fly, including
eccentric ones; it is linear and, under J2, time-varying. Without J2 it is Clohessy-Wiltshire on
a circular orbit and Yamanaka-Ankersen on an elliptic one. Arguments are those of
`TwoBodyDynamics` without `perturbations`; names are `linear_j2` and `linear_twobody`.

Limits: the second-order terms dominate the error of a linear J2 model (Alfriend et al.,
sec. 7.4). For a projected circular orbit of radius $\rho$ at phase $\alpha$ the along-track
drift per orbit is $-(9\pi\rho^2/4a)(2 + \cos 2\alpha)$ (their eq 5.64). Each segment costs one
integration.

## Validation

Tests: `tests/test_elliptic_targets.py`, `tests/test_relative.py`, `tests/test_variational.py`
and `tests/test_fidelity.py`; published values in [validation.md](validation.md).

- Elliptic targets: the target's elements round-trip to 1e-12, target and chaser follow Kepler
  propagation over one orbit (1e-5 m), and J2 elements are mean elements.
- Yamanaka-Ankersen: the matrix is in the paper's frame and order; its closed-form inverse is
  exact to rounding at $e$ = 0, 0.3 and 0.9; with thrust and samples inside the segment it matches
  an integration of the linearized equations to 1e-11 of the state over two orbits at $e$ = 0.1,
  0.5 and 0.8 (Alfriend et al., example 5.5); at $e = 0$ it equals Clohessy-Wiltshire to 1e-13;
  its error against the two-body model grows as $\rho^2$ while Clohessy-Wiltshire's on the same
  elliptic target grows as $e\rho$; bounded starts repeat over three orbits; segments compose and
  run backward.
- Curvilinear Clohessy-Wiltshire: the maps round-trip to rounding from 1 mm to 50 km; a chaser on
  the target's circle is stationary in it and in the two-body model; V-bar errors fall as the
  fourth power of the offset while Cartesian Clohessy-Wiltshire's match $6\pi D^2/a$ to 1%; a
  radial offset drifts $-12\pi X^2/a$; short burns match Cartesian Clohessy-Wiltshire.
- Linearized model: without J2 it reproduces Clohessy-Wiltshire on a circular orbit and
  Yamanaka-Ankersen on an elliptic one to 1e-11; with J2, its 15-orbit along-track error against
  the nonlinear J2 model on Alfriend et al.'s examples 7.3 to 7.6 (0.5 km projected circular
  orbit, mean $a$ = 7100 km, $i$ = 70 deg) equals eq 5.64 to 0.1%, and the book's -11 m and
  -3.5 m to 0.5 m.

## Example

Clohessy-Wiltshire and Yamanaka-Ankersen against the two-body model for a chaser 100 m from a
target with $e$ = 0.1, after one orbit:

```python
import numpy as np

from vbar.dynamics.models import CWDynamics, TwoBodyDynamics
from vbar.dynamics.relative import YamanakaAnkersenDynamics

a, e, mean_anomaly = 7500e3, 0.1, 0.4
truth = TwoBodyDynamics(elements=[a, e, 0.9, 0.3, 1.1, mean_anomaly])
ya = YamanakaAnkersenDynamics(a, e, mean_anomaly)
cw = CWDynamics(truth.n)
start = ya.make_bounded([30.0, 90.0, 20.0, 0.0, 0.0, 0.01])
period = 2 * np.pi / truth.n
final = {}
for model in (truth, ya, cw):
    model.reset(start)
    final[model.name] = model.advance(np.zeros(3), period)[0][:3]
for name in ("ya", "cw"):
    print(name, "error after one orbit (m):", np.abs(final[name] - final["twobody"]).max())
```
