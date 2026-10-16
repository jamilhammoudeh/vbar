# Transition matrices and fidelity metrics

Code: `TwoBodyDynamics.transition_matrices` and `gravity_gradient` in `vbar.dynamics.models`;
`vbar.dynamics.fidelity`. States are Hill states $[x, y, z, v_x, v_y, v_z]$
([frames.md](frames.md)).

## Transition matrices from the variational equations

`model.transition_matrices(accel_hill, duration)` returns the transition and input matrices of the
segment that `advance(accel_hill, duration)` would fly next,

$$
\Phi = \frac{\partial\mathbf x(t + \Delta t)}{\partial\mathbf x(t)}, \qquad
\Gamma = \frac{\partial\mathbf x(t + \Delta t)}{\partial\mathbf a_{\text{Hill}}}
$$

shapes (6, 6) and (6, 3), without changing the model's state. `TwoBodyDynamics`, with or without
J2 and perturbations, and `LinearizedTwoBodyDynamics` provide it. The linear models give theirs in
closed form or from stored matrices instead: `cw_discrete`, `YamanakaAnkersenDynamics.matrices`,
`libration_discrete` and `PeriodicLinearDynamics.matrices`.

**Equations.** The matrices come from the variational equations of the same model, integrated
along the chaser's own trajectory together with the state (Montenbruck and Gill, *Satellite
Orbits*, sec. 7.2). For the inertial offset $\mathbf y = (\delta\mathbf r, \delta\mathbf v)$:

$$
\dot\Phi_y = A(t)\,\Phi_y, \qquad \dot\Gamma_y = A(t)\,\Gamma_y + \begin{bmatrix} 0 \\ C(t) \end{bmatrix},
\qquad
A = \begin{bmatrix} 0 & I \\
G(\mathbf r + \delta\mathbf r) + \sum_k \partial\Delta\mathbf a_k/\partial\delta\mathbf r &
\sum_k \partial\Delta\mathbf a_k/\partial\delta\mathbf v \end{bmatrix}
$$

from $\Phi_y = I$ and $\Gamma_y = 0$, with $C(t)$ the Hill axes and $G$ the gravity gradient. Using
the same model matters: a Keplerian transition matrix integrated next to a J2 state drifts from
it within a few orbits (Montenbruck and Gill, sec. 7.2.3).

- **Gravity gradient.** `gravity_gradient(r, body, j2=False)` is analytic, shape `(..., 3, 3)`:
  $-\frac{\mu}{r^3}(I - 3\hat{\mathbf r}\hat{\mathbf r}^T)$ plus, with J2, the derivative of the
  J2 acceleration of [dynamics.md](dynamics.md).
- **Perturbation partials.** Drag and solar pressure partials come from central differences with
  steps of 1 m and 1e-3 m/s. The forces vary over tens of kilometres and kilometres per second, so
  the truncation error is below 1e-9. The twelve shifted offsets go through each force in one
  stacked evaluation.
- **Hill frame.** With $M(t)$ the linear map from $(\delta\mathbf r, \delta\mathbf v)$ to the Hill
  state, $\begin{bmatrix} C^T & 0 \\ -[\boldsymbol\omega\times]\,C^T & C^T \end{bmatrix}$ with
  $\boldsymbol\omega$ the frame's rate from the target's full acceleration,
  $\Phi = M(t + \Delta t)\,\Phi_y\,M(t)^{-1}$ and $\Gamma = M(t + \Delta t)\,\Gamma_y$.
- **Integration.** DOP853 at the model's `rtol`, with absolute tolerances of 1e-12 scaled to each
  block's units ($1/n$ for position per velocity, $n$ for velocity per position, $1/n^2$ and
  $1/n$ for the input matrix).

Limits: shadow crossings make the solar pressure partials noisy, since the penumbra is shorter
than an integration step ([perturbations.md](perturbations.md)). A transition matrix over one day
at 650 km costs about three times a coasting segment of the same length.

## STM adequacy

`stm_adequacy(phi_true, phi_model, n)` measures how far a model's transition matrix is from the
truth's over the same horizon (Montenbruck and Gill, eq 7.109):

$$
m = \max_{ij}\,\bigl|\bigl(D\,(I - \Phi_{\text{true}}\Phi_{\text{model}}^{-1})\,D^{-1}\bigr)_{ij}\bigr|,
\qquad D = \operatorname{diag}(1, 1, 1, 1/n, 1/n, 1/n)
$$

The scaling counts a velocity error as the position error it causes in $1/n$ seconds. $m$ is zero
when the matrices agree; Montenbruck and Gill take 0.1 as the limit of an adequate model
(exercise 7.1). Stacked matrices `(..., 6, 6)` give one value each.
`validity_horizon(times, adequacy, limit=0.1)` returns the first of increasing `times` at which
the adequacy exceeds `limit`, or `inf`.

Clohessy-Wiltshire against the J2 model at 500 km and 51.6 deg has $m$ = 0.053 after one orbit
and 0.106 after two. Within each orbit $m$ oscillates, with the along-track response to a radial
offset dominating, and it first exceeds 0.1 at under half an orbit (0.111 at 0.45 orbit).
Sampling only at whole orbits would put the horizon near two orbits.

## Modeling-error index

`modeling_error_index(truth, model, n)` compares trajectories rather than matrices (Alfriend et
al., *Spacecraft Formation Flying*, eqs 8.58 to 8.60). `truth` and `model` are the two models'
states, shape (k, 6), at one time, flown from the same k start states:

$$
\nu = \max_k\left|\frac{w(\bar{\mathbf x}_k)}{w(\mathbf x_k)} - 1\right|, \qquad
w(\mathbf x) = |D\,\mathbf x|^2
$$

with $\bar{\mathbf x}$ the model's state and $D$ as above. For a small error along the state, $\nu$
is about twice its fractional size. It stays flat as the separation grows when the model leaves
out effects linear in it (J2, eccentricity) and grows with it when the model leaves out
nonlinearity.

## Validation

Tests: `tests/test_variational.py` and `tests/test_fidelity.py`; published values in
[validation.md](validation.md).

- The analytic gravity gradient, with and without J2, matches central differences of the gravity
  to 1e-9 of its size.
- Transition and input matrices of the J2 model, alone and with drag or solar pressure, match
  central differences of the model's own `advance` to 1e-9 in scaled units; computing them leaves
  the state unchanged; with conservative forces $\det\Phi = 1$ over an orbit.
- Montenbruck and Gill's exercise 7.1, the inertial transition matrix of a 650 km, 51 deg orbit
  after one day: the Keplerian first row to 6e-6; with J2 the row agrees to 4e-5 using WGS 84's
  J2, since the book states neither its J2 nor its integrator settings.
- STM adequacy is zero for the true matrix and scales velocities by $1/n$; Clohessy-Wiltshire's
  horizon under J2 at 500 km lies between 0.4 and 0.5 orbit, and its misfit doubles from one orbit
  to two.
- The modeling-error index of Clohessy-Wiltshire against J2 changes by less than 20% from 100 m to
  1 km (Alfriend et al., sec. 8.3); that of Yamanaka-Ankersen against the two-body model grows
  tenfold over the same range; for a 0.1% error it is 2e-3.

## Example

The horizon over which Clohessy-Wiltshire stays adequate against the J2 model at 500 km:

```python
import numpy as np

from vbar.dynamics.cw import cw_stm
from vbar.dynamics.fidelity import stm_adequacy, validity_horizon
from vbar.dynamics.models import TwoBodyDynamics

truth = TwoBodyDynamics(altitude=500e3, inclination=np.radians(51.6), j2=True)
truth.reset(np.zeros(6))
period = 2 * np.pi / truth.n
times = period * np.linspace(0.1, 1.0, 10)
adequacy = [
    stm_adequacy(truth.transition_matrices(np.zeros(3), t)[0], cw_stm(truth.n, t), truth.n)
    for t in times
]
print("adequate for", validity_horizon(times, adequacy) / period, "orbit")
```
