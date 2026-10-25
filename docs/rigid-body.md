# Rigid-body attitude

Code: `vbar.dynamics.rigid_body`. It gives the attitude of a tumbling target, a rigid body free of
torques, in closed form, for the target attitude modes of the scenarios.

## Principal axes

`principal_axes(inertia)` diagonalizes an inertia tensor about the centre of mass and returns the
moments in ascending order with the principal axes as the columns of a rotation matrix. An
eigensolver may return a left-handed set of eigenvectors, as Schaub and Junkins show in their
example 4.2; the last axis is flipped when it does, so the axes always form a rotation
(determinant +1). The tensor must be symmetric and positive definite.

## Torque-free motion in closed form

**Integrals.** Free of torques, a body keeps its rotational energy $T = \tfrac12 \sum I_i\omega_i^2$
and its angular momentum $\mathbf H$, fixed in inertial space. With principal moments
$I_1 < I_2 < I_3$ the motion is set by the effective inertia $D = H^2 / (2T)$, which lies between
$I_1$ and $I_3$, and the rate $\omega_l = 2T / H = H / D$ (Scheeres, *Orbital Motion in Strongly
Perturbed Environments*, 2012, section 3.2):

- $D > I_2$: the body rate circulates about the major axis (short-axis mode);
- $D < I_2$: it circulates about the minor axis (long-axis mode);
- $D = I_2$: the separatrix through the unstable spin about the intermediate axis.

**Body rate.** About the axis the rate circulates about (axis 3 below; for a long-axis tumble the
axes are relabelled $e_1' = e_3$, $e_2' = -e_2$, $e_3' = e_1$, which keeps them right-handed), the
rate is a pair of Jacobi elliptic functions (Landau and Lifshitz, *Mechanics*, 3rd ed., section 37):

$$
\boldsymbol\omega = \big(s A_1\,\mathrm{cn}\,\tau,\ \sigma A_2\,\mathrm{sn}\,\tau,\ s A_3\,\mathrm{dn}\,\tau\big),
\qquad \tau = \tau_0 + \lambda t,
$$

$$
A_1^2 = \frac{2TI_3 - H^2}{I_1(I_3 - I_1)},\quad
A_2^2 = \frac{2TI_3 - H^2}{I_2(I_3 - I_2)},\quad
A_3^2 = \frac{H^2 - 2TI_1}{I_3(I_3 - I_1)},\quad
\lambda^2 = \frac{(I_3 - I_2)(H^2 - 2TI_1)}{I_1 I_2 I_3},\quad
m = k^2 = \frac{(I_2 - I_1)(2TI_3 - H^2)}{(I_3 - I_2)(H^2 - 2TI_1)},
$$

with $s$ the sign of $\omega_3$ and $\sigma = \operatorname{sign}(I_3 - I_1)$. The numerators
$2TI_3 - H^2$ and $H^2 - 2TI_1$ are computed as sums of terms of one sign, and the mode from
$I_1(I_1 - I_2)\omega_1^2 + I_3(I_3 - I_2)\omega_3^2$, so none of them loses digits to
cancellation. The body rate has period $4K(m)/\lambda$; $\tau_0$ comes from the initial rate
through the amplitude $\operatorname{atan2}(\sigma\omega_2/A_2, s\omega_1/A_1)$ and the incomplete
integral of the first kind.

**Attitude.** With 3-1-3 angles about $\mathbf H$ (body axis 3 the last), $\theta$ and $\psi$ are
algebraic in the body components of $\mathbf h = I\boldsymbol\omega / H$:
$\cos\theta = h_3$, $\sin\theta\sin\psi = h_1$, $\sin\theta\cos\psi = h_2$. The precession angle
integrates in closed form,

$$
\phi(t) = \frac{H t}{I_3} + \frac{H (I_3 - I_1)}{I_3 I_1 \lambda}
\big[\Pi(n; \mathrm{am}\,\tau \mid m) - \Pi(n; \mathrm{am}\,\tau_0 \mid m)\big],
\qquad n = -\frac{I_3 (I_2 - I_1)}{I_1 (I_3 - I_2)},
$$

an incomplete elliptic integral of the third kind with a characteristic that depends on the
inertia alone. $\Pi$ is evaluated through Carlson's symmetric integrals, $\Pi(n; \varphi \mid m) =
\sin\varphi\, R_F(\cos^2\varphi, 1 - m\sin^2\varphi, 1) + \tfrac{n}{3}\sin^3\varphi\,
R_J(\cos^2\varphi, 1 - m\sin^2\varphi, 1, 1 - n\sin^2\varphi)$ (DLMF 19.25), continued through every
half period by $2\Pi(n \mid m)$, so it holds for any time. Scheeres' appendix F writes the third
kind with $1 + n\sin^2$, the opposite sign of this characteristic. The attitude is

$$
R_{nb}(t) = R_{nH}\, R_z(\phi)\, R_x(\theta)\, R_z(\psi),
$$

with $R_{nH}$ fixed by the attitude at $t = 0$. Attitudes are stored inertially, `matrix(t)` returning
$R_{nb}$ (`v_n = R_nb v_b`), and rates are in body axes.

**Pure spins.** A rate along a principal axis, or any rate of a body with three equal moments, is a
spin at a constant rate about a fixed axis and is taken apart. The exact separatrix is refused: it
has no periodic solution, and a rate one rounding step either side of it falls in one of the two
modes.

**Tumble parameters.** `TorqueFreeMotion.from_tumble(moments, D, omega_l, phase, direction,
spin_angle)` builds a motion from the parameters that describe it rather than from a rate vector
(Scheeres, section 3.2.8): the inertia and $D$ fix its shape, $\omega_l$ its pace, `phase` the
fraction of the body-rate period at $t = 0$, `direction` the inertial direction of $\mathbf H$ and
`spin_angle` the turn of the body about it.

**Bounds.** `rate_bound()` is the least of $\sqrt{2T/I_{\min}}$, $H/I_{\min}$ and
$\sqrt{A_1^2 + A_2^2 + A_3^2}$; `acceleration_bound()` bounds $|\dot{\boldsymbol\omega}|$ from Euler's
equations, $\dot\omega_i = (I_j - I_k)\omega_j\omega_k / I_i$, with each $|\omega_j| \le A_j$. Both
hold for all time and are within a factor 2 of the extremes on the test cases.

## Assumptions and limits

- No torques: gravity gradient, drag, solar pressure and outgassing are left out. The
  gravity-gradient pitch libration of a dead stage (Schaub and Junkins, section 4.4.3) needs a
  numerical model.
- The body is rigid with no internal dissipation, so a tumble keeps its mode; real debris drifts
  toward a spin about the major axis over long times.
- Moments are principal moments about the body axes; a full inertia tensor goes through
  `principal_axes` first.

## Validation

`tests/test_rigid_body.py`:

- the principal inertias of Schaub and Junkins example 4.2 (2.30798, 2.64310 and 7.04892 kg m^2)
  with right-handed axes;
- the axisymmetric case, whose transverse rate turns at $(I_3/I_T - 1)\omega_3$ in the body
  (Schaub and Junkins, eq. 4.42), to 1e-10;
- energy 8.3616 J and angular momentum 55.8134 N m s of Schaub and Junkins problem 4.9, with the
  inertial momentum constant to 1e-10;
- Scheeres' Toutatis tumble (section 3.2, table 8.1): $k^2$ = 0.037658, a body-rate period of
  5.4253 d, the mean precession rate of 2.3331e-5 rad/s (the node of the minor axis winds once less
  per period than the angle Scheeres reports) and a whole period that turns the body about
  $\mathbf H$ alone;
- the small-nutation limit of the period, $\sqrt{I_xI_y/((I_z - I_y)(I_z - I_x))}\,2\pi/\omega_l$
  (Scheeres, eqs. 3.84 to 3.87), to 1e-8;
- the closed form against Euler's equations and the quaternion kinematics integrated with DOP853
  at a relative tolerance of 1e-12, over three periods for short- and long-axis tumbles, an oblate
  and a prolate body, $D$ one part in a thousand either side of the separatrix ($m$ = 0.996),
  a small nutation, pure spins and rest, and over two periods for twelve random bodies and rates:
  attitudes agree to 1e-8 rad and rates to 1e-9 relative;
- the tumble parameters recovered from `from_tumble`, and the separatrix refused.

## Example

```python
import numpy as np

from vbar.dynamics.rigid_body import TorqueFreeMotion

# An upper stage tumbling about its major axis at about 1 deg/s.
motion = TorqueFreeMotion.from_tumble((1000.0, 4200.0, 4700.0), 4600.0, np.radians(1.0))
print(round(motion.period, 1), motion.long_axis)
attitude = motion.matrix([0.0, 60.0])
print(np.allclose(attitude[1] @ (motion.moments * motion.omega(60.0)), motion.momentum))
```
