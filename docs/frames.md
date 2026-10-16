# Frames, units and rotations

Code: `vbar.conventions`.

## Units

SI throughout: m, m/s, m/s^2, s, kg, N. Angles are radians. `G0` = 9.80665 m/s^2 converts a
specific impulse in seconds to an exhaust velocity.

## Hill frame

All relative states are expressed in the Hill frame of the target's orbit (also called RIC, RSW
or RTN). For a target at inertial position $\mathbf r$ with velocity $\mathbf v$ and angular
momentum $\mathbf h = \mathbf r \times \mathbf v$:

$$
\hat{\mathbf x} = \frac{\mathbf r}{|\mathbf r|}, \qquad
\hat{\mathbf z} = \frac{\mathbf h}{|\mathbf h|}, \qquad
\hat{\mathbf y} = \hat{\mathbf z} \times \hat{\mathbf x}
$$

- x: radial, away from the central body (the R-bar)
- y: along-track, in the direction of motion for a circular orbit (the V-bar)
- z: orbit normal

`hill_frame(r, v)` returns the matrix $C$ whose columns are these axes in inertial coordinates,
so $\mathbf u_{\text{inertial}} = C\,\mathbf u_{\text{Hill}}$, and the frame's angular velocity
in Hill coordinates:

$$
\boldsymbol\omega = \left( \frac{|\mathbf r|\,a_h}{|\mathbf h|},\; 0,\; \frac{|\mathbf h|}{|\mathbf r|^2} \right)
$$

where $a_h$ is the component of the target's inertial acceleration along $\hat{\mathbf z}$. It is
zero for Keplerian motion. Under J2 it is not, and it turns the orbit plane about x; leaving it
out puts an error of about 0.8 mm/s per km of separation into relative velocities, which is the
size of typical navigation noise.

## Relative state

A relative state is $[x, y, z, \dot x, \dot y, \dot z]$: position $\boldsymbol\rho$ and the rate
of change of position seen in the rotating frame, not the difference of inertial velocities. With
$\delta\mathbf r$ and $\delta\mathbf v$ the chaser's inertial position and velocity minus the
target's:

$$
\boldsymbol\rho = C^{\mathsf T} \delta\mathbf r, \qquad
\dot{\boldsymbol\rho} = C^{\mathsf T} \delta\mathbf v - \boldsymbol\omega \times \boldsymbol\rho
$$

`inertial_to_hill` and `hill_to_inertial` implement this pair. Both accept arrays stacked along
leading axes.

## ISS-style LVLH frame

Much of the rendezvous operations literature (for example Fehse, *Automated Rendezvous and
Docking of Spacecraft*) uses a different local frame: x along the velocity, y opposite the orbit
normal, z toward the central body. In Hill axes:

| LVLH axis | Hill expression |
|---|---|
| x (V-bar) | +y |
| y (H-bar) | -z |
| z (R-bar, toward Earth) | -x |

`hill_to_iss` and `iss_to_hill` convert vectors (..., 3) and states (..., 6). Two traps when
reading that literature: plots often draw +R-bar (toward Earth) upward, and the Sun's beta angle is
sometimes defined with the opposite sign.

## Rotations

Rotation matrices map components between frames and follow scipy's active convention:

- the matrix of a quaternion `q_a_b` (scalar last) takes components in frame b to frame a,
  $\mathbf u_a = R(q_{a/b})\,\mathbf u_b$;
- `hill_frame` returns $C$ with $\mathbf u_{\text{inertial}} = C\,\mathbf u_{\text{Hill}}$.

Textbooks usually print passive rotations. Montenbruck and Gill's $R_x$, $R_y$, $R_z$ and Schaub
and Junkins' direction cosine matrices are transposes of the active rotation by the same angle.
Schaub and Junkins' $[BN]$ with Euler parameters $\beta_{B/N}$ is $R(q_{n/b})^{\mathsf T}$ with
$q_{n/b} = [\beta_1, \beta_2, \beta_3, \beta_0]$.
