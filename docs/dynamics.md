# Dynamics

Code: `vbar.dynamics`. Frames and units: [frames.md](frames.md).

## Interface

Every relative-motion model implements the `Dynamics` protocol, so a scenario can change fidelity
without changing anything else:

| Member | Meaning |
|---|---|
| `name` | the model's name, from the table below |
| `n` | the rate of the model's reference frame (rad/s), listed per model below |
| `reset(rel_state)` | set the chaser's state relative to the target, `[x, y, z, vx, vy, vz]` in the model's frame |
| `advance(accel_hill, duration, n_samples=0)` | hold an acceleration (m/s^2) in the model's frame for `duration` s; returns the final state and, if `n_samples > 0`, the states at `n_samples` evenly spaced times in `[0, duration]`, endpoints included |
| `state_in_last_segment(tau)` | the state `tau` s into the most recent `advance` |

The acceleration is held constant in the model's frame over each segment. For the Earth-orbit
models that frame is the target's Hill frame, which rotates with the target; for the three-body
models it is the synodic frame of the primaries, whose axes near the smaller primary are the Hill
axes of its orbit. Every model accepts these arguments by name as well as by position.

| `name` | Class | Frame | `n` | Page |
|---|---|---|---|---|
| `cw` | `CWDynamics(n)` | Hill | the given mean motion | this page |
| `twobody`, `twobody_j2` | `TwoBodyDynamics` | Hill, exact rotation | Keplerian mean motion of the (mean) semi-major axis | this page |
| `twobody_drag`, `twobody_j2_drag_srp`, ... | `TwoBodyDynamics(perturbations=...)` | Hill, exact rotation | as above | [perturbations.md](perturbations.md) |
| `ya` | `YamanakaAnkersenDynamics` | Hill of an elliptic Keplerian target | mean motion | [relative-motion.md](relative-motion.md) |
| `cw_curvilinear` | `CurvilinearCWDynamics` | Hill states in and out | circular mean motion | [relative-motion.md](relative-motion.md) |
| `linear_twobody`, `linear_j2` | `LinearizedTwoBodyDynamics` | Hill, exact rotation | as `TwoBodyDynamics` | [relative-motion.md](relative-motion.md) |
| `cr3bp` | `CR3BPDynamics` | synodic | synodic rate, $2\pi$ over the primaries' period | [three-body.md](three-body.md) |
| `libration` | `LibrationPointDynamics` | synodic | synodic rate | [three-body.md](three-body.md) |
| `cr3bp_linear` | `PeriodicLinearDynamics` | synodic | synodic rate | [three-body.md](three-body.md) |
| `bicircular` | `BicircularDynamics` | synodic | synodic rate | [three-body.md](three-body.md) |

A perturbed two-body model's name is `twobody` or `twobody_j2` followed by `_` and each force's
name (`drag`, `srp`, `srp_plate`) in the order given. `make_dynamics(kind, body="earth",
altitude=500e3, inclination=0.0)` builds `"cw"`, `"twobody"` or `"twobody_j2"` by name; the other
models take parameters it cannot carry and are built from their classes. Transition matrices of
the models and the metrics that compare them are in
[transition-matrices.md](transition-matrices.md).

## Clohessy-Wiltshire (`"cw"`)

Relative motion linearized about a circular target orbit with mean motion $n$
(Clohessy and Wiltshire, 1960):

$$
\begin{aligned}
\ddot x &= 3n^2 x + 2n\dot y + a_x \\
\ddot y &= -2n\dot x + a_y \\
\ddot z &= -n^2 z + a_z
\end{aligned}
$$

Assumptions: circular target orbit, separation small compared with the orbit radius, point-mass
gravity, no other forces.

**Propagation.** With the state $\mathbf x$ and the equations written as
$\dot{\mathbf x} = A\mathbf x + B\mathbf a$, an acceleration held for $\Delta t$ gives exactly

$$
\mathbf x(t + \Delta t) = \Phi(\Delta t)\,\mathbf x(t) + \Gamma(\Delta t)\,\mathbf a, \qquad
\Phi = e^{A\Delta t}, \qquad \Gamma = \int_0^{\Delta t} e^{As}\,ds\,B
$$

`cw_discrete(n, dt)` evaluates $\Phi$ and $\Gamma$ in closed form, so there is no integration
error at all. The closed form is written in terms of $k = 1 - \cos n\Delta t = 2\sin^2(n\Delta t/2)$
and $d = n\Delta t - \sin n\Delta t$, with $d$ taken from its Taylor series below $n\Delta t = 0.5$.
The textbook form subtracts nearly equal numbers on short steps and loses about four digits below
$n\Delta t = 10^{-2}$; this form keeps full precision at any step length. A numerical matrix
exponential (`scipy.linalg.expm`) is less accurate, about 4e-9 m off over one orbit on some
supported SciPy versions, and is used only as a cross-check in the tests. `CWDiscretizer` caches
the matrices by step length. `cw_states(n, dt, state, accel)` evaluates
$\Phi(\Delta t)\mathbf x + \Gamma(\Delta t)\mathbf a$ row by row for many different durations,
states and accelerations at once without forming the matrices, as the vectorized environment's
refinement needs; it agrees with the matrix product to 1e-13 of the largest term (tested in
`tests/test_cw.py` for 200 random steps from 1e-6 s to 1e4 s and for zero length).

**Accuracy.** The model error against nonlinear two-body motion grows with the square of the
separation. For a chaser held at along-track offset $D$ from a target with semi-major axis $a$,
the true motion drifts by $-6\pi D^2/a$ per orbit, which CW does not see: about 2.7 m per orbit at
1 km and 500 km altitude, and 2.5 mm at 30 m.

## Nonlinear two-body (`"twobody"`, `"twobody_j2"`)

**State.** The model carries the target's inertial state $(\mathbf r, \mathbf v)$ and the chaser's
inertial offset from it $(\delta\mathbf r, \delta\mathbf v)$, twelve numbers in all:

$$
\ddot{\mathbf r} = \mathbf g(\mathbf r), \qquad
\delta\ddot{\mathbf r} = \mathbf g(\mathbf r + \delta\mathbf r) - \mathbf g(\mathbf r) + C\,\mathbf a_{\text{Hill}}
$$

Integrating the offset directly, rather than two 7000 km positions, lets it have its own error
tolerances, so relative motion is resolved far below a millimetre. $C$ is the current Hill frame
of the target, so the commanded acceleration stays fixed in the rotating frame as in CW.

**Gravity difference.** Subtracting two accelerations of about 8 m/s^2 leaves rounding of about
1e-15 m/s^2, a relative error of 1e-9 at 1 m of separation. The point-mass part is computed instead
with Battin's function (*An Introduction to the Mathematics and Methods of Astrodynamics*):

$$
q = \frac{\delta\mathbf r \cdot (\delta\mathbf r + 2\mathbf r)}{|\mathbf r|^2}, \qquad
f(q) = \frac{q\,(3 + 3q + q^2)}{1 + (1 + q)^{3/2}}, \qquad
\mathbf g(\mathbf r + \delta\mathbf r) - \mathbf g(\mathbf r) = -\frac{\mu}{|\mathbf r + \delta\mathbf r|^3}\left(\delta\mathbf r - f(q)\,\mathbf r\right)
$$

which equals the direct difference exactly but has no cancellation. With J2, its contribution is
differenced directly; its rounding is a factor J2 smaller.

**J2.** The oblateness acceleration, with $z$ along the body's spin axis:

$$
\mathbf a_{J_2} = \frac{3}{2}\,\frac{J_2\,\mu R^2}{r^5}
\begin{bmatrix} x\,(5z^2/r^2 - 1) \\ y\,(5z^2/r^2 - 1) \\ z\,(5z^2/r^2 - 3) \end{bmatrix}
$$

**Integration.** DOP853 (scipy `solve_ivp`) with relative tolerance `rtol` = 1e-11 by default and
absolute tolerances of 1e-6 m and 1e-9 m/s on the target and 1e-9 m and 1e-12 m/s on the offset.
At the default the relative state stays within about 1e-9 of its own size over one orbit. Each
segment starts from the step size the previous one reached rather than from scipy's initial
guess, which cuts the cost of a 10 s step from about 84 gravity evaluations to 16.

**Target orbit.** The target starts at the ascending node (on +x) of a circular orbit with the
given altitude and inclination. With J2, "circular" means circular in first-order mean elements
(see below); the osculating state then carries J2's short-period terms instead of a spurious
eccentricity that would make the target's own orbit wobble. Given classical elements `elements`
`[a, e, i, raan, argp, M]` (m, rad), the target starts from them instead, and with J2 they are
mean elements ([relative-motion.md](relative-motion.md)).

**Other forces.** `perturbations` adds drag and solar radiation pressure on both spacecraft
([perturbations.md](perturbations.md)).

**Output.** The Hill state is formed with the exact frame rotation, including the out-of-plane
rate J2 causes ([frames.md](frames.md)).

**Mean motion.** `n` is the Keplerian mean motion of the mean semi-major axis for every model.
Under J2 the target's mean argument of latitude advances faster, by
$\tfrac{3}{2} J_2 (R/a)^2 (4\cos^2 i - 1)\,n$ (7.6e-4 n at 500 km and 51.6 deg). A controller
that uses `n` sees that difference as model error.

## Kepler propagation

Code: `vbar.dynamics.kepler`. Classical elements are `[a, e, i, raan, argp, M]`.

- `solve_kepler(M, e)`: Newton's method on $E - e\sin E = M$, started from $E = M$ for $e < 0.8$
  and from $\pi$ above, which prevents overshoot at high eccentricity. The result stays in the
  same revolution as $M$.
- `elements_from_state` and `state_from_elements`: conversions for elliptic orbits. On circular or
  equatorial orbits only the sums $\omega + M$ and $\Omega + \omega + M$ are defined, and angles
  are measured from the x axis when there is no node line.
- `propagate_kepler(r0, v0, t, mu)`: universal-variable propagation through the Lagrange
  coefficients $f, g, \dot f, \dot g$, valid on any conic (Bate, Mueller and White,
  *Fundamentals of Astrodynamics*, ch. 4). The Stumpff functions use their series below
  $|z| = 0.1$, where the closed forms cancel. Hyperbolic starting values follow Vallado,
  *Fundamentals of Astrodynamics and Applications*, algorithm 8. On an ellipse the time is first
  reduced to within half a period with an exact remainder, since the coefficients are periodic;
  subtracting whole periods instead loses a short arc's time entirely when the period is long
  (near-parabolic ellipses have periods up to $10^{18}$ s). The universal anomaly is found by
  Newton's method inside a bracket (the Kepler function rises with it at the rate of the radius),
  with a bisection whenever a step leaves the bracket; plain Newton from the elliptic starting
  value failed at $e \gtrsim 0.9$ over many periods. It stops when the step falls below
  $10^{-12}$ of the anomaly or the bracket closes to rounding. The second case occurs far out on
  near-radial hyperbolas, where the terms of the Kepler function reach $10^{17}$ and their
  difference never gets below its rounding; the returned state then carries that rounding, about
  $10^{-12}$ of the radius. The tests fly ellipses up to $e = 0.98$ over twenty periods and 500
  random hyperbolas up to $10^6$ s out and back.

## J2 mean elements

Code: `vbar.dynamics.mean_elements`.

- `j2_secular_rates(a, e, i, body)`: first-order secular rates of the node, perigee and mean
  anomaly. With $p = a(1 - e^2)$ and $\eta = \sqrt{1 - e^2}$:

$$
\dot\Omega = -\tfrac{3}{2} n J_2 \left(\tfrac{R}{p}\right)^2 \cos i, \qquad
\dot\omega = \tfrac{3}{4} n J_2 \left(\tfrac{R}{p}\right)^2 (5\cos^2 i - 1), \qquad
\dot M = n + \tfrac{3}{4} n J_2 \left(\tfrac{R}{p}\right)^2 \eta\,(3\cos^2 i - 1)
$$

- `mean_to_osculating` and `osculating_to_mean`: Brouwer's first-order theory in Lyddane's
  nonsingular form, as given by Schaub and Junkins (*Analytical Mechanics of Space Systems*).

Limits:

- The map is first order in J2, so a round trip leaves errors of order $J_2^2$, tens of metres in
  semi-major axis in low Earth orbit. Compare only mean elements computed the same way.
- The eccentricity terms are resonant at the critical inclination (63.43 deg). The factor
  $1 - 5\cos^2 i$ is clamped to at least 0.05 in magnitude, about 0.7 deg either side. Every
  clamped term carries a factor $e$, so circular orbits are mapped exactly.
- The inclination variable is $\sin(i/2)$, so accuracy in $i$ falls off toward 180 deg (round trip
  4e-6 rad at 150 deg).

## Bodies

Code: `vbar.dynamics.bodies`.

| Body | $\mu$ (m^3/s^2) | Radius (m) | $J_2$ | Source |
|---|---|---|---|---|
| `EARTH` | 3.986004418e14 | 6378137 | 1.08262982131e-3 | WGS 84 (NIMA TR8350.2), one matched set |
| `KERBIN` | 3.5316e12 | 600000 | 0 | Kerbal Space Program, as used by KSPDG |

Earth's J2 is the value of WGS 84's normal gravity field, which follows from its GM, radius,
flattening and spin rate. It differs from EGM96's own J2 by 3e-6 relative. `EARTH` also carries
the WGS 84 spin rate (7.292115e-5 rad/s) and flattening (1/298.257223563), used by the drag model
and the geodetic height.

The module also holds the Sun's constants for solar pressure (`AU` = 1.495978707e11 m, IAU 2012;
`SUN_RADIUS` = 6.957e8 m, IAU 2015; `SOLAR_PRESSURE` = 4.56e-6 N/m^2) and the three-body constant
sets ([three-body.md](three-body.md)).
