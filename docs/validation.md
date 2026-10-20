# Validation

The test suite (`python -m pytest -q`) checks the models against closed-form solutions, an
independent high-order integrator, universal-variable Kepler propagation, conserved quantities and
the published examples below. Each test names its source in a comment next to the check.

## How good the tests are

- Coverage: the suite executes 97% of the package's statements; CI fails below 95%. The only
  module without tests is the plotting helper `vbar.viz`.
- Every python example in README.md and these pages runs as part of the suite
  (`tests/test_docs.py`).
- Mutation testing (`mutmut`) plants small changes in the code (a flipped sign, a changed
  constant, a dropped argument) and checks that some test fails. In `vbar.dynamics.cw` the
  tests catch 359 of 389 mutants; the 30 that survive change nothing observable (slices past the
  end of an array, the cache key's rounding, the switch point between a series and its closed
  form, settings of the reference integrator). In `vbar.constraints` they catch 131 of 194: most
  survivors are in the arithmetic of the certified in-step bound, because the tests check that
  the bound never exceeds the true minimum but not how tight it is; the keep-out sphere's
  second-order bound now has a tightness table, and halving or doubling its curvature bound fails
  the suite. A weekly CI run repeats this for the core modules.

## Published examples reproduced

Each test below recomputes a printed result through the package and compares it at the precision
it is printed. Where a printed value is a misprint, the test uses the corrected value and says so.

**Montenbruck and Gill, *Satellite Orbits: Models, Methods and Applications*** (Springer, 2000)

- Exercise 2.1: Hohmann transfer from 750 to 775 km altitude (13.08 m/s), flown through the
  two-body model.
- Exercise 2.2: Kepler's equation for M = 50 deg, e = 0.72.
- Exercise 2.3: classical elements from a position and velocity.
- Exercise 2.5: the sun-synchronous orbit with 43 revolutions in 3 days, solved from the J2
  secular rates (a = 7153.123 km, i = 98.498 deg).
- Exercise 4.1 and section 4.1.6: RK4 errors on Hull et al. problem D1, which checks the set-up
  used for D1 to D5.
- Exercise 7.1: the first row of the Keplerian state transition matrix for a 650 km, 51 deg orbit
  over one day.
- Eq. 2.50: the rotation from perifocal to inertial axes.

**Schaub and Junkins, *Analytical Mechanics of Space Systems*** (AIAA). Identified by their
values, since example numbers differ between editions.

- Kepler's equation for e = 0.8, M = 1.5 rad.
- The 3-2-1 attitude examples with angles (30, -45, 60) deg and (10, 25, -15) deg: direction
  cosine matrices, the relative matrix [BF], Euler parameters, principal rotation and modified
  Rodrigues parameters with their shadow set.
- The Hohmann transfer from 7000 to 7200 km: 105.5392 m/s. The printed 104.351 m/s comes from a
  wrong circular speed.

**Alfriend, Vadali, Gurfil, How and Breger, *Spacecraft Formation Flying: Dynamics, Control and
Navigation*** (Butterworth-Heinemann, 2010)

- Example 3.1: mean to osculating elements for a mean orbit with a = 7100 km, i = 70 deg.
- Example 5.2: a two-impulse Clohessy-Wiltshire rendezvous. The printed numbers correspond to a
  transfer time of n t = 2 rad.
- Examples 7.3 and 7.4: inertial to Hill conversion for a 500 m projected circular orbit at phases
  0 and 90 deg.
- Example 9.2: short Clohessy-Wiltshire targeting on a 7170 km orbit, nulling the position in
  0.01 orbit.
- Eqs. 5.63, 5.64 and 5.73: second-order drift coefficients for along-track, radial and projected
  circular orbit starts.
- Eqs. 4.77, 5.70 and 5.71: the exact co-orbital equilibrium that Clohessy-Wiltshire mistakes for
  a drift.
- Page 92: the transfer times at which in-plane Clohessy-Wiltshire targeting is singular.

**Fehse, *Automated Rendezvous and Docking of Spacecraft*** (Cambridge University Press, 2003).
Recomputed for a 400 km orbit with this package's Earth constants; Fehse's axes are the ISS-style
LVLH frame, converted with `iss_to_hill`.

- Eq. 3.22: motion under a constant force.
- Eqs. 3.24 to 3.26: drift ahead of 94.25 m and 376.99 m per orbit from 10 m below the target.
- Eq. 3.27: an out-of-plane release crossing the orbit plane at 0.01131 m/s. The printed
  0.0138 m/s is a misprint.
- Eq. 3.28 and fig. 3.15: a 0.1 m/s forward push that comes back 1666.1 m behind, and the
  tangential impulse responses.
- Eq. 3.34 and fig. 3.20: radial impulse responses.
- Eq. 3.55: holding 100 m below the target takes a constant radial acceleration of 3 n^2 |x|
  (3.840e-4 m/s^2).
- Eqs. 3.59 to 3.74: one orbit of constant thrust from rest.
- Figs. 4.17 and 4.18: a 0.3 m/s radial collision-avoidance impulse from d = 20, 50 and 100 m
  below the target, drifting 12 pi d per orbit.

**Hull, Enright, Fellen and Sedgwick, "Comparing numerical methods for ordinary differential
equations"** (*SIAM Journal on Numerical Analysis* 9, 1972)

- Two-body test problems D1 to D5 (eccentricity 0.1 to 0.9), integrated by the two-body model.

## Published examples behind the fidelity models

The tests of the perturbation, relative-motion, transition-matrix and three-body models
([perturbations.md](perturbations.md), [relative-motion.md](relative-motion.md),
[transition-matrices.md](transition-matrices.md), [three-body.md](three-body.md)) reproduce the
examples below. Where a printed value cannot be reproduced from the book's own numbers, the test
says what it checks instead.

**Montenbruck and Gill, *Satellite Orbits: Models, Methods and Applications***

- Sec. 3.4.1: the Earth's umbra and penumbra half-cones, 0.264 and 0.269 deg, and the umbra's
  vertex 1.384e6 km behind the Earth; the cylindrical shadow's eclipse fractions at 500 km
  (37.8%, 35.8% and 23.1% at beta angles of 0, 30 and 60 deg) follow from the same geometry.
- Eqs. 3.85 to 3.94: the conical shadow function, checked against a numerical integration of the
  overlap of the solar and Earth disks.
- Table 3.8: Harris-Priester minimum and maximum densities at 400, 500 and 600 km.
- Exercise 5.3: the geodetic height of (1917032.190, 6029782.349, -801376.113) m on the WGS 84
  ellipsoid, -63.667 m.
- Eq. 3.69: solar radiation pressure of 4.56e-6 N/m^2 at 1 AU.
- Eq. 3.72: a specular plate at incidence $\theta$ pushed along its normal by
  $2P\,(A/m)\cos^2\theta$; absorbing, specular and diffuse plates facing the Sun equal cannonballs
  with $C_R$ = 1, 2 and 5/3.
- Fig. 3.1 and secs. 3.2 to 3.7: the forces left out of relative motion; the lunar and solar
  tidal accelerations at 1 km (1.7e-10 and 7.9e-11 m/s^2) are recomputed from IAU and JPL GMs.
- Exercise 7.1: the first row of the inertial state transition matrix of a 650 km, 51 deg orbit
  after one day, now from the variational equations of the two-body model: Keplerian to 6e-6 and,
  with J2, to 2e-4 (4e-5 with WGS 84's J2; the book states neither its J2 nor its integrator
  settings).
- Eq. 7.109 and exercise 7.1: the STM adequacy metric and its 0.1 limit, applied to
  Clohessy-Wiltshire against the J2 model at 500 km and 51.6 deg.

**Fehse, *Automated Rendezvous and Docking of Spacecraft***

- Eq. 4.3: drag of 1.3676e-6 m/s^2 at 400 km with 1e-11 kg/m^3 and a ballistic coefficient of
  215 kg/m^2 (printed rounded to 1.38e-6).
- Eq. 4.3 and fig. 4.3: a chaser with ballistic coefficient 470 kg/m^2, 1.5 times the target's,
  drifts 9.213 m up and 130.24 m behind in three orbits, as Clohessy-Wiltshire with the constant
  differential acceleration predicts. The text's "about 70 m behind and 5 m above" does not follow
  from its numbers.
- Fig. 4.15: a chaser started at -0.06 m/s along-track with five times the target's ballistic
  coefficient at 9.4e-12 kg/m^3 returns after 9.08 orbits ("about ten" in the book).

**Alfriend, Vadali, Gurfil, How and Breger, *Spacecraft Formation Flying***

- Example 5.5: Yamanaka-Ankersen matches an integration of the linearized equations, here at
  $e$ = 0.1, 0.5 and 0.8 with thrust and samples inside the segment.
- Eqs. 4.77, 5.70 and 5.71: the co-orbital equilibrium, stationary in curvilinear
  Clohessy-Wiltshire and in the two-body model.
- Eqs. 5.79 to 5.88: Clohessy-Wiltshire in curvilinear coordinates.
- Examples 7.3 to 7.6 and eq. 5.64: the linearized J2 model's 15-orbit along-track error on a
  0.5 km projected circular orbit about a mean 7100 km, 70 deg, node 45 deg target, -11 m and
  -3.5 m at phases 0 and 90 deg (read from figures), and the second-order drift
  $-(9\pi\rho^2/4a)(2 + \cos 2\alpha)$ per orbit to 0.1%.
- Sec. 8.3 and eqs. 8.58 to 8.60: the modeling-error index, flat in separation for J2 left out by
  Clohessy-Wiltshire on a projected circular orbit at 8000 km and 50 deg.

**Yamanaka and Ankersen, "New State Transition Matrix for Relative Motion on an Arbitrary
Elliptical Orbit"** (*Journal of Guidance, Control, and Dynamics* 25(1), 2002)

- The fundamental matrix in the paper's frame and state order, its pseudo-initial-value inverse,
  and its reduction to Clohessy-Wiltshire at $e = 0$.

**Koon, Lo, Marsden and Ross, *Dynamical Systems, the Three-Body Problem and Space Mission
Design*** (version 1.3, 2022)

- Table 2.2.1: the Earth-Moon, Sun-Jupiter and Sun-Earth velocity units, 1.025, 13.102 and
  29.784 km/s. The table's Sun-Earth period, 3.147e7 s, is a misprint for 3.156e7 s.
- Sec. 2.3, eqs. 2.3.1 to 2.3.5 and 2.3.14: the synodic equations against an inertial-frame
  integration, and the two Jacobi constant conventions, which differ by $\mu(1 - \mu)$.
- Sec. 2.5: the Sun-Jupiter Hill radius 6.825e-2 and L1 at 0.9324.
- Sec. 4.4: the Sun-Jupiter L2 and L3 energies, about -1.519 and -1.501.
- Fig. 2.5.1: linear stability of L4 below Routh's mass ratio 0.038521.
- Eqs. 2.6.1 to 2.6.4 and 6.4.1: the linearization at a massless secondary, and the
  libration-point equations at $c_2 = 1$, are Clohessy-Wiltshire.
- Eq. 3.3.1: the time-reversal mirror symmetry, in its spatial form.
- Secs. 6.3 and 6.4: the Sun-Earth L1 constants $\gamma_1$ = 1.001090475e-2, $c_2$, $c_3$, $c_4$,
  $\omega_p$, $\omega_v$ and $\kappa$, with Richardson's mass ratio 3.040357143e-6.
- Sec. 6.6: Richardson's coefficients $s_1$, $s_2$, $l_1$, $l_2$ and $\Delta$ at Sun-Earth L1,
  reproduced to 1e-6 with $b_{22} = +3c_3\omega_p/d_1$ and $a_{32}$ containing
  $4c_3(\kappa a_{24} - b_{22})$ (both misprinted), and $l_1$ = -15.96560314 as printed in
  sec. 6.6 (sec. 6.4's -15.9650314 is a misprint); the ISEE-3 halo with $A_z$ = 110,000 km,
  $A_x$ about 206,000 km, $A_y$ about 665,000 km and a period of 177.73 days.
- Secs. 4.3 and 6.7: the transition matrix is symplectic in canonical coordinates, and so keeps
  $J = [[2W, I], [-I, 0]]$ in position and velocity.
- Sec. 6.7: continuing the Earth-Moon L2 halo family in $z_0$ lands on an unrelated orbit past the
  family's largest $z_0$; the tests detect the jump and pass the turning point by
  pseudo-arclength continuation.
- Eqs. 5.3.4 to 5.3.7: the bicircular Sun's mass, distance and synodic rate, consistent with
  $\omega_S = 1 - \sqrt{(m_S + 1)/a_S^3}$.

Values described in the tests as computed were obtained with a separate implementation of Koon et
al.'s methods and constants: the Earth-Moon Lagrange points and Jacobi constants, the collinear
constants at Earth-Moon L1 and L2, a planar Lyapunov orbit at Earth-Moon L1, a small Earth-Moon L2
halo from Richardson's seed, and the 9:2 near-rectilinear halo orbit's state, period, perilune and
apolune radii, Jacobi constant and monodromy eigenvalues.

**Scheeres, *Orbital Motion in Strongly Perturbed Environments*** (Springer, 2012)

- Sec. 5.1: the change of the Jacobi constant in a time-periodic system, checked on the bicircular
  model with thrust.

## Published examples behind the engine

The tests and defaults of the spacecraft, sensor and constraint models and the environment
([spacecraft.md](spacecraft.md), [sensors.md](sensors.md), [constraints.md](constraints.md),
[environment.md](environment.md)) rest on the published values below. Statistical checks are set
to fail a correct model with probability 1e-6 and use fixed seeds.

**Montenbruck and Gill, *Satellite Orbits: Models, Methods and Applications***

- Eq. 3.129 and table 3.13: the delta-v of a constant-thrust burn,
  $-(F / \dot m)\ln(1 - \dot m t / m_0)$, for a 10 N, Isp 350 s station-keeping thruster on
  1000 kg for 600 s, to 1e-12.
- Eqs. 3.126 to 3.130: constant-thrust flight with the mass equation; one orbit of 10 s burns at
  0.02 m/s^2 matches DOP853 on the Clohessy-Wiltshire equations with $\mathbf F / m(t)$ to
  4.5e-9 m.

**Fehse, *Automated Rendezvous and Docking of Spacecraft***

- Page 97: a 1 deg thrust misalignment puts 0.017452 m/s across a 1 m/s burn, reproduced by the
  execution error model to 5e-7.
- Secs. 2.4 and 5.3.2: relative navigation good to about 1% of range, the default of the
  range-scaled sensor (5 m at 500 m).

**Alfriend, Vadali, Gurfil, How and Breger, *Spacecraft Formation Flying***

- Page 272: PRISMA's 0.7 mm/s minimum impulse bit, 0.35 N s on 500 kg, as the minimum impulse
  test's parameter.
- Eqs. 11.1 to 11.3: the per-burn execution error model, checked for its mean and covariance over
  20000 burns.
- Ch. 12: the carrier-phase differential GPS accuracies at 100 m, 1 km and 10 km behind the
  `:nav2` defaults (4 cm and 0.4 mm/s at 1 km).

The rocket equation, the minimum range of Clohessy-Wiltshire flybys and the Gauss-Markov
covariances are checked against closed forms and independent integrations rather than printed
examples; the pages above give the numbers.

## Published examples behind the metrics

The tests of the constraint types and the metrics ([constraints.md](constraints.md),
[metrics.md](metrics.md)) reproduce the values below, recomputed at 400 km with this package's
Earth constants and converted from Fehse's ISS-style axes.

**Fehse, *Automated Rendezvous and Docking of Spacecraft***

- Eq. 3.28 and fig. 3.13: thrust stopped at 0.1 m/s on the V-bar coasts forward 0.47755 v / n,
  42.21 m, the basis of the V-bar passive-speed envelope 2.094 n (gap).
- Eq. 3.22 and sec. 5.7.2: the R-bar passive-speed envelope $n\sqrt{(4h - d)^2 - 9h^2}$ gives
  0.1225, 0.0612 and 0.0283 m/s at 50, 30 and 20 m with 15 m of structure; the book's approach
  flies 0.1, 0.05 and 0.02 m/s there.
- Figs. 4.16 to 4.20: collision-avoidance impulses. Radially away from 20, 50 and 100 m below, the
  chaser is $12\pi d$ ahead after an orbit whatever the impulse; 0.5 m/s away with 0.10 m/s
  retrograde from 15 m below ends 2231.6 m ahead; 0.3 m/s further back from 20 m behind passes
  under the target at $x$ = -396.220 m and ends 4978.3 m ahead.
- Appendix A.1.5: the travelling ellipse. Eq. A.69 is printed with $+2b$; with $-2b$ it matches
  propagation to 1e-9 m on the appendix's case, where the printed form is 176 m off after 300 s.
- Eqs. 4.12 and 4.13: the Gaussian plume normalized to the thrust over a unit sphere, 13 deg
  one-sigma.
- Secs. 7.1.1, 8.3.5 and appendix C: the docking contact ranges behind the default contact limits
  (lateral 0.1 m and 0.03 m/s, probe-drogue axial window 0.05 to 0.10 m/s).

**Dunlap et al., "Comparing Run Time Assurance Approaches for Safe Spacecraft Docking"** (2021)

- Eq. 14: the distance-dependent speed limit $0.2 + 4n\rho$ m/s used in the speed tests.

**Boyd and Vandenberghe, *Convex Optimization*** (Cambridge, 2004)

- Exercise 8.3: past $\varphi = \theta + \pi/2$ the nearest point of a cone is its apex; the cone
  margin equals a direct search for the signed distance on 12 random cones.
- Secs. 5.1, 5.5 and 3.1.8: lower bounds need the dual norm of the cost (the optimality gap rejects
  bounds in the wrong norm), and zero-mean dither cannot lower expected fuel (the dither mutants).

**Goodman, *History of Space Shuttle Rendezvous*** (NASA JSC-63400, revision 3, 2011)

Units converted from ft and ft/s; mean motion at 185 n.mi. where the book flies the ISS approach.

- P. 137: the twice-orbital-rate circle flown clockwise needs only the radial hover thrust and
  lies on the no-drift manifold $v_y + 2nx = 0$. After a thrust loss 90, 120, 150 or 180 deg into
  it, the free ellipse keeps its range between $R$ and 7.00, 6.20, 4.15 and $2R$, and free drift is
  certified clear of a sphere just inside $R$ and not of one just outside (derived).
- Fig. 25.22: the STS-130 +V-bar gates (0.20, 0.15, 0.10 and 0.07 ft/s at 320, 110, 75 and 30 ft)
  sit inside the V-bar passive-speed envelope by 3.84, 1.76, 1.80 and 1.03 times.
- Table 24.3: the +R-bar gates from ORBT sit 3.2, 3.4, 4.1, 5.7, 7.7 and 11.0 times below
  $0.2 + 4n\rho$ at 500 km at 2000, 1000, 600, 500, 200 and 100 ft, which is why the closing-rate
  law is the scenario's (`GateTable`).
- P. 244: the measured Shuttle-ISS contact speed, 0.113 ft/s = 0.0344 m/s, lies outside the
  probe-drogue window 0.05 to 0.10 m/s and inside a window of 0.025 to 0.05 m/s (the upper bound
  from Fehse appendix C, the lower ours).

**Kochenderfer, Wheeler and Wray, *Algorithms for Decision Making*** (MIT Press, 2022)

- Sec. 14.1, eq. 14.5: 3 failures in 1e4, whose normal interval goes below 0. The tests check the
  exact Clopper-Pearson interval recomputed for it, (6.187e-5, 8.765e-4), and for two cases of ours
  at the 100-episode default, 90 and 95 of 100, (0.8238, 0.9510) and (0.887, 0.984), and the
  resolution table on the cards.
## Published results behind the agent layer

The reward library ([rewards.md](rewards.md)), the training interfaces
([training.md](training.md)) and the agent tools ([agents.md](agents.md),
[llm-agents.md](llm-agents.md)) are checked against the cases below.

**Bradtke and Duff, "Reinforcement learning methods for continuous-time Markov decision
problems"** (NIPS 1995)

- The semi-Markov return: a trajectory's discounted return is the same for 10 s steps, 5 s steps
  and an uneven partition, to 1e-12; on the environment, one burn through the keep-out sphere
  earns the same return at 10, 5 and 2.5 s steps, and one burn-and-coast decision earns what the
  steps it spans earn.

**Ng, Harada and Russell, "Policy invariance under reward transformations"** (ICML 1999), with
the absorbing terminal state of Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd
ed., 2018), sec. 3.4

- Potential-based shaping telescopes to $-\Phi(s_0)$ at a terminal and to
  $\Gamma_T\Phi(s_T) - \Phi(s_0)$ at a time limit, on random transitions and on PD captures in
  the environment.

**Bradtke, Ydstie and Barto, "Adaptive linear quadratic control using policy iteration"** (ACC
1994)

- For the Riccati gain of the exactly discretized discounted quadratic cost, the 5-step return of
  the environment plus the bootstrap $\Gamma_N V(x_N)$ equals $V(x_0) = -x_0^T P x_0$ to 1e-9.

**Alfriend, Vadali, Gurfil, How and Breger, *Spacecraft Formation Flying***

- Page 92: the LLM agent's transfer tool refuses durations where Clohessy-Wiltshire targeting is
  singular (a half orbit in the test) and arrives at its target otherwise.

**Sign-convention self-check**: the PD agent passes in every
continuous mode, ISS-style axes and a flipped radial axis are caught.

## Published examples behind the scenarios

**Goodman, *History of Space Shuttle Rendezvous*** (NASA JSC-63400 Rev 3, 2011; tables in the
LVLH frame of its appendix G in ft and ft/s, converted with `iss_to_hill`; n at 185 n.mi.).

- p. 137 (derived): the twice-orbital-rate circle needs only the radial hover force, and the
  closed loop from the +R-bar reaches the +V-bar after T/8 keeping the range to 1e-9 m; after
  thrust loss the free ellipse never comes closer than the circle; the quarter lap costs 5.500 nR
  against 4.422 nR at the inertial rate, with break-even near 200 deg.
- Tables 25.8 and 25.4: T/8 = 685.4 s against the flown 690 s; the pitch manoeuvre's 480 s; the
  final approach rates 0.200 and 0.10 ft/s.
- Table 24.3 and fig. 9.3: the gate time constants, 1000 s inside 500 ft on the HST profiles
  (600 s at 120 ft), 200 and 100 s on the dual-coelliptic profile; the closing-speed law
  0.2 + 4 n rho is 3.2 to 11.0 times the ORBT gates.
- p. 244: the measured contact speed 0.113 ft/s = 0.0344 m/s fits only the `apds` window.
- pp. 140 to 141 and 241 (derived): the R-bar free drift through the pitch manoeuvre stays within
  581 to 605 ft and the closing speed that returns to the start radius is 0.318 ft/s.
- Fig. 25.22: the flown V-bar rates sit inside the passive-speed envelope by factors 3.84 to 1.03.

**Fehse, *Automated Rendezvous and Docking of Spacecraft*** (Cambridge, 2003), in addition to the
examples behind the metrics:

- Eqs 3.29 to 3.32 and 3.44 to 3.48: a 2700 m V-bar transfer at 400 km costs 0.3241 m/s
  (tangential two-impulse), 1.5273 m/s (radial) and 7.082 m/s (forced straight line over one
  orbit; the printed "44x" ratio is 21.85).
- Eqs 3.33 and 3.38 to 3.39: the tangential and radial fly-arounds to a point 500 m below cost
  2 n dz = 1.1314 m/s and 3 n dz = 1.697 m/s.
- Section 5.3.1: 0.414 m of radial drift in 120 s from rest 15 m below the V-bar (the berthing
  box); the lateral speed r n on the axis of an inertially pointing target.
- Section 5.7.3: a port 20 m along-track on a target pitched 30 deg sits 20 tan(30 deg) =
  11.547 m off the V-bar.

**Montenbruck and Gill**, section 3.4: with the Sun in the orbit plane a 500 km target is in
eclipse 37.8% of each orbit, 23.1% at beta = 60 deg.

**Rawlings, Mayne and Diehl, *Model Predictive Control*** (2nd ed., 2022), sections 2.9 and 3.1:
the hold-box audit (a stationary dispersion of 0.22 m and 8.2 mm/s per axis for raw 1 m, 1 mm/s
noise into a 10 m, 0.1 m/s LQR at 10 s steps) and the undetectable along-track position bias.

**Kochenderfer, Katz, Corso and Moss, *Algorithms for Validation*** (2026), section 8.1: the
per-draw truncation level for an episode-level exceedance probability (3 sigma per draw leaves
94.6% of 1080-draw episodes outside the bounded model; 1e-3 needs 4.91 and 5.04 sigma for 1080
and 2160 draws).

## Published examples behind the solvers

The projection and the horizon solvers of [solvers.md](solvers.md) are checked against the cases
below.

**Boyd and Vandenberghe, *Convex Optimization*** (Cambridge University Press, 2004)

- Exercise 4.3: the QP with $P = [[13, 12, -2], [12, 17, 6], [-2, 6, 12]]$,
  $q = (-22, -14.5, 13)$ and $-1 \le x_i \le 1$, solved as a projection through the Cholesky
  factor of $P$, gives $x^* = (1, 0.5, -1)$, $p^* = -21.625$ and multipliers 1 on $x_1 \le 1$, 2
  on $x_3 \ge -1$ and 0 on the other four bounds, to 1e-12.
- Sec. 8.1.1: the projection onto a halfspace, $u_0 + \max(0, b - a \cdot u_0)\,a/|a|^2$, on 200
  random instances to 1e-12.
- Exercise 5.26: a feasible set that is a single point, where Slater's condition fails; its
  analogue, a row tangent to the thrust ball, is solved and reported without finite multipliers.

**Rawlings, Mayne and Diehl, *Model Predictive Control: Theory, Computation, and Design*** (2nd
ed., Nob Hill, 2017)

- Sec. 8.8.3, eqs. (8.51) to (8.54): the Riccati recursion equals the sparse KKT solve of the
  whole horizon to 1e-10 relative on exact CW with 40 time-varying stages, affine terms and
  $R = 10^3 I$; the same recursion solving the unsymmetrized stage system by LU is off by 33% in
  $u$ on that problem (our case, recomputed with our constants), and the test asserts it is off by
  more than 10%, while the shipped one-triangle Cholesky recursion keeps $P_k$ symmetric to
  rounding.
- Sec. 8.8.4: condensing is avoided; at 200 stages of 60 s the condensed Hessian's condition
  number exceeds 1e10 while the stage-wise solve matches the sparse solve to 1e-9.

## Methods

- Clohessy and Wiltshire, "Terminal guidance system for satellite rendezvous", *Journal of the
  Aerospace Sciences* 27, 1960: the linearized relative-motion model.
- Battin, *An Introduction to the Mathematics and Methods of Astrodynamics* (AIAA, revised
  edition, 1999): the cancellation-free gravity difference f(q) used by the two-body model.
- Bate, Mueller and White, *Fundamentals of Astrodynamics* (Dover, 1971), chapter 4, and Vallado,
  *Fundamentals of Astrodynamics and Applications*, algorithm 8: universal-variable Kepler
  propagation and its starting value for hyperbolas.
- Brouwer (*Astronomical Journal* 64, 1959) and Lyddane (*Astronomical Journal* 68, 1963), in the
  form given by Schaub and Junkins: the first-order J2 mean-osculating element map.
- NIMA TR8350.2, *Department of Defense World Geodetic System 1984*: Earth's GM, radius and J2,
  and its spin rate and flattening.
- Goodman, *History of Space Shuttle Rendezvous* (NASA JSC-63400 Rev 3, 2011): the flown gate
  tables, corridors, holds, contact speed and separation profile behind the Shuttle-practice
  scenarios.
- Fehse, *Automated Rendezvous and Docking of Spacecraft* (Cambridge, 2003), chapters 2, 3, 5, 7
  and 8: the mission phases, hold points, approach corridors, contact conditions and berthing box
  behind the scenario catalogue.
- Dunlap, Hibbard, Mote and Hobbs, "Comparing Run Time Assurance Approaches for Safe Spacecraft
  Docking" (2021), eq. 14: the distance-dependent speed limit of `docking-afrl`.
- Montenbruck and Gill, *Satellite Orbits* (Springer, 2000): the Harris-Priester density model
  (sec. 3.5.2), the geodetic height iteration (eqs 5.83 to 5.88), cannonball and flat-plate solar
  pressure (eqs 3.69 to 3.75), drag in a co-rotating atmosphere (eq 3.97), the conical shadow
  function (eqs 3.85 to 3.94) and the variational equations (sec. 7.2).
- Yamanaka and Ankersen (*Journal of Guidance, Control, and Dynamics* 25(1), 2002): the elliptic
  relative-motion transition matrix.
- Richardson, "Analytic Construction of Periodic Orbits about the Collinear Points" (*Celestial
  Mechanics* 22, 1980), and Koon, Lo, Marsden and Ross (secs. 2 to 8): the CR3BP, the collinear
  constants, the third-order halo seed, differential correction, monodromy and Floquet analysis,
  and the bicircular model.
- IAU 2012 resolution B2 and IAU 2015 resolution B3: the astronomical unit and the nominal solar
  radius.
- Tsiolkovsky's rocket equation, and Montenbruck and Gill (eqs 3.126 to 3.130), for propellant use
  at constant acceleration and at constant thrust.
- Alfriend et al. (eqs 11.1 to 11.3): the thrust execution error model.
- Brent, *Algorithms for Minimization without Derivatives* (Prentice-Hall, 1973): the bounded
  scalar minimization that refines the in-step constraint minimum.
- Kochenderfer, Katz, Corso and Moss, *Algorithms for Validation* (secs. 9.2 and 9.3): the
  second-order Taylor inclusion behind the keep-out sphere's in-step certificate, taken on the
  squared range.
- Donzé and Maler (FORMATS 2010): constraint margins as the signed distance to violation of
  signal temporal logic.
- Van Loan, "Computing integrals involving the matrix exponential" (*IEEE Transactions on
  Automatic Control* 23, 1978): the added covariance of the `:nav2` error between observations.
- Desplats (1988), as given by Fehse (eqs 4.12 and 4.13): the Gaussian plume pressure model.
- Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice"
  (NeurIPS 2021): the interquartile mean.
- Clopper and Pearson, "The use of confidence or fiducial limits illustrated in the case of the
  binomial" (*Biometrika* 26, 1934): the exact interval of a rate.
- Kochenderfer, Wheeler and Wray (sec. 14.1): ranking safety metrics by their unsafe tail, and
  sections 19.3 and 20.1: the navigation posterior as the reference for a stated belief.
- Rawlings, Mayne and Diehl, *Model Predictive Control: Theory, Computation, and Design* (2nd ed.,
  Nob Hill, 2017): bounds hold only for the model they are computed on (sec. 8.10), the
  preparation and feedback split of real-time iteration (sec. 8.9), the Riccati recursion for
  stage-structured QPs (sec. 8.8.3).
- Boyd and Vandenberghe, *Convex Optimization* (2004): the KKT conditions behind the active-set
  projection (sec. 5.5.3), Farkas certificates of infeasibility (sec. 5.8.3), projection onto a
  halfspace (sec. 8.1.1), the sum-of-infeasibilities phase I and its conflicting rows
  (sec. 11.4.1), and the conditioning argument for nondimensional units (secs. 9.3 and 9.5).
- Ames, Coogan, Egerstedt, Notomista, Sreenath and Tabuada, "Control Barrier Functions: Theory and
  Applications" (ECC 2019): safety filters, the activity of their rows, and the conservatism of
  safe sets under bounded inputs.
  Automatic Control* 23, 1978): the added covariance of the `:nav2` error between observations
  and the integrated quadratic reward.
- Ng, Harada and Russell, "Policy invariance under reward transformations" (ICML 1999):
  potential-based shaping.
- Bradtke and Duff, "Reinforcement learning methods for continuous-time Markov decision
  problems" (NIPS 1995): discounting per second.
