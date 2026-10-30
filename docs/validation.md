# Validation

The test suite (`python -m pytest -q`) checks the models against closed-form solutions, an
independent high-order integrator, universal-variable Kepler propagation, conserved quantities and
the published examples below. Each test names its source in a comment next to the check.

## How good the tests are

- Coverage: the suite executes 97% of the package's statements; CI fails below 95%.
- Every python example in README.md and these pages runs as part of the suite
  (`tests/test_docs.py`).
- The Rerun replay export is checked by reading its files back with Rerun's own reader
  (`docs/replay.md`, "Validation").
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

## Published examples behind the rigid-body attitude

**Schaub and Junkins, *Analytical Mechanics of Space Systems*** (4th ed.):

- Example 4.2: the principal inertias 2.30798, 2.64310 and 7.04892 kg m^2, with a right-handed
  set of axes.
- Eq. 4.42: the transverse rate of an axisymmetric body turns at (I3 / I_T - 1) w3 in the body.
- Problem 4.9: I = diag(210, 200, 118) kg m^2 and w0 = (0.2, 0.15, -0.18) rad/s keep T = 8.3616 J
  and |H| = 55.8134 N m s, the momentum fixed in inertial space.
- Eq. 4.158, in V-BAR axes (the largest moment about the orbit normal z, the smallest radial): a
  small pitch about z librates under the gravity-gradient torque at n sqrt(3 (I_yy - I_xx) / I_zz).

**Scheeres, *Orbital Motion in Strongly Perturbed Environments*** (2012), section 3.2:

- The Toutatis long-axis tumble (table 8.1 ratios, I_D/Iz = 0.51983, w_l = 1.8548e-5 rad/s):
  k^2 = 0.037658, a body-rate period of 5.4253 d and a mean precession rate of 2.3331e-5 rad/s.
- Eqs. 3.84 to 3.87: near a spin about the major axis the period tends to
  sqrt(Ix Iy / ((Iz - Iy)(Iz - Ix))) 2 pi / w_l.

The closed form of Landau and Lifshitz (*Mechanics*, section 37) is checked against Euler's
equations integrated numerically for every tumble mode, near the separatrix and on a random sweep
([rigid-body.md](rigid-body.md)).

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
  $R = 10^3 I$; the same recursion solving the unsymmetrized stage system by LU is off by 6.7% to
  33% in $u$ on that problem depending on the BLAS build (our case, recomputed with our constants),
  and the test asserts it is off by more than 1%, while the shipped one-triangle Cholesky recursion
  keeps $P_k$ symmetric to rounding.
- Sec. 8.8.4: condensing is avoided; at 200 stages of 60 s the condensed Hessian's condition
  number exceeds 1e10 while the stage-wise solve matches the sparse solve to 1e-9.

**Boyd and Vandenberghe**, additional items for the horizon solver:

- Sec. 11.7, Algorithm 11.2: on a 10-stage CW approach with an active thrust box the objective
  matches SLSQP on the condensed problem to 1e-8 relative, with KKT residuals below 1e-7.
- Sec. 11.4.1 with sec. 5.8.3: an unreachable terminal row is certified infeasible, and the
  certificate verifies by Farkas' lemma against the assembled matrices.

## Published examples behind the baselines

The tests of the classical baselines ([lqr.md](lqr.md), [lqg.md](lqg.md),
[lyapunov.md](lyapunov.md), [two-impulse.md](two-impulse.md), [glideslope.md](glideslope.md),
[trajectory-elements.md](trajectory-elements.md), [shuttle-baselines.md](shuttle-baselines.md),
[lambert.md](lambert.md), [mpc.md](mpc.md), [nonlinear-mpc.md](nonlinear-mpc.md)) reproduce the
examples below. Each item names its test.

**Alfriend, Vadali, Gurfil, How and Breger, *Spacecraft Formation Flying***

- Example 5.2, in the book's normalized units: the two impulses at $nt_f$ = 2 rad,
  (-0.00180, -0.01927, 0.005865) and (0.00562, -0.00173, 0.0165), total 0.03774
  (`test_alfriend_example_5_2_two_impulse_rendezvous`). The printed transfer time (3.22 h) is a
  misprint: the printed impulses belong to $nt_f$ = 2 rad. The printed second impulse has the
  wrong sign; the rendezvous needs $-\mathbf v(t_f)$.
- Example 5.2, the cheapest transfer: the printed minimum (0.0375 at $nt_f$ = 2) is a misprint;
  recomputed at the stated inputs the least total is 0.03572 at $nt_f$ = 4.64
  (`test_the_cheapest_transfer_of_example_5_2_lies_past_the_printed_transfer_time`), and the
  two-impulse fuel bound reaches it
  (`test_bound_reaches_the_cheapest_two_impulse_transfer_of_alfriend_example_5_2`).
- Example 9.2: from (25, 25, 50) m to the origin in 0.01 orbits at $a$ = 7170 km, the departure
  velocity (-0.38888, -0.43924, -0.82643) m/s, the fifth digit recomputed
  (`test_alfriend_example_9_2_short_transfer_departure_velocity`).
- P. 92: the transfer times where targeting is singular, $nt_f = 2k\pi$ and the roots of
  $8\cos x + 3x\sin x = 8$ in plane (2.81346 and 4.89060), $k\pi$ across the track
  (`test_conditioning_vanishes_at_the_singular_transfer_times`).
- Section 10.1.2: LQR gains scale with the mean motion when the step scales with the period
  (`test_gains_scale_with_mean_motion_when_the_step_scales_with_the_period`). Eq. 10.8: the
  tracking law at rest on the hold point is the CW-cancelling PD law
  (`test_tracking_law_at_rest_on_the_hold_point_is_the_cw_cancelling_pd_law`). The printed sign of
  the innovation in the filter update is a misprint; the update adds the gain times
  $\mathbf y - C\hat{\mathbf x}$ (`test_update_adds_the_gain_times_the_innovation`).

**Fehse, *Automated Rendezvous and Docking of Spacecraft*** (at his $n$ = 1.131367e-3 rad/s)

- Section 3.3.2, eqs. 3.41 to 3.43: CW targeting from 3000 m to 300 m behind the target in 0.37
  orbits reaches the end point (`test_fehse_in_plane_targeting_reaches_the_end_point`).
- Section 3.3, transfers of 2700 m along the V-bar: tangential 0.3241 m/s, printed as 0.33
  (`test_tangential_transfer_burns_n_dy_over_6_pi_and_costs_n_dy_over_3_pi`); radial 1.5273 m/s,
  $3\pi/2$ times the tangential one
  (`test_radial_transfer_costs_3_pi_over_2_times_the_tangential_one`).
- Eqs. 3.33, 3.38 and 3.39: fly-arounds to a point 500 m below the target cost 1.1314 m/s
  tangentially and 1.697 m/s radially (`test_fly_arounds_cost_2_n_dz_and_3_n_dz`).
- Eqs. 3.44 to 3.48: the forced straight-line approach from 3000 m to 300 m costs 7.082 m/s over
  an orbit and 8.054 m/s over half an orbit, 21.85 and 24.85 times the tangential transfer
  (`test_straight_vbar_approach_costs_2_v_plus_2_n_dy`). The printed half-orbit ratio (44) is a
  misprint.
- Eqs. 3.61 and 3.68: a finite burn lasts $\Delta v/\gamma$
  (`test_finite_burn_lasts_dv_over_the_acceleration`).
- Section 4.4.1: tracking the nominal arrives closer than flying the plan open loop
  (`test_tracking_the_nominal_arrives_closer_than_the_open_loop_plan`).
- Section 5.7: the final approach keeps its speed, 0.04 m/s, up to capture
  (`test_fehse_sequence_final_leg_keeps_constant_speed_up_to_capture`).
- Section 6.2.3, p. 183, and appendix B, p. 443: exponential braking with $\lambda$ = 1000 s gives
  0.3, 0.1 and 0.01 m/s at 300, 100 and 10 m, the limit of the glideslope profile as the arrival
  rate vanishes (`test_shuttle_exponential_braking_is_the_limit_of_a_vanishing_arrival_rate`).

**Curtis, *Orbital Mechanics for Engineering Students***, section 7.5 (first edition Examples 7.2
and 7.3, later editions 7.4 and 7.5)

- Example 7.5: from 2 km behind at rest, rendezvous in 1.49 h with two impulses of 0.1226 m/s,
  total 0.2452 m/s (`test_curtis_example_7_5_two_impulses_from_two_kilometres_behind`,
  `test_curtis_example_7_5_departure_and_arrival_velocities`).
- Example 7.4: from (20, 20, 20) km in 8 h, impulses of 0.0740440 and 0.0355649 km/s, total
  0.109609 km/s (`test_curtis_example_7_4_two_impulses_from_twenty_kilometres_in_eight_hours`,
  `test_curtis_example_7_4_final_impulse_cancels_the_arrival_velocity`). The first edition's
  printed final impulse is a misprint; the later editions' figures are reproduced.

**Montenbruck and Gill, *Satellite Orbits: Models, Methods and Applications***

- Exercise 2.6: the orbit through two positions 9000 s apart, $a$ = 28196.776 km,
  $e$ = 0.7679436, $i$ = 20.315 deg, $\Omega$ = 359.145 deg, $\omega$ = 179.425 deg,
  $M$ = 29.236 deg (`test_montenbruck_exercise_2_6_orbit_from_two_positions`); its velocities
  close the arc under Kepler propagation (`test_montenbruck_exercise_2_6_velocities_close_the_arc`).

**Schaub and Junkins, *Analytical Mechanics of Space Systems***

- The Lambert identities on 200 random problems each way: the conic through both positions
  (`test_random_solutions_lie_on_a_conic_through_both_positions`), the velocity locus
  $v_c v_\rho = \mu c \sec^2(\Delta f/2)/(2r_1r_2)$
  (`test_random_solutions_satisfy_the_lambert_velocity_locus`) and
  $f\dot g - g\dot f = 1$ (`test_lagrange_coefficients_of_the_solution_have_unit_determinant`).
  The printed value 0 is a misprint.
- Section 13.7, eq. 13.229: the Lyapunov law makes the closed-loop acceleration the linear error
  law on the declared two-body model, with and without J2
  (`test_closed_loop_acceleration_is_the_linear_error_law_on_the_declared_model`).

**Rawlings, Mayne and Diehl, *Model Predictive Control: Theory, Computation, and Design*** (2nd
ed.), at 500 km with a 10 s step

- Chapter 1 (sections 1.2.5 and 1.5.1, exercise 1.61), the hold point: an R-bar hold 100 m out
  needs $3n^2 \cdot 100$ m = 3.6749087912e-4 m/s²
  (`test_r_bar_hold_needs_three_n_squared_times_the_offset`); regulating without the target input
  settles (-3.702, 0.041, 0) m off at scales (10 m, 0.01 m/s)
  (`test_regulating_to_the_hold_point_without_feedforward_leaves_an_offset`,
  `test_regulating_without_target_input_settles_at_the_known_offset`), and the target input
  removes the offset (`test_feedforward_removes_the_hold_point_offset`).
- Chapter 1, the one-step delay: ignored by the design, it moves the closed-loop spectral radius to
  1.3921, 1.0057 and 0.9012 at three weightings
  (`test_one_step_delay_ignored_by_the_design_changes_the_closed_loop_radius`); the augmented
  design recovers the undelayed radius
  (`test_delay_augmented_design_recovers_the_undelayed_radius`) and captures where the undelayed
  one does not (`test_delay_augmented_design_captures_where_the_undelayed_design_does_not`).
- Section 5.5: a constant unmodelled acceleration leaves the steady offset
  (1.288762e-4, 5.842058e-4, -1.759851e-4) m
  (`test_constant_disturbance_leaves_the_known_steady_offset`).
- Chapter 8 (section 8.5.2, with Van Loan 1978): the sampled-data stage cost gives the exact
  zero-order-hold optimum, a cost of 116.65796914 at 10 s and 122.99249126 at 60 s, and differs
  from weighting the samples by 0.55 % and 6.6 % of the gain
  (`test_sampled_data_cost_gives_the_exact_zoh_optimum`).
- Lemma 1.8, constant biases with full-state measurements: an along-track position bias is not
  detectable, rank 6 of 7 (`test_along_track_position_bias_is_not_detectable`); every other single
  channel is (`test_constant_bias_on_one_channel_is_detectable_except_along_track_position`); all
  six together reach rank 11 of 12, and the five without the along-track position are detectable
  (`test_biases_on_all_six_channels_are_not_detectable_together`).
- Section 2.5.1, the horizon QP with the DARE terminal cost: with no binding row the first move is
  the LQR law for $N$ = 1, 2, 5, 20 and 60 over 50 states, to 1e-10 relative, for discrete
  weights $Q = \operatorname{diag}(10^{-2} I_3, 10^4 I_3)$, $R = 2500 I$, the sampled cost, a
  one-step delay and a rate weight of 0.1
  (`test_unconstrained_first_move_is_the_lqr_law_for_every_horizon`); the two-step first-move loop
  has spectral radius 0.99005 with the DARE terminal cost and 0.99995 with none
  (`test_dare_terminal_cost_gives_the_lqr_loop_at_two_steps_and_none_does_not`).
- Section 2.5.4, the thrust box and the terminal set: from rest 0.5 and 0.99 times the along-track
  reach of the set (201.509569 m for a 0.02 m/s² box, 116.341604 m for the box inscribed in the
  ball) the first move is the LQR law (`test_mpc_equals_the_lqr_inside_the_terminal_set`); at 3
  and 20 times, $N$ = 30, the box cuts the LQR move by 0.040 and 0.380 m/s² (box) and 0.023 and
  0.219 m/s² (ball), to 1e-6 relative of a bounded-variable least-squares solve
  (`test_box_rows_cut_the_lqr_move_outside_the_terminal_set`).
- Proposition 2.18: 604.528706 m along-track from the hold point at rest, the optimal cost is
  1.855711116e5 for $N$ = 5, 10, 20 and 40 to 1e-8 relative, above the unconstrained value
  1.845593979e5 (`test_constrained_cost_is_the_infinite_horizon_value_for_every_horizon`), and
  does not rise with $N$ (`test_constrained_cost_does_not_rise_with_the_horizon`).
- Section 8.5.2, the sampled-data cost in the horizon QP: $Q_c = \operatorname{diag}(10^{-4}
  I_3, I_3)$, $R_c = 10^3 I$, five steps from (50, -80, 10) m, (0.02, -0.01, 0.005) m/s give twice
  the cost 116.65796914 at 10 s and 122.99249126 at 60 s, to 1e-10 relative
  (`test_sampled_cost_objective_is_the_van_loan_infinite_horizon_value`).
- Section 8.5.1, path constraints between grid points: zero-thrust coasts with both ends on the
  keep-out sphere dip inside it by 1.783e-2 m (0.2 m/s, 10 s, $R$ = 50 m), 2.806e-1 m (1 m/s) and
  6.491e-2 m (0.2 m/s, 5 s, $R$ = 2 m), and the tightened keep-out rows reject each of them
  (`test_rows_reject_a_coast_that_dips_inside_between_samples`).
- Sections 2.4.2 and 2.7, recursive feasibility: for 20 converged plans with the terminal set on
  `approach-vbar`, 55 to 120 m from the target, the shifted plan with the LQR move appended meets
  every row of the next problem (`test_shifted_plan_meets_every_row_of_the_next_problem`).
- Sections 8.5.2 and 8.6.3, direct multiple shooting with one Gauss-Newton step; the book prints
  no number for this case, so the values come from our own independent computation. A
  minimum-energy rendezvous to the origin at rest from (100, -2000, 50) m, (0.05, 0.2, -0.02) m/s
  on two-body at 51.6 deg: over four 60 s segments the CW plan flown open loop misses by
  0.017893 m, and one step from the shooting linearization leaves continuity below 1e-10 and a
  miss below 1e-9 m
  (`test_one_gauss_newton_step_from_the_cw_plan_closes_a_short_two_body_rendezvous`); over twenty
  segments of half an orbit the CW plan misses by 3.9909 m, and one step leaves continuity below
  3e-10, a miss below 1e-7 m and 0.07806 % more energy than the CW plan
  (`test_one_gauss_newton_step_closes_the_half_orbit_two_body_rendezvous`). The node Jacobians
  from the variational equations agree with central differences of the flown segment to 1e-5
  relative, with and without J2
  (`test_two_body_jacobians_equal_central_differences_of_the_flown_segment`).
- Section 8.9.2, eq. 8.60, the real-time iteration: on `approach-vbar` flown on two-body from
  300 m behind the hold point, the matched declared model predicts the next observation to 1e-8 m
  and 1e-9 m/s over three decisions of a two-step horizon
  (`test_matched_two_body_model_predicts_the_next_observation`), and to 1e-8 m over five decisions
  of a ten-step horizon
  (`test_matched_two_body_model_predicts_the_next_observation_over_a_ten_step_horizon`); declared
  CW mispredicts by more than 5e-7 m
  (`test_declared_cw_on_two_body_motion_mispredicts_by_the_gravity_remainder`). The preparation
  linearizes at the shifted plan without reading the new observation
  (`test_preparation_linearizes_at_the_shifted_plan_without_the_new_observation`).

**Boyd and Vandenberghe, *Convex Optimization***

- Exercise 5.16, the exact penalty: on `approach-vbar`, for ten approach starts and ten passes 55
  to 90 m from the target, the softened keep-out rows give the hard solution with slack below
  1e-9 m at the default price, 10 times the largest stage multiplier sum of 205.958 per m
  (`test_default_penalty_gives_the_hard_solution`,
  `test_default_penalty_is_ten_times_the_largest_keep_out_multiplier_sum`); at a hundredth of it a
  pass takes slack (`test_penalty_below_the_keep_out_multipliers_leaves_slack`).

**Goodman, *History of Space Shuttle Rendezvous*** (NASA JSC-63400, revision 3, 2011)

- Chapter 23, pp. 178 to 180, Tables 23.4 and 23.5: the NSR transfer takes $4/(3n)$, 19.03 min at
  140 n.mi. (`test_nsr_transfer_takes_4_over_3n_which_is_19_03_min_at_140_nmi`); the printed
  expression $(4/3)\omega$ is a misprint. The football transfer takes a quarter period, 22.42 min
  (`test_football_transfer_takes_a_quarter_period_which_is_22_42_min_at_140_nmi`).
- P. 137: the twice-orbital-rate circle needs only the radial hover thrust
  (`test_fehse_twice_rate_circle_needs_only_the_rbar_hover_force`). After a thrust loss 180, 150,
  120 or 90 deg from the start, the free ellipse stays between $R$ and 2, 4.145751, 6.203627 and
  7 $R$, which the book rounds to 4.15 and 6.20
  (`test_thrust_loss_on_the_twice_rate_circle_never_comes_closer_than_its_radius`,
  `test_free_ellipse_after_torva_thrust_loss_never_comes_closer_than_the_range_at_loss`). The
  flown quarter turn takes $T/8$ (`test_torva_reaches_plus_vbar_after_an_eighth_of_an_orbit`).
- Pp. 150 and 154: the STORRM terminal-phase burn at 42 deg elevation is mostly radial up
  (`test_terminal_phase_burn_at_42_deg_elevation_is_mostly_radial_up`).
- P. 143, STS-92: pilot pulses of 0.1 ft/s (`test_each_pilot_pulse_delivers_0_1_ft_per_s`), and
  12 closing pulses add up to 1.2 ft/s
  (`test_twelve_pulses_from_far_behind_add_up_to_1_2_ft_per_s`); the gate never drops below the
  0.1 ft/s final rate of Table 25.8 (`test_pilot_gate_never_drops_below_0_1_ft_per_s`).

**Hablani, Tapper and Dana-Bashian, "Guidance and control of an autonomous rendezvous
spacecraft"** (*Journal of Guidance, Control, and Dynamics* 25(3), 2002), as reproduced in Yuan,
Li and Wang (*Aerospace* 9, 242, 2022), eqs. 14 to 25

- The glideslope profile $\dot\rho = a\rho + b$ arrives at $T = \ln(\dot\rho_T/\dot\rho_0)/a$,
  $1000\ln 11$ s from 200 m at -0.22 m/s to -0.02 m/s
  (`test_the_profile_closes_at_a_negative_rate_and_arrives_at_the_closed_form_time`), and meets
  its boundary range and rates (`test_the_profile_meets_its_boundary_range_and_rates`).

## Published examples behind the evaluation harness

- Kochenderfer, Wheeler and Wray, *Algorithms for Decision Making* (2022), sec. 14.1: the
  Clopper-Pearson intervals for 3 failures in 10,000 runs, (6.187e-5, 8.765e-4), and for 90 of 100,
  (0.8238, 0.9510), and the normal interval's negative lower end for the rare case, -3.945e-5
  (`tests/test_eval_tails.py`).
- Kochenderfer, Katz, Corso and Moss, *Algorithms for Validation* (2026), secs. 7.1 and 12.2: the
  one-sided (95%, 95%) tolerance order statistics, the 2nd of 100 runs (confidence 0.963) and the
  39th of 1000 (0.957); the zero-failure counts 59, 230 and 299 for 95/95, 99/90 and 99/95; the
  probability that a split-conformal bound covers less than 95% given its calibration runs, 0.436
  at 100 runs and 0.480 at 1000 (`tests/test_eval_tails.py`).
- Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice" (NeurIPS
  2021): the interquartile mean, the stratified bootstrap and the probability of improvement as the
  Mann-Whitney fraction (`tests/test_eval_stats.py`).
- Hamilton, Dunlap and Hobbs (2024, arXiv 2405.12355), van Wijk, Dunlap, Majji and Hobbs (AAS 23-342,
  2023, arXiv 2308.02743) and Supreeth and Mehta (CVPR 2026 Workshops): every constant of the four
  comparison tasks, and the inspection paper's minimum completion time of about 3060 s, pi / n,
  reproduced at 3060 s (`tests/test_comparison_tasks.py`).

## Published problem behind the pursuit-evasion game

- KSPDG v0.15.0 (github.com/mit-ll/spacegym-kspdg, commit 6251ca3), public
  source only: the pe1 observation, score, 240 s timeout, 5 m capture distance and RCS model
  (`src/kspdg/pe1/pe1_base.py`), the action dictionary and frames (`base_envs.py`), the agent and
  runner signatures (`agent_api/`), the I3 orbits, parts and propellant of the mission save
  `pe1_i3_init.sfs`, and the e3 evasion distance and throttle (`e3_envs.py`). The I3 range
  a dM = 2617.994 m and the burn's propellant use 8000 N / (240 s g0) are reproduced
  (`tests/test_pursuit_evasion.py`).
- KSP's resource definitions (`ResourcesGeneric.cfg`): monopropellant density 0.004 t per unit.
- Carrasco, Rodriguez-Fernandez and Linares (arXiv 2408.08676, 2024): the PE1_I3_E3 scenario as
  the one their LLM agents were evaluated on.
## Public reports behind the mission reconstructions

Facts only, each cited in [reconstructions.md](reconstructions.md); none is an operator's data.

- Ellis, "Collision in space" (NASA Ames, 2004 print): the Progress M-34 takeover at 6 km, the
  closing rate of 6.5 m/s against a planned 5.0 m/s, the aim point 1000 m behind the station and the
  400 m out-of-plane offset, which set the start of `recon-progress-m34`; its arc through the aim
  point is checked on 20 validation seeds (`tests/test_reconstructions.py`).
- NASA, "Overview of the DART Mishap Investigation Results" (2006), and Howard and Bryan, "DART AVGS
  Performance" (NTRS 20070031729): the 1 km start, the 6.3 m waypoint sphere 200 m behind the
  target, the 0.6 m/s GPS velocity bias and the 50 ft station-keeping point of `recon-dart`; the
  bias ending exactly on sphere entry is tested (`tests/test_reconstructions.py`).
- Kawano et al., J. JSASS 50(578) (2002), and Oda, i-SAIRAS 2001: the 520 m V-bar start from ahead,
  the 2 m holding point held about 15 min, the 550 km orbit and the 2.5 t chaser of `recon-ets7`.

## Agreement with Basilisk and Orekit

V-BAR's two-body plus J2 rung, Basilisk 2.12 and Orekit 13.1 fly the same initial inertial states
with the same constants (`benchmarks/agreement_table.py`, [backends.md](backends.md)); the largest
Hill-frame differences over each arc, sampled every 100 s for the coasts and 10 s for the episode:

| Case | Backend | Position (m) | Velocity (m/s) | Loosened backend (m) | Loosened V-BAR (m) |
|---|---|---|---|---|---|
| coast 100 m, 1 orbit | Basilisk | 6.28e-08 | 4.05e-11 | 6.28e-08 | 6.28e-08 |
| coast 100 m, 1 orbit | Orekit | 3.43e-08 | 3.21e-11 | 1.35e-07 | 3.42e-08 |
| coast 100 m, 10 orbits | Basilisk | 5.20e-06 | 2.92e-10 | 5.20e-06 | 5.20e-06 |
| coast 100 m, 10 orbits | Orekit | 7.44e-07 | 2.02e-10 | 2.37e-06 | 7.43e-07 |
| coast 1000 m, 1 orbit | Basilisk | 1.23e-07 | 6.93e-11 | 1.23e-07 | 1.23e-07 |
| coast 1000 m, 1 orbit | Orekit | 2.27e-08 | 1.80e-11 | 8.51e-08 | 2.26e-08 |
| coast 1000 m, 10 orbits | Basilisk | 1.32e-06 | 1.52e-10 | 1.32e-06 | 1.32e-06 |
| coast 1000 m, 10 orbits | Orekit | 3.09e-06 | 3.28e-10 | 4.04e-06 | 3.09e-06 |
| coast 10000 m, 1 orbit | Basilisk | 1.31e-07 | 5.03e-11 | 1.31e-07 | 1.23e-07 |
| coast 10000 m, 1 orbit | Orekit | 3.90e-08 | 2.76e-11 | 5.98e-08 | 4.98e-08 |
| coast 10000 m, 10 orbits | Basilisk | 5.72e-06 | 4.99e-10 | 5.72e-06 | 5.45e-06 |
| coast 10000 m, 10 orbits | Orekit | 8.09e-07 | 3.81e-10 | 6.05e-06 | 9.99e-07 |

Every coast agrees to 6e-6 m over ten orbits and 1.4e-7 m over one, at every range: the residual
is the integrators' and does not grow with the range, so no force-model difference shows. The
last two columns repeat the arc with one tolerance loosened. Orekit's residual grows by up to a
factor of 7.5 when its own tolerance is loosened from 1e-6 m to 1e-4 m, so its integrator sets it.
Loosening Basilisk's (from 1e-12 to 1e-10, inside a fixed 10 s task step) changes nothing, and
loosening V-BAR's (rtol from 1e-12 to 1e-10) changes the residual by at most 6% against Basilisk
and 28% against Orekit, at 10 km: V-BAR's own error is a smaller share. The frame conversion is
one formula applied to every tool's states, so it adds nothing.

| Case | Backend | Position, 1 s substep (m) | Position, 0.5 s substep (m) | Velocity, 1 s (m/s) |
|---|---|---|---|---|
| PD episode on the V-bar approach, 62 burns | Basilisk | 1.07e-04 | 3.17e-05 | 2.56e-07 |
| PD episode on the V-bar approach, 62 burns | Orekit | 1.07e-04 | 3.17e-05 | 2.56e-07 |

Replayed open loop, the episode agrees to 1.1e-4 m with both tools, and the residual falls by a
factor of 3.4 when the substep over which a backend holds a burn's inertial direction halves (the
second-order term of that approximation), identically in both: it comes from that approximation,
not from V-BAR. The tests assert a one-orbit coast within 1e-4 m and a burn replay within 1e-3 m
whose residual falls by more than three when the substep halves (`tests/test_backends_external.py`,
run when the extras are installed).

## Coverage of the scoring intervals

Synthetic outcomes with a known truth, 1000 experiments per cell, each interval at 95%
(`benchmarks/coverage_table.py`; the cells at 50 and 100 runs and the narrowed interval are
asserted in `tests/test_eval_coverage.py`). An interval holds when its coverage is within three
binomial standard errors (2.1 points) of 95%. Success and keep-out rates use Clopper-Pearson;
the continuous metric (normal, mean 2, sd 1) and the paired difference (two controllers sharing
90% of each seed's variance, true difference 0.3) use the percentile bootstrap of the IQM with
1000 resamples; McNemar's exact test compares two equal controllers (success 0.8) flown on the
same seeds.

| Runs per cell | Success 0.9 | Keep-out 0.05 | IQM | Paired difference | McNemar false positives |
|---|---|---|---|---|---|
| 5 | 0.992 | 0.980 | 0.889 (short) | 0.886 (short) | 0.000 |
| 10 | 0.989 | 0.992 | 0.927 (short) | 0.925 (short) | 0.000 |
| 20 | 0.991 | 0.984 | 0.939 | 0.943 | 0.000 |
| 50 | 0.976 | 0.992 | 0.938 | 0.945 | 0.014 |
| 100 | 0.962 | 0.987 | 0.948 | 0.954 | 0.034 |

The exact rate intervals cover at every count (they are conservative); the bootstrap intervals
fall short below 20 runs, so a leaderboard row needs at least 20 runs per cell for its IQM
interval to mean what it says, and the default of 100 holds with room. McNemar's false-positive
rate stays below 5% at every count. An IQM interval narrowed to 0.7 of its width covers 0.849 at
100 runs, so the test catches an interval that is too narrow.
## Examples behind the oracles

The tests of the primer-vector certificates ([oracle-certificates.md](oracle-certificates.md),
`tests/test_oracle_certificate.py`) reproduce the published examples below and check the
constructed cases. Values marked independent were
computed in a separate implementation, not with the module under test. Each item names its test.

**Fehse, *Automated Rendezvous and Docking of Spacecraft***, section 3.3, at 500 km with WGS 84
constants

- Eqs. 3.29 to 3.32 and 3.37: from 500 m behind the target to the target point, in one orbit with
  two tangential burns of $n\Delta y/(6\pi)$ or in half an orbit with two radial burns of
  $n\Delta y/4$. Both certify as optimal for box thrusters, with the primer peak in
  $[1 - 10^{-12}, 1 + 10^{-9}]$ and costs $n\Delta y/(3\pi)$ and $n\Delta y/2$ to $10^{-12}$
  (`test_fehse_transfers_certify_as_box_optimal`). On the tangential transfer every certifying
  primer reaches a 2-norm of at least 1.0014, so the wrong dual norm would reject this optimum.
  The radial transfer also certifies for one engine
  (`test_radial_transfer_certifies_for_a_single_engine`); with its second burn 1% too large it
  misses the target and is rejected (`test_plan_that_misses_the_target_is_not_certified`).

**Alfriend, Vadali, Gurfil, How and Breger, *Spacecraft Formation Flying***, Example 5.2, in units
with $n = 1$, from (0.01, 0.02, 0.015, 0.001, 0.001, 0.001) to the origin at rest

- The least two-impulse transfer, at $nt_f$ = 4.6404, costs 0.0357202 in the 2-norm and 0.0464255
  in the 1-norm (independent) (`test_cost_is_the_actuator_norm_of_the_impulses`). Its six burning
  components fix the multiplier; the primer equals the burn direction at both burns but peaks
  inside the transfer at 5.5929 for box thrusters and 1.9217 for one engine (independent), so the
  plan is optimal in neither norm, and its certified lower bound stays below 0.0377879, the cost
  of a 2000-interval grid plan
  (`test_best_two_impulse_plan_of_alfriend_example_5_2_is_not_optimal`). A 40-step fuel-oracle
  plan for one engine costs at most 0.96 of it, 0.0339611 (independent)
  (`test_a_verified_ball_plan_beats_the_two_impulse_plan_the_certificate_rejects`).
- The 1-norm optimum over $nT$ = 5 (independent) has four burns: along $y$ at 0 and 3.27422,
  along $z$ at $\pi - \arctan 15$ and along $x$ at 5. It certifies for box thrusters at a cost of
  0.0375601, with the coast Hamiltonian constant to $10^{-8}$; its primer at $T$ has a 2-norm of
  1.566, so a 2-norm check would reject it
  (`test_multi_impulse_optimum_of_alfriend_example_5_2_certifies`).

**Constructed cases on Lawden's conditions**, not book examples: the conditions from Lawden,
*Optimal Trajectories for Space Navigation* (1963), with their sufficiency on linear dynamics from
Longuski, Guzman and Prussing, *Optimal Control with Aerospace Applications* (2014), section 10.5

- The dual norm: burns along (cos 30 deg, 0, sin 30 deg) at 0 and (cos 150 deg, 0, sin 150 deg)
  at 1.8 from rest, $n = 1$, fix a primer that equals both burn directions to $10^{-9}$ and whose
  components stay below 0.8661 while its 2-norm reaches 1.1606 (independent); the one-engine
  certificate rejects the plan (`test_ball_certificate_measures_the_primer_in_the_2_norm`).
- A peak between samples: two $+z$ burns whose primer $A\cos(t - t_m)$, $A = 1 + 10^{-6}$,
  exceeds 1 only on the $2.8\times10^{-3}$ between them, where the largest of 1001 uniform
  samples is $1 + 9.994\times10^{-7}$ (independent). Both norms report a peak in
  $[A - 10^{-12}, A + 10^{-9}]$ and reject the plan (`test_primer_peak_between_burns_is_found`).
- A burn against the primer: from $v_z = -10^{-3}$ through the origin, $n = 1$, burns of
  $+2\times10^{-3}$ at 0 and $+10^{-3}$ at $\pi$ cost three times the one-burn optimum. No primer
  is $+1$ at both, so both norms reject the plan on direction, with a lower bound in
  $[0, 10^{-3}]$ (`test_burn_against_the_primer_fails_the_certificate`).

**A computed case**, not a book example: half an orbit of a 6778 km circle from (-200, -1500, 50) m
to (0, -100, 0) m at rest with a $2\times10^{-3}$ m/s^2 limit. The step-grid optima are LPs whose
equality multipliers give lower bounds by LP duality (Boyd and Vandenberghe, *Convex
Optimization*, 2004, sections 4.3 and 5.1.3).

- Box thrusters: the switching control gives 0.8947125 m/s to $10^{-7}$ (independent), below the
  160-step and 640-step LP optima, 0.8949323 and 0.8947293 m/s, by $2.5\times10^{-4}$ and
  $1.9\times10^{-5}$ of them (independent).
  Flown arc by arc it lands within $10^{-6}$ of the distance to cover, and the dual bound at 20000
  cells lies within $2\times10^{-3}$ below it
  (`test_indirect_box_optimum_agrees_with_the_direct_lp_optima`).
- One engine: 0.8289721 m/s to $10^{-6}$ (independent), below the 0.8324776 m/s of a 40-step
  plan, with the same miss check and the dual bound within $3\times10^{-3}$
  (`test_indirect_ball_optimum_lies_inside_the_lp_bracket`).
- The Hamiltonian along the box optimum spreads by at most $10^{-8}$ of its largest term; with the
  costate's sign flipped it spreads by 1.77 times that term (independent)
  (`test_hamiltonian_is_constant_along_the_indirect_optimum`).

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
  stage-structured QPs (sec. 8.8.3) and why an infeasible-start iterate is not a plan (sec. 6.1.2).
- Nocedal and Wright, *Numerical Optimization* (2nd ed., Springer, 2006), sec. 16.6: Mehrotra's
  predictor-corrector step used by the horizon solver.
- Boyd and Vandenberghe, *Convex Optimization* (2004): the KKT conditions behind the active-set
  projection (sec. 5.5.3), Farkas certificates of infeasibility (sec. 5.8.3), projection onto a
  halfspace (sec. 8.1.1), the sum-of-infeasibilities phase I and its conflicting rows
  (sec. 11.4.1), the primal-dual interior-point method and its line search (secs. 11.7 and 9.2),
  and the conditioning argument for nondimensional units (secs. 9.3 and 9.5).
- Ames, Coogan, Egerstedt, Notomista, Sreenath and Tabuada, "Control Barrier Functions: Theory and
  Applications" (ECC 2019): safety filters, the activity of their rows, and the conservatism of
  safe sets under bounded inputs.
  Automatic Control* 23, 1978): the added covariance of the `:nav2` error between observations
  and the integrated quadratic reward.
- Ng, Harada and Russell, "Policy invariance under reward transformations" (ICML 1999):
  potential-based shaping.
- Bradtke and Duff, "Reinforcement learning methods for continuous-time Markov decision
  problems" (NIPS 1995): discounting per second.
