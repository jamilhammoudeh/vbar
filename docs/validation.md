# Validation

The test suite (`python -m pytest -q`) checks the models against closed-form solutions, an
independent high-order integrator, universal-variable Kepler propagation, conserved quantities and
the published examples below. Each test names its source in a comment next to the check.

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
- NIMA TR8350.2, *Department of Defense World Geodetic System 1984*: Earth's GM, radius and J2.
