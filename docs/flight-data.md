# Flight-data replay

Code: `vbar.flightdata`, `TwoBodyDynamics.reset_inertial`. Two spacecraft's precise orbits from
flight are replayed through V-BAR's relative-motion models: the first flown sample starts the
model, and the model's relative state is compared with the flown one at every later sample.

The flown pair is GRACE-FO on 3 December 2019: two satellites in one near-polar orbit whose
semi-major axis is 483 km above the equatorial radius, 182 km apart, their positions known from GPS
orbit determination to millimetres. **At this separation the replay validates the nonlinear models
(two-body, J2, drag), not Clohessy-Wiltshire.** The CW model's curvature error rho^2 / (2 r) is 2.4
km here, and the run below shows CW off by more than 10 km within an orbit. No public dataset of a
formation under 1 km with open terms was found.

## Data

| Item | Value |
|---|---|
| Collection | GRACEFO_L1B_ASCII_GRAV_JPL_RL04, version 4, PO.DAAC, doi:10.5067/GFL1B-ASJ04 (the files' own product id is 10.5067/GFJPL-L1B04) |
| Licence | CC0, under the NASA Earthdata data use policy for NASA-led missions; no endorsement by NASA is implied |
| Citation | GRACE-FO (2019), GRACEFO_L1B_ASCII_GRAV_JPL_RL04, ver. 4, PO.DAAC, CA, USA, doi:10.5067/GFL1B-ASJ04 |
| Day | 2019-12-03, daily bundle `gracefo_1B_2019-12-03_RL04.ascii.noLRI.tgz` (Earthdata login needed to download) |
| Products used | GNV1B (1 Hz states in the Earth-fixed frame, ITRF2014) for GRACE-C (target) and GRACE-D (chaser); KBR1B (K-band range every 5 s) |
| Why this day | after the 2019-06-16 correction of an antenna calibration error; no orbit-control thruster firing on either satellite (THR1B); no events in the mission's event notes for 1 to 3 December; one continuous K-band segment for the day; geomagnetically quiet |
| Excerpt installed | `src/vbar/flightdata/gracefo_2019-12-03.csv`, installed with the package and read by `gracefo_excerpt()` (`tests/test_package_data.py`): every 120 s of the first 16 h, both satellites' Earth-fixed states and the corrected K-band range (89 kB). `examples/gracefo_excerpt.py` writes it from the extracted bundle. |

The full daily files are not part of V-BAR.

## Method

The Earth-fixed states are turned inertial by the Earth's rotation alone, about the Earth-fixed z
axis at the WGS 84 rate from the Greenwich mean sidereal angle of the first sample
(`earth_fixed_to_inertial`, `gmst`): r_i = Rz(theta) r and v_i = Rz(theta) (v + w x r), theta =
theta0 + w t. The frame's z axis is the axis J2 is symmetric about, to the polar motion (about
1.5e-6 rad). Precession and nutation turn it against the stars at about 1e-11 rad/s, a fictitious
relative acceleration below 1e-8 m/s^2 at this separation. The inertial-frame product GNI1B is not
used: J2 acts about the Earth's spin axis of date, which the Earth-fixed frame carries directly.

`replay(t, target, chaser, dynamics)` starts the model with `reset_inertial` from the first sample
and coasts it from sample to sample. Flown and modelled relative states are both expressed in the
Hill frame of their target orbit (`hill_states`), with the frame's rate about the radial axis from
point-mass plus J2 gravity, so the two are compared in one convention.

The rungs: `TwoBodyDynamics()` (point mass), `TwoBodyDynamics(j2=True)`, and the J2 rung with
`Drag(HarrisPriester(sun, 6), B, B)`, the Sun direction from the low-precision solar position of
Montenbruck and Gill (Satellite Orbits, eqs 3.43 and 3.44; `sun_direction`) and B = 601.2 kg /
(2.3 x 0.955 m^2) for both satellites (launch mass and front area from the GRACE-FO Level-1
handbook, table 5; the drag coefficient 2.3 is an assumption). The CW rung is
`cw_stm(n, t) @ x0` from the first flown relative state.

## The bound, stated before the run

The bound was fixed before the flight data were read, from a budget of what the models leave out:

- **Geopotential beyond J2.** A simulation with V-BAR's point mass and J2 plus the EGM96
  harmonics to degree 30 as the truth, for a GRACE-FO-like pair 220 km apart at 485 km and 89 deg
  over six starting phases, gave a J2-rung error of at most 42, 81, 211 and 440 m after 1, 2, 5 and
  10 orbits; degree 45 changed it by under 2%. Its envelope is 20 m + 44 m per orbit; with a factor
  1.5 for geometry outside the sweep, **30 m + 66 m per orbit**.
- **Everything else, as a bounded acceleration** A = 5e-8 m/s^2: drag model error (the whole drag
  on one satellite at solar minimum, 2e-8), lunar and solar tides on average along the separation
  (2.5e-8), the frame's precession (0.5e-8). Its effect is bounded by A times the integral of the
  largest singular value of the CW block that maps a velocity change to a later position
  (`error_envelope`).

So B(t) = 30 m + 66 m per orbit + A int_0^t ||Phi_rv(s)|| ds, about 99 m after one orbit and 930 m
after ten at this orbit's mean motion. With it: the error's least-squares growth over ten orbits at
most 95 m per orbit; the J2 rung at least five times better than two-body in the first orbit; drag
changing the J2 result by at most 100 m; CW off by more than 10 km after one orbit; and the
GPS-derived range agreeing with the K-band range to under 1 cm RMS.

## Result

| Rung | Largest relative position error after 1 / 2 / 5 / 10 orbits | Growth (least squares) | Inside B |
|---|---|---|---|
| two-body | 2911 / 5820 / 14551 / 29115 m | 2910 m per orbit | no |
| J2 | 53 / 106 / 254 / 501 m | 50 m per orbit | yes, by at least 28 m |
| J2 and drag | 53 / 106 / 254 / 501 m | 50 m per orbit | yes |
| Clohessy-Wiltshire | 89 km within the first orbit | | no |

The relative velocity error of the J2 rung after ten orbits is 4 mm/s. Drag changes the J2 result
by 0.13 m: the twins have the same ballistic coefficient and fly one orbit, so their drag nearly
cancels in the relative motion. The J2 rung's error is of the size the geopotential budget
predicted, which the models leave out.

The K-band check is independent of the GPS orbits: the corrected K-band range and the range
between the two GNV1B positions differ by a constant (the K-band bias) plus 1.3 mm RMS over the
16 h, and the modelled range change follows the K-band range change inside B.

The test asserts the bound, and pins this run's numbers as a regression record (to 1%, the J2
rung's numbers and the bound's values to 0.5 m), so a change to the models that moves them shows.

## Validation

`tests/test_flightdata.py`:

- Known answers: a point fixed on the Earth turns with it, at the right speed; the sidereal angle
  returns after one sidereal day; the Sun's direction at the December 2019 solstice, the March 2020
  equinox and the June 2020 solstice (times from the US Naval Observatory) within 0.05 deg in
  declination and 0.1 deg in right ascension; the envelope's force term follows its short-time
  limit A t^2 / 2.
- The reader takes column names from the file's header and refuses a file without a header end
  or with a short record; the excerpt round trip keeps the written precision; the fixture is the
  cited day at 120 s steps with its DOI and licence.
- Fast: the GPS range agrees with the K-band range to under 1 cm RMS; the J2 rung stays inside B
  over two orbits.
- Slow (`-m slow`): every bound above over ten orbits, for the J2 and J2-plus-drag rungs, and
  the result table as a regression record.

## Limits

- One day and one pair. Solar activity was low (December 2019); drag would matter more at solar
  maximum, but nearly cancels for identical satellites in one orbit either way.
- The geopotential beyond J2 is not in V-BAR; at 182 km it costs tens of metres per orbit
  ([perturbations.md](perturbations.md) gives its size at 1 km).
- The bound's geopotential term comes from a simulation against EGM96 over six phases, not a
  proof.

## Example

Replay the J2 rung over the first orbit of the excerpt:

```python
import numpy as np

from vbar.dynamics import EARTH, TwoBodyDynamics
from vbar.flightdata import earth_fixed_to_inertial, gmst, gracefo_excerpt, replay

excerpt = gracefo_excerpt()
t = excerpt["t"] - excerpt["t"][0]
theta0 = float(gmst(excerpt["t"][0]))
target = np.hstack(earth_fixed_to_inertial(t, *np.hsplit(excerpt["target"], 2), theta0))
chaser = np.hstack(earth_fixed_to_inertial(t, *np.hsplit(excerpt["chaser"], 2), theta0))
first = t <= 5700.0
result = replay(t[first], target[first], chaser[first], TwoBodyDynamics(EARTH, j2=True))
print(f"{result.position_error.max():.1f} m after one orbit")
```
