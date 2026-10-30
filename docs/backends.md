# External propagators

`vbar.dynamics.backends` flies the same target and chaser as V-BAR's two-body plus J2 rung with
two independent tools, Basilisk and Orekit, so V-BAR's propagation can be checked against
software the field already trusts. Both are optional extras and never needed to run V-BAR:

```
pip install "vbar[basilisk]"
pip install "vbar[orekit]"    # also needs a Java runtime (11 or later) on JAVA_HOME
```

## What each backend does

`propagate(target, chaser, times, body, j2=True, burns=(), tolerance, substep, mass)` takes the
initial inertial states of both spacecraft (m, m/s), flies them under the central body's point
mass and, with `j2`, its J2 term, and returns both inertial states at `times`. Both backends use
V-BAR's constants for the body (`vbar.dynamics.bodies`: WGS 84 GM, equatorial radius and J2 for
the Earth), so a difference comes from the integrator or the force model's implementation, not
from the constants.

- **Basilisk** (AVS Lab, University of Colorado; ISC licence): two `spacecraft.Spacecraft` hubs
  under one central body whose gravity is a degree-2 spherical-harmonic model holding only the
  normalized coefficient C20 = -J2 / sqrt(5), integrated with Basilisk's adaptive RKF78 at the
  relative and absolute `tolerance` inside each task step of `substep` seconds.
- **Orekit** (through orekit-jpype; Apache 2.0 licence): one `NumericalPropagator` per
  spacecraft in GCRF with a Dormand-Prince 8(5,3) integrator whose tolerances come from Orekit's
  rule for a position error of `tolerance` metres, under `NewtonianAttraction` and
  `J2OnlyPerturbation` about the frame's z axis.

A burn (`Burn(start, duration, accel_hill)`) is a constant acceleration on the chaser held in the
target's Hill axes, as in V-BAR. Neither tool holds a force in a rotating frame, so each backend
sets the burn's inertial direction once per `substep`, from the target's Hill frame turned to the
substep's middle (the Hill axes turn about the orbit normal at h / r^2), and holds it for the
substep: an error second order in the angle the frame turns in a substep. Basilisk applies it as
an `extForceTorque` force on a hub of `mass`; Orekit as a `ConstantThrustManeuver` in inertial
attitude with a specific impulse of 1e12 s, so the mass does not change.

Both `propagate` functions and `vbar_arc` check their arguments first and raise `ValueError`
before the extra is imported, so the checks need neither tool installed. They refuse a state that
is not six finite real numbers, sample times that are empty, not 1-D, not finite, negative or not
in time order (a repeated time is fine), a `tolerance`, `substep`, `mass` or (for `vbar_arc`)
`rtol` that is not a finite positive number, and burns that are not `Burn` objects, start before
t = 0 or overlap by more than 1e-9 s; burns that touch are accepted, in any order. Overlapping
burns are refused because only one acceleration can act at a time, and the backends and `vbar_arc`
would not pick the same one.

`hill_states(target, chaser, body, j2)` turns the inertial states back into Hill relative states
with the same conversion V-BAR uses (`vbar.conventions.inertial_to_hill`, the frame's rate from the
target's gravity), so every tool's output is compared in one frame by one formula.

## The agreement table

`vbar.dynamics.backends.agreement` flies the same initial states with V-BAR (`vbar_arc`,
`TwoBodyDynamics` at `rtol` 1e-12) and each backend and reports the largest Hill position and
velocity differences over the arc. `benchmarks/agreement_table.py` regenerates the table in
[validation.md](validation.md): coast arcs from (0.1 r, -r, 0.05 r) m with an out-of-plane speed
of 1e-4 r m/s at r = 100 m, 1 km and 10 km, over one and ten orbits of a 500 km, 51.6 deg target,
and an episode of the PD baseline on the V-bar approach replayed open loop on its recorded thrust
segments (`episode_burns`). Each residual is attributed by changing one setting at a time:
loosening V-BAR's tolerance to 1e-10, the backend's (Basilisk 1e-10, Orekit 1e-4 m), and halving
the substep over which a burn's direction is held.

```python
import math

import numpy as np

from vbar.dynamics import EARTH
from vbar.dynamics.backends import Burn, hill_states
from vbar.dynamics.backends.agreement import vbar_arc

times = np.arange(0.0, 601.0, 100.0)
burns = [Burn(0.0, 50.0, (0.001, 0.0, 0.0))]
ours, target, chaser = vbar_arc(np.array([0.0, -1000.0, 0.0, 0.0, 0.0, 0.0]), times, burns)
# With the basilisk extra: from vbar.dynamics.backends import basilisk
# t, c = basilisk.propagate(target, chaser, times, EARTH, True, burns=burns)
# theirs = hill_states(t, c, EARTH, True)
assert ours.shape == (7, 6) and math.isclose(ours[0, 1], -1000.0)
```

## Assumptions and limits

- Only the central body's point mass and J2: no drag, solar pressure, third bodies or higher
  harmonics, on any side.
- The burns' held inertial direction is the backends' one approximation; it is second order in
  n times the substep and is measured in the table by halving the substep.
- The Orekit backend needs a Java runtime; `jdk4py` provides one when it is installed, and the
  backend uses it if `JAVA_HOME` is not set.

## Validation

`tests/test_backends.py` (no extras needed): the shared conversion returns V-BAR's own Hill
states, the held direction is turned to the hold's middle, V-BAR's arc splits at burn edges, an
episode's burns are its thrusting segments, malformed input and overlapping burns are refused
before the extra is imported, and a missing extra names its install command.
`tests/test_backends_external.py` (each with its extra): a one-orbit coast at 1 km agrees to
1e-4 m, the tolerance set before any run from the integrators' settings, and a burn replay agrees
to 1e-3 m with a residual that falls by more than a factor of three when the substep halves.
