# Solvers

Code: `vbar.solvers` (`HillScaling`). The
optimization problems that safety filters and horizon methods solve, built on numpy and scipy only,
deterministic and with a defined status for every outcome.

## Hill-frame units

`HillScaling(length, n)` gives nondimensional units for Hill-frame problems: length $L$ (m), time
$1/n$ (s), speed $nL$ and acceleration $n^2L$. Hill-frame quantities mix metres, seconds and a mean
motion near 1e-3 rad/s, so a problem posed in SI units has entries spread over many orders of
magnitude. Newton-type methods are affine invariant in exact arithmetic, but their stopping tests
and factorizations are not, and first-order methods stall on such conditioning (Boyd and
Vandenberghe, *Convex Optimization*, secs. 9.3 and 9.5). In these units positions, speeds and
accelerations of a proximity operation are of order one and the Clohessy-Wiltshire equations have
unit mean motion:

$$
\ddot x' - 2\dot y' - 3x' = a_x', \qquad \ddot y' + 2\dot x' = a_y', \qquad \ddot z' + z' = a_z',
$$

with $x' = x/L$, $t' = nt$ and $a' = a/(n^2L)$. `state` and `state_si` convert states of shape
`(..., 6)` (m, m/s), `accel` and `accel_si` accelerations of any shape (m/s^2), `time` and
`time_si` durations (s). $L$ and $n$ must be finite and positive; $L$ is usually the scenario's
range scale.

The tests check the units of each quantity, the round trips, that `cw_rhs` with $n = 1$ on scaled
states equals the scaled SI derivative, and that the exact transition over 1 ms, 10 s, 600 s and
5400 s in scaled units equals the SI propagation to 1e-12 relative.

```python
from vbar.dynamics import EARTH, mean_motion
from vbar.solvers import HillScaling

units = HillScaling(length=100.0, n=mean_motion(EARTH.radius + 400e3))
print(units.state([-100.0, -500.0, 0.0, 0.0, 0.2, 0.0]).round(4))  # 0.2 m/s is 1.7678 n L
print(round(float(units.accel(2e-3)), 3))  # 2 mm/s^2 is 15.625 n^2 L
print(round(units.time(600.0), 4))  # 600 s is 0.6788 rad of orbit
```
