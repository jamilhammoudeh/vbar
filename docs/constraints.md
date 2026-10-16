# Constraints and the in-step minimum

Code: `vbar.constraints` (`Constraint`, `KeepOut`, `segment_minimum`, `interval_bounds`,
`range_lower_bounds`). A
constraint's margin is $\ge 0$ where it holds and $< 0$ where it is violated, in the constraint's
units, as in the quantitative semantics of signal temporal logic (Donzé and Maler, 2010). The
environment finds each constraint's lowest margin over every constant-acceleration segment of a
step, not only at the step's samples ([environment.md](environment.md)).

## Constraints

A constraint has a `name`, a `margin(states, t)` vectorized over stacked states `(..., 6)`, a
`rate_bound(speed, accel, rho_min)` that bounds $|dm/dt|$ given bounds on $|\mathbf v|$ and
$|\dot{\mathbf v}|$ and a lower bound on the range, and a `resolution` in margin units. The first
one is:

- `KeepOut(radius)`: stay outside a sphere around the target. Margin $\rho - r_{\text{ko}}$ (m),
  rate bound $|d\rho/dt| = |\mathbf r \cdot \mathbf v| / \rho \le |\mathbf v|$, resolution 1 mm.

## Sampling and refinement

`segment_minimum(constraint, state_at, times, states, thrust, n, t0)` takes evenly spaced samples
of one segment, endpoints included; the environment uses $\max(2, \operatorname{round}(N_s\,\tau /
\Delta t))$ of them for a segment of $\tau$ s, with $N_s$ = `violation_samples` = 20 per full step.
The lowest sample is the first estimate. Sample intervals flagged by the bound below are refined
with a bounded Brent search (`scipy.optimize.minimize_scalar`, method `"bounded"`, bracket
tolerance 1e-5 s) on `state_at(tau)`, the dynamics model's state inside the segment: the exact
closed form for Clohessy-Wiltshire and the integrator's dense output for the two-body models.

## The Lipschitz certificate

On a sample interval $[t_i, t_{i+1}]$ of length $h$ where $|dm/dt| \le L$, the margin lies above
both $m_i - L(t - t_i)$ and $m_{i+1} - L(t_{i+1} - t)$, so its lowest value is at least where the
two lines meet:

$$
m(t) \ge \tfrac12\,(m_i + m_{i+1} - L h) \ge \min_i m_i - \tfrac12 L h
$$

$L$ is `rate_bound` evaluated on bounds $V$ and $A$ for the speed and the acceleration over the
interval. The same two-line argument gives $|\mathbf v| \le \tfrac12(|\mathbf v_i| + |\mathbf
v_{i+1}| + A h)$, $\rho \le \max(\rho_i, \rho_{i+1}) + \tfrac12 V h$ and $\rho \ge \max(0,
\tfrac12(\rho_i + \rho_{i+1} - V h))$. The last is the `rho_min` passed to `rate_bound`; constraints
whose rate grows as the range falls (closing speed, viewing angles) need it. In Clohessy-Wiltshire
motion the natural acceleration $(3n^2x + 2n\dot y,\ -2n\dot x,\ -n^2 z)$ is at most
$3n^2\rho + 2n|\mathbf v|$, so with thrust acceleration $a_T$ and a safety factor $c$ = 2

$$
A = a_T + c\,(3n^2\rho + 2n V), \qquad
V = \frac{\bar v + \tfrac12 h\,(a_T + 3cn^2\rho_{\text{end}})}{1 - g}, \qquad
g = \tfrac12\,c\,h\,(\tfrac32 n^2 h + 2n)
$$

with $\bar v$ the mean of the two end speeds and $\rho_{\text{end}}$ the larger end range. This
solves the linear inequality the true maximum speed satisfies, so it holds when $g < 1$. For longer
intervals, $h$ above about 350 s at 500 km, the bounds are infinite and so is the certificate
(`lower_bound` is $-\infty$); every interval is then refined.

**Which intervals are refined.** Intervals are visited in order of their bound, and each whose
bound is more than `resolution` below the best value found so far is refined; the first that is
not ends the search. A pass between two samples cannot go unnoticed, because its interval's bound
falls below the samples on either side.

**What is guaranteed.** `lower_bound` is a valid lower bound on the margin of the simulated
trajectory over the segment. The reported `margin` is the lowest value found: the true minimum
whenever each refined interval holds a single dip. Refinement does not tighten the bound, which
can stay up to $\tfrac12 L h$ below the reported minimum; over 200 coasting flybys at 0.5 to
20 m/s sampled 20 times per 10 s the gap has a median of 1.7 m and a maximum of 5.1 m.

## In the environment

Each step's per-constraint minima (time, margin, lower bound) are in
`env.record.segment.minima`, and the episode's lowest values in `info["min_margins"]` and
`info["min_margin_bounds"]`. The approach scenario counts a step as entering the keep-out zone when
its minimum margin is negative.

## Assumptions and limits

- $c$ = 2 covers the two-body nonlinearity and J2 at separations below 1% of the orbit radius,
  about 69 km at 500 km; this is not checked at run time.
- Forces not proportional to range or speed, such as differential drag or solar pressure
  ([perturbations.md](perturbations.md)), are not in $A$. The environment's dynamics models do not
  include them.
- The certificate and the refinement apply to the simulated trajectory, not to the errors of the
  dynamics model itself.

## How it is tested

`tests/test_constraints.py` and `tests/test_base_env.py`:

- Over 20 random Clohessy-Wiltshire flybys, coasting or thrusting, the refined minimum matches the
  exact one (closed-form transition matrix, or DOP853 at a relative tolerance of 1e-13, with the
  root of $\mathbf r \cdot \mathbf v$) to 3.2e-9 m against a test bound of 1 mm; the best of the
  20 samples alone is up to 1.7 m off.
- Over 20 more flybys the certificate lies below the exact minimum and is never looser than
  $\min_i m_i - \tfrac12 L h$ with the largest $L$.
- A 12 m/s pass sampled only at the two ends of a 10 s segment is flagged by the bound and found
  by refinement.
- On two-body motion, with and without J2, the speed bounds hold against 3801 dense samples per
  minute and the certificate lies below the dense minimum.
- With 1000 s between samples the bound is infinite, not wrong, and refinement still finds the
  minimum to 1 mm.
- Over 10 flybys the range bound lies below the range of 1901 dense samples on every interval,
  and it is zero when the speed bound is infinite.
- A constant margin on the V-bar is not refined.

## Example

```python
import numpy as np

from vbar.constraints import KeepOut, segment_minimum
from vbar.dynamics import EARTH, CWDynamics, mean_motion

n = mean_motion(EARTH.radius + 500e3)
model = CWDynamics(n)
model.reset([0.0, 60.0, 5.0, 0.0, -12.0, 0.0])  # a 12 m/s pass 5 m from the target
_, samples = model.advance(np.zeros(3), 10.0, n_samples=20)
times = np.linspace(0.0, 10.0, 20)
keep_out = KeepOut(50.0)
found = segment_minimum(keep_out, model.state_in_last_segment, times, samples, 0.0, n)
print(keep_out.margin(samples).min().round(3))  # -44.078 m, the best sample
print(round(found.margin, 3), round(found.t, 3))  # -44.989 m at 5.0 s
print(found.lower_bound <= found.margin)  # True
```
