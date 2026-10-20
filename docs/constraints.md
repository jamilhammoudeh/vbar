# Constraints and the in-step minimum

Code: `vbar.constraints` (`Constraint`, the constraint types below, `segment_minimum`,
`segment_minima`, `interval_bounds`, `range_lower_bounds`, `violation_intervals`,
`sampled_negative_intervals`). A
constraint's margin is $\ge 0$ where it holds and $< 0$ where it is violated, in the constraint's
units, as in the quantitative semantics of signal temporal logic (Donzé and Maler, 2010). The
environment finds each constraint's lowest margin over every constant-acceleration segment of a
step, not only at the step's samples ([environment.md](environment.md)).

## Constraints

A constraint has a `name`, a `margin(states, t)` vectorized over stacked states `(..., 6)`, a
`rate_bound(speed, accel, rho_min)` that bounds $|dm/dt|$ given bounds on $|\mathbf v|$ and
$|\dot{\mathbf v}|$ and a lower bound on the range, and a `resolution` in margin units. With
$\rho = |\mathbf r|$ and $\dot\rho = \mathbf r \cdot \mathbf v / \rho$:

- `KeepOut(radius)`: stay outside a sphere around the target. Margin $\rho - r_{\text{ko}}$ (m),
  rate bound $|d\rho/dt| = |\mathbf r \cdot \mathbf v| / \rho \le |\mathbf v|$, resolution 1 mm.
- `SpeedLimit(v0, v1)`: margin $v_0 + v_1\rho - |\mathbf v|$ (m/s), the distance-dependent limit of
  Dunlap et al., "Comparing Run Time Assurance Approaches for Safe Spacecraft Docking" (2021),
  eq. 14, which uses $v_0$ = 0.2 m/s and $v_1 = 4n$. Rate bound $|v_1| V + A$, resolution
  1e-5 m/s.
- `ClosingSpeedLimit(v0, v1)`: margin $v_0 + v_1\rho - \max(0, -\dot\rho)$ (m/s), the same limit on
  the speed toward the target only (`closing_speed`). Since $\ddot\rho = (|\mathbf v|^2 -
  \dot\rho^2)/\rho + \hat{\mathbf r}\cdot\dot{\mathbf v}$, the rate bound is $|v_1| V + V^2 /
  \rho_{\min} + A$, infinite when $\rho_{\min}$ = 0. Resolution 1e-5 m/s.
- `GateTable(gates)`: the closing speed below a table of range-rate gates, given as (range m,
  closing speed m/s) pairs, the form the Space Shuttle flew (Goodman, *History of Space Shuttle
  Rendezvous*, JSC-63400, 2011, table 24.3 and figs. 14.12, 25.17 and 25.22). The allowed speed is
  linearly interpolated between gates and held at the end gates beyond the table; a speed of 0 is
  a hold. Margin allowed $- \max(0, -\dot\rho)$ (m/s); rate bound as `ClosingSpeedLimit` with
  $|v_1|$ replaced by the steepest slope between gates. Resolution 1e-5 m/s.
- `ApproachCone(half_angle, axis, apex, active_range)`: stay inside a cone whose apex is the mating
  point, since the corridor starts at the docking port or berthing box (Fehse, *Automated Rendezvous
  and Docking of Spacecraft*, 2003, sections 2.4.2 and 5.6; ISS corridors are 10 to 15 deg). With
  $\mathbf d = \mathbf r - \mathbf p$, $d = |\mathbf d|$ and $\varphi$ the angle between
  $\mathbf d$ and the axis $\hat{\mathbf a}$ (pointing from the apex toward the approach side), the
  signed distance to the cone of half-angle $\theta$ is
  $$
  s = \begin{cases} d \sin(\theta - \varphi) & \varphi < \theta + \pi/2 \\ -d & \text{otherwise}
  \end{cases}
  $$
  where the second case is the region whose nearest cone point is the apex (Boyd and Vandenberghe,
  *Convex Optimization*, exercise 8.3). The cone applies within `active_range` $R_c$ of the apex,
  and the margin is the robustness of "$d \le R_c$ implies inside the cone",
  $\max(d - R_c,\ s)$ (m), so it is continuous where the cone turns on. Both terms are 1-Lipschitz
  in position, so the rate bound is $V$; resolution 1 mm; $0 < \theta < \pi/2$.
- `SunExclusion(half_angle, sun_direction, rate)`: keep the Sun out of a sensor that looks from the
  chaser at the target. Margin $\angle(-\mathbf r, \hat{\mathbf s}(t)) - \theta_s$ (rad). The Sun is
  fixed in inertial space and the Hill frame turns at `rate` (the mean motion of a circular target
  orbit) about z, so $\hat{\mathbf s}(t) = R_z(-n t)\,\hat{\mathbf s}_0$. The line of sight turns
  at most $V/\rho$, so the rate bound is $V / \rho_{\min} + n$; resolution 1e-5 rad.
- `PassiveSpeed(axis, extent, n)`: the closing speed below which a thrust failure on a
  straight-line approach still misses structure `extent` $d$ m out, from Fehse's free-drift
  solutions (eqs. 3.22 and 3.28). On the V-bar a chaser whose thrust stops at closing speed $v$
  coasts forward at most $\kappa v / n$ before falling back, at $nt = \arccos(3/4)$, with
  $\kappa = \sqrt7 - 3\arccos(3/4) = 0.47755$; it stays clear while $v \le n(\rho - d)/\kappa
  = 2.094\,n(\rho - d)$. On the R-bar, from range $h$ at closing speed $v$ the closest radial
  approach is $4h - \sqrt{9h^2 + v^2/n^2}$, clear while
  $$
  v \le n\sqrt{(4h - d)^2 - 9h^2}, \qquad h > d,
  $$
  and zero for $h \le d$. With $d$ = 15 m at 400 km this gives 0.1225, 0.0612 and 0.0283 m/s at
  50, 30 and 20 m; Fehse's section 5.7.2 approach flies 0.1, 0.05 and 0.02 m/s there, inside the
  envelope. The margin is the envelope at $\rho$ minus the closing speed (m/s). The R-bar
  envelope's slope $n f'/(2\sqrt f)$, $f = 7h^2 - 8hd + d^2$, falls with $h$, so its rate bound is
  taken at $\rho_{\min}$ (infinite for $\rho_{\min} \le d$).
- `ZoneEntry(semi_axes, center, go_time)`: stay out of an ellipsoid until the GO event at
  `go_time`. With $q = |D^{-1}(\mathbf r - \mathbf c)|$, $D$ the semi-axes and $a_{\min}$ the
  smallest, the margin is $a_{\min}(q - 1)$ (m) up to and including `go_time`, so the step that
  ends at GO still sees the zone, and $+\infty$ after it, allowing $10^{-9}\max(1, |t_{\text{go}}|)$ s past
  `go_time` since the environment's time is a sum of steps. Since
  $|q(\mathbf x) - q(\mathbf y)| \le |\mathbf x - \mathbf y| / a_{\min}$, the margin is 1-Lipschitz
  (rate bound $V$) and never above the distance to the ellipsoid outside it, equal to it for a
  sphere. The ISS approach ellipsoid is $\pm$2000 m along-track and $\pm$1000 m otherwise
  (Fehse 5.6), `ZoneEntry((1000, 2000, 1000))`. The GO event must fall on a step boundary, since
  the rate bound does not cover a jump inside a segment.

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
whose rate grows as the range falls (closing speed, viewing angles) need it. Both bounds take stacked
trajectories, shape `(..., samples, 6)`, so many arcs are bounded in one call. In Clohessy-Wiltshire
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

## Batches of segments

`segment_minima(constraint, state_at, times, states, thrust, n, t0)` does the same for $N$
segments sampled at the same times, as the vectorized environment needs: `states` is
$(N, m, 6)$, `thrust` and `t0` are $(N,)$, and `state_at(rows, tau)` evaluates the segments
`rows` at the times `tau`. The bounds are the same. Since the best value only falls during
refinement, every interval the sequential rule could visit has a bound more than `resolution`
below the best sample; all of those, across the batch, are searched in one golden-section search
to 1e-6 s instead of one Brent search each. The rule is then replayed on the results: intervals
in order of their bound, stopping at the first within `resolution` of the best value so far. Each
segment's minimum therefore comes from the intervals `segment_minimum` would refine, and the two
agree to the accuracy of the searches; near a dip the margin's error is second order in the
bracket, about 1e-7 m or better. One search for the whole batch keeps the cost per step nearly
independent of the batch size. The constraint's `margin` and `rate_bound` must accept arrays, as
`KeepOut`'s do.

## Violation intervals

`violation_intervals(constraint, state_at, times, states, minimum, t0, thrust, n)` returns the
times inside a segment where the margin is negative, as (start, end) pairs in seconds from the
episode start. It takes the samples and the result of `segment_minimum`, with the segment's thrust
acceleration and the mean motion for the rate bounds, and returns nothing when the certified lower
bound is not negative. The search, `sampled_negative_intervals`, is the one the metrics use
(`vbar.metrics.negative_intervals`), so the reward and the violation metrics see the same
intervals. Each boundary is the root of the margin on a sample interval whose ends have opposite
signs, found with Brent's method (`scipy.optimize.brentq`) to 1e-9 s. A sample interval whose ends
share a sign is searched for a dip below zero, or a rise above it, only when its Lipschitz bound
allows one; like the refinement, this assumes at most one dip per sample interval. An interval
open at the segment's start or end runs to that end. The environment merges intervals that meet at
a segment boundary, so a pass through the keep-out zone during an actuation delay's switch is one
interval. The reward library builds its constraint costs from these intervals
([rewards.md](rewards.md)).

## In the environment

Each step's per-constraint minima (time, margin, lower bound) are in
`env.record.segment.minima`, its violation intervals in `env.record.segment.violations`, and the
episode's lowest values in `info["min_margins"]` and `info["min_margin_bounds"]`. The approach scenario counts a step as entering the keep-out zone when
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
- Bounds on four stacked trajectories with different thrust equal the bounds on each one.
- `tests/test_constraint_types.py`: for every constraint type, over 12 flybys (coasting and
  thrusting at 0.1 to 3 m/s) the certificate is finite and lies below the minimum of 4001 dense
  samples, and the refined minimum is within 1e-6 of it. Hand-built states check each margin
  formula.
- The cone margin equals the signed distance found by a direct search over the cone's generators
  (Nelder-Mead from 63 starts) to 1e-6 m for 12 random cones and points, including points behind
  the apex; it is continuous across the active range.
- The sun-exclusion margin is checked looking into the Sun and from above, and the Sun direction
  turns from +y to +x in a quarter orbit.
- The R-bar envelope gives 0.1225, 0.0612 and 0.0283 m/s at 50, 30 and 20 m (Fehse 5.7.2, from eq. 3.22);
  free drift from the envelope at 20, 50 and 300 m comes within 15 m of the target to 1e-8
  relative over 200001 samples of one orbit. The V-bar coast at 0.1 m/s and 400 km is 42.21 m
  (eq. 3.28), and free drift from the V-bar envelope with gaps of 5, 50 and 500 m reaches the structure.
- The zone margin is zero on the ISS ellipsoid, infinite after GO, below the distance to 8 random
  ellipsoids (200000 surface samples) and equal to it for a sphere.
- A constant margin on the V-bar is not refined.
- Over 40 random flybys sampled 20 times or only at their ends, the batched minimum agrees with
  the scalar one to 1e-7 m and its bound to 1e-12.
- Violation intervals of coasting passes match the entry and exit times of the closed-form
  transition matrix to 1e-8 s, including a pass seen only by the refinement between two samples
  and intervals open at a segment's start or end; a pass across an actuation-delay switch is one
  interval within a step.

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
