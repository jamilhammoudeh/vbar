# Constraints and the in-step minimum

Code: `vbar.constraints` (`Constraint`, the constraint types below, `segment_minimum`,
`segment_minima`, `interval_bounds`, `range_lower_bounds`, `violation_intervals`,
`sampled_negative_intervals`). A constraint's margin is $\ge 0$ where it holds and $< 0$ where it is
violated, in the constraint's units, as in the quantitative semantics of signal temporal logic
(Donzé and Maler, 2010). The environment finds each constraint's lowest margin over every
constant-acceleration segment of a step, not only at the step's samples
([environment.md](environment.md)).

## Constraints

A constraint has a `name`, a `margin(states, t)` vectorized over stacked states `(..., 6)`, a
`rate_bound(speed, accel, rho_min)` that bounds $|dm/dt|$ given bounds on $|\mathbf v|$ and
$|\dot{\mathbf v}|$ and a lower bound on the range, and a `resolution` in margin units. With
$\rho = |\mathbf r|$ and $\dot\rho = \mathbf r \cdot \mathbf v / \rho$:

- `KeepOut(radius)`: stay outside a sphere around the target. Margin $\rho - r_{\text{ko}}$ (m),
  rate bound $|d\rho/dt| = |\mathbf r \cdot \mathbf v| / \rho \le |\mathbf v|$, resolution 1 mm;
  it also carries the second-order certificate below.
- `SpeedLimit(v0, v1)`: margin $v_0 + v_1\rho - |\mathbf v|$ (m/s), the distance-dependent limit of
  Dunlap et al., "Comparing Run Time Assurance Approaches for Safe Spacecraft Docking" (2021),
  eq. 14, which uses $v_0$ = 0.2 m/s and $v_1 = 4n$. Rate bound $|v_1| V + A$, resolution
  1e-5 m/s; it also carries a second-order certificate, as do the next two.
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
- `ZoneEntry(semi_axes, center, go_time)`: stay out of an ellipsoid until the GO event at `go_time`.
  With $q = |D^{-1}(\mathbf r - \mathbf c)|$, $D$ the semi-axes and $a_{\min}$ the smallest, the
  margin is $a_{\min}(q - 1)$ (m) up to and including `go_time`, so the step that ends at GO still
  sees the zone, and $+\infty$ after it, allowing $10^{-9}\max(1, |t_{\text{go}}|)$ s past `go_time`
  since the environment's time is a sum of steps. Since
  $|q(\mathbf x) - q(\mathbf y)| \le |\mathbf x - \mathbf y| / a_{\min}$, the margin is 1-Lipschitz
  (rate bound $V$) and never above the distance to the ellipsoid outside it, equal to it for a
  sphere. The ISS approach ellipsoid is $\pm$2000 m along-track and $\pm$1000 m otherwise
  (Fehse 5.6), `ZoneEntry((1000, 2000, 1000))`. The rate bound does not cover the jump at GO, so a
  segment that GO falls inside, as with 15 s steps and a GO at 610 s, is checked only up to GO:
  `segment_minimum`, `segment_minima` and `violation_intervals` resample it evenly from its start
  to GO at the same spacing or finer, one model evaluation per sample once per episode, and nothing
  can be violated after GO.

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
interval. The same two-line argument gives
$|\mathbf v| \le \tfrac12(|\mathbf v_i| + |\mathbf v_{i+1}| + A h)$,
$\rho \le \max(\rho_i, \rho_{i+1}) + \tfrac12 V h$ and
$\rho \ge \max(0, \tfrac12(\rho_i + \rho_{i+1} - V h))$. The last is the `rho_min` passed to
`rate_bound`; constraints whose rate grows as the range falls (closing speed, viewing angles) need
it. Both bounds take stacked trajectories, shape `(..., samples, 6)`, so many arcs are bounded in
one call. In Clohessy-Wiltshire motion the natural acceleration
$(3n^2x + 2n\dot y,\ -2n\dot x,\ -n^2 z)$ is at most $3n^2\rho + 2n|\mathbf v|$, so with thrust
acceleration $a_T$ and a safety factor $c$ = 2

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
not ends the search. A pass between two samples is flagged, because its interval's bound falls
below the samples on either side, and the search finds it when the interval holds that one dip.

**Intervals that may still hold a violation.** On a coarse interval with two dips, Brent's method
can settle in the shallower one while the bound stays negative, so a violation would go unseen
(two passes 1 m and 5 m from the target, 77 s apart, sampled only at the ends of 100 s: the
search finds the 5 m pass). An interval is therefore cut while the best value found is not
negative and its bound is below $\min(0, m_{\text{best}} - \text{resolution})$. The interval with
the lowest such bound is cut into 4 equal pieces, whose 5 ends are sampled and which are bounded
the same way, each piece's bound raised to its parent's where that is higher; the piece with the
lowest bound is refined by the Brent search if its bound still meets the rule, and every piece
whose bound does joins the intervals to cut. This repeats until no bound meets the rule, a
violation is found or 16 cuts are spent. A segment therefore costs at most 80 more evaluations
of the model and 16 more Brent searches, and only where its bound allows a violation that its
minimum does not show. At the environment's default sampling no
segment was cut in 60 steps of the PD agent and 100 random steps in each of 12 scenarios. The
pieces' bounds steer this search alone; `lower_bound` is still the bound of the segment's own
samples. A segment whose bound is not negative, or whose minimum is already negative, is not cut,
so where every interval holds one dip nothing changes.

**What is guaranteed.** `lower_bound` is a valid lower bound on the margin of the simulated
trajectory over the segment. The reported `margin` is the lowest value found: the true minimum
whenever each refined interval holds a single dip, and otherwise negative wherever the cut search
above finds a violation the bound allows. Where the rule skips the refinement, the
reported margin is the lowest sample and is accurate only to the constraint's `resolution`, 1 mm
for the keep-out sphere. `KeepOut`, `SpeedLimit`, `ClosingSpeedLimit` and `GateTable` lower
`lower_bound` by 3 eps times the magnitudes their margins are computed from (the range, the
radius or allowed speed, and the start range and speed plus what the speed and acceleration bounds
allow over the segment), since rounding in the propagated state and in $\rho - r_{\text{ko}}$
with $\rho$ near $r_{\text{ko}}$ can put a sample's margin above the exact one even where the
margin is near zero. Against a 50-digit reference on the seeded test segments
(`benchmarks/rounding_bounds.py`), the worst sample error is 1.751 eps of that scale, for the
speed limit, so the factor of 3 leaves a margin of about 42%. The worst case moves with the
sample times, from 1.505 eps with them a nanosecond off, so no sample set gives a tight
supremum and the factor is set well above it. Neither the refinement nor the cut
search tightens the bound. The Lipschitz bound can stay up to $\tfrac12 L h$ below the reported
minimum: over 200 coasting flybys at 0.5 to 20 m/s sampled 20 times per 10 s its gap to the exact
minimum has a median of 1.76 m and a maximum of 5.02 m. The keep-out sphere therefore also
carries the second-order bound below, and its certificate is the larger of the two on each
interval.

## The second-order certificate for the keep-out sphere

A constraint may define `second_order_bounds(states, h, speed, accel, jerk)`, one lower bound on
its margin per sample interval from a curvature argument; `segment_minimum` and `segment_minima`
then take the larger of it and the Lipschitz bound on each interval. `KeepOut` has one, which
does not use the jerk bound and may be called without it. Its margin
$\rho - r_{\text{ko}}$ has an unbounded second derivative near the origin, so the bound is taken on
$f = |\mathbf r|^2$, whose second derivative is $f'' = 2(|\mathbf v|^2 + \mathbf r \cdot
\dot{\mathbf v})$. With the interval's bounds $V$ and $A$ from above and $R = \max(\rho_i,
\rho_{i+1}) + \tfrac12 V h$ on the range,

$$
f'' \le M = 2\,(V^2 + R A)
$$

(`curvature_bound(speed, accel, extent)`). Then $f - \tfrac12 M s^2$ is concave in the time $s$
from the interval's start, so $f$ lies above its chord minus $\tfrac12 M s (h - s)$, a convex
quadratic in $s$ whose minimum over $[0, h]$ is closed form:

$$
f(s) \ge f_i + s\,\frac{f_{i+1} - f_i}{h} - \tfrac12 M s (h - s), \qquad
m \ge \sqrt{\max(0, f_{\text{lb}})} - r_{\text{ko}}
$$

with $f_{\text{lb}}$ the quadratic's minimum. This is a second-order Taylor inclusion (Kochenderfer,
Katz, Corso and Moss, *Algorithms for Validation*, sections 9.2 and 9.3), whose width shrinks with
$h^2$ where the Lipschitz inclusion shrinks with $h$. The bound is exact for a straight-line pass
with $n = 0$ and no thrust, where $f'' = 2|\mathbf v|^2$ and $V$ is the exact speed. It follows the
validity rule of the speed and acceleration bounds: where those are infinite the bound is $-\infty$,
never the finite value $-r_{\text{ko}}$.

Measured against exact minima from the closed-form solution, with the same 200-flyby generator
(0.5 to 20 m/s aimed near the target from 20 to 200 m):

| Case | Lipschitz bound, gap median / max | Second-order bound, gap median / max |
|---|---|---|
| 200 coasting flybys, 10 s, 20 samples | 1.76 m / 5.02 m | 4.9e-5 m / 0.019 m |
| 200 flybys, half thrusting at up to 0.02 m/s^2 per axis, 5 to 30 s, 20 samples | 3.17 m / 13.1 m | 2.1e-3 m / 0.143 m |
| 200 coasting flybys, 10 s, 5 samples | 5.28 m / 20.6 m | 1.4e-14 m / 1.24 m |

Neither bound replaces the other. The second-order bound was never below the Lipschitz bound on
these 8400 intervals, but on the 320 random segments of the test suite (1 to 60 s at 0.5 to
20 m/s with 2 to 20 samples, a third thrusting at up to 0.02 m/s^2 per axis, half aimed within
5 m of the target and 103 passing there) it is the looser of the two on 23 segments, with 2 to 6 samples or passes close
to the origin, and the reported gap over those 320 has a median of 0.20 m, a 95th percentile of
23 m and a maximum of 102 m (sound on all 320, and above the Lipschitz bound on 297). Where neither
bound certifies a margin to the constraint's resolution, the interval has to be sampled more
finely.

## Second-order certificates for the speed limits

For a speed-type margin the Lipschitz width $\tfrac12 L h$ is dominated by the acceleration, which
is far larger than the margin's change over an interval, so without a second bound nearly every
interval is refined. `SpeedLimit`, `ClosingSpeedLimit` and `GateTable` bound their margins by a
chord less a curvature term, whose width shrinks with $h^2$. Besides $V$ and $A$ they need a bound
on the jerk $|\ddot{\mathbf v}|$, which `segment_minimum` and `segment_minima` pass as the fifth
argument of `second_order_bounds`; without it their bounds are $-\infty$. With the thrust held in
Hill axes, the rate of the natural acceleration is
$(3n^2\dot x + 2n\ddot y,\ -2n\ddot x,\ -n^2\dot z)$, so with the same factor $c$

$$
J = c\,(3n^2 V + 2n A).
$$

Every bound below has the form $m(s) \ge \ell(s) - \tfrac12 K s (h - s)$, with $\ell$ the chord
between the margin's (or a term's) end values and $s$ the time from the interval's start; its
minimum over $[0, h]$ is the closed form of the keep-out bound, at most $\tfrac18 K h^2$ below
the lower end.

**Speed.** The velocity lies within $\tfrac12 J s (h - s)$ of the line between its end values,
since each component of the difference is zero at both ends with second derivative at most $J$,
and the norm of that line is below the chord of $|\mathbf v|$ by the triangle inequality. So
$$
|\mathbf v(s)| \le \left(1 - \tfrac{s}{h}\right)|\mathbf v_i| + \tfrac{s}{h}\,|\mathbf v_{i+1}|
+ \tfrac12 J s (h - s),
$$
with no division by the speed, so it holds through $\mathbf v = 0$, where $|\mathbf v|$ has a
kink. The range has $\ddot\rho = (|\mathbf v|^2 - \dot\rho^2)/\rho + \hat{\mathbf r}\cdot
\dot{\mathbf v} \in [-A,\ V^2/\rho_{\min} + A]$, the lower end holding through $\rho = 0$ too, so
$v_1 \rho$ lies above its chord less $\tfrac12 K_\rho s(h - s)$ with
$K_\rho = v_1 (V^2/\rho_{\min} + A)$ for $v_1 > 0$ (which needs $\rho_{\min} > 0$), $|v_1| A$ for
$v_1 < 0$ and 0 for $v_1 = 0$. The `SpeedLimit` margin lies above its chord less
$\tfrac12 (K_\rho + J) s (h - s)$.

**Closing speed.** The margin $a(\rho) - \max(0, -\dot\rho)$ equals
$\min(a(\rho),\ a(\rho) + \dot\rho)$, which avoids the kink at $\dot\rho = 0$; each term is bounded
separately and the certificate is the smaller. For an allowed speed linear in $\rho$ the first is
the range bound above. For the second,
$$
\dddot\rho = \frac{3\,\mathbf v_\perp \cdot \dot{\mathbf v}}{\rho}
- \frac{3\,|\mathbf v_\perp|^2 \dot\rho}{\rho^2} + \hat{\mathbf r}\cdot\ddot{\mathbf v},
\qquad
|\dddot\rho| \le \frac{3 V A}{\rho_{\min}} + \frac{2 V^3}{\sqrt3\,\rho_{\min}^2} + J,
$$
with $\mathbf v_\perp$ the velocity across the line of sight, since $|\mathbf v_\perp|^2
|\dot\rho|$ is at most $2V^3/(3\sqrt3)$ for $|\mathbf v| \le V$. Both terms need
$\rho_{\min} > 0$; an interval whose range bound is zero gets $-\infty$. `GateTable` uses the same
bound on an interval whose range, between $\rho_{\min}$ and $\max(\rho_i, \rho_{i+1}) + \tfrac12
V h$, stays between two consecutive gates or beyond the table, where the allowed speed is linear in
$\rho$; an interval that can reach a gate, where the allowed speed has a kink, gets $-\infty$.

As for the keep-out sphere, the certificate on each interval is the larger of this bound and the
Lipschitz bound, and the refinement rule is unchanged, so the bound decides only which intervals
are refined. Where it ends the search, the reported minimum is the lowest sample, within
`resolution` of the true minimum. `ApproachCone`, `SunExclusion`, `PassiveSpeed` and `ZoneEntry`
keep the Lipschitz bound alone.

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
independent of the batch size. The segments whose bound still allows a violation get the cut
search above one at a time, and those a GO falls inside go through `segment_minimum`. The
constraint's `margin` and `rate_bound` must accept arrays, as `KeepOut`'s do.

## Violation intervals

`violation_intervals(constraint, state_at, times, states, minimum, t0, thrust, n)` returns the
times inside a segment where the margin is negative, as (start, end) pairs in seconds from the
episode start. It takes the samples and the result of `segment_minimum`, with the segment's thrust
acceleration and the mean motion for the rate bounds, and returns nothing when the certified lower
bound is not negative. The search, `sampled_negative_intervals`, is the one the metrics use
(`vbar.metrics.negative_intervals`), so the reward and the violation metrics see the same
intervals. The time of the reported minimum is added to the samples when its margin is negative,
so a violation found by refinement or by the cut search is reported even where the samples on
either side are positive. Each boundary is the root of the margin on a sample interval whose ends
have opposite signs, found with Brent's method (`scipy.optimize.brentq`) to 1e-9 s. The ends are
evaluated again with the function the root search uses; a sample on the boundary can round to
the other side of zero there, and the end nearer zero is then taken as the crossing instead of
the search failing. A sample interval whose ends share a sign is searched for a dip below zero,
or a rise above it, only when its Lipschitz bound allows one; like the refinement, this assumes
at most one dip per sample interval. An interval open at the segment's start or end runs to that
end. The environment merges intervals that meet at a segment boundary, so a pass through the
keep-out zone during an actuation delay's switch is one interval. The reward library builds its
constraint costs from these intervals ([rewards.md](rewards.md)).

## In the environment

Each step's per-constraint minima (time, margin, lower bound) are in `env.record.segment.minima`,
its violation intervals in `env.record.segment.violations`, and the episode's lowest values in
`info["min_margins"]` and `info["min_margin_bounds"]`. The approach scenario counts a step as
entering the keep-out zone when its minimum margin is negative.

## Assumptions and limits

- $c$ = 2 covers the two-body nonlinearity and J2 at separations below 1% of the orbit radius,
  about 69 km at 500 km, in the acceleration bound and in the jerk bound; this is not checked at
  run time.
- The jerk bound assumes the thrust acceleration is constant in Hill axes within a segment, as the
  environment's dynamics models apply it.
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
- The second-order keep-out bound: on the three flyby tables above and on 320 random segments
  (1 to 60 s, 0.5 to 20 m/s, 2 to 20 samples, half passing within 5 m of the target) the certified
  bound lies below the exact minimum of the closed-form solution on every segment, is never below
  the Lipschitz bound, and keeps to the table's gaps with headroom (median 1e-4 m and maximum
  0.03 m dense, 0.2 m thrusting, 1.5 m coarse), which the Lipschitz bound alone misses; a
  straight-line pass with $n = 0$ sampled at 2 and 20 points gives the exact closest approach to
  1e-9 m; 1000 s between samples gives $-\infty$, as does any interval with an infinite speed or
  acceleration bound; a stationary chaser gives the margin without dividing by zero; stacked
  trajectories give the bounds of each; halving the curvature bound makes the certificate unsound
  on the flyby tables and doubling it breaks the tightness table while staying sound, so both
  mutations fail the suite; the batched minimum reports the same bound as the scalar one. The
  tests run the first 100 segments of each table for the first-order, never-below and doubled
  curvature claims; `benchmarks/keep_out_bound_tables.py` runs the full tables.
- `tests/test_constraint_types.py`: for every constraint type, over 12 flybys (coasting and
  thrusting at 0.1 to 3 m/s) the certificate is finite and lies below the minimum of 4001 dense
  samples, and the refined minimum is within 1e-6 of it. Hand-built states check each margin
  formula.
- The cone margin equals the signed distance found by a direct search over the cone's generators
  (Nelder-Mead from 63 starts) to 1e-6 m for 12 random cones and points, including points behind
  the apex; it is continuous across the active range.
- The sun-exclusion margin is checked looking into the Sun and from above, and the Sun direction
  turns from +y to +x in a quarter orbit.
- Beyond the closing-speed term, each passive-speed rate bound equals the envelope's slope at
  `rho_min` times the speed, to 1e-6 against a central difference, at 16, 20, 50 and 300 m.
- The R-bar envelope gives 0.1225, 0.0612 and 0.0283 m/s at 50, 30 and 20 m (Fehse 5.7.2, from
  eq. 3.22); free drift from the envelope at 20, 50 and 300 m comes within 15 m of the target to
  1e-8 relative over 200001 samples of one orbit. The V-bar coast at 0.1 m/s and 400 km is 42.21 m
  (eq. 3.28), and free drift from the V-bar envelope with gaps of 5, 50 and 500 m reaches the
  structure.
- The zone margin is zero on the ISS ellipsoid, infinite after GO, below the distance to 8 random
  ellipsoids (200000 surface samples) and equal to it for a sphere.
- A constant margin on the V-bar is not refined.
- Over 40 random flybys sampled 20 times or only at their ends, the batched minimum agrees with
  the scalar one to 1e-7 m and its bound to 1e-12.
- Violation intervals of coasting passes match the entry and exit times of the closed-form
  transition matrix to 1e-8 s, including a pass seen only by the refinement between two samples
  and intervals open at a segment's start or end; a pass across an actuation-delay switch is one
  interval within a step.

`tests/test_constraints_edge_cases.py`:

- With GO 1.4 s into a 10 s step sampled 20 times, between two samples, the zone bound lies below
  the closed-form margin at GO (-0.400 m), the minimum is within 1 mm of it, and the violation runs
  from the closed-form entry to GO within 1e-6 s. With 15 s steps and seed 30's GO at 610 s, the
  closing scenario's bound lies below the exact minimum on every step through GO.
- 40 keep-out radii set to the range at a sample, one ulp either side, give violation intervals
  without an error, and on a 5 m/s V-bar closing a radius equal to the range at each of samples
  1 to 18 starts the violation at that sample to 1e-8 s.
- Two passes 1 m and 5 m from the target in one 100 s segment, sampled at its ends or 191 times:
  the bound lies below the exact minimum (-2 m), the minimum is within 1 mm of it, and the one
  violation matches the closed-form entry and exit to 1e-6 s.

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
