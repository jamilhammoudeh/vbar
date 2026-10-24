# Two-impulse CW targeting

Code: `conditioning`, `cw_lambert`, `two_impulse`, `TwoImpulseAgent` and `TwoImpulseBound` in
`vbar.agents.cw_targeting`. Hill frame, metres, seconds.

The classical way to reach a point near the target in a fixed time: one impulse puts the chaser
on the Clohessy-Wiltshire (CW) coast that passes through the point at that time, and a second
impulse stops it there. `TwoImpulseAgent` flies such a transfer to the hold point of a scenario,
either re-planned at every decision, planned once and flown open loop, or tracked as a nominal
trajectory. `TwoImpulseBound` gives the cheapest such transfer from a start state as an upper
bound on the delta-v a scenario needs.

## How it is built

**Targeting.** With the CW transition matrix over the transfer time $t_f$ split into position
and velocity blocks,

$$
\begin{bmatrix}\mathbf r(t_f) \\ \mathbf v(t_f)\end{bmatrix} =
\begin{bmatrix}\Phi_{rr} & \Phi_{rv} \\ \Phi_{vr} & \Phi_{vv}\end{bmatrix}
\begin{bmatrix}\mathbf r_0 \\ \mathbf v_0\end{bmatrix},
$$

the departure velocity that reaches $\mathbf r_f$ solves the position rows, and the arrival
velocity follows from the velocity rows (Fehse, *Automated Rendezvous and Docking of
Spacecraft*, section 3.3.2, eqs 3.41 to 3.43; Curtis, *Orbital Mechanics for Engineering
Students*, section 7.5):

$$
\mathbf v_0^+ = \Phi_{rv}^{-1}\left(\mathbf r_f - \Phi_{rr}\mathbf r_0\right), \qquad
\mathbf v_f = \Phi_{vr}\mathbf r_0 + \Phi_{vv}\mathbf v_0^+ .
$$

`cw_lambert(n, r0, rf, tf)` returns $(\mathbf v_0^+, \mathbf v_f)$; `two_impulse(n, state,
target, tf)` returns the impulses $\Delta\mathbf v_1 = \mathbf v_0^+ - \mathbf v_0$ and
$\Delta\mathbf v_2 = -\mathbf v_f$ that end at rest on the target. $\Phi$ comes from
`cw_discrete`, whose closed form keeps $1 - \cos nt$ free of cancellation at short transfers,
and the $3 \times 3$ system is solved with `numpy.linalg.solve`, never by forming the inverse.

**Singular set.** With $x = n t_f$, $\det\Phi_{rv} = (8 - 8\cos x - 3x\sin x)\sin x / n^3$.
`conditioning(n, tf)` returns the two factors scaled to 1 at short transfers,

$$
g = \frac{8 - 8\cos x - 3x\sin x}{x^2}, \qquad h = \frac{\sin x}{x},
$$

the in-plane and cross-track measures. $g$ vanishes at $x = 2k\pi$ and at the other roots of
$8\cos x + 3x\sin x = 8$ ($2.81346\pi$, $4.89060\pi$, ...), $h$ at $x = k\pi$ (Alfriend, Vadali,
Gurfil, How and Breger, *Spacecraft Formation Flying*, p. 92). Near these times the transfer
needs impulses that grow without bound, so `cw_lambert` refuses a transfer where $|g|$ or $|h|$
is below `tol` (default $10^{-2}$, which bounds the magnification of the solve by about 100).

**Sign of the first command.** From rest on the V-bar the departure velocity toward a goal
$\delta$ away is $n\sin x\,\delta_y / D$ along track and $n(4\sin x - 3x)\,\delta_x / D$
radially, with $D = 8 - 8\cos x - 3x\sin x > 0$ on $(0, 2\pi)$. Along track and across it the
first burn points at the goal for $x < \pi$; radially only below the root of $4\sin x = 3x$,
$x = 1.2757$. A longer transfer correctly starts by burning away from a goal above or below.

**Execution.** `TwoImpulseAgent` plans a transfer of $t_f = $ `n_tf` $/ n$ from its first
observation to rest at the scenario's hold point. Impulses go through
`vbar.agents.planning.impulse_action`: in the `impulse_coast` action mode an impulse that one
step can deliver is one step followed by a coast, and a larger one is spread over several equal
steps; in the continuous modes each step commands the impulse over the step, clipped to the
thrusters. The modes are:

- `"retarget"` solves `two_impulse` again at every decision for the time left and fires its
  first impulse. Fewer than `t_min_steps` steps before the arrival it brakes, cancelling the
  measured velocity until that burn is delivered, since the re-planned impulse grows as the
  error over the shrinking time left. If the time left passes through the singular set (for
  `n_tf` above $\pi$) the agent coasts those steps, since it is already on a transfer.
- `"open_loop"` plans both impulses at the first decision and flies them at their steps, with
  one committed coast in between.
- `"track"` (continuous action modes only) flies the open-loop plan and adds the feedback
  $-\omega_n^2(\mathbf r - \mathbf r_\text{nom}) - 2\zeta\omega_n(\mathbf v - \mathbf v_\text{nom})$
  toward the nominal propagated from the first observation through the plan as executed. Fehse,
  section 4.4.1, warns that re-targeting near the target leaves deviations from the nominal
  uncontrolled, so the passive-safety analysis of the nominal no longer holds; tracking keeps it.

After the arrival every mode holds the hold point with its hover acceleration
$(-3n^2x_h, 0, n^2z_h)$, zero on the V-bar (Fehse eqs 3.55, 3.58); `"track"` adds the same
feedback toward the hold point.

**Log.** `agent.log` holds one entry per decision: the time `t`, the `phase` (`transfer`,
`singular`, `departure`, `coast`, `arrival` or `hold`) and whether the command `saturated` the
thrusters. The decision that ends the departure burn of `"open_loop"` or `"track"` commits a coast
to the arrival burn and adds `coast_s`, the predicted minimum keep-out margin over it,
`predicted_margin`, and a certified lower bound on that minimum, `predicted_margin_bound` (m), both
from `vbar.agents.planning.predicted_margin` on the state after the burn step. The coast is sampled
20 times per step and passed to `vbar.constraints.segment_minimum`, so the bound is, on each sample
interval, the larger of the speed bound and the keep-out sphere's second-order bound
([constraints.md](constraints.md)), and the margin lies within the 1 mm keep-out resolution of the
true minimum. A predicted violation is logged and the plan flown as is: the agent is an unfiltered
reference, and the metrics score the true trajectory.

**Fuel bound.** `TwoImpulseBound().bounds(view, start)` evaluates $|\Delta\mathbf v_1|_p +
|\Delta\mathbf v_2|_p$ from `start` to rest at the hold point over a grid of transfer angles,
$p = 1$ for box thrusters and $p = 2$ for a ball, skipping angles where either `conditioning`
measure is below $10^{-2}$ in magnitude and transfers longer than
$(\text{max\_steps} - 1)\,dt$, and returns
`FuelBounds(lower=0.0, upper=<least cost>, norm="l1" or "l2", rung="cw")`. The upper bound is
a feasible cost for impulsive burns under exact CW motion; the lower bound 0 is the trivial one,
so only the upper optimality ratio is informative.

## Parameters and defaults

| Parameter | Default | Reason |
|---|---|---|
| `n_tf` | 1.0 rad | Scale-free; $g = 1.153$ and $h = 0.841$, far from the singular set; the first command points at the goal on every axis; 903 s at 500 km fits the 3600 s approach episode |
| `mode` | `"retarget"` | Closes the loop on navigation noise |
| `t_min_steps` | 2 | Stops re-planning before the time left makes the first impulse of order error over time |
| `singular_tol` | $10^{-2}$ | Bounds the magnification of the solve by about 100 |
| `wn`, `zeta` | 0.01 rad/s, 1.0 | The `PDAgent` values: critically damped, about nine times the mean motion at 500 km |
| `TwoImpulseBound(n_tf_grid=None, grid_points=400)` | 400 angles from 0.05 to 5 rad | Covers transfers up to most of an orbit, below the first in-plane singular time $2\pi$ |

## Assumptions and limits

- CW motion: a circular target orbit and separations small against its radius. On two-body or
  J2 truth the planned transfer misses by the model error.
- Burns are impulses in the plan and one or more steps of constant thrust in flight. A burn
  spread over several steps starts at the impulse time instead of being centred on it, so the
  arrival is off by that timing error.
- The environment rounds a coast to whole steps, so an open-loop arrival burn fires up to half a
  step early or late.
- `"open_loop"` plans on one noisy observation and never corrects it; `"retarget"` corrects every
  step. `"track"` needs a continuous action mode and is refused in `impulse_coast`; every mode is
  refused in `disc27`.
- `TwoImpulseAgent.reset` and `TwoImpulseBound.bounds` refuse goals other than a hold or a
  berthing, and goals without a hold point, with a `ValueError`.
- Re-targeting does not always capture. Single measured cases in the approach scenario, mode
  `impulse_coast`, $dt$ = 10 s, `max_accel` = 0.02 m/s^2, default agent, episode ended at
  3600 s without capture: from rest 1 km behind the target with exact navigation, 121 m from the
  hold point at the end; from rest 2 km behind, 1032 m; from the default random start of episode
  seed 8 (agent seed 1008) with the default navigation, 61 m.
- `n_tf` within `singular_tol` of the singular set is refused at reset. A transfer longer than the
  episode is cut at the episode's end.
- The predicted margin covers the keep-out constraint over committed coasts only, under exact CW
  motion from the planned state; its certified bound is $-\infty$ where the samples are too far
  apart for the speed bound to hold.
- Only `n_tf` below about 1.2757 keeps the radial first command toward the goal (see above), so
  the sign check fails in $\pm x$ for longer transfers, by physics rather than a sign error.

## Validation

Tests: `tests/test_cw_targeting.py` and `tests/test_two_impulse_agent.py`.

- Alfriend et al., example 5.2, in normalized units ($n = 1$): the impulses at $n t_f = 2$
  reproduce $\Delta\mathbf v_1 = (-0.00180, -0.01927, 0.005865)$ and
  $\Delta\mathbf v_2 = (0.00562, -0.00173, 0.0165)$ with total 0.03774, recomputed at the stated
  inputs (the printed transfer time and the sign of the printed second impulse are misprints);
  the cheapest transfer over $n t_f \in [0.05, 10]$ costs 0.03572 at $n t_f = 4.64$, past the
  printed time (`test_alfriend_example_5_2_two_impulse_rendezvous`,
  `test_the_cheapest_transfer_of_example_5_2_lies_past_the_printed_transfer_time`).
  `TwoImpulseBound` reaches 0.03572 to $10^{-3}$ in the 2-norm and 0.04573 in the 1-norm, and a
  window cut at $n t_f = 2$ stays above 0.0365.
- Curtis, *Orbital Mechanics for Engineering Students*, section 7.5, its two worked examples
  (first edition Examples 7.2 and 7.3; later editions 7.4 and 7.5), with the book's
  $\mu$ = 398600 km^3/s^2 and 300 km target orbit: from (20, 20, 20) km moving at
  (-0.02, 0.02, -0.005) km/s, rendezvous in 8 h costs 0.0740440 and 0.0355649 km/s, total
  0.109609 km/s, with every impulse component to its printed digits; from rest 2 km behind,
  rendezvous in 1.49 h costs 0.1226 m/s per impulse, 0.2452 m/s in total. The first edition's
  printed final impulse of the 8 h example has a misprint and uses a rounded mean motion; the
  later editions' figures are reproduced (`tests/test_curtis_rendezvous.py`).
- Alfriend et al., example 9.2: the departure velocity for a 60.42 s transfer from (25, 25, 50) m
  to the origin at $a$ = 7170 km is $(-0.38888, -0.43924, -0.82643)$ m/s to $10^{-4}$.
- Fehse, section 3.3.2, eqs 3.41 to 3.43: the transfer from 3000 m to 300 m behind the target in
  0.37 orbits reaches the end point within $10^{-9}$ of its length.
- The singular set: `conditioning` vanishes at $n t_f = 2\pi$, $2.81346\pi$, $4.89060\pi$ (in
  plane) and $\pi$ (across the track), found by root finding to $10^{-5}\pi$; `cw_lambert`
  refuses one second off an orbit and accepts one minute off it.
- The plan is linear from 1 cm to 100 km to $10^{-12}$, a 1 ms transfer needs the displacement
  over the time, and the textbook transition matrix gives the same velocity to $10^{-12}$.
- The radial first command from rest changes sign at the root of $4\sin x = 3x$: toward the goal
  at $n t_f = 1.2$, away at 1.35; `sign_check` passes on all six axes at the default and fails
  exactly in $\pm x$ at $n t_f = 1.5$.
- Flown in the approach scenario on exact CW from 300 m to the hold point 100 m behind the
  target: `"open_loop"` captures at the planned arrival, with a miss after the arrival burn under
  twice the 0.842 m explained by flying the 89.35-step coast as 89 steps, and its delivered
  delta-v equals the planned two-impulse cost to $10^{-6}$. With 1 m and 1 mm/s navigation noise,
  `"retarget"` captures on all five tested seeds while `"open_loop"` misses on one, and
  `"track"` arrives closer to the hold point than `"open_loop"` (median over five seeds).
- On random starts, every coast whose certified predicted margin is non-negative keeps a
  non-negative true minimum margin in continuous time, and the bound never exceeds the predicted
  margin. Over the 850 to 880 s coasts of seeds 0 to 29, against a minimum from 0.5 s samples
  refined by Brent's method, the bound lies 0.05 to 0.19 mm below the true minimum and the
  predicted margin at most 0.29 mm above it, in `"open_loop"` and `"track"`.

## Example

An open-loop transfer from 300 m behind the target to the hold point 100 m behind, scored
against the plan's own two-impulse cost:

```python
from vbar import ApproachConfig, ApproachEnv
from vbar.agents.cw_targeting import TwoImpulseAgent, TwoImpulseBound
from vbar.metrics import evaluate, record_episode
from vbar.spacecraft import PerfectNav

config = ApproachConfig(
    start_state=(0.0, -300.0, 0.0, 0.0, 0.0, 0.0),
    action_mode="impulse_coast",
    sensor=PerfectNav(),
)
ep = record_episode(ApproachEnv(config), TwoImpulseAgent(mode="open_loop"), seed=1, agent_seed=2)
bounds = TwoImpulseBound(n_tf_grid=(1.0,)).bounds(ep.view, ep.start)
scored = evaluate(ep, bounds=bounds)
print(f"captured: {scored['success']}, delta-v over the plan: {scored['optimality_gap_upper']:.6f}")
assert scored["success"]
```
