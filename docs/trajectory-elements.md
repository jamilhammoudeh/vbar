# Trajectory elements

Code: `Element`, the element functions and the agents `ElementAgent`, `TorvaAgent`,
`FehseSequenceAgent` and `NominalTrackingAgent` in `vbar.agents.elements`. Hill frame, metres,
seconds.

The building blocks of rendezvous practice as flown since the Shuttle: impulsive transfers along
the V-bar, fly-arounds to a point below the target, straight-line approaches held by thrust, and
the twice-orbital-rate circle about the target. Each element is a plan of impulses and, for forced
motion, a thrust law between them. The agents fly one element, the Shuttle's twice-rate
fly-around, Fehse's closing sequence of a radial hop and a straight-line final approach, and the
tracking of a planned nominal trajectory.

## How it is built

**Frame.** Fehse, *Automated Rendezvous and Docking of Spacecraft*, writes his local orbital
frame with x along the V-bar, y opposite the orbit normal and z toward Earth. In the Hill frame
used here (x radial out, y along-track, z along the orbit normal) his $(x, y, z)$ is
$(-z_F, x_F, -y_F)$; `vbar.conventions.iss_to_hill` does the mapping, so his point $d_z$ below
the target is Hill $(-d_z, 0, 0)$. All equations below are written in Hill coordinates, on the
Clohessy-Wiltshire (CW) equations

$$
\ddot x = 3n^2 x + 2n\dot y + a_x, \qquad \ddot y = -2n\dot x + a_y, \qquad
\ddot z = -n^2 z + a_z .
$$

**Elements.** `Element(impulses, duration, accel)` holds the impulses (`Impulse(t, dv)`, time
since the element's start), the time to the last impulse, and an optional thrust law
`accel(t, state)`. With $T = 2\pi / n$:

- `tangential_transfer(n, dy)`: along-track impulses $-n\,\Delta y / 6\pi$ at 0 and the opposite
  at $T$ move a chaser at rest on the V-bar by $\Delta y$; total $n|\Delta y| / 3\pi$ (Fehse
  section 3.3, eqs 3.29 to 3.32).
- `radial_transfer(n, dy)`: radial impulses $-n\,\Delta y / 4$ at 0 and at $T/2$; a radial
  impulse $\Delta v_x$ moves the chaser by $-4\Delta v_x / n$ in half an orbit, so a forward hop
  starts toward Earth; total $n|\Delta y| / 2$, $3\pi/2$ times the tangential one (eqs 3.34,
  3.37).
- `cw_hohmann(n, dx)`: along-track impulses $n\,\Delta x / 4$ at 0 and $T/2$ raise a chaser from
  the coelliptic orbit $\Delta x$ below onto the target orbit and advance it by
  $3\pi\Delta x / 4$.
- `tangential_flyaround(n, dz)`: from rest at $y = -3\pi d_z / 4$, an along-track impulse
  $-n d_z / 4$ reaches $(-d_z, 0, 0)$ after half an orbit with along-track speed $7 n d_z / 4$,
  which the second impulse cancels; total $2 n d_z$ (eq 3.33).
- `radial_flyaround(n, dz)`: from rest at $y = -2 d_z$, a radial impulse $-n d_z$ reaches
  $(-d_z, 0, 0)$ after a quarter orbit with along-track speed $2 n d_z$; total $3 n d_z$
  (eqs 3.38, 3.39).
- `forced_motion_accel(n, r, v, a_path)`: the thrust that makes the chaser follow a path with
  acceleration $\mathbf a_\text{path}$ is that acceleration minus the CW terms,
  $\mathbf a = \mathbf a_\text{path} - (3n^2 x + 2n\dot y,\ -2n\dot x,\ -n^2 z)$. At constant
  speed on the V-bar it is $(-2n\dot y, 0, 0)$ (eq 3.44), on the R-bar $(-3n^2 x, 2n\dot x, 0)$
  (eqs 3.49 to 3.53).
- `straight_line(n, r0, r1, speed)`: a start impulse of `speed` toward `r1`, the forced
  acceleration above at the current state, and a stop impulse at `r1`. The speed stays constant
  to the end point, with no braking near it. On the V-bar the cost is
  $2|\Delta y| / t + 2n|\Delta y|$ (`straight_line_cost`, eqs 3.44 to 3.48).
- `twice_rate_circle(n, R)`: from rest at $(-R, 0, 0)$ below the target, an impulse $2nR$ along
  $+y$ puts the chaser on the circle flown clockwise in the Hill x-y plane (below, ahead, above,
  behind) at rate $2n$, where $\mathbf v = (2n y, -2n x, 0)$. Fehse's forced circle (eq 3.79)
  then needs only the R-bar hover force $(-3n^2 x, 0, 0)$ (Goodman, *History of Space Shuttle
  Rendezvous*, JSC-63400, p. 137). `twice_rate_cost(n, R, angle)` is the cost of an arc from
  rest to rest: $4nR + \tfrac{3}{2} nR \int_0^\theta |\cos u|\,du$.
- `finite_burn_duration(dv, gamma)`: $\tau = |\Delta\mathbf v| / \gamma$; an impulse stands for
  such a burn while $\tau$ is much shorter than the time between burns (eqs 3.61, 3.68).

**Thrust-loss safety of the twice-rate circle.** On the circle $\dot y + 2n x = 0$. The CW
equations give $\frac{d}{dt}(\dot y + 2n x) = a_y$, so the radial hover force keeps it, and with
$\dot y + 2nx = 0$ the along-track drift of free CW motion vanishes. After a thrust loss the
chaser flies a closed ellipse whose least range is the range at the loss: it never comes closer
to the target.

**Executing an element.** `ElementAgent(element_factory)` calls `element_factory(view, state)`
at the first decision with the observed Hill state and flies the returned element. Each impulse
fires at the step nearest its time through `vbar.agents.planning.impulse_action`: in the
`impulse_coast` action mode an impulse one step can deliver is one step followed by a committed
coast to the next impulse, and a larger one is spread over several steps; in the continuous
modes each step commands the impulse over the step, clipped to the thrusters, and what was not
delivered carries over to the next step. Between the first and last impulses the element's
`accel`, evaluated at the observed state, is added to every step. After the last impulse the
agent holds its observed position with the hover acceleration $(-3n^2 x, 0, n^2 z)$ (Fehse
eqs 3.55, 3.58).

**Twice-rate fly-around.** `TorvaAgent(stop_angle=...)` flies the twice-rate circle through the
chaser's first observed position. The start burn sets the circle's velocity
$(2n y, -2n x, 0)$ against the observed one; on the arc the command is the hover force
$-3n^2 x$ at the observed $x$ and nothing else; the stop burn, at the step nearest
$\theta_\text{stop} / 2n$, cancels the observed velocity. A quarter turn from below the target
reaches the $+$V-bar in $T/8$.

**Fehse sequence.** `FehseSequenceAgent` flies the closing sequence of Fehse, sections 3.3 and
5.7, as legs logged under `"mode"` in `agent.log`:

1. `"to_vbar"`, only when the first observation lies more than `gate` from the V-bar: a CW
   two-impulse transfer (`vbar.agents.cw_targeting.cw_lambert`) to $(0, y, 0)$ in
   `n_tf` $/ n$.
2. `"hop"`: a radial transfer along the V-bar, in half an orbit, to the point `final_leg` from
   the hold point on the chaser's side: from 600 m behind a hold point 100 m behind, to 150 m
   behind.
3. `"final"`: a straight line to the hold point at `approach_speed`, held by the forced
   acceleration, with no braking before capture.
4. `"hold"`: the hover acceleration where the chaser is.

Each leg starts by setting the velocity its element plans against the observed velocity and ends
by cancelling the observed velocity. A leg that ends more than `gate` from its waypoint is
followed by `"contingency"`, a two-impulse transfer back to the waypoint in `n_tf` $/ n$, before
the sequence goes on.

**Nominal tracking.** `NominalTrackingAgent(planner)` lets an open-loop planner such as
`TwoImpulseAgent(mode="open_loop")` plan at the first decision and then flies
`vbar.agents.lyapunov.CWTrackingAgent` toward the plan's nominal state at each step, with the
reference acceleration the CW acceleration of the nominal plus the plan's thrust:

$$
\mathbf u = \mathbf a_\text{nom} - \mathbf f_\text{CW}(\mathbf x)
  - n^2(\mathbf r - \mathbf r_\text{nom}) - k n(\mathbf v - \mathbf v_\text{nom}).
$$

On the nominal the command is the planned one; off it the error obeys
$\ddot{\mathbf e} + kn\dot{\mathbf e} + n^2\mathbf e = 0$ on CW. Fehse, section 4.4.1, warns that
re-planning near the target leaves deviations from the nominal uncontrolled, so the safety
analysis of the nominal no longer holds; tracking keeps the chaser on it. After the plan the
reference is the hold point at rest.

**Log.** Each agent keeps `agent.log`, one entry per decision, with the time `t`, the `phase`
(`"burn"`, `"forced"`, `"coast"` or `"hold"`; `"track"` or `"hold"` for the tracker) and whether
the command `saturated` the thrusters; the Fehse sequence adds its leg as `mode`. A burn followed
by a free coast adds `coast_s` and the keep-out margin predicted over the coast,
`predicted_margin`, with its certified lower bound, `predicted_margin_bound` (m, from
`vbar.agents.planning.predicted_margin` on the state after the burn step). A predicted violation
is logged and the plan flown as is: these agents are unfiltered references, and the metrics score
the true trajectory.

## Parameters and defaults

| Parameter | Default | Reason |
|---|---|---|
| `ElementAgent(mode)` | `"open_loop"` | Every impulse is flown as planned; the only mode |
| `TorvaAgent(stop_angle)` | none | The arc depends on the task; $\pi/2$ is the Shuttle's quarter turn from below to ahead |
| `FehseSequenceAgent(final_leg)` | 50 m | The project's own choice: with the hold point 100 m behind the target the hop ends 150 m behind, and the whole final leg stays outside the 50 m keep-out sphere |
| `approach_speed` | 0.04 m/s | The project's own choice: below the 0.05 m/s capture speed of the approach scenario, so the chaser captures without braking |
| `gate` | 10 m | The project's own choice: twice the 5 m capture radius, far above the arrival error of one-step burns |
| `n_tf` | 1.0 rad | As for `TwoImpulseAgent`: far from the singular set of CW targeting |
| `NominalTrackingAgent(k)` | 2.0 | Critically damped error dynamics on CW |

## Assumptions and limits

- Exact CW motion: a circular target orbit and separations small against its radius. On
  two-body or J2 truth the elements and their predicted margins carry the model error.
- Burns are impulses in the plan and one or more steps of constant thrust in flight, starting
  at the impulse's step; with 10 s steps the radial hop ends about 0.2 m from its end point.
- `ElementAgent` plans once, from the first observation, and flies open loop.
- `TorvaAgent` times the stop burn from the arc's rate instead of measuring the angle, and
  leaves the out-of-plane motion free. Holding the hover force over each step lets the range
  wander from the circle by the zero-order-hold error.
- `FehseSequenceAgent` refuses at reset a hop and final leg longer than the episode,
  $\pi / n + $ `final_leg` $/$ `approach_speed`, naming the duration. A `"to_vbar"` or
  `"contingency"` leg adds `n_tf` $/ n$ and can run past the end of the episode.
- `TorvaAgent`, `FehseSequenceAgent` and `NominalTrackingAgent` need continuous thrust and refuse
  the `impulse_coast` action mode; every agent here refuses `disc27`.
- `FehseSequenceAgent` and `NominalTrackingAgent` refuse at reset goals other than a hold or a
  berthing, and goals without a hold point, with a `ValueError`; `ElementAgent` and
  `TorvaAgent` never read the goal.
- The twice-rate circle's start burn and the hop's first burn are set by where the chaser is, not
  by the goal: from rest on the V-bar both point radially, so `vbar.agents.sign_check` fails in
  the along-track directions by design.
- The predicted margin covers the keep-out constraint over free coasts only, under exact CW
  motion; its certified bound is $-\infty$ where the samples are too far apart for the speed
  bound to hold.

## Validation

Tests: `tests/test_elements.py` (the element functions) and `tests/test_element_agents.py` (the
agents in the approach scenario).

- Fehse, section 3.3, at $n = 1.131367 \times 10^{-3}$ rad/s (a 400 km circular orbit; the book
  used $r = 6766$ km): the tangential
  transfer over 2700 m costs 0.3241 m/s, $n|\Delta y| / 3\pi$ at that $n$ (the book prints 0.33);
  the radial transfer costs $3\pi/2$ times as much; the CW Hohmann transfer advances
  $3\pi\Delta x / 4$; the fly-arounds to 500 m below cost 1.1314 m/s and 1.697 m/s; every
  impulsive element ends at rest at its end point to $10^{-9}$ of its length, and all scale
  linearly from 1 cm to 100 km.
- Fehse eqs 3.44 to 3.48: the straight V-bar approach from 3000 m to 300 m costs 7.082 m/s over
  an orbit and 8.054 m/s over half an orbit, 21.85 and 24.85 times the tangential transfer,
  recomputed at the stated constants; the flown line holds the path to $10^{-8}$ of its length
  under an independent integrator, and the integral of its thrust matches the formula to
  $10^{-9}$. The forced acceleration reduces to eqs 3.44, 3.49 to 3.53 and 3.58 on the V-bar,
  the R-bar and out of plane, and to the differentiated path minus the CW terms on a circle
  (eqs 3.77 to 3.80).
- Goodman, JSC-63400, p. 137, at 185 n.mi. and $R$ = 600 ft: the twice-rate circle is held by the
  hover force alone; after a thrust loss at 180, 150, 120 and 90 deg the free ellipse keeps a
  least range of $R$ and reaches 2, 4.145751, 6.203627 and 7 $R$, recomputed at the stated
  constants; arcs of 90, 180, 270 and 360 deg cost 5.5, 7.0, 8.5 and 10.0 $nR$.
- The radial hop flown by `ElementAgent` from 600 m to the hold point 100 m behind, at 500 km
  with 10 s steps, in both `impulse_coast` and `accel` modes: it departs toward Earth with no
  along-track part, ends within twice the 0.209 m that the one-step burns explain and slower than
  0.05 m/s within two steps of $T/2$, captures, and costs $n\Delta y / 2$ to 1%; in
  `impulse_coast` the predicted coast margin equals the margin the environment records to
  $10^{-6}$ and its certified bound lies below both. The coast's minimum falls at its end, where
  the second-order bound of `vbar.constraints.segment_minimum` is the sampled margin itself: from
  rest 400 to 700 m behind on the V-bar, over seeds 0 to 29, it meets a minimum from 0.5 s samples
  refined by Brent's method to $2 \times 10^{-13}$ m, on either side by rounding.
- `TorvaAgent` with $R$ = 100 m at 500 km: the start burn is along $+y$ and the chaser turns
  clockwise; the arc commands only the hover force at the observed $x$, to $10^{-9}$; the quarter
  turn reaches the $+$V-bar in $T/8$ = 709.6 s within a step, within 3 m of $(0, R, 0)$ and
  slower than 0.01 m/s; the range stays within 3 m of $R$; the quarter turn costs $5.5\,nR$ to 3%; and after a thrust
  loss at 0, 30, 60 and 90 deg along the flown arc the free ellipse never comes closer than the
  range at the loss and reaches 2, 4.15, 6.20 and 7 $R$ within $0.1R$.
- `FehseSequenceAgent` from 600 m behind with 900 steps: the modes are `"hop"` then `"final"`,
  the hop starts toward Earth and ends within 1 m of 150 m behind, the final leg keeps
  0.04 m/s to 1 mm/s up to capture (Fehse, section 5.7), and the chaser captures. With 360 steps
  the 4088.5 s plan is refused at reset with its duration in the message.
- `NominalTrackingAgent` on `TwoImpulseAgent(mode="open_loop")` from 300 m behind with 1 m and
  1 mm/s navigation noise: its first command is the plan's to $10^{-9}$, and over seeds 10 to 14
  it arrives closer to the hold point than the open-loop plan (median), Fehse, section 4.4.1.
- Every agent keeps one log entry per decision, takes the protocol arguments by name, sees only
  the time in `info` and holds no simulator object.

## Example

The radial hop from 600 m to the hold point 100 m behind the target, flown with one committed
coast and scored against the transfer's delta-v:

```python
import numpy as np

from vbar import ApproachConfig, ApproachEnv
from vbar.agents.elements import ElementAgent, radial_transfer
from vbar.metrics import record_episode, success
from vbar.spacecraft import PerfectNav


def hop(view, state):
    return radial_transfer(view.n, view.goal["hold_point"][1] - state[1])


config = ApproachConfig(
    start_state=(0.0, -600.0, 0.0, 0.0, 0.0, 0.0),
    action_mode="impulse_coast",
    sensor=PerfectNav(),
)
env = ApproachEnv(config)
ep = record_episode(env, ElementAgent(hop), seed=1, agent_seed=2)
dv = sum(r.actuation.dv_l1 for r in ep.records)
planned = env.agent_view().n * 500.0 / 2.0
print(f"captured: {success(ep)}, delta-v {dv:.4f} m/s against {planned:.4f} m/s planned")
assert success(ep) and np.isclose(dv, planned, rtol=1e-2)
```
