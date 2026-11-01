# Supervisory commands

Code: `vbar.agents.supervision` (`Command`, `Order`, `Rung`, `ladder`, `Situation`,
`SupervisionStep`, `Supervisor`, `SupervisedAgent`, `PHASES`, `MODES`), `vbar.agents.tracker`
(`SetpointTracker`, `speed_cap`), `vbar.agents.rules` (`RuleSupervisor`),
`vbar.agents.llm_supervisor` (`LLMSupervisor`, `parse_order`, `SUPERVISOR_PROMPT`,
`MAX_REPLY_CHARS`), `vbar.agents.learned_supervisor` (`PolicySupervisor`, `supervisor_features`),
`vbar.training` (`SupervisorEnv`). The supervisory channel tests a decision-maker that gives
operator commands at a few instants, as a flight controller does, instead of thrust at each step.
The commands are CONTINUE, GO, HOLD, RETREAT, CAM and THRUST_INHIBIT (RECONFIGURE is refused). A set-point tracker flies the accepted commands, so the supervisor can be a rule book, a
trained policy or a language model and is scored on the same flights.

## Structure

A `SupervisedAgent` holds a supervisor and a `SetpointTracker`. At a consult it builds a
`Situation` from the observation and the agent-safe info, works out the valid commands and asks the
supervisor for a `Command` or an `Order`. A valid order sets the tracker's reference; an invalid
one is recorded with its reason and ignored. Between consults the tracker flies the last accepted
order. The supervisor and the tracker see the observation, the agent-safe info and the scenario's
public description (`AgentView`), so the situation is as noisy as the sensor model makes it.

The task is cut into a ladder of rungs, the places where the chaser stops and waits for the
go-ahead, after the staged approaches and hold points of Fehse (*Automated Rendezvous and Docking
of Spacecraft*, 2003, section 6.5).

## Commands

| Command | Value | Effect | Valid when |
|---|---|---|---|
| CONTINUE | 0 | nothing changes | in each phase |
| GO | 1 | ladder mode: fly to the next rung (after a HOLD or THRUST_INHIBIT, to the current one); waypoint and velocity modes: fly to the order's point or velocity | see below |
| HOLD | 2 | stop where the chaser is and keep station | not already holding or stopped, and station keeping here is affordable |
| RETREAT | 3 | fly back to the previous rung | not retreating, and not on the first rung |
| CAM | 4 | fire a collision avoidance burn; ends the approach | an abort impulse exists and its arc clears the keep-out sphere |
| THRUST_INHIBIT | 5 | command no thrust and drift | the free drift clears the keep-out sphere; inside the commit radius, on a docking only |
| RECONFIGURE | 6 | switch to a redundant unit | in no phase |

`Order(command, point=, velocity=, speed=, unit=, reason=)` adds a destination in the waypoint and
velocity modes; `reason` is logged and not scored. `unit` is carried for RECONFIGURE. A return
value that is neither a `Command` nor an `Order` is refused.

## The ladder

`ladder(view, obs0)` gives the rungs of a task from the public goal and the first observation. A
`Rung` has a position (m, Hill axes at t = 0), a kind and the radius (m) and relative speed (m/s)
within which the chaser has arrived.

| Goal | Rungs |
|---|---|
| `hold`, `berth` with the hold point $\mathbf h \ne 0$ | a "hold" rung at $2\mathbf h$, then the "goal" rung at $\mathbf h$ with the goal's capture radius and speed |
| `hold`, `berth` with $\mathbf h = 0$ | a "hold" rung 50 m along the first observed line of sight, then the "goal" rung at the target |
| `dock` | a "hold" rung at the standoff $\mathbf p + (R_\text{cone} + 10)\hat{\mathbf a}$, a second "hold" rung at $\mathbf p + h_0 \hat{\mathbf a}$ if the goal has a station-keeping point, then the "contact" rung at the port $\mathbf p$ with the goal's lateral limit as radius and the upper end of its axial speed window as speed |
| `inspect` | a "hold" rung at the entry point, 60 m along the first line of sight, then an "orbit" rung at 60 m in the direction of the Sun, projected on the orbit plane |

$\mathbf p$ and $\hat{\mathbf a}$ are the port and its axis, $R_\text{cone}$ is the largest active
range of the approach cones (0 with none), and $h_0$ is the first entry of the goal's `hold`. A
"hold" or "orbit" rung counts as reached within 5 m and 0.05 m/s of its reference
(`ARRIVAL_RADIUS`, `ARRIVAL_SPEED`). An "orbit" rung is the point of the circle that keeps the Sun
behind the inspector at t = 0 and turns with the Sun afterwards ([Tracker](#set-point-tracker)).

`ladder` raises `ValueError` if the goal lacks a key its task needs, if a line of sight or the Sun
direction is zero, if an approach cone has an infinite active range, or if the task has no ladder.
The departure and capture tasks have none.

## Phases

| Phase | Meaning |
|---|---|
| `transfer` | flying to the rung, point or velocity of the last accepted GO; the start in ladder mode |
| `holding` | arrived at the rung or point |
| `stopped` | after HOLD; the start in waypoint and velocity modes, where the tracker holds the first observed position |
| `retreating` | flying to the previous rung after RETREAT |
| `inhibited` | after THRUST_INHIBIT |
| `commit` | inside the goal's `commit_radius` (default 0, which switches the phase off), the last metres of a docking; shown to the supervisor in place of the phase underneath |
| `aborted` | after CAM |

A `transfer` or `retreating` agent becomes `holding` when its observed position is within the
rung's radius of the tracker's reference position and its observed velocity is within the rung's
speed of the reference velocity. In waypoint mode the radius and speed are the "hold" rung's; in
velocity mode the agent does not arrive. A GO from `holding` or `retreating` in ladder mode
increments the rung before the tracker flies to it, so a GO while retreating turns the chaser
back to the rung it left.

## Modes

`SupervisedAgent(mode=)` is one of `MODES`:

- `"ladder"` (default): GO takes no point, velocity or speed. The supervisor chooses only when to
  go on.
- `"waypoint"`: GO carries a `point` (m, Hill) and a `speed` (m/s). The speed must be in
  $(0, v_\text{max}]$ and the point not inside the keep-out sphere. The tracker flies to the point
  and holds it. The rungs are reported, not flown.
- `"velocity"`: GO carries a `velocity` (m/s, Hill) of length at most $v_\text{max}$. The tracker
  flies at it until the next accepted order.

The other commands take no point, velocity or speed.

## The validity mask

The mask is worked out at each consult from the observation, the phase and the public goal, and
is part of the `Situation` with a reason for each refused command (`Situation.reasons`). The
keep-out radius $R$ is the largest radius of the view's `keep_out` constraints (0 with none); the
commit radius is the goal's `commit_radius`.

| Command | Refused when |
|---|---|
| CONTINUE | none |
| GO | `aborted`; `commit`. In ladder mode also: `transfer` (not arrived); `holding` with the arrival no longer held; `holding` on the last rung |
| HOLD | `aborted`; `commit`; `holding`; `stopped`; $\sqrt{(3n^2x)^2 + (n^2z)^2} > f_h\, a_\text{max}$, where the left side is the tidal acceleration a chaser at rest at $(x, y, z)$ must cancel to stay there (Fehse 3.3, Clohessy-Wiltshire) and $f_h$ is the tracker's `hold_accel` |
| RETREAT | `aborted`; `commit`; `retreating`; on rung 0 |
| CAM | `aborted`; `commit`; no abort impulse; the best arc enters the keep-out sphere (below) |
| THRUST_INHIBIT | `aborted`; `inhibited`; `commit` unless the task is a docking; the free drift enters the keep-out sphere (below). Inside the commit radius of a docking the drift is not checked |
| RECONFIGURE | in each phase, "no redundant units" |

The abort impulses are the goal's `cam_impulses`, the set the abort metrics check
([scenarios.md](scenarios.md), [metrics.md](metrics.md)): by default one 0.3 m/s impulse opposite
the approach (Fehse 4.4). With $R > 0$ each impulse $\Delta\mathbf v$ gets the margin

$$m(\Delta\mathbf v) = \min\left(\min_k \rho_k,\ \min_k\left[\tfrac12(\rho_k + \rho_{k+1}) -
\tfrac12\,\delta\,\max(v_k, v_{k+1})\right]\right) - R,$$

where $\rho_k$ and $v_k$ are the range and speed at $t_k = k\delta$ of the free
Clohessy-Wiltshire drift from the observed state after $\Delta\mathbf v$, $\delta$ = 5 s, over two
orbits. The second term lowers each interval's mean range by the most the range can fall between
samples if the speed changes little over it. CAM is valid when the largest margin is not
negative, and the agent fires that impulse. With $R = 0$ the first impulse is used and no arc is
checked. THRUST_INHIBIT uses the same margin without an impulse.

## Consults

The supervisor is consulted at the first observation, at each `period_s` (default 60 s; a whole
number of the scenario's steps), and whenever the phase it would be shown, or the rung, differs
from that at the previous consult, at most once per step. An arrival and an accepted order change
the phase, so they are consulted on the next step. `SupervisedAgent.pending(obs, info)` gives the
`Situation` the next `act` would show, or None between consults, and changes no state beyond
starting the episode on the first call after `reset`; `SupervisorEnv` uses it to find the next
consult.

An order is refused, without raising, when it is not a `Command` or an `Order`, carries fields its
mode or command does not take, has a point or speed or velocity outside the ranges above, or names
a command the mask refuses. The agent records `accepted=False` and the reason in `last_step`
(`SupervisionStep`), leaves the tracker's reference as it was, and shows the reason in
`Situation.refused` at the next consult.

## Set-point tracker

`SetpointTracker` turns a reference into a continuous command in the view's action mode and reads
only the `AgentView`, the observation, the time and its own commands. `reset` raises `ValueError` for a discrete or
burn-and-coast action mode. The command is the acceleration $\mathbf a$ as a fraction of the
thruster limit: clipped per axis for box thrusters, scaled to unit length if longer for ball
thrusters.

| Parameter | Default | Meaning |
|---|---|---|
| `v_max` | 0.5 m/s | highest speed along a leg |
| `gain` | 0.01 1/s | speed per metre to the last point of a leg |
| `k_v` | 0.05 1/s | gain of the velocity loop |
| `wn`, `zeta` | 0.02 rad/s, 1 | natural frequency and damping of the PD hold |
| `capture_range` | 20 m | distance to the last point inside which the PD hold acts |
| `speed_margin` | 0.8 | fraction of the public speed laws the demanded speed may use |
| `clearance` | 1.2 | keep-out radii a leg must pass the target at |
| `hold_accel` | 0.5 | fraction of the thruster limit that holding a place may cost against the tidal acceleration |
| `sensor` | "scenario" | noise of the navigation filter: "scenario", None or `{"sigma_pos", "sigma_vel"}` |
| `lateral_sigma` | 0.045 m | standard deviation across the port axis below which the descent starts |

**Leg.** The acceleration cancels the natural Clohessy-Wiltshire acceleration and drives the
velocity to the leg's (Fehse 3.3, 6.5):

$$\mathbf a = -\begin{pmatrix}3n^2 x + 2n v_y\\ -2n v_x\\ -n^2 z\end{pmatrix} + k\,(\mathbf v_p + s\,\hat{\mathbf u} - \mathbf v),$$

with $\hat{\mathbf u}$ the unit vector to the leg's current point, $\mathbf v_p$ the point's
velocity (0 for a point at rest) and $k$ = `k_v`. The speed $s$ is the least of the leg speed
(`v_max` or the order's), `speed_margin` times the lowest full-speed law at the range, and
`speed_margin` times the lowest closing-speed law divided by the component of $\hat{\mathbf u}$
toward the target. On the last point of a leg it is also at most `gain` times the distance, so the
leg slows to a stop. The laws are the view's closing-speed and speed limits, gate tables
(interpolated between gates, held beyond them) and passive-safety speed envelopes: on the V-bar
$n(\rho - d)/0.47755$, on the R-bar $n\sqrt{(4\rho - d)^2 - 9\rho^2}$, zero within the extent $d$
([llm-agents.md](llm-agents.md)). `speed_cap(view, rho)` is the lowest of them at range $\rho$,
`math.inf` if the view has none.

**Hold.** Within `capture_range` of the last point the law is the PD law $-\omega_n^2(\mathbf r -
\mathbf p) - 2\zeta\omega_n(\mathbf v - \mathbf v_p)$ written as the same velocity loop with $k =
2\zeta\omega_n$ and gain $\omega_n/2\zeta$, so the speed laws still bound the speed it asks for.
`hold(r)` is a leg of one point.

**Detours.** If the straight segment from the start of a leg to its point passes the target
nearer than `clearance` keep-out radii, the leg goes through points on the sphere of radius
$\max(3, 1.3\,\text{clearance})$ keep-out radii, 45 degrees apart along the great circle (through
a perpendicular one if the two directions are opposite), then to the point. The tracker turns to
the next point within $\max(3\ \text{m}, v_\text{leg}/k_v)$ of the current one, $v_\text{leg}$
the leg's speed. The passive-safety speed law falls to zero at the keep-out radius, so a detour
nearer would crawl.

**Docking descent** (kind "contact"). The tracker flies to the standoff $\mathbf p + (R_\text{cone}
+ 10)\hat{\mathbf a}$ as a leg, first walking round the sphere about the port in steps of at most
0.35 rad if the chaser is farther than `capture_range` from the standoff and more than 0.15 rad
off the port axis. Within 5 m of the standoff it descends the axis with desired velocity

$$\mathbf v_\text{des} = \boldsymbol\omega \times (\mathbf p + \max(d, 0)\,\hat{\mathbf a}) - s_d\,\hat{\mathbf a} - 0.03\,\mathbf e,\qquad s_d = \min\left(v_\text{max},\ v_c + 0.003\max(d, 0)\right),$$

where $d = (\mathbf r - \mathbf p)\cdot\hat{\mathbf a}$ is the depth along the axis, $\mathbf e$ the
lateral offset, $v_c$ the middle of the goal's axial speed window and $\boldsymbol\omega$ the
target's angular velocity; the port and axis turn with the target as its briefed attitude mode
says (Fehse 5.3.1). The velocity loop gain is `k_v` while $d > 2$ m and 0.2 1/s inside, so the
speed settles in the window.

**Navigation filter.** On a noisy sensor the tracker flies on a Kalman estimate of the state
instead of the raw observation; the supervisor and the `Situation` keep the raw observation. The
filter is the Clohessy-Wiltshire transition (`vbar.dynamics.cw_discrete`) driven by the tracker's
own commands as accelerations, with a $10^{-9}$ m/s$^2$ process noise floor against rounding and
a Joseph-form update on every `act`, in every mode. Its noise is the white `sigma_pos` and
`sigma_vel` of the catalogue sensor that the view's `scenario_id` names (read by `reset` through
`parse_id`, so `:nav1` gives 1 m and 1 mm/s), or the mapping given as `sensor`; it runs no filter
for an identifier outside the catalogue, a scenario without a Gaussian sensor, or `sensor=None`.
The estimate is `SetpointTracker.estimate`. A docking leg holds the standoff until the estimate's
standard deviation across the port axis is below `lateral_sigma`, or only twice the descent's
planned time (half the standoff over the window's middle speed, plus 100 s) is left. Tests:
`test_the_default_sensor_reads_the_noise_of_the_scenarios_variant`,
`test_the_position_error_of_a_hold_falls_as_one_over_the_root_of_the_fixes`,
`test_the_estimate_follows_a_manoeuvre_by_the_trackers_own_commands`,
`test_two_runs_from_different_true_starts_fed_the_same_observations_agree_at_every_step`,
`test_the_tracker_holds_the_standoff_while_the_lateral_sigma_is_above_the_limit`,
`test_the_tracker_descends_without_waiting_when_only_the_descent_time_is_left`, and the `slow`
flights `test_the_supervised_tracker_docks_on_the_noisy_axis_as_often_as_the_reference_and_cleanly`
and `test_the_supervised_tracker_succeeds_on_as_many_other_noisy_runs_as_the_reference`.

**Orbit** (kind "orbit"). The reference is the circle $\mathbf r_c(t) = R_z(-nt)\,\mathbf r_0$ with
$\mathbf r_0$ the rung position, which turns at $-n$ about the orbit normal and so keeps its angle
to the Sun, fixed in inertial space (Fehse 5.7.3). Over the first 900 s the reference moves from
the start's direction and range to the circle's by a raised cosine in time; the command is the
reference acceleration, minus the natural acceleration at the reference, plus the PD terms on the
position and velocity error.

**Other modes.** `track_velocity(r, v)` flies $\mathbf a = -\mathbf a_\text{cw} + k_v(\mathbf v_\text{cmd}
- \mathbf v)$. `coast()` commands nothing. `fire(dv)` burns $\Delta\mathbf v$ over as many steps as
the thrust limit needs, then coasts; a CAM is such a burn, so it is finite, not an impulse.

## Rule supervisor

`RuleSupervisor(settle_s=60.0, overspeed=1.2, warning_s=600.0, brake_margin=0.5, strikes=2)`
decides by the first rule that applies, after the staged-approach discipline of Fehse 6.5:

1. In the `commit` phase of a docking, THRUST_INHIBIT if valid.
2. Predicted collision, outside the `commit` phase, on a rung that is not "contact" and farther
   out than the goal's `commit_radius`, while closing. The free drift of the observed state is
   propagated with the exact transition matrix (`cw_stm`) at the view's mean motion and sampled
   every 5 s from 5 s up to and including `warning_s`. If a sample comes strictly within the
   largest keep-out radius $R$, and either the chaser is already inside $R$ or stopping at the
   sphere needs more than `brake_margin` times `max_accel`, that is
   $\dot\rho^2 / (2(\rho - R)) > \mathrm{brake\_margin}\cdot a_{max}$, then CAM if valid, else
   RETREAT if valid. A view without a keep-out sphere has no such rule. The matrices are built at
   reset.
3. Braking deficit, in the same places: in `stopped` or `holding`, closing faster than
   `overspeed` times `speed_cap` at the range and not slower than at the previous consult, on
   `strikes` consecutive consults: RETREAT if valid, else CAM if valid. A consult where the
   closing speed fell restarts the count at itself; leaving `stopped` and `holding`, dropping
   under the threshold, or a reset clears it. The first consult after a reset counts as not
   slower.
4. Closer than 1.1 keep-out radii and closing, CAM if valid. The "contact" rung is exempt, since a
   port's keep-out sphere is entered on purpose.
5. In `transfer`, closing faster than `overspeed` times `speed_cap` at the range: HOLD if valid.
   `stopped` and not closing that fast: GO if valid.
6. `holding` with GO valid at each consult for at least `settle_s` seconds: GO.
7. Otherwise CONTINUE.

`warning_s` and `brake_margin` are positive finite numbers and `strikes` a positive integer;
anything else raises `ValueError`. The rules read only the situation and the view.
`tests/test_rule_supervisor_aborts.py` asserts rules 2 and 3 (sample times, the braking
condition, `brake_margin`, strikes and their resets, mask fallbacks, exemptions, priorities,
parameter checks) and, in a slow test, that the supervisor still succeeds without a keep-out
violation on the core scenarios.

## Language-model supervisor

`LLMSupervisor(complete, prompt=SUPERVISOR_PROMPT, price=None, mode="ladder")` asks a model at
each consult. `complete(messages, tools)` is the model function of `LLMAgent`
([llm-agents.md](llm-agents.md)): it gets one user message and an empty tool list, and returns a
`Reply` or text. `RecordedModel` and `ReplayModel` repeat a run without the model.

The prompt is `SUPERVISOR_PROMPT` filled by `str.format` over `SUPERVISOR_FIELDS`: the frame, the
task briefing of `vbar.agents.briefing` ([llm-agents.md](llm-agents.md)), the rungs numbered from
1, what GO means in the mode, the time, phase and rung, the valid commands, each other command with
its reason, the reason the last order was refused, and the reply format.

The reply is a JSON object, `{"command": "GO", "reason": "settled"}`; in waypoint mode a GO also
carries `"point": [x, y, z]` (m) and `"speed"` (m/s), in velocity mode `"velocity": [vx, vy, vz]`
(m/s). `parse_order(text, mode)` reads the first JSON object in the reply, wherever it stands
among prose. The command is a `Command` name in any case with blanks around it. The point and
velocity need three finite numbers and the speed a finite number; a missing, quoted, non-finite or
too large field makes the reply unusable. Fields of another mode, and a point, speed or velocity
on any command but GO, are ignored. A speed or point outside the tracker's range is passed on and
refused by the agent. A reason that is not a string is dropped.

A reply that holds no order is a parse failure: it is logged and answered with CONTINUE. So is a
reply of more than `MAX_REPLY_CHARS` (8192) characters, which is not read; an order is a few
hundred characters. The order is the first JSON object among the reply's first eight opening
braces, so prose before it may hold a few braces and the object may nest. Each log entry holds the
time, prompt, reply, the order's reason (None on a parse failure), the failure flag, model calls,
tokens, cost (None without a `price`) and latency. Errors the model raises, including
`ReplayError`, are not caught.

A reply is invalid when it holds no order or orders a command that is not valid now. With
`LLMSupervisor(..., repairs=k)` an invalid reply is sent back: the model gets the conversation so
far, its reply, the reason ("Your reply was not a command in the requested format." or "RETREAT
is not available now: ..." with the agent's reason), the commands valid now and the reply format,
at most k times per consult. The default is 0, no repair. A final reply with no order is CONTINUE,
as above; a final order for a command that is not valid now is passed on and the agent refuses it,
so the supervisor fails closed either way. The log entry adds `invalid` (the final reply was
invalid) and `repair_replies` (the replies sent back, in order), and its calls, tokens, cost and
latency cover every call of the consult. `usage()` totals decisions, calls, tokens, cost, parse
failures, `invalid_actions` (consults whose final reply was invalid), `invalid_replies` (every
invalid reply, repaired or not), `repair_attempts` and `repaired` (consults an invalid reply was
repaired at). The invalid-action rate of a run is `invalid_actions / decisions`, and the repair
attempt rate `repair_attempts / decisions`. The supervisor judges a reply against the mask only: a
GO whose destination the agent then refuses (a waypoint inside the keep-out sphere, a speed out of
the tracker's range) is valid here and shows as a refused step of the `SupervisedAgent`, so a
run's refusals are the invalid actions plus the agent's refused steps. A consult whose model call
raises, `BudgetExceeded` included, is not logged; a `BudgetedModel` still counts every call made.

In real time (`run_episode(..., real_time=RealTime(time_scale=s))`) the clock runs while the
supervisor's model answers: each step's order takes effect `s` times the agent's wall-clock
decision time later, so a slow model's orders arrive late, and a consult costs flight time that a
rule supervisor's does not. The result's `latencies_s` and a `RecordedModel` transcript repeat such
a run exactly through `RealTime(latencies=..., time_scale=s)` and `ReplayModel`.

## Learned supervisor

`supervisor_features(situation, view)` is the vector a learned policy acts on, built from a
`Situation` alone: the observation (m, m/s), the observed position minus the position of the rung
being flown to (3, m), a one-hot of the phase in the order of `PHASES` (7), the rung index over
the number of rungs, in $[0, 1)$ (1), and the mask, 1 for a valid command, in the order of
`Command` (7).

`PolicySupervisor(policy)` calls `policy(features)` and takes the integer as a `Command`
(`ValueError` for a value that is not one). The policy is shared between the agents built on it.

`SupervisorEnv(env, tracker=None, period_s=60.0, refused_penalty=0.0, reward_hook=None)` is a
Gymnasium wrapper whose action is one of the seven `Command` values (`Discrete(7)`) and whose
observation is `supervisor_features`, so a trained policy flies the same trajectory in a
`SupervisedAgent`. A step gives the command to a `SupervisedAgent` and runs the inner environment
until the agent next consults. The reward is the inner rewards summed with each discounted by the
inner discounts before it; `info["discount"]` is the product of the inner discounts (1 if the inner
environment has none). A command the mask refuses is ignored and costs `refused_penalty` more.
`reward_hook(step)` is called once per step with the `SupervisionStep` of the consult and its
result is added. `action_masks()` is the bool mask for the sb3-contrib convention and
`info["action_mask"]` the same as int8.

## Limits

- Only hold, berthing, docking and inspection goals have a ladder. The departure and capture tasks
  have none: `ladder` raises `ValueError` for them.
- `SupervisorEnv` is a training interface: like the base environment and `FilteredEnv`, it passes
  the inner environment's `info`, `true_state` included, to the learner. The features a
  `PolicySupervisor` is scored on come from the `Situation` only.
- RECONFIGURE is refused: the chaser has no redundant units.
- The drift and CAM checks of the mask sample the arc each 5 s over two orbits and pad each
  interval by the greater end speed. That is a heuristic, not a certificate: a command the mask
  allows can still enter the keep-out sphere between samples or after two orbits, and one it
  refuses can be safe. The metrics, not the mask, score safety.
- A CAM ends the approach. Afterwards GO, HOLD, RETREAT, CAM and THRUST_INHIBIT are refused and
  the chaser coasts.
- The mask and the situation come from the observation, so they carry the sensor's error.
- The mode of an `LLMSupervisor` has to be the mode of its `SupervisedAgent`: the situation
  carries the agent's mode, and the first consult raises `ValueError` on a mismatch before the
  model is called (`test_a_supervisor_whose_mode_is_not_the_agents_is_refused_before_the_model_is_called`).
- A model reply of more than `MAX_REPLY_CHARS` characters is a parse failure, however good the
  order inside it.
- The orbit rung in the learning features is at its position at t = 0. It turns with the Sun, so
  the gap to it matches the rung's position only at t = 0.
- `SupervisedAgent`, `SetpointTracker` and `SupervisorEnv` need a continuous action mode.
- The tracker flies the reference it is given; a command that asks for more than the thrusters
  give is clipped.
- A keep-out radius of zero (no `keep_out` constraint) switches off the arc checks of CAM and
  THRUST_INHIBIT and the detour; a commit radius of zero switches off the `commit` phase.

## How it is tested

Each line is a claim of this page and the test that asserts it; "(p)" marks a parametrized test.
Flights are marked `slow`.

`tests/test_supervision.py`, ladder:

- Hold goal off the origin, hold rung at 2h: `test_a_hold_goal_off_the_origin_gets_a_hold_rung_twice_as_far_and_the_goal`.
- Berthing goal, same rungs, goal rung with the capture radius and speed: `test_a_berthing_goal_gets_a_hold_rung_twice_as_far_and_the_goal`.
- Goal rung radius and speed on approach-vbar, approach-rbar and berthing-box (p): `test_the_goal_rung_arrives_within_the_capture_radius_and_speed_of_the_goal`.
- Hold goal at the origin, rung 50 m along the line of sight, and the line of sight of a drawn seed: `test_a_hold_goal_at_the_origin_gets_a_rung_50_m_along_the_line_of_sight`, `test_the_line_of_sight_rung_follows_the_first_observation_of_a_drawn_seed`.
- Docking standoff outside every cone, the station-keeping rung, and the standoff 10 m out without a cone: `test_a_docking_goal_gets_a_standoff_outside_every_cone_and_the_port_as_contact`, `test_a_docking_goal_with_a_station_keeping_point_gets_a_rung_there_before_the_port`, `test_a_docking_without_an_approach_cone_gets_its_standoff_10_m_out_on_the_axis`.
- Contact rung radius is the lateral limit and speed the top of the axial window: `test_the_contact_rung_arrives_within_the_lateral_limit_and_below_the_top_axial_speed`.
- Inspection entry and orbit rungs: `test_an_inspection_goal_gets_an_entry_point_at_60_m_and_the_sun_behind_orbit_point`.
- Hold and orbit rungs arrive within 5 m and 0.05 m/s: `test_a_hold_rung_arrives_within_5_m_and_0_05_m_s`, `test_the_orbit_rung_arrives_within_5_m_and_0_05_m_s_like_the_entry_rung`.
- The five core scenarios are the ones the ladder covers: `test_the_base_core_scenarios_are_the_five_the_ladder_covers`.
- `ValueError` for a goal missing a key of the hold, docking or inspection task, naming it (p): `test_a_goal_missing_a_key_its_task_needs_raises_value_error_naming_the_key`, `test_an_inspection_goal_without_a_sun_direction_raises_value_error_naming_it`.
- `ValueError` for a zero line of sight, hold goal at the target and inspection: `test_a_hold_goal_at_the_target_with_a_zero_first_line_of_sight_raises_value_error`, `test_an_inspection_with_a_zero_first_line_of_sight_raises_value_error`.
- `ValueError` for a zero Sun direction and one along the orbit normal (p): `test_an_inspection_goal_with_no_sun_direction_in_the_orbit_plane_raises_value_error`.
- `ValueError` for an infinite cone range and for the departure and capture tasks: `test_an_approach_cone_with_an_infinite_active_range_raises_value_error`, `test_the_departure_and_capture_tasks_have_no_ladder`.

`tests/test_supervision.py`, phases, mask and orders:

- Seven commands valued 0 to 6 and seven phases: `test_the_commands_and_phases_are_the_declared_ones`.
- First consult at t = 0 in `transfer` on rung 0: `test_the_first_consult_is_at_t_0_in_transfer_to_rung_0_with_the_ladder_and_mask`.
- Each phase is reached by its orders and arrivals (p): `test_the_orders_and_arrivals_that_reach_each_phase_put_the_agent_in_it`.
- Every cell of the mask table, command by phase: GO before and after arrival and on the last rung, HOLD against the station-keeping cost on the radial and normal axes, RETREAT on rung 0, CAM and THRUST_INHIBIT with arcs that clear and enter the keep-out sphere, `commit` on a hold and a docking, `aborted`, a reason for each refusal (p): `test_the_mask_allows_a_command_exactly_when_the_phase_and_state_conditions_hold`.
- The drift states of those cases have the drift they are built for, against the Clohessy-Wiltshire transition matrix: `test_the_free_drift_states_of_the_mask_cases_have_the_drift_they_are_built_for`.
- CONTINUE is valid and RECONFIGURE refused with "no redundant units" in each phase (p): `test_continue_is_always_valid_and_reconfigure_is_refused_for_want_of_a_redundant_unit`.
- A refused order is recorded and shown at the next consult: `test_an_order_the_mask_refuses_is_recorded_as_refused_and_shown_at_the_next_consult`.
- Malformed orders of each mode are refused and move no reference (p): `test_an_invalid_order_is_refused_and_recorded_without_moving_the_reference`.
- A command other than GO given a point, a velocity or a speed is refused, with the reason "takes no point, velocity or speed", in the three modes (p): `test_a_command_other_than_go_given_a_point_a_velocity_or_a_speed_is_refused_and_changes_nothing`; the bare command is accepted (p): `test_the_bare_command_is_accepted_where_the_same_command_with_a_point_is_refused`.
- Valid orders move the reference: `test_a_valid_waypoint_order_moves_the_reference_to_the_point`, `test_a_valid_velocity_order_sets_the_reference_velocity`, `test_a_valid_go_moves_the_ladder_reference_to_the_next_rung`.
- A situation is frozen and holds copies: `test_a_situation_is_frozen_and_holds_copies_of_the_observation`.
- The discrete and burn-and-coast modes are refused at reset (p): `test_a_supervised_agent_refuses_an_action_mode_that_is_not_continuous`.
- Equal records from equal seeds, and agents from one tracker stepped alternately: `test_the_same_seeds_give_equal_supervision_records_and_commands`, `test_two_agents_from_one_config_stepped_interleaved_equal_each_flying_alone`.

`tests/test_supervision.py`, consults and truth:

- Consults at t = 0 and every period for 10, 30, 60 and 120 s (p): `test_the_supervisor_is_consulted_at_t_0_and_every_period_while_nothing_changes`.
- Extra consults on an arrival and on an accepted order, at most one per step: `test_a_phase_change_is_consulted_between_periods_by_arrival_and_by_an_order`, `test_the_supervisor_is_consulted_at_most_once_per_step_however_often_the_phase_changes`.
- `pending` gives the situation the next `act` records and changes no state: `test_pending_gives_the_situation_the_next_act_records_and_changes_no_state`.
- A period that is not a whole number of steps, zero or negative is refused: `test_a_period_that_is_not_a_whole_number_of_steps_raises_value_error`.
- Starts more than 10 m apart give identical situations and commands: `test_runs_from_true_starts_more_than_10_m_apart_give_identical_situations_and_commands`.
- No number of the true state is in a situation or step repr, and the scan finds a planted leak: `test_the_repr_of_a_situation_and_of_a_step_holds_no_number_of_the_true_state`, `test_the_repr_scan_finds_a_leaked_true_state`.
- `slow`: the rule supervisor succeeds without a keep-out violation on the five core scenarios and five seeds each, and a CONTINUE-only supervisor does not succeed on any: `test_the_rule_supervisor_over_the_tracker_succeeds_without_a_keep_out_violation`, `test_a_supervisor_that_only_continues_never_succeeds_and_violates_nothing`.

`tests/test_tracker.py`:

- `speed_cap` gives the closing-speed and full-speed laws, the gate table between and beyond its gates, and the V-bar and R-bar passive-safety laws, and takes the lowest: `test_a_closing_speed_limit_gives_its_linear_law_in_the_range`, `test_a_full_speed_limit_gives_its_linear_law_in_the_range`, `test_a_gate_table_is_interpolated_between_its_gates_and_held_beyond_them`, `test_a_v_bar_passive_speed_law_is_zero_at_the_extent_and_grows_with_the_range`, `test_an_r_bar_passive_speed_law_is_zero_within_the_extent_and_follows_the_briefed_formula`, `test_the_cap_is_the_lowest_of_the_laws_that_apply`.
- On each catalogue scenario with a speed law the cap is the lowest public law at seven ranges, and the catalogue has each kind of law: `test_the_cap_of_every_catalogue_scenario_is_the_lowest_of_its_public_laws`, `test_the_catalogue_has_scenarios_for_every_kind_of_speed_law`.
- The command stays within a full command per axis or in length, in the three continuous action modes: `test_the_command_never_exceeds_a_full_command_per_axis_or_in_length`.
- The discrete and burn-and-coast modes are refused at reset (p): `test_the_tracker_refuses_an_action_mode_that_is_not_continuous_at_reset`.
- Holding at rest cancels the natural acceleration, and flying at a velocity cancels the Coriolis terms, to a relative 1e-12: `test_holding_at_rest_off_v_bar_cancels_the_cw_acceleration`, `test_tracking_a_velocity_cancels_the_coriolis_terms`.
- A leg through the target goes round the keep-out sphere: `test_the_tracker_flies_round_the_keep_out_sphere_when_the_leg_would_cross_it`.
- `slow`: a GO at each consult flies the five core scenarios, which exercise the docking descent and the orbit reference, to success without a keep-out violation: `test_the_tracker_given_go_at_every_consult_flies_each_base_scenario`.

`tests/test_rule_supervisor.py`, each rule alone on constructed situations:

- Defaults `settle_s` 60 and `overspeed` 1.2: `test_the_defaults_are_a_settle_time_of_60_s_and_an_overspeed_factor_of_1_2`.
- Rule 1 acts in the `commit` phase of a docking only, and not when the mask refuses it: `test_a_chaser_inside_the_commit_radius_of_a_docking_is_told_to_inhibit_thrust`, `test_the_commit_phase_of_a_task_other_than_a_docking_does_not_inhibit_thrust`, `test_a_docking_outside_the_commit_phase_does_not_inhibit_thrust`, `test_thrust_is_not_inhibited_when_the_mask_refuses_it`.
- Rule 2 fires inside 1.1 keep-out radii when closing (p) and not when opening, at rest, on the "contact" rung, when refused or without a keep-out sphere: `test_a_chaser_closing_inside_1_1_keep_out_radii_is_told_to_fire_a_cam_and_not_outside`, `test_a_chaser_that_is_opening_inside_the_cam_range_is_not_told_to_fire_a_cam`, `test_a_chaser_at_rest_inside_the_cam_range_is_not_told_to_fire_a_cam`, `test_the_contact_rung_is_exempt_from_the_cam_rule`, `test_a_cam_is_not_ordered_when_the_mask_refuses_it`, `test_a_view_without_a_keep_out_sphere_never_gets_a_cam`.
- Rule 3 holds above 1.2 speed caps (p), scales with `overspeed`, and acts in `transfer` only and not when refused: `test_a_transferring_chaser_closing_faster_than_1_2_speed_caps_is_told_to_hold`, `test_overspeed_scales_the_threshold_on_the_speed_cap`, `test_an_overspeeding_chaser_is_not_told_to_hold_when_the_mask_refuses_it`, `test_overspeed_outside_the_transfer_phase_is_not_answered_with_a_hold`.
- Rule 3 sends a stopped chaser on below the threshold and not above it or when refused: `test_a_stopped_chaser_below_the_overspeed_threshold_is_told_to_go`, `test_a_stopped_chaser_still_closing_above_the_threshold_is_not_told_to_go`, `test_a_stopped_chaser_is_not_told_to_go_when_the_mask_refuses_it`.
- Rule 4 sends on after 60 s held, after `settle_s` when set, and restarts the wait when GO is lost, the phase changes or a new episode starts: `test_a_holding_chaser_is_told_to_go_once_the_arrival_has_been_held_for_60_s`, `test_settle_s_sets_how_long_the_arrival_is_held_before_the_go_ahead`, `test_losing_the_go_ahead_restarts_the_settle_time`, `test_leaving_the_holding_phase_restarts_the_settle_time`, `test_a_new_episode_starts_the_settle_time_afresh`.
- Rule 5, CONTINUE in `transfer`, `inhibited` and `aborted` without a trigger (p): `test_a_situation_no_rule_covers_gets_continue`.
- Priority: `test_the_commit_rule_beats_the_cam_rule`, `test_the_cam_rule_beats_the_overspeed_rule`, `test_the_cam_rule_beats_the_go_ahead_of_a_stopped_chaser`, `test_the_cam_rule_beats_the_go_ahead_of_a_settled_holding_chaser`.

`tests/test_llm_supervisor.py`:

- Each command name parses in each mode, in any case and trimmed (p): `test_every_command_name_parses_in_every_mode`, `test_command_names_are_read_case_insensitively_and_trimmed`.
- An unknown name, a value that is not a name and a reply without an object are not orders: `test_an_unknown_command_name_is_not_an_order`, `test_a_command_that_is_not_a_name_is_not_an_order`, `test_a_reply_without_a_command_object_is_not_an_order`.
- Prose around the object is ignored and the first object counts: `test_prose_around_the_json_object_is_ignored`, `test_the_first_object_in_a_reply_is_the_one_that_counts`.
- The point, speed and velocity are read; a GO without a well-formed field, a non-finite number or one too large for a float is not an order; fields of another mode are ignored; a destination the mode cannot fly is passed on: `test_waypoint_mode_reads_the_point_and_the_speed`, `test_velocity_mode_reads_the_velocity`, `test_a_waypoint_go_without_a_well_formed_point_and_speed_is_not_an_order`, `test_a_velocity_go_without_a_well_formed_velocity_is_not_an_order`, `test_a_non_finite_number_makes_the_reply_not_an_order`, `test_a_number_too_large_for_a_float_makes_the_reply_not_an_order`, `test_fields_that_belong_to_another_mode_are_ignored`, `test_a_destination_the_mode_cannot_fly_is_left_for_the_agent_to_refuse`.
- A reason is kept, and one that is not a string (a number, boolean, list, null or object) becomes empty, in each mode (p): `test_a_reason_is_kept_and_defaults_to_empty`, `test_a_reason_that_is_not_a_string_gives_an_order_with_an_empty_reason`, `test_a_reason_that_is_an_object_gives_an_order_with_an_empty_reason`.
- An order is read within the first eight opening braces and not after, one nested deeper than `MAX_JSON_DEPTH` is not read, and deep brackets after the order do not hide it: `test_an_order_is_read_only_within_the_first_eight_opening_braces`, `test_an_order_nested_deeper_than_the_json_limit_is_not_read`, `test_deep_brackets_after_the_order_do_not_hide_it`.
- `parse_order` does not raise on two million characters, 5000 braces or 5000 levels of nesting, and returns only orders the agent could fly on generated replies: `test_an_over_long_reply_does_not_raise`, `test_parse_order_never_raises_and_returns_only_orders_it_could_fly`.
- The prompt holds the frame, briefing, ladder and situation lines of each core scenario, names the phase and rung, asks for a JSON command, and the model gets one message and no tools: `test_the_prompt_holds_the_frame_the_briefing_the_ladder_and_the_situation_lines`, `test_the_prompt_names_the_phase_and_the_rung_of_the_situation`, `test_the_prompt_asks_for_a_json_command_and_the_model_gets_one_message_and_no_tools`.
- It lists the valid commands and each other with its reason, and the last refusal appears in the next prompt only, also in a flown agent: `test_the_prompt_lists_exactly_the_valid_commands_and_each_other_with_its_reason`, `test_the_last_refusal_is_in_the_next_prompt_only`, `test_the_prompt_of_a_flown_agent_lists_the_commands_its_mask_allows`, `test_a_refused_order_reaches_the_next_prompt_of_a_flown_agent`.
- A goal without a key its task needs is refused at reset: `test_a_goal_without_a_key_its_task_needs_is_refused_at_reset`.
- Usage totals decisions, tokens, cost and parse failures; an unparseable reply is CONTINUE, leaves the reference and is counted; a reply of `MAX_REPLY_CHARS` parses and one more does not: `test_usage_counts_decisions_tokens_cost_and_parse_failures`, `test_an_unparseable_reply_is_continue`, `test_an_unparseable_reply_leaves_the_tracker_reference_and_is_counted`, `test_a_reply_longer_than_the_bound_is_a_parse_failure`.
- An unparseable reply or a command not valid now is one invalid action without repairs; a repair sends back the reply with its reason and the valid commands, the repaired order counts, a valid first reply is not repaired, a reply still invalid after every repair is CONTINUE or the refused order, and `repairs` must be a non-negative integer: `test_without_repairs_an_unparseable_reply_is_one_call_and_one_invalid_action`, `test_without_repairs_a_command_not_valid_now_is_passed_on_and_counted_invalid`, `test_a_repaired_reply_is_the_order_and_the_model_saw_its_reply_and_the_reason`, `test_the_repair_of_a_command_not_valid_now_names_it_and_its_reason`, `test_a_valid_first_reply_is_not_repaired`, `test_a_reply_still_invalid_after_every_repair_is_continue_or_the_refused_order`, `test_repairs_must_be_a_non_negative_integer`.
- `slow`: in real time the model's wait is measured as the decision latency, a decision latency changes a supervised flight (and the same latencies give the same flight), and the run replays exactly from its recorded replies and latencies: `test_the_models_wait_is_measured_as_the_decision_latency`, `test_a_decision_latency_delays_the_supervised_flight`, `test_a_real_time_supervised_run_replays_with_its_recorded_replies_and_latencies`.
- `slow`: a stand-in model that reads only the prompt gives the rule supervisor's orders: `test_a_stand_in_reading_only_the_prompt_gives_the_rule_supervisors_orders_early_in_a_flight`, `test_a_stand_in_reading_only_the_prompt_gives_the_rule_supervisors_orders_on_a_drawn_seed`.
- A recorded run replays to identical steps, and one byte changed in the prompt or request raises `ReplayError`: `test_a_recorded_run_replays_to_identical_supervision_steps_without_the_model`, `test_a_replay_with_one_byte_of_the_prompt_changed_raises_replay_error`, `test_a_replay_with_one_byte_of_the_request_changed_raises_replay_error`.
- Starts more than 10 m apart give identical prompts, no number of the true state is in a prompt, and a misbriefed target's prompt equals its briefing's: `test_runs_from_true_starts_more_than_10_m_apart_give_identical_prompts`, `test_no_number_of_the_true_state_is_in_the_prompts_of_a_flight`, `test_the_prompt_of_a_misbriefed_target_is_the_prompt_of_its_briefing`.

`tests/test_supervisor_env.py`:

- The wrapper passes `check_env`; the action space is `Discrete(7)` and the observation a flat float box holding the observations: `test_the_wrapper_passes_the_gymnasium_environment_checker`, `test_the_action_space_is_one_of_the_seven_commands`, `test_the_observation_space_is_a_flat_float_box_that_holds_the_observations`.
- The features are the observation, rung gap, phase, rung and mask, follow an arrival and a GO, hold nothing of the info beyond the situation, and equal the situation's features at each consult: `test_the_features_are_the_observation_the_rung_gap_the_phase_the_rung_and_the_mask`, `test_the_features_follow_the_phase_and_the_rung_after_an_arrival_and_a_go`, `test_the_features_hold_nothing_of_the_info_beyond_what_the_situation_was_given`, `test_the_wrapper_observations_are_the_features_of_the_situation_at_each_consult`.
- The masks are the situations' masks, as bool and int8; CONTINUE is valid and RECONFIGURE is not: `test_the_masks_are_the_situations_masks_at_every_decision_as_bool_and_as_int8`, `test_continue_is_always_valid_and_reconfigure_never`.
- A step runs one period, ends early at a phase change, and a CONTINUE episode truncates at the inner step limit: `test_a_step_runs_the_inner_environment_for_one_period_and_gives_the_time_reached`, `test_the_period_sets_the_inner_steps_of_a_step_while_nothing_changes`, `test_a_step_ends_at_the_next_consult_whether_by_the_period_or_by_a_phase_change`, `test_an_episode_of_continue_commands_ends_truncated_at_the_inner_step_limit`.
- The reward is the discounted sum and the discount their product; a refused command costs the penalty once for each of the seven commands (p); the hook's value is added; a given tracker flies the commands: `test_the_reward_is_the_discounted_sum_of_the_inner_rewards_and_the_discount_their_product`, `test_a_refused_order_costs_the_refused_penalty_and_an_accepted_one_does_not`, `test_the_penalty_is_paid_once_for_each_refused_order_and_never_for_a_valid_one`, `test_the_reward_hook_is_called_for_each_step_with_the_supervision_step_and_added`, `test_a_tracker_given_to_the_wrapper_flies_the_commands`.
- Starts more than 10 m apart give identical observations and masks: `test_runs_from_true_starts_more_than_10_m_apart_give_identical_observations_and_masks`.
- A `PolicySupervisor` gives the policy's command and flies the wrapper's trajectory for the same policy: `test_a_policy_supervisor_gives_the_policys_command_on_the_features_of_the_situation`, `test_a_policy_supervisor_flies_the_trajectory_of_the_wrapper_with_the_same_policy`.
- A period that is not a positive whole number of steps, a penalty that is not a finite number not below zero, a non-continuous action mode and a non-integer action are refused before anything is flown: `test_a_period_that_is_not_a_positive_whole_number_of_steps_raises_value_error`, `test_a_refused_penalty_that_is_not_a_finite_number_not_below_zero_raises_value_error`, `test_an_action_mode_that_is_not_continuous_raises_value_error_naming_it`, `test_an_action_that_is_not_an_integer_command_is_refused_before_anything_is_flown`, `test_a_whole_number_period_and_an_integer_penalty_are_accepted`.

## Example

A rule supervisor flies 150 steps of the V-bar approach. The chaser reaches the first rung, the
arrival holds for 60 s, and the supervisor sends it on; the loop prints each order but CONTINUE
with the time (s), the phase and the rung in effect after it:

```python
from vbar.agents import RuleSupervisor, SupervisedAgent
from vbar.scenarios import make

env = make("approach-vbar", max_steps=150)
agent = SupervisedAgent(RuleSupervisor(), period_s=60.0)
obs, info = env.reset(seed=0)
agent.reset(env.agent_view())
done = False
while not done:
    action = agent.act(obs, env.agent_info())
    step = agent.last_step
    if step.consulted and step.order.command.name != "CONTINUE":
        print(step.t, step.order.command.name, step.phase, step.rung)  # 1440.0 GO transfer 1
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

A language model supervisor with a stand-in model that needs no API. It asks to go on at once,
which the mask refuses because the chaser has not reached the first rung:

```python
from vbar.agents import LLMSupervisor, Price, Reply, SupervisedAgent
from vbar.scenarios import make


def complete(messages, tools):
    """Replace with a call to a model; this one always asks for GO."""
    return Reply('{"command": "GO", "reason": "ready"}', input_tokens=1200, output_tokens=15)


supervisor = LLMSupervisor(complete, price=Price(3.0, 15.0))
agent = SupervisedAgent(supervisor, period_s=60.0)
env = make("approach-vbar", max_steps=3)
obs, info = env.reset(seed=0)
agent.reset(env.agent_view())
agent.act(obs, env.agent_info())
step = agent.last_step
print(step.accepted, step.reason)  # False GO: not arrived at the rung
print(round(supervisor.usage()["cost_usd"], 6))  # 0.003825
```

A learner picks the commands. The first mask allows CONTINUE, HOLD, CAM and THRUST_INHIBIT: the
chaser has not reached a rung, so GO is refused, there is no previous rung, so RETREAT is refused,
and RECONFIGURE is refused. The refused GO is ignored and the step runs one period:

```python
from vbar.scenarios import make
from vbar.training import SupervisorEnv

env = SupervisorEnv(make("approach-vbar", max_steps=60), period_s=60.0)
obs, info = env.reset(seed=0)
print(obs.shape, info["action_mask"])  # (24,) [1 0 1 0 1 1 0]
obs, reward, terminated, truncated, info = env.step(1)
print(info["time_s"])  # 60.0
```
