# Space Shuttle rendezvous baselines

Code: `nsr_aim`, `football_aim`, `elevation`, `ElevationTimedAgent` and `PilotAgent` in
`vbar.agents.shuttle`. Hill frame, metres, seconds.

The Space Shuttle flew rendezvous with a small set of operational rules: a position-only CW
targeter turned into a coelliptic or a no-drift burn by the choice of aim point and transfer
time, burns timed on the elevation of the target instead of the clock, and a final approach
flown by hand in fixed pulses against a range-rate gate. These baselines reproduce those rules
so a learned or language-model controller can be compared against flight practice. The source
is D. Goodman, *History of Space Shuttle Rendezvous*, JSC-63400 Rev 3 (NASA, 2011).

## How it is built

**Frames and units.** Goodman works in an LVLH frame with $+X$ along the V-bar, $+Y$ against the
orbit normal and $+Z$ toward the Earth, in ft, ft/s and nautical miles. A book point
$(X, Y, Z)$ is $(x, y, z) = (-Z, X, -Y)$ in the Hill frame (`vbar.conventions.iss_to_hill`), so
a chaser $Z_0$ "above" the target in the book is at $x_0 = -Z_0$ here. Conversions use
1 ft = 0.3048 m and 1 n.mi. = 1852 m; an orbit at altitude $h$ has radius the WGS 84 equatorial
radius plus $h$.

**NSR aim point.** For a chaser at $\mathbf r_0 = (x_0, y_0, z_0)$, `nsr_aim(r0, n)` returns

$$
\mathbf r_f = (x_0,\; y_0 - 2x_0,\; 0), \qquad t_f = \frac{4}{3n},
$$

Goodman's $(X_0 + 2Z_0, Z_0)$ in Hill coordinates. Coelliptic motion at height $x_0$ drifts along
track at $-\tfrac32 n x_0$, which covers $-2x_0$ in $4/(3n)$, so CW targeting to this point
(`vbar.agents.cw_targeting.cw_lambert`) returns $\mathbf v_0 = \mathbf v_f =
(0, -\tfrac32 n x_0, 0)$: the coelliptic burn.

**Football aim point.** `football_aim(r0, n)` returns

$$
\mathbf r_f = (0,\; y_0 - 2x_0,\; 0), \qquad t_f = \frac{T}{4} = \frac{\pi}{2n},
$$

Goodman's $(X_0 + 2Z_0, 0)$. The targeted velocity is $(0, -2nx_0, 0)$, which meets the no-drift
condition $\dot y + 2nx = 0$ and centres the relative ellipse on the V-bar, so the chaser
returns to $\mathbf r_0$ after one period. Both rules aim a start out of the orbit plane back
into it.

**Elevation.** The elevation of the target above the chaser's local horizontal is

$$
\mathrm{el} = \arcsin\left(-\frac{x}{|\mathbf r|}\right),
$$

positive when the chaser is below the target and zero on the V-bar (`elevation(r)`).

**Elevation-timed burn.** `ElevationTimedAgent(trigger=...)` flies a CW transfer of
$t_f = $ `n_tf` $/n$ to the scenario's hold point, fired when the elevation reaches `trigger`
(rad). At its first decision it propagates the unforced CW coast from the observation, sampled
every quarter step up to the episode's last decision, finds the first sample at or above the
trigger and refines the crossing time with Brent's method (`scipy.optimize.brentq`). The burn is
scheduled at that time rounded to a step, so it can fire up to half a step before the elevation
gets there. Every later decision re-predicts the crossing: a crossing within `slip_s` of the one
that set the schedule moves the burn, one further away is ignored and logged as
`slip_exceeded`. If the first prediction finds no crossing before the episode's last decision,
the burn fires at once and the entry is logged as `trigger_not_reached`, so the agent degrades
to two-impulse targeting. From the burn on, a re-targeting
[`TwoImpulseAgent`](two-impulse.md) flies the transfer, its arrival burn and the hold.

**Pilot model.** `PilotAgent()` flies the final approach the way Shuttle crews did by hand.
With $\mathbf d$ the vector from the chaser to the hold point, $\rho = |\mathbf d|$,
$\mathbf u = \mathbf d / \rho$, closing rate $\dot\rho_c = \mathbf v \cdot \mathbf u$ and gate
rate

$$
\dot\rho^* = \max\left(\frac{\rho}{\tau},\ \dot\rho_\text{min}\right),
$$

each decision outside a read period applies the first rule that holds:

1. inside the capture radius, while $|\mathbf v|$ is at least half a pulse: a pulse against
   $\mathbf v$;
2. line-of-sight rate $|\mathbf d \times \mathbf v| / \rho^2$ above `los_rate_limit`: a pulse
   against the velocity across the line of sight, $\mathbf v - \dot\rho_c\mathbf u$;
3. $\dot\rho_c < (1 - b)\,\dot\rho^*$: a pulse along $\mathbf u$;
4. $\dot\rho_c > (1 + b)\,\dot\rho^*$: a pulse against $\mathbf u$;

and otherwise coasts. Each pulse is `pulse` m/s delivered in one step (a fraction 0.1524 of a
full command at the default 0.02 m/s² and 10 s step), followed by `read_s` seconds without
thrust, which the `impulse_coast` action mode receives as the coast of the action. The gate
profile $\rho/\tau$ closes exponentially with time constant $\tau$ until the floor
$\dot\rho_\text{min}$ takes over at $\rho = \tau\dot\rho_\text{min}$ (30.48 m at the defaults).

**Log.** Both agents keep `agent.log`, one entry per decision with the time `t`.
`ElevationTimedAgent` adds the `phase` (`wait` before the burn, then the two-impulse agent's
phase and `saturated` flag) and, before the burn, `predicted_crossing_s` and the scheduled
`burn_t`; the flags `trigger_not_reached` and `slip_exceeded` are set on the decision where they
occur. A decision that commits a coast adds `coast_s`, `predicted_margin` and
`predicted_margin_bound`: a wait that sets or moves the burn, over the coast from the observed
state to the burn, and the first transfer decision whose impulse fits in one step, over the
coast from the state after that step to the arrival burn, cut at the end of the episode.
`PilotAgent` adds the `rule` that decided (`read`, `capture`, `line_of_sight`,
`closing`, `opening` or `coast`) and `saturated`; a pulse adds the read coast `coast_s`, the
keep-out margin predicted over it, `predicted_margin`, and its certified lower bound,
`predicted_margin_bound` (m, from `vbar.agents.planning.predicted_margin`). A predicted
violation is logged and the agent flies on; the metrics score the true trajectory.

## Parameters and defaults

| Parameter | Default | Source and reason |
|---|---|---|
| `ElevationTimedAgent.trigger` | required | Elevation in rad; the STORRM terminal-phase burn used 42 deg (Goodman p. 150) |
| `aim` | `"hold"` | The transfer ends at the scenario's hold point |
| `n_tf` | 1.0 rad | As for `TwoImpulseAgent`: far from the singular set, and from rest on the V-bar the first command points at the goal on every axis |
| `slip_s` | 60 s | The project's own value: six steps at the default 10 s step |
| `PilotAgent.tau` | 1000 s | Gate time constant of the Shuttle range-rate tables (Goodman pp. 57, 74, Table 24.3) |
| `band` | 0.25 | The project's own value: a quarter of the gate rate either side |
| `pulse` | 0.03048 m/s | 0.1 ft/s, the pulse size flown on STS-92 (Goodman p. 143) |
| `min_rate` | 0.03048 m/s | 0.1 ft/s, the final closing rate from 10 ft to contact on STS-130 (Goodman Table 25.8) |
| `los_rate_limit` | $10^{-4}$ rad/s | The project's own value: 2 cm/s across the line of sight at 200 m. Goodman names line-of-sight drift as a reason to pulse but gives no threshold; `None` turns the rule off |
| `read_s` | 60 s | The project's own value for the quiet period the crew needed to read the new range rate (Goodman pp. 57, 74) |

## Assumptions and limits

- CW motion: a circular target orbit and separations small against its radius. The aim-point
  rules are exact under CW and approximate on curved or perturbed truth.
- `cw_lambert` is singular in plane at $n t_f = 2k\pi$ and at the other roots of
  $8\cos x + 3x\sin x = 8$, and across the track at $n t_f = k\pi$ (see
  [two-impulse targeting](two-impulse.md)). `ElevationTimedAgent` refuses an `n_tf` within
  $10^{-2}$ of that set at reset, so transfers near a whole revolution, and the 180 deg
  transfer, are refused.
- The crossing is predicted on the unforced CW coast from a single observation, so navigation
  noise moves it; the slip limit bounds how far one re-prediction may move the burn. A crossing
  that enters and leaves the trigger within a quarter step can be missed. The agent waits one
  step at a time and commits no coast before the burn.
- The pilot model has no memory beyond its read period and acts on the observation it is given.
  Without the line-of-sight rule its closing pulses feed the CW coupling between radial and
  along-track motion, which nothing nulls, and on the standard approach from 300 m behind it
  drifts away instead of capturing.
- A pulse larger than one step of thrust is refused at reset; the predicted margin covers the
  whole read coast even where the episode ends inside it.
- `disc27` actions are refused at reset by both agents.
- Both agents refuse at reset goals other than a hold or a berthing, and goals without a hold
  point, with a `ValueError`.

## Validation

Tests: `tests/test_shuttle.py`. Book values are in ft and n.mi., converted as above.

- Goodman chapter 23 (STS-39), Tables 23.4 and 23.5: the NSR transfer time at 140 n.mi. is
  $4/(3n)$ = 19.03 min, matching the tables' 19.0 min; the expression printed on p. 180,
  $(4/3)\,\omega$, is corrected to $4/(3\omega)$
  (`test_nsr_transfer_takes_4_over_3n_which_is_19_03_min_at_140_nmi`). The football transfer
  takes $T/4$ = 22.42 min with $T$ = 89.69 min
  (`test_football_transfer_takes_a_quarter_period_which_is_22_42_min_at_140_nmi`).
- Goodman pp. 178 to 180: from the four starts of Tables 23.4 and 23.5 (10 km behind at
  2000 ft below and 1200 ft above, the near-field point and the Ti point), CW targeting to the
  NSR aim point returns the coelliptic velocity with $\mathbf v_f = \mathbf v_0$ to
  $10^{-12}$ m/s and a propagated miss under $10^{-9}$ m, and the speed is $1.5\,nZ_0$
  (`test_nsr_aim_turns_cw_targeting_into_a_coelliptic_burn`); from 2000 ft below it is
  3.50261 ft/s (`test_nsr_burn_from_2000_ft_below_is_3_50261_ft_per_s`). The football aim gives
  $\dot x_0 = 0$ and $\dot y_0 = -2nx_0$ to $10^{-12}$ m/s and returns to the start after one
  period within $10^{-6}$ m
  (`test_football_aim_gives_a_no_drift_ellipse_that_returns_after_one_period`).
- Goodman pp. 150 and 154 (STORRM): at 190 n.mi., from coelliptic 4000 and 5000 ft below with
  the burn at 42 deg elevation, aiming 330 ft behind and 1000 ft below over 90 and 120 deg, the
  burn is 2.35 to 3.61 ft/s with more than 0.9 of it radial up, as flown on p. 154, (0.3, 0.1,
  -3.4) ft/s in the book's frame; the text on p. 150 calls it mostly posigrade, which the flown
  components contradict
  (`test_terminal_phase_burn_at_42_deg_elevation_is_mostly_radial_up`). `ElevationTimedAgent`
  flown in those cases fires at the predicted crossing step with the burn CW targeting gives
  (`test_elevation_timed_terminal_phase_burn_matches_cw_targeting_at_190_nmi`).
- Timing and slip: the burn fires at the crossing rounded to a step, before the elevation gets
  there; a re-predicted crossing 40 s later moves it, one 200 s later is ignored and logged; a
  trigger never reached fires two-impulse targeting at once and is logged
  (`test_elevation_timed_burn_fires_at_the_predicted_crossing_step`,
  `test_a_re_predicted_crossing_within_the_slip_limit_moves_the_burn`,
  `test_a_re_predicted_crossing_beyond_the_slip_limit_is_ignored_and_logged`,
  `test_a_trigger_never_reached_fires_two_impulse_targeting_at_once`).
- Goodman p. 143 (STS-92): twelve 0.1 ft/s pulses add up to 1.2 ft/s; the pilot model from
  1400 m behind the hold point gives twelve pulses within 10 deg of the V-bar that sum to
  1.2 ft/s to $10^{-12}$
  (`test_twelve_pulses_from_far_behind_add_up_to_1_2_ft_per_s`), and every pulse delivers
  0.1 ft/s in both `impulse_coast` and `accel` (`test_each_pilot_pulse_delivers_0_1_ft_per_s`).
- The gate rule: every decision outside a read period follows the line-of-sight and gate rules
  in order over ten minutes of flight (`test_every_pilot_decision_outside_the_read_time_follows_the_gate_rule`),
  the floor holds at 0.1 ft/s (`test_pilot_gate_never_drops_below_0_1_ft_per_s`), and the read
  period follows every pulse (`test_pilot_waits_the_read_time_after_every_pulse`).
- From rest 300 m behind the target, the pilot captures at the hold point 100 m behind within
  the hour in both `impulse_coast` and `accel`
  (`test_pilot_captures_at_the_hold_point_within_the_hour`); with the line-of-sight rule off it
  ends the hour more than 1 km from the hold point
  (`test_pilot_without_line_of_sight_nulling_drifts_away_instead_of_capturing`).
- Every committed coast of `ElevationTimedAgent` logs its predicted margin and bound
  (`test_the_wait_and_the_transfer_each_log_their_predicted_margin`), and on the random starts
  of seeds 0 to 3 in the `impulse_coast` mode every episode whose bounds are all non-negative
  keeps a non-negative continuous-time keep-out margin
  (`test_a_coast_predicted_clear_of_the_keep_out_zone_stays_clear`).
- Both agents pass the sign check on all six axes
  (`test_elevation_timed_agent_first_command_points_at_the_goal_on_every_axis`,
  `test_pilot_first_pulse_points_at_the_goal_on_every_axis`), keep one log entry per decision
  (`test_agent_keeps_one_log_entry_per_decision`), take the protocol arguments by name
  (`test_agent_takes_the_protocol_arguments_by_name`), and see only the time
  (`test_agent_sees_only_the_time_and_holds_no_simulator_object`).

## Example

The pilot model on the standard approach, and the NSR burn from 2000 ft below at 140 n.mi.:

```python
import numpy as np

from vbar import ApproachConfig, ApproachEnv
from vbar.agents.cw_targeting import cw_lambert
from vbar.agents.shuttle import PilotAgent, nsr_aim
from vbar.dynamics import EARTH, mean_motion
from vbar.metrics import record_episode, success
from vbar.spacecraft import PerfectNav

config = ApproachConfig(
    start_state=(0.0, -300.0, 0.0, 0.0, 0.0, 0.0),
    action_mode="impulse_coast",
    sensor=PerfectNav(),
)
ep = record_episode(ApproachEnv(config), PilotAgent(), seed=1, agent_seed=2)
pulses = sum(1 for entry in ep.agent_log if "coast_s" in entry)
print(f"captured: {success(ep)} at {ep.records[-1].t:.0f} s after {pulses} pulses")
assert success(ep)

n = mean_motion(EARTH.radius + 140.0 * 1852.0)
r0 = np.array([-2000.0, -32808.0, 0.0]) * 0.3048
rf, tf = nsr_aim(r0, n)
v0, _ = cw_lambert(n, r0, rf, tf)
print(f"NSR burn {np.linalg.norm(v0) / 0.3048:.5f} ft/s, transfer {tf / 60.0:.2f} min")
```
