# Glideslope guidance

Code: `GlideslopeProfile`, `glideslope_profile`, `glideslope_pulses` and `GlideslopeAgent` in
`vbar.agents.glideslope`. Hill frame, metres, seconds.

A glideslope approach flies straight down the line of sight to an aim point, closing fast when
far and slowly near arrival, with a pulse at regular intervals that puts the chaser back on the
line at the range the profile asks for. It is the classical pulsed final-approach law of
Hablani, Tapper and Dana-Bashian, and a reference for agents that approach along a line.

## How it is built

**Profile.** The range to go $\rho = |\mathbf r - \mathbf r_T|$ to the aim point $\mathbf r_T$
follows

$$
\dot\rho = a\rho + b, \qquad a = \frac{\dot\rho_0 - \dot\rho_T}{\rho_0} < 0, \qquad b = \dot\rho_T < 0,
$$

so that

$$
\rho(t) = \left(\rho_0 + \frac{b}{a}\right) e^{a t} - \frac{b}{a}, \qquad
T = \frac{1}{a}\ln\frac{\dot\rho_T}{\dot\rho_0},
$$

with $\rho(T) = 0$ and $\dot\rho(T) = \dot\rho_T$. `glideslope_profile(rho0, rho_dot0, rho_dotT)`
returns $(a, b, T)$. It refuses a range that is not positive, an arrival rate that is not
negative (a zero rate never arrives), and $\dot\rho_0 \ge \dot\rho_T$, which gives $a \ge 0$: the
law needs the closing rate to decay. As $\dot\rho_T \to 0$ the profile becomes exponential
braking, $\dot\rho = -\rho / \lambda$ with $\lambda = -1/a$, the Shuttle's law with
$\lambda = 1000$ s inside 300 m.

**Pulses.** `glideslope_pulses(n, state, hold, profile, pulse_interval)` fixes the line of sight
$\mathbf u = (\mathbf r_0 - \mathbf r_T) / \rho_0$ and places waypoints
$\mathbf r_k = \mathbf r_T + \rho(t_k)\,\mathbf u$ at $t_k = k\,\Delta t$ for
$k < N = \lceil T / \Delta t \rceil$, with $\Delta t$ the pulse interval, and $\mathbf r_N =
\mathbf r_T$ at $t_N = T$. Each segment's departure velocity comes from the position rows of the
CW transition matrix,

$$
\mathbf v_k^+ = \Phi_{rv}^{-1}(t_{k+1} - t_k)\left[\mathbf r_{k+1} - \Phi_{rr}(t_{k+1} - t_k)\,\mathbf r_k\right],
$$

through `vbar.agents.cw_targeting.cw_lambert`, and the pulse is
$\Delta\mathbf v_k = \mathbf v_k^+ - \mathbf v_k^-$, with $\mathbf v_0^-$ the initial velocity and
$\mathbf v_{k+1}^-$ the arrival velocity of segment $k$. The last pulse at $T$ cancels the arrival
velocity, leaving the chaser at rest on the aim point. The result is $N + 1$ `Impulse`s.

**Agent.** `GlideslopeAgent(*, rate=1e-3, arrival_speed=0.02, pulse_interval=60.0,
mode="retarget", name="glideslope")` fixes the line of sight from the hold point to the chaser at
its first decision and flies the profile with $a = -$`rate`, $b = -$`arrival_speed` and
$\dot\rho_0 = -(\text{rate}\,\rho_0 + \text{arrival\_speed})$. Pulses fall every
`pulse_interval`, which must be a whole number of steps, and at $T$ rounded to a whole step, so
every pulse lands on a decision.

An environment burn is a constant acceleration held over whole steps, not an impulse. Each pulse
is therefore solved for the burn as flown: the acceleration $\mathbf a$ held for the fewest $m$
steps that the thrusters deliver (per axis for a box layout, in 2-norm for a ball) solves

$$
\Phi_{r\cdot}(\tau - m\,dt)\left[\Phi(m\,dt)\,\mathbf x_k + \Gamma(m\,dt)\,\mathbf a\right] = \mathbf r_{k+1},
$$

with $\tau$ the segment length and $\Phi$, $\Gamma$ the exact zero-order-hold CW matrices
(`vbar.dynamics.cw_discrete`); the last pulse solves the velocity rows of
$\Phi(m\,dt)\,\mathbf x + \Gamma(m\,dt)\,\mathbf a = 0$ instead. Each burn step goes out through
`vbar.agents.planning.impulse_action` as a one-step impulse $\mathbf a\,dt$. A burn that no number
of steps within its segment can deliver is flown at the thruster limit over the whole segment and
logged as saturated.

- `mode="open_loop"` solves every burn from the state the plan predicts, starting from the first
  observation.
- `mode="retarget"` solves each pulse from the observed state, to the same waypoints, so
  navigation and model errors are corrected at every pulse.

In `"impulse_coast"` the last step of each burn commits the coast to the next pulse; in the
continuous modes every step is a decision and the agent commands zero between burns. After the
last pulse it commands no thrust. `"disc27"` is refused at reset.

`log` holds one entry per decision: `time_s`, `pulse` (the index of the pulse a burn starts, else
None), `dv` (the step's velocity change, m/s), `saturated`, `coast_s`, and, over a committed
coast, `predicted_margin` and `predicted_margin_bound`, the predicted minimum keep-out margin
and its certified lower bound from `vbar.agents.planning.predicted_margin`, computed from the
observed state carried through the burn step (None without a coast). A coast predicted to enter
the keep-out zone is logged and flown; the metrics score what happened.

**Parameters and defaults.**

| Parameter | Default | Reason |
|---|---|---|
| `rate` | 1e-3 1/s | The Shuttle's 1000 s time constant for the final approach inside 300 m |
| `arrival_speed` | 0.02 m/s | Below the approach scenario's 0.05 m/s capture speed |
| `pulse_interval` | 60 s | Six 10 s steps; $n\,\Delta t = 0.066$ rad at 500 km, far from the singular transfer times |
| `mode` | `"retarget"` | Corrects navigation and model errors at every pulse |

## Assumptions and limits

- The paper of Hablani, Tapper and Dana-Bashian (*Journal of Guidance, Control, and Dynamics*
  25(3), 2002) could not be consulted directly; the law and the pulse equations are taken as
  reproduced in Yuan, Li and Wang, *Aerospace* 9, 242 (2022), eqs 14 to 25, and in Wang, Li, Lin
  and Huo, *Aerospace* 8, 137 (2021). Yuan places pulses at arbitrary times and Wang at equal intervals
  $T/N$; the pulses here fall every `pulse_interval` with a shorter last segment, so that they
  land on decision steps.
- The plan assumes Clohessy-Wiltshire motion about a circular target orbit. Under other dynamics
  the open-loop plan misses its waypoints by the model error, which retargeting corrects at the
  next pulse.
- The line of sight is fixed at the first decision. The chaser is on it only at the pulses; between
  them it follows CW arcs, which bow off the line by an amount that grows with the pulse interval.
- The last pulse stops the chaser near the hold point. A hold point off the V-bar is not an
  equilibrium, and the chaser drifts from it after arrival.
- `pulse_interval` must be shorter than half an orbit, with the CW targeting measures of
  `vbar.agents.cw_targeting.conditioning` at least 1e-2 there; reset refuses it otherwise. Every
  segment is then shorter than the first singular transfer time.
- The agent does not avoid the keep-out zone; an approach line through the target goes through it.
- The agent refuses at reset goals other than a hold or a berthing, and goals without a hold
  point, with a `ValueError`.
- A chaser that starts within the capture radius of the hold point, or so close that the arrival
  rounds to no whole step, is treated as arrived: the agent commands no thrust and logs each
  decision.
- An arrival later than the episode is cut by the time limit.

## Validation

Tests: `tests/test_glideslope.py`.

- The profile from $\rho_0$ = 200 m at $-0.22$ m/s to $-0.02$ m/s has $a = -10^{-3}$ 1/s,
  $b = -0.02$ m/s and $T = 1000 \ln 11$ s to $10^{-12}$; from 420 m at $-0.35$ m/s to $-0.015$ m/s
  it starts at $\rho_0$ and the initial rate and ends at zero range and the arrival rate.
- Fehse, *Automated Rendezvous and Docking of Spacecraft*, section 6.2.3, p. 183, and appendix B,
  p. 443: exponential braking with $\lambda = 1000$ s from 300 m gives 0.3, 0.1 and 0.01 m/s at
  300, 100 and 10 m and $\rho(t) = 300\,e^{-t/\lambda}$. The profile with an arrival rate of
  $10^{-9}$ m/s reproduces the speeds to $10^{-6}$ m/s and the ranges to $10^{-6}$ relative.
- A zero, opening or slower initial rate, an opening arrival rate and a range that is not positive
  are refused.
- Pulse plans from rest on the V-bar and from an off-axis drifting start, and a property test over
  starts, rates and arrival speeds: $\lceil T / \Delta t \rceil + 1$ pulses from 0 to $T$, at most
  $\Delta t$ apart; flown impulsively on the exact CW matrices, the chaser is on the line of sight
  at the profile range before every pulse to $10^{-9}$ of $\rho_0$, the range falls at every
  pulse, and the last pulse leaves it at rest on the aim point.
- The total pulse cost from 300 m to the hold point 100 m behind the target rises as the pulse
  interval shrinks and converges at first order: extrapolated to zero interval from 60, 30 and
  10 s it is 0.72579 m/s to $10^{-3}$, the value an independent computation at 5 and 1 s gives.
- Flown open loop in `"impulse_coast"` from rest at $(0, -300, 0)$ m to the hold point
  $(0, -100, 0)$ m: the run captures without entering the keep-out zone, every burn starts within
  $10^{-6}$ m of the V-bar, the last burn starts at $T$ rounded to a step, and every committed
  coast has a positive predicted margin bound below the predicted margin.
- Started on the hold point, the agent commands no thrust and the run captures
  (`tests/test_glideslope.py`).
- The first command points toward the goal on all six axes of the sign check; a pulse interval off
  the step grid and `"disc27"` actions are refused at reset.

## Example

The profile and pulse cost of the default approach from 300 m behind the target, then the agent
flying it:

```python
import numpy as np

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.glideslope import GlideslopeAgent, glideslope_profile, glideslope_pulses
from vbar.dynamics import EARTH, mean_motion

n = mean_motion(EARTH.radius + 500e3)
start = np.array([0.0, -300.0, 0.0, 0.0, 0.0, 0.0])
hold = np.array([0.0, -100.0, 0.0])
profile = glideslope_profile(200.0, -(1e-3 * 200.0 + 0.02), -0.02)
print(f"arrival after {profile.duration:.1f} s")
for interval in (60.0, 30.0, 10.0):
    pulses = glideslope_pulses(n, start, hold, profile, interval)
    cost = sum(np.linalg.norm(burn.dv) for burn in pulses)
    print(f"{interval:4.0f} s: {len(pulses)} pulses, {cost:.4f} m/s")

config = ApproachConfig(action_mode="impulse_coast", start_state=tuple(start))
result = run_episode(ApproachEnv(config), GlideslopeAgent(), seed=1, agent_seed=2)
print(result["captured"], result["keep_out_violated"], result["time_s"])
assert result["captured"]
```
