# Fault injection

Code: `vbar.faults`. A fault changes what the spacecraft delivers or what the navigation sensor
reports, for part of an episode. Each fault pairs an effect with an activation that decides when
it acts. Faults are listed in `EnvConfig.faults`, and the base environment applies them in every
scenario built on it ([environment.md](environment.md)).

## When a fault acts

The environment decides each fault's activity at every decision boundary, from the step just
flown:

1. At reset, each fault is reset with its own random stream and updated with the start state.
2. Before a step's command is issued, each active fault's `on_command` edits the command
   fraction as issued, after the scenario's own limits (thrust inhibit windows) and before the
   thruster limits, the minimum impulse bit, the actuation delay and the execution error.
3. After the step is flown, each fault's `update(t, states)` sees the step's end time and its
   sampled true states (20 per step by default, [environment.md](environment.md)) and sets its
   activity until the next boundary.
4. The observation at the end of the step passes the sensor's reading through each active
   fault's `on_observation`, in the order the faults are listed; the observation channels that
   follow it are not touched.

So a fault whose condition is met during a step corrupts the reading made at the end of that step
and acts on the next command. A command fault is decided when the command is issued: with an
actuation delay, the delivered thrust reflects the fault's state at the issue time.

Activations (`vbar.faults`):

| Activation | Active |
|---|---|
| `Always()` | for the whole episode (the default) |
| `Window(start_s, end_s)` | at decision times t with start_s <= t < end_s, an edge within rounding of a decision counting as reached there |
| `After(condition)` | from the first step in which `condition` holds, for the rest of the episode |
| `Until(condition)` | until the first step in which `condition` holds |
| `Intermittent(probability, within)` | at each decision with `probability` while `within` is active |
| `AllOf(activations)` | while every member is active; each member is updated at every decision |

Conditions test a step's sampled true states. `RangeBelow(range_m)` holds when a sample is closer
to the target than `range_m`. `InSphere(center_m, radius_m)` holds when the path enters a sphere in
the Hill frame; the path between consecutive samples is taken as the straight chord, so a pass
between two samples that both lie outside the sphere still counts. The arc departs from its chord
by at most $a h^2 / 8$ for an acceleration $a$ over a sample interval $h$: with the Coriolis term
$2 n v$ at 1 m/s, $n$ = 1.1e-3 rad/s and the default interval of 10/19 s, that is 8e-5 m.

`Intermittent` draws once per decision from the fault's own stream, whether or not `within` is
active, so the pattern of later draws does not depend on the window.

## Effects

| Fault | Effect while active |
|---|---|
| `ThrusterOff(axis, direction)` | the thrusters pushing along `direction` (+1 or -1) of Hill `axis` (0 x, 1 y, 2 z) stop firing: that sign of the axis's command fraction is delivered as zero, the other sign and the other axes are untouched, and no propellant is spent on the lost part |
| `SensorBias(position_m, velocity_mps)` | a constant bias is added to the reading: its position part has a magnitude uniform on `position_m` (low, high) in m and its velocity part one uniform on `velocity_mps` in m/s, each along a direction uniform on the sphere, drawn once per episode |
| `RangeLoss(assumed_range_m)` | range and range rate are lost: the reading becomes the line of sight and its rate scaled to `assumed_range_m` |

`RangeLoss` keeps the observation's layout and removes its range information. With the measured
position $\mathbf p$ and velocity $\mathbf v$ and the assumed range $R$, the reading is

$$
\mathbf p' = R \frac{\mathbf p}{|\mathbf p|}, \qquad
\mathbf v' = \frac{R}{|\mathbf p|} \left( \mathbf v - \frac{\mathbf p \cdot \mathbf v}{|\mathbf p|^2} \mathbf p \right)
$$

the unit line of sight and its time derivative, each multiplied by $R$. This is what a camera or a
bearing-only sensor gives. A reading at zero range passes unchanged.

Each fault has a `name`, and `info["faults_active"]` lists the names of the faults active after
each step (and at reset); the entry is present only when faults are configured.

## Random streams and copies

The fault stream is derived from the episode's seed sequence under a key of its own rather than
spawned from it, so the sensor, actuator and target streams are spawned exactly as without faults,
on seeded and unseeded resets alike; it has one child per fault in the order listed. A fault's
draws therefore repeat on the same seed, differ between seeds and between successive unseeded
resets, and do not move when a fault is appended to the list. A configured fault that never acts
leaves the episode identical to one without it. Each environment
deep-copies its configuration's faults, because activations and biases hold episode state.

## What agents see

Agents receive no fault channel: the agent view and `agent_info()` are the same with and without
faults, and a fault shows only through the observations and the spacecraft's response. The
recorded `command` is the agent's command; the actuation shows what was delivered. Activations
read the true state, which never leaves the environment. A scenario's identifier lists faults
added over its spec's own among the changed settings, as it does for every setting that changes
results, so records tell the runs apart; the identifier in the agent's view leaves them out, so an
agent is not told what will fail or when. Faults a catalogue spec declares are public with the
catalogue, as its dispersions are.

## Supported environments

Every scenario built on the base environment takes faults. The vectorized approach environment
refuses them.

## Custom faults

A fault is any object with `name`, `active`, `reset(rng)`, `update(t, states)`,
`on_command(t, command)` and `on_observation(t, obs)`. `on_command` receives the command fraction
(shape (3,)) and must return a new array rather than edit its argument; `on_observation` receives
the six-element navigated state. The configuration refuses an object without these methods.

## Validation

Tests in `tests/test_faults.py` check each effect against a hand-computed answer on a fixed start:

- A failed +y thruster under a +y command leaves the state on the Clohessy-Wiltshire coast and
  spends no delta-v or propellant; a -y command and the other axes are delivered unchanged, against
  the closed-form discrete solution.
- A sensor bias is the same constant at every observation, of the stated position magnitude and a
  velocity magnitude inside its range; it repeats on a seed and differs between seeds.
- `RangeLoss` maps p = (30, 40, 0) m, v = (1, 0, 0) m/s with R = 100 m to (60, 80, 0) m and
  (1.28, -0.96, 0) m/s, and in an episode its reading always has norm R and the true line of
  sight.
- A window from 20 s to 50 s with 10 s steps acts at 20, 30 and 40 s only, and the delta-v spent
  drops to zero exactly on those steps.
- A bias triggered by `RangeBelow(150)` while closing at 2 m/s from 200 m becomes active after the
  step ending at 30 s and stays active; `Until` ends when its condition first holds; a sphere
  pass between two outside samples counts and a pass 0.7 m outside it does not.
- An intermittent fault acts on 30% of 2000 decisions within the binomial bound, repeats on a seed
  and keeps its pattern when another fault is appended.
- `AllOf` of a trigger below 500 m and an `Until` on a sphere 200 m out is active only between
  them, and stays off after the sphere is entered.
- A fault that never acts leaves observations and the final state bit-identical over a seeded and
  an unseeded reset; unseeded resets draw new fault parameters; environments
  built from one configuration do not share a latched trigger; agents see the same view and info
  with and without faults; environments with faults pass Gymnasium's `check_env`; the vector
  environment refuses faults; bad settings are refused.

## Example

A thruster that fails on the +V-bar side from 20 s on, and a velocity bias of 0.4 to 0.8 m/s once
the chaser is inside 300 m:

```python
import numpy as np

from vbar.faults import After, RangeBelow, SensorBias, ThrusterOff, Window
from vbar.scenarios import ApproachConfig, ApproachEnv

faults = (
    ThrusterOff(axis=1, direction=1, when=Window(start_s=20.0)),
    SensorBias(velocity_mps=(0.4, 0.8), when=After(condition=RangeBelow(range_m=300.0))),
)
env = ApproachEnv(ApproachConfig(faults=faults))
obs, info = env.reset(seed=1)
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.5, 0.0]))
print(info["faults_active"])
```
