# Pursuit-evasion in KSPDG's format

Code: `vbar.games` (`PursuitEvasionEnv`, `PE1_E1_I3_Env`, `PE1_E3_I3_Env`, `PursuitEvasionGymEnv`,
`AgentEnvRunner`, `BaseAgent`, `fly`, `PursuitTask`, the bots in `vbar.games.bots` and the
reference pursuers in `vbar.games.pursuers`).

KSPDG (github.com/mit-ll/spacegym-kspdg) poses spacecraft differential games inside Kerbal Space
Program and flies agents through kRPC in real time. `vbar.games` rebuilds its pursuit-evasion
problem pe1 from KSPDG's public definition on two-body dynamics, with the same observation list,
action dictionary, step return and agent interface. An agent written for KSPDG runs unchanged,
headless, faster than real time and in parallel. The evader bots are V-BAR's versions of KSPDG's
public evaders, written from their source (`pe1/e1_envs.py` and `e3_envs.py`, MIT licence) without
copying it; they fly the same rules on V-BAR's model, and the differences that remain are listed
under [Limits](#limits). No V-BAR episode has yet been compared with one flown in the game.

## The problem

A pursuer and an evader orbit Kerbin. The agent flies the pursuer and tries to come close to the
evader soon, slowly and with little propellant. The episode ends when the range falls below the
capture distance (5 m) or the mission elapsed time reaches the timeout (240 s, the value in
KSPDG's code; its README says 10 minutes). The score, lower is better, is KSPDG's

$$
S = (0.1\,d)^{2} + (0.5\,v)^{1.5} + (0.1\,f)^{1.25} + 0.01\,t
$$

at the closest approach: $d$ the range (m), $v$ the relative speed (m/s), $f$ the pursuer's
propellant used so far (kg) and $t$ the mission elapsed time (s). The reward is $-S$ on the step
that ends the episode and 0 before it.

The four starts are KSPDG's, read from the ORBIT blocks of its mission saves
(`pe1_i1_init.sfs` to `pe1_i4_init.sfs`: semi-major axis, eccentricity, inclination, argument of
periapsis, longitude of the ascending node, mean anomaly at epoch 0), with the saved digits:

| Start | Pursuer | Evader | Range at the start |
|---|---|---|---|
| I1 | ellipse, a = 724999.9 m, e = 0.0345, apoapsis on the evader's circle | 750 km circle | 4510 m; unpowered, they meet within 0.2 m at 121.7 s where the ellipse touches the circle |
| I2 | 750 km circle inclined 1 deg | 750 km circle | 456.8 m, both 2 deg past the node they share; unpowered, they meet again at the far node after 1073.75 s |
| I3 | 750 km circle, 0.2 deg behind | 750 km circle tilted 1e-4 deg | 2618 m |
| I4 | ellipse, a = 754500.2 m, e = 0.0073, inclined 1 deg, another period | the I3 evader | 2660 m |

With the 5 m capture distance, a pursuer that only coasts captures the passive evader from I1 at
121.6 s. KSPDG describes the starts as "elliptical and circular orbits with a tangential
conjunction", "inclined orbits with an unactuated conjunction", "about 3 km apart in the same
circular orbit" and "inclined, out of phase, unsynchronized periods". Each start is flown against
each evader; a class is named for both, as in KSPDG (`PE1_E3_I2_Env` is evader e3 from start I2),
and the classes of one evader share a parent (`PE1_E1_ParentEnv`, `PE1_E2_ParentEnv`,
`PE1_E3_ParentEnv`, `PE1_E4_ParentEnv`). The e3
classes take KSPDG's `evade_dist` and `evade_throttle` arguments.

| Classes | Gymnasium IDs | Evader |
|---|---|---|
| `PE1_E1_I1_Env` to `PE1_E1_I4_Env` | `vbar/PursuitEvasionE1I1-v1` to `vbar/PursuitEvasionE1I4-v1` | `PassiveEvader`: never thrusts (KSPDG's e1 is "passive") |
| `PE1_E2_I1_Env` to `PE1_E2_I4_Env` | `vbar/PursuitEvasionE2I1-v1` to `vbar/PursuitEvasionE2I4-v1` | `RandomEvader`: random throttles, held for random durations, while the pursuer is within 2200 m |
| `PE1_E3_I1_Env` to `PE1_E3_I4_Env` | `vbar/PursuitEvasionE3I1-v1` to `vbar/PursuitEvasionE3I4-v1` | `NormalEscapeEvader`: full thrust along its orbit normal while the pursuer is within 200 m, acting only inside 2200 m |
| `PE1_E4_I1_Env` to `PE1_E4_I4_Env` | `vbar/PursuitEvasionE4I1-v1` to `vbar/PursuitEvasionE4I4-v1` | `ProgradeEvader`: a fifth of its forward thrust along its velocity from 0.5 s on |

`NormalEscapeEvader` is written for V-BAR from KSPDG's description of e3 ("heuristic evasion
maneuvers") and its public code: it acts only while the pursuer is inside the 2200 m control range
of KSPDG's pe1 (`CONTROL_RANGE`), where it points along the angular momentum of its own orbit and
fires its eight forward nozzles (8000 N) while the range is below the evasion distance (200 m, full
throttle), stopping when the range rises back through it. Outside the control range it changes
nothing and keeps its last command, as KSPDG's does; with the default distances that is always
nothing, but with an evasion distance wider than the control range a burn begun inside continues
after the pursuer leaves. Its propellant runs out like the pursuer's. The class names are KSPDG's,
so a KSPDG script needs only its imports changed; the LLM agents of Carrasco, Rodriguez-Fernandez
and Linares (arXiv 2408.08676) were evaluated on PE1_I3_E3.

`RandomEvader` is V-BAR's version of KSPDG's e2 (`e2_envs.py`, "randomized evasive maneuvers").
While the pursuer is inside the control range it points prograde and repeats: draw a hold time
uniform on (0, 2) s and, with probability one half, a throttle on each axis uniform on [-1, 1]
(otherwise none), and hold them for that time. It looks at the range only when a hold ends; outside
the control range it keeps its last throttles until the pursuer is back inside, as KSPDG's does.
KSPDG's bot draws from numpy's unseeded global generator and times its holds with wall-clock
sleeps, so no two of its runs are alike; V-BAR's draws from the episode seed on simulated time, so
a seed replays an episode exactly, and scores against it are compared over many seeds. KSPDG's e2
also sets random pitch, yaw and roll inputs with its throttles; V-BAR's keeps its forward axis
prograde and draws none.

`ProgradeEvader` is V-BAR's version of KSPDG's e4 (`e4_envs.py`, "constant low-thrust burn in
prograde direction, regardless of pursuer"). KSPDG's bot engages its autopilot toward its orbital
prograde, waits 0.5 s for it to turn, then sets forward throttle 0.2 for the rest of the episode;
V-BAR's coasts until mission time 0.5 s and then fires 0.2 forward (1600 N) along its inertial
velocity, pointed exactly, whatever the pursuer does.

## Observation

A list of 15 floats, as in KSPDG:

| Index | Entry |
|---|---|
| 0 | mission elapsed time (s) |
| 1 | pursuer mass (kg) |
| 2 | pursuer monopropellant (units; one unit is 4 kg) |
| 3 to 5 | pursuer position (m) |
| 6 to 8 | pursuer velocity (m/s) |
| 9 to 11 | evader position (m) |
| 12 to 14 | evader velocity (m/s) |

Positions and velocities are relative to Kerbin's centre in a right-handed inertial frame with z
along the north pole (KSPDG's `rhcbci`). Entry 2 is in units because KSPDG reads it from kRPC's
resource amount, although its docstring says kg; KSP's monopropellant density is 0.004 t per unit
(the game's `ResourcesGeneric.cfg`). The x axis lies in the equator toward the
direction the mission save measures its node longitudes from; kRPC documents KSPDG's x axis only as
"an arbitrary direction through the equator", and nothing in pe1 depends on it.

pe1 is a game of full information: the observation is the exact state of both vessels, as in
KSPDG. It is a new list on every call, so an agent that changes it changes nothing in the
environment, and the runner passes the agent nothing else.

## Actions

KSPDG's dictionary:

- `burn_vec`: three components and a burn duration (s).
- `ref_frame`: 0 the pursuer's body frame (forward, right, down), 1 the inertial frame above, 2
  the pursuer's NTW frame (N in the orbit plane and outward on a circular orbit, T along the
  velocity, W along the angular momentum).
- `vec_type` (optional, 0 when absent): 0 throttle, 1 thrust in newtons.

Any other sequence of four numbers is body-frame throttle and duration, as in KSPDG. An inertial
or NTW vector is turned into body components at the start of the step, a thrust is divided by the
axis's limit, and each body throttle is clipped to [-1, 1] and held for the burn. The RCS model is
KSPDG's declared one (`pe1_base.py`): 1000 N per nozzle at 240 s specific impulse, eight nozzles
along the forward axis each way and four along each lateral axis each way, so the limits are
8000 N forward and back and 4000 N on each side. Each axis has its own nozzles, so the propellant
flow is

$$
\dot m = \frac{|F_x| + |F_y| + |F_z|}{I_{sp}\, g_0}
$$

with $g_0$ = 9.80665 m/s^2. A full forward burn uses 3.399 kg/s. Each vessel has 1885 kg dry mass
and 1200 units (4800 kg) of monopropellant, as in the mission save; even all six axes at full
throttle (6.80 kg/s) take 706 s to empty it, longer than the 240 s timeout. A tank that does run
dry, on a smaller `Vessel`, stops the thrust at that instant.

The pursuer holds its forward axis on the evader, as KSPDG's pursuer does with SAS in target
mode. KSP leaves the roll about that axis to SAS; V-BAR fixes it so that down is the part of the
direction to Kerbin's centre across the line of sight.

A malformed action (wrong shape, a number that is not finite, a negative duration, an unknown
frame or vector type) raises `ValueError` and leaves the episode as it was.

## Evader bots

An evader bot flies a sequence of segments (`vbar.games.Segment`). In a segment its throttles
along its forward, right and down axes are constant, and its forward axis points prograde (along
its inertial velocity) or along its orbit normal (its angular momentum), with down toward Kerbin
as for the pursuer; the evader is the same vessel as the pursuer, with the same thrust limits and
per-axis propellant flow. A segment ends at a mission time (`until`) or where a function of the
two vessels' states crosses zero in a given direction (`triggers`), and the game then asks the bot
for the next one with `segment(t, r_p, v_p, r_e, v_e, previous, cause, rng)`, where `cause` is the
index of the trigger that fired (None at the start and at `until`) and `rng` is the episode's
evader stream. A bot decides a switch from `cause` rather than by testing its triggers again,
because at the located crossing they are zero to rounding. A bot that keeps changing segment
without letting time pass (1000 segments in a row each ending within a microsecond of its start)
raises `RuntimeError` instead of hanging the episode, and one that returns anything but a
`Segment` raises `TypeError`.

```python
from dataclasses import dataclass

from vbar.games import PursuitEvasionEnv, Segment


@dataclass(frozen=True)
class ShortBurn:
    """Full forward thrust prograde for the first 2.5 s, then nothing."""

    name: str = "short-burn"

    def segment(self, t, r_p, v_p, r_e, v_e, previous, cause, rng):
        if t < 2.5:
            return Segment(throttle=(1.0, 0.0, 0.0), pointing="prograde", until=2.5)
        return Segment()


env = PursuitEvasionEnv(evader=ShortBurn(), seed=1)
env.reset()
for _ in range(3):
    observation, reward, done, info = env.step([0.0, 0.0, 0.0, 1.0])
print(info["evader_fuel_usage"])  # 2.5 s of 8000 N at 240 s: about 8.50 kg
```

## Seeds

The evader's random stream is drawn from the episode seed: `reset(seed=...)`, else the
`seed` given to the environment's constructor (then every reset replays the same episode), else
a fresh 128-bit seed from the operating system. The seed used is kept in `env.seed`. The
Gymnasium environment passes its `reset(seed=...)` on, and without one draws the stream's seed
from its `np_random`; `fly(env, agent, seed=...)` and `PursuitTask` pass theirs on too. Nothing the
agent receives is derived from the seed other than the observation itself. The passive and
normal-escape bots draw nothing, so their episodes do not depend on it; the random evader draws
everything it does from it. An agent that tries seeds by flying twin episodes and matching the
evader's first seconds finds a small seed among 16 candidates and no 128-bit drawn one.

## Time and integration

Simulated time advances only inside `step`. A step lasts the burn duration or `min_step` (0.1 s by
default), whichever is longer, coasting after a shorter burn, and stops early at capture or the
timeout. `min_step` must be at least 1e-6 s and long enough to move the clock at the timeout, so
every step advances time; settings that are not positive finite numbers, and an evader, vessel or
body of the wrong kind, are refused when the environment is built. Both vessels are point masses
under two-body gravity with Kerbin's $\mu$ = 3.5316e12 m^3/s^2 (no oblateness, as in KSP) plus
thrust; their 14 states (positions, velocities, masses) are
integrated together with DOP853 at relative tolerance 1e-12, which keeps positions near 7.5e5 m
accurate to about a micrometre. Events found by root finding on the integrator's dense output end
or mark a segment: capture (range falls through the capture distance; a fast pass that dips
under it and out again within one integrator step shows as a range minimum below it, and the
crossing before that minimum is then found by Brent's method), each local minimum of
range (range rate rising through zero), a tank running dry, and the end of an evader segment.
The closest approach is therefore found between steps, not only at step ends; KSPDG samples it
when each step ends, so for the same trajectory V-BAR's $d$ is never larger.

## Agents

`get_action(observation)` returns an action, or None to keep the last one, as in KSPDG. KSPDG's
runner re-executes the last action until a new one arrives, while game time keeps running; here
the runner calls the agent once per step, flies the returned (or last) action, and before the
first action the pursuer coasts one shortest step. The agent's thinking time does not cost
simulated time.

```python
from vbar.games import PE1_E3_I3_Env, AgentEnvRunner, BaseAgent


class NaivePursuitAgent(BaseAgent):
    def get_action(self, observation):
        return {"burn_vec": [1.0, 0, 0, 1.0], "vec_type": 0, "ref_frame": 0}


runner = AgentEnvRunner(
    agent=NaivePursuitAgent(),
    env_cls=PE1_E3_I3_Env,
    env_kwargs={"episode_timeout": 30.0},
    runner_timeout=100,
    debug=False,
)
results = runner.run()["agent_env_results"]
print(results["closest_approach"], results["weighted_score"])
```

This is KSPDG's example agent; with KSPDG, `BaseAgent`, `AgentEnvRunner` and the environment class
come from `kspdg.agent_api.base_agent`, `kspdg.agent_api.runner` and `kspdg`. The runner takes any
object with `get_action`, so a subclass of KSPDG's own `KSPDGBaseAgent` runs as it is. `run()`
returns `{"agent_env_results": info}`, KSPDG's form, with `info` holding `is_episode_done`,
`closest_approach`, `closest_approach_time`, `closest_approach_speed`,
`closest_approach_pursuer_fuel_usage`, `pursuer_fuel_usage`, `evader_fuel_usage` and
`weighted_score`, plus `steps` and `wall_s`. `enable_telemetry=True` only warns, as there is no
telemetry stream.

For reinforcement learning, the Gymnasium environments take the same dictionary, return the
observation as a float64 array, terminate at capture and truncate at the timeout. The action
space has KSPDG's declared bounds (components in [-1, 1], durations up to 10 s); `step` also
accepts thrusts in newtons and longer burns, which lie outside it:

```python
import gymnasium as gym
import numpy as np

import vbar  # noqa: F401  (registers the environments)

env = gym.make("vbar/PursuitEvasionE3I3-v1", episode_timeout=20.0)
observation, info = env.reset(seed=0)
action = {"burn_vec": np.array([1.0, 0.0, 0.0, 1.0]), "ref_frame": 0, "vec_type": 0}
terminated = truncated = False
while not (terminated or truncated):
    observation, reward, terminated, truncated, info = env.step(action)
print(reward, info["closest_approach"])
```

The same agent and seed give the same episode on every run, whatever the number of workers; with
the passive and normal-escape evaders the seed changes nothing.

## Published agents

`ClosestApproachAgent` is V-BAR's reimplementation of Jason Cheng's deterministic agent in
github.com/ARCLab-MIT/kspdg (commit a45e5c6,
`arclab_mit/agents/extended_obs_agent/deterministic_agent.py` and `simulate.py`, MIT licence); Cheng
is an author of "Language Models are Spacecraft Operators" (Rodriguez-Fernandez, Carrasco, Cheng,
Scharf, Siew and Linares, arXiv 2404.00413). Each call predicts the closest approach of the two
vessels if both coast, sampling their two-body conics every second for 300 s (`closest_approach`,
V-BAR's Kepler propagation in place of poliastro), and fires a 5 s inertial-frame throttle burn:

| Predicted approach | Throttle vector |
|---|---|
| now, or 80 s or more ahead | 0.7 toward the evader |
| 25 s to 80 s ahead | 0.7 along the predicted miss vector (pursuer to evader at the approach) |
| under 25 s ahead | 0.4 along the miss vector minus 0.8 along the predicted relative velocity |

The constants and the order of the arithmetic are the original's. `closest_approach(state,
max_time)` has the original's signature and returns its first sample of least range, so the true
minimum may lie up to half a second from the sample.

```python
from vbar.games import AgentEnvRunner, PE1_E1_I3_Env
from vbar.games.pursuers import ClosestApproachAgent

info = AgentEnvRunner(ClosestApproachAgent(), PE1_E1_I3_Env, {"episode_timeout": 60.0}).run()
print(info["agent_env_results"]["closest_approach"])
```

## Running many episodes

`PursuitTask` makes the scenarios a task of the evaluation harness ([evaluation.md](evaluation.md)),
so `run_jobs` flies them in parallel processes, appends results to a file and records each run's
provenance. Success is a capture; the metrics are the score terms, the fuel used and the step count.

```python
from vbar.eval import Job, run_jobs
from vbar.games import PursuitTask

jobs = [Job("pe1-e1-i3", "validation", 0, 2**100, 2**101)]
(result,) = run_jobs(jobs, "vbar.games.pursuers:VelocityMatchingPursuer", task=PursuitTask())
print(result["metrics"]["success"], result["metrics"]["weighted_score"])
```

`LineOfSightPursuer` is the strategy of KSPDG's example agent (full forward thrust toward the
evader).
`VelocityMatchingPursuer` closes along the line of sight at 20 m/s, slowing in proportion to range
inside 200 m; it captures the passive evader.

## Speed

`benchmarks/pursuit_evasion.py` times the E3 scenario, whose 240 s episodes neither reference
pursuer ends early. On a hosted two-core Ubuntu CI runner with Python 3.15, one episode
took 0.075 s with `LineOfSightPursuer` (240 steps of 1 s), about 3200 times real time, and 0.22 s
with `VelocityMatchingPursuer` (480 steps of 0.5 s), about 1100 times. Four episodes on two worker
processes through `run_jobs` took 2.3 s, about 400 times real time in total: the episodes need
about 0.45 s per worker, and the rest is the harness starting its workers and recording each run.
KSPDG runs one second of game time per second of wall time. The figures depend on the machine;
rerun the script to measure yours.

## Limits

- Two-body point masses: no KSP physics engine, part physics, reaction-wheel slew or roll
  dynamics; the pursuer points at the evader exactly.
- The evaders point exactly and at once, on simulated time; KSPDG's turn under SAS or the
  autopilot in game time, so they fire before they finish turning.
- The random evader holds its forward axis prograde instead of applying random attitude inputs,
  and draws from a seeded stream on simulated time instead of an unseeded one on wall-clock time.
- Lockstep time: a slow agent loses no simulated time, unlike in KSPDG, where game time runs on
  while it decides.
- The episode starts at mission time 0 at the saved orbits' epoch; the saves were written 0.02 s to
  0.14 s of game time later, and KSPDG loads the save and pauses about 3 s
  of game time before the first observation.
- `vec_type` takes 0 and 1; KSPDG declares a third value it does not define. Burn durations above
  10 s are accepted, as KSPDG does not enforce its declared bound.

## Validation

`tests/test_pursuit_evasion.py`:

- The I3 start from the mission save's elements: both speeds equal $\sqrt{\mu/a}$ and the range
  equals $a\,\Delta M$ = 2617.994 m to 2 mm (the chord is 1.3 mm shorter).
- A coast agrees with universal-variable Kepler propagation to 1e-6 m after 34 s of mixed step
  lengths.
- A one-second full forward burn uses exactly 8000 N / (240 s g0) of propellant, 1/4 of that in
  units, and its velocity change against a coasting twin matches the rocket equation to 1e-5.
- One command given in the body frame, the inertial frame, the NTW frame and as thrust in newtons
  flies the same trajectory; throttle above 1 saturates.
- A tank that runs dry mid-burn stops the thrust there: the velocity change matches the rocket
  equation for exactly the propellant on board.
- At a pass inside a 10 s step, the recorded range, relative speed and propellant used equal a
  twin's stopped exactly at the pass, and the score equals the formula on them.
- Capture ends the step at the capture distance to 1e-6 m; the closest approach of a pass inside a
  10 s step matches a run in 0.25 s steps to 1e-6 m and is 20 m below every 10 s step end.
- `None` repeats the last action and coasts before the first; a KSPDG-shaped agent class runs
  through `AgentEnvRunner` with KSPDG's arguments; the agent's list is a copy.
- Parallel and serial runs give identical metrics (`tests/test_pursuit_scenarios.py`, which also
  runs `check_env` on every registered pe1 Gymnasium ID, enumerated from the registry).

`tests/test_pursuit_starts.py`:

- Each start's two states equal the save's elements mapped through the perifocal frame and three
  rotations, written out in the test, to 1e-6 m and 1e-9 m/s.
- The start ranges are 4510 m (I1), 456.8 m (I2) and 2660 m (I4) to 1 m.
- I1: the pursuer's apoapsis is the evader's radius to 0.1 m, and an unpowered coast passes within
  0.2 m at 121.7 s. I2: the start range equals the chord 2 r sin(u) sin(di / 2) to 1e-12, and an
  unpowered coast meets again within 0.1 m at 178/360 of a period, 1073.75 s. I4: the pursuer's
  semi-major axis, eccentricity and inclination recovered from its state by energy and angular
  momentum match the save.
- Every scenario class flies its start and its evader.
- A fast pass (the I1 coast at 36 m/s) is captured at the capture distance to 1e-6 m at the same
  instant, to 1e-6 s, whether it is flown in 10 s or 0.5 s steps.

`tests/test_pursuit_random_evader.py`:

- 4000 draws inside the control range: hold times uniform on (0, 2) s (mean 1 s to 0.04 s), on
  half the time (to 0.035), throttles uniform on [-1, 1] (mean 0 and mean magnitude 0.5); every
  draw points prograde and ends only at its time.
- Outside the control range the last throttles are kept and the segment ends where the range
  falls through 2200 m, after which the bot draws at once; it never burns on the I3 coast, which
  stays beyond 2618 m.
- The same seed replays an I2 episode exactly and another seed differs; a draw on all three axes
  uses each axis's flow.
- A cheat agent that flies twin episodes for 16 candidate seeds and matches the evader's first
  three seconds finds seed 7 and does not find a drawn 128-bit seed.

`tests/test_pursuit_prograde_evader.py`:

- No propellant before 0.5 s, then exactly a fifth of the forward flow, 239.5 s of it by the end.
- After 10 s of burning the evader's velocity differs from a passive twin's by the rocket-equation
  amount to 1e-3, along its velocity, and not at all with the pursuer burning; over the episode its
  semi-major axis rises by 20 km to 60 km. Every E4 start flies this evader.

`tests/test_pursuit_closest_approach.py`:

- `closest_approach` returns the whole second of least range of an independent DOP853 coast from
  I1 and I4, with that sample's states to 1e-9; a pair already separating gives time 0.
- Each branch fires the original's burn: toward the evader for the I3 start (approach 121 s
  ahead), along the miss vector 40 s ahead, and the braking combination 10 s ahead.
- From I3 the agent captures the passive evader, using all three branches.

`tests/test_pursuit_bots.py`:

- The I3 episode of `VelocityMatchingPursuer` against the normal-escape evader matches the one
  recorded with vbar 0.4.0 (closest approach, both vessels' propellant and the score) to 1e-9.
- A segment ending at 2.5 s inside a 1 s step uses exactly 2.5 s of full forward flow; lateral
  throttles draw each axis's flow (8000 + 4000 + 2000 N at 240 s); a forward burn pointed prograde
  or along the orbit normal changes the evader's velocity along that direction by the
  rocket-equation amount to 1e-5.
- The same seed flies the same episode with a random bot and another seed differs; a constructor
  seed holds for every reset, and without one each reset draws a different seed of 128 bits.
- The normal-escape evader with a 2500 m evasion distance burns nothing until the pursuer is
  inside 2200 m, then burns to the end of the episode, after the pursuer has left; its segment
  choice follows the crossing that fired, not a fresh test of the range.
- Each body-axis throttle (forward, right, down) pushes the evader along that axis with its
  thrust limit.
- The Gymnasium seed and `fly`'s seed reach the evader's stream: equal seeds give equal episodes,
  and unseeded Gymnasium resets follow from the last seeded one.
- Malformed segments and seeds are refused; a bot that never lets time pass, or re-arms the
  trigger that just fired, raises instead of hanging.

Sources: KSPDG v0.15.0 (commit 6251ca3): `src/kspdg/pe1/pe1_base.py` (observation, score, timeout,
capture distance, thrusters, target-hold), `base_envs.py` (actions, frames), `agent_api/` (agent
and runner signatures), `pe1/README.md` (evader descriptions), `e3_envs.py` (evasion distance and
throttle) and the mission saves `ksp_files/saves/missions/pe1_i1/pe1_i1_init.sfs` to
`pe1_i4/pe1_i4_init.sfs` (the orbits of every start; parts and propellant from `pe1_i3_init.sfs`).
