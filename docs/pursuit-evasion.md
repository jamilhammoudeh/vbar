# Pursuit-evasion in KSPDG's format

Code: `vbar.games` (`PursuitEvasionEnv`, `PE1_E1_I3_Env`, `PE1_E3_I3_Env`, `PursuitEvasionGymEnv`,
`AgentEnvRunner`, `BaseAgent`, `fly`, `PursuitTask`, the bots in `vbar.games.bots` and the
reference pursuers in `vbar.games.pursuers`).

KSPDG (github.com/mit-ll/spacegym-kspdg) poses spacecraft differential games inside Kerbal Space
Program and flies agents through kRPC in real time. `vbar.games` rebuilds its pursuit-evasion
problem pe1 from KSPDG's public definition on two-body dynamics, with the same observation list,
action dictionary, step return and agent interface. An agent written for KSPDG runs unchanged,
headless, faster than real time and in parallel. The evader bots are V-BAR's own, written from
KSPDG's one-line descriptions of its evaders; KSPDG's bots are not used or reproduced, so scores
against them are not comparable with scores in KSPDG.

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

Two scenarios start from KSPDG's I3 conditions, read from its `pe1_i3_init.sfs` mission save:
both vessels on a 750 km circular equatorial orbit, the pursuer 0.2 deg (2618 m) behind, the
evader's orbit tilted by 1e-4 deg.

| Class | Gymnasium ID | Evader |
|---|---|---|
| `PE1_E1_I3_Env` | `vbar/PursuitEvasionE1I3-v1` | `PassiveEvader`: never thrusts (KSPDG's e1 is "passive") |
| `PE1_E3_I3_Env` | `vbar/PursuitEvasionE3I3-v1` | `NormalEscapeEvader`: full thrust along its orbit normal while the pursuer is within 200 m |

`NormalEscapeEvader` is written for V-BAR from KSPDG's description of e3 ("heuristic evasion
maneuvers") and the evasion distance and throttle in its public code (200 m, full throttle); its
thrust is 8000 N along the angular momentum of its own orbit, it stops when the range rises back
through 200 m, and its propellant runs out like the pursuer's. The class names are KSPDG's, so a
KSPDG script needs only its imports changed; the LLM agents of Carrasco, Rodriguez-Fernandez and
Linares (arXiv 2408.08676) were evaluated on PE1_I3_E3.

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
direction to Kerbin's centre across the line of sight. The evader turns its forward axis to its
thrust, so its limit is 8000 N.

A malformed action (wrong shape, a number that is not finite, a negative duration, an unknown
frame or vector type) raises `ValueError` and leaves the episode as it was.

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
or mark a segment: capture (range falls through the capture distance), each local minimum of
range (range rate rising through zero), a tank running dry, and the evader bot switching modes.
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

The game is deterministic: the same agent gives the same episode on every run, whatever the
number of workers, and the seed changes nothing.

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
pursuer ends early. On a GitHub-hosted Ubuntu runner with two cores and Python 3.15, one episode
took 0.075 s with `LineOfSightPursuer` (240 steps of 1 s), about 3200 times real time, and 0.22 s
with `VelocityMatchingPursuer` (480 steps of 0.5 s), about 1100 times. Four episodes on two worker
processes through `run_jobs` took 2.3 s, about 400 times real time in total: the episodes need
about 0.45 s per worker, and the rest is the harness starting its workers and recording each run.
KSPDG runs one second of game time per second of wall time. The figures depend on the machine;
rerun the script to measure yours.

## Limits

- Two-body point masses: no KSP physics engine, part physics, reaction-wheel slew or roll
  dynamics; the pursuer points at the evader exactly.
- Lockstep time: a slow agent loses no simulated time, unlike in KSPDG, where game time runs on
  while it decides.
- The episode starts at mission time 0 at the I3 orbits; KSPDG loads the save and pauses about 3 s
  of game time before the first observation.
- `vec_type` takes 0 and 1; KSPDG declares a third value it does not define. Burn durations above
  10 s are accepted, as KSPDG does not enforce its declared bound.
- Only the I3 start and the e1 and e3 evaders exist so far; scores against V-BAR's evaders are not
  comparable with KSPDG's.

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
- Capture ends the step at the capture distance to 1e-6 m; the closest approach of a pass inside a
  10 s step matches a run in 0.25 s steps to 1e-6 m and is 20 m below every 10 s step end.
- `None` repeats the last action and coasts before the first; a KSPDG-shaped agent class runs
  through `AgentEnvRunner` with KSPDG's arguments; the agent's list is a copy.
- Both Gymnasium IDs pass `check_env`; parallel and serial runs give identical metrics.

Sources: KSPDG v0.15.0 (commit 6251ca3): `src/kspdg/pe1/pe1_base.py` (observation, score, timeout,
capture distance, thrusters, target-hold), `base_envs.py` (actions, frames), `agent_api/` (agent
and runner signatures), `pe1/README.md` (evader descriptions), `e3_envs.py` (evasion distance and
throttle) and `ksp_files/saves/missions/pe1_i3/pe1_i3_init.sfs` (orbits, parts, propellant).
