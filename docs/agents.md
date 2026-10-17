# Agents

Code: `vbar.agents` (`Agent`, `AgentView`, `PDAgent`, `DriftAgent`, `LLMAgent`),
`vbar.agents.sign_check`, `vbar.runner` (`RealTime`, `Episode`). The agent interface and the
built-in agents are described with the environment ([environment.md](environment.md)); this page
covers the tools around them.

## Sign-convention check

`sign_check(agent, config)` runs VALIDATION V12 before an agent is scored. Six times, the agent
is reset with the approach scenario's view and asked for its first command from rest at
$(0, -300, 0)$ m, on the V-bar and so at rest under Clohessy-Wiltshire motion, with the goal
50 m away along $+x$, $-x$, $+y$, $-y$, $+z$ and $-z$. The command, read in fractions of a full
command whatever the action mode (`command_fraction`), passes when its component on that axis
has the goal's sign; zero fails. The scenario keeps `config`'s action mode, thrusters, delay and
observation channels, so the agent sees its normal observation layout, but navigation is exact,
so only the agent's conventions decide the result.

`SignCheck.passed` is metric A6; `failures` lists the directions that failed and `commands` the
first command of each. The check catches an agent that reads the Hill frame as ISS-style LVLH
($x_{\text{ISS}} = y$, $y_{\text{ISS}} = -z$, $z_{\text{ISS}} = -x$, [frames.md](frames.md)) or
flips the radial axis. A drift agent fails every direction by construction; it is a floor
baseline, scored with the `sign_error` label rather than refused. The harness records the result
with each entry.

## Real-time runs

By default the simulation waits for the agent: its command takes effect at the observation's
time, however long it took to decide. `run_episode(env, agent, seed, real_time=RealTime(...))`
lets the clock run instead. The runner measures the agent's wall-clock time $w_k$ for decision
$k$, and the command takes effect $s\,w_k$ simulated seconds after the observation, plus the
configured actuation delay, with $s$ = `time_scale` (1 by default; above 1 makes a slow agent
proportionally slower against the spacecraft). Until then the previous command stays on, so the
step splits into segments at the arrival time, exactly as with an actuation delay
([spacecraft.md](spacecraft.md)), and the keep-out check covers both. A command never takes
effect before one issued earlier: if a slow decision is overtaken, the newer command acts from
the older one's arrival time and the older never acts. Decisions still come every `dt`; a
latency longer than `dt` carries into the next step.

Every result carries `latencies_s`, the wall time of each decision (metric A1), real time or
not. `RealTime(latencies=result["latencies_s"], time_scale=s)` repeats a recorded run exactly,
as transcript replay of an LLM agent needs. `env.record.agent_latency_s` holds the simulated
latency of the last step.

`Episode(env, seed, agent_seed)` is the loop `run_episode` steps, for callers that drive an
episode themselves (the agent servers): it refuses training environments, seeds the agent's
stream, never returns the reward, and builds the same result.

Limits: the pending-command observation channel has $\lceil$delay$/\Delta t\rceil$ slots, sized
for the configured delay, so a latency longer than that can push commands out of it; latency
measured on a shared machine includes everything else the machine was doing.

## How it is tested

`tests/test_sign_check.py`:

- The PD agent passes in normalized, acceleration and thrust modes, with ball thrusters, and with
  a 3 s delay and the time and pending-command channels.
- The drift agent fails all six directions, in the continuous and the discrete mode.
- An agent that treats Hill commands as ISS-style axes fails all six; one with the radial axis
  flipped fails exactly $\pm x$.
- A discrete agent that picks the bang-off-bang command toward the goal passes.
- An LLM agent passes when its model reads the position and goal from the prompt and points at
  the goal, and fails five directions when the model always answers $+x$.

`tests/test_real_time.py`:

- A run with recorded latencies of 3.7 s flies exactly the trajectory of a 3.7 s actuation delay.
- A PD agent that sleeps 10 ms per decision at `time_scale` 200 gets a latency of at least 2 s,
  recorded in the step record, and the previous command acts for exactly that long in the step.
- A command overtaken by a newer one never acts; negative latencies, a negative time scale and a
  recording shorter than the run are refused; every run records its decision times.

## Example

```python
from vbar import ApproachConfig
from vbar.agents import PDAgent
from vbar.agents.sign_check import sign_check

check = sign_check(PDAgent(), ApproachConfig(action_mode="accel"))
print(check.passed, check.commands["-y"])  # True (0.0, -0.25, 0.0)
```
