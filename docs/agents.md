# Agents

Code: `vbar.agents` (`Agent`, `AgentView`, `PDAgent`, `DriftAgent`, `LLMAgent`),
`vbar.agents.sign_check`, `vbar.agents.plan` (`PlanAgent`, `PlanVerifier`, `Verdict`,
`LLMPlanner`), `vbar.runner` (`RealTime`, `Episode`). The agent interface and the
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

## Plan submission

Instead of one command per step, a planner can submit a whole plan: rows of fractions of a full
command, one per step. `PlanVerifier(view).verify(obs, plan, steps_left)` is the external check:
it flies the plan from the observed state under the Clohessy-Wiltshire model with the view's mean
motion, thrust layout and limit, checks each public constraint over every step with the
certified in-step minimum ([constraints.md](constraints.md)), and stops at the first step whose
end meets the goal. The `Verdict` holds the predicted lowest margin of each constraint, the
capture step, the final distance and speed, the delta-v (2-norm) and a list of problems: an empty
or non-finite plan, more steps than remain, a violated constraint (with the first step and the
lowest margin), or the goal not reached. `ok` means no problems; `summary()` puts the verdict
in words for a planner to revise from.

The verifier knows what the agent knows and nothing more: it starts from the observation, so a
noisy navigation fix makes its prediction wrong by that error, and it does not model the
minimum impulse bit, delay, propellant limits or the nonlinear dynamics. With exact navigation
and the CW model it predicts the flight exactly.

`PlanAgent(planner, max_revisions=3)` calls `planner(obs, view, verdict, steps_left)` when it has
no plan left, with `verdict` None at first and the last rejection after that, until a plan
passes or `max_revisions` revisions are used. It then flies the accepted plan open loop, one row
per step, converted to the action mode. A plan that never passes is not flown: the spacecraft
coasts and the planner is asked again at the next decision. Each planning point's number of
verifier calls and last verdict go to `log`, the data for plan quality against verifier calls.

`LLMPlanner(complete, price)` is a planner backed by a model with the same `complete(messages,
tools)` function as the LLM agent ([llm-agents.md](llm-agents.md)). Its prompt gives the frame,
the goal, the keep-out radius, the observed state, the step, the thrust limit and the steps left,
asks for a JSON list of segments `{"ax", "ay", "az", "steps"}`, and on a revision includes the
verdict's summary. `parse_plan(text, max_rows=10_000)` expands the segments into rows; numbers
may be quoted. A reply that cannot be read (bad JSON, nesting too deep, a number too large for a
float, steps that are not a whole number of at least zero, more than `max_rows` steps) is an
empty plan, which the verifier rejects. A plan longer than the steps left is parsed, so the
verifier can tell the model it is too long, and the verifier predicts at most the steps left.
Every call's prompt, reply, tokens, cost and latency go to its `log`.

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

`tests/test_plan.py`:

- A PD plan predicted from exact navigation, flown open loop by the environment, captures at the
  predicted step, with the predicted minimum range (1e-12 relative), final distance and delta-v.
- A plan through the keep-out zone is rejected with the step and margin of the violation; the
  same planner routed around the zone passes.
- With 20 m of navigation noise the prediction starts from the observation, not the truth.
- A planner that cuts through the zone first and goes around after the rejection is accepted on
  the second call and the run captures without a violation; a planner whose plans never pass is
  not flown, using every allowed call at each decision.
- The LLM planner's revision prompt carries the verdict and the steps left, and tells a model
  whose 100-step plan exceeds the 30 steps left that it is too long; segments are expanded, with
  quoted numbers; an unreadable reply is an empty plan, as are infinite, negative, fractional or
  400-digit steps, a 400-digit fraction, 3000 levels of nesting, more than `max_rows` steps and
  a negative segment paired with a long one; a 2000-row plan with 5 steps left is rejected after
  predicting exactly 5 steps, and a negative number of steps left predicts none.
- The plan agent rejects the discrete mode, and the verifier unknown constraint types.

`tests/test_real_time.py`:

- A run with recorded latencies of 3.7 s flies exactly the trajectory of a 3.7 s actuation delay.
- A PD agent that sleeps 10 ms per decision at `time_scale` 200 gets a latency of at least 2 s,
  recorded in the step record, and the previous command acts for exactly that long in the step.
- A command overtaken by a newer one never acts; negative latencies, a negative time scale and a
  recording shorter than the run are refused; every run records its decision times.

## Examples

The sign check:

```python
from vbar import ApproachConfig
from vbar.agents import PDAgent
from vbar.agents.sign_check import sign_check

check = sign_check(PDAgent(), ApproachConfig(action_mode="accel"))
print(check.passed, check.commands["-y"])  # True (0.0, -0.25, 0.0)
```

A real-time run in which each decision's wall time counts twenty times over, repeated from its
recorded latencies:

```python
from vbar import ApproachConfig, ApproachEnv, RealTime, run_episode
from vbar.agents import PDAgent

env = ApproachEnv(ApproachConfig(start_state=(-150.0, -400.0, 0.0, 0.0, 0.0, 0.0)))
result = run_episode(env, PDAgent(), seed=0, real_time=RealTime(time_scale=20.0))
repeat = RealTime(time_scale=20.0, latencies=tuple(result["latencies_s"]))
again = run_episode(env, PDAgent(), seed=0, real_time=repeat)
print((again["states"] == result["states"]).all())  # True: recorded latencies repeat the run
```

Plan submission with a planner that flies the PD controller on the CW model:

```python
import numpy as np

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent
from vbar.agents.plan import PlanAgent
from vbar.dynamics import CWDynamics


def pd_plan(obs, view, verdict, steps_left):
    pd = PDAgent()
    pd.reset(view)
    model = CWDynamics(view.n)
    model.reset(obs[:6])
    state, rows = np.asarray(obs[:6]), []
    for _ in range(min(steps_left, 200)):
        rows.append(pd.act(state, {}))
        state, _ = model.advance(rows[-1] * view.max_accel, view.dt)
    return np.array(rows)


agent = PlanAgent(pd_plan)
env = ApproachEnv(ApproachConfig(start_state=(-150.0, -400.0, 0.0, 0.0, 0.0, 0.0)))
result = run_episode(env, agent, seed=0)
print(agent.log[0]["verdict"].capture_step, result["captured"])  # the predicted step, True
```
