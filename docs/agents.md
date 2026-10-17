# Agents

Code: `vbar.agents` (`Agent`, `AgentView`, `PDAgent`, `DriftAgent`, `LLMAgent`),
`vbar.agents.sign_check`. The agent interface and the built-in agents are described with the
environment ([environment.md](environment.md)); this page covers the tools around them.

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

## Example

```python
from vbar import ApproachConfig
from vbar.agents import PDAgent
from vbar.agents.sign_check import sign_check

check = sign_check(PDAgent(), ApproachConfig(action_mode="accel"))
print(check.passed, check.commands["-y"])  # True (0.0, -0.25, 0.0)
```
