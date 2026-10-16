# Training interfaces

Code: `vbar.scenarios.base` (action modes), `vbar.agents.base` (`DISC27`, `command_fraction`,
`zero_action`). Options that make reinforcement learning easier without changing what is
scored. Rewards are on their own page ([rewards.md](rewards.md)).

## Discrete bang-off-bang actions

`action_mode="disc27"` replaces the continuous action with `Discrete(27)`: index $i$ selects the
command `DISC27[i]` from $\{-1, 0, 1\}^3$, fractions of a full command per Hill axis, held for the
step. Index 13 is no thrust. Value-based learners (the DQN family) can then act without a
maximization over a continuous action (Sutton and Barto, *Reinforcement Learning*, 2nd ed. draft,
sec. 9.4), and little is lost: the fuel-optimal bounded control of the linear problem is
bang-off-bang. The scenario ID gains the suffix `:disc27`, so results in this mode are never
mixed with continuous ones.

With ball thrusters a diagonal command has length up to $\sqrt 3$ and is scaled to the unit
sphere, as any over-long command is ([spacecraft.md](spacecraft.md)).

Agents that produce continuous commands reject the mode: `PDAgent` and `LLMAgent` raise a
`ValueError` at reset, because `action_scale(view)` has no scale for it. `DriftAgent` returns
`zero_action(view)`, the no-thrust action of any mode. `command_fraction(view, action)` converts an
action of any mode into fractions of a full command at the initial mass.

## How it is tested

`tests/test_training.py`:

- Thirty random indices fly exactly the trajectory of the same commands given as continuous
  fractions; the 27 commands are distinct and index 13 is zero.
- The scenario ID ends in `:disc27`, Gymnasium's environment checker passes, and an index outside
  0 to 26 is rejected.
- The PD and LLM agents reject the mode; the drift agent coasts.
- `command_fraction` reads every action mode, and `zero_action` asks for no thrust in each.

## Example

```python
from vbar import ApproachConfig, ApproachEnv
from vbar.agents import DISC27

env = ApproachEnv(ApproachConfig(action_mode="disc27"))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(4)
print(env.scenario_id, DISC27[4])  # ...:v0:disc27 [-1.  0.  0.], full thrust toward -x
```
