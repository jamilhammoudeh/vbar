# Training interfaces

Code: `vbar.scenarios.base` (the `"disc27"` and `"impulse_coast"` action modes),
`vbar.agents.base` (`DISC27`, `command_fraction`, `zero_action`). Options that make
reinforcement learning easier without changing what is scored. Rewards are on their own page
([rewards.md](rewards.md)).

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

## Burn and coast decisions

`action_mode="impulse_coast"` makes one decision a burn followed by a coast: the action is
$(f_x, f_y, f_z, t_c)$, fractions of a full command held for one step $\Delta t$, then no thrust
for $t_c$ seconds rounded to whole steps and cut at the time limit. The agent sees nothing
between the burn and the end of the coast; the environment checks constraints over the whole
interval in continuous time, as in any step. One decision can span hundreds of steps, so a
transfer of two burns takes two or three decisions instead of hundreds, the semi-Markov
formulation of Sutton and Barto (sec. 14.4).

The step's duration is $(1 + k)\Delta t$ for $k$ coast steps, `info["time_s"]` and the step count
advance by it, and with a library reward the transition is discounted by
$e^{-(1 + k)\Delta t/\tau}$, given in `info["discount"]`; its reward is the discounted integral
over the whole interval, so a decision earns exactly what the steps it replaces would have
earned ([rewards.md](rewards.md)). Success is checked at the end of each decision, not inside a
coast. The action box is $[-1, 1]^3 \times [0, T_{\max}]$ s; a learner that expects
$[-1, 1]$ per entry can rescale it, for example with Gymnasium's `RescaleAction` wrapper.

`PDAgent` and `LLMAgent` reject the mode; `DriftAgent` returns zero thrust and no coast.

## Training-only starts

An environment built with `training=True` accepts `reset(seed=..., options=...)` with:

- `"start_state"`: the start state itself (m, m/s), for exploring starts or replaying a state;
- `"start_pos_range"` and `"start_vel_range"` (approach scenario): per-axis (low, high) ranges
  that replace the configured ones, for wider starts or a curriculum.

Options apply to that episode only; the next `reset` without them draws from the configuration
again. Without `training=True`, any option raises a `ValueError`, and `run_episode`, the scored
path, refuses an environment built for training, so these starts never reach a scored run
(Sutton and Barto, sec. 5.3, exploring starts; sec. 15.5, shaping by a curriculum of easier
tasks). Unknown option names are rejected.

## How it is tested

`tests/test_training.py`:

- Thirty random indices fly exactly the trajectory of the same commands given as continuous
  fractions; the 27 commands are distinct and index 13 is zero.
- The scenario ID ends in `:disc27`, Gymnasium's environment checker passes, and an index outside
  0 to 26 is rejected.
- The PD and LLM agents reject the mode; the drift agent coasts.
- `command_fraction` reads every action mode, and `zero_action` asks for no thrust in each.
- Eight random burn-and-coast decisions through the keep-out sphere end in the same states as the
  same burns flown one step per decision (to 1e-11 of the largest component, since one long coast
  and several short ones round differently), with the same discounted reward, the same discount
  and the same minimum range, with and without discounting.
- A coast is rounded to whole steps and cut at the time limit, which truncates the episode; a
  non-finite coast is rejected. Gymnasium's environment checker passes; the PD agent rejects the
  mode and the drift agent coasts.
- Reset options raise without `training=True` and for unknown names; an injected start is used
  exactly, wider ranges reach beyond the default ones over 200 seeds, and the next plain reset
  draws the same start as an environment never given options; `run_episode` refuses a training
  environment.

## Examples

```python
from vbar import ApproachConfig, ApproachEnv
from vbar.agents import DISC27

env = ApproachEnv(ApproachConfig(action_mode="disc27"))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(4)
print(env.scenario_id, DISC27[4])  # ...:v0:disc27 [-1.  0.  0.], full thrust toward -x
```

```python
from vbar import ApproachConfig, ApproachEnv
from vbar.rewards import Reward

env = ApproachEnv(ApproachConfig(action_mode="impulse_coast", reward=Reward(tau=3600.0)))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step([0.0, -0.5, 0.0, 600.0])
print(info["time_s"], round(info["discount"], 4))  # 610.0 s in one decision, 0.8441
```
