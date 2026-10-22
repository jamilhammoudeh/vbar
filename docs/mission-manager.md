# Mission manager

Code: `Mode` and `MissionManagerAgent` in `vbar.agents.mission`.

A rendezvous is flown as a timeline of modes: each mode has its own guidance and control, runs
until a gate on time and state says the next mode may start, and has a contingency mode for when
its time runs out and an abort mode for when safety demands it. `MissionManagerAgent` is that
mode manager as an agent: it forwards each decision to the active mode's agent and switches modes
at their exits. It is the classical reference for multi-phase approaches and for agents that act
as an operator choosing modes.

## How it is built

The structure follows Fehse, *Automated Rendezvous and Docking of Spacecraft*, section 6.3: a
timeline requests modes in order, a transition table checks time and state criteria against the
navigation data, and each phase has a contingency mode loaded when its criteria are not met.

**Modes.** `Mode(*, name, agent, exit=None, t_max=None, next=None, contingency=None, abort=None)`
is a frozen dataclass. `agent` is any agent; `exit(obs, t_in_mode)` is the gate to the mode named
`next`, with `t_in_mode` the time in the mode (s); `t_max` is the time in the mode (s) after which
the mode named `contingency` runs if the gate does not hold; `abort` names the mode taken when
safety demands it.

**Manager.** `MissionManagerAgent(modes, *, abort_when=None, name="mission")` checks at
construction that mode names are unique, that every `next`, `contingency` and `abort` names a mode
of the timeline, that a mode with a `contingency` has a `t_max`, and that the last mode has no
`next`, and raises `ValueError` otherwise.

At reset the first mode becomes active. Each time a mode is entered its agent is reset with the
manager's view and random stream, so a mode entered twice starts afresh; its clock starts at its
first decision. Each decision goes to the active agent, which sees the observation and the
agent-safe info only. After the agent acts, the exits are checked in this order, and the first
that holds is taken:

1. **Abort**, to `abort`: `abort_when(obs, time_s)` holds, with `time_s` the episode time (s), or,
   in the `"impulse_coast"` action mode, the coast the action commits is not predicted clear of the
   keep-out zone: the certified lower bound of `vbar.agents.planning.predicted_margin` over that
   coast, from the observed state carried through the burn step on the exact CW matrices, is
   negative.
2. **Contingency**, to `contingency`: the time in the mode is strictly above `t_max` and the exit
   gate does not hold.
3. **Next**, to `next`: the exit gate holds, or the agent has set `done = True`.

An exit whose mode is None is not taken, so without an abort mode a coast predicted to enter the
keep-out zone is flown, and the metrics score what happened. On any transition a coast the action
commits is cut to zero, so the new mode takes over one step later.

`log` holds one entry per decision: `mode`, `t_in_mode` (s), `transition` (`"abort"`,
`"contingency"`, `"next"` or None), `to` (the mode entered next, or None), and, over a committed
coast, `predicted_margin` and `predicted_margin_bound` (m; None without a coast).

## Assumptions and limits

- The manager adds no guidance of its own: each mode is only as good as its agent, and the
  transitions are only as good as the gates the user writes. Gates see the observation, so with
  navigation noise they fire on the estimate, not the truth.
- `t_max` without a `contingency` has no effect.
- The coast prediction covers the keep-out sphere only, and assumes Clohessy-Wiltshire motion; the
  burn step before the coast is not itself checked.
- Fehse's modes also schedule equipment and fault detection; only guidance and control modes are
  modelled here.

## Validation

Tests: `tests/test_mission.py` and `tests/test_mission_transitions.py`. From rest at
$(0, -300, 0)$ m with 0.1 m/s across the track, the cross-track position is 1, 2, 3, 4 and 5 m at
the decisions 10 to 50 s, so gates on it fire at known decisions.

- A four-mode timeline moves on at a position gate, runs its contingency when a mode's time
  exceeds its limit, and moves on at a time gate, each at the expected decision and logged there;
  `t_in_mode` restarts at zero in each mode, and a second episode starts again from the first
  mode.
- The contingency fires only once the time in the mode is above its limit, not at it; a gate that
  holds after the limit moves on instead.
- An agent that sets `done` moves the timeline on.
- When the abort predicate, the gate and the time limit all hold at one decision, the abort wins.
- Timelines that name missing modes, give a contingency without a time limit, or whose last mode
  has a next mode, are refused.
- A next transition cuts the coast the leaving mode's agent committed.
- A mode agent is reset with the manager's view and random stream each time its mode is entered,
  and mode agents see only the observation and the time.
- From 100 m behind the target closing at 0.5 m/s, a 300 s coast passes 21 m from it: with an
  abort mode the manager cuts the coast and switches to it; without one it flies the coast, logs a
  predicted margin below $-20$ m with its bound below it, and the run enters the keep-out zone.

## Example

A glideslope approach to 20 m from the hold point, then a PD hold:

```python
import numpy as np

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent
from vbar.agents.glideslope import GlideslopeAgent
from vbar.agents.mission import MissionManagerAgent, Mode

hold = np.array([0.0, -100.0, 0.0])
manager = MissionManagerAgent(
    [
        Mode(
            name="approach",
            agent=GlideslopeAgent(),
            exit=lambda obs, t: np.linalg.norm(obs[:3] - hold) < 20.0,
            next="hold",
        ),
        Mode(name="hold", agent=PDAgent()),
    ]
)
config = ApproachConfig(start_state=(0.0, -300.0, 0.0, 0.0, 0.0, 0.0))
result = run_episode(ApproachEnv(config), manager, seed=1, agent_seed=2)
switch = next(entry for entry in manager.log if entry["transition"])
print(f"hold entered after {switch['t_in_mode']:.0f} s of approach")
print(result["captured"], result["time_s"])
assert result["captured"]
```
