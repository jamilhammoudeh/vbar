# Mutant agents

Code: `Mutant` and `MUTANT_KINDS` in `vbar.agents.mutants`. Hill frame ([frames.md](frames.md)),
SI units.

A mutant is a deliberately broken controller: `Mutant(agent, kind, severity, noise_seed=7)` wraps
any agent with continuous commands and adds one defect of a chosen kind and size. Mutants check
the scoring rather than the agent: a metric responsible for a defect must move the wrong way when
the defect is present and, for defects with an ordered size, rank larger defects worse. A metric
that does not notice a sign error or a delay would let a broken controller score as a good one.

## How it is built

The wrapper changes the observation before the inner agent sees it, or the inner agent's command
after it is issued. Commands are handled in fractions of a full command (`command_fraction`), so
a defect means the same thing in the `"normalized"`, `"accel"` and `"thrust"` modes, and the
result is converted back to the view's action mode. With $\mathbf u$ the inner command in
fractions, $s$ the severity, $n$ the view's mean motion and $a_{\max}$ its `max_accel`:

| Kind | Rule | Severity |
|---|---|---|
| `position_bias` | the observed radial position becomes $x + s$ | m |
| `velocity_bias` | the observed radial velocity becomes $\dot x + s$ | m/s |
| `delay` | the inner agent sees the observation of $\lfloor s \rfloor$ decisions before (the first one until that many exist) | steps |
| `sign_flip` | $-\mathbf u$ | not used |
| `dither` | $\operatorname{clip}(\mathbf u + s\,\boldsymbol\xi_k, -1, 1)$, $\boldsymbol\xi_k$ standard normal per axis | fraction |
| `chatter` | $\operatorname{clip}(\mathbf u + ((-1)^k s, 0, 0), -1, 1)$ at decision $k = 1, 2, \dots$ | fraction |
| `radial_sign` | $(-u_x, u_y, u_z)$ | not used |
| `no_coriolis` | $\operatorname{clip}(\mathbf u + s\,(2n\dot y, -2n\dot x, 0)/a_{\max}, -1, 1)$, observed velocity | fraction of the Coriolis term |
| `bang_bang` | $s\,\operatorname{sign}(\mathbf u)$ per axis | fraction |

At $s = 1$, `no_coriolis` undoes the cancellation of the Coriolis acceleration that a law such as
`PDAgent` performs. The inner agent gets the view and the generator given to `reset` unchanged,
so a random inner agent draws what it would unwrapped. The dither draws come from a child
generator spawned from that one (spawning leaves the parent's stream untouched), or from
`numpy.random.default_rng(noise_seed)` when `reset` gets none; they never depend on the episode
seed or on the inner agent's draws, and for one stream the levels of $s$ are scaled copies of one
sequence. The wrapper copies the observation
and works on a new array for the command, so neither the caller's observation nor the inner
agent's returned array is changed in place. It reads no `info` key; `info` is passed to the
inner agent as given.

## Which metric catches which mutant

From the approach 400 m behind the hold point, (0, -400, 30) m at rest, 200 steps of 10 s, exact
navigation, with `PDAgent` inside. "Ordered" means the metric moves strictly with the severity at
the levels listed; "detected" means it differs from the unmutated agent in the stated direction.

| Kind | Levels | Metric | Direction |
|---|---|---|---|
| `position_bias` | 2, 10, 40 m | terminal position error; composite score | ordered, up; nonincreasing, last below the unmutated |
| `velocity_bias` | 0.01, 0.03, 0.1 m/s | terminal position error; composite score | ordered, up; nonincreasing, last below the unmutated |
| `delay` | 1, 3, 6 steps | delta-v | ordered, up |
| `delay` | 6 steps | keep-out margin | detected, negative (enters the zone) |
| `dither` | 0.05, 0.15, 0.45 | delta-v | ordered, up |
| `dither` | 0.15, 0.45 | keep-out margin | ordered, down (0.05 is inside the seed noise) |
| `chatter` | 0.02, 0.1, 0.4 | chatter | ordered, up |
| `sign_flip` | | success; progress | detected, failure with zero progress |
| `sign_flip` | | sign check | fails all six directions |
| `radial_sign` | | success; terminal position error | detected, failure with a larger error |
| `radial_sign` | | sign check | fails +x and -x |
| `no_coriolis` | 1.0 | time to complete; delta-v | detected, both up; detection only |
| `bang_bang` | 0.1, 0.3, 1.0 | chatter | detected, above the unmutated at every level |
| `bang_bang` | 0.1, 0.3, 1.0 | delta-v | ordered, up |

`no_coriolis` is detection only. The Coriolis force is perpendicular to the velocity and does no
work, so no fuel or error ordering follows from the physics, and the terminal position error is
not monotone in its severity on this approach. The sign check passes `no_coriolis` and
`bang_bang`, whose first command from rest keeps its sign. Filter metrics judge a safety filter,
not the agent, so no agent mutant is ranked by them.

## Assumptions and limits

- Only the continuous action modes: `reset` raises `ValueError` from `action_scale` for
  `"disc27"` and `"impulse_coast"`, whatever the inner agent accepts.
- `position_bias`, `velocity_bias` and `chatter` act on the radial axis only; `delay` counts
  decisions, not seconds, and a fractional delay severity is rounded down to whole steps.
- The additive kinds clip each axis to a full command, as a box thrust layout does; on a ball
  layout the environment's own limit applies after that.
- The orderings above are established for the stated start, step count and inner agent. A
  different agent or scenario can reorder close levels; detection of `sign_flip` and
  `radial_sign` does not depend on them.
- `bang_bang` with $s > 1$ asks for more than a full command; the environment clips it.

## How it is tested

`tests/test_mutants.py` checks each kind's rule against `PDAgent` commands on hand-made
observations near the hold point: the biases shift what the inner agent sees, the delay feeds the
observation from $s$ steps before, the sign flips negate, `bang_bang` gives $s\,\operatorname{sign}(\mathbf u)$,
`no_coriolis` adds back $s$ times the Coriolis acceleration, `chatter` alternates by $s$ on the
radial axis, and `dither` levels are scaled copies of one noise sequence. Commands agree in
fractions across the three continuous modes to 1e-12 for every kind; the discrete and impulsive
modes are refused even around an inner agent that accepts any view; neither the observation nor
the inner command is changed in place; reset wipes the delay history, the chatter phase and the
dither stream; the dither follows the agent seed and its `noise_seed`, never the episode seed;
and the wrapper reads no `info` key but `time_s`. The metric rows of the table above are
`test_mission_metrics_catch_a_flipped_radial_command`,
`test_dropping_the_coriolis_cancellation_is_detected`,
`test_bang_bang_raises_chatter_and_delta_v_with_its_amplitude`,
`test_dither_lowers_the_keep_out_margin_with_its_size`,
`test_a_long_decision_delay_enters_the_keep_out_zone` and
`test_sign_check_fails_flipped_axes_and_passes_the_other_kinds` there, and the bias, delay,
dither, chatter and sign-flip rows in `tests/test_metrics_mutants.py`. Around `LQRAgent`, which
captures from 50 m behind the hold point, the sign flip fails the mission with zero progress
(`test_wrapper_breaks_an_lqr_agent_that_succeeds`).

## Example

```python
from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent
from vbar.agents.mutants import Mutant

behind = (0.0, -400.0, 30.0, 0.0, 0.0, 0.0)
env = ApproachEnv(ApproachConfig(start_state=behind, max_steps=200))
for agent in (PDAgent(), Mutant(PDAgent(), "delay", 3), Mutant(PDAgent(), "radial_sign", 1)):
    out = run_episode(env, agent, seed=1, agent_seed=2)
    print(agent.name, out["captured"], round(out["dv_used_mps"], 3))
```
