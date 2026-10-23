# Expert datasets

Code: `OracleAgent` in `vbar.oracles.expert`; `expert_dataset` and `ExpertDataset` in
`vbar.oracles.datasets`; the generator `benchmarks/make_datasets.py`. Hill frame, metres,
seconds.

An expert dataset holds what an expert agent saw and did on public seeds of one scenario: its
observations, actions, rewards and episode ends, for behaviour cloning, DAgger-style imitation
or offline reinforcement learning. The expert is `OracleAgent`, which flies the plan behind the
[fuel oracle's](oracles.md) upper bound, so on the CW rung its delta-v lies inside a certified
bracket.

## How it is built

**The expert.** `OracleAgent` is an ordinary agent. It gets the scenario's `AgentView`, its own
random stream, the observations and the agent-safe info, and never the environment. At its first
decision it calls `FuelOracle(config=config).solve(view, obs[:6])` at a free final time with
every listed constraint hard (only keep-out is modelled by the plan). The plan holds one Hill
acceleration per step and reaches the capture ball at its arrival step, then holds it for the
dwell, verified by exact CW propagation. Each step returns the next row in the view's action
mode: over `max_accel` for `"normalized"`, as is for `"accel"`, and times `mass_kg` for
`"thrust"`. After the plan ends it asks for no thrust. With perfect navigation the first
observation is the true start, so the agent flies exactly what the oracle verified.

By default the agent flies open loop. With `replan_steps`, it solves again from the observation
every `replan_steps` steps until the plan's arrival step, for the steps left in the episode with
the thrust inhibit windows shifted to the new start; a solve that verifies no plan keeps the
current one. The random stream is not used.

**The dataset.** `expert_dataset(scenario_id, split, indices, *, agent=None, **config)` builds
the environment with `make(scenario_id, **config)` and flies episode $k$ of `indices` with the
episode seed `seeds(scenario_id, split, ...).seed(k)` and the agent seed `agent_seed(k)` of the
same split, both public. The agent is reset with `agent_generator(agent_seed)`, as
`run_episode` does. Copies of every observation and action are kept as the agent receives and
returns them, so an agent that changes its arrays in place cannot change the record. The test
split is refused before any seed is derived, and the hidden key variable is never read.

**The file.** `ExpertDataset.save(path)` writes an NPZ file (a zip of `.npy` arrays, readable
with `numpy.load`) and the dataset card at `path.with_suffix(".md")`. The zip members are
written in a fixed order, each deflated at level 6 with a fixed time stamp (1980-01-01), so the
bytes depend only on the arrays and the zlib build. `ExpertDataset.load(path)` reads both back
and refuses a file whose SHA-256 differs from the card's.

## Fields

| Array | Shape | Contents |
|---|---|---|
| `observations` | (steps + episodes, d) | Every observation the agent received, the first of each episode included (m, m/s, then any enabled channels) |
| `actions` | (steps, 3) | Commands in the scenario's action mode |
| `rewards` | (steps,) | Rewards the environment returned |
| `terminations`, `truncations` | (steps,) | Episode ends the environment returned, bool |
| `episode_starts` | (episodes,) | First step of each episode, int64; episode $i$'s observations start at row `episode_starts[i] + i` |
| `seeds`, `agent_seeds` | (episodes,) | Episode and agent seeds as 32 lowercase hex digits |

No true state, start state, step record or hidden seed is stored. Resetting the scenario with
`int(seed, 16)` and stepping the stored actions reproduces the stored observations, rewards and
episode ends bit for bit.

## The card

The card's front matter has one `key: value` line per field:

| Field | Value |
|---|---|
| `vbar_version` | Package version |
| `git_revision` | Commit of the source tree the package ran from, with `-dirty` if tracked files had changed, or `unknown` outside a checkout |
| `scenario_ids` | Full identifier of the environment, configuration tokens included |
| `split` | `train` or `validation` |
| `indices` | Episode indices as runs, `0-49` or `3-5, 9` |
| `agent` | The agent's name |
| `oracle_config` | Every `OracleConfig` setting and `replan_steps`, or `none` for another agent |
| `numpy_version`, `scipy_version`, `highs_version` | Library versions; HiGHS as scipy bundles it, `unknown` if scipy does not expose it |
| `platform` | Operating system, release, machine and Python |
| `sha256` | SHA-256 of the NPZ file |
| `e1_summary` | Mean, least and greatest E1 (m/s) over the episodes |

The body describes the arrays and states that the data have "deterministic behaviour, no action
probabilities". Each stored action is the only action the expert takes in its state, so offline
value-based reinforcement learning sees no other action and has no coverage to estimate their
values from; behaviour cloning and DAgger need only the expert's action and are unaffected.
Agents trained on a dataset name it and its SHA-256 in their results.

## Generating and hosting

`python benchmarks/make_datasets.py OUTPUT_DIR` writes 50 validation episodes of the V-bar and
R-bar approach scenarios, one NPZ file and card each; `--scenario`, `--split`,
`--start`, `--count` and `--replan-steps` change that. It refuses to run without an existing
output directory and has no default location. The data files are hosted outside this
repository; the script and the seeds rebuild them.

## Assumptions and limits

- Perfect navigation only. `expert_dataset` refuses a scenario whose sensor is not
  `PerfectNav`, which covers the `:nav1` and `:nav2` variants: an open-loop plan from a noisy
  first observation is not the oracle's plan, and a closed-loop expert would carry solver ties
  into the trajectory.
- Continuous action modes only. `OracleAgent` refuses `"disc27"` and `"impulse_coast"`, and
  `expert_dataset` refuses those modes for any agent, since the arrays hold three numbers per
  step.
- Hold and berthing goals only, with a capture ball and a briefed target that does not
  manoeuvre; `OracleAgent` refuses docking, inspection and departure goals. The plan is exact on
  the CW rung; on other rungs it is flown open loop without correction unless `replan_steps` is
  set, and may miss the goal.
- Replanning carries no guarantee of staying under the oracle's upper bound. Each solve is a
  new free-final-time problem from the observed state, and the costs of the flown pieces need
  not add up to the cost of the first plan. With perfect navigation on the tested seed every
  replan found the rest of the first plan again and the flight stayed within the bracket; that
  is one case, not a property.
- The plan is an acceleration history, flown exactly in `"normalized"` and `"accel"` modes with
  the default acceleration drive. In `"thrust"` mode the agent converts at the initial mass
  `mass_kg`, so the delivered acceleration exceeds the plan by the ratio of the initial to the
  current mass; with the thrust drive it rises within each burn. Execution errors are not
  corrected in open loop.
- The plan verifies keep-out only. Soft constraints such as closing-speed limits are scored,
  not planned around.
- Platform dependence. The expert's actions come from linear programs solved by HiGHS, whose
  ties between plans of equal cost can resolve differently with other CPU kernels, BLAS or
  library versions. Files are identical byte for byte only for the same seeds, release, library
  versions and platform; across platforms actions and E1 may differ, by an amount no test
  measures. The zip's deflate stream also depends on the zlib build.
- If the first solve verifies no plan, `OracleAgent.act` raises `RuntimeError` rather than
  recording a non-expert episode.

## Validation

Tests: `tests/test_expert_datasets.py`, on `vbar:approach-rbar:cw:box:dv:v1` with the oracle
settings `arrival_candidates=4` and `bank_stride=20`. Flown oracle episodes are 34 steps of 30 s
(1020 s) with 2 constraint samples per step, through `make`'s `dt`, `max_steps` and
`violation_samples`; validation seeds 0 and 1 arrive at step 14 and complete the 600 s dwell by
the last step. Tests that only solve use 100 steps of 10 s.

- On validation seeds 0 and 1 the open-loop oracle agent succeeds, and its E1b lies in the
  certified bracket `bounds_for` gives, within $10^{-6}$ relative of the upper bound.
- The agent's `reset` takes only the view and a random stream and `act` only the observation and
  info. Its first three actions equal the oracle's plan from the first observation to $10^{-12}$
  relative, with no environment built; shown a shifted observation it plans from that, not from
  the true state; two random streams give the same actions.
- `"accel"` mode returns the plan in m/s^2, and `"thrust"` mode its first three rows times the
  initial mass, both to $10^{-12}$ relative. A dataset in `"thrust"` mode records the mode in
  `scenario_ids`, starts from the same observation as the `"normalized"` one, stores the same
  plan rows in newtons, and its E1 exceeds the `"normalized"` flight's.
- With 70 steps of 10 s (100 s before the dwell) validation seed 0 has no verified plan, and
  `act` raises `RuntimeError`.
- Replanning every 10 steps on validation seed 1 solves at steps 0, 10, ... up to the plan's
  arrival and no later, at least twice, and the flight succeeds with E1b in the open-loop
  bracket.
- It refuses docking, inspection and departure goals, the `"disc27"` and `"impulse_coast"`
  modes and a briefed manoeuvring target.
- Saved files hold exactly the documented arrays, none of dtype object, with the documented
  shapes and episode ends; the seeds are the public validation seeds in hex; replaying each
  seed with the stored actions reproduces the observations, rewards and episode ends bit for
  bit.
- Save, load and save again gives identical NPZ and card bytes, and regenerating the oracle
  dataset gives identical bytes on the same machine.
- The card has every field above other than `agent`, the versions numpy and scipy report, the
  file's SHA-256, the oracle settings, the index range and the behaviour sentence with its
  consequences for value-based learning, behaviour cloning and DAgger.
- The test split is refused with and without a hidden key set, an unknown split is refused, and
  so is a `:nav1` scenario. Generation never reads the hidden key variable.
- An attacker holding the first ten training episodes finds no hidden-split seed written in hex,
  decimal or binary in the files, no token that derives one as the hidden key, and no hidden
  episode's first observation in the data; the files are identical under two random hidden keys
  and none.
- The generator script exits with an error and writes nothing when given no arguments.

## Example

```python
import tempfile
from pathlib import Path

import numpy as np

from vbar.oracles.datasets import ExpertDataset, expert_dataset
from vbar.oracles.expert import OracleAgent
from vbar.oracles.fuel import OracleConfig
from vbar.scenarios import make

sid = "vbar:approach-rbar:cw:box:dv:v1"
# A shortened episode and a lighter oracle keep the example quick.
short = {"dt": 30.0, "max_steps": 34, "violation_samples": 2}
agent = OracleAgent(config=OracleConfig(arrival_candidates=4, bank_stride=20))
dataset = expert_dataset(sid, "validation", [0], agent=agent, **short)

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "approach-rbar-validation-0.npz"
    dataset.save(path)
    card = path.with_suffix(".md").read_text(encoding="utf-8")
    loaded = ExpertDataset.load(path)

assert "deterministic behaviour, no action probabilities" in card
assert np.array_equal(loaded.actions, dataset.actions)

# The stored actions replay the stored observations from the public seed.
env = make(sid, **short)
obs, _ = env.reset(seed=int(loaded.seeds[0], 16))
for k, action in enumerate(loaded.actions):
    assert np.array_equal(obs, loaded.observations[k])
    obs, reward, terminated, truncated, _ = env.step(action)
assert terminated and np.array_equal(obs, loaded.observations[-1])
print(loaded.card["e1_summary"])
```
