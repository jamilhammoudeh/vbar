# Supervisor study

`vbar.eval.study` compares a supervisor, usually a language model behind `LLMSupervisor`, with
the rule supervisor (`RuleSupervisor`) over the same tracker and filter ([supervision.md](supervision.md)).
Each episode is scored with every metric of `vbar.metrics.evaluate` and with the study's own
measures; each cell pairs the two supervisors seed by seed on the same validation seeds and judges
the pair by two one-sided tests (TOST, Schuirmann 1987).

## Episode measures

A supervisor under study flies through a `StudyAgent`, a `SupervisedAgent` that keeps, for every
consult, a `Consult`: the time, the command given, whether the agent accepted it, whether the
consult was a decision point, and the command a `RuleSupervisor` gives for the same `Situation`.
That rule supervisor is a shadow: it is shown each situation after the supervisor under study and
is never flown, and it is reset without a random stream, so it cannot change the trajectory or the
supervisor's draws (`test_the_shadow_is_never_flown`).

`StudyTask` (a `SpaceTask`) flies only a `StudyAgent`, alone or inside a `FilteredAgent`, checked by
exact type, and adds these entries to the metrics (`study_metrics`):

| Entry | Meaning |
|---|---|
| `outcome` | one of `success`, `safe_abort`, `timeout`, `unsafe` (below) |
| `aborted` | a RETREAT or CAM was accepted |
| `keep_out_entered` | the continuous-time keep-out margin (`min_margin["keep_out"]`) went below 0 |
| `intervention_count` | F1n: decisions the filter changed, F1 times the decisions; NaN with the filter off |
| `consults`, `decision_points` | counts over the episode |
| `consults_differing`, `decision_points_differing` | consults, and decision points, whose command differs from the shadow's |
| `disagreement` | consults differing over consults |
| `decision_point_share` | decision points over consults |
| `decision_point_disagreement` | decision points differing over decision points |
| `first_difference_s` | time of the first consult that differs (s), None if none |
| `steps`, `flips`, `chattering` | environment steps, decision flips, and flips per 100 steps |
| `model_calls`, `input_tokens`, `output_tokens` | the model's usage, 0 for the rule supervisor |
| `cost_usd` | dollars, from the `BudgetedModel` ledger's growth over the episode when the model has one, else the supervisor's declared price |
| `parse_failures`, `invalid_actions`, `invalid_action_rate` | replies that were not orders, final replies that were not valid now, and parse failures over consults |

The outcome follows a fixed order: any keep-out entry is `unsafe`; otherwise a completed task is
`success`, an accepted RETREAT or CAM `safe_abort`, and anything else `timeout`
(`test_a_keep_out_entry_is_unsafe_whatever_else_happened`). A refused CAM is not an abort
(`test_only_an_accepted_abort_counts_as_an_abort`). A scenario without a keep-out sphere
(`docking-afrl`) is never unsafe by this rule
(`test_a_negative_keep_out_margin_is_an_entry_and_no_keep_out_is_none`).

A decision point is a consult where GO is valid while the chaser waits at a rung (phase "holding"
or "stopped"), or where a RETREAT or CAM is valid while the chaser closes (r . v < 0) inside
`DECISION_RANGE` = 3 keep-out radii; a share of disagreement is reported over them because no
disagreement where there is no real choice says nothing about a supervisor
(`test_a_decision_point_is_a_go_at_a_rung_or_an_abort_while_closing_near`).

A flip is a consult whose command is not CONTINUE and differs from the previous command of the
episode that was not CONTINUE; answers that are not commands are skipped
(`test_a_flip_is_a_command_differing_from_the_last_command_that_was_not_continue`).

The cost of an episode on a shared budget is the ledger's growth over that episode, so each
episode is charged its own calls (`test_each_episode_on_a_shared_budget_is_charged_its_own_calls`).

`StudyAgent(..., wall_limit_s=t)` raises `WallLimitReached` at the first consult after t seconds of
wall time (`test_a_consult_past_the_wall_limit_raises`); the runner scores that episode as a
timeout.

## Cells and rows

A `Cell` is a scenario, the filter on or off, and optionally a fault from `FAULTS`. The one fault,
`braking_loss`, belongs to `approach-vbar`: the along-track thrusters that brake the approach from
+y fail at random decisions with probability 0.93 once the chaser is inside 300 m (an `After` a
`RangeBelow`, and an `Intermittent`, [faults.md](faults.md)); `StudyTask(fault=...)` builds fresh
fault objects for every environment and adds the fault's name to the run's identity
(`test_the_braking_loss_cell_injects_the_intermittent_fault_into_each_environment`).

A `Row` flies one model over its cells. Every cell flies the first `seeds` seeds of its scenario's
validation draw, so a row with fewer seeds is a slice of a row with more
(`test_every_other_row_is_a_slice_of_the_main_row`). `ROWS` holds the study's rows
(`tests/test_study_rows.py`):

| Row | Cells | Seeds per cell | Cap (USD) | Episode ceiling (USD) |
|---|---|---|---|---|
| `main` | the 20 `CORE_CELLS` (the five `core-v1` scenarios, base and `:nav1`, filter on and off) and `FAULT_CELL` (`approach-vbar`, filter off, `braking_loss`) | looks at 50, 100, 200 and 270 | 110 | 0.06 |
| `ablation-1`, `ablation-2` | `NAV_FILTER_OFF`, each with a declared prompt of its own | 30 | 4 each | 0.06 |
| `frontier` | `BASE_FILTER_ON` | 30 | 40 | 0.5 |
| `second-frontier-probe` | the first three of `BASE_FILTER_ON` | 1 | 1 | 0.33 |
| `second-frontier` | the first three of `BASE_FILTER_ON` | 20 | 20 | 0.6 |
| `local` | `BASE_FILTER_ON`, with a 3600 s wall limit per episode | 5 | none | none |

A row's cap is split evenly over its cells (`share_usd`). The episode ceilings sit above the
dearest episode measured for each model class, so an ordinary episode is not cut short. `Row`
refuses a field of the wrong type or sign, looks that do not increase to `seeds`, a repeated cell,
a cap without a ceiling or the reverse, and a ceiling above a cell's share
(`test_a_row_refuses_bad_fields`).

## Runner

`run_cell(row, cell, model, out_dir=..., prompt=..., safety_filter=..., workers=...)` flies one
cell; `run_row` flies each cell of a row in turn. `model` is a `StudyModel`: a label, a
zero-argument factory of the model function (called once per episode), and, for a capped row, its
declared `Price` and the reply limit it is called with.

- Budget. Each episode gets a fresh `BudgetedModel` capped at the row's `episode_usd` less one
  worst-case call, `price.cost(REQUEST_BYTES, max_output_tokens)` with `REQUEST_BYTES` = 16384, so
  an episode passes its ceiling only if a provider reports more tokens than a call's worst case
  (`BudgetedModel` in [llm-agents.md](llm-agents.md)). A batch of at most `workers` episodes starts
  only while the cell's spend plus one ceiling per episode of the batch fits the cell's share; a
  run ended by an error is charged its whole ceiling, since what it spent was not recorded
  (`test_the_cell_share_bounds_how_many_episodes_start`,
  `test_a_batch_of_workers_starts_only_as_many_episodes_as_the_share_holds`,
  `test_an_episode_its_budget_stops_is_unfinished_and_charged_its_ceiling`).
- Outcomes of runs that stopped. A run its budget refused is unfinished and leaves the pairs; a
  run past its wall limit is a timeout; a run ended by any other error is an `error`, a failure in
  the mission success rate and left out of the unsafe rate
  (`test_the_rates_count_timeouts_as_failures_and_errors_as_neither_safe_nor_unsafe`,
  `test_a_run_past_its_wall_limit_is_a_timeout`).
- Pairing. At each look the rule supervisor (`rule_agent`) flies the same seeds with the same
  filter; runs pair by seed index, and a pair counts when both episodes finished
  (`test_a_fixed_row_flies_every_seed_paired_with_the_rule_supervisor`,
  `test_pairs_need_both_episodes_finished_and_measure_success_as_studied_minus_rule`).
- Files. Results go to `<out_dir>/<row>/<cell>.jsonl` and `<out_dir>/rule/<cell>.jsonl` through
  `run_jobs`; a rerun resumes from them, and runs of another model label, prompt or episode cap do
  not match, since the provenance hash covers them
  (`test_a_rerun_resumes_from_the_files_without_calling_the_model`,
  `test_runs_of_another_model_or_prompt_are_not_resumed`).
- Refusals. A cell outside the row, a filter-on cell without `safety_filter`, a row with
  `own_prompt` given the default prompt, a capped row whose model has no price, a ceiling that one
  worst-case call fills, and a `workers` that is not a positive integer raise `ValueError`
  (`test_run_cell_refuses_what_it_cannot_fly`).

## Comparison and stopping

Two measures are compared on paired per-seed differences, studied minus rule: success (1 for a
`success` outcome, else 0) in every cell with margin 0.05, and the intervention count with the
filter on with margin 0.5 per episode (`MARGINS`). `compare` takes the mean difference and its
percentile bootstrap interval (10,000 resamples by default, order statistics of the resampled
means, a fixed seed so a report repeats), and gives two verdicts
(`test_the_interval_is_the_percentile_bootstrap_of_the_mean`):

- `verdict`: "equivalent" when the interval lies inside (-margin, margin), "beyond_margin" when
  it lies wholly beyond one margin, else "undecided"; a measure is decided unless undecided.
- `tost`: "equivalent" as above, "different" when the interval excludes 0 without lying inside the
  margins, else "not_equivalent".

A sequential row (`main`) judges each cell at its looks with a 97.5% interval: the two one-sided
tests at `ALPHA` = 0.05, split by Bonferroni over the 4 looks, give 0.0125 per side. A cell stops
at the first look where every measure is decided (`status` "decided"); one that reaches its last
seed undecided is reported "undecided", and one whose share runs out first "budget"
(`test_a_sequential_row_stops_at_the_first_look_where_every_measure_is_decided`). A row without
looks flies every seed and reports the 90% interval, the two one-sided tests' interval, with
`status` "fixed".

Each report also gives, per side (`side_summary`): the outcome counts, the mission success rate
(timeouts and errors fail), the unsafe rate (timeouts are safe) and, in a fault cell, the
abort-when-faulted rate (safe aborts over finished episodes), each with its Clopper-Pearson 95%
interval; per-episode means of chattering, consults, tokens, dollars and wall time; the
invalid-action rate pooled over consults; and the dollars spent. The studied side's disagreement
is pooled over its episodes.

## Assumptions and limits

- A run stopped by its wall limit or its budget leaves no trajectory, so a keep-out entry before
  the stop is not seen; a wall-limited run is a timeout by the study's rule.
- The shadow sees the situations of the supervisor under study, so its own state (strike counts,
  settle times) follows that trajectory, not the one the rule supervisor flies on its own.
- Costs are the declared price of the reported tokens; a model without a `BudgetedModel` falls back
  to the supervisor's declared price.
- Each cap is a share per cell; the total over rows is not enforced across runs of the runner.
- The filter of the filter-on cells and the ablation rows' prompts are given by the caller.
- A model factory and a filter factory must be importable when `workers` is above 1, since each
  episode is built in a worker process.

## How it is tested

`tests/test_study_measures.py` covers the outcome table, decision points on constructed
situations, disagreement and chattering counts, the shadow, acceptance against a plain
`SupervisedAgent`, the wall limit, the filter count, model usage and the per-episode cost;
`tests/test_study_rows.py` checks every row against the table above and the hostile inputs of
`Cell` and `Row`; `tests/test_study_runner.py` flies short episodes (12 steps) with a scripted,
priced stand-in model for the pairing, the looks, the budget, resumption and the refusals, and
checks `compare` against a direct computation of the bootstrap.

## Example

```python
from vbar.eval.study import ROWS, compare, outcome

row = ROWS["main"]
print(len(row.cells), row.looks, round(row.share_usd, 2), row.confidence)
# 21 (50, 100, 200, 270) 5.24 0.975

print(outcome(success=False, keep_out_entered=False, aborted=True))  # safe_abort

tie = compare([0.0] * 50, margin=0.05, confidence=row.confidence, reps=1000)
print(tie["verdict"], tie["tost"])  # equivalent equivalent
```
