# Evaluation harness

`vbar.eval` turns a suite, an agent and an optional safety filter into result files and a report:
it flies every episode of the suite, in parallel if asked, records each result with a provenance
hash, and summarises the results with intervals. A suite is reported in full or not at all.

## Running a suite

`run_suite(suite, agent, *, task=SpaceTask(), safety_filter=None, workers=1, out=None)` flies
every (scenario, seed) of a suite and returns one result per run, in suite order. `agent` and
`safety_filter` are factories, zero-argument callables (a class such as `PDAgent` is one) or
import paths `"module:attribute"`, so every episode gets a fresh agent and filter and worker
processes can build their own. `run_jobs(jobs, agent, ...)` does the same for any list of `Job`s
(scenario, split, seed index, seed, agent seed).

- **Independent episodes.** Each episode builds its environment, agent and filter from scratch, so
  a result depends only on the scenario, the seeds, the agent and the filter, never on the number
  of workers or the order the episodes run in. Workers are separate processes started with
  `spawn`; agents given as classes or functions must be importable by them.
- **Resuming.** With `out`, each result is appended to a JSON-lines file as it finishes; running
  again skips every run the file holds with the same provenance, so an interrupted suite resumes.
- **Errors.** An episode that raises is recorded with `error` set to the exception's type and
  message, `success` False and the failure label `"error"`. It is never dropped or run again, so a
  crashing controller cannot shrink the suite it is scored on.

The harness knows a domain only through the `Task` protocol: `make(scenario)` builds an
environment, `record(env, agent, seed, agent_seed)` flies an episode, `score(env, episode)`
returns the metrics by name, and `identity(env)` gives what the provenance covers beyond the seeds,
agent and filter. `SpaceTask` is V-BAR's: catalogue environments, `vbar.metrics.record_episode`
and `vbar.metrics.evaluate` with the scenario's declared settings. A planar double integrator with
a keep-out disk runs through the same runner and report in the tests, unchanged.

```python
from dataclasses import dataclass

from vbar.agents import PDAgent
from vbar.eval import Job, SpaceTask, run_jobs
from vbar.eval.stats import summary_interval
from vbar.scenarios import make, seeds


@dataclass(frozen=True)
class FiveSteps(SpaceTask):
    """The catalogue scenarios cut to five steps, to keep the example quick."""

    def make(self, scenario):
        return make(scenario, max_steps=5)


pairs = seeds("approach-vbar", "validation", 3).pairs()
jobs = [Job("approach-vbar", "validation", k, s, a) for k, (s, a) in enumerate(pairs)]
results = run_jobs(jobs, PDAgent, task=FiveSteps())
assert [r["error"] for r in results] == [None, None, None]
dv = summary_interval([r["metrics"]["delta_v"] for r in results], reps=200, rng=0)
assert dv.lower <= dv.estimate <= dv.upper
```

## Result files

One JSON object per episode (schema 1):

| Field | Content |
|---|---|
| `schema`, `task`, `suite` | schema version, the task's name, the suite identifier |
| `scenario`, `scenario_id`, `split`, `index` | entry name with variants, identifier flown, seed split, index in the split |
| `seed`, `agent_seed` | the seeds, as decimal strings: 128-bit integers do not survive JSON readers that parse numbers as doubles; null on the test split |
| `seed_digest`, `agent_seed_digest` | keyed digests of the two seeds, 64 hex digits; null only for a split with no key, never on the test split |
| `agent`, `filter` | names; `filter` is null without one |
| `metrics`, `failure_label`, `error` | the task's score; the label of a failed run; the exception of a crashed one |
| `provenance`, `environment`, `wall_s` | the run's hash, the software and machine it ran on, its wall time |

The files are strict JSON: NaN and infinite metric values are written as the strings `"nan"`,
`"inf"` and `"-inf"` and read back as floats by `read_results`. `save_trajectories(path, episodes)`
stores the true states, times, observations and commands of recorded episodes in a compressed
`.npz` file keyed by provenance hash; `load_trajectories` reads them back.

## The hidden test split

The seeds of the test split are secret: whoever knows one can rebuild that episode's start state
([scenarios.md](scenarios.md#seeds)). Result files are published, so the lines of the test split
hold no seed:

- `seed` and `agent_seed` are written as null, whatever the result held in memory, and
  `read_results` returns None for them. Null is read only for the test split; in any other split
  it is a refused line. A test split result needs both digests, each 64 lowercase hex digits, when
  it is written and when it is read.
- No other field may hold a secret either. An exception message can carry the seeds of the run
  that raised it, or the key, so in the `error` text of a test split result they are replaced by
  `<hidden>`, in decimal and in hexadecimal (8 characters or more, any case). Anywhere else in
  the line, the hidden key in `VBAR_HIDDEN_SEED_KEY` or either seed makes `encode` raise
  `ValueError` before anything is written. Only those two spellings are searched for: a seed
  written in another base, with digit groups or encoded is not found. This stops a publication,
  not a theft: an agent that
  runs in the harness's own process can read the key from the environment. Run submitted agents in
  a process of their own, started without the key in its environment, such as a remote agent
  ([remote-agents.md](remote-agents.md)).
- `seed_digest` and `agent_seed_digest` name the run instead. `seed_digest(split, seed, index,
  stream="episode")` in `vbar.scenarios.seeds` is the HMAC-SHA256, keyed with the split's key, of
  the text `vbar seed digest|<stream>|<index>|<seed>`; the agent seed uses the stream `"agent"`.
  `run_job` stores both digests in every result of every split, and leaves them None where the
  split has no key (a custom split name). Without the key a guessed seed cannot be tested against
  a digest.
- The provenance hash covers the seeds one way, and a 128-bit seed cannot be searched for, so it
  reveals nothing. Resuming works on a test split file as on any other.

A maintainer checks a published file with the hidden key in `VBAR_HIDDEN_SEED_KEY`. A line belongs
to the run of the suite whose seed it holds or, where its seed is null, whose digest it holds, so
`Suite.require_complete` and `suite_report` accept the published lines of `core-test-v1` as they
accept the lines of `core-v1`, and `latest_results` identifies a run by its digest where the seed
is null. A line whose digest matches no run of the suite belongs to none, so a file with an altered
digest is refused as missing that run. The message about a missing run of a test split entry
names the scenario and the index in the entry, never the seed. The same check, line by line:

```python
from vbar.scenarios import seeds
from vbar.scenarios.seeds import seed_digest

for result in results:
    seed = seeds(result["scenario"], result["split"], result["index"] + 1).seed(result["index"])
    assert result["seed_digest"] == seed_digest(result["split"], seed, result["index"])
```

The block runs on the validation results flown above; on a published test file, whose lines hold
no seed, it is the same with the key set.

The seeds themselves are kept privately. If the environment variable `VBAR_HIDDEN_SEED_LOG`
(`HIDDEN_LOG_VARIABLE` in `vbar.scenarios.seeds`) names a file, `write_results`, and with it
`run_jobs`, `run_suite` and `python -m vbar.eval run`, first appends one JSON line per test split
result to it: `suite`, `scenario_id`, `split`, `index`, `seed` and `agent_seed` as decimal strings,
and both digests. Nothing is written when the variable is unset or empty or the result is of
another split. The file only grows, and a write that cannot open it raises before the result's
line is written, so no private record is lost. It belongs next to the hidden key in the
maintainers' private storage, never beside the published results; `write_results` refuses a log
that is the results file itself. A run killed between the log and
the results file is flown again and logged twice.

`save_trajectories` stores the true states of the episodes, and the start state of a hidden
episode follows from them. Trajectory files of the test split are as secret as the seeds and are
not published.

## Provenance

`provenance_hash(provenance_fields(...))` is the SHA-256 of the canonical JSON (sorted keys, no
spaces) of: the package version, the git revision of the source tree, whether it had local changes
and a SHA-256 digest of those changes (staged and unstaged edits of tracked files, `git diff HEAD`;
None for a clean tree, False and None for an installed package), the task, the identifier flown and
its dynamics model, both seeds, and the agent's and the filter's names and settings. Settings are a declared
`config` mapping, or else the component's public attributes of plain types (numbers, strings,
booleans and tuples of them), which for the built-in agents are their constructor arguments. The
interpreter, numpy, scipy and platform are recorded in `environment` but not hashed: results
agree across them within the reproducibility tolerances of [validation.md](validation.md), not
bit for bit. A report carries `combined_hash`, the hash of its runs' sorted hashes.

## Statistics

`vbar.eval.stats` follows Agarwal et al., "Deep Reinforcement Learning at the Edge of the
Statistical Precipice", NeurIPS 2021:

- `iqm(values)`: the interquartile mean, the mean after a quarter is trimmed from each end.
- `stratified_bootstrap(samples, statistic, reps=10_000, confidence=0.95, rng)`: `samples` maps
  each scenario to its runs; every resample draws as many runs as each scenario has, with
  replacement, within that scenario, so the spread between scenarios does not widen the interval.
  The interval is the percentile interval of the resampled statistic, taken as order statistics so
  infinite values stay values. Scenarios are taken in sorted order, so a mapping's order never
  changes a result, and a seed or generator makes it reproducible.
- `aggregate_iqm(samples)`: the IQM of every run of every scenario, pooled.
- `performance_profile(samples, taus)`: the fraction of runs scoring above each threshold, each
  scenario weighted equally.
- `probability_of_improvement(x, y)`: within each scenario P(X > Y) + P(X = Y) / 2 over every
  pair of runs (the Mann-Whitney U over n m), averaged over scenarios, with a stratified bootstrap
  interval.
- `paired_difference(x, y)`: the IQM of per-seed differences when both agents flew the same seeds
  (common random numbers), with an interval that resamples pairs, so the variation the two runs of
  a seed share cancels.

- `mcnemar(only_a, only_b)` (in `vbar.eval.coverage`): the exact McNemar test of two success
  rates on the same seeds, from the seeds where only one controller succeeded: twice the smaller
  tail of Binomial(only_a + only_b, 1/2).

`vbar.eval.coverage` checks that these intervals hold their level: it draws synthetic outcomes
with a known rate or mean, builds the interval a report would, and counts how often it covers the
truth over 1000 repeats (`rate_coverage`, `iqm_coverage`, `paired_coverage`, and
`mcnemar_false_positives` for two equal controllers). The table, by number of runs, is in
[validation.md](validation.md): the bootstrap intervals hold at 20, 50 and 100 runs per cell and
fall short at 5 (in the table's run also, narrowly, at 10), the exact rate intervals cover at least
95% at every number of runs from 1 to 200 on the rates tested, and McNemar's test calls equal
controllers different in at most 5% of repeats at 5, 10, 20, 50 and 100 runs.

The normal-approximation interval is never used: for rare events its lower end falls below zero
(3 failures in 10,000 runs give -3.9e-5; Kochenderfer, Wheeler and Wray, *Algorithms for Decision
Making*, 2022, section 14.1), where the Clopper-Pearson interval is (6.19e-5, 8.76e-4).

## Safety tails

`tail_summary(values, worse)` gives a safety metric's unsafe tail, the worst run, the 5th
percentile on the unsafe side and the mean beyond it (as `vbar.metrics.unsafe_tail`), with
distribution-free intervals (Kochenderfer, Katz, Corso and Moss, *Algorithms for Validation*,
2026, sections 7.1 and 12.2):

- `quantile_interval`: the pair of order statistics (l, u) with equal tails whose coverage of the
  5th percentile, P(l <= B < u) for B ~ Binomial(n, 0.05), is at least the confidence. With too
  few runs an end is infinite: 20 runs cannot bound the 5th percentile from below at 95%.
- `worst_bound`: a one-sided (95%, 95%) tolerance bound, the k-th worst run with k the largest
  rank for which P(Binomial(n, 0.05) >= k) >= 0.95 (`tolerance_order`). It is the 2nd worst of 100
  runs (confidence 0.963) and the 39th of 1000 (0.957); at least 59 runs are needed, the
  zero-failure count. It is NaN with fewer.
- `tail_mean_interval`: a percentile bootstrap of the tail mean.

Split-conformal bounds are not used: their coverage holds on average over calibration sets, and
given one set it falls below 95% with probability 0.436 at n = 100 and 0.480 at n = 1000
(`conformal_coverage_risk`). `compare_rates(successes_a, n_a, successes_b, n_b)` gives both
Clopper-Pearson intervals and calls an unpaired difference resolved only when it exceeds the wider
interval's width; with `require_resolved=True` an unresolved claim raises. 70 of 100 against 50 of
100 differ by 0.20, inside the 0.203 width at 50 of 100, so the claim is refused.

## Learned agents

`vbar.eval.learned` holds the rules for agents trained from data. Training runs are the unit:
`LearnedEntry` declares every run it launched (`TrainingRun`: seed, status, scored checkpoint,
environment steps, simulated and wall time, violations and propellant during training), at least
five. A run that diverged, saturated its cap or did not terminate stays in N as a failed run:
`run_scores` gives it the failure score and refuses scores for runs the entry did not declare, so a
rerun cannot replace it. The entry also declares the scored action mode (deterministic, or sampled
with fixed seeds), the discount as a time constant `tau_s` (gamma = exp(-dt / tau)), lambda, the
training-reward components and weights, the shaping potential, stabilisers, whether the
hyperparameters were shared (with the scenarios they were tuned on) or tuned per scenario, the
number of configurations tried (each reward variant counts), and for offline training the data
source. `normalized` maps scores per start state onto two references (0 and 1) before a
cross-scenario IQM, and `by_tuning` keeps shared and per-scenario entries apart.

## Reports and reproduction

`suite_report(suite, results)` first calls `Suite.require_complete`, then summarises each
scenario: success as a rate with its Clopper-Pearson interval, performance metrics as the IQM with
a bootstrap interval, safety metrics as their tails, never an IQM, plus the failure labels and the
count of errors. It refuses results that mix agents or filters or hold a run twice. `latest_results` keeps the last result of each run, which the `report` command applies, since a results file resumed after a commit holds each run once per revision. `to_markdown` renders the table.
From the command line:

```
python -m vbar.eval run --suite core-v1 --agent vbar.agents:DriftAgent --workers 8 --out drift.jsonl
python -m vbar.eval report drift.jsonl --markdown drift.md
python -m vbar.eval reproduce reproduce.toml
```

`reproduce.toml` lists every published table: its suite, agent and filter factories, and where its
results and table go. `reproduce` flies only what the results files lack and rewrites every table,
so one command regenerates all of them.

The notebook `examples/quickstart.ipynb`
([open in Colab](https://colab.research.google.com/github/jamilhammoudeh/vbar/blob/main/examples/quickstart.ipynb))
installs the package from PyPI, flies the PD baseline, prints its metrics and rebuilds a table of
two episodes. The command line knows only the built-in suites, so the notebook registers a small
suite with `register_suite` and calls `vbar.eval.__main__.main(["reproduce", "quickstart.toml"])`,
the entry point of `python -m vbar.eval`, with a one-table manifest. Running that cell again flies
nothing, since the results file already holds both episodes.

## Inputs

The statistics raise `ValueError` for a NaN score (the message names the scenario where there is
one), a NaN `performance_profile` threshold, a bootstrap statistic that is NaN on the data or on a
resample, and two paired scores that are the same infinity; infinite scores are allowed. Counts
(`reps`, `n`, `repeats`, the counts of `mcnemar`, seeds, a `Job`'s index, `configurations_tried`, a
training run's steps and violations) must be integers in range: numpy integers pass, floats (even
`100.0`) and bools do not. `compare_rates` and `clopper_pearson` are the exception: they read a
whole float such as `3.0` as its integer. `confidence`, `level`, `coverage` and `alpha` lie in (0, 1), `p` in
[0, 1], and `widen` is finite and positive. A result line needs every schema field, an object for
`metrics` and a seed that is decimal text or a JSON integer of at least 0 (null for the test split),
or `read_results` raises `ValueError` naming the line. A `Job` takes integer seeds, and `encode`
writes them from integers only; decimal text is read by `decode` but not accepted by `encode`.
On the test split `encode` takes a seed that is None or a whole number of at least 0 and refuses
any other, since it could not search for it.

## Assumptions and limits

- Wall-clock metrics (latency, compute, filter time) differ between repeats of a run;
  `reproducible(result)` leaves them out when two runs are compared.
- The provenance hash names the code by its git revision and, for a tree with local changes, by a
  digest of the changes to tracked files, read once per process, so a resumed run is served only
  results flown with the same edits. Untracked files are not covered, an installed package is
  named by its version alone, and where git is missing or fails on a checkout the revision is
  unknown and every edit hashes the same.
- Trajectories are stored as `.npz`; Parquet output is not provided.

## Validation

`tests/test_eval_runner.py`: serial and parallel results agree; an interrupted run resumes
without flying a run twice; a crashing agent is recorded, not dropped; the provenance hash changes
with every field it covers; 128-bit seeds and non-finite metrics survive the file.
`tests/test_eval_hidden_split.py`: the digest against a standard-library HMAC and fixed vectors,
test split files holding no seed or key in decimal or hexadecimal, the private log, resuming, and
complete, partial and altered reports from published lines.
`tests/test_eval_stats.py`, `tests/test_eval_tails.py`: the IQM and the stratified bootstrap
against hand cases and their invariances, the Mann-Whitney identity of the probability of
improvement, the order-statistic and tolerance ranks and their coverage by simulation, the
Clopper-Pearson cases of Kochenderfer et al. section 14.1, the conformal risk.
`tests/test_eval_coverage.py`: the coverage table's cells at 50 and 100 runs and a narrowed
interval failing it. `tests/test_eval_documented_figures.py`: the other figures this page and
validation.md quote: the bootstrap coverage at 5 and 20 runs, McNemar's false-positive rate from 5
to 100 runs, the exact rate interval's coverage from 1 to 200 runs, the documented defaults and the
safety-tail figures.
`tests/test_eval_report.py`: complete reports, refusals, the command line and the manifest.
`tests/test_eval_learned.py`: the learned-agent rules. `tests/test_eval_domain.py`: the toy domain.
Each module's `test_hostile_*` test runs the refusals of the Inputs section.
`tests/test_quickstart_notebook.py`: the notebook's cells run headless without the install cell,
the notebook holds no stored outputs, its install cell is the unpinned package name, and a second
run of the table cell flies nothing. `tests/test_quickstart_links.py`: the README and this page
link the notebook, and the notebook's own badge points at it.
