# Agent metric cards

Definitions in full: [docs/metrics.md](../../../../docs/metrics.md). Tests: `tests/test_metrics_agent.py`.
Wall-clock figures are not reproducible by nature and are left out of the golden values.

## A1 latency_median (v1)
- Definition: median wall time of the agent's decisions (s), filter time excluded.
- Summary: interquartile mean across episodes.
- Why it matters: whether the agent fits a control cycle.
- Known-answer tests: 0.3 s for set latencies 0.1, 0.2, 0.3, 0.4 and 2.0 s.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: measured by the recorder around `act`, not reported by the agent.
- Reproducibility: wall clock.
- Power: starts behind the target: SD 7.59e-06 s about a mean of 3.145e-05 s, 92 episodes per group for a 10% change; the spec's start box: SD 9.59e-06 s about a mean of 3.192e-05 s, 142 episodes per group for a 10% change.
- Validity: not applicable.
- Changelog: v1, first version.

## A1p latency_p95 (v1)
- Definition: 95th percentile of the decision wall time (s).
- Summary: interquartile mean across episodes.
- Why it matters: the tail decides missed cycles.
- Known-answer tests: numpy's 95th percentile of the set latencies.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: as A1.
- Reproducibility: wall clock.
- Power: starts behind the target: SD 1.19e-05 s about a mean of 6.316e-05 s, 56 episodes per group for a 10% change; the spec's start box: SD 2.18e-05 s about a mean of 6.602e-05 s, 172 episodes per group for a 10% change.
- Validity: not applicable.
- Changelog: v1, first version.

## A2 compute (v1)
- Definition: total agent wall time over the episode (s).
- Summary: interquartile mean across episodes.
- Why it matters: cost of running the agent.
- Known-answer tests: 3.0 s for the set latencies.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: as A1.
- Reproducibility: wall clock.
- Power: starts behind the target: SD 0.000479 s about a mean of 0.002228 s, 73 episodes per group for a 10% change; the spec's start box: SD 0.000908 s about a mean of 0.002619 s, 189 episodes per group for a 10% change.
- Validity: not applicable.
- Changelog: v1, first version.

## A3i input_tokens (v1)
- Definition: input tokens summed over the agent's log; NaN unless every entry has a count.
- Summary: interquartile mean across episodes.
- Why it matters: language-model cost and context use.
- Known-answer tests: 4000 for four entries of 1000.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: all entries or none, so a partial log cannot under-report.
- Reproducibility: exact for cached replies.
- Power: no token-logging agent in the suite yet.
- Validity: as reported by the model provider.
- Changelog: v1, first version.

## A3o output_tokens (v1)
- Definition: output tokens summed over the agent's log; NaN unless every entry has a count.
- Summary: interquartile mean across episodes.
- Why it matters: as A3i.
- Known-answer tests: 200 for four entries of 50.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: as A3i.
- Reproducibility: exact for cached replies.
- Power: no token-logging agent in the suite yet.
- Validity: as reported by the model provider.
- Changelog: v1, first version.

## A3 token_cost (v1)
- Definition: tokens times a declared price in USD per million input and output tokens; NaN without a price.
- Summary: interquartile mean across episodes.
- Why it matters: comparable cost across models.
- Known-answer tests: (4000 x 3 + 200 x 15) / 1e6 USD.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: the price table is declared with the result, not inferred.
- Reproducibility: exact for cached replies.
- Power: no token-logging agent in the suite yet.
- Validity: as good as the declared prices.
- Changelog: v1, first version.

## A4 parse_failure_rate (v1)
- Definition: share of log entries whose reply could not be parsed into an action.
- Summary: interquartile mean across episodes.
- Why it matters: a language-model agent that cannot answer in format is coasting.
- Known-answer tests: 0.5 for alternating unreadable and valid replies.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: drives the parse_failure label (all replies unreadable, `tests/test_metrics_taxonomy.py`).
- Anti-gaming: the default action on a failure is to coast, which is counted here rather than scored as a choice.
- Reproducibility: exact for cached replies.
- Power: no language-model agent in the baseline runs.
- Validity: not applicable.
- Changelog: v1, first version.

## A5 reasoning_faithfulness (v1)
- Definition: share of decisions whose stated belief (range within 10%, closing flag) matches the observation the agent received.
- Summary: interquartile mean across episodes.
- Why it matters: whether the stated reasoning is consistent with the agent's own inputs.
- Known-answer tests: 1 for an agent that restates range and closing from its prompt, 0 for one that doubles the range; under 50 m navigation noise the honest agent still scores 1 although its beliefs are off the truth.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: the doubled-range agent scores 0.
- Anti-gaming: judged against the observation, not the truth, so an agent cannot be faulted for sensor noise or rewarded for guessing the truth.
- Reproducibility: exact for cached replies.
- Power: no language-model agent in the baseline runs.
- Validity: measures consistency, not correctness.
- Changelog: v1, first version.

## A5b belief_consistency (v1)
- Definition: root mean square of (stated range - posterior range) / posterior range sigma over decisions with a stated belief, against a reference navigation posterior built from the same observations (truth-free); sigma = sqrt(u' P u) with u the unit posterior mean; NaN without a posterior.
- Summary: interquartile mean across episodes.
- Why it matters: under navigation noise the rational belief is the filter posterior given the observations, not the latest fix (Kochenderfer et al., Algorithms for Decision Making, 2022, sections 19.3 and 20.1); an agent that restates each noisy fix scores A5 = 1 while being badly calibrated.
- Known-answer tests: sqrt(5) for stated ranges 2, -2, 6 and -6 m off a posterior with 2 m sigma; at rest on the V-bar with 50 m fixes, an agent that restates them scores A5 = 1 and A5b above 1, one that states the mean of its fixes below 1e-3; 1 for a 10 m error where the covariance along the line of sight is 100 m^2 and 4 and 9 m^2 across it; a posterior with no range variance or with its mean at the target is rejected.
- Convergence: per decision.
- Invariance: mirror (range and covariance are unchanged).
- Mutation detection: the restating agent scores above 1.
- Anti-gaming: judged against a posterior from the agent's own observations, so it neither rewards guessing the truth nor faults sensor noise; only decisions with a stated belief count, so an agent that states one belief and then none is judged on that one, and A5b is read with A5, which counts the missing beliefs; a log with more entries than decisions is rejected.
- Reproducibility: exact for cached replies and a deterministic posterior.
- Power: not given.
- Validity: computed only where a reference posterior is supplied; the shipped navigation models have none yet.
- Changelog: v1, first version.

## A6 sign_check_pass (v1)
- Definition: the result of the convention self-check, as reported.
- Summary: a rate across episodes, with its Clopper-Pearson interval (resolution table in README.md).
- Why it matters: frame and sign mistakes waste a full evaluation.
- Known-answer tests: passed through; False drives the sign_error label.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: a reported failure labels the run sign_error.
- Anti-gaming: reported by the harness that ran the check.
- Reproducibility: exact.
- Power: not applicable.
- Validity: not applicable.
- Changelog: v1, first version.

## A7 training_cost (v1)
- Definition: env steps, simulated time, wall clock and compute of training, as reported; all four keys required.
- Summary: not summarised across episodes (a label, a record or already an aggregate).
- Why it matters: learned agents' results mean little without their training budget.
- Known-answer tests: passed through; incomplete figures are rejected.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: required keys, so a budget cannot be reported in part.
- Reproducibility: as reported.
- Power: not applicable.
- Validity: as reported.
- Changelog: v1, first version.

## A8 cost_of_learning (v1)
- Definition: constraint violations and propellant during training and the area under the learning curve, as reported; all three keys required.
- Summary: not summarised across episodes (a label, a record or already an aggregate).
- Why it matters: a policy that learned by crashing hundreds of times is a different result.
- Known-answer tests: a report missing `propellant_kg` is rejected.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: as A7.
- Reproducibility: as reported.
- Power: not applicable.
- Validity: as reported.
- Changelog: v1, first version.

## A1r preparation_median (v1)
- Definition: median of the log's `prep_s` (s), the work done before the observation arrives, over the entries that report it; NaN if none does.
- Summary: interquartile mean across episodes.
- Why it matters: real-time iteration schemes split each decision so that only the feedback part delays the command (Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed., 2017, section 8.9); A1 alone mixes the two.
- Known-answer tests: 0.02 s for 0.02, 0.03 and 0.01 s.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: not exercised; no shipped agent reports it.
- Anti-gaming: as reported by the agent; A1 still measures the whole decision.
- Reproducibility: machine dependent.
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## A1f feedback_median (v1)
- Definition: median of the log's `feedback_s` (s), the time from the observation to the command, over the entries that report it; NaN if none does.
- Summary: interquartile mean across episodes.
- Why it matters: the feedback delay is what the loop sees; real-time iteration schemes split each decision so that only the feedback part delays the command (Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed., 2017, section 8.9).
- Known-answer tests: 0.002 s for 0.001, 0.002 and 0.004 s, the fourth entry not reporting it.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: not exercised.
- Anti-gaming: as A1r.
- Reproducibility: machine dependent.
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## A1fp feedback_p95 (v1)
- Definition: 95th percentile of `feedback_s` (s), as A1f.
- Summary: interquartile mean across episodes.
- Why it matters: a rare long feedback delay can miss the step.
- Known-answer tests: the 95th percentile of 0.001, 0.002 and 0.004 s.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: not exercised.
- Anti-gaming: as A1r.
- Reproducibility: machine dependent.
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## A9 solver_metadata (v1)
- Definition: the solvers and tolerances the log reports, sorted, and the count of each solver status.
- Summary: not summarised across episodes (a label, a record or already an aggregate).
- Why it matters: latency depends on the solver: active-set and SQP methods warm-start better than interior-point methods, and a fixed-barrier interior-point method solves a more conservative problem (Rawlings, Mayne and Diehl, section 8.7), so results are compared with the solver named.
- Known-answer tests: ("ipopt", "osqp"), (1e-8, 1e-6) and {max_iter: 1, solved: 3} for a scripted log; empty for an agent that reports none.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: as reported by the agent.
- Reproducibility: exact (labels).
- Power: not applicable.
- Validity: not applicable.
- Changelog: v1, first version.

## A9i solver_iterations_median (v1)
- Definition: median of the log's `iterations` over the entries that report it; NaN if none does.
- Summary: interquartile mean across episodes.
- Why it matters: iterations, unlike wall time, compare across machines.
- Known-answer tests: 10 for 12, 5, 40 and 8.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: not exercised.
- Anti-gaming: as reported by the agent.
- Reproducibility: exact.
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## A9w warm_start_rate (v1)
- Definition: share of the entries reporting `warm_start` that were warm-started.
- Summary: interquartile mean across episodes.
- Why it matters: a warm start from the shifted previous solution is the main lever on latency for a receding-horizon agent.
- Known-answer tests: 0.75 for three warm starts in four.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: not exercised.
- Anti-gaming: as reported by the agent.
- Reproducibility: exact (ratio of counts).
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## A9f agent_fallback_count (v1)
- Definition: decisions the log marks `fallback`, the agent's own (not the filter's, F5f) backup answer; NaN if no entry reports it.
- Summary: the unsafe tail across episodes (highest value, 95th percentile, mean at or above it), never the interquartile mean.
- Why it matters: an agent whose solver often fails and falls back is not the controller its nominal design describes.
- Known-answer tests: 1 for one fallback in four.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: not exercised.
- Anti-gaming: reported apart from the filter's fallbacks, so neither hides the other.
- Reproducibility: exact.
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.
