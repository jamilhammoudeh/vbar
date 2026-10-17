# Agent metric cards

Definitions in full: [docs/metrics.md](../../../../docs/metrics.md). Tests: `tests/test_metrics_agent.py`.
Wall-clock figures are not reproducible by nature and are left out of the golden values.

## A1 latency_median (v1)
- Definition: median wall time of the agent's decisions (s), filter time excluded.
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

## A6 sign_check_pass (v1)
- Definition: the result of the convention self-check, as reported.
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
