# Composite score and failure label cards

Definitions in full: [docs/metrics.md](../../../../docs/metrics.md). Tests:
`tests/test_metrics_composite.py` and `tests/test_metrics_taxonomy.py`.

## C1 composite_score (v1)
- Definition: completion / (1 + sum_j w_j n_j), completion 1 on success else M5; n_j one per contiguous violation interval and one per timeout; default weights keep-out 0.7, cone 0.5, speed and closing speed 0.3, timeout 0.4.
- Summary: interquartile mean across episodes.
- Why it matters: one number that combines completion and infractions, never reported without its breakdown.
- Known-answer tests: 1 for a clean success; the flyby's progress over 2.1; the formula with custom weights.
- Convergence: the infraction counts come from continuous-time intervals; the same at 20, 10 and 5 s steps (`tests/test_metrics_gate.py`).
- Invariance: mirror; unchanged when all lengths double.
- Mutation detection: never rises with position or velocity bias, and falls at the largest (`tests/test_metrics_mutants.py`).
- Anti-gaming: a long violation counts once but a violation that is left and re-entered counts each time, so repeated grazing costs more.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: no spread over 40 runs (all 1); the spec's start box: SD 0.0012 about a mean of 0.5826, under 2 episodes per group for a 10% change.
- Validity: the weights are a convention, not a measurement. Sensitivity check: the PD baseline, PD through the target and free drift keep their order under each weight halved and raised by half (ten changes); two agents built to swap under a halved keep-out weight are reported as unstable.
- Changelog: v1, first version.

## C1c completion (v1)
- Definition: 1 on success, else M5.
- Summary: interquartile mean across episodes.
- Why it matters: the numerator of C1.
- Known-answer tests: 1 for a clean success.
- Convergence: as M5.
- Invariance: as M5.
- Mutation detection: as M5.
- Anti-gaming: as M5.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: no spread over 40 runs (all 1); the spec's start box: SD 0.00204 about a mean of 0.9903, under 2 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## C1n infractions (v1)
- Definition: counts per constraint of contiguous violation intervals, and one for a timeout without success.
- Summary: not summarised across episodes (a label, a record or already an aggregate).
- Why it matters: the breakdown C1 is reported with.
- Known-answer tests: one keep-out entry and one timeout for the flyby; a relative orbit centred on the target counts each entry into the sphere as the closed form does.
- Convergence: continuous-time intervals merged across steps.
- Invariance: mirror.
- Mutation detection: one keep-out entry when PD flies through the target.
- Anti-gaming: unweighted constraints are still counted, so soft breaches stay visible.
- Reproducibility: exact (counts).
- Power: not applicable.
- Validity: not applicable.
- Changelog: v1, first version.

## T1 failure_label (v1)
- Definition: for a failed run, the first matching label of filter_defect, filter_infeasible, keep_out_entry, speed_limit_breach, cone_violation, propellant_exhaustion, parse_failure, sign_error, oscillation, contact_out_of_tolerance, hold_not_maintained, overshoot, timeout_near, timeout_far, other; None on success.
- Summary: not summarised across episodes (a label, a record or already an aggregate).
- Why it matters: failures differ in kind, and a rate alone does not say which.
- Known-answer tests: one constructed run per label reaches it through the order.
- Convergence: inherits the metrics it reads.
- Invariance: mirror (labels are exact in the golden values).
- Mutation detection: unreadable replies, a failed sign check and a command that flips every step each get their own label.
- Anti-gaming: first match wins, so a run that both enters the keep-out zone and times out is labelled by the keep-out entry.
- Reproducibility: exact (label).
- Power: not applicable.
- Validity: not yet measured.
- Changelog: v1, first version.
