# Safety-filter metric cards

Definitions in full: [docs/metrics.md](../../../../docs/metrics.md). Tests: `tests/test_metrics_filter.py`.
An episode without a filter gives NaN. The only built-in filter so far is the pass-through, so
power figures need real filters and are not given.

## F1 intervention_rate (v2)
- Definition: decisions where a safety row was active (as the filter reports it, or a change that was not only actuator clipping) over decisions.
- Summary: interquartile mean across episodes.
- Why it matters: how often the agent needed correcting.
- Known-answer tests: 1 for a filter that halves every command; 0 for the pass-through; 0 for a filter that only clips; 1 for a filter that reports an active row without moving the action.
- Convergence: per decision, so it depends on the step by definition.
- Invariance: not applicable.
- Mutation detection: 0 against 1 for pass-through against halving (`tests/test_metrics_mutants.py`).
- Anti-gaming: actuator clipping does not count, since the spacecraft clips anyway (Ames et al., control barrier function survey); a bare action is judged by whether any component moved by more than 1e-6.
- Reproducibility: exact (ratio of counts).
- Power: not given.
- Validity: not yet measured.
- Changelog: v1, any change above 1e-6; v2, safety-row activity, with clipping counted in F1c.

## F1c clip_rate (v1)
- Definition: decisions whose change was only actuator clipping, over decisions.
- Summary: interquartile mean across episodes.
- Why it matters: keeps clipping visible without counting it as a safety intervention.
- Known-answer tests: 1 for a filter that reports every change as clipping.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: as F1.
- Reproducibility: exact (ratio of counts).
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## F2 intervention_size (v2)
- Definition: mean of |filtered - command|_2 over decisions with an active safety row, in m/s^2 at the initial mass.
- Summary: interquartile mean across episodes.
- Why it matters: a filter that nudges and one that overrides can have the same F1.
- Known-answer tests: the mean half-command size in m/s^2 for the halving filter, in normalized, acceleration and thrust action modes; NaN with no interventions.
- Convergence: per decision.
- Invariance: in m/s^2, so it is the same in every action mode.
- Mutation detection: positive for halving, NaN for the pass-through.
- Anti-gaming: averaged over decisions with an active row only, so many tiny changes cannot dilute a few large ones.
- Reproducibility: 1e-9 relative.
- Power: not given.
- Validity: not yet measured.
- Changelog: v1, fractions of a full command over sqrt(3); v2, m/s^2, with the normalised form in F2n.

## F2n intervention_size_normalized (v1)
- Definition: F2 over the widest command change the actuator allows, from a full command to its opposite: 2 sqrt(3) a_max for box limits, 2 a_max for a ball; in [0, 1].
- Summary: interquartile mean across episodes.
- Why it matters: compares filters across actuators; sqrt(3) alone under-states a box change, which can reach 2 sqrt(3) a_max, and over-states a ball's.
- Known-answer tests: F2 over 2 sqrt(3) x 0.02 m/s^2 for box limits and over 2 x 0.02 m/s^2 for a ball.
- Convergence: per decision.
- Invariance: the same in every action mode.
- Mutation detection: as F2.
- Anti-gaming: as F2.
- Reproducibility: 1e-9 relative.
- Power: not given.
- Validity: not yet measured.
- Changelog: v1, first version.

## F3 safety_dv_cost (v1)
- Definition: per seed, E1 with the filter minus E1 without; reported as the interquartile mean with the per-seed values.
- Summary: interquartile mean across episodes.
- Why it matters: the fuel price of safety.
- Known-answer tests: equals the trimmed mean of per-seed differences over 5 seeds in any order of the runs; unpaired runs are rejected.
- Convergence: inherits E1's.
- Invariance: run order.
- Mutation detection: a filter that wastes fuel raises it.
- Anti-gaming: pairing by seed removes start-state luck.
- Reproducibility: 1e-9 relative.
- Power: not given.
- Validity: not yet measured.
- Changelog: v1, first version.

## F4 filter_dependence (v1)
- Definition: success rate with the filter minus without, paired by seed.
- Summary: not summarised across episodes (a label, a record or already an aggregate).
- Why it matters: how much of an agent's success the filter supplies.
- Known-answer tests: within [-1, 1] on the paired PD runs.
- Convergence: inherits M1's.
- Invariance: run order.
- Mutation detection: not exercised (the pass-through cannot change success).
- Anti-gaming: paired by seed.
- Reproducibility: exact.
- Power: not given.
- Validity: not yet measured.
- Changelog: v1, first version.

## F5 infeasible_steps (v1)
- Definition: decisions outside the commit zone with status infeasible_certified.
- Summary: the unsafe tail across episodes (highest value, 95th percentile, mean at or above it), never the interquartile mean.
- Why it matters: when no safe action exists, the filter must prove it (Boyd and Vandenberghe, sections 5.8 and 11.4); a stalled solver proves nothing.
- Known-answer tests: 4 of 8 scripted decisions; none counted inside a commit zone.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: drives the filter_infeasible label.
- Anti-gaming: solver failures are counted apart, so a broken filter cannot pass its failures off as infeasibility.
- Reproducibility: exact.
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## F5b solver_failures (v1)
- Definition: decisions outside the commit zone with status solver_failed.
- Summary: the unsafe tail across episodes (highest value, 95th percentile, mean at or above it), never the interquartile mean.
- Why it matters: a filter implementation defect, not an agent failure.
- Known-answer tests: 2 of 8 scripted decisions; 4 of 4, and none inside a 500 m commit zone.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: drives the filter_defect label.
- Anti-gaming: as F5.
- Reproducibility: exact.
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## F6 filter_compute_median (v1)
- Definition: median of the filter's compute time per decision (s).
- Summary: interquartile mean across episodes.
- Why it matters: a filter must run within the control cycle.
- Known-answer tests: 0.003 s for scripted times 0.001 to 0.004 and 0.1 s.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: measured around the filter call when the filter does not report it.
- Reproducibility: wall clock, so not reproducible by nature.
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## F6p filter_compute_p95 (v1)
- Definition: 95th percentile of the filter's compute time (s).
- Summary: interquartile mean across episodes.
- Why it matters: the tail decides whether a cycle is missed.
- Known-answer tests: numpy's 95th percentile of the scripted times.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: as F6.
- Reproducibility: wall clock.
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.
