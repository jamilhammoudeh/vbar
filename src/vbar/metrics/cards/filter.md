# Safety-filter metric cards

Definitions in full: [docs/metrics.md](../../../../docs/metrics.md). Tests: `tests/test_metrics_filter.py`.
An episode without a filter gives NaN. The only built-in filter so far is the pass-through, so
power figures need real filters and are not given.

## F1 intervention_rate (v2)
- Definition: decisions where a safety row was active over decisions: the filter reports an active row, or the acceleration the actuator flies changed (command and filtered action both projected onto the thrust limit).
- Summary: interquartile mean across episodes.
- Why it matters: how often the agent needed correcting.
- Known-answer tests: 1 for a filter that halves every command; 0 for the pass-through; 0 for a filter that clips a command of 3 to the box limit; 1 for a filter that reports an active row without moving the action; 1 for a filter that zeroes every command while reporting no active row.
- Convergence: per decision, so it depends on the step by definition.
- Invariance: not applicable.
- Mutation detection: 0 against 1 for pass-through against halving (`tests/test_metrics_mutants.py`).
- Anti-gaming: judged from the recorded command and action, not from the filter's own flags, so a filter cannot hide a change by reporting no active row; clipping does not count, since the actuator clips anyway (Ames et al., control barrier function survey); a reported row only adds decisions; changes below 1e-6 of a full command are rounding.
- Reproducibility: exact (ratio of counts).
- Power: not given.
- Validity: not yet measured.
- Changelog: v1, any change above 1e-6; v2, a reported row or a change in flown acceleration, with clipping counted in F1c.

## F1c clip_rate (v1)
- Definition: decisions whose action changed but whose flown acceleration did not (clipping to the thrust limit only), over decisions.
- Summary: interquartile mean across episodes.
- Why it matters: keeps clipping visible without counting it as a safety intervention.
- Known-answer tests: 1 for a filter that clips a command of 3 to the box limit.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: as F1.
- Reproducibility: exact (ratio of counts).
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## F2 intervention_size (v2)
- Definition: mean change in flown acceleration, |a(filtered) - a(command)|_2 with both projected onto the thrust limit, over decisions with an active safety row, in m/s^2 at the initial mass.
- Summary: interquartile mean across episodes.
- Why it matters: a filter that nudges and one that overrides can have the same F1.
- Known-answer tests: half of a constant command inside the limit, in m/s^2, in acceleration and thrust action modes and for box and ball limits; 0.25 x 0.02 m/s^2 for halving 1.5 times the box limit, which flies 0.75 instead of 1; NaN with no interventions.
- Convergence: per decision.
- Invariance: in m/s^2, so it is the same in every action mode.
- Mutation detection: positive for halving, NaN for the pass-through.
- Anti-gaming: averaged over decisions with an active row only, so many tiny changes cannot dilute a few large ones; the part of a command beyond the thrust limit is never counted, since it is never flown.
- Reproducibility: 1e-9 relative.
- Power: not given.
- Validity: not yet measured.
- Changelog: v1, fractions of a full command over sqrt(3); v2, change in flown acceleration in m/s^2, with the normalised form in F2n.

## F2n intervention_size_normalized (v1)
- Definition: F2 over the widest command change the actuator allows, from a full command to its opposite: 2 sqrt(3) a_max for box limits, 2 a_max for a ball; in [0, 1].
- Summary: interquartile mean across episodes.
- Why it matters: compares filters across actuators; sqrt(3) alone under-states a box change, which can reach 2 sqrt(3) a_max, and over-states a ball's.
- Known-answer tests: F2 over 2 sqrt(3) x 0.02 m/s^2 for box limits and over 2 x 0.02 m/s^2 for a ball; exactly 1 for reversing a command ten times the box limit on every axis.
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

## F7 filter_conservatism (v1)
- Definition: share of a declared region's viable states (a reference such as the exact braking set) that the filter's safe set leaves out; from `filter_conservatism(in_filter_set, in_viable_set, states)`, not per episode.
- Summary: not summarised across episodes (a label, a record or already an aggregate).
- Why it matters: constructions for bounded thrust shrink a barrier's safe set below what is viable, and a filter that is safe by refusing to move is not useful (Ames et al., control barrier function survey, section III).
- Known-answer tests: 40/101 when a filter keeps x <= 0.6 m of a viable x <= 1 m on a 201-point grid; NaN with no viable sample.
- Convergence: converges as the region's sampling is refined.
- Invariance: not applicable.
- Mutation detection: not exercised; no shipped filter has a safe set.
- Anti-gaming: measured against a reference set, not the filter's own claim.
- Reproducibility: exact (ratio of counts).
- Power: not given.
- Validity: not yet measured.
- Changelog: v1, first version.

## F8 unnecessary_intervention_rate (v1)
- Definition: share of decisions with an active safety row whose own command, held for the step and then left to free drift over the drift horizon, stays clear of the collision radius.
- Summary: interquartile mean across episodes.
- Why it matters: a filter that intervenes when the agent was safe costs fuel and progress for nothing.
- Known-answer tests: 0 when zeroing full thrust 100 m behind the target (the held command coasts to 45 m, inside 50 m), equal to closed-form escape arcs at every decision; 1 for a tenth of that from 400 m.
- Convergence: the escape arcs agree at 50 and 800 drift samples per orbit, as for S4.
- Invariance: mirror across the orbit plane.
- Mutation detection: not exercised; no shipped filter intervenes.
- Anti-gaming: judged on the agent's own command, not on what the filter returned.
- Reproducibility: exact (ratio of counts).
- Power: not given.
- Validity: not yet measured.
- Changelog: v1, first version.

## F9 reentry_time (v1)
- Definition: mean length (s) of the stretches of decisions where the filter reported `recovering` (the state already outside its safe set); infinite if the episode ends inside one; 0 if it never did; NaN if it does not report it.
- Summary: the unsafe tail across episodes (highest value, 95th percentile, mean at or above it), never the interquartile mean.
- Why it matters: navigation or execution error, or a start outside the safe set, puts the state where the barrier is negative; how fast the filter gets back matters.
- Known-answer tests: 20 s for stretches of 3 and 1 decisions of 10 s; infinite when still recovering at the end; NaN for a filter that does not report it.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: not exercised; no shipped filter reports it.
- Anti-gaming: taken from the filter's own flag.
- Reproducibility: exact.
- Power: not given.
- Validity: not yet measured.
- Changelog: v1, first version.

## F10 stall_time (v1)
- Definition: time (s) with an active safety row while the chaser moves slower than `stall_speed` (0.01 m/s, ours).
- Summary: the unsafe tail across episodes (highest value, 95th percentile, mean at or above it), never the interquartile mean.
- Why it matters: with the target between the chaser and its goal, a barrier's push and the agent's pull can cancel and hold the chaser in place for good; a filter that is safe by never arriving is not useful.
- Known-answer tests: 60 s for a filter that zeroes every command from rest over six 10 s steps; 0 when the commands pass and the chaser moves.
- Convergence: per decision.
- Invariance: mirror across the orbit plane.
- Mutation detection: not exercised; no shipped filter stalls.
- Anti-gaming: counts only time with an active row, so a planned hold does not count.
- Reproducibility: exact.
- Power: not given.
- Validity: not yet measured.
- Changelog: v1, first version.

## F5f fallback_count (v1)
- Definition: decisions answered by a fallback (`fallback_level` > 0: the shifted previous plan or the declared backup), over the decisions that report a level; NaN if none does.
- Summary: the unsafe tail across episodes (highest value, 95th percentile, mean at or above it), never the interquartile mean.
- Why it matters: an optimisation-based filter that times out must fall back to something safe; how often it does is a measure of the filter, not the agent (Rawlings, Mayne and Diehl, Algorithm 6.1).
- Known-answer tests: 2 of 4 for a scripted sequence 0, 1, 0, 2.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: not exercised.
- Anti-gaming: reported next to F5b, so a filter's failures are not hidden as fallbacks.
- Reproducibility: exact.
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## F5l fallback_level_max (v1)
- Definition: the deepest fallback used (0 the converged solution, 1 the shifted previous plan, 2 the declared backup); NaN if no decision reports a level.
- Summary: the unsafe tail across episodes (highest value, 95th percentile, mean at or above it), never the interquartile mean.
- Why it matters: the declared backup is the last line; reaching it says the filter ran out of better answers.
- Known-answer tests: 2 for the scripted sequence 0, 1, 0, 2.
- Convergence: per decision.
- Invariance: not applicable.
- Mutation detection: not exercised.
- Anti-gaming: as F5f.
- Reproducibility: exact.
- Power: not given.
- Validity: not applicable.
- Changelog: v1, first version.

## F11 filter_certificates (v1)
- Definition: the kinds of guarantee the filter reported over the episode (for example explicit, implicit), sorted.
- Summary: not summarised across episodes (a label, a record or already an aggregate).
- Why it matters: an implicit (backup) filter certifies only up to its backup horizon unless that ends in an invariant set (Rawlings, Mayne and Diehl, section 2.7).
- Known-answer tests: ("implicit",) for a filter that reports it.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: as reported by the filter.
- Reproducibility: exact (labels).
- Power: not applicable.
- Validity: not applicable.
- Changelog: v1, first version.

## F11h certificate_horizon_min (v1)
- Definition: the shortest certificate horizon (s) the filter reported; infinite for an invariant set; NaN if not reported.
- Summary: the unsafe tail across episodes (lowest value, 5th percentile, mean at or below it), never the interquartile mean.
- Why it matters: a finite horizon means safety is proven only that far ahead.
- Known-answer tests: 450 s for horizons 600 and 450 s.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: as reported by the filter.
- Reproducibility: exact.
- Power: not applicable.
- Validity: not applicable.
- Changelog: v1, first version.
