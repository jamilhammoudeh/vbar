# Efficiency metric cards

Definitions in full: [docs/metrics.md](../../../../docs/metrics.md). Tests:
`tests/test_metrics_efficiency.py` unless named. Power as in the mission cards.

## E1 delta_v (v1)
- Definition: sum over executed segments of |a|_2 tau (m/s), as the actuator integrated it.
- Summary: interquartile mean across episodes.
- Why it matters: fuel for one steerable engine.
- Known-answer tests: a half-command burn on two axes for 30 s gives 0.01 sqrt(2) 30 m/s; a constant-thrust burn gives Isp g0 ln(m0 / m1) (1e-12).
- Convergence: the same at 10 and 2.5 s steps (1e-12); same trajectory at 20, 10 and 5 s (`tests/test_metrics_gate.py`).
- Invariance: unchanged when a ball thruster's command is rotated; doubles with all lengths and the thrust.
- Mutation detection: rises in order with decision delay (1, 3, 6 steps) and with dither at 0.05, 0.15, 0.45 of one noise sequence (`tests/test_metrics_mutants.py`).
- Anti-gaming: reads executed, not commanded, acceleration, so a filter or delay cannot hide thrust; a half-step delay moves the burn but not its cost.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: SD 0.844 m/s about a mean of 2.533 m/s, 175 episodes per group for a 10% change; the spec's start box: SD 0.747 m/s about a mean of 4.102 m/s, 53 episodes per group for a 10% change.
- Validity: equals the actuator's own propellant accounting through E2.
- Changelog: v1, first version.

## E1b delta_v_axes (v1)
- Definition: the same with |a|_1 (m/s), the cost of a thruster pair per axis.
- Summary: interquartile mean across episodes.
- Why it matters: with box limits the 2-norm under-counts fuel by up to sqrt(3).
- Known-answer tests: 0.6 m/s for the half-command burn on two axes for 30 s.
- Convergence: as E1.
- Invariance: not rotation invariant by design (per-axis thrusters).
- Mutation detection: as E1.
- Anti-gaming: the optimality gap of a box actuator must use this norm.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: SD 1.06 m/s about a mean of 3.38 m/s, 154 episodes per group for a 10% change; the spec's start box: SD 0.948 m/s about a mean of 5.308 m/s, 51 episodes per group for a 10% change.
- Validity: E1 <= E1b <= sqrt(3) E1 on the PD baseline.
- Changelog: v1, first version.

## E2 propellant_mass (v1)
- Definition: m0 (1 - exp(-dv / (Isp g0))) with the layout's delta-v (kg).
- Summary: interquartile mean across episodes.
- Why it matters: what the tank actually loses.
- Known-answer tests: 500 (1 - exp(-0.6 / (220 g0))) kg for the box burn; equals the actuator's per-burn sum to 1e-12; ball thrusters spend on the 2-norm.
- Convergence: as E1.
- Invariance: as E1 and E1b.
- Mutation detection: as E1.
- Anti-gaming: computed from executed delta-v.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: SD 0.244 kg about a mean of 0.7827 kg, 153 episodes per group for a 10% change; the spec's start box: SD 0.219 kg about a mean of 1.229 kg, 50 episodes per group for a 10% change.
- Validity: matches the spacecraft model's mass bookkeeping.
- Changelog: v1, first version.

## E3 optimality_gap (v1)
- Definition: layout delta-v over a lower bound certified for the episode's dynamics rung, the same success set, time window and norm; NaN for failures, without bounds, or when the bound is for another rung (then see E3m).
- Summary: interquartile mean across episodes.
- Why it matters: fuel against what physics allows, at least 1 by theorem.
- Known-answer tests: 2 with a lower bound at half the delta-v.
- Convergence: inherits E1's.
- Invariance: the bound's norm must match the layout; a mismatch is rejected.
- Mutation detection: not exercised; no oracle bounds are in the suite.
- Anti-gaming: bounds are for the success set, not the exact hold point, so finishing at the edge of the capture ball cannot score below 1 (Boyd and Vandenberghe, sections 4.1.1 and 5.5).
- Reproducibility: 1e-9 relative.
- Power: oracle bounds not yet available.
- Validity: E3f flags any run below its bound.
- Changelog: v1, first version; the bound now names its rung and E3 is reported only on it.

## E3m optimality_gap_model (v1)
- Definition: layout delta-v over a lower bound certified for a different dynamics rung than the episode's; NaN otherwise.
- Summary: interquartile mean across episodes.
- Why it matters: a CW bound is a theorem for CW only; on two-body truth the optimum can lie on either side of it, so the ratio can fall below 1 with no bug (Rawlings, Mayne and Diehl, sections 8.1.2 and 8.10).
- Known-answer tests: 1/1.01 for a CW bound 1% above the delta-v of a two-body run, which is not flagged.
- Convergence: inherits E1's.
- Invariance: as E3.
- Mutation detection: not exercised; no oracle bounds are in the suite.
- Anti-gaming: kept apart from E3 so a model ratio below 1 is never read as a certified gap.
- Reproducibility: 1e-9 relative.
- Power: oracle bounds not yet available.
- Validity: not applicable.
- Changelog: v1, first version.

## E3u optimality_gap_upper (v1)
- Definition: layout delta-v over the best known feasible cost.
- Summary: interquartile mean across episodes.
- Why it matters: progress against the best known solution, which can be below 1.
- Known-answer tests: 1.25 with an upper bound at 0.8 of the delta-v.
- Convergence: inherits E1's.
- Invariance: as E3.
- Mutation detection: not exercised; no oracle bounds are in the suite.
- Anti-gaming: reported next to E3, never instead of it.
- Reproducibility: 1e-9 relative.
- Power: oracle bounds not yet available.
- Validity: not yet measured.
- Changelog: v1, first version.

## E3f below_lower_bound (v1)
- Definition: a successful run spent less than the certified lower bound by more than 1e-9 relative.
- Summary: a rate across episodes, with its Clopper-Pearson interval.
- Why it matters: a theorem says this cannot happen, so it exposes simulator or accounting bugs.
- Known-answer tests: flagged 1% below the bound; not flagged at the bound within rounding; never flagged for a failure or for a bound certified for another rung.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: a bug that under-counts delta-v would raise it.
- Anti-gaming: failures are excluded, since the bound is for reaching the success set.
- Reproducibility: exact (boolean).
- Power: not applicable.
- Validity: not applicable.
- Changelog: v1, first version.

## E3c bound_certified (v1)
- Definition: whether the oracle's lower bound is certified for the episode's dynamics rung.
- Summary: a rate across episodes, with its Clopper-Pearson interval.
- Why it matters: says whether E3 >= 1 is a theorem for this run.
- Known-answer tests: False for a CW bound on a two-body run, True for a two-body bound there and for a CW bound on CW.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: a bound must name its rung; there is no default.
- Reproducibility: exact (boolean).
- Power: not applicable.
- Validity: not applicable.
- Changelog: v1, first version.

## E3r bound_rung (v1)
- Definition: the dynamics rung the oracle's lower bound is certified for, empty without bounds.
- Summary: not summarised across episodes (a label, a record or already an aggregate).
- Why it matters: carried with the result so a later reader can tell a certified gap from a model ratio.
- Known-answer tests: "cw" for a CW bound.
- Convergence: not applicable.
- Invariance: not applicable.
- Mutation detection: not applicable.
- Anti-gaming: as E3c.
- Reproducibility: exact (label).
- Power: not applicable.
- Validity: not applicable.
- Changelog: v1, first version.

## E4 burn_count (v1)
- Definition: maximal runs of consecutive steps with any executed thrust.
- Summary: interquartile mean across episodes.
- Why it matters: thruster cycling and operational complexity.
- Known-answer tests: 3 burns in a scripted on, on, off, on, off, on pattern; a half-step delay keeps one burn one burn.
- Convergence: defined per step, so it changes with the step length by definition.
- Invariance: mirror.
- Mutation detection: the PD mutants all thrust continuously (one burn), so E4 does not separate them.
- Anti-gaming: reads executed thrust.
- Reproducibility: exact (count).
- Power: starts behind the target: no spread over 40 runs (all 1); the spec's start box: no spread over 40 runs (all 1).
- Validity: not yet measured.
- Changelog: v1, first version.

## E5 duty_cycle (v1)
- Definition: time with any executed thrust over T.
- Summary: interquartile mean across episodes.
- Why it matters: share of the episode spent thrusting.
- Known-answer tests: 1 for a constant burn; 1/2 for the scripted pattern.
- Convergence: same at 10 and 2.5 s steps; same trajectory at 20, 10 and 5 s.
- Invariance: unchanged when all lengths double.
- Mutation detection: not separating for the PD mutants (all thrust continuously).
- Anti-gaming: reads executed segments, so a delay or the minimum impulse bit counts as flown.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: no spread over 40 runs (all 1); the spec's start box: no spread over 40 runs (all 1).
- Validity: not yet measured.
- Changelog: v1, first version.

## E6 chatter (v1)
- Definition: per axis, sign changes between consecutive nonzero executed accelerations, summed over axes, over T (1/s).
- Summary: interquartile mean across episodes.
- Why it matters: oscillating commands wear thrusters and waste fuel.
- Known-answer tests: 3 changes in 80 s for the scripted pattern; 0 for a constant burn.
- Convergence: same trajectory at 20, 10 and 5 s.
- Invariance: mirror.
- Mutation detection: rises in order with alternating commands of 0.02, 0.1 and 0.4.
- Anti-gaming: counts executed accelerations, so a filter that passes chatter through is still charged.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: SD 0.000852 1/s about a mean of 0.004547 1/s, 56 episodes per group for a 10% change; the spec's start box: SD 0.000833 1/s about a mean of 0.003657 1/s, 82 episodes per group for a 10% change.
- Validity: drives the oscillation failure label.
- Changelog: v1, first version.
