# Safety metric cards

Definitions in full: [docs/metrics.md](../../../../docs/metrics.md) and, for the margins,
[docs/constraints.md](../../../../docs/constraints.md). Tests: `tests/test_metrics_safety.py` unless
named. Power as in the mission cards.

## S1 min_margin (v1)
- Definition: per constraint, the lowest margin over the episode in continuous time, start included (constraint units). Truth.
- Summary: the unsafe tail across episodes (lowest value, 5th percentile, mean at or below it), never the interquartile mean.
- Why it matters: how close each constraint came to being broken, as signal temporal logic robustness.
- Known-answer tests: keep-out S1 of the flyby equals the closed-form closest approach minus 50 m to 1 mm; speed and closing-speed margins found on the rebuilt trajectory equal the environment's own search to 1e-9 m/s; each constraint type's margin matches a direct reference (`tests/test_constraint_types.py`).
- Convergence: the same at 4, 10 and 25 s steps to 1e-6 for keep-out, speed, closing-speed and cone; the same trajectory at 20, 10 and 5 s (`tests/test_metrics_properties.py`).
- Invariance: mirror across the orbit plane; doubles with all lengths.
- Mutation detection: negative when PD flies through the target, positive for the clean approach (`tests/test_metrics_mutants.py`).
- Anti-gaming: a dip between step boundaries or between samples is found (Lipschitz-flagged refinement), so burning then coasting through a constraint does not hide it.
- Reproducibility: to the resolution (1 mm for range margins).
- Power: starts behind the target: SD 0.403 m about a mean of 54.43 m, under 2 episodes per group for a 10% change; the spec's start box: SD 10.9 m about a mean of -32.62 m, 176 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## S1b min_margin_bound (v3)
- Definition: certified lower bound on S1, per sample interval the larger of the Lipschitz bound and the constraint's second-order bound where it has one (the squared range for the keep-out sphere; the speed, closing-speed and gate-table margins with a jerk bound).
- Changelog: v1, the Lipschitz bound alone; v2, the keep-out sphere adds the second-order bound, which closes the gap to S1 from metres to millimetres on dense samples; v3, the speed, closing-speed and gate-table limits add theirs.
- Summary: the unsafe tail across episodes (lowest value, 5th percentile, mean at or below it), never the interquartile mean.
- Why it matters: S1 is the lowest value found; S1b is a value the trajectory provably never went below.
- Known-answer tests: S1b <= S1 for the flyby; the certificate lies below the dense minimum for 12 flybys of every constraint type (`tests/test_constraint_types.py`).
- Convergence: tightens as samples get closer, by design; excluded from the step-size comparison.
- Invariance: mirror.
- Mutation detection: as S1.
- Anti-gaming: infinite (no claim) rather than wrong when the bounds fail, for example with very long sample intervals.
- Reproducibility: 1e-9 relative.
- Power: as S1.
- Validity: not applicable.
- Changelog: v1, first version.

## S2 violation_depth (v1)
- Definition: max(0, -S1) per constraint.
- Summary: the unsafe tail across episodes (highest value, 95th percentile, mean at or above it), never the interquartile mean.
- Why it matters: how deep the worst violation went.
- Known-answer tests: 50 m minus the closed-form closest approach for the flyby (1 mm).
- Convergence: as S1.
- Invariance: as S1.
- Mutation detection: positive when PD flies through the target.
- Anti-gaming: as S1.
- Reproducibility: to the resolution.
- Power: starts behind the target: no spread over 40 runs (all 0 m); the spec's start box: SD 10.9 m about a mean of 32.62 m, 176 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## S3 violation_duration (v1)
- Definition: total time with the margin below zero, per constraint (s), with roots refined at sign changes.
- Summary: the unsafe tail across episodes (highest value, 95th percentile, mean at or above it), never the interquartile mean.
- Why it matters: a brief graze and a long violation have the same S1 sign but not the same risk.
- Known-answer tests: the flyby's time inside the 50 m sphere equals the closed-form exit minus entry time to 1e-6 s; a dip between two samples 10 s apart is found; a rise above zero between two negative samples splits the interval (`tests/test_metrics_episode.py`).
- Convergence: the same at 3.7, 10 and 25 s steps to 1e-6 s.
- Invariance: mirror.
- Mutation detection: positive when PD flies through the target.
- Anti-gaming: continuous time, so step-boundary checks cannot be timed around.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: no spread over 40 runs (all 0 s); the spec's start box: SD 16 s about a mean of 132.8 s, 23 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## S5 closing_speed_ratio (v2)
- Definition: highest closing speed over the speed the scenario's closing-rate law allows at that range, the linear v0 + v1 rho or a gate table; infinite when closing at a hold gate or where the law allows less than 0; the higher ratio when both laws are declared; NaN without a law.
- Summary: the unsafe tail across episodes (highest value, 95th percentile, mean at or above it), never the interquartile mean.
- Why it matters: how close the approach came to the speed rule, scale-free.
- Known-answer tests: equals the maximum over 120001 closed-form samples to 1e-7, for the linear law and for a gate table; infinite closing through a hold gate; the STS-130 +V-bar gates sit 3.84, 1.76, 1.80 and 1.03 times inside the V-bar passive-speed envelope at 185 n.mi. (Goodman, fig. 25.22).
- Convergence: the same at 4, 10 and 25 s steps to 1e-7.
- Invariance: mirror.
- Mutation detection: not exercised by the PD mutants (the approach scenario has no closing-speed limit yet).
- Anti-gaming: refined around the highest sample; the certified quantity is S1 of the closing-speed limit.
- Reproducibility: 1e-6.
- Power: no closing-speed limit in the baseline scenario yet.
- Validity: the Shuttle flew gate tables 3 to 11 times below the linear 0.2 + 4 n rho law (Goodman, table 24.3), so the law is the scenario's, not fixed.
- Changelog: v1, the linear law only; v2, the scenario's law, linear or a gate table.

## S6 min_time_to_keep_out (v1)
- Definition: least (rho - R) / closing speed over the step boundaries while closing; 0 inside; infinite if never closing.
- Summary: the unsafe tail across episodes (lowest value, 5th percentile, mean at or below it), never the interquartile mean.
- Why it matters: time available to react before entering the keep-out zone.
- Known-answer tests: 500 s for 250 m at 0.5 m/s straight in; infinite at rest on the V-bar.
- Convergence: defined at step boundaries.
- Invariance: mirror; unchanged when all lengths double.
- Mutation detection: 0 for PD through the target and 199 s for the clean approach in the golden values.
- Anti-gaming: 0 once inside, so entering the zone cannot score a long time.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: SD 25.2 s about a mean of 210 s, 23 episodes per group for a 10% change; the spec's start box: no spread over 40 runs (all 0 s).
- Validity: not yet measured.
- Changelog: v1, first version.
