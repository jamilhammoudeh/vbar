# Mission metric cards

Definitions in full: [docs/metrics.md](../../../../docs/metrics.md). Tests: `tests/test_metrics_mission.py`
unless named. Power figures come from the PD baseline over seeds 0 to 39 of approach-vbar (360
steps), from starts 200 to 600 m behind the target, where it succeeds, and from the spec's start
box 200 to 600 m ahead, where the target is in its way: the spread over seeds and the episodes per
group needed to detect a change of 10% of the mean (two-sided alpha 0.05, power 0.8, normal
approximation, unpaired; paired runs on common seeds need fewer).

## M1 success (v1)
- Definition: the goal is met (capture at a step boundary; a dwell of `hold_time`; contact with M10 >= 0) and no hard constraint's certified in-step minimum fell below zero. Truth.
- Why it matters: the headline completion rate.
- Known-answer tests: PD from 400 m behind succeeds; rest on the V-bar hold point succeeds at 10 s; capture after flying through the keep-out zone fails; a hold that lasts 0.9 of the required dwell fails and 1.1 succeeds; contact at 0.08 m/s on axis succeeds, at 0.2 m/s fails.
- Convergence: decided at step boundaries for capture, in continuous time for dwell and contact; contact found between samples 10 s apart (`test_a_contact_between_two_samples_ten_seconds_apart_is_found`).
- Invariance: unchanged by mirroring across the orbit plane and by doubling all lengths and the thrust (`tests/test_metrics_gate.py`).
- Mutation detection: sign flip fails; position biases of 10 and 40 m and velocity biases of 0.03 and 0.1 m/s fail (`tests/test_metrics_mutants.py`).
- Anti-gaming: a violation between step boundaries still fails the run (continuous-time minima); passing the port between boundaries is caught by the contact event; a hold must last its dwell.
- Reproducibility: exact (boolean) in `tests/golden/metrics_*.json`.
- Power: PD succeeds from 40 of 40 starts behind the target and 0 of 40 in the spec's start box, where the target is in the way; a drop of 0.1 from the first rate needs 74 episodes per group.
- Validity: not yet measured.
- Changelog: v1, first version.

## M2 terminal_pos_error (v1)
- Definition: |r - r_goal| at the end (m), r_goal the hold point or the port. Truth.
- Why it matters: how far from the goal the run ended.
- Known-answer tests: zero at the V-bar equilibrium; equal to the closed-form free-drift end point for the flyby (1e-12 relative).
- Convergence: same at 5, 10 and 20 s steps (1e-9 relative).
- Invariance: mirror across the orbit plane; doubles when all lengths double.
- Mutation detection: rises in order with position bias 2, 10, 40 m and velocity bias 0.01, 0.03, 0.1 m/s.
- Anti-gaming: not a success criterion on its own; reported with M1 and M4.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: SD 0.106 m about a mean of 4.779 m, under 2 episodes per group for a 10% change; the spec's start box: SD 0.107 m about a mean of 4.812 m, under 2 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## M3 terminal_vel_error (v1)
- Definition: |v - v_goal| at the end (m/s). Truth.
- Why it matters: a run can end near the goal but moving fast.
- Known-answer tests: zero at the V-bar equilibrium; under the capture speed for PD's captures.
- Convergence: same at 5, 10 and 20 s steps (1e-9 relative).
- Invariance: mirror; doubles with all lengths.
- Mutation detection: the biased mutants settle at rest with an offset, so M3 does not see them; M2 does.
- Anti-gaming: reported with M1.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: SD 0.000897 m/s about a mean of 0.03908 m/s, under 2 episodes per group for a 10% change; the spec's start box: SD 0.000861 m/s about a mean of 0.03954 m/s, under 2 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## M4 time_to_complete (v1)
- Definition: when the goal is met (contact time, end of the first full dwell, first capturing step boundary), else T (s).
- Why it matters: speed of completion.
- Known-answer tests: 10 s at the V-bar equilibrium; the closed-form crossing time for contact (1e-6 s); the required dwell for a hold that succeeds.
- Convergence: step-boundary resolution for capture; continuous for dwell and contact.
- Invariance: mirror; unchanged when all lengths double.
- Mutation detection: equals T for the mutants that fail (sign flip, the larger biases).
- Anti-gaming: equals T for failures, so a fast failure does not score well.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: SD 52.5 s about a mean of 616.2 s, 12 episodes per group for a 10% change; the spec's start box: SD 36.2 s about a mean of 700 s, 5 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## M5 progress (v1)
- Definition: clip(1 - d_end / d_0, 0, 1), d the distance to the goal.
- Why it matters: partial credit for failed runs; the composite score's completion.
- Known-answer tests: 1 at the equilibrium; 0 when moving away; 1 - d_end / 250 for the flyby.
- Convergence: same at 5, 10 and 20 s steps.
- Invariance: mirror; unchanged when all lengths double.
- Mutation detection: 0 for the sign flip; falls with position bias.
- Anti-gaming: clipped at 0 so moving away is not negative credit; a run that ends near the goal after a violation still fails M1.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: SD 0.00755 about a mean of 0.9826, under 2 episodes per group for a 10% change; the spec's start box: SD 0.00204 about a mean of 0.9903, under 2 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## M6 contact_lateral_error (v1)
- Definition: lateral offset from the docking axis at the first port-plane crossing (m). Truth, continuous time.
- Why it matters: reception range of the mechanism (Fehse 7.1.1: 0.05 to 0.2 m).
- Known-answer tests: equals the closed-form offset at the closed-form crossing time to 1e-9 m.
- Convergence: the crossing is found between samples 10 s apart.
- Invariance: mirror (NaN without a port).
- Mutation detection: not exercised by a mutant yet; the contact cases set it directly.
- Anti-gaming: evaluated at the event, not at a step boundary, so a pass through the port between boundaries is scored.
- Reproducibility: 1e-9 relative.
- Power: no docking scenario yet.
- Validity: not yet measured.
- Changelog: v1, first version.

## M7 contact_axial_speed (v1)
- Definition: -v . a_hat at contact (m/s).
- Why it matters: impact docking needs a minimum and a maximum contact speed (Fehse 8.3.5, appendix C).
- Known-answer tests: equals the closed-form speed at the crossing to 1e-9 m/s.
- Convergence: continuous-time event.
- Invariance: mirror.
- Mutation detection: a 0.2 m/s contact against a 0.10 m/s window gives M10 = -1.
- Anti-gaming: the window's lower end stops hovering at the port from counting as docking.
- Reproducibility: 1e-9 relative.
- Power: no docking scenario yet.
- Validity: not yet measured.
- Changelog: v1, first version.

## M8 contact_lateral_speed (v1)
- Definition: |v - (v . a_hat) a_hat| at contact (m/s).
- Why it matters: lateral speed limits are 4 to 20 times tighter than total-speed limits (Fehse 7.1.1).
- Known-answer tests: equals the closed-form lateral speed at the crossing to 1e-9 m/s.
- Convergence: continuous-time event.
- Invariance: mirror.
- Mutation detection: set directly in the contact cases.
- Anti-gaming: kept apart from the axial speed so one cannot hide in the other.
- Reproducibility: 1e-9 relative.
- Power: no docking scenario yet.
- Validity: not yet measured.
- Changelog: v1, first version.

## M9 hold_dwell (v1)
- Definition: longest continuous time inside the capture predicate (s), NaN without one.
- Why it matters: a state inside the capture box can leave it within a fraction of an orbit (Fehse 3.3.3).
- Known-answer tests: from rest 2 m above the hold point, the closed-form time to leave a 10 m ball (1e-6 s).
- Convergence: continuous time from the hold signal's negative intervals.
- Invariance: mirror.
- Mutation detection: approach episodes end at capture, so the dwell is short for all baselines; exercised by the hold cases.
- Anti-gaming: stopping at the edge of the box does not count as a hold unless it lasts.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: SD 2.71 s about a mean of 5.552 s, 374 episodes per group for a 10% change; the spec's start box: SD 2.72 s about a mean of 4.696 s, 526 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## M10 capture_margin (v1)
- Definition: least over the contact criteria of (limit - value) / limit, the window's lower end as (value - limit) / limit; negative is out of tolerance.
- Why it matters: a continuous margin next to the boolean contact verdict.
- Known-answer tests: equals the formula on closed-form contact values to 1e-8; -1 for a 0.2 m/s impact against 0.10 m/s.
- Convergence: continuous-time event.
- Invariance: mirror.
- Mutation detection: negative for the fast contact.
- Anti-gaming: each criterion normalised by its own limit so none dominates by units.
- Reproducibility: 1e-9 relative.
- Power: no docking scenario yet.
- Validity: not yet measured.
- Changelog: v1, first version.
