# Passive-safety and abort metric cards

Definitions in full: [docs/metrics.md](../../../../docs/metrics.md). Tests:
`tests/test_metrics_passive.py` unless named. All read the declared `MetricSettings` (collision
radius, exclusion and commit zones, horizons, the CAM set). Power as in the mission cards.

## S4 passive_safety_margin (v1)
- Definition: least free-drift collision margin over the drift horizon from every step boundary beyond the exclusion zone (m), on the episode's dynamics model (Fehse 4.4).
- Why it matters: whether a loss of thrust at any point would end in a collision.
- Known-answer tests: 50 m everywhere at rest on the V-bar hold point; equal to a closed-form reference (20001 samples and Brent) over the flyby to 1 mm; within 1 cm of Clohessy-Wiltshire on the two-body model at 150 m.
- Convergence: the same at 50 and 800 drift samples per orbit to 1 mm; as steps nest from 40 to 5 s it can only fall, by less each time (`tests/test_metrics_gate.py`).
- Invariance: mirror; doubles with all lengths.
- Mutation detection: falls in order as PD gains rise from 0.005 to 0.04 rad/s; the 0.02 rad/s approach succeeds while S4 is negative, which M1 cannot show (`tests/test_metrics_mutants.py`).
- Anti-gaming: evaluated from every step boundary, not only the end; partial burns are covered by S4o.
- Reproducibility: to 1 mm.
- Power: starts behind the target: SD 26.3 m about a mean of 1.586 m, 15.7 (SD / delta)^2 episodes per group for a change delta; the spec's start box: SD 10.2 m about a mean of -39.37 m, 106 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## S4b passive_safe_fraction (v1)
- Definition: share of the checked step boundaries whose free drift stays clear.
- Why it matters: how much of the approach was passively safe.
- Known-answer tests: equals the share of closed-form references that stay clear over the flyby.
- Convergence: the same at 50 and 800 drift samples per orbit.
- Invariance: mirror; unchanged when all lengths double.
- Mutation detection: never rises as the PD gain rises from 0.005 to 0.04 rad/s, and is below 1 at 0.04.
- Anti-gaming: arcs whose sign the samples cannot settle are refined, so the fraction is exact up to the single-dip assumption.
- Reproducibility: exact (ratio of counts).
- Power: starts behind the target: SD 0.255 about a mean of 0.8029, 159 episodes per group for a 10% change; the spec's start box: SD 0.0456 about a mean of 0.6037, 9 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## S4i passive_forever_fraction (v1)
- Definition: share certified clear for all time by the travelling ellipse (Clohessy-Wiltshire only).
- Why it matters: a two-orbit check misses slow drift toward the target (Fehse appendix A.1.5).
- Known-answer tests: the ellipse matches propagation to 1e-9 m on Fehse's appendix case, where his printed sign is 176 m off; a parallel orbit 10 m below drifting at the target is not safe forever, drifting away it is, and a relative orbit whose tube clears the target by 60 m is.
- Convergence: exact by construction for the certified cases; passages up to 10 orbits are sampled 400 times per orbit and refined.
- Invariance: mirror.
- Mutation detection: never rises with the PD gain, and is below 1 at 0.04 rad/s.
- Anti-gaming: a long passage that cannot be sampled is not certified, so slow drift toward the target is never called safe.
- Reproducibility: exact (ratio of counts).
- Power: starts behind the target: SD 0.254 about a mean of 0.8022, 158 episodes per group for a 10% change; the spec's start box: SD 0.0463 about a mean of 0.6023, 10 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## S4t passive_threat_time (v1)
- Definition: earliest time a drift not certified forever could reach the envelope (s); infinite if all are certified.
- Why it matters: separates "unsafe in two minutes" from "unsafe in a year".
- Known-answer tests: the closed-form entry time of the drifting parallel orbit (0.2%).
- Convergence: sampled at 400 per orbit within the passage window.
- Invariance: mirror.
- Mutation detection: infinite at 0.005 rad/s and finite at 0.04 rad/s.
- Anti-gaming: reports when an uncertified passage begins rather than calling it safe.
- Reproducibility: 1e-9 relative.
- Power: starts behind the target: SD 2.79e+07 s about a mean of 6.232e+06 s, 15.7 (SD / delta)^2 episodes per group for a change delta; the spec's start box: no spread over 40 runs (all 0 s).
- Validity: not yet measured.
- Changelog: v1, first version.

## S4o stuck_thruster_margin (v1)
- Definition: as S4 with the acceleration in effect continuing for the detection time, from step boundaries and from points inside thrusting segments, where a burn cut short is also checked (m).
- Why it matters: Fehse's analysis is of thrusters failing closed during a burn and stuck open until detected (4.3.3, 4.4.2).
- Known-answer tests: a full-thrust command toward the target stuck for 30 s from 300 m matches its closed form to 1 mm and lies below S4; adding partial-burn points can only lower it.
- Convergence: the same at 50 and 800 drift samples per orbit.
- Invariance: mirror.
- Mutation detection: falls in order with the PD gain.
- Anti-gaming: checks inside thrusting segments, so a burn that is only unsafe midway is caught.
- Reproducibility: to 1 mm.
- Power: starts behind the target: SD 23.8 m about a mean of -6.45 m, 15.7 (SD / delta)^2 episodes per group for a change delta; the spec's start box: SD 10.2 m about a mean of -39.5 m, 106 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## S8 cam_margin (v1)
- Definition: least over step boundaries beyond the commit zone of the best margin over the CAM set (drift, impulse, drift) (m).
- Why it matters: where passive safety cannot hold, a collision avoidance manoeuvre must (Fehse 4.5).
- Known-answer tests: escape arcs reproduce Fehse's R-bar CAM with a retrograde part (2231.6 m after an orbit), the radial CAM's 12 pi times its depth, and the -V-bar CAM passing under the target at -396.220 m; S8 equals a closed-form reference to 1 mm over the flyby.
- Convergence: the same at 50 and 800 drift samples per orbit.
- Invariance: mirror; doubles with all lengths when the CAM size doubles too.
- Mutation detection: not monotone in the PD gain, so not used to rank it; S9 combines it with S4.
- Anti-gaming: a step counts as protected only once one CAM of the set is shown clear.
- Reproducibility: to 1 mm.
- Power: starts behind the target: SD 22.1 m about a mean of -18.17 m, 15.7 (SD / delta)^2 episodes per group for a change delta; the spec's start box: SD 9.86 m about a mean of -38.78 m, 102 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.

## S9 unprotected_time (v1)
- Definition: time in steps that start beyond the commit zone with neither a clear drift nor a clear CAM (s).
- Why it matters: Fehse's rule is that it is zero outside the last metre.
- Known-answer tests: equals 10 s times the steps whose closed-form drift and CAM references both fail over the flyby.
- Convergence: counted per step, so it is a step-boundary quantity.
- Invariance: mirror.
- Mutation detection: never falls as the PD gain rises from 0.005 to 0.04 rad/s, and is positive at 0.04.
- Anti-gaming: inside the exclusion zone only aborts count, so passive safety cannot be claimed where it cannot hold.
- Reproducibility: exact for whole steps.
- Power: starts behind the target: SD 94.8 s about a mean of 59.5 s, 15.7 (SD / delta)^2 episodes per group for a change delta; the spec's start box: SD 41.5 s about a mean of 277.8 s, 36 episodes per group for a 10% change.
- Validity: not yet measured.
- Changelog: v1, first version.
