# Plume metric cards

Definitions in full: [docs/metrics.md](../../../../docs/metrics.md). Tests: `tests/test_metrics_plume.py`.
All need a declared plume range; without one they are NaN.

## S10 plume_impulse (v1)
- Definition: time integral of the Gaussian plume pressure on a declared target area from thrust inside the plume range (N s), thrust from the mass at the start of the step (Fehse 4.2.4, eqs. 4.12 to 4.13).
- Why it matters: plume loads on the target can exceed drag by one to two orders close in, which is why braking toward the target is avoided.
- Known-answer tests: braking at full thrust 30 m behind the target matches an independent quadrature on the closed-form trajectory to 1e-8; the pressure normalisation matches a midpoint rule over the sphere to 1e-6 and tends to 2 pi theta0^2 for a narrow plume; thrust outside the range counts nothing.
- Convergence: changes by less than 1e-4 between 10 and 2.5 s steps, the mass at the start of each step tracking the falling mass more closely.
- Invariance: mirror (box thrusters are symmetric under z to -z).
- Mutation detection: braking toward the target against thrusting away gives about 1e-3 against 1e-43 N s.
- Anti-gaming: the plume range entry and exit are found in continuous time, so a burn that starts outside and ends inside is counted for the time inside.
- Reproducibility: 1e-9 relative.
- Power: no plume range in the baseline scenario yet.
- Validity: a load indicator on a flat area facing the plume, not a surface force model.
- Changelog: v1, first version.

## S10t plume_time (v1)
- Definition: time with some exhaust within the plume half-angle (39 deg, three times the 13 deg 1-sigma) of the line of sight to the target, inside the plume range (s).
- Why it matters: exposure time, independent of the force model.
- Known-answer tests: 10 s for a full step of braking 30 m behind; 0 for thrusting toward the target; entering the range mid-step counts the time inside to 2e-4 s against 100001 closed-form samples.
- Convergence: the same at 10 and 2.5 s steps to 1e-6 s.
- Invariance: mirror.
- Mutation detection: as S10.
- Anti-gaming: per thruster for box layouts, so firing two axes cannot average the exhaust direction away.
- Reproducibility: 1e-9 relative.
- Power: no plume range in the baseline scenario yet.
- Validity: not yet measured.
- Changelog: v1, first version.

## S10m plume_margin (v1)
- Definition: least angle between a firing thruster's exhaust and the line of sight to the target, minus the plume half-angle, inside the plume range (rad): the soft plume constraint.
- Why it matters: a continuous margin for the "no exhaust toward the target" rule.
- Known-answer tests: -39 deg while braking straight at the target; for thrust toward the target, the least angle over 10001 closed-form samples minus 39 deg to 1e-6 rad.
- Convergence: the same at 10 and 2.5 s steps to 1e-6 rad.
- Invariance: mirror.
- Mutation detection: as S10.
- Anti-gaming: infinite only when no thrust falls inside the range.
- Reproducibility: to 1e-6 rad.
- Power: no plume range in the baseline scenario yet.
- Validity: not yet measured.
- Changelog: v1, first version.
