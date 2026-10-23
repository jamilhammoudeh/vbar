# Minimum-time oracle

Code: `minimum_time`, `MinimumTime` and `verify_infeasible` in `vbar.oracles.minimum_time`,
exported from `vbar.oracles`. Hill frame, metres, seconds.

`minimum_time` answers how few steps a scenario needs: the smallest step $k$ at whose end a plan
of accelerations, each held for one step within the thrust limit, puts the chaser in the capture
ball, with the plan verified by exact propagation. For every smaller step it either proves that
no acceleration history within the limit gets there, held per step or not, or lists the step as
uncertified. A run that captures before a proved step has hit a simulator or actuator bug.

## How it is built

**Proof that a step is out of reach.** Let $T = k\,\Delta t$, $\Phi$ the CW transition matrix,
$\mathbf x_g$ the hold point and goal velocity and $\rho_r$, $\rho_v$ the `capture_radius` and
`capture_speed`. For a multiplier $\nu \in \mathbb R^6$ with primer
$\mathbf p(t) = B^T\Phi(T - t)^T\nu$ (closed form on the [fuel oracles](oracles.md) page), every
acceleration history satisfies

$$
\nu^T\big(\mathbf x(T) - \Phi(T)\mathbf x_0\big) = \int_0^T \mathbf p(t)^T\mathbf a(t)\,dt
\le a \int_0^T |\mathbf p(t)|_1\,dt
$$

when $|a_i(t)| \le a$ on each Hill axis (box thrusters: the dual of the per-axis limit is the
1-norm of $\mathbf p$). Entering the closed ball needs at least the support margin

$$
D(\nu) = \nu^T\big(\mathbf x_g - \Phi(T)\mathbf x_0\big) - \rho_r|\nu_r|_2 - \rho_v|\nu_v|_2 .
$$

So $D(\nu) > a\int_0^T|\mathbf p|_1\,dt$ proves the ball out of reach at $T$, for exact CW motion
and no other forcing. This is a theorem of alternatives (Boyd and Vandenberghe, *Convex
Optimization*, 2004, section 5.8); it holds for any $\nu$, and positive multiples of $\nu$ give
the same answer. `verify_infeasible(view, start, nu, step)` checks it.

**Certified integral.** The integral is bounded from above over cells of width $h$: on each cell
the larger endpoint value of $|p_i|$ plus $L_i h / 2$, with the global Lipschitz constants $L_i$
of the primer from the [fuel oracles](oracles.md) page, summed over the axes, then multiplied by
$1 + 10^{-12}$. The margin is reduced by 64 machine epsilons of the summed magnitudes of its
terms. Cells start at 8 per step and halve, down to 128 per step, while the check fails and the
trapezoid estimate of the integral still leaves a positive margin: the bound converges to the
integral, so a multiplier with no estimated margin is not refined. The slack $L h / 2$ matters
near the minimum: on the 100 m case below, the multiplier for 141 steps has a margin of 0.595 m,
while 8 cells per step cost 1.76 m of slack and 64 cost 0.22 m.

**Step-grid programs.** Positions are scaled by
$\ell = \max(|\mathbf r_0|, |\mathbf r_g|, \rho_r, T|\mathbf v_0|, T|\mathbf v_g|, T\rho_v)$ and
velocities by $\ell / T$, so the programs stay well scaled as $n \to 0$. With $R$ the exact
zero-order-hold response at step $k$ to the held inputs and $\mathbf m = \mathbf x_g -
\Phi(T)\mathbf x_0$, a phase I linear program finds the least violation $s \ge 0$ of the ball's
bounding box,

$$
\min s \quad\text{subject to}\quad |(R\mathbf u - \mathbf m)_i| \le \rho_i + s,\quad |u_i| \le a,
$$

by HiGHS at feasibility tolerance $10^{-10}$. Its terminal multipliers, unscaled, are the $\nu$
of the proof. If $s$ exceeds $10^{-9}$ (scaled), no held plan lands and the multiplier is
checked; otherwise the least 1-norm plan aiming inside $0.9\rho_i/\sqrt 3$ per component is
solved. HiGHS drops matrix entries below $10^{-9}$, which on short windows are the cross-axis
couplings ($nT$ is $10^{-7}$ at $n = 10^{-9}$ rad/s), so up to three least-squares steps on the
inputs inside the limit move the exact landing to the goal, each followed by clipping to the
limit. The plan is accepted when exact propagation, step by step with `cw_discrete`, ends within
$(1 + 0.9)/2$ of the capture radius and speed.

**Search.** Steps are tried in order from 1. A multiplier that proved an earlier step is tried
first at 8 cells per step, since it usually proves the next steps too until its margin runs
out; only then is the phase I program solved. The first step with an accepted plan is the
answer. `certified` is True when every smaller step was proved out of reach, and the others are
listed in `uncertified_steps`.

## Parameters and defaults

| Parameter | Default | Reason |
|---|---|---|
| `config.cells_per_step` | 8 | Starting cell density of the proof; refinement only where needed |
| `config.aim` | 0.9 | Plans aim inside $0.9\rho/\sqrt 3$ per component and must land within $0.95\rho$ |
| Finest cells | 128 per step | Four times the 32 per step that the step-grid multiplier of the 100 m case below needs at step 141; the 700 m case needs 16 at step 35 |
| Phase I threshold | $10^{-9}$ scaled | Ten times the LP's feasibility tolerance |
| LP tolerances | $10^{-10}$ | The tightest HiGHS accepts; a ball can be $10^{-8}$ of the scaled distance |

`minimum_time(view, start, config=OracleConfig(...))` reads only `cells_per_step` and `aim` of
the [fuel oracle](oracles.md) settings. `verify_infeasible` always starts at 8 cells per step.

## Assumptions and limits

- Exact CW motion: a circular target orbit and no other forcing. The proof is a theorem on the
  `"cw"` rung only.
- Box thrusters only, with $|a_i| \le$ `max_accel` on each Hill axis; a ball layout raises
  `ValueError`. The proof needs the flown acceleration within that limit, which a thrust drive
  that gains acceleration as mass falls, or an execution-error scale bias, can break.
- The answer is the arrival step. A dwell (`hold_time`) and path constraints such as keep-out
  are not modelled: the plan may violate them, while a step proved out of reach stays out of
  reach with them. Goals with thrust inhibit windows raise `ValueError`.
- Proofs are about continuous thrust; plans hold each input for one step. A step that
  continuous thrust can reach but the step grid cannot is neither proved nor reached, and goes
  to `uncertified_steps`. On a 1-axis rest-to-rest move of $d$ this happens when
  $d / (a\,\Delta t^2)$ lies in $(\lfloor k^2/4 \rfloor, k^2/4]$ for an odd $k$.
- The capture ball is open for plans (the capture predicate is strict), so a zero radius or
  speed never yields a plan; the proofs treat it as closed, which makes them stronger.
- `verify_infeasible` returning False says nothing about the step: the multiplier may be poor,
  or the finest cells may leave too much slack.

## Validation

Tests: `tests/test_minimum_time.py`. Values marked as independent were computed with scipy's
matrix exponential of the CW system and scipy HiGHS, separately from the module.

- With $n = 10^{-9}$ rad/s the motion over a few minutes is a double integrator, whose
  rest-to-rest minimum time over $d$ is $2\sqrt{d/a}$ (bang-bang with one switch), and the
  largest rest-to-rest move in $k$ held steps is $a\,\Delta t^2\lfloor k^2/4\rfloor$.
- 100 m along V-bar, $a = 0.02$ m/s^2, 1 s steps, a $10^{-6}$ m and m/s ball: 142 steps, with all
  141 smaller steps proved ($2\sqrt{100/0.02} = 141.42$ s; 141 steps move at most 99.40 m and 142
  move 100.82 m, independent). The plan, flown exactly, lands inside the ball within the limit.
- The switching multiplier $\nu_y = 1$, $\nu_{vy} = -T/2$, whose integral of $|p_y|$ is $T^2/4$,
  proves 141 steps and not 142, also with the primer charged on both axes for a two-axis move,
  under positive scaling from $10^{-6}$ to $10^4$, and not when negated or zero. Of 100 random
  multipliers at 142 steps none passes; 10 small perturbations of the 141-step multiplier all
  pass.
- 30 m along R-bar with 2 s steps: 39 steps, certified ($2\sqrt{30/0.02} = 77.46$ s; 39 steps
  move 30.40 m, 38 move 28.88 m, independent). An episode of 30 steps (at most 18 m) returns no
  step and no plan, certified.
- 22.04 m with $a = 0.05$ m/s^2 and 2 s steps: continuous thrust arrives in 41.990 s, within
  step 21, but 21 held steps move at most 22.00 m. The answer is 22 steps with step 21 in
  `uncertified_steps` and `certified` False.
- 700 m along V-bar on a 500 km circular orbit (WGS 84), 10 s steps: 36 steps, certified. The
  step-grid program first reaches the point at 36 steps and a 2 s grid needs 358 s, so the
  continuous minimum lies in $(350, 358]$ s; the best multiplier for 350 s passes the cell rule
  at 64 cells per step but not at 8 (independent).
- A start at rest at the hold point arrives at step 1.
- Unknown thrust layouts, goals without a capture ball, bad start states, thrust limits that are
  not positive and finite, malformed multipliers and steps outside the episode raise
  `ValueError`.

## Example

```python
import math

import numpy as np

from vbar.agents.base import AgentView
from vbar.oracles import minimum_time, verify_infeasible

view = AgentView(
    scenario_id="radial-30m",
    dt=2.0,
    max_steps=60,
    n=1e-9,
    action_mode="accel",
    max_accel=0.02,
    thrust_limit="box",
    goal={"hold_point": (0.0, 0.0, 0.0), "capture_radius": 1e-6, "capture_speed": 1e-6},
)
start = np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0])

result = minimum_time(view, start)
assert result.steps == math.ceil(2 * math.sqrt(30.0 / 0.02) / view.dt) == 39
assert result.certified and result.uncertified_steps == ()
assert result.plan.shape == (39, 3)

# The bang-bang switching multiplier along -x proves 38 steps (76 s) out of reach; negated,
# it proves nothing.
t = 38 * view.dt
nu = np.array([-1.0, 0.0, 0.0, t / 2, 0.0, 0.0])
assert verify_infeasible(view, start, nu, 38)
assert not verify_infeasible(view, start, -nu, 38)
```
