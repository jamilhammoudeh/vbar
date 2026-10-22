# Run time assurance filters

Code: `vbar.safety.rta` (`AfrlSetup`, `AfrlConstraints`, `NmtBackup`, `BackupRollout`,
`ExplicitSwitchingFilter`, `ExplicitOptimizationFilter`, `ImplicitSwitchingFilter`,
`ImplicitOptimizationFilter`).

Dunlap, Hibbard, Mote and Hobbs, "Comparing Run Time Assurance Approaches for Safe Spacecraft
Docking" (IEEE Control Systems Letters 6, 2022, arXiv:2110.00447v1), compare four filters on a
docking problem with a distance-dependent speed limit: explicit and implicit switching, and
explicit and implicit optimization. This module rebuilds them from the paper, on the paper's Euler
plant, so that they can be run beside the filters of [safety-filters.md](safety-filters.md). Values
the paper does not give come from the authors' published code
(github.com/kyle-dunlap/rta-for-spacecraft-docking) and are marked "code" below. None of the four
filters counts toward certification (see Assumptions and limits).

## The docking problem

The deputy moves on the Clohessy-Wiltshire equations in the Hill frame (x radial, y along-track,
z orbit normal), $\dot x = A x + B u$ with $A, B$ from `cw_system` and $u$ an acceleration in
m/s^2. The paper integrates them by Euler steps, $x_{k+1} = x_k + (A x_k + B u_k)\,\Delta t$
(sec. IV), and the filters predict with the same step. An Euler step drops the term
$a\,\Delta t^2/2$ from the position that the exact step of the environments keeps.

`AfrlSetup` holds the problem; `AfrlSetup.paper()` and `AfrlSetup.authors_code()` differ only in
$\nu_1$.

| Field | Value | Source |
|---|---|---|
| `mass_kg`, `u_max` | 12 kg, 1 N per axis (box), so $a_{\max} = 1/12$ m/s^2 | Assumption 2 |
| `n` | 0.001027 rad/s | Assumption 2 |
| `nu0`, `nu1` | 0.2 m/s; $4n$ (`paper`) or $2.15n$ (`authors_code`) | Assumption 2; code |
| `v_max`, `r_max` | 10 m/s, 10 km | Assumption 2 |
| `dt`, `horizon_s` | 1 s, 5 s | sec. IV |
| `start_range`, `start_speed` | 9.85 km, 0.866 m/s (directions not given) | sec. IV |
| `alpha_speed`, `alpha_velocity` | (0.05, 0.1), (0.0005, 0.001) | code |
| `nmt_grid` | 20 x 20 x 20 over $\theta_1, \theta_2, \psi$ | code |
| `backup_q`, `backup_r` | $Q = 10^{-5} I_6$, $R = 10^7 I_3$ on the force in N | code |
| `track_radius` | 1.5 m | code |
| `primary_q`, `primary_r` | $Q = 0.05 I_6$, $R = 1000 I_3$ on the force in N | code |
| `stop_radius`, `stop_time` | 1 m, 5000 s | code |

`setup.view()` is the `AgentView` the filters reset from: scenario "dunlap2021-euler", commands
in m/s^2 under a box of $a_{\max}$, one `speed_limit` constraint with $v_0 = \nu_0$,
$v_1 = \nu_1$, and `stop_time / dt` steps.

### Allowable set

`AfrlConstraints` evaluates the four constraints of eq. 16 and 19,

$$
\varphi_1 = \nu_0 + \nu_1 |r| - |v| \;\text{(m/s)}, \qquad
\varphi_{2,3,4} = v_{\max}^2 - v_i^2 \;\text{(m}^2\text{/s}^2\text{)},
$$

with gradients $\nabla\varphi_1 = (\nu_1 \hat r, -\hat v)$ and $\nabla\varphi_{i+1} = -2 v_i e_{v_i}$,
and the class-K function $\alpha(h) = a_1 h + a_3 h^3$ of each with the gains of the table.
$C_A$ is the set where all four are nonnegative. At $r = 0$ or $v = 0$ the undefined part of
$\nabla\varphi_1$ is set to zero, and every barrier row on $\varphi_1$ is dropped below a speed of
$10^{-9}$ m/s.

### The lemmas

Lemma 1 asks $(3n^2 + 2n\nu_1 + \nu_1^2) R_{\max} + (2n + \nu_1)\nu_0 \le u_{\max}$ for $C_S = C_A$.
The left side is an acceleration (`lemma1_accel`): 0.2860 m/s^2 at $\nu_1 = 4n$, against the
$1/12 = 0.0833$ m/s^2 the thruster gives, so the paper's values do not satisfy it. The authors'
code checks instead $v_{\max}(\sqrt3\,\nu_1 + 3\sqrt3\,n^2/\nu_1 + 2n) - 3n^2\nu_0/\nu_1 \le a_{\max}$
(`code_lemma_accel`): 0.08332 m/s^2 at $2.15n$, which passes by 0.02%, and 0.1049 m/s^2 at $4n$,
which fails.

## Backup controller

`NmtBackup` steers toward closed natural motion trajectories (sec. III.D and III.E). A point of
such a trajectory is

$$
x = b\sin\nu,\; y = 2b\cos\nu,\; z = c\sin\psi,\;
\dot x = bn\cos\nu,\; \dot y = -2bn\sin\nu,\; \dot z = nc\cos\psi,
$$

$$
c = \frac{b}{\sin\theta_1}\sqrt{\tan^2\theta_2 + 4\cos^2\theta_1}, \qquad
\nu = \tan^{-1}\frac{2\cos\theta_1}{\tan\theta_2} - \psi .
$$

- **Grid.** $\theta_1$ takes the upper ends and $\theta_2$ the lower ends of 20 equal intervals of
  $[0, \pi/2]$, and $\psi$ 20 equal steps of $[0, 2\pi)$; the ranges are ours. A point is kept
  when $(c/b)^2 \le \min((\nu_1/n)^2 - 4,\; 4)$: Lemma 3, with $c \le v_{\max}/n$ required for
  every $b$ up to $v_{\max}/(2n)$. That keeps 780 points at $2.15n$ and 2640 at $4n$.
- **Every phase.** The proof of Lemma 3 evaluates the speed limit at $\nu = \pi/2$, $\psi = 0$
  only, but Lemma 3 gives it at every phase. On the unit orbit, with $q = (\nu_1/n)^2$ and
  $Z = \tfrac32 e^{2i\nu} - \tfrac{c^2}{2} e^{2i\psi}$,
  $q|r|^2 - |v|^2/n^2 = (q - 4 - c^2) + (q + 1)(\tfrac32 + \tfrac{c^2}{2} + \mathrm{Re}(Z e^{2i\tau}))$,
  and $|Z| \le \tfrac32 + \tfrac{c^2}{2}$ bounds it below by $q - 4 - c^2 \ge 0$.
- **Target.** At a state $x$, $b = \min(\sqrt{x^2 + (y/2)^2},\; v_{\max}/(2n))$ and the target
  $x_{\text{des}}$ is the kept point scaled by $b$ nearest the position.
- **Command.** $u_b = \mathrm{clip}(K(x_{\text{des}} - x))$ on the box, with $K$ the LQR gain of
  the code's force weights, $K = R^{-1} B_F^T P / m$ with $B_F = B/m$ and $P$ from the continuous
  algebraic Riccati equation.
- **Rollout.** `rollout(x, k)` takes $k$ Euler steps of the closed loop. The target is chosen at
  the rollout's start and held, and moves by Euler free motion once the position comes within
  1.5 m of it. It is chosen again at every rollout start, so the backup keeps no memory between
  decisions and the implicit safe set is a set of states. The code's integral term and its
  reselection every 20 steps are not reproduced.
- **Sensitivities.** Along the rollout, $S_{j+1} = S_j + \Delta t\,(A S_j + B D_j K (T_j - S_j))$,
  $S_0 = I$, with $D_j$ zero on the saturated axes and $T_j$ the Jacobian of the target with
  respect to the start: $p\,\nabla b^T$ where $b$ is not capped, zero where it is, and multiplied
  by $I + A\Delta t$ at each step once the target moves. $S_j$ is the exact Jacobian of the Euler
  rollout between the switches of saturation, target and track radius.

## Filters

Every filter takes `setup=` (keyword only) and has `reset(view)`, `filter(obs, action)` returning a
`FilterResult`, `contains(x)` and `declaration`.

- **Commands.** The action is converted to m/s^2 and clipped onto the box; clipping is not an
  intervention. `intervened` is True when the returned acceleration differs from the clipped one
  by more than $10^{-6} a_{\max}$.
- **Refusals at reset** (`ValueError`): an action mode that is not continuous, a ball thrust limit,
  a frame other than "hill" or an observation that does not start with the state, a
  pending-commands channel, a dynamics part of the scenario ID other than "cw" (an ID without
  one, such as "dunlap2021-euler", is accepted), any constraint but one `speed_limit`, a keep-out
  included, and a view whose $n$, dt, `max_accel` or speed limit differs from the setup's by more
  than $10^{-12}$ relative.
- **Declaration.** Certificate None and `counts_toward_certification` False for all four; the
  safe set in words, with the class-K gains for the optimization filters; `barriers` holds the
  `AfrlConstraints`; buffer 0, no declared error sets. `vbar.safety.audit.audit_filter` refuses
  every one of them.

### Explicit switching

Eq. 4 with $C_S = C_A$: the clipped command passes when the Euler prediction
$x_1 = x + (Ax + Bu)\Delta t$ lies in $C_A$. Otherwise the backup sets each violated constraint to
zero at $x_1$ by inverting the Euler step at $x$ (sec. III.E): a velocity component beyond
$v_{\max}$ is put on $\pm v_{\max}$, then, if the speed limit is still violated, the predicted
velocity is scaled along itself to $\nu_0 + \nu_1 |r_1|$. The speed limit goes last, so it wins,
as the paper orders it. The paper does not say how the speed is reduced; the scaling is ours. The
backup is clipped onto the box, where it may stop short of the boundary.

### Explicit optimization

Eq. 7 and 10: the clipped command is projected with `vbar.solvers.project` onto

$$
\nabla\varphi_i(x)\,(Ax + Bu) + \alpha(\varphi_i(x)) \ge 0, \qquad i = 1..4,
$$

under the box. `rows(x)` returns them, named "speed_limit", "vx_limit", "vy_limit", "vz_limit".
Rows the box meets for every command ($b_i \le -a_{\max}|a_i|_1$) are dropped first, which leaves
the projection unchanged. The speed-limit gradient is that of $\varphi_1$ itself; the code's
gradient of the squared form is not used.

### Implicit switching

Eq. 4 with the implicit safe set of eq. 6, sampled: the clipped command passes when its Euler
step $x_1$ and the 5 backup steps from $x_1$ all lie in $C_A$, and is replaced by $u_b(x)$
otherwise. Every point of the horizon is checked; the authors' code checks only the one-step
state.

### Implicit optimization

Eq. 11, with its bracket corrected: for every constraint $i$ and every rollout point
$\phi_j$, $j = 0..5$, of the backup from $x$,

$$
\nabla\varphi_i(\phi_j)\, S_j\, B\,(u - u_b(x)) + \alpha(\varphi_i(\phi_j)) \ge 0,
$$

24 rows named "speed_limit@j" and so on, projected under the box as above. At $u = u_b(x)$ each
row reduces to $\alpha(\varphi_i(\phi_j)) \ge 0$, so the projection is feasible whenever the
rollout from $x$ stays in $C_A$.

### Outcomes

| Filter | `fallback_level` | `active` | `recovering` | `certificate` |
|---|---|---|---|---|
| Explicit switching | 0 command, 2 backup | backup used | $x \notin C_A$ | None |
| Explicit optimization | 0 (`solver_failed` returns `project`'s command) | a positive multiplier | $x \notin C_A$ | None |
| Implicit switching | 0 command, 2 backup | backup used | rollout from $x$ leaves $C_A$ | "finite-horizon", 5 s, when the applied command's step and the rollout after it stay in $C_A$ |
| Implicit optimization | 0, or 2 with $u_b(x)$ on `solver_failed` | a positive multiplier | rollout from $x$ leaves $C_A$ | as implicit switching, on feasible decisions |

When the rows of an optimization filter cannot all be met in the box, the result is
"infeasible_certified" with `project`'s least-violation command, the conflicting rows and an
`InfeasibilityProof` that `verify()` accepts, as for `CbfFilter`.

## Assumptions and limits

- **Not certified.** With the paper's $\nu_1 = 4n$ Lemma 1 fails under the force limit, so $C_A$ is
  not shown to be control invariant; at $2.15n$ the code's check passes by 0.02% and is not
  published; the plant is an Euler model; the implicit filters look 5 s ahead. Every declaration
  therefore has certificate None and does not count toward certification.
- **Samples only.** Every check and every "finite-horizon" certificate is on the Euler samples.
  Nothing is claimed between samples or for the exact Clohessy-Wiltshire step.
- **Eq. 11 corrected.** As printed, eq. 11 evaluates the closed loop at the backup point,
  $D(\phi_j)[f(x) + g(x)u - f(\phi_j) - g(\phi_j)u_b(\phi_j)]$, which is not the time derivative
  of $\varphi_i(\phi_j)$ along any flow. By the flow property the bracket is
  $f(x) + g(x)u - f(x) - g(x)u_b(x) = g(x)(u - u_b(x))$, the form of Chen, Jankovic, Santillo and
  Ames, "Backup Control Barrier Functions: Formulation and Comparative Study"
  (arXiv:2104.11332v1, sec. III.A), which these rows use.
- **No terminal row and no margin.** As in the paper, the implicit optimization filter has no row
  that ends the rollout in an invariant backup set and no margin for the motion between rollout
  samples (Chen et al. add a terminal row; Gurriet et al., IEEE Access 8, 187249, 2020, a
  margin). Its rows are continuous-time conditions on a sampled rollout; recursive feasibility is
  not shown.
  The authors' code constrains the points at 0 to 4 s; here all six points, 0 to 5 s.
- **Backup.** The target is memoryless between decisions, so the code's integral action and its
  reselection every 20 steps are left out.
- **Choices the paper leaves open**: the grid ranges, the scaling of the speed in the explicit
  switching backup, and the start directions. The published letter was not read (it is not open
  access); the arXiv version may differ from it.

## Validation

The tests in `tests/test_rta.py` check:

- The presets and the view; refusals of a zero mass, a negative $\nu_1$, a NaN step, a horizon
  that is not a whole number of steps, zero or negative class-K gains and a grid with an empty
  axis.
- Lemma 1 at $4n$ is 0.2860 m/s^2, above $1/12$; the code's check is 0.08332 m/s^2 at $2.15n$,
  below $1/12$, and 0.1049 m/s^2 at $4n$, above it.
- $\varphi$ equals eq. 16 and 19 written out on 20 states and $\alpha$ a hand value; the gradients
  match central differences to 1e-6 relative on 20 states.
- The grid equals the ellipse of sec. III.D built point by point from the paper's formulas on the
  Lemma 3 points, 780 at $2.15n$ and 2640 at $4n$; every kept point, scaled by 1 m, 100 m and the
  cap, returns to itself after $2\pi/n$ on `cw_states` to 1e-9 relative; every kept point meets
  Lemma 3 and $c/b \le 2$, meets $\nu_1 |r| \ge |v|$ at 3600 phases of its unit orbit at
  $2.15n$ and $4n$, has a nonnegative exact minimum of $\nu_1^2|r|^2 - |v|^2$ over phase,
  and stays in $C_A$ at 73 phases on `cw_states` at $b$ = 5 m and the cap.
- The backup gain equals the Riccati solution of the force weights divided by the mass, and the
  closed loop is stable.
- $S_j$ matches central differences of the rollout to 1e-6 relative with $b$ capped, with $b$ free
  and with the target moving from the first step, and with stiffer weights that saturate some
  axes and not others.
- Each implicit row's normal equals the central-difference gradient of $\varphi_i(\phi_j(x))$ times
  $B$ to 1e-6 relative, in the same three cases.
- On 150 seeded states whose rollout stays in $C_A$, $u_b(x)$ meets every implicit row, and each
  row's slack there equals $\alpha(\varphi_i(\phi_j))$ written out, to 1e-9 relative.
- Explicit switching passes a safe command unchanged, puts a violated velocity limit and a violated
  speed limit on the boundary at the next sample, and on a prediction of (10.08, 3, 0) m/s, where
  the order matters, puts $v_x$ on $v_{\max}$ before scaling to the speed limit.
- Explicit optimization meets every row on 40 seeded states, for a random command and for each
  box corner that does least for a row, and equals the projection onto all rows; its rows equal
  $\nabla\varphi(Ax + Bu) + \alpha(\varphi)$ by central differences; the speed-limit row is dropped
  at rest.
- Both optimization filters return "infeasible_certified" with a proof that verifies 2.6 m/s above
  the speed limit.
- Implicit switching uses the backup when the first two horizon points lie in $C_A$ and a later
  one does not, a state that a one-step check passes; it passes a command whose horizon stays in
  $C_A$ with the "finite-horizon" certificate.
- Implicit optimization is feasible and meets its rows on 30 seeded states inside its set; for
  both implicit filters `certificate` and `recovering` follow the rollouts, on a state inside and
  one at the edge of $v_{\max}$.
- The "accel", "thrust" and "normalized" modes give the same acceleration; the declarations; the
  reset refusals above for every filter; `audit_filter` refuses all four; a horizon of 8 s is
  refused by the implicit optimization filter (36 rows).

## Example

```python
import numpy as np

from vbar.safety.rta import (
    AfrlSetup,
    ExplicitOptimizationFilter,
    ExplicitSwitchingFilter,
    ImplicitOptimizationFilter,
    ImplicitSwitchingFilter,
)

setup = AfrlSetup.authors_code()  # the paper's values with nu1 = 2.15 n
view = setup.view()
print(round(setup.code_lemma_accel, 5), round(setup.a_max, 5))  # 0.08332 0.08333

x = np.array([5000.0, 0.0, 0.0, 9.99, 0.0, 0.0])  # m and m/s, 10 mm/s below v_max on x
push = np.array([setup.a_max, 0.0, 0.0])  # m/s^2, full thrust along x

switching = ExplicitSwitchingFilter(setup=setup)
switching.reset(view)
result = switching.filter(x, push)
print(result.action.round(5), result.fallback_level)  # [-0.00582  0.       0.     ] 2

optimization = ExplicitOptimizationFilter(setup=setup)
optimization.reset(view)
result = optimization.filter(x, push)
print(result.action.round(5), result.active)  # [-0.01582  0.       0.     ] True

implicit = ImplicitSwitchingFilter(setup=setup)
implicit.reset(view)
y = np.array([3000.0, 2000.0, -500.0, -1.0, 0.5, 0.2])
result = implicit.filter(y, np.array([0.02, -0.05, 0.01]))
print(result.intervened, result.certificate, result.certificate_horizon)  # False finite-horizon 5.0

rows = ImplicitOptimizationFilter(setup=setup)
rows.reset(view)
print(len(rows.rows(y).names), rows.declaration.counts_toward_certification)  # 24 False
```
