# Run time assurance filters

Code: `vbar.safety.rta` (`AfrlSetup`, `AfrlConstraints`, `NmtBackup`, `BackupRollout`,
`ExplicitSwitchingFilter`, `ExplicitOptimizationFilter`, `ImplicitSwitchingFilter`,
`ImplicitOptimizationFilter`, `start_state`, `reproduce`, `Reproduction`) and
`benchmarks/afrl_reproduction.py`.

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
$a\,\Delta t^2/2$ from the position that the exact step of the "cw" environments keeps. The plant
is `vbar.dynamics.EulerCWDynamics`, the `euler1` dynamics rung of [dynamics.md](dynamics.md), and
the scenario `docking-afrl-paper` flies it ([scenarios.md](scenarios.md)).

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
| `backup_q`, `backup_r` | $Q = 10^{-5} I_{12}$ on the state and its integral, $R = 10^7 I_3$ on the force in N | code |
| `track_radius` | 1.5 m | code |
| `primary_q`, `primary_r` | $Q = 0.05 I_6$, $R = 1000 I_3$ on the force in N | code |
| `stop_radius`, `stop_time` | 1 m, 5000 s | code |

`setup.view()` is the `AgentView` the filters reset from: commands in m/s^2 under a box of
$a_{\max}$, one `speed_limit` constraint with $v_0 = \nu_0$, $v_1 = \nu_1$, and `stop_time / dt`
steps. Its scenario is the registered one that flies the setup,
"vbar:docking-afrl-paper:euler1:box:dv:v1" (Gymnasium `vbar/DockingAfrlPaper-v1`) for
`AfrlSetup.paper()` and the same ID with ":code" (`vbar/DockingAfrlPaperCode-v1`) for
`AfrlSetup.authors_code()`. Any other setup keeps "dunlap2021-euler", which no environment flies.

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
  every $b$ up to $v_{\max}/(2n)$. That keeps 780 points at $2.15n$ and 2640 at $4n$. At $4n$,
  $\theta_1 = \pi/4$ with $\theta_2 = 0$ gives $(c/b)^2 = 4$ exactly, which sine and cosine round
  to either side depending on the platform, so a point within $10^{-12}$ of the bound counts as
  on it and is kept with $c/b$ set onto the bound.
- **Every phase.** The proof of Lemma 3 evaluates the speed limit at $\nu = \pi/2$, $\psi = 0$
  only, but Lemma 3 gives it at every phase. On the unit orbit, with $q = (\nu_1/n)^2$ and
  $Z = \tfrac32 e^{2i\nu} - \tfrac{c^2}{2} e^{2i\psi}$,
  $q|r|^2 - |v|^2/n^2 = (q - 4 - c^2) + (q + 1)(\tfrac32 + \tfrac{c^2}{2} + \mathrm{Re}(Z e^{2i\tau}))$,
  and $|Z| \le \tfrac32 + \tfrac{c^2}{2}$ bounds it below by $q - 4 - c^2 \ge 0$.
- **Target.** At a state $x$, $b = \min(\sqrt{x^2 + (y/2)^2},\; v_{\max}/(2n))$ and the target
  $x_{\text{des}}$ is the kept point scaled by $b$ nearest the position.
- **Command.** $u_b = \mathrm{clip}(K(x_{\text{des}} - x))$ on the box. The authors design their
  backup on the state augmented with the integral $z$ of the tracking error,
  $\dot z = x - x_{\text{des}}$, with $Q = 10^{-5} I_{12}$ and $R = 10^7 I_3$ on the force; $K$ is
  the block $K_1$ of that gain on $x$, divided by the mass, and the integral term itself is left
  out. The augmented system is not stabilizable (the velocity integral repeats the position), so
  the integrators get a leak, $\dot z = x - x_{\text{des}} - 10^{-6} z$, before the continuous
  algebraic Riccati equation is solved. The leak is an assumption; down to $10^{-8}$ 1/s it moves
  $K_1$ by under $2 \cdot 10^{-4}$ of its largest entry. The velocity gains are about 0.1 N per
  m/s.
- **Rollout.** `rollout(x, k)` takes $k$ Euler steps of the closed loop. The target is chosen at
  the rollout's start and held, and moves by Euler free motion once the position comes within
  1.5 m of it. It is chosen again at every rollout start, so the backup keeps no memory between
  decisions and the implicit safe set is a set of states. The code's integral term and its
  reselection every 20 steps are not reproduced.
- **Sensitivities.** Along the rollout, $S_{j+1} = S_j + \Delta t\,(A S_j + B D_j K (T_j - S_j))$,
  $S_0 = I$, with $D_j$ zero on the saturated axes and $T_j$ the Jacobian of the target with
  respect to the start: $p\,\nabla b^T$ where $b$ is not capped, zero where it is, and multiplied
  by $I + A\Delta t$ at each step once the target moves; `BackupRollout.target_jacobians` holds
  them. $S_j$ is the exact Jacobian of the Euler rollout between the switches of saturation,
  target and track radius.

## Filters

Every filter takes `setup=` (keyword only) and has `reset(view)`, `filter(obs, action)` returning a
`FilterResult`, `contains(x)` and `declaration`.

- **Commands.** The action is converted to m/s^2 and clipped onto the box; clipping is not an
  intervention. `intervened` is True when the returned acceleration differs from the clipped one
  by more than $10^{-6} a_{\max}$.
- **Refusals at reset** (`ValueError`): an action mode that is not continuous, a ball thrust limit,
  a frame other than "hill" or an observation that does not start with the state, a
  pending-commands channel, a dynamics part of the scenario ID other than "cw" or "euler1" (an ID
  without one, such as "dunlap2021-euler", is accepted), "euler1" with a setup dt other than the
  rung's 1 s step, any constraint but one `speed_limit`, a keep-out included, and a view whose
  $n$, dt, `max_accel` or speed limit differs from the setup's by more than $10^{-12}$ relative.
  A filter of `AfrlSetup.paper()` therefore accepts the views of `docking-afrl-paper` and refuses
  those of its `:code` variant, whose $v_1$ differs, and the other way round for
  `AfrlSetup.authors_code()`.
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

## Reproduction

`reproduce(filter, setup, x0)` flies the paper's run: from `x0`, each Euler step the paper's
primary gives a command, `vbar.safety.run_filter` passes it to the filter and times it, and the
returned command, clipped to the box, is applied, until $|r| \le$ `stop_radius` or `stop_time`.
The primary is "an unconstrained LQR controller" (sec. IV) with the code's weights
$Q = 0.05 I_6$, $R = 1000 I_3$ on the force and the code's clipping to 1 N: the paper's
controller, not a V-BAR baseline. `vbar.safety.PassThroughFilter` flies it alone. The result
holds the states, the primary and applied commands, a backup flag per step (`fallback_level` 2),
the `FilterResult`s and the filter's total wall time.

The plant is `EulerCWDynamics(setup.n, step_s=setup.dt)`, with $n = 0.001027$ rad/s exactly, and
the run is not flown through an environment: the environment's $n$ comes from the catalogue's
altitude and differs from 0.001027 by a few parts in $10^{15}$, its starts are drawn by its own
dispersions, and a scored reset refuses a given start, so the published numbers would move.
Replaying a run's applied commands in `docking-afrl-paper` from the run's start (training mode,
"accel" actions) gives its states within $10^{-6}$ m and $10^{-9}$ m/s. The environment ends an
episode within 1 m of the target at under 0.2 m/s; `reproduce` stops at $|r| \le 1$ m on
position alone.

The paper starts every run at $|r| = 9.85$ km and $|v| = 0.866$ m/s without giving directions.
`start_state(setup, seed)` puts the position uniformly on that sphere and the velocity in an
independent uniform direction, from a seed in $[0, 2^{128})$.

`benchmarks/afrl_reproduction.py` flies 100 starts, each from a 128-bit seed spawned from a
printed root seed, with the primary alone and inside each filter, at $2.15n$ and $4n$, and prints
the scenario ID of each slope. A sample
violates when $\varphi_1 < -10^{-5}$ m/s or some $|v_i| > v_{\max} + 10^{-5}$ m/s. At $2.15n$ it
prints thirteen criteria, each with its threshold and whether it is met. The paper's safety claim
(sec. III) becomes no violating sample for any filter, an intervention in at least 95 runs for each filter
and a violation in at least 95 runs for the primary alone. The behaviours it reports (sec. IV)
become min $\varphi_1 \ge 10^{-3}$ m/s in every explicit optimization run; a median of at least 10 switches between command and backup per
implicit switching run and at least 5 times that of explicit switching; and the two optimization
filters within 98.5 m and 0.02 m/s of each other over their common samples in at least 95 runs.
At $4n$ it prints the same counts. It prints each filter's per-run filter time over the mean of
explicit switching beside Table I. The script reports and does not gate: it exits with status 0
whether or not the criteria are met, and with status 1 only when the run is broken, that is when a
run's min $\varphi_1$, speed excess or filter time is not finite, or a compute table entry is not
finite and positive.

Two more cases from the authors' code are flown at $2.15n$ and reported: its
standard start, $(-5800, 6690, 4700)$ m at rest (main.py:40), and its random rule, position
uniform in $[-5000, 5000]$ m and velocity uniform in $[-1, 1]$ m/s per axis (main.py:34-37).

Measured with root seed 81998954318777406897742997449538877993 (1505 runs in 4459 s on 6 worker
processes). Runs that violate at least once, out of 100 (out of 1 for the standard start):

| | Paper rule, $2.15n$ | Worst $\varphi_1$ (m/s) | Worst $\lvert v_i\rvert - v_{\max}$ (m/s) | Paper rule, $4n$ | Code's start | Code's random rule |
|---|---|---|---|---|---|---|
| Explicit switching | 31 | $-0.021$ | $2 \cdot 10^{-15}$ | 17 | 0 | 3 |
| Explicit optimization | 71 | $-0.0063$ | $-0.045$ | 40 | 1 | 97 |
| Implicit switching | 61 | $-13.4$ | 14.4 | 48 | 0 | 49 |
| Implicit optimization | 60 | $-0.37$ | 0.28 | 60 | 0 | 33 |
| Primary alone | 100 | $-38.1$ | 28.2 | 100 | 1 | 100 |

Every filter intervenes in every run, and every filtered run reaches the stop radius. At $2.15n$
explicit optimization keeps min $\varphi_1 \ge 10^{-3}$ m/s in no run; implicit switching switches
a median 833 times per run against 2 for explicit switching; the two optimization filters stay
within 98.5 m and 0.02 m/s of each other in no run (largest gaps 311.7 m and 2.76 m/s). Six of the
thirteen criteria at $2.15n$ are not met. An independent
re-implementation of the four filters matches these runs, and finds:

- Explicit switching: the box clips the backup. The command that puts $\varphi_1$ on zero at the
  next sample lies outside the box, and the clipped one stops short; every first violation
  follows a clipped backup.
- Explicit optimization: the rows of eq. 10 are linear in the step, and on the Euler step the
  speed $|v + a\,\Delta t|$ exceeds its linearization by about $|a_\perp|^2 \Delta t^2 / (2|v|)$, so
  the filter rides the boundary a few mm/s outside it.
- Implicit optimization: the rows of eq. 11, as printed and as corrected, do not keep the set
  invariant without a row on the rollout's end; they stay feasible and tight on the last rollout
  point while a velocity component crosses $v_{\max}$.

- Implicit switching: the backup, an LQR clipped to the box that tracks a natural-motion point
  kilometres away, accelerates past the per-axis limit $v_{\max}$ on its way there; Lemma 3 covers
  the target orbit, not that transient. Most starts of the paper's rule therefore lie outside the
  implicit set $C_S$ of eq. 6, but the six-point check over 5 s cannot see an exit that comes later,
  so the agent's command is accepted up to the edge and the backup then crosses the limit. In all
  61 violating runs the first exit is under the backup after at least 6 backup steps (60 on a
  velocity component, 1 on the speed limit), and the last state that passed the check lies 5 steps
  earlier (4 in one run), with the backup rollout leaving $C_A$ one step past the horizon. A longer
  horizon (up to 100 s) or holding the target for 20 steps leaves the count at 59 to 61. From the
  authors' code start the backup rollout stays inside $C_A$ (lowest $\varphi_1$ +11.9 m/s, largest
  $|v_i|$ 7.97 m/s), which matches no violation there.

The authors' code uses different rows: for explicit optimization the gradient of the squared
speed limit $(\nu_0 + \nu_1|r|)^2 - |v|^2$ with $\alpha$ of the unsquared value, and for implicit
optimization rows of the form of Gurriet et al. at fixed offsets along the rollout. In the
independent check they stay within the limit, or far closer to it, on the same starts. They are
not reproduced because they are not the paper's equations.

Per-run filter time over the mean of explicit switching (mean and SD), beside Table I:

| Filter | $2.15n$ | $4n$ | Table I |
|---|---|---|---|
| Explicit switching | 1.00 (0.13) | 1.00 (0.11) | 1.00 (0.03) |
| Explicit optimization | 7.20 (0.97) | 8.15 (0.44) | 3.25 (0.13) |
| Implicit switching | 8.57 (0.93) | 11.38 (1.57) | 11.31 (1.05) |
| Implicit optimization | 25.14 (7.63) | 28.61 (7.30) | 13.33 (0.52) |

The runs shared 6 processes on a loaded machine, and run lengths differ between filters, so the
ratios follow run length and load as well as cost per decision.

## Assumptions and limits

- **Not certified.** With the paper's $\nu_1 = 4n$ Lemma 1 fails under the force limit, so $C_A$ is
  not shown to be control invariant; at $2.15n$ the code's check passes by 0.02% and is not
  published; the plant is an Euler model; the implicit filters look 5 s ahead. Every declaration
  therefore has certificate None and does not count toward certification.
- **Not safe on the Euler plant.** None of the four keeps $C_A$ in every run of the
  reproduction (section above); the paper's figures show one run each.
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
  reselection every 20 steps are left out; only the state block of its gain is kept, and the leak
  that makes that gain computable is an assumption.
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
- The backup gain equals the block $K_1$ of the Riccati solution on the augmented state with the
  leak of $10^{-6}$ 1/s, divided by the mass, to 1e-12; its velocity gains exceed 0.1 N per m/s;
  leaks of $10^{-7}$ and $10^{-8}$ 1/s move it by under $2 \cdot 10^{-4}$ of its largest entry;
  every mode of $A - BK$ decays faster than $3n$.
- $S_j$ matches central differences of the rollout to 1e-6 relative with $b$ capped, with $b$ free
  and with the target moving from the first step, and with stiffer weights that saturate some
  axes and not others.
- Each implicit row's normal equals the central-difference gradient of $\varphi_i(\phi_j(x))$ times
  $B$ to 1e-6 relative, in the same three cases.
- On 80 seeded states whose rollout stays in $C_A$, $u_b(x)$ meets every implicit row, and each
  row's slack there equals $\alpha(\varphi_i(\phi_j))$ written out, to 1e-9 relative.
- Explicit switching passes a safe command unchanged, puts a violated velocity limit and a violated
  speed limit on the boundary at the next sample, and on a prediction of (10.08, 3, 0) m/s, where
  the order matters, puts $v_x$ on $v_{\max}$ before scaling to the speed limit.
- Explicit optimization meets every row on 25 seeded states, for a random command and for each
  box corner that does least for a row, and equals the projection onto all rows; its rows equal
  $\nabla\varphi(Ax + Bu) + \alpha(\varphi)$ by central differences; the speed-limit row is dropped
  at rest.
- Both optimization filters return "infeasible_certified" with a proof that verifies 2.6 m/s above
  the speed limit.
- Implicit switching uses the backup at a state 6.4 km out of plane at 9.76 m/s toward it, where
  only the last of the six horizon points leaves $C_A$ (by more than 0.4 m^2/s^2, the others lie
  inside by more than 0.4), so any shorter check passes the command; it passes a command whose
  horizon stays in $C_A$ with the "finite-horizon" certificate.
- Implicit optimization is feasible and meets its rows on 30 seeded states inside its set, with
  `certificate` set exactly when the applied command's horizon stays in $C_A$.
- For both implicit filters `certificate` and `recovering` follow the rollouts in three cases, all
  in $C_A$: a state whose rollout stays inside (not recovering, certified); the out-of-plane state
  with no thrust (recovering, not certified); and the same state with full thrust away from
  $-v_{\max}$ on z (recovering and certified).
- `BackupRollout.target_jacobians` match central differences of the targets to 1e-6 relative,
  with $b$ free and with the target moving from the first step.
- `start_state` gives 9.85 km and 0.866 m/s exactly and the same start for the same seed, and
  bits of the seed above the 64th change it. Over 2000 seeds each direction component and the dot
  product of the two directions pass a Kolmogorov-Smirnov test for the uniform law on [-1, 1]
  (a uniform direction has uniform components) at p > 1e-4, and components above 0.9 in size
  number 1200 within five standard deviations, which normalized draws from a cube would miss. It
  refuses -1, $2^{128}$ and True.
- `reproduce` steps the Euler plant with the applied commands from 1 km on V-bar at 2.3 m/s
  inbound, where explicit switching passes the first command and replaces every later one: the
  primary equals the LQR of the force weights clipped to the box, every applied command equals a
  second filter's decision at that sample, the backup flags follow `fallback_level`, and the wall
  time is positive and the sum of the decisions' `compute_s`. It flies no step from a start on
  the stop radius, stops at the first state within 1 m from 5 m out, stops after `stop_time`,
  and refuses a start of five numbers or with a NaN.
- The "accel", "thrust" and "normalized" modes give the same acceleration; the declarations; the
  reset refusals above for every filter; `audit_filter` refuses all four; a horizon of 8 s is
  refused by the implicit optimization filter (36 rows).

`tests/test_rta_euler_scenario.py` checks that `reproduce` equals an Euler loop written out in the
test, bit for bit, at both slopes, and equals `EulerCWDynamics` stepped with the applied commands;
and that `docking-afrl-paper`, replaying the applied commands from the run's start, matches the
run within $10^{-6}$ m and $10^{-9}$ m/s over 600 steps of the primary alone at $4n$, and over
whole explicit switching runs at both slopes (marked slow). `tests/test_scenario_afrl_paper.py`
checks that the presets' views name the registered IDs and another setup keeps
"dunlap2021-euler", and that each filter accepts the environment's view with its own preset and
refuses the other slope's.

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
print(view.scenario_id)  # vbar:docking-afrl-paper:euler1:box:dv:v1:code
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

```python
import dataclasses

from vbar.safety import PassThroughFilter
from vbar.safety.rta import reproduce, start_state

short = dataclasses.replace(setup, stop_time=200.0)  # s
x0 = start_state(short, 2**100 + 7)
run = reproduce(ExplicitSwitchingFilter(setup=short), short, x0)
alone = reproduce(PassThroughFilter(), short, x0)
print(run.applied.shape, int(run.backup.sum()))  # (200, 3) 96
print(abs(alone.states[:, 3:]).max().round(3), abs(run.states[:, 3:]).max().round(3))  # 19.88 10.0
```
