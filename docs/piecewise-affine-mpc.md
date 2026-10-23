# Piecewise-affine MPC

Code: `mpqp_regions`, `Region`, `lookup`, `PiecewiseAffineConfig` and `PiecewiseAffineMPCAgent`
in `vbar.agents.mpc_pwa`. Hill frame ([frames.md](frames.md)), SI units.

A model predictive controller to a hold point whose optimization is done once, at reset. The
box-constrained regulator on the exact zero-order-hold CW model is solved for every state in a
box around the hold point, as a multiparametric quadratic program, and stored as a table of
polyhedral regions, each with an affine law. A decision is a table lookup and one affine
evaluation, with no solver in the loop. Where no thrust limit binds over the horizon it gives
the LQR command of [lqr.md](lqr.md); where limits bind it gives the first move of the
constrained plan, which clipping the LQR command does not.

## How it is built

**Regulator.** `regulator_model` (the design `LQRAgent` uses) gives the step $(\Phi, \Gamma)$,
the stage cost $(Q, R, S)$ on the deviations $\delta\mathbf x = \mathbf x - \mathbf x_s$ and
$\delta\mathbf u = \mathbf u - \mathbf u_s$ from the hold point at rest and its hold input, and
the DARE solution $P$. Over $N$ moves the controller minimizes

$$
\sum_{k=0}^{N-1}\left(\delta\mathbf x_k^T Q\,\delta\mathbf x_k + \delta\mathbf u_k^T R\,
\delta\mathbf u_k + 2\,\delta\mathbf x_k^T S\,\delta\mathbf u_k\right) +
\delta\mathbf x_N^T P\,\delta\mathbf x_N
\quad\text{s.t.}\quad
-a_{\max} - u_{s,i} \le \delta u_{k,i} \le a_{\max} - u_{s,i},
$$

so the limit $|u_{k,i}| \le a_{\max}$ applies to the absolute command. With no binding limit
the first move is $-K\,\delta\mathbf x_0$ for every $N$, the LQR law, since $P$ is the
infinite-horizon cost-to-go (dynamic programming).

**Decoupling.** The CW model and the diagonal weights of `LQRConfig` split exactly into the
in-plane subsystem ($x, y, v_x, v_y$ driven by $a_x, a_y$) and the cross-track one ($z, v_z$
driven by $a_z$), and the box limit is per axis. Each subsystem is condensed separately into

$$
\min_{U}\ \tfrac12 U^T H U + (F\,\delta\mathbf x)^T U
\quad\text{s.t.}\quad G U \le \mathbf w + S_x\,\delta\mathbf x,
$$

with $U$ the stacked moves, $G = [I; -I]$, $\mathbf w$ the limits above and $S_x = 0$: two
parametric problems with 4 and 2 parameters instead of one with 6.

**Regions.** For a set $A$ of active rows the minimizer is affine in the parameter, $U = K_A
\delta\mathbf x + \mathbf k_A$, and so are the multipliers $\boldsymbol\lambda_A$. The states
where $A$ is optimal form the critical region: the inactive rows hold at $U$ and
$\boldsymbol\lambda_A \ge 0$, intersected with the parameter box (Bemporad, Morari, Dua and
Pistikopoulos, "The explicit linear quadratic regulator for constrained systems", *Automatica*
38(1), 2002; Rawlings, Mayne and Diehl, *Model Predictive Control: Theory, Computation, and
Design*, 2nd ed., sections 7.3 and 7.4). `mpqp_regions` builds them as
follows.

- The law of a set is solved on the free inputs: the active rows are solved for as many inputs
  as there are rows (chosen by QR with column pivoting; for limit rows, the inputs at their
  limits), and the cost is minimized over the rest by a Cholesky solve of the reduced Hessian.
  An input at its limit then takes the limit exactly. The multipliers follow from stationarity,
  $H U + F\delta\mathbf x + G_A^T\boldsymbol\lambda_A = 0$, on the solved-for inputs.
- A set whose rows are linearly dependent has no unique multipliers and is skipped.
- A region is kept when its Chebyshev ball, in coordinates where the box is $[-1, 1]$ on every
  axis, has a radius above $10^{-9}$ (`scipy.optimize.linprog`, HiGHS). Sets optimal only on a
  lower-dimensional set, such as two rows meeting at a point, are dropped.
- Sets are explored breadth first from the empty set, and a set is extended by one row only
  when its own region is kept. The regions come out in that order, by number of active rows.

Each `Region` holds `active`, `gain` and `offset` (the whole plan $U$), and the rows `a`, `b`
of $\{\delta\mathbf x : a\,\delta\mathbf x \le b\}$: the inactive rows of $G$, the multipliers,
then the box. Rows that a saturated region cannot violate are kept as rows of zeros.

**Lookup.** `lookup(regions, x)` tests the regions in order and returns the first whose rows hold
to $10^{-12}$ of their magnitudes ($|b_i| + \sum_j |a_{ij} x_j|$), with the number of regions
tested; outside the box it returns `None` after testing all of them. The tolerance lets a state
on a shared facet be found despite the rounding of the rows; the law is continuous, so either
region gives the same command there.

**Agent.** `PiecewiseAffineMPCAgent.reset` builds both tables for the view's orbit, step, thrust
and hold point. Each decision looks up the in-plane and the cross-track deviation and applies
$\mathbf u_s$ plus the first move of each region's law, clipped to $a_{\max}$ against rounding.
If either deviation lies outside the box the command is the LQR law $\mathbf u_s - K\delta
\mathbf x$ clipped per axis, as `LQRAgent` flies it.

**Log.** `agent.log` holds one entry per decision: `time_s`; `level`, 0 for a region's law and 2
for the clipped LQR law outside the box; `regions`, the in-plane and cross-track region indices,
or `None` at level 2; `visits`, the regions tested; `elapsed_s`, the lookup's wall time; and
`fallback`, true at level 2. `agent.fallbacks` counts decisions by level (level 1 is unused).
Both are the agent's own bookkeeping, which no metric reads as a score.

## Parameters and defaults

`PiecewiseAffineConfig` is a frozen, keyword-only dataclass.

| Parameter | Default | Reason |
|---|---|---|
| `regulator` | `LQRConfig()` | The weights `LQRAgent` uses, so the two agree wherever no limit binds; its `rate_weight` must be 0 |
| `horizon` | 2 | Regions are rebuilt at every reset: two moves build in under a second in the cases measured below, three take 1.5 to 2.2 s |
| `box` | `(600, 800, 700, 1, 1, 1)` m and m/s | Half widths around the hold point that hold the start dispersions of `approach-rbar` (500 m radially), `approach-vbar` (700 m along track) and `approach-hbar` (600 m cross track) with at least 100 m to spare, and ten times the 0.1 m/s per axis of their start velocities |

Fixed in the module: the Chebyshev radius $10^{-9}$ of the scaled box, far below the smallest
kept region of the tested cases and the default tables ($5\times10^{-5}$) while the dropped sets
measure 0; the lookup tolerance $10^{-12}$, above the
rounding of the rows (near $10^{-16}$ of them) and below the $10^{-12}\,a_{\max}$ the tests
allow on the command.

## Assumptions and limits

- CW motion on a circular target orbit at the view's mean motion. On two-body or J2 truth the
  laws are those of the CW model.
- Box thrusters only. A ball limit couples the three axes, so the problem neither splits into
  the two subsystems nor keeps linear rows; `reset` refuses it with a `ValueError`.
- Convex limits on the inputs only. The keep-out sphere is not convex and is not modelled; the
  other path constraints are not modelled either.
- No command delay and no rate weight. The regulator is not augmented with past commands, so
  `reset` refuses a view whose observation carries pending commands and a `regulator` with a
  positive `rate_weight`, each with a `ValueError`. A delay that the observation does not
  announce is not modelled, as for `LQRAgent`.
- Holds and berthings only: other tasks and goals without a hold point are refused, as are the
  `"disc27"` and `"impulse_coast"` action modes.
- The breadth-first search reaches a set only through sets with nonempty regions. On every case
  in the tests the regions cover the box, and the counts equal an independent enumeration of
  every active set.
- The table grows quickly with the horizon, and so does the build. Measured once on one machine
  under varying load, at the default weights and box (500 km circular orbit of the WGS 84 Earth,
  10 s steps, 0.02 m/s$^2$ per axis), for the views of `approach-vbar`, `approach-rbar` and
  `approach-hbar`; the counts were the same for the three holds and for the box
  `(300, 600, 300, 1, 1, 1)` at one and two moves:

  | Horizon | In-plane regions | Cross-track regions | Build time per reset |
  |---|---|---|---|
  | 1 | 9 | 3 | 0.04 to 0.12 s |
  | 2 | 25 | 5 | 0.2 to 0.8 s |
  | 3 | 49 | 7 | 1.5 to 2.2 s |
  | 4 | 81 | 9 | 5.0 to 5.3 s |
  | 5 | 121 | 11 | 9.3 to 10.5 s |

  About two thirds of the build at three moves went to the Chebyshev LPs, one per candidate
  set with independent rows. With the heavier weights of the
  region-count tests below the tables are larger: at $2\times10^{-3}$ m/s$^2$ the cross-track
  axis has 33 regions at four moves and the in-plane axes 81 at two.
- The lookup is sequential, so its cost grows with the region count. In nine default episodes
  (seeds 0, 1 and 2 of `approach-vbar`, `approach-rbar` and `approach-hbar`, measured once) every
  decision was at level 0, at most 18 regions were tested in one lookup (`approach-vbar`, seed 2;
  at most 8 on `approach-rbar` and 6 on `approach-hbar`), and the median lookup took 0.1 to
  0.2 ms.

## Validation

Tests: `tests/test_mpc_pwa.py`. References marked independent come from a bounded-variable
least-squares solve of the same QP (`scipy.optimize.lsq_linear`), from trying every face of the
input box, or from an enumeration of every active set, not from the module.

- Rawlings, Mayne and Diehl, Example 7.11 ($V = x^2/2 - ux + u^2$ with $u \ge 1$,
  $u \ge 2 - x/2$, $u \ge 2 - x$): three regions, with the second row active on $[0, 2]$ and
  $u = 2 - x/2$ to $10^{-12}$; the first row alone, optimal only at $x = 2$, is dropped, and
  every pair of rows is dependent (the rows bound the same input) and skipped.
  The law equals $\max(x/2, 1, 2 - x/2, 2 - x)$ on a $10^{-3}$ grid over $[-1, 4]$ to
  $10^{-12}$.
- Example 7.12 ($H = [[3, 1], [1, 2]]$, linear term $(2x, x)$, $|u_i| \le 1$): five regions
  mirrored about zero, with only $u_1$ at its lower limit on $[5/3, 3]$ and
  $u = (-1, 1/2 - x/2)$, to $10^{-12}$; the law equals the independent face enumeration on a
  $10^{-3}$ grid over $[-4, 4]$ to $10^{-12}$.
- Section 7.2, $\min a + bu + cu^2/2$ over $|u| \le 1$ with $b$ the parameter: the values on the
  saturated pieces are $a - b + c/2$ and $a + b + c/2$ to $10^{-14}$ relative, for
  $c = 0.5, 2, 3$ and 200 draws each. The book prints $c^2/2$ there, which these values of $c$
  would expose.
- CW region counts (ours; 500 km, 10 s steps, $|a| \le 2\times10^{-3}$ m/s$^2$, stage weights
  $\mathrm{diag}(1, 10^3)$ cross-track and $\mathrm{diag}(1, 1, 10^3, 10^3)$ in-plane,
  $R = 10^4$ per input, DARE terminal cost, box $|x| \le 300$ m, $|y| \le 600$ m, $|z| \le 300$
  m, $|v| \le 1$ m/s): cross-track 9 and 33 regions at two and four moves, in-plane 9 and 81 at
  one and two, equal to the independent enumeration. The regions come breadth first from the
  empty set. At 200 random states the law equals the independent QP solve to
  $10^{-12}\,a_{\max}$ (measured $5.9\times10^{-15}$ cross-track and $6.5\times10^{-16}$
  in-plane), every state lies in exactly one region, and the lookup returns the first region
  containing it after `index + 1` visits.
- Boundaries: states outside the box find no region after visiting all; states where regions
  meet (Example 7.11 at 0 and 2, Example 7.12 at $\pm5/3$ and $\pm3$, and the box edges) go to
  the first region they touch and take the QP minimizer; points on every facet of the CW tables
  are found and take the minimizer to $10^{-12}\,a_{\max}$.
- Agent: at the V-bar and R-bar holds at rest the command is the hold input (zero, and
  $3.7\times10^{-4}$ m/s$^2$ radially) to $10^{-12}\,a_{\max}$; for offsets under a metre it
  equals `LQRAgent` to $10^{-12}\,a_{\max}$ at level 0; at 50 states drawn over three decades of
  the box it equals the first move of the coupled six-state QP, solved independently, to
  $10^{-12}\,a_{\max}$ (measured $1.4\times10^{-15}$), with some draws saturating an input and
  some not; 700 m along track, beyond the box, it equals the clipped `LQRAgent` command and logs
  level 2.
- Refusals: a ball limit, the discrete and impulsive modes, tasks other than a hold, a hold goal
  without a hold point, an empty horizon and a degenerate box.
- Agent contract: equal commands in fractions in the three continuous action modes, reset
  clears the previous episode, two agents from one configuration on interleaved environments
  match separate runs, the agent runs on a hand-made view and reads only `time_s`, and the
  recorder keeps one log entry per decision.

## Usage

```python
import numpy as np

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.mpc_pwa import PiecewiseAffineConfig, PiecewiseAffineMPCAgent
from vbar.agents.mpc_pwa import lookup, mpqp_regions

# Rawlings, Mayne and Diehl, Example 7.12.
regions = mpqp_regions(
    h=np.array([[3.0, 1.0], [1.0, 2.0]]),
    f=np.array([[2.0], [1.0]]),
    g=np.vstack([np.eye(2), -np.eye(2)]),
    w=np.ones(4),
    s=np.zeros((4, 1)),
    bounds=[(-4.0, 4.0)],
)
index, visits = lookup(regions, np.array([2.0]))
region = regions[index]
print(len(regions), region.active, region.gain @ [2.0] + region.offset)

agent = PiecewiseAffineMPCAgent(PiecewiseAffineConfig(horizon=2))
env = ApproachEnv(ApproachConfig(start_state=(0.0, -400.0, 30.0, 0.0, 0.0, 0.0), max_steps=20))
run_episode(env, agent, seed=1, agent_seed=2)
print(agent.fallbacks, agent.log[0]["regions"], agent.log[0]["visits"])
```
