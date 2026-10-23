# Fuel-optimal MPC

Code: `fuel_plan`, `FuelPlan`, `FuelMPCConfig` and `FuelMPCAgent` in `vbar.agents.mpc_fuel`.
Hill frame ([frames.md](frames.md)), SI units.

A model predictive controller whose only cost is propellant: at every decision it plans the
thrust sequence with the least 1-norm delta-v that brings the chaser to rest at the hold point
at a set time, on the exact zero-order-hold CW model, and applies the first move. It is the
fuel baseline for the hold-point scenarios with box thrusters, and the reference for how far a
receding-horizon fuel controller drifts from the open-loop optimum.

## How it is built

**Plan.** With the moves $\mathbf u_k$ held over steps of $\Delta t$, the state after $N$ steps
is $\mathbf x_N = \Phi^N\mathbf x_0 + G\mathbf u$, $G = [\Phi^{N-1}\Gamma, \dots, \Phi\Gamma,
\Gamma]$, with $(\Phi, \Gamma)$ from `cw_discrete`. `fuel_plan(n, dt, x0, x_goal, steps, a_max)`
solves

$$
\min_{\mathbf u}\ \sum_{k=0}^{N-1} \Delta t\left(\|\mathbf u_k\|_1 +
\frac{\tau}{2a_{\max}}\|\mathbf u_k\|_2^2\right)
\quad\text{s.t.}\quad G\mathbf u = \mathbf x_\text{goal} - \Phi^N\mathbf x_0,\qquad
|u_{ki}| \le a_{\max}.
$$

Without the quadratic term this is the 1-norm regulator, a linear program (Rawlings, Mayne and
Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., 2022, exercise
2.19). Its minimizer can be a whole face (section 7.7.1 and Fig. 7.7): on the cross-track case
below, HiGHS returns plans whose first moves are 0 or $-2.54\times10^{-4}$ m/s$^2$ depending on
the order of the variables, so the applied command would depend on the solver. The term with
weight $\tau$ (`tie_break`) makes the cost strictly convex and the plan unique. Since
$u_{ki}^2 \le a_{\max}|u_{ki}|$, the exact minimizer of the regularized problem costs at most
$\tau p^*/2$ more than the linear program's optimum $p^*$; a computed plan is not held to that
figure, and near the goal it can exceed it by far.

**Fuel guarantee.** The guarantee is checked at run time. For any multipliers $\boldsymbol\nu$
the linear program's dual value $d(\boldsymbol\nu) = \mathbf r\cdot\boldsymbol\nu -
a_{\max}\sum_i \max(0, |\mathbf g_i^T\boldsymbol\nu| - \Delta t)$ is at most $p^*$ (weak
duality). `FuelPlan.fuel_lower` is $d(\boldsymbol\nu)$ at the plan's multipliers less a bound on
the rounding of its terms, so a converged plan's fuel exceeds $p^*$ by at most
$\text{fuel} - \text{fuel\_lower}$. At every converged decision the agent recomputes the plan's
fuel, logs `fuel_lower_mps` and `fuel_excess_bound` $= \text{fuel}/\text{fuel\_lower} - 1$, and
sets `fuel_bound_certified` to false, with no bound, when the lower bound is not positive or
exceeds the fuel. At the regularized optimum the gap is at most $\tau$ times the fuel of the
moves inside the band, so the relative bound is of the order of $\tau$ while the moves stand well
above the rounding of the prices: on the 60-step transfer from $y = -160$ m the tests assert at
most $2\times10^{-6}$, and it measures $6.2\times10^{-7}$ there, $4.5\times10^{-7}$ on the
half-orbit transfer and $9.6\times10^{-7}$ on the cross-track case. Close to the goal the moves
fall to the rounding of the prices and the plan itself can cost more: on the test's sign case,
$10^{-6}$ m from the goal over 284 steps, the plan costs $2.5806668\times10^{-9}$ m/s, 1.86%
above a feasible plan of $2.5335205\times10^{-9}$ m/s, and the relative bound it logs is
$1.89\times10^{-2}$, which the test checks against an independent lower bound on the optimum.

**Dual.** For multipliers $\boldsymbol\nu$ of the six terminal equalities, each move minimizes
$\Delta t(|u| + \tau u^2/(2a_{\max})) - w u$ with $w = \mathbf g_i^T\boldsymbol\nu$, the primer
of that move, in closed form:

$$
u_i(\boldsymbol\nu) = \operatorname{sign}(w)\,\min\!\left(a_{\max},\
\frac{a_{\max}}{\tau}\max\!\left(0, \frac{|w|}{\Delta t} - 1\right)\right).
$$

A move is zero while its primer is below the fuel price $\Delta t$, grows linearly over a band
of width $\tau\Delta t$ above it, and saturates at $a_{\max}$. The dual function of
$\boldsymbol\nu$ is therefore concave and continuously differentiable with a piecewise-linear
gradient $\mathbf r - G\mathbf u(\boldsymbol\nu)$, and the plan is found by maximizing it over
six variables:

- Newton steps on the generalized Hessian $(a_{\max}/(\tau\Delta t))\,G_B G_B^T$ over the moves
  $B$ strictly inside the band, with a Levenberg term of $10^{-12}$ of its trace plus the
  residual norm where it is singular (fewer than six independent moves inside), on rows of $G$
  scaled to unit norm;
- an exact line search: along a direction the gradient's slope is piecewise linear and
  decreasing, so its root is found between two breakpoints of the moves by bisection over the
  sorted breakpoints and one interpolation. A slope that stays positive past the last
  breakpoint proves the problem infeasible.

**Start.** A cold start begins at the multipliers of the linear program without the tie-break,
solved by the HiGHS dual simplex (`scipy.optimize.linprog`) for a right-hand side scaled to unit
norm, which leaves the multipliers unchanged and keeps the solver's absolute tolerances
relative to the plan. From there the plans of the test cases converge in one to four Newton
steps; from zero multipliers the half-orbit plan below took 50 and the one-orbit plan more
than 300. HiGHS reporting infeasibility ends the plan with status `"infeasible"`. A warm start
(`warm`, the previous plan's multipliers) gets a quarter of the iterations; if it has not
converged, the cold start takes the rest. `iterations` counts updates of the multipliers: Newton
steps, and one for the jump to the linear program's multipliers.

**Stop rule.** The plan stops when $\|G\mathbf u - \mathbf r\| \le 10^{-10}(\|\mathbf r\| +
\|G\|_F\|\mathbf u\|)$, or at `max_iter` with status `"iteration_cap"`. The closed form
amplifies the rounding of each primer by $a_{\max}/\tau$; at $\Delta t$ = 600 s the along-track
column of $G$ reaches $2.2\times10^7$ m per m/s$^2$ and the moves alone stop at a relative
residual of $2.4\times10^{-7}$, so the rule could not be met. Two corrections make it reachable:

- When the full Newton step keeps every move on its piece of the closed form, the step is exact
  on that piece, and it is taken in the moves directly: the moves inside the band receive the
  minimum-norm change onto the terminal rows, $G_B^T(G_BG_B^T)^+(\mathbf r - G\mathbf u)$.
- Moves below about $\epsilon\,a_{\max}/\tau = 4.4\times10^{-12}$ m/s$^2$ (a plan from within
  about $10^{-5}$ m of the goal) sit inside the rounding of $|w|/\Delta t - 1$, so their pieces
  jitter. The same minimum-norm correction over the moves at the fuel price up to that rounding
  is accepted when every corrected move keeps the sign of its primer and is optimal at a price
  within the rounding of its own: it is then the exact plan for prices that differ by rounding
  alone.

**Horizons.** `FuelMPCAgent` plans from each observation to rest at the scenario's hold point.

- `"receding"`: every decision plans to arrive one horizon later, so the planned arrival moves
  with time and the controller is economic MPC: with a stage cost that is not positive definite
  a terminal equality gives an average-cost bound only, and stability needs strict
  dissipativity (Rawlings, Mayne and Diehl, section 2.8, Proposition 2.52 and Theorem 2.56). The
  arrival is not the horizon.
- `"shrinking"`: the arrival is fixed one horizon after the first decision and each decision
  plans for the steps left. With a unique plan the tail of an optimal plan is the optimal plan
  from where it leads, so on the model the controller flies its first plan and arrives on time.
  With one step left it applies the plan's last move, and after the arrival the hold input
  $\mathbf u_s$, which holds the hold point under the step (zero on the V-bar).

**Fallback.** An unconverged plan is never applied. The agent then flies its previous plan
shifted by one step, with $\mathbf u_s$ appended once it runs out, which is feasible in both
modes since the goal is an equilibrium; with no previous plan it commands zero thrust. The
planner's iteration cap is the only budget, so runs reproduce across machines.

**Log.** `agent.log` holds one entry per decision: `time_s`; `status` (`"converged"`,
`"iteration_cap"` or `"infeasible"` from the planner, `"last_move"` with one step left,
`"after_arrival"` after a shrinking horizon); `iterations`; `elapsed_s`, the planner's wall time;
`level`, the source of the command (0 a plan solved at this decision, 1 the previous plan
shifted, 2 zero thrust); `fallback`, true when a plan was solved and did not converge;
`warm_start`; `solver` and `tolerance`; `planned_arrival_s`; `flown_arrival_s`, the time of the
first observation within 1 cm of the hold point, `None` before it; `planned_fuel_mps`;
`fuel_lower_mps`, `fuel_excess_bound` and `fuel_bound_certified`, the fuel guarantee above; and
`predicted_margin` and `certified_bound` (m), the lowest keep-out margin of the flown plan from
the observation at 20 samples per step, as the environment checks, and a lower bound on it
between samples, the first-order bound of `vbar.constraints.segment_minimum` ($-\infty$ where
the samples are too far apart). The recorder copies the log, and the agent metrics read its
solver fields (`iterations`, `status`, `warm_start`, `fallback`, `solver`, `tolerance`) as the
agent reports them. `agent.fallbacks` counts the decisions by level; it is the agent's own
bookkeeping, which neither the recorder nor any metric reads.

## Parameters and defaults

`FuelMPCConfig` is a frozen, keyword-only dataclass.

| Parameter | Default | Reason |
|---|---|---|
| `mode` | `"receding"` | The usual MPC form; `"shrinking"` is the variant that flies the open-loop optimum |
| `horizon_s` | `None`: half the orbit period, rounded to whole steps (2840 s at 500 km and 10 s) | The receding case measured below; a full orbit arrives closer to its horizon at twice the planning work per decision |
| `tie_break` | $10^{-6}$ | Fuel guarantee measured at about $10^{-6}$ of the optimum away from the goal, while rounding in the moves, amplified by $a_{\max}/\tau$, stays at $5\times10^{-9}\,a_{\max}$ |
| `max_iter` | 100 | In the tests cold plans take two to five updates and converging warm ones up to 23; a stalled warm start leaves 75 for the cold start |

Fixed in the module: the stop rule $10^{-10}$, six decades above the $4\times10^{-16}$ that
converged plans of 284 and 568 steps reach; the arrival radius 1 cm, a
fiftieth of the smallest capture radius in the catalogue (0.5 m); 20 keep-out samples per step,
the environment's rate.

## Assumptions and limits

- CW motion on a circular target orbit. On two-body or J2 truth each plan misses by the model
  error and the receding controller corrects it at the next decision; the shrinking controller
  re-plans too, but its arrival time is fixed, so close to it a perturbed state can have no
  plan within the box, and the agent then flies its previous plan.
- Box thrusters only. Under a ball limit the 2-norm fuel is a second-order cone program, and
  `reset` refuses a ball view with a `ValueError`. The `"disc27"` and `"impulse_coast"` action
  modes are refused, as are goals other than a hold or a berthing and goals without a hold point.
- Constraints are not modelled: the plan may cross the keep-out sphere, and the logged margins
  only report it. Command delay and pending commands are not modelled either.
- No stability theorem for `"receding"` (economic MPC). Measured on the exact CW model from rest
  at $y = -500$ m to the hold point at $y = -100$ m (500 km circular orbit of the WGS 84 Earth,
  10 s steps, 0.02 m/s$^2$ per axis), by `tests/test_mpc_fuel.py`:

  | Horizon | Arrival within 1 cm | Delta-v | Open-loop optimum for that arrival | Excess |
  |---|---|---|---|---|
  | 284 steps (half an orbit) | 401 steps, 0.7064 orbit | 0.170431 m/s | 0.135625 m/s | 26% |
  | 568 steps (one orbit) | 569 steps, 1.0023 orbit | 0.047112 m/s | 0.0469733 m/s | 0.3% |

  The shrinking horizon of 284 steps arrives at step 284 with 0.222907 m/s, the open-loop
  optimum. A comparison with the fuel oracle has to use the controller's actual arrival time.
- The plan's uniqueness is numerical: two formulations of one problem agree to
  $4.7\times10^{-9}\,a_{\max}$, set by rounding in the moves amplified by $a_{\max}/\tau$.
- Within about $10^{-5}$ m of the goal the moves fall below what the multipliers resolve. The
  rounding correction converges for most such plans; the rest end at the cap and the agent flies
  its previous plan. In the one-orbit run above this happened at 1 of 571 decisions, after the
  arrival.
- Cost: one linear program per cold start (`scipy.optimize.linprog`) and a few Newton steps of
  $O(N)$ work per decision. Measured on one shared machine under varying load: 425 decisions
  with a 284-step horizon in 5 to 9 s, 571 decisions with a 568-step horizon in 17 to 24 s.

## Validation

Tests: `tests/test_mpc_fuel.py`. Reference values marked HiGHS come from
`scipy.optimize.linprog` on the split-variable linear program, a path separate from the dual
solver.

- Cross-track case (ours, after Rawlings, Mayne and Diehl, section 7.7.1): start at the node with
  0.03 m/s out of plane, step one 96th of the period, 150 steps, $2\times10^{-3}$ m/s$^2$ per
  axis, terminal state at rest. HiGHS gives 0.0300214291 m/s; the plan's fuel lies in
  $[p^* - 10^{-10},\ p^*(1 + 5\times10^{-7}) + 10^{-10}]$ m/s and ends within $10^{-9}$ of the
  motion's scale of rest.
- Uniqueness, the reproducibility check that permuted rows give the same action: the
  time-reversed problem, with the stages reversed and the
  along-track axis reflected, returns the reversed plan to $10^{-6}\,a_{\max}$ on the
  cross-track case and on it with an in-plane offset, where HiGHS vertices differ by
  $0.127\,a_{\max}$.
- Half-orbit transfer above: the plan's fuel equals the HiGHS optimum, 0.22291 m/s, within
  $5\times10^{-7}$ of it, and arrives to $10^{-9}$ of the 400 m span. Plans at steps of
  $10^{-3}$ s and 600 s arrive to $10^{-9}$ of their span.
- Shrinking horizon: a 60-step transfer from $y = -160$ m (and, marked slow, the 284-step one)
  flies the open-loop optimum within $5\times10^{-7}$ of it and arrives at its planned step, not
  one step early. Receding horizon: the 60-step controller is still more than 1 m away when its
  first horizon ends; the slow tests reproduce the table above, the receding-horizon figures
  of economic MPC (Rawlings, Mayne and Diehl, section 2.8): arrival at 0.706 orbit with
  0.17043 m/s, 26% above the open-loop optimum for that arrival, for half an orbit, and at 1.002
  orbits with 0.04711 m/s for one.
- Refusals: one-step plans (three moves cannot meet six terminal equalities), a zero or negative
  tie-break, a ball limit, discrete and impulsive modes, tasks other than a hold.
- Agent contract: equal commands in fractions in the three continuous action modes, reset
  clears the previous episode, two agents from one configuration on interleaved environments
  match separate runs, the agent runs on a hand-made view and reads only `time_s`, an
  unconverged plan is never applied and the shifted previous plan replaces it, and the recorder
  keeps one log entry per decision.

## Usage

```python
import numpy as np

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.mpc_fuel import FuelMPCAgent, FuelMPCConfig, fuel_plan
from vbar.dynamics import EARTH, mean_motion

n = mean_motion(EARTH.radius + 500e3)
start = np.array([0.0, -160.0, 0.0, 0.0, 0.0, 0.0])
goal = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
plan = fuel_plan(n, 10.0, start, goal, 60, 0.02)
print(plan.status, f"{plan.fuel:.6f} m/s", plan.u[0])

agent = FuelMPCAgent(FuelMPCConfig(mode="shrinking", horizon_s=600.0))
env = ApproachEnv(ApproachConfig(start_state=(0.0, -300.0, 0.0, 0.0, 0.0, 0.0), max_steps=20))
out = run_episode(env, agent, seed=1, agent_seed=2)
print(agent.log[0]["planned_arrival_s"], agent.fallbacks)
```
