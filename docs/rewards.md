# Rewards

Code: `vbar.rewards` (`Reward`, `Transition`, `QuadraticCost`). A reward is a training signal for
reinforcement learning, never the benchmark score; scored runs never pass it to an agent. The
library builds rewards from terms that each serve one benchmark metric, integrated over the
step in continuous time and discounted per second, so a learner that maximizes the return is
pushed toward what the metrics measure.

## Discounting per second

A step of duration $\Delta t_k$ is discounted by

$$
\gamma_k = e^{-\Delta t_k / \tau}
$$

with the time constant $\tau$ (s) given as `Reward(tau=...)`, or $\gamma_k = 1$ when `tau` is
`None`. Every episode ends at the time limit, so the undiscounted return is finite. A rate
$r(t)$ held over the step contributes $\int_{t_k}^{t_{k+1}} e^{-(t - t_k)/\tau} r(t)\,dt$ to the
step's reward and an event at time $t$ in the step contributes its size times
$e^{-(t - t_k)/\tau}$. The return $G = \sum_k \Gamma_k R_k$, with $\Gamma_k = \prod_{j<k}\gamma_j$,
is then $\int_0^T e^{-t/\tau} r(t)\,dt$ plus the discounted events, whatever the decision
interval: the semi-Markov return of Sutton and Barto, *Reinforcement Learning: An Introduction*
(2nd ed. draft, 2015), sec. 14.4, after Bradtke and Duff (1995). With a fixed per-step $\gamma$,
halving $\Delta t$ would double the effective horizon. $\tau$ = 3600 s gives $\gamma$ = 0.99723
at 10 s steps and 0.99861 at 5 s.

`Reward.discount(duration)` returns $\gamma_k$; a learner should discount each transition by it.

## Terms

All weights default to 0 (off) and must not be negative.

| Term | Field | Contribution to the step reward | Metric |
|---|---|---|---|
| Time | `time_weight` (per s) | $-w_t \int_0^{\Delta t} e^{-s/\tau} ds$ | M4 |
| Fuel | `fuel_weight` (per m/s) | $-w_f \sum_{\text{segments}} \lvert\mathbf a\rvert \int e^{-s/\tau} ds$, $\lvert\mathbf a\rvert$ the 1-norm for box thrusters and the 2-norm for ball | E1, E1b |
| Constraint time | `violation_weights[name]` (per s) | $-w_c \int_{\text{margin} < 0} e^{-s/\tau} ds$ | S3 |
| Constraint entries | `entry_weights[name]` (each) | $-w_e \sum_{\text{intervals begun}} e^{-s_{\text{entry}}/\tau}$ | C1 |
| Success | `success_bonus` | $+B\,\gamma_k$ at a terminal step whose success holds | M1 |
| Progress | `progress_weight` (per m) | $\gamma_k \Phi(s_{k+1}) - \Phi(s_k)$ with $\Phi = -w_p d$, $\Phi = 0$ at a terminal | shaping only |
| Quadratic | `quadratic` | $-\int e^{-s/\tau} [(\mathbf x - \mathbf x_{\text{ref}})^T Q (\mathbf x - \mathbf x_{\text{ref}}) + \mathbf a^T R\,\mathbf a]\,ds$ | sanity checks |

$s$ is the time from the start of the step and $d$ the distance to the goal.

- **Success** is paid only at a step that ends the episode (`terminal`, not the time limit) with
  the scenario's success predicate holding and no hard constraint violated in the episode, the
  definition of M1. A trajectory through the keep-out zone earns no bonus.
- **Progress** is potential-based. Over an episode the shaped terms sum to
  $\Gamma_T \Phi(s_T) - \Phi(s_0)$, which at a true terminal is $-\Phi(s_0)$, the same for every
  policy, so shaping changes no policy's ranking (Ng, Harada and Russell, 1999). Distance-progress
  terms of the form $\Phi(s') - \Phi(s)$ with $\gamma < 1$ are not: they add $(1 - \gamma)\Phi(s')$ per
  step, a distance penalty that rewards cutting straight through the target.
- **Constraint costs** come from the violation intervals of each step's segment trace
  ([constraints.md](constraints.md)), the same intervals behind S3 (time in violation) and C1
  (one count per contiguous interval). An interval that continues across a step boundary is one
  entry.
- **Quadratic.** `QuadraticCost(n, q, r, reference)` integrates the cost rate exactly under
  Clohessy-Wiltshire motion: over a segment held at acceleration $\mathbf a$ from state
  $\mathbf x_0$ the discounted integral is $\mathbf z^T W \mathbf z$ with
  $\mathbf z = (\mathbf x_0, \mathbf a, 1)$, where $W$ comes from Van Loan's block exponential
  (Van Loan, *Computing integrals involving the matrix exponential*, 1978): with
  $F = \begin{bmatrix} A & B & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0\end{bmatrix} - \tfrac{1}{2\tau} I$
  and $C$ the cost matrix of $\mathbf z$, $\exp\left(\begin{bmatrix} -F^T & C \\ 0 & F\end{bmatrix}
  d\right)$ has blocks $G_{12}$, $G_{22}$ and $W = G_{22}^T G_{12}$. The $\tfrac{1}{2\tau}$ shift
  folds the discount into the state. It is exact for the CW model only.

## Transitions

`Reward` is called on a `Transition`, which holds one step as the truth saw it: the step
duration, its segments (start offset, duration, acceleration), the true state at each segment's
start, each segment's delta-v rate in the actuator's norm, the violation intervals and entry
times per constraint, the goal distance at the step's start and end, and the `success` and
`terminal` flags. Every field may carry leading batch axes; violation intervals are padded with
empty intervals and entry times with `inf`.

## How it is tested

`tests/test_rewards.py`:

- The discount reproduces the per-second values in the Sutton and Barto review (0.99723 and
  0.99861 for $\tau$ = 3600 s).
- KAT-2 (Sutton and Barto 14.4): the return of one trajectory with burns and keep-out passes,
  under time, fuel, constraint-time and entry costs, equals the closed-form integral to 1e-12
  relative for 10 s steps, 5 s steps and an uneven partition, with and without discounting.
- KAT-1 (Sutton and Barto eq. 3.3 with an absorbing terminal): progress shaping over 50 random
  steps sums to $\Gamma_T\Phi(s_T) - \Phi(s_0)$ when truncated and to $-\Phi(s_0)$ at a terminal,
  to 1e-12 relative.
- The success bonus needs both a terminal step and success.
- The quadratic cost matches adaptive quadrature of the cost rate along the closed-form CW
  trajectory to 1e-10 relative, with and without discounting.
- A batch of transitions gives each row its own reward; invalid time constants and weights are
  rejected.

## Example

```python
import numpy as np

from vbar.rewards import Reward, Transition

reward = Reward(tau=3600.0, fuel_weight=10.0, time_weight=1e-3, progress_weight=0.01)
step = Transition(
    duration=10.0,
    segments=np.array([[0.0, 4.0, 0.02, 0.0, 0.0], [4.0, 6.0, 0.0, 0.0, 0.0]]),
    starts=np.zeros((2, 6)),
    fuel_rates=np.array([0.02, 0.0]),
    goal_distance=300.0,
    next_goal_distance=298.0,
)
print(round(float(reward(step)), 6))  # fuel, time and progress terms of one 10 s step
print(round(float(reward.discount(step.duration)), 6))  # 0.997226
```
