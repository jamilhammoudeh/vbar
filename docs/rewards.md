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
interval: the semi-Markov return of Bradtke and Duff (1995). With a fixed per-step $\gamma$,
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
  terms of the form $\Phi(s') - \Phi(s)$ with $\gamma < 1$ are not: they add $(1 - \gamma)\Phi(s')$
  per step, a distance penalty that rewards cutting straight through the target.
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
  folds the discount into the state; since it makes the exponential grow as $e^{d/2\tau}$, and
  a long segment spans many orbits of CW motion, a segment is integrated in $k$ pieces
  $h = d/k \le \min(\tau, 1/4n)$,
  $W(d) = \sum_j e^{-jh/\tau} E_j^T W(h) E_j$ with $E_j$ the undiscounted transition of
  $\mathbf z$ over $jh$, taken from the closed-form CW matrices; pieces starting past $60\tau$
  are skipped, since a held acceleration makes the cost grow like $s^4$ and
  $e^{-x}x^4/24 < 10^{-20}$ beyond $x = 60$. SciPy 1.11, the oldest supported, computes one-radian blocks to only
  $10^{-11}$ and quarter-radian ones to $4 \times 10^{-15}$, and powers of its one-piece
  transition drift by $10^{-11}$ over a hundred pieces; with both choices the cost is good to
  $3 \times 10^{-14}$ there and $10^{-15}$ on current SciPy. The duration is integrated as given,
  since real-time latency makes it arbitrary. It is exact for the CW model only.

## Transitions

`Reward` is called on a `Transition`, which holds one step as the truth saw it: the step
duration, its segments (start offset, duration, acceleration), the true state at each segment's
start, each segment's delta-v rate in the actuator's norm, the violation intervals and entry
times per constraint, the goal distance at the step's start and end, and the `success` and
`terminal` flags. Every field may carry leading batch axes; violation intervals are padded with
empty intervals and entry times with `inf`.

## In the environment

`EnvConfig(reward=Reward(...))` makes `step` return this reward instead of the scenario's legacy
one, and adds the step's `discount` to `info`. `env.transition(record, terminated)` builds the
step's `Transition` from `env.record`: segments, segment start states (`record.segment.starts`),
fuel rates in the configured thruster layout's norm, the violation intervals of
`record.segment.violations` as offsets, the entries (an interval open at the step's start is a
continuation when the previous step ended inside), the goal distance from the scenario, and
the success flag. The approach scenario's success is capture with the keep-out zone never
entered, and its goal distance is the distance to the hold point.

The time limit is a truncation, not a terminal: `terminated` is false, the last potential is
kept, and a learner should bootstrap from the final state. `terminated` wins when both happen
on the same step. With `terminate_on_violation`, a violation is a terminal with
$\Phi = 0$; a learner paying a time cost may then prefer to end the episode early, so keep the
entry weight above the time cost of the rest of the episode.

The legacy reward of the approach scenario, used when `reward` is not set, is the first
release's signal: progress in units of 100 m without a discount, delta-v, 1 per step that enters
the keep-out zone and 10 on capture even after a violation. It is kept for comparison with
earlier results and is not aligned with the metrics.

## How it is tested

`tests/test_rewards.py`:

- The discount gives the per-second values 0.99723 and 0.99861 for $\tau$ = 3600 s at 10 s and
  5 s steps.
- The semi-Markov return (Bradtke and Duff, 1995): the return of one trajectory with burns and
  keep-out passes, under time, fuel, constraint-time and entry costs, equals the closed-form
  integral to 1e-12 relative for 10 s steps, 5 s steps and an uneven partition, with and without
  discounting.
- Potential-based shaping with an absorbing terminal state (Ng, Harada and Russell, 1999; Sutton
  and Barto, 2nd ed., 2018, sec. 3.4): progress shaping over 50 random
  steps sums to $\Gamma_T\Phi(s_T) - \Phi(s_0)$ when truncated and to $-\Phi(s_0)$ at a terminal,
  to 1e-12 relative.
- The success bonus needs both a terminal step and success.
- The quadratic cost matches adaptive quadrature of the cost rate along the closed-form CW
  trajectory to 1e-10 relative, with and without discounting.
- It stays at 1e-10 on segments 100, 2000 and 5000 times longer than $\tau$, where one block
  exponential lost five digits or overflowed, and to 1e-12 on 1e4, 3e4 and 1e5 s segments
  without discounting or with $\tau$ = 1e5 s, where one block was up to 1e-3 off with SciPy 1.11.
- It matches quadrature to 1e-13 on segments of 7.4 ms and 9.988 s, durations off any grid that
  were 6e-8 off when rounded to 1 ns.
- Under $\tau$ = 1000 s a 1e5 s segment with a held acceleration matches quadrature to 1e-13 (the
  former cut at a discount of 1e-17 lost 1.7e-12), and a 1e6 s segment under $\tau$ = 0.01 s, 1e8
  pieces long, costs what a 1 s segment does without building them.
  A mean motion that is not positive is refused.
- A batch of transitions gives each row its own reward; invalid time constants and weights are
  rejected.

`tests/test_reward_env.py`, on episodes of the approach scenario:

- The semi-Markov return: an open-loop burn through the keep-out sphere earns the same return, under
  every term including the quadratic cost, at 10, 5 and 2.5 s steps, to 1e-10 relative (2e-14
  observed).
- Potential-based shaping: progress shaping changes the return of PD captures by exactly $w_p d_0$,
  from a clear path and after cutting through the keep-out zone; a truncated episode keeps
  $\Gamma_T \Phi(s_T)$.
- The linear-quadratic value (Bradtke, Ydstie and Barto, 1994): for the Riccati gain of the exactly
  discretized discounted quadratic cost, the 5-step return plus the bootstrap $\Gamma_N V(x_N)$
  equals $V(x_0) = -x_0^T P x_0$ to 1e-9 relative, and without the bootstrap the gap is the
  discounted value of the last state.
- The success bonus is withheld from a capture after a keep-out violation; under an aligned
  reward a PD flight around the keep-out sphere earns more than one cutting through it, while
  the legacy reward still pays the capture bonus after the violation.
- Gymnasium's environment checker passes with a reward configured.

## Examples

Training reward for the approach scenario:

```python
from vbar import ApproachConfig, ApproachEnv
from vbar.rewards import Reward

reward = Reward(
    tau=3600.0,
    success_bonus=10.0,
    fuel_weight=1.0,
    time_weight=1e-3,
    violation_weights={"keep_out": 0.1},
    entry_weights={"keep_out": 5.0},
    progress_weight=0.01,
)
env = ApproachEnv(ApproachConfig(reward=reward))
obs, info = env.reset(seed=0)
obs, r, terminated, truncated, info = env.step([0.0, -0.2, 0.0])
print(round(r, 4), round(info["discount"], 6))  # the learner discounts this step by 0.997226
```

A single transition:

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
