# Navigation sensors

Code: `vbar.spacecraft.sensors` (`PerfectNav`, `GaussianNav`, `RangeScaledNav`, `GaussMarkovNav`).
A sensor turns the true relative state into the navigated state the agent sees, in the Hill frame
(m, m/s, [frames.md](frames.md)). It has two methods: `reset(rng)`, called at the start of each
episode with the sensor's random stream, and `observe(true_state, t)`, which returns
$[x, y, z, \dot x, \dot y, \dot z]$ as navigated at time `t` (s). The environment observes once at
reset and once after every step, and gives each environment its own copy of the configured sensor
([environment.md](environment.md)). The agent's observation is built only from this output and
the optional channels; the true state never reaches it.

## Perfect navigation

`PerfectNav` reports the true state. It is the default.

## White noise with an episode bias (`:nav1`)

`GaussianNav(sigma_pos, sigma_vel, bias_pos, bias_vel)` adds to each axis a bias drawn once per
episode and fresh white noise at every observation:

$$
\hat{\mathbf x} = \mathbf x + \mathbf b + \mathbf n, \qquad
\mathbf b \sim \mathcal N(0, \operatorname{diag}(\sigma_{b,r}^2 I_3, \sigma_{b,v}^2 I_3)), \qquad
\mathbf n \sim \mathcal N(0, \operatorname{diag}(\sigma_r^2 I_3, \sigma_v^2 I_3))
$$

The defaults, 1 m and 1 mm/s with no bias, are the synthetic `:nav1` track. They are not meant to
resemble a real sensor and are not valid inside about 20 m, where 1 m is 5% of the range.

`bound`, in units of sigma, truncates every draw of the noise and of the bias by rejection (a
draw outside the bound is drawn again), each per axis, so the error on an axis never exceeds
`bound` times the sum of the noise and bias sigmas, and a certificate for bounded disturbances
has a bounded set to work with. On `RangeScaledNav` the bound applies to the norm of each
three-dimensional draw, because the draw is projected on the line of sight: the position error
then stays within `bound` times the range sigma along the line of sight and `bound` times the
lateral sigma across it, and the velocity error's norm within `bound` times its sigma. The thrust
execution error has the same option ([spacecraft.md](spacecraft.md)). The level that gives a
stated exceedance probability per episode for one-dimensional draws comes from
`vbar.scenarios.per_draw_level` ([scenarios.md](scenarios.md)); for a norm bound use the
chi distribution with three degrees of freedom instead.

## Range-scaled errors in the sensor frame

`RangeScaledNav(k_range, k_lateral, floor, c_range, k_vel, floor_vel)` scales white position noise
with the range $\rho$ and orients it along the line of sight
$\hat{\boldsymbol\ell} = \mathbf r / \rho$:

$$
\sigma_\parallel = \max(\text{floor}, k_{\text{range}}\,\rho, c_{\text{range}}\,\rho^2), \qquad
\sigma_\perp = \max(\text{floor}, k_{\text{lateral}}\,\rho), \qquad
\sigma_v = \max(\text{floor}_v, k_v\,\rho)
$$

One standard normal draw $\mathbf d$ in three dimensions gives the position error

$$
\mathbf e = \sigma_\perp\,\mathbf d + (\sigma_\parallel - \sigma_\perp)\,(\mathbf d \cdot \hat{\boldsymbol\ell})\,\hat{\boldsymbol\ell}
$$

whose component along the line of sight has standard deviation $\sigma_\parallel$ and whose two
components across it have $\sigma_\perp$, independent of each other. At $\rho = 0$ the error is
isotropic at the floor. Velocity errors are isotropic with $\sigma_v$.

The position defaults follow Fehse's rule that relative navigation is good to about 1% of range
(*Automated Rendezvous and Docking of Spacecraft*, secs 2.4 and 5.3.2): $k_{\text{range}}$ =
$k_{\text{lateral}}$ = 0.01 with a 1 cm floor, so 5 m at 500 m. `c_range` models camera ranging
from the apparent size of a target pattern, whose error grows as $\rho^2$; it is off by default.
The velocity defaults, $k_v$ = 1e-4 1/s with a 0.1 mm/s floor (0.05 m/s at 500 m, 1 mm/s at 10 m),
are synthetic, not taken from a sensor.

## Time-correlated errors (`:nav2`)

A navigation filter's output error is correlated in time, and its velocity error is the rate of
its position error. `GaussMarkovNav(sigma_pos, sigma_vel, damping)` models each Hill axis's error
$e$ as a stationary second-order Gauss-Markov process:

$$
\ddot e = -2\zeta\omega\,\dot e - \omega^2 e + q\,w(t), \qquad \omega = \frac{\sigma_v}{\sigma_r}
$$

with $w$ unit white noise. In the state $(e, \dot e)$, with
$A = [[0, 1], [-\omega^2, -2\zeta\omega]]$, the stationary covariance solves
$AP + PA^\top + \operatorname{diag}(0, q^2) = 0$, which gives

$$
P = \operatorname{diag}(\sigma_r^2, \sigma_v^2), \qquad \sigma_v = \omega\,\sigma_r, \qquad
q^2 = 4\zeta\omega\,\sigma_v^2
$$

The cross-covariance is zero because $\frac{d}{dt}E[e^2] = 2E[e\dot e]$ vanishes for a stationary
process. Between observations $\Delta t$ apart the process is propagated exactly:

$$
\mathbf e_{k+1} = F\,\mathbf e_k + L\,\mathbf n_k, \qquad F = e^{A\Delta t}, \qquad
L L^\top = Q = \int_0^{\Delta t} e^{As}\operatorname{diag}(0, q^2)\,e^{A^\top s}\,ds
$$

with $\mathbf n_k$ standard normal. $Q$ equals $P - F P F^\top$ in exact arithmetic, but that
difference cancels on short steps, where the added position variance is about $q^2\Delta t^3/3$
against $\sigma_r^2$. $Q$ is computed instead by Van Loan's method, from one exponential of the
block matrix $[[-A, \operatorname{diag}(0, q^2)], [0, A^\top]]\,\Delta t$, and $L$ from its
eigendecomposition with rounding below zero clipped, so steps down to microseconds stay exact. Each
episode starts from a draw of $P$, so the error is stationary from the first observation, and the
lag covariance is $E[\mathbf e_{k+1}\mathbf e_k^\top] = F P$. Integrating the observed velocity
reproduces the observed position, and averaging over steps does not remove the error: at the
defaults the correlation time $1 / (\zeta\omega)$ is 100 s.

The defaults, 4 cm, 0.4 mm/s and $\zeta$ = 1 ($\omega$ = 0.01 rad/s), are the carrier-phase
differential GPS levels at 1 km reported by Alfriend et al. (*Spacecraft Formation Flying*,
ch. 12: 0.2 to 2.5 cm and 0.2 to 0.5 mm/s at 100 m, 5 to 40 cm and 0.5 to 3 mm/s at 10 km). This
is the `:nav2` track.

## Parameters

| Sensor | Parameters and defaults |
|---|---|
| `GaussianNav` | `sigma_pos` 1 m, `sigma_vel` 1e-3 m/s, `bias_pos` 0 m, `bias_vel` 0 m/s; none negative; `bound` none (sigma, positive) |
| `RangeScaledNav` | `k_range` 0.01, `k_lateral` 0.01, `floor` 0.01 m, `c_range` 0 1/m, `k_vel` 1e-4 1/s, `floor_vel` 1e-4 m/s; none negative; `bound` none |
| `GaussMarkovNav` | `sigma_pos` 0.04 m, `sigma_vel` 4e-4 m/s, `damping` 1; all positive |

## Assumptions and limits

- All errors are Gaussian. There are no outliers, dropouts, field-of-view limits, operating ranges
  or handovers between sensors.
- `RangeScaledNav` uses the true line of sight and range to set the error, as a sensor's error
  depends on the true geometry.
- `GaussMarkovNav` has the same levels on every axis and at every range. For intervals much
  shorter than $1/\omega$, $Q$ loses its positive definiteness to rounding (with 10 m and 1 mm/s,
  at 1 ms), and `observe` raises.

## How it is tested

`tests/test_sensors.py` and `tests/test_base_env.py`. Each statistical check fails a correct sensor
with probability 1e-6 and uses a fixed seed:

- `:nav1` mean and covariance over 20000 draws, the episode bias over 4000 episodes, and noise
  uncorrelated with the true state.
- Range-scaled errors along and across the line of sight at 0.5, 30, 100 and 1000 m, and defaults
  of 1% of range.
- `:nav2` stationary covariance at the first observation and 10 s later, the lag covariance $F P$
  over 4000 episodes, the velocity error matching a finite difference of the position error,
  and the added covariance matching its short-step series at 1 microsecond and 1 ms and
  $P - F P F^\top$ at 50 s.
- In the environment, the observation minus the truth has the sensor's mean and covariance and is
  uncorrelated with the true state and the past actions over 4000 steps; two environments built
  from one configuration draw their own noise.

## Example

```python
import numpy as np

from vbar.spacecraft import GaussMarkovNav, RangeScaledNav

print(RangeScaledNav().sigmas(500.0))  # (5.0, 5.0, 0.05): along, across (m), velocity (m/s)

nav2 = GaussMarkovNav()
nav2.reset(np.random.default_rng(0))
truth = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
errors = np.array([nav2.observe(truth, 10.0 * k) - truth for k in range(2000)])
print(errors[:, :3].std(axis=0).round(3))  # near 0.04 m on each axis
```
