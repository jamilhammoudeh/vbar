# Lambert's problem

Code: `lambert` in `vbar.dynamics.lambert`, and the targeting agent `LambertAgent` in
`vbar.agents.lambert`. Inertial vectors in metres and seconds, any body-centred inertial frame;
the agent works in the Hill frame.

`lambert(r1, r2, tof, mu, *, prograde=True, normal=(0, 0, 1), tol=1e-10, max_iter=100)` returns
the velocities $(\mathbf v_1, \mathbf v_2)$ at two positions $\mathbf r_1$, $\mathbf r_2$ on the
single-revolution two-body conic that carries a spacecraft from the first to the second in `tof`
seconds. It is the targeting step of an impulsive transfer: $\mathbf v_1 - \mathbf v_1^-$ is the
first burn and $\mathbf v_2^+ - \mathbf v_2$ the second.

## How it is built

The universal-variable formulation of Curtis, *Orbital Mechanics for Engineering Students*,
algorithm 5.2, which follows Bate, Mueller and White, *Fundamentals of Astrodynamics*, chapter 5,
rewritten from the equations.

**Transfer angle.** With $\theta_s = \operatorname{atan2}(|\mathbf r_1 \times \mathbf r_2|,
\mathbf r_1 \cdot \mathbf r_2) \in (0, \pi)$ the short way round, the transfer angle is
$\Delta\theta = \theta_s$ or $2\pi - \theta_s$. A prograde transfer sweeps counterclockwise about
`normal`: it takes the short way when $(\mathbf r_1 \times \mathbf r_2) \cdot \hat{\mathbf n} > 0$
and the long way otherwise; a retrograde transfer takes the other branch. `atan2` keeps the angle
of a 1 s arc to full precision where `acos` of the dot product would lose half its digits.

**Equations.** With $r_1 = |\mathbf r_1|$, $r_2 = |\mathbf r_2|$ and the Stumpff functions
$C(z)$, $S(z)$ (series below $|z| = 0.1$, as in `vbar.dynamics.kepler`):

$$
A = \sin\Delta\theta \sqrt{\frac{r_1 r_2}{1 - \cos\Delta\theta}}
  = \pm\sqrt{2 r_1 r_2}\,\cos\frac{\theta_s}{2}, \qquad
y(z) = r_1 + r_2 + A\,\frac{z S(z) - 1}{\sqrt{C(z)}},
$$

$$
F(z) = \left(\frac{y}{C}\right)^{3/2} S + A\sqrt{y} - \sqrt{\mu}\,t = 0,
$$

with the minus sign of $A$ on the long way. The half-angle form of $A$ is exact algebra and
avoids the cancellation in $1 - \cos\Delta\theta$ on short arcs and near a full revolution. The
factor multiplying $A$ in $y$ is $-\sqrt 2\cos(\sqrt z / 2)$, continued to $-\sqrt 2\cosh(\sqrt{-z} / 2)$
for $z < 0$, so with the half-angle identities $y$ is evaluated as

$$
y = \bigl(\sqrt{r_1} - \sqrt{r_2}\bigr)^2 + 4\sqrt{r_1 r_2}\,\bigl[s \pm \cos\tfrac{\theta_s}{2}\, q(z)\bigr],
\qquad
q(z) = \sin^2\frac{\sqrt z}{4} \;\text{or}\; -\sinh^2\frac{\sqrt{-z}}{4},
$$

with $s = \sin^2(\theta_s / 4)$ and the plus sign on the short way, $s = \cos^2(\theta_s / 4)$ and
the minus sign on the long way. The plain form subtracts two numbers of size $r_1 + r_2$ whenever
$y$ is small against them; this one subtracts terms of size $4\sqrt{r_1 r_2}\,s$, which for a
short arc is $r\,\theta_s^2 / 4$, so a 1 s arc keeps its full precision. Once $z$ is known, the
Lagrange coefficients $f = 1 - y / r_1$, $g = A\sqrt{y / \mu}$ and $\dot g = 1 - y / r_2$ give

$$
\mathbf v_1 = \frac{\mathbf r_2 - f\,\mathbf r_1}{g}, \qquad
\mathbf v_2 = \frac{\dot g\,\mathbf r_2 - \mathbf r_1}{g}.
$$

$z = \Delta E^2$ on an ellipse, $-\Delta H^2$ on a hyperbola and 0 on a parabola, so one
equation covers every conic.

**Root.** $F$ rises with $z$ wherever $y > 0$ and goes to $+\infty$ at $z = 4\pi^2$, where
$C$ vanishes. For $A > 0$ (short way) $y$ falls through zero at some negative $z$ and the root
lies above it; for $A < 0$ (long way) $y$ stays positive and $F$ falls toward $-\sqrt\mu\,t$ as
$z \to -\infty$. The solver takes $z = -1$, doubling downward, until $y \le 0$ or $F < 0$, as the
lower end of a bracket whose upper end is the largest double below $4\pi^2$; any point with
$y \le 0$ or $F < 0$ lies below the root. From $z = 0$ it then runs safeguarded Newton's method
(Press et al., *Numerical Recipes*, 3rd ed., sec. 9.4) with $dF/dz$ from the chain rule and the
Stumpff derivatives $C' = (1 - z S - 2C) / 2z$, $S' = (C - 3S) / 2z$ (series near zero): a step
that leaves the bracket is replaced by bisection, and so is the next step whenever a Newton step
has not halved $|F|$ (the usual rule compares step sizes instead, and rejects every Newton step
when the root sits at one end of the bracket, where the step equals the last bisection). A
point where $y \le 0$ only tightens the bracket, so the next step bisects, and so does a point
where $dF/dz$ is not positive and finite: $F$ rises with $z$, so such a slope is rounding noise,
as on fast long-way transfers where $F$ is a small difference of large terms, and the bracket
then closes and the rounding guard below decides whether the result stands. It stops once a Newton
step has changed $y$ by less than `tol` of itself and the point it reached has $y > 0$, or when
the bracket has closed to four ulp, in which case the upper end, where $y > 0$, is taken; it
raises `RuntimeError` after `max_iter` iterations. A tolerance on $z$ alone would not do: on a
short transfer the root sits within $10^{-8}$ of the zero of $y$, and a step of that size still
leaves $F$ at half its size. Below $z = -10^4$ the two leading terms of $F$ cancel to their
rounding, so a bracket search reaching it raises `ValueError`.

**Rounding guard.** At the root the solver estimates its own rounding: the two terms of $y$ and
the two terms of $F$ each carry a few roundings, counted as four, and $y$ also moves by $dy/dz$
over one ulp of $z$. The error in $y$ reaches the velocities through $g$ at half its relative
size and the error in $F$ through the time of flight. When the estimate exceeds $5 \times
10^{-10}$, half the level the round trips are validated to, the transfer is refused with
`ValueError("tof too short ...")`; the short-transfer tests check the round trips of whatever
passes.

**Parameters.** `prograde` and `normal` fix the branch of the transfer angle as above. `tol`
(default 1e-10) bounds the relative change of $y$ over the last Newton step; the quadratic
convergence leaves the error far below it. `max_iter` defaults to 100.

## Assumptions and limits

- Single revolution only: $z < 4\pi^2$. Transfers of more than one revolution have their own
  roots above $4\pi^2$ and are not returned.
- Collinear positions have no transfer plane. The solver raises `ValueError` when
  $|\sin\theta_s| < 10^{-12}$, so arcs of $10^{-12}$ rad or within that of $\pi$ are refused.
  `normal` in the transfer plane, `tof` not positive, and zero or non-finite inputs (including
  `mu`) raise `ValueError` too.
- Near a full revolution $y$ cancels: at $2\pi - 0.01$ rad on an ellipse of eccentricity 0.3 the
  plain form of $y$ shrinks to $2.5 \times 10^{-5}$ of $r_1 + r_2$, and the rounding of the
  residual then moves the velocities by about $10^{-9}$ relative, a floor that grows as the
  fourth power of the inverse gap to $2\pi$. The test on that arc allows $10^{-7}$; the
  half-angle form measures $8 \times 10^{-12}$ there. Arcs up to $2\pi - 0.1$ are recovered to
  $10^{-9}$.
- Short transfers, those much faster than the orbital time scale of the positions, put the root
  where $y$ is a small remainder of its two terms, or, the long way round, at very negative $z$,
  where $F$ is a small difference of terms that grow as $e^{\sqrt{-z}/4}$ and the solution is a
  near-rectilinear hyperbola through the centre. The rounding guard refuses these with
  `ValueError("tof too short ...")` rather than return velocities off by more than $10^{-9}$;
  on a 6878 km circle the refusal starts near $10^{-4}$ of the period for a 1 rad arc and
  $10^{-5}$ for a 3 rad arc, while arcs small enough that the chord stays near the circle
  (a 1e-2 s arc at 500 km) keep full precision. Below $z = -10^4$ the bracket search itself
  refuses.
- The result is a two-body transfer. Under J2 or other perturbations the second position is
  missed by the perturbation's effect over `tof`.

## Validation

Tests: `tests/test_lambert.py`.

- Montenbruck and Gill, *Satellite Orbits*, exercise 2.6: the orbit through the two given
  positions 9000 s apart, with the book's GM, has $a$ = 28196.776 km, $e$ = 0.7679436, $i$ =
  20.315 deg, $\Omega$ = 359.145 deg, $\omega$ = 179.425 deg and $M$ = 29.236 deg from
  `elements_from_state` to the printed digits, and `propagate_kepler` from $\mathbf v_1$ reaches
  $\mathbf r_2$ and $\mathbf v_2$ to $10^{-9}$.
- Schaub and Junkins, *Analytical Mechanics of Space Systems*, Lambert section, on 200 random
  pairs of positions between 6800 and 42000 km with times of flight from 0.05 to 0.9 of the
  period at their mean radius, both ways round: with $p$ from the returned velocity, the conic
  identity $\mathbf r_2 = F\mathbf r_1 + G\mathbf v_1$ with $F = 1 - r_2(1 - \cos\Delta\theta)/p$
  and $G = r_1 r_2 \sin\Delta\theta / \sqrt{\mu p}$ holds to $10^{-10}$; the chord and radial
  components of $\mathbf v_1$ obey the velocity locus $v_c v_\rho = \mu c \sec^2(\Delta\theta/2) /
  (2 r_1 r_2)$ to $10^{-10}$; and the Lagrange coefficients of the returned pair satisfy
  $f\dot g - g\dot f = 1$ to $10^{-12}$.
- Round trips through `propagate_kepler`, an independent universal-variable solver: the same
  200 random problems both ways round reach $\mathbf r_2$ and $\mathbf v_2$ to $10^{-9}$; arcs on
  a 500 km circle of 0.01 rad, 1 s and 0.95 period, on a Molniya ellipse ($e$ = 0.74, short and
  long arcs), on an $e$ = 0.3 ellipse at $\pi \pm 10^{-3}$ and $2\pi - 0.01$, and on hyperbolas
  of $e$ = 1.5 (short and long arcs) and $e$ = 3 reproduce the conic's own velocities to $10^{-9}$
  ($10^{-7}$ at $2\pi - 0.01$); a property test over $e$ in [0, 0.9] and arcs from 0.01 to
  $2\pi - 0.1$ rad, excluding $10^{-3}$ rad around $\pi$, recovers the velocities to $10^{-9}$.
- Prograde and retrograde transfers between the same positions reach $\mathbf r_2$ with opposite
  angular momentum, retrograde equals prograde about the opposite `normal`, and the refusals above
  raise.
- Short transfers (`tests/test_lambert_short_arcs.py`): on a 6878 km circle, arcs of 0.1, 1 and
  3 rad with times of flight from $10^{-7}$ to $10^{-3}$ of the period either round trip through
  `propagate_kepler` to $10^{-9}$ or raise the "tof too short" refusal, with both outcomes
  present; a 1 ms transfer over 1 rad is refused; 1 s and 1e-2 s arcs on a 500 km circle return
  the circular velocity to $10^{-9}$.
- Against Clohessy-Wiltshire two-impulse targeting on a 500 km circular orbit with the chaser at
  rest on the V-bar from 1 cm to 100 km behind, over $n t_f = 1$: the Lambert burns differ from
  the linear ones by less than $0.5\,\rho / a$ of their size plus a rounding floor of
  $32\,\epsilon\,n a$, and the first burn by more than $0.2\,\rho / a$ at 100 km, where the
  curvature terms the linear equations drop are visible.

## Example

The orbit through two positions of Montenbruck and Gill's exercise 2.6:

```python
import numpy as np

from vbar.dynamics.kepler import elements_from_state
from vbar.dynamics.lambert import lambert

mu = 398600.4415e9
r1 = np.array([11959.978e3, -16289.478e3, -5963.827e3])
r2 = np.array([39863.390e3, -13730.547e3, -4862.350e3])
v1, v2 = lambert(r1, r2, 9000.0, mu)
a, e, i, raan, argp, M = elements_from_state(r1, v1, mu)
print(f"a = {a / 1e3:.3f} km, e = {e:.7f}, i = {np.degrees(i):.3f} deg")
assert abs(a - 28196.776e3) < 0.5
```

## Lambert targeting agent

`LambertAgent(*, n_tf=1.0, mu=EARTH.mu, mode="retarget", name="lambert")` flies a two-impulse
transfer to rest at the scenario's hold point, planned by Lambert's problem instead of the
linear CW equations of `TwoImpulseAgent`.

**Declared model knowledge.** The agent sees the view's mean motion $n$ but not the central
body, so the gravitational parameter `mu` is declared configuration, Earth's by default, supplied
from the public scenario parameters. With it the agent rebuilds the target's circular orbit of
radius $a = (\mu / n^2)^{1/3}$. A declared `mu` that differs from the flown body's moves that
orbit, and the resulting miss is part of the agent's score.

**Plan.** At each plan the target is placed at angle 0 on the declared circle, in the inertial
x-y plane, with $\mathbf r_t = (a, 0, 0)$ and $\mathbf v_t = (0, na, 0)$; its Hill axes there are
the inertial axes. The chaser's inertial state is the target's plus the observed Hill state
(`vbar.conventions.hill_to_inertial`), and the aim point is the hold point relative to the
target at angle $n t_f$. `lambert` (prograde about $+z$) gives $\mathbf v_1$ and $\mathbf v_2$,
and

$$
\Delta\mathbf v_1 = \mathbf v_1 - \mathbf v_c, \qquad
\Delta\mathbf v_2 = C(t_f)^T\left(\mathbf v_\text{hold} - \mathbf v_2\right),
$$

with $\mathbf v_\text{hold}$ the inertial velocity of the hold point at rest in the Hill frame
and $C(t_f)$ the Hill axes at arrival. Lambert's problem is invariant under rotation, so the
true orbit's inclination and node do not enter, and the agent reads only the observation and
`info["time_s"]`.

**Modes.** `"retarget"` solves again at every decision for the time left and fires the departure
impulse; fewer than two steps before the arrival it cancels the measured velocity until that
burn is delivered, and it coasts through decisions whose transfer angle lies within
$10^{-2}$ rad of $\pi$. `"open_loop"` plans both impulses at the first decision and flies them as
a `vbar.agents.elements.ElementAgent`, committing the coast between them in the `impulse_coast`
action mode. Impulses are executed by `vbar.agents.planning.impulse_action`. After the arrival
both modes hover where they are with $(-3n^2x, 0, n^2z)$.

**Log.** `agent.log` holds one entry per decision with the time `t`, the `phase` (`"transfer"`,
`"singular"`, `"arrival"` or `"hold"` in `"retarget"`; `"burn"`, `"coast"` or `"hold"` in
`"open_loop"`) and whether the command `saturated`. In `"open_loop"` the departure burn adds
`coast_s` and the keep-out margin predicted over the coast with its certified lower bound,
`predicted_margin` and `predicted_margin_bound` (m). The prediction propagates CW motion, not the
two-body model of the plan. A predicted violation is logged and the plan flown as is.

| Parameter | Default | Reason |
|---|---|---|
| `n_tf` | 1.0 rad | As for `TwoImpulseAgent`: 903 s at 500 km, and the first command points at the goal on every axis |
| `mu` | `EARTH.mu` | The body of the default scenarios |
| `mode` | `"retarget"` | Closes the loop on navigation noise |

**Assumptions and limits.**

- The model is an unperturbed circular target orbit. J2 and the eccentricity of the true orbit
  are outside it and show up as model error on `twobody_j2` truth.
- Single revolution only: `n_tf` must lie below $2\pi$, and at least $10^{-2}$ rad from $\pi$ and
  $2\pi$, where the transfer plane is undetermined.
- Burns are impulses in the plan and steps of constant thrust in flight, starting at the impulse
  time, so large impulses flown with long steps miss by the finite-burn error, as for
  `TwoImpulseAgent`.
- `disc27` actions are refused at reset.
- The agent refuses at reset goals other than a hold or a berthing, and goals without a hold
  point, with a `ValueError`.

**Validation.** Tests: `tests/test_lambert_agent.py`, at 500 km.

- On CW truth from rest 300 m and 3000 m behind the target, the first burn agrees with
  `TwoImpulseAgent`'s to a relative difference below $\rho / a$, the curvature the linear plan
  drops.
- On two-body truth from rest 10 km behind, with 0.1 s burns and a 900 s transfer, the open-loop
  plan arrives less than 0.2 times as far from the hold point as the CW plan, and within twice
  the 0.594 m that the 0.1 s burn explains. Declaring eight times Earth's `mu` more than doubles
  the miss.
- On two-body truth the commands at 0 and 51.6 deg inclination agree to $10^{-9}$ in both modes;
  a run on `twobody_j2` truth completes with finite commands.
- `vbar.agents.sign_check` passes on all six axes; committed coasts log a bound no larger than
  the predicted margin; the agent sees only the time in `info` and holds no simulator object.

**Example.** An open-loop transfer from 300 m behind the target on two-body truth:

```python
from vbar import ApproachConfig, ApproachEnv
from vbar.agents.lambert import LambertAgent
from vbar.metrics import record_episode, success
from vbar.spacecraft import PerfectNav

config = ApproachConfig(
    start_state=(0.0, -300.0, 0.0, 0.0, 0.0, 0.0),
    action_mode="impulse_coast",
    dynamics="twobody",
    sensor=PerfectNav(),
)
ep = record_episode(ApproachEnv(config), LambertAgent(mode="open_loop"), seed=1, agent_seed=2)
print(f"captured: {success(ep)}, predicted margin {ep.agent_log[0]['predicted_margin']:.2f} m")
assert success(ep)
```
