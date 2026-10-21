# Spacecraft and propulsion

Code: `vbar.spacecraft.propulsion` (`Spacecraft`, `Actuation`, `ExecutionError`). The environment
builds one `Spacecraft` from its configuration ([environment.md](environment.md)); it can also be
used on its own. Units are SI; accelerations and commands are along the Hill axes
([frames.md](frames.md)).

## Commands and thruster layouts

A command $\mathbf u$ is three fractions of `max_accel` $a_{\max}$, one per Hill axis. The thruster
layout sets the limit and the norm that propellant is spent on:

- **box** (default): one thruster pair per axis. Each component is clipped to $[-1, 1]$, and each
  axis spends propellant for its own delta-v, so the total follows the 1-norm.
- **ball**: one steerable engine. A command longer than 1 is scaled to length 1 in the same
  direction, and propellant follows the 2-norm.

`limit(command, duration)` returns the command as the thrusters deliver it, whether it was
saturated, and whether an impulse was dropped as below the minimum bit.

**Minimum impulse bit.** A thruster cannot deliver less than `min_impulse` (N s). Over a step of
$\Delta t$ the impulse is $|u_i| F_{\max} \Delta t$ on each box axis and
$\min(|\mathbf u|, 1)\, F_{\max} \Delta t$ for the ball, with $F_{\max}$ the thrust of a full
command at the current mass (`max_force`). An impulse above zero and below the minimum is dropped:
the axis for the box, the whole command for the ball. The default 0 turns this off. PRISMA's 0.7
mm/s bit (Alfriend et al., *Spacecraft Formation Flying*, p. 272) is 0.35 N s on 500 kg, against 200
mm/s for a full 10 s command at 0.02 m/s^2.

## Mass and the rocket equation

The exhaust speed is $v_e = I_{sp} g_0$ with $g_0$ = 9.80665 m/s^2. Below, $\|\cdot\|$ is the
layout's norm. `propellant` defaults to the whole mass, so the dry mass is zero and the default
spacecraft never runs out.

**Constant acceleration** (`drive="accel"`, the default). The actuator holds
$\mathbf a = a_{\max}\mathbf u$ over a burn, so the thrust falls with the mass:

$$
\dot m = -\frac{m\,\|\mathbf a\|}{v_e}, \qquad
\Delta m = -m_0\,\operatorname{expm1}\!\left(-\frac{\|\mathbf a\|\,\tau}{v_e}\right), \qquad
\Delta\mathbf v = \mathbf a\,\tau
$$

which is the rocket equation $m_1 = m_0 e^{-\|\Delta\mathbf v\| / v_e}$. With a dry mass the burn
stops after $\tau_{\text{dry}} = (v_e / \|\mathbf a\|) \ln(m_0 / m_{\text{dry}})$.

**Constant thrust** (`drive="thrust"`). The thrust $\mathbf F = F_{\max}\mathbf u$ is fixed, with
$F_{\max} = a_{\max} m_{\text{initial}}$, so the mass flow $\dot m = -\|\mathbf F\| / v_e$ is
constant and the acceleration $\mathbf F / m(t)$ rises (Montenbruck and Gill, *Satellite Orbits*,
eqs 3.126 to 3.130). From rest, with initial acceleration $\mathbf a_0 = \mathbf F / m_0$ and burned
fraction $x = |\dot m|\,\tau / m_0$:

$$
\Delta\mathbf v = \mathbf a_0\,\tau\,H(x), \qquad \Delta\mathbf r = \mathbf a_0\,\tau^2\,G(x), \qquad
H(x) = \frac{-\ln(1 - x)}{x}, \qquad G(x) = \frac{x + (1 - x)\ln(1 - x)}{x^2}
$$

Below $x = 0.1$ both come from their power series, since the closed forms cancel to $x^2$. Every
dynamics model holds one acceleration per segment, so each constant-thrust burn becomes two
half-length segments:

$$
\mathbf a_1 = \mathbf a_0\,(4G - H), \qquad \mathbf a_2 = \mathbf a_0\,(3H - 4G)
$$

Their delta-v is $(\mathbf a_1 + \mathbf a_2)\,\tau / 2 = \mathbf a_0 \tau H$ and their displacement
$(3\mathbf a_1 + \mathbf a_2)\,\tau^2 / 8 = \mathbf a_0 \tau^2 G$, both exact. One segment at the
mean acceleration would keep the delta-v but leave $a^2\tau^3 / (12 v_e)$ of position error per
burn, 1.5e-5 m at 0.02 m/s^2 over 10 s. Constant thrust needs a dry mass (`propellant` below
`mass`).

**Run-out.** When the propellant reaches the dry mass during a burn, the burn stops there, the rest
of the interval coasts, and `Actuation.empty` is set. Later commands deliver nothing.

## Actuation delay and segments

A command takes effect `delay` s after it is issued; until then the previous one stays on. The
spacecraft keeps a queue of commands with their effect times, and `apply(command, duration)`
returns the segments that cover the interval, split at every switch:

- a delay that is not a whole number of steps splits each step in two;
- a delay longer than a step keeps several commands queued;
- a switch closer than 1e-9 s to either end of the interval falls on that end, so no segment is
  shorter than that.

Run-out and the two constant-thrust halves add boundaries inside a piece. The environment
propagates each segment with its own call to the dynamics model, so no integrator steps across a
thrust change. `pending()` lists the commands not yet in effect with the seconds until they act,
and `in_effect()` the one acting now, both as limited, never as delivered.

## Execution error

`ExecutionError(scale_bias, scale_sigma, noise_sigma, misalignment, bound)` perturbs each burn,
one per command (Alfriend et al., eqs 11.1 to 11.3):

$$
\mathbf u_{\text{delivered}} = R\,\bigl(\mathbf u\,(1 + \delta) + \mathbf w\bigr), \qquad
\delta = b + \sigma_\delta\,\nu, \qquad \mathbf w \sim \mathcal N(0, \sigma_w^2 I)
$$

with $b$ = `scale_bias`, $\sigma_\delta$ = `scale_sigma`, $\nu$ a standard normal draw per burn,
$\sigma_w$ = `noise_sigma` in fractions of a full command, and $R$ the rotation by the fixed
rotation vector `misalignment` (rad, Hill axes). It acts after the limits and the minimum impulse
bit; zero commands are not perturbed; the delivered command is not limited again, since a scale bias
means the thrusters deliver more or less than their rating; propellant follows the delivered thrust.
Draws come from the actuator stream. A 1 deg misalignment puts $\sin 1° = 0.017452$ of a burn across
it (Fehse, *Automated Rendezvous and Docking of Spacecraft*, p. 97): 3.49e-4 m/s^2 at 0.02 m/s^2,
close to the $3n^2|x|$ = 3.7e-4 m/s^2 it takes to hold 100 m below the target at 500 km.

## What `apply` returns

`Actuation` holds `segments`, the consecutive (duration s, Hill acceleration m/s^2) pieces;
`dv_l2` and `dv_l1`, the delta-v in the 2-norm and the 1-norm (m/s); `propellant_kg`; and the
flags `saturated`, `below_mib` and `empty`.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_accel` | 0.02 m/s^2 | acceleration of a full command at the initial mass (per axis for box, total for ball) |
| `layout` | `"box"` | `"box"` or `"ball"` |
| `isp` | 220 s | specific impulse |
| `mass` | 500 kg | initial wet mass |
| `propellant` | whole mass | propellant at the start (kg) |
| `min_impulse` | 0 N s | minimum impulse bit |
| `delay` | 0 s | actuation delay |
| `drive` | `"accel"` | hold the acceleration (`"accel"`) or the thrust (`"thrust"`) over a burn |
| `execution` | none | an `ExecutionError`; its `bound` (in sigma, optional) truncates the scale and noise draws by rejection, as for the sensors ([sensors.md](sensors.md)) |

`max_accel`, `isp` and `mass` must be positive, `min_impulse` and `delay` not negative.

## Assumptions and limits

- Thrust acts along the commanded Hill direction with instant on and off: no attitude dynamics,
  thruster geometry, ramps or plume effects.
- The default constant-acceleration drive is an idealization of a fixed thruster: it needs the
  thrust to fall as the mass falls. Use `drive="thrust"` for a fixed thruster.
- The minimum impulse bit is checked for the whole step at the mass when the command is issued.
- Execution errors are drawn per command, not per thruster.

## How it is tested

`tests/test_propulsion.py`:

- Propellant over 360 random steps matches $m_0(1 - e^{-\sum\|\Delta\mathbf v\| / v_e})$ to 1e-9
  relative for both layouts and both drives, and with a 7 s delay.
- Montenbruck and Gill eq 3.129: the delta-v of their 10 N, Isp 350 s station-keeping thruster
  (table 3.13) on 1000 kg for 600 s, to 1e-12.
- One orbit of constant thrust at 0.02 m/s^2 in 10 s steps, against DOP853 on the
  Clohessy-Wiltshire equations with $\mathbf F / m(t)$ and the mass equation: 4.5e-9 m and
  3.2e-12 m/s (the test allows 1e-7 m and 1e-10 m/s), where one mean-acceleration segment per step
  is off by 1.0e-3 m.
- Run-out stops the burn at $\tau_{\text{dry}}$ to 1e-12 for both drives.
- Each segment carries the command issued `delay` earlier, for delays of 0, 10, 25 and 33.3 s,
  and a switch within the tolerance of a step boundary falls on it.
- The minimum impulse bit with PRISMA's 0.7 mm/s, per axis and for the ball.
- Execution error: mean and covariance of 20000 burns at a 1e-6 false-alarm probability, and the
  1 deg misalignment of Fehse p. 97.

## Example

```python
from vbar.spacecraft import Spacecraft

craft = Spacecraft(max_accel=0.02, layout="ball", mass=500.0, propellant=0.1, delay=2.5)
craft.apply([1.0, 0.0, 0.0], 10.0)
act = craft.apply([0.0, -1.0, 0.0], 10.0)
for duration, accel in act.segments:
    print(round(duration, 3), accel.tolist())  # 2.5 s of the first command, then the second
print(round(craft.propellant, 4))  # 0.0189 kg left
```
