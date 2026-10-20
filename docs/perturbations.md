# Perturbations and shadow

Code: `vbar.dynamics.perturbations`. Positions are body-centred inertial in m with z along the
spin axis, and may be stacked as `(..., 3)`. `sun` is the inertial unit vector from the body
toward the Sun. Over a rendezvous episode it moves by about 1 deg per day, so every model here
holds it fixed.

## How the forces enter the two-body model

`TwoBodyDynamics(..., perturbations=[...])` ([dynamics.md](dynamics.md)) takes any objects with
a `name` and a method `accelerations(r, v, dr, dv)` that returns two inertial accelerations
(m/s^2): the target's at $(\mathbf r, \mathbf v)$, and the chaser's at
$(\mathbf r + \delta\mathbf r, \mathbf v + \delta\mathbf v)$ minus the target's. The target's
orbit feels the first and the chaser's offset the second:

$$
\ddot{\mathbf r} = \mathbf g(\mathbf r) + \sum_k \mathbf a_k, \qquad
\delta\ddot{\mathbf r} = \mathbf g(\mathbf r + \delta\mathbf r) - \mathbf g(\mathbf r)
+ C\,\mathbf a_{\text{Hill}} + \sum_k \Delta\mathbf a_k
$$

Each spacecraft has its own coefficients, so relative motion feels only the difference of the
forces. The model's name gains each force's name (`twobody_j2_drag_srp`). The Hill frame of the
output turns with the target's full acceleration, so a force with a component normal to the orbit
turns the frame as J2 does ([frames.md](frames.md)).

## Atmosphere density

`HarrisPriester(sun, exponent, body=EARTH)` maps positions to density in kg/m^3: the
Harris-Priester model for mean solar activity (Montenbruck and Gill, *Satellite Orbits*,
sec. 3.5.2).

- **Table.** Minimum and maximum densities at 50 heights from 100 to 1000 km (their table 3.8,
  in g/km^3 = 1e-12 kg/m^3). Between table heights the logarithm of each is interpolated linearly,
  which is exponential interpolation of the density.
- **Diurnal bulge.** The density is blended between the two over the bulge:

$$
\rho = \rho_{\min}(h) + \bigl(\rho_{\max}(h) - \rho_{\min}(h)\bigr)\cos^n\frac{\psi}{2},
\qquad \cos\psi = \hat{\mathbf r}\cdot\hat{\mathbf u}_{\text{apex}}
$$

  The apex has the Sun's declination and lies 30 deg east of it in right ascension (2 pm local
  time). `exponent` $n$ has no default: 2 suits low inclinations and 6 polar orbits.
- **Height.** $h$ is the geodetic height above the body's ellipsoid (below).
- **Range.** The density is zero above 1000 km. Below 100 km the model raises `ValueError` rather
  than extrapolate.

Limits: mean solar activity only, with no solar-cycle or geomagnetic dependence. Density models of
this kind agree with each other only to 15 to 40% (Montenbruck and Gill, sec. 3.5.4), so treat
the density, and the drag coefficient with it, as uncertain.

**Geodetic height.** `geodetic_height(r, body=EARTH)` iterates on where the ellipsoid normal
through $\mathbf r$ meets the spin axis (Montenbruck and Gill, eqs 5.83 to 5.88). With
$e^2 = f(2 - f)$ from the flattening $f$ and $\Delta z = e^2 z$ to start:

$$
\sin\phi = \frac{z + \Delta z}{\sqrt{x^2 + y^2 + (z + \Delta z)^2}}, \qquad
N = \frac{R}{\sqrt{1 - e^2\sin^2\phi}}, \qquad
\Delta z \leftarrow N e^2 \sin\phi
$$

until $\Delta z$ changes by less than 1e-9 m; then
$h = \sqrt{x^2 + y^2 + (z + \Delta z)^2} - N$. Each pass shrinks the error by about $e^2$,
6.7e-3 for the Earth (WGS 84 flattening 1/298.257223563).

## Drag

`Drag(density, ballistic_target, ballistic_chaser, body=EARTH)` applies cannonball drag to both
spacecraft in an atmosphere turning with the body (Montenbruck and Gill, eq 3.97):

$$
\mathbf a = -\frac{\rho\,|\mathbf v_r|\,\mathbf v_r}{2B}, \qquad
\mathbf v_r = \mathbf v - \boldsymbol\omega_\oplus \times \mathbf r, \qquad
B = \frac{m}{C_D A}
$$

with $\boldsymbol\omega_\oplus$ the body's spin (7.292115e-5 rad/s about z for the Earth) and $B$
each spacecraft's ballistic coefficient in kg/m^2. `density` is any function of positions, for
example a `HarrisPriester`.

Drag is the largest non-gravitational relative force in low orbit. Ballistic coefficients of
similar spacecraft differ by 10 to 50%, and those of a compact chaser and a space station by a
factor of 4 to 8; typical chaser values are 215 and 470 kg/m^2 (Fehse, *Automated Rendezvous and
Docking of Spacecraft*, sec. 4.2.1). At 500 km a 10% mismatch for $C_D$ = 2.2 and
$A/m$ = 0.01 m^2/kg gives 2e-8 to 1e-7 m/s^2 of differential drag between the table's minimum
and maximum densities, and 1.1 to 5.9 m of along-track drift per orbit.

Limits: the area does not depend on attitude, the air has no winds, and $C_D$ is a fixed input.

## Solar radiation pressure

**Cannonball.** `SolarPressure(sun, target, chaser, body=EARTH, distance=AU)`, with `target` and
`chaser` each spacecraft's $C_R A/m$ in m^2/kg (Montenbruck and Gill, eq 3.75):

$$
\mathbf a = -\nu\,P_\odot\,C_R\,\frac{A}{m}\,\frac{\mathrm{AU}^2}{d^2}\,\hat{\mathbf d}
$$

$\mathbf d$ runs from the spacecraft to the Sun, placed at `distance` along `sun`.
$P_\odot$ = 4.56e-6 N/m^2 is the pressure on an absorbing surface at 1 AU, 1367 W/m^2 over the speed
of light (their eq 3.69). $\nu$ is the visible fraction of the Sun from the conical shadow. $C_R$ is
1 for an absorbing surface; their table 3.5 gives 1.21 for solar panels, 1.30 for an antenna and
1.88 for a sail. With equal coefficients two spacecraft 1 km apart feel the same force to about 1e-8
of it, except while one is in the penumbra and the other is not.

**Flat plates.** `FlatPlatePressure(sun, target, chaser, body=EARTH, distance=AU)` takes each
spacecraft as a sequence of `Plate(area_to_mass, normal, specular=0.0, diffuse=0.0)`: the plate's
area over the spacecraft's mass (m^2/kg), its unit normal in the target's Hill axes, and the
fractions of light reflected specularly and diffusely; the rest is absorbed. A lit plate feels
(Montenbruck and Gill, eq 3.72, with a Lambertian diffuse term)

$$
\mathbf a = -\nu\,P_\odot\,\frac{\mathrm{AU}^2}{d^2}\,\frac{A}{m}\cos\theta
\left[(1 - s)\,\hat{\mathbf u} + 2\left(s\cos\theta + \frac{f}{3}\right)\hat{\mathbf n}\right]
$$

with $\hat{\mathbf u}$ the direction to the Sun, $\hat{\mathbf n}$ the normal on the lit side,
$\theta$ the angle between them, and $s$ and $f$ the specular and diffuse fractions. A plate is
lit on whichever side faces the Sun. Facing the Sun it is a cannonball with
$C_R = 1 + s + 2f/3$. `Plate` raises `ValueError` if the normal is not a unit vector.

Limits: both spacecraft hold their plates fixed in the target's Hill axes (the chaser's own axes
differ by its angular separation, 1.4e-4 rad at 1 km). Plates do not shade one another, the
target does not shade the chaser, and absorbed light is not re-radiated.

## Earth shadow

**Cylindrical.** `cylindrical_shadow(r, sun, body=EARTH)` is 0 behind the body
($\mathbf r\cdot\hat{\mathbf s} < 0$) within the body's radius of the Sun line, and 1 elsewhere.
It is meant for lighting, where the few seconds of penumbra do not matter (Montenbruck and Gill,
sec. 3.4.1). At 500 km a circular orbit spends 37.8% of its period in it with the Sun in the
orbit plane, 35.8% at a beta angle of 30 deg and 23.1% at 60 deg.

**Conical.** `conical_shadow(r, sun, body=EARTH, distance=AU)` is the visible fraction of the
solar disk, with the Sun and the body treated as disks of their apparent radii (Montenbruck and
Gill, eqs 3.85 to 3.94):

$$
a = \arcsin\frac{R_\odot}{|\mathbf d|}, \qquad
b = \arcsin\frac{R}{|\mathbf r|}, \qquad
c = \arccos\frac{-\mathbf r\cdot\mathbf d}{|\mathbf r|\,|\mathbf d|}
$$

$$
\nu = \begin{cases}
1 & c \ge a + b \\
0 & c \le b - a \quad\text{(umbra)} \\
1 - b^2/a^2 & c \le a - b \quad\text{(annular, beyond the umbra's vertex)} \\
1 - \dfrac{a^2\arccos(x/a) + b^2\arccos\bigl((c - x)/b\bigr) - c\,y}{\pi a^2} & \text{otherwise}
\end{cases}
$$

with $x = (c^2 + a^2 - b^2)/(2c)$ and $y = \sqrt{a^2 - x^2}$. For the Earth the umbra's
half-cone is 0.264 deg, the penumbra's 0.269 deg, and the umbra's vertex lies 1.384e6 km behind
the Earth. At 500 km the penumbra takes about 8 s to cross. The fraction varies continuously, so
solar pressure built on it needs no event handling at shadow entry. Integrator steps are longer
than the penumbra, though: a run of 10 s segments through it at 500 km is off by about 1e-5 m at
the default tolerance and 1e-7 m at `rtol` = 1e-13, against 1e-9 of the separation elsewhere.

## Forces left out

Relative motion in low orbit is driven by differential gravity with J2, differential drag and
differential solar pressure. The other forces matter for the absolute orbit but not for the chaser
relative to the target. Their differential accelerations at 1 km of separation (Montenbruck and
Gill, fig. 3.1 and secs. 3.2 to 3.7):

| Force | Differential acceleration at 1 km | Effect on relative motion |
|---|---|---|
| Lunar and solar gravity | 1.7e-10 and 8e-11 m/s^2 (tidal, $2\mu d/D^3$) | under 1 cm per orbit |
| Harmonics beyond J2 | 2e-8 to 5e-8 m/s^2, periodic | centimetres per orbit |
| Earth albedo and infrared | 10 to 35% of solar pressure, which largely cancels between the two | below the solar pressure difference |
| Relativity, solid and ocean tides | nothing measurable | none |

## Validation

Tests: `tests/test_perturbations.py`, with the published values listed in
[validation.md](validation.md).

- The cylindrical shadow's eclipse fractions at 500 km match the geometry at three beta angles;
  the conical shadow's umbra and penumbra half-cones and the umbra's vertex match Montenbruck and
  Gill's sec. 3.4.1; its penumbra fraction matches a numerical integration of the overlap of the
  two disks to 1e-9; the conical and cylindrical shadows agree on the mean over an orbit.
- The geodetic height reproduces Montenbruck and Gill's exercise 5.3 to 0.5 mm.
- Harris-Priester densities at the bulge apex and its antipode equal table 3.8 at 400, 500 and
  600 km; between table heights the density is the geometric mean at the midpoint; the bulge
  narrows with the exponent as $\cos^n(\psi/2)$.
- Drag on a spacecraft at 400 km with 1e-11 kg/m^3 and 215 kg/m^2 is 1.3676e-6 m/s^2 (Fehse
  eq 4.3 rounds it to 1.38e-6); prograde on the equator the co-rotating air reduces it by
  $(1 - \omega r / v)^2$.
- Differential drag in the setting of Fehse's eq 4.3 and fig. 4.3 drifts the chaser as Clohessy-
  Wiltshire with the constant differential acceleration predicts; in the setting of fig. 4.15 a
  chaser started backward returns after 9.08 orbits ("about ten" in the book).
- Solar pressure pushes away from the Sun at 4.56e-6 N/m^2 times $C_R A/m$ and is zero in the
  umbra; a Sun-facing plate equals a cannonball with $C_R$ = 1, 2 and 5/3 for absorbing,
  specular and diffuse plates; an oblique mirror is pushed along its normal by
  $2P\,(A/m)\cos^2\theta$.
- Equal spacecraft feel no differential force and, started together, stay together; forces of
  zero strength leave the two-body model bit for bit unchanged; the Hill velocity under an
  out-of-plane solar force follows the frame's full rate.
- Lunisolar tides are below 1% of the differential drag of a 10% ballistic mismatch at 500 km.

## Example

Drag with a 10% ballistic-coefficient mismatch and equal solar pressure on a chaser holding
100 m behind the target at 500 km, over one orbit:

```python
import numpy as np

from vbar.dynamics.models import TwoBodyDynamics
from vbar.dynamics.perturbations import Drag, HarrisPriester, SolarPressure

sun = [0.5, 0.5, 0.7]
b = 1 / (2.2 * 0.01)  # ballistic coefficient for C_D 2.2 and A/m 0.01 m^2/kg
drag = Drag(HarrisPriester(sun, exponent=4), ballistic_target=b, ballistic_chaser=1.1 * b)
srp = SolarPressure(sun, target=0.013, chaser=0.013)
model = TwoBodyDynamics(
    altitude=500e3, inclination=np.radians(51.6), j2=True, perturbations=[drag, srp]
)
model.reset([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
state, _ = model.advance(np.zeros(3), 2 * np.pi / model.n)
print(model.name, np.round(state[:3], 2))
```
