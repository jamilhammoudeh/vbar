# Reproduced results

Code: `vbar.agents.glideslope_lp.lp_glideslope`, `vbar.reproductions`. Published rendezvous results
rebuilt inside V-BAR from the paper alone, flown on the paper's stated case, and compared number by
number. `python -m vbar.reproductions` prints the comparison, which [validation.md](validation.md)
quotes.

## Ariba et al. (2016): glideslope by linear programming

Source: Ariba, Arzelier, Urbina and Louembet, "V-bar and R-bar glideslope guidance algorithms for
fixed-time rendezvous: a linear programming approach", IFAC-PapersOnLine 49(17), 385-390 (2016),
doi:10.1016/j.ifacol.2016.09.066, case 1, which the authors take from Hablani, Tapper and
Dana-Bashian (Journal of Guidance, Control, and Dynamics 25(3), 2002).

**The case.** A V-bar transfer from 500 m to 100 m behind the target, 20 m above it, from rest to
rest in T = 540 s with N impulses at equal times, on a circular orbit at the rate the paper uses,
0.001 rad/s. The paper's axes are x along-track, y out of plane and z toward the Earth (its
equation 2: x'' = 2 z', z'' = 3 z - 2 x', y'' = -y), so its [x, y, z] is V-BAR's [-z, x, -y] in
V-BAR's [x, y, z] ([frames.md](frames.md)): the start [-500, 0, -20] m becomes (20, -500, 0) m and
the aim [-100, 0, -20] m becomes (20, -100, 0) m. Costs are the sum of the impulses' 1-norms, the
paper's criterion.

**The LP glideslope** (`lp_glideslope(n, start, aim, duration, pulses, bound)`). Following the
paper's section 3, the variables are the along-track positions at which the chaser is back on the
line between impulses, its velocity after each impulse, and the last impulse. The arcs between
impulses are exact Clohessy-Wiltshire arcs of length T / N that must end on the line; the last
impulse leaves the chaser at rest at the aim. Each arc's radial deviation from the line at its
midpoint, where a V-bar hump is highest (the paper's lemma 1), is bounded by `bound`; that holds
only while each arc is shorter than one orbit, which `lp_glideslope` requires. The cost
is linear after splitting each impulse component into its magnitude, and the program is solved
with SciPy's HiGHS. Only lines parallel to the V-bar are accepted, the case the paper's V-bar
formulation covers; an infeasible bound raises `ValueError`.

**The standard glideslope** is the baseline `glideslope_pulses` ([glideslope.md](glideslope.md))
with the pulse interval T / N. The paper states its transfer time but not its initial and final
closing rates, which leaves one free parameter, their ratio. It is fitted so that the N = 10 cost
is the paper's 8.09 m/s (`fitted_rate_ratio`): a ratio of 1.195e-3, closing at 4.99 m/s at the start
and 6.0 mm/s at the aim. The other four costs and all five excursions are then predictions.

**Excursion** (`excursion`) is the largest radial distance from the line over the flight, from
exact CW arcs sampled at 400 points each.

## Result

All six LP costs reproduce to the paper's printed digits, the case's headline (2.31 m/s for N =
10 and a 1 m bound) included, and so do four of the five LP excursions and four of five
standard-glideslope costs with all five of its excursions. Two numbers differ in their last digit:
the LP excursion at N = 20 (0.140 m against 0.15 m) and the standard cost at N = 20 (9.306 m/s
against 9.32 m/s). The paper's two-impulse transfer (0.12 m/s, about 125 m of excursion) is not
reproduced: it is Fehse's tangential transfer, and the paper gives no transfer time for it; a
two-impulse transfer in the case's 540 s costs 2.12 m/s. The table with every row is in
[validation.md](validation.md).

At the 400 km orbital rate, 1.131e-3 rad/s, the LP case costs 2.43 m/s instead: the paper's
numbers belong to its rounded 0.001 rad/s.

## Assumptions and limits

- The R-bar formulation of the paper's section 4 and its second case (a cone-constrained approach)
  are not rebuilt.
- The midpoint bound limits the deviation where a V-bar hump is highest on an arc that starts and
  ends on the line; the excursion is checked on the flown arcs.
- The original paper of Hablani, Tapper and Dana-Bashian was not read; the case and its numbers
  are Ariba et al.'s.

## Validation

`tests/test_reproductions.py`:

- The case's axes map from the paper's frame to V-BAR's as stated.
- The LP plan, flown on exact CW arcs, takes every impulse on the line 20 m above the target, keeps
  every midpoint within the bound and ends at rest at the aim; its N = 10, 1 m cost is 2.31 m/s; at
  N = 10 a 0.5614 m bound cannot be met and a 0.5616 m bound is met by the flown arcs; settings off
  the V-bar line or out of range are refused and an impossible bound is reported. `excursion`
  refuses bad inputs.
- Slow: the table in [validation.md](validation.md) is exactly what `python -m vbar.reproductions`
  prints, and the rows that do not match are the four named above.

## Example

The LP glideslope of the case:

```python
import numpy as np

from vbar.agents.glideslope_lp import lp_glideslope

plan = lp_glideslope(1e-3, (20.0, -500.0, 0.0), (20.0, -100.0, 0.0), 540.0, 10, 1.0)
print(len(plan), "impulses,", round(sum(float(np.abs(p.dv).sum()) for p in plan), 3), "m/s")
```
