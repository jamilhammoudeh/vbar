# Mission reconstructions

Code: `vbar.scenarios.reconstructions`. Three real rendezvous events, rebuilt as scenarios
**reconstructed from public reports**. They use only facts published in the sources below; they
are not the operators' data, and nothing in them claims to match what any operator flew or
computed. Where a report gives a value, the scenario uses it and this page cites it. Where it
does not, the value is a dispersion over a stated range or marked "ours", a choice of this
benchmark. Each scenario has a `:nominal` variant with the failure removed, so the same
controller can be scored with and without it, and each failure is built from the fault models of
[faults.md](faults.md).

| Scenario | Gymnasium ID | Event | Failure modelled |
|---|---|---|---|
| `recon-progress-m34` | `vbar/ReconProgressM34-v0` | Progress M-34 and Mir, 25 June 1997 | range and range rate lost on a fast approach |
| `recon-ets7` | `vbar/ReconEts7-v0` | ETS-VII second docking experiment, August 1998 | braking thrusters failing intermittently |
| `recon-dart` | `vbar/ReconDart-v0` | DART and MUBLCOM, 15 April 2005 | velocity reading biased until a waypoint that was missed |

All three use Clohessy-Wiltshire dynamics by default and the hold-point task
([scenarios.md](scenarios.md)): reach a point near the target, stay inside its capture ball below
its capture speed for the dwell, and never enter the keep-out sphere. Entering it ends the
episode (`terminate_on_violation`), since in each event it stands for hitting the target.

## Progress M-34 and Mir

During a test of the remote manual docking system, the station's commander flew the Progress
cargo ship toward Mir with the Kurs rendezvous radar switched off. The only range cue was the TV
image from the Progress camera. The closing rate was higher than planned and was recognised too
late; the Progress hit the Spektr module.

| Item | Value | Source |
|---|---|---|
| Manual control began at | 6 km | Ellis |
| Closing rate at takeover | estimated 6.5 m/s against a planned 5.0 m/s | Ellis |
| Initial burn aimed at | a point about 1000 m behind the station | Ellis |
| Out-of-plane offset | 400 m | Ellis |
| Approach direction | from above the station | Ellis (figure); NASA SMA case study |
| Kurs radar | off; no laser ranging at the right times; TV image the only range-rate cue | Ellis |
| Rate recognised as too high | at about 250 m, about a minute before contact | Ellis |
| Manual control lasted | about 16 min | Ellis |
| Contact speed | about 3 m/s | Ellis |
| Doing nothing | post-event simulations avoided the collision in many cases, including with no manoeuvring at all | Ellis (citing NASA documents of 1997 to 1999) |
| Orbit | 300 to 400 km at 51.6 deg | Ellis |
| Masses | Mir about 100 t, Progress about 7 t | Ellis |

The scenario:

| Setting | Value | Basis |
|---|---|---|
| Start range | uniform on 6 to 7 km | Ellis gives 6 km; the NASA SMA case study puts the start about 7 km from the station |
| Start elevation | uniform on 30 to 70 deg above the local horizontal, the horizontal part behind the station | ours, "above" and "behind" from the reports |
| Out-of-plane offset | 400 m, either side | Ellis gives the size, not the side |
| Closing rate at the start | uniform on 5.0 to 6.5 m/s | Ellis |
| Start velocity | the coasting arc through the aim point 1000 m behind the station (at the same out-of-plane offset) whose initial range rate is minus the drawn closing rate | Ellis's aim point; the arc is ours |
| Fault | `RangeLoss`, the reading scaled to an assumed 6.5 km for the whole episode | Kurs off, TV only |
| Hold point | 200 m behind the station, capture 10 m and 0.1 m/s, dwell 60 s | ours; Ellis notes earlier takeovers were made inside 200 m |
| Keep-out | 25 m | ours, the size of the station's modules and arrays |
| Orbit | 390 km circular, 51.6 deg | inside Ellis's range |
| Chaser | 7000 kg, 0.05 m/s^2 per axis | mass from Ellis; the authority is ours |
| Step, length | 10 s, 300 steps | ours, three times the reported 16 min |

The coasting arcs this builds take 650 to 1600 s to reach the aim point, which brackets the
reported 16 min of manual control. With the radar off, the reading carries no range or range
rate at all, so a closing speed is invisible until the line of sight swings.

## ETS-VII second docking experiment

ETS-VII was a chaser and target pair that made the first unmanned autonomous docking in July
1998. In the second experiment the chaser withdrew to 520 m ahead of the target and approached
along the V-bar; during the final approach its gas-jet thrusters intermittently failed to fire
properly, the approach was interrupted, and the docking succeeded only on 27 August 1998 after
the flight software was changed and the approach slowed.

| Item | Value | Source |
|---|---|---|
| Orbit | 550 km circular | Oda; Kawano et al. |
| Masses | chaser 2.5 t, target 0.4 t | Oda |
| Sensors | GPS relative navigation beyond 500 m, laser radar 2 to 500 m | Oda; Kawano et al. |
| Second experiment | withdrawal to 520 m, then a V-bar approach from ahead at the same altitude | Kawano et al. |
| Firings in the final approach | pulses every 1 to several seconds | Kawano et al. |
| Anomaly | thrusters failing to fire properly, mostly in the final approach; several thrusters (Oda) | Kawano et al.; Oda |
| Holding point and approach in the first experiment | about 2 m, held about 15 min, approached at about 1 cm/s | Kawano et al. |
| Collision avoidance rule | a manoeuvre required for consecutive failures inside 30 m | Kawano et al. |
| Docking contact | about 10 mm/s | Kawano et al. |

The scenario:

| Setting | Value | Basis |
|---|---|---|
| Start | 520 m ahead on the V-bar, uniform within 5 m per axis, at rest within 2 mm/s per axis | Kawano et al.; the spread is ours |
| Fault | `ThrusterOff` on the thrusters pushing along +V-bar, which brake an approach from ahead, at each 5 s decision with probability 0.3 | which thrusters failed and how often is not public; the braking side is the one that matters, and the rate is ours |
| Hold point | 2 m ahead of the target, capture 0.2 m and 0.02 m/s, dwell 900 s | Kawano et al.'s holding point and hold time; the capture ball is ours |
| Keep-out | 1 m | ours |
| Orbit | 550 km circular, 35 deg | altitude from Oda; the inclination is from a secondary source and does not enter the CW model |
| Chaser | 2500 kg, 4e-4 m/s^2 per axis | mass from Oda; one 1 N thruster per direction is ours |
| Step, length | 5 s, 2880 steps (4 h) | ours |

The collision avoidance rule inside 30 m is not modelled.

## DART and MUBLCOM

DART was an autonomous rendezvous demonstrator. Its navigation estimator kept resetting to GPS
velocity that carried a bias, it used propellant fast, and it missed by under 2 m a small
waypoint sphere whose entry would have switched its camera to range measurements; it struck the
MUBLCOM satellite while its navigation believed it was 130 m away and retreating.

| Item | Value | Source |
|---|---|---|
| Proximity operations began | about 1 km behind the target | NASA mishap overview |
| Planned approach | to 50 ft behind on the V-bar, station-keeping there, never within 5 m | Howard and Bryan |
| Camera modes | bearing only at 200 to 500 m; range and attitude inside 200 m | NASA mishap overview |
| Mode-switch waypoint | sphere of 6.3 m radius 200 m behind the target, missed by less than 2 m | NASA mishap overview |
| GPS velocity bias | about 0.6 m/s | NASA mishap overview |
| Estimator resets | to GPS, about every 3 min | NASA mishap overview |
| Contact | about 1.5 m/s (telemetry) or 1.4 m/s (6-DOF reconstruction), while navigation believed 130 m away and retreating at 0.3 m/s | NASA mishap overview; Howard and Bryan |
| Orbit | polar; a 500 km parking orbit and a phasing orbit about 250 km below the target | Howard and Bryan; Rumford |
| Thrusters | 16 cold-gas nitrogen thrusters of 3.6 N, specific impulse 60 s | Rumford |

The scenario:

| Setting | Value | Basis |
|---|---|---|
| Start | 1 km behind on the V-bar, uniform within 10 m per axis, within 1 cm/s per axis | mishap overview; the spread is ours |
| Fault | `SensorBias`, a velocity bias of 0.4 to 0.8 m/s along a random direction, until the path enters the 6.3 m sphere 200 m behind the target | the 0.6 m/s bias and the sphere from the mishap overview; the range and direction are ours |
| Hold point | 50 ft (15.24 m) behind, capture 2 m and 0.05 m/s, dwell 600 s | Howard and Bryan; capture and dwell ours |
| Keep-out | 5 m | Howard and Bryan |
| Orbit | 750 km circular, 90 deg | ours: Rumford's phasing orbit about 250 km below the target, taken at the 500 km parking altitude; "polar" taken as 90 deg |
| Chaser | 360 kg, 0.01 m/s^2 per axis, specific impulse 60 s | specific impulse from Rumford; the mass and one 3.6 N thruster per direction are ours |
| Step, length | 10 s, 720 steps | ours |

The scenario carries the bias and the missed mode switch. It does not carry DART's estimator,
whose resets to the biased GPS velocity made the error persist; an agent here sees the bias in
every velocity reading, as the reset estimator did.

## What the reconstructions show

Tests in `tests/test_reconstructions.py` fly reference controllers and free drift on each
scenario over drawn seeds:

| Scenario | Controller | Result asserted | Test |
|---|---|---|---|
| Progress M-34 | free drift | never enters the keep-out sphere and stays more than 300 m away, on 20 validation seeds | `test_progress_free_drift_never_hits_the_station` |
| Progress M-34 `:nominal` | reference glide to the hold point | holds on all 20 seeds | `test_progress_reference_holds_with_range_and_mostly_hits_without_it` |
| Progress M-34 | the same glide, range lost | hits the station on at least half of the 20 seeds | same test |
| DART `:nominal` | reference glide | holds on all 10 seeds | `test_dart_bias_defeats_a_controller_that_trusts_the_velocity_reading` |
| DART | the same glide, trusting the velocity reading | holds on none of the 10 seeds | same test |
| DART | the same glide fed velocity from differenced positions | holds on all 10 seeds | same test |
| ETS-VII `:nominal` and faulted | slow reference glide | holds on 5 seeds with and without the fault; with it the closest approach is smaller | `test_ets7_slow_approach_holds_with_the_fault_at_less_margin` |

Which design choice would have caught each failure, as far as these runs show:

- Progress M-34: range and range rate. With them the same controller holds on every seed; the
  line of sight and its rate alone do not show a closing speed, and the controller flies into the
  station. Doing nothing was safer than flying blind, as the post-event simulations Ellis cites
  found: the arc aimed 1000 m behind the station misses it.
- DART: an independent velocity source. A controller that differences its position readings is
  not fooled by the bias; one that trusts the biased velocity never reaches its hold point. The
  real collision also needed the estimator's resets (not modelled), so these runs do not
  reproduce the contact itself.
- ETS-VII: a slower approach, which the real recovery also chose. The slow reference holds with
  the braking thrusters failing three decisions in ten, at less margin than without the fault.
  The thrust authority and failure rate are assumptions, so the margin is not a statement about
  the flight vehicle.

## Example

Free drift on the Progress M-34 reconstruction, with and without the radar:

```python
import numpy as np

from vbar.scenarios import make

for name in ("recon-progress-m34", "recon-progress-m34:nominal"):
    env = make(name)
    obs, info = env.reset(seed=3)
    closest = np.linalg.norm(info["true_state"][:3])
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.zeros(3))
        closest = min(closest, np.linalg.norm(info["true_state"][:3]))
    print(name, round(float(closest)), "m closest", round(float(np.linalg.norm(obs[:3]))), "m read")
```

## Sources

- S. R. Ellis, "Collision in space", NASA Ames Research Center web publication (2004 print);
  the published version is in Ergonomics in Design (2000).
- NASA Office of Safety and Mission Assurance, "Spektr of Failure", significant incident case
  study (2010).
- NASA, "Overview of the DART Mishap Investigation Results" (2006).
- R. T. Howard and T. C. Bryan, "DART AVGS Performance", NASA technical report, NTRS 20070031729.
- T. E. Rumford, "Demonstration of Autonomous Rendezvous Technology (DART) project summary", SPIE
  (2003), NTRS 20030062183.
- I. Kawano, M. Mokuno, T. Suzuki, H. Koyama, M. Kunugi, "Result of rendezvous docking
  experiment of ETS-VII", Journal of the Japan Society for Aeronautical and Space Sciences
  50(578), 2002.
- M. Oda, "ETS-VII: achievements, troubles and future", i-SAIRAS 2001.
