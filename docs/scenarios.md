# Scenarios

Code: `vbar.scenarios.spec` (`ScenarioSpec`, the dispersion types, `Observation`,
`per_draw_level`), `vbar.scenarios.registry` (`SCENARIOS`, `make`, `register`, `parse_id`,
`gymnasium_id`, tags, plugins), `vbar.scenarios.seeds` (`seeds`, `SeedSplit`),
`vbar.scenarios.suites` (`Suite`, `SUITES`), `vbar.scenarios.targets` (target attitude modes and
the constraints that follow them), `vbar.scenarios.scenario` (`ScenarioEnv`, identifiers), the
scenario kinds `vbar.scenarios.hold`, `docking`, `inspection`, `berthing`, `departure`, and the
catalogue `vbar.scenarios.catalog`. Every scenario is built on the base environment of
[environment.md](environment.md).

## Specs are data

A scenario is a `ScenarioSpec`: a name and version, the environment kind that flies it
(`entry_point`), the shared `EnvConfig` fields (`config`), the kind's own parameters
(`parameters`), the per-episode draws (`dispersions`), the target's attitude mode (`target`),
tags, operational rules as temporal-logic formulas (`rules`), an observation-completeness
statement (`observation`), reference figures with their sources (`oracle`) and named variants
(`variants`), each a set of overrides of the config, parameters, dispersions, target and tags.

Dispersions are typed distributions: `Fixed(value)`, `Uniform(low, high)` per component,
`Normal(mean, sigma, bound)` with an optional truncation at `bound` sigma by rejection,
`Choice(options, weights)` and `Shell(low, high)`, a vector of uniform length in a direction
uniform on the sphere. `spec.draw(rng)` makes one episode's parameters in the spec's order and
`spec.episode(seed)` does the same through the generator `reset(seed=...)` builds, so the
environment and the spec agree on every draw (`tests/test_scenario_spec.py`). Every built-in
scenario draws `start_pos`, `start_vel` and `argument_of_latitude` (the target's mean argument of
latitude, used only on the `twobody_j2` rung, where the drift depends on it, Alfriend et al.,
*Spacecraft Formation Flying*, ch. 8); docking variants with a Sun window add `beta` and
`sun_phase`, the inspection adds the same, `closing` adds `go_time`, offset and inertial targets
add `target_pitch` or `target_phase`, and the departure adds `separation` and
`separation_speed`.

`Observation` names what the agent does not see and what it needs to make the task Markov in
its observation; `horizon` is `"indefinite"` (the time cap is a truncation, the default) or
`"finite"` (time to go is observed through `t_remaining` and the cap is terminal). All v1
scenarios are indefinite with the dwell clock, the contact event or the imaged set as the only
hidden quantities, each of which an agent keeps itself.

## Identifiers

`vbar:{scenario}:{dynamics}:{actuator}:{cost}:v{n}`, then the variants applied, then one
`field=value` token for every `EnvConfig` field that differs from the spec's declared
configuration, so two environments with the same identifier and seed fly the same episode:

```text
vbar:approach-vbar:cw:box:dv:v1
vbar:approach-vbar:cw:box:dv:v1:iss:nav1
vbar:approach-vbar:twobody_j2-i51.6:box:dv:v1
vbar:docking:cw:box:dv:v2:lit:dt=2.5:sensor=gaussian(0.5+0.0005+0+0)
```

The dynamics field carries the inclination on the J2 rung, the only rung whose relative motion
depends on it. Named sensors keep their track names (`nav1`, `nav2`); others are written with
their parameters. The reward and the training flag change no trajectory and are left out. A
custom sensor without a `label` attribute appears by its type name only, so give one when its
parameters matter. A spec that differs from the registered spec of its name and variants (a
spec rebuilt with `dataclasses.replace`, or one never registered) adds a `spec=` token with the
first 12 hex digits of a hash over its kind, configuration, parameters, dispersions, target mode,
cost and version, so a changed spec never borrows the catalogue's identifier.
`make(name_or_id, **config)` builds an environment from a name, a name with variant suffixes
(`"docking:axis"`) or a full identifier; configuration and spec tokens are not rebuilt from an
identifier, pass the fields to `make` instead.

Importing `vbar` registers every spec and each of its variants with Gymnasium as
`vbar/<Name>-v<n>` (`vbar/ApproachVbar-v1`, `vbar/ApproachVbarIss-v1`, `vbar/Docking-v2`); all
pass `gymnasium.utils.env_checker.check_env`. A behaviour change bumps the version.

## Seeds

`seeds(scenario, split, count)` gives the first `count` seeds of a scenario in the `train`,
`validation` or `test` split. Each seed is the first 128 bits of SHA-256 over a key, the scenario
name, a stream label and the index, so it is the same on every platform and release. The train and
validation keys are public and fixed in the code; the test key is never in the repository and
comes from the environment variable `VBAR_HIDDEN_SEED_KEY`. `pairs()` adds an agent seed per
episode from a different stream label, so `agent_seed` never equals `seed`.

The public keys make every train and validation seed public: an agent can list them, reset a twin
of the scenario with each and keep the one whose first observation matches its own, or hash the
public key with the agent label until it reproduces its own stream, and in either case know the
true state at every step ([environment.md](environment.md)). Public splits are for development and
for numbers reported as development results; scored comparisons use the test split, whose seeds
neither search finds without the key.

## Suites

A `Suite` is a versioned list of entries (scenario with variants, split, number of seeds);
`runs()` lists every (scenario identifier, seed, agent seed) and `missing(results)` the runs a
report lacks, so a suite is reported in full or not at all. Built in: `v1-v1` (the four v1
scenarios, 50 validation seeds each), `v1-nav1-v1` (the same on the `:nav1` track),
`v1-operations-v1` (closing, R-bar acquisition, berthing box, the on-axis, lit and offset
dockings, the inertial target and the orbit-normal approach) and `v1-shuttle-v1` (the six
Shuttle-practice scenarios and the Shuttle corridor docking).

## Tags and plugins

Every spec carries tags (`approach`, `docking`, `hold`, `target-in-path`, `iss`, `shuttle`,
`gates`, `lighting`, ...); `by_tag(tag)` lists the scenarios carrying one, `tags_of(identifier)`
the tags of a run, and `group_by_tag(results)` groups result rows under each tag for per-tag
reporting. Third-party packages add scenarios, agents and filters through the entry-point groups
`vbar.scenarios`, `vbar.agents` and `vbar.filters`; `load_plugins()` registers every spec a
package advertises (a `ScenarioSpec` or a callable returning specs), and `plugins(group)` loads
the objects of a group by name. A plugin that fails to import is logged and skipped.

## Target attitude modes

The target's body axes coincide with the Hill axes at zero attitude; a port on the -V-bar face
has body position $(0, -R, 0)$ and axis $(0, -1, 0)$. A mode gives the rotation from body to Hill
axes at any time (Fehse, *Automated Rendezvous and Docking of Spacecraft*, 2003, section 5.3.1),
and `angular_velocity(t, n)`, the rate of the body axes relative to the Hill frame in Hill axes,
which a mode written for a plugin must provide as well. A point fixed to the target moves at
$\boldsymbol\omega \times R\,\mathbf p$ (`body_point_velocity`), the velocity a capture interface
on a turning target has (Fehse 8.3.6):

- `Passive`: LVLH-stabilised, the identity.
- `FixedOffset(pitch, amplitude, period)`: a constant pitch about the orbit normal, such as a
  torque-equilibrium attitude of 15 to 30 deg, with an optional limit cycle
  `amplitude` sin(2 pi t / `period`) standing in for an attitude controller's deadband.
- `InertialPointing(phase)`: body axes fixed in inertial space, so they turn at $-n$ about z in
  the Hill frame; on the docking axis of such a target the chaser needs a lateral speed $rn$,
  0.113 m/s at 100 m (Fehse 5.3.1).
- `Tumbling(axis, rate, phase)`: rotation at a constant rate about a body axis, an interface for
  the tumbling-capture scenarios; a free tumble about a non-principal axis nutates and needs the
  6-DOF target.
- `Maneuvering(schedule, attitude)`: a target that burns on a schedule of (start, duration, Hill
  acceleration). The chaser's relative state feels the opposite acceleration, exact in the
  Clohessy-Wiltshire model and first order in the target's orbit change otherwise; the
  environment cuts its thrust segments at the burns' edges so no segment straddles one
  (`ManeuveringDynamics`).
- `Misbriefed(actual, briefed)`: the truth follows one mode while the goal tells agents another
  (Goodman, *History of Space Shuttle Rendezvous*, 2011: SPARTAN, STS-51G). In 3-DOF a wrong
  briefing cannot be detected from the relative state, so no catalogue scenario uses it yet.
  Everything an agent receives is what an honest briefing of `briefed` gives: the constraint
  specs are built from the briefed mode and the view's identifier hashes the briefed spec, while
  the environment's `scenario_id` and the recorded episode's `scenario_id` name the true mode.

The goal's `target` entry is the mode as agents are told it. Constraints fixed to the target
follow the mode: `RotatingCone` is the approach cone of [constraints.md](constraints.md) with
its apex and axis moving, whose rate bound adds the attitude rate times the reach of the cone
($|\mathbf p| + R_c / (1 - \sin\theta)$, since the cone term is active only that close to the
apex); `BodyKeepOut` is the target's keep-out sphere open at the port, with margin
$\max(\rho - R, \min(d, 0) + d_r)$ where $d$ is the signed depth in front of the port plane and
$d_r$ the reception depth of the mechanism (0.1 m, ours), so the reception volume is not part of
the keep-out and a chaser that has just crossed the plane at the port reads a positive margin;
the clipped depth is the larger term only within $R + d_r$ of the centre, which bounds the rate
the turning plane adds; `LightingWindow` requires, within `active_range` of the port, the Sun's
in-plane angle from the approach axis to lie in a window and the target to be on the day side,
as the robustness $\max((d - R_c) / R_c, w(t))$ with $w$ the least angular margin to the window's
edges and the lit margin $(p - R_B) / R_B$, where $R_B$ is the configured central body's radius
and $p$ the target's distance from its shadow cylinder's axis on the night side, and that distance
plus $r s_x$ on the day side, continuous across the terminator.
`sunlit` is the cylindrical shadow of [perturbations.md](perturbations.md) at the target's
position; at 500 km with the Sun in the orbit plane the target is in eclipse 37.8% of each orbit,
23.1% at beta = 60 deg (Montenbruck and Gill, *Satellite Orbits*, section 3.4).

## Scenario kinds

Every kind reads its parameters from the spec, draws the episode in `_draw_episode`, builds
its constraints and goal, and ends the episode as its task demands. The goal names its task
(`"hold"`, `"dock"`, `"inspect"`, `"berth"` or `"depart"`); the built-in PD, LLM and plan agents
fly only holds and berthings and refuse the other tasks at `reset`, since the inspection and
departure goals carry a hold point for the mission metrics, not as a place to fly to. All kinds
support `thrust_inhibit` windows, (start, end) in seconds on step boundaries, during which every
command is zeroed (an attitude manoeuvre or a procedural inhibit; `info["inhibited"]` reports
it); the windows are public in the goal under `thrust_inhibit`. `soft` names the constraints
scored without failing the run.

**Hold point** (`HoldPointEnv`, approaches, closing, acquisition, the Shuttle holds). The goal is a
hold point with a capture ball and a dwell: success when $|\mathbf r - \mathbf r_h| <$
`capture_radius` and $|\mathbf v| <$ `capture_speed` hold without a break for `hold_time`, since
a state inside such a box leaves it within a fraction of an orbit unless it is controlled
(Fehse 3.3.3). The predicate is checked on every sample of every step; the metrics settle the
dwell in continuous time ([metrics.md](metrics.md)). Constraints follow the parameters: a
keep-out sphere, the linear closing-speed law or a table of range-rate gates, the passive-speed
envelope with the keep-out radius as the structure extent, a limit on the whole relative speed,
an approach ellipsoid closed until the episode's drawn GO time, and a corridor with its apex at
the target's centre and its axis toward the hold point, turning with the target. Holding off the
V-bar costs thrust: $3n^2|x|$ on the R-bar and $n^2|z|$ on the orbit normal, 2.09 m/s per orbit
at 100 m below a 500 km target (Fehse 3.3.3, eqs 3.55 and 3.58), which the delta-v metric shows.

**Docking** (`DockingEnv`). Contact is an event: the first crossing of the port plane
$(\mathbf r - \mathbf p) \cdot \hat{\mathbf a} = 0$ from the approach side within
`contact_range` of the port, found by a root search in the exact segment trace, and the step
ends a nanometre past the plane (so the record shows the crossing whatever the rounding of the
propagation between samples) with the propellant of
the unflown rest given back (at 0.1 m/s and 5 s steps the
chaser moves 0.5 m per step, so a step-boundary test would miss or overshoot the plane). Success
at contact: lateral offset within `lateral_limit`, lateral speed within `lateral_speed_limit`,
axial closing speed inside the mechanism's window, no hard constraint violated on the way
(Fehse 7.1.1, 8.3.5, appendix C). The window comes from a preset: `probe-drogue-atv` 0.05 to
0.10 m/s (ATV, the default), `impact` 0.10 to 0.35 m/s (Soyuz and Progress), `low-impact` 0.01 to
0.03 m/s (Hermes-Columbus, the ESA active latch) and `apds` 0.025 to 0.05 m/s (the Shuttle's
mechanism: below 0.05 m/s per Fehse appendix C; the lower end is this benchmark's choice just
below the 0.030 m/s planned for STS-130, whose measured 0.0344 m/s fits no other preset,
Goodman p. 242 and 244). A capture margin, the least over the criteria of
(limit - value) / limit, is reported next to the verdict. The corridor cones have their apex
set back from the port by `lateral_limit` / tan(half-angle), so the corridor is exactly as wide
as the reception envelope at the port plane and a contact inside the envelope is inside the
corridor; 50 m out the 10 deg cone is 0.57 m wider than one with its apex at the port. The hard
speed limit is $v_0 + v_1 \rho$ on the whole relative speed with $v_0$ the window's upper end and
$v_1 = 4n$; gate tables and the closing-speed law are soft options. `commit_radius` (1 m, the
last metre in which no abort is expected), `passive_exclusion` and `plume_range` go to the
metrics through `metric_settings()`. The commit zone satisfies the cone barrier's feasibility
condition $R_{\text{commit}} \ge \cot\theta\, v_{\text{lat}}^2 / a_{\max}$ = 0.255 m at 10 deg,
0.03 m/s and 0.02 m/s^2; the environment's cone margin is the max-composition of
[constraints.md](constraints.md), continuous where the cone turns on, and a filter chooses its
own treatment of the activation.

**Inspection** (`InspectionEnv`). The target carries 100 surface points on a 5 m sphere (a
Fibonacci lattice, equal weights). A point with outward normal $\mathbf n_i$ at $\mathbf p_i$ is
imaged at time $t$ when the chaser faces it, $\mathbf n_i \cdot (\mathbf r - \mathbf p_i) > 0$,
the Sun lights it, $\mathbf n_i \cdot \hat{\mathbf s}(t) > 0$ with the target outside the
central body's shadow (the configured `body`: on Kerbin at 80 km the target is in shadow 34.4% of
each orbit at beta = 0), the range is within 200 m and the Sun is more than 30 deg from the
camera's line of sight. The Sun's elevation above the orbit plane (beta, positive toward +z) is
dispersed over $\pm 75$ deg, since $|\beta| \le i + 23.44$ deg (Fehse 5.4.1), with its in-plane
phase. Points whose normal never faces the Sun during the episode cannot be imaged by anyone, so
success is a coverage of 95% of the points lit at some time in the episode (sampled every minute at
reset);
`info` reports both fractions, the coverage per m/s of delta-v and the time to the goal. The
imaged count is in the agent's `info` under `coverage`, since the camera knows what it has
imaged; the set itself is not. The points follow the target mode, so the inertially pointing and
offset variants rotate them.

**Berthing** (`BerthingEnv`). Bring the chaser to rest inside a capture ball and let the arm take
it: once the capture predicate holds at a step boundary the environment inhibits the thrusters
for `capture_time` (Fehse 8.1.2: thrust is inhibited at capture and the arm grapples within one
to two minutes), and success needs the predicate to hold on every sample until the window ends.
Released at rest 15 m below the V-bar the chaser drifts 0.414 m in 120 s (Fehse 5.3.1), a quarter
of that in 60 s, so a 0.5 m ball with a 60 s window is reached from rest and lost with a
residual along-track speed near 0.01 m/s. The corridor's apex sits behind the box by
`capture_radius` / tan(half-angle), as for docking.

**Departure** (`DepartureEnv`). The chaser starts at the port, displaced along the axis by the
drawn separation and moving out at the drawn speed, and must leave through a departure corridor
on the port axis. Success at a step boundary when the range exceeds `clear_range` and the free
Clohessy-Wiltshire drift from that state never comes within the body radius over two orbits
(`drift_minimum_range`, with the least range refined by a root of $\mathbf r \cdot \mathbf v$).
The public goal names the clear point on the axis as a hold point with a wide capture ball and a
capture speed no departure reaches (10 m/s), so the mission metrics score the arrival in the
clear zone; the drift verdict is the scenario's own
and is in `info["drift_margin_m"]`.

## The catalogue

Shared defaults: Earth, a circular 500 km orbit at 51.6 deg (ISS-like; the J2 rung needs an
inclination, and 0 deg hides every out-of-plane effect), exact Clohessy-Wiltshire propagation, a
500 kg chaser with 0.02 m/s^2 per axis and Isp 220 s, perfect navigation, 10 s steps. `:iss` and
Shuttle-practice scenarios fly at 342.62 km (185 n.mi., the STS-130 mean altitude, Goodman
p. 227), since every n-scaled limit and timing taken from Shuttle data belongs to that orbit.
Values marked "ours" are this benchmark's choices inside published ranges. Every scenario has
`:nav1` (`GaussianNav`, 1 m and 1 mm/s white noise) and `:nav2` (`GaussMarkovNav`, the CDGPS
levels) variants as separate leaderboard tracks ([sensors.md](sensors.md)). Each spec's
`oracle` holds reference figures: the two-impulse and forced-motion delta-v of a V-bar transfer
(Fehse eqs 3.29 to 3.32 and 3.44 to 3.48), the fly-around costs, the hold costs and the flown
Shuttle values.

| Scenario (version) | Task | Start | Goal and constraints | Steps | Sources |
|---|---|---|---|---|---|
| `approach-vbar` (v1) | hold 100 m behind the target | x in [-200, 200], y in [200, 600], z in [-50, 50] m, each velocity in [-0.1, 0.1] m/s: the target lies in the way | ball 5 m, 0.05 m/s, dwell 600 s (ours); keep-out 50 m hard (about 10 times the target extent, Fehse p. 142); closing-speed law 0.2 + 4n rho (soft, ours in the form of Dunlap et al. 2021) and the V-bar passive-speed envelope (soft); plume range 100 m | 360 x 10 s | Fehse 5.7.3: S3 at 100 to 200 m for a small target |
| `approach-vbar:iss` | the same at ISS class | y in [600, 1200] m | hold 400 m behind, keep-out 200 m, plume range 300 m, 342.62 km | 720 x 10 s | Fehse 5.7.1 (S3 300 to 500 m), fig. 5.24 (200 m zone); Goodman p. 240 (Low Z from 1000 ft) |
| `approach-rbar` (v1) | hold 100 m below the target | x in [-600, -200], y in [-200, 200] m | as the V-bar approach with the R-bar passive-speed envelope; holding costs 3 n^2 |x| | 360 x 10 s | Fehse 5.7.2 (S4 300 to 500 m below; safe speeds 0.1, 0.05, 0.02 m/s at 50, 30, 20 m inside the derived envelope) |
| `approach-hbar` (v1) | hold 100 m past the target along the orbit normal | z in [300, 500] m, within 20 m of the axis | as the V-bar approach without the passive envelope; a liveness probe where the Clohessy-Wiltshire coupling does not break the head-on symmetry | 360 x 10 s | a head-on probe along the orbit normal (ours), after the barrier-function stalls of Ames et al. 2019 |
| `docking` (v2) | contact at the -V-bar port of a 2 m sphere | random direction, 300 to 800 m, velocities within 0.05 m/s | ATV window; lateral 0.1 m and 0.03 m/s (ours, inside Fehse's 0.05 to 0.2 m and 0.01 to 0.05 m/s); 10 deg corridor within 50 m (ISS corridors 10 to 15 deg, Fehse 5.6); body keep-out 2 m; speed limit 0.10 + 4n rho hard; commit zone 1 m, passive-safety exclusion 50 m, plume range 100 m | 1200 x 5 s | Fehse 7.1.1, 8.3.5, app. C; Goodman pp. 207 to 208 (Low Z 200 to 400 ft for a small target) |
| `docking:axis` | the final approach from the axis | on the axis 100 to 500 m out, within 20 m laterally, 0.01 m/s | as docking | 1200 x 5 s | Fehse 5.7.3 |
| `docking:corridor-shuttle` | the Shuttle's corridors and mechanism | as `:axis` | 8 deg corridor from 400 ft, then 5 deg from 30 m (ours); `apds` window; optional 300 s hold at 30 ft; 342.62 km | 1200 x 5 s | Goodman p. 137, fig. 20.15, table 25.8 |
| `docking:lit` | contact inside a Sun window | on the axis 100 to 300 m out; beta in [-70, 70] deg | within 10 m of the port the Sun's in-plane angle from the axis must lie in [15, 75] deg (Fehse's 105 to 165 deg from +V-bar) on the day side; the agent times the approach from a hold | 1500 x 5 s | Fehse 5.4.1 |
| `docking-offset` (v1) | dock at a port 20 m from the centre of a pitched target | on the V-bar 150 to 300 m behind | pitch in [15, 30] deg; body keep-out 20 m; the approach line is off the V-bar and drifts 3 pi h per orbit | 1200 x 5 s | Fehse 5.3.1, 5.7.3 |
| `inertial-target` (v1) | dock at the port of an inertially pointing target | random direction, 150 to 300 m | the axis turns at -n from a random phase; on it the chaser needs lateral speed r n | 1200 x 5 s | Fehse 5.3.1 |
| `docking-afrl` (v1) | the run time assurance docking task | as docking | reach the origin within 0.5 m at under 0.2 m/s; the whole relative speed under 0.2 + 4n rho (hard); n = 0.001027 rad/s (851.8 km) | 1200 x 5 s | Dunlap et al. 2021, eq. 14 |
| `inspection` (v1) | image 95% of the lit points | random direction, 50 to 150 m; beta in [-75, 75] deg | keep-out 10 m; range 200 m; Sun exclusion 30 deg (ours) | 1140 x 10 s (about two orbits) | Fehse 5.4.1; Montenbruck and Gill 3.4 |
| `closing` (v1) | from the S2 hold to the S3 hold of an ISS-class target | 3000 m behind within 20 m, 0.01 m/s | hold 400 m behind, dwell 600 s; keep-out 200 m; the approach ellipsoid (1000 x 2000 x 1000 m) may be entered only after the GO drawn between 600 and 2400 s; plume range 300 m | 1140 x 10 s | Fehse 5.6, 5.7.1, eqs 3.29 to 3.32 and 3.44 to 3.48 |
| `rbar-acquisition` (v1) | from a V-bar hold to a hold 500 m below | 1 to 5 km behind | dwell 300 s (ours; the hold costs 0.59 m/s); keep-out 200 m | 1140 x 10 s | Fehse eqs 3.33, 3.38 to 3.39 (2 n dz and 3 n dz) |
| `berthing-box` (v1) | forced R-bar approach to a box 15 m below | 50 to 80 m below, within 5 m of the axis | ball 0.5 m, 0.01 m/s, 60 s window (ours, inside Fehse's 0.1 to 0.5 m, 0.01 m/s, 60 to 120 s); keep-out 10 m; 10 deg corridor | 360 x 10 s | Fehse 5.3.1, 8.1.3 |
| `orbt-rbar` (v1) | the ORBT rendezvous from Ti to a hold 600 ft below the ISS | 8 n.mi. behind and 0.2 n.mi. below, coelliptic, within 200 m and 0.05 m/s | dwell 300 s; keep-out 100 ft (ours); fig. 25.17 gates (soft); MC-4 waypoint 885 ft behind and 1823 ft below; Low Z from 1000 ft | 600 x 10 s | Goodman tables 25.4, 25.7, 25.8 |
| `rpm-hold` (v1) | hold through the pitch manoeuvre | 600 ft below, 0.3 ft/s closing | thrust inhibited from 300 to 570 s; range floor 580 ft (hard); dwell 300 s after | 180 x 10 s | Goodman pp. 140 to 141, 241 (the free drift derived from them) |
| `torva` (v1) | twice-orbital-rate transition from 575 ft below to 310 ft ahead | 575 ft below at rest | dwell 300 s; keep-out 100 ft (ours); the oracle is the quarter lap at 5.5 n R | 240 x 10 s | Goodman p. 137, table 25.8 (the circle's thrust law and costs derived from them) |
| `vbar-final-shuttle` (v1) | the +V-bar final approach to contact | 320 ft ahead of the port, 0.2 ft/s closing | `apds` window; 8 deg then 5 deg corridors; fig. 25.22 gates (soft, ranges to the centre); optional 30 ft hold | 720 x 5 s | Goodman fig. 25.22, table 25.8 |
| `hst-rbar-grapple` (v1) | +R-bar approach to an inertially pointing telescope and station keeping on its axis | 2000 ft below at rest | hold at 35 ft within 1 m and 0.01 m/s for 120 s inside a 10 deg cone on the turning axis (active within 50 ft); keep-out 8 m; STS-109/125 gates (soft); Low Z from 1500 ft; 180 min deadline; 560 km | 1080 x 10 s | Goodman table 24.3, p. 208, ch. 24 |
| `separation-flyaround` (v1) | undock from the +V-bar port and reach 2000 ft on a non-returning path | at the port, 0.15 to 0.2 ft/s out | 10 deg departure corridor within 400 ft; body 30 m; the `:reboost` variant has the target burn 0.5 mm/s^2 along-track for 600 s from 900 s | 540 x 10 s | Goodman pp. 246 to 252 |

Gate tables (`GATES` in the catalogue) are the flown range-rate gates in m and m/s:
`shuttle-rbar-orbt` (HST +R-bar from ORBT, table 24.3), `shuttle-vbar-iss` (fig. 25.22),
`dual-coelliptic` (fig. 9.3) and `sts-130-orbt` (fig. 25.17); the approaches have `:gates-*`
variants that replace the closing-speed law with one as a soft constraint.

## Audit

Each scenario was audited with a reference controller (`tests/scenario_references.py`:
forced-motion waypoint flights, re-solved two-impulse transfers and reference tracking with
Clohessy-Wiltshire feedforward, built on the observation and the public view alone). It
succeeds on the first three validation seeds of every scenario and variant, the audit that
`pytest -m slow` runs, and on ten more drawn seeds each in the audit sweep; free drift reaches no
scenario's goal even with the constraints removed, the same seed repeats the episode and the
metrics agree with the environment on the run (`tests/test_scenarios.py`). The docking
reference walks round the sphere of its standoff's radius about the port, so it never crosses a
corridor's activation sphere off the axis; the inspection reference joins its circle along a
path that keeps its range; the grapple reference starts its descent only just inside the lead
angle, so it arrives as the telescope's axis comes round. The hold boxes
contain the dispersion a reference LQR reaches with raw `:nav1` noise fed back: a stationary
1-sigma of 0.22 m and 8.2 mm/s per axis at 10 s steps, so 3 sigma per axis is 0.66 m and
0.025 m/s against the 5 m and 0.05 m/s box, and the 3-sigma norm over three axes 1.13 m and
0.043 m/s is still inside; the speed criterion is the tight one (`tests/test_hold_audit.py`,
after Rawlings, Mayne and Diehl, *Model Predictive Control*, 2nd ed., sections 2.9 and 3.1). The
grapple box (1 m, 0.01 m/s) is not reachable through `:nav1`, and its `:nav1` variant is a
hardness track. A constant along-track position bias is undetectable on the Clohessy-Wiltshire
model (a biased y is another V-bar equilibrium), so a hold under such a bias carries the bias as
an offset. Bounded variants of the error models exist for certificates that need them:
`GaussianNav`, `RangeScaledNav` and `ExecutionError` take `bound` (in sigma), and
`per_draw_level(episode_probability, draws)` derives it from an episode-level exceedance
probability by the union over draws (at 3 sigma per draw 94.6% of 1080-draw episodes leave the
bounded model; 1e-3 per episode needs 4.91 sigma for 1080 draws and 5.04 for 2160; Kochenderfer,
Katz, Corso and Moss, *Algorithms for Validation*, section 8.1).

## Assumptions and limits

- Relative attitude is perfect in 3-DOF: the docking criteria have no angular terms, and a
  misbriefed attitude cannot be detected from the relative state.
- The goal's port and axis are their Hill-frame values at t = 0, and the mission metrics judge
  contact there, so for a turning port (`inertial-target`) the environment's `info["success"]`
  is the verdict; the metrics' contact entries are not valid there. The departure metric scores
  only the arrival in the clear zone.
- The GO time of `closing` is public in the zone constraint; a GO revealed at the event needs the
  supervisory channel.
- `docking-afrl` keeps this benchmark's exact propagation; the published benchmark integrates
  with Euler steps of 1 s, which the safety-filter reproduction adds as its own dynamics rung.
- A target burn changes the relative state only; the reference orbit stays the target's unburned
  one, exact in the Clohessy-Wiltshire model.
- The inspection's inspectable set is sampled every minute of the episode; a point lit for less
  than a minute could be counted either way.
- An episode cut by the contact event ends with the spacecraft's internal clock ahead of the
  environment's by the unflown part of the step; the environment is reset before the next
  episode.

## How it is tested

`tests/test_scenario_spec.py`, `test_targets.py`, `test_scenarios.py`, `test_hold_audit.py` and
`test_shuttle_known_answers.py`:

- Dispersions stay inside their bounds and reject bad parameters; `per_draw_level` reproduces
  the union-bound figures above.
- `spec.episode(seed)` draws what `reset(seed)` draws; variants apply in order and appear in the
  identifier; every result-changing configuration field changes the identifier and the reward
  and training flag do not; identifiers parse back to their scenario; every scenario and variant
  passes the Gymnasium checker and `gymnasium.make` flies the same episode as `make`.
- Seeds are deterministic, distinct across splits and scenarios, above 2^100, the test split
  refuses to run without the hidden key, and the key is not in the package.
- Suites list their runs and find missing ones; tags group results; plugins register through
  entry points and a conflicting re-registration is refused.
- Target modes: the inertially pointing port moves at $rn$ (Fehse 5.3.1), the pitched port sits
  at $R\sin$(pitch), a tumble returns after its period, the rotating cone equals the fixed cone
  turned by the attitude and its certified minimum through a segment never exceeds the true one,
  the body keep-out is zero at the port and negative inside, the eclipse fractions match
  Montenbruck and Gill, the lighting window opens only near the port inside the window on the day
  side, and a target burn is the opposite acceleration on the relative state. Each mode's
  angular velocity equals the rate of its attitude matrix, a body point's velocity equals the
  rate of its position, and the turning cone and body keep-out certificates fail a segment test
  when their rotation term is removed.
- The audit above; the v1 spec values of the approaches and the docking; contact is an event
  inside the step that ends it with the keep-out margin at zero and the delta-v of the cut step;
  a contact too fast or off axis is recorded and not a success; a far crossing of the port plane
  is no contact; the dwell must be unbroken; inhibit windows zero the command; berthing inhibits
  thrust in the capture window and needs the drift to stay; the closing zone opens at the drawn
  GO; the J2 rung draws the argument of latitude; the inspection counts lit, facing points within
  range and outside the glare; bounded error models stay within their stated bounds; a target
  burn moves the relative state.
- Goodman's flown figures and the ones derived from them (the twice-orbital-rate thrust law,
  the transition and pitch-manoeuvre timings, the thrust-loss ellipses, the arc costs, the gate
  time constants, the contact speed, the pitch-manoeuvre drift, the V-bar envelope ratios) and
  Fehse's transfer, fly-around and berthing-box figures, recomputed from the Clohessy-Wiltshire
  solution ([validation.md](validation.md)).

## Example

```python
import numpy as np

from vbar.scenarios import SUITES, make, seeds

env = make("docking:axis")
print(env.scenario_id)  # vbar:docking:cw:box:dv:v2:axis
obs, info = env.reset(seed=seeds("docking:axis", "validation", 1).seed(0))
goal = env.agent_view().goal
print(goal["port"], goal["axial_speed_window"], [c["type"] for c in env.agent_view().constraints])
obs, reward, terminated, truncated, info = env.step(np.zeros(3))
print(round(info["dist_to_hold_m"], 1), info["contact"])
print(len(SUITES["v1-v1"].runs()))  # 200 runs a report must cover in full
```
