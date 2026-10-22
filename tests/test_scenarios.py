"""The scenario audit: every catalogue scenario is solvable, not trivial and deterministic.

A reference controller (`scenario_references.py`) succeeds on each scenario, free drift fails,
the same seed repeats the episode, and the metrics agree with the environment on the run. The
audit runs whole episodes, so this module is the slowest in the suite; each scenario is run on
one seed with its reference and one with drift.
"""

import math

import numpy as np
import pytest
from scenario_references import reference_for

from vbar import run_episode
from vbar.agents import DriftAgent, PDAgent
from vbar.metrics import contact, evaluate, record_episode
from vbar.scenarios import SCENARIOS, get, make
from vbar.scenarios.registry import entry_class
from vbar.spacecraft import ExecutionError, GaussianNav

# One seed per scenario and variant; the reference succeeds on every seed tried, these are the
# cheapest in wall time.
AUDIT = {
    "approach-vbar": 3,
    "approach-vbar:iss": 1,
    "approach-rbar": 3,
    "approach-rbar:iss": 1,
    "approach-hbar": 1,
    "docking": 2,
    "docking:axis": 1,
    "docking:corridor-shuttle": 2,
    "docking:lit": 2,
    "docking-offset": 1,
    "inertial-target": 1,
    "docking-afrl": 2,
    "inspection": 3,
    "closing": 1,
    "rbar-acquisition": 1,
    "berthing-box": 2,
    "orbt-rbar": 3,
    "rpm-hold": 1,
    "torva": 1,
    "vbar-final-shuttle": 3,
    "hst-rbar-grapple": 3,
    "separation-flyaround": 1,
    "separation-flyaround:reboost": 1,
}


def base_name(name):
    return name.split(":")[0]


@pytest.mark.parametrize("name", sorted(AUDIT))
def test_the_reference_controller_succeeds_and_the_metrics_agree(name):
    env = make(name)
    seed = AUDIT[name]
    episode = record_episode(env, reference_for(base_name(name)), seed=seed, agent_seed=seed + 1)
    info = env._info()
    assert info["success"], info
    assert not info["keep_out_violated"]
    metrics = evaluate(episode, env.metric_settings())
    if base_name(name) not in ("inspection", "inertial-target", "separation-flyaround"):
        # The inspection goal is not a hold, a turning port is not yet a metrics goal, and
        # the departure metric scores only the arrival in the clear zone.
        assert metrics["success"] is True
    if base_name(name) in ("docking", "docking-offset", "vbar-final-shuttle"):
        assert contact(episode).t == pytest.approx(info["contact_time_s"], abs=1e-6)
    assert metrics["delta_v"] == pytest.approx(info["dv_used_mps"], rel=1e-9)
    assert math.isfinite(metrics["composite_score"])


def unconstrained(name, **config):
    """The scenario with its constraints removed, so a run costs no in-step refinement."""
    spec = get(name)
    kind = entry_class(spec)

    class Unconstrained(kind):
        def _constraints(self):
            return ()

    return Unconstrained(spec, config)


def test_a_dwell_broken_between_samples_restarts_the_clock():
    # From 4.95 m off the hold point at 0.045 m/s outward, braking at 0.009 m/s^2 for one step
    # crosses the 5 m ball's edge at 5.06 m mid-step and ends back inside at 4.95 m and 0.045 m/s:
    # the step ends inside the ball, but the dwell does not count it.
    env = unconstrained("approach-vbar", training=True)
    env.reset(seed=0, options={"start_state": (4.95, -100.0, 0.0, 0.045, 0.0, 0.0)})
    _, _, _, _, info = env.step(np.array([-0.45, 0.0, 0.0]))
    assert info["captured"] and info["dwell_s"] == 0.0
    _, _, _, _, info = env.step(np.zeros(3))
    assert info["dwell_s"] == pytest.approx(env.cfg.dt)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_free_drift_does_not_reach_any_scenario_goal(name):
    # Constraints aside, since a drifting chaser cannot even meet the goal; without them a
    # long episode costs milliseconds per step instead of the refinement of several margins.
    # The departure is the exception: a push along the V-bar drifts up, over the station and
    # out to the clear zone, and only the departure corridor forbids it.
    env = make(name) if name.startswith("separation-flyaround") else unconstrained(name)
    run_episode(env, DriftAgent(), seed=1, agent_seed=2)
    assert not env._info()["success"]


@pytest.mark.parametrize("name", ["approach-vbar", "docking:lit", "inspection", "closing"])
def test_the_same_seed_repeats_the_episode_and_other_seeds_differ(name):
    def fly(seed, steps=15):
        env = make(name)
        agent = PDAgent() if "hold_point" in env.agent_view().goal else DriftAgent()
        obs, info = env.reset(seed=seed)
        agent.reset(env.agent_view(), np.random.default_rng(0))
        states = [info["true_state"]]
        for _ in range(steps):
            obs, _, term, trunc, info = env.step(agent.act(obs, env.agent_info()))
            states.append(info["true_state"])
            if term or trunc:
                break
        return np.array(states), dict(env.draw, constraints=env.agent_view().constraints)

    steps = 5 if name == "docking:lit" else 15
    first, draw_a = fly(7, steps)
    second, draw_b = fly(7, steps)
    other, _ = fly(8, steps)
    assert np.array_equal(first, second)
    assert draw_a["constraints"] == draw_b["constraints"]
    assert not np.array_equal(first[: len(other)], other[: len(first)])


def test_the_approach_scenarios_keep_their_v1_spec_values():
    env = make("approach-vbar")
    goal = env.agent_view().goal
    assert goal["hold_point"] == (0.0, -100.0, 0.0)
    assert (goal["capture_radius"], goal["capture_speed"], goal["hold_time"]) == (5.0, 0.05, 600.0)
    kinds = {c["type"]: c for c in env.agent_view().constraints}
    assert kinds["keep_out"]["radius"] == 50.0
    assert kinds["closing_speed_limit"]["v0"] == 0.2
    assert kinds["closing_speed_limit"]["v1"] == pytest.approx(4.0 * env.n)
    assert env.soft_constraints == ("closing_speed_limit", "passive_speed")
    assert math.degrees(env.cfg.inclination) == pytest.approx(51.6)
    iss = make("approach-vbar:iss")
    assert {c["type"]: c for c in iss.agent_view().constraints}["keep_out"]["radius"] == 200.0
    assert iss.cfg.altitude_m == pytest.approx(185.0 * 1852.0)
    rbar = make("approach-rbar")
    assert rbar.agent_view().goal["hold_point"] == (-100.0, 0.0, 0.0)
    assert {c["type"]: c for c in rbar.agent_view().constraints}["passive_speed"]["axis"] == "rbar"


def test_the_docking_scenario_keeps_its_v2_spec_values():
    env = make("docking")
    goal = env.agent_view().goal
    assert goal["port"] == (0.0, -2.0, 0.0) and goal["port_axis"] == (0.0, -1.0, 0.0)
    assert goal["axial_speed_window"] == (0.05, 0.10)
    assert (goal["lateral_limit"], goal["lateral_speed_limit"]) == (0.1, 0.03)
    assert (goal["commit_radius"], goal["passive_exclusion"]) == (1.0, 50.0)
    kinds = {c["type"]: c for c in env.agent_view().constraints}
    assert kinds["approach_cone"]["half_angle"] == pytest.approx(math.radians(10.0))
    # The apex sits 0.1 / tan(10 deg) = 0.567 m behind the port, so the corridor is 0.1 m wide
    # at the port plane and still 8.9 m wide 50 m out.
    assert kinds["approach_cone"]["apex"][1] == pytest.approx(
        -2.0 + 0.1 / math.tan(math.radians(10))
    )
    assert kinds["speed_limit"]["v0"] == 0.10 and kinds["speed_limit"]["v1"] == pytest.approx(
        4 * env.n
    )
    assert kinds["keep_out"]["radius"] == 2.0
    assert env.cfg.dt == 5.0 and env.cfg.max_steps == 1200
    # The commit zone condition of the cone barrier: R_commit >= cot(theta) v_lat^2 / a_max.
    needed = goal["lateral_speed_limit"] ** 2 / (math.tan(math.radians(10.0)) * env.cfg.max_accel)
    assert needed == pytest.approx(0.255, abs=1e-3) and goal["commit_radius"] >= needed


def test_contact_is_an_event_inside_the_step_and_ends_it_there():
    # Coast through the port at 0.08 m/s from 1.5 m out: the crossing falls inside a 5 s step,
    # the step ends at the crossing with the keep-out margin exactly zero, and the delta-v of the
    # cut step counts only the part flown.
    env = make("docking", training=True)
    env.reset(seed=0, options={"start_state": (0.0, -3.5, 0.0, 0.0, 0.08, 0.0)})
    done = False
    while not done:
        obs, _, done, trunc, info = env.step(np.array([0.0, 0.0, 0.0]))
    assert info["contact"] and info["success"]
    assert info["time_s"] < env.record.segment.segments[0][0] + env.cfg.dt
    assert info["min_margins"]["keep_out"] >= 0.0
    assert info["contact_axial_speed_mps"] == pytest.approx(0.08, abs=2e-3)
    # Past the plane by a nanometre, toward the target (+y), so the record shows the crossing.
    assert -2.0 + 1e-9 < info["true_state"][1] < -2.0 + 1e-7
    # From 0.2 m out the crossing comes 2.5 s into a 5 s step thrusting across the axis: only
    # the flown part counts, in delta-v and in propellant by the rocket equation.
    env.reset(seed=0, options={"start_state": (0.0, -2.2, 0.0, 0.0, 0.08, 0.0)})
    mass = env.spacecraft.mass
    _, _, done, _, info = env.step(np.array([0.0, 0.0, 1.0]))
    flown = info["time_s"]
    assert done and 2.0 < flown < 3.0
    dv = env.cfg.max_accel * flown
    assert info["dv_used_mps"] == pytest.approx(dv, rel=1e-6)
    burned = mass * -math.expm1(-dv / (env.cfg.isp * 9.80665))
    assert info["propellant_used_kg"] == pytest.approx(burned, rel=1e-9)
    assert env.spacecraft.mass == pytest.approx(mass - burned, rel=1e-12)


def test_a_contact_too_fast_or_off_axis_is_recorded_but_not_a_success():
    env = make("docking", training=True)
    env.reset(seed=0, options={"start_state": (0.0, -3.0, 0.0, 0.0, 0.15, 0.0)})
    done = False
    while not done:
        _, _, done, _, info = env.step(np.zeros(3))
    assert info["contact"] and not info["success"] and info["capture_margin"] < 0.0
    env.reset(seed=0, options={"start_state": (0.2, -3.0, 0.0, 0.0, 0.07, 0.0)})
    done = False
    while not done:
        _, _, done, _, info = env.step(np.zeros(3))
    # The Clohessy-Wiltshire coupling moves the offset by about 1.5 cm in the 14 s of the coast.
    assert info["contact"] and info["contact_lateral_error_m"] == pytest.approx(0.2, abs=0.02)
    assert not info["success"]


def test_a_crossing_of_the_port_plane_far_from_the_port_is_not_a_contact():
    env = make("docking", training=True)
    env.reset(seed=0, options={"start_state": (80.0, -3.0, 0.0, 0.0, 0.5, 0.0)})
    for _ in range(4):
        _, _, done, _, info = env.step(np.zeros(3))
    assert not info["contact"] and not done


def test_the_hold_dwell_must_be_unbroken():
    # The dwell logic needs no constraints, and without them a step costs milliseconds.
    env = unconstrained("approach-vbar", training=True)
    rest = (0.0, -100.0, 0.0, 0.0, 0.0, 0.0)
    env.reset(seed=0, options={"start_state": rest})
    steps = 0
    done = trunc = False
    while not (done or trunc) and steps < 100:
        _, _, done, trunc, info = env.step(np.zeros(3))
        steps += 1
    assert done and info["success"] and steps == 60 and info["dwell_s"] == pytest.approx(600.0)
    # Leaving the ball for one step restarts the clock.
    env.reset(seed=0, options={"start_state": rest})
    for _ in range(30):
        env.step(np.zeros(3))
    env.step(np.array([0.0, 0.0, 1.0]))
    env.step(np.array([0.0, 0.0, -1.0]))
    _, _, _, _, info = env.step(np.zeros(3))
    assert info["dwell_s"] < 300.0 and not info["success"]


def test_inhibit_windows_zero_the_command_and_report_it():
    env = make("rpm-hold")
    obs, info = env.reset(seed=1)
    # The windows are public: an agent knows when its commands will be zeroed.
    assert env.agent_view().goal["thrust_inhibit"] == ((300.0, 570.0),)
    assert "thrust_inhibit" not in make("approach-vbar").agent_view().goal
    for _ in range(28):
        env.step(np.ones(3))
    _, _, _, _, before = env.step(np.ones(3))
    assert not before["inhibited"] and before["time_s"] == 290.0
    _, _, _, _, armed = env.step(np.ones(3))
    assert armed["inhibited"] and armed["time_s"] == 300.0
    assert armed["dv_used_mps"] > before["dv_used_mps"]
    _, _, _, _, during = env.step(np.ones(3))
    assert during["inhibited"] and during["time_s"] == 310.0
    assert during["dv_used_mps"] == pytest.approx(armed["dv_used_mps"])
    assert np.all(env.record.actuation.segments[0][1] == 0.0)
    with pytest.raises(ValueError):
        make("rpm-hold", dt=7.0)


def test_berthing_inhibits_thrust_in_the_capture_window_and_needs_the_drift_to_stay():
    env = make("berthing-box", training=True)
    env.reset(seed=0, options={"start_state": (-15.0, 0.0, 0.0, 0.0, 0.0, 0.0)})
    _, _, done, _, info = env.step(np.ones(3))
    assert info["armed"] and info["inhibited"] and info["dv_used_mps"] == 0.0
    steps = 1
    while not done:
        _, _, done, _, info = env.step(np.ones(3))
        steps += 1
    assert info["success"] and steps == 6 and info["dv_used_mps"] == 0.0
    # With a residual along-track speed of 0.0095 m/s the drift leaves the 0.5 m ball before
    # 60 s pass (a radial residual is braked by the 3 n^2 x term and stays inside).
    env.reset(seed=0, options={"start_state": (-15.0, 0.0, 0.0, 0.0, 0.0095, 0.0)})
    for _ in range(6):
        _, _, done, _, info = env.step(np.zeros(3))
    assert not info["success"] and not info["armed"]


def test_the_closing_zone_opens_at_the_drawn_go_time():
    env = make("closing")
    env.reset(seed=4)
    zone = next(c for c in env.constraints if c.name == "zone_entry")
    assert zone.go_time == env.draw["go_time"] and 600.0 <= zone.go_time <= 2400.0
    inside = np.array([0.0, -1500.0, 0.0, 0.0, 0.0, 0.0])
    assert zone.margin(inside, zone.go_time - 1.0) < 0.0 < zone.margin(inside, zone.go_time + 1.0)
    assert "zone_entry" not in env.soft_constraints


def test_the_j2_rung_draws_the_argument_of_latitude():
    env = make("approach-vbar", dynamics="twobody_j2")
    env.reset(seed=2)
    u0 = env.draw["argument_of_latitude"]
    assert 0.0 <= u0 < 2 * math.pi and env.dynamics.elements[5] == u0
    env.reset(seed=3)
    assert env.dynamics.elements[5] != u0
    assert env.scenario_id.startswith("vbar:approach-vbar:twobody_j2-i51.6:")


def test_inspection_counts_lit_facing_points_within_range_and_outside_the_glare():
    # The Sun along +x at t = 0, so the target is on the day side. A chaser 60 m out, 20 deg off
    # the anti-Sun line, looks within 30 deg of the Sun: points it faces are lit, but glare hides
    # them all. 60 deg off the line it sees exactly the facing, lit points.
    from vbar.scenarios.targets import sun_direction

    env = make("inspection")
    env.reset(seed=1)
    env.sun0 = sun_direction(0.0, 0.0)
    assert env.target_lit(0.0) == 1.0
    normals = env.normals(0.0)
    lit = normals @ env.sun0 > 0.0
    for off, glare in ((20.0, True), (60.0, False)):
        a = math.radians(off)
        r = 60.0 * np.array([-math.cos(a), math.sin(a), 0.0])
        facing = normals @ r - env.params.target_radius > 0.0
        assert np.any(facing & lit)
        seen = env.inspected(np.concatenate([r, np.zeros(3)]), 0.0)
        assert np.array_equal(seen, np.zeros_like(seen) if glare else facing & lit)
    far = np.array([-250.0 * math.cos(1.0), 250.0 * math.sin(1.0), 0.0, 0.0, 0.0, 0.0])
    assert not np.any(env.inspected(far, 0.0))
    assert 0.0 <= env.inspectable_coverage() <= 1.0


@pytest.mark.parametrize("name", ["docking", "docking-offset", "vbar-final-shuttle"])
def test_built_in_agents_refuse_a_docking_goal_at_reset(name):
    from vbar.agents import LLMAgent
    from vbar.agents.llm import text_model

    view = make(name).agent_view()
    with pytest.raises(ValueError):
        PDAgent().reset(view, np.random.default_rng(0))
    with pytest.raises(ValueError):
        LLMAgent(text_model(lambda prompt: "{}")).reset(view, np.random.default_rng(0))
    # The run time assurance task has a hold point but no keep-out sphere.
    with pytest.raises(ValueError):
        LLMAgent(text_model(lambda prompt: "{}")).reset(
            make("docking-afrl").agent_view(), np.random.default_rng(0)
        )


def test_lighting_follows_the_configured_central_body():
    # Kerbin (radius 600 km) at 80 km: with the Sun in the orbit plane the shadow cylinder hides
    # the target for asin(600 / 680) / pi = 34.4% of each orbit, whatever the dynamics model.
    from vbar.dynamics import KERBIN
    from vbar.scenarios.targets import sun_direction

    radius = KERBIN.radius + 80e3
    for dynamics in ("cw", "twobody"):
        env = make("inspection", body="kerbin", altitude_m=80e3, dynamics=dynamics)
        env.reset(seed=1)
        env.sun0 = sun_direction(0.0, 0.0)
        times = np.linspace(0.0, 2 * math.pi / env.n, 20001)[:-1]
        shadow = 1.0 - np.mean(env.target_lit(times))
        assert shadow == pytest.approx(math.asin(KERBIN.radius / radius) / math.pi, abs=2e-3)
    lit = make("docking:lit", body="kerbin", altitude_m=80e3)
    lit.reset(seed=2)
    window = next(c for c in lit.constraints if c.name == "lighting_window")
    assert window.body is KERBIN and window.orbit_radius == pytest.approx(radius)


def test_an_agent_sees_the_coverage_count_but_not_which_points():
    env = make("inspection")
    env.reset(seed=1)
    info = env.agent_info()
    assert set(info) == {"time_s", "coverage"}


def test_bounded_error_models_never_exceed_their_bound():
    rng = np.random.default_rng(0)
    nav = GaussianNav(1.0, 1e-3, bound=2.0)
    nav.reset(rng)
    errors = np.array([nav.observe(np.zeros(6), t) for t in range(5000)])
    assert np.max(np.abs(errors[:, :3])) <= 2.0 and np.max(np.abs(errors[:, 3:])) <= 2e-3
    error = ExecutionError(scale_sigma=0.1, noise_sigma=0.01, bound=3.0)
    delivered = np.array([error.apply(np.array([1.0, 0.0, 0.0]), rng) for _ in range(5000)])
    assert np.all(np.abs(delivered[:, 0] - 1.0) <= 0.3 + 0.03)
    with pytest.raises(ValueError):
        GaussianNav(bound=0.0)
    with pytest.raises(ValueError):
        ExecutionError(bound=-1.0)
    assert ":sensor=gaussian(1+0.001+0+0,b2)" in make("approach-vbar", sensor=nav).scenario_id


def test_a_misbriefed_target_is_not_revealed_by_the_view():
    # The truth turns at -n from phase 0.7 while the briefing says the target is LVLH-stabilised:
    # nothing an agent receives names the true mode, the goal geometry is the briefed one, and
    # the truth still moves the port.
    from dataclasses import replace

    from vbar.scenarios import DockingEnv, InertialPointing, Misbriefed, Passive

    spec = replace(SCENARIOS["docking"], target=Misbriefed(InertialPointing(0.7), Passive()))
    env = DockingEnv(spec)
    env.reset(seed=1)
    view = env.agent_view()
    assert view.goal["target"] == {"mode": "passive"}
    assert view.goal["port"] == pytest.approx((0.0, -2.0, 0.0))
    public = repr((view.goal, view.constraints, env.agent_info()))
    assert "Misbriefed" not in public and "inertial" not in public and "0.7" not in public
    assert all("target" not in c for c in view.constraints)
    assert env.port_at(0.0) != pytest.approx((0.0, -2.0, 0.0))


def test_a_departure_whose_drift_comes_back_within_the_body_is_not_clear():
    # 768 m out and moving away, but the free drift passes 15 m from the centre of the 30 m body
    # within the next orbit: past the clear range, and still not a departure.
    env = make("separation-flyaround", training=True)
    start = (667.47856, 379.90829, 0.0, -0.64444, -1.19533, 0.0)
    env.reset(seed=1, options={"start_state": start})
    _, _, done, _, info = env.step(np.zeros(3))
    assert info["range_to_target_m"] > env.params.clear_range
    assert not done and not info["success"]
    assert info["drift_margin_m"] == pytest.approx(15.0 - env.params.body_radius, abs=0.05)


def test_a_target_burn_moves_the_relative_state_in_the_departure_variant():
    plain, boosted = make("separation-flyaround"), make("separation-flyaround:reboost")
    states = []
    for env in (plain, boosted):
        obs, info = env.reset(seed=1)
        for _ in range(160):
            obs, _, _, _, info = env.step(np.zeros(3))
        states.append(info["true_state"])
    assert states[0][1] != pytest.approx(states[1][1], abs=1.0)
    assert states[1][1] < states[0][1]


class StopAtSegmentEnd(make("approach-vbar").__class__):
    """A hold scenario whose first segment ends the step at its last sample."""

    def _event_time(self, tau, samples, t0):
        return float(tau[-1]) if t0 == 0.0 else None


def test_an_event_at_the_end_of_a_segment_ends_the_step_there():
    from vbar.scenarios import SCENARIOS

    env = StopAtSegmentEnd(SCENARIOS["approach-vbar"], {"action_mode": "impulse_coast"})
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.array([0.5, 0.2, 0.0, 3 * env.cfg.dt]))
    assert info["time_s"] == env.cfg.dt
    assert len(env.record.segment.segments) == 1
    assert info["dv_used_mps"] == pytest.approx(np.hypot(0.5, 0.2) * env.cfg.max_accel * env.cfg.dt)
    with pytest.raises(ValueError):
        type(env)._event_time = lambda self, tau, samples, t0: float(tau[-1]) + 1.0
        env.reset(seed=1)
        env.step(np.array([0.0, 0.0, 0.0, 0.0]))
