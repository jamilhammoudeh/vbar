"""The tumbling-capture kind: the turning capsule keep-out and the capture-ready predicate."""

import math
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from scipy.optimize import brentq

from vbar.agents import DriftAgent, action_scale
from vbar.constraints import segment_minimum
from vbar.dynamics import EARTH, CWDynamics, cw_discrete, cw_stm, mean_motion
from vbar.metrics import mission_metrics, record_episode
from vbar.metrics.mission import goal_errors, hold_signal
from vbar.scenarios import SCENARIOS, Fixed, make
from vbar.scenarios.registry import entry_class
from vbar.scenarios.targets import Passive, TorqueFree, Tumbling, body_point, body_point_velocity
from vbar.scenarios.tumbling import CaptureParams, RotatingCapsule

N = mean_motion(EARTH.radius + 500e3)
KEYS = ("moments", "effective_inertia", "rate", "phase", "momentum_direction", "spin_angle")
SPIN = TorqueFree((1000.0, 4200.0, 4700.0), 4650.0, math.radians(2.0), 0.3, (0.2, -0.4, 0.9), 0.7)


def test_the_capsule_margin_is_the_distance_to_the_turned_segment():
    capsule = RotatingCapsule(((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0)), 1.5, SPIN, N)
    t = 140.0
    a, b = (body_point(SPIN, end, t, N) for end in capsule.ends)
    rng = np.random.default_rng(4)
    points = rng.uniform(-8.0, 8.0, (300, 3))
    expected = []
    for p in points:
        s = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
        expected.append(np.linalg.norm(p - (a + s * (b - a))) - 1.5)
    states = np.concatenate([points, np.zeros((300, 3))], axis=1)
    assert capsule.margin(states, t) == pytest.approx(expected, abs=1e-12)
    # Beyond the aft end on the long axis the nearest point is the end itself.
    beyond = np.r_[body_point(SPIN, (-5.0, 0.0, 0.0), t, N), np.zeros(3)]
    assert capsule.margin(beyond, t) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(
    "ends", [((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0)), ((-1.0, 0.0, 0.0), (5.0, 0.0, 0.0))]
)
def test_the_capsule_rate_bound_covers_its_own_turning(ends):
    # A chaser at rest next to the turning body sees the margin change only because the body
    # turns, at up to the rate times the farther end's distance from the centre (3 m, or 5 m for
    # the capsule off the centre, whose nearer end would give too small a bound).
    capsule = RotatingCapsule(ends, 1.5, SPIN, N)
    rng = np.random.default_rng(5)
    points = rng.uniform(-8.0, 8.0, (800, 3))
    states = np.concatenate([points, np.zeros((800, 3))], axis=1)
    h = 1e-3
    for t in (0.0, 300.0, 1700.0):
        rates = np.abs(capsule.margin(states, t + h) - capsule.margin(states, t)) / h
        assert np.max(rates) <= capsule.rate_bound(0.0, 0.0)
        # Tight to a factor 1.5: points beside the far end move almost as fast as the end.
        assert np.max(rates) > capsule.rate_bound(0.0, 0.0) / 1.5


def test_the_capsule_certificate_holds_for_a_chaser_passing_the_spinning_body():
    target = Tumbling((0.0, 0.0, 1.0), 0.05)
    capsule = RotatingCapsule(((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0)), 1.5, target, N)
    start = np.array([-6.0, -4.0, 0.0, 0.02, 0.0, 0.0])
    model = CWDynamics(N)
    model.reset(start)
    _, states = model.advance(np.zeros(3), 600.0, n_samples=21)
    tau = np.linspace(0.0, 600.0, 21)
    found = segment_minimum(capsule, model.state_in_last_segment, tau, states, 0.0, N, 0.0)
    fine = np.linspace(0.0, 600.0, 60001)
    path = np.einsum("tij,j->ti", cw_discrete(N, fine)[0], start)
    truth = float(np.min(capsule.margin(path, fine)))
    assert truth < 1.0
    assert found.lower_bound <= truth
    assert found.margin == pytest.approx(truth, abs=1e-3)


def test_a_capture_ball_reaching_into_the_body_is_refused():
    with pytest.raises(ValueError):
        CaptureParams(capture_point=(-4.6, 0.0, 0.0), capture_radius=0.2)
    assert CaptureParams(capture_point=(-5.0, 0.0, 0.0)).dwell == 30.0


class Follower:
    """Rides the capture point of a briefed tumble with its acceleration fed forward."""

    def reset(self, view, rng=None):
        public = view.goal["target"]
        self.mode = TorqueFree(*(public[key] for key in KEYS))
        self.point, self.n, self.dt = view.goal["capture_point"], view.n, view.dt
        self.scale, self.max_accel = action_scale(view), view.max_accel

    def feedforward(self, obs, t):
        """The command that gives the chaser the point's acceleration over the coming step."""
        h, mid = 1e-2, t + 0.5 * self.dt

        def c(s):
            return body_point(self.mode, self.point, s, self.n)

        accel = (c(mid + h) - 2.0 * c(mid) + c(mid - h)) / h**2
        x, _, z, vx, vy, _ = obs
        natural = np.array(
            [3 * self.n**2 * x + 2 * self.n * vy, -2 * self.n * vx, -(self.n**2) * z]
        )
        return accel - natural

    def act(self, obs, info):
        t = info["time_s"]
        error = obs[:3] - body_point(self.mode, self.point, t, self.n)
        rate = obs[3:6] - body_point_velocity(self.mode, self.point, t, self.n)
        a = self.feedforward(obs, t) - 0.0225 * error - 0.3 * rate
        return np.clip(a / self.max_accel, -1.0, 1.0) * self.scale


def capture_spec(velocity, start=None, steps=20):
    """The flagship with the tumble SPIN and a start at `start` (its capture point by default)."""
    spec = SCENARIOS["tumbling-capture"]
    start = body_point(SPIN, (-5.0, 0.0, 0.0), 0.0, N) if start is None else start
    return replace(
        spec,
        config={**spec.config, "max_steps": steps},
        target=SPIN,
        dispersions={"start_pos": Fixed(tuple(start)), "start_vel": Fixed(tuple(velocity))},
    )


def test_following_the_capture_point_is_capture_ready_and_holding_still_is_not():
    # The capture point is 5 m out on a body spinning at 2 deg/s, so it moves at 0.17 m/s; a
    # chaser at rest there is far outside the 0.02 m/s relative speed even at its start.
    moving = body_point_velocity(SPIN, (-5.0, 0.0, 0.0), 0.0, N)
    assert np.linalg.norm(moving) > 0.1
    for velocity, agent, held in ((moving, Follower(), True), (np.zeros(3), DriftAgent(), False)):
        spec = capture_spec(velocity)
        env = entry_class(spec)(spec)
        ep = record_episode(env, agent, seed=3, agent_seed=4)
        info = env._info()
        assert info["success"] is held
        m = mission_metrics(ep)
        assert m["success"] is held
        if held:
            # The dwell ends at the first sample 30 s after the start; the metrics' continuous
            # dwell agrees to the 0.1 s sample spacing.
            assert info["time_s"] == pytest.approx(30.0, abs=1e-9)
            assert m["time_to_complete"] == pytest.approx(30.0, abs=0.1)
            assert m["terminal_pos_error"] < 0.01 and m["terminal_vel_error"] < 0.005
            assert m["hold_dwell"] >= 30.0 - 0.1
        else:
            assert m["hold_dwell"] == 0.0


def test_capture_metrics_need_the_target_mode():
    goal = {"capture_point": (-5.0, 0.0, 0.0), "capture_radius": 0.2, "capture_speed": 0.02}
    with pytest.raises(ValueError):
        goal_errors(np.zeros(6), goal)
    distance, speed = goal_errors(np.zeros(6), goal, 0.0, Passive(), N)
    assert distance == pytest.approx(5.0) and speed == pytest.approx(0.0)


@pytest.mark.parametrize("variant", ["", ":coning", ":separatrix", ":intermediate"])
def test_every_tumble_preset_passes_the_gymnasium_checker(variant):
    env = make("tumbling-capture" + variant)
    check_env(env, skip_render_check=True)
    _, info = env.reset(seed=11)
    # The tumble is drawn and briefed: the public parameters rebuild the true attitude.
    public = env.agent_view().goal["target"]
    rebuilt = TorqueFree(*(public[key] for key in KEYS))
    times = np.array([0.0, 100.0, 1000.0])
    assert rebuilt.matrix(times, env.n) == pytest.approx(env.target.matrix(times, env.n), abs=1e-12)
    assert info["captured"] is False


def test_the_dwell_ends_when_the_drifting_chaser_falls_behind_the_moving_point():
    # Starting on the capture point with its velocity and then drifting, the chaser keeps the
    # point's velocity while the point turns away: the relative speed passes 0.02 m/s within a
    # few seconds, at the root the metrics find in continuous time.
    velocity = body_point_velocity(SPIN, (-5.0, 0.0, 0.0), 0.0, N)
    spec = capture_spec(velocity)
    env = entry_class(spec)(spec)
    ep = record_episode(env, DriftAgent(), seed=3, agent_seed=4)
    start = ep.start

    def margin(t):
        state = cw_stm(N, t) @ start
        point = body_point(SPIN, (-5.0, 0.0, 0.0), t, N)
        point_velocity = body_point_velocity(SPIN, (-5.0, 0.0, 0.0), t, N)
        return min(
            (0.2 - np.linalg.norm(state[:3] - point)) / 0.2,
            (0.02 - np.linalg.norm(state[3:] - point_velocity)) / 0.02,
        )

    leave = brentq(margin, 0.1, 30.0, xtol=1e-10)
    m = mission_metrics(ep)
    assert 1.0 < leave < 10.0
    assert m["hold_dwell"] == pytest.approx(leave, abs=1e-6)
    assert not m["success"]


def test_the_capture_signal_rate_bound_covers_the_moving_point():
    # A chaser at rest in the Hill frame (speed and acceleration bounds zero) sees the capture
    # margin change only because the point moves and turns; the bound must cover that.
    goal = {
        "capture_point": (-5.0, 0.0, 0.0),
        "capture_radius": 0.2,
        "capture_speed": 0.02,
        "hold_time": 30.0,
    }
    signal = hold_signal(goal, SPIN, N)
    # 10 ms apart over 1.8 body-rate periods: the body-frame rate, and with it the extremes of the
    # point's speed and acceleration, repeats every period.
    times = np.linspace(0.0, 500.0, 50001)
    for point in (body_point(SPIN, (-5.0, 0.0, 0.0), 37.0, N), np.array([0.0, 5.0, 0.0])):
        state = np.r_[point, 0.0, 0.0, 0.0]
        values = signal.margin(np.broadcast_to(state, (len(times), 6)), times)
        rates = np.abs(np.diff(values)) / np.diff(times)
        assert np.max(rates) <= signal.rate_bound(0.0, 0.0)
    assert np.max(rates) > 0.1 * signal.rate_bound(0.0, 0.0)


def test_the_capsule_keeps_the_true_target_out_of_its_public_spec():
    env = make("tumbling-capture")
    env.reset(seed=5)
    (spec,) = env.agent_view().constraints
    assert spec["type"] == "keep_out" and "target" not in spec
    assert spec["radius"] == 1.5 and spec["ends"] == ((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0))


def test_the_capture_signal_bound_needs_the_point_acceleration_on_a_fast_tumble():
    # Above w = capture_speed / capture_radius = 0.1 rad/s the point's acceleration, not its speed,
    # moves the speed margin fastest: a chaser on the point with its velocity, on a body turning
    # at 0.3 rad/s, sees the margin change faster than the speed terms alone allow.
    fast = TorqueFree((1000.0, 4200.0, 4700.0), 4650.0, 0.3, 0.3, (0.2, -0.4, 0.9), 0.7)
    goal = {"capture_point": (-5.0, 0.0, 0.0), "capture_radius": 0.2, "capture_speed": 0.02}
    signal = hold_signal(goal, fast, N)
    t0 = 40.0
    point = body_point(fast, (-5.0, 0.0, 0.0), t0, N)
    velocity = body_point_velocity(fast, (-5.0, 0.0, 0.0), t0, N)
    state = np.r_[point, velocity]
    times = t0 + np.linspace(-0.05, 0.05, 1001)
    values = signal.margin(np.broadcast_to(state, (len(times), 6)), times)
    rate = float(np.max(np.abs(np.diff(values)) / np.diff(times)))
    speed = float(np.linalg.norm(velocity))
    point_speed = fast.rate_bound(N) * 5.0
    assert rate <= signal.rate_bound(speed, 0.0)
    assert rate > (speed + point_speed) / 0.2


def test_the_keep_out_turns_with_the_drawn_target():
    # The capsule's ends follow the episode's own tumble: the end at +3 m on the body axis sits
    # 1.5 m inside the surface at every time.
    env = make("tumbling-capture")
    env.reset(seed=7)
    (capsule,) = env.constraints
    for t in (0.0, 233.0, 1500.0):
        end = body_point(env.target, (3.0, 0.0, 0.0), t, env.n)
        assert capsule.margin(np.r_[end, np.zeros(3)], t) == pytest.approx(-1.5, abs=1e-12)


def test_a_capture_dwell_broken_between_samples_restarts_the_clock():
    # From 0.18 m out from the capture point, at rest relative to it: one step holding there, one
    # pushing out at 0.0075 m/s^2 (ending 0.195 m out, moving out at 0.015 m/s), one braking at
    # 0.015 m/s^2 (peaking at 0.2025 m, outside the 0.2 m ball, at mid-step and ending at
    # 0.195 m moving in), one pushing out again (ending at rest 0.18 m out). Every step ends
    # inside the ball; the third is broken between its ends, so the clock restarts there.
    point = body_point(SPIN, (-5.0, 0.0, 0.0), 0.0, N)
    out = point / np.linalg.norm(point)
    velocity = body_point_velocity(SPIN, (-5.0, 0.0, 0.0), 0.0, N)
    spec = capture_spec(velocity)
    env = entry_class(spec)(spec, {"training": True})
    env.reset(seed=1, options={"start_state": np.r_[point + 0.18 * out, velocity]})
    follower = Follower()
    follower.reset(env.agent_view())
    dwells = []
    for push in (0.0, 0.0075, -0.015, 0.0075):
        relative = follower.feedforward(env.obs, env.t) + push * out
        _, _, _, _, info = env.step(relative / env.cfg.max_accel)
        assert info["captured"]
        dwells.append(info["dwell_s"])
    assert dwells == pytest.approx([2.0, 4.0, 0.0, 2.0])


def test_a_capture_dwell_after_entering_the_body_is_not_a_success():
    # Starting 5 cm inside the body's capsule on the ray to the capture point, the follower climbs
    # out to the point and holds it for the dwell: the episode ends held, but the keep-out was
    # entered, so neither the environment nor the metrics call it a success.
    point = body_point(SPIN, (-5.0, 0.0, 0.0), 0.0, N)
    inside = point * 4.45 / 5.0
    velocity = body_point_velocity(SPIN, (-4.45, 0.0, 0.0), 0.0, N)
    spec = capture_spec(velocity, start=inside, steps=60)
    env = entry_class(spec)(spec)
    ep = record_episode(env, Follower(), seed=3, agent_seed=4)
    info = env._info()
    assert ep.terminated and info["dwell_s"] >= 30.0 - 1e-9
    assert info["keep_out_violated"] and not info["success"]
    assert not mission_metrics(ep)["success"]


def test_a_second_reset_reports_the_new_episodes_capture_point():
    # The capture point at a time is kept for the step; a reset draws a new tumble and must not
    # reuse the last episode's.
    env = make("tumbling-capture")
    env.reset(seed=1)
    _, again = env.reset(seed=2)
    _, fresh = make("tumbling-capture").reset(seed=2)
    assert again["dist_to_hold_m"] == fresh["dist_to_hold_m"]
    assert again["speed_to_capture_mps"] == fresh["speed_to_capture_mps"]
