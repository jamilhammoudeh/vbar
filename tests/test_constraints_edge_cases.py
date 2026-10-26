import numpy as np
import pytest
from scipy.optimize import brentq

from vbar.constraints import KeepOut, ZoneEntry, segment_minimum, violation_intervals
from vbar.dynamics import EARTH, CWDynamics, cw_states, cw_stm, mean_motion
from vbar.scenarios.registry import make

N = mean_motion(EARTH.radius + 500e3)


def exact(n, x0, accel, t):
    """Closed-form CW states (len(t), 6) at times `t` (s) from `x0` under a held `accel`."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return cw_states(n, t, np.tile(x0, (t.size, 1)), np.tile(accel, (t.size, 1)))


def flown(x0, accel, duration, samples):
    model = CWDynamics(N)
    model.reset(x0)
    _, states = model.advance(accel, duration, n_samples=samples)
    return model, np.linspace(0.0, duration, samples), states


def test_zone_bound_is_sound_when_go_falls_between_two_samples():
    # A 101 m start closing at 1 m/s enters the 100 m sphere about 1 s into the step; GO comes
    # at 1.4 s, between the samples at 1.05 s and 1.58 s, so the margin falls a further 0.35 m
    # after the last finite sample.
    t0, go = 610.0, 611.4
    x0 = np.array([0.0, 101.0, 0.0, 0.0, -1.0, 0.0])
    zone = ZoneEntry((100.0, 100.0, 100.0), go_time=go)
    model, times, states = flown(x0, np.zeros(3), 10.0, 20)
    found = segment_minimum(zone, model.state_in_last_segment, times, states, 0.0, N, t0)
    at_go = float(zone.margin(exact(N, x0, np.zeros(3), go - t0)[0], go))
    assert at_go == pytest.approx(-0.4, abs=1e-3)
    assert found.lower_bound <= at_go
    assert found.margin == pytest.approx(at_go, abs=zone.resolution)


def test_zone_violation_runs_to_go_when_go_falls_between_two_samples():
    t0, go = 610.0, 611.4
    x0 = np.array([0.0, 101.0, 0.0, 0.0, -1.0, 0.0])
    zone = ZoneEntry((100.0, 100.0, 100.0), go_time=go)
    model, times, states = flown(x0, np.zeros(3), 10.0, 20)
    found = segment_minimum(zone, model.state_in_last_segment, times, states, 0.0, N, t0)
    spans = violation_intervals(zone, model.state_in_last_segment, times, states, found, t0, 0.0, N)
    entry = t0 + brentq(
        lambda tau: float(zone.margin(exact(N, x0, np.zeros(3), tau)[0], t0 + tau)),
        0.0,
        go - t0,
        xtol=1e-12,
    )
    assert len(spans) == 1
    assert spans[0][0] == pytest.approx(entry, abs=1e-6)
    assert spans[0][1] == pytest.approx(go, abs=1e-6)


def test_closing_with_15_s_steps_refuses_its_go_time_or_certifies_every_step_soundly():
    # Seed 30 draws GO at 610 s, 40.67 steps of 15 s. The start is chosen so that the chaser
    # is 15 m outside the ellipsoid at GO and closing at 1 m/s along-track, so the margin falls
    # 0.5 m/s, and about 0.26 m between the last sample before GO and GO itself.
    seed, dt = 30, 15.0
    try:
        env = make("closing", dt=dt, training=True)
        env.reset(seed=seed)
        go = env.draw["go_time"]
        n = env.n
        start = cw_stm(n, -go) @ np.array([0.0, -2030.0, 0.0, 0.0, 1.0, 0.0])
        env.reset(seed=seed, options={"start_state": tuple(start)})
    except ValueError:
        return
    assert go % dt != 0.0
    zone = next(c for c in env.constraints if c.name == "zone_entry")
    t_prev = 0.0
    while t_prev < go + dt:
        _, _, terminated, truncated, _ = env.step(np.zeros(3))
        assert not (terminated or truncated)
        t = np.linspace(t_prev, env.t, 301)
        if t_prev < go < env.t:
            t = np.append(t, go)
        true_min = float(np.min(zone.margin(exact(n, start, np.zeros(3), t), t)))
        bound = env.record.segment.minima["zone_entry"].lower_bound
        assert bound <= true_min + 1e-9, f"step ending at {env.t} s"
        t_prev = env.t


def test_violation_intervals_do_not_raise_when_a_sample_grazes_the_boundary():
    # The keep-out radius is the range at one sample up to its last bit, so rounding can put
    # the sample's margin and the margin re-evaluated at the same time on opposite sides of 0.
    rng = np.random.default_rng(0)
    raised = 0
    trials = 40
    for trial in range(trials):
        x0 = np.r_[rng.normal(size=3) * 30.0, rng.normal(size=3) * 5.0]
        model, times, states = flown(x0, np.zeros(3), 10.0, 20)
        j = int(rng.integers(1, 20))
        rho_j = float(np.linalg.norm(states[j, :3]))
        radius = np.nextafter(rho_j, -np.inf) if trial % 2 else np.nextafter(rho_j, np.inf)
        ko = KeepOut(float(radius))
        found = segment_minimum(ko, model.state_in_last_segment, times, states, 0.0, N)
        try:
            violation_intervals(ko, model.state_in_last_segment, times, states, found, 0.0, 0.0, N)
        except ValueError:
            raised += 1
    assert raised == 0, f"{raised} of {trials} grazing segments raised"


def test_a_crossing_on_a_sample_starts_the_violation_at_that_sample():
    # Closing along the V-bar at 5 m/s, the range falls through the whole segment; a radius
    # equal to the range at sample j puts the entry exactly on that sample.
    x0 = np.array([0.0, 60.0, 0.0, 0.0, -5.0, 0.0])
    for j in range(1, 19):
        model, times, states = flown(x0, np.zeros(3), 10.0, 20)
        ko = KeepOut(float(np.linalg.norm(states[j, :3])))
        found = segment_minimum(ko, model.state_in_last_segment, times, states, 0.0, N)
        spans = violation_intervals(
            ko, model.state_in_last_segment, times, states, found, 0.0, 0.0, N
        )
        assert len(spans) == 1, f"sample {j}"
        assert spans[0][0] == pytest.approx(times[j], abs=1e-8), f"sample {j}"
        assert spans[0][1] == 10.0


# Thrusting at 0.08 m/s^2 against a 4 m/s along-track approach, the chaser passes the target
# at about 11.3 s (1.0 m off) and again at about 88.6 s (5.0 m off) within one 100 s segment.
TWO_PASSES = np.array([0.4, 40.0, 0.0, 0.0995, -4.0, 0.0])
TWO_PASSES_ACCEL = np.array([0.0, 0.08, 0.0])


def two_pass_truth(ko):
    t = np.linspace(0.0, 100.0, 100001)
    margins = ko.margin(exact(N, TWO_PASSES, TWO_PASSES_ACCEL, t))
    k = int(np.argmin(margins))

    def margin(tau):
        return float(ko.margin(exact(N, TWO_PASSES, TWO_PASSES_ACCEL, tau)[0]))

    entry = brentq(margin, 0.0, t[k], xtol=1e-12)
    exit_ = brentq(margin, t[k], 50.0, xtol=1e-12)
    return float(margins[k]), entry, exit_


@pytest.mark.parametrize("samples", [2, 191])
def test_a_pass_between_two_samples_is_found_and_its_violation_reported(samples):
    # 191 samples is the environment's default density, 20 per 10 s; 2 leaves both passes in
    # one interval, which is flagged (its bound is -3 m) but refined to the shallower pass.
    ko = KeepOut(3.0)
    true_min, entry, exit_ = two_pass_truth(ko)
    assert true_min == pytest.approx(-2.0, abs=0.01)
    model, times, states = flown(TWO_PASSES, TWO_PASSES_ACCEL, 100.0, samples)
    thrust = float(np.linalg.norm(TWO_PASSES_ACCEL))
    found = segment_minimum(ko, model.state_in_last_segment, times, states, thrust, N)
    spans = violation_intervals(
        ko, model.state_in_last_segment, times, states, found, 0.0, thrust, N
    )
    assert found.lower_bound <= true_min
    assert found.margin <= true_min + ko.resolution
    assert len(spans) == 1
    assert spans[0] == pytest.approx((entry, exit_), abs=1e-6)
