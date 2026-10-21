"""Impulse execution and coast prediction shared by the classical planners."""

import dataclasses
import math

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv
from vbar.agents.base import command_fraction
from vbar.agents.planning import Impulse, impulse_action, predicted_margin
from vbar.constraints import KeepOut, segment_minimum
from vbar.dynamics import EARTH, CWDiscretizer, cw_discrete, mean_motion

N = mean_motion(EARTH.radius + 500e3)
# At the defaults one step delivers max_accel dt = 0.2 m/s per axis.
STEP_DV = 0.02 * 10.0
FAR = (0.0, -300.0, 0.0, 0.0, 0.0, 0.0)


def view_for(**config):
    return ApproachEnv(ApproachConfig(**config)).agent_view()


def chord_state(n, radius=50.0, dt=10.0):
    """Coast with both ends on the keep-out sphere, chord above the target, moving toward +y.

    The departure velocity is the CW targeting solution for the hop, about 0.2 m/s along +y. A
    constraint imposed at the sample times alone holds at both ends and fails between them
    (Rawlings, Mayne and Diehl, Model Predictive Control, section 8.5.1).
    """
    x = math.sqrt(radius**2 - 1.0)
    r0, rf = np.array([x, -1.0, 0.0]), np.array([x, 1.0, 0.0])
    phi, _ = cw_discrete(n, dt)
    v0 = np.linalg.solve(phi[:3, 3:], rf - phi[:3, :3] @ r0)
    return np.r_[r0, v0]


def reference_minimum(n, state, dt, samples=20):
    """The environment's own routine on `samples` sub-step points of a coast of `dt` s."""
    disc = CWDiscretizer(n)
    phis, _ = disc.sample_matrices(dt, samples)
    times = np.linspace(0.0, dt, samples)
    return segment_minimum(
        KeepOut(50.0), lambda tau: disc.matrices(tau)[0] @ state, times, phis @ state, 0.0, n
    )


def test_an_impulse_holds_its_time_and_hill_delta_v_and_cannot_be_changed():
    burn = Impulse(t=120.0, dv=np.array([0.1, -0.2, 0.0]))
    assert burn.t == 120.0
    assert np.array_equal(burn.dv, [0.1, -0.2, 0.0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        burn.t = 0.0


def test_an_impulse_within_one_step_is_its_fraction_of_a_full_step_and_the_coast():
    view = view_for(action_mode="impulse_coast")
    dv = np.array([0.1, -0.05, 0.02])
    action, saturated = impulse_action(view, dv, 120.0)
    assert action.shape == (4,) and not saturated
    assert np.allclose(action[:3], dv / STEP_DV, rtol=1e-14, atol=0.0)
    assert action[3] == 120.0


def test_an_impulse_above_one_step_is_split_into_equal_steps_and_marked_saturated():
    view = view_for(action_mode="impulse_coast")
    dv = np.array([0.3, -0.05, 0.01])
    k = math.ceil(np.max(np.abs(dv)) / STEP_DV)
    assert k == 2
    action, saturated = impulse_action(view, dv, 120.0)
    assert saturated and action.shape == (4,)
    assert np.allclose(action[:3], dv / (k * STEP_DV), rtol=1e-14, atol=0.0)
    assert action[3] == 0.0


def test_a_split_burn_delivers_the_whole_impulse_on_exact_cw():
    env = ApproachEnv(ApproachConfig(action_mode="impulse_coast", start_state=FAR, max_steps=20))
    env.reset(seed=0)
    dv = np.array([0.3, -0.05, 0.01])
    action, _ = impulse_action(env.agent_view(), dv, 0.0)
    for _ in range(2):
        info = env.step(action)[4]
    # The environment multiplies the same fraction by max_accel and dt, then sums the axes.
    assert info["dv_axes_mps"] == pytest.approx(np.sum(np.abs(dv)), rel=1e-12)


def test_a_ball_layout_splits_by_the_two_norm_where_a_box_delivers_in_one_step():
    dv = np.array([0.15, 0.15, 0.0])
    box, box_saturated = impulse_action(view_for(action_mode="impulse_coast"), dv, 0.0)
    ball_view = view_for(action_mode="impulse_coast", thrust_limit="ball")
    ball, ball_saturated = impulse_action(ball_view, dv, 0.0)
    assert not box_saturated and ball_saturated
    assert np.allclose(box[:3], dv / STEP_DV, rtol=1e-14, atol=0.0)
    assert np.allclose(ball[:3], dv / (2 * STEP_DV), rtol=1e-14, atol=0.0)
    cfg = ApproachConfig(
        action_mode="impulse_coast", thrust_limit="ball", start_state=FAR, max_steps=20
    )
    env = ApproachEnv(cfg)
    env.reset(seed=0)
    for _ in range(2):
        info = env.step(ball)[4]
    assert info["dv_used_mps"] == pytest.approx(np.linalg.norm(dv), rel=1e-12)


@pytest.mark.parametrize(
    ("mode", "scale"), [("normalized", 1.0), ("accel", 0.02), ("thrust", 10.0)]
)
def test_continuous_modes_command_the_clipped_rate_in_the_modes_units(mode, scale):
    view = view_for(action_mode=mode)
    full, saturated = impulse_action(view, np.array([0.0, 0.5, 0.0]), 0.0)
    half, unsaturated = impulse_action(view, np.array([0.0, -0.1, 0.0]), 0.0)
    assert saturated and not unsaturated
    assert full.shape == (3,) and half.shape == (3,)
    assert full == pytest.approx([0.0, scale, 0.0], rel=1e-14, abs=0.0)
    assert half == pytest.approx([0.0, -0.5 * scale, 0.0], rel=1e-14, abs=0.0)


@pytest.mark.parametrize("mode", ["normalized", "accel", "thrust"])
def test_continuous_modes_fly_an_impulse_as_saturated_steps_then_the_remainder(mode):
    env = ApproachEnv(ApproachConfig(action_mode=mode, start_state=FAR, max_steps=20))
    env.reset(seed=0)
    view = env.agent_view()
    residual = np.array([0.0, 0.5, 0.0])
    fractions, flags = [], []
    while np.max(np.abs(residual)) > 1e-12:
        action, saturated = impulse_action(view, residual, 0.0)
        fraction = command_fraction(view, action)
        fractions.append(float(fraction[1]))
        flags.append(saturated)
        info = env.step(action)[4]
        residual = residual - fraction * STEP_DV
    assert fractions == pytest.approx([1.0, 1.0, 0.5], rel=1e-12)
    assert flags == [True, True, False]
    # In thrust mode the commanded newtons act on a mass that falls by 2e-4 of itself over
    # the burn, so the delivered delta-v differs from the plan by that much.
    assert info["dv_axes_mps"] == pytest.approx(0.5, rel=1e-3 if mode == "thrust" else 1e-12)


def test_disc27_views_and_non_finite_impulses_are_refused():
    with pytest.raises(ValueError, match="disc27"):
        impulse_action(view_for(action_mode="disc27"), np.array([0.1, 0.0, 0.0]), 0.0)
    view = view_for(action_mode="impulse_coast")
    for bad in ([np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, -np.inf]):
        with pytest.raises(ValueError):
            impulse_action(view, np.array(bad), 0.0)


@pytest.mark.parametrize("mode", ["impulse_coast", "normalized", "accel", "thrust"])
def test_a_negative_or_non_finite_coast_is_refused_in_every_mode(mode):
    view = view_for(action_mode=mode)
    for bad in (-5.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="coast_s"):
            impulse_action(view, np.array([0.1, 0.0, 0.0]), bad)


def test_an_unknown_action_mode_is_refused():
    view = dataclasses.replace(view_for(action_mode="accel"), action_mode="torque")
    with pytest.raises(ValueError, match="unknown action mode"):
        impulse_action(view, np.array([0.1, 0.0, 0.0]), 0.0)


def test_predicted_margin_refuses_a_bad_state_a_bad_coast_and_too_few_samples():
    view = view_for(action_mode="impulse_coast")
    state = np.array([0.0, -300.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="state"):
        predicted_margin(view, state[:5], 10.0)
    with pytest.raises(ValueError, match="state"):
        predicted_margin(view, np.r_[state[:5], np.nan], 10.0)
    with pytest.raises(ValueError, match="coast_s"):
        predicted_margin(view, state, -1.0)
    with pytest.raises(ValueError, match="samples"):
        predicted_margin(view, state, 10.0, samples=1)


def test_a_nanometre_per_second_impulse_is_an_exact_fraction_without_saturation():
    view = view_for(action_mode="impulse_coast")
    action, saturated = impulse_action(view, np.array([1e-9, 0.0, 0.0]), 30.0)
    assert not saturated
    assert action[0] == pytest.approx(1e-9 / STEP_DV, rel=1e-14, abs=0.0)
    assert action[1] == 0.0 and action[2] == 0.0 and action[3] == 30.0


def test_a_hundred_metres_per_second_impulse_is_five_hundred_full_steps():
    view = view_for(action_mode="impulse_coast")
    action, saturated = impulse_action(view, np.array([0.0, 0.0, -100.0]), 30.0)
    assert saturated and action[3] == 0.0
    # 100 m/s over ceil(100 / 0.2) = 500 steps is a full command on every step.
    assert action[2] == pytest.approx(-1.0, rel=1e-14, abs=0.0)
    assert action[0] == 0.0 and action[1] == 0.0


def test_predicted_margin_finds_the_dip_between_the_sample_times_of_the_chord_coast():
    view = view_for(action_mode="impulse_coast")
    assert view.n == N and view.dt == 10.0
    state = chord_state(N)
    margin, bound = predicted_margin(view, state, view.dt)
    found = reference_minimum(N, state, view.dt)
    # Both paths evaluate the same samples, and the second-order bound puts every interval
    # within the 1 mm keep-out resolution of the best one, so neither refines.
    assert margin == pytest.approx(found.margin, abs=1e-9)
    # The sag of a 2 m chord of a 50 m sphere is 1.00 cm and the CW curvature adds 0.78 cm:
    # -1.783 cm, recomputed independently at the stated constants; the best sample is 0.05 mm
    # above it. Sampling the step ends alone reports 0.
    assert margin == pytest.approx(-0.01783, abs=1e-4)
    assert bound <= margin
    assert bound == pytest.approx(found.lower_bound, rel=1e-9)


def test_more_samples_keep_the_margin_and_tighten_the_certified_bound():
    view = view_for(action_mode="impulse_coast")
    state = chord_state(N)
    margin, bound = predicted_margin(view, state, view.dt)
    finer, tighter = predicted_margin(view, state, view.dt, samples=40)
    # Both are a best sample within the 1 mm keep-out resolution of the true minimum.
    assert finer == pytest.approx(margin, abs=1e-3)
    assert bound < tighter <= finer
    # The second-order gap shrinks like h^2, to a quarter per halving (7.1e-5 to 1.7e-5 m); the
    # Lipschitz term |v| h / 2 alone would only halve.
    assert finer - tighter < 0.3 * (margin - bound)


def test_the_predicted_chord_dip_is_the_margin_the_environment_scores():
    state = chord_state(N)
    cfg = ApproachConfig(action_mode="impulse_coast", start_state=tuple(state), max_steps=3)
    env = ApproachEnv(cfg)
    env.reset(seed=0)
    view = env.agent_view()
    margin, bound = predicted_margin(view, state, view.dt)
    info = env.step(np.zeros(4))[4]
    assert info["keep_out_violated"]
    assert info["min_margins"]["keep_out"] == pytest.approx(margin, abs=1e-9)
    assert info["min_margin_bounds"]["keep_out"] == pytest.approx(bound, rel=1e-9)


def test_a_multi_step_coast_is_predicted_to_the_environments_minimum_and_certified_tightly():
    # From 55 m above and 40 m behind the target at 0.5 m/s along +y, the range dips to 58.2 m
    # near t = 62 s (recomputed independently), well inside a 90 s coast.
    state = np.array([55.0, -40.0, 0.0, 0.0, 0.5, 0.0])
    cfg = ApproachConfig(action_mode="impulse_coast", start_state=tuple(state), max_steps=12)
    env = ApproachEnv(cfg)
    env.reset(seed=0)
    view = env.agent_view()
    margin, bound = predicted_margin(view, state, 9 * view.dt)
    info = env.step(np.array([0.0, 0.0, 0.0, 8 * view.dt]))[4]
    assert info["time_s"] == 9 * view.dt
    assert 5.0 < margin < 15.0 and not info["keep_out_violated"]
    # The prediction samples the whole coast and the environment each step; neither refines,
    # since the second-order bound is within the 1 mm keep-out resolution, so the two best
    # samples agree to that resolution (2.1e-5 m apart).
    assert info["min_margins"]["keep_out"] == pytest.approx(margin, abs=1e-3)
    # The Lipschitz term |v| h / 2 alone would be 0.13 m per 10 s step at 0.5 m/s; the
    # second-order bound sits 0.13 mm below the margin.
    assert bound <= margin
    assert margin - bound < 1e-3


def test_a_sample_spacing_too_long_for_the_rate_bound_gives_an_honest_minus_infinity():
    # The speed bound between samples solves a fixed point that needs
    # h (1.5 n^2 h + 2 n) < 1, false for the 421 s spacing of 20 samples over an 8000 s step,
    # so the certified bound is -inf while the sampled margin stays finite.
    view = view_for(action_mode="impulse_coast", dt=8000.0, max_steps=2)
    state = np.array([0.0, -300.0, 0.0, 0.0, 0.0, 0.0])
    margin, bound = predicted_margin(view, state, view.dt)
    assert np.isfinite(margin) and margin > 0.0
    assert bound == -np.inf
