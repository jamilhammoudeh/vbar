import numpy as np
import pytest
from metric_cases import ConstantAgent, episode, pd_episode

from vbar.conventions import G0
from vbar.metrics import FuelBounds, efficiency_metrics, evaluate, optimality_gap


class ScriptedAgent:
    """Plays a fixed list of normalized commands, then coasts."""

    name = "scripted"

    def __init__(self, commands):
        self.commands = [np.asarray(c, dtype=float) for c in commands]

    def reset(self, view, rng=None):
        self.k = 0

    def act(self, obs, info):
        u = self.commands[self.k] if self.k < len(self.commands) else np.zeros(3)
        self.k += 1
        return u


def test_constant_burn_has_closed_form_delta_v_and_propellant():
    ep = episode(ConstantAgent([0.5, -0.5, 0.0]), max_steps=3)
    m = efficiency_metrics(ep)
    # 0.02 m/s^2 per axis at half command for 30 s.
    assert m["delta_v"] == pytest.approx(0.01 * np.sqrt(2.0) * 30.0, rel=1e-12)
    assert m["delta_v_axes"] == pytest.approx(0.02 * 30.0, rel=1e-12)
    expected = 500.0 * (1.0 - np.exp(-0.6 / (220.0 * G0)))
    assert m["propellant_mass"] == pytest.approx(expected, rel=1e-12)
    # The actuator's own bookkeeping, burn by burn, gives the same mass.
    flown = sum(rec.actuation.propellant_kg for rec in ep.records)
    assert m["propellant_mass"] == pytest.approx(flown, rel=1e-12)
    assert m["burn_count"] == 1 and m["duty_cycle"] == 1.0 and m["chatter"] == 0.0


def test_ball_layout_spends_propellant_on_the_two_norm():
    ep = episode(ConstantAgent([0.6, 0.8, 0.0]), max_steps=2, thrust_limit="ball")
    m = efficiency_metrics(ep)
    assert m["delta_v"] == pytest.approx(0.02 * 20.0, rel=1e-12)
    expected = 500.0 * (1.0 - np.exp(-m["delta_v"] / (220.0 * G0)))
    assert m["propellant_mass"] == pytest.approx(expected, rel=1e-12)


def test_constant_thrust_delta_v_is_the_rocket_equation():
    ep = episode(ConstantAgent([1.0, 0.0, 0.0]), max_steps=5, drive="thrust", propellant_kg=50.0)
    m = efficiency_metrics(ep)
    m1 = 500.0 - m["propellant_mass"]
    assert m["delta_v"] == pytest.approx(220.0 * G0 * np.log(500.0 / m1), rel=1e-12)


def test_burns_duty_cycle_and_chatter_from_a_scripted_pattern():
    commands = [(1, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0.5, 0), (0, 0, 0), (0, -0.5, 0)]
    m = efficiency_metrics(episode(ScriptedAgent(commands), max_steps=8))
    assert m["burn_count"] == 3
    assert m["duty_cycle"] == pytest.approx(4 / 8)
    # x: +, -, + is two changes; y: +, - is one; over 80 s.
    assert m["chatter"] == pytest.approx(3 / 80.0)


def test_delay_moves_the_executed_burn_not_its_cost():
    commands = [(1, 0, 0), (0, 0, 0), (0, 0, 0)]
    prompt = efficiency_metrics(episode(ScriptedAgent(commands), max_steps=3))
    late = efficiency_metrics(episode(ScriptedAgent(commands), max_steps=3, delay_s=5.0))
    assert late["delta_v"] == pytest.approx(prompt["delta_v"], rel=1e-12)
    # Delayed by half a step, the burn spans two steps but is still one burn.
    assert late["burn_count"] == 1 and late["duty_cycle"] == pytest.approx(prompt["duty_cycle"])


def test_delta_v_does_not_depend_on_the_decision_step():
    reference = efficiency_metrics(episode(ConstantAgent([0.3, 0.2, -0.1]), max_steps=6))
    m = efficiency_metrics(episode(ConstantAgent([0.3, 0.2, -0.1]), dt=2.5, max_steps=24))
    for name in ("delta_v", "delta_v_axes", "propellant_mass", "duty_cycle"):
        assert m[name] == pytest.approx(reference[name], rel=1e-12)


def test_two_norm_delta_v_is_unchanged_by_rotating_the_command_for_a_ball_thruster():
    a = efficiency_metrics(
        episode(ConstantAgent([0.6, 0.0, 0.0]), max_steps=3, thrust_limit="ball")
    )
    b = efficiency_metrics(
        episode(ConstantAgent([0.0, 0.36, 0.48]), max_steps=3, thrust_limit="ball")
    )
    assert a["delta_v"] == pytest.approx(b["delta_v"], rel=1e-12)
    assert a["propellant_mass"] == pytest.approx(b["propellant_mass"], rel=1e-12)


def test_pd_spends_less_than_its_box_cost_on_the_two_norm():
    m = evaluate(pd_episode())
    assert 0.0 < m["delta_v"] <= m["delta_v_axes"] <= np.sqrt(3.0) * m["delta_v"]


def test_optimality_gap_is_delta_v_over_each_bound_for_a_success():
    ep = pd_episode()
    dv = efficiency_metrics(ep)["delta_v_axes"]
    gap = optimality_gap(ep, FuelBounds(0.5 * dv, 0.8 * dv, "l1"))
    assert gap["optimality_gap"] == pytest.approx(2.0)
    assert gap["optimality_gap_upper"] == pytest.approx(1.25)
    assert not gap["below_lower_bound"]


def test_a_run_below_its_certified_lower_bound_is_flagged():
    ep = pd_episode()
    dv = efficiency_metrics(ep)["delta_v_axes"]
    gap = optimality_gap(ep, FuelBounds(1.01 * dv, 2.0 * dv, "l1"))
    assert gap["below_lower_bound"] and gap["optimality_gap"] < 1.0
    # Rounding at the bound is not a bug.
    assert not optimality_gap(ep, FuelBounds(dv * (1 + 1e-12), 2 * dv, "l1"))["below_lower_bound"]


def test_optimality_gap_is_nan_for_a_failure_or_without_bounds():
    # The bounds are for reaching the success set, so a failure below them is not flagged.
    failed = optimality_gap(episode(), FuelBounds(0.1, 0.2, "l1"))
    assert np.isnan(failed["optimality_gap"]) and np.isnan(failed["optimality_gap_upper"])
    assert not failed["below_lower_bound"]
    assert np.isnan(evaluate(pd_episode())["optimality_gap"])


def test_bounds_must_use_the_actuator_norm():
    # Boyd and Vandenberghe review C3: box thrusters are scored on the 1-norm.
    with pytest.raises(ValueError, match="l1"):
        optimality_gap(pd_episode(), FuelBounds(0.1, 0.2, "l2"))
    with pytest.raises(ValueError, match="lower <= upper"):
        FuelBounds(0.3, 0.2, "l1")
