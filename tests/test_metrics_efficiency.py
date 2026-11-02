import math
import warnings

import numpy as np
import pytest
from metric_cases import ConstantAgent, episode, pd_episode

from vbar.conventions import G0
from vbar.metrics import FuelBounds, efficiency_metrics, evaluate, optimality_gap
from vbar.metrics.mission import success
from vbar.oracles import FuelOracle, OracleConfig, fuel
from vbar.oracles.fuel import bounds_for


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
    gap = optimality_gap(ep, FuelBounds(0.5 * dv, 0.8 * dv, "l1", "cw"))
    assert gap["optimality_gap"] == pytest.approx(2.0)
    assert gap["optimality_gap_upper"] == pytest.approx(1.25)
    assert not gap["below_lower_bound"]


def test_a_run_below_its_certified_lower_bound_is_flagged():
    ep = pd_episode()
    dv = efficiency_metrics(ep)["delta_v_axes"]
    gap = optimality_gap(ep, FuelBounds(1.01 * dv, 2.0 * dv, "l1", "cw"))
    assert gap["below_lower_bound"] and gap["optimality_gap"] < 1.0
    # Rounding at the bound is not a bug.
    assert not optimality_gap(ep, FuelBounds(dv * (1 + 1e-12), 2 * dv, "l1", "cw"))[
        "below_lower_bound"
    ]


def test_optimality_gap_is_nan_for_a_failure_or_without_bounds():
    # The bounds are for reaching the success set, so a failure below them is not flagged.
    failed = optimality_gap(episode(), FuelBounds(0.1, 0.2, "l1", "cw"))
    assert np.isnan(failed["optimality_gap"]) and np.isnan(failed["optimality_gap_upper"])
    assert not failed["below_lower_bound"]
    assert np.isnan(evaluate(pd_episode())["optimality_gap"])


def test_bounds_must_use_the_actuator_norm():
    # A bound is valid only in the cost's own norm (Boyd and Vandenberghe, eqs. 5.11 to 5.12);
    # box thrusters are scored on the 1-norm.
    with pytest.raises(ValueError, match="l1"):
        optimality_gap(pd_episode(), FuelBounds(0.1, 0.2, "l2", "cw"))
    with pytest.raises(ValueError, match="lower <= upper"):
        FuelBounds(0.3, 0.2, "l1", "cw")


def test_a_bound_for_another_dynamics_rung_is_a_model_ratio_and_never_flagged():
    # A CW bound on two-body truth: the two-body optimum can lie on either side of it, so a run
    # below it is no bug (Rawlings, Mayne and Diehl, 8.1.2 and 8.10).
    ep = pd_episode(dynamics="twobody")
    dv = efficiency_metrics(ep)["delta_v_axes"]
    gap = optimality_gap(ep, FuelBounds(1.01 * dv, 2.0 * dv, "l1", "cw"))
    assert not gap["bound_certified"] and gap["bound_rung"] == "cw"
    assert np.isnan(gap["optimality_gap"]) and not gap["below_lower_bound"]
    assert gap["optimality_gap_model"] == pytest.approx(1 / 1.01)
    certified = optimality_gap(ep, FuelBounds(0.5 * dv, 2.0 * dv, "l1", "twobody"))
    assert certified["bound_certified"] and certified["optimality_gap"] == pytest.approx(2.0)
    assert np.isnan(certified["optimality_gap_model"])


def infeasible_bounds(ep, reason, **fields):
    dv = efficiency_metrics(ep)["delta_v_axes"]
    return FuelBounds(0.0, math.inf, "l1", "cw", infeasible=reason, **fields), dv


@pytest.mark.parametrize(
    "reason", ["infeasible_within_limit", "infeasible_without_limit", "no_feasible_window"]
)
def test_a_bound_on_an_infeasible_verdict_is_uncertified_with_no_infinite_ratio(reason):
    ep = pd_episode()
    bounds, _ = infeasible_bounds(ep, reason, accel_limit=ep.view.max_accel)
    with pytest.warns(RuntimeWarning, match=reason):
        gap = optimality_gap(ep, bounds)
    assert not gap["bound_certified"] and gap["bound_reason"] == reason
    assert gap["bound_rung"] == "cw" and not gap["below_lower_bound"]
    for key in ("optimality_gap", "optimality_gap_model", "optimality_gap_upper"):
        assert np.isnan(gap[key])
    assert not any(isinstance(v, float) and math.isinf(v) for v in gap.values())


def test_a_success_the_infeasible_verdict_forbids_is_a_solver_error():
    ep = pd_episode()
    with pytest.warns(RuntimeWarning, match="solver error"):
        gap = optimality_gap(ep, infeasible_bounds(ep, "infeasible_without_limit")[0])
    assert gap["bound_solver_error"]


def test_a_run_over_the_limit_does_not_contradict_an_infeasible_verdict_within_it():
    ep = pd_episode()
    bounds, _ = infeasible_bounds(ep, "infeasible_within_limit", accel_limit=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gap = optimality_gap(ep, bounds)
    assert not gap["bound_solver_error"] and not gap["bound_certified"]


def test_a_run_that_splits_its_steps_does_not_contradict_an_infeasible_verdict():
    # A delayed command spans two steps, so the run is no point of the LP with one input a step.
    ep = pd_episode(delay_s=5.0)
    assert success(ep)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gap = optimality_gap(ep, infeasible_bounds(ep, "infeasible_without_limit")[0])
    assert not gap["bound_solver_error"] and not gap["bound_certified"]


def test_a_failure_or_another_rung_does_not_contradict_an_infeasible_verdict():
    failed = episode()
    other = pd_episode(dynamics="twobody")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for ep in (failed, other):
            gap = optimality_gap(ep, infeasible_bounds(ep, "infeasible_without_limit")[0])
            assert not gap["bound_solver_error"] and gap["bound_reason"]


def test_a_forced_infeasible_oracle_verdict_flags_the_successful_run(monkeypatch):
    # The window LP of a start the PD controller flies to success is made to report infeasible;
    # the run satisfies the outer relaxation, so the verdict is the error.
    ep = pd_episode()
    assert success(ep)
    real = fuel._outer

    def outer(win, norm, config):
        _, _, cuts, _ = real(win, norm, config)
        return None, math.inf, cuts, fuel.INFEASIBLE_WITHOUT_LIMIT

    monkeypatch.setattr(fuel, "_outer", outer)
    bounds = bounds_for(ep, oracle=FuelOracle(config=OracleConfig(bank_stride=50)))
    assert bounds.infeasible == "no_feasible_window"
    with pytest.warns(RuntimeWarning, match="solver error"):
        gap = optimality_gap(ep, bounds)
    assert gap["bound_solver_error"] and np.isnan(gap["optimality_gap"])


def test_a_start_without_an_infeasible_verdict_keeps_its_certified_gap():
    ep = pd_episode()
    gap = optimality_gap(ep, FuelBounds(0.5, 2.0, "l1", "cw"))
    assert gap["bound_reason"] == "" and not gap["bound_solver_error"]
    assert gap["bound_certified"]
