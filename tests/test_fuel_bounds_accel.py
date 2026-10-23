"""E3 with the acceleration limit a fuel bound assumed.

A finite-thrust lower bound holds only for acceleration histories within its limit. Under the
constant-thrust drive the acceleration rises as propellant burns, and an execution scale bias is
not limited again, so a run can fly harder than `max_accel` and legally spend less than a bound
computed for `max_accel`. Such a bound is a model bound for that run, never a certified one.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from metric_cases import GuardedApproach, episode, pd_episode

from vbar import ApproachConfig
from vbar.agents import PDAgent
from vbar.metrics import (
    METRICS,
    FuelBounds,
    efficiency_metrics,
    evaluate,
    optimality_gap,
    record_episode,
    success,
)
from vbar.spacecraft import ExecutionError

GAP_KEYS = (
    "optimality_gap",
    "optimality_gap_model",
    "optimality_gap_upper",
    "below_lower_bound",
    "bound_certified",
    "bound_rung",
)
THRUST = {"drive": "thrust", "propellant_kg": 50.0}
BIAS = {"execution": ExecutionError(scale_bias=0.05)}


def gap_before_accel_limit(ep, bounds):
    """E3 as defined before bounds named their acceleration limit, kept to pin the defaults."""
    gap = {
        "optimality_gap": math.nan,
        "optimality_gap_model": math.nan,
        "optimality_gap_upper": math.nan,
        "below_lower_bound": False,
        "bound_certified": False,
        "bound_rung": "",
    }
    if bounds is None:
        return gap
    certified = bounds.rung == ep.dynamics
    gap.update(bound_certified=certified, bound_rung=bounds.rung)
    if not success(ep):
        return gap
    dv = layout_dv(ep)
    ratio = dv / bounds.lower if bounds.lower > 0.0 else math.inf
    gap["optimality_gap" if certified else "optimality_gap_model"] = ratio
    gap["below_lower_bound"] = certified and bool(dv < bounds.lower * (1.0 - 1e-9))
    gap["optimality_gap_upper"] = dv / bounds.upper if bounds.upper > 0.0 else math.inf
    return gap


def layout_dv(ep):
    m = efficiency_metrics(ep)
    return m["delta_v_axes"] if ep.view.thrust_limit == "box" else m["delta_v"]


def flown_peak(ep, order):
    """Largest executed acceleration in the record, in the given vector norm (m/s^2)."""
    return max(
        float(np.linalg.norm(accel, ord=order))
        for rec in ep.records
        for _, accel in rec.actuation.segments
    )


def assert_same_gap(actual, expected):
    assert set(GAP_KEYS) <= set(actual)
    for key in GAP_KEYS:
        a, b = actual[key], expected[key]
        if isinstance(b, float) and math.isnan(b):
            assert isinstance(a, float) and math.isnan(a), key
        else:
            assert a == b, key


def norm_of(ep):
    return "l1" if ep.view.thrust_limit == "box" else "l2"


@pytest.fixture(scope="module")
def runs():
    return {
        "box": pd_episode(),
        "ball": pd_episode(thrust_limit="ball"),
        "twobody": pd_episode(dynamics="twobody"),
        "failed": episode(),
        "thrust": pd_episode(**THRUST),
        "bias": pd_episode(**BIAS),
        "ball_thrust": pd_episode(thrust_limit="ball", **THRUST),
        "ball_bias": pd_episode(thrust_limit="ball", **BIAS),
    }


def test_new_fields_default_to_no_limit_and_no_impulsive_bound():
    bounds = FuelBounds(0.1, 0.2, "l1", "cw")
    assert bounds.accel_limit == math.inf
    assert bounds.impulsive_lower is None


@pytest.mark.parametrize(
    "name", ["box", "ball", "twobody", "failed", "thrust", "bias", "ball_thrust", "ball_bias"]
)
def test_defaults_reproduce_every_e3_entry_of_today(runs, name):
    ep = runs[name]
    dv = layout_dv(ep)
    norm = norm_of(ep)
    cases = [
        None,
        FuelBounds(0.5 * dv, 0.8 * dv, norm, "cw"),
        FuelBounds(1.01 * dv, 2.0 * dv, norm, "cw"),
        FuelBounds(dv * (1 + 1e-12), 2.0 * dv, norm, "cw"),
        FuelBounds(0.5 * dv, 2.0 * dv, norm, "twobody"),
        FuelBounds(0.0, 2.0 * dv, norm, "cw"),
        FuelBounds(0.0, 0.0, norm, "cw"),
    ]
    for bounds in cases:
        assert_same_gap(optimality_gap(ep, bounds), gap_before_accel_limit(ep, bounds))


def test_runs_used_below_fly_harder_than_max_accel_only_under_thrust_drive_or_bias(runs):
    # Preconditions of the certification tests: each run succeeds, and the record shows the
    # acceleration the bound must cover.
    for ep in runs.values():
        assert ep is runs["failed"] or success(ep)
    a_max = runs["box"].view.max_accel
    assert flown_peak(runs["box"], np.inf) == pytest.approx(a_max, rel=1e-12)
    assert flown_peak(runs["ball"], 2) == pytest.approx(a_max, rel=1e-12)
    assert flown_peak(runs["thrust"], np.inf) > a_max * (1 + 1e-6)
    assert flown_peak(runs["bias"], np.inf) == pytest.approx(1.05 * a_max, rel=1e-12)
    assert flown_peak(runs["ball_thrust"], 2) > a_max * (1 + 1e-6)
    assert flown_peak(runs["ball_bias"], 2) == pytest.approx(1.05 * a_max, rel=1e-12)


@pytest.mark.parametrize("name", ["box", "ball"])
def test_bound_at_max_accel_is_certified_for_a_constant_acceleration_run(runs, name):
    # Box thrusters reach max_accel on two axes at once, so the box limit is per axis; a ball
    # command scaled onto the sphere lands within rounding of it.
    ep = runs[name]
    dv = layout_dv(ep)
    a_max = ep.view.max_accel
    gap = optimality_gap(ep, FuelBounds(0.5 * dv, 2.0 * dv, norm_of(ep), "cw", accel_limit=a_max))
    assert gap["bound_certified"] and gap["bound_rung"] == "cw"
    assert gap["optimality_gap"] == pytest.approx(2.0, rel=1e-12)
    assert math.isnan(gap["optimality_gap_model"])
    below = optimality_gap(
        ep, FuelBounds(1.01 * dv, 2.0 * dv, norm_of(ep), "cw", accel_limit=a_max)
    )
    assert below["below_lower_bound"]


@pytest.mark.parametrize("name", ["thrust", "bias", "ball_thrust", "ball_bias"])
def test_bound_for_less_acceleration_than_flown_is_a_model_bound(runs, name):
    ep = runs[name]
    dv = layout_dv(ep)
    bounds = FuelBounds(1.01 * dv, 2.0 * dv, norm_of(ep), "cw", accel_limit=ep.view.max_accel)
    gap = optimality_gap(ep, bounds)
    assert not gap["bound_certified"]
    assert math.isnan(gap["optimality_gap"])
    assert gap["optimality_gap_model"] == pytest.approx(1 / 1.01, rel=1e-12)
    assert not gap["below_lower_bound"]
    assert gap["optimality_gap_upper"] == pytest.approx(0.5, rel=1e-12)


@pytest.mark.parametrize(
    ("name", "limit"),
    [
        # The thrust runs burn under 1 kg of 500 kg, so 0.0201 m/s^2 covers both the flown peak
        # and the most the thrust could give at the final mass.
        ("thrust", 0.0201),
        ("ball_thrust", 0.0201),
        ("bias", 1.05 * 0.02),
        ("ball_bias", 1.05 * 0.02),
    ],
)
def test_bound_for_the_acceleration_flown_is_certified(runs, name, limit):
    ep = runs[name]
    dv = layout_dv(ep)
    gap = optimality_gap(ep, FuelBounds(0.5 * dv, 2.0 * dv, norm_of(ep), "cw", accel_limit=limit))
    assert gap["bound_certified"]
    assert gap["optimality_gap"] == pytest.approx(2.0, rel=1e-12)
    assert math.isnan(gap["optimality_gap_model"])


def test_ball_limit_is_checked_on_the_two_norm(runs):
    # The biased ball run flies 0.021 m/s^2 in the 2-norm, about 0.0209 on its largest axis.
    ep = runs["ball_bias"]
    dv = layout_dv(ep)
    gap = optimality_gap(ep, FuelBounds(0.5 * dv, 2.0 * dv, "l2", "cw", accel_limit=0.0205))
    assert not gap["bound_certified"] and math.isnan(gap["optimality_gap"])


def test_the_rung_rule_still_applies_with_a_matching_limit(runs):
    ep = runs["twobody"]
    dv = layout_dv(ep)
    gap = optimality_gap(ep, FuelBounds(1.01 * dv, 2.0 * dv, "l1", "cw", accel_limit=0.02))
    assert not gap["bound_certified"] and not gap["below_lower_bound"]
    assert gap["optimality_gap_model"] == pytest.approx(1 / 1.01, rel=1e-12)


def test_an_exceeded_limit_is_scored_against_the_impulsive_bound(runs):
    # The impulsive bound holds for any acceleration history; the run lies above it, so no
    # theorem is broken even though it lies below the finite-thrust bound.
    ep = runs["thrust"]
    dv = layout_dv(ep)
    impulsive = 0.5 * dv
    bounds = FuelBounds(
        1.01 * dv, 2.0 * dv, "l1", "cw", accel_limit=0.02, impulsive_lower=impulsive
    )
    gap = optimality_gap(ep, bounds)
    assert gap["bound_certified"] and not gap["below_lower_bound"]
    assert gap["optimality_gap"] == dv / impulsive
    assert gap["optimality_gap_model"] == pytest.approx(1 / 1.01, rel=1e-12)


def test_a_run_below_the_impulsive_bound_is_flagged_whatever_it_flew(runs):
    ep = runs["thrust"]
    dv = layout_dv(ep)
    bounds = FuelBounds(
        1.02 * dv, 2.0 * dv, "l1", "cw", accel_limit=0.02, impulsive_lower=1.01 * dv
    )
    gap = optimality_gap(ep, bounds)
    assert gap["bound_certified"] and gap["below_lower_bound"]
    assert gap["optimality_gap"] == pytest.approx(1 / 1.01, rel=1e-12)


def test_the_impulsive_bound_is_not_certified_on_another_rung(runs):
    # A CW impulsive bound is no theorem for two-body motion either.
    ep = runs["twobody"]
    dv = layout_dv(ep)
    bounds = FuelBounds(1.01 * dv, 2.0 * dv, "l1", "cw", accel_limit=0.02, impulsive_lower=0.5 * dv)
    gap = optimality_gap(ep, bounds)
    assert not gap["bound_certified"] and not gap["below_lower_bound"]
    assert math.isnan(gap["optimality_gap"])
    assert gap["optimality_gap_model"] == pytest.approx(1 / 1.01, rel=1e-12)


def test_thrust_drive_limit_covers_the_acceleration_at_the_end_of_each_burn(runs):
    # Under constant thrust the acceleration at the end of a step is the thrust over the mass
    # left, above both recorded half-burn values, so a bound between the recorded peak and that
    # peak scaled by its step's mass ratio is not certified.
    ep = runs["thrust"]
    mass = ep.view.mass_kg - np.cumsum([0.0, *(r.actuation.propellant_kg for r in ep.records)])
    recorded = flown_peak(ep, np.inf)
    reached = max(
        mass[k] / mass[k + 1] * float(np.max(np.abs(accel)))
        for k, rec in enumerate(ep.records)
        for _, accel in rec.actuation.segments
    )
    assert reached > recorded * (1 + 1e-6)
    dv = layout_dv(ep)
    limit = math.sqrt(recorded * reached)
    gap = optimality_gap(ep, FuelBounds(0.5 * dv, 2.0 * dv, "l1", "cw", accel_limit=limit))
    assert not gap["bound_certified"] and math.isnan(gap["optimality_gap"])


def test_infinite_upper_bound_gives_nan_upper_ratio(runs):
    ep = runs["box"]
    dv = layout_dv(ep)
    gap = optimality_gap(ep, FuelBounds(0.5 * dv, math.inf, "l1", "cw"))
    assert math.isnan(gap["optimality_gap_upper"])
    assert gap["optimality_gap"] == pytest.approx(2.0, rel=1e-12)
    model = optimality_gap(runs["twobody"], FuelBounds(0.0, math.inf, "l1", "cw"))
    assert math.isnan(model["optimality_gap_upper"])


class DishonestAgent:
    """Flies the PD law and claims, every way it can, that its bound holds.

    It reports an unlimited constant-acceleration drive with no scale bias on itself, in its
    per-decision log and in the info it is handed, carries a fuel bound of its own with an
    unlimited acceleration, and rewrites the view it was given.
    """

    name = "pd"
    drive = "accel"
    scale_bias = 0.0
    accel_limit = math.inf
    max_accel = math.inf
    bound_certified = True

    def __init__(self):
        self.pd = PDAgent()
        self.bounds = FuelBounds(1e-9, 1.0, "l1", "cw", accel_limit=math.inf)
        self.fuel_bounds = self.bounds
        self.log = []

    def reset(self, view, rng=None):
        self.pd.reset(view, rng)
        self.log = []
        object.__setattr__(view, "max_accel", math.inf)

    def act(self, obs, info):
        info.update(
            accel_limit=math.inf,
            drive="accel",
            scale_bias=0.0,
            bound_certified=True,
            fuel_bounds=self.bounds,
        )
        self.log.append(
            {
                "accel_limit": math.inf,
                "max_accel": math.inf,
                "drive": "accel",
                "bound_certified": True,
                "fuel_bounds": self.bounds,
            }
        )
        return self.pd.act(obs, info)


def dishonest_run(**config):
    """The PD run of `pd_episode` flown by `DishonestAgent`, with the same seeds."""
    config.setdefault("start_state", (0.0, -400.0, 30.0, 0.0, 0.0, 0.0))
    config.setdefault("max_steps", 200)
    env = GuardedApproach(ApproachConfig(**config))
    return record_episode(env, DishonestAgent(), seed=0, agent_seed=1)


def test_reported_drive_limit_or_bound_cannot_certify_a_bound(runs):
    ep = dishonest_run(**THRUST)
    honest = runs["thrust"]
    assert ep.agent_log and ep.agent_log[0]["accel_limit"] == math.inf
    dv = layout_dv(ep)
    assert dv == layout_dv(honest)
    bounds = FuelBounds(1.01 * dv, 2.0 * dv, "l1", "cw", accel_limit=ep.view.max_accel)
    gap = optimality_gap(ep, bounds)
    assert not gap["bound_certified"] and not gap["below_lower_bound"]
    assert math.isnan(gap["optimality_gap"])
    assert_same_gap(gap, optimality_gap(honest, bounds))
    assert_same_gap(evaluate(ep, bounds=bounds), gap)


def test_episode_records_the_drive_the_environment_flew(runs):
    assert runs["box"].drive == "accel"
    assert runs["bias"].drive == "accel"
    assert runs["thrust"].drive == "thrust"
    assert runs["ball_thrust"].drive == "thrust"


def test_an_agent_cannot_change_the_recorded_drive():
    assert dishonest_run(**THRUST).drive == "thrust"


def test_a_bound_the_agent_carries_is_never_used(runs):
    scored = evaluate(dishonest_run(**THRUST))
    for key in ("optimality_gap", "optimality_gap_model", "optimality_gap_upper"):
        assert math.isnan(scored[key])
    assert not scored["bound_certified"] and not scored["below_lower_bound"]
    assert scored["bound_rung"] == ""


def doctored(ep, accel_scale, duration_scale):
    """`ep` with every executed segment rescaled, so the record claims a different delta-v."""
    scale = accel_scale * duration_scale
    records = tuple(
        replace(
            rec,
            actuation=replace(
                rec.actuation,
                segments=tuple(
                    (duration * duration_scale, np.asarray(accel) * accel_scale)
                    for duration, accel in rec.actuation.segments
                ),
                dv_l2=rec.actuation.dv_l2 * scale,
                dv_l1=rec.actuation.dv_l1 * scale,
                propellant_kg=rec.actuation.propellant_kg * scale,
            ),
        )
        for rec in ep.records
    )
    return replace(ep, records=records)


def test_record_with_half_the_delta_v_is_flagged_only_within_the_limit(runs):
    # Honest runs give the bounds; each doctored record keeps the true trajectory, so it still
    # succeeds, but claims half the delta-v.
    box = runs["box"]
    dv = layout_dv(box)
    bounds = FuelBounds(0.9 * dv, 2.0 * dv, "l1", "cw", accel_limit=box.view.max_accel)
    gentle = doctored(box, accel_scale=0.5, duration_scale=1.0)
    assert success(gentle) and layout_dv(gentle) == pytest.approx(0.5 * dv, rel=1e-12)
    assert optimality_gap(gentle, bounds)["below_lower_bound"]

    thrust = runs["thrust"]
    dv = layout_dv(thrust)
    bounds = FuelBounds(0.9 * dv, 2.0 * dv, "l1", "cw", accel_limit=thrust.view.max_accel)
    hard = doctored(thrust, accel_scale=1.0, duration_scale=0.5)
    assert success(hard) and layout_dv(hard) == pytest.approx(0.5 * dv, rel=1e-12)
    gap = optimality_gap(hard, bounds)
    assert not gap["below_lower_bound"] and not gap["bound_certified"]


@pytest.mark.parametrize("value", [-1e-3, -math.inf, math.nan])
def test_accel_limit_must_be_a_nonnegative_number(value):
    with pytest.raises(ValueError, match="accel_limit"):
        FuelBounds(0.1, 0.2, "l1", "cw", accel_limit=value)


@pytest.mark.parametrize("value", [-1e-3, -math.inf, math.inf, math.nan])
def test_impulsive_lower_must_be_finite_and_nonnegative(value):
    with pytest.raises(ValueError, match="impulsive_lower"):
        FuelBounds(0.1, 0.2, "l1", "cw", impulsive_lower=value)


def test_impulsive_lower_above_upper_is_an_inconsistent_bound():
    with pytest.raises(ValueError, match="impulsive_lower"):
        FuelBounds(0.1, 0.2, "l1", "cw", impulsive_lower=0.2 * (1 + 1e-9))
    assert FuelBounds(0.1, 0.2, "l1", "cw", impulsive_lower=0.2).impulsive_lower == 0.2


@pytest.mark.parametrize(
    "name",
    [
        "optimality_gap",
        "optimality_gap_model",
        "optimality_gap_upper",
        "below_lower_bound",
        "bound_certified",
    ],
)
def test_changed_e3_definitions_bump_their_versions(name):
    assert METRICS[name].version >= 2
