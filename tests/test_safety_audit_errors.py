"""The audit under truncated ("bounded") and untruncated ("gaussian") error models.

The interface these tests assume:

- `audit_filter(..., errors=...)` takes "none" (the default, perfect navigation and no
  execution error), "bounded" or "gaussian". "bounded" needs truncated `GaussianNav` and
  `ExecutionError` models (or none of either) whose truncation equals the filter's declared
  `per_draw_level`, and declared error sets that contain the sets `tightening.declared_errors`
  derives from the models at that level. "gaussian" needs the same models untruncated.
- `FilterAudit` gains `errors`, `exceedance` (p), `per_draw_level` (b) and `draws` (D, from
  `tightening.episode_draws` over the episode's steps), and `uncovered`, as `RunAudit` does: the
  decisions without a certificate, except those commanding zero thrust before the last
  certified decision's time plus its horizon.
- Under "gaussian" `passed` is None, and `exceedance_rates` maps each covered constraint and
  "any" to a rate with `count` (failing runs), `trials`, `rate`, `interval` (95%
  Clopper-Pearson) and `level`: p per constraint and min(1, sum of p over the constraints) for
  "any".
- A filter, and a `FilterDeclaration`, takes `exceedance` and `per_draw_level` together, and only
  with a declared error set.
"""

import dataclasses

import numpy as np
import pytest
from scipy.stats import beta

from vbar.agents.base import action_scale
from vbar.constraints import KeepOut, SegmentMinimum, SpeedLimit
from vbar.dynamics.cw import cw_states
from vbar.safety import FilterResult, tightening
from vbar.safety.audit import audit_filter
from vbar.safety.cbf import BackupFilter, CbfFilter
from vbar.safety.predictive import PredictiveFilter
from vbar.safety.tightening import BoxError
from vbar.scenarios import base as base_module
from vbar.scenarios import per_draw_level
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.spacecraft import ExecutionError, GaussianNav, GaussMarkovNav, RangeScaledNav

# A fixed 128-bit root seed, so the drawn starts and seeds are the same on every run.
SEED = 0x9E3779B97F4A7C15F39CC0605CEDC834
P = 1e-3
A_MAX = 0.02
STEPS = 3
# Hill states (m, m/s) from just outside the 50 m keep-out to 150 m, where the filters act.
NEAR = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))
# Test settings, not a model of a real sensor or thruster: a navigation box of a few tenths of a
# metre and a burn error of a few percent keep the braking barrier and its start set wide.
NAV = {"sigma_pos": 0.1, "sigma_vel": 1e-4}
BURN = {"scale_sigma": 0.01, "noise_sigma": 0.005}
DRAWS = tightening.episode_draws(GaussianNav(**NAV), ExecutionError(**BURN), steps=STEPS)
LEVEL = per_draw_level(P, DRAWS)


def truncated(level=LEVEL):
    return GaussianNav(bound=level, **NAV), ExecutionError(bound=level, **BURN)


def declared_braking(level=LEVEL, nav_scale=1.0, burn_scale=1.0, **kw):
    """The braking filter with the sets derived at `level`, each scaled by the factor given."""
    navigation, da_max = tightening.declared_errors(
        *truncated(level), a_max=A_MAX, layout="box", level=level
    )
    navigation = BoxError(r_pos=nav_scale * navigation.r_pos, r_vel=nav_scale * navigation.r_vel)
    declared = {
        "navigation": navigation,
        "execution": burn_scale * da_max,
        "exceedance": P,
        "per_draw_level": level,
        **kw,
    }
    return BackupFilter(barrier="braking", **declared)


def bounded_env(**models):
    sensor, execution = truncated()
    config = {"sensor": sensor, "execution": execution, "max_steps": STEPS, **models}
    return ApproachEnv(ApproachConfig(**config))


def bounded_audit(filt, env, **kw):
    return audit_filter(
        filt, env, **{"errors": "bounded", "starts": 2, "box": NEAR, "seed": SEED, **kw}
    )


def test_bounded_audit_flies_truncated_models_inside_the_declared_sets():
    filt = declared_braking()
    found = bounded_audit(filt, bounded_env())
    assert found.errors == "bounded"
    assert len(found.runs) == 2
    assert isinstance(found.passed, bool)
    assert (found.exceedance, found.per_draw_level, found.draws) == (P, LEVEL, DRAWS)
    # Every estimate of a start must lie in the declared set, so starts come from the start set.
    assert all(filt.contains_start(run.start) for run in found.runs)


REFUSED = {
    "untruncated navigation": lambda: (declared_braking(), bounded_env(sensor=GaussianNav(**NAV))),
    "untruncated execution": lambda: (
        declared_braking(),
        bounded_env(execution=ExecutionError(**BURN)),
    ),
    "range-scaled navigation": lambda: (
        declared_braking(),
        bounded_env(sensor=RangeScaledNav(bound=LEVEL)),
    ),
    "gauss-markov navigation": lambda: (declared_braking(), bounded_env(sensor=GaussMarkovNav())),
    "navigation set too small": lambda: (declared_braking(nav_scale=0.5), bounded_env()),
    "execution set too small": lambda: (declared_braking(burn_scale=0.5), bounded_env()),
    "no navigation set": lambda: (declared_braking(navigation=None), bounded_env()),
    # Sets derived at the higher level contain those at LEVEL, so only the level differs.
    "per-draw level differs": lambda: (declared_braking(level=1.1 * LEVEL), bounded_env()),
    "no declared level": lambda: (
        declared_braking(exceedance=None, per_draw_level=None),
        bounded_env(),
    ),
}


@pytest.mark.parametrize("case", REFUSED)
def test_bounded_audit_refuses_models_its_declared_sets_do_not_bound(case):
    # Each case changes one thing of the accepted setup above.
    filt, env = REFUSED[case]()
    with pytest.raises(ValueError):
        bounded_audit(filt, env)


def navigation_box():
    return tightening.declared_errors(*truncated(), a_max=A_MAX, layout="box", level=LEVEL)[0]


# One box corner with mixed signs, so neither the first corner nor an axis picks it out.
CORNER = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
TILT = 0.01


def tilted_ellipsoid(stretch):
    """An ellipsoid through the navigation box's corners, long toward +-CORNER, times `stretch`.

    With the corners scaled to sign vectors s, the ellipsoid is s' A s <= 1 for
    A = c (I + TILT q q'), q = CORNER / sqrt(6), c = 1 / (6 + 16 TILT / 6): every other corner
    has (s . CORNER)^2 <= 16 and lies inside, CORNER lies outside by the factor
    sqrt((6 + 6 TILT) / (6 + 16 TILT / 6)) = 1.0028, and the extent along each axis is 2.45
    half-widths.
    """
    box = navigation_box()
    half = np.repeat([box.r_pos, box.r_vel], 3)
    q = CORNER / np.sqrt(6.0)
    c = 1.0 / (6.0 + 16.0 * TILT / 6.0)
    root = (np.eye(6) + ((1.0 + TILT) ** -0.5 - 1.0) * np.outer(q, q)) / np.sqrt(c)
    return tightening.EllipsoidError(P=stretch * half[:, None] * root)


def ball(pos_scale, vel_scale):
    """Balls through the navigation box's corners, sqrt(3) half-widths, times the factors."""
    box = navigation_box()
    k = np.sqrt(3.0)
    return tightening.BallError(r_pos=k * pos_scale * box.r_pos, r_vel=k * vel_scale * box.r_vel)


HOLDING = {
    "tilted ellipsoid": lambda: tilted_ellipsoid(1.01),
    "balls": lambda: ball(1.001, 1.001),
}


@pytest.mark.parametrize("case", HOLDING)
def test_bounded_audit_accepts_a_declared_set_holding_the_navigation_box(case):
    found = bounded_audit(declared_braking(navigation=HOLDING[case]()), bounded_env())
    assert found.errors == "bounded"
    assert len(found.runs) == 2


MISSING = {
    "ellipsoid misses one corner": lambda: tilted_ellipsoid(1.0),
    "position ball misses the corners": lambda: ball(0.999, 1.001),
    "velocity ball misses the corners": lambda: ball(1.001, 0.999),
}


@pytest.mark.parametrize("case", MISSING)
def test_bounded_audit_refuses_a_declared_set_missing_a_navigation_box_corner(case):
    with pytest.raises(ValueError, match="does not hold"):
        bounded_audit(declared_braking(navigation=MISSING[case]()), bounded_env())


def test_bounded_audit_refuses_a_level_below_the_one_its_exceedance_needs():
    # Models truncated at the declared level and sets derived there fit; only b < b(p, D) fails.
    level = 0.9 * LEVEL
    sensor, execution = truncated(level)
    env = bounded_env(sensor=sensor, execution=execution)
    with pytest.raises(ValueError, match="per_draw_level .* is below"):
        bounded_audit(declared_braking(level=level), env)


@pytest.mark.parametrize("errors", ["bounded", "gaussian"])
def test_audit_under_errors_refuses_starts_outside_the_set(errors):
    with pytest.raises(ValueError, match="mode='inside'"):
        bounded_audit(declared_braking(), bounded_env(), errors=errors, mode="outside")


def test_bounded_audit_draws_only_starts_whose_every_estimate_is_in_the_set():
    filt = declared_braking()
    # At rest on V-bar from 50 to 50.8 m: the declared set holds every state, the start set,
    # 0.41 m of navigation error in, only those beyond about 50.6 m.
    lo, hi = (0.0, 50.0, 0.0, 0.0, 0.0, 0.0), (0.0, 50.8, 0.0, 0.0, 0.0, 0.0)
    found = bounded_audit(filt, bounded_env(), box=(lo, hi), starts=4, max_steps=1)
    assert filt.contains(np.array(lo)) and not filt.contains_start(np.array(lo))
    assert all(filt.contains_start(run.start) for run in found.runs)
    assert found.rejected > 0


def test_gaussian_audit_refuses_truncated_models():
    with pytest.raises(ValueError):
        audit_filter(declared_braking(), bounded_env(), errors="gaussian", starts=2, box=NEAR)


def test_audit_refuses_an_unknown_error_mode():
    with pytest.raises(ValueError):
        audit_filter(declared_braking(), bounded_env(), errors="truncated", starts=1, box=NEAR)


class Push:
    """Constant command of 0.9 a_max along -y, toward the target from a V-bar start."""

    name = "push"

    def reset(self, view, rng=None):
        self.scale = action_scale(view)

    def act(self, obs, info):
        return np.array([0.0, -0.9, 0.0]) * self.scale


class PassThrough(BackupFilter):
    """Declares the passive filter's sets and passes every command through."""

    def filter(self, obs, action):
        return FilterResult(np.array(action, dtype=float), intervened=False)


# Test setting, not a thruster model: a +50% scale bias turns 10 m of commanded travel in 30 s
# into 15 m delivered, so the delivered trajectory from 60 m crosses the 50 m keep-out and the
# commanded one does not.
PUSH = {"scale_bias": 0.5, "noise_sigma": 1e-3}
PUSH_LEVEL = per_draw_level(P, tightening.episode_draws(None, ExecutionError(**PUSH), steps=3))


def passing_through():
    _, da_max = tightening.declared_errors(
        None, ExecutionError(bound=PUSH_LEVEL, **PUSH), a_max=A_MAX, layout="box", level=PUSH_LEVEL
    )
    # A coarse drift grid keeps each start-set test to a few dozen samples.
    return PassThrough(
        barrier="passive", grid=100.0, t_cap=1000.0, execution=da_max, exceedance=P,
        per_draw_level=PUSH_LEVEL,
    )  # fmt: skip


def test_refly_uses_delivered_acceleration(monkeypatch):
    # One step from rest at 51.1 m: the commanded 0.9 a_max moves the chaser 0.9 m in the 10 s
    # step and the delivered one, 50% larger, 1.35 m, so only the delivered step enters the
    # keep-out. Commands in m/s^2, so a re-fly on the recorded command would use that command.
    x0 = (0.0, 51.1, 0.0, 0.0, 0.0, 0.0)
    env = ApproachEnv(
        ApproachConfig(
            execution=ExecutionError(bound=PUSH_LEVEL, **PUSH), max_steps=1, action_mode="accel"
        )
    )
    tau = np.linspace(0.0, env.cfg.dt, 1001)
    commanded = np.array([0.0, -0.9 * A_MAX, 0.0])
    path = cw_states(
        env.n, tau, np.broadcast_to(x0, (1001, 6)), np.broadcast_to(commanded, (1001, 3))
    )
    assert np.linalg.norm(path[:, :3], axis=1).min() > 50.1
    # The environment's refinement is made to miss every dip below the resolution, so only the
    # audit's re-fly of the step, on what the thrusters delivered, can find the violation.
    real = base_module.segment_minimum

    def missed(*args, **kwargs):
        found = real(*args, **kwargs)
        if found.margin >= -KeepOut.resolution:
            return found
        return SegmentMinimum(found.t, 0.0, min(found.lower_bound, -1.0))

    monkeypatch.setattr(base_module, "segment_minimum", missed)
    found = bounded_audit(passing_through(), env, nominal=Push(), box=(x0, x0))
    assert [run.steps for run in found.runs] == [1, 1]
    refinements = [r for run in found.runs for r in run.refinements]
    assert [r.step for r in refinements] == [0, 0]
    assert all(r.margin >= -KeepOut.resolution for r in refinements)
    assert all(r.outcome == "violation" for r in refinements)
    # The delivered step reaches about 49.75 m.
    assert all(r.refined_margin < -0.2 for r in refinements)
    assert found.violations == 2
    assert not found.passed


# (certificate horizon s or None, thrust) per decision, one every 10 s.
SCRIPT = (
    (35.0, False),
    (None, False),  # t 10 s, coasting to 20 s inside the horizon to 35 s: covered
    (None, False),  # t 20 s, to 30 s: covered
    (1000.0, False),
    (None, True),  # t 40 s: thrust without a certificate, uncovered
    (5.0, False),  # the last certificate now ends at 55 s, the earlier one no longer counts
    (None, False),  # t 60 s: uncovered
    (None, False),  # t 70 s: uncovered
)


class Scripted(BackupFilter):
    """Returns the decisions of `script` in turn, certified with the horizon given or not."""

    script = SCRIPT

    def reset(self, view):
        super().reset(view)
        self.k = 0

    def filter(self, obs, action):
        horizon, thrust = self.script[self.k]
        self.k += 1
        u = np.array([0.0, 0.01, 0.0]) if thrust else np.zeros(3)
        if horizon is None:
            return FilterResult(u, intervened=True)
        return FilterResult(
            u, intervened=True, certificate="bounded-W", certificate_horizon=horizon
        )


def scripted_audit(script):
    burn = {"noise_sigma": 1e-3}
    steps = len(script)
    level = per_draw_level(P, tightening.episode_draws(None, ExecutionError(**burn), steps=steps))
    _, da_max = tightening.declared_errors(
        None, ExecutionError(bound=level, **burn), a_max=A_MAX, layout="box", level=level
    )
    filt = Scripted(
        barrier="passive", grid=100.0, t_cap=1000.0, execution=da_max, exceedance=P,
        per_draw_level=level,
    )  # fmt: skip
    filt.script = script
    env = ApproachEnv(
        ApproachConfig(execution=ExecutionError(bound=level, **burn), max_steps=steps)
    )
    # At rest on V-bar 300 m out, far from the keep-out whatever the script commands.
    x0 = (0.0, 300.0, 0.0, 0.0, 0.0, 0.0)
    return audit_filter(filt, env, errors="bounded", starts=2, box=(x0, x0), seed=SEED)


def test_zero_thrust_inside_a_certificate_is_covered():
    found = scripted_audit(SCRIPT)
    assert found.violations == found.uncertified == 0
    assert [run.uncovered for run in found.runs] == [3, 3]
    assert found.uncovered == 6
    assert not found.passed
    coasting = scripted_audit(SCRIPT[:3])
    assert coasting.uncovered == 0
    assert coasting.passed


# A certificate at 0 s, then coasts decided at 10 s (step to 20 s) and 20 s (step to 30 s).
HORIZON_END = {
    # The horizon ends mid-step: the coast from 20 s runs past 25 s although it starts before.
    "mid-step": ((25.0, False), (None, False), (None, False)),
    # The horizon ends exactly at the end of the first coast, which it still covers.
    "at a step end": ((20.0, False), (None, False), (None, False)),
}


@pytest.mark.parametrize("case", HORIZON_END)
def test_coast_is_covered_only_when_its_step_ends_within_the_horizon_from_the_certificate(case):
    # Counting the horizon from the certified step's end (30 or 35 s), or comparing a coast's
    # decision time instead of its step end, would cover both coasts.
    found = scripted_audit(HORIZON_END[case])
    assert [run.uncovered for run in found.runs] == [1, 1]


def test_gaussian_line_reports_cp_against_level():
    env = ApproachEnv(ApproachConfig(execution=ExecutionError(**PUSH), max_steps=3))
    # At rest on V-bar from 56 to 68 m: the delivered push crosses the keep-out from below about
    # 62 m, so some runs fail and some do not.
    box = ((0.0, 56.0, 0.0, 0.0, 0.0, 0.0), (0.0, 68.0, 0.0, 0.0, 0.0, 0.0))
    trials = 20
    found = audit_filter(
        passing_through(), env, nominal=Push(), errors="gaussian", starts=trials, box=box,
        seed=SEED,
    )  # fmt: skip
    assert found.errors == "gaussian"
    # A rate is reported against the declared level, never passed or failed.
    assert found.passed is None
    assert (found.exceedance, found.per_draw_level) == (P, PUSH_LEVEL)
    assert found.draws == 9
    failing = sum(run.violations > 0 for run in found.runs)
    assert 0 < failing < trials
    lower = beta.ppf(0.025, failing, trials - failing + 1)
    upper = beta.ppf(0.975, failing + 1, trials - failing)
    rates = found.exceedance_rates
    assert set(rates) == {"keep_out", "any"}
    for name in rates:
        assert (rates[name].count, rates[name].trials) == (failing, trials)
        assert rates[name].rate == failing / trials
        assert rates[name].interval == pytest.approx((lower, upper), rel=1e-12)
    assert rates["keep_out"].level == P
    # The union bound over the one covered constraint.
    assert rates["any"].level == min(1.0, 1 * P)


class SpeedLimitedApproach(ApproachEnv):
    """The approach with a speed limit after the keep-out, v1 = 4n as in Dunlap et al. (2021)
    eq. 14; v0 = 0.3 m/s is a test setting that the push from rest breaks only from the nearer
    starts."""

    def _constraints(self):
        return (*super()._constraints(), SpeedLimit(0.3, 4.0 * self.n))


class PassingCbf(CbfFilter):
    """Declares the CBF filter's sets, keep-out and speed limit, and passes every command."""

    def filter(self, obs, action):
        return FilterResult(np.array(action, dtype=float), intervened=False)


def test_gaussian_any_rate_counts_either_constraint_against_the_union_bound():
    burn = {"noise_sigma": 1e-3}
    level = per_draw_level(P, tightening.episode_draws(None, ExecutionError(**burn), steps=3))
    _, da_max = tightening.declared_errors(
        None, ExecutionError(bound=level, **burn), a_max=A_MAX, layout="box", level=level
    )
    env = SpeedLimitedApproach(ApproachConfig(execution=ExecutionError(**burn), max_steps=3))
    # At rest on V-bar from 56 to 68 m: 30 s of push enter the keep-out from below about 58 m
    # and break the speed limit from below about 62 m.
    box = ((0.0, 56.0, 0.0, 0.0, 0.0, 0.0), (0.0, 68.0, 0.0, 0.0, 0.0, 0.0))
    filt = PassingCbf(execution=da_max, exceedance=P, per_draw_level=level)
    found = audit_filter(
        filt, env, nominal=Push(), errors="gaussian", starts=20, box=box, seed=SEED
    )
    assert found.covered == ("keep_out", "speed_limit")
    keep_out = sum(run.violated["keep_out"] > 0 for run in found.runs)
    speed = sum(run.violated["speed_limit"] > 0 for run in found.runs)
    either = sum(run.violated["keep_out"] + run.violated["speed_limit"] > 0 for run in found.runs)
    # Some runs break only the speed limit, so "any" is more than the keep-out's count.
    assert 0 < keep_out < either < 20
    rates = found.exceedance_rates
    assert [rates[name].count for name in ("keep_out", "speed_limit", "any")] == [
        keep_out,
        speed,
        either,
    ]
    assert rates["keep_out"].level == rates["speed_limit"].level == P
    assert rates["any"].level == 2 * P
    # Above p = 1/2 the sum of the two levels passes 1, where the union bound stops.
    saturated = dataclasses.replace(found, exceedance=0.6).exceedance_rates
    assert saturated["speed_limit"].level == 0.6
    assert saturated["any"].level == 1.0


DECLARING = {
    "passive": lambda **kw: BackupFilter(barrier="passive", **kw),
    "braking": lambda **kw: BackupFilter(barrier="braking", **kw),
    "ecbf": lambda **kw: CbfFilter(poles=(0.06, 0.06), gamma=0.02, v_cap=0.1, **kw),
    "predictive": lambda **kw: PredictiveFilter(**kw),
}


@pytest.mark.parametrize(
    "kw",
    [
        {"execution": 1e-4, "exceedance": P},
        {"execution": 1e-4, "per_draw_level": 4.0},
        {"exceedance": P, "per_draw_level": 4.0},
    ],
    ids=["exceedance alone", "level alone", "no error set"],
)
@pytest.mark.parametrize("name", DECLARING)
def test_exceedance_and_level_come_together_and_only_with_an_error_set(name, kw):
    with pytest.raises(ValueError):
        DECLARING[name](**kw)


def test_declaration_takes_exceedance_and_level_together_and_only_with_an_error_set():
    filt = BackupFilter(barrier="passive", execution=1e-4, exceedance=P, per_draw_level=4.0)
    filt.reset(ApproachEnv().agent_view())
    declared = filt.declaration
    with pytest.raises(ValueError):
        dataclasses.replace(declared, per_draw_level=None)
    with pytest.raises(ValueError):
        dataclasses.replace(declared, exceedance=None)
    bare = dict(declared.tightening, execution=0.0)
    with pytest.raises(ValueError):
        dataclasses.replace(declared, tightening=bare)
