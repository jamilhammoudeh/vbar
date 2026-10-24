"""Certificate vocabulary: the kinds of guarantee a filter decision or declaration may claim.

The interface these tests assume:

- `vbar.safety.CERTIFICATE_KINDS` is the tuple ("nominal", "bounded-W", "chance-epsilon",
  "finite-horizon") and is listed in `vbar.safety.__all__`.
- `FilterResult` raises ValueError when `certificate` is neither None nor one of the kinds,
  when exactly one of `certificate` and `certificate_horizon` is None, or when `certificate` is
  "finite-horizon" and `certificate_horizon` is inf. Any other kind takes a finite or infinite
  positive horizon.
- `FilterDeclaration` raises ValueError when `certificate` is neither None nor one of the kinds.
- `run_filter` rebuilds the filter's result, so a result whose certificate was set after its
  construction is checked again and raises ValueError.
- A `BackupFilter` with a navigation `EllipsoidError` of exceedance above 0 declares
  "chance-epsilon" and certifies its decisions as "chance-epsilon", never "bounded-W".
"""

import dataclasses
import itertools
import math

import numpy as np
import pytest

import vbar.safety as safety
from vbar.agents.base import action_scale
from vbar.dynamics.cw import cw_system
from vbar.safety import FilteredAgent, FilterResult, run_filter
from vbar.safety.cbf import BackupFilter, CbfFilter, SoftCbfFilter
from vbar.safety.corridor import CorridorCamFilter
from vbar.safety.predictive import PredictiveFilter
from vbar.safety.rta import (
    AfrlSetup,
    ExplicitOptimizationFilter,
    ExplicitSwitchingFilter,
    ImplicitOptimizationFilter,
    ImplicitSwitchingFilter,
    start_state,
)
from vbar.safety.tightening import EllipsoidError
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.spacecraft import PerfectNav

KINDS = ("nominal", "bounded-W", "chance-epsilon", "finite-horizon")
UNKNOWN = ("explicit", "", "Nominal", "bounded_W", "finite horizon", "chance")
A_MAX = ApproachConfig().max_accel
CODE = AfrlSetup.authors_code()
AFRL_FILTERS = (
    ExplicitSwitchingFilter,
    ExplicitOptimizationFilter,
    ImplicitSwitchingFilter,
    ImplicitOptimizationFilter,
)
# Closing on the target along V-bar at 0.2 m/s from 120 m, where the filters act.
CLOSING = (0.0, 120.0, 0.0, 0.0, -0.2, 0.0)
# At rest 400 m out on V-bar, inside every tightened set these tests declare.
REST = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)
# 0.5 m in position and 0.1 mm/s in velocity: small enough that the drift from REST certifies.
NAV_P = np.diag([0.5, 0.5, 0.5, 1e-4, 1e-4, 1e-4])
# About a third of an orbit of drift instead of the default two keeps the tightened barrier small.
T_CAP = 2000.0


def allowed(certificate, horizon):
    """The rules of the module docstring, written out."""
    if certificate is None or horizon is None:
        return certificate is None and horizon is None
    if certificate not in KINDS:
        return False
    return not (certificate == "finite-horizon" and math.isinf(horizon))


def result(certificate, horizon):
    return FilterResult(
        np.zeros(3), intervened=False, certificate=certificate, certificate_horizon=horizon
    )


class HoldPD:
    """PD toward the hold point, clipped to the box."""

    name = "hold-pd"

    def reset(self, view, rng=None):
        self.hold = np.array(view.goal["hold_point"])
        self.scale = action_scale(view) / view.max_accel
        self.a_max = view.max_accel

    def act(self, obs, info):
        u = np.clip(-2e-4 * (obs[:3] - self.hold) - 0.03 * obs[3:6], -self.a_max, self.a_max)
        return u * self.scale


class ExplicitClaim:
    """Claims an "explicit" certificate, a kind outside the vocabulary."""

    name = "explicit-claim"

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(
            np.array(action, dtype=float),
            intervened=False,
            certificate="explicit",
            certificate_horizon=math.inf,
        )


class ForgedClaim:
    """Builds an honest result, then writes an "explicit" certificate past its checks."""

    name = "forged-claim"

    def reset(self, view):
        pass

    def filter(self, obs, action):
        out = FilterResult(
            np.array(action, dtype=float),
            intervened=False,
            certificate="nominal",
            certificate_horizon=math.inf,
        )
        object.__setattr__(out, "certificate", "explicit")
        return out


def approach(start, steps):
    return ApproachEnv(
        ApproachConfig(dt=10.0, max_steps=steps, start_state=start, sensor=PerfectNav())
    )


def fly(filt, start, steps):
    """The filter's decisions around the hold-point PD over `steps` steps of 10 s."""
    env = approach(start, steps)
    pilot = FilteredAgent(HoldPD(), filt)
    obs, _ = env.reset(seed=0)
    pilot.reset(env.agent_view(), np.random.default_rng(0))
    results = []
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(pilot.act(obs, env.agent_info()))
        results.append(pilot.last_result)
        if terminated or truncated:
            break
    return results


def fly_afrl(cls, steps):
    """The filter's decisions on zero commands along the Euler plant from the seed-0 start."""
    filt = cls(setup=CODE)
    filt.reset(CODE.view())
    a, b = cw_system(CODE.n)
    x = start_state(CODE, 0)
    results = []
    for _ in range(steps):
        out, _ = run_filter(filt, x, np.zeros(3))
        u = np.clip(np.asarray(out.action, dtype=float), -CODE.a_max, CODE.a_max)
        x = x + CODE.dt * (a @ x + b @ u)
        results.append(out)
    return filt, results


def assert_valid(declaration, results):
    claims = (*safety.CERTIFICATE_KINDS, None)
    assert declaration.certificate in claims
    for out in results:
        assert out.certificate in claims
        assert allowed(out.certificate, out.certificate_horizon)


def test_vocabulary_is_the_four_kinds_and_is_exported():
    assert safety.CERTIFICATE_KINDS == KINDS
    assert "CERTIFICATE_KINDS" in safety.__all__


@pytest.mark.parametrize("kind", KINDS[:3])
@pytest.mark.parametrize("horizon", (math.inf, 5.0))
def test_result_accepts_every_kind_but_finite_horizon_with_any_horizon(kind, horizon):
    out = result(kind, horizon)
    assert out.certificate == kind
    assert out.certificate_horizon == horizon


def test_result_accepts_finite_horizon_with_a_finite_horizon():
    assert result("finite-horizon", 5.0).certificate == "finite-horizon"


def test_result_accepts_no_certificate_and_no_horizon():
    out = result(None, None)
    assert out.certificate is None
    assert out.certificate_horizon is None


@pytest.mark.parametrize("kind", UNKNOWN)
def test_result_refuses_a_kind_outside_the_vocabulary(kind):
    with pytest.raises(ValueError):
        result(kind, math.inf)


@pytest.mark.parametrize("kind", KINDS)
def test_result_refuses_a_kind_without_a_horizon(kind):
    with pytest.raises(ValueError):
        result(kind, None)


@pytest.mark.parametrize("horizon", (math.inf, 5.0))
def test_result_refuses_a_horizon_without_a_kind(horizon):
    with pytest.raises(ValueError):
        result(None, horizon)


def test_result_refuses_finite_horizon_with_an_infinite_horizon():
    with pytest.raises(ValueError):
        result("finite-horizon", math.inf)


@pytest.mark.parametrize(
    ("kind", "horizon"),
    list(itertools.product((*KINDS, *UNKNOWN[:2], None), (None, 5.0, math.inf))),
)
def test_result_accepts_exactly_the_pairs_the_rules_allow(kind, horizon):
    if allowed(kind, horizon):
        assert result(kind, horizon).certificate == kind
    else:
        with pytest.raises(ValueError):
            result(kind, horizon)


def passive_declaration():
    filt = BackupFilter(barrier="passive")
    filt.reset(approach(REST, 1).agent_view())
    return filt.declaration


@pytest.mark.parametrize("kind", (*KINDS, None))
def test_declaration_accepts_every_kind_and_none(kind):
    assert dataclasses.replace(passive_declaration(), certificate=kind).certificate == kind


@pytest.mark.parametrize("kind", UNKNOWN)
def test_declaration_refuses_a_kind_outside_the_vocabulary(kind):
    declaration = passive_declaration()
    with pytest.raises(ValueError):
        dataclasses.replace(declaration, certificate=kind)


def test_run_filter_refuses_a_filter_claiming_an_explicit_certificate():
    with pytest.raises(ValueError):
        run_filter(ExplicitClaim(), np.array(REST), np.zeros(3))


def test_run_filter_refuses_a_certificate_written_after_construction():
    with pytest.raises(ValueError):
        run_filter(ForgedClaim(), np.array(REST), np.zeros(3))


def test_filtered_agent_refuses_a_filter_claiming_an_explicit_certificate():
    env = approach(REST, 1)
    pilot = FilteredAgent(HoldPD(), ForgedClaim())
    obs, _ = env.reset(seed=0)
    pilot.reset(env.agent_view(), np.random.default_rng(0))
    with pytest.raises(ValueError):
        pilot.act(obs, env.agent_info())


HELD_FILTERS = {
    "cbf": (lambda: CbfFilter(poles=(0.05, 0.05), v_cap=0.1, gamma=0.02), CLOSING, False),
    "soft-cbf": (
        lambda: SoftCbfFilter(poles=(0.05, 0.05), alpha=1.0, v_cap=0.1, gamma=0.02),
        CLOSING,
        False,
    ),
    "passive": (lambda: BackupFilter(barrier="passive"), CLOSING, True),
    "passive-bounded": (
        lambda: BackupFilter(barrier="passive", execution=1e-5, t_cap=T_CAP),
        REST,
        True,
    ),
    "braking": (
        lambda: BackupFilter(barrier="braking", a_b=0.01044, rho_cap=2000.0, v_cap=1.0),
        CLOSING,
        True,
    ),
    "cam": (lambda: BackupFilter(barrier="cam"), CLOSING, True),
    "predictive": (lambda: PredictiveFilter(horizon=5), CLOSING, True),
    "corridor-cam": (
        lambda: CorridorCamFilter(phases=((0.0, math.inf, (0.0, A_MAX, 0.0), 2),)),
        CLOSING,
        False,
    ),
}


@pytest.mark.parametrize("name", HELD_FILTERS)
def test_every_approach_filter_claims_only_valid_certificates(name):
    make, start, certifies = HELD_FILTERS[name]
    filt = make()
    results = fly(filt, start, 3)
    assert_valid(filt.declaration, results)
    # A filter that certifies here must have done so, or the check above saw no certificate.
    assert any(out.certificate is not None for out in results) == certifies


@pytest.mark.parametrize("cls", AFRL_FILTERS)
def test_every_afrl_filter_claims_only_valid_certificates(cls):
    filt, results = fly_afrl(cls, 3)
    assert_valid(filt.declaration, results)


@pytest.mark.parametrize("cls", (ImplicitSwitchingFilter, ImplicitOptimizationFilter))
def test_implicit_afrl_filters_claim_finite_horizon_over_the_setup_horizon(cls):
    _, results = fly_afrl(cls, 3)
    certified = [out for out in results if out.certificate is not None]
    assert certified
    assert all(out.certificate == "finite-horizon" for out in certified)
    assert all(out.certificate_horizon == CODE.horizon_s for out in certified)


def ellipsoid_filter(exceedance):
    navigation = EllipsoidError(P=NAV_P, exceedance=exceedance)
    return BackupFilter(barrier="passive", navigation=navigation, t_cap=T_CAP)


def test_backup_filter_with_a_hard_ellipsoid_declares_and_certifies_bounded_w():
    filt = ellipsoid_filter(0.0)
    results = fly(filt, REST, 2)
    assert filt.declaration.certificate == "bounded-W"
    assert [out.certificate for out in results] == ["bounded-W", "bounded-W"]


def test_backup_filter_with_an_exceedance_declares_and_certifies_chance_epsilon():
    filt = ellipsoid_filter(0.01)
    results = fly(filt, REST, 2)
    assert filt.declaration.certificate == "chance-epsilon"
    certified = [out for out in results if out.certificate is not None]
    assert certified
    assert all(out.certificate == "chance-epsilon" for out in certified)
    assert_valid(filt.declaration, results)
