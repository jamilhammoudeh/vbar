import dataclasses
import functools
import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from vbar.agents.base import action_scale
from vbar.constraints import KeepOut, segment_minimum
from vbar.dynamics import cw_discrete
from vbar.dynamics.models import CWDynamics
from vbar.safety import FilteredAgent, FilterResult
from vbar.safety import cbf as cbf_module
from vbar.safety.barriers import (
    BrakingBarrier,
    KeepOutEcbf,
    PassiveBarrier,
    SpeedCbf,
    braking_deceleration,
)
from vbar.safety.cbf import BackupFilter, CbfFilter
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.solvers.qp import ActuatorSet, Rows, project
from vbar.spacecraft import PerfectNav

R = 50.0
A_MAX = 0.02
N = ApproachEnv().n
# From 1 s to twice the scenario step.
DTS = (1.0, 5.0, 10.0, 20.0)
# A violation is a certified in-step bound deeper than the keep-out's resolution.
DEPTH = KeepOut.resolution
FAR = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])


def cw_accel(x):
    return np.array([3 * N * N * x[0] + 2 * N * x[4], -2 * N * x[3], -N * N * x[2]])


def approach(dt=10.0, steps=1, start=FAR, **cfg):
    start = tuple(float(s) for s in start)
    return ApproachEnv(ApproachConfig(dt=dt, max_steps=steps, start_state=start, **cfg))


def view(dt=10.0, **cfg):
    return approach(dt, **cfg).agent_view()


def ready(filt, dt=10.0, **cfg):
    filt.reset(view(dt, **cfg))
    return filt


def cbf(p=0.05, v_cap=0.1, gamma=0.02, **kw):
    return CbfFilter(poles=(p, p), v_cap=v_cap, gamma=gamma, **kw)


def braking(a_b=0.01044, rho_cap=2000.0, v_cap=1.0, **kw):
    return BackupFilter(barrier="braking", a_b=a_b, rho_cap=rho_cap, v_cap=v_cap, **kw)


def fields(result):
    """Every FilterResult field but the wall time, comparable with ==."""
    return (
        tuple(np.asarray(result.action, dtype=float).tolist()),
        result.intervened,
        result.status,
        result.slack,
        result.conflicting,
        result.active,
        result.recovering,
        result.fallback_level,
        result.certificate,
        result.certificate_horizon,
        result.iterations,
    )


class CentrePD:
    """PD toward the target centre, so the target is always in the way."""

    name = "centre-pd"

    def reset(self, view, rng=None):
        self.scale = action_scale(view) / view.max_accel

    def act(self, obs, info):
        return (-4e-4 * obs[:3] - 0.04 * obs[3:6]) * self.scale


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


class Reckless:
    """Pushes at 1 m/s^2 toward the target, far beyond any thrust limit."""

    name = "reckless"

    def reset(self, view, rng=None):
        self.scale = action_scale(view) / view.max_accel

    def act(self, obs, info):
        return -obs[:3] / np.linalg.norm(obs[:3]) * self.scale


class ScriptedSensor:
    """Returns a fixed sequence of observations whatever the true state."""

    name = "scripted"

    def __init__(self, states):
        self.states = [np.array(s, dtype=float) for s in states]

    def reset(self, rng):
        self.k = 0

    def observe(self, true_state, t):
        x = self.states[min(self.k, len(self.states) - 1)]
        self.k += 1
        return x.copy()


def push_inward(env, da):
    """Adds a constant acceleration of da m/s^2 toward the target to every segment of a step."""
    apply = env.spacecraft.apply

    def pushed(command, duration, latency=0.0):
        actuation = apply(command, duration, latency)
        r = env.state[:3]
        d = -da * r / np.linalg.norm(r)
        segments = tuple((t, np.asarray(a) + d) for t, a in actuation.segments)
        return dataclasses.replace(actuation, segments=segments)

    env.spacecraft.apply = pushed


@dataclasses.dataclass(frozen=True)
class Flight:
    worst: float
    max_speed: float
    captured: bool
    seen: tuple
    commands: tuple
    results: tuple


def assert_in_actuator_set(action, layout):
    """The returned command lies in the thrust limit exactly, in fractions of a full command."""
    action = np.asarray(action, dtype=float)
    if layout == "box":
        assert np.all(np.abs(action) <= 1.0)
    else:
        assert np.linalg.norm(action) <= 1.0


def fly(filt, agent, start, dt, steps, *, push=0.0, sensor=None, **cfg):
    env = approach(dt, steps, start, sensor=sensor or PerfectNav(), **cfg)
    if push:
        push_inward(env, push)
    pilot = FilteredAgent(agent, filt)
    obs, _ = env.reset(seed=0)
    pilot.reset(env.agent_view(), np.random.default_rng(0))
    worst, max_speed = math.inf, 0.0
    seen, commands, results = [], [], []
    for _ in range(steps):
        seen.append(obs.copy())
        obs, _, terminated, truncated, _ = env.step(pilot.act(obs, env.agent_info()))
        segment = env.record.segment
        worst = min(worst, segment.minima["keep_out"].lower_bound)
        max_speed = max(max_speed, float(np.linalg.norm(segment.states[:, 3:], axis=1).max()))
        commands.append(pilot.last_command)
        results.append(pilot.last_result)
        assert_in_actuator_set(pilot.last_result.action, cfg.get("thrust_limit", "box"))
        if terminated or truncated:
            break
    return Flight(worst, max_speed, env.captured, tuple(seen), tuple(commands), tuple(results))


def in_declared_set(x, declaration, v_cap):
    """Membership of the declared safe set: the ECBF start set, the speed cap and an
    acceleration region where the box can meet every row with 10% speed overshoot."""
    p_max = max(declaration.poles)
    keep_out = KeepOutEcbf(radius=R + declaration.buffer_m, poles=declaration.poles)
    return (
        keep_out.contains(x)
        and np.linalg.norm(x[3:]) <= v_cap
        and np.linalg.norm(cw_accel(x)) + 2 * p_max * 1.1 * v_cap <= declaration.a_max
    )


def ecbf_draw(rng, p):
    """One start from shells up to 150 m outside the keep-out, kept if rho_dot + p h >= 0."""
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    h0 = rng.uniform(0.0, 150.0)
    v = rng.uniform(-0.3, 0.3, size=3)
    return np.concatenate([(R + h0) * d, v]) if d @ v + p * h0 >= 0.0 else None


@functools.cache
def cbf_starts(p, dt, v_cap=0.1, count=4):
    declaration = ready(cbf(p, v_cap), dt).declaration
    rng = np.random.default_rng(1903)
    starts = []
    while len(starts) < count:
        x = ecbf_draw(rng, p)
        if x is not None and in_declared_set(x, declaration, v_cap):
            starts.append(x)
    return tuple(starts)


def sample_region(rng, count):
    d = rng.normal(size=(count, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    h0 = rng.uniform(0.0, 150.0, size=count)
    v = rng.uniform(-0.3, 0.3, size=(count, 3))
    return np.hstack([(R + h0)[:, None] * d, v])


@functools.cache
def braking_region():
    rng = np.random.default_rng(1903)
    sample_region(rng, 200000)
    return sample_region(rng, 5000)


def braking_starts(barrier, count=4):
    return tuple(x for x in braking_region() if barrier.contains(x))[:count]


@functools.cache
def passive_filter():
    return ready(BackupFilter(barrier="passive"))


@functools.cache
def passive_starts(count=4):
    env = ApproachEnv()
    starts, seed = [], 0
    while len(starts) < count:
        _, info = env.reset(seed=seed)
        if passive_filter().contains(info["true_state"]):
            starts.append(info["true_state"])
        seed += 1
    return tuple(starts)


# Construction and reset refusals.


def test_filters_take_keyword_arguments_only():
    with pytest.raises(TypeError):
        CbfFilter((0.05, 0.05))
    with pytest.raises(TypeError):
        BackupFilter("passive")


def test_names_are_fixed_at_construction():
    assert cbf(0.05).name == "cbf-ecbf-p0.05"
    assert BackupFilter(barrier="passive").name == "cbf-passive"
    assert braking().name == "cbf-brake"


def test_pole_rule_refuses_p_dt_above_0_6_and_names_the_largest_pole():
    # 0.6 lies below the double-integrator hold threshold (sqrt 5 - 1) / 2 of the ECBF.
    with pytest.raises(ValueError, match=r"0\.06"):
        ready(cbf(0.1))
    with pytest.raises(ValueError, match=r"0\.06"):
        ready(cbf(0.0601))
    ready(cbf(0.0599))
    ready(cbf(0.05))
    with pytest.raises(ValueError, match=r"0\.03"):
        ready(cbf(0.05), dt=20.0)


def test_pole_rule_uses_the_faster_of_two_unequal_poles():
    with pytest.raises(ValueError, match=r"0\.06"):
        ready(CbfFilter(poles=(0.02, 0.07), v_cap=0.1, gamma=0.02))
    ready(CbfFilter(poles=(0.02, 0.059), v_cap=0.1, gamma=0.02))


def test_speed_gain_rule_refuses_gamma_dt_above_0_6():
    with pytest.raises(ValueError, match=r"0\.06"):
        ready(cbf(gamma=0.0601))
    ready(cbf(gamma=0.0599))


FILTERS = {
    "ecbf": lambda **kw: cbf(**kw),
    "passive": lambda **kw: BackupFilter(barrier="passive", **kw),
    "brake": lambda **kw: braking(**kw),
}


@pytest.mark.parametrize("kind", FILTERS)
@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_reset_refuses_discrete_and_impulse_action_modes(kind, mode):
    with pytest.raises(ValueError, match="continuous|mode"):
        ready(FILTERS[kind](), action_mode=mode)


@pytest.mark.parametrize("kind", FILTERS)
@pytest.mark.parametrize("mode", ["normalized", "accel", "thrust"])
def test_reset_accepts_every_continuous_action_mode(kind, mode):
    ready(FILTERS[kind](), action_mode=mode)


@pytest.mark.parametrize("kind", FILTERS)
def test_reset_refuses_a_frame_other_than_hill(kind):
    with pytest.raises(ValueError, match="frame|hill"):
        FILTERS[kind]().reset(dataclasses.replace(view(), frame="eci"))


@pytest.mark.parametrize("kind", FILTERS)
def test_reset_refuses_observations_not_led_by_the_hill_state(kind):
    names = ("y", "x", "z", "vx", "vy", "vz")
    with pytest.raises(ValueError, match="observ"):
        FILTERS[kind]().reset(dataclasses.replace(view(), observation=names))


@pytest.mark.parametrize("kind", FILTERS)
def test_reset_refuses_a_pending_commands_channel(kind):
    with pytest.raises(ValueError, match="pending"):
        ready(FILTERS[kind](), delay_s=10.0, observation_channels=("pending_commands",))
    # Channels after the six state entries are allowed.
    ready(FILTERS[kind](), observation_channels=("t_remaining",))


@pytest.mark.parametrize("kind", FILTERS)
def test_reset_refuses_non_cw_dynamics_without_a_declared_model_error(kind):
    with pytest.raises(ValueError, match="model_error|dynamics"):
        ready(FILTERS[kind](), dynamics="twobody")
    ready(FILTERS[kind](model_error=1e-6), dynamics="twobody")


@pytest.mark.parametrize("kind", FILTERS)
def test_reset_refuses_an_unenforced_constraint_unless_ignored(kind):
    sun = {"type": "sun_exclusion", "half_angle": 0.3, "sun_direction": (1.0, 0.0, 0.0), "rate": N}
    widened = dataclasses.replace(view(), constraints=(*view().constraints, sun))
    with pytest.raises(ValueError, match="sun_exclusion"):
        FILTERS[kind]().reset(widened)
    filt = FILTERS[kind](ignore=("sun_exclusion",))
    filt.reset(widened)
    assert filt.declaration.ignored == ("sun_exclusion",)


def test_backup_filters_refuse_a_speed_limit_the_cbf_filter_enforces():
    limit = {"type": "speed_limit", "v0": 0.5, "v1": 0.0}
    limited = dataclasses.replace(view(), constraints=(*view().constraints, limit))
    cbf(v_cap=0.1).reset(limited)
    for make in (FILTERS["passive"], FILTERS["brake"]):
        with pytest.raises(ValueError, match="speed_limit"):
            make().reset(limited)
        make(ignore=("speed_limit",)).reset(limited)


@pytest.mark.parametrize("kind", FILTERS)
def test_reset_refuses_a_view_without_keep_out(kind):
    with pytest.raises(ValueError, match="keep_out"):
        FILTERS[kind]().reset(dataclasses.replace(view(), constraints=()))


def test_braking_filter_refuses_a_default_deceleration_that_is_not_positive():
    args = dict(a_max=A_MAX, layout="box", v_cap=1.0, radius=R, rho_cap=2000.0, n=N)
    assert braking_deceleration(dt=20.0, **args) <= 0.0 < braking_deceleration(dt=10.0, **args)
    filt = BackupFilter(barrier="braking", rho_cap=2000.0, v_cap=1.0)
    with pytest.raises(ValueError, match="a_b"):
        ready(filt, dt=20.0)
    ready(BackupFilter(barrier="braking", rho_cap=2000.0, v_cap=1.0), dt=10.0)


def test_cbf_declaration_lists_its_design():
    filt = ready(cbf(0.05, v_cap=0.1, gamma=0.02))
    d = filt.declaration
    assert d.name == filt.name == "cbf-ecbf-p0.05"
    assert d.poles == (0.05, 0.05) and d.gamma == 0.02
    assert d.design_dt == 10.0 and d.layout == "box" and d.a_max == A_MAX
    # Boundary riding misses the change of the CW acceleration over a held step by at most
    # 3 n^2 v dt / p^2, with a 20% margin.
    assert d.buffer_m == pytest.approx(1.2 * 3 * N * N * 0.1 * 10.0 / 0.05**2, rel=1e-9)
    assert d.certificate is None
    assert isinstance(d.counts_toward_certification, bool)
    assert d.safe_set and d.barriers
    assert d.ignored == ()


def test_cbf_declaration_adds_the_execution_bias_to_the_buffer():
    # A persistent error da on h'' = -p^2 h - 2 p h' settles at h = -da / p^2.
    d = ready(cbf(0.05, execution=2e-4)).declaration
    riding = 1.2 * 3 * N * N * 0.1 * 10.0 / 0.05**2
    assert d.buffer_m == pytest.approx(riding + 2e-4 / 0.05**2, rel=1e-9)


@pytest.mark.parametrize("barrier, name", [("passive", "cbf-passive"), ("braking", "cbf-brake")])
def test_backup_declaration_lists_a_nominal_certificate(barrier, name):
    filt = ready(braking() if barrier == "braking" else BackupFilter(barrier="passive"))
    d = filt.declaration
    assert d.name == name
    assert d.certificate == "nominal"
    assert d.counts_toward_certification is True
    assert d.design_dt == 10.0 and d.layout == "box" and d.a_max == A_MAX
    assert d.backup and d.safe_set and d.barriers


def test_backup_declaration_with_an_execution_error_is_bounded_w():
    assert ready(braking(execution=2e-4)).declaration.certificate == "bounded-W"


# Clipping onto the actuator set comes before the filter and is never an intervention.


CLIP_CASES = [
    ("normalized", "box", (0.0, 0.0, 3.0), (0.0, 0.0, 1.0)),
    ("accel", "box", (0.0, 0.0, 0.06), (0.0, 0.0, 0.02)),
    ("thrust", "box", (0.0, 0.0, 30.0), (0.0, 0.0, 10.0)),
    ("normalized", "ball", (0.0, 0.0, 3.0), (0.0, 0.0, 1.0)),
]


@pytest.mark.parametrize("kind", FILTERS)
@pytest.mark.parametrize("mode, layout, command, clipped", CLIP_CASES)
def test_a_clipped_command_is_not_an_intervention(kind, mode, layout, command, clipped):
    # A 2 s step at the full command ends at 0.04 m/s, inside the 0.1 m/s speed cap.
    filt = ready(FILTERS[kind](), dt=2.0, action_mode=mode, thrust_limit=layout)
    result = filt.filter(FAR.copy(), np.array(command))
    assert isinstance(result, FilterResult)
    np.testing.assert_allclose(result.action, clipped, rtol=0, atol=1e-12)
    assert result.intervened is False and result.active is False


def test_ball_clipping_scales_the_command_radially():
    filt = ready(cbf(), dt=2.0, thrust_limit="ball")
    result = filt.filter(FAR.copy(), np.array([3.0, 0.0, 3.0]))
    np.testing.assert_allclose(result.action, [math.sqrt(0.5), 0.0, math.sqrt(0.5)], atol=1e-12)
    assert result.intervened is False


# Outcomes and FilterResult fields.


def test_cbf_with_inactive_rows_passes_the_command():
    result = ready(cbf()).filter(FAR.copy(), np.zeros(3))
    np.testing.assert_array_equal(result.action, np.zeros(3))
    assert fields(result)[1:] == (
        False, "feasible", 0.0, (), False, False, 0, None, None, None
    )  # fmt: skip


def test_cbf_with_an_active_row_meets_it():
    filt = ready(cbf())
    x = np.array([0.0, 55.0, 0.0, 0.0, -0.05, 0.0])
    assert in_declared_set(x, filt.declaration, 0.1)
    result = filt.filter(x.copy(), np.array([0.0, -1.0, 0.0]))
    assert fields(result)[1:] == (
        True, "feasible", 0.0, (), True, False, 0, None, None, None
    )  # fmt: skip
    keep_out = KeepOutEcbf(radius=R + filt.declaration.buffer_m, poles=(0.05, 0.05))
    a, b = keep_out.row(x, cw_accel(x))
    assert a @ (A_MAX * np.asarray(result.action)) >= b - 1e-9 * A_MAX


def test_cbf_with_rows_beyond_the_box_reports_certified_infeasibility():
    filt = ready(cbf())
    # Closing at 0.5 m/s from 1 m: the keep-out row asks rhat . u >= 0.0475 m/s^2.
    x = np.array([0.0, 51.0, 0.0, 0.0, -0.5, 0.0])
    result = filt.filter(x.copy(), np.array([0.0, -1.0, 0.0]))
    assert result.status == "infeasible_certified"
    assert result.active is True and result.recovering is True
    assert result.conflicting and result.fallback_level == 0
    assert result.certificate is None and result.certificate_horizon is None
    assert result.iterations is None
    keep_out = KeepOutEcbf(radius=R + filt.declaration.buffer_m, poles=(0.05, 0.05))
    speed = SpeedCbf(v0=0.1, v1=0.0, gamma=0.02)
    rows = [keep_out.row(x, cw_accel(x)), speed.row(x, cw_accel(x))]
    rows = Rows(np.array([a for a, _ in rows]), np.array([b for _, b in rows]), ("k", "s"))
    best = project(np.array([0.0, -A_MAX, 0.0]), rows, ActuatorSet(A_MAX, "box"))
    assert result.slack == pytest.approx(best.phase1_margin, rel=1e-6)
    # The least violation pushes outward at the full limit.
    assert result.action[1] == pytest.approx(1.0, abs=1e-9)


def test_cbf_solver_failure_falls_back_to_the_keep_out_support_maximiser(monkeypatch):
    real = project

    def failing(*args, **kwargs):
        return dataclasses.replace(real(*args, **kwargs), status="solver_failed")

    monkeypatch.setattr(cbf_module, "project", failing)
    filt = ready(cbf())
    x = np.array([0.0, 55.0, 0.0, 0.0, -0.05, 0.0])
    result = filt.filter(x.copy(), np.array([0.0, -1.0, 0.0]))
    assert result.status == "solver_failed" and result.active is True
    assert result.fallback_level == 2 and result.conflicting == ()
    np.testing.assert_allclose(result.action, [0.0, 1.0, 0.0], atol=1e-12)
    keep_out = KeepOutEcbf(radius=R + filt.declaration.buffer_m, poles=(0.05, 0.05))
    a, b = keep_out.row(x, cw_accel(x))
    assert result.slack == pytest.approx(max(0.0, b - a @ (A_MAX * result.action)), abs=1e-12)
    assert result.certificate is None and result.iterations is None


@pytest.mark.parametrize("kind", ["passive", "brake"])
def test_backup_accepting_the_candidate_reports_one_check(kind):
    filt = ready(FILTERS[kind]())
    result = filt.filter(FAR.copy(), np.zeros(3))
    np.testing.assert_array_equal(result.action, np.zeros(3))
    assert result.status == "feasible" and result.intervened is False
    assert result.active is False and result.recovering is False
    assert result.slack == 0.0 and result.conflicting == () and result.fallback_level == 0
    assert result.certificate == "nominal" and result.iterations == 1


def test_braking_certificate_holds_for_all_time():
    assert ready(braking()).filter(FAR.copy(), np.zeros(3)).certificate_horizon == math.inf


@pytest.mark.parametrize(
    "x",
    [
        FAR,
        # x = vy = 0: a 2:1 ellipse about the target, nearest it at 60 m.
        np.array([0.0, -120.0, 0.0, -60.0 * N, 0.0, 0.0]),
    ],
    ids=["v-bar hold", "ellipse"],
)
def test_passive_certificate_on_a_closed_relative_orbit_clear_of_the_keep_out_holds_for_all_time(
    x,
):
    # The drift minimum repeats every orbit and never falls below R, so the certificate has no
    # horizon.
    result = ready(BackupFilter(barrier="passive")).filter(x.copy(), np.zeros(3))
    assert result.recovering is False and result.certificate == "nominal"
    assert result.certificate_horizon == math.inf


def test_passive_certificate_with_secular_drift_holds_for_all_time():
    barrier = PassiveBarrier(radius=R, n=N)
    x = np.array([0.0, 400.0, 0.0, 0.0, -0.05, 0.0])
    assert barrier.contains(x) and barrier.horizon(x)[1] is True
    result = ready(BackupFilter(barrier="passive")).filter(x.copy(), np.zeros(3))
    assert result.certificate == "nominal" and result.certificate_horizon == math.inf


def test_braking_line_search_stopping_short_of_the_candidate():
    filt = ready(braking())
    x = np.array([0.0, 55.0, 0.0, 0.0, -0.2, 0.0])
    assert BrakingBarrier(radius=R, a_b=0.01044, rho_cap=2000.0, v_cap=1.0).contains(x)
    result = filt.filter(x.copy(), np.array([0.0, -1.0, 0.0]))
    # u = theta u_c + (1 - theta) beta with beta = (0, 1, 0) of a full command.
    assert result.action[0] == 0.0 and result.action[2] == 0.0
    assert -1.0 < result.action[1] < 1.0
    assert result.status == "feasible" and result.intervened is True and result.active is True
    assert result.recovering is False and result.slack == 0.0 and result.fallback_level == 0
    assert result.certificate == "nominal" and result.certificate_horizon == math.inf
    assert result.iterations == 13


def test_braking_line_search_falling_back_to_the_backup():
    # With a_b just below the box's outward limit, every theta > 2^-12 brakes too weakly from the
    # braking boundary, so only the backup itself is left.
    a_b, c = 0.019999, 0.2
    filt = ready(braking(a_b=a_b))
    x = np.array([0.0, R + c * c / (2 * a_b) + 1e-9, 0.0, 0.0, -c, 0.0])
    assert BrakingBarrier(radius=R, a_b=a_b, rho_cap=2000.0, v_cap=1.0).contains(x)
    result = filt.filter(x.copy(), np.array([0.0, -1.0, 0.0]))
    np.testing.assert_allclose(result.action, [0.0, 1.0, 0.0], atol=1e-12)
    assert result.status == "feasible" and result.active is True
    assert result.fallback_level == 2 and result.iterations == 13
    assert result.recovering is False and result.certificate == "nominal"


def test_braking_from_outside_its_set_reports_recovery_and_the_predicted_violation():
    filt = ready(braking())
    # h_b = 1 - 0.5^2 / (2 0.01044) < 0: even full braking enters the keep-out.
    x = np.array([0.0, 51.0, 0.0, 0.0, -0.5, 0.0])
    result = filt.filter(x.copy(), np.zeros(3))
    assert result.status == "feasible" and result.recovering is True
    assert result.active is True and result.fallback_level in (0, 2)
    assert result.slack > 0.0
    assert result.certificate is None and result.certificate_horizon is None


@pytest.mark.parametrize("kind", FILTERS)
def test_inside_the_commit_radius_the_filter_only_monitors(kind):
    base = view()
    committed = dataclasses.replace(base, goal={**base.goal, "commit_radius": 500.0})
    filt = FILTERS[kind]()
    filt.reset(committed)
    x = np.array([0.0, 51.0, 0.0, 0.0, -0.5, 0.0])
    result = filt.filter(x.copy(), np.array([0.0, -3.0, 0.0]))
    np.testing.assert_allclose(result.action, [0.0, -1.0, 0.0], atol=1e-12)
    assert fields(result)[1:] == (
        False, "feasible", 0.0, (), False, None, None, None, None, None
    )  # fmt: skip


def assert_backup_outcomes(flight, beta_of):
    """Each decision matches its row of the outcome table, theta read off the action."""
    for x, command, r in zip(flight.seen, flight.commands, flight.results, strict=True):
        candidate = np.clip(command, -1.0, 1.0)
        beta = beta_of(x)
        action = np.asarray(r.action)
        assert r.status == "feasible"
        if r.iterations == 1:
            np.testing.assert_allclose(action, candidate, atol=1e-12)
            assert r.active is False and r.fallback_level == 0
        else:
            assert r.iterations == 13 and r.active is True
            at_backup = np.allclose(action, beta, atol=1e-12)
            if not np.allclose(candidate, beta, atol=1e-12):
                assert (r.fallback_level == 2) == at_backup


# Closed loop on the exact CW environment at 500 km, keep-out 50 m, box 0.02 m/s^2.

CBF_STEPS = {1.0: 60, 5.0: 60, 10.0: 60, 20.0: 30}


def cbf_pole(dt):
    # p = 0.05 meets the pole rule up to dt 12 s; at 20 s the largest allowed pole is 0.03.
    return 0.05 if dt <= 10.0 else 0.025


@functools.cache
def cbf_flights(dt):
    p = cbf_pole(dt)
    return tuple(
        fly(cbf(p), CentrePD(), x0, dt, CBF_STEPS[dt]) for x0 in cbf_starts(p, dt)
    )  # fmt: skip


@pytest.mark.parametrize("dt", DTS)
def test_cbf_filter_keeps_starts_in_its_declared_set_out_of_the_keep_out(dt):
    for flight in cbf_flights(dt):
        assert flight.worst >= -DEPTH
        assert all(r.status == "feasible" for r in flight.results)
        assert flight.results[0].recovering is False


@pytest.mark.parametrize("dt", DTS)
def test_cbf_filter_speed_overshoot_stays_below_ten_percent(dt):
    for flight in cbf_flights(dt):
        assert flight.max_speed < 1.1 * 0.1


def riding_pole(dt):
    return 0.04 if dt <= 10.0 else 0.025


@functools.cache
def cbf_riding(dt):
    p = riding_pole(dt)
    filt = ready(cbf(p, v_cap=0.2), dt)
    x0 = np.array([0.0, R + filt.declaration.buffer_m + 1e-6, 0.0, 0.2, 0.0, 0.0])
    assert in_declared_set(x0, filt.declaration, 0.2)
    return fly(cbf(p, v_cap=0.2), Reckless(), x0, dt, 25)


@pytest.mark.parametrize("dt", DTS)
def test_cbf_filter_rides_the_boundary_without_entering(dt):
    flight = cbf_riding(dt)
    assert flight.worst >= -DEPTH
    assert all(r.status == "feasible" for r in flight.results)
    assert any(r.active for r in flight.results)


@pytest.mark.parametrize("dt", DTS)
def test_cbf_filter_riding_the_boundary_overshoots_its_speed_cap_by_under_ten_percent(dt):
    assert cbf_riding(dt).max_speed < 1.1 * 0.2


@settings(max_examples=20, deadline=None)
@given(
    direction=st.tuples(*[st.floats(-1.0, 1.0, allow_subnormal=False)] * 3),
    h=st.floats(0.0, 300.0, allow_subnormal=False),
    velocity=st.tuples(*[st.floats(-0.1, 0.1, allow_subnormal=False)] * 3),
    command=st.tuples(*[st.floats(-5.0, 5.0, allow_subnormal=False)] * 3),
    layout=st.sampled_from(["box", "ball"]),
)
def test_cbf_rows_are_feasible_everywhere_in_the_declared_set(
    direction, h, velocity, command, layout
):
    d = np.array(direction)
    assume(np.linalg.norm(d) > 0.1)
    filt = ready(cbf(), thrust_limit=layout)
    x = np.concatenate([(R + filt.declaration.buffer_m + h) * d / np.linalg.norm(d), velocity])
    assume(in_declared_set(x, filt.declaration, 0.1))
    result = filt.filter(x.copy(), np.array(command))
    assert result.status == "feasible" and result.recovering is False
    assert_in_actuator_set(result.action, layout)
    u = A_MAX * np.asarray(result.action)
    keep_out = KeepOutEcbf(radius=R + filt.declaration.buffer_m, poles=(0.05, 0.05))
    a, b = keep_out.row(x, cw_accel(x))
    assert a @ u >= b - 1e-9 * A_MAX
    speed_row = SpeedCbf(v0=0.1, v1=0.0, gamma=0.02).row(x, cw_accel(x))
    if speed_row is not None:
        assert speed_row[0] @ u >= speed_row[1] - 1e-9 * A_MAX


def ecbf_first_violation(x0, p, dt, t_end=1500.0):
    """Step of the first violation with the bare ECBF row and no thrust limit, or None."""
    keep_out = KeepOut(R)
    barrier = KeepOutEcbf(radius=R, poles=(p, p))
    dyn = CWDynamics(N)
    dyn.reset(x0)
    x = np.array(x0, dtype=float)
    times = np.linspace(0.0, dt, 20)
    for k in range(math.ceil(t_end / dt)):
        a, b = barrier.row(x, cw_accel(x))
        u = project(-4e-4 * x[:3] - 0.04 * x[3:], Rows(a[None], np.array([b]), ("k",))).u
        x_new, states = dyn.advance(u, dt, 20)
        m = segment_minimum(
            keep_out, dyn.state_in_last_segment, times, states, float(np.linalg.norm(u)), N, k * dt
        )
        if m.margin < -DEPTH:
            return k
        x = x_new
    return None


def test_the_bare_ecbf_row_with_p_dt_one_fails_from_every_start():
    # The continuous-time ECBF condition (Ames et al. 2019, sec. IV, eq. 19) held for 10 s at
    # p = 0.1 1/s, past the sampled-data threshold 0.787 / p, which the pole rule refuses.
    rng = np.random.default_rng(1903)
    for _ in range(100):
        while ecbf_draw(rng, 0.05) is None:
            pass
    starts = []
    while len(starts) < 100:
        x = ecbf_draw(rng, 0.1)
        if x is not None:
            starts.append(x)
    assert all(ecbf_first_violation(x0, 0.1, 10.0) is not None for x0 in starts)
    with pytest.raises(ValueError):
        ready(cbf(0.1))


def braking_row(x, a_b, gamma=0.02):
    """First-order row on the braking barrier while closing, None otherwise."""
    r, v = x[:3], x[3:]
    rho = np.linalg.norm(r)
    rhat = r / rho
    rho_dot = rhat @ v
    c = max(0.0, -rho_dot)
    if c == 0.0:
        return None
    cent = (v @ v - rho_dot**2) / rho
    h_b = rho - R - c * c / (2 * a_b)
    return rhat, a_b * (-gamma * h_b + c) / c - cent - rhat @ cw_accel(x)


def test_the_braking_barrier_as_a_qp_row_fails_under_sample_and_hold():
    # L_g h_b vanishes at rho_dot = 0, so the row says nothing at the turn and a held command
    # overshoots: the barrier needs the line search on the exact held segment.
    a_b = A_MAX - (3 * N * N * 2000.0 + 2 * N * 1.0)
    barrier = BrakingBarrier(radius=R, a_b=a_b, rho_cap=2000.0, v_cap=1.0)
    keep_out, actuator, dt = KeepOut(R), ActuatorSet(A_MAX, "box"), 10.0
    times = np.linspace(0.0, dt, 20)
    for x0 in braking_starts(barrier):
        dyn = CWDynamics(N)
        dyn.reset(x0)
        x, violated = np.array(x0), False
        for k in range(60):
            u = np.clip(-4e-4 * x[:3] - 0.04 * x[3:], -A_MAX, A_MAX)
            row = braking_row(x, a_b)
            if row is not None:
                u = project(u, Rows(row[0][None], np.array([row[1]]), ("b",)), actuator).u
            x, states = dyn.advance(u, dt, 20)
            thrust = float(np.linalg.norm(u))
            m = segment_minimum(
                keep_out, dyn.state_in_last_segment, times, states, thrust, N, k * dt
            )
            if m.margin < -DEPTH:
                violated = True
                break
        assert violated


BRAKING_STEPS = {1.0: 60, 5.0: 60, 10.0: 60, 20.0: 30}


def braking_beta(x):
    """Full outward braking inside the box, along rhat, in fractions of a full command."""
    return x[:3] / np.max(np.abs(x[:3]))


@pytest.mark.parametrize("dt", BRAKING_STEPS)
def test_braking_backup_filter_keeps_starts_out_at_every_dt(dt):
    barrier = BrakingBarrier(radius=R, a_b=0.01044, rho_cap=2000.0, v_cap=1.0)
    for x0 in braking_starts(barrier):
        flight = fly(braking(), CentrePD(), x0, dt, BRAKING_STEPS[dt])
        assert flight.worst >= -DEPTH
        assert flight.results[0].recovering is False
        # This a_b carries no allowance for the turn of rhat within a step, so the set need
        # not be invariant; only decisions made inside it carry the certificate.
        inside = [r for r in flight.results if r.recovering is False]
        assert all(r.certificate == "nominal" for r in inside)
        assert all(r.certificate_horizon == math.inf for r in inside)
        assert_backup_outcomes(flight, braking_beta)


@pytest.mark.parametrize("dt", [10.0, 20.0])
def test_braking_backup_filter_with_the_default_deceleration_keeps_starts_out(dt):
    a_b = braking_deceleration(
        a_max=A_MAX, layout="box", v_cap=0.6, dt=dt, radius=R, rho_cap=2000.0, n=N
    )
    barrier = BrakingBarrier(radius=R, a_b=a_b, rho_cap=2000.0, v_cap=0.6)
    for x0 in braking_starts(barrier):
        filt = BackupFilter(barrier="braking", rho_cap=2000.0, v_cap=0.6)
        flight = fly(filt, CentrePD(), x0, dt, BRAKING_STEPS[dt])
        assert flight.worst >= -DEPTH
        assert all(r.status == "feasible" for r in flight.results)
        # The default a_b covers the turn of rhat, so full braking keeps the set invariant.
        assert all(r.recovering is False for r in flight.results)


@pytest.mark.parametrize("dt", DTS)
def test_braking_backup_filter_rides_the_boundary_without_entering(dt):
    x0 = np.array([0.0, R + 1e-6, 0.0, 0.2, 0.0, 0.0])
    flight = fly(braking(a_b=None, v_cap=0.3), Reckless(), x0, dt, 30)
    assert flight.worst >= -DEPTH
    assert all(r.status == "feasible" for r in flight.results)
    assert any(r.active for r in flight.results)
    assert_backup_outcomes(flight, braking_beta)


@functools.cache
def passive_flights(candidate=None, steps=360, count=4):
    extra = {} if candidate is None else {"candidate": candidate}
    return tuple(
        fly(BackupFilter(barrier="passive", **extra), HoldPD(), x0, 10.0, steps)
        for x0 in passive_starts()[:count]
    )


def test_passive_backup_filter_keeps_starts_out_and_reaches_the_hold_point():
    # Free drift is the backup of Ames et al. 2019, sec. III, eq. 12, with beta = 0.
    for flight in passive_flights():
        assert flight.worst >= -DEPTH
        assert all(r.status == "feasible" for r in flight.results)
        assert flight.captured


def test_passive_backup_filter_acts_on_some_steps_only():
    results = [r for flight in passive_flights() for r in flight.results]
    fraction = np.mean([r.active for r in results])
    assert 0.0 < fraction < 1.0
    for flight in passive_flights():
        assert_backup_outcomes(flight, lambda x: np.zeros(3))


def test_passive_certificate_holds_for_all_time_on_every_step_inside_the_set():
    inside = [r for flight in passive_flights() for r in flight.results if not r.recovering]
    assert inside
    for r in inside:
        assert r.certificate == "nominal" and r.certificate_horizon == math.inf


def mean_correction(flights, steps):
    sizes = [
        np.linalg.norm(np.asarray(r.action) - np.clip(c, -1.0, 1.0))
        for flight in flights
        for c, r in list(zip(flight.commands, flight.results, strict=True))[:steps]
        if r.active
    ]
    return float(np.mean(sizes))


def test_projection_candidate_corrects_less_than_the_clipped_nominal():
    projected = passive_flights("projection", 30)
    for flight in projected:
        assert flight.worst >= -DEPTH
        assert all(r.status == "feasible" for r in flight.results)
    assert mean_correction(projected, 30) < mean_correction(passive_flights(), 30)


def test_cbf_filter_with_a_declared_execution_error_absorbs_a_persistent_push():
    filt = ready(cbf(0.05, execution=2e-4))
    x0 = np.array([0.0, R + filt.declaration.buffer_m + 1e-6, 0.0, 0.1, 0.0, 0.0])
    flight = fly(cbf(0.05, execution=2e-4), Reckless(), x0, 10.0, 30, push=2e-4)
    assert flight.worst >= -DEPTH
    assert all(r.status == "feasible" for r in flight.results)


def test_braking_filter_with_a_declared_execution_error_absorbs_a_persistent_push():
    x0 = np.array([0.0, R + 1e-6, 0.0, 0.2, 0.0, 0.0])
    filt = braking(a_b=None, v_cap=0.3, execution=2e-4)
    flight = fly(filt, Reckless(), x0, 10.0, 30, push=2e-4)
    assert flight.worst >= -DEPTH
    assert all(r.status == "feasible" for r in flight.results)
    assert all(r.certificate in ("bounded-W", None) for r in flight.results)


# Truth isolation: a filter sees the view and the observations, never the environment.

SCRIPT = [(0.0, 62.0 - k, 0.0, 0.0, -0.1, 0.0) for k in range(10)]
TRUTHS = [FAR, np.array([150.0, -300.0, 30.0, 0.05, -0.05, 0.0])]


@pytest.mark.parametrize("kind", FILTERS)
def test_two_true_starts_seen_through_one_script_give_identical_decisions(kind):
    runs = [
        fly(FILTERS[kind](), CentrePD(), x0, 10.0, len(SCRIPT), sensor=ScriptedSensor(SCRIPT))
        for x0 in TRUTHS
    ]
    first, second = ([fields(r) for r in run.results] for run in runs)
    assert len(first) == len(SCRIPT) and first == second


@pytest.mark.parametrize("kind", FILTERS)
def test_recovery_is_judged_on_the_estimate(kind):
    outside = (0.0, 51.0, 0.0, 0.0, -0.5, 0.0)
    flight = fly(FILTERS[kind](), CentrePD(), FAR, 10.0, 1, sensor=ScriptedSensor([outside]))
    assert flight.results[0].recovering is True
    flight = fly(FILTERS[kind](), CentrePD(), outside, 10.0, 1, sensor=ScriptedSensor([FAR]))
    assert flight.results[0].recovering is False


# The free-drift set is kept by the drift step: a member's backup is admissible at every step.
START_BOX = ((-200.0, 200.0, -50.0, -0.1, -0.1, -0.1), (200.0, 600.0, 50.0, 0.1, 0.1, 0.1))
NEAR_BOX = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))
BOXES = {"start": START_BOX, "near": NEAR_BOX}
# x_c = delta and vy = -1.5 n delta: the drift keeps x = delta and closes along-track at
# 1.5 n delta m/s. With these delta the drift that enters at t_cap starts inside each box.
SLOW_DRIFT = {"start": (2.0, 5.0), "near": (0.5, 1.0)}
# Enters the keep-out about 1.4 s after the ten-orbit cap.
X_STAR = np.array([5.0, 521.0, 0.0, 0.0, -1.5 * N * 5.0, 0.0])
DRIFT_STEP = cw_discrete(N, 10.0)[0]
T_CAP = 20.0 * math.pi / N


def slow_drift(delta, y):
    return np.array([delta, y, 0.0, 0.0, -1.5 * N * delta, 0.0])


def entering_after_the_cap(delta):
    """The slow drifter that reaches the keep-out 2 s after the ten-orbit cap."""
    return slow_drift(delta, math.sqrt(R * R - delta * delta) + 1.5 * N * delta * (T_CAP + 2.0))


@functools.cache
def box_draws(box):
    lo, hi = BOXES[box]
    # About 2% of the start box lies outside the set, so it needs more draws to pair with.
    count = 400 if box == "start" else 200
    xs = np.random.default_rng(1135).uniform(lo, hi, (count, 6))
    inside = np.array([passive_filter().contains(x) for x in xs])
    return xs, inside


@functools.cache
def passive_filter_at(dt):
    return ready(BackupFilter(barrier="passive"), dt)


def bisect_to_boundary(inside, outside, contains, steps=28):
    """The in-set end of [inside, outside] after halving it `steps` times."""
    for _ in range(steps):
        mid = 0.5 * (inside + outside)
        if contains(mid):
            inside = mid
        else:
            outside = mid
    return inside


def test_slow_drift_entering_the_keep_out_after_t_cap_is_recovering_not_solver_failed():
    filt = ready(BackupFilter(barrier="passive"))
    assert not filt.contains(X_STAR)
    result = filt.filter(X_STAR.copy(), np.zeros(3))
    assert result.status == "feasible" and result.recovering is True
    assert result.certificate is None and result.certificate_horizon is None


@pytest.mark.parametrize("box", BOXES)
def test_free_drift_step_keeps_in_set_states_of_the_box_in_the_set(box):
    xs, inside = box_draws(box)
    planted = X_STAR if box == "start" else entering_after_the_cap(1.0)
    members = [x for x in (*xs[inside][:100], planted) if passive_filter().contains(x)]
    assert len(members) >= 80
    left = [x for x in members if not passive_filter().contains(DRIFT_STEP @ x)]
    assert left == []


@pytest.mark.parametrize("half", [0, 1])
@pytest.mark.parametrize("box", BOXES)
def test_free_drift_step_keeps_boundary_states_in_the_set(box, half):
    xs, inside = box_draws(box)
    members, others = xs[inside], xs[~inside]
    assert len(others) > 0
    pairs = range(5 * half, 5 * half + 5)
    contains = passive_filter().contains
    boundary = [bisect_to_boundary(members[i], others[i % len(others)], contains) for i in pairs]
    lo, hi = BOXES[box][0][1], BOXES[box][1][1]
    for delta in SLOW_DRIFT[box]:
        a, b = slow_drift(delta, hi), slow_drift(delta, lo)
        if contains(a) and not contains(b):
            boundary.append(bisect_to_boundary(a, b, contains))
    left = [x for x in boundary if not contains(DRIFT_STEP @ x)]
    assert left == []


@pytest.mark.parametrize(("dt", "count"), [(7.0, 5), (1.0, 3)])
def test_free_drift_step_keeps_boundary_states_in_the_set_when_the_grid_does_not_divide_dt(
    dt, count
):
    # The default 5 s grid divides neither step; the dt 1 s case gets fewer states because its
    # 1 s grid makes every check five times longer.
    filt = passive_filter_at(dt)
    xs, inside = box_draws("start")
    members, others = xs[inside], xs[~inside]
    boundary = []
    for i in range(count):
        a, b = members[i], others[i % len(others)]
        assert filt.contains(a) and not filt.contains(b)
        boundary.append(bisect_to_boundary(a, b, filt.contains))
    step = cw_discrete(N, dt)[0]
    left = [x for x in boundary if not filt.contains(step @ x)]
    assert left == []


def test_a_solver_failed_step_carries_no_certificate():
    # a_b = 0.03 m/s^2 overstates the box's outward limit of 0.02, so full braking from the
    # braking boundary cannot hold h_b >= 0 and the backup fails its own check inside the set.
    a_b, c = 0.03, 0.2
    filt = ready(braking(a_b=a_b))
    x = np.array([0.0, R + c * c / (2 * a_b) + 1e-9, 0.0, 0.0, -c, 0.0])
    assert filt.contains(x)
    result = filt.filter(x.copy(), np.array([0.0, -1.0, 0.0]))
    assert result.status == "solver_failed" and result.recovering is False
    assert result.certificate is None and result.certificate_horizon is None
