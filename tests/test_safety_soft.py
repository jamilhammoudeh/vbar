import dataclasses

import numpy as np
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from vbar.safety.barriers import KeepOutEcbf, SpeedCbf
from vbar.safety.cbf import CbfFilter, SoftCbfFilter
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.solvers.qp import ActuatorSet, Rows, project

R = 50.0
A_MAX = 0.02
FAR = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])


def view(**cfg):
    return ApproachEnv(ApproachConfig(start_state=tuple(FAR), **cfg)).agent_view()


def soft(alpha, p=0.05, **kw):
    return SoftCbfFilter(poles=(p, p), alpha=alpha, v_cap=0.1, gamma=0.02, **kw)


def hard(p=0.05):
    return CbfFilter(poles=(p, p), v_cap=0.1, gamma=0.02)


def ready(filt, **cfg):
    filt.reset(view(**cfg))
    return filt


def cw_accel(x, n):
    return np.array([3 * n * n * x[0] + 2 * n * x[4], -2 * n * x[3], -n * n * x[2]])


def hard_rows(x, filt, n):
    """The keep-out and speed-cap rows the filter softens, in m/s^2."""
    keep_out = KeepOutEcbf(radius=R + filt.declaration.buffer_m, poles=(0.05, 0.05))
    rows = [keep_out.row(x, cw_accel(x, n))]
    speed = SpeedCbf(v0=0.1, v1=0.0, gamma=0.02).row(x, cw_accel(x, n))
    if speed is not None:
        rows.append(speed)
    names = tuple(f"row{i}" for i in range(len(rows)))
    return Rows(np.array([a for a, _ in rows]), np.array([b for _, b in rows]), names)


def penalty_objective(u, u_nom, rows, alpha):
    return 0.5 * float((u - u_nom) @ (u - u_nom)) + alpha * float(
        np.maximum(0.0, rows.b - rows.a @ u).sum()
    )


def test_soft_filter_takes_keyword_arguments_only():
    with pytest.raises(TypeError):
        SoftCbfFilter((0.05, 0.05), 1.0)


def test_soft_filter_is_named_apart_and_never_counts_toward_certification():
    filt = soft(1.0)
    assert filt.name == "cbf-ecbf-soft-p0.05"
    d = ready(filt).declaration
    assert d.name == "cbf-ecbf-soft-p0.05"
    assert d.counts_toward_certification is False
    assert d.certificate is None


def test_soft_filter_keeps_the_hard_filters_pole_rule():
    with pytest.raises(ValueError, match=r"0\.06"):
        ready(soft(1.0, p=0.1))


def test_soft_filter_clipping_is_not_an_intervention():
    result = ready(soft(1.0)).filter(FAR.copy(), np.array([0.0, 0.0, 3.0]))
    np.testing.assert_allclose(result.action, [0.0, 0.0, 1.0], atol=1e-12)
    assert result.intervened is False and result.slack == 0.0


def test_inside_the_commit_radius_the_soft_filter_only_monitors():
    base = view()
    committed = dataclasses.replace(base, goal={**base.goal, "commit_radius": 500.0})
    filt = soft(1e-4)
    filt.reset(committed)
    x = np.array([0.0, 51.0, 0.0, 0.0, -0.5, 0.0])
    result = filt.filter(x.copy(), np.array([0.0, -3.0, 0.0]))
    np.testing.assert_allclose(result.action, [0.0, -1.0, 0.0], atol=1e-12)
    assert result.status == "feasible" and result.intervened is False
    assert result.active is False and result.recovering is None
    assert result.slack == 0.0 and result.fallback_level is None


def test_soft_filter_accepts_slack_where_the_hard_rows_are_infeasible():
    x = np.array([0.0, 51.0, 0.0, 0.0, -0.5, 0.0])
    command = np.array([0.0, -1.0, 0.0])
    assert ready(hard()).filter(x.copy(), command).status == "infeasible_certified"
    result = ready(soft(1.0)).filter(x.copy(), command)
    assert result.status != "solver_failed"
    assert result.slack > 0.0 and result.certificate is None


STATES = dict(
    direction=st.tuples(*[st.floats(-1.0, 1.0, allow_subnormal=False)] * 3),
    h=st.floats(0.0, 20.0, allow_subnormal=False),
    velocity=st.tuples(*[st.floats(-0.1, 0.1, allow_subnormal=False)] * 3),
    command=st.tuples(*[st.floats(-3.0, 3.0, allow_subnormal=False)] * 3),
)


def build_state(direction, h, velocity, filt):
    d = np.array(direction)
    assume(np.linalg.norm(d) > 0.1)
    rho = R + filt.declaration.buffer_m + h
    x = np.concatenate([rho * d / np.linalg.norm(d), velocity])
    keep_out = KeepOutEcbf(radius=R + filt.declaration.buffer_m, poles=(0.05, 0.05))
    assume(keep_out.contains(x))
    return x


@settings(max_examples=40, deadline=None)
@given(alpha=st.floats(1e-5, 0.1, allow_subnormal=False), **STATES)
def test_soft_filter_matches_the_hard_projection_exactly_when_alpha_covers_the_multipliers(
    alpha, direction, h, velocity, command
):
    # The l1 penalty is exact: the hard projection meets the optimality conditions of the
    # penalised problem when alpha is at least every row multiplier, and fails them otherwise.
    soft_filter = ready(soft(alpha))
    n = view().n
    x = build_state(direction, h, velocity, soft_filter)
    u_nom = A_MAX * np.clip(command, -1.0, 1.0)
    reference = project(u_nom, hard_rows(x, soft_filter, n), ActuatorSet(A_MAX, "box"))
    assume(reference.status == "feasible")
    lam = float(np.max(reference.multipliers, initial=0.0))
    assume(abs(alpha - lam) > 1e-6)
    soft_result = soft_filter.filter(x.copy(), np.array(command))
    assert np.all(np.abs(np.asarray(soft_result.action)) <= 1.0)
    if alpha > lam:
        expected = reference.u / A_MAX
        np.testing.assert_allclose(soft_result.action, expected, rtol=0, atol=1e-9)
        assert soft_result.slack == 0.0
    else:
        assert soft_result.slack > 0.0


@settings(max_examples=30, deadline=None)
@given(alpha=st.floats(1e-5, 0.1, allow_subnormal=False), **STATES)
# Once returned |u_z| = 0.02 (1 + 7.5e-11), beyond the box.
@example(
    alpha=0.0625,
    direction=(0.0, 0.0, 1.0),
    h=8.0,
    velocity=(0.0625, 0.0625, 9.5e-12),
    command=(0.0, 1.0, -1.0),
)
def test_soft_filter_action_minimises_the_penalised_objective(
    alpha, direction, h, velocity, command
):
    filt = ready(soft(alpha))
    n = view().n
    x = build_state(direction, h, velocity, filt)
    u_nom = A_MAX * np.clip(command, -1.0, 1.0)
    rows = hard_rows(x, filt, n)
    action = np.asarray(filt.filter(x.copy(), np.array(command)).action)
    # The returned command lies in the box exactly, in fractions of a full command.
    assert np.all(np.abs(action) <= 1.0)
    u = A_MAX * action
    best = penalty_objective(u, u_nom, rows, alpha)
    rng = np.random.default_rng(0)
    trials = rng.uniform(-A_MAX, A_MAX, size=(500, 3))
    trials = np.vstack([trials, project(u_nom, rows, ActuatorSet(A_MAX, "box")).u, u_nom])
    for trial in trials:
        assert best <= penalty_objective(trial, u_nom, rows, alpha) + 1e-12


def test_the_audit_refuses_a_soft_filter():
    audit = pytest.importorskip(
        "vbar.safety.audit", reason="the filter audit arrives with the audit harness"
    )
    with pytest.raises(ValueError, match="soft|certif"):
        audit.audit_filter(soft(1.0), ApproachEnv())
