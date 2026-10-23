import dataclasses

import numpy as np
import pytest
from scipy.linalg import solve_continuous_are
from scipy.stats import kstest

from vbar.dynamics.cw import cw_states, cw_system
from vbar.safety import InfeasibilityProof, PassThroughFilter
from vbar.safety.audit import audit_filter
from vbar.safety.rta import (
    AfrlConstraints,
    AfrlSetup,
    ExplicitOptimizationFilter,
    ExplicitSwitchingFilter,
    ImplicitOptimizationFilter,
    ImplicitSwitchingFilter,
    NmtBackup,
    reproduce,
    start_state,
)
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.solvers.qp import ActuatorSet, project

# Dunlap et al. 2021, Assumption 2: n = 0.001027 rad/s, 1 N on 12 kg, nu0 = 0.2 m/s and
# v_max = 10 m/s.
N = 0.001027
A_MAX = 1.0 / 12.0
CODE = AfrlSetup.authors_code()
PAPER = AfrlSetup.paper()
CW_A, CW_B = cw_system(N)
FILTERS = (
    ExplicitSwitchingFilter,
    ExplicitOptimizationFilter,
    ImplicitSwitchingFilter,
    ImplicitOptimizationFilter,
)
# States away from saturation, target and latch switches: b capped, b free, and within the
# 1.5 m track radius of a grid point (built in `latched`).
FAR = np.array([6000.0, -8000.0, 1500.0, 0.3, -0.5, 0.1])
NEAR = np.array([800.0, -1500.0, 300.0, 0.2, 0.4, -0.1])
# 6.4 km out of plane at 9.76 m/s toward it: the backup pulls z toward its target, 3.6 km nearer
# the plane, harder than it damps vz, and with the gravity gradient it adds about 0.05 m/s to
# |vz| per step, so a rollout's last point passes -v_max.
EDGE = np.array([-3674.0, 2010.7, 6403.8, -5.95, 3.675, -9.76])


def euler(x, u, dt=1.0):
    return x + dt * (CW_A @ x + CW_B @ u)


def phi(x, nu1=2.15 * N):
    """Eq. 16 and 19, written out."""
    v = x[3:]
    return np.array(
        [0.2 + nu1 * np.linalg.norm(x[:3]) - np.linalg.norm(v), *(100.0 - v**2)], dtype=float
    )


def steps_for(x):
    # Relative steps keep the truncation of the central difference near 1e-8 and its rounding
    # below 1e-7 of entries of order one.
    pos = 1e-4 * max(float(np.linalg.norm(x[:3])), 1.0)
    vel = 1e-4 * max(float(np.linalg.norm(x[3:])), 0.1)
    return (pos,) * 3 + (vel,) * 3


def jacobian(fun, x):
    """Central differences of `fun` at `x`, one column per state entry."""
    cols = []
    for k, h in enumerate(steps_for(x)):
        e = np.zeros(6)
        e[k] = h
        cols.append((np.asarray(fun(x + e)) - np.asarray(fun(x - e))) / (2.0 * h))
    return np.stack(cols, axis=-1)


def assert_close_columns(got, want, rtol=1e-6):
    got, want = np.atleast_2d(got), np.atleast_2d(want)
    for k in range(want.shape[-1]):
        scale = max(np.linalg.norm(want[..., k]), np.linalg.norm(got[..., k]))
        assert np.linalg.norm(got[..., k] - want[..., k]) <= rtol * scale + 1e-14


def ready(cls, setup=CODE, **view):
    filt = cls(setup=setup)
    filt.reset(dataclasses.replace(setup.view(), **view))
    return filt


def latched(backup):
    """A state 0.5 m and 1 cm/s from the backup's target, so the desired point moves at once."""
    target = backup.rollout(NEAR, 1).targets[0]
    return target + np.array([0.3, -0.3, 0.2, 0.01, -0.005, 0.002])


def draw_inside(filt, count, seed):
    """Seeded states inside the filter's declared set, up to 9 km and 90% of the speed cap."""
    rng = np.random.default_rng(seed)
    found = []
    while len(found) < count:
        rho = rng.uniform(10.0, 9000.0)
        cap = min(0.2 + 2.15 * N * rho, 10.0)
        r = rng.normal(size=3)
        v = rng.normal(size=3)
        x = np.concatenate(
            [
                rho * r / np.linalg.norm(r),
                rng.uniform(0.0, 0.9 * cap) * v / np.linalg.norm(v),
            ]
        )
        if filt.contains(x):
            found.append(x)
    return found


def test_presets_hold_the_paper_values_and_the_code_slope():
    assert PAPER.nu1 == pytest.approx(4.0 * N, rel=1e-15)
    assert CODE.nu1 == pytest.approx(2.15 * N, rel=1e-15)
    for setup in (PAPER, CODE):
        assert setup.a_max == pytest.approx(A_MAX, rel=1e-15)
        assert setup.horizon_steps == 5
        view = setup.view()
        assert view.scenario_id == "dunlap2021-euler"
        assert (view.action_mode, view.thrust_limit, view.dt, view.n) == ("accel", "box", 1.0, N)
        assert view.max_accel == pytest.approx(A_MAX, rel=1e-15)
        assert view.max_steps == 5000
        assert [dict(c) for c in view.constraints] == [
            {"type": "speed_limit", "v0": 0.2, "v1": setup.nu1}
        ]


@pytest.mark.parametrize(
    "change",
    [
        {"mass_kg": 0.0},
        {"nu1": -1e-3},
        {"dt": float("nan")},
        {"horizon_s": 4.5, "dt": 2.0},
        {"alpha_speed": (0.0, 0.0)},
        {"alpha_velocity": (-1e-4, 1e-3)},
        {"nmt_grid": (20, 0, 20)},
    ],
)
def test_setup_refuses_values_out_of_range(change):
    with pytest.raises(ValueError):
        dataclasses.replace(CODE, **change)


def test_lemma_numbers_of_the_paper_and_the_code_check():
    # Lemma 1 at nu1 = 4 n: (3 n^2 + 2 n nu1 + nu1^2) R_max + (2 n + nu1) nu0 = 27 n^2 R_max +
    # 6 n nu0, above the 1/12 m/s^2 the thruster gives.
    assert PAPER.lemma1_accel == pytest.approx(0.2860, abs=5e-5)
    assert PAPER.lemma1_accel > PAPER.a_max
    # The code's check passes at 2.15 n by 0.02% and fails at 4 n.
    assert CODE.code_lemma_accel == pytest.approx(0.08332, abs=5e-6)
    assert CODE.code_lemma_accel < CODE.a_max
    assert PAPER.code_lemma_accel == pytest.approx(0.1049, abs=5e-5)
    assert PAPER.code_lemma_accel > PAPER.a_max


def test_constraint_values_follow_eq_16_and_19():
    cons = AfrlConstraints(CODE)
    rng = np.random.default_rng(3)
    states = np.concatenate([rng.uniform(-9e3, 9e3, (20, 3)), rng.uniform(-12, 12, (20, 3))], 1)
    want = np.array([phi(x) for x in states])
    np.testing.assert_allclose(cons.values(states), want, rtol=1e-13, atol=1e-12)
    assert np.array_equal(cons.contains(states), np.all(want >= 0.0, axis=1))
    # alpha(h) = a1 h + a3 h^3 with (0.05, 0.1) on the speed limit and (5e-4, 1e-3) per axis.
    h = np.array([-0.5, 2.0, 50.0, -3.0])
    np.testing.assert_allclose(cons.alpha(h), [-0.0375, 0.009, 125.025, -0.0285], rtol=1e-13)


def test_constraint_gradients_match_central_differences():
    cons = AfrlConstraints(CODE)
    rng = np.random.default_rng(5)
    for _ in range(20):
        x = np.concatenate([rng.uniform(-9e3, 9e3, 3), rng.uniform(-9.0, 9.0, 3)])
        fd = jacobian(cons.values, x)
        got = cons.gradients(x)
        for i in range(4):
            assert_close_columns(got[i][:, None], fd[i][:, None])


def test_kept_nmt_points_return_to_themselves_after_one_period():
    period = 2.0 * np.pi / N
    for setup in (PAPER, CODE):
        backup = NmtBackup(setup)
        assert len(backup.points) > 0
        for b in (1.0, 100.0, backup.b_cap):
            start = b * backup.points
            end = cw_states(N, np.full(len(start), period), start, np.zeros((len(start), 3)))
            dr = np.linalg.norm(end[:, :3] - start[:, :3], axis=1)
            dv = np.linalg.norm(end[:, 3:] - start[:, 3:], axis=1)
            assert np.all(dr <= 1e-9 * np.linalg.norm(start[:, :3], axis=1))
            assert np.all(dv <= 1e-9 * np.linalg.norm(start[:, 3:], axis=1))


@pytest.mark.parametrize(("setup", "count"), [(PAPER, 2640), (CODE, 780)])
def test_nmt_grid_is_the_papers_ellipse_on_lemma_3_points(setup, count):
    # Sec. III.D: x = b sin nu, y = 2 b cos nu, z = c sin psi with c / b and nu from theta1 and
    # theta2, nu = atan(2 cos theta1 / tan theta2) - psi, on a 20 x 20 x 20 grid with theta1 in
    # (0, pi/2], theta2 in [0, pi/2) and psi in [0, 2 pi).
    theta1 = np.linspace(0.0, np.pi / 2, 21)[1:]
    theta2 = np.linspace(0.0, np.pi / 2, 21)[:-1]
    psi = np.arange(20) * np.pi / 10
    want = []
    for t1 in theta1:
        for t2 in theta2:
            c2 = (np.tan(t2) ** 2 + 4.0 * np.cos(t1) ** 2) / np.sin(t1) ** 2
            if c2 > min((setup.nu1 / N) ** 2 - 4.0, 4.0):
                continue
            c = np.sqrt(c2)
            beta = np.pi / 2 if t2 == 0.0 else np.arctan(2.0 * np.cos(t1) / np.tan(t2))
            for p in psi:
                nu = beta - p
                want.append(
                    [
                        np.sin(nu),
                        2 * np.cos(nu),
                        c * np.sin(p),
                        N * np.cos(nu),
                        -2 * N * np.sin(nu),
                        N * c * np.cos(p),
                    ]
                )
    got = NmtBackup(setup).points
    assert len(got) == len(want) == count
    np.testing.assert_allclose(np.array(got), np.array(want), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("setup", [PAPER, CODE], ids=["4n", "2.15n"])
def test_kept_nmt_points_meet_the_speed_limit_at_3600_phases(setup):
    points = NmtBackup(setup).points
    tau = 2.0 * np.pi * np.arange(3600) / 3600
    # On the unit orbit (b = 1) at phase tau, r = p cos tau + w sin tau and
    # v / n = w cos tau - p sin tau, with p the position and w the velocity over n.
    pos, w = points[:, :3], points[:, 3:] / N
    pp, pw, ww = (np.sum(a * b, axis=1)[:, None] for a, b in ((pos, pos), (pos, w), (w, w)))
    # nu1^2 |r|^2 - |v|^2 = A cos^2 + B sin cos + C sin^2, with nu0 = 0, which b scales out;
    # the limit with nu0 > 0 then holds too.
    q, n2 = setup.nu1**2, N**2
    a, b, c = q * pp - n2 * ww, 2.0 * (q + n2) * pw, q * ww - n2 * pp
    margin = np.outer(a, np.cos(tau) ** 2) + np.outer(b, np.sin(2.0 * tau) / 2.0)
    margin += np.outer(c, np.sin(tau) ** 2)
    assert np.all(margin >= -1e-12 * q * (pp + ww))


def test_kept_nmt_points_meet_lemma_3_and_stay_allowable_at_every_phase():
    for setup in (PAPER, CODE):
        backup = NmtBackup(setup)
        cons = AfrlConstraints(setup)
        p = backup.points
        # On the unit orbit (b = 1), x = sin nu, y = 2 cos nu and |z| peaks at c / b.
        assert np.allclose(p[:, 0] ** 2 + (p[:, 1] / 2.0) ** 2, 1.0, rtol=1e-14)
        c = np.hypot(p[:, 2], p[:, 5] / N)
        assert np.all(c**2 <= (setup.nu1 / N) ** 2 - 4.0 + 1e-12)
        assert np.all(c <= 2.0 + 1e-12)
        # The exact minimum over phase of nu1^2 |r|^2 - |v|^2 on a unit orbit is
        # (q - 1)(5/2 + c^2/2) - (q + 1)|3/2 - (c^2/2) e^(2 i delta)| times n^2, with
        # q = (nu1/n)^2 and delta the phase of z behind x.
        q = (setup.nu1 / N) ** 2
        delta = np.arctan2(p[:, 2], p[:, 5] / N) - np.arctan2(p[:, 0], p[:, 3] / N)
        spread = np.abs(1.5 - 0.5 * c**2 * np.exp(2j * delta))
        assert np.all((q - 1.0) * (2.5 + 0.5 * c**2) - (q + 1.0) * spread >= -1e-12)
        times = np.linspace(0.0, 2.0 * np.pi / N, 73)
        for b in (5.0, backup.b_cap):
            start = np.repeat(b * p, len(times), axis=0)
            dt = np.tile(times, len(p))
            states = cw_states(N, dt, start, np.zeros((len(start), 3)))
            assert np.all(cons.values(states) >= -1e-9)


@pytest.mark.parametrize("name", ["far", "near", "latched"])
def test_backup_sensitivities_match_central_differences(name):
    backup = NmtBackup(CODE)
    x = {"far": FAR, "near": NEAR, "latched": latched(backup)}[name]
    roll = backup.rollout(x, 5)
    assert roll.states.shape == (6, 6) and roll.sensitivities.shape == (6, 6, 6)
    np.testing.assert_array_equal(roll.sensitivities[0], np.eye(6))
    moved = not np.array_equal(roll.targets[1], roll.targets[0])
    assert moved == (name == "latched")
    fd = jacobian(lambda s: backup.rollout(s, 5).states, x)
    for j in range(6):
        assert_close_columns(roll.sensitivities[j], fd[j])


def test_backup_sensitivities_drop_the_feedback_on_saturated_axes():
    # Stiff weights saturate some axes of the rollout, so both branches of the Jacobian are used.
    stiff = dataclasses.replace(CODE, backup_q=1e-2, backup_r=1.0)
    backup = NmtBackup(stiff)
    x = backup.target(NEAR)[0] + np.array([30.0, 0.5, 0.2, 0.0, 0.0, 0.0])
    roll = backup.rollout(x, 5)
    raw = (roll.targets[:-1] - roll.states[:-1]) @ backup.gain.T
    saturated = np.abs(raw) >= stiff.a_max
    assert saturated.any() and not saturated.all()
    # No axis within 1% of its saturation limit, so the steps of the differences keep the pattern.
    assert np.all(np.abs(np.abs(raw) / stiff.a_max - 1.0) > 1e-2)
    fd = jacobian(lambda s: backup.rollout(s, 5).states, x)
    for j in range(6):
        assert_close_columns(roll.sensitivities[j], fd[j])


@pytest.mark.parametrize("name", ["far", "near", "latched"])
def test_implicit_row_normals_are_gradients_of_the_constraints_along_the_backup(name):
    filt = ready(ImplicitOptimizationFilter)
    backup = filt.backup
    x = {"far": FAR, "near": NEAR, "latched": latched(backup)}[name]
    rows = filt.rows(x)
    assert len(rows.names) == 24
    names = AfrlConstraints.names
    for a, row in zip(rows.a, rows.names, strict=True):
        constraint, j = row.split("@")
        i = names.index(constraint)

        def along(s, i=i, j=int(j)):
            return filt.constraints.values(backup.rollout(s, 5).states[j])[i]

        # g(x) = B selects the velocity entries of the gradient.
        assert_close_columns(a[:, None], (jacobian(along, x) @ CW_B)[:, None])


def test_backup_command_meets_every_implicit_row_inside_the_declared_set():
    filt = ready(ImplicitOptimizationFilter)
    for x in draw_inside(filt, 80, seed=17):
        rows = filt.rows(x)
        u_b = filt.backup.command(x)
        assert np.all(np.abs(u_b) <= A_MAX)
        slack = rows.a @ u_b - rows.b
        scale = np.maximum(np.abs(rows.b), np.abs(rows.a) @ np.full(3, A_MAX))
        assert np.all(slack >= -1e-12 * np.maximum(scale, 1e-12))
        # Each row's slack at u_b(x) is alpha(phi_i(phi_j)), written out from the constraints.
        states = filt.backup.rollout(x, 5).states
        want = []
        for row in rows.names:
            constraint, j = row.split("@")
            i = AfrlConstraints.names.index(constraint)
            h = phi(states[int(j)])[i]
            a1, a3 = (0.05, 0.1) if i == 0 else (5e-4, 1e-3)
            want.append(a1 * h + a3 * h**3)
        np.testing.assert_allclose(slack, want, rtol=1e-9, atol=1e-12 * scale.max())


def test_explicit_switching_passes_a_command_whose_next_state_is_allowable():
    filt = ready(ExplicitSwitchingFilter)
    x = np.array([3000.0, 2000.0, -500.0, -1.0, 0.5, 0.2])
    u = np.array([0.02, -0.05, 0.01])
    result = filt.filter(x, u)
    np.testing.assert_array_equal(result.action, u)
    assert not result.intervened and not result.active
    assert (result.fallback_level, result.certificate, result.recovering) == (0, None, False)


def test_explicit_switching_puts_a_violated_velocity_limit_on_its_boundary():
    filt = ready(ExplicitSwitchingFilter)
    x = np.array([5000.0, 0.0, 0.0, 9.99, 0.0, 0.0])
    u = np.array([A_MAX, 0.0, 0.0])
    assert phi(euler(x, u))[1] < 0.0
    result = filt.filter(x, u)
    assert result.intervened and result.active and result.fallback_level == 2
    assert np.all(np.abs(result.action) < A_MAX)
    after = phi(euler(x, result.action))
    assert after[1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(after >= -1e-12)


def test_explicit_switching_puts_a_violated_speed_limit_on_its_boundary():
    filt = ready(ExplicitSwitchingFilter)
    rho = 1000.0
    x = np.array([0.0, rho, 0.0, 0.0, -(0.2 + 2.15 * N * rho - 0.01), 0.0])
    u = np.array([0.0, -A_MAX, 0.0])
    assert phi(euler(x, u))[0] < 0.0
    result = filt.filter(x, u)
    assert np.all(np.abs(result.action) < A_MAX)
    assert phi(euler(x, result.action))[0] == pytest.approx(0.0, abs=1e-12)


def test_explicit_switching_puts_the_axes_on_v_max_before_the_speed_limit():
    filt = ready(ExplicitSwitchingFilter)
    nu1 = 2.15 * N
    # Predicted velocity (10.08, 3, 0) m/s with a speed cap between `lo` and `hi`: putting vx on
    # v_max first leaves the speed above the cap, which the scaling then reaches; scaling first
    # would leave vx above v_max, so the clamp would pull the speed below the cap.
    v1 = np.array([10.08, 3.0, 0.0])
    lo = 10.0 * np.linalg.norm(v1) / v1[0]
    hi = np.hypot(10.0, v1[1])
    cap = 0.5 * (lo + hi)
    # On V-bar under (a_max, 0, 0) the step adds (2 n vy + a_max, -2 n vx, 0) dt.
    vx, vy = np.linalg.solve([[1.0, 2.0 * N], [-2.0 * N, 1.0]], [v1[0] - A_MAX, v1[1]])
    rho = np.sqrt(((cap - 0.2) / nu1) ** 2 - vx**2) - vy
    x = np.array([0.0, rho, 0.0, vx, vy, 0.0])
    u = np.array([A_MAX, 0.0, 0.0])
    x1 = euler(x, u)
    np.testing.assert_allclose(x1[3:], v1, rtol=1e-12, atol=1e-12)
    assert 0.2 + nu1 * np.linalg.norm(x1[:3]) == pytest.approx(cap, rel=1e-12)
    clamped = np.array([10.0, v1[1], 0.0])
    want = u + clamped * cap / np.linalg.norm(clamped) - v1
    result = filt.filter(x, u)
    # The command is a difference of speeds near 10 m/s, which round at 2e-15 m/s.
    np.testing.assert_allclose(result.action, want, rtol=0.0, atol=1e-13)
    assert np.all(np.abs(result.action) < A_MAX)
    after = phi(euler(x, result.action))
    assert after[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(after[1:] > 0.0)


def test_explicit_optimization_meets_every_row_and_equals_the_full_projection():
    filt = ready(ExplicitOptimizationFilter)
    rng = np.random.default_rng(23)
    box = ActuatorSet(A_MAX, "box")
    states = draw_inside(filt, 25, seed=29)
    for x in states:
        rows = filt.rows(x)
        assert set(rows.names) <= set(AfrlConstraints.names)
        # A random command and, for each row, the box corner that does least for it, where a
        # row the box does not always meet binds.
        commands = [rng.uniform(-2.0 * A_MAX, 2.0 * A_MAX, 3), *(-A_MAX * np.sign(rows.a))]
        for u in commands:
            result = filt.filter(x, u)
            assert result.status == "feasible" and result.certificate is None
            assert np.all(rows.a @ result.action - rows.b >= -1e-10 * (np.abs(rows.b) + A_MAX))
            assert np.all(np.abs(result.action) <= A_MAX)
            # Rows the box meets for every command are dropped before the projection; with
            # them kept the projection is the same.
            full = project(np.clip(u, -A_MAX, A_MAX), rows, box)
            np.testing.assert_allclose(result.action, full.u, rtol=0.0, atol=1e-9 * A_MAX)


def test_explicit_optimization_rows_are_the_barrier_conditions_of_eq_9():
    filt = ready(ExplicitOptimizationFilter)
    cons = filt.constraints
    x = np.array([2000.0, -3000.0, 400.0, -1.5, 2.0, 0.3])
    rows = filt.rows(x)
    assert rows.names == AfrlConstraints.names
    grad = jacobian(cons.values, x)
    alpha = cons.alpha(phi(x))
    for u in np.eye(3) * 0.05:
        # grad phi (A x + B u) + alpha(phi) >= 0 is a . u - b >= 0.
        want = grad @ (CW_A @ x + CW_B @ u) + alpha
        np.testing.assert_allclose(rows.a @ u - rows.b, want, rtol=1e-6, atol=1e-9)


def test_speed_limit_row_is_dropped_at_rest():
    x = np.array([2000.0, -3000.0, 400.0, 0.0, 0.0, 0.0])
    rows = ready(ExplicitOptimizationFilter).rows(x)
    assert "speed_limit" not in rows.names
    implicit = ready(ImplicitOptimizationFilter).rows(x)
    assert "speed_limit@0" not in implicit.names and "speed_limit@1" in implicit.names


@pytest.mark.parametrize("cls", [ExplicitOptimizationFilter, ImplicitOptimizationFilter])
def test_optimization_filters_prove_an_unmeetable_speed_limit(cls):
    filt = ready(cls)
    # 2.6 m/s above the limit: alpha(phi1) = -1.89 m/s^2, beyond the 1/12 m/s^2 box.
    x = np.array([1000.0, 0.0, 0.0, 0.0, -5.0, 0.0])
    result = filt.filter(x, np.zeros(3))
    assert result.status == "infeasible_certified" and result.recovering
    assert isinstance(result.proof, InfeasibilityProof) and result.proof.verify()
    assert set(result.proof.names) <= set(filt.rows(x).names)
    assert result.conflicting and result.certificate is None
    assert np.all(np.abs(result.action) <= A_MAX)


def test_implicit_switching_uses_the_backup_when_a_later_backup_point_leaves():
    filt = ready(ImplicitSwitchingFilter)
    backup, cons = filt.backup, filt.constraints
    x, u = EDGE, np.zeros(3)
    # Only the last of the six horizon points leaves C_A, so any shorter check passes u.
    values = cons.values(backup.rollout(euler(x, u), 5).states)
    assert np.all(values[:5] > 0.4) and values[5, 3] < -0.4
    result = filt.filter(x, u)
    assert result.fallback_level == 2 and result.active and result.intervened
    np.testing.assert_allclose(result.action, backup.command(x), rtol=1e-15, atol=0.0)
    # The certificate states whether the backup's own step and rollout stay allowable.
    holds = cons.contains(backup.rollout(euler(x, result.action), 5).states).all()
    assert (result.certificate == "finite-horizon") == holds


def test_implicit_switching_passes_a_command_whose_horizon_stays_allowable():
    filt = ready(ImplicitSwitchingFilter)
    x = np.array([3000.0, 2000.0, -500.0, -1.0, 0.5, 0.2])
    u = np.array([0.02, -0.05, 0.01])
    result = filt.filter(x, u)
    np.testing.assert_array_equal(result.action, u)
    assert not result.intervened and result.fallback_level == 0
    assert (result.certificate, result.certificate_horizon) == ("finite-horizon", 5.0)
    assert filt.constraints.contains(filt.backup.rollout(euler(x, u), 5).states).all()


def test_implicit_optimization_is_feasible_inside_and_meets_its_rows():
    filt = ready(ImplicitOptimizationFilter)
    rng = np.random.default_rng(31)
    for x in draw_inside(filt, 30, seed=37):
        u = rng.uniform(-A_MAX, A_MAX, 3)
        result = filt.filter(x, u)
        assert result.status == "feasible" and result.recovering is False
        rows = filt.rows(x)
        scale = np.maximum(
            np.abs(rows.b) / np.maximum(np.linalg.norm(rows.a, axis=1), 1e-300), A_MAX
        )
        tol = 1e-9 * scale * np.linalg.norm(rows.a, axis=1)
        assert np.all(rows.a @ result.action - rows.b >= -tol)
        points = filt.backup.rollout(euler(x, result.action), 5).states
        holds = filt.constraints.contains(points).all()
        assert (result.certificate, result.certificate_horizon) == (
            ("finite-horizon", 5.0) if holds else (None, None)
        )


@pytest.mark.parametrize("cls", [ImplicitSwitchingFilter, ImplicitOptimizationFilter])
def test_implicit_certificate_and_recovery_follow_the_backup_rollouts(cls):
    filt = ready(cls)
    backup, cons = filt.backup, filt.constraints
    # All three states lie in C_A. EDGE's own rollout leaves at its last point; full thrust
    # away from -v_max on z keeps the horizon after one step inside, no thrust does not.
    results = []
    for x, u in [
        (np.array([3000.0, 2000.0, -500.0, -1.0, 0.5, 0.2]), np.array([0.02, -0.05, 0.01])),
        (EDGE, np.zeros(3)),
        (EDGE, np.array([0.0, 0.0, A_MAX])),
    ]:
        assert cons.contains(x)
        result = filt.filter(x, u)
        assert result.recovering == (not cons.contains(backup.rollout(x, 5).states).all())
        holds = cons.contains(backup.rollout(euler(x, result.action), 5).states).all()
        assert (result.certificate == "finite-horizon") == holds
        results.append((result.recovering, result.certificate))
    # Each flag takes both values, and the last decision is certified while recovering.
    assert results == [(False, "finite-horizon"), (True, None), (True, "finite-horizon")]


@pytest.mark.parametrize("cls", FILTERS)
def test_action_modes_give_the_same_acceleration(cls):
    x = np.array([5000.0, 0.0, 0.0, 9.99, 0.0, 0.0])
    u = np.array([A_MAX, -0.02, 0.01])
    accel = ready(cls).filter(x, u).action
    thrust = ready(cls, action_mode="thrust").filter(x, u * 12.0).action / 12.0
    normalized = ready(cls, action_mode="normalized").filter(x, u / A_MAX).action * A_MAX
    np.testing.assert_allclose(thrust, accel, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(normalized, accel, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("cls", FILTERS)
def test_declarations_make_no_certified_claim(cls):
    filt = ready(cls)
    dec = filt.declaration
    assert dec.counts_toward_certification is False and dec.certificate is None
    assert (dec.design_dt, dec.layout, dec.buffer_m) == (1.0, "box", 0.0)
    assert dec.a_max == pytest.approx(A_MAX, rel=1e-15)
    assert dec.name == filt.name and isinstance(dec.barriers[0], AfrlConstraints)
    assert (dec.backup is None) == (cls is ExplicitOptimizationFilter)


@pytest.mark.parametrize("cls", FILTERS)
@pytest.mark.parametrize(
    "change",
    [
        {"thrust_limit": "ball"},
        {"action_mode": "disc27"},
        {"action_mode": "impulse_coast"},
        {"observation": ("x", "y", "z", "vx", "vy", "vz", "pending_ax")},
        {
            "constraints": (
                {"type": "keep_out", "radius": 1.0},
                {"type": "speed_limit", "v0": 0.2, "v1": 2.15 * N},
            )
        },
        {"scenario_id": "dunlap2021-euler:default:twobody"},
        {"constraints": ()},
        {"constraints": ({"type": "speed_limit", "v0": 0.2, "v1": 4.0 * N},)},
        {"n": 0.0011},
        {"dt": 2.0},
        {"max_accel": 0.1},
    ],
)
def test_reset_refuses_what_the_filters_do_not_model(cls, change):
    filt = cls(setup=CODE)
    with pytest.raises(ValueError):
        filt.reset(dataclasses.replace(CODE.view(), **change))
    with pytest.raises(RuntimeError):
        _ = filt.declaration


@pytest.mark.parametrize("cls", FILTERS)
def test_reset_accepts_a_cw_dynamics_part(cls):
    ready(cls, scenario_id="dunlap2021-euler:default:cw")


@pytest.mark.parametrize("cls", FILTERS)
def test_audit_refuses_every_rta_filter(cls):
    class AfrlViewEnv(ApproachEnv):
        def agent_view(self):
            return CODE.view()

    env = AfrlViewEnv(ApproachConfig(action_mode="accel", dt=1.0))
    with pytest.raises(ValueError, match="certification"):
        audit_filter(cls(setup=CODE), env, starts=1)


def test_filters_need_a_reset_and_a_setup():
    with pytest.raises(TypeError):
        ExplicitSwitchingFilter(setup=CODE.view())
    with pytest.raises(RuntimeError):
        ExplicitSwitchingFilter(setup=CODE).filter(np.zeros(6), np.zeros(3))
    with pytest.raises(ValueError):
        ready(ExplicitSwitchingFilter).filter(np.zeros(6), np.array([np.nan, 0.0, 0.0]))


def test_implicit_optimization_refuses_a_horizon_beyond_the_row_limit():
    with pytest.raises(ValueError):
        ImplicitOptimizationFilter(setup=dataclasses.replace(CODE, horizon_s=8.0))


def integral_lqr_gain(leak):
    """K_1 (N per m and per m/s) of the LQR on x and z' = x - leak z, Q = 1e-5 I12, R = 1e7 I3."""
    a = np.block([[CW_A, np.zeros((6, 6))], [np.eye(6), -leak * np.eye(6)]])
    b = np.vstack([CW_B / 12.0, np.zeros((6, 3))])
    p = solve_continuous_are(a, b, 1e-5 * np.eye(12), 1e7 * np.eye(3))
    return (b.T @ p / 1e7)[:, :6]


def test_backup_gain_is_the_state_block_of_the_integral_lqr_in_acceleration_units():
    backup = NmtBackup(CODE)
    k1 = integral_lqr_gain(1e-6)
    np.testing.assert_allclose(backup.gain * 12.0, k1, rtol=1e-12, atol=1e-12 * np.abs(k1).max())
    assert np.all(np.diag(k1[:, 3:]) > 0.1)
    # The leak only makes the Riccati equation solvable: down to 1e-8 1/s it moves K_1 by under
    # 2e-4 of its largest entry.
    for leak in (1e-7, 1e-8):
        assert np.abs(integral_lqr_gain(leak) - k1).max() < 2e-4 * np.abs(k1).max()
    # Every mode decays faster than 3 n; the six-state LQR of the same weights leaves one at
    # 0.04 n.
    closed = np.linalg.eigvals(CW_A - CW_B @ backup.gain)
    assert np.all(closed.real < -3.0 * N)


def test_target_jacobians_match_central_differences_after_the_target_moves():
    backup = NmtBackup(CODE)
    for x, moves in ((NEAR, False), (latched(backup), True)):
        roll = backup.rollout(x, 5)
        assert (not np.array_equal(roll.targets[1], roll.targets[0])) == moves
        assert np.any(roll.target_jacobians[0] != 0.0)
        fd = jacobian(lambda s: backup.rollout(s, 5).targets, x)
        for j in range(6):
            assert_close_columns(roll.target_jacobians[j], fd[j])


def test_start_state_has_the_papers_range_and_speed_in_independent_uniform_directions():
    starts = np.array([start_state(CODE, seed) for seed in range(2000)])
    np.testing.assert_allclose(np.linalg.norm(starts[:, :3], axis=1), 9850.0, rtol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(starts[:, 3:], axis=1), 0.866, rtol=1e-14)
    seed = 2**127 + 5
    np.testing.assert_array_equal(start_state(CODE, seed), start_state(CODE, seed))
    # Bits of the seed above the 64th change the start.
    assert not np.array_equal(start_state(CODE, seed), start_state(CODE, seed - 2**64))
    r_hat = starts[:, :3] / 9850.0
    v_hat = starts[:, 3:] / 0.866
    # Each component of a uniform direction, and the dot product of two independent ones, is
    # uniform on [-1, 1] (Archimedes).
    for sample in (*r_hat.T, *v_hat.T, np.sum(r_hat * v_hat, axis=1)):
        assert kstest(sample, "uniform", args=(-1.0, 2.0)).pvalue > 1e-4
    # So each component exceeds 0.9 in size with chance 0.1, and at most one does: 4000
    # directions give 1200 with SD 29. Normalized draws from a cube give about 736.
    hits = np.sum(np.abs(r_hat) > 0.9) + np.sum(np.abs(v_hat) > 0.9)
    assert abs(hits - 1200) < 5 * 29
    for seed in (-1, 2**128, True):
        with pytest.raises(ValueError):
            start_state(CODE, seed)


def test_reproduce_steps_the_euler_plant_with_the_filtered_commands():
    short = dataclasses.replace(CODE, stop_time=20.0)
    # 1 km out on V-bar at 2.3 m/s inbound, 0.11 m/s inside a speed limit that shrinks with the
    # range: the filter passes the first command and replaces every later one.
    x0 = np.array([0.0, 1000.0, 0.0, 0.0, -2.3, 0.0])
    run = reproduce(ExplicitSwitchingFilter(setup=short), short, x0)
    assert run.states.shape == (21, 6) and run.applied.shape == (20, 3)
    np.testing.assert_array_equal(run.states[0], x0)
    assert not run.backup[0] and run.backup[1:].all()
    # The paper's primary: LQR of Q = 0.05 I6 and R = 1000 I3 on the force, clipped to 1 N.
    b_f = CW_B / 12.0
    p = solve_continuous_are(CW_A, b_f, 0.05 * np.eye(6), 1000.0 * np.eye(3))
    gain = b_f.T @ p / 1000.0 / 12.0
    # Explicit switching has no memory, so a second filter decides the same at each sample.
    twin = ready(ExplicitSwitchingFilter, setup=short)
    for k in range(20):
        x = run.states[k]
        np.testing.assert_allclose(
            run.primary[k], np.clip(-gain @ x, -A_MAX, A_MAX), rtol=1e-9, atol=1e-15
        )
        np.testing.assert_array_equal(run.applied[k], twin.filter(x, run.primary[k]).action)
        np.testing.assert_array_equal(run.applied[k], run.results[k].action)
        np.testing.assert_allclose(run.states[k + 1], euler(x, run.applied[k]), rtol=1e-15)
        assert run.backup[k] == (run.results[k].fallback_level == 2)
    assert run.filter_s > 0.0
    assert run.filter_s == pytest.approx(sum(r.compute_s for r in run.results), rel=1e-12)


def test_reproduce_stops_at_the_first_state_within_the_stop_radius_and_at_the_stop_time():
    on_radius = np.array([0.0, 1.0, 0.0, 0.0, -0.1, 0.0])
    assert reproduce(PassThroughFilter(), CODE, on_radius).applied.shape == (0, 3)
    # 5 m out at 0.5 m/s inbound the primary crosses 1 m within a few steps.
    run = reproduce(PassThroughFilter(), CODE, np.array([0.0, 5.0, 0.0, 0.0, -0.5, 0.0]))
    ranges = np.linalg.norm(run.states[:, :3], axis=1)
    assert len(run.results) == len(ranges) - 1
    assert ranges[-1] <= 1.0 and np.all(ranges[:-1] > 1.0)
    short = dataclasses.replace(CODE, stop_time=12.0)
    assert len(reproduce(PassThroughFilter(), short, start_state(short, 3)).results) == 12
    for x0 in (np.zeros(5), np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])):
        with pytest.raises(ValueError):
            reproduce(PassThroughFilter(), CODE, x0)
