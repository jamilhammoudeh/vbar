"""Bounded-W mode of the predictive safety filter: declared navigation and execution error sets.

The interface these tests assume, beyond that of `tests/test_safety_predictive.py`:

- `PredictiveFilter(*, navigation=None, execution=0.0, grid=5.0, ...)` keeps each as an
  attribute of the same name. `navigation` is a `BoxError`, `BallError` or `EllipsoidError` on
  the observed state; `execution` (m/s^2) bounds the execution error of each held command,
  entering the sets as the box W of half-width `execution` per axis (generators Gamma da I);
  `grid` (s) spaces the samples of the terminal check. Either error set makes the filter
  bounded: the declaration has certificate "bounded-W" and tightening
  {"navigation": navigation, "execution": execution, "model_error": None}. backup="cam" with
  an error set raises ValueError at construction or reset.
- With E = `tightening.error_zonotope(navigation)` (a zonotope about 0; a box exactly) and
  t_cap = 4 pi / n, `contains(x)` is
  `drift_clearance(n, Zonotope(x, G_E), R, t_cap, ceil(t_cap / grid) + 1).certified >= 0`.
  A certified decision reports "bounded-W" with horizon t_cap; a decision at a state outside
  the set reports `recovering` True; a level 2 that fails the tightened check has no
  certificate.
- The error of the plan's state x_k lies in Z_k = Phi^k Z_0 + sum_{j<k} Phi^j Gamma W, with
  Z_0 = Phi(delay_s) E + sum_{j<delay_s/dt} Phi^j Gamma W (the delay window). Each keep-out row
  a_x . x_k + a_u . u_k >= b of the program is the nominal filter's row at the same reference,
  with b raised by h_{Z_k}(a_x) + h_W(a_u), the support of the error of that sample.

Kochenderfer et al., Algorithms for Validation, eq. 8.29 with the keep-out ball (our case,
recomputed on CW, `docs/validation.md`): V-bar hold (0, -100, 0) at rest, 500 km, R 50 m, two
orbits. The point clears by 50.000 m; the box (3 m, 3 mm/s) reaches the target; a velocity box
is certified clear below 1.4148 mm/s. These tests use a 30 s grid, not the 60 s first set for
this case: at 60 s the certified bound of the 1.38 mm/s box is -0.150 m (found +1.231 m), so it
is not contained; at 30 s it is +0.854 m and the 1.45 mm/s box gives -1.630 m.
"""

import math

import numpy as np
import pytest

from vbar.agents.base import action_scale
from vbar.dynamics import cw_discrete
from vbar.reach import Zonotope, cw_loop, drift_clearance, loop_clearance, propagate
from vbar.safety import predictive as predictive_module
from vbar.safety.tightening import BoxError, EllipsoidError, delay_margin
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.solvers.ipm import HorizonSolution
from vbar.spacecraft import PerfectNav

PredictiveFilter = predictive_module.PredictiveFilter

R = 50.0
A_MAX = 0.02
DT = 10.0
N = ApproachEnv().n
PHI, GAMMA = cw_discrete(N, DT)
TWO_ORBITS = 4.0 * math.pi / N
GRID = 30.0
SAMPLES = math.ceil(TWO_ORBITS / GRID) + 1
HOLD = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
V_BAR = np.array([0.0, 150.0, 0.0, 0.0, 0.0, 0.0])
ZERO = np.zeros(3)
# test_safety_predictive.py: X1 passes the target at 1.5 m/s, 55.2 m from its centre, and PUSH
# takes it into the keep-out, so every filter solves there; its rows all lie beyond R + b.
X1 = np.array([60.0, 78.935, 0.0, 0.0, -1.5, 0.0])
PUSH = np.array([-1.0, 0.0, 0.0])


def approach(dt=DT, steps=1, start=V_BAR, **cfg):
    start = tuple(float(s) for s in start)
    return ApproachEnv(ApproachConfig(dt=dt, max_steps=steps, start_state=start, **cfg))


def ready(filt, dt=DT, **cfg):
    filt.reset(approach(dt, **cfg).agent_view())
    return filt


def bounded(navigation=None, execution=0.0, **kw):
    return PredictiveFilter(navigation=navigation, execution=execution, grid=GRID, **kw)


def execution_box(da):
    return Zonotope.box(np.zeros(3), np.full(3, da))


def stopped(qp, *args, **kwargs):
    return HorizonSolution(None, None, "iteration_cap", 7, math.nan, 1.0, 1.0, 1.0, None)


# Declaration.


def test_error_sets_make_the_declaration_bounded_w():
    nav = BoxError(r_pos=0.1, r_vel=1e-4)
    declared = ready(bounded(nav, 1e-4)).declaration
    assert declared.certificate == "bounded-W"
    assert dict(declared.tightening) == {
        "navigation": nav,
        "execution": 1e-4,
        "model_error": None,
    }
    assert ready(PredictiveFilter()).declaration.certificate == "nominal"


@pytest.mark.parametrize(
    "kw",
    [{"execution": 1e-4}, {"navigation": BoxError(r_pos=0.1, r_vel=0.0)}],
    ids=["execution", "navigation"],
)
def test_the_cam_backup_refuses_error_sets(kw):
    with pytest.raises(ValueError):
        ready(PredictiveFilter(backup="cam", grid=GRID, **kw))


# The navigation boxes at the V-bar hold.


def test_three_metre_box_at_the_hold_is_not_contained_though_the_point_clears_by_50_m(
    monkeypatch,
):
    point = drift_clearance(N, Zonotope.point(HOLD), R, TWO_ORBITS, SAMPLES)
    assert point.found == pytest.approx(50.0, abs=1e-3)
    # A filter on the point estimate takes the hold as safe.
    assert ready(PredictiveFilter(), start=HOLD).contains(HOLD) is True
    filt = ready(bounded(BoxError(r_pos=3.0, r_vel=3e-3)), start=HOLD)
    assert filt.contains(HOLD) is False
    monkeypatch.setattr(predictive_module, "solve_horizon", stopped)
    result = filt.filter(HOLD.copy(), ZERO.copy())
    assert result.status == "feasible" and result.fallback_level == 2
    assert result.recovering is True
    assert result.certificate is None and result.certificate_horizon is None


@pytest.mark.parametrize(("r_vel", "inside"), [(1.38e-3, True), (1.45e-3, False)])
def test_velocity_box_at_the_hold_is_contained_below_1_4148_mm_per_s(r_vel, inside):
    filt = ready(bounded(BoxError(r_pos=0.0, r_vel=r_vel)), start=HOLD)
    assert filt.contains(HOLD) is inside
    if not inside:
        return
    result = filt.filter(HOLD.copy(), ZERO.copy())
    assert result.status == "feasible" and result.fallback_level == 0
    assert result.recovering is False
    assert result.certificate == "bounded-W"
    assert result.certificate_horizon == pytest.approx(TWO_ORBITS, rel=1e-12)
    np.testing.assert_array_equal(result.action, ZERO)


# Row tightening, against the nominal filter's program at the same state and reference.


def first_program(filt, obs, action, monkeypatch):
    seen = []
    solve = predictive_module.solve_horizon

    def record(qp, *args, **kwargs):
        seen.append(qp)
        return solve(qp, *args, **kwargs)

    monkeypatch.setattr(predictive_module, "solve_horizon", record)
    filt.filter(obs.copy(), action.copy())
    monkeypatch.setattr(predictive_module, "solve_horizon", solve)
    assert seen
    return seen[0]


def keep_out_rows(qp, k):
    """The rows of stage k that read the state, (rows, bounds)."""
    rows, bounds = np.asarray(qp.C[k]), np.asarray(qp.e[k])
    on_state = np.any(rows[:, :6] != 0.0, axis=1)
    return rows[on_state], bounds[on_state]


def tightening_by_row(bounded_qp, nominal_qp, k):
    """Each keep-out row of stage k with its bound less the nominal bound of the same row."""
    rows, bounds = keep_out_rows(bounded_qp, k)
    nominal_rows, nominal_bounds = keep_out_rows(nominal_qp, k)
    assert len(rows) == len(nominal_rows) >= 4
    raised = []
    for row, bound in zip(rows, bounds, strict=True):
        same = np.flatnonzero(np.all(np.isclose(nominal_rows, row, rtol=1e-12, atol=1e-15), axis=1))
        assert same.size == 1
        raised.append(bound - nominal_bounds[same[0]])
    return rows, np.array(raised)


def delayed_obs(delay_s):
    env = approach(
        DT,
        1,
        X1,
        delay_s=delay_s,
        observation_channels=("pending_commands",),
        sensor=PerfectNav(),
    )
    obs, _ = env.reset(seed=0)
    return env.agent_view(), obs


# The difference of two bounds near 50 m carries about 1e-14 m of rounding.
ATOL = 1e-12


def test_delay_window_support_equals_the_delay_margin(monkeypatch):
    da, delay_s = 1e-3, 20.0
    view, obs = delayed_obs(delay_s)
    nominal = PredictiveFilter(delay_s=delay_s)
    nominal.reset(view)
    filt = bounded(None, da, delay_s=delay_s)
    filt.reset(view)
    action = PUSH
    nominal_qp = first_program(nominal, obs, action, monkeypatch)
    bounded_qp = first_program(filt, obs, action, monkeypatch)
    w = execution_box(da)
    rows, raised = tightening_by_row(bounded_qp, nominal_qp, 0)
    for row, lift in zip(rows, raised, strict=True):
        window = delay_margin(N, DT, delay_s, w, row[:6])
        assert window > 0.0
        assert lift - w.support(row[6:]) == pytest.approx(window, rel=1e-12, abs=ATOL)


def test_row_tightening_is_the_support_of_the_propagated_error_sets(monkeypatch):
    nav, da = BoxError(r_pos=0.2, r_vel=2e-4), 1e-4
    nominal_qp = first_program(ready(PredictiveFilter()), X1, PUSH, monkeypatch)
    bounded_qp = first_program(ready(bounded(nav, da)), X1, PUSH, monkeypatch)
    w = execution_box(da)
    e = Zonotope.box(np.zeros(6), [0.2] * 3 + [2e-4] * 3)
    steps = bounded_qp.C.shape[0]
    errors = propagate(cw_loop(N, DT), e, w, steps)
    for k in range(steps):
        rows, raised = tightening_by_row(bounded_qp, nominal_qp, k)
        for row, lift in zip(rows, raised, strict=True):
            expected = errors[k].support(row[:6]) + w.support(row[6:])
            assert lift == pytest.approx(expected, rel=1e-12, abs=ATOL)


# Closed loop with errors drawn inside the declared sets.


class Reckless:
    name = "reckless"

    def reset(self, view, rng=None):
        self.scale = action_scale(view) / view.max_accel

    def act(self, obs, info):
        return -obs[:3] / np.linalg.norm(obs[:3]) * self.scale


# 201 exact samples 0.05 s apart: at 1 m/s the range dips by under 1e-5 m between them.
TAU = np.linspace(0.0, DT, 201)
PHIS, GAMMAS = cw_discrete(N, TAU)


def check_errors_in_the_declared_sets_stay_out(decisions):
    """Fly the reckless agent from V-bar with errors drawn inside the declared sets.

    Returns the lowest sampled keep-out margin of the true path (m), the number of decisions
    the filter moved and the number of thrusting steps flown with an execution error.
    """
    r_pos, r_vel, da = 0.05, 5e-5, 5e-5
    filt = ready(bounded(BoxError(r_pos=r_pos, r_vel=r_vel), da))
    agent = Reckless()
    view = approach().agent_view()
    agent.reset(view)
    rng = np.random.default_rng(1913)
    x, lowest, intervened, thrusting = V_BAR.copy(), math.inf, 0, 0
    for _ in range(decisions):
        e = rng.uniform(-1.0, 1.0, 6) * np.array([r_pos] * 3 + [r_vel] * 3)
        obs = x + e
        result = filt.filter(obs.copy(), agent.act(obs, {}))
        intervened += bool(result.intervened)
        u = np.asarray(result.action, dtype=float) * A_MAX / action_scale(view)
        if np.any(u):
            # Execution error of norm at most da, the declared bound, drawn in the ball.
            w = rng.normal(size=3)
            u = u + w / np.linalg.norm(w) * da * rng.uniform() ** (1.0 / 3.0)
            thrusting += 1
        held = PHIS @ x + GAMMAS @ u
        lowest = min(lowest, float(np.linalg.norm(held[:, :3], axis=1).min()) - R)
        x = PHI @ x + GAMMA @ u
    return lowest, intervened, thrusting


# Twenty decisions, which reach the close pass, cost about 9 s of CPU (each held step of a plan
# is a set clearance), so the test flies three: 1.5 s, three thrusting steps with both errors
# drawn, two of them moved by the filter, about 90 m out. A fourth takes the test past 2 s; the
# check function takes any number of decisions.
LOOP_DECISIONS = 3


def test_decisions_with_errors_in_the_declared_sets_stay_out_of_the_keep_out():
    lowest, intervened, thrusting = check_errors_in_the_declared_sets_stay_out(LOOP_DECISIONS)
    assert intervened >= 2 and thrusting >= 2
    assert lowest >= -1e-3


# Which error set each check uses. On R-bar, closing at 0.1 m/s, a full outward step of
# 0.02 m/s^2 turns the approach around: the range dips 0.25 m at mid-step and the step ends at
# the start range, receding at 0.1 m/s. The free drift from the start enters the keep-out, so
# each start is outside the declared set. A 10 s grid keeps the drift bound's sample loss near
# 1 cm.
FINE = 10.0
FINE_SAMPLES = math.ceil(TWO_ORBITS / FINE) + 1
OUTWARD = np.array([A_MAX, 0.0, 0.0])
LOOP = cw_loop(N, DT)


def turnaround(r0):
    return np.array([r0, 0.0, 0.0, -5.0 * A_MAX, 0.0, 0.0])


def held_clearance(x, generators, da):
    """Certified clearance of the held outward step from x + G beta under OUTWARD + W, m."""
    found = loop_clearance(
        LOOP, Zonotope(x, generators), Zonotope.box(OUTWARD, np.full(3, da)), R, 1, 20
    )
    return found.certified


def end_clearance(x1, generators):
    return drift_clearance(N, Zonotope(x1, generators), R, TWO_ORBITS, FINE_SAMPLES).certified


def fine(navigation=None, execution=0.0, **kw):
    return PredictiveFilter(navigation=navigation, execution=execution, grid=FINE, **kw)


def decide(filt, x):
    """One decision on the outward step from `x`, the filter reset for it."""
    filt.reset(approach(start=x).agent_view())
    return filt.filter(x.copy(), OUTWARD / A_MAX)


def test_a_held_step_clear_for_the_estimate_but_not_for_its_navigation_box_is_rejected():
    x = turnaround(50.65)
    e = np.diag([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])[:, :3]
    # Clear by 0.400 m for the point, short by 0.100 m for the 0.5 m box; the drift after the
    # step clears by 0.137 m for the box carried through the step.
    assert held_clearance(x, np.zeros((6, 0)), 0.0) == pytest.approx(0.400, abs=1e-3)
    assert held_clearance(x, e, 0.0) == pytest.approx(-0.100, abs=1e-3)
    assert end_clearance(PHI @ x + GAMMA @ OUTWARD, PHI @ e) == pytest.approx(0.137, abs=1e-3)
    result = decide(fine(BoxError(r_pos=0.5, r_vel=0.0)), x)
    # A rejected command costs a solve; the command passed through has none.
    assert result.iterations is not None


def test_a_held_step_clear_without_the_execution_error_but_not_with_it_is_rejected():
    x, da = turnaround(50.255), 5e-4
    # 5.2 mm clear for the exact command, 3.2 mm short with the 0.5 mm/s^2 box around it; the
    # drift after the step clears by 0.115 m with that box carried through.
    assert held_clearance(x, np.zeros((6, 0)), 0.0) == pytest.approx(0.0052, abs=2e-4)
    assert held_clearance(x, np.zeros((6, 0)), da) == pytest.approx(-0.0032, abs=2e-4)
    assert end_clearance(PHI @ x + GAMMA @ OUTWARD, da * GAMMA) == pytest.approx(0.115, abs=1e-3)
    assert decide(fine(None, da), x).iterations is not None


def test_an_end_state_whose_grown_error_set_drifts_in_is_rejected():
    x, da = turnaround(50.6), 4e-3
    x1 = PHI @ x + GAMMA @ OUTWARD
    # The held step clears by 0.261 m with W; the drift after it clears by 0.588 m for the
    # exact end state, which Z_0 is without navigation or delay, and is short by 0.430 m for
    # Z_1 = Gamma W.
    assert held_clearance(x, np.zeros((6, 0)), da) == pytest.approx(0.261, abs=1e-3)
    assert end_clearance(x1, np.zeros((6, 0))) == pytest.approx(0.588, abs=1e-3)
    assert end_clearance(x1, da * GAMMA) == pytest.approx(-0.430, abs=1e-3)
    assert decide(fine(None, da), x).iterations is not None


def test_a_certified_command_from_outside_the_declared_set_is_recovering():
    x = turnaround(50.6)
    filt = fine(None, 1e-5)
    result = decide(filt, x)
    assert filt.contains(x) is False
    assert result.iterations is None and result.fallback_level == 0
    assert result.certificate == "bounded-W"
    assert result.recovering is True


def test_an_ellipsoid_with_a_stated_exceedance_certifies_chance_epsilon():
    # The velocity ellipsoid's outer box is the 1.38 mm/s box contained at the hold.
    nav = EllipsoidError(P=np.diag([0.0] * 3 + [1.38e-3] * 3), exceedance=0.01)
    filt = ready(bounded(nav), start=HOLD)
    assert filt.declaration.certificate == "chance-epsilon"
    result = filt.filter(HOLD.copy(), ZERO.copy())
    assert result.fallback_level == 0 and result.recovering is False
    assert result.certificate == "chance-epsilon"
    assert result.certificate_horizon == pytest.approx(TWO_ORBITS, rel=1e-12)


def test_error_sets_are_refused_in_the_finite_horizon_mode():
    with pytest.raises(ValueError):
        PredictiveFilter(mode="finite-horizon", execution=1e-4)


def test_an_execution_error_with_a_delay_of_part_of_a_step_is_refused():
    channels = ("pending_commands",)
    ready(bounded(None, 1e-4, delay_s=20.0), delay_s=20.0, observation_channels=channels)
    with pytest.raises(ValueError):
        ready(bounded(None, 1e-4, delay_s=15.0), delay_s=15.0, observation_channels=channels)
