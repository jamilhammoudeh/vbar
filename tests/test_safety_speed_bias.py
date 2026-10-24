"""The CBF filter's speed rows under a declared persistent acceleration error.

The interface these tests assume:

- `CbfFilter` lowers the v0 of every speed row, its own speed cap and each declared
  "speed_limit" and "closing_speed_limit", by `tightening.first_order_bias(da_max, gamma)`,
  da_max = execution + model_error, and lists the lowered `SpeedCbf` and `ClosingSpeedCbf`
  barriers in `declaration.barriers` after the keep-out, in the order of the view; v1 and gamma
  are unchanged. `v_cap` keeps its value and still sets the riding buffer, the safe set and the
  end-of-step speed cut. A bias that leaves a v0 at or below zero makes reset raise ValueError.

Under a bias b held along the speed, s' = -gamma s - b settles at s = -b / gamma, so the
unlowered cap rides 2e-4 / 0.03 = 6.7 mm/s over v_cap.
"""

import dataclasses

import numpy as np
import pytest
from test_safety_cbf import A_MAX, N, view

from vbar.agents.base import action_scale
from vbar.dynamics import cw_discrete, cw_states
from vbar.safety.barriers import ClosingSpeedCbf, SpeedCbf
from vbar.safety.cbf import CbfFilter
from vbar.safety.tightening import first_order_bias

DT = 10.0
GAMMA = 0.03
V_CAP = 0.2
BIAS = 2e-4
SPEED_LIMIT = {"type": "speed_limit", "v0": 0.15, "v1": 0.001}
CLOSING_LIMIT = {"type": "closing_speed_limit", "v0": 0.1, "v1": 0.002}


def limited_view():
    base = view(DT)
    return dataclasses.replace(base, constraints=(*base.constraints, SPEED_LIMIT, CLOSING_LIMIT))


def speed_barriers(filt):
    """(type, v0, v1, gamma) of each speed barrier the filter declares."""
    return [
        (type(b), b.v0, b.v1, b.gamma)
        for b in filt.declaration.barriers
        if isinstance(b, (SpeedCbf, ClosingSpeedCbf))
    ]


def assert_speed_rows(filt, expected):
    found = speed_barriers(filt)
    assert [kind for kind, *_ in found] == [kind for kind, *_ in expected]
    for (_, *got), (_, *want) in zip(found, expected, strict=True):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-15)


def test_every_speed_row_is_lowered_by_the_first_order_bias():
    filt = CbfFilter(v_cap=V_CAP, gamma=GAMMA, execution=BIAS, model_error=1e-4)
    filt.reset(limited_view())
    # (2e-4 + 1e-4) / 0.03 = 0.01 m/s.
    depth = first_order_bias(3e-4, GAMMA)
    assert depth == pytest.approx(0.01, rel=1e-12)
    assert_speed_rows(
        filt,
        [
            (SpeedCbf, V_CAP - depth, 0.0, GAMMA),
            (SpeedCbf, 0.15 - depth, 0.001, GAMMA),
            (ClosingSpeedCbf, 0.1 - depth, 0.002, GAMMA),
        ],
    )
    assert filt.v_cap == V_CAP


def test_without_declared_errors_the_speed_rows_keep_their_limits():
    filt = CbfFilter(v_cap=V_CAP, gamma=GAMMA)
    filt.reset(limited_view())
    assert_speed_rows(
        filt,
        [
            (SpeedCbf, V_CAP, 0.0, GAMMA),
            (SpeedCbf, 0.15, 0.001, GAMMA),
            (ClosingSpeedCbf, 0.1, 0.002, GAMMA),
        ],
    )


def test_a_bias_that_uses_up_a_speed_limit_is_refused():
    # 0.003 / 0.03 = 0.1 m/s, the whole closing-speed limit.
    with pytest.raises(ValueError):
        CbfFilter(v_cap=V_CAP, gamma=GAMMA, execution=3e-3).reset(limited_view())


def test_declared_set_budget_takes_off_the_execution_error():
    # Radial offsets at rest, where |a_cw| = 3 n^2 x; the speed budget is
    # 2 max(p) 1.1 v_cap = 0.0132 m/s^2, and the lowered rows ask up to da_max more.
    da = 1e-3
    budget = 2.0 * 0.03 * 1.1 * V_CAP
    plain = CbfFilter(v_cap=V_CAP, gamma=GAMMA)
    plain.reset(view(DT))
    biased = CbfFilter(v_cap=V_CAP, gamma=GAMMA, execution=da)
    biased.reset(view(DT))
    edges = {
        "plain": (plain, (A_MAX - budget) / (3.0 * N * N)),
        "biased": (biased, (A_MAX - budget - da) / (3.0 * N * N)),
    }
    for filt, edge in edges.values():
        assert filt.contains(np.array([(1.0 - 1e-6) * edge, 0.0, 0.0, 0.0, 0.0, 0.0])) is True
        assert filt.contains(np.array([(1.0 + 1e-6) * edge, 0.0, 0.0, 0.0, 0.0, 0.0])) is False
    between = np.array([0.5 * (edges["plain"][1] + edges["biased"][1]), 0.0, 0.0, 0.0, 0.0, 0.0])
    assert plain.contains(between) is True and biased.contains(between) is False


def fly_with_bias(filt, steps=60):
    """Largest speed (m/s) on dense samples of the held steps under a bias along +y.

    The chaser starts 1000 m out on the V-bar moving away at 0.15 m/s, and the agent pushes at
    full thrust along its velocity, so the keep-out row never acts and the speed rows bind.
    """
    v = view(DT)
    filt.reset(v)
    to_accel = v.max_accel / action_scale(v)
    phi, gamma = cw_discrete(N, DT)
    tau = np.linspace(0.0, DT, 11)
    d = np.array([0.0, BIAS, 0.0])
    x = np.array([0.0, 1000.0, 0.0, 0.0, 0.15, 0.0])
    fastest = 0.0
    for _ in range(steps):
        command = x[3:] / np.linalg.norm(x[3:])
        u = np.asarray(filt.filter(x.copy(), command).action) * to_accel
        held = np.tile(u + d, (len(tau), 1))
        states = cw_states(N, tau, np.tile(x, (len(tau), 1)), held)
        fastest = max(fastest, float(np.linalg.norm(states[:, 3:], axis=1).max()))
        x = phi @ x + gamma @ (u + d)
    return fastest


def test_speed_stays_under_the_cap_with_the_bias_declared():
    assert fly_with_bias(CbfFilter(v_cap=V_CAP, gamma=GAMMA, execution=BIAS)) <= V_CAP


def test_an_undeclared_bias_rides_over_the_cap():
    # The same flight without the declared error: 6.3 mm/s over at the samples, against the
    # continuous-time 6.7 mm/s.
    over = fly_with_bias(CbfFilter(v_cap=V_CAP, gamma=GAMMA)) - V_CAP
    assert 5e-3 < over < 7.5e-3
