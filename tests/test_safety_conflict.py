"""The keep-out and approach-cone rows, each feasible in the thrust box, in conflict together.

The interface these tests assume, beyond `project` and `InfeasibilityProof` as they are:

- `CbfFilter` enforces "approach_cone" constraints with one
  `ConeEcbf(half_angle, axis, apex, active_range, poles=poles)` per cone, its row named
  "approach_cone", and reports an "infeasible_certified" step with `conflicting` as `project`
  names it and the `InfeasibilityProof` of those rows.

The state was found by a search over positions within 12 m of the apex outside the cone and
velocities up to 0.4 m/s, for a margin of at least 5% of a_max on each row alone and a joint
shortfall above 5e-4 m/s^2; it is rounded to 0.1 m and 0.1 m/s.
"""

import dataclasses
import math

import numpy as np
import pytest
from scipy.optimize import linprog
from test_safety_cbf import A_MAX, cbf, cw_accel, view

from vbar.constraints import ApproachCone
from vbar.safety import InfeasibilityProof
from vbar.safety.barriers import ConeEcbf, KeepOutEcbf, SpeedCbf
from vbar.scenarios.base import public_spec
from vbar.solvers.qp import BOX_FACES, ActuatorSet, Rows, project

R = 50.0
P = (0.05, 0.05)
GAMMA = 0.02
# A speed cap well above the chaser's 0.283 m/s, so its row is far from binding.
V_CAP = 0.5
CONE = ApproachCone(
    half_angle=math.radians(10.0), axis=(0.0, -1.0, 0.0), apex=(0.0, -50.0, 0.0), active_range=50.0
)
# 12 m beside the apex at the keep-out's edge (range 51.906 m), outside the cone (h = -11.33 m),
# closing on the target at 0.241 m/s.
X = np.array([12.0, -50.5, 0.0, -0.2, 0.2, 0.0])
BOX = ActuatorSet(A_MAX, "box")
NAMES = ("keep_out", "speed_cap", "approach_cone")


def filter_for_conflict():
    filt = cbf(p=P[0], v_cap=V_CAP, gamma=GAMMA)
    base = view()
    filt.reset(dataclasses.replace(base, constraints=(*base.constraints, public_spec(CONE))))
    return filt


def rows_at(x):
    """The filter's rows at `x`, built from the barriers with the filter's keep-out buffer."""
    plain = cbf(p=P[0], v_cap=V_CAP, gamma=GAMMA)
    plain.reset(view())
    keep_out = KeepOutEcbf(radius=R + plain.declaration.buffer_m, poles=P)
    speed = SpeedCbf(v0=V_CAP, v1=0.0, gamma=GAMMA)
    cone = ConeEcbf(
        half_angle=CONE.half_angle,
        axis=CONE.axis,
        apex=CONE.apex,
        active_range=CONE.active_range,
        poles=P,
    )
    a_cw = cw_accel(x)
    found = [barrier.row(x, a_cw) for barrier in (keep_out, speed, cone)]
    return Rows(np.array([a for a, _ in found]), np.array([b for _, b in found]), NAMES)


def test_conflict_state_is_outside_the_cone_within_its_active_range_and_closing():
    d = X[:3] - np.array(CONE.apex)
    assert np.linalg.norm(d) <= CONE.active_range
    assert float(CONE.margin(X)) < 0.0
    assert np.linalg.norm(X[:3]) > R
    assert X[:3] @ X[3:] < 0.0


@pytest.mark.parametrize("name", NAMES)
def test_each_barrier_row_alone_is_feasible_in_the_box(name):
    rows = rows_at(X)
    k = NAMES.index(name)
    alone = Rows(rows.a[k][None], rows.b[k : k + 1], (name,))
    assert project(np.zeros(3), alone, BOX).status == "feasible"
    # Each row is met with at least 0.0048 m/s^2 to spare.
    assert A_MAX * np.sum(np.abs(rows.a[k])) - rows.b[k] > 4e-3


def test_keep_out_and_cone_rows_exceed_the_box_support_once_combined():
    # Weights (1, lam) on the keep-out and cone rows with lam = rhat_x / -g_x = 0.2350 cancel
    # the x components: the combined normal is (0, -1.198, 0), whose box support is
    # 0.02396 m/s^2 against a combined bound of 0.02669 m/s^2.
    rows = rows_at(X)
    a_k, b_k = rows.a[0], rows.b[0]
    a_c, b_c = rows.a[2], rows.b[2]
    lam = a_k[0] / -a_c[0]
    assert lam == pytest.approx(0.2350, abs=1e-4)
    w = a_k + lam * a_c
    support = A_MAX * float(np.sum(np.abs(w)))
    assert support == pytest.approx(0.02396, abs=1e-5)
    assert b_k + lam * b_c == pytest.approx(0.02669, abs=1e-5)
    assert b_k + lam * b_c - support > 2e-3


def test_keep_out_cone_and_speed_rows_together_are_certified_infeasible():
    found = project(np.zeros(3), rows_at(X), BOX)
    assert found.status == "infeasible_certified"
    # The least total violation in the box, by an independent linear program.
    rows = rows_at(X)
    lp = linprog(
        np.concatenate([np.zeros(3), np.ones(3)]),
        A_ub=np.hstack([-rows.a, -np.eye(3)]),
        b_ub=-rows.b,
        bounds=[(-A_MAX, A_MAX)] * 3 + [(0.0, None)] * 3,
        method="highs",
    )
    assert lp.status == 0 and lp.fun > 2e-3
    assert found.phase1_margin == pytest.approx(lp.fun, rel=1e-6)


def test_conflict_names_the_keep_out_and_the_cone_and_not_the_slack_speed_row():
    found = project(np.zeros(3), rows_at(X), BOX)
    assert set(found.conflicting) - set(BOX_FACES) == {"keep_out", "approach_cone"}


def conflict_proof(found, rows):
    lam = np.asarray(found.certificate, dtype=float)[: len(rows.b)]
    keep = np.array([name in found.conflicting for name in rows.names])
    return InfeasibilityProof(
        normals=rows.a[keep],
        bounds=rows.b[keep],
        multipliers=lam[keep],
        layout="box",
        a_max=A_MAX,
        names=tuple(np.array(rows.names, dtype=object)[keep]),
    )


def test_proof_on_the_named_rows_verifies():
    rows = rows_at(X)
    proof = conflict_proof(project(np.zeros(3), rows, BOX), rows)
    assert proof.names == ("keep_out", "approach_cone")
    assert proof.verify() is True
    lam = np.asarray(proof.multipliers)
    w = np.asarray(proof.normals).T @ lam
    assert np.asarray(proof.bounds) @ lam > A_MAX * np.sum(np.abs(w))


def test_cbf_filter_at_the_conflict_names_the_same_rows_while_recovering():
    filt = filter_for_conflict()
    result = filt.filter(X.copy(), np.zeros(3))
    assert result.status == "infeasible_certified"
    assert result.recovering is True
    assert set(result.conflicting) - set(BOX_FACES) == {"keep_out", "approach_cone"}
    proof = result.proof
    assert proof is not None and proof.verify() is True
    assert set(proof.names) == {"keep_out", "approach_cone"}
    rows = rows_at(X)
    for name, a, b in zip(proof.names, proof.normals, proof.bounds, strict=True):
        k = NAMES.index(name)
        np.testing.assert_allclose(a, rows.a[k], rtol=1e-9, atol=1e-15)
        assert b == pytest.approx(rows.b[k], rel=1e-9)
