import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.optimize import linprog
from test_safety_cbf import (
    A_MAX,
    DTS,
    FAR,
    CentrePD,
    Reckless,
    braking,
    braking_starts,
    cbf,
    cbf_flights,
    cw_accel,
    fly,
    ready,
)

from vbar.safety import FilterResult, InfeasibilityProof
from vbar.safety.barriers import KeepOutEcbf, SpeedCbf
from vbar.solvers.qp import BOX_FACES, ActuatorSet, Rows, project

R = 50.0
STATUSES = {"feasible", "infeasible_certified"}
# verify() lets a multiplier be negative by 1e-12 of the largest one, the solver's rounding
# (_PROOF_SIGN_TOL in vbar.safety), and treats it as zero.
SIGN_TOL = 1e-12


def support(normals, multipliers, layout, a_max):
    """max of (A^T lam) . u over the thrust limit."""
    w = np.asarray(normals, dtype=float).T @ np.asarray(multipliers, dtype=float)
    return a_max * float(np.sum(np.abs(w)) if layout == "box" else np.linalg.norm(w))


def farkas_holds(proof):
    """The proof's own arrays rule out every command in the thrust limit."""
    lam = np.asarray(proof.multipliers, dtype=float)
    b = np.asarray(proof.bounds, dtype=float)
    largest = float(np.max(np.abs(lam), initial=0.0))
    if not largest > 0.0 or np.any(lam < -SIGN_TOL * largest):
        return False
    lam = np.maximum(lam, 0.0)
    if not lam.sum() > 0.0:
        return False
    margin = b @ lam - support(proof.normals, lam, proof.layout, proof.a_max)
    return bool(margin > 1e-9 * (np.abs(b) @ lam))


def assert_proved(result, x, filt):
    """An infeasible step carries a valid proof over the filter's own rows at the estimate."""
    proof = result.proof
    assert proof is not None
    assert proof.verify() is True and farkas_holds(proof)
    assert proof.layout == "box" and proof.a_max == A_MAX
    assert len(proof.names) == len(proof.bounds) == len(proof.multipliers)
    keep_out = KeepOutEcbf(radius=R + filt.declaration.buffer_m, poles=(0.05, 0.05))
    references = {"keep_out": keep_out.row(x, cw_accel(x))}
    speed = SpeedCbf(v0=0.1, v1=0.0, gamma=0.02).row(x, cw_accel(x))
    if speed is not None:
        references["speed_cap"] = speed
    for name, a, b in zip(proof.names, proof.normals, proof.bounds, strict=True):
        if name in references:
            a_ref, b_ref = references[name]
            scale = np.linalg.norm(a) / np.linalg.norm(a_ref)
            np.testing.assert_allclose(a, scale * a_ref, rtol=1e-9, atol=1e-15)
            assert b == pytest.approx(scale * b_ref, rel=1e-9, abs=1e-15)


# One row under a thrust limit: the exact feasibility rule, the dual search and the support
# maximiser.


def dual_search(u_nom, w, b, layout, a_max, steps=200):
    """Least lam >= 0 with w . P(u_nom + lam w) >= b by bisection, for stacked cases (k, 3).

    w . P(u_nom + lam w) is nondecreasing in lam because the projection P onto a convex set is
    monotone, so its least feasible lam gives the projection onto the row and the limit.
    """

    def onto(u):
        if layout == "box":
            return np.clip(u, -a_max, a_max)
        size = np.linalg.norm(u, axis=1, keepdims=True)
        return np.where(size <= a_max, u, a_max * u / np.maximum(size, a_max))

    def meets(lam):
        return np.sum(w * onto(u_nom + lam[:, None] * w), axis=1) >= b

    lo = np.zeros(len(b))
    hi = np.where(meets(lo), 0.0, 1.0)
    while not np.all(meets(hi)):
        hi = np.where(meets(hi), hi, 2.0 * hi)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        ok = meets(mid)
        hi, lo = np.where(ok, mid, hi), np.where(ok, lo, mid)
    return onto(u_nom + hi[:, None] * w)


@pytest.mark.parametrize("layout, norm", [("box", 1), ("ball", 2)])
def test_one_row_is_infeasible_exactly_beyond_the_support_and_answers_its_maximiser(layout, norm):
    rng = np.random.default_rng(7)
    k = 300
    w = rng.normal(size=(k, 3))
    reach = A_MAX * np.linalg.norm(w, ord=norm, axis=1)
    b = rng.uniform(-1.5, 1.5, size=k) * reach
    u_nom = rng.uniform(-2 * A_MAX, 2 * A_MAX, size=(k, 3))
    clear = np.abs(reach - b) > 1e-9 * reach
    assert clear.sum() > 295
    w, reach, b, u_nom = w[clear], reach[clear], b[clear], u_nom[clear]
    feasible = reach >= b
    expected = dual_search(u_nom[feasible], w[feasible], b[feasible], layout, A_MAX)
    maximiser = A_MAX * (np.sign(w) if layout == "box" else w / np.linalg.norm(w, axis=1)[:, None])
    actuator = ActuatorSet(A_MAX, layout)
    answers = iter(expected)
    for wi, bi, ui, ok, best in zip(w, b, u_nom, feasible, maximiser, strict=True):
        found = project(ui, Rows(wi[None], np.array([bi]), ("row",)), actuator)
        if ok:
            assert found.status == "feasible"
            np.testing.assert_allclose(found.u, next(answers), rtol=0, atol=1e-9 * A_MAX)
        else:
            assert found.status == "infeasible_certified"
            np.testing.assert_allclose(found.u, best, rtol=0, atol=1e-9 * A_MAX)


# Proofs.


def row_proof(normals, bounds, multipliers, layout="box"):
    names = tuple(f"row{i}" for i in range(len(bounds)))
    return InfeasibilityProof(
        normals=np.array(normals, dtype=float),
        bounds=np.array(bounds, dtype=float),
        multipliers=np.array(multipliers, dtype=float),
        layout=layout,
        a_max=A_MAX,
        names=names,
    )


def test_a_row_beyond_the_box_support_is_a_valid_proof():
    assert row_proof([[1.0, 0.0, 0.0]], [0.03], [1.0]).verify() is True


def test_a_wrong_sign_multiplier_is_not_a_proof():
    assert row_proof([[1.0, 0.0, 0.0]], [0.03], [-1.0]).verify() is False


def test_a_bound_inside_the_support_is_not_a_proof():
    assert row_proof([[1.0, 0.0, 0.0]], [0.01], [1.0]).verify() is False


def test_the_support_is_the_layouts_own():
    # Normal (1, 1, 0): the box reaches 0.04 m/s^2 along it, the ball 0.0283 m/s^2.
    assert row_proof([[1.0, 1.0, 0.0]], [0.03], [1.0], "box").verify() is False
    assert row_proof([[1.0, 1.0, 0.0]], [0.03], [1.0], "ball").verify() is True


def test_opposite_rows_need_both_multipliers():
    normals, bounds = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [0.01, 0.001]
    assert row_proof(normals, bounds, [1.0, 1.0]).verify() is True
    assert row_proof(normals, bounds, [1.0, 0.0]).verify() is False


@settings(max_examples=60, deadline=None)
@given(
    normals=st.lists(
        st.tuples(*[st.floats(-1.0, 1.0, allow_subnormal=False)] * 3), min_size=1, max_size=4
    ),
    data=st.data(),
    layout=st.sampled_from(["box", "ball"]),
)
def test_verify_agrees_with_an_independent_farkas_check(normals, data, layout):
    k = len(normals)
    bounds = data.draw(
        st.lists(st.floats(-0.1, 0.1, allow_subnormal=False), min_size=k, max_size=k)
    )
    lam = data.draw(st.lists(st.floats(-1.0, 1.0, allow_subnormal=False), min_size=k, max_size=k))
    # With unit-sized normals the stated margins of both checks lie below 1e-9 m/s^2.
    assume(all(np.linalg.norm(a) >= 0.1 for a in normals))
    proof = row_proof(normals, bounds, lam, layout)
    clipped = np.maximum(lam, 0.0)
    margin = np.array(bounds) @ clipped - support(normals, clipped, layout, A_MAX)
    if abs(margin) > 1e-9 or min(lam) < -SIGN_TOL * max(np.abs(lam)):
        assert proof.verify() is farkas_holds(proof)


def test_verify_accepts_rounding_sized_negative_multipliers_and_no_more():
    normals, bounds = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [0.03, 0.0]
    assert row_proof(normals, bounds, [1.0, -SIGN_TOL]).verify() is True
    assert row_proof(normals, bounds, [1.0, -10 * SIGN_TOL]).verify() is False


def test_a_proof_belongs_only_to_a_certified_infeasible_result():
    proof = row_proof([[1.0, 0.0, 0.0]], [0.03], [1.0])
    with pytest.raises(ValueError):
        FilterResult(np.zeros(3), False, status="feasible", proof=proof)
    with pytest.raises(ValueError):
        FilterResult(np.zeros(3), False, status="solver_failed", proof=proof)
    result = FilterResult(np.ones(3), True, status="infeasible_certified", proof=proof)
    assert result.proof is proof
    assert FilterResult(np.zeros(3), False).proof is None


# The CBF filter outside its declared set.

OUTSIDE = {
    # 5 m inside the keep-out, closing at 0.3 m/s.
    "inside-closing": np.array([0.0, 45.0, 0.0, 0.0, -0.3, 0.0]),
    # Far out, three times over the 0.1 m/s speed cap, heading in.
    "over-the-cap": np.array([0.0, 200.0, 0.0, 0.0, -0.3, 0.0]),
    # 10 m inside the keep-out moving out at 0.15 m/s: the rows pull opposite ways.
    "inside-opening": np.array([0.0, 40.0, 0.0, 0.0, 0.15, 0.0]),
}


@pytest.mark.parametrize("start", OUTSIDE)
@pytest.mark.parametrize("nominal", [CentrePD, Reckless])
def test_cbf_filter_outside_its_set_reports_feasible_or_certified_with_a_proof(start, nominal):
    flight = fly(cbf(), nominal(), OUTSIDE[start], 10.0, 20)
    filt = ready(cbf())
    assert flight.results[0].recovering is True
    for x, r in zip(flight.seen, flight.results, strict=True):
        assert r.status in STATUSES
        if r.status == "infeasible_certified":
            assert_proved(r, x, filt)
            # Box faces and the ball are the thrust limit, carried by the support, not rows.
            rows = set(r.conflicting) - {*BOX_FACES, "ball"}
            assert rows and rows <= set(r.proof.names)
        else:
            assert r.proof is None


def test_cbf_filter_certifies_the_step_inside_the_keep_out_closing():
    filt = ready(cbf())
    x = OUTSIDE["inside-closing"]
    result = filt.filter(x.copy(), np.zeros(3))
    assert result.status == "infeasible_certified"
    assert_proved(result, x, filt)


def test_two_rows_that_conflict_in_the_box_are_both_named():
    filt = ready(cbf())
    x = OUTSIDE["inside-opening"]
    keep_out = KeepOutEcbf(radius=R + filt.declaration.buffer_m, poles=(0.05, 0.05))
    speed = SpeedCbf(v0=0.1, v1=0.0, gamma=0.02)
    rows = [keep_out.row(x, cw_accel(x)), speed.row(x, cw_accel(x))]
    box = ActuatorSet(A_MAX, "box")
    # Each row alone can be met in the box; together they cannot.
    for a, b in rows:
        assert project(np.zeros(3), Rows(a[None], np.array([b]), ("r",)), box).status == "feasible"
    both = Rows(np.array([a for a, _ in rows]), np.array([b for _, b in rows]), ("k", "s"))
    assert project(np.zeros(3), both, box).status == "infeasible_certified"
    result = filt.filter(x.copy(), np.zeros(3))
    assert result.status == "infeasible_certified"
    assert {"keep_out", "speed_cap"} <= set(result.conflicting)
    assert {"keep_out", "speed_cap"} <= set(result.proof.names)
    assert_proved(result, x, filt)


def test_least_violation_action_minimises_the_sum_of_violations_inside_the_box():
    filt = ready(cbf())
    x = OUTSIDE["inside-opening"]
    command = np.array([0.5, -1.0, 0.25])
    result = filt.filter(x.copy(), command)
    assert result.status == "infeasible_certified"
    action = np.asarray(result.action)
    assert np.all(np.abs(action) <= 1.0)
    proof = result.proof
    a, b = np.asarray(proof.normals), np.asarray(proof.bounds)
    rows = Rows(a, b, tuple(proof.names))
    u = A_MAX * action
    # The phase I point of the projection on the same rows.
    expected = project(A_MAX * np.clip(command, -1.0, 1.0), rows, ActuatorSet(A_MAX, "box")).u
    np.testing.assert_allclose(u, expected, rtol=0, atol=1e-9 * A_MAX)
    # Independently: no command in the box violates the rows less in total.
    k = len(b)
    lp = linprog(
        np.concatenate([np.zeros(3), np.ones(k)]),
        A_ub=np.hstack([-a, -np.eye(k)]),
        b_ub=-b,
        bounds=[(-A_MAX, A_MAX)] * 3 + [(0.0, None)] * k,
        method="highs",
    )
    assert lp.status == 0
    violation = np.maximum(0.0, b - a @ u).sum()
    assert violation == pytest.approx(lp.fun, rel=1e-9, abs=1e-12)


# Inside the declared set there is still nothing to prove.


@pytest.mark.parametrize("dt", DTS)
def test_cbf_filter_from_its_declared_set_has_no_infeasible_steps(dt):
    for flight in cbf_flights(dt):
        assert all(r.status == "feasible" and r.proof is None for r in flight.results)


@pytest.mark.parametrize("dt", DTS)
def test_braking_backup_filter_from_its_set_has_no_infeasible_steps(dt):
    for x0 in braking_starts(ready(braking(), dt).contains, 2):
        flight = fly(braking(), CentrePD(), x0, dt, 30)
        assert all(r.status == "feasible" and r.proof is None for r in flight.results)


def test_a_far_start_gives_a_feasible_step_without_a_proof():
    result = ready(cbf()).filter(FAR.copy(), np.zeros(3))
    assert result.status == "feasible" and result.proof is None
