"""Approach-cone rows in the CBF filter and the feasibility of the cone row near its apex.

The interface these tests assume:

- `ConeEcbf(half_angle, axis, apex, active_range, poles)` accepts active_range = inf, the
  `ApproachCone` default, and then gives its row at every distance from the apex but zero.
- `CbfFilter` enforces "approach_cone" constraints. Reset builds one
  `ConeEcbf(half_angle, axis, apex, active_range, poles=poles)` per cone of the view, from the
  fields `public_spec` gives, and lists each in `declaration.barriers` in the order of the view.
  The rows passed to `vbar.solvers.qp.project` include each cone's `ConeEcbf.row(x, a_cw)`
  as the barrier returns it, named "approach_cone", "approach_cone#1", ... as the speed rows are
  numbered, and left out where the row is None.
- `contains(x)` also requires the cone's start set, `ConeEcbf.contains`, wherever x lies within
  active_range of the apex.
- With a cone the declaration has counts_toward_certification False and its safe_set says
  "cone activation untreated", so `audit_filter` refuses the filter.
- The docstring of `ConeEcbf.infeasible_radius` no longer claims the radius for the box.
"""

import dataclasses
import math

import numpy as np
import pytest
from test_safety_cbf import A_MAX, FAR, cbf, cw_accel, view

from vbar.constraints import ApproachCone
from vbar.safety import cbf as cbf_module
from vbar.safety.audit import audit_filter
from vbar.safety.barriers import ConeEcbf
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.scenarios.base import public_spec
from vbar.solvers.qp import ActuatorSet, Rows, project

THETA = math.radians(10.0)
P = (0.05, 0.05)


def cone_spec(axis, apex, half_angle=THETA, active_range=math.inf):
    cone = ApproachCone(half_angle=half_angle, axis=axis, apex=apex, active_range=active_range)
    return public_spec(cone)


def cone_view(*cones, dt=10.0):
    base = view(dt)
    return dataclasses.replace(base, constraints=(*base.constraints, *cones))


def ready_with(*cones, **kw):
    filt = cbf(**kw)
    filt.reset(cone_view(*cones))
    return filt


def first_rows(monkeypatch, filt, x, action):
    """The rows of the filter's first projection at `x`, and its result."""
    seen = []
    real = cbf_module.project

    def spy(u_nom, rows, *args, **kwargs):
        seen.append(rows)
        return real(u_nom, rows, *args, **kwargs)

    monkeypatch.setattr(cbf_module, "project", spy)
    result = filt.filter(x.copy(), np.asarray(action, dtype=float))
    assert seen
    return seen[0], result


def named(rows, name):
    k = rows.names.index(name)
    return rows.a[k], float(rows.b[k])


# A cone along +y with its apex 300 m out, clear of the keep-out.
AXIS = (0.0, 1.0, 0.0)
APEX = (0.0, 300.0, 0.0)


def test_cone_barrier_accepts_an_unbounded_active_range():
    barrier = ConeEcbf(half_angle=THETA, axis=AXIS, apex=APEX, active_range=math.inf, poles=P)
    x = np.array([500.0, 1e4, 0.0, 0.0, 0.0, 0.0])
    assert barrier.row(x, cw_accel(x)) is not None


def test_cbf_declaration_lists_one_cone_barrier_per_cone_constraint():
    wide = cone_spec(AXIS, APEX, half_angle=math.radians(20.0), active_range=150.0)
    filt = ready_with(cone_spec(AXIS, APEX), wide, p=P[0])
    cones = [b for b in filt.declaration.barriers if isinstance(b, ConeEcbf)]
    assert cones == [
        ConeEcbf(half_angle=THETA, axis=AXIS, apex=APEX, active_range=math.inf, poles=P),
        ConeEcbf(half_angle=math.radians(20.0), axis=AXIS, apex=APEX, active_range=150.0, poles=P),
    ]


def test_a_cone_constraint_takes_the_filter_out_of_certification():
    filt = ready_with(cone_spec(AXIS, APEX))
    assert filt.declaration.counts_toward_certification is False
    assert "cone activation untreated" in filt.declaration.safe_set
    plain = cbf()
    plain.reset(view())
    assert plain.declaration.counts_toward_certification is True


class ConeApproachEnv(ApproachEnv):
    """The approach scenario with a 10 degree cone about the V-bar from the hold point."""

    def _constraints(self):
        cone = ApproachCone(half_angle=THETA, axis=(0.0, -1.0, 0.0), apex=(0.0, -50.0, 0.0))
        return (*super()._constraints(), cone)


def test_audit_refuses_the_cbf_filter_on_a_scenario_with_a_cone():
    env = ConeApproachEnv(ApproachConfig(dt=10.0, max_steps=2))
    assert any(spec["type"] == "approach_cone" for spec in env.agent_view().constraints)
    with pytest.raises(ValueError, match="certification") as refused:
        audit_filter(cbf(), env, starts=1)
    assert "cone activation untreated" in str(refused.value)
    assert "soft filter" not in str(refused.value)


def test_cbf_cone_row_on_the_surface_has_gradient_sin_theta_and_the_curvature_term(monkeypatch):
    # Our cone case after Ames et al. 2019, sec. IV, Def. 7: theta 10 deg, 5 m from the apex,
    # circumferential speed 0.03 m/s.
    filt = ready_with(cone_spec(AXIS, APEX), p=P[0])
    e = math.cos(THETA) * np.array(AXIS) + math.sin(THETA) * np.array([1.0, 0.0, 0.0])
    x = np.concatenate([np.array(APEX) + 5.0 * e, [0.0, 0.0, 0.03]])
    rows, _ = first_rows(monkeypatch, filt, x, np.zeros(3))
    a, b = named(rows, "approach_cone")
    assert np.linalg.norm(a) == pytest.approx(0.173648, rel=1e-5)
    # With h = h' = 0 the row is -(h'' at zero thrust); the CW part of h'' is a . a_cw.
    assert -b - a @ cw_accel(x) == pytest.approx(-1.77265e-4, rel=1e-5)


def inside_moving_out():
    """100 m from the apex at 9 deg off the axis, moving outward across it at 0.08 m/s."""
    off = math.radians(9.0)
    d = 100.0 * np.array([math.sin(off), math.cos(off), 0.0])
    return np.concatenate([np.array(APEX) + d, [0.08, 0.0, 0.0]])


def test_cbf_filter_command_meets_an_active_cone_row(monkeypatch):
    filt = ready_with(cone_spec(AXIS, APEX), p=P[0])
    x = inside_moving_out()
    reference = ConeEcbf(half_angle=THETA, axis=AXIS, apex=APEX, active_range=math.inf, poles=P)
    a, b = reference.row(x, cw_accel(x))
    # The zero command breaks the row: b is about 5.9e-4 m/s^2.
    assert b > 1e-4
    rows, result = first_rows(monkeypatch, filt, x, np.zeros(3))
    np.testing.assert_allclose(named(rows, "approach_cone")[0], a, rtol=1e-12, atol=1e-15)
    assert named(rows, "approach_cone")[1] == pytest.approx(b, rel=1e-12)
    assert result.status == "feasible" and result.intervened is True
    assert a @ (A_MAX * np.asarray(result.action)) >= b - 1e-9 * A_MAX


def test_cone_row_is_left_out_beyond_the_active_range(monkeypatch):
    # FAR is 100 m from the apex and 90 deg off the axis, far outside the cone.
    filt = ready_with(cone_spec((1.0, 0.0, 0.0), APEX, active_range=50.0))
    rows, result = first_rows(monkeypatch, filt, FAR, np.zeros(3))
    assert "approach_cone" not in rows.names
    assert result.intervened is False and result.recovering is False


def test_rows_of_several_cones_are_numbered(monkeypatch):
    wide = cone_spec(AXIS, APEX, half_angle=math.radians(20.0))
    filt = ready_with(cone_spec(AXIS, APEX), wide)
    rows, _ = first_rows(monkeypatch, filt, inside_moving_out(), np.zeros(3))
    assert {"approach_cone", "approach_cone#1"} <= set(rows.names)
    assert sum(name.startswith("approach_cone") for name in rows.names) == 2


def test_contains_requires_the_cone_start_set_within_the_active_range():
    plain = cbf()
    plain.reset(view())
    assert plain.contains(FAR)
    across = (1.0, 0.0, 0.0)
    assert not ready_with(cone_spec(across, APEX, active_range=150.0)).contains(FAR)
    assert ready_with(cone_spec(across, APEX, active_range=50.0)).contains(FAR)
    assert ready_with(cone_spec(AXIS, APEX)).contains(FAR)


# The cone row near the apex under the two thrust limits, with a_cw = 0 as in the surface case
# above. On the surface with h = h' = 0 the row is g . u >= cos(theta) v^2 / rho with
# |g|_2 = sin(theta); the ball offers a_max |g|_2 and the box a_max |g|_1, up to sqrt(3) times more.


def best_box_case(rho, v_lat=0.03):
    """A cone whose surface gradient at `rho` from the apex lies along (1, 1, 1)."""
    w = np.ones(3) / math.sqrt(3.0)
    q = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    axis = math.sin(THETA) * w + math.cos(THETA) * q
    barrier = ConeEcbf(half_angle=THETA, axis=axis, apex=(0.0, 0.0, 0.0), active_range=1.0, poles=P)
    # The point rho q lies on the surface and its gradient is sin(theta) w; v is normal to both.
    x = np.concatenate([rho * q, v_lat * np.cross(w, q)])
    return barrier, x


def cone_row_status(barrier, x, layout):
    a, b = barrier.row(x, np.zeros(3))
    return project(np.zeros(3), Rows(a[None], np.array([b]), ("cone",)), ActuatorSet(A_MAX, layout))


def test_best_direction_case_is_on_the_surface_with_the_gradient_along_the_diagonal():
    barrier, x = best_box_case(0.2)
    assert abs(barrier.value(x)) <= 1e-15
    g = barrier.gradient(x)
    np.testing.assert_allclose(g, math.sin(THETA) * np.ones(3) / math.sqrt(3.0), atol=1e-15)
    assert abs(g @ x[3:]) <= 1e-15


def test_at_0_2_m_a_box_command_meets_the_cone_row_and_no_ball_command_does():
    # At 0.03 m/s circumferential speed the ball cannot meet the row inside 0.255 m, the box
    # along the best direction only inside 0.147 m.
    barrier, x = best_box_case(0.2)
    box = cone_row_status(barrier, x, "box")
    assert box.status == "feasible"
    a, b = barrier.row(x, np.zeros(3))
    assert a @ box.u >= b - 1e-12
    assert cone_row_status(barrier, x, "ball").status == "infeasible_certified"
    assert barrier.infeasible_radius(0.03, A_MAX) > 0.2


def test_box_infeasible_radius_along_the_best_direction_is_0_147_m():
    rho_box = math.cos(THETA) * 0.03**2 / (A_MAX * math.sin(THETA) * math.sqrt(3.0))
    assert round(rho_box, 3) == 0.147
    assert cone_row_status(*best_box_case(1.02 * rho_box), "box").status == "feasible"
    inside = cone_row_status(*best_box_case(0.98 * rho_box), "box")
    assert inside.status == "infeasible_certified"


def test_infeasible_radius_is_not_claimed_for_the_box():
    doc = ConeEcbf.infeasible_radius.__doc__
    assert "holds for both" not in doc
