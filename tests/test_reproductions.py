"""Published results rebuilt from their papers: Ariba et al. (2016), case 1.

The match table in docs/validation.md is the output of `python -m vbar.reproductions`; the test
that compares them builds every row and is behind the `slow` marker.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from vbar.agents.glideslope_lp import lp_glideslope
from vbar.agents.planning import Impulse
from vbar.dynamics import cw_stm
from vbar.reproductions import (
    ARIBA_CASE_ONE,
    ariba_case_one,
    excursion,
    fitted_rate_ratio,
    match_table,
    matches,
)

DOCS = Path(__file__).parent.parent / "docs" / "validation.md"
CASE = ARIBA_CASE_ONE


def fly(n, start, plan):
    """States just before each impulse and the final state, flying the plan on exact CW arcs."""
    state = np.concatenate([np.asarray(start, dtype=float), np.zeros(3)])
    before = []
    for k, impulse in enumerate(plan):
        before.append(state.copy())
        state = state + np.concatenate([np.zeros(3), impulse.dv])
        if k + 1 < len(plan):
            state = cw_stm(n, plan[k + 1].t - impulse.t) @ state
    return np.array(before), state


def test_case_one_axes_map_from_the_paper_to_the_hill_frame():
    # The paper's x is along-track, z points to the Earth and y completes the frame, so
    # V-BAR's [x, y, z] is [-z_p, x_p, -y_p].
    def hill(paper):
        x_p, y_p, z_p = paper
        return (-z_p, x_p, -y_p)

    assert hill((-500.0, 0.0, -20.0)) == CASE.start
    assert hill((-100.0, 0.0, -20.0)) == CASE.aim


def test_lp_plan_flies_the_line_and_stops_at_the_aim():
    plan = lp_glideslope(CASE.n, CASE.start, CASE.aim, CASE.duration, 10, 1.0)
    assert len(plan) == 11
    np.testing.assert_allclose([p.t for p in plan], np.linspace(0.0, CASE.duration, 11))
    before, final = fly(CASE.n, CASE.start, plan)
    # Every impulse is taken on the V-bar line 20 m above the target; the flight ends at rest.
    np.testing.assert_allclose(before[:, 0], 20.0, atol=1e-9)
    np.testing.assert_allclose(final, [*CASE.aim, 0.0, 0.0, 0.0], atol=1e-9)
    for k in range(10):
        mid = cw_stm(CASE.n, 27.0) @ (before[k] + np.concatenate([np.zeros(3), plan[k].dv]))
        assert abs(mid[0] - 20.0) <= 1.0 + 1e-9


def test_lp_case_one_cost_is_the_published_value():
    plan = lp_glideslope(CASE.n, CASE.start, CASE.aim, CASE.duration, 10, 1.0)
    cost = sum(float(np.sum(np.abs(p.dv))) for p in plan)
    # Ariba et al. (2016), sec. 5: 2.31 m/s for N = 10 and a 1 m bound.
    assert cost == pytest.approx(2.31, abs=0.005)


def test_the_published_cost_belongs_to_the_rounded_orbital_rate():
    # The paper rounds the 400 km rate to 0.001 rad/s; at the exact rate the cost is 2.43 m/s.
    from vbar.dynamics import R_EARTH, mean_motion

    n = mean_motion(R_EARTH + 400e3)
    plan = lp_glideslope(n, CASE.start, CASE.aim, CASE.duration, 10, 1.0)
    assert sum(float(np.sum(np.abs(p.dv))) for p in plan) == pytest.approx(2.43, abs=0.005)


def test_the_midpoint_bound_is_active_near_its_threshold():
    # At N = 10 the free optimum bows 0.5615 m off the line at mid-arc: a bound 0.1 mm under that
    # cannot be met, and one 0.1 mm over it is met by the flown arcs, sampled finely.
    with pytest.raises(ValueError, match="bound"):
        lp_glideslope(CASE.n, CASE.start, CASE.aim, CASE.duration, 10, 0.5614)
    plan = lp_glideslope(CASE.n, CASE.start, CASE.aim, CASE.duration, 10, 0.5616)
    assert excursion(CASE.n, np.asarray(CASE.start), plan, samples=2000) <= 0.5616 + 1e-9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": (20.0, -500.0, 1.0)},
        {"aim": (21.0, -100.0, 0.0)},
        {"start": (20.0, -500.0)},
        {"start": (math.nan, -500.0, 0.0)},
        {"n": 0.0},
        {"duration": -1.0},
        {"bound": math.inf},
        {"pulses": 0},
        {"pulses": 2.0},
        {"pulses": True},
        {"n": True},
    ],
)
def test_lp_glideslope_refuses_bad_settings(kwargs):
    settings = {
        "n": CASE.n,
        "start": CASE.start,
        "aim": CASE.aim,
        "duration": CASE.duration,
        "pulses": 10,
        "bound": 1.0,
    } | kwargs
    with pytest.raises(ValueError):
        lp_glideslope(**settings)


def test_arcs_of_an_orbit_or_longer_are_refused_and_shorter_ones_hold_the_bound():
    # Two arcs of 1.01 orbits each: the midpoint no longer marks the highest point of a hump.
    orbit = 2.0 * math.pi / CASE.n
    with pytest.raises(ValueError, match="orbit"):
        lp_glideslope(CASE.n, CASE.start, CASE.aim, 2.02 * orbit, 2, 200.0)
    plan = lp_glideslope(CASE.n, CASE.start, CASE.aim, 1.98 * orbit, 2, 200.0)
    assert excursion(CASE.n, np.asarray(CASE.start), plan, samples=4000) <= 200.0 + 1e-6


def test_an_impossible_bound_is_reported():
    # One arc from rest 400 m along the V-bar bows far more than a millimetre off the line.
    with pytest.raises(ValueError, match="bound"):
        lp_glideslope(CASE.n, CASE.start, CASE.aim, CASE.duration, 1, 1e-3)


PLAN = (Impulse(0.0, np.array([0.0, 0.1, 0.0])), Impulse(60.0, np.array([0.0, -0.1, 0.0])))


@pytest.mark.parametrize(
    "args",
    [
        (math.nan, CASE.start, PLAN),
        (-1e-3, CASE.start, PLAN),
        (True, CASE.start, PLAN),
        (1e-3, (math.nan, 0.0, 0.0), PLAN),
        (1e-3, (1.0, 2.0), PLAN),
        (1e-3, CASE.start, ()),
        (1e-3, CASE.start, (PLAN[1], PLAN[0])),
        (1e-3, CASE.start, (Impulse(0.0, np.array([math.inf, 0.0, 0.0])),)),
        (1e-3, CASE.start, PLAN, 1),
    ],
)
def test_excursion_refuses_bad_inputs(args):
    with pytest.raises(ValueError):
        excursion(*args)


def test_fitted_standard_glideslope_rates():
    # Quoted in docs/reproductions.md: the one parameter the paper leaves out.
    ratio = fitted_rate_ratio()
    initial = 400.0 * math.log(ratio) / (CASE.duration * (1.0 - ratio))
    assert ratio == pytest.approx(1.195e-3, rel=1e-3)
    assert initial == pytest.approx(-4.991, abs=1e-3)
    assert ratio * initial == pytest.approx(-0.00596, abs=1e-5)


def test_match_rule_reads_the_paper_digits():
    from vbar.reproductions import MatchRow

    assert matches(MatchRow("a", "2.31", 2.3149, "m/s", ""))
    assert not matches(MatchRow("a", "2.31", 2.3151, "m/s", ""))
    assert matches(MatchRow("a", "125", 125.4, "m", ""))
    # A trailing zero is a printed digit: 2.30 asks for 2.295 to 2.305.
    assert not matches(MatchRow("a", "2.30", 2.306, "m/s", ""))
    with pytest.raises(ValueError):
        matches(MatchRow("a", "1e-05", 1e-5, "m", ""))


@pytest.mark.slow
def test_docs_table_is_the_regenerated_table():
    rows = ariba_case_one()
    text = DOCS.read_text(encoding="utf-8")
    begin, end = "<!-- ariba-case-one begin -->\n", "<!-- ariba-case-one end -->"
    quoted = text.split(begin)[1].split(end)[0].strip()
    assert quoted == match_table(rows)
    # Which rows reproduce is part of the claim, so a change shows here too.
    missed = [row.quantity for row in rows if not matches(row)]
    assert missed == [
        "LP excursion, N = 20, 20 m bound",
        "Standard glideslope, N = 20",
        "Two-impulse transfer",
        "Two-impulse excursion",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0.0},
        {"n": math.nan},
        {"n": True},
        {"duration": 0.0},
        {"duration": math.inf},
        {"start": (20.0, math.nan, 0.0)},
        {"start": (20.0, -500.0)},
        {"aim": (20.0, -500.0, 0.0)},
    ],
)
def test_ariba_case_refuses_bad_settings(kwargs):
    from vbar.reproductions import AribaCase

    with pytest.raises(ValueError):
        AribaCase(**kwargs)
