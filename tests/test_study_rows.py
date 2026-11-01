"""The supervisor study's cells and rows, and the fault cell's environment."""

import math

import pytest

from vbar.eval.study import (
    BASE_FILTER_ON,
    CORE_CELLS,
    FAULT_CELL,
    MARGINS,
    NAV_FILTER_OFF,
    ROWS,
    Cell,
    Row,
    StudyTask,
)
from vbar.scenarios import SUITES, seeds

MAIN = ROWS["main"]
SLICES = [row for row in ROWS.values() if row.name != "main"]


def test_the_core_cells_are_every_core_suite_entry_with_the_filter_on_and_off():
    entries = {entry.scenario for entry in SUITES["core-v1"].entries}
    assert len(CORE_CELLS) == 20
    assert {c.scenario for c in CORE_CELLS} == entries
    assert {(c.scenario, c.filter) for c in CORE_CELLS} == {
        (s, on) for s in entries for on in (True, False)
    }
    assert all(c.fault is None for c in CORE_CELLS)


def test_the_main_row_is_the_core_cells_and_the_braking_loss_cell_seeded_in_four_looks():
    assert MAIN.cells == (*CORE_CELLS, FAULT_CELL)
    assert FAULT_CELL == Cell("approach-vbar", False, "braking_loss")
    assert MAIN.looks == (50, 100, 200, 270) and MAIN.seeds == 270 and MAIN.sequential
    assert MAIN.cap_usd == 110.0
    assert MAIN.share_usd == pytest.approx(110.0 / 21)
    assert not MAIN.own_prompt and MAIN.wall_limit_s is None


def test_a_sequential_row_uses_a_bonferroni_interval_and_a_fixed_row_the_tost_interval():
    # Two one-sided tests at 0.05 split over 4 looks: 0.0125 per side, a 97.5% interval.
    assert MAIN.confidence == pytest.approx(0.975)
    assert all(row.confidence == pytest.approx(0.90) for row in SLICES)
    assert MARGINS == {"success": 0.05, "intervention_count": 0.5}


def test_the_ablation_rows_fly_the_navigation_cells_with_the_filter_off_on_their_own_prompts():
    ablation = [ROWS["ablation-1"], ROWS["ablation-2"]]
    for row in ablation:
        assert row.cells == NAV_FILTER_OFF and row.seeds == 30 and row.own_prompt
        assert all(c.scenario.endswith(":nav1") and not c.filter for c in row.cells)
    assert sum(row.cap_usd for row in ablation) == 8.0


def test_the_cross_model_rows_fly_the_base_cells_with_the_filter_on():
    assert {c.scenario for c in BASE_FILTER_ON} == {
        "approach-vbar",
        "approach-rbar",
        "docking:axis",
        "docking-afrl",
        "inspection",
    }
    assert all(c.filter and c.fault is None for c in BASE_FILTER_ON)
    frontier, second, probe, local = (
        ROWS["frontier"],
        ROWS["second-frontier"],
        ROWS["second-frontier-probe"],
        ROWS["local"],
    )
    assert frontier.cells == BASE_FILTER_ON and frontier.seeds == 30 and frontier.cap_usd == 40.0
    trio = ("approach-vbar", "approach-rbar", "docking:axis")
    assert tuple(c.scenario for c in second.cells) == trio
    assert second.seeds == 20 and second.cap_usd == 20.0
    assert probe.cells == second.cells and probe.seeds == 1 and probe.cap_usd == 1.0
    assert local.cells == BASE_FILTER_ON and local.seeds == 5
    assert local.cap_usd is None and local.wall_limit_s == 3600.0


def test_the_caps_split_as_decided_and_leave_the_reserve():
    capped = sum(
        r.cap_usd for r in ROWS.values() if r.cap_usd and r.name != "second-frontier-probe"
    )
    # 110 main, 8 ablation, 40 frontier, 20 second frontier; the reserve of 20 is not a row.
    assert capped == 178.0


def test_every_other_row_is_a_slice_of_the_main_row():
    for row in SLICES:
        assert set(row.cells) <= set(MAIN.cells), row.name
        assert row.seeds <= MAIN.seeds and not row.sequential
    # A slice flies the first seeds of the same validation draw.
    cell = BASE_FILTER_ON[2]
    assert (
        seeds(cell.scenario, "validation", 30).pairs()
        == seeds(cell.scenario, "validation", 270).pairs()[:30]
    )


def test_every_episode_ceiling_fits_its_cells_share():
    for row in ROWS.values():
        if row.cap_usd is not None:
            assert 0.0 < row.episode_usd <= row.share_usd, row.name


def test_the_braking_loss_cell_injects_the_intermittent_fault_into_each_environment():
    task = StudyTask(fault="braking_loss")
    first, second = task.make("approach-vbar"), task.make("approach-vbar")
    assert "braking_intermittent" in first.scenario_id
    assert "intermittent(0.93" in first.scenario_id and "rangebelow(300)" in first.scenario_id
    assert first.faults[0] is not second.faults[0]
    assert task.identity(first)["fault"] == "braking_loss"
    assert StudyTask().identity(StudyTask().make("approach-vbar"))["fault"] is None


def test_a_cell_label_names_the_scenario_the_filter_and_the_fault():
    assert Cell("docking:axis:nav1", True).label == "docking:axis:nav1 filter on"
    assert FAULT_CELL.label == "approach-vbar filter off braking_loss"


@pytest.mark.parametrize(
    "fields",
    [
        {"scenario": "", "filter": True},
        {"scenario": 3, "filter": True},
        {"scenario": "approach-vbar", "filter": 1},
        {"scenario": "approach-vbar", "filter": False, "fault": "sensor_bias"},
        {"scenario": "docking:axis", "filter": False, "fault": "braking_loss"},
    ],
)
def test_a_cell_refuses_bad_fields(fields):
    with pytest.raises(ValueError):
        Cell(**fields)


GOOD = {"name": "r", "cells": BASE_FILTER_ON, "seeds": 4}


@pytest.mark.parametrize(
    "fields",
    [
        {"name": ""},
        {"cells": ()},
        {"cells": list(BASE_FILTER_ON)},
        {"cells": (BASE_FILTER_ON[0], BASE_FILTER_ON[0])},
        {"cells": ("approach-vbar",)},
        {"seeds": 0},
        {"seeds": 2.0},
        {"seeds": True},
        {"looks": [2, 4]},
        {"looks": (2, 3)},
        {"looks": (3, 2, 4)},
        {"looks": (0, 4)},
        {"looks": (2, 2, 4)},
        {"cap_usd": 1.0},
        {"episode_usd": 0.1},
        {"cap_usd": math.nan, "episode_usd": 0.1},
        {"cap_usd": -1.0, "episode_usd": 0.1},
        {"cap_usd": math.inf, "episode_usd": 0.1},
        {"cap_usd": 1.0, "episode_usd": 0.0},
        # A share of 1/5 dollar cannot hold a 0.3 dollar episode.
        {"cap_usd": 1.0, "episode_usd": 0.3},
        {"wall_limit_s": 0.0},
        {"wall_limit_s": math.nan},
        {"own_prompt": 1},
    ],
)
def test_a_row_refuses_bad_fields(fields):
    with pytest.raises(ValueError):
        Row(**{**GOOD, **fields})


def test_a_row_accepts_an_episode_ceiling_equal_to_its_share():
    assert Row(**GOOD, cap_usd=1.0, episode_usd=0.2).share_usd == pytest.approx(0.2)
