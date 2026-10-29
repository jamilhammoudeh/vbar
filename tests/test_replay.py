"""Replay export to Rerun: what a .rrd file holds when it is read back."""

import logging
import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("rerun")

from metric_cases import ConstantAgent, episode  # noqa: E402
from replay_read import Recording  # noqa: E402

from vbar.agents import PDAgent  # noqa: E402
from vbar.agents.base import command_fraction  # noqa: E402
from vbar.metrics import evaluate, record_episode
from vbar.metrics.efficiency import efficiency_metrics
from vbar.metrics.filter import filter_metrics  # noqa: E402
from vbar.replay import export_rrd  # noqa: E402
from vbar.safety import FilteredAgent, FilterResult  # noqa: E402
from vbar.scenarios import make  # noqa: E402

# The PD baseline crosses the 50 m keep-out sphere flying through the target to the hold point.
BLOCKED = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)
CLEAR = (-150.0, -400.0, 0.0, 0.0, 0.0, 0.0)
# Times are stored as whole nanoseconds.
NS = 1e-9


@pytest.fixture(scope="module")
def blocked(tmp_path_factory):
    ep = episode(PDAgent(), start_state=BLOCKED, max_steps=80)
    path = export_rrd(ep, tmp_path_factory.mktemp("blocked") / "ep.rrd")
    return ep, Recording(path), evaluate(ep)


def merged(spans, tol=1e-9):
    """Intervals joined where they touch, written apart from the exporter."""
    out = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1] + tol:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def snapshot(ep):
    """Every array and violation interval of an episode as bytes and tuples."""
    arrays = [ep.start, ep.first_obs]
    for rec in ep.records:
        arrays += [rec.true_state, rec.obs, rec.command, rec.filtered, rec.segment.states]
        arrays += [a for _, a in rec.actuation.segments]
    spans = [sorted(rec.segment.violations.items()) for rec in ep.records]
    return [a.tobytes() for a in arrays], spans, ep.terminated, ep.truncated


def recorded_violations(ep, name):
    return merged(s for rec in ep.records for s in rec.segment.violations.get(name, ()))


class HalvingFilter:
    """Halves every command and says it did not intervene."""

    name = "halving"

    def __init__(self, status="feasible", report=False):
        self.status, self.report = status, report

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(0.5 * action, self.report, self.status)


class CloseFilter:
    """Stops all thrust inside 200 m of the target."""

    name = "close"

    def reset(self, view):
        pass

    def filter(self, obs, action):
        if np.linalg.norm(obs[:3]) < 200.0:
            return FilterResult(np.zeros(3), True)
        return FilterResult(action, False)


def test_positions_and_velocities_read_back_exactly(blocked):
    ep, rec, _ = blocked
    assert np.array_equal(rec.column("hill/chaser", "position_m"), ep.states[:, :3])
    assert np.array_equal(rec.column("hill/chaser", "velocity_mps"), ep.states[:, 3:])


def test_times_are_the_record_times_to_a_nanosecond(blocked):
    ep, rec, _ = blocked
    assert np.abs(rec.times("hill/chaser") - ep.times).max() <= NS
    steps = [row["step"] for row in rec.timed["hill/chaser"]]
    assert steps == list(range(len(ep.records) + 1))


def test_drawn_positions_are_the_record_to_single_precision(blocked):
    ep, rec, _ = blocked
    drawn = rec.column("hill/chaser", "Points3D:positions")
    assert np.allclose(drawn, ep.states[:, :3], rtol=1e-6, atol=1e-6)
    assert not np.array_equal(drawn, ep.states[:, :3])


def test_trail_grows_by_one_recorded_state_per_step(blocked):
    ep, rec, _ = blocked
    strips = rec.timed["hill/chaser/trail"]
    assert [len(np.asarray(row["LineStrips3D:strips"][0])) for row in strips] == list(
        range(2, len(ep.records) + 2)
    )
    last = np.asarray(strips[-1]["LineStrips3D:strips"][0])
    assert np.allclose(last, ep.states[:, :3], rtol=1e-6, atol=1e-6)


def test_hill_axes_are_x_right_y_forward_z_up(blocked):
    _, rec, _ = blocked
    # Rerun's ViewDir codes: 3 right, 5 forward, 1 up.
    assert rec.static["hill"]["ViewCoordinates:xyz"] == [[3, 5, 1]]


def test_keep_out_sphere_goal_and_target_are_stored_in_metres(blocked):
    ep, rec, _ = blocked
    keep_out = rec.static["hill/constraints/keep_out"]
    assert keep_out["radius_m"] == [ep.constraint("keep_out").radius]
    assert keep_out["Ellipsoids3D:half_sizes"] == [[50.0] * 3]
    assert keep_out["Ellipsoids3D:centers"] == [[0.0, 0.0, 0.0]]
    assert rec.static["hill/goal/hold_point"]["position_m"] == [list(ep.view.goal["hold_point"])]
    assert rec.static["hill/target"]["Points3D:positions"] == [[0.0, 0.0, 0.0]]


def test_violation_events_land_at_the_recorded_times(blocked):
    ep, rec, _ = blocked
    spans = recorded_violations(ep, "keep_out")
    # The predicate must be true somewhere in this episode, or the test observes nothing.
    assert spans
    rows = rec.timed["events/violation/keep_out"]
    starts = np.array([row["start_s"][0] for row in rows])
    ends = np.array([row["end_s"][0] for row in rows])
    assert np.array_equal(starts, [lo for lo, _ in spans])
    assert np.array_equal(ends, [hi for _, hi in spans])
    assert np.abs(rec.times("events/violation/keep_out") - starts).max() <= NS
    assert all(row["TextLog:level"] == ["ERROR"] for row in rows)


def test_a_run_that_never_enters_the_keep_out_zone_has_no_violation_events(tmp_path):
    ep = episode(PDAgent(), start_state=CLEAR, max_steps=60)
    assert not recorded_violations(ep, "keep_out")
    rec = Recording(export_rrd(ep, tmp_path / "clear.rrd", card={}))
    assert not any(path.startswith("events/violation") for path in rec.timed)


def test_violation_time_agrees_with_the_cards_violation_duration(blocked):
    ep, rec, card = blocked
    rows = rec.timed["events/violation/keep_out"]
    total = sum(row["end_s"][0] - row["start_s"][0] for row in rows)
    assert total == pytest.approx(card["violation_duration"]["keep_out"], rel=1e-6)


def test_a_violation_spanning_two_steps_is_one_event(blocked):
    ep, rec, _ = blocked
    per_step = sum(len(r.segment.violations.get("keep_out", ())) for r in ep.records)
    assert per_step > len(rec.timed["events/violation/keep_out"])


def test_commanded_and_flown_accelerations_read_back_exactly(blocked):
    ep, rec, _ = blocked
    view = ep.view
    expected = np.array([command_fraction(view, r.command) * view.max_accel for r in ep.records])
    assert np.array_equal(rec.column("hill/chaser/command", "accel_mps2"), expected)
    dv = np.array([sum(d * a for d, a in r.actuation.segments) for r in ep.records])
    assert np.array_equal(rec.column("hill/chaser/flown", "dv_mps"), dv)
    span = np.array([sum(d for d, _ in r.actuation.segments) for r in ep.records])
    assert np.array_equal(rec.column("hill/chaser/flown", "accel_mps2"), dv / span[:, None])
    # The delta-v vector has the norm of the record's delta-v.
    assert np.allclose(
        np.linalg.norm(dv, axis=1), [r.actuation.dv_l2 for r in ep.records], rtol=1e-12
    )


@pytest.mark.parametrize("mode", ["accel", "thrust"])
def test_the_first_commanded_acceleration_does_not_depend_on_the_action_mode(mode, tmp_path):
    # Later commands differ in thrust mode: the mass has fallen, so the same force flies a
    # different trajectory.
    normalized = episode(PDAgent(), start_state=BLOCKED, max_steps=2)
    other = episode(PDAgent(), start_state=BLOCKED, max_steps=2, action_mode=mode)
    a = Recording(export_rrd(normalized, tmp_path / "a.rrd", card={}))
    b = Recording(export_rrd(other, tmp_path / "b.rrd", card={}))
    assert np.allclose(
        a.column("hill/chaser/command", "accel_mps2"),
        b.column("hill/chaser/command", "accel_mps2"),
        rtol=1e-12,
        atol=1e-15,
    )


def test_arrow_length_scales_the_drawing_and_not_the_stored_acceleration(tmp_path):
    ep = episode(PDAgent(), start_state=BLOCKED, max_steps=10)
    short = Recording(export_rrd(ep, tmp_path / "s.rrd", card={}, arrow_m=10.0))
    long = Recording(export_rrd(ep, tmp_path / "l.rrd", card={}, arrow_m=40.0))
    assert np.array_equal(
        short.column("hill/chaser/command", "accel_mps2"),
        long.column("hill/chaser/command", "accel_mps2"),
    )
    drawn_short = short.column("hill/chaser/command", "Arrows3D:vectors")
    drawn_long = long.column("hill/chaser/command", "Arrows3D:vectors")
    assert np.allclose(drawn_long, 4.0 * drawn_short, rtol=1e-6, atol=1e-6)
    expected = [command_fraction(ep.view, r.command) * 10.0 for r in ep.records]
    assert np.allclose(drawn_short, expected, rtol=1e-6, atol=1e-6)


def test_interventions_are_events_at_the_decisions_that_changed_the_command(tmp_path):
    ep = episode(FilteredAgent(PDAgent(), CloseFilter()), start_state=BLOCKED, max_steps=60)
    changed = [k for k, r in enumerate(ep.records) if np.any(r.filtered != r.command)]
    assert changed and len(changed) < len(ep.records)
    rec = Recording(export_rrd(ep, tmp_path / "f.rrd", card={}))
    rows = rec.timed["events/filter/intervention"]
    assert [row["step"] for row in rows] == changed
    assert np.abs(rec.times("events/filter/intervention") - ep.times[changed]).max() <= NS
    assert np.array_equal(
        rec.column("hill/chaser/filtered", "accel_mps2")[changed],
        [command_fraction(ep.view, ep.records[k].filtered) * ep.view.max_accel for k in changed],
    )


def test_a_filter_that_denies_intervening_still_gets_its_events(tmp_path):
    # The record shows the command was halved, whatever the filter reports.
    ep = episode(
        FilteredAgent(PDAgent(), HalvingFilter(report=False)), start_state=BLOCKED, max_steps=12
    )
    rec = Recording(export_rrd(ep, tmp_path / "h.rrd", card={}))
    moved = [k for k, r in enumerate(ep.records) if np.any(r.command != 0.0)]
    assert [row["step"] for row in rec.timed["events/filter/intervention"]] == moved


def test_filter_statuses_other_than_feasible_are_error_events(tmp_path):
    agent = FilteredAgent(PDAgent(), HalvingFilter(status="solver_failed"))
    ep = episode(agent, start_state=BLOCKED, max_steps=6)
    rec = Recording(export_rrd(ep, tmp_path / "s.rrd", card={}))
    rows = rec.timed["events/filter/status"]
    assert len(rows) == 6 and all(row["TextLog:level"] == ["ERROR"] for row in rows)
    assert all(row["status"] == ["solver_failed"] for row in rows)


def test_an_unfiltered_run_has_no_filter_entities(blocked):
    _, rec, _ = blocked
    assert not any(path.startswith("events/filter") for path in rec.timed)
    assert "hill/chaser/filtered" not in rec.timed


def test_the_corridor_is_drawn_on_the_cone_the_scenario_declares(tmp_path):
    env = make("docking:axis", max_steps=4)
    ep = record_episode(env, ConstantAgent([0.0, 0.0, 0.0]), seed=3, agent_seed=4)
    cone = next(c for c in ep.constraints if type(c).__name__ == "ApproachCone")
    rec = Recording(export_rrd(ep, tmp_path / "d.rrd", card={}))
    stored = rec.static[f"hill/constraints/{cone.name}"]
    assert np.array_equal(np.asarray(stored["apex_m"]).ravel(), cone.apex)
    assert np.asarray(stored["half_angle_rad"]).ravel()[0] == cone.half_angle
    strips = [np.asarray(s) for s in stored["LineStrips3D:strips"]]
    apex, axis = np.asarray(cone.apex), np.asarray(cone.axis) / np.linalg.norm(cone.axis)
    points = np.concatenate(strips) - apex
    points = points[np.linalg.norm(points, axis=1) > 1e-9]
    angle = np.arccos(np.clip(points @ axis / np.linalg.norm(points, axis=1), -1.0, 1.0))
    # Every vertex lies on the cone's surface; the points are single precision.
    assert np.allclose(angle, cone.half_angle, atol=1e-4)
    port = np.asarray(rec.static["hill/goal/port"]["position_m"]).ravel()
    assert np.array_equal(port, ep.view.goal["port"])


def test_the_card_series_end_at_the_values_of_the_full_card(blocked):
    ep, rec, card = blocked
    ends = ep.times[1:]
    for name in (
        "delta_v",
        "delta_v_axes",
        "propellant_mass",
        "burn_count",
        "duty_cycle",
        "chatter",
    ):
        series = rec.column(f"card/{name}", "Scalars:scalars")[:, 0]
        assert len(series) == len(ep.records)
        assert series[-1] == card[name], name
        assert np.abs(rec.times(f"card/{name}") - ends).max() <= NS
    assert np.all(np.diff(rec.column("card/delta_v", "Scalars:scalars")[:, 0]) >= 0.0)
    low = rec.column("card/min_margin/keep_out", "Scalars:scalars")[:, 0]
    assert low[-1] == card["min_margin"]["keep_out"] and np.all(np.diff(low) <= 0.0)
    depth = rec.column("card/violation_depth/keep_out", "Scalars:scalars")[:, 0]
    assert depth[-1] == card["violation_depth"]["keep_out"]
    duration = rec.column("card/violation_duration/keep_out", "Scalars:scalars")[:, 0]
    assert duration[-1] == pytest.approx(card["violation_duration"]["keep_out"], rel=1e-6)
    assert np.all(np.diff(duration) >= 0.0)


def test_running_filter_metrics_end_at_the_full_cards_values(tmp_path):
    ep = episode(FilteredAgent(PDAgent(), CloseFilter()), start_state=BLOCKED, max_steps=40)
    card = evaluate(ep)
    rec = Recording(export_rrd(ep, tmp_path / "f.rrd", card=card))
    for name in ("intervention_rate", "intervention_size"):
        series = rec.column(f"card/{name}", "Scalars:scalars")[:, 0]
        assert series[-1] == card[name], name


def test_running_series_equal_the_metrics_of_every_prefix_of_the_record(tmp_path):
    ep = episode(FilteredAgent(PDAgent(), CloseFilter()), start_state=BLOCKED, max_steps=40)
    rec = Recording(export_rrd(ep, tmp_path / "p.rrd", card={}))
    for k in (1, 2, 7, 23, len(ep.records)):
        prefix = replace(ep, records=ep.records[:k], terminated=False, truncated=False)
        metrics = {**efficiency_metrics(prefix), **filter_metrics(prefix)}
        for name in ("delta_v", "delta_v_axes", "propellant_mass", "burn_count", "duty_cycle"):
            assert rec.column(f"card/{name}", "Scalars:scalars")[k - 1, 0] == metrics[name], (
                name,
                k,
            )
        assert rec.column("card/chatter", "Scalars:scalars")[k - 1, 0] == metrics["chatter"]
        assert (
            rec.column("card/intervention_rate", "Scalars:scalars")[k - 1, 0]
            == metrics["intervention_rate"]
        )


def test_final_card_values_are_stored_with_their_units(blocked):
    ep, rec, card = blocked
    end = ep.times[-1]
    assert rec.column("card/final/delta_v", "Scalars:scalars")[0, 0] == card["delta_v"]
    assert rec.times("card/final/delta_v")[0] == pytest.approx(end, abs=NS)
    assert (
        rec.column("card/final/min_margin/keep_out", "Scalars:scalars")[0, 0]
        == card["min_margin"]["keep_out"]
    )
    assert rec.static["card/delta_v"]["SeriesLines:names"] == ["E1 delta_v (m/s)"]
    # NaN entries and labels are left out of the scalars.
    assert "card/final/optimality_gap" not in rec.timed
    assert "card/final/failure_label" not in rec.timed
    assert rec.timed["events/outcome"][0]["TextLog:text"] == [
        f"not successful: {card['failure_label']}"
    ]


def test_range_and_speed_series_are_read_from_the_states(blocked):
    ep, rec, _ = blocked
    assert np.array_equal(
        rec.column("state/range_m", "Scalars:scalars")[:, 0],
        np.linalg.norm(ep.states[:, :3], axis=1),
    )
    assert np.array_equal(
        rec.column("state/speed_mps", "Scalars:scalars")[:, 0],
        np.linalg.norm(ep.states[:, 3:], axis=1),
    )


def test_the_file_holds_no_seed(tmp_path):
    seed = 987654321987654321
    ep = episode(PDAgent(), seed=seed, agent_seed=seed + 1, start_state=BLOCKED, max_steps=3)
    rec = Recording(export_rrd(ep, tmp_path / "s.rrd", card={}))
    text = rec.static["meta/episode"]["TextDocument:text"][0]
    assert ep.view.scenario_id in text and str(seed) not in text and "seed" not in text
    assert str(seed).encode() not in (tmp_path / "s.rrd").read_bytes()


def test_export_changes_neither_the_episode_nor_its_metrics(tmp_path):
    ep = episode(PDAgent(), start_state=BLOCKED, max_steps=30)
    before = snapshot(ep), repr(evaluate(ep))
    export_rrd(ep, tmp_path / "x.rrd")
    assert (snapshot(ep), repr(evaluate(ep))) == before


def test_export_leaves_the_environments_next_episode_unchanged(tmp_path):
    first = episode(PDAgent(), start_state=BLOCKED, max_steps=20)
    export_rrd(first, tmp_path / "x.rrd", card={})
    second = episode(PDAgent(), start_state=BLOCKED, max_steps=20)
    assert np.array_equal(first.states, second.states)


def test_the_file_is_written_where_asked_and_replaced(tmp_path):
    ep = episode(PDAgent(), start_state=BLOCKED, max_steps=5)
    target = tmp_path / "out.rrd"
    target.write_bytes(b"stale")
    assert export_rrd(ep, str(target), card={}) == target
    assert Recording(target).timed["hill/chaser"]


def test_export_logs_and_does_not_print(tmp_path, caplog, capsys):
    ep = episode(PDAgent(), start_state=BLOCKED, max_steps=5)
    with caplog.at_level(logging.INFO, logger="vbar.replay"):
        export_rrd(ep, tmp_path / "x.rrd", card={})
    assert "wrote 5 steps" in caplog.text
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("arrow", [0.0, -1.0, math.nan, math.inf])
def test_a_bad_arrow_length_is_refused_before_any_file_is_written(arrow, tmp_path):
    ep = episode(PDAgent(), start_state=BLOCKED, max_steps=3)
    with pytest.raises(ValueError, match="arrow_m"):
        export_rrd(ep, tmp_path / "x.rrd", arrow_m=arrow)
    assert not (tmp_path / "x.rrd").exists()


def test_an_episode_without_steps_is_refused(tmp_path):
    ep = episode(PDAgent(), start_state=BLOCKED, max_steps=3)
    empty = type(ep)(**{**ep.__dict__, "records": ()})
    with pytest.raises(ValueError, match="no steps"):
        export_rrd(empty, tmp_path / "x.rrd")
