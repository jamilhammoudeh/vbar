"""Corridor monitor with a phase-specific open-loop CAM, and the paired CAM rates.

Contract assumed of `vbar.safety.corridor.CorridorCamFilter`:
    CorridorCamFilter(phases=((rho_lo, rho_hi, a_c, m), ...)) with range bands in m, a_c the
    CAM's Hill acceleration (m/s^2) and m the whole steps it is held; `reset(view)` builds the
    default corridor `PassiveSpeed("vbar", extent=keep-out radius, n=view.n)` and clears the
    latch; `filter(obs, action)` returns a `FilterResult` whose action is in the view's units.
    It passes the command through until the first decision outside the goal's commit radius with
    a negative corridor margin, then returns a_c of the phase whose band holds rho for m decisions
    (that one included) and zero thrust for every later decision. `declaration.certificate` is
    None and `declaration.counts_toward_certification` False.

Contract assumed of `vbar.safety.audit.cam_rates(pairs)`, pairs from `record_pair`:
    returns an object with `false_cam`, `missed_cam` and `cam_induced`, each with `count`,
    `trials`, `rate` (count / trials) and `interval` (95% Clopper-Pearson, `clopper_pearson`).
    needed: the off run's S1 (lowest keep-out margin, not its bound) is below -1 mm. fired: some
    step of the on run has a filtered command different from the recorded raw command; the
    filter's own flags are never read. false_cam counts fired among not needed; missed_cam counts
    (not fired, or on-run S1 below -1 mm) among needed; cam_induced counts on runs that violate
    among not needed.
"""

import functools
import math
from dataclasses import replace

import numpy as np
import pytest

from vbar.agents.base import action_scale
from vbar.constraints import SegmentMinimum
from vbar.metrics import clopper_pearson
from vbar.metrics.episode import FilterPair, record_episode, record_pair
from vbar.safety import FilteredAgent, FilterResult, PassThroughFilter, audit
from vbar.safety.audit import audit_filter
from vbar.scenarios.approach import ApproachConfig, ApproachEnv

# Fehse's V-bar forward coast after thrust stops at speed v is kappa v / n (eq. 3.28, fig. 3.13).
KAPPA = math.sqrt(7.0) - 3.0 * math.acos(0.75)
A_MAX = ApproachConfig().max_accel
CAM = (0.0, A_MAX, 0.0)
SEED = 0x6F1D2C3B4A5968778695A4B3C2D1E0F1
AGENT_SEED = 0x1F0E2D3C4B5A69788796A5B4C3D2E1F0


class Coast:
    name = "coast"

    def reset(self, view, rng=None):
        pass

    def act(self, obs, info):
        return np.zeros(3)


class Reckless:
    """Full command straight at the target."""

    name = "reckless"

    def reset(self, view, rng=None):
        self.scale = action_scale(view)

    def act(self, obs, info):
        return -obs[:3] / np.linalg.norm(obs[:3]) * self.scale


class SilentCam:
    """Thrusts full +V-bar from its second decision on and reports no intervention."""

    name = "silent-cam"

    def reset(self, view):
        self.k = 0

    def filter(self, obs, action):
        self.k += 1
        if self.k == 1:
            return FilterResult(np.array(action, dtype=float), intervened=False)
        return FilterResult(np.array([0.0, 1.0, 0.0]), intervened=False, active=False)


class LoudPassThrough:
    """Reports an intervention on every decision and passes every command through."""

    name = "loud-pass-through"

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(np.array(action, dtype=float), intervened=True, active=True)


def corridor(phases=((0.0, math.inf, CAM, 2),), view=None):
    from vbar.safety.corridor import CorridorCamFilter

    filt = CorridorCamFilter(phases=phases)
    filt.reset(ApproachEnv().agent_view() if view is None else view)
    return filt


def vbar_state(rho, closing):
    """On the +V-bar at range rho (m), closing on the target at `closing` (m/s)."""
    return np.array([0.0, rho, 0.0, 0.0, -closing, 0.0])


def envelope(rho):
    """Fehse's V-bar passive-safety speed at range rho on approach-vbar (m/s)."""
    n = ApproachEnv().agent_view().n
    return n * (rho - ApproachConfig().keep_out_radius) / KAPPA


def acts(filt, obs, command=(0.0, 0.0, 0.0)):
    return filt.filter(np.asarray(obs, dtype=float), np.array(command, dtype=float)).action


def test_fehse_vbar_envelope_at_100_m_on_approach_vbar_is_0_116():
    # Fehse 2003, eq. 3.28 and fig. 3.13: 2.094 n (rho - 50 m) at 500 km.
    assert envelope(100.0) == pytest.approx(0.116, abs=5e-4)
    assert 1.0 / KAPPA == pytest.approx(2.094, abs=5e-4)


@pytest.mark.parametrize("side", [1.0, -1.0], ids=["plus_vbar", "minus_vbar"])
def test_corridor_margin_is_zero_at_the_vbar_envelope_at_100_m(side):
    v0 = envelope(100.0)
    away = (0.0, side * A_MAX, 0.0)
    command = (0.1, -0.2, 0.05)
    for closing, fires in ((v0 - 1e-6, False), (v0 + 1e-6, True), (0.1155, False), (0.116, True)):
        filt = corridor(phases=((0.0, math.inf, away, 2),))
        u = acts(filt, vbar_state(100.0, closing) * side, command)
        expected = np.array(away) / A_MAX if fires else np.array(command)
        np.testing.assert_allclose(u, expected, atol=1e-12)


def test_commands_inside_the_corridor_pass_through_unchanged():
    filt = corridor()
    for k, rho in enumerate((400.0, 300.0, 200.0, 120.0)):
        command = (0.3, -0.5 + 0.1 * k, 0.1)
        np.testing.assert_allclose(
            acts(filt, vbar_state(rho, 0.5 * envelope(rho)), command), command, atol=1e-12
        )


def test_abort_latch_holds_the_cam_then_zero_thrust_whatever_the_state():
    filt = corridor(phases=((0.0, math.inf, CAM, 3),))
    cam = np.array(CAM) / A_MAX
    np.testing.assert_allclose(acts(filt, vbar_state(100.0, 0.3), (0.0, -1.0, 0.0)), cam)
    # Back inside the corridor and far out, the latch keeps the abort.
    safe = vbar_state(400.0, 0.0)
    for _ in range(2):
        np.testing.assert_allclose(acts(filt, safe, (0.0, -1.0, 0.0)), cam, atol=1e-12)
    for _ in range(6):
        np.testing.assert_allclose(acts(filt, safe, (0.4, -1.0, 0.2)), np.zeros(3), atol=1e-12)


def test_reset_clears_the_abort_latch():
    filt = corridor()
    acts(filt, vbar_state(100.0, 0.3))
    filt.reset(ApproachEnv().agent_view())
    command = (0.2, -0.3, 0.0)
    np.testing.assert_allclose(acts(filt, vbar_state(300.0, 0.0), command), command, atol=1e-12)


@pytest.mark.parametrize(
    "rho, cam_y, held", [(100.0, 1.0, 2), (200.0, 0.5, 1)], ids=["inner", "outer"]
)
def test_the_cam_is_that_of_the_phase_whose_band_holds_the_range(rho, cam_y, held):
    phases = ((0.0, 150.0, CAM, 2), (150.0, math.inf, (0.0, 0.5 * A_MAX, 0.0), 1))
    filt = corridor(phases=phases)
    assert 0.5 > envelope(rho)
    outputs = [acts(filt, vbar_state(rho, 0.5), (0.0, -1.0, 0.0)) for _ in range(held + 2)]
    for u in outputs[:held]:
        np.testing.assert_allclose(u, (0.0, cam_y, 0.0), atol=1e-12)
    for u in outputs[held:]:
        np.testing.assert_allclose(u, np.zeros(3), atol=1e-12)


def test_no_abort_inside_the_commit_radius():
    base = ApproachEnv().agent_view()
    view = replace(base, goal={**base.goal, "commit_radius": 150.0})
    filt = corridor(view=view)
    command = (0.1, -0.4, 0.0)
    np.testing.assert_allclose(acts(filt, vbar_state(100.0, 0.5), command), command, atol=1e-12)
    # No latch was set inside the zone, so a corridor-safe state outside it passes through.
    np.testing.assert_allclose(acts(filt, vbar_state(300.0, 0.1), command), command, atol=1e-12)
    np.testing.assert_allclose(
        acts(filt, vbar_state(300.0, 1.0), command), np.array(CAM) / A_MAX, atol=1e-12
    )


def test_a_declared_approach_cone_bounds_the_corridor_at_its_surface():
    # A 10 deg cone about +V-bar from the target: at y = 200 m its surface is at
    # x = 200 tan(10 deg) = 35.2654 m, and 1 cm off it the signed distance is 1 cm cos(10 deg).
    half_angle = math.radians(10.0)
    surface = 200.0 * math.tan(half_angle)
    assert surface == pytest.approx(35.2654, abs=1e-4)
    base = ApproachEnv().agent_view()
    cone = {"type": "approach_cone", "half_angle": half_angle, "axis": (0.0, 1.0, 0.0)}
    view = replace(base, constraints=(*base.constraints, cone))
    at_rest = np.array([surface, 200.0, 0.0, 0.0, 0.0, 0.0])
    assert envelope(np.linalg.norm(at_rest[:3])) > 0.3
    margins = {c.name: c for c in corridor(view=view).declaration.barriers}
    offset = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert float(margins["approach_cone"].margin(at_rest)) == pytest.approx(0.0, abs=1e-9)
    assert float(margins["approach_cone"].margin(at_rest + offset)) == pytest.approx(
        -0.01 * math.cos(half_angle), abs=1e-9
    )
    command = (0.1, -0.2, 0.05)
    inside = acts(corridor(view=view), at_rest - offset, command)
    np.testing.assert_allclose(inside, command, atol=1e-12)
    outside = acts(corridor(view=view), at_rest + offset, command)
    np.testing.assert_allclose(outside, np.array(CAM) / A_MAX, atol=1e-12)


def test_a_corridor_margin_of_exactly_zero_is_inside_the_corridor():
    # At rest on the keep-out surface the V-bar envelope n (rho - R) / kappa and the closing
    # speed are both exactly 0, so the margin is exactly 0; only a negative margin aborts.
    on_surface = vbar_state(ApproachConfig().keep_out_radius, 0.0)
    (envelope_margin,) = corridor().declaration.barriers
    assert float(envelope_margin.margin(on_surface)) == 0.0
    command = (0.1, -0.2, 0.05)
    np.testing.assert_allclose(acts(corridor(), on_surface, command), command, atol=1e-12)
    closing = vbar_state(ApproachConfig().keep_out_radius, 1e-9)
    np.testing.assert_allclose(acts(corridor(), closing, command), np.array(CAM) / A_MAX)


def test_corridor_filter_declares_no_certificate():
    declaration = corridor().declaration
    assert declaration.certificate is None
    assert declaration.counts_toward_certification is False


def test_the_audit_refuses_to_certify_the_corridor_filter():
    from vbar.safety.corridor import CorridorCamFilter

    with pytest.raises((ValueError, TypeError)):
        audit_filter(
            CorridorCamFilter(phases=((0.0, math.inf, CAM, 2),)), ApproachEnv(), starts=1,
            max_steps=1, seed=SEED,
        )  # fmt: skip


def test_flown_cam_overrides_the_agent_then_holds_zero_thrust():
    from vbar.safety.corridor import CorridorCamFilter

    env = ApproachEnv(ApproachConfig(start_state=tuple(vbar_state(100.0, 0.3)), max_steps=5))
    flown = record_pair(
        env, Reckless, CorridorCamFilter(phases=((0.0, math.inf, CAM, 2),)), SEED, AGENT_SEED
    )
    records = flown.on.records
    for rec in records[:2]:
        np.testing.assert_allclose(rec.filtered, np.array(CAM) / A_MAX, atol=1e-12)
    for rec in records[2:]:
        assert np.linalg.norm(rec.command) > 0.5
        np.testing.assert_allclose(rec.filtered, np.zeros(3), atol=1e-12)


@functools.cache
def base_episode():
    env = ApproachEnv(ApproachConfig(start_state=(0.0, 300.0, 0.0, 0.0, 0.0, 0.0), max_steps=5))
    return record_episode(env, FilteredAgent(Coast(), PassThroughFilter()), SEED, AGENT_SEED)


def episode(margin=None, fired=False, reported=False, step=2):
    """The base coast run with step `step` given a keep-out minimum `margin` (m), a filtered
    command that differs from the raw one when `fired`, and the filter's report `reported`."""
    ep = base_episode()
    records = list(ep.records)
    rec = records[step]
    segment = rec.segment
    if margin is not None:
        minima = dict(segment.minima)
        t = minima["keep_out"].t
        # A certified bound below the minimum, as the environment records it.
        minima["keep_out"] = SegmentMinimum(t, margin, margin - 0.005)
        segment = replace(segment, minima=minima)
    filtered = np.array(rec.command, dtype=float)
    if fired:
        filtered = filtered + np.array(CAM) / A_MAX
    result = FilterResult(filtered.copy(), intervened=reported, active=reported)
    records[step] = replace(rec, segment=segment, filtered=filtered, filter_result=result)
    return replace(ep, records=tuple(records))


def pair(off_margin=None, fired=False, on_margin=None, reported=None):
    """A constructed pair; the filter reports the opposite of what it did unless told."""
    reported = (not fired) if reported is None else reported
    on = episode(on_margin, fired=fired, reported=reported)
    return FilterPair(on=on, off=episode(off_margin), filter_name="corridor-cam", first="on")


CLEAN = None
VIOLATED = -0.5


def check(rate, count, trials):
    assert rate.count == count
    assert rate.trials == trials
    assert rate.rate == pytest.approx(count / trials, abs=1e-15)
    assert tuple(rate.interval) == pytest.approx(clopper_pearson(count, trials), abs=1e-12)


def test_cam_rates_count_each_definition_exactly():
    pairs = [
        *[pair(CLEAN, fired=False, on_margin=CLEAN) for _ in range(3)],
        *[pair(CLEAN, fired=True, on_margin=CLEAN) for _ in range(2)],
        pair(CLEAN, fired=True, on_margin=VIOLATED),
        *[pair(VIOLATED, fired=True, on_margin=CLEAN) for _ in range(2)],
        # Not fired is missed even with a clean on run.
        pair(VIOLATED, fired=False, on_margin=CLEAN),
        # Fired too late: the on run still violates.
        pair(VIOLATED, fired=True, on_margin=VIOLATED),
    ]
    rates = audit.cam_rates(pairs)
    check(rates.false_cam, 3, 6)
    check(rates.missed_cam, 2, 4)
    check(rates.cam_induced, 1, 6)


def test_cam_rates_ignore_the_filters_own_flags():
    honest = [pair(CLEAN, fired=True, reported=True), pair(VIOLATED, fired=True, reported=True)]
    lying = [pair(CLEAN, fired=True, reported=False), pair(VIOLATED, fired=True, reported=False)]
    loud = [pair(CLEAN, fired=False, reported=True), pair(VIOLATED, fired=False, reported=True)]
    check(audit.cam_rates(honest).false_cam, 1, 1)
    check(audit.cam_rates(lying).false_cam, 1, 1)
    check(audit.cam_rates(lying).missed_cam, 0, 1)
    check(audit.cam_rates(loud).false_cam, 0, 1)
    check(audit.cam_rates(loud).missed_cam, 1, 1)


def anchors():
    """One needed pair whose CAM saved the run and one quiet pair that needed none."""
    return [pair(VIOLATED, fired=True), pair(CLEAN, fired=False)]


@pytest.mark.parametrize(
    "off_margin, needed",
    [(-0.9e-3, False), (-1.0e-3, False), (-1.1e-3, True)],
    ids=["above", "at", "below"],
)
def test_a_cam_is_needed_only_below_minus_1_mm_on_the_off_run(off_margin, needed):
    probe = pair(off_margin, fired=True, on_margin=CLEAN)
    rates = audit.cam_rates([*anchors(), probe])
    if needed:
        check(rates.false_cam, 0, 1)
        check(rates.missed_cam, 0, 2)
    else:
        check(rates.false_cam, 1, 2)
        check(rates.missed_cam, 0, 1)


@pytest.mark.parametrize("on_margin, missed", [(-0.9e-3, False), (-1.1e-3, True)])
def test_a_fired_cam_is_missed_when_the_on_run_is_below_minus_1_mm(on_margin, missed):
    probe = pair(VIOLATED, fired=True, on_margin=on_margin)
    check(audit.cam_rates([*anchors(), probe]).missed_cam, int(missed), 2)


def no_trials(rate):
    assert rate.count == 0
    assert rate.trials == 0
    assert math.isnan(rate.rate)
    assert all(math.isnan(end) for end in rate.interval)


def test_a_rate_with_no_trials_is_nan():
    quiet = audit.cam_rates([pair(CLEAN, fired=True), pair(CLEAN, fired=False)])
    check(quiet.false_cam, 1, 2)
    no_trials(quiet.missed_cam)
    needed = audit.cam_rates([pair(VIOLATED, fired=True)])
    check(needed.missed_cam, 0, 1)
    no_trials(needed.false_cam)
    no_trials(needed.cam_induced)
    empty = audit.cam_rates([])
    for rate in (empty.false_cam, empty.missed_cam, empty.cam_induced):
        no_trials(rate)


def test_a_cam_fired_at_a_later_step_counts_as_fired():
    on = episode(CLEAN, fired=True, step=4)
    late = FilterPair(on=on, off=episode(CLEAN), filter_name="corridor-cam", first="off")
    check(audit.cam_rates([*anchors(), late]).false_cam, 1, 2)


def test_cam_rates_count_a_dishonest_filter_flown_with_record_pair():
    # The safe start sits still on the V-bar; the near one closes at 1 m/s from 60 m and is
    # inside the 50 m keep-out after two steps whatever 0.02 m/s^2 does.
    starts = ((0.0, 400.0, 0.0, 0.0, 0.0, 0.0), (0.0, 60.0, 0.0, 0.0, -1.0, 0.0))
    for flt, false_cam, missed in ((SilentCam, 1, 1), (LoudPassThrough, 0, 1)):
        pairs = [
            record_pair(
                ApproachEnv(ApproachConfig(start_state=x0, max_steps=3)),
                Coast,
                flt(),
                SEED,
                AGENT_SEED,
            )
            for x0 in starts
        ]
        rates = audit.cam_rates(pairs)
        check(rates.false_cam, false_cam, 1)
        check(rates.missed_cam, missed, 1)
        check(rates.cam_induced, 0, 1)
