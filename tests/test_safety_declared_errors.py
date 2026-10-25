"""Error sets a filter declares from the truncated navigation and execution models.

The interface these tests assume:

- `tightening.declared_errors(sensor, execution, *, a_max, layout, level)` returns
  `(navigation, da_max)`: `BoxError(r_pos=b (sigma_pos + bias_pos), r_vel=b (sigma_vel +
  bias_vel))` for a `GaussianNav` (None for no sensor or `PerfectNav`) and
  da_max = a_max [s (|scale_bias| + b scale_sigma + 2 sin(theta / 2)) + sqrt(3) b noise_sigma]
  (m/s^2, 0 for no execution error), with b = `level`, theta the misalignment angle and s the
  largest command length, sqrt(3) for "box" and 1 for "ball".
- `tightening.episode_draws(sensor, execution, *, steps)` counts the Gaussian draws of an
  episode of `steps` decisions with a nonzero sigma: per step three for each nonzero noise sigma
  of the sensor, three for a nonzero `noise_sigma` and one for a nonzero `scale_sigma`; once per
  episode three for each nonzero bias sigma.
- The filters take `exceedance` and `per_draw_level` keywords (None by default) and report them
  in `FilterDeclaration` fields of the same names; `contains_start(x)` tells whether every
  estimate x + e, e in the navigation set, lies in the filter's declared set.
"""

import itertools
import math

import numpy as np
import pytest
from test_safety_cbf import braking, cbf, ready, view

from vbar.safety import tightening
from vbar.safety.cbf import BackupFilter
from vbar.safety.predictive import PredictiveFilter
from vbar.safety.tightening import BoxError
from vbar.scenarios import per_draw_level
from vbar.spacecraft import ExecutionError, GaussianNav, PerfectNav

A_MAX = 0.02
DT = 10.0
STEPS = 360
HOLD = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
VERTICES = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))
# Test settings, not a model of a real thruster: a -1% scale bias, 1% scale and 0.5% noise
# sigmas and a 0.023 rad misalignment, so every term of the bound is exercised.
MISALIGNMENT = (0.01, -0.02, 0.005)
THETA = math.sqrt(sum(a * a for a in MISALIGNMENT))
EXECUTION = {"scale_bias": -0.01, "scale_sigma": 0.01, "noise_sigma": 0.005}
NAV = {"sigma_pos": 1.0, "sigma_vel": 1e-3, "bias_pos": 0.5, "bias_vel": 2e-4}
# The truncation acts on a third of the draws at 1 sigma; 3 sigma checks a level near the
# audit's 5 sigma without needing millions of draws to reach it.
LEVELS = (1.0, 3.0)
# 1e5 draws take tens of seconds of Python calls; the default run takes 2000.
DRAWS = (pytest.param(100_000, marks=pytest.mark.slow), 2_000)


def burn_error(level):
    return ExecutionError(misalignment=MISALIGNMENT, bound=level, **EXECUTION)


@pytest.mark.parametrize("draws", DRAWS)
@pytest.mark.parametrize("level", LEVELS)
def test_declared_box_contains_every_truncated_nav_draw(level, draws):
    sensor = GaussianNav(bound=level, **NAV)
    navigation, da_max = tightening.declared_errors(
        sensor, None, a_max=A_MAX, layout="box", level=level
    )
    assert isinstance(navigation, BoxError)
    assert navigation.r_pos == pytest.approx(level * 1.5, rel=1e-12)
    assert navigation.r_vel == pytest.approx(level * 1.2e-3, rel=1e-12)
    assert da_max == 0.0
    radii = np.repeat([navigation.r_pos, navigation.r_vel], 3)
    rng = np.random.default_rng(2025)
    worst = np.zeros(6)
    # Episodes of 100 observations, so the bias is drawn once per 100 noise draws.
    for _ in range(draws // 100):
        sensor.reset(rng)
        for _ in range(100):
            worst = np.maximum(worst, np.abs(sensor.observe(np.zeros(6), 0.0)))
    assert np.all(worst <= radii)


def commands(layout, rng, count):
    """Commands as fractions of a_max on the layout's limit: the corners, then random ones."""
    if layout == "box":
        corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
        inside = rng.uniform(-1.0, 1.0, (count, 3))
    else:
        corners = np.vstack([np.eye(3), -np.eye(3)])
        inside = rng.standard_normal((count, 3))
        inside /= np.linalg.norm(inside, axis=1, keepdims=True)
        inside *= rng.uniform(0.0, 1.0, (count, 1)) ** (1.0 / 3.0)
    return np.vstack([np.repeat(corners, count // (4 * len(corners)), axis=0), inside])[:count]


@pytest.mark.parametrize("draws", DRAWS)
@pytest.mark.parametrize("layout, longest", [("box", math.sqrt(3.0)), ("ball", 1.0)])
@pytest.mark.parametrize("level", LEVELS)
def test_declared_ball_contains_every_truncated_burn_error(layout, longest, level, draws):
    execution = burn_error(level)
    navigation, da_max = tightening.declared_errors(
        None, execution, a_max=A_MAX, layout=layout, level=level
    )
    assert navigation is None
    expected = A_MAX * (
        longest * (0.01 + level * 0.01 + 2.0 * math.sin(THETA / 2.0))
        + math.sqrt(3.0) * level * 0.005
    )
    assert da_max == pytest.approx(expected, rel=1e-12)
    rng = np.random.default_rng(2026)
    worst = 0.0
    for c in commands(layout, rng, draws):
        delivered = A_MAX * execution.apply(c, rng)
        worst = max(worst, float(np.linalg.norm(delivered - A_MAX * c)))
    assert worst <= da_max


def test_declared_sets_combine_both_models_and_vanish_without_them():
    navigation, da_max = tightening.declared_errors(
        GaussianNav(bound=2.0, **NAV), burn_error(2.0), a_max=A_MAX, layout="ball", level=2.0
    )
    assert navigation.r_pos == pytest.approx(3.0, rel=1e-12)
    assert da_max > 0.0
    for sensor in (None, PerfectNav()):
        navigation, da_max = tightening.declared_errors(
            sensor, None, a_max=A_MAX, layout="box", level=2.0
        )
        assert navigation is None
        assert da_max == 0.0


def test_declared_errors_do_not_depend_on_the_episode_seed():
    sensor = GaussianNav(bound=3.0, **NAV)
    execution = burn_error(3.0)

    def declared():
        return tightening.declared_errors(sensor, execution, a_max=A_MAX, layout="box", level=3.0)

    before = declared()
    for seed in (1, 2, 3):
        sensor.reset(np.random.default_rng(seed))
        assert declared() == before


@pytest.mark.parametrize(
    "sensor, execution, draws",
    [
        (GaussianNav(), None, 6 * STEPS),
        (None, ExecutionError(noise_sigma=0.005), 3 * STEPS),
        (GaussianNav(bias_pos=0.5, bias_vel=2e-4), None, 6 * STEPS + 6),
        (GaussianNav(bias_pos=0.5), None, 6 * STEPS + 3),
        (GaussianNav(sigma_vel=0.0), None, 3 * STEPS),
        (None, ExecutionError(scale_sigma=0.01, noise_sigma=0.005), 4 * STEPS),
        (None, ExecutionError(scale_sigma=0.01), STEPS),
        (None, ExecutionError(scale_bias=0.01, misalignment=MISALIGNMENT), 0),
        (GaussianNav(), ExecutionError(scale_sigma=0.01, noise_sigma=0.005), 10 * STEPS),
        (PerfectNav(), None, 0),
        (None, None, 0),
    ],
)
def test_episode_draws_count_every_draw_with_a_nonzero_sigma(sensor, execution, draws):
    assert tightening.episode_draws(sensor, execution, steps=STEPS) == draws


def test_episode_draws_of_nav1_and_burn_noise_give_the_known_levels():
    # 360 decisions: :nav1 draws 2160 noise values and a 3-axis burn noise 1080, which need
    # 5.04 and 4.91 sigma per draw for an episode-level 1e-3.
    nav_draws = tightening.episode_draws(GaussianNav(), None, steps=STEPS)
    burn_draws = tightening.episode_draws(None, ExecutionError(noise_sigma=0.005), steps=STEPS)
    assert per_draw_level(1e-3, nav_draws) == pytest.approx(5.04, abs=0.005)
    assert per_draw_level(1e-3, burn_draws) == pytest.approx(4.91, abs=0.005)


NAVIGATION = BoxError(r_pos=0.3, r_vel=1e-4)
# Coarse drift grids keep each clearance to a few dozen samples; the predictive filter drifts
# two orbits, on which a 200 s grid still certifies most of the starts below.
FILTERS = {
    "passive": lambda **kw: BackupFilter(barrier="passive", t_cap=1000.0, grid=100.0, **kw),
    "braking": braking,
    "predictive": lambda **kw: PredictiveFilter(grid=200.0, **kw),
}


def starts(rng, count):
    """Slow states between 80 and 300 m behind the target on V-bar, then the hold."""
    spread = rng.uniform(-1.0, 1.0, (count, 6)) * np.array([5.0, 0.0, 5.0, 1e-3, 1e-3, 1e-3])
    spread[:, 1] = rng.uniform(-300.0, -80.0, count)
    return np.vstack([spread, HOLD])


def unit_errors(x, rng, sweep):
    """Navigation errors as fractions of the box radii, the one toward the target first.

    Each `contains` is a drift clearance, so the default run takes the vertex that moves the
    estimate toward the target and four seeded vertices and interior points; the slow run
    takes every vertex and 64 interior points.
    """
    toward = np.where(x < 0.0, 1.0, -1.0)
    if sweep == "full":
        return np.vstack([toward, VERTICES, rng.uniform(-1.0, 1.0, (64, 6))])
    picked = VERTICES[rng.choice(len(VERTICES), 4, replace=False)]
    return np.vstack([toward, picked, rng.uniform(-1.0, 1.0, (4, 6))])


@pytest.mark.parametrize("sweep", [pytest.param("full", marks=pytest.mark.slow), "sampled"])
@pytest.mark.parametrize("name", FILTERS)
def test_start_set_keeps_every_estimate_inside(name, sweep):
    filt = FILTERS[name](navigation=NAVIGATION)
    filt.reset(view(DT, start=HOLD))
    radii = np.repeat([NAVIGATION.r_pos, NAVIGATION.r_vel], 3)
    rng = np.random.default_rng(71)
    for x in starts(rng, 10):
        if filt.contains_start(x):
            break
    else:
        pytest.fail("no start is in the start set")
    for e in unit_errors(x, rng, sweep) * radii:
        assert filt.contains(x + e), e


def at_rest(y):
    return np.array([0.0, y, 0.0, 0.0, 0.0, 0.0])


def edge(inside, kept, refused, step):
    """Bisect V-bar rest states to `step` (m) between one `inside` passes and one it fails."""
    assert inside(at_rest(kept))
    assert not inside(at_rest(refused))
    while refused - kept > step:
        mid = 0.5 * (kept + refused)
        if inside(at_rest(mid)):
            kept = mid
        else:
            refused = mid
    return kept, refused


# The drift from rest on V-bar is unchanged by z, vz -> -z, -vz, so the vertices with a
# positive z error stand for all of them.
MIRRORED = VERTICES[VERTICES[:, 2] > 0.0]


@pytest.mark.parametrize("vertices", [pytest.param(VERTICES, marks=pytest.mark.slow), MIRRORED])
@pytest.mark.parametrize("name", FILTERS)
def test_start_set_refuses_a_declared_start_whose_estimate_leaves_the_set(name, vertices):
    filt = FILTERS[name](navigation=NAVIGATION)
    filt.reset(view(DT, start=HOLD))
    errors = vertices * np.repeat([NAVIGATION.r_pos, NAVIGATION.r_vel], 3)
    kept, refused = edge(filt.contains_start, -300.0, -50.0, 1e-3)
    declared, _ = edge(filt.contains, refused, -1.0, 1e-2)
    # Halfway between the two edges the start is declared but refused: the start set is an
    # outer bound on the estimates, 0.1 m to 10 m inside their edge here, so halfway (0.5 m to
    # 18 m) is past it and some estimate leaves the declared set.
    x = at_rest(0.5 * (refused + declared))
    assert filt.contains(x)
    assert not filt.contains_start(x)
    assert any(not filt.contains(x + e) for e in errors)
    # 1 mm inside the edge clears the search's 1e-6 relative tolerance, 0.3 mm at 300 m.
    x = at_rest(kept - 1e-3)
    assert filt.contains_start(x)
    for e in errors:
        assert filt.contains(x + e), e


def test_braking_start_set_refuses_starts_whose_estimates_cross_its_caps():
    filt = braking(navigation=NAVIGATION)
    filt.reset(view(DT, start=HOLD))
    rho_cap, v_cap = filt.rho_cap, filt.v_cap
    errors = VERTICES * np.repeat([NAVIGATION.r_pos, NAVIGATION.r_vel], 3)
    # Half a box radius inside a cap: the vertex error along that axis carries the estimate
    # half a radius past it, while h_b stays far above zero.
    near_v_cap = np.array([0.0, -1000.0, 0.0, 0.0, v_cap - 0.5 * NAVIGATION.r_vel, 0.0])
    near_rho_cap = at_rest(-(rho_cap - 0.5 * NAVIGATION.r_pos))
    for x in (near_v_cap, near_rho_cap):
        assert filt.contains(x)
        assert not filt.contains_start(x)
        assert any(not filt.contains(x + e) for e in errors)
    # Along the box diagonal a vertex reaches |r| + sqrt(3) r_pos and |v| + sqrt(3) r_vel, so
    # 1 mm and 1 um/s inside those margins is just inside both caps.
    d = np.array([1.0, -1.0, 1.0]) / math.sqrt(3.0)
    rho = rho_cap - math.sqrt(3.0) * NAVIGATION.r_pos - 1e-3
    speed = v_cap - math.sqrt(3.0) * NAVIGATION.r_vel - 1e-6
    x = np.concatenate([rho * d, -speed * d])
    assert filt.contains_start(x)
    for e in errors:
        assert filt.contains(x + e), e


@pytest.mark.parametrize("name", FILTERS)
def test_start_set_without_navigation_error_is_the_declared_set(name):
    filt = FILTERS[name](execution=1e-4)
    filt.reset(view(DT, start=HOLD))
    for x in starts(np.random.default_rng(72), 10):
        assert filt.contains_start(x) == filt.contains(x)


@pytest.mark.parametrize("name", FILTERS)
def test_start_set_needs_a_reset(name):
    with pytest.raises(RuntimeError):
        FILTERS[name](navigation=NAVIGATION).contains_start(HOLD)


DECLARING = {
    **FILTERS,
    "ecbf": lambda **kw: cbf(p=0.06, **kw),
}


@pytest.mark.parametrize("name", DECLARING)
def test_declaration_reports_exceedance_and_per_draw_level(name):
    level = per_draw_level(1e-3, 6 * STEPS)
    filt = ready(DECLARING[name](execution=1e-4, exceedance=1e-3, per_draw_level=level), DT)
    declaration = filt.declaration
    assert declaration.exceedance == 1e-3
    assert declaration.per_draw_level == level
    assert "exceedance=0.001" in repr(declaration)
    assert f"per_draw_level={level!r}" in repr(declaration)


@pytest.mark.parametrize("name", DECLARING)
def test_declaration_without_a_truncation_reports_none(name):
    declaration = ready(DECLARING[name](execution=1e-4), DT).declaration
    assert declaration.exceedance is None
    assert declaration.per_draw_level is None


@pytest.mark.parametrize(
    "kw",
    [
        {"exceedance": 0.0},
        {"exceedance": 1.0},
        {"exceedance": math.nan},
        {"per_draw_level": 0.0},
        {"per_draw_level": -5.0},
        {"per_draw_level": math.inf},
    ],
)
def test_declaration_refuses_an_exceedance_outside_zero_one_or_a_level_not_positive(kw):
    with pytest.raises(ValueError):
        BackupFilter(barrier="passive", execution=1e-4, **kw)
