"""Scripted navigation and execution errors, and the falsifier that searches over them.

The interface these tests assume, in `vbar.stress.disturbances`:

- `ScriptedNav(offsets, box=..., duration=...)`: `offsets` (4, 6) in [-1, 1], one row per
  quarter of an episode of `duration` s (the last row from 3/4 of it on); `observe(x, t)`
  returns x plus the row for t times the radii of the `BoxError` `box`. `reset(rng)` draws
  nothing.
- `ScriptedExecution(direction, da_max=..., a_max=...)`: `direction` in [-1, 1]^3 is mapped
  radially onto the ball of radius `da_max` (m/s^2), w = da_max max|d_i| d / |d|; `apply(u, rng)`
  returns u + w / a_max for a command u (fractions of a_max) that thrusts, and u unchanged for a
  coast, and draws nothing.
- `falsify_filter(..., errors="bounded")` searches z = (start, 4 x 6 navigation offsets, 3
  execution directions), 33 numbers, starts drawn in the filter's start set and the rest in
  [-1, 1], and flies each z with the scripted models scaled by the filter's declared sets.
"""

import itertools

import numpy as np
import pytest

from vbar.agents.base import action_scale
from vbar.safety import FilterResult, tightening
from vbar.safety.audit import falsify_filter
from vbar.safety.cbf import BackupFilter
from vbar.safety.tightening import BoxError
from vbar.scenarios import per_draw_level
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.spacecraft import ExecutionError, GaussianNav
from vbar.stress import disturbances

SEED = 0xD1B54A32D192ED03ABA6F0B1C3E5A7F9
P = 1e-3
A_MAX = 0.02
BOX = BoxError(r_pos=0.4, r_vel=4e-4)
RADII = np.repeat([BOX.r_pos, BOX.r_vel], 3)
DURATION = 40.0
DA_MAX = 1.3e-3
CORNERS = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))


def offsets(rng):
    """Four rows of offsets: two box corners, then two interior points."""
    picked = CORNERS[rng.choice(len(CORNERS), 2, replace=False)]
    return np.vstack([picked, rng.uniform(-1.0, 1.0, (2, 6))])


def test_scripted_nav_errors_stay_inside_the_declared_box():
    rng = np.random.default_rng(81)
    for _ in range(20):
        nav = disturbances.ScriptedNav(offsets(rng), box=BOX, duration=DURATION)
        for t in np.linspace(0.0, 1.5 * DURATION, 13):
            x = rng.uniform(-200.0, 200.0, 6) * np.repeat([1.0, 1e-3], 3)
            error = nav.observe(x, t) - x
            # The error is one rounding of x away from offset times radius, which is at most
            # the radius.
            assert np.all(np.abs(error) <= RADII + 1e-12 * np.abs(x).max())


def test_scripted_nav_holds_each_row_for_a_quarter_of_the_episode():
    rows = np.linspace(-1.0, 1.0, 24).reshape(4, 6)
    nav = disturbances.ScriptedNav(rows, box=BOX, duration=DURATION)
    x = np.zeros(6)
    for t, row in [(0.0, 0), (9.9, 0), (10.0, 1), (25.0, 2), (39.9, 3), (40.0, 3), (100.0, 3)]:
        np.testing.assert_allclose(nav.observe(x, t), rows[row] * RADII, rtol=1e-15, atol=0.0)


def test_scripted_nav_is_stateless():
    rows = offsets(np.random.default_rng(82))
    nav = disturbances.ScriptedNav(rows, box=BOX, duration=DURATION)
    rng = np.random.default_rng(83)
    before = rng.bit_generator.state
    nav.reset(rng)
    assert rng.bit_generator.state == before
    x = np.array([5.0, 120.0, -3.0, 0.01, -0.02, 0.0])
    times = (35.0, 5.0, 15.0, 35.0, 5.0)
    seen = [nav.observe(x, t) for t in times]
    np.testing.assert_array_equal(seen[0], seen[3])
    np.testing.assert_array_equal(seen[1], seen[4])
    fresh = disturbances.ScriptedNav(rows, box=BOX, duration=DURATION)
    for t, obs in zip(times, seen, strict=True):
        np.testing.assert_array_equal(fresh.observe(x, t), obs)


@pytest.mark.parametrize("rows", [np.full((4, 6), 1.01), np.zeros((3, 6)), np.full((4, 6), np.nan)])
def test_scripted_nav_refuses_offsets_outside_the_unit_box(rows):
    with pytest.raises(ValueError):
        disturbances.ScriptedNav(rows, box=BOX, duration=DURATION)


def burn_error(direction, command=(0.3, -1.0, 0.5)):
    """The delivered minus the commanded acceleration, m/s^2."""
    u = np.array(command)
    delivered = disturbances.ScriptedExecution(direction, da_max=DA_MAX, a_max=A_MAX).apply(
        u, np.random.default_rng(84)
    )
    return (delivered - u) * A_MAX


def test_scripted_burn_error_maps_the_cube_radially_onto_the_declared_ball():
    rng = np.random.default_rng(85)
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    for d in np.vstack([corners, np.eye(3), rng.uniform(-1.0, 1.0, (50, 3))]):
        w = burn_error(d)
        assert np.linalg.norm(w) == pytest.approx(DA_MAX * np.abs(d).max(), rel=1e-12)
        np.testing.assert_allclose(w / np.linalg.norm(w), d / np.linalg.norm(d), atol=1e-12)
        assert np.linalg.norm(w) <= DA_MAX * (1.0 + 1e-12)
    np.testing.assert_allclose(
        burn_error([0.5, 0.0, 0.0]), [0.5 * DA_MAX, 0.0, 0.0], rtol=1e-12, atol=1e-18
    )


def test_scripted_burn_error_is_constant_and_leaves_a_coast_unperturbed():
    d = np.array([0.2, -0.7, 0.4])
    for command in ([1.0, 1.0, 1.0], [0.0, -0.1, 0.0], [-0.5, 0.25, 1.0]):
        np.testing.assert_allclose(burn_error(d, command), burn_error(d), rtol=1e-12, atol=1e-18)
    coast = disturbances.ScriptedExecution(d, da_max=DA_MAX, a_max=A_MAX).apply(
        np.zeros(3), np.random.default_rng(86)
    )
    np.testing.assert_array_equal(coast, np.zeros(3))


def test_scripted_burn_error_is_stateless():
    execution = disturbances.ScriptedExecution([0.9, 0.1, -0.3], da_max=DA_MAX, a_max=A_MAX)
    rng = np.random.default_rng(87)
    before = rng.bit_generator.state
    u = np.array([0.4, -0.2, 0.1])
    first = execution.apply(u, rng)
    assert rng.bit_generator.state == before
    np.testing.assert_array_equal(execution.apply(u, rng), first)
    np.testing.assert_array_equal(u, [0.4, -0.2, 0.1])


class Reckless:
    """Full command straight at the target."""

    name = "reckless"

    def reset(self, view, rng=None):
        self.scale = action_scale(view)

    def act(self, obs, info):
        return -obs[:3] / np.linalg.norm(obs[:3]) * self.scale


class IgnoresExecution(BackupFilter):
    """Declares its execution set and tightens every check with zero."""

    @property
    def da_max(self):
        return self.model_error or 0.0

    @property
    def bounded(self):
        return self.navigation is not None or self.execution > 0.0


# Hill states (m, m/s) up to 1 m outside the 50 m keep-out, where the braking filter holds the
# reckless agent's full thrust on the keep-out's edge.
CLOSE = ((-8.0, 50.2, -1.0, -0.01, -0.01, -0.01), (8.0, 51.0, 1.0, 0.01, 0.01, 0.01))
STEPS = 2
# Test setting, not a thruster model: 1% burn noise gives a 1.3 mm/s^2 ball, whose inward half
# carries a step held on the keep-out's edge 1 to 6 cm inside it when the filter ignores it.
BURN = {"noise_sigma": 0.01}
LEVEL = per_draw_level(P, tightening.episode_draws(None, ExecutionError(**BURN), steps=STEPS))


def burn_filters():
    _, da_max = tightening.declared_errors(
        None, ExecutionError(bound=LEVEL, **BURN), a_max=A_MAX, layout="box", level=LEVEL
    )
    declared = {"execution": da_max, "exceedance": P, "per_draw_level": LEVEL}
    return (
        BackupFilter(barrier="braking", **declared),
        IgnoresExecution(barrier="braking", **declared),
    )


def search(filt, env, box, budget):
    return falsify_filter(
        filt, env, nominal=Reckless(), errors="bounded", box=box, budget=budget, population=10,
        chains=1, seed=SEED, max_steps=STEPS,
    )  # fmt: skip


def test_falsifier_finds_ignored_execution_set():
    env = ApproachEnv(
        ApproachConfig(execution=ExecutionError(bound=LEVEL, **BURN), max_steps=STEPS)
    )
    real, mutant = burn_filters()
    view = env.agent_view()
    real.reset(view)
    mutant.reset(view)
    assert mutant.declaration.tightening == real.declaration.tightening
    assert mutant.declaration.certificate == real.declaration.certificate
    caught = search(mutant, env, CLOSE, budget=10)
    assert caught.verdict == "violation"
    assert caught.score < -1.0
    assert caught.run.violations > 0
    clean = search(real, env, CLOSE, budget=10)
    # Braking issues no certificate under error sets, so a run without a violation is "uncovered".
    assert clean.verdict == "uncovered"
    assert clean.run.violations == 0
    assert clean.score >= -1.0


# Test setting: a navigation box of a few tenths of a metre keeps the braking start set wide.
NAV = {"sigma_pos": 0.1, "sigma_vel": 1e-4}
NEAR = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))


def nav_and_burn(env_type=ApproachEnv):
    """A braking filter declaring the truncated navigation and burn errors of its scenario."""
    draws = tightening.episode_draws(GaussianNav(**NAV), ExecutionError(**BURN), steps=STEPS)
    level = per_draw_level(P, draws)
    sensor, execution = GaussianNav(bound=level, **NAV), ExecutionError(bound=level, **BURN)
    navigation, da_max = tightening.declared_errors(
        sensor, execution, a_max=A_MAX, layout="box", level=level
    )
    filt = BackupFilter(
        barrier="braking", navigation=navigation, execution=da_max, exceedance=P,
        per_draw_level=level,
    )  # fmt: skip
    env = env_type(ApproachConfig(sensor=sensor, execution=execution, max_steps=STEPS))
    return filt, env


def test_falsifier_searches_start_navigation_and_burn_error_together():
    filt, env = nav_and_burn()
    found = search(filt, env, NEAR, budget=4)
    points = found.search.points
    assert points.shape == (4, 33)
    assert np.all(np.abs(points[:, 6:]) <= 1.0)
    lo, hi = (np.array(corner) for corner in NEAR)
    assert np.all((lo <= points[:, :6]) & (points[:, :6] <= hi))
    assert all(filt.contains_start(z[:6]) for z in points)
    np.testing.assert_array_equal(found.search.x[:6], found.start)


def test_bounded_falsifier_flies_every_run_with_scripted_errors_only():
    flown = []

    class Recording(ApproachEnv):
        def __init__(self, config):
            flown.append((config.sensor, config.execution))
            super().__init__(config)

    filt, env = nav_and_burn(Recording)
    flown.clear()
    found = search(filt, env, NEAR, budget=4)
    # The search's runs and the re-fly of the worst point.
    assert len(flown) == found.evaluations + 1
    for sensor, execution in flown:
        assert isinstance(sensor, disturbances.ScriptedNav)
        assert isinstance(execution, disturbances.ScriptedExecution)


def test_bounded_falsifier_reflies_the_worst_point_to_its_searched_score():
    filt, env = nav_and_burn()
    found = search(filt, env, NEAR, budget=4)
    resolutions = {c.name: c.resolution for c in env.constraints}
    reflown = min(bound / resolutions[name] for name, bound in found.run.worst_bound.items())
    # The scripted errors draw nothing, so the same point flies the same run.
    assert reflown == found.score


# Hill states just outside the keep-out, where about two thirds of the starts in the safe set
# have an estimate within the navigation box that is not.
EDGE = ((-2.0, 50.0, -2.0, -0.02, -0.02, -0.02), (2.0, 51.0, 2.0, 0.02, 0.02, 0.02))


def test_bounded_falsifier_flies_only_starts_whose_every_estimate_is_safe():
    filt, env = nav_and_burn()
    filt.reset(env.agent_view())
    lo, hi = (np.array(corner) for corner in EDGE)
    draws = np.random.default_rng(88).uniform(lo, hi, (200, 6))
    assert any(filt.contains(x) and not filt.contains_start(x) for x in draws)
    found = search(filt, env, EDGE, budget=10)
    assert all(filt.contains_start(z[:6]) for z in found.search.points)


def test_bounded_falsifier_refuses_untruncated_models():
    real, _ = burn_filters()
    env = ApproachEnv(ApproachConfig(execution=ExecutionError(**BURN), max_steps=STEPS))
    with pytest.raises(ValueError):
        search(real, env, CLOSE, budget=4)


def test_falsifier_refuses_gaussian_errors():
    # Untruncated models, which `audit_filter` accepts for gaussian errors, so only the error
    # mode itself is refused.
    real, _ = burn_filters()
    env = ApproachEnv(ApproachConfig(execution=ExecutionError(**BURN), max_steps=STEPS))
    with pytest.raises(ValueError, match="gaussian"):
        falsify_filter(
            real, env, nominal=Reckless(), errors="gaussian", box=CLOSE, budget=4, population=10,
            chains=1, seed=SEED, max_steps=STEPS,
        )  # fmt: skip


class Scripted(BackupFilter):
    """Returns the decisions of `script` in turn, certified with the horizon given or not."""

    script = ()

    def reset(self, view):
        super().reset(view)
        self.k = 0

    def filter(self, obs, action):
        horizon, thrust = self.script[self.k]
        self.k += 1
        u = np.array([0.0, 0.01, 0.0]) if thrust else np.zeros(3)
        if horizon is None:
            return FilterResult(u, intervened=True)
        return FilterResult(
            u, intervened=True, certificate="bounded-W", certificate_horizon=horizon
        )


# Near rest on V-bar about 300 m out, far from the keep-out whatever the script commands.
FAR = ((-1.0, 299.0, -1.0, -1e-4, -1e-4, -1e-4), (1.0, 301.0, 1.0, 1e-4, 1e-4, 1e-4))


def scripted_search(script):
    real, _ = burn_filters()
    declared = {"execution": real.execution, "exceedance": P, "per_draw_level": LEVEL}
    filt = Scripted(barrier="passive", grid=100.0, t_cap=1000.0, **declared)
    filt.script = script
    env = ApproachEnv(
        ApproachConfig(execution=ExecutionError(bound=LEVEL, **BURN), max_steps=len(script))
    )
    return search(filt, env, FAR, budget=4)


def test_bounded_falsifier_reports_an_uncovered_decision():
    # A certificate at 0 s with a 1000 s horizon, then a burn decided at 10 s without one.
    burn = scripted_search(((1000.0, False), (None, True)))
    assert burn.run.violations == burn.run.uncertified == 0
    assert burn.run.uncovered == 1
    assert burn.verdict == "uncovered"
    # The same decision as a coast ends at 20 s, inside the horizon, so it is covered.
    coast = scripted_search(((1000.0, False), (None, False)))
    assert coast.run.uncovered == 0
    assert coast.verdict == "none_found"
    # Braking under error sets certifies nothing, so its thrusting decisions are all uncovered.
    real, _ = burn_filters()
    env = ApproachEnv(
        ApproachConfig(execution=ExecutionError(bound=LEVEL, **BURN), max_steps=STEPS)
    )
    clean = search(real, env, CLOSE, budget=10)
    assert clean.run.uncovered > 0
    assert clean.verdict == "uncovered"
