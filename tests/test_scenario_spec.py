import math
import os
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from scipy.stats import norm

import vbar.scenarios as scenarios
from vbar.scenarios import (
    SCENARIOS,
    SUITES,
    Choice,
    Fixed,
    HoldPointEnv,
    Normal,
    Observation,
    ScenarioSpec,
    SeedSplit,
    Shell,
    Suite,
    SuiteEntry,
    Uniform,
    make,
    parse_id,
    per_draw_level,
    scenario_id,
    seeds,
)
from vbar.scenarios.registry import gymnasium_id
from vbar.scenarios.seeds import HIDDEN_KEY_VARIABLE
from vbar.spacecraft import ExecutionError, GaussianNav, GaussMarkovNav, RangeScaledNav


def test_uniform_and_normal_draws_stay_inside_their_bounds_and_report_them():
    rng = np.random.default_rng(0)
    box = Uniform((-1.0, 0.0), (1.0, 2.0))
    draws = np.array([box.sample(rng) for _ in range(500)])
    assert np.all(draws >= [-1.0, 0.0]) and np.all(draws <= [1.0, 2.0])
    assert np.allclose(box.nominal(), [0.0, 1.0])
    scalar = Uniform(3.0, 5.0)
    assert isinstance(scalar.sample(rng), float) and scalar.bounds() == (3.0, 5.0)
    bounded = Normal(0.0, 2.0, bound=1.5)
    values = np.array([bounded.sample(rng) for _ in range(2000)])
    assert np.max(np.abs(values)) <= 3.0 and bounded.bounds() == (-3.0, 3.0)
    assert Shell(10.0, 20.0).sample(rng).shape == (3,)
    assert 10.0 <= np.linalg.norm(Shell(10.0, 20.0).sample(rng)) <= 20.0
    assert Choice((1, 2, 3), weights=(0, 0, 1)).sample(rng) == 3
    assert Fixed(7).sample(rng) == 7


def test_dispersions_reject_bad_parameters():
    with pytest.raises(ValueError):
        Uniform(1.0, 0.0)
    with pytest.raises(ValueError):
        Normal(0.0, -1.0)
    with pytest.raises(ValueError):
        Normal(0.0, 1.0, bound=0.0)
    with pytest.raises(ValueError):
        Shell(5.0, 1.0)
    with pytest.raises(ValueError):
        Observation(horizon="episodic")


def test_per_draw_level_matches_the_union_bound_known_answers():
    # Kochenderfer et al., Algorithms for Validation, sec. 8.1: 3 sigma per draw leaves 94.6% of
    # 1080-draw episodes outside the bounded model; 1e-3 per episode needs 4.91 and 5.04 sigma.
    assert 1.0 - (1.0 - 2.0 * norm.sf(3.0)) ** 1080 == pytest.approx(0.946, abs=1e-3)
    assert per_draw_level(1e-3, 1080) == pytest.approx(4.91, abs=0.005)
    assert per_draw_level(1e-3, 2160) == pytest.approx(5.04, abs=0.005)
    level = per_draw_level(0.05, 360)
    assert 1.0 - (1.0 - 2.0 * norm.sf(level)) ** 360 == pytest.approx(0.05, rel=1e-9)


def test_spec_episode_draws_what_reset_draws():
    spec = SCENARIOS["docking"]
    env = make("docking")
    env.reset(seed=11)
    drawn = spec.episode(11)
    assert np.allclose(np.concatenate([drawn["start_pos"], drawn["start_vel"]]), env.state)
    assert set(drawn) == set(env.draw)


def test_variants_apply_in_order_and_are_recorded_in_the_identifier():
    base = SCENARIOS["approach-vbar"]
    iss = base.variant("iss", "nav1")
    assert iss.applied == ("iss", "nav1")
    assert iss.parameters["keep_out"] == 200.0 and isinstance(iss.config["sensor"], GaussianNav)
    assert "iss" in iss.tags and "nav1" in iss.tags
    env = make("approach-vbar:iss:nav1")
    assert env.scenario_id == "vbar:approach-vbar:cw:box:dv:v1:iss:nav1"
    with pytest.raises(KeyError):
        base.variant("mars")


def test_every_result_changing_config_field_shows_in_the_identifier():
    base = make("approach-vbar").scenario_id
    changed = {
        "dt": 5.0,
        "max_steps": 100,
        "dynamics": "twobody",
        "altitude_m": 400e3,
        "action_mode": "disc27",
        "max_accel": 0.01,
        "thrust_limit": "ball",
        "isp": 300.0,
        "mass_kg": 600.0,
        "propellant_kg": 50.0,
        "min_impulse": 0.5,
        "delay_s": 2.0,
        "execution": ExecutionError(scale_sigma=0.01),
        "sensor": GaussianNav(0.5, 5e-4),
        "violation_samples": 40,
        "observation_channels": ("t_remaining",),
    }
    ids = {k: make("approach-vbar", **{k: v}).scenario_id for k, v in changed.items()}
    ids["drive"] = make("approach-vbar", drive="thrust", propellant_kg=100.0).scenario_id
    assert all(value != base for value in ids.values())
    assert len(set(ids.values())) == len(ids)
    assert ids["sensor"] == base + ":sensor=gaussian(0.5+0.0005+0+0)"
    assert ids["dynamics"].startswith("vbar:approach-vbar:twobody:box:dv:v1")
    j2 = make("approach-vbar", dynamics="twobody_j2").scenario_id
    assert j2.startswith("vbar:approach-vbar:twobody_j2-i51.6:box:dv:v1")
    assert make("approach-vbar", sensor=GaussianNav()).scenario_id == base + ":sensor=nav1"
    assert make("approach-vbar", sensor=GaussMarkovNav()).scenario_id == base + ":sensor=nav2"
    assert "range_scaled(" in make("approach-vbar", sensor=RangeScaledNav()).scenario_id


def test_the_reward_and_the_training_flag_do_not_change_the_identifier():
    from vbar.rewards import Reward

    base = make("approach-vbar").scenario_id
    assert make("approach-vbar", training=True).scenario_id == base
    assert make("approach-vbar", reward=Reward(success_bonus=1.0)).scenario_id == base


def test_identifiers_parse_back_to_their_scenario():
    for name in ("approach-vbar", "docking", "closing:nav1", "docking:corridor-shuttle"):
        env = make(name)
        base, variants, overrides = parse_id(env.scenario_id)
        rebuilt = make(env.scenario_id)
        assert rebuilt.scenario_id == env.scenario_id
        assert replace(rebuilt.cfg, sensor=None) == replace(env.cfg, sensor=None)
        assert type(rebuilt.cfg.sensor) is type(env.cfg.sensor)
    with pytest.raises(ValueError):
        parse_id("vbar:docking:cw:box:dv:v9")
    with pytest.raises(ValueError):
        make("vbar:docking:cw:box:dv:v2:sensor=nav1")
    with pytest.raises(ValueError):
        make("vbar:docking:cw:box:dv:v2:mars")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_and_variant_passes_the_gymnasium_checker(name):
    spec = SCENARIOS[name]
    for variants in [(), *((v,) for v in spec.variants)]:
        env = gym.make(gymnasium_id(spec, variants))
        check_env(env.unwrapped, skip_render_check=True)
        assert env.unwrapped.scenario_id.startswith(f"vbar:{name}:")


def test_made_and_constructed_scenarios_fly_the_same_episode():
    made, direct = gym.make("vbar/Docking-v2"), make("docking")
    obs_a, _ = made.reset(seed=5)
    obs_b, _ = direct.reset(seed=5)
    assert np.array_equal(obs_a, obs_b)
    for action in ([0.2, -0.5, 0.1], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.3]):
        assert np.array_equal(made.step(action)[0], direct.step(action)[0])


def test_seed_splits_are_deterministic_distinct_and_unguessable():
    train = seeds("approach-vbar", "train", 5)
    assert list(train) == list(seeds("approach-vbar", "train", 5))
    assert len(set(train) | set(seeds("approach-vbar", "validation", 5))) == 10
    assert set(train).isdisjoint(seeds("approach-rbar", "train", 5))
    assert all(s >= 2**100 for s in train)
    assert all(s != a for s, a in train.pairs())
    with pytest.raises(ValueError):
        SeedSplit("approach-vbar", "secret", 1)


def test_test_split_needs_the_hidden_key(monkeypatch):
    monkeypatch.delenv(HIDDEN_KEY_VARIABLE, raising=False)
    with pytest.raises(ValueError):
        seeds("approach-vbar", "test", 2).seed(0)
    monkeypatch.setenv(HIDDEN_KEY_VARIABLE, "0" * 32)
    first = seeds("approach-vbar", "test", 2).seed(0)
    monkeypatch.setenv(HIDDEN_KEY_VARIABLE, "1" * 32)
    assert seeds("approach-vbar", "test", 2).seed(0) != first
    assert first not in set(seeds("approach-vbar", "train", 100))


def test_suites_list_their_runs_and_find_missing_ones():
    suite = SUITES["v1-v1"]
    runs = suite.runs()
    assert len(runs) == 4 * 50 and len({(sid, seed) for sid, seed, _ in runs}) == len(runs)
    results = [{"scenario_id": sid, "seed": seed} for sid, seed, _ in runs[:-3]]
    assert len(suite.missing(results)) == 3 and not suite.complete(results)
    assert suite.complete(results + [{"scenario_id": s, "seed": k} for s, k, _ in runs[-3:]])
    with pytest.raises(KeyError):
        Suite("broken", 1, (SuiteEntry("no-such-scenario"),))
    with pytest.raises(ValueError):
        Suite("twice", 1, (SuiteEntry("docking"), SuiteEntry("docking")))


def test_tags_group_results_by_scenario_tag():
    ids = [make("approach-vbar").scenario_id, make("approach-rbar:iss").scenario_id]
    groups = scenarios.group_by_tag([{"scenario_id": i} for i in ids])
    assert len(groups["approach"]) == 2 and len(groups["iss"]) == 1 and len(groups["vbar"]) == 1
    assert "approach-hbar" in scenarios.by_tag("liveness")


def test_plugins_register_third_party_specs(monkeypatch):
    spec = ScenarioSpec(
        name="plugin-hold",
        version=1,
        entry_point=HoldPointEnv,
        summary="a plugin scenario",
        config={"max_steps": 10},
        parameters={"hold_point": (0.0, -50.0, 0.0), "keep_out": 10.0},
        dispersions={
            "start_pos": Fixed(np.array([0.0, -80.0, 0.0])),
            "start_vel": Fixed(np.zeros(3)),
        },
    )

    class Point:
        name = "plugin-hold"

        def load(self):
            return lambda: [spec]

    monkeypatch.setattr(scenarios.registry, "entry_points", lambda group: [Point()])
    try:
        assert scenarios.load_plugins() == ["plugin-hold"]
        env = make("plugin-hold")
        env.reset(seed=0)
        assert env.scenario_id == "vbar:plugin-hold:cw:box:dv:v1"
        with pytest.raises(ValueError):
            scenarios.register(replace(spec, summary="a different spec"))
    finally:
        SCENARIOS.pop("plugin-hold", None)
    with pytest.raises(ValueError):
        scenarios.plugins("vbar.rewards")


def test_scenario_id_function_matches_the_environment():
    env = make("closing")
    assert scenario_id(env.scenario, env.cfg) == env.scenario_id


def test_spec_rejects_bad_names_and_variants():
    with pytest.raises(ValueError):
        ScenarioSpec(name="bad:name", version=1, entry_point=HoldPointEnv, summary="")
    with pytest.raises(ValueError):
        ScenarioSpec(
            name="ok", version=1, entry_point=HoldPointEnv, summary="", variants={"v": {"goal": {}}}
        )


def test_hidden_key_is_not_in_the_repository():
    root = os.path.dirname(os.path.dirname(__file__))
    for folder, _, files in os.walk(os.path.join(root, "src")):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(folder, file), encoding="utf-8") as handle:
                    assert "VBAR_HIDDEN_SEED_KEY=" not in handle.read()
    assert math.isfinite(SCENARIOS["docking"].config["dt"])
