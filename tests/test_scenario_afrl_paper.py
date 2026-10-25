import dataclasses

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import vbar  # noqa: F401  (registers the environments)
from vbar.agents import DriftAgent
from vbar.dynamics import cw_system
from vbar.metrics import evaluate, record_episode
from vbar.metrics.passive import passive_metrics
from vbar.metrics.stl import Trace
from vbar.safety.rta import (
    SCENARIO_ID,
    AfrlSetup,
    ExplicitOptimizationFilter,
    ExplicitSwitchingFilter,
    ImplicitOptimizationFilter,
    ImplicitSwitchingFilter,
)
from vbar.scenarios import SCENARIOS, make
from vbar.scenarios.catalog import AFRL_N
from vbar.scenarios.spec import Shell
from vbar.scenarios.suites import SUITES

PAPER_ID = "vbar:docking-afrl-paper:euler1:box:dv:v1"
CODE_ID = PAPER_ID + ":code"
GYM_IDS = {"vbar/DockingAfrlPaper-v1": PAPER_ID, "vbar/DockingAfrlPaperCode-v1": CODE_ID}
FILTERS = (
    ExplicitSwitchingFilter,
    ExplicitOptimizationFilter,
    ImplicitSwitchingFilter,
    ImplicitOptimizationFilter,
)


@pytest.mark.parametrize("gym_id", sorted(GYM_IDS))
def test_afrl_paper_scenarios_pass_check_env_and_start_on_the_paper_shell(gym_id):
    env = gym.make(gym_id)
    check_env(env.unwrapped, skip_render_check=True)
    assert env.unwrapped.scenario_id == GYM_IDS[gym_id]
    for seed in range(5):
        _, info = env.reset(seed=seed)
        r, v = info["true_state"][:3], info["true_state"][3:]
        assert np.linalg.norm(r) == pytest.approx(9850.0, rel=1e-12)
        assert np.linalg.norm(v) == pytest.approx(0.866, rel=1e-12)


def test_afrl_paper_entry_is_docking_afrl_with_the_papers_euler_plant():
    paper, afrl = SCENARIOS["docking-afrl-paper"], SCENARIOS["docking-afrl"]
    assert paper.version == 1
    assert paper.entry_point == afrl.entry_point
    config = paper.config
    assert config["dynamics"] == "euler1"
    assert config["dt"] == 1.0
    assert config["max_steps"] == 5000
    assert config["mass_kg"] == 12.0
    assert config["max_accel"] == pytest.approx(1.0 / 12.0, rel=1e-15)
    assert config.get("drive", "accel") == "accel"
    assert config["altitude_m"] == afrl.config["altitude_m"]
    assert paper.parameters["capture_radius"] == 1.0
    for name, value in afrl.parameters.items():
        if name != "capture_radius":
            assert paper.parameters[name] == value, name
    assert paper.dispersions["start_pos"] == Shell(9850.0, 9850.0)
    assert paper.dispersions["start_vel"] == Shell(0.866, 0.866)
    assert "reproduction" in paper.tags
    assert set(paper.variants) >= {"code"}


def test_afrl_paper_code_variant_changes_only_the_speed_limit_slope():
    paper = SCENARIOS["docking-afrl-paper"]
    code = paper.variant("code")
    assert code.parameters["full_speed_limit"] == pytest.approx((0.2, 2.15 * AFRL_N), rel=1e-15)
    assert paper.parameters["full_speed_limit"] == pytest.approx((0.2, 4.0 * AFRL_N), rel=1e-15)
    rest = {k: v for k, v in code.parameters.items() if k != "full_speed_limit"}
    assert rest == {k: v for k, v in paper.parameters.items() if k != "full_speed_limit"}
    assert code.config == paper.config
    assert code.dispersions == paper.dispersions


def test_afrl_paper_scenario_is_in_no_suite():
    for suite in SUITES.values():
        assert all(e.scenario.split(":")[0] != "docking-afrl-paper" for e in suite.entries)


def test_docking_afrl_keeps_its_v1_continuous_plant():
    afrl = SCENARIOS["docking-afrl"]
    assert afrl.version == 1
    assert afrl.config.get("dynamics", "cw") == "cw"
    assert afrl.config["dt"] == 5.0
    assert make("docking-afrl").scenario_id == "vbar:docking-afrl:cw:box:dv:v1"


def test_docking_afrl_can_be_flown_on_the_euler_rung():
    env = make("vbar:docking-afrl:euler1:box:dv:v1")
    assert env.cfg.dynamics == "euler1"
    _, info = env.reset(seed=3)
    x = np.array(info["true_state"], dtype=float)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    a, _ = cw_system(env.n)
    # docking-afrl decides every 5 s, so one coasting step is five 1 s Euler steps.
    for _ in range(5):
        x = x + a @ x
    np.testing.assert_allclose(info["true_state"], x, rtol=1e-12, atol=1e-12)


def test_setup_views_name_the_registered_scenarios():
    assert AfrlSetup.paper().view().scenario_id == PAPER_ID
    assert AfrlSetup.authors_code().view().scenario_id == CODE_ID
    custom = dataclasses.replace(AfrlSetup.paper(), nu1=3.0 * AFRL_N)
    assert custom.view().scenario_id == SCENARIO_ID


@pytest.mark.parametrize("cls", FILTERS)
def test_filters_accept_the_paper_scenario_only_with_its_own_setup(cls):
    paper_view = make(PAPER_ID).agent_view()
    code_view = make(CODE_ID).agent_view()
    cls(setup=AfrlSetup.paper()).reset(paper_view)
    cls(setup=AfrlSetup.authors_code()).reset(code_view)
    cls(setup=AfrlSetup.paper()).reset(AfrlSetup.paper().view())
    with pytest.raises(ValueError, match="v1"):
        cls(setup=AfrlSetup.authors_code()).reset(paper_view)
    with pytest.raises(ValueError, match="v1"):
        cls(setup=AfrlSetup.paper()).reset(code_view)


def _euler_episode(steps=5):
    env = make(PAPER_ID, max_steps=steps)
    return record_episode(env, DriftAgent(), seed=11, agent_seed=12)


def _cw_episode(steps=5):
    env = make("docking-afrl", max_steps=steps)
    return record_episode(env, DriftAgent(), seed=11, agent_seed=12), env.metric_settings()


def test_cw_interpolating_metrics_refuse_euler_trace():
    episode = _euler_episode()
    assert episode.dynamics == "euler1"
    with pytest.raises(ValueError):
        Trace(episode)
    Trace(_cw_episode()[0])


def test_cw_interpolating_metrics_refuse_euler_passive_metrics():
    episode = _euler_episode()
    with pytest.raises(ValueError):
        passive_metrics(episode)
    assert "passive_safety_margin" in passive_metrics(*_cw_episode())


def test_cw_interpolating_metrics_refuse_euler_evaluate():
    episode = _euler_episode()
    with pytest.raises(ValueError):
        evaluate(episode)
    assert "success" in evaluate(*_cw_episode())


@pytest.mark.parametrize("cls", FILTERS)
def test_afrl_filters_refuse_an_euler_view_at_another_step(cls):
    # docking-afrl decides every 5 s; its euler1 rung still steps the plant every 1 s, which the
    # filters' 5 s Euler prediction would not model.
    cw_view = make("docking-afrl").agent_view()
    euler_view = make("vbar:docking-afrl:euler1:box:dv:v1").agent_view()
    assert euler_view.dt == cw_view.dt == 5.0
    setup = dataclasses.replace(
        AfrlSetup.paper(),
        dt=cw_view.dt,
        mass_kg=cw_view.mass_kg,
        u_max=cw_view.max_accel * cw_view.mass_kg,
    )
    cls(setup=setup).reset(cw_view)
    with pytest.raises(ValueError, match="euler1"):
        cls(setup=setup).reset(euler_view)
