import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import DriftAgent, LLMAgent, PDAgent, parse_action
from vbar.safety import FilteredAgent, FilterResult, PassThroughFilter


def test_gymnasium_api_compliance():
    check_env(ApproachEnv(), skip_render_check=True)


def test_reset_is_reproducible_with_seed():
    env = ApproachEnv()
    a, _ = env.reset(seed=3)
    b, _ = env.reset(seed=3)
    assert np.allclose(a, b)


def test_dv_accounting():
    env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0)))
    env.reset()
    _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0]))
    assert info["dv_used_mps"] == pytest.approx(env.cfg.max_accel * env.cfg.dt)


def test_drift_on_vbar_stays_safe():
    env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0)))
    r = run_episode(env, DriftAgent())
    assert not r["keep_out_violated"]
    assert r["dv_used_mps"] == 0.0


def test_pd_reaches_hold_point_on_clear_path():
    env = ApproachEnv(ApproachConfig(start_state=(-150, -400, 0, 0, 0, 0)))
    r = run_episode(env, PDAgent())
    assert r["captured"] and not r["keep_out_violated"]


def test_pd_violates_keep_out_when_target_is_in_the_way():
    # Documents the motivating failure: a naive controller flies straight
    # through the keep-out zone. A safety filter should prevent this.
    env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0)))
    r = run_episode(env, PDAgent())
    assert r["keep_out_violated"]


def test_violation_detected_between_steps():
    # Fast pass through the keep-out zone within a single step must be caught.
    cfg = ApproachConfig(start_state=(0, 60, 0, 0, -12, 0), dt=10.0)
    env = ApproachEnv(cfg)
    env.reset()
    _, _, _, _, info = env.step(np.zeros(3))
    assert info["range_to_target_m"] > cfg.keep_out_radius  # end point is outside...
    assert info["keep_out_violated"]  # ...but the path went through


def test_parse_action():
    assert np.allclose(parse_action('{"ax": 0.1, "ay": -2, "az": 0}'), [0.1, -1.0, 0.0])
    assert parse_action("no json here") is None
    assert parse_action('{"ax": "fast"}') is None


def test_llm_agent_with_fake_model_logs_everything():
    env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0), max_steps=5))
    agent = LLMAgent(lambda prompt: '{"ax": 0, "ay": -0.2, "az": 0}', name="fake-llm")
    r = run_episode(env, agent)
    assert len(agent.log) == 5
    assert not any(e["parse_failed"] for e in agent.log)
    assert r["dv_used_mps"] > 0


def test_filtered_agent_counts_interventions():
    env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0), max_steps=5))
    agent = FilteredAgent(PDAgent(), PassThroughFilter())
    r = run_episode(env, agent)
    assert r["filter_interventions"] == 0


class HalvingFilter:
    """Halves every command; reports itself as certified infeasible when asked to."""

    name = "halve"

    def __init__(self, status="feasible"):
        self.status = status

    def reset(self, view):
        pass

    def filter(self, obs, action):
        if self.status == "feasible":
            return 0.5 * action
        return FilterResult(0.5 * action, True, self.status, slack=0.1, conflicting=("keep_out",))


def test_filtered_agent_keeps_the_raw_command_and_a_result_for_a_bare_filter():
    env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0), max_steps=3))
    agent = FilteredAgent(PDAgent(), HalvingFilter())
    r = run_episode(env, agent)
    assert r["filter_interventions"] == 3
    assert agent.last_result.status == "feasible" and agent.last_result.intervened
    assert np.allclose(agent.last_result.action, 0.5 * agent.last_command)
    assert agent.last_result.compute_s >= 0.0


def test_filtered_agent_passes_a_filter_result_through():
    env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0), max_steps=2))
    agent = FilteredAgent(PDAgent(), HalvingFilter("infeasible_certified"))
    run_episode(env, agent)
    assert agent.last_result.status == "infeasible_certified"
    assert agent.last_result.conflicting == ("keep_out",)


def test_filter_result_rejects_an_unknown_status():
    with pytest.raises(ValueError, match="status"):
        FilterResult(np.zeros(3), False, "maybe")


def test_filter_result_rejects_an_unknown_fallback_level_or_a_horizon_that_is_not_positive():
    with pytest.raises(ValueError, match="fallback level"):
        FilterResult(np.zeros(3), False, fallback_level=3)
    for horizon in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError, match="horizon"):
            FilterResult(np.zeros(3), False, certificate_horizon=horizon)
    assert FilterResult(np.zeros(3), False, certificate_horizon=float("inf")).certificate_horizon
