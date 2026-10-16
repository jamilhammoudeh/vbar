import numpy as np
import pytest
from metric_cases import FLYBY, ConstantAgent, drift_range_crossings, episode

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import LLMAgent, PDAgent
from vbar.constraints import KeepOut
from vbar.dynamics import TwoBodyDynamics, cw_discrete
from vbar.metrics import Signal, Trace, complement, merge, negative_intervals, record_episode, total
from vbar.safety import FilteredAgent


class HalvingFilter:
    name = "halve"

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return 0.5 * action


def test_recorded_states_match_run_episode_for_the_same_seeds():
    config = ApproachConfig(max_steps=30)
    recorded = record_episode(ApproachEnv(config), PDAgent(), seed=4, agent_seed=9)
    ran = run_episode(ApproachEnv(config), PDAgent(), seed=4, agent_seed=9)
    assert np.array_equal(recorded.states, ran["states"])
    assert recorded.hard == frozenset({"keep_out"})
    assert recorded.agent_seed == 9 and recorded.truncated and not recorded.terminated


def test_records_keep_the_raw_command_and_the_filter_result():
    env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0), max_steps=4))
    ep = record_episode(env, FilteredAgent(PDAgent(), HalvingFilter()), seed=0, agent_seed=0)
    for rec in ep.records:
        assert np.allclose(rec.filtered, 0.5 * rec.command)
        assert rec.filter_result.intervened
        assert rec.agent_latency_s >= 0.0


def test_an_llm_agent_log_is_kept():
    reply = '{"ax": 0.1, "ay": 0.0, "az": 0.0}'
    ep = episode(LLMAgent(lambda prompt: reply), max_steps=3)
    assert len(ep.agent_log) == 3 and not ep.agent_log[0]["parse_failed"]


def test_trace_returns_the_samples_at_sample_times():
    ep = episode(ConstantAgent([0.3, -0.2, 0.1]), max_steps=3)
    trace = Trace(ep)
    times, states = trace.samples()
    for k in (0, 7, 19, len(times) - 1):
        assert np.array_equal(trace.state_at(times[k]), states[k])


def test_trace_matches_the_cw_closed_form_between_samples():
    command = np.array([0.3, -0.2, 0.1])
    ep = episode(ConstantAgent(command), max_steps=3)
    trace, n = Trace(ep), ep.view.n
    accel = command * ep.view.max_accel
    for t in np.random.default_rng(0).uniform(0.0, 30.0, 25):
        k = int(t // 10.0)
        phi, gamma = cw_discrete(n, t - 10.0 * k)
        exact = phi @ ep.states[k] + gamma @ accel
        # Same closed form from a different starting sample; differences are rounding only.
        assert np.allclose(trace.state_at(t), exact, rtol=0.0, atol=1e-9 * np.abs(exact).max())


@pytest.mark.parametrize(
    ("dynamics", "tol"),
    # Without J2 only the two-body nonlinearity is dropped (rho^2 / r); with J2 the Hill frame
    # turns faster than n by 1.5 J2 (R/a)^2 n, a Coriolis difference of about 5e-6 m/s^2 at
    # 1.5 m/s held for up to 0.53 s. Both far below the 1 mm resolution of range constraints.
    [("twobody", 1e-8), ("twobody_j2", 5e-6)],
)
def test_trace_follows_the_two_body_models_between_samples(dynamics, tol):
    env = ApproachEnv(ApproachConfig(start_state=FLYBY, max_steps=1, dynamics=dynamics))
    ep = record_episode(env, ConstantAgent([0.5, 0.5, -0.5]), seed=0, agent_seed=0)
    model: TwoBodyDynamics = env.dynamics
    trace = Trace(ep)
    for tau in np.linspace(0.13, 9.91, 17):
        gap = trace.state_at(tau) - model.state_in_last_segment(tau)
        assert np.linalg.norm(gap[:3]) < tol


def test_keep_out_intervals_match_the_closed_form_crossings():
    ep = episode(max_steps=20)
    entry, exit_ = drift_range_crossings(FLYBY, ep.view.n, 50.0, 200.0)
    found = negative_intervals(KeepOut(50.0), Trace(ep))
    assert len(found) == 1
    assert found[0][0] == pytest.approx(entry, abs=1e-6)
    assert found[0][1] == pytest.approx(exit_, abs=1e-6)


def test_keep_out_intervals_do_not_depend_on_the_decision_step():
    reference = negative_intervals(KeepOut(50.0), Trace(episode(max_steps=20)))
    for dt in (3.7, 25.0):
        found = negative_intervals(KeepOut(50.0), Trace(episode(dt=dt, max_steps=int(200 / dt))))
        assert np.allclose(found, reference, atol=1e-6)


def test_a_dip_between_two_positive_samples_is_found():
    # A 12 m/s pass that dips 5 m into the keep-out zone between two samples 10 s apart.
    ep = episode(start_state=(0.0, 60.0, 45.0, 0.0, -12.0, 0.0), max_steps=1, violation_samples=2)
    found = negative_intervals(KeepOut(50.0), Trace(ep))
    entry, exit_ = drift_range_crossings((0.0, 60.0, 45.0, 0.0, -12.0, 0.0), ep.view.n, 50.0, 10.0)
    assert np.allclose(found, [(entry, exit_)], atol=1e-6)


def test_a_rise_above_zero_between_two_negative_samples_splits_the_interval():
    # A signal negative at both samples, 10 s apart, that is positive between 4 and 6 s.
    ep = episode(start_state=(0.0, 400.0, 0.0, 0.0, 0.0, 0.0), max_steps=1, violation_samples=2)
    bump = Signal("bump", lambda s, t: 1.0 - (np.asarray(t) - 5.0) ** 2, lambda v, a, r: 20.0, 1e-9)
    found = negative_intervals(bump, Trace(ep))
    assert np.allclose(found, [(0.0, 4.0), (6.0, 10.0)], atol=1e-6)


def test_merge_complement_and_total():
    merged = merge([(3.0, 4.0), (0.0, 1.0), (1.0, 2.0), (3.5, 5.0)])
    assert merged == [(0.0, 2.0), (3.0, 5.0)]
    assert complement(merged, 0.0, 10.0) == [(2.0, 3.0), (5.0, 10.0)]
    assert total(merged) == 4.0
