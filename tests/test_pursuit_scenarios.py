import pytest
from test_pursuit_evasion import FLOW, NaivePursuitAgent

from vbar.games import (
    AgentEnvRunner,
    NormalEscapeEvader,
    PassiveEvader,
    PE1_E1_I3_Env,
    PE1_E3_I3_Env,
)


def test_the_scenarios_fly_the_i3_start_against_their_bots():
    assert isinstance(PE1_E1_I3_Env().evader, PassiveEvader)
    assert isinstance(PE1_E3_I3_Env().evader, NormalEscapeEvader)
    assert PE1_E3_I3_Env().reset()[0] == PE1_E1_I3_Env().reset()[0]


def test_kspdg_example_runner_call_works_as_written():
    runner = AgentEnvRunner(
        agent=NaivePursuitAgent(),
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=100,
        debug=True,
        enable_telemetry=False,
    )
    results = runner.run()["agent_env_results"]
    assert results["is_episode_done"] and results["steps"] == 240
    assert results["pursuer_fuel_usage"] == pytest.approx(240 * FLOW, rel=1e-12)
