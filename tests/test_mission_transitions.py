"""Mode changes cut committed coasts, a contingency needs a time limit, and the abort predicate
reads the episode time."""

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.base import DriftAgent
from vbar.agents.mission import MissionManagerAgent, Mode

REST = (0.0, -300.0, 0.0, 0.0, 0.0, 0.0)


class CoastAway(DriftAgent):
    """Commits a 300 s coast at every decision."""

    name = "coast-away"

    def act(self, obs, info):
        return np.array([0.0, 0.0, 0.0, 300.0])


def test_a_next_transition_cuts_the_coast_the_leaving_mode_committed():
    manager = MissionManagerAgent(
        [
            Mode(name="leave", agent=CoastAway(), exit=lambda obs, t: True, next="final"),
            Mode(name="final", agent=DriftAgent()),
        ]
    )
    cfg = ApproachConfig(action_mode="impulse_coast", start_state=REST, max_steps=5)
    result = run_episode(ApproachEnv(cfg), manager, seed=1, agent_seed=2)
    assert manager.log[0]["transition"] == "next"
    assert result["actions"][0][3] == 0.0
    assert [entry["mode"] for entry in manager.log] == ["leave"] + ["final"] * 4


def test_a_contingency_without_a_time_limit_is_refused():
    with pytest.raises(ValueError):
        MissionManagerAgent(
            [
                Mode(name="hold", agent=DriftAgent(), contingency="final", next="final"),
                Mode(name="final", agent=DriftAgent()),
            ]
        )


def test_the_abort_predicate_reads_the_episode_time_not_the_time_in_mode():
    # "second" is entered at 30 s. The predicate t > 45 s holds first at the 50 s decision by the
    # episode clock; by the mode clock it would hold only at 80 s.
    manager = MissionManagerAgent(
        [
            Mode(name="first", agent=DriftAgent(), exit=lambda obs, t: t > 15.0, next="second"),
            Mode(name="second", agent=DriftAgent(), next="final", abort="safe"),
            Mode(name="safe", agent=DriftAgent(), next="final"),
            Mode(name="final", agent=DriftAgent()),
        ],
        abort_when=lambda obs, t: t > 45.0,
    )
    cfg = ApproachConfig(start_state=REST, max_steps=10)
    run_episode(ApproachEnv(cfg), manager, seed=1, agent_seed=2)
    assert [entry["mode"] for entry in manager.log] == (
        ["first"] * 3 + ["second"] * 3 + ["safe"] * 4
    )
    assert manager.log[5]["transition"] == "abort"
