import dataclasses

import numpy as np
import pytest

from vbar.dynamics import models
from vbar.dynamics.cw import cw_system
from vbar.safety import PassThroughFilter
from vbar.safety.rta import AfrlSetup, ExplicitSwitchingFilter, reproduce, start_state
from vbar.scenarios import make

PAPER_ID = "vbar:docking-afrl-paper:euler1:box:dv:v1"
CODE_ID = PAPER_ID + ":code"


def _inline_euler(setup, x0, applied):
    a, b = cw_system(setup.n)
    states = [np.asarray(x0, dtype=float)]
    for u in applied:
        states.append(states[-1] + setup.dt * (a @ states[-1] + b @ u))
    return np.array(states)


@pytest.mark.parametrize("setup", [AfrlSetup.paper(), AfrlSetup.authors_code()])
def test_reproduce_matches_an_inline_euler_loop(setup):
    x0 = start_state(setup, 5)
    run = reproduce(ExplicitSwitchingFilter(setup=setup), setup, x0)
    assert len(run.applied) > 0
    np.testing.assert_array_equal(run.states, _inline_euler(setup, x0, run.applied))


def test_reproduce_plant_is_the_euler_rung_with_the_setups_step():
    setup = dataclasses.replace(AfrlSetup.paper(), stop_time=300.0)
    x0 = start_state(setup, 7)
    run = reproduce(PassThroughFilter(), setup, x0)
    rung = models.EulerCWDynamics(setup.n, step_s=setup.dt)
    rung.reset(x0)
    for k, u in enumerate(run.applied):
        state, _ = rung.advance(u, setup.dt)
        np.testing.assert_array_equal(state, run.states[k + 1])


def _replay(scenario_id, run, steps):
    env = make(scenario_id, training=True, action_mode="accel")
    _, info = env.reset(seed=0, options={"start_state": run.states[0]})
    flown = [info["true_state"]]
    for u in run.applied[:steps]:
        _, _, terminated, truncated, info = env.step(u)
        flown.append(info["true_state"])
        if terminated or truncated:
            break
    return np.array(flown)


def _assert_close(flown, states):
    # The environment's mean motion comes from the catalogue's altitude, a few ulps from
    # 0.001027; over a run that moves the state by far less than these tolerances.
    k = len(flown)
    assert np.max(np.linalg.norm(flown[:, :3] - states[:k, :3], axis=1)) <= 1e-6
    assert np.max(np.linalg.norm(flown[:, 3:] - states[:k, 3:], axis=1)) <= 1e-9


@pytest.mark.parametrize(("scenario_id", "setup"), [(PAPER_ID, AfrlSetup.paper())])
def test_env_replay_matches_reproduce(scenario_id, setup):
    run = reproduce(PassThroughFilter(), setup, start_state(setup, 5))
    flown = _replay(scenario_id, run, 600)
    assert len(flown) == 601
    _assert_close(flown, run.states)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("scenario_id", "setup"), [(PAPER_ID, AfrlSetup.paper()), (CODE_ID, AfrlSetup.authors_code())]
)
def test_env_replay_matches_reproduce_over_the_whole_run(scenario_id, setup):
    run = reproduce(ExplicitSwitchingFilter(setup=setup), setup, start_state(setup, 5))
    flown = _replay(scenario_id, run, len(run.applied))
    assert len(flown) >= len(run.states) - 1
    _assert_close(flown, run.states)
