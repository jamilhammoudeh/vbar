import math

import numpy as np
import pytest

from vbar.conventions import G0
from vbar.games import SCENARIOS, PE1_E4_I3_Env, ProgradeEvader, fly

FLOW = 0.2 * 8000.0 / (240.0 * G0)  # kg/s at a fifth of the forward thrust


class Coast:
    def get_action(self, observation):
        return [0.0, 0.0, 0.0, 0.25]


def test_the_burn_starts_after_half_a_second_at_a_fifth_of_forward_thrust():
    env = PE1_E4_I3_Env()
    env.reset()
    assert env.step([0.0, 0.0, 0.0, 0.25])[3]["evader_fuel_usage"] == 0.0
    assert env.step([0.0, 0.0, 0.0, 0.25])[3]["evader_fuel_usage"] == 0.0
    info = env.step([0.0, 0.0, 0.0, 0.3])[3]
    assert info["evader_fuel_usage"] == pytest.approx(0.3 * FLOW, rel=1e-12)
    info = fly(PE1_E4_I3_Env(), Coast())
    assert info["evader_fuel_usage"] == pytest.approx(239.5 * FLOW, rel=1e-12)


def test_the_burn_pushes_prograde_by_the_rocket_equation_whatever_the_pursuer_does():
    coast = SCENARIOS["pe1-e1-i3"]()
    start = np.array(coast.reset()[0])
    runs = []
    for throttle in (0.0, 1.0):
        env = PE1_E4_I3_Env()
        env.reset()
        for _ in range(21):
            observation = env.step([throttle, 0.0, 0.0, 0.5])[0]
        runs.append(np.array(observation))
    for _ in range(21):
        passive = np.array(coast.step([0.0, 0.0, 0.0, 0.5])[0])
    # The pursuer's burns change nothing in the evader's flight.
    assert np.allclose(runs[0][9:15], runs[1][9:15], rtol=0, atol=1e-6)
    dv = runs[0][12:15] - passive[12:15]
    wet = 6685.0
    # The twins drift about 12 m apart in the 10 s, and the gravity gradient across that adds
    # about 1e-3 m/s, 5e-4 of the 2.39 m/s.
    assert np.linalg.norm(dv) == pytest.approx(
        240.0 * G0 * math.log(wet / (wet - 10.0 * FLOW)), rel=1e-3
    )
    # Along the velocity, to the turn of the orbit in 10 s (about 0.03 rad).
    v = start[12:15]
    assert dv @ v / (np.linalg.norm(dv) * np.linalg.norm(v)) > math.cos(0.03)
    # Over the whole episode the orbit rises: about 57 m/s prograde adds tens of kilometres.
    env = PE1_E4_I3_Env()
    env.reset()
    done = False
    while not done:
        end, _, done, _ = env.step([0.0, 0.0, 0.0, 10.0])
    mu = 3.5316e12
    sma = [
        1.0 / (2.0 / np.linalg.norm(o[9:12]) - np.dot(o[12:15], o[12:15]) / mu)
        for o in (start, end)
    ]
    assert 20e3 < sma[1] - sma[0] < 60e3


def test_every_e4_start_is_flown_against_the_prograde_evader():
    keys = [key for key in SCENARIOS if key.startswith("pe1-e4-")]
    assert sorted(keys) == [f"pe1-e4-i{i}" for i in (1, 2, 3, 4)]
    assert all(isinstance(SCENARIOS[key]().evader, ProgradeEvader) for key in keys)


@pytest.mark.parametrize(
    "kwargs", [{"throttle": 1.5}, {"throttle": -0.1}, {"delay": -1.0}, {"delay": float("nan")}]
)
def test_bad_settings_of_the_prograde_evader_are_refused(kwargs):
    with pytest.raises(ValueError):
        ProgradeEvader(**kwargs)
