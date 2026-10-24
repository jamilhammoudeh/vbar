"""Each metric detects the defect it is responsible for and ranks severities in order."""

import numpy as np
import pytest
from metric_cases import episode
from mutants import Mutant

from vbar.agents import PDAgent
from vbar.metrics import (
    MetricSettings,
    composite_metrics,
    efficiency_metrics,
    filter_metrics,
    mission_metrics,
    passive_metrics,
    safety_metrics,
)
from vbar.safety import FilteredAgent, PassThroughFilter

BEHIND = (0.0, -400.0, 30.0, 0.0, 0.0, 0.0)
TARGET_IN_THE_WAY = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)


def run(agent, start=BEHIND):
    return episode(agent, start_state=start, max_steps=200)


def increasing(values):
    return all(a < b for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize(
    ("kind", "levels"),
    [("position_bias", (2.0, 10.0, 40.0)), ("velocity_bias", (0.01, 0.03, 0.1))],
)
def test_navigation_biases_raise_the_terminal_error_with_their_size(kind, levels):
    eps = [run(PDAgent())] + [run(Mutant(kind, s)) for s in levels]
    errors = [mission_metrics(ep)["terminal_pos_error"] for ep in eps]
    scores = [composite_metrics(ep)["composite_score"] for ep in eps]
    assert increasing(errors)
    assert all(a >= b for a, b in zip(scores, scores[1:], strict=False)) and scores[-1] < scores[0]


def test_decision_delay_raises_delta_v_with_its_length():
    dv = [efficiency_metrics(run(Mutant("delay", k)))["delta_v"] for k in (0, 1, 3, 6)]
    assert increasing(dv)


def test_dither_raises_delta_v_along_scaled_copies_of_one_noise_law():
    # Zero-mean dither cannot lower expected fuel (Jensen; Boyd and Vandenberghe 3.1.8), and along
    # t v for one fixed v the cost rises with t (exercise 3.10); the levels share one sequence of
    # draws.
    dv = [efficiency_metrics(run(Mutant("dither", s)))["delta_v"] for s in (0.0, 0.05, 0.15, 0.45)]
    assert increasing(dv)


def test_alternating_commands_raise_chatter_with_their_amplitude():
    chatter = [
        efficiency_metrics(run(Mutant("chatter", s)))["chatter"] for s in (0.0, 0.02, 0.1, 0.4)
    ]
    assert increasing(chatter)


def test_a_sign_flip_fails_with_no_progress():
    flipped = mission_metrics(run(Mutant("sign_flip", 1)))
    assert mission_metrics(run(PDAgent()))["success"]
    assert not flipped["success"] and flipped["progress"] == 0.0


def test_ignoring_the_keep_out_zone_is_caught_by_its_margin():
    straight = run(PDAgent(), TARGET_IN_THE_WAY)
    assert safety_metrics(straight)["min_margin"]["keep_out"] < 0.0
    assert safety_metrics(run(PDAgent()))["min_margin"]["keep_out"] > 0.0
    assert composite_metrics(straight)["infractions"]["keep_out"] == 1


def test_a_more_aggressive_approach_lowers_passive_safety():
    # Higher PD gains close faster, so drift after a thrust failure comes closer to the target.
    settings = MetricSettings(partial_burn_samples=0)
    found = [passive_metrics(run(PDAgent(wn=wn)), settings) for wn in (0.005, 0.01, 0.02, 0.04)]
    margins = [m["passive_safety_margin"] for m in found]
    stuck = [m["stuck_thruster_margin"] for m in found]
    exposed = [m["unprotected_time"] for m in found]
    assert increasing(margins[::-1]) and increasing(stuck[::-1])
    assert all(a <= b for a, b in zip(exposed, exposed[1:], strict=False)) and exposed[-1] > 0.0
    for name in ("passive_safe_fraction", "passive_forever_fraction"):
        shares = [m[name] for m in found]
        assert all(a >= b for a, b in zip(shares, shares[1:], strict=False)) and shares[-1] < 1.0
    assert found[0]["passive_threat_time"] == np.inf and found[-1]["passive_threat_time"] < np.inf


def test_a_filter_that_changes_every_command_is_seen_by_the_intervention_metrics():
    class Halving:
        name = "halve"

        def reset(self, view):
            pass

        def filter(self, obs, action):
            return 0.5 * action

    passing = filter_metrics(run(FilteredAgent(PDAgent(), PassThroughFilter())))
    halving = filter_metrics(run(FilteredAgent(PDAgent(), Halving())))
    assert passing["intervention_rate"] == 0.0 and np.isnan(passing["intervention_size"])
    assert halving["intervention_rate"] == 1.0 and halving["intervention_size"] > 0.0
