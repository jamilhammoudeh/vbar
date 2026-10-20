import math

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv
from vbar.agents import DriftAgent, PDAgent
from vbar.metrics import Trace, merge, negative_intervals, record_episode
from vbar.rewards import Reward


@pytest.mark.parametrize("seed", range(8))
def test_metrics_and_environment_find_the_same_keep_out_violations(seed):
    # The metrics bracket roots on the rebuilt trajectory and the environment on each step's
    # samples; both locate boundaries to about 1e-9 s.
    ep = record_episode(ApproachEnv(ApproachConfig()), PDAgent(), seed=seed, agent_seed=seed)
    keep_out = ep.constraints[0]
    from_metrics = negative_intervals(keep_out, Trace(ep))
    from_env = merge(
        [s for rec in ep.records for s in rec.segment.violations.get(keep_out.name, ())]
    )
    assert from_metrics, "the default PD run should enter the keep-out zone"
    assert len(from_metrics) == len(from_env)
    for (a0, a1), (b0, b1) in zip(from_metrics, from_env, strict=True):
        assert a0 == pytest.approx(b0, abs=1e-8) and a1 == pytest.approx(b1, abs=1e-8)


def test_both_find_two_dips_in_one_long_step():
    # A natural-motion ellipse of radial amplitude 100 m skims a keep-out sphere 1e-4 larger
    # twice in one 3000 s step; the dips lie between samples, around the refined minimum and away
    # from it.
    b = 100.0
    n = ApproachEnv(ApproachConfig()).n
    phi = -0.1
    start = (
        b * math.cos(phi),
        -2 * b * math.sin(phi),
        0.0,
        -b * n * math.sin(phi),
        -2 * b * n * math.cos(phi),
        0.0,
    )
    reward = Reward(violation_weights={"keep_out": 1.0}, entry_weights={"keep_out": 1.0})
    config = ApproachConfig(
        dt=3000.0,
        max_steps=1,
        keep_out_radius=b * (1 + 1e-4),
        start_state=start,
        capture_radius=1e-3,
        reward=reward,
    )
    ep = record_episode(ApproachEnv(config), DriftAgent(), seed=0, agent_seed=1)
    keep_out = ep.constraints[0]
    from_metrics = negative_intervals(keep_out, Trace(ep))
    from_env = list(ep.records[0].segment.violations[keep_out.name])
    assert len(from_metrics) == len(from_env) == 2
    for (a0, a1), (b0, b1) in zip(from_metrics, from_env, strict=True):
        assert a0 == pytest.approx(b0, abs=1e-8) and a1 == pytest.approx(b1, abs=1e-8)
    env = ApproachEnv(config)
    env.reset(seed=0)
    _, r, *_ = env.step(np.zeros(3))
    inside = sum(hi - lo for lo, hi in from_metrics)
    # One unit per second inside the zone and one per entry, both charged by the reward.
    assert r == pytest.approx(-(inside + 2.0), rel=1e-9)
