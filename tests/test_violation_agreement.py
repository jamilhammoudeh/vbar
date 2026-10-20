import pytest

from vbar import ApproachConfig, ApproachEnv
from vbar.agents import PDAgent
from vbar.metrics import Trace, merge, negative_intervals, record_episode


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
