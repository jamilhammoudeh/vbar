"""ONNX export of policies; skipped without the onnx extra."""

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.policy import PolicyAgent

onnx = pytest.importorskip("onnx", reason="needs the onnx extra")
pytest.importorskip("onnxruntime", reason="needs the onnx extra")

from test_policy import observations, policy  # noqa: E402

from vbar.agents.policy import OnnxAgent, export_onnx, profile_latency  # noqa: E402


@pytest.mark.parametrize("options", [{}, {"activation": "relu", "squash": "clip"}])
def test_export_round_trip_reproduces_the_policy(options, tmp_path):
    p = policy(**options)
    obs = observations()
    path = tmp_path / "policy.onnx"
    export_onnx(p, path)
    session = OnnxAgent(path).session
    single = session.run(None, {"obs": obs.astype(np.float32)})[0]
    # The graph does the numpy policy's arithmetic in float32: agreement to float32 rounding.
    assert np.allclose(single, p(obs, np.float32), rtol=0.0, atol=2e-6)
    assert np.allclose(single, p(obs), rtol=0.0, atol=2e-5)
    double = OnnxAgent(export_onnx(p, dtype="float64")).session
    assert np.allclose(double.run(None, {"obs": obs})[0], p(obs), rtol=0.0, atol=1e-12)
    assert onnx.load(path).producer_name == "vbar"


def test_onnx_agent_commands_what_the_policy_commands():
    p = policy()
    cfg = ApproachConfig(start_state=(0.0, 400.0, 0.0, 0.0, 0.0, 0.0), max_steps=30)
    flown = run_episode(ApproachEnv(cfg), PolicyAgent(p))
    agent = OnnxAgent(export_onnx(p, dtype="float64"))
    agent.reset(ApproachEnv(cfg).agent_view())
    actions = [agent.act(obs, {}) for obs in flown["states"][:-1]]
    assert np.allclose(actions, flown["actions"], rtol=0.0, atol=1e-12)


def test_latency_profile_reports_the_machine_and_the_timings():
    report = profile_latency(export_onnx(policy()), observations(200), warmup=10)
    assert report["profile"] == "single-thread" and report["runs"] == 200
    assert 0.0 < report["median_s"] <= report["p95_s"] <= report["max_s"] < 0.1
    assert {"machine", "processor", "cpu_count", "python", "onnxruntime"} <= set(report)
    with pytest.raises(ValueError, match="unknown profile"):
        profile_latency(export_onnx(policy()), observations(5), profile="flight")
