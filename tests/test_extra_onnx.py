"""Without the onnx extra, the ONNX features name the command that installs it."""

import re
import sys

import pytest
from test_policy import observations, policy

from vbar.agents.policy import OnnxAgent, export_onnx, profile_latency

INSTALL = re.escape('pip install "vbar[onnx]"')


def test_export_without_onnx_names_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(ModuleNotFoundError, match=f"needs onnx: {INSTALL}"):
        export_onnx(policy())


def test_inference_without_onnxruntime_names_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ModuleNotFoundError, match=f"needs onnxruntime: {INSTALL}"):
        OnnxAgent(b"")
    with pytest.raises(ModuleNotFoundError, match=f"needs onnxruntime: {INSTALL}"):
        profile_latency(b"", observations(10))
