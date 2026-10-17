"""Neural policies and their export to ONNX with normalization and squashing built in.

`MLPPolicy` is a multilayer perceptron in numpy: the observation is normalized, passed through
the hidden layers, and the output squashed to fractions of a full command. `export_onnx` writes
the same computation as one ONNX graph, so a policy trained in any framework (its weights copied
into an `MLPPolicy`) is flown from a single file with no preprocessing left outside it, the
mismatch Sutton and Barto's input-scaling examples warn about (2nd ed. draft, 2015, secs. 14.1
and 14.5). `OnnxAgent` flies an exported file with onnxruntime, and `profile_latency` measures
its inference time under fixed CPU profiles. ONNX support needs the optional `onnx` extra.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .base import AgentView, action_scale

ACTIVATIONS = ("tanh", "relu")
SQUASHES = ("tanh", "clip")
# Execution settings named so latencies can be compared between runs on one machine.
PROFILES = {
    "single-thread": {"intra_op_threads": 1, "inter_op_threads": 1},
    "all-threads": {"intra_op_threads": 0, "inter_op_threads": 0},
}
# Opset 17 with IR version 8 loads in every onnxruntime since 1.13; newer onnx releases default
# to IR versions that older runtimes refuse.
OPSET, IR_VERSION = 17, 8


@dataclass(frozen=True)
class MLPPolicy:
    """Fractions of a full command from an observation, (..., d) to (..., 3).

    `weights` are (inputs, outputs) matrices and `biases` vectors, one pair per layer; hidden
    layers use `activation`, the last layer's output is squashed by tanh or clipped to [-1, 1].
    The observation is normalized as (obs - obs_mean) / obs_std first.
    """

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    obs_mean: np.ndarray
    obs_std: np.ndarray
    activation: str = "tanh"
    squash: str = "tanh"
    name: str = field(default="mlp", compare=False)

    def __post_init__(self):
        if self.activation not in ACTIVATIONS or self.squash not in SQUASHES:
            raise ValueError(f"activation is one of {ACTIVATIONS}, squash one of {SQUASHES}")
        if not self.weights or len(self.weights) != len(self.biases):
            raise ValueError("give one bias per weight matrix")
        width = len(self.obs_mean)
        if np.shape(self.obs_std) != (width,) or not np.all(np.asarray(self.obs_std) > 0.0):
            raise ValueError("obs_std must be positive and match obs_mean")
        for w, b in zip(self.weights, self.biases, strict=True):
            if np.shape(w)[0] != width or np.shape(b) != (np.shape(w)[1],):
                raise ValueError("layer shapes do not chain")
            width = np.shape(w)[1]
        if width != 3:
            raise ValueError("the last layer must have 3 outputs")

    @classmethod
    def random(
        cls,
        obs_mean,
        obs_std,
        hidden: tuple[int, ...] = (64, 64),
        rng: np.random.Generator | None = None,
        **options,
    ) -> MLPPolicy:
        """Glorot-uniform weights and zero biases, for tests and examples."""
        rng = rng or np.random.default_rng()
        sizes = [len(obs_mean), *hidden, 3]
        weights, biases = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            biases.append(np.zeros(fan_out))
        mean, std = np.asarray(obs_mean, float), np.asarray(obs_std, float)
        return cls(tuple(weights), tuple(biases), mean, std, **options)

    def __call__(self, obs, dtype=np.float64) -> np.ndarray:
        """The policy's output computed in `dtype`, the arithmetic of an export in that type."""
        x = (np.asarray(obs, dtype) - np.asarray(self.obs_mean, dtype)) / np.asarray(
            self.obs_std, dtype
        )
        last = len(self.weights) - 1
        for k, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ np.asarray(w, dtype) + np.asarray(b, dtype)
            if k < last:
                x = np.tanh(x) if self.activation == "tanh" else np.maximum(x, 0.0)
        return np.tanh(x) if self.squash == "tanh" else np.clip(x, -1.0, 1.0)


class PolicyAgent:
    """Flies an `MLPPolicy` in process, in any continuous action mode."""

    def __init__(self, policy: MLPPolicy, name: str | None = None):
        self.policy = policy
        self.name = name or policy.name

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        self.scale = action_scale(view)

    def act(self, obs, info):
        return self.policy(obs) * self.scale


def export_onnx(policy: MLPPolicy, path: str | Path | None = None, dtype: str = "float32") -> bytes:
    """The policy as an ONNX model, input "obs" (N, d) and output "action" (N, 3) fractions.

    Normalization, every layer and the squashing are in the graph; weights are stored in `dtype`
    ("float32" or "float64"). Returns the serialized model and writes it to `path` if given.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if dtype not in ("float32", "float64"):
        raise ValueError("dtype is float32 or float64")
    elem = TensorProto.FLOAT if dtype == "float32" else TensorProto.DOUBLE
    width = len(policy.obs_mean)

    def constant(name: str, value) -> onnx.TensorProto:
        return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)

    inits = [constant("obs_mean", policy.obs_mean), constant("obs_std", policy.obs_std)]
    nodes = [
        helper.make_node("Sub", ["obs", "obs_mean"], ["shifted"]),
        helper.make_node("Div", ["shifted", "obs_std"], ["layer_0"]),
    ]
    last = len(policy.weights) - 1
    for k, (w, b) in enumerate(zip(policy.weights, policy.biases, strict=True)):
        inits += [constant(f"w_{k}", w), constant(f"b_{k}", b)]
        nodes += [
            helper.make_node("MatMul", [f"layer_{k}", f"w_{k}"], [f"product_{k}"]),
            helper.make_node("Add", [f"product_{k}", f"b_{k}"], [f"sum_{k}"]),
        ]
        if k < last:
            op = "Tanh" if policy.activation == "tanh" else "Relu"
            nodes.append(helper.make_node(op, [f"sum_{k}"], [f"layer_{k + 1}"]))
    if policy.squash == "tanh":
        nodes.append(helper.make_node("Tanh", [f"sum_{last}"], ["action"]))
    else:
        inits += [constant("low", -1.0), constant("high", 1.0)]
        nodes.append(helper.make_node("Clip", [f"sum_{last}", "low", "high"], ["action"]))
    graph = helper.make_graph(
        nodes,
        policy.name,
        [helper.make_tensor_value_info("obs", elem, ["batch", width])],
        [helper.make_tensor_value_info("action", elem, ["batch", 3])],
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION
    )
    model.producer_name = "vbar"
    onnx.checker.check_model(model)
    data = model.SerializeToString()
    if path is not None:
        Path(path).write_bytes(data)
    return data


def _session(model: str | Path | bytes, profile: str):
    import onnxruntime as ort

    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}, expected {tuple(PROFILES)}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = PROFILES[profile]["intra_op_threads"]
    options.inter_op_num_threads = PROFILES[profile]["inter_op_threads"]
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    source = model if isinstance(model, bytes) else str(model)
    return ort.InferenceSession(source, options, providers=["CPUExecutionProvider"])


class OnnxAgent:
    """Flies an exported policy (a file path or the serialized model) with onnxruntime."""

    def __init__(self, model: str | Path | bytes, profile: str = "single-thread", name="onnx"):
        self.session = _session(model, profile)
        self.input = self.session.get_inputs()[0]
        self.dtype = np.float64 if self.input.type == "tensor(double)" else np.float32
        self.name = name

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        self.scale = action_scale(view)

    def act(self, obs, info):
        batch = np.asarray(obs, dtype=self.dtype)[None, :]
        fraction = self.session.run(None, {self.input.name: batch})[0][0]
        return np.asarray(fraction, dtype=float) * self.scale


def profile_latency(
    model: str | Path | bytes,
    observations: np.ndarray,
    profile: str = "single-thread",
    warmup: int = 50,
) -> dict[str, Any]:
    """Wall time (s) of one inference on each row of `observations`, under a CPU profile.

    Returns the median, 95th percentile and maximum with the profile and the machine, Python and
    onnxruntime versions, since latencies compare only within one machine.
    """
    import onnxruntime as ort

    session = _session(model, profile)
    name = session.get_inputs()[0].name
    dtype = np.float64 if session.get_inputs()[0].type == "tensor(double)" else np.float32
    rows = np.asarray(observations, dtype=dtype)
    for k in range(warmup):
        session.run(None, {name: rows[k % len(rows)][None, :]})
    times = []
    for row in rows:
        t0 = time.perf_counter()
        session.run(None, {name: row[None, :]})
        times.append(time.perf_counter() - t0)
    return {
        "profile": profile,
        "runs": len(times),
        "median_s": float(np.median(times)),
        "p95_s": float(np.percentile(times, 95)),
        "max_s": float(np.max(times)),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
    }
