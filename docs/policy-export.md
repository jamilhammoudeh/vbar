# Policy export

Code: `vbar.agents.policy` (`MLPPolicy`, `PolicyAgent`, `export_onnx`, `OnnxAgent`,
`profile_latency`). A learned policy is only reproducible if everything between the
observation and the command travels with it. V-BAR's export puts the observation
normalization, the network and the action squashing into one ONNX file, and measures how long
one inference takes. ONNX support needs the optional extra: `pip install "vbar[onnx]"`
(onnx, Apache-2.0, and onnxruntime, MIT).

## The policy

`MLPPolicy(weights, biases, obs_mean, obs_std, activation, squash)` maps an observation to
fractions of a full command:

$$
\mathbf x_0 = (\mathbf o - \boldsymbol\mu) / \boldsymbol\sigma, \qquad
\mathbf x_{k+1} = f(\mathbf x_k W_k + \mathbf b_k), \qquad
\mathbf u = s(\mathbf x_{L-1} W_{L-1} + \mathbf b_{L-1})
$$

with $f$ = tanh or ReLU for the hidden layers and $s$ = tanh, or a clip to $[-1, 1]$, for the
output. Weight matrices are (inputs, outputs), so the weights of most frameworks' linear layers
go in transposed; the last layer has three outputs. A network trained elsewhere is brought in by
copying its weights and its observation statistics. `MLPPolicy.random(obs_mean, obs_std,
hidden)` gives Glorot-uniform weights for tests. `PolicyAgent(policy)` flies it in process in
any continuous action mode.

Normalization and squashing belong to the policy, not to the code around it. A policy exported
without them acts on raw metres and returns unbounded numbers, and the scaling it was trained with
would live outside the file and be easy to lose.

## Export

`export_onnx(policy, path, dtype)` builds the graph `Sub`, `Div`, then `MatMul` and `Add` per
layer with `Tanh` or `Relu` between them, and `Tanh` or `Clip` at the end, with input `obs`
`(batch, d)` and output `action` `(batch, 3)`. Weights are stored in float32 (default) or
float64. The model uses opset 17 and IR version 8, which every onnxruntime release since 1.13
loads, is checked with `onnx.checker`, and is returned serialized (and written to `path` if
given).

`OnnxAgent(model, profile)` flies an exported file or serialized model with onnxruntime's CPU
provider, in any continuous action mode. Its commands are what the exported graph computes, so
the in-process `PolicyAgent` and the `OnnxAgent` differ only by the arithmetic type.

## Latency

`profile_latency(model, observations, profile)` times one inference per observation row after a
warm-up and returns the median, 95th percentile and maximum (s) with the machine, processor, CPU
count, Python and onnxruntime versions. Profiles fix the execution so runs on one machine are
comparable:

| Profile | Threads | Use |
|---|---|---|
| `single-thread` (default) | one intra-op and one inter-op thread | the cost on one core |
| `all-threads` | onnxruntime's default | throughput on a workstation |

Both run sequentially with all graph optimizations. A latency is a property of the machine it
was measured on: the profiles fix the threading, not the processor, and do not emulate a slower
one. For latency inside an episode, see real-time runs ([agents.md](agents.md)).

## How it is tested

`tests/test_policy.py` (always) and `tests/test_onnx_export.py` (with the extra; a CI job installs
it):

- The policy normalizes before the first layer (shifting the mean shifts its input), its
  outputs stay within $(-1, 1)$ for tanh and within $[-1, 1]$ for the clip, and layer shapes are
  checked.
- The policy agent flies the same trajectory in normalized and acceleration modes.
- Round trip: over 500 observations at the approach scenario's scale, the float32 export
  reproduces the numpy policy computed in float32 to 2e-6 and in float64 to 2e-5, for tanh and for
  ReLU with clipping; the float64 export reproduces it to 1e-12.
- The ONNX agent's commands along a 30-step episode equal the in-process policy's to 1e-12
  (float64 export).
- The latency report holds 200 runs with ordered statistics and the machine fields; an unknown
  profile is refused.

## Example

```python
import numpy as np

from vbar.agents.policy import MLPPolicy, PolicyAgent

mean = np.array([0.0, 300.0, 0.0, 0.0, 0.0, 0.0])
std = np.array([150.0, 200.0, 50.0, 0.1, 0.1, 0.1])
policy = MLPPolicy.random(mean, std, hidden=(64, 64), rng=np.random.default_rng(0))
print(policy(np.array([0.0, 400.0, 0.0, 0.0, -0.1, 0.0])).round(3))  # fractions in (-1, 1)
agent = PolicyAgent(policy)

# With the onnx extra:
#   from vbar.agents.policy import OnnxAgent, export_onnx, profile_latency
#   export_onnx(policy, "policy.onnx")
#   rows = mean + std * np.random.default_rng(1).normal(size=(500, 6))
#   report = profile_latency("policy.onnx", rows)
```
