"""Golden trajectories: frozen outputs that every platform and later version must reproduce.

Each fixture in `tests/golden/` holds its inputs and outputs. After an intended change to the
physics, run `python tests/test_golden.py` to regenerate them and review the diff.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent
from vbar.dynamics import make_dynamics

GOLDEN = Path(__file__).parent / "golden"

MODEL_CASES = {
    "cw": {"kind": "cw", "body": "earth", "altitude": 500e3, "inclination": 0.0},
    "twobody": {"kind": "twobody", "body": "earth", "altitude": 500e3, "inclination": 0.0},
    "twobody_j2": {"kind": "twobody_j2", "body": "earth", "altitude": 500e3, "inclination": 0.9},
    "twobody_kerbin": {"kind": "twobody", "body": "kerbin", "altitude": 150e3, "inclination": 0.0},
}
APPROACH_CASES = {
    "approach_clear_path": [-150.0, -400.0, 0.0, 0.0, 0.0, 0.0],
    "approach_target_in_the_way": [0.0, 400.0, 0.0, 0.0, 0.0, 0.0],
}


def fly(case, initial_state, accels, dt):
    m = make_dynamics(case["kind"], case["body"], case["altitude"], case["inclination"])
    m.reset(initial_state)
    states = [np.asarray(initial_state, dtype=float)]
    for accel in accels:
        state, _ = m.advance(accel, dt)
        states.append(state)
    return np.array(states)


def fly_approach(start_state):
    return run_episode(ApproachEnv(ApproachConfig(start_state=tuple(start_state))), PDAgent())


def load(name):
    return json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))


def assert_trajectory_matches(actual, golden):
    golden = np.asarray(golden)
    assert actual.shape == golden.shape
    for block in (slice(0, 3), slice(3, 6)):
        scale = np.max(np.abs(golden[:, block]))
        assert np.max(np.abs(actual[:, block] - golden[:, block])) <= 1e-9 * scale


@pytest.mark.parametrize("name", MODEL_CASES)
def test_dynamics_reproduce_golden_trajectory(name):
    golden = load(name)
    case = {key: golden[key] for key in ("kind", "body", "altitude", "inclination")}
    states = fly(case, golden["initial_state"], golden["accel"], golden["dt"])
    assert_trajectory_matches(states, golden["states"])


@pytest.mark.parametrize("name", APPROACH_CASES)
def test_approach_episode_reproduces_golden(name):
    golden = load(name)
    result = fly_approach(golden["start_state"])
    for key in ("captured", "keep_out_violated", "violation_steps"):
        assert result[key] == golden[key]
    assert result["dv_used_mps"] == pytest.approx(golden["dv_used_mps"], rel=1e-9)
    assert_trajectory_matches(result["states"], golden["states"])


def dump(record):
    """JSON with one trajectory row per line, so diffs show which steps moved."""
    parts = []
    for key, value in record.items():
        if isinstance(value, list) and value and isinstance(value[0], list):
            rows = ",\n    ".join(json.dumps(row) for row in value)
            parts.append(f'  "{key}": [\n    {rows}\n  ]')
        else:
            parts.append(f'  "{key}": {json.dumps(value)}')
    return "{\n" + ",\n".join(parts) + "\n}\n"


def regenerate():
    GOLDEN.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261015)
    dt = 10.0
    for name, case in MODEL_CASES.items():
        initial_state = [250.0, -800.0, 120.0, 0.1, 0.2, -0.05]
        accels = rng.uniform(-0.02, 0.02, (60, 3)).tolist()
        states = fly(case, initial_state, accels, dt)
        record = {
            **case,
            "dt": dt,
            "initial_state": initial_state,
            "accel": accels,
            "states": states.tolist(),
        }
        (GOLDEN / f"{name}.json").write_text(dump(record), encoding="utf-8", newline="\n")
    for name, start_state in APPROACH_CASES.items():
        result = fly_approach(start_state)
        record = {
            "start_state": start_state,
            "captured": bool(result["captured"]),
            "keep_out_violated": bool(result["keep_out_violated"]),
            "violation_steps": int(result["violation_steps"]),
            "dv_used_mps": float(result["dv_used_mps"]),
            "states": result["states"].tolist(),
        }
        (GOLDEN / f"{name}.json").write_text(dump(record), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    regenerate()
