"""Decision steps per second of the approach scenario for each dynamics model.

Run with `python benchmarks/steps.py`. The numbers depend on the machine; compare branches on the
same machine.
"""

import time

import numpy as np

from vbar import ApproachConfig, ApproachEnv


def steps_per_second(dynamics: str, steps: int = 360) -> float:
    env = ApproachEnv(ApproachConfig(dynamics=dynamics))
    env.reset(seed=0)
    action = np.array([0.1, -0.2, 0.05])
    start = time.perf_counter()
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=0)
    return steps / (time.perf_counter() - start)


if __name__ == "__main__":
    for dynamics in ("cw", "twobody", "twobody_j2"):
        print(f"{dynamics:12s} {steps_per_second(dynamics):10.0f} steps/s")
