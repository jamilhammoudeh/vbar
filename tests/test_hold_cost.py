"""Cost of holding a position in the CW frame, flown through the environment and the E1 metric."""

import numpy as np
import pytest
from metric_cases import ConstantAgent, GuardedApproach

from vbar import ApproachConfig
from vbar.metrics import record_episode
from vbar.metrics.efficiency import delta_v

MAX_ACCEL = 0.02
STEPS = 568  # 5680 s, one orbit at 500 km


def hold(offset, accel):
    """Episode at rest at `offset` (m, Hill) commanding `accel(n)` (m/s^2, Hill) at every step."""
    config = ApproachConfig(start_state=(*offset, 0.0, 0.0, 0.0), max_steps=STEPS)
    env = GuardedApproach(config)
    command = np.asarray(accel(env.n)) / MAX_ACCEL
    return env.n, record_episode(env, ConstantAgent(command), seed=1, agent_seed=2)


def stray(ep, offset):
    """Largest distance (m) any recorded state is from the starting position."""
    states = np.array([rec.true_state for rec in ep.records])
    return float(np.max(np.linalg.norm(states[:, :3] - offset, axis=1)))


@pytest.mark.parametrize("x0", [-100.0, 250.0])
def test_radial_hover_costs_3_n_squared_x_per_second(x0):
    # Fehse, Automated Rendezvous and Docking of Spacecraft, sec. 3.3.3, eqs. 3.55 and 3.56:
    # a radial offset is held by 3 n^2 |x0| and costs 3 n^2 |x0| T.
    offset = np.array([x0, 0.0, 0.0])
    n, ep = hold(offset, lambda n: [-3.0 * n**2 * x0, 0.0, 0.0])
    # The command passes through float32, 6e-8 relative; the 1e-3 m bound is 1e-5 of the offset.
    assert delta_v(ep) == pytest.approx(3.0 * n**2 * abs(x0) * ep.duration, rel=1e-6)
    assert stray(ep, offset) < 1e-3


@pytest.mark.parametrize("z0", [-40.0, 60.0])
def test_cross_track_hover_costs_n_squared_z_per_second(z0):
    # Fehse sec. 3.3.3, eq. 3.58: an offset along H-bar (z here) is held by n^2 |z0| and costs
    # n^2 |z0| T.
    offset = np.array([0.0, -100.0, z0])
    n, ep = hold(offset, lambda n: [0.0, 0.0, n**2 * z0])
    assert delta_v(ep) == pytest.approx(n**2 * abs(z0) * ep.duration, rel=1e-6)
    assert stray(ep, offset) < 1e-3


@pytest.mark.parametrize("y0", [-100.0, 300.0])
def test_vbar_hold_costs_no_delta_v_and_stays_put(y0):
    # Fehse sec. 2.3.3: a V-bar hold point is kept at zero delta-v.
    offset = np.array([0.0, y0, 0.0])
    _, ep = hold(offset, lambda n: np.zeros(3))
    assert delta_v(ep) == 0.0
    assert stray(ep, offset) < 1e-6
