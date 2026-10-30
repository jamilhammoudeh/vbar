import numpy as np
import pytest

from vbar.agents.base import AgentView
from vbar.agents.pd import PDAgent

A_MAX = 1e-3
N = 0.0011


def _command(limit: str, hold: tuple[float, float, float]) -> np.ndarray:
    view = AgentView(
        scenario_id="hand-made",
        dt=10.0,
        max_steps=360,
        n=N,
        action_mode="accel",
        max_accel=A_MAX,
        thrust_limit=limit,
        goal={"hold_point": hold},
    )
    agent = PDAgent()
    agent.reset(view)
    # At rest at the origin the CW terms vanish, so the command is -Kp times the position error.
    return np.asarray(agent.act(np.zeros(6), {}))


def test_ball_saturated_command_keeps_direction_at_max_accel():
    # Kp = 1e-4 s^-2 on a 100 m error per axis asks for 1e-2 m/s^2 on each of y and z.
    u = _command("ball", (0.0, 100.0, 100.0))
    assert np.linalg.norm(u) == pytest.approx(A_MAX, rel=1e-12)
    np.testing.assert_allclose(u / np.linalg.norm(u), [0.0, 1.0, 1.0] / np.sqrt(2.0), rtol=1e-12)


DIAGONAL = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)


def _ball_command_of_size(size: float) -> np.ndarray:
    # Kp = 1e-4 s^-2, so a hold point 10 m out asks for size times the limit.
    return _command("ball", tuple(size * 10.0 * DIAGONAL))


@pytest.mark.parametrize("size", [1.01, 1.27, 1.49, 2.0])
def test_ball_command_over_the_limit_is_scaled_to_the_limit(size):
    u = _ball_command_of_size(size)
    assert np.linalg.norm(u) == pytest.approx(A_MAX, rel=1e-12)
    np.testing.assert_allclose(u / np.linalg.norm(u), DIAGONAL, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("size", [0.5, 0.99, 0.999999])
def test_ball_command_under_the_limit_is_untouched(size):
    np.testing.assert_allclose(
        _ball_command_of_size(size), size * A_MAX * DIAGONAL, rtol=1e-12, atol=1e-15
    )


def test_box_saturated_command_clips_each_axis():
    u = _command("box", (0.0, 100.0, 100.0))
    np.testing.assert_array_equal(u, [0.0, A_MAX, A_MAX])


@pytest.mark.parametrize("limit", ["ball", "box"])
def test_command_within_limit_is_untouched(limit):
    u = _command(limit, (2.0, 3.0, -1.0))
    np.testing.assert_allclose(u, [2e-4, 3e-4, -1e-4], rtol=1e-12)


def test_unknown_thrust_limit_is_rejected():
    with pytest.raises(ValueError, match="unknown thrust limit"):
        _command("cone", (0.0, 100.0, 0.0))
