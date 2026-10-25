import numpy as np
import pytest

from vbar.dynamics import mean_motion, models
from vbar.dynamics.bodies import EARTH
from vbar.dynamics.cw import cw_system
from vbar.dynamics.models import CWDynamics, make_dynamics
from vbar.scenarios import make

# The run time assurance benchmark's mean motion, rad/s (Dunlap et al., 2021, eq. 14).
N = 0.001027
X0 = np.array([5000.0, -7000.0, 3000.0, 0.5, -0.4, 0.3])
ACCEL = np.array([0.05, -0.03, 0.02])
ZERO = np.zeros(3)


def euler(n, step_s=1.0):
    return models.EulerCWDynamics(n, step_s=step_s)


def started(dyn, x0):
    dyn.reset(x0)
    return dyn


def test_one_euler_step_is_x_plus_h_times_the_cw_derivative():
    a, b = cw_system(N)
    state, _ = started(euler(N), X0).advance(ACCEL, 1.0)
    np.testing.assert_array_equal(state, X0 + 1.0 * (a @ X0 + b @ ACCEL))


def test_five_euler_steps_equal_one_five_second_advance():
    stepped = started(euler(N), X0)
    for _ in range(5):
        state, _ = stepped.advance(ACCEL, 1.0)
    whole, _ = started(euler(N), X0).advance(ACCEL, 5.0)
    np.testing.assert_array_equal(whole, state)


@pytest.mark.parametrize("k", [1, 5, 600])
def test_unforced_euler_steps_are_powers_of_i_plus_h_a(k):
    a, _ = cw_system(N)
    state, _ = started(euler(N), X0).advance(ZERO, float(k))
    expected = np.linalg.matrix_power(np.eye(6) + a, k) @ X0
    np.testing.assert_allclose(state, expected, rtol=1e-12)


def test_euler_advance_returns_every_node():
    a, b = cw_system(N)
    state, nodes = started(euler(N), X0).advance(ACCEL, 3.0, n_samples=20)
    expected = [X0]
    for _ in range(3):
        expected.append(expected[-1] + (a @ expected[-1] + b @ ACCEL))
    assert nodes.shape == (4, 6)
    np.testing.assert_array_equal(nodes, np.array(expected))
    np.testing.assert_array_equal(state, nodes[-1])


def test_vbar_point_is_an_euler_equilibrium():
    # At rest on the V-bar every column of A that the state touches is zero, so each step adds
    # an exact zero.
    point = np.array([0.0, -250.0, 0.0, 0.0, 0.0, 0.0])
    dyn = started(euler(N), point)
    state, _ = dyn.advance(ZERO, 5000.0)
    np.testing.assert_array_equal(state, point)


def test_euler_out_of_plane_energy_grows_by_one_plus_n2h2_per_step():
    # The out-of-plane block maps (z, vz) to (z + h vz, vz - h n^2 z), so
    # (z + h vz)^2 + (vz / n - h n z)^2 = (1 + n^2 h^2) (z^2 + (vz / n)^2): the cross terms
    # cancel and each step multiplies the invariant of the exact motion by 1 + n^2 h^2.
    x0 = np.array([0.0, 0.0, 120.0, 0.0, 0.0, -0.08])
    dyn = started(euler(N), x0)

    def energy(x):
        return x[2] ** 2 + (x[5] / N) ** 2

    for k in range(1, 5001):
        state, _ = dyn.advance(ZERO, 1.0)
        if k in (1, 10, 1000, 5000):
            assert energy(state) == pytest.approx(energy(x0) * (1.0 + N * N) ** k, rel=1e-10)
    np.testing.assert_array_equal(state[[0, 1, 3, 4]], np.zeros(4))


def test_euler_one_step_error_falls_fourfold_when_the_step_halves():
    # Forward Euler's local error is h^2 / 2 times the second derivative plus O(h^3), so halving
    # h divides the one-step error by 4 to within O(n h).
    def error(h):
        exact, _ = started(CWDynamics(N), X0).advance(ACCEL, h)
        approx, _ = started(euler(N, step_s=h), X0).advance(ACCEL, h)
        return np.linalg.norm(approx - exact)

    assert 3.9 <= error(1.0) / error(0.5) <= 4.1


@pytest.mark.parametrize("duration", [0.5, 2.5, 1.0 + 1e-6])
def test_euler_advance_refuses_a_partial_step(duration):
    dyn = started(euler(N), X0)
    with pytest.raises(ValueError):
        dyn.advance(ACCEL, duration)


def test_euler_state_exists_only_at_the_nodes():
    a, b = cw_system(N)
    dyn = started(euler(N), X0)
    dyn.advance(ACCEL, 3.0)
    first = X0 + (a @ X0 + b @ ACCEL)
    np.testing.assert_array_equal(dyn.state_in_last_segment(0.0), X0)
    np.testing.assert_array_equal(dyn.state_in_last_segment(1.0), first)
    for tau in (0.5, 1.25, 2.999):
        with pytest.raises(ValueError):
            dyn.state_in_last_segment(tau)


def test_euler_rung_is_built_by_name_with_the_keplerian_mean_motion():
    dyn = make_dynamics("euler1", altitude=600e3)
    assert isinstance(dyn, models.EulerCWDynamics)
    assert dyn.name == "euler1"
    assert dyn.step_s == 1.0
    assert dyn.n == mean_motion(EARTH.radius + 600e3, EARTH.mu)
    assert models.EulerCWDynamics.step_s == 1.0
    assert not hasattr(make_dynamics("cw"), "step_s")


def _euler_nodes(n, x0, steps):
    a, _ = cw_system(n)
    nodes = [np.asarray(x0, dtype=float)]
    for _ in range(steps):
        nodes.append(nodes[-1] + (a @ nodes[-1]))
    return np.array(nodes)


@pytest.mark.parametrize("violation_samples", [2, 200])
def test_euler_violations_are_judged_at_the_nodes(violation_samples):
    # Closing at 0.6 m/s from 100 m: the speed limit 0.2 + 4 n |r| shrinks by 4 n 0.6 m/s each
    # second, so nodes 0 to 4 are inside it and node 5 is the first outside. A continuous model
    # would put the crossing near 4.39 s; the Euler plant only has the node at 5 s.
    env = make(
        "vbar:docking-afrl:euler1:box:dv:v1",
        training=True,
        action_mode="accel",
        dt=5.0,
        violation_samples=violation_samples,
    )
    x0 = np.array([0.0, 100.0, 0.0, 0.0, -0.6, 0.0])
    env.reset(seed=0, options={"start_state": x0})
    (limit,) = (c for c in env.constraints if c.name == "speed_limit")
    nodes = _euler_nodes(env.n, x0, 5)
    margins = np.array([float(limit.margin(x, 0.0)) for x in nodes])
    assert np.all(margins[:5] > 0.0) and margins[5] < 0.0

    _, _, _, _, info = env.step(ZERO)
    np.testing.assert_array_equal(info["true_state"], nodes[-1])
    assert env.record.segment.violations["speed_limit"] == ((4.0, 5.0),)
    assert info["min_margins"]["speed_limit"] == pytest.approx(margins.min(), rel=1e-12)
    bound = info["min_margin_bounds"]["speed_limit"]
    assert 0.0 <= info["min_margins"]["speed_limit"] - bound < 1e-9


def test_euler_env_takes_one_second_substeps_of_its_step():
    env = make("vbar:docking-afrl:euler1:box:dv:v1", training=True, action_mode="accel", dt=5.0)
    x0 = np.array([300.0, -400.0, 50.0, 0.1, 0.2, -0.05])
    env.reset(seed=0, options={"start_state": x0})
    _, _, _, _, info = env.step(ZERO)
    np.testing.assert_array_equal(info["true_state"], _euler_nodes(env.n, x0, 5)[-1])
    assert info["time_s"] == 5.0


def test_euler_env_refuses_a_step_that_is_not_whole_seconds():
    assert make_dynamics("euler1").step_s == 1.0
    with pytest.raises(ValueError):
        env = make("vbar:docking-afrl:euler1:box:dv:v1", training=True, action_mode="accel", dt=2.5)
        env.reset(seed=0)
        env.step(ZERO)


def test_maneuvering_target_on_euler_is_judged_at_the_nodes():
    env = make("vbar:separation-flyaround:euler1:box:dv:v1:reboost", training=True)
    env.reset(seed=0)
    ((burn_start, burn_duration, _),) = env.dynamics.target.burns()
    spans: dict[str, set[float]] = {c.name: set() for c in env.constraints}
    violated: dict[str, set[float]] = {c.name: set() for c in env.constraints}
    while env.t <= burn_start + burn_duration:
        step_start = env.t
        _, _, terminated, truncated, _ = env.step(ZERO)
        trace = env.record.segment
        # The Euler plant exists only at whole seconds; nothing between them may be sampled.
        np.testing.assert_array_equal(trace.times, np.round(trace.times))
        for c in env.constraints:
            for start, end in trace.violations[c.name]:
                assert start == round(start) and end == round(end) and end > start
                spans[c.name].update(np.arange(start + 1.0, end + 0.5))
            margins = np.asarray(c.margin(trace.states, trace.times), dtype=float)
            violated[c.name].update(trace.times[(margins < 0.0) & (trace.times > step_start)])
        assert not terminated and not truncated
    # A violation is the span (t_{j-1}, t_j] ending at each node with a negative margin.
    assert spans == violated
