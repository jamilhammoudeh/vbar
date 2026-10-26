"""What the external propagators share, testable without them."""

import math
import sys
import types

import numpy as np
import pytest

from vbar.conventions import hill_frame
from vbar.dynamics import EARTH, TwoBodyDynamics
from vbar.dynamics.backends import Burn, hill_states, orekit
from vbar.dynamics.backends.agreement import episode_burns, vbar_arc
from vbar.dynamics.backends.common import accel_at, check_aligned, inertial_accel
from vbar.dynamics.models import gravity


def test_the_shared_conversion_gives_vbars_own_hill_states():
    model = TwoBodyDynamics("earth", 500e3, math.radians(51.6), j2=True)
    rel = np.array([12.0, -340.0, 7.0, 0.01, -0.02, 0.003])
    model.reset(rel)
    end, _ = model.advance(np.zeros(3), 900.0)
    found = hill_states(model.target_state(), model.chaser_state(), EARTH, True)
    # Same formula on the same inertial states; the difference is rounding of 7e6 m positions.
    np.testing.assert_allclose(found, end, rtol=0.0, atol=1e-8)


def test_a_burn_held_in_inertial_axes_is_aimed_at_the_middle_of_its_hold():
    state = np.array([6.878e6, 0.0, 0.0, 0.0, 7612.6, 0.0])
    burns = [Burn(0.0, 10.0, (0.0, 1.0, 0.0))]
    C, omega = hill_frame(state[:3], state[3:], gravity(state[:3], EARTH, True))
    held = inertial_accel(burns, 0.0, state, EARTH, True, hold=2.0)
    # The along-track axis turns toward -x at h / r^2; after 1 s it is rotated by omega_z.
    expected = C @ np.array([-math.sin(omega[2]), math.cos(omega[2]), 0.0])
    np.testing.assert_allclose(held, expected, rtol=0.0, atol=1e-15)
    assert np.array_equal(inertial_accel(burns, 0.0, state, EARTH, True), C @ [0.0, 1.0, 0.0])
    assert not accel_at(burns, 10.0).any() and accel_at(burns, 9.99)[1] == 1.0


def test_burns_need_a_positive_duration_and_finite_values():
    with pytest.raises(ValueError, match="positive"):
        Burn(0.0, 0.0, (1.0, 0.0, 0.0))
    for start, duration, accel in (
        (0.0, math.inf, (1.0, 0.0, 0.0)),
        (math.nan, 1.0, (1.0, 0.0, 0.0)),
        (0.0, 1.0, (math.nan, 0.0, 0.0)),
        (0.0, 1.0, (1.0, 0.0)),
    ):
        with pytest.raises(ValueError):
            Burn(start, duration, accel)


def test_vbars_arc_splits_at_burn_edges_between_samples():
    rel = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    burns = [Burn(25.0, 30.0, (0.001, 0.0, 0.0))]
    times = np.arange(0.0, 101.0, 50.0)
    split, _, _ = vbar_arc(rel, times, burns)
    model = TwoBodyDynamics("earth", 500e3, math.radians(51.6), j2=True, rtol=1e-12)
    model.reset(rel)
    model.advance(np.zeros(3), 25.0)
    model.advance(np.array([0.001, 0.0, 0.0]), 30.0)
    end, _ = model.advance(np.zeros(3), 45.0)
    np.testing.assert_allclose(split[-1], end, rtol=0.0, atol=1e-9)


def test_an_episodes_burns_are_its_thrusting_segments():
    from vbar.agents import PDAgent
    from vbar.metrics import record_episode
    from vbar.scenarios import ApproachConfig, ApproachEnv

    env = ApproachEnv(ApproachConfig(start_state=(30.0, -400.0, 10.0, 0.0, 0.0, 0.0), max_steps=5))
    episode = record_episode(env, PDAgent(), seed=1, agent_seed=2)
    burns = episode_burns(episode.records)
    assert len(burns) == 5 and burns[0].start == 0.0 and burns[-1].start == 40.0
    np.testing.assert_allclose(burns[0].accel_hill, episode.records[0].segment.segments[0][2])


@pytest.mark.parametrize("module, package", [("basilisk", "Basilisk"), ("orekit", "orekit_jpype")])
def test_a_missing_backend_names_its_install_command(module, package, monkeypatch):
    import importlib

    # Every loaded module of the package too, so the import fails even after another test ran it.
    for loaded in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
        monkeypatch.setitem(sys.modules, loaded, None)
    monkeypatch.setitem(sys.modules, package, None)
    backend = importlib.import_module(f"vbar.dynamics.backends.{module}")
    if module == "orekit":
        backend._orekit.cache_clear()
    with pytest.raises(ImportError, match=r"vbar\[" + module):
        backend.propagate(np.zeros(6), np.zeros(6), [0.0], EARTH)


def test_burn_edges_inside_a_substep_are_refused():
    check_aligned([Burn(1.0, 3.0, (0.0, 0.0, 0.0))], 1.0)
    with pytest.raises(ValueError, match="whole numbers of substeps"):
        check_aligned([Burn(0.5, 3.0, (0.0, 0.0, 0.0))], 1.0)
    with pytest.raises(ValueError, match="whole numbers of substeps"):
        check_aligned([Burn(1.0, 2.5, (0.0, 0.0, 0.0))], 1.0)


def test_orekit_without_a_java_runtime_names_the_install_command(monkeypatch):
    class NoJava(ValueError):
        pass

    def init_vm():
        raise NoJava("no JVM found")

    fake = types.SimpleNamespace(initVM=init_vm)
    monkeypatch.setitem(sys.modules, "orekit_jpype", fake)
    orekit._orekit.cache_clear()
    try:
        with pytest.raises(ImportError, match=r"vbar\[orekit"):
            orekit._orekit()
    finally:
        orekit._orekit.cache_clear()
