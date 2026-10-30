"""What the external propagators share, testable without them."""

import math
import sys
import threading
import types

import numpy as np
import pytest

from vbar.conventions import hill_frame
from vbar.dynamics import EARTH, TwoBodyDynamics
from vbar.dynamics.backends import Burn, basilisk, common, hill_states, orekit
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


STATE = np.array([6.878e6, 0.0, 0.0, 0.0, 7612.6, 0.0])
ARC_REL = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])


def thrust(start, duration):
    return Burn(start, duration, (0.0, 1e-3, 0.0))


def state_with(value):
    state = STATE.copy()
    state[2] = value
    return state


@pytest.fixture
def no_extras(monkeypatch):
    for package in ("Basilisk", "orekit_jpype"):
        for loaded in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
            monkeypatch.setitem(sys.modules, loaded, None)
        monkeypatch.setitem(sys.modules, package, None)
    orekit._orekit.cache_clear()
    yield
    orekit._orekit.cache_clear()


def within(seconds, function, *args, **kwargs):
    """Call `function` on a daemon thread; a call that does not return fails only its own test."""
    outcome = {}

    def run():
        try:
            outcome["value"] = function(*args, **kwargs)
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        pytest.fail(f"no answer after {seconds:g} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def labelled(argument, cases):
    return [pytest.param(argument, value, id=f"{argument}-{label}") for label, value in cases]


BAD_STATES = [
    ("three-values", np.zeros(3)),
    ("seven-values", np.zeros(7)),
    ("two-rows", np.zeros((2, 6))),
    ("nan", state_with(math.nan)),
    ("inf", state_with(math.inf)),
    ("complex", STATE.astype(complex)),
    ("strings", ["1"] * 6),
]
BAD_NUMBERS = [("zero", 0.0), ("negative", -1.0), ("nan", math.nan), ("inf", math.inf)]
BAD_TIMES = [
    ("empty", []),
    ("two-dimensional", [[0.0, 1.0], [2.0, 3.0]]),
    ("scalar", 1.0),
    ("unsorted", [0.0, 2.0, 1.0]),
    ("nan", [0.0, math.nan, 2.0]),
    ("inf", [0.0, 1.0, math.inf]),
    ("negative", [-1.0, 0.0, 1.0]),
    ("complex", [0j, 1 + 0j]),
    ("strings", ["0", "1"]),
]
BAD_BURNS = [
    ("non-burn", (thrust(0.0, 5.0), "thrust")),
    ("bare-tuple", ((0.0, 5.0, (0.0, 0.0, 0.0)),)),
    ("overlapping", (thrust(0.0, 10.0), thrust(5.0, 10.0))),
    ("before-the-start", (thrust(-5.0, 10.0),)),
]
BACKENDS = pytest.mark.parametrize("backend", [basilisk, orekit], ids=["basilisk", "orekit"])


@BACKENDS
@pytest.mark.parametrize(
    "argument, bad",
    [
        *labelled("target", BAD_STATES),
        *labelled("chaser", BAD_STATES),
        *labelled("times", BAD_TIMES),
        *labelled("tolerance", BAD_NUMBERS),
        *labelled("substep", BAD_NUMBERS),
        *labelled("mass", BAD_NUMBERS),
        *labelled("burns", BAD_BURNS),
    ],
)
def test_hostile_backend_inputs_are_refused_before_the_extra_is_imported(
    backend, argument, bad, no_extras
):
    kwargs = {"target": STATE, "chaser": STATE, "times": [0.0, 1.0, 2.0], "body": EARTH}
    kwargs[argument] = bad
    with pytest.raises(ValueError):
        backend.propagate(**kwargs)


@BACKENDS
@pytest.mark.parametrize(
    "argument, good",
    [
        *labelled("times", [("repeated-sample", [0.0, 1.0, 1.0]), ("array", np.arange(3.0))]),
        *labelled("target", [("list", STATE.tolist())]),
        *labelled("tolerance", [("loose", 1e-3)]),
        *labelled(
            "burns",
            [
                ("touching", (thrust(0.0, 10.0), thrust(10.0, 5.0))),
                ("rounding-overlap", [thrust(0.0, 10.0), thrust(10.0 - 1e-12, 5.0)]),
                ("listed-out-of-order", [thrust(20.0, 5.0), thrust(0.0, 10.0)]),
            ],
        ),
    ],
)
def test_valid_backend_inputs_reach_the_install_check(backend, argument, good, no_extras):
    kwargs = {"target": STATE, "chaser": STATE, "times": [0.0, 1.0, 2.0], "body": EARTH}
    kwargs[argument] = good
    with pytest.raises(ImportError, match=r"vbar\[(basilisk|orekit)"):
        backend.propagate(**kwargs)


@pytest.mark.parametrize(
    "argument, bad",
    [
        *labelled("times", [(label, np.asarray(t)) for label, t in BAD_TIMES]),
        *labelled("times", [("unsorted-list", [0.0, 100.0, 50.0])]),
        *labelled("rel", BAD_STATES),
        *labelled("rtol", BAD_NUMBERS),
        *labelled("burns", BAD_BURNS),
    ],
)
def test_hostile_arc_inputs_are_refused_before_integrating(argument, bad):
    kwargs = {"rel": ARC_REL, "times": np.array([0.0, 50.0, 100.0])}
    kwargs[argument] = bad
    with pytest.raises(ValueError):
        within(20.0, vbar_arc, **kwargs)


def test_vbars_arc_accepts_a_list_of_times_with_a_repeated_sample():
    times = [0.0, 50.0, 50.0, 100.0]
    from_list, _, _ = vbar_arc(ARC_REL, times)
    from_array, _, _ = vbar_arc(ARC_REL, np.array(times))
    np.testing.assert_array_equal(from_list, from_array)
    assert np.array_equal(from_list[1], from_list[2])


@pytest.mark.parametrize(
    "burns",
    [
        [thrust(0.0, 10.0), thrust(9.5, 5.0)],
        [thrust(0.0, 10.0), thrust(10.0 - 1e-6, 5.0)],
    ],
    ids=["half-a-second", "a-microsecond"],
)
def test_burns_that_overlap_are_refused(burns):
    with pytest.raises(ValueError):
        common.check_burns(burns)


@pytest.mark.parametrize(
    "burns",
    [
        [thrust(0.0, 10.0), thrust(10.0 - 1e-12, 5.0)],
        [thrust(0.0, 10.0), thrust(10.0, 5.0), thrust(15.0, 5.0)],
        [thrust(20.0, 5.0), thrust(0.0, 10.0), thrust(10.0, 5.0)],
        [thrust(0.0, 10.0), thrust(11.0, 5.0)],
        [],
    ],
    ids=["rounding-overlap", "touching-chain", "listed-out-of-order", "gap", "none"],
)
def test_burns_that_do_not_overlap_are_accepted(burns):
    common.check_burns(burns)


@pytest.mark.parametrize(
    "burns",
    [
        [thrust(0.0, 100.0), thrust(10.0, 5.0), thrust(50.0, 5.0)],
        [thrust(50.0, 5.0), thrust(10.0, 5.0), thrust(0.0, 100.0)],
    ],
    ids=["in-order", "listed-out-of-order"],
)
def test_a_long_burn_containing_two_short_ones_is_refused(burns):
    with pytest.raises(ValueError):
        common.check_burns(burns)


def test_the_burns_of_a_recorded_episode_do_not_overlap():
    from vbar.agents import PDAgent
    from vbar.metrics import record_episode
    from vbar.scenarios import ApproachConfig, ApproachEnv

    env = ApproachEnv(ApproachConfig(start_state=(30.0, -400.0, 10.0, 0.0, 0.0, 0.0), max_steps=20))
    episode = record_episode(env, PDAgent(), seed=1, agent_seed=2)
    burns = episode_burns(episode.records)
    assert len(burns) == 20
    common.check_burns(burns)


def test_adjacent_burns_have_one_acceleration_at_the_join_in_both_paths():
    first = Burn(0.0, 30.0, (1e-3, 0.0, 0.0))
    second = Burn(30.0, 30.0, (0.0, -2e-3, 5e-4))
    burns = [first, second]
    common.check_burns(burns)
    np.testing.assert_array_equal(accel_at(burns, 30.0), second.accel_hill)
    np.testing.assert_array_equal(accel_at(burns, 29.999), first.accel_hill)
    arc, _, _ = vbar_arc(ARC_REL, np.array([0.0, 30.0, 60.0]), burns)
    model = TwoBodyDynamics("earth", 500e3, math.radians(51.6), j2=True, rtol=1e-12)
    model.reset(ARC_REL)
    joined, _ = model.advance(np.array(first.accel_hill), 30.0)
    end, _ = model.advance(np.array(second.accel_hill), 30.0)
    np.testing.assert_allclose(arc[1], joined, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(arc[2], end, rtol=1e-9, atol=1e-9)
