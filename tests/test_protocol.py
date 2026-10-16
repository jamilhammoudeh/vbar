"""Every dynamics model behaves the same through the `Dynamics` interface."""

import inspect

import numpy as np
import pytest

from vbar.dynamics import cr3bp
from vbar.dynamics.bodies import EARTH_MOON_KLMR, SUN_FOR_EARTH_MOON_KLMR
from vbar.dynamics.models import CWDynamics, Dynamics, TwoBodyDynamics
from vbar.dynamics.perturbations import Drag, HarrisPriester, SolarPressure
from vbar.dynamics.relative import (
    CurvilinearCWDynamics,
    LinearizedTwoBodyDynamics,
    YamanakaAnkersenDynamics,
)

NRHO = np.array([0.9873849199042153, 0.0, 0.008369210996156513, 0.0, 1.6744665364075588, 0.0])
SUN = [0.3, 0.5, 0.8]

MODELS = {
    "cw": lambda: CWDynamics(1.1e-3),
    "twobody_j2_drag_srp": lambda: TwoBodyDynamics(
        inclination=0.9,
        j2=True,
        perturbations=[Drag(HarrisPriester(SUN, 2), 40.0, 60.0), SolarPressure(SUN, 0.02, 0.01)],
    ),
    "ya": lambda: YamanakaAnkersenDynamics(7500e3, 0.1, 0.4),
    "cw_curvilinear": lambda: CurvilinearCWDynamics(6878e3),
    "linear_j2": lambda: LinearizedTwoBodyDynamics(inclination=0.9),
    "cr3bp": lambda: cr3bp.CR3BPDynamics(EARTH_MOON_KLMR, NRHO),
    "libration": lambda: cr3bp.LibrationPointDynamics.at_point(EARTH_MOON_KLMR, 2),
    "cr3bp_linear": lambda: cr3bp.PeriodicLinearDynamics(EARTH_MOON_KLMR, NRHO, 1.5088887454509277),
    "bicircular": lambda: cr3bp.BicircularDynamics(SUN_FOR_EARTH_MOON_KLMR, NRHO, 0.5),
}


def drive(model: Dynamics, start, accel, duration):
    model.reset(start)
    return model.advance(accel, duration, n_samples=5)


@pytest.mark.parametrize("name", MODELS)
def test_model_satisfies_the_dynamics_interface(name):
    model = MODELS[name]()
    assert model.name == name
    assert isinstance(model.n, float) and model.n > 0
    start = np.array([150.0, -400.0, 30.0, 0.05, 0.1, -0.02])
    model.reset(start)
    with pytest.raises(RuntimeError, match="no segment"):
        model.state_in_last_segment(1.0)
    final, samples = drive(model, start, [2e-4, -1e-4, 5e-5], 120.0)
    assert final.shape == (6,) and samples.shape == (5, 6)
    assert np.allclose(samples[0], start, rtol=0, atol=1e-9)
    assert np.allclose(samples[-1], final, rtol=0, atol=1e-9)
    assert np.allclose(model.state_in_last_segment(60.0), samples[2], rtol=0, atol=1e-9)
    # A model holds its own copy: changing the returned state changes nothing.
    final[:] = 0.0
    assert not np.allclose(model.state_in_last_segment(120.0), 0.0)
    _, none = model.advance(np.zeros(3), 10.0)
    assert none is None


@pytest.mark.parametrize("name", MODELS)
def test_model_takes_the_interface_arguments_by_name(name):
    positional = MODELS[name]()
    for method in ("reset", "advance", "state_in_last_segment"):
        expected = list(inspect.signature(getattr(Dynamics, method)).parameters)[1:]
        assert list(inspect.signature(getattr(positional, method)).parameters) == expected
    start = np.array([150.0, -400.0, 30.0, 0.05, 0.1, -0.02])
    accel = np.array([2e-4, -1e-4, 5e-5])
    final, samples = drive(positional, start, accel, 120.0)
    model = MODELS[name]()
    model.reset(rel_state=start)
    by_name, named_samples = model.advance(accel_hill=accel, duration=120.0, n_samples=5)
    assert np.array_equal(by_name, final) and np.array_equal(named_samples, samples)
    assert np.array_equal(
        model.state_in_last_segment(tau=60.0), positional.state_in_last_segment(60.0)
    )
