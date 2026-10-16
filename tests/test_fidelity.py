import numpy as np
import pytest
from reference_states import pco_state

from vbar.dynamics.cw import cw_stm
from vbar.dynamics.fidelity import modeling_error_index, stm_adequacy, validity_horizon
from vbar.dynamics.models import CWDynamics, TwoBodyDynamics
from vbar.dynamics.relative import YamanakaAnkersenDynamics

ZERO = np.zeros(3)


def test_adequacy_is_zero_for_the_true_matrix_and_scales_velocities_by_n():
    n = 1.1e-3
    phi = cw_stm(n, 1234.0)
    assert stm_adequacy(phi, phi, n) == pytest.approx(0.0, abs=1e-12)
    wrong = phi.copy()
    wrong[0, 3] += 10.0
    misfit = np.eye(6) - phi @ np.linalg.inv(wrong)
    d = np.diag([1, 1, 1, 1 / n, 1 / n, 1 / n])
    assert stm_adequacy(phi, wrong, n) == pytest.approx(
        np.max(np.abs(d @ misfit @ np.linalg.inv(d)))
    )
    assert stm_adequacy(np.stack([phi, phi]), np.stack([phi, wrong]), n)[0] == pytest.approx(0.0)


def test_cw_stops_being_adequate_under_j2_within_one_orbit():
    # Montenbruck and Gill, Satellite Orbits, eq 7.109, with their 0.1 limit: CW against the J2
    # model at 500 km and 51.6 deg. The along-track response to a radial offset dominates; it
    # oscillates within each orbit and grows from orbit to orbit, so sampling at whole orbits
    # alone (0.053, 0.106) would put the horizon near two orbits instead of under one.
    truth = TwoBodyDynamics(inclination=np.radians(51.6), j2=True)
    truth.reset(np.zeros(6))
    period = 2 * np.pi / truth.n

    def adequacy(t):
        return stm_adequacy(truth.transition_matrices(ZERO, t)[0], cw_stm(truth.n, t), truth.n)

    times = period * np.arange(0.1, 1.01, 0.1)
    values = [adequacy(t) for t in times]
    assert 0.4 * period <= validity_horizon(times, values) <= 0.5 * period
    assert validity_horizon(times, values, limit=1.0) == np.inf
    assert adequacy(2 * period) / adequacy(period) == pytest.approx(2.0, rel=1e-2)


def index_after_one_orbit(truth, model, rho):
    """Modeling-error index over projected circular orbits of radius `rho` at four phases."""
    period = 2 * np.pi / truth.n
    finals = {id(truth): [], id(model): []}
    for alpha in np.radians([0.0, 90.0, 180.0, 270.0]):
        for m in (truth, model):
            m.reset(pco_state(rho, alpha, truth.n))
            finals[id(m)].append(m.advance(ZERO, period)[0])
    return modeling_error_index(np.array(finals[id(truth)]), np.array(finals[id(model)]), truth.n)


def test_index_is_flat_in_separation_for_effects_linear_in_it():
    # Alfriend et al., Spacecraft Formation Flying, sec. 8.3: J2 left out by CW is linear in the
    # separation, so CW's index barely changes from 100 m to 1 km.
    truth = TwoBodyDynamics(j2=True, elements=[8000e3, 0.0, np.radians(50.0), 0.0, 0.0, 0.0])
    cw = CWDynamics(truth.n)
    near, far = (index_after_one_orbit(truth, cw, rho) for rho in (100.0, 1000.0))
    assert near > 1e-3
    assert far / near == pytest.approx(1.0, abs=0.2)


def test_index_grows_with_separation_for_left_out_nonlinearity():
    elements = [8000e3, 0.01, 0.9, 0.0, 0.0, 0.0]
    truth = TwoBodyDynamics(elements=elements)
    ya = YamanakaAnkersenDynamics(8000e3, 0.01)
    near, far = (index_after_one_orbit(truth, ya, rho) for rho in (100.0, 1000.0))
    assert far / near == pytest.approx(10.0, rel=0.02)


def test_index_is_twice_a_small_relative_error():
    truth = np.array([[100.0, -50.0, 20.0, 0.01, 0.02, -0.01]])
    assert modeling_error_index(truth, 1.001 * truth, 1e-3) == pytest.approx(2.001e-3)
