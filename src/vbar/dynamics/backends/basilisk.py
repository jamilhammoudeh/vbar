"""Basilisk (AVS Lab, University of Colorado) as an independent propagator.

Two `spacecraft.Spacecraft` hubs, the target and the chaser, fly under one central body whose
gravity is a degree-2 spherical-harmonic model holding only the normalized C20 = -J2 / sqrt(5)
of V-BAR's body (or the point mass alone), integrated with Basilisk's adaptive RKF78. A burn is
an `extForceTorque` force on the chaser whose inertial direction is set every `substep` from the
target's Hill frame at the substep's middle and held over it, the one approximation the backend
adds (second order in the angle the frame turns in a substep).
Requires the optional extra: pip install "vbar[basilisk]".
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..bodies import Body
from .common import Burn, check_aligned, inertial_accel

INSTALL = 'pip install "vbar[basilisk]"'


def _modules():
    try:
        from Basilisk.simulation import (
            extForceTorque,
            gravityEffector,
            spacecraft,
            svIntegrators,
        )
        from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
    except ImportError as error:
        raise ImportError(f"the Basilisk backend needs Basilisk: {INSTALL}") from error
    return (
        extForceTorque,
        gravityEffector,
        spacecraft,
        svIntegrators,
        SimulationBaseClass,
        macros,
        simIncludeGravBody,
    )


def propagate(
    target: ArrayLike,
    chaser: ArrayLike,
    times: ArrayLike,
    body: Body,
    j2: bool = True,
    burns: Sequence[Burn] = (),
    tolerance: float = 1e-12,
    substep: float = 1.0,
    mass: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fly a target and a chaser from inertial states and sample both at `times`.

    Parameters
    ----------
    target, chaser : array_like
        Initial inertial [r, v] (m, m/s), shape (6,).
    times : array_like
        Sample times (s) from the start, increasing, each a multiple of `substep`.
    body : Body
        Central body: its GM, equatorial radius and J2.
    j2 : bool
        Whether gravity includes J2.
    burns : sequence of Burn
        Chaser accelerations, held in the target's Hill axes.
    tolerance : float
        Relative and absolute tolerance of the RKF78 integrator.
    substep : float
        Basilisk task step (s); also how often a burn's inertial direction is updated.
    mass : float
        Chaser mass (kg), for the force that gives the burns' acceleration.

    Returns
    -------
    tuple of numpy.ndarray
        Target and chaser inertial states at `times`, each shape (len(times), 6).

    Raises
    ------
    ImportError
        If Basilisk is not installed.
    ValueError
        If a time or a burn edge is not a whole number of substeps.
    """
    ext, grav_effector, spacecraft, integrators, base, macros, grav_bodies = _modules()
    samples = np.asarray(times, dtype=float)
    steps = np.round(samples / substep).astype(int)
    if not np.allclose(steps * substep, samples, rtol=0.0, atol=1e-9):
        raise ValueError("sample times must be whole numbers of substeps")
    check_aligned(burns, substep)
    sim = base.SimBaseClass()
    process = sim.CreateNewProcess("dynamics")
    process.addTask(sim.CreateNewTask("step", macros.sec2nano(substep)))
    factory = grav_bodies.gravBodyFactory()
    planet = factory.createCustomGravObject(body.name, body.mu)
    planet.isCentralBody = True
    planet.mu = body.mu
    planet.radEquator = body.radius
    if j2:
        model = grav_effector.SphericalHarmonicsGravityModel()
        model.muBody = body.mu
        model.radEquator = body.radius
        model.cBar = [[1.0], [0.0, 0.0], [-body.j2 / math.sqrt(5.0), 0.0, 0.0]]
        model.sBar = [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0]]
        model.maxDeg = 2
        planet.gravityModel = model
    hubs, kept = [], []
    for name, state in (("target", target), ("chaser", chaser)):
        x = np.asarray(state, dtype=float)
        hub = spacecraft.Spacecraft()
        hub.ModelTag = name
        hub.hub.mHub = mass
        hub.hub.r_CN_NInit = [[c] for c in x[:3]]
        hub.hub.v_CN_NInit = [[c] for c in x[3:]]
        integrator = integrators.svIntegratorRKF78(hub)
        integrator.relTol = tolerance
        integrator.absTol = tolerance
        hub.setIntegrator(integrator)
        factory.addBodiesTo(hub)
        sim.AddModelToTask("step", hub)
        hubs.append(hub)
        kept.append(integrator)
    force = ext.ExtForceTorque()
    force.ModelTag = "burns"
    hubs[1].addDynamicEffector(force)
    sim.AddModelToTask("step", force)
    sim.InitializeSimulation()
    # A coast runs straight from sample to sample; a burn needs its direction updated every
    # substep.
    stops = set(steps.tolist()) | {0}
    if burns:
        stops |= set(range(int(steps.max()) + 1))
    out_t, out_c = [], []
    k = 0
    for stop in sorted(stops):
        while k < len(steps) and steps[k] == stop:
            out_t.append(_state(hubs[0]))
            out_c.append(_state(hubs[1]))
            k += 1
        if stop == steps.max():
            break
        accel = inertial_accel(burns, stop * substep, _state(hubs[0]), body, j2, substep)
        force.extForce_N = [[mass * c] for c in accel]
        following = min(s for s in stops if s > stop)
        sim.ConfigureStopTime(macros.sec2nano(following * substep))
        sim.ExecuteSimulation()
    return np.array(out_t), np.array(out_c)


def _state(hub) -> np.ndarray:
    states = hub.scStateOutMsg.read()
    return np.concatenate(
        [np.array(states.r_BN_N, dtype=float), np.array(states.v_BN_N, dtype=float)]
    )


__all__ = ["INSTALL", "propagate"]
