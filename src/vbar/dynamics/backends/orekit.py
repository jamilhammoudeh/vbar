"""Orekit (CS GROUP, through orekit-jpype) as an independent propagator.

Each spacecraft is flown by its own `NumericalPropagator` in GCRF with a Dormand-Prince 8(5,3)
integrator, whose tolerances come from Orekit's own rule for a position error `tolerance`, under
the central body's `NewtonianAttraction` and, optionally, `J2OnlyPerturbation` with V-BAR's GM,
radius and J2 about the frame's z axis. A burn is a `ConstantThrustManeuver` on the chaser per
`substep`, in inertial attitude, with the target's Hill direction at the substep's middle, and
a specific impulse so high that the mass does not change. Requires the optional extra and a Java
runtime (11 or later) on JAVA_HOME: pip install "vbar[orekit]".
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cache

import numpy as np
from numpy.typing import ArrayLike

from ..bodies import Body
from .common import Burn, check_aligned, inertial_accel

INSTALL = 'pip install "vbar[orekit]" and a Java runtime on JAVA_HOME'
# High enough that a burn's propellant is below the mass's rounding.
_ISP = 1e12


@cache
def _orekit():
    try:
        import orekit_jpype
    except ImportError as error:
        raise ImportError(f"the Orekit backend needs orekit-jpype: {INSTALL}") from error
    if "JAVA_HOME" not in os.environ:
        try:
            import jdk4py

            os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
        except ImportError:
            pass
    try:
        orekit_jpype.initVM()
    except Exception as error:  # jpype raises a ValueError subclass when no JVM is found
        raise ImportError(f"the Orekit backend needs a Java runtime: {INSTALL}") from error
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.attitudes import FrameAlignedProvider
    from org.orekit.forces.gravity import J2OnlyPerturbation
    from org.orekit.forces.maneuvers import ConstantThrustManeuver
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import CartesianOrbit, OrbitType
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.utils import PVCoordinates

    return {
        "Vector3D": Vector3D,
        "Integrator": DormandPrince853Integrator,
        "Inertial": FrameAlignedProvider,
        "J2": J2OnlyPerturbation,
        "Maneuver": ConstantThrustManeuver,
        "frame": FramesFactory.getGCRF(),
        "Orbit": CartesianOrbit,
        "OrbitType": OrbitType,
        "State": SpacecraftState,
        "Propagator": NumericalPropagator,
        "epoch": AbsoluteDate(2026, 1, 1, 0, 0, 0.0, TimeScalesFactory.getTAI()),
        "PV": PVCoordinates,
    }


def _propagator(o, state: np.ndarray, start: float, body: Body, j2: bool, tolerance: float, mass):
    frame, epoch = o["frame"], o["epoch"]
    pv = o["PV"](o["Vector3D"](*map(float, state[:3])), o["Vector3D"](*map(float, state[3:])))
    orbit = o["Orbit"](pv, frame, epoch.shiftedBy(float(start)), float(body.mu))
    tol = o["Propagator"].tolerances(float(tolerance), orbit, o["OrbitType"].CARTESIAN)
    integrator = o["Integrator"](1e-6, 300.0, tol[0], tol[1])
    propagator = o["Propagator"](integrator)
    propagator.setOrbitType(o["OrbitType"].CARTESIAN)
    propagator.setMu(float(body.mu))
    if j2:
        propagator.addForceModel(o["J2"](float(body.mu), float(body.radius), float(body.j2), frame))
    propagator.setInitialState(o["State"](orbit, float(mass)))
    return propagator


def _read(o, spacecraft_state) -> np.ndarray:
    pv = spacecraft_state.getPVCoordinates(o["frame"])
    p, v = pv.getPosition(), pv.getVelocity()
    return np.array([p.getX(), p.getY(), p.getZ(), v.getX(), v.getY(), v.getZ()])


def propagate(
    target: ArrayLike,
    chaser: ArrayLike,
    times: ArrayLike,
    body: Body,
    j2: bool = True,
    burns: Sequence[Burn] = (),
    tolerance: float = 1e-6,
    substep: float = 1.0,
    mass: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fly a target and a chaser from inertial states and sample both at `times`.

    Parameters
    ----------
    target, chaser : array_like
        Initial inertial [r, v] (m, m/s), shape (6,).
    times : array_like
        Sample times (s) from the start, increasing; with burns, whole numbers of `substep`.
    body : Body
        Central body: its GM, equatorial radius and J2.
    j2 : bool
        Whether gravity includes J2.
    burns : sequence of Burn
        Chaser accelerations, held in the target's Hill axes.
    tolerance : float
        Position error (m) Orekit's tolerance rule sets the integrator for.
    substep : float
        How long a burn's inertial direction is held (s).
    mass : float
        Chaser mass (kg).

    Returns
    -------
    tuple of numpy.ndarray
        Target and chaser inertial states at `times`, each shape (len(times), 6).

    Raises
    ------
    ImportError
        If orekit-jpype or a Java runtime is missing.
    ValueError
        If burns are given and a time or a burn edge is not a whole number of substeps.
    """
    o = _orekit()
    samples = np.asarray(times, dtype=float)
    x_t, x_c = np.asarray(target, dtype=float), np.asarray(chaser, dtype=float)
    epoch = o["epoch"]
    if not burns:
        out = []
        for state in (x_t, x_c):
            propagator = _propagator(o, state, 0.0, body, j2, tolerance, mass)
            out.append(
                np.array(
                    [_read(o, propagator.propagate(epoch.shiftedBy(float(t)))) for t in samples]
                )
            )
        return out[0], out[1]
    steps = np.round(samples / substep).astype(int)
    if not np.allclose(steps * substep, samples, rtol=0.0, atol=1e-9):
        raise ValueError("with burns, sample times must be whole numbers of substeps")
    check_aligned(burns, substep)
    out_t, out_c = [], []
    k = 0
    for step in range(int(steps.max()) + 1):
        while k < len(steps) and steps[k] == step:
            out_t.append(x_t.copy())
            out_c.append(x_c.copy())
            k += 1
        if step == steps.max():
            break
        start, end = step * substep, (step + 1) * substep
        accel = inertial_accel(burns, start, x_t, body, j2, substep)
        target_prop = _propagator(o, x_t, start, body, j2, tolerance, mass)
        chaser_prop = _propagator(o, x_c, start, body, j2, tolerance, mass)
        size = float(np.linalg.norm(accel))
        if size > 0.0:
            direction = o["Vector3D"](*map(float, accel / size))
            maneuver = o["Maneuver"](
                epoch.shiftedBy(float(start)),
                float(substep),
                float(mass * size),
                _ISP,
                o["Inertial"](o["frame"]),
                direction,
            )
            chaser_prop.addForceModel(maneuver)
        x_t = _read(o, target_prop.propagate(epoch.shiftedBy(float(end))))
        x_c = _read(o, chaser_prop.propagate(epoch.shiftedBy(float(end))))
    return np.array(out_t), np.array(out_c)


__all__ = ["INSTALL", "propagate"]
