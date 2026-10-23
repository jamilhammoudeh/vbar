"""Two-impulse targeting of the hold point by Lambert's problem on a declared circular orbit.

The agent rebuilds the target's circular orbit from the view's mean motion and a declared
gravitational parameter, a = (mu / n^2)^(1/3), places the observed Hill state and the hold point
on it, and solves Lambert's problem between them (`vbar.dynamics.lambert.lambert`; Curtis,
Orbital Mechanics for Engineering Students, algorithm 5.2). The plan is exact for two-body motion
about a circular target; J2 and eccentricity of the true orbit are outside its model.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..conventions import hill_frame, hill_to_inertial
from ..dynamics import EARTH
from ..dynamics.lambert import lambert
from .base import AgentView, action_scale, hold_point
from .elements import Element, ElementAgent
from .planning import Impulse, impulse_action

MODES = ("retarget", "open_loop")
# Transfer angles this close to half an orbit (rad) leave the transfer plane undetermined.
SINGULAR_TOL = 1e-2
# Steps before the arrival below which "retarget" brakes instead of solving again, since the
# first impulse grows as the error over the shrinking time left.
_BRAKE_STEPS = 2


def _circular(a: float, n: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Inertial position (m) and velocity (m/s) at `angle` (rad) on a circle in the x-y plane."""
    c, s = math.cos(angle), math.sin(angle)
    return a * np.array([c, s, 0.0]), n * a * np.array([-s, c, 0.0])


class LambertAgent:
    """Two-impulse transfer to rest at the hold point, planned by Lambert's problem.

    The plan uses declared model knowledge: the gravitational parameter `mu`, which with the
    view's mean motion n fixes the radius a = (mu / n^2)^(1/3) of the target's circular orbit.
    At each plan the target is placed at angle 0 on that orbit, the chaser at the observed Hill
    state relative to it (`vbar.conventions.hill_to_inertial`), and the aim point at the hold
    point relative to the target after the transfer time tf. Lambert's problem between the
    chaser and the aim point gives the departure impulse, the inertial velocity change, which
    equals its Hill components because the target's Hill axes at angle 0 are the inertial ones;
    the arrival impulse is the velocity change that leaves the chaser at rest at the hold point,
    in the Hill axes at arrival. Lambert's problem is invariant under rotation, so the orbit's
    inclination and node do not enter. The agent reads only the observation and
    `info["time_s"]`.

    "retarget" solves again at every decision for the time left and fires the departure
    impulse; fewer than two steps before the arrival it cancels the measured velocity until
    that burn is delivered, and it coasts through decisions whose time left puts the transfer
    within `SINGULAR_TOL` of half an orbit. "open_loop" plans both impulses at the first decision
    and flies them as an `vbar.agents.elements.ElementAgent`, committing the coast between them
    in "impulse_coast". After the arrival both modes hover where they are with
    (-3 n^2 x, 0, n^2 z).

    `log` holds one dict per decision with the time "t" (s), the "phase" and whether the
    command "saturated". "retarget" phases are "transfer", "singular", "arrival" and "hold";
    "open_loop" phases are those of `ElementAgent`, and its departure burn adds "coast_s" and the
    keep-out margin predicted over the coast, "predicted_margin", with its certified lower bound,
    "predicted_margin_bound" (m, from `planning.predicted_margin`). The prediction propagates CW
    motion, not the two-body model of the plan. A predicted violation is logged and the plan
    flown as is.

    Parameters
    ----------
    n_tf : float
        Transfer angle n tf (rad), below 2 pi (single revolution) and at least `SINGULAR_TOL`
        from pi and from 2 pi.
    mu : float
        Declared gravitational parameter of the central body (m^3/s^2).
    mode : str
        "retarget" or "open_loop".
    name : str
        Name used in results.

    Raises
    ------
    ValueError
        If `mode` is unknown, `mu` is not positive and finite, or `n_tf` is not positive and
        finite, is within `SINGULAR_TOL` of pi, or exceeds 2 pi - `SINGULAR_TOL`.
    """

    def __init__(
        self,
        *,
        n_tf: float = 1.0,
        mu: float = EARTH.mu,
        mode: str = "retarget",
        name: str = "lambert",
    ):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        if not (math.isfinite(mu) and mu > 0.0):
            raise ValueError("mu must be positive and finite")
        if not (math.isfinite(n_tf) and n_tf > 0.0):
            raise ValueError("n_tf must be positive and finite")
        if abs(n_tf - math.pi) < SINGULAR_TOL or n_tf > 2.0 * math.pi - SINGULAR_TOL:
            raise ValueError("n_tf must be away from pi and below 2 pi")
        self.n_tf = n_tf
        self.mu = mu
        self.mode = mode
        self.name = name
        self.log: list[dict[str, Any]] = []

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode with the scenario's public description.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, the action mode is
            "disc27" or unknown, or thrust actions come without `mass_kg`.
        """
        self.hold = hold_point(view, "Lambert agent")
        if view.action_mode == "disc27":
            raise ValueError("disc27 actions cannot deliver a planned impulse")
        if view.action_mode != "impulse_coast":
            action_scale(view)
        self.view = view
        self.radius = (self.mu / view.n**2) ** (1.0 / 3.0)
        self.t0: float | None = None
        self.braking = False
        self.arrived = False
        if self.mode == "open_loop":
            self.flight = ElementAgent(self._element, name=self.name)
            self.flight.reset(view, rng)
            self.log = self.flight.log
        else:
            self.log = []

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """The action for this decision in the view's action mode; appends one `log` entry."""
        if self.mode == "open_loop":
            return self.flight.act(obs, info)
        view = self.view
        t = float(info["time_s"])
        state = np.asarray(obs[:6], dtype=float)
        if self.t0 is None:
            self.t0 = t
            self.t_arrival = t + self.n_tf / view.n
        entry: dict[str, Any] = {"t": t}
        time_left = self.t_arrival - t
        if self.arrived:
            entry["phase"] = "hold"
            hover = np.array([-3.0 * view.n**2 * state[0], 0.0, view.n**2 * state[2]])
            dv = hover * view.dt
        elif not self.braking and round(time_left / view.dt) >= _BRAKE_STEPS:
            if abs(view.n * time_left - math.pi) < SINGULAR_TOL:
                # The chaser is already on a transfer; coasting crosses the half-orbit time.
                entry["phase"] = "singular"
                dv = np.zeros(3)
            else:
                entry["phase"] = "transfer"
                dv, _ = self._impulses(state, time_left)
        else:
            # A saturated arrival burn carries on, re-measured, until it is delivered.
            self.braking = True
            entry["phase"] = "arrival"
            dv = -state[3:6]
        action, saturated = impulse_action(view, dv, 0.0)
        entry["saturated"] = saturated
        if entry["phase"] == "arrival" and not saturated:
            self.arrived = True
        self.log.append(entry)
        return action

    def _impulses(self, state: np.ndarray, tf: float) -> tuple[np.ndarray, np.ndarray]:
        """Departure and arrival impulses (Hill, m/s) from `state` to rest at the hold point.

        Parameters
        ----------
        state : numpy.ndarray, shape (6,)
            Chaser state [x, y, z, vx, vy, vz] in the Hill frame (m, m/s).
        tf : float
            Transfer time (s).

        Returns
        -------
        dv1 : numpy.ndarray, shape (3,)
            Departure impulse in the Hill axes at departure (m/s).
        dv2 : numpy.ndarray, shape (3,)
            Arrival impulse in the Hill axes at arrival (m/s).

        Raises
        ------
        ValueError
            If the chaser and the aim point are collinear with the centre of the orbit.
        RuntimeError
            If Lambert's equation does not converge.
        """
        n, a = self.view.n, self.radius
        r_target, v_target = _circular(a, n, 0.0)
        dr, dv = hill_to_inertial(r_target, v_target, state)
        r_aim_target, v_aim_target = _circular(a, n, n * tf)
        dr_aim, dv_aim = hill_to_inertial(r_aim_target, v_aim_target, np.r_[self.hold, 0, 0, 0])
        v1, v2 = lambert(r_target + dr, r_aim_target + dr_aim, tf, self.mu)
        axes, _ = hill_frame(r_aim_target, v_aim_target)
        return v1 - (v_target + dv), axes.T @ (v_aim_target + dv_aim - v2)

    def _element(self, view: AgentView, state: np.ndarray) -> Element:
        tf = self.n_tf / view.n
        dv1, dv2 = self._impulses(state, tf)
        return Element(impulses=(Impulse(0.0, dv1), Impulse(tf, dv2)), duration=tf)
