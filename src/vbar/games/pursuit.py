"""KSPDG's pe1 pursuit-evasion problem on two-body dynamics, in KSPDG's formats.

The pursuer and the evader orbit a central body (Kerbin by default) as point masses under
two-body gravity. The pursuer's agent commands RCS throttles or thrusts in its body frame, the
inertial frame or its NTW frame for a burn duration, as KSPDG's `burn_vec` actions do; the
evader is flown by one of V-BAR's bots (`vbar.games.bots`), which draws any random choice from
the episode seed. The pursuer holds its forward axis
on the evader, as KSPDG's pursuer does under target-hold, and its body throttles are held for
the burn while that axis turns.

Simulated time advances only inside `step`, so an episode runs as fast as the agent and the
integrator allow and gives the same result every time. Each step lasts the action's burn
duration or `min_step` seconds, whichever is longer, coasting after a shorter burn. A step ends
early at capture (range below `capture_dist`) and at the timeout.

Sources: KSPDG v0.15.0 (github.com/mit-ll/spacegym-kspdg, commit 6251ca3), `src/kspdg/pe1/
pe1_base.py` (observation, score, capture distance, timeout, thrusters), `base_envs.py`
(actions), the `pe1_i*_init.sfs` mission saves (initial orbits) and `pe1/README.md`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from ..dynamics.bodies import KERBIN, Body
from ..dynamics.kepler import state_from_elements
from .bots import EvaderBot, PassiveEvader, Segment
from .frames import axes_along, body_axes, ntw_axes
from .vessels import MONOPROPELLANT_KG, PE1_VESSEL, Vessel, _number

log = logging.getLogger(__name__)

EPISODE_TIMEOUT = 240.0  # s, KSPDG pe1 DEFAULT_EPISODE_TIMEOUT
CAPTURE_DIST = 5.0  # m, KSPDG pe1 DEFAULT_CAPTURE_DIST
# Positions are near 7.5e5 m, so this keeps integration error near a micrometre.
_RTOL = 1e-12
_ATOL = 1e-9
# A bot whose segments end this many times in a row, each this soon after it began, is looping:
# re-arming a trigger at its own crossing advances the clock by rounding steps (about 1e-13 s).
_MAX_SHORT_SEGMENTS = 1000
_SHORT_SEGMENT = 1e-6  # s

OBSERVATION = (
    "mission_elapsed_time",
    "pursuer_mass",
    "pursuer_monopropellant",
    "pursuer_x",
    "pursuer_y",
    "pursuer_z",
    "pursuer_vx",
    "pursuer_vy",
    "pursuer_vz",
    "evader_x",
    "evader_y",
    "evader_z",
    "evader_vx",
    "evader_vy",
    "evader_vz",
)


@dataclass(frozen=True)
class Orbit:
    """A Keplerian orbit as KSP mission saves give it.

    Parameters
    ----------
    sma : float
        Semi-major axis (m).
    ecc : float
        Eccentricity.
    inc, lpe, lan : float
        Inclination, argument of periapsis and longitude of the ascending node (deg).
    mna : float
        Mean anomaly at the start (rad).
    """

    sma: float
    ecc: float
    inc: float
    lpe: float
    lan: float
    mna: float

    def state(self, mu: float) -> tuple[np.ndarray, np.ndarray]:
        """Inertial position (m) and velocity (m/s) for gravitational parameter `mu`."""
        elements = [
            self.sma,
            self.ecc,
            math.radians(self.inc),
            math.radians(self.lan),
            math.radians(self.lpe),
            self.mna,
        ]
        return state_from_elements(elements, mu)


# The four starts of KSPDG pe1, from its mission saves `pe1_i1_init.sfs` to `pe1_i4_init.sfs`
# (each vessel's ORBIT block: SMA, ECC, INC, LPE, LAN, MNA at epoch 0), digits as saved.
# I1: the pursuer on an ellipse whose apoapsis touches the evader's 750 km circle; both are about
# 20 deg short of the point of contact and reach it together.
I1_PURSUER = Orbit(724999.90000000002, 0.034483000636100769, 0.0001, 180.0, 0.0, 2.7743753789701864)
I1_EVADER = Orbit(750000.0, 0.0, 0.0001, 180.0, 0.0, 2.7925268031909272)
# I2: one 750 km circle at 1 deg relative inclination, both 2 deg past the node they share.
I2_PURSUER = Orbit(750000.0, 0.0, 1.0, 0.0, 0.0, 0.034906585039886591)
I2_EVADER = Orbit(750000.0, 0.0, 0.0001, 0.0, 0.0, 0.034906585039886591)
# I3: one 750 km circle, the pursuer 0.2 deg (2618 m) behind; the evader tilted by 1e-4 deg.
I3_PURSUER = Orbit(750000.0, 0.0, 0.0, 0.0, 0.0, 5.9306287982767323)
I3_EVADER = Orbit(750000.0, 0.0, 0.0001, 0.0, 0.0, 5.9341194567807207)
# I4: the I3 evader; the pursuer on an inclined ellipse of another period, out of phase.
I4_PURSUER = Orbit(754500.1875, 0.0072900000959634781, 1.0, 355.0, 350.0, 6.1959188445798699)
I4_EVADER = I3_EVADER
INITIAL_CONDITIONS = {
    "i1": (I1_PURSUER, I1_EVADER),
    "i2": (I2_PURSUER, I2_EVADER),
    "i3": (I3_PURSUER, I3_EVADER),
    "i4": (I4_PURSUER, I4_EVADER),
}


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be {size} numbers") from None
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be {size} finite numbers")
    return array


def _choice(value: Any, choices: range, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer")
    if int(value) not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}")
    return int(value)


def _seed(value: Any) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError("seed must be an integer")
    if value < 0:
        raise ValueError("seed must not be negative")
    return int(value)


def parse_action(action: Any) -> tuple[np.ndarray, float, int, int]:
    """The vector, burn duration, frame and vector type of a KSPDG action.

    A mapping has `burn_vec` (three vector components and a duration in s), `ref_frame` (0 body,
    1 inertial, 2 NTW) and optionally `vec_type` (0 throttle, 1 thrust in N; 0 when absent).
    Any other sequence of four numbers is body-frame throttle and duration, as in KSPDG.

    Parameters
    ----------
    action : mapping or sequence
        The action.

    Returns
    -------
    vector : numpy.ndarray, shape (3,)
    duration : float
        Burn duration (s).
    ref_frame, vec_type : int

    Raises
    ------
    ValueError
        If the action is not of either form, a number is not finite or the duration is
        negative.
    """
    if isinstance(action, Mapping):
        missing = {"burn_vec", "ref_frame"} - set(action)
        if missing:
            raise ValueError(f"action is missing {sorted(missing)}")
        burn = _finite_vector(action["burn_vec"], 4, "burn_vec")
        ref_frame = _choice(action["ref_frame"], range(3), "ref_frame")
        vec_type = _choice(action.get("vec_type", 0), range(2), "vec_type")
    else:
        burn = _finite_vector(action, 4, "an action")
        ref_frame, vec_type = 0, 0
    if burn[3] < 0.0:
        raise ValueError("burn duration must not be negative")
    return burn[:3], float(burn[3]), ref_frame, vec_type


class PursuitEvasionEnv:
    """KSPDG's pe1 game on two-body dynamics, with KSPDG's observation, action and step forms.

    `reset()` returns `(observation, info)`; `step(action)` returns `(observation, reward,
    done, info)`. The observation is a list of 15 floats (`OBSERVATION`): mission elapsed time
    (s), pursuer mass (kg), pursuer monopropellant (units of 4 kg, as kRPC reports it), then the
    pursuer's and the evader's inertial positions (m) and velocities (m/s) relative to the
    central body, in the right-handed body-centred inertial frame with z north. The reward is
    the negative weighted score on the final step and 0 before it.

    Parameters
    ----------
    initial : str
        Initial conditions, a key of `INITIAL_CONDITIONS`: KSPDG's starts "i1" to "i4".
    evader : EvaderBot, optional
        The evader's bot; `PassiveEvader()` by default.
    seed : int, optional
        Seed of the evader's random stream for every reset that is not given one. Without it,
        each such reset draws a fresh 128-bit seed from the operating system. The seed used is
        kept in `seed`.
    episode_timeout : float
        Mission elapsed time (s) at which the episode ends.
    capture_dist : float
        Range (m) below which the evader is captured and the episode ends.
    min_step : float
        Shortest step (s); a shorter burn is followed by a coast to this length.
    body : Body
        The central body; Kerbin by default.
    vessel : Vessel
        The pursuer's and the evader's vessel.
    debug : bool
        Log each step at DEBUG level, as KSPDG's `debug` flag does.
    """

    def __init__(
        self,
        *,
        initial: str = "i3",
        evader: EvaderBot | None = None,
        seed: int | None = None,
        episode_timeout: float = EPISODE_TIMEOUT,
        capture_dist: float = CAPTURE_DIST,
        min_step: float = 0.1,
        body: Body = KERBIN,
        vessel: Vessel = PE1_VESSEL,
        debug: bool = False,
    ):
        if initial not in INITIAL_CONDITIONS:
            raise ValueError(f"initial must be one of {sorted(INITIAL_CONDITIONS)}")
        self.episode_timeout = _number(episode_timeout, "episode_timeout")
        self.capture_dist = _number(capture_dist, "capture_dist")
        self.min_step = _number(min_step, "min_step")
        # A shorter step could leave the mission clock unchanged, and an agent that only coasts
        # would then never reach the timeout.
        if self.min_step < 1e-6 or self.min_step <= 8 * math.ulp(self.episode_timeout):
            raise ValueError("min_step is too short to advance the mission clock")
        self.evader = evader if evader is not None else PassiveEvader()
        if not callable(getattr(self.evader, "segment", None)):
            raise TypeError("evader must have a segment method")
        self._fixed_seed = None if seed is None else _seed(seed)
        self.seed: int | None = None
        if not isinstance(vessel, Vessel):
            raise TypeError("vessel must be a Vessel")
        if not isinstance(body, Body):
            raise TypeError("body must be a Body")
        self.initial = initial
        self.body = body
        self.vessel = vessel
        self.debug = debug
        self._y: np.ndarray | None = None
        self._closest: dict[str, float] = {}

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        """Start the episode at the initial conditions.

        Parameters
        ----------
        seed : int, optional
            Seed of the evader's random stream; by default the constructor's, else a fresh
            128-bit seed. The seed used is kept in `seed`.

        Returns
        -------
        observation : list of float
        info : dict
        """
        if seed is not None:
            seed = _seed(seed)
        elif self._fixed_seed is not None:
            seed = self._fixed_seed
        else:
            seed = int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]
        pursuer, evader = INITIAL_CONDITIONS[self.initial]
        r_p, v_p = pursuer.state(self.body.mu)
        r_e, v_e = evader.state(self.body.mu)
        wet = self.vessel.wet_mass
        self._y = np.concatenate([r_p, v_p, r_e, v_e, [wet, wet]])
        self.t = 0.0
        self.done = False
        self.captured = False
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._segment: Segment | None = None
        self._next_segment(self._y, None)
        self._empty = [False, False]
        self._closest = {}
        self._sample(self.t, self._y)
        return self.get_observation(), self.get_info()

    def get_observation(self) -> list[float]:
        """The current observation, a new list of 15 floats."""
        y = self._state()
        propellant = (y[12] - self.vessel.dry_mass) / MONOPROPELLANT_KG
        return [self.t, float(y[12]), float(propellant), *(float(x) for x in y[:12])]

    def get_info(self) -> dict[str, Any]:
        """Closest approach so far, fuel used and the weighted score.

        Returns
        -------
        dict
            `is_episode_done`; `closest_approach` (m), `closest_approach_time` (s),
            `closest_approach_speed` (m/s) and `closest_approach_pursuer_fuel_usage` (kg) at the
            smallest range so far; `pursuer_fuel_usage` and `evader_fuel_usage` (kg) now; and
            `weighted_score`, KSPDG's pe1 score of the closest approach.
        """
        y = self._state()
        closest = self._closest
        return {
            "is_episode_done": self.done,
            "closest_approach": closest["d"],
            "closest_approach_time": closest["t"],
            "closest_approach_speed": closest["v"],
            "closest_approach_pursuer_fuel_usage": closest["f"],
            "pursuer_fuel_usage": self.vessel.wet_mass - float(y[12]),
            "evader_fuel_usage": self.vessel.wet_mass - float(y[13]),
            "weighted_score": weighted_score(
                closest["d"], closest["v"], closest["f"], closest["t"]
            ),
        }

    def step(self, action: Any) -> tuple[list[float], float, bool, dict[str, Any]]:
        """Fly one KSPDG action.

        Parameters
        ----------
        action : mapping or sequence
            A KSPDG action (`parse_action`).

        Returns
        -------
        observation : list of float
        reward : float
            The negative weighted score on the step that ends the episode, else 0.
        done : bool
        info : dict

        Raises
        ------
        ValueError
            If the action is malformed (the episode is unchanged).
        RuntimeError
            If the episode has not been reset or has ended.
        """
        vector, duration, ref_frame, vec_type = parse_action(action)
        y = self._state()
        if self.done:
            raise RuntimeError("the episode has ended; call reset")
        r_p, v_p, r_e = y[0:3], y[3:6], y[6:9]
        body = body_axes(r_p, r_e)
        if ref_frame == 1:
            vector = body @ vector
        elif ref_frame == 2:
            vector = body @ (ntw_axes(r_p, v_p).T @ vector)
        throttle = self.vessel.throttle(vector) if vec_type == 1 else vector
        throttle = np.clip(throttle, -1.0, 1.0)
        start = self.t
        self._fly(throttle, duration)
        self._fly(np.zeros(3), max(self.min_step - (self.t - start), 0.0))
        if self.t >= self.episode_timeout:
            self.done = True
        info = self.get_info()
        reward = -info["weighted_score"] if self.done else 0.0
        if self.debug:
            log.debug("t %.3f s, throttle %s, done %s", self.t, throttle, self.done)
        return self.get_observation(), reward, self.done, info

    def close(self) -> None:
        """Nothing to release; present because KSPDG environments have it."""

    def _state(self) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("call reset before using the environment")
        return self._y

    def _sample(self, t: float, y: np.ndarray) -> None:
        dr = y[6:9] - y[0:3]
        d = float(np.linalg.norm(dr))
        if not self._closest or d < self._closest["d"]:
            self._closest = {
                "d": d,
                "v": float(np.linalg.norm(y[9:12] - y[3:6])),
                "f": self.vessel.wet_mass - float(y[12]),
                "t": float(t),
            }

    def _next_segment(self, y: np.ndarray, cause: int | None) -> None:
        segment = self.evader.segment(
            self.t, y[0:3], y[3:6], y[6:9], y[9:12], self._segment, cause, self._rng
        )
        if not isinstance(segment, Segment):
            raise TypeError("an evader bot must return a Segment")
        self._segment = segment

    def _change_segment(self, y: np.ndarray, cause: int | None, changes: int, start: float) -> int:
        """Take the bot's next segment; count the segments in a row that ended almost at once."""
        changes = changes + 1 if self.t - start < _SHORT_SEGMENT else 0
        if changes > _MAX_SHORT_SEGMENTS:
            raise RuntimeError("the evader bot keeps changing segment with no time passing")
        self._next_segment(y, cause)
        return changes

    def _rhs(self, throttle: np.ndarray, pursuer_thrusts: bool, evader_thrusts: bool):
        mu = self.body.mu
        vessel = self.vessel
        thrust_body = vessel.thrust(throttle)
        flow = vessel.mass_flow(thrust_body)
        thrust = thrust_body.tolist()
        segment = self._segment
        assert segment is not None
        thrust_e_body = vessel.thrust(np.array(segment.throttle))
        flow_e = vessel.mass_flow(thrust_e_body)
        thrust_e = thrust_e_body.tolist()
        prograde = segment.pointing == "prograde"

        def rhs(t, y):
            xp, yp, zp, vxp, vyp, vzp, xe, ye, ze, vxe, vye, vze, m_p, m_e = y.tolist()
            k_p = -mu / (xp * xp + yp * yp + zp * zp) ** 1.5
            k_e = -mu / (xe * xe + ye * ye + ze * ze) ** 1.5
            a_p = [k_p * xp, k_p * yp, k_p * zp]
            a_e = [k_e * xe, k_e * ye, k_e * ze]
            dm_p = dm_e = 0.0
            if pursuer_thrusts:
                axes = body_axes((xp, yp, zp), (xe, ye, ze))
                for i in range(3):
                    a_p[i] += (
                        axes[0, i] * thrust[0] + axes[1, i] * thrust[1] + axes[2, i] * thrust[2]
                    ) / m_p
                dm_p = -flow
            if evader_thrusts:
                if prograde:
                    forward = (vxe, vye, vze)
                else:
                    forward = (ye * vze - ze * vye, ze * vxe - xe * vze, xe * vye - ye * vxe)
                axes = axes_along((xe, ye, ze), forward)
                for i in range(3):
                    a_e[i] += (
                        axes[0, i] * thrust_e[0]
                        + axes[1, i] * thrust_e[1]
                        + axes[2, i] * thrust_e[2]
                    ) / m_e
                dm_e = -flow_e
            return np.array([vxp, vyp, vzp, *a_p, vxe, vye, vze, *a_e, dm_p, dm_e])

        return rhs

    def _fly(self, throttle: np.ndarray, duration: float) -> None:
        """Advance `duration` s (less at capture or timeout) holding the pursuer's throttles."""
        end = min(self.t + duration, self.episode_timeout)
        dry = self.vessel.dry_mass
        changes = 0
        while not self.done and self.t < end:
            y = self._state()
            # The crossing event cannot see a range that is already inside the capture distance.
            if np.linalg.norm(y[6:9] - y[0:3]) < self.capture_dist:
                self.done = self.captured = True
                break
            segment = self._segment
            assert segment is not None
            if segment.until <= self.t:
                changes = self._change_segment(y, None, changes, self.t)
                continue
            stop = min(end, segment.until)
            pursuer_thrusts = bool(np.any(throttle != 0.0)) and not self._empty[0]
            evader_thrusts = any(x != 0.0 for x in segment.throttle) and not self._empty[1]

            def capture(t, y):
                return np.linalg.norm(y[6:9] - y[0:3]) - self.capture_dist

            def closing(t, y):
                return (y[6:9] - y[0:3]) @ (y[9:12] - y[3:6])

            def pursuer_empty(t, y):
                return y[12] - dry

            def evader_empty(t, y):
                return y[13] - dry

            capture.terminal, capture.direction = True, -1  # type: ignore[attr-defined]
            closing.terminal, closing.direction = False, 1  # type: ignore[attr-defined]
            pursuer_empty.terminal = pursuer_thrusts  # type: ignore[attr-defined]
            evader_empty.terminal = evader_thrusts  # type: ignore[attr-defined]
            pursuer_empty.direction = evader_empty.direction = -1  # type: ignore[attr-defined]
            events = [capture, closing, pursuer_empty, evader_empty]
            for g, direction in segment.triggers:

                def trigger(t, y, g=g):
                    return g(y[0:3], y[3:6], y[6:9], y[9:12])

                trigger.terminal = True  # type: ignore[attr-defined]
                trigger.direction = direction  # type: ignore[attr-defined]
                events.append(trigger)
            start = self.t
            solution = solve_ivp(
                self._rhs(throttle, pursuer_thrusts, evader_thrusts),
                (self.t, stop),
                y,
                method="DOP853",
                rtol=_RTOL,
                atol=_ATOL,
                events=events,
                # A whole step of a few seconds is a small fraction of the 36-minute orbit, so
                # trying it in one integrator step usually succeeds and saves the warm-up steps.
                first_step=stop - self.t,
            )
            if solution.status < 0:
                raise RuntimeError(f"integration failed: {solution.message}")
            for t_event, y_event in zip(solution.t_events[1], solution.y_events[1], strict=True):
                self._sample(t_event, y_event)
            y_end = solution.y[:, -1].copy()
            self.t = float(solution.t[-1])
            self._y = y_end
            self._sample(self.t, y_end)
            if solution.t_events[0].size:
                self.done = self.captured = True
            if pursuer_thrusts and solution.t_events[2].size:
                self._empty[0] = True
            if evader_thrusts and solution.t_events[3].size:
                self._empty[1] = True
            if self.done:
                continue
            fired = [i for i, e in enumerate(solution.t_events[4:]) if e.size]
            if fired:
                changes = self._change_segment(y_end, fired[0], changes, start)
            elif self.t >= segment.until:
                changes = self._change_segment(y_end, None, changes, start)


def weighted_score(d: float, v: float, f: float, t: float) -> float:
    """KSPDG's pe1 score: (0.1 d)^2 + (0.5 v)^1.5 + (0.1 f)^1.25 + 0.01 t, lower is better.

    Parameters
    ----------
    d : float
        Closest approach (m).
    v : float
        Relative speed at the closest approach (m/s).
    f : float
        Pursuer propellant used by then (kg).
    t : float
        Mission elapsed time of the closest approach (s).

    Returns
    -------
    float
    """
    return (0.1 * d) ** 2.0 + (0.5 * v) ** 1.5 + (0.1 * f) ** 1.25 + 0.01 * t


__all__ = [
    "CAPTURE_DIST",
    "EPISODE_TIMEOUT",
    "INITIAL_CONDITIONS",
    "OBSERVATION",
    "Orbit",
    "PursuitEvasionEnv",
    "parse_action",
    "weighted_score",
]
