"""Trajectory elements of rendezvous practice in the Hill frame.

Impulsive transfers and fly-arounds, forced straight-line motion and finite-burn durations follow
Fehse, Automated Rendezvous and Docking of Spacecraft, sections 3.3 and 4.4; the twice-orbital-rate
circle follows Goodman, History of Space Shuttle Rendezvous (JSC-63400), p. 137. Fehse's local
orbital frame (x along the V-bar, y = -h, z toward Earth) maps to the Hill frame by
`vbar.conventions.iss_to_hill`, so his point dz below the target is Hill (-dz, 0, 0). Every
element's impulse times count from its own start.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..dynamics import cw_discrete
from .base import AgentView, action_scale, command_fraction, hold_point
from .cw_targeting import SINGULAR_SET, conditioning, cw_lambert
from .lyapunov import CWTrackingAgent
from .planning import Impulse, impulse_action, predicted_margin


@dataclass(frozen=True, kw_only=True)
class Element:
    """A planned trajectory element in the Hill frame.

    Parameters
    ----------
    impulses : tuple of Impulse
        Impulsive velocity changes (m/s), each at a time (s) since the element's start.
    duration : float
        Time from the element's start to its last impulse (s).
    accel : callable or None
        Thrust acceleration ``accel(t, state)`` (m/s^2, Hill) between the impulses, with `t` in
        s since the element's start and `state` = [x, y, z, vx, vy, vz] (m, m/s); None for
        free coasts.
    """

    impulses: tuple[Impulse, ...]
    duration: float
    accel: Callable[[float, np.ndarray], np.ndarray] | None = None


def _check_n(n: float) -> None:
    if not (math.isfinite(n) and n > 0.0):
        raise ValueError("n must be positive and finite")


def _along(dv: float, axis: int) -> np.ndarray:
    """Impulse of size `dv` (m/s) along Hill axis `axis`, exactly zero on the others."""
    vec = np.zeros(3)
    vec[axis] = dv
    return vec


def tangential_transfer(n: float, dy: float) -> Element:
    """Two along-track impulses one orbit apart that move a chaser along the V-bar by `dy`.

    A +y impulse dvy moves the chaser by -6 pi dvy / n in one orbit, so the transfer burns
    -n dy / (6 pi) along y at the start and the opposite at the end; the total is
    n |dy| / (3 pi) (Fehse eqs 3.29 to 3.32).

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    dy : float
        Along-track displacement (m), from rest on the V-bar to rest on the V-bar.

    Returns
    -------
    Element
        Impulses at 0 and 2 pi / n (s); no forced acceleration.

    Raises
    ------
    ValueError
        If `n` is not positive and finite.
    """
    _check_n(n)
    burn = -n * dy / (6.0 * math.pi)
    period = 2.0 * math.pi / n
    return Element(
        impulses=(Impulse(0.0, _along(burn, 1)), Impulse(period, _along(-burn, 1))),
        duration=period,
    )


def radial_transfer(n: float, dy: float) -> Element:
    """Two radial impulses half an orbit apart that move a chaser along the V-bar by `dy`.

    A radial impulse dvx moves the chaser by -4 dvx / n in half an orbit and arrives with the
    radial velocity reversed, so both burns are -n dy / 4 along x; the total is n |dy| / 2,
    3 pi / 2 times the tangential transfer (Fehse eqs 3.34, 3.37).

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    dy : float
        Along-track displacement (m), from rest on the V-bar to rest on the V-bar.

    Returns
    -------
    Element
        Impulses at 0 and pi / n (s); no forced acceleration.

    Raises
    ------
    ValueError
        If `n` is not positive and finite.
    """
    _check_n(n)
    burn = _along(-n * dy / 4.0, 0)
    return Element(
        impulses=(Impulse(0.0, burn), Impulse(math.pi / n, burn.copy())), duration=math.pi / n
    )


def cw_hohmann(n: float, dx: float) -> Element:
    """Hohmann transfer in CW form that raises a chaser by `dx` onto the target orbit.

    The chaser starts on the coelliptic orbit dx below, at x = -dx with along-track speed
    1.5 n dx. Two along-track burns of n dx / 4 half an orbit apart bring it to rest at x = 0
    and advance it along the V-bar by 3 pi dx / 4 (Fehse section 3.3).

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    dx : float
        Height of the target orbit above the starting orbit (m); negative lowers the chaser.

    Returns
    -------
    Element
        Impulses at 0 and pi / n (s); no forced acceleration.

    Raises
    ------
    ValueError
        If `n` is not positive and finite.
    """
    _check_n(n)
    burn = _along(n * dx / 4.0, 1)
    return Element(
        impulses=(Impulse(0.0, burn), Impulse(math.pi / n, burn.copy())), duration=math.pi / n
    )


def tangential_flyaround(n: float, dz: float) -> Element:
    """Along-track impulses that carry a chaser from behind the target to the point dz below.

    From rest on the V-bar at y = -3 pi dz / 4, a burn of -n dz / 4 along y reaches Hill
    (-dz, 0, 0) after half an orbit with along-track speed 7 n dz / 4, which the second burn
    cancels; the total is 2 n |dz| (Fehse eq 3.33).

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    dz : float
        Depth of the end point below the target (m).

    Returns
    -------
    Element
        Impulses at 0 and pi / n (s); no forced acceleration.

    Raises
    ------
    ValueError
        If `n` is not positive and finite.
    """
    _check_n(n)
    return Element(
        impulses=(
            Impulse(0.0, _along(-n * dz / 4.0, 1)),
            Impulse(math.pi / n, _along(-7.0 * n * dz / 4.0, 1)),
        ),
        duration=math.pi / n,
    )


def radial_flyaround(n: float, dz: float) -> Element:
    """A radial and an along-track impulse that carry a chaser from behind to dz below.

    From rest on the V-bar at y = -2 dz, a burn of -n dz along x reaches Hill (-dz, 0, 0)
    after a quarter orbit with along-track speed 2 n dz and no radial speed, which the second
    burn cancels; the total is 3 n |dz| (Fehse eqs 3.38, 3.39).

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    dz : float
        Depth of the end point below the target (m).

    Returns
    -------
    Element
        Impulses at 0 and pi / (2 n) (s); no forced acceleration.

    Raises
    ------
    ValueError
        If `n` is not positive and finite.
    """
    _check_n(n)
    quarter = 0.5 * math.pi / n
    return Element(
        impulses=(
            Impulse(0.0, _along(-n * dz, 0)),
            Impulse(quarter, _along(-2.0 * n * dz, 1)),
        ),
        duration=quarter,
    )


def forced_motion_accel(n: float, r: ArrayLike, v: ArrayLike, a_path: ArrayLike) -> np.ndarray:
    """Thrust acceleration that makes a chaser follow a path with acceleration `a_path`.

    The prescribed path acceleration minus the CW terms (3 n^2 x + 2 n vy, -2 n vx, -n^2 z).
    On the V-bar at constant speed it is (-2 n vy, 0, 0) (Fehse eq 3.44), on the R-bar
    (-3 n^2 x, 2 n vx, 0) (eqs 3.49 to 3.53), and along a circle it is Fehse eq 3.79.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    r, v : array_like, shape (3,)
        Position (m) and velocity (m/s) on the path, Hill frame.
    a_path : array_like, shape (3,)
        Acceleration of the path at that point, seen in the Hill frame (m/s^2).

    Returns
    -------
    numpy.ndarray, shape (3,)
        Thrust acceleration in the Hill frame (m/s^2).
    """
    x, _, z = np.asarray(r, dtype=float)
    vx, vy, _ = np.asarray(v, dtype=float)
    cw = np.array([3.0 * n**2 * x + 2.0 * n * vy, -2.0 * n * vx, -(n**2) * z])
    return np.asarray(a_path, dtype=float) - cw


def straight_line(n: float, r0: ArrayLike, r1: ArrayLike, speed: float) -> Element:
    """Forced motion from `r0` to `r1` along the straight line between them at constant speed.

    A start impulse of `speed` toward `r1`, thrust that cancels the CW terms on the way, and a
    stop impulse at `r1`; the speed stays constant up to arrival, with no braking near the end
    point (Fehse eqs 3.44 to 3.53).

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    r0, r1 : array_like, shape (3,)
        Start and end points in the Hill frame (m), both at rest.
    speed : float
        Speed along the line (m/s).

    Returns
    -------
    Element
        Impulses at 0 and |r1 - r0| / speed (s); `accel` is `forced_motion_accel` with zero
        path acceleration at the current state.

    Raises
    ------
    ValueError
        If `n` or `speed` is not positive and finite, or `r0` and `r1` are not distinct
        points of three finite numbers.
    """
    _check_n(n)
    if not (math.isfinite(speed) and speed > 0.0):
        raise ValueError("speed must be positive and finite")
    r0, r1 = np.asarray(r0, dtype=float), np.asarray(r1, dtype=float)
    if r0.shape != (3,) or r1.shape != (3,) or not np.all(np.isfinite(np.r_[r0, r1])):
        raise ValueError("r0 and r1 must be three finite numbers")
    length = float(np.linalg.norm(r1 - r0))
    if length == 0.0:
        raise ValueError("r0 and r1 must differ")
    velocity = speed * (r1 - r0) / length
    duration = length / speed
    no_path_accel = np.zeros(3)

    def accel(t: float, state: np.ndarray) -> np.ndarray:
        return forced_motion_accel(n, state[:3], state[3:6], no_path_accel)

    return Element(
        impulses=(Impulse(0.0, velocity), Impulse(duration, -velocity)),
        duration=duration,
        accel=accel,
    )


def straight_line_cost(n: float, distance: float, duration: float) -> float:
    """Delta-v of a constant-speed straight-line approach along the V-bar.

    Start and stop impulses of the speed each and the along-track integral of the -2 n vy
    thrust: 2 |dy| / duration + 2 n |dy| (Fehse eqs 3.44 to 3.48).

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    distance : float
        Length of the approach along the V-bar (m).
    duration : float
        Time of the approach (s).

    Returns
    -------
    float
        Total delta-v (m/s).

    Raises
    ------
    ValueError
        If `n` or `duration` is not positive and finite.
    """
    _check_n(n)
    if not (math.isfinite(duration) and duration > 0.0):
        raise ValueError("duration must be positive and finite")
    return 2.0 * abs(distance) / duration + 2.0 * n * abs(distance)


def twice_rate_circle(n: float, radius: float) -> Element:
    """One lap of the twice-orbital-rate circle about the target, from rest on the +R-bar.

    The chaser starts at rest at Hill (-radius, 0, 0), below the target, and a burn of
    2 n radius along +y puts it on the circle flown clockwise in the Hill x-y plane (below,
    ahead, above, behind) at rate 2 n, where v = (2 n y, -2 n x, 0). There the forced
    acceleration of Fehse eq 3.79 is the R-bar hover force (-3 n^2 x, 0, 0) alone (Goodman,
    p. 137), which `accel` applies at the current state. The relation vy + 2 n x = 0 then holds
    along the arc, so free motion after a thrust loss is a closed ellipse with no drift. The
    lap takes pi / n and ends with a stop burn of -2 n radius along y.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    radius : float
        Radius of the circle (m).

    Returns
    -------
    Element
        Impulses at 0 and pi / n (s) and the hover force as `accel`.

    Raises
    ------
    ValueError
        If `n` is not positive and finite.
    """
    _check_n(n)
    burn = _along(2.0 * n * radius, 1)
    hover_gain = -3.0 * n**2

    def accel(t: float, state: np.ndarray) -> np.ndarray:
        return _along(hover_gain * state[0], 0)

    lap = math.pi / n
    return Element(impulses=(Impulse(0.0, burn), Impulse(lap, -burn)), duration=lap, accel=accel)


def twice_rate_cost(n: float, radius: float, angle: float) -> float:
    """Delta-v of an arc of the twice-rate circle from rest on the +R-bar to rest.

    Start and stop impulses of 2 n R each plus the hover force 3 n^2 |x| integrated over the
    arc at rate 2 n: 4 n R + (3 n R / 2) times the integral of |cos| over the arc angle
    (Goodman, p. 137).

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    radius : float
        Radius of the circle (m).
    angle : float
        Arc angle flown from the +R-bar (rad).

    Returns
    -------
    float
        Total delta-v (m/s).

    Raises
    ------
    ValueError
        If `n` is not positive and finite, or `angle` is negative or not finite.
    """
    _check_n(n)
    if not (math.isfinite(angle) and angle >= 0.0):
        raise ValueError("angle must be non-negative and finite")
    # The integral of |cos| from 0 to angle in closed form: each half-period centred on a
    # multiple of pi contributes 2.
    half_periods = math.floor(angle / math.pi + 0.5)
    integral = 2.0 * half_periods + (-1.0) ** half_periods * math.sin(angle)
    return abs(radius) * n * (4.0 + 1.5 * integral)


def finite_burn_duration(dv: ArrayLike, gamma: float) -> float:
    """Duration of a burn that delivers the impulse `dv` at constant acceleration `gamma`.

    tau = |dv| / gamma; replacing an impulse by such a burn holds while tau is much shorter
    than the time between burns (Fehse eqs 3.61, 3.68).

    Parameters
    ----------
    dv : array_like, shape (3,)
        Impulse in the Hill frame (m/s).
    gamma : float
        Thrust acceleration (m/s^2).

    Returns
    -------
    float
        Burn duration (s).

    Raises
    ------
    ValueError
        If `gamma` is not positive and finite, or `dv` is not finite.
    """
    if not (math.isfinite(gamma) and gamma > 0.0):
        raise ValueError("gamma must be positive and finite")
    size = float(np.linalg.norm(np.asarray(dv, dtype=float)))
    if not math.isfinite(size):
        raise ValueError("dv must be finite")
    return size / gamma


def _hover(n: float, state: np.ndarray) -> np.ndarray:
    """Acceleration (Hill, m/s^2) that holds a chaser at rest where it is (Fehse eqs 3.55, 3.58)."""
    return np.array([-3.0 * n**2 * state[0], 0.0, n**2 * state[2]])


def _check_planned_impulses(view: AgentView) -> None:
    if view.action_mode == "disc27":
        raise ValueError("disc27 actions cannot deliver a planned impulse")
    if view.action_mode != "impulse_coast":
        action_scale(view)


class _Flight:
    """An element flown from decision step `start`, each impulse at its nearest step.

    With `measured`, the element is flown from rest to rest on the observation: the first
    impulse sets the velocity it plans relative to the observed one, and the last cancels the
    observed velocity instead of the planned arrival velocity.
    """

    def __init__(self, element: Element, start: int, dt: float, *, measured: bool):
        due: dict[int, np.ndarray] = {}
        for impulse in element.impulses:
            k = start + round(impulse.t / dt)
            due[k] = due.get(k, np.zeros(3)) + np.asarray(impulse.dv, dtype=float)
        self.due = sorted(due.items(), key=lambda item: item[0])
        self.accel = element.accel
        self.start = start
        self.end = start + round(element.duration / dt)
        self.dt = dt
        self.measured = measured
        self.fired = 0
        self.pending = np.zeros(3)

    @property
    def finished(self) -> bool:
        return self.fired == len(self.due) and not np.any(self.pending)

    def command(self, k: int, state: np.ndarray) -> tuple[np.ndarray, str, int | None]:
        """Impulse (Hill, m/s) for step `k`, its phase, and the next impulse's step if a free
        coast leads to it."""
        velocity = state[3:6]
        stopping = self.measured and len(self.due) > 1 and self.fired == len(self.due)
        # A saturated stop goes on against the observed velocity, not the planned remainder.
        dv = -velocity if stopping and np.any(self.pending) else self.pending.copy()
        burn = bool(np.any(self.pending))
        while self.fired < len(self.due) and self.due[self.fired][0] <= k:
            planned = self.due[self.fired][1]
            if self.measured and self.fired == 0:
                dv = dv + planned - velocity
            elif self.measured and self.fired == len(self.due) - 1:
                dv = -velocity
            else:
                dv = dv + planned
            self.fired += 1
            burn = True
        forced = self.accel is not None and self.start <= k < self.end
        if self.accel is not None and forced:
            dv = dv + self.accel((k - self.start) * self.dt, state) * self.dt
        if burn:
            phase = "burn"
        elif forced:
            phase = "forced"
        else:
            phase = "coast"
        next_step = None
        if burn and self.accel is None and self.fired < len(self.due):
            next_step = self.due[self.fired][0]
        return dv, phase, next_step

    def settle(self, dv: np.ndarray, delivered: np.ndarray, saturated: bool) -> None:
        """Carry what a saturated step could not deliver to the next step."""
        self.pending = dv - delivered if saturated else np.zeros(3)


class _FlightAgent:
    """Execution shared by the element agents: burns, coasts, coast prediction and the hover."""

    view: AgentView
    log: list[dict[str, Any]]

    def _start(self, view: AgentView) -> None:
        self.view = view
        self.phi, self.gamma = cw_discrete(view.n, view.dt)
        self.log = []
        self.t0: float | None = None

    def _step(self, t: float) -> int:
        assert self.t0 is not None
        return round((t - self.t0) / self.view.dt)

    def _fly(
        self, flight: _Flight, k: int, t: float, state: np.ndarray, entry: dict[str, Any]
    ) -> np.ndarray:
        view = self.view
        dv, phase, next_step = flight.command(k, state)
        coast_s = 0.0
        if next_step is not None:
            # The coast the environment flies: whole steps, cut at the end of the episode.
            steps_left = max(view.max_steps - round(t / view.dt) - 1, 0)
            coast_s = min(next_step - k - 1, steps_left) * view.dt
        action, saturated = impulse_action(
            view, dv, coast_s if view.action_mode == "impulse_coast" else 0.0
        )
        delivered = command_fraction(view, action) * view.max_accel * view.dt
        flight.settle(dv, delivered, saturated)
        entry.update(phase=phase, saturated=saturated)
        if coast_s > 0.0 and not saturated:
            after = self.phi @ state + self.gamma @ (delivered / view.dt)
            margin, bound = predicted_margin(view, after, coast_s)
            entry.update(coast_s=coast_s, predicted_margin=margin, predicted_margin_bound=bound)
        return action

    def _hold(self, state: np.ndarray, entry: dict[str, Any]) -> np.ndarray:
        view = self.view
        action, saturated = impulse_action(view, _hover(view.n, state) * view.dt, 0.0)
        entry.update(phase="hold", saturated=saturated)
        return action


_ELEMENT_MODES = ("open_loop",)


class ElementAgent(_FlightAgent):
    """Flies one trajectory element from the first observation, then hovers where it ends.

    At the first decision `element_factory(view, state)` returns the `Element` to fly, with
    `state` the observed Hill state [x, y, z, vx, vy, vz] (m, m/s). Each impulse fires at the
    step nearest its time and goes through `planning.impulse_action`, so an impulse larger than
    a step can deliver is spread over the following steps. Between the first and the last
    impulse the element's `accel`, evaluated at the observed state, is added to every step's
    command; an element without one coasts, and in "impulse_coast" the coast to the next
    impulse is committed in the action. After the last impulse the agent holds its observed
    position with the hover acceleration (-3 n^2 x, 0, n^2 z).

    `log` holds one dict per decision with the time "t" (s), the "phase" ("burn", "forced",
    "coast" or "hold") and whether the command "saturated". A burn followed by a free coast
    adds "coast_s" (s), the keep-out margin predicted over that coast, "predicted_margin", and
    its certified lower bound, "predicted_margin_bound" (m, from `planning.predicted_margin`).
    A predicted violation is logged and the element flown as is.

    Parameters
    ----------
    element_factory : callable
        `element_factory(view, state)` returning the `Element` to fly, its impulse times
        counted from the first decision.
    mode : str
        "open_loop": every impulse is flown as planned.
    name : str
        Name used in results.

    Raises
    ------
    ValueError
        If `mode` is unknown.
    """

    def __init__(
        self,
        element_factory: Callable[[AgentView, np.ndarray], Element],
        *,
        mode: str = "open_loop",
        name: str = "element",
    ):
        if mode not in _ELEMENT_MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {_ELEMENT_MODES}")
        self.element_factory = element_factory
        self.mode = mode
        self.name = name
        self.log: list[dict[str, Any]] = []

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode with the scenario's public description.

        Raises
        ------
        ValueError
            If the action mode is "disc27" or unknown, or thrust actions come without
            `mass_kg`.
        """
        _check_planned_impulses(view)
        self._start(view)
        self.flight: _Flight | None = None

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """The action for this decision in the view's action mode; appends one `log` entry."""
        t = float(info["time_s"])
        state = np.asarray(obs[:6], dtype=float)
        if self.flight is None:
            self.t0 = t
            element = self.element_factory(self.view, state)
            self.flight = _Flight(element, 0, self.view.dt, measured=False)
        entry: dict[str, Any] = {"t": t}
        if self.flight.finished:
            action = self._hold(state, entry)
        else:
            action = self._fly(self.flight, self._step(t), t, state, entry)
        self.log.append(entry)
        return action


class TorvaAgent(_FlightAgent):
    """Twice-orbital-rate fly-around of the target, stopped at a given angle.

    At the first decision the chaser at Hill (x, y) is put on the circle about the target
    through it, flown clockwise in the Hill x-y plane (below, ahead, above, behind) at rate
    2 n, where the velocity is (2 n y, -2 n x, 0): the start burn sets that velocity against
    the observed one. On the arc the only thrust is the R-bar hover force (-3 n^2 x, 0, 0) at
    the observed x (Goodman, History of Space Shuttle Rendezvous, JSC-63400, p. 137). The arc
    keeps vy + 2 n x = 0, so after a thrust loss the chaser flies a closed ellipse with no
    drift that never comes closer than the range at the loss. The stop burn, at the step
    nearest stop_angle / (2 n), cancels the observed velocity; the agent then holds its
    position with the hover acceleration. Out-of-plane position is left free.

    The start burn depends on where the chaser is, not on the goal, so from rest on the V-bar
    it points radially and `sign_check` fails along track by design.

    `log` holds one dict per decision with "t" (s), "phase" ("burn", "forced" or "hold") and
    "saturated".

    Parameters
    ----------
    stop_angle : float
        Arc flown before the stop burn (rad), counted clockwise from the start position.
    name : str
        Name used in results.

    Raises
    ------
    ValueError
        If `stop_angle` is not positive and finite.
    """

    def __init__(self, *, stop_angle: float, name: str = "torva"):
        if not (math.isfinite(stop_angle) and stop_angle > 0.0):
            raise ValueError("stop_angle must be positive and finite")
        self.stop_angle = stop_angle
        self.name = name
        self.log: list[dict[str, Any]] = []

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode with the scenario's public description.

        Raises
        ------
        ValueError
            If the action mode takes no continuous command, thrust actions come without
            `mass_kg`, or the arc to `stop_angle` takes less than half a step.
        """
        action_scale(view)
        self.stop_step = round(self.stop_angle / (2.0 * view.n * view.dt))
        if self.stop_step < 1:
            raise ValueError("stop_angle is flown in less than half a step")
        self._start(view)
        self.flight: _Flight | None = None

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """The action for this decision in the view's action mode; appends one `log` entry."""
        t = float(info["time_s"])
        state = np.asarray(obs[:6], dtype=float)
        if self.flight is None:
            self.t0 = t
            self.flight = _Flight(self._arc(state), 0, self.view.dt, measured=True)
        entry: dict[str, Any] = {"t": t}
        if self.flight.finished:
            action = self._hold(state, entry)
        else:
            action = self._fly(self.flight, self._step(t), t, state, entry)
        self.log.append(entry)
        return action

    def _arc(self, state: np.ndarray) -> Element:
        """The arc from the observed position to the stop, as an element from rest to rest."""
        n = self.view.n
        x, y = float(state[0]), float(state[1])
        c, s = math.cos(self.stop_angle), math.sin(self.stop_angle)
        x_stop, y_stop = c * x + s * y, -s * x + c * y
        stop_t = self.stop_step * self.view.dt
        circle = twice_rate_circle(n, math.hypot(x, y))
        return Element(
            impulses=(
                Impulse(0.0, np.array([2.0 * n * y, -2.0 * n * x, 0.0])),
                Impulse(stop_t, np.array([-2.0 * n * y_stop, 2.0 * n * x_stop, 0.0])),
            ),
            duration=stop_t,
            accel=circle.accel,
        )


# The leg that follows each leg of the Fehse sequence once its waypoint gate is passed.
_NEXT_LEG = {"to_vbar": "hop", "hop": "final", "final": "hold"}


class FehseSequenceAgent(_FlightAgent):
    """Fehse's closing sequence: a radial hop along the V-bar, then a straight-line approach.

    The legs, each flown from rest to rest on the observation (Fehse, Automated Rendezvous and
    Docking of Spacecraft, sections 3.3 and 5.7):

    - "to_vbar", only when the first observation lies more than `gate` from the V-bar: a CW
      two-impulse transfer to (0, y, 0) in `n_tf` / n.
    - "hop": a radial transfer (`radial_transfer`) along the V-bar, in half an orbit, to the
      point `final_leg` from the hold point on the chaser's side.
    - "final": the forced straight line (`straight_line`) from there to the hold point at
      `approach_speed`, with no braking before capture.

    Each leg's first burn sets the velocity the element plans against the observed one, and its
    last burn cancels the observed velocity. A leg that ends more than `gate` from its waypoint
    is followed by "contingency", a two-impulse transfer back to the waypoint in `n_tf` / n,
    before the sequence goes on. After the final leg the agent hovers where it is ("hold").

    `log` holds one dict per decision with "t" (s), the leg as "mode", the "phase" ("burn",
    "forced", "coast" or "hold") and "saturated"; a burn followed by a free coast adds
    "coast_s", "predicted_margin" and "predicted_margin_bound" as `ElementAgent` does.

    Parameters
    ----------
    final_leg : float
        Length of the straight-line approach (m).
    approach_speed : float
        Speed along the final leg (m/s).
    gate : float
        Distance from a waypoint (m) within which a leg counts as arrived.
    n_tf : float
        Transfer angle n tf (rad) of the two-impulse legs.
    name : str
        Name used in results.

    Raises
    ------
    ValueError
        If `final_leg`, `approach_speed`, `gate` or `n_tf` is not positive and finite.
    """

    def __init__(
        self,
        *,
        final_leg: float = 50.0,
        approach_speed: float = 0.04,
        gate: float = 10.0,
        n_tf: float = 1.0,
        name: str = "fehse-sequence",
    ):
        positive = {
            "final_leg": final_leg,
            "approach_speed": approach_speed,
            "gate": gate,
            "n_tf": n_tf,
        }
        for label, value in positive.items():
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{label} must be positive and finite")
        self.final_leg = final_leg
        self.approach_speed = approach_speed
        self.gate = gate
        self.n_tf = n_tf
        self.name = name
        self.log: list[dict[str, Any]] = []

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode with the scenario's public description.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, the action mode takes
            no continuous command, thrust actions come without `mass_kg`, `n_tf` lies within
            1e-2 of the CW singular set, or the hop and the final leg take longer than the
            episode.
        """
        self.hold = hold_point(view, "Fehse sequence agent")
        action_scale(view)
        g, h = conditioning(view.n, self.n_tf / view.n)
        if abs(g) < 1e-2 or abs(h) < 1e-2:
            raise ValueError(f"n_tf = {self.n_tf} is near the singular set ({SINGULAR_SET})")
        duration = math.pi / view.n + self.final_leg / self.approach_speed
        episode = view.max_steps * view.dt
        if duration > episode:
            raise ValueError(
                f"the hop and final leg take {duration:.1f} s, "
                f"longer than the {episode:.1f} s episode"
            )
        self._start(view)
        self.leg = ""
        self.mode = ""
        self.waypoint = np.zeros(3)
        self.hop_end = np.zeros(3)
        self.flight: _Flight | None = None

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """The action for this decision in the view's action mode; appends one `log` entry."""
        t = float(info["time_s"])
        state = np.asarray(obs[:6], dtype=float)
        if self.t0 is None:
            self.t0 = t
            side = math.copysign(self.final_leg, state[1] - self.hold[1])
            self.hop_end = np.array([0.0, self.hold[1] + side, 0.0])
            first = "to_vbar" if math.hypot(state[0], state[2]) > self.gate else "hop"
            self._begin(first, 0, state)
        k = self._step(t)
        if self.flight is not None and self.flight.finished:
            if np.linalg.norm(state[:3] - self.waypoint) > self.gate:
                self._begin("contingency", k, state)
            else:
                self._begin(_NEXT_LEG[self.leg], k, state)
        entry: dict[str, Any] = {"t": t, "mode": self.mode}
        if self.flight is None:
            action = self._hold(state, entry)
        else:
            action = self._fly(self.flight, k, t, state, entry)
        self.log.append(entry)
        return action

    def _begin(self, mode: str, k: int, state: np.ndarray) -> None:
        """Plan the leg `mode` from the observed `state` at step `k`."""
        n = self.view.n
        r = state[:3]
        self.mode = mode
        if mode == "hold":
            self.flight = None
            return
        if mode == "hop":
            self.waypoint = self.hop_end
            element = radial_transfer(n, self.hop_end[1] - r[1])
        elif mode == "final":
            self.waypoint = self.hold
            element = straight_line(n, r, self.hold, self.approach_speed)
        else:
            if mode == "to_vbar":
                self.waypoint = np.array([0.0, r[1], 0.0])
            tf = self.n_tf / n
            v0, vf = cw_lambert(n, r, self.waypoint, tf)
            element = Element(impulses=(Impulse(0.0, v0), Impulse(tf, -vf)), duration=tf)
        if mode != "contingency":
            self.leg = mode
        self.flight = _Flight(element, k, self.view.dt, measured=True)


class NominalTrackingAgent:
    """Tracks the nominal trajectory of an open-loop plan with the CW tracking law.

    At the first decision the planner plans from the observation; its first command is not
    used. The agent then flies `vbar.agents.lyapunov.CWTrackingAgent` toward the plan's nominal
    state at each step, feeding forward the plan's thrust and the CW acceleration of the
    nominal, so the command is the planned one wherever the chaser is on the nominal (Fehse,
    Automated Rendezvous and Docking of Spacecraft, section 4.4.1). After the plan the target is
    the hold point at rest. Continuous action modes only.

    `log` holds one dict per decision with "t" (s) and "phase" ("track" or "hold").

    Parameters
    ----------
    planner : agent
        A planner that, after its first `act`, exposes `schedule`, the Hill impulse (m/s)
        delivered over each step, and `nominal`, the planned state at the start of each step
        and after the last, as `TwoImpulseAgent` does in "open_loop".
    k : float
        Damping factor of the tracking law; the velocity gain is k n and the position gain n^2,
        critically damped at 2.
    name : str
        Name used in results.

    Raises
    ------
    ValueError
        If `k` is not positive and finite.
    """

    def __init__(self, planner: Any, *, k: float = 2.0, name: str = "nominal-tracking"):
        self.planner = planner
        self.law = CWTrackingAgent(k=k, reference=self._reference, name=name)
        self.name = name
        self.log: list[dict[str, Any]] = []

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode with the scenario's public description.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, the action mode takes
            no continuous command, the thrust layout is unknown, or the planner refuses the view.
        """
        self.hold = hold_point(view, "nominal tracking agent")
        self.law.reset(view, rng)
        self.planner.reset(view, rng)
        self.view = view
        self.log = []
        self.t0: float | None = None

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """The action for this decision in the view's action mode; appends one `log` entry.

        Raises
        ------
        ValueError
            If the planner exposes no `schedule` and `nominal` after its first decision.
        """
        t = float(info["time_s"])
        if self.t0 is None:
            self.t0 = t
            self.planner.act(obs, info)
            if not (hasattr(self.planner, "schedule") and hasattr(self.planner, "nominal")):
                raise ValueError("the planner exposes no schedule and nominal to track")
        k = round((t - self.t0) / self.view.dt)
        self.log.append({"t": t, "phase": "track" if k < len(self.planner.schedule) else "hold"})
        return self.law.act(obs, info)

    def _reference(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nominal position (m), velocity (m/s) and acceleration (m/s^2) at `t`, Hill frame."""
        assert self.t0 is not None
        view = self.view
        k = round((t - self.t0) / view.dt)
        schedule = self.planner.schedule
        if k >= len(schedule):
            return self.hold, np.zeros(3), np.zeros(3)
        nominal = np.asarray(self.planner.nominal[k], dtype=float)
        r, v = nominal[:3], nominal[3:6]
        # The plan holds each step's impulse as a constant thrust on top of the CW motion.
        natural = -forced_motion_accel(view.n, r, v, np.zeros(3))
        return r, v, natural + np.asarray(schedule[k], dtype=float) / view.dt
