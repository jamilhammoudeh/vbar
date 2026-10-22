"""Glideslope guidance: a pulsed approach along the line of sight with an exponential range profile.

The range to go rho along the fixed line from the aim point to the start follows
rho_dot = a rho + b with a = (rho_dot0 - rho_dotT) / rho0 < 0 and b = rho_dotT < 0, so
rho(t) = (rho0 + b / a) exp(a t) - b / a and the arrival time is T = ln(rho_dotT / rho_dot0) / a.
Pulses place the chaser on the line at the profile's range, each segment's velocity coming from
the CW transition matrices (Hablani, Tapper and Dana-Bashian, Guidance and control of an
autonomous rendezvous spacecraft, Journal of Guidance, Control, and Dynamics 25(3), 2002, with
the equations as reproduced in Yuan, Li and Wang, Aerospace 9, 242, 2022, eqs 14 to 25).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..dynamics import cw_discrete
from .base import CONTINUOUS_MODES, AgentView, action_scale
from .cw_targeting import conditioning, cw_lambert
from .planning import Impulse, impulse_action, predicted_margin

EXECUTION_MODES = ("open_loop", "retarget")


@dataclass(frozen=True)
class GlideslopeProfile:
    """Exponential range profile rho_dot = a rho + b.

    `a` is the rate (1/s, negative), `b` the closing rate at arrival (m/s, negative) and
    `duration` the time to reach zero range (s).
    """

    a: float
    b: float
    duration: float


def glideslope_profile(rho0: float, rho_dot0: float, rho_dotT: float) -> GlideslopeProfile:
    """The exponential profile from range `rho0` at rate `rho_dot0` to zero range at `rho_dotT`.

    Parameters
    ----------
    rho0 : float
        Initial range to the aim point (m).
    rho_dot0 : float
        Initial range rate (m/s), negative when closing.
    rho_dotT : float
        Range rate at arrival (m/s), negative when closing.

    Returns
    -------
    GlideslopeProfile
        a = (rho_dot0 - rho_dotT) / rho0, b = rho_dotT and duration ln(rho_dotT / rho_dot0) / a.

    Raises
    ------
    ValueError
        If an input is not finite, `rho0` is not positive, `rho_dotT` is not negative, or
        `rho_dot0` is not below `rho_dotT` (the profile must close faster at the start, a < 0).
    """
    if not all(math.isfinite(value) for value in (rho0, rho_dot0, rho_dotT)):
        raise ValueError("profile inputs must be finite")
    if not rho0 > 0.0:
        raise ValueError("rho0 must be positive")
    if not rho_dotT < 0.0:
        raise ValueError("rho_dotT must be negative: a zero or opening rate never arrives")
    if not rho_dot0 < rho_dotT:
        raise ValueError("rho_dot0 must be below rho_dotT, so the range rate decays (a < 0)")
    a = (rho_dot0 - rho_dotT) / rho0
    return GlideslopeProfile(a, rho_dotT, math.log(rho_dotT / rho_dot0) / a)


def _range(profile: GlideslopeProfile, rho0: float, t: float) -> float:
    """Range to go (m) at `t` (s) on `profile` from `rho0`."""
    shift = profile.b / profile.a
    return (rho0 + shift) * math.exp(profile.a * t) - shift


def glideslope_pulses(
    n: float,
    state: ArrayLike,
    hold: ArrayLike,
    profile: GlideslopeProfile,
    pulse_interval: float,
) -> tuple[Impulse, ...]:
    """Impulses that fly `profile` down the line of sight from `state` to rest at `hold`.

    With N = ceil(T / pulse_interval), pulses fall at k `pulse_interval` for k < N and at the
    arrival time T. Each pulse except the last sets the velocity that coasts to the next
    waypoint hold + rho(t_k+1) u on the line u from `hold` to the start, found by inverting the
    CW transition matrices; the next waypoint after the last coast is `hold` itself, and the last
    pulse cancels the arrival velocity.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    state : array_like, shape (6,)
        Chaser state [x, y, z, vx, vy, vz] in the Hill frame (m, m/s) at t = 0.
    hold : array_like, shape (3,)
        Aim point in the Hill frame (m).
    profile : GlideslopeProfile
        Range profile, normally `glideslope_profile` of the range from `hold` to `state`.
    pulse_interval : float
        Longest time between pulses (s).

    Returns
    -------
    tuple of Impulse
        N + 1 impulses (Hill frame, m/s), times in s from the start.

    Raises
    ------
    ValueError
        If `state` or `hold` has the wrong shape or is not finite, `state` is at `hold`,
        `pulse_interval` is not positive and finite, or a segment is too close to the singular
        set of CW targeting (see `cw_lambert`).
    """
    state = np.asarray(state, dtype=float)
    hold = np.asarray(hold, dtype=float)
    if state.shape != (6,) or hold.shape != (3,):
        raise ValueError("state must have 6 entries and hold 3")
    if not (np.all(np.isfinite(state)) and np.all(np.isfinite(hold))):
        raise ValueError("state and hold must be finite")
    if not (math.isfinite(pulse_interval) and pulse_interval > 0.0):
        raise ValueError("pulse_interval must be positive and finite")
    offset = state[:3] - hold
    rho0 = float(np.linalg.norm(offset))
    if rho0 == 0.0:
        raise ValueError("state is at the aim point: no line of sight")
    u = offset / rho0
    count = math.ceil(profile.duration / pulse_interval)
    times = [k * pulse_interval for k in range(count)] + [profile.duration]
    waypoints = [state[:3]]
    waypoints += [hold + _range(profile, rho0, t) * u for t in times[1:-1]]
    waypoints.append(hold)
    impulses = []
    v_before = state[3:]
    for k in range(count):
        v_after, v_before_next = cw_lambert(
            n, waypoints[k], waypoints[k + 1], times[k + 1] - times[k]
        )
        impulses.append(Impulse(times[k], v_after - v_before))
        v_before = v_before_next
    impulses.append(Impulse(times[-1], -v_before))
    return tuple(impulses)


def _limited(view: AgentView, accel: np.ndarray) -> np.ndarray:
    """`accel` (m/s^2) cut to the thruster limit, per axis (box) or in 2-norm (ball)."""
    if view.thrust_limit == "box":
        return np.clip(accel, -view.max_accel, view.max_accel)
    # A part in 1e12 below the limit keeps the rounded 2-norm inside it.
    return accel * min(1.0, (1.0 - 1e-12) * view.max_accel / float(np.linalg.norm(accel)))


class GlideslopeAgent:
    """Flies the glideslope profile from the first observation to rest at the hold point.

    At its first decision the agent fixes the line of sight from the hold point to the chaser
    and the profile a = -`rate`, b = -`arrival_speed`, rho_dot0 = -(`rate` rho0 +
    `arrival_speed`). Pulses fall every `pulse_interval` and at the arrival time rounded to a
    whole step. Each pulse is a burn at constant acceleration over the fewest whole steps the
    thrusters can deliver, solved on the exact CW transition matrices of the burn as the
    environment flies it, so that the chaser reaches the next waypoint on the line at the next
    pulse; the last pulse brings the velocity to zero, and the agent then commands no thrust.
    A chaser that starts within the capture radius of the hold point, or so close that the
    arrival rounds to no whole step, is already there: the agent commands no thrust throughout.
    A burn that cannot be delivered within its segment is flown at the thruster limit and logged
    as saturated.

    In "open_loop" mode every burn is solved from the state the plan predicts; in "retarget"
    mode each pulse is solved from the observed state, to the same waypoints. In
    "impulse_coast" the coast to the next pulse is committed with the last burn step; in the
    continuous modes each step is a decision and the agent commands zero between burns.

    `log` holds one entry per decision: `time_s`, `pulse` (the index of the pulse a burn starts,
    else None), `dv` (the step's velocity change, Hill frame, m/s), `saturated`, `coast_s`, and
    over a committed coast `predicted_margin` and `predicted_margin_bound` (m, from
    `planning.predicted_margin` on the observed state carried through the burn step; None
    without a coast). A coast predicted to enter the keep-out zone is logged and flown.
    """

    def __init__(
        self,
        *,
        rate: float = 1e-3,
        arrival_speed: float = 0.02,
        pulse_interval: float = 60.0,
        mode: str = "retarget",
        name: str = "glideslope",
    ):
        """Store the profile and pulse parameters.

        Parameters
        ----------
        rate : float
            Decay rate -a of the range rate (1/s); 1e-3 is a 1000 s time constant.
        arrival_speed : float
            Closing speed -b at arrival (m/s).
        pulse_interval : float
            Time between pulses (s), a whole number of decision steps.
        mode : str
            "open_loop" or "retarget".
        name : str
            Agent name.

        Raises
        ------
        ValueError
            If `rate`, `arrival_speed` or `pulse_interval` is not positive and finite, or
            `mode` is unknown.
        """
        for label, value in (
            ("rate", rate),
            ("arrival_speed", arrival_speed),
            ("pulse_interval", pulse_interval),
        ):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{label} must be positive and finite")
        if mode not in EXECUTION_MODES:
            raise ValueError(f"unknown mode {mode!r}, expected {EXECUTION_MODES}")
        self.rate = rate
        self.arrival_speed = arrival_speed
        self.pulse_interval = pulse_interval
        self.mode = mode
        self.name = name

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Check the view and clear the plan.

        Raises
        ------
        ValueError
            If the action mode is "disc27" or unknown, `pulse_interval` is not a whole number of
            steps, or n `pulse_interval` is not below pi with CW targeting conditioned to 1e-2.
        """
        if view.action_mode != "impulse_coast":
            if view.action_mode not in CONTINUOUS_MODES:
                raise ValueError(f"action mode {view.action_mode!r} cannot fly planned burns")
            action_scale(view)
        steps = round(self.pulse_interval / view.dt)
        if steps < 1 or abs(steps * view.dt - self.pulse_interval) > 1e-9 * self.pulse_interval:
            raise ValueError("pulse_interval must be a whole number of steps")
        # No segment is longer than the pulse interval, so below half an orbit none reaches a
        # singular CW transfer time; the tolerance matches cw_lambert's.
        g, h = conditioning(view.n, self.pulse_interval)
        if not (view.n * self.pulse_interval < math.pi and min(g, h) >= 1e-2):
            raise ValueError("pulse_interval is too close to half an orbit")
        self.view = view
        self.pulse_steps = steps
        self.hold = np.array(view.goal["hold_point"], dtype=float)
        self.one_step = cw_discrete(view.n, view.dt)
        self.log: list[dict[str, Any]] = []
        self.schedule: list[int] = []
        self.waypoints: list[np.ndarray] = []
        self.pulse = 0
        self.predicted = np.zeros(6)
        self.burn: list[tuple[np.ndarray, float]] = []
        self.planned = False

    def act(self, obs, info):
        state = np.asarray(obs[:6], dtype=float)
        step = round(info["time_s"] / self.view.dt)
        if not self.planned:
            self._plan(state, step)
        started = None
        saturated = False
        if not self.burn and self.pulse < len(self.schedule) and step >= self.schedule[self.pulse]:
            # A decision that comes after a later pulse time aims at the next waypoint ahead.
            while self.pulse + 1 < len(self.schedule) and self.schedule[self.pulse + 1] <= step:
                self.pulse += 1
            started = self.pulse
            saturated = self._start_burn(state if self.mode == "retarget" else self.predicted, step)
            self.pulse += 1
        if self.burn:
            accel, coast_s = self.burn.pop(0)
        else:
            accel, coast_s = np.zeros(3), self._coast_to_next(step)
        if self.view.action_mode != "impulse_coast":
            coast_s = 0.0
        action, _ = impulse_action(self.view, accel * self.view.dt, coast_s)
        margin = bound = None
        if coast_s > 0.0:
            phi, gamma = self.one_step
            margin, bound = predicted_margin(self.view, phi @ state + gamma @ accel, coast_s)
        self.log.append(
            {
                "time_s": info["time_s"],
                "pulse": started,
                "dv": accel * self.view.dt,
                "saturated": saturated,
                "coast_s": coast_s,
                "predicted_margin": margin,
                "predicted_margin_bound": bound,
            }
        )
        return action

    def _plan(self, state: np.ndarray, step: int) -> None:
        """Fix the line of sight, the profile, the pulse steps and the waypoints."""
        self.predicted = state.copy()
        self.planned = True
        offset = state[:3] - self.hold
        rho0 = float(np.linalg.norm(offset))
        # A chaser already at the hold point has nothing to fly; the empty schedule commands zero.
        if rho0 == 0.0 or rho0 < self.view.goal.get("capture_radius", 0.0):
            return
        profile = glideslope_profile(
            rho0, -(self.rate * rho0 + self.arrival_speed), -self.arrival_speed
        )
        u = offset / rho0
        dt = self.view.dt
        arrival = round(profile.duration / dt)
        if arrival == 0:
            return
        # Both counts are whole steps, so every pulse before the arrival falls strictly before it.
        count = math.ceil(arrival / self.pulse_steps)
        offsets = [k * self.pulse_steps for k in range(count)] + [arrival]
        self.schedule = [step + k for k in offsets]
        self.waypoints = [self.hold + _range(profile, rho0, k * dt) * u for k in offsets[1:-1]]
        self.waypoints.append(self.hold)

    def _start_burn(self, state: np.ndarray, step: int) -> bool:
        """Queue the burn of the current pulse from `state`; True if it had to be limited."""
        n, dt = self.view.n, self.view.dt
        last = self.pulse == len(self.schedule) - 1
        span = self.pulse_steps if last else self.schedule[self.pulse + 1] - step
        saturated = True
        for steps in range(1, span + 1):
            accel = self._solve(state, steps, None if last else span)
            if not impulse_action(self.view, accel * dt, 0.0)[1]:
                saturated = False
                break
        if saturated:
            accel = _limited(self.view, accel)
        coast_steps = 0 if last else span - steps
        self.burn = [(accel, 0.0)] * (steps - 1) + [(accel, coast_steps * dt)]
        phi, gamma = cw_discrete(n, steps * dt)
        self.predicted = cw_discrete(n, coast_steps * dt)[0] @ (phi @ state + gamma @ accel)
        return saturated

    def _solve(self, state: np.ndarray, steps: int, span: int | None) -> np.ndarray:
        """Constant acceleration (m/s^2) held `steps` steps from `state`.

        It reaches the next waypoint `span` steps later, or with `span` None brings the
        velocity to zero at the end of the burn.
        """
        n, dt = self.view.n, self.view.dt
        phi, gamma = cw_discrete(n, steps * dt)
        if span is None:
            return np.linalg.solve(gamma[3:], -phi[3:] @ state)
        after = cw_discrete(n, (span - steps) * dt)[0][:3]
        miss = self.waypoints[self.pulse] - after @ phi @ state
        return np.linalg.solve(after @ gamma, miss)

    def _coast_to_next(self, step: int) -> float:
        """Coast (s) after a zero-thrust step that ends at the next pulse, 0 after arrival."""
        if self.pulse < len(self.schedule):
            return max(self.schedule[self.pulse] - step - 1, 0) * self.view.dt
        return 0.0
