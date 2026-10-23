"""Space Shuttle rendezvous practice: CW aim points, elevation-timed burns and the pilot model.

Goodman, History of Space Shuttle Rendezvous, JSC-63400 Rev 3 (2011). The book works in an LVLH
frame with +X along the V-bar and +Z toward the Earth, in ft and n.mi.; everything here is in the
Hill frame (x radial out, y along track) in metres, so a book point (X, Z) is (-Z, X).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from ..dynamics import cw_discrete
from .base import AgentView, action_scale, command_fraction, hold_point, zero_action
from .cw_targeting import TwoImpulseAgent
from .planning import impulse_action, predicted_margin

AIMS = ("hold",)


def _position(r: ArrayLike, name: str) -> np.ndarray:
    out = np.asarray(r, dtype=float)
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be three finite numbers")
    return out


def _mean_motion(n: float) -> float:
    if not (math.isfinite(n) and n > 0.0):
        raise ValueError("n must be a positive finite number")
    return n


def nsr_aim(r0: ArrayLike, n: float) -> tuple[np.ndarray, float]:
    """Aim point and transfer time that make CW targeting fly a coelliptic (NSR) burn.

    Goodman's rule aims at (X0 + 2 Z0, Z0) in LVLH after 4 / (3 n); in Hill coordinates the aim
    point is (x0, y0 - 2 x0, 0), the start height two heights further along track, and the
    targeted velocity is the coelliptic drift (0, -1.5 n x0, 0).

    Parameters
    ----------
    r0 : array_like, shape (3,)
        Chaser position in the Hill frame (m). A cross-track offset is aimed back to the orbit
        plane.
    n : float
        Mean motion of the target orbit (rad/s).

    Returns
    -------
    rf : numpy.ndarray, shape (3,)
        Aim point in the Hill frame (m).
    tf : float
        Transfer time 4 / (3 n) (s).

    Raises
    ------
    ValueError
        If `r0` is not three finite numbers or `n` is not a positive finite number.
    """
    r0 = _position(r0, "r0")
    n = _mean_motion(n)
    return np.array([r0[0], r0[1] - 2.0 * r0[0], 0.0]), 4.0 / (3.0 * n)


def football_aim(r0: ArrayLike, n: float) -> tuple[np.ndarray, float]:
    """Aim point and transfer time that make CW targeting fly a no-drift ellipse (football).

    Goodman's rule aims at (X0 + 2 Z0, 0) in LVLH after a quarter period; in Hill coordinates
    the aim point is (0, y0 - 2 x0, 0) on the V-bar, and the targeted velocity (0, -2 n x0, 0)
    stops the along-track drift, so the chaser returns to `r0` after one period.

    Parameters
    ----------
    r0 : array_like, shape (3,)
        Chaser position in the Hill frame (m). A cross-track offset is aimed back to the orbit
        plane.
    n : float
        Mean motion of the target orbit (rad/s).

    Returns
    -------
    rf : numpy.ndarray, shape (3,)
        Aim point in the Hill frame (m).
    tf : float
        Transfer time pi / (2 n), a quarter of the orbital period (s).

    Raises
    ------
    ValueError
        If `r0` is not three finite numbers or `n` is not a positive finite number.
    """
    r0 = _position(r0, "r0")
    n = _mean_motion(n)
    return np.array([0.0, r0[1] - 2.0 * r0[0], 0.0]), 0.5 * math.pi / n


def elevation(r: ArrayLike) -> float:
    """Elevation of the target above the chaser's local horizontal, asin(-x / |r|).

    Parameters
    ----------
    r : array_like, shape (3,)
        Chaser position in the Hill frame (m).

    Returns
    -------
    float
        Elevation (rad) in [-pi/2, pi/2]: positive when the chaser is below the target, zero on
        the V-bar.

    Raises
    ------
    ValueError
        If `r` is not three finite numbers or is zero.
    """
    r = _position(r, "r")
    rho = float(np.linalg.norm(r))
    if rho == 0.0:
        raise ValueError("r must be nonzero")
    return math.asin(-r[0] / rho)


class ElevationTimedAgent:
    """CW targeting to the hold point, fired when the target's elevation reaches `trigger`.

    The Shuttle timed its terminal-phase burn on the line-of-sight elevation of the target, not on
    the clock (Goodman pp. 150 and 154, a 42 deg trigger for the STORRM re-rendezvous). At its
    first decision the agent propagates the coast from the observation in closed-form CW, finds
    the first time the `elevation` reaches `trigger` (sampled at a quarter step and refined with
    Brent's method) and schedules the burn at that time rounded to a step. Each later decision
    re-predicts the crossing: a crossing within `slip_s` of the one that set the schedule moves
    the burn, and one further away is ignored and logged. A first prediction with no crossing
    before the episode's last decision fires the burn at once and is logged, so the agent
    degrades to two-impulse targeting. A chaser already at or above the trigger burns at once.

    The burn and everything after it are flown by a re-targeting `TwoImpulseAgent` started at the
    burn: the first impulse puts the chaser on the CW transfer of `n_tf` / n to the hold point,
    later decisions correct it, and the arrival burn ends at rest there.

    `log` holds one dict per decision with the time "t" (s) and the "phase", "wait" before the
    burn or the transfer agent's phase from then on (with its "saturated" flag). Before the burn
    each entry carries the "predicted_crossing_s" (s since the episode start, None when there is
    none) and the scheduled "burn_t" (s); "trigger_not_reached" or "slip_exceeded" is set to True
    on the decision where that happens. The decision that commits a coast adds "coast_s" (s),
    the keep-out margin predicted over it, "predicted_margin", and its certified lower bound,
    "predicted_margin_bound" (m, from `planning.predicted_margin`): a wait that sets or moves
    the burn, over the coast from the observed state to the burn, and the first transfer
    decision whose impulse fits in one step, over the coast from the state after that step to
    the arrival burn, cut at the end of the episode. A predicted violation is logged and the
    agent flies on.

    Parameters
    ----------
    trigger : float
        Elevation at which to burn (rad), in [-pi/2, pi/2].
    aim : str
        Where the transfer ends; "hold" is the scenario's hold point.
    n_tf : float
        Transfer angle n tf (rad).
    slip_s : float
        Largest move of a re-predicted crossing from the one that set the schedule (s).
    name : str
        Name used in results.

    Raises
    ------
    ValueError
        If `trigger` is not finite or lies outside [-pi/2, pi/2], `aim` is unknown, `n_tf` is not
        positive and finite, or `slip_s` is negative or not finite.
    """

    def __init__(
        self,
        *,
        trigger: float,
        aim: str = "hold",
        n_tf: float = 1.0,
        slip_s: float = 60.0,
        name: str = "elevation-timed",
    ):
        if not (math.isfinite(trigger) and abs(trigger) <= 0.5 * math.pi):
            raise ValueError("trigger must be a finite angle in [-pi/2, pi/2]")
        if aim not in AIMS:
            raise ValueError(f"unknown aim {aim!r}, expected one of {AIMS}")
        if not (math.isfinite(n_tf) and n_tf > 0.0):
            raise ValueError("n_tf must be positive and finite")
        if not (math.isfinite(slip_s) and slip_s >= 0.0):
            raise ValueError("slip_s must be finite and non-negative")
        self.trigger = trigger
        self.aim = aim
        self.n_tf = n_tf
        self.slip_s = slip_s
        self.name = name
        self.log: list[dict[str, Any]] = []

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode with the scenario's public description.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, the action mode is
            "disc27" or unknown, thrust actions come without `mass_kg`, or `n_tf` lies within
            1e-2 of the singular set of CW targeting.
        """
        # The transfer reads the hold point too; checking here names this agent in the refusal.
        hold_point(view, "elevation-timed agent")
        self.view = view
        self.transfer = TwoImpulseAgent(n_tf=self.n_tf, name=self.name)
        self.transfer.reset(view, rng)
        self.phi, self.gamma = cw_discrete(view.n, view.dt)
        self.log = []
        self.crossing_t: float | None = None
        self.burn_step: int | None = None
        self.committed_step: int | None = None
        self.fired = False
        self.transfer_committed = False

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """The action for this decision in the view's action mode; appends one `log` entry."""
        view = self.view
        t = float(info["time_s"])
        state = np.asarray(obs[:6], dtype=float)
        k = round(t / view.dt)
        entry: dict[str, Any] = {"t": t}
        if not self.fired:
            burn_step = self._schedule(state, t, entry)
            if k < burn_step:
                entry["phase"] = "wait"
                if burn_step != self.committed_step:
                    self.committed_step = burn_step
                    self._predict(entry, state, (burn_step - k) * view.dt)
                self.log.append(entry)
                return zero_action(view)
            self.fired = True
        action = self.transfer.act(obs, info)
        entry.update({key: v for key, v in self.transfer.log[-1].items() if key != "t"})
        if entry["phase"] == "transfer" and not entry["saturated"] and not self.transfer_committed:
            # The first impulse delivered in one step commits the coast to the arrival burn, in
            # whole steps cut at the end of the episode; later decisions only correct it.
            self.transfer_committed = True
            steps = round((self.transfer.t_arrival - t) / view.dt) - 1
            steps = min(steps, view.max_steps - k - 1)
            if steps > 0:
                accel = command_fraction(view, action) * view.max_accel
                after = self.phi @ state + self.gamma @ accel
                self._predict(entry, after, steps * view.dt)
        self.log.append(entry)
        return action

    def _predict(self, entry: dict[str, Any], state: np.ndarray, coast_s: float) -> None:
        """Log the keep-out margin predicted over a coast of `coast_s` from `state`."""
        margin, bound = predicted_margin(self.view, state, coast_s)
        entry.update(coast_s=coast_s, predicted_margin=margin, predicted_margin_bound=bound)

    def _schedule(self, state: np.ndarray, t: float, entry: dict[str, Any]) -> int:
        """The burn step after re-predicting the crossing and applying the slip limit."""
        dt = self.view.dt
        crossing = self._crossing(state, t)
        entry["predicted_crossing_s"] = crossing
        step = self.burn_step
        if step is None:
            if crossing is None:
                entry["trigger_not_reached"] = True
                step = round(t / dt)
            else:
                self.crossing_t = crossing
                step = round(crossing / dt)
        elif crossing is not None and self.crossing_t is not None:
            if abs(crossing - self.crossing_t) > self.slip_s:
                entry["slip_exceeded"] = True
            else:
                self.crossing_t = crossing
                step = round(crossing / dt)
        self.burn_step = step
        entry["burn_t"] = step * dt
        return step

    def _crossing(self, state: np.ndarray, t: float) -> float | None:
        """Time (s) the coast from `state` at `t` first reaches the trigger; None past the end."""
        view = self.view
        # Crossings up to half a step past the last decision still round to it.
        last = view.max_steps - 1
        span = (last + 0.5) * view.dt - t
        if span <= 0.0:
            return None
        count = max(2, math.ceil(4.0 * span / view.dt) + 1)
        taus = np.linspace(0.0, span, count)
        phis, _ = cw_discrete(view.n, taus)
        r = (phis @ state)[:, :3]
        rho = np.linalg.norm(r, axis=1)
        # A sample at zero range has no elevation and counts as not reached.
        with np.errstate(divide="ignore", invalid="ignore"):
            above = np.arcsin(-r[:, 0] / rho) >= self.trigger
        reached = np.flatnonzero(above)
        if not len(reached):
            return None
        i = int(reached[0])
        if i == 0:
            return t

        def miss(tau: float) -> float:
            r_tau = (cw_discrete(view.n, tau)[0] @ state)[:3]
            return math.asin(-r_tau[0] / float(np.linalg.norm(r_tau))) - self.trigger

        crossing = t + brentq(miss, taus[i - 1], taus[i], xtol=1e-6)
        return crossing if round(crossing / view.dt) <= last else None


class PilotAgent:
    """Manual proximity operations: fixed pulses that hold the closing rate on a gate profile.

    Shuttle crews flew the final approach by hand with small translational pulses (0.1 ft/s,
    Goodman p. 143), nulling line-of-sight drift and keeping the range rate on a gate profile of
    range over a 1000 s time constant (pp. 57 and 74), with a quiet period after each pulse so the
    new range rate could be read. With d the vector from the chaser to the hold point, rho = |d|,
    u = d / rho and the closing rate v . u, each decision outside a read applies the first rule
    that holds:

    1. inside the capture radius, while |v| is at least half a pulse: a pulse against v;
    2. line-of-sight rate |d x v| / rho^2 above `los_rate_limit`: a pulse against the velocity
       across the line of sight;
    3. closing rate below (1 - `band`) times the gate max(rho / `tau`, `min_rate`): a pulse
       along u;
    4. closing rate above (1 + `band`) times the gate: a pulse against u;

    and otherwise coasts. A pulse of `pulse` m/s is one step of thrust, followed by `read_s`
    seconds without thrust, asked for as the coast of the action in "impulse_coast".

    `log` holds one dict per decision with the time "t" (s), the "rule" ("read", "capture",
    "line_of_sight", "closing", "opening" or "coast") and whether the command "saturated". A pulse
    adds the read coast "coast_s" (s), the keep-out margin predicted over it, "predicted_margin",
    and its certified lower bound, "predicted_margin_bound" (m, from `planning.predicted_margin`
    on the state after the pulse step, over the whole read even where the episode ends inside it).
    A predicted violation is logged and the pilot flies on.

    Parameters
    ----------
    tau : float
        Time constant of the gate profile (s).
    band : float
        Half-width of the gate band as a fraction of the gate rate, in (0, 1).
    pulse : float
        Pulse size (m/s); it must fit in one step of thrust.
    min_rate : float
        Lowest gate closing rate (m/s).
    los_rate_limit : float or None
        Line-of-sight rate above which drift is nulled (rad/s); None turns the rule off.
    read_s : float
        Coast after every pulse (s).
    name : str
        Name used in results.

    Raises
    ------
    ValueError
        If `tau` or `pulse` is not positive and finite, `band` is outside (0, 1), `min_rate` or
        `read_s` is negative or not finite, or `los_rate_limit` is neither None nor positive and
        finite.
    """

    def __init__(
        self,
        *,
        tau: float = 1000.0,
        band: float = 0.25,
        pulse: float = 0.03048,
        min_rate: float = 0.03048,
        los_rate_limit: float | None = 1e-4,
        read_s: float = 60.0,
        name: str = "pilot",
    ):
        for label, value in {"tau": tau, "pulse": pulse}.items():
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{label} must be positive and finite")
        if not (0.0 < band < 1.0):
            raise ValueError("band must lie in (0, 1)")
        for label, value in {"min_rate": min_rate, "read_s": read_s}.items():
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{label} must be finite and non-negative")
        if los_rate_limit is not None and not (
            math.isfinite(los_rate_limit) and los_rate_limit > 0.0
        ):
            raise ValueError("los_rate_limit must be None or positive and finite")
        self.tau = tau
        self.band = band
        self.pulse = pulse
        self.min_rate = min_rate
        self.los_rate_limit = los_rate_limit
        self.read_s = read_s
        self.name = name
        self.log: list[dict[str, Any]] = []

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode with the scenario's public description.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, the action mode is
            "disc27" or unknown, thrust actions come without `mass_kg`, or `pulse` is more than
            one step of thrust, max_accel dt.
        """
        self.hold = hold_point(view, "pilot agent")
        if view.action_mode == "disc27":
            raise ValueError("disc27 actions cannot deliver a pulse")
        if view.action_mode != "impulse_coast":
            action_scale(view)
        if self.pulse > view.max_accel * view.dt:
            raise ValueError("pulse exceeds one step of thrust")
        self.view = view
        self.capture_radius = float(view.goal.get("capture_radius", 0.0))
        self.phi, self.gamma = cw_discrete(view.n, view.dt)
        self.quiet_until = -math.inf
        self.log = []

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """The action for this decision in the view's action mode; appends one `log` entry."""
        view = self.view
        t = float(info["time_s"])
        state = np.asarray(obs[:6], dtype=float)
        entry: dict[str, Any] = {"t": t}
        # The tolerance absorbs rounding in times built from repeated steps.
        if t < self.quiet_until - 1e-9:
            entry["rule"] = "read"
            dv = np.zeros(3)
        else:
            dv = self._pulse(state, entry)
        pulsed = bool(np.any(dv != 0.0))
        coast_s = self.read_s if pulsed and view.action_mode == "impulse_coast" else 0.0
        action, saturated = impulse_action(view, dv, coast_s)
        entry["saturated"] = saturated
        if pulsed:
            self.quiet_until = t + view.dt + self.read_s
        # The environment rounds an impulse_coast coast to whole steps.
        read = round(self.read_s / view.dt) * view.dt
        if pulsed and read > 0.0:
            after = self.phi @ state + self.gamma @ (dv / view.dt)
            margin, bound = predicted_margin(view, after, read)
            entry.update(coast_s=read, predicted_margin=margin, predicted_margin_bound=bound)
        self.log.append(entry)
        return action

    def _pulse(self, state: np.ndarray, entry: dict[str, Any]) -> np.ndarray:
        """Pulse (Hill, m/s) the first rule that holds asks for; zero when none does."""
        r, v = state[:3], state[3:6]
        d = self.hold - r
        rho = float(np.linalg.norm(d))
        if rho < self.capture_radius or rho == 0.0:
            entry["rule"] = "capture"
            speed = float(np.linalg.norm(v))
            # Below half a pulse a further pulse would leave more velocity than it removes.
            if speed < 0.5 * self.pulse:
                return np.zeros(3)
            return -self.pulse * v / speed
        u = d / rho
        closing = float(v @ u)
        if self.los_rate_limit is not None:
            los_rate = float(np.linalg.norm(np.cross(d, v))) / rho**2
            if los_rate > self.los_rate_limit:
                entry["rule"] = "line_of_sight"
                across = v - closing * u
                return -self.pulse * across / np.linalg.norm(across)
        gate = max(rho / self.tau, self.min_rate)
        if closing < (1.0 - self.band) * gate:
            entry["rule"] = "closing"
            return self.pulse * u
        if closing > (1.0 + self.band) * gate:
            entry["rule"] = "opening"
            return -self.pulse * u
        entry["rule"] = "coast"
        return np.zeros(3)
