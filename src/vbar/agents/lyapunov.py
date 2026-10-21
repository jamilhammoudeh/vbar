"""Control-Lyapunov tracking baselines: the CW tracking law and the Cartesian Lyapunov law.

Both laws cancel the natural relative acceleration of a model, feed forward the reference
acceleration and close a linear spring and damper on the error, so on their own model the
Hill-frame error obeys e'' + K2 e' + K1 e = 0. `CWTrackingAgent` cancels the CW terms
(Alfriend et al., Spacecraft Formation Flying, eq 10.8); `LyapunovAgent` cancels the
acceleration of a declared two-body model (Schaub and Junkins, Analytical Mechanics of Space
Systems, eq 13.229). Neither handles the keep-out zone or any other constraint.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..dynamics import TwoBodyDynamics, make_dynamics, relative_acceleration
from .base import AgentView, action_scale

Reference = Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]


def _cw_terms(n: float, x: np.ndarray) -> np.ndarray:
    """CW acceleration F1 r + N(v) = (3 n^2 x + 2 n vy, -2 n vx, -n^2 z) in m/s^2."""
    return np.array([3 * n**2 * x[0] + 2 * n * x[4], -2 * n * x[3], -(n**2) * x[2]])


def _issue(view: AgentView, scale: float, accel: np.ndarray) -> np.ndarray:
    """`accel` (m/s^2) clipped per axis (box) or scaled onto the ball, in the view's mode."""
    fraction = accel / view.max_accel
    if view.thrust_limit == "box":
        fraction = np.clip(fraction, -1.0, 1.0)
    else:
        size = np.linalg.norm(fraction)
        if size > 1.0:
            fraction = fraction / size
    return fraction * scale


def _check_gain(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be positive and finite")


def _check_thrust_limit(view: AgentView) -> None:
    if view.thrust_limit not in ("box", "ball"):
        raise ValueError(f"unknown thrust limit {view.thrust_limit!r}")


class CWTrackingAgent:
    """CW control-Lyapunov tracking law toward a reference trajectory or the hold point.

    The command (Hill frame, m/s^2) is

        u = a_d - F1 r - N(v) - n^2 e - k n e',   e = r - r_d,   e' = v - v_d,

    with F1 = diag(3 n^2, 0, -n^2) the CW stiffness and N(v) = (2 n v_y, -2 n v_x, 0) the
    Coriolis terms, so F1 r + N(v) is the CW acceleration of the observed state and the error
    obeys e'' + k n e' + n^2 e = 0 on CW, critically damped at k = 2. This is Alfriend et al.,
    Spacecraft Formation Flying, sec. 10.1.1, eq 10.8, u = -(n^2 I + F1) e - N - k n e', with
    F1 and N derived here from the CW equations rather than copied from the book; for a natural
    CW reference a_d - F1 r_d - N(v_d) vanishes and the two forms agree, and for any other
    reference the a_d term feeds its acceleration forward. With no reference the target is the
    hold point at rest, and the law equals `PDAgent(wn=n, zeta=k/2)`. It is also the hybrid
    Hill-frame Lyapunov law of Schaub and Junkins, Analytical Mechanics of Space Systems,
    eq 13.266, on a circular target. The command is clipped per axis (box layout) or scaled
    onto the ball (ball layout) at `max_accel`, then issued in the view's action mode.

    Parameters
    ----------
    k : float
        Damping factor; the velocity gain is k n (1/s) and the position gain n^2 (1/s^2).
    reference : callable, optional
        `reference(t)` returns the desired position (m), velocity (m/s) and acceleration
        (m/s^2), each shape (3,) in the Hill frame, at `t` s since the episode start. None
        holds `view.goal["hold_point"]` at rest.
    name : str
        Name in results.

    Raises
    ------
    ValueError
        If `k` is not positive and finite.
    """

    def __init__(
        self, k: float = 2.0, reference: Reference | None = None, name: str = "cw_tracking"
    ):
        _check_gain("k", k)
        self.k = k
        self.reference = reference
        self.name = name

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Take the orbit rate, limit, mode and hold point from the view.

        Parameters
        ----------
        view : AgentView
            Scenario description; uses `n` (rad/s), `max_accel` (m/s^2), `thrust_limit`,
            `action_mode`, `mass_kg` and `goal["hold_point"]` (Hill frame, m).
        rng : numpy.random.Generator, optional
            Unused; the law is deterministic.

        Raises
        ------
        ValueError
            If the action mode takes no continuous command or the thrust layout is unknown.
        """
        self.scale = action_scale(view)
        _check_thrust_limit(view)
        self.view = view
        self.n = view.n
        self.hold = np.array(view.goal["hold_point"], dtype=float)

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Command toward the reference.

        Parameters
        ----------
        obs : numpy.ndarray
            Observation whose first six entries are the Hill state (m, m/s).
        info : dict
            `time_s` (s since the episode start), read only when a reference is given.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Command in the view's action mode.
        """
        x = np.asarray(obs, dtype=float)[:6]
        if self.reference is None:
            r_d, v_d, a_d = self.hold, np.zeros(3), np.zeros(3)
        else:
            r_d, v_d, a_d = (
                np.asarray(part, dtype=float) for part in self.reference(info["time_s"])
            )
        n = self.n
        u = a_d - _cw_terms(n, x) - n**2 * (x[:3] - r_d) - self.k * n * (x[3:] - v_d)
        return _issue(self.view, self.scale, u)


@dataclass(frozen=True, kw_only=True)
class DeclaredModel:
    """The relative-motion model a model-based baseline is told, from the public scenario fields.

    The fields mirror `EnvConfig.dynamics`, `altitude_m`, `inclination` and `body` with the same
    defaults, so a harness passes the scenario's values. A declared model that differs from the
    flown one is a model mismatch, and its effect is part of the agent's score.

    Parameters
    ----------
    dynamics : str
        "cw", "twobody" or "twobody_j2".
    altitude_m : float
        Altitude of the target's circular orbit above the body's equatorial radius (m).
    inclination : float
        Inclination of the target's orbit (rad).
    body : str
        Central body name.
    """

    dynamics: str = "cw"
    altitude_m: float = 500e3
    inclination: float = 0.0
    body: str = "earth"


@dataclass(frozen=True, kw_only=True)
class LyapunovConfig:
    """Gains and declared model of `LyapunovAgent`.

    The default gains are those of `PDAgent` at wn = 0.01 rad/s and zeta = 1 (critically
    damped), so the CW-cancelling baselines coincide at defaults on the CW model.

    Parameters
    ----------
    k1 : float
        Position gain (1/s^2).
    k2 : float
        Velocity gain (1/s).
    model : DeclaredModel
        Model whose natural acceleration the law cancels.

    Raises
    ------
    ValueError
        If `k1` or `k2` is not positive and finite.
    """

    k1: float = 1e-4
    k2: float = 0.02
    model: DeclaredModel = field(default_factory=DeclaredModel)

    def __post_init__(self):
        _check_gain("k1", self.k1)
        _check_gain("k2", self.k2)


class LyapunovAgent:
    """Cartesian Lyapunov law to the hold point on a declared relative-motion model.

    The command (Hill frame, m/s^2) is

        u = -f(x, t) - K1 (r - r_hold) - K2 v,

    Schaub and Junkins, Analytical Mechanics of Space Systems, sec. 13.7, eq 13.229, with the
    desired motion the hold point at rest, whose acceleration is zero. f is the Hill-frame
    acceleration of the observed state with no thrust on the declared model: the CW terms for
    "cw", where the law equals `CWTrackingAgent(k=K2/n)` when K1 = n^2, and
    `vbar.dynamics.relative_acceleration` for "twobody" and "twobody_j2". On the matched model the
    error obeys e'' + K2 e' + K1 e = 0 in continuous time. For the two-body models the agent
    propagates its own target orbit from the declared fields, starting at the episode start as
    the scenario's target does, and advances it to each `info["time_s"]`; it is never given an
    environment object. The command is clipped per axis (box layout) or scaled onto the ball
    (ball layout) at `max_accel`, then issued in the view's action mode.

    Parameters
    ----------
    config : LyapunovConfig, optional
        Gains and declared model; the defaults of `LyapunovConfig` when omitted.
    name : str
        Name in results.
    """

    def __init__(self, config: LyapunovConfig | None = None, name: str = "lyapunov"):
        self.config = LyapunovConfig() if config is None else config
        self.name = name

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Build the declared model with its target at the episode start.

        Parameters
        ----------
        view : AgentView
            Scenario description; uses `max_accel` (m/s^2), `thrust_limit`, `action_mode`,
            `mass_kg` and `goal["hold_point"]` (Hill frame, m). The orbit comes from the
            declared model, not from `view.n`.
        rng : numpy.random.Generator, optional
            Unused; the law is deterministic.

        Raises
        ------
        ValueError
            If the action mode takes no continuous command, the thrust layout is unknown, or
            the declared dynamics or body is unknown.
        """
        self.scale = action_scale(view)
        _check_thrust_limit(view)
        self.view = view
        self.hold = np.array(view.goal["hold_point"], dtype=float)
        m = self.config.model
        self.model = make_dynamics(m.dynamics, m.body, m.altitude_m, m.inclination)
        self.n = self.model.n
        self.model.reset(np.zeros(6))
        self.model_time = 0.0

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Command toward the hold point.

        Parameters
        ----------
        obs : numpy.ndarray
            Observation whose first six entries are the Hill state (m, m/s).
        info : dict
            `time_s` (s since the episode start), read for the two-body models.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Command in the view's action mode.

        Raises
        ------
        RuntimeError
            If the integration of the declared target orbit fails.
        """
        x = np.asarray(obs, dtype=float)[:6]
        model = self.model
        if isinstance(model, TwoBodyDynamics):
            time_s = float(info["time_s"])
            if time_s != self.model_time:
                model.advance(np.zeros(3), time_s - self.model_time)
                self.model_time = time_s
            f = relative_acceleration(x, model.target_state(), model.body, j2=model.j2)
        else:
            f = _cw_terms(self.n, x)
        cfg = self.config
        u = -f - cfg.k1 * (x[:3] - self.hold) - cfg.k2 * x[3:]
        return _issue(self.view, self.scale, u)
