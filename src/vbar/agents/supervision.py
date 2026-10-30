"""Supervisory commands: a supervisor gives operator commands, a tracker flies them.

A `SupervisedAgent` lets a supervisor (rules, a learned policy, a language model) decide at a
few instants among the commands CONTINUE, GO, HOLD, RETREAT, CAM and THRUST_INHIBIT, and turns
the accepted ones into set points for a `SetpointTracker`. The supervisor sees a `Situation`
built from the observation and the agent-safe info alone, with the validity of each command
worked out from them, and an order the mask refuses is recorded and ignored. Wrapped in a
`vbar.safety.FilteredAgent` the tracker's command passes the safety filter like any other.

The task is cut into a ladder of rungs (`ladder`): the places a supervisor stops the chaser
before it gives the go-ahead for the next, after the staged approaches of Fehse 6.5 and the
hold points of Shuttle rendezvous.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from ..dynamics import cw_discrete
from .base import AgentView
from .briefing import check_goal
from .tracker import ARRIVAL_RADIUS, ARRIVAL_SPEED, SetpointTracker


class Command(IntEnum):
    """What a supervisor can tell the chaser to do."""

    CONTINUE = 0
    GO = 1
    HOLD = 2
    RETREAT = 3
    CAM = 4
    THRUST_INHIBIT = 5
    RECONFIGURE = 6


PHASES = ("transfer", "holding", "stopped", "retreating", "inhibited", "commit", "aborted")
MODES = ("ladder", "waypoint", "velocity")
_RUNG_KINDS = ("hold", "goal", "contact", "orbit")

# Range (m) of the inspection entry point and the orbit, and of a hold rung placed along the line
# of sight when the goal is at the target.
_ENTRY_RANGE = 60.0
_SIGHT_RANGE = 50.0
# A docking standoff lies this far (m) outside the farthest approach cone's active range.
_STANDOFF_MARGIN = 10.0
# Drift and CAM arcs are sampled every _SAMPLE_S seconds over two orbits.
_SAMPLE_S = 5.0
_ORBITS = 2.0


@dataclass(frozen=True)
class Order:
    """A supervisor's order: a command, with a destination in the non-ladder modes.

    The fields are what the supervisor said; the agent checks them and records a refusal
    instead of raising.

    Parameters
    ----------
    command : Command
        The command.
    point : sequence of float, optional
        Waypoint mode: destination (m, Hill).
    velocity : sequence of float, optional
        Velocity mode: velocity to fly at (m/s, Hill).
    speed : float, optional
        Waypoint mode: speed along the leg (m/s).
    unit : int, optional
        Unit a RECONFIGURE names.
    reason : str
        The supervisor's reason, logged and never scored.
    """

    command: Command
    _: KW_ONLY
    point: Sequence[float] | None = None
    velocity: Sequence[float] | None = None
    speed: float | None = None
    unit: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class Rung:
    """A place on the ladder where the chaser stops and waits for the go-ahead.

    Parameters
    ----------
    position : array_like
        Position (m, Hill), shape (3,), at t = 0; an "orbit" rung turns with the Sun.
    kind : str
        "hold" (a stop of the ladder's own), "goal" (the scenario's hold point), "contact" (the
        docking port) or "orbit" (the inspection circle).
    radius : float
        Distance (m) from the rung inside which the chaser has arrived.
    speed : float
        Speed (m/s) relative to the rung below which the chaser has arrived.
    """

    position: ArrayLike
    kind: str
    radius: float
    speed: float

    def __post_init__(self):
        position = tuple(float(c) for c in self.position)
        if len(position) != 3 or not all(math.isfinite(c) for c in position):
            raise ValueError("a rung position needs three finite coordinates")
        if self.kind not in _RUNG_KINDS:
            raise ValueError(f"unknown rung kind {self.kind!r}")
        if not (math.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError("a rung radius must be positive and finite")
        if not (math.isfinite(self.speed) and self.speed > 0.0):
            raise ValueError("a rung speed must be positive and finite")
        object.__setattr__(self, "position", position)


def ladder(view: AgentView, obs0: ArrayLike) -> tuple[Rung, ...]:
    """The rungs of a task, from the public goal and the first observation.

    A hold goal off the target gets a stop twice as far out and the goal itself; one at the
    target a stop 50 m along the first line of sight, then the target. A docking goal gets a
    standoff outside every approach cone, the goal's station-keeping point if it has one, and the
    port. An inspection goal gets an entry point at 60 m on the first line of sight and the
    point of the 60 m circle that keeps the Sun behind the inspector.

    Parameters
    ----------
    view : AgentView
        The scenario's public description; goal positions in m, Hill axes at t = 0.
    obs0 : array_like
        The first observation; its first three entries are the observed position (m, Hill).

    Returns
    -------
    tuple of Rung
        The rungs in the order they are flown.

    Raises
    ------
    ValueError
        If the goal lacks a key its task needs, the task has no ladder, or a rung is at the target
        along a line of sight that is undefined.
    """
    check_goal(view)
    goal = view.goal
    task = goal.get("task", "hold")
    r0 = np.asarray(obs0, dtype=float)[:3]
    if task in ("hold", "berth"):
        goal_rung = Rung(goal["hold_point"], "goal", goal["capture_radius"], goal["capture_speed"])
        h = np.asarray(goal["hold_point"], dtype=float)
        size = float(np.linalg.norm(h))
        if size > 0.0:
            out = h + size * h / size
        else:
            out = _SIGHT_RANGE * _unit(r0, "the first observation")
        return (Rung(out, "hold", ARRIVAL_RADIUS, ARRIVAL_SPEED), goal_rung)
    if task == "dock":
        port = np.asarray(goal["port"], dtype=float)
        axis = np.asarray(goal["port_axis"], dtype=float)
        reach = max(
            (c["active_range"] for c in view.constraints if c["type"].startswith("approach_cone")),
            default=0.0,
        )
        if not math.isfinite(reach):
            raise ValueError("an approach cone with no active range leaves no standoff")
        rungs = [
            Rung(port + (reach + _STANDOFF_MARGIN) * axis, "hold", ARRIVAL_RADIUS, ARRIVAL_SPEED)
        ]
        if "hold" in goal:
            rungs.append(Rung(port + goal["hold"][0] * axis, "hold", ARRIVAL_RADIUS, ARRIVAL_SPEED))
        low, high = goal["axial_speed_window"]
        rungs.append(Rung(port, "contact", goal["lateral_limit"], high))
        return tuple(rungs)
    if task == "inspect":
        if "sun_direction" not in goal:
            raise ValueError("the inspect goal has no sun_direction")
        sun = np.asarray(goal["sun_direction"], dtype=float)
        planar = _unit(np.array([sun[0], sun[1], 0.0]), "the Sun direction")
        entry = _ENTRY_RANGE * _unit(r0, "the first observation")
        return (
            Rung(entry, "hold", ARRIVAL_RADIUS, ARRIVAL_SPEED),
            Rung(_ENTRY_RANGE * planar, "orbit", ARRIVAL_RADIUS, ARRIVAL_SPEED),
        )
    raise ValueError(f"the {task} task has no ladder")


def _unit(vector: np.ndarray, what: str) -> np.ndarray:
    size = float(np.linalg.norm(vector))
    if size == 0.0:
        raise ValueError(f"{what} has no direction")
    return vector / size


@dataclass(frozen=True, eq=False)
class Situation:
    """What a supervisor is told at a consult; all of it from the observation and agent info.

    Parameters
    ----------
    t : float
        Time (s).
    obs : numpy.ndarray
        Copy of the observation (m, m/s, Hill).
    phase : str
        One of `PHASES`.
    rung : int
        Index of the rung being flown to, held, or retreated to.
    ladder : tuple of Rung
        The rungs of the task.
    mask : tuple of bool
        Whether each command, indexed by `Command`, may be given now.
    reasons : mapping
        For each command the mask refuses, why.
    refused : str or None
        Why the previous consult's order was refused, if it was.
    info : mapping
        Copy of the agent-safe info the agent was given (time, coverage).
    """

    t: float
    obs: np.ndarray
    phase: str
    rung: int
    ladder: tuple[Rung, ...]
    mask: tuple[bool, ...]
    reasons: Mapping[Command, str]
    refused: str | None
    info: Mapping[str, Any]


@dataclass(frozen=True, eq=False)
class SupervisionStep:
    """What a `SupervisedAgent` did in its latest step.

    `order`, `accepted` and `reason` are None, None and "" when the supervisor was not consulted;
    `phase` and `rung` are those in effect once the order is applied, `mask` that of the latest
    consult, and `r_ref` and `v_ref` the reference the tracker flies to (m, m/s, Hill).
    """

    t: float
    consulted: bool
    order: Order | None
    accepted: bool | None
    reason: str
    phase: str
    rung: int
    mask: tuple[bool, ...]
    r_ref: tuple[float, ...]
    v_ref: tuple[float, ...]


class Supervisor(Protocol):
    name: str

    def reset(
        self, view: AgentView, ladder: tuple[Rung, ...], rng: np.random.Generator | None = None
    ) -> None:
        """Start an episode with the public view and the task's rungs."""

    def decide(self, situation: Situation) -> Order | Command:
        """Give a command (or an order with a destination) for the situation."""


def _drift_margin(
    n: float, state: np.ndarray, radius: float, dv: np.ndarray | None = None
) -> float:
    """Least range minus the collision radius over two orbits of free drift (m).

    The drift starts from `state` after the impulse `dv` (m/s). Ranges are sampled every
    `_SAMPLE_S` s and each interval is lowered by half its length times the greater end speed, the
    most the range can fall between samples. The pad assumes the speed varies little over an
    interval, so the margin is a heuristic, not a certificate.
    """
    x0 = np.array(state[:6], dtype=float)
    if dv is not None:
        x0[3:] += dv
    times = np.arange(0.0, _ORBITS * 2.0 * math.pi / n + _SAMPLE_S, _SAMPLE_S)
    phi, _ = cw_discrete(n, times)
    path = phi @ x0
    rho = np.linalg.norm(path[:, :3], axis=1)
    speed = np.linalg.norm(path[:, 3:], axis=1)
    pad = 0.5 * _SAMPLE_S * np.maximum(speed[:-1], speed[1:])
    lowest = np.minimum(rho.min(), (0.5 * (rho[:-1] + rho[1:]) - pad).min())
    return float(lowest) - radius


def _abort_impulses(view: AgentView) -> np.ndarray:
    """The collision avoidance impulses (m/s, Hill) the goal declares, else the metric's rule."""
    declared = view.goal.get("cam_impulses")
    if declared is not None:
        return np.asarray(declared, dtype=float).reshape(-1, 3)
    # Imported here: the metrics import the agents package.
    from ..metrics.passive import abort_set
    from ..metrics.settings import MetricSettings

    try:
        return abort_set(view.goal, MetricSettings())
    except ValueError:
        return np.zeros((0, 3))


class SupervisedAgent:
    """Runs a supervisor's commands through a set-point tracker.

    The supervisor is consulted at t = 0, every `period_s` seconds and whenever the phase or rung
    changes, at most once per step. Between consults the tracker keeps flying the last accepted
    order. In "ladder" mode GO sends the chaser to the next rung; in "waypoint" mode GO carries a
    point and a speed, in "velocity" mode a velocity, and the ladder is only reported.

    Parameters
    ----------
    supervisor : Supervisor
        Gives a `Command` or an `Order` for each `Situation`; anything else is refused.
    tracker : SetpointTracker, optional
        The tracker; copied, so one tracker can be given to several agents. Default gains if None.
    period_s : float
        Time between scheduled consults (s); a whole number of the scenario's steps.
    mode : str
        "ladder", "waypoint" or "velocity".
    name : str, optional
        The agent's name; default from the supervisor's.

    Raises
    ------
    ValueError
        If `period_s` is not positive and finite, `mode` is unknown, or the supervisor or tracker
        has the wrong type.
    """

    def __init__(
        self,
        supervisor: Supervisor,
        *,
        tracker: SetpointTracker | None = None,
        period_s: float = 60.0,
        mode: str = "ladder",
        name: str | None = None,
    ):
        if not (
            callable(getattr(supervisor, "reset", None))
            and callable(getattr(supervisor, "decide", None))
        ):
            raise ValueError("the supervisor needs reset and decide methods")
        if tracker is not None and not isinstance(tracker, SetpointTracker):
            raise ValueError("tracker must be a SetpointTracker")
        if isinstance(period_s, bool) or not isinstance(period_s, (int, float)):
            raise ValueError(f"period_s must be a number, got {period_s!r}")
        if not (math.isfinite(period_s) and period_s > 0.0):
            raise ValueError(f"period_s must be positive and finite, got {period_s!r}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
        self.supervisor = supervisor
        self.period_s = float(period_s)
        self.mode = mode
        self.name = name or f"supervised-{getattr(supervisor, 'name', type(supervisor).__name__)}"
        self._tracker = copy.copy(tracker if tracker is not None else SetpointTracker())
        self.last_step: SupervisionStep | None = None
        self._ready = False

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode; the supervisor is reset with the ladder at the first observation.

        Parameters
        ----------
        view : AgentView
            The scenario's public description.
        rng : numpy.random.Generator, optional
            The agent's stream, passed on to the supervisor.

        Raises
        ------
        ValueError
            If the action mode is not continuous, or `period_s` is not a whole number of steps.
        """
        steps = round(self.period_s / view.dt)
        if steps < 1 or abs(steps * view.dt - self.period_s) > 1e-9 * max(1.0, self.period_s):
            raise ValueError(
                f"period_s {self.period_s} is not a whole number of the {view.dt} s steps"
            )
        self._tracker.reset(view)
        self._ready = True
        self._view = view
        self._rng = rng
        self._steps = steps
        self._ladder: tuple[Rung, ...] = ()
        self._phase = "transfer" if self.mode == "ladder" else "stopped"
        self._rung = 0
        self._seen: tuple[str, int] | None = None
        self._refused: str | None = None
        self._mask = (True,) + (False,) * (len(Command) - 1)
        self._cam = np.zeros(3)
        self._collision = max(
            (float(c["radius"]) for c in view.constraints if c["type"].startswith("keep_out")),
            default=0.0,
        )
        self._commit = float(view.goal.get("commit_radius", 0.0))
        self._impulses = _abort_impulses(view)
        self.last_step = None

    def act(self, obs, info) -> np.ndarray:
        """The tracker's command for observation `obs`, after consulting the supervisor if due.

        Parameters
        ----------
        obs : numpy.ndarray
            The observation (m, m/s, Hill); the first six entries are the observed state.
        info : mapping
            The agent-safe info; "time_s" (s) is the time.

        Returns
        -------
        numpy.ndarray
            The command in the view's action mode.
        """
        if not self._ready:
            raise RuntimeError("reset the agent before acting")
        view = self._view
        t = float(info["time_s"])
        obs = np.array(obs, dtype=float)
        if not self._ladder:
            self._start(obs)
        self._arrive(obs, t)
        state = (self._shown(obs), self._rung)
        k = round(t / view.dt)
        consulted = self.last_step is None or k % self._steps == 0 or state != self._seen
        order = accepted = None
        reason = ""
        if consulted:
            order, accepted, reason = self._consult(obs, t, info, state)
        action = self._tracker.act(obs, t)
        r_ref, v_ref = self._tracker.reference(t)
        self.last_step = SupervisionStep(
            t=t,
            consulted=consulted,
            order=order,
            accepted=accepted,
            reason=reason,
            phase=self._shown(obs),
            rung=self._rung,
            mask=self._mask,
            r_ref=tuple(float(c) for c in r_ref),
            v_ref=tuple(float(c) for c in v_ref),
        )
        return action

    def _start(self, obs: np.ndarray) -> None:
        self._ladder = ladder(self._view, obs)
        self.supervisor.reset(self._view, self._ladder, self._rng)
        if self.mode == "ladder":
            self._fly_rung()
        else:
            self._tracker.hold(obs[:3])

    def _fly_rung(self) -> None:
        rung = self._ladder[self._rung]
        self._tracker.fly_to(rung.position, rung.kind)

    def _shown(self, obs: np.ndarray) -> str:
        """The phase a supervisor sees: commit inside the commit radius, unless aborted."""
        if self._phase != "aborted" and float(np.linalg.norm(obs[:3])) < self._commit:
            return "commit"
        return self._phase

    def _arrived(self, obs: np.ndarray, t: float) -> bool:
        """Whether the observation is within the rung's radius and speed of its reference."""
        if self.mode == "velocity":
            return False
        rung = self._ladder[self._rung]
        radius, speed = (
            (rung.radius, rung.speed)
            if self.mode == "ladder"
            else (
                ARRIVAL_RADIUS,
                ARRIVAL_SPEED,
            )
        )
        r_ref, v_ref = self._tracker.reference(t)
        return bool(
            np.linalg.norm(obs[:3] - r_ref) <= radius and np.linalg.norm(obs[3:6] - v_ref) <= speed
        )

    def _arrive(self, obs: np.ndarray, t: float) -> None:
        if self._phase in ("transfer", "retreating") and self._arrived(obs, t):
            self._phase = "holding"

    def _validity(self, obs: np.ndarray, t: float) -> tuple[tuple[bool, ...], dict[Command, str]]:
        """Whether each command may be given now, and why not, from the observation alone."""
        view = self._view
        phase, shown = self._phase, self._shown(obs)
        ok = [True] * len(Command)
        why: dict[Command, str] = {}

        def refuse(command: Command, text: str) -> None:
            ok[command] = False
            why[command] = text

        refuse(Command.RECONFIGURE, "no redundant units")
        if phase == "aborted":
            for command in (Command.GO, Command.HOLD, Command.RETREAT, Command.CAM):
                refuse(command, "aborted after a CAM")
            refuse(Command.THRUST_INHIBIT, "aborted after a CAM")
            return tuple(ok), why
        if shown == "commit":
            for command in (Command.GO, Command.HOLD, Command.RETREAT, Command.CAM):
                refuse(command, "inside the commit radius")
            if view.goal.get("task") != "dock":
                refuse(Command.THRUST_INHIBIT, "inhibit inside the commit radius is for docking")
            elif phase == "inhibited":
                refuse(Command.THRUST_INHIBIT, "already inhibited")
            return tuple(ok), why

        last = self._rung == len(self._ladder) - 1
        if self.mode == "ladder":
            if phase == "transfer":
                refuse(Command.GO, "not arrived at the rung")
            elif phase == "holding":
                if not self._arrived(obs, t):
                    refuse(Command.GO, "arrival at the rung not held")
                elif last:
                    refuse(Command.GO, "the last rung")
        if phase in ("holding", "stopped"):
            refuse(Command.HOLD, "already holding" if phase == "holding" else "already stopped")
        else:
            x, z = obs[0], obs[2]
            keeping = math.hypot(3.0 * view.n**2 * x, view.n**2 * z)
            if keeping > self._tracker.hold_accel * view.max_accel:
                refuse(Command.HOLD, "station keeping here costs more than the hold limit")
        if phase == "retreating":
            refuse(Command.RETREAT, "already retreating")
        elif self._rung < 1:
            refuse(Command.RETREAT, "no previous rung")

        if len(self._impulses) == 0:
            refuse(Command.CAM, "no abort impulse")
        elif self._collision <= 0.0:
            self._cam = self._impulses[0]
        else:
            margins = [_drift_margin(view.n, obs, self._collision, dv) for dv in self._impulses]
            if max(margins) < 0.0:
                refuse(Command.CAM, "the CAM arc enters the keep-out sphere")
            else:
                self._cam = self._impulses[int(np.argmax(margins))]
        if phase == "inhibited":
            refuse(Command.THRUST_INHIBIT, "already inhibited")
        elif self._collision > 0.0 and _drift_margin(view.n, obs, self._collision) < 0.0:
            refuse(Command.THRUST_INHIBIT, "the free drift enters the keep-out sphere")
        return tuple(ok), why

    def _consult(
        self, obs: np.ndarray, t: float, info, state: tuple[str, int]
    ) -> tuple[Order | None, bool, str]:
        self._seen = state
        self._mask, reasons = self._validity(obs, t)
        situation = Situation(
            t=t,
            obs=_frozen(obs),
            phase=state[0],
            rung=self._rung,
            ladder=self._ladder,
            mask=self._mask,
            reasons=MappingProxyType(dict(reasons)),
            refused=self._refused,
            info=MappingProxyType(copy.deepcopy(dict(info))),
        )
        answer = self.supervisor.decide(situation)
        if isinstance(answer, Command):
            answer = Order(answer)
        if not isinstance(answer, Order):
            self._refused = f"not a Command or an Order: {answer!r:.60}"
            return None, False, self._refused
        reason = self._check(answer, reasons)
        if reason:
            self._refused = reason
            return answer, False, reason
        self._refused = None
        self._apply(answer, obs)
        return answer, True, ""

    def _check(self, order: Order, reasons: Mapping[Command, str]) -> str:
        """Why the order is refused, or "" if it may be applied."""
        command = order.command
        if not isinstance(command, Command):
            return f"unknown command {command!r:.40}"
        if command != Command.GO:
            if order.point is not None or order.velocity is not None or order.speed is not None:
                return f"{command.name} takes no point, velocity or speed"
        elif self.mode == "ladder":
            if order.point is not None or order.velocity is not None or order.speed is not None:
                return "ladder mode takes no point, velocity or speed"
        elif self.mode == "waypoint":
            reason = self._check_waypoint(order)
            if reason:
                return reason
        else:
            reason = self._check_velocity(order)
            if reason:
                return reason
        if not self._mask[command]:
            return f"{command.name}: {reasons[command]}"
        return ""

    def _check_waypoint(self, order: Order) -> str:
        if order.velocity is not None:
            return "waypoint mode takes no velocity"
        if order.point is None or order.speed is None:
            return "waypoint GO needs a point and a speed"
        point = _vector(order.point)
        if point is None:
            return "point is not three finite numbers"
        speed = order.speed
        if (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not math.isfinite(speed)
        ):
            return "speed is not a finite number"
        if not 0.0 < speed <= self._tracker.v_max:
            return f"speed outside 0 to {self._tracker.v_max} m/s"
        if float(np.linalg.norm(point)) < self._collision:
            return "point inside the keep-out sphere"
        return ""

    def _check_velocity(self, order: Order) -> str:
        if order.point is not None or order.speed is not None:
            return "velocity mode takes no point or speed"
        if order.velocity is None:
            return "velocity GO needs a velocity"
        velocity = _vector(order.velocity)
        if velocity is None:
            return "velocity is not three finite numbers"
        if float(np.linalg.norm(velocity)) > self._tracker.v_max:
            return f"velocity above {self._tracker.v_max} m/s"
        return ""

    def _apply(self, order: Order, obs: np.ndarray) -> None:
        command = order.command
        if command == Command.GO:
            if self.mode == "ladder":
                if self._phase in ("holding", "retreating"):
                    self._rung += 1
                self._fly_rung()
            elif self.mode == "waypoint":
                self._tracker.fly_to(_vector(order.point), "waypoint", speed=order.speed)
            else:
                self._tracker.track_velocity(obs[:3], _vector(order.velocity))
            self._phase = "transfer"
        elif command == Command.HOLD:
            self._tracker.hold(obs[:3])
            self._phase = "stopped"
        elif command == Command.RETREAT:
            self._rung -= 1
            self._fly_rung()
            self._phase = "retreating"
        elif command == Command.CAM:
            self._tracker.fire(self._cam)
            self._phase = "aborted"
        elif command == Command.THRUST_INHIBIT:
            self._tracker.coast()
            self._phase = "inhibited"


def _vector(values) -> np.ndarray | None:
    """Three finite floats from `values`, or None."""
    try:
        v = np.array(values, dtype=float)
    except (TypeError, ValueError):
        return None
    return v if v.shape == (3,) and np.all(np.isfinite(v)) else None


def _frozen(array: np.ndarray) -> np.ndarray:
    out = np.array(array, dtype=float)
    out.setflags(write=False)
    return out
