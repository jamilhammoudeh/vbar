"""A language model as the supervisor of a `SupervisedAgent`.

The model gets one message per consult: the frame, the task briefing, the ladder of rungs, the
phase and the observed state, the commands it may give now and why the others are refused, and the
reason the previous order was refused. It answers with a JSON object, `{"command": "GO"}`; in
waypoint mode a GO also carries a "point" and a "speed", in velocity mode a "velocity", and an
optional "reason" is logged and never scored. A reply that is not an order counts as a parse
failure and is CONTINUE. Every number in the prompt comes from the observation, the agent-safe info
and the scenario's public description.

The model is `complete(messages, tools) -> Reply | str`, as for `vbar.agents.LLMAgent`, so
`RecordedModel` and `ReplayModel` repeat a run without the model.
"""

from __future__ import annotations

import json
import math
import string
import time
from typing import Any

import numpy as np

from .base import AgentView
from .briefing import briefing
from .briefing import situation as situation_text
from .llm import MAX_JSON_DEPTH, Model, Price, as_reply
from .supervision import MODES, Command, Order, Rung, Situation

# Longer replies are refused before parsing; an order is a few hundred characters.
MAX_REPLY_CHARS = 8192
# A reply's first few objects are tried; prose that opens a brace before the order costs one each.
_OBJECT_STARTS = 8
_DECODER = json.JSONDecoder()


def _depth_from(text: str, start: int) -> int:
    """Deepest nesting of the JSON value that opens at `start`, up to where it closes."""
    depth = peak = 0
    in_string = escaped = False
    for ch in text[start:]:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            peak = max(peak, depth)
        elif ch in "]}":
            depth -= 1
            if depth <= 0:
                break
    return peak


def _first_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in `text` among its first few opening braces, nesting included."""
    start = text.find("{")
    for _ in range(_OBJECT_STARTS):
        if start < 0:
            return None
        if _depth_from(text, start) <= MAX_JSON_DEPTH:
            try:
                value, _ = _DECODER.raw_decode(text, start)
            except (ValueError, RecursionError):
                value = None
            if isinstance(value, dict):
                return value
        start = text.find("{", start + 1)
    return None


SUPERVISOR_PROMPT = """You supervise the approach of a chaser spacecraft to a target spacecraft. A
controller flies the chaser to the reference your last command set. You do not thrust; you give
operator commands at a few instants.

Frame (centered on the target): x = radial (up, away from Earth), y = along-track
(direction of flight), z = cross-track. Units: meters, meters per second and seconds.

{briefing}

The rungs of the ladder are the places where the chaser stops and waits for the go-ahead, from
the first to the last:
{ladder}

Commands:
CONTINUE: change nothing; the controller keeps flying its reference.
GO: {go}
HOLD: stop where the chaser is and keep station there.
RETREAT: fly back to the previous rung.
CAM: fire a collision avoidance burn. It ends the approach; no command is valid afterwards.
THRUST_INHIBIT: stop all thrust and drift.
RECONFIGURE: switch to a redundant unit. The chaser has none, so it is never valid.

Phases: transfer (flying to the rung or destination), holding (arrived and holding), stopped
(after HOLD), retreating (flying to the previous rung), inhibited (after THRUST_INHIBIT), commit
(inside the commit radius), aborted (after CAM).

Status at t = {t:.1f} s, phase: {phase}, rung {rung}. Observed state:
{situation}

Commands you may give now: {commands}

{unavailable}{refused}{reply_format}"""

SUPERVISOR_FIELDS = (
    "briefing",
    "ladder",
    "go",
    "t",
    "phase",
    "rung",
    "situation",
    "commands",
    "unavailable",
    "refused",
    "reply_format",
)

_GO = {
    "ladder": (
        "fly to the next rung once the chaser holds at the current one; after a HOLD or a "
        "THRUST_INHIBIT, resume flying to the current rung."
    ),
    "waypoint": (
        "fly to the point you name at the speed you name, then hold there. The rungs are for "
        "reference only."
    ),
    "velocity": (
        "fly at the velocity you name until the next command. The rungs are for reference only."
    ),
}

_REPLY = {
    "ladder": (
        'Reply with a JSON object only, for example {"command": "GO", "reason": "settled at the '
        'rung"}. The "reason" is optional.'
    ),
    "waypoint": (
        'Reply with a JSON object only. A GO names the destination "point" [x, y, z] in meters and '
        'the "speed" in m/s along the leg, for example {"command": "GO", "point": [0.0, -100.0, '
        '0.0], "speed": 0.3}. Any other command is {"command": "HOLD"}. A "reason" is optional.'
    ),
    "velocity": (
        'Reply with a JSON object only. A GO names the "velocity" [vx, vy, vz] in m/s, for '
        'example {"command": "GO", "velocity": [0.0, 0.1, 0.0]}. Any other command is '
        '{"command": "HOLD"}. A "reason" is optional.'
    ),
}

_KIND = {
    "hold": "a stop on the way",
    "goal": "the hold point of the task",
    "contact": "the docking port",
    "orbit": "a point of the inspection circle, turning with the Sun",
}


def _number(value: Any) -> float | None:
    """`value` as a finite float if it is a JSON number, else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _vector(value: Any) -> tuple[float, float, float] | None:
    """Three finite floats from a JSON list of three numbers, else None."""
    if not isinstance(value, list) or len(value) != 3:
        return None
    x, y, z = (_number(c) for c in value)
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def parse_order(text: str, mode: str) -> Order | None:
    """The order in a model's reply, or None when the reply holds no usable one.

    The first JSON object in the reply counts, nested values included, if it opens at one of the
    reply's first eight "{"; prose before it may hold the others, and an object nested more than
    `vbar.agents.llm.MAX_JSON_DEPTH` levels deep is skipped. Its
    "command" must be the name of a `Command`, in any case and with surrounding blanks; a number
    or another spelling is not a command. In "waypoint" mode a GO also needs a "point" of three
    finite numbers (m, Hill) and a finite "speed" (m/s), in "velocity" mode a "velocity" of three
    finite numbers (m/s, Hill); the fields of the other modes, and a point, speed or velocity
    on any other command, are ignored. A speed outside the tracker's range or a point inside the
    keep-out sphere is passed on for the agent to refuse. A "reason" that is not a string is
    dropped.

    Parameters
    ----------
    text : str
        The model's reply. A reply of more than `MAX_REPLY_CHARS` characters is not read.
    mode : str
        "ladder", "waypoint" or "velocity", as the `SupervisedAgent`'s.

    Returns
    -------
    Order or None
        None if there is no object, it has no valid command, or a field the mode needs is missing,
        not a number or not finite.

    Raises
    ------
    ValueError
        If `mode` is unknown.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
    if not isinstance(text, str) or len(text) > MAX_REPLY_CHARS:
        return None
    data = _first_object(text)
    if data is None:
        return None
    name = data.get("command")
    if not isinstance(name, str) or not name.isascii():
        return None
    command = Command.__members__.get(name.strip().upper())
    if command is None:
        return None
    reason = data.get("reason")
    fields: dict[str, Any] = {"reason": reason if isinstance(reason, str) else ""}
    if command == Command.GO and mode == "waypoint":
        point, speed = _vector(data.get("point")), _number(data.get("speed"))
        if point is None or speed is None:
            return None
        fields.update(point=point, speed=speed)
    elif command == Command.GO and mode == "velocity":
        velocity = _vector(data.get("velocity"))
        if velocity is None:
            return None
        fields.update(velocity=velocity)
    return Order(command, **fields)


class LLMSupervisor:
    """Asks a language model for the supervisor's command at each consult.

    The log holds, for each consult, "time_s", "prompt", "reply", the order's "reason" (None
    when the reply was not an order), "parse_failed", "model_calls", "input_tokens",
    "output_tokens", "cost_usd" and "latency_s". An unparseable reply, or one of more than
    `MAX_REPLY_CHARS` characters, is logged as a parse failure and answered with CONTINUE. Errors
    the model raises, `ReplayError` included, are not caught.

    `mode` has to be the mode of the `SupervisedAgent` this supervisor is given to. A supervisor
    is reset with the view and the ladder only, which are the same in every mode, so a mismatch
    cannot be detected here: the agent refuses the orders of the other mode and the model is told
    why in the next prompt.

    Parameters
    ----------
    complete : callable
        `complete(messages, tools) -> Reply | str`. It gets one user message and no tools.
    prompt : str
        A `str.format` template over `SUPERVISOR_FIELDS`; the default is `SUPERVISOR_PROMPT`.
    price : Price, optional
        USD per million tokens; the cost is None without it.
    mode : str
        "ladder", "waypoint" or "velocity": which reply the prompt asks for and `parse_order` reads.

    Raises
    ------
    ValueError
        If `mode` is unknown or the template is malformed or uses a field outside
        `SUPERVISOR_FIELDS`.
    """

    name = "llm"

    def __init__(
        self,
        complete: Model,
        *,
        prompt: str = SUPERVISOR_PROMPT,
        price: Price | None = None,
        mode: str = "ladder",
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
        fields = {f for _, f, _, _ in string.Formatter().parse(prompt) if f is not None}
        unknown = fields - set(SUPERVISOR_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown prompt fields {sorted(unknown)}, expected {SUPERVISOR_FIELDS}"
            )
        self.complete = complete
        self.prompt = prompt
        self.price = price
        self.mode = mode
        self.log: list[dict[str, Any]] = []

    def reset(
        self, view: AgentView, ladder: tuple[Rung, ...], rng: np.random.Generator | None = None
    ) -> None:
        """Start an episode: write the briefing and clear the log.

        Parameters
        ----------
        view : AgentView
            The scenario's public description.
        ladder : tuple of Rung
            The task's rungs; the prompt takes them from each situation.
        rng : numpy.random.Generator, optional
            Not used.

        Raises
        ------
        ValueError
            If the goal lacks a key its task needs, or cannot be described.
        """
        self._view = view
        self._briefing = briefing(view)
        self.log = []

    def decide(self, situation: Situation) -> Order | Command:
        """The model's order for the situation, or CONTINUE if its reply is not one.

        Parameters
        ----------
        situation : Situation
            The consult's situation (Hill axes, m and m/s).

        Returns
        -------
        Order or Command
            The parsed order; `Command.CONTINUE` for an unreadable reply.
        """
        prompt = self._prompt(situation)
        t0 = time.perf_counter()
        reply = as_reply(self.complete([{"role": "user", "content": prompt}], []))
        latency = time.perf_counter() - t0
        order = parse_order(reply.text, self.mode)
        self.log.append(
            {
                "time_s": situation.t,
                "prompt": prompt,
                "reply": reply.text,
                "reason": None if order is None else order.reason,
                "parse_failed": order is None,
                "model_calls": 1,
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "cost_usd": (
                    None
                    if self.price is None
                    else self.price.cost(reply.input_tokens, reply.output_tokens)
                ),
                "latency_s": latency,
            }
        )
        return Command.CONTINUE if order is None else order

    def usage(self) -> dict[str, Any]:
        """Episode totals: decisions, model calls, tokens, cost (None without a price) and parse
        failures."""
        tokens_in = sum(e["input_tokens"] for e in self.log)
        tokens_out = sum(e["output_tokens"] for e in self.log)
        return {
            "decisions": len(self.log),
            "model_calls": sum(e["model_calls"] for e in self.log),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost_usd": None if self.price is None else self.price.cost(tokens_in, tokens_out),
            "parse_failures": sum(e["parse_failed"] for e in self.log),
        }

    def _prompt(self, sit: Situation) -> str:
        allowed = [c for c in Command if sit.mask[c]]
        refused = [c for c in Command if not sit.mask[c]]
        unavailable = ""
        if refused:
            lines = "\n".join(f"{c.name}: {sit.reasons[c]}" for c in refused)
            unavailable = f"Not available now, with the reason:\n{lines}\n\n"
        last = "" if sit.refused is None else f"Your last order was refused: {sit.refused}\n\n"
        return self.prompt.format(
            briefing=self._briefing,
            ladder="\n".join(_rung_line(k, rung) for k, rung in enumerate(sit.ladder, start=1)),
            go=_GO[self.mode],
            t=sit.t,
            phase=sit.phase,
            rung=sit.rung + 1,
            situation=situation_text(self._view, sit.obs, sit.info),
            commands=", ".join(c.name for c in allowed),
            unavailable=unavailable,
            refused=last,
            reply_format=_REPLY[self.mode],
        )


def _rung_line(number: int, rung: Rung) -> str:
    x, y, z = np.asarray(rung.position, dtype=float)
    return (
        f"{number}. [{x:.2f}, {y:.2f}, {z:.2f}] m, {_KIND[rung.kind]}; arrived within "
        f"{rung.radius:.4g} m and {rung.speed:.4g} m/s of it"
    )


__all__ = [
    "MAX_REPLY_CHARS",
    "SUPERVISOR_FIELDS",
    "SUPERVISOR_PROMPT",
    "LLMSupervisor",
    "parse_order",
]
