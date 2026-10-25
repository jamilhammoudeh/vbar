"""LLM agent: asks a language model for a thrust command at each decision step.

The model is passed in as `complete(messages, tools) -> Reply | str`, so any hosted API or local
model works through a small adapter; `text_model` wraps a plain `prompt -> text` function.
`messages` is the decision's conversation as provider-neutral dicts with a "role" ("user",
"assistant" or "tool") and a "content" string; `tools` lists the JSON-schema descriptions of the
tools offered. A reply may call tools; the agent runs them on its own observation, adds the
results to the conversation and asks again, at most `max_tool_rounds` times, then asks once more
without tools. Each decision's messages, tool calls, reply, parsed action, parse failure, token
counts, cost and wall-clock latency are stored in `self.log`. With `belief=True` the model also
reports its belief about the range and whether it is closing, which is checked against the
observation it was given (metric A5). `RecordedModel` keeps a run's
requests and replies and `ReplayModel` plays them back, so a run repeats exactly without the
model.

The default prompt describes the goal and constraints of every task kind and the observed state
measured against them (`vbar.agents.briefing`). The simulation clock is paused during the model
call unless the run is in real time. Every number in the prompt comes from the observation, the
agent's `info` or the scenario's public description, never from the true state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import string
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .base import AgentView, action_scale
from .tools import Tool, call_tool

PROMPT_TEMPLATE = """You are the guidance system of a chaser spacecraft near a target spacecraft.

Frame (centered on the target): x = radial (up, away from Earth), y = along-track
(direction of flight), z = cross-track. Units: meters, meters per second and seconds.
Orbital mechanics apply (Clohessy-Wiltshire dynamics, mean motion n = {n:.6f} rad/s).

{briefing}

Current state (t = {t:.0f} s):
  position = [{x:.2f}, {y:.2f}, {z:.2f}] m
  velocity = [{vx:.4f}, {vy:.4f}, {vz:.4f}] m/s
{situation}

Each command is held for {dt:.0f} s. {thrust}

{reply_format}"""

REPLY_FORMAT = 'Reply with ONLY a JSON object, for example: {"ax": 0.1, "ay": -0.5, "az": 0.0}'
BELIEF_FORMAT = (
    "Reply with ONLY a JSON object with the command and what you believe about the target: "
    '"range_m", your distance to the target in meters, and "closing", true if that distance '
    'is shrinking. For example: {"ax": 0.1, "ay": -0.5, "az": 0.0, "range_m": 350.0, '
    '"closing": true}'
)
# A belief matches the observation when its range is within this fraction of the observed one.
BELIEF_RANGE_TOL = 0.1

# Fields a prompt template may use.
PROMPT_FIELDS = (
    "briefing",
    "situation",
    "thrust",
    "hold",
    "cap_speed",
    "keep_out",
    "n",
    "t",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "dist",
    "rng",
    "dt",
    "max_accel",
    "reply_format",
)
# Fields only some goals can fill, with what they need.
GOAL_FIELDS = {
    "hold": "hold_point",
    "dist": "hold_point",
    "cap_speed": "capture_speed",
    "keep_out": "keep-out radius",
}


@dataclass(frozen=True)
class ToolCall:
    """One tool call in a reply; `id` pairs it with its result message."""

    name: str
    arguments: Mapping[str, Any]
    id: str = ""


@dataclass(frozen=True)
class Reply:
    """A model's answer: its text, any tool calls, and the tokens it counted (0 if unknown)."""

    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class Price:
    """A declared price table for metric A3: USD per million input and output tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


Model = Callable[[list[dict[str, Any]], list[dict[str, Any]]], "Reply | str"]


def _plain(arguments: Any) -> Any:
    """Tool-call arguments for the log: a dict copy, or the value as the model sent it."""
    return dict(arguments) if isinstance(arguments, Mapping) else arguments


def text_model(complete: Callable[[str], str]) -> Model:
    """Adapt a `prompt -> text` function: it receives the last message's content."""

    def model(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        return complete(messages[-1]["content"])

    return model


def as_reply(answer: Reply | str) -> Reply:
    return answer if isinstance(answer, Reply) else Reply(text=str(answer))


def request_digest(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """SHA-256 of a request's canonical JSON; equal requests have equal digests."""
    canonical = json.dumps({"messages": messages, "tools": tools}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reply_to_dict(reply: Reply) -> dict[str, Any]:
    return {
        "text": reply.text,
        "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
        "tool_calls": [
            {"name": c.name, "arguments": _plain(c.arguments), "id": c.id} for c in reply.tool_calls
        ],
    }


def _reply_from_dict(entry: Mapping[str, Any]) -> Reply:
    calls = tuple(ToolCall(c["name"], c["arguments"], c.get("id", "")) for c in entry["tool_calls"])
    return Reply(entry["text"], entry["input_tokens"], entry["output_tokens"], calls)


class RecordedModel:
    """Calls `model` and records each request's digest with its reply, in order."""

    def __init__(self, model: Model):
        self.model = model
        self.entries: list[dict[str, Any]] = []

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        reply = as_reply(self.model(messages, tools))
        self.entries.append(
            {"request": request_digest(messages, tools), "reply": _reply_to_dict(reply)}
        )
        return reply

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.entries, indent=1), encoding="utf-8")


class ReplayError(RuntimeError):
    """A replayed run asked for something the transcript does not hold."""


class ReplayModel:
    """Answers each request with the recorded reply, in order, without calling a model.

    Raises `ReplayError` when a request differs from the recorded one (the run has diverged) or
    the transcript is exhausted. Replies carry the recorded token counts.
    """

    def __init__(self, entries: Sequence[Mapping[str, Any]]):
        self.entries = list(entries)
        self.position = 0

    @classmethod
    def load(cls, path: str | Path) -> ReplayModel:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        if self.position >= len(self.entries):
            raise ReplayError(f"the run made more than the {len(self.entries)} recorded requests")
        entry = self.entries[self.position]
        if request_digest(messages, tools) != entry["request"]:
            raise ReplayError(f"request {self.position} differs from the transcript")
        self.position += 1
        return _reply_from_dict(entry["reply"])


# Everything converting a model's JSON can raise; a reply that triggers any of them is unusable.
REPLY_ERRORS = (ValueError, KeyError, TypeError, OverflowError, RecursionError)

# Deeper replies are refused before parsing: how deep `json.loads` goes before raising
# RecursionError differs between Python versions and builds (some stop short of 3000 levels).
MAX_JSON_DEPTH = 32


def json_depth(text: str) -> int:
    """Deepest nesting of JSON arrays and objects in `text`, ignoring brackets inside strings."""
    depth = peak = 0
    in_string = escaped = False
    for ch in text:
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
    return peak


def first_json(text: str, pattern: str = r"\{.*?\}") -> Any:
    """The first JSON value in `text` matching `pattern`, or None when there is none or it
    cannot be parsed (malformed, or nested too deeply)."""
    if not isinstance(text, str):
        return None
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match or json_depth(match.group(0)) > MAX_JSON_DEPTH:
        return None
    try:
        return json.loads(match.group(0))
    except (ValueError, RecursionError):
        return None


def parse_belief(text: str) -> dict[str, Any] | None:
    """The first JSON object in a reply with a finite number `range_m` and a boolean `closing`.

    Objects are read in order and those without both are skipped, so a belief may follow the
    command in its own object; quoted numbers do not count. The agent and metric A5 both use it.
    """
    if not isinstance(text, str):
        return None
    for match in re.finditer(r"\{[^{}]*\}", text):
        if json_depth(match.group(0)) > MAX_JSON_DEPTH:
            continue
        try:
            data = json.loads(match.group(0))
        except (ValueError, RecursionError):
            continue
        range_m, closing = data.get("range_m"), data.get("closing")
        if isinstance(range_m, bool) or not isinstance(range_m, int | float):
            continue
        try:
            range_m = float(range_m)
        except OverflowError:
            continue
        if math.isfinite(range_m) and isinstance(closing, bool):
            return {"range_m": range_m, "closing": closing}
    return None


def belief_matches(belief: Mapping[str, Any] | None, obs: np.ndarray) -> bool:
    """Whether a belief agrees with the observation it answered (metric A5).

    The range must be within 10% of the observed range, and "closing" must equal r . v < 0 for
    the observed position r and velocity v. It measures consistency with the agent's own
    inputs, not with the truth. A missing belief does not match.
    """
    if belief is None:
        return False
    observed = float(np.linalg.norm(obs[:3]))
    close_enough = abs(belief["range_m"] - observed) <= BELIEF_RANGE_TOL * observed
    return close_enough and belief["closing"] == bool(obs[:3] @ obs[3:6] < 0.0)


def parse_action(text: str) -> np.ndarray | None:
    """Extract {"ax":..,"ay":..,"az":..} from a model reply. Returns None on failure."""
    data = first_json(text)
    try:
        a = np.array([float(data["ax"]), float(data["ay"]), float(data["az"])])
    except REPLY_ERRORS:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return np.clip(a, -1.0, 1.0)


@dataclass
class LLMAgent:
    """Agent that asks `complete(messages, tools)` for each command.

    `prompt` is a `str.format` template over `PROMPT_FIELDS`; the default describes the goal and
    constraints of every task kind (`vbar.agents.briefing`), and `reset` refuses a goal that
    lacks a key the template's fields need. `price` turns the reported tokens into dollars;
    `tools` are offered to the model for at most `max_tool_rounds` rounds of calls per decision;
    `belief` asks for the model's range and closing belief and checks it. An unparseable reply
    is logged and the spacecraft coasts.
    """

    complete: Model
    _: KW_ONLY
    name: str = "llm"
    prompt: str = PROMPT_TEMPLATE
    price: Price | None = None
    tools: Sequence[Tool] = ()
    max_tool_rounds: int = 4
    belief: bool = False
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.fields = {f for _, f, _, _ in string.Formatter().parse(self.prompt) if f}
        unknown = self.fields - set(PROMPT_FIELDS)
        if unknown:
            raise ValueError(f"unknown prompt fields {sorted(unknown)}, expected {PROMPT_FIELDS}")

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        # The briefing reads the target modes of `vbar.scenarios`, which imports this module.
        from .briefing import briefing, check_goal

        # Refuse a goal the prompt cannot describe now, not at the first decision.
        check_goal(view)
        radii = [c["radius"] for c in view.constraints if c["type"] == "keep_out" and "radius" in c]
        available = {"hold_point", "capture_speed"} & set(view.goal)
        if radii:
            available.add("keep-out radius")
        missing = sorted({GOAL_FIELDS[f] for f in self.fields if f in GOAL_FIELDS} - available)
        if missing:
            task = view.goal.get("task", "hold")
            raise ValueError(
                f"the prompt needs a {', '.join(missing)}, which the {task} goal lacks"
            )
        self.view = view
        self.scale = action_scale(view)
        self.keep_out = radii[0] if radii else None
        self.briefing = briefing(view)
        if view.thrust_limit == "ball":
            self.thrust = (
                f"The command is a vector of fractions of the maximum ({view.max_accel} m/s^2); a "
                "vector longer than 1 is shortened to length 1."
            )
        else:
            self.thrust = (
                f"Thrust per axis is a fraction of the maximum ({view.max_accel} m/s^2), between "
                "-1 and 1."
            )
        self.log = []

    def prompt_text(self, obs: np.ndarray, info: Mapping[str, Any]) -> str:
        """The prompt for observation `obs`, from the observation and the public view only."""
        view = self.view
        x, y, z, vx, vy, vz = obs[:6]
        values: dict[str, Any] = {
            "briefing": self.briefing,
            "thrust": self.thrust,
            "n": view.n,
            "t": info["time_s"],
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "rng": float(np.linalg.norm(obs[:3])),
            "dt": view.dt,
            "max_accel": view.max_accel,
            "reply_format": self.reply_format(),
        }
        if "situation" in self.fields:
            from .briefing import situation

            values["situation"] = situation(view, obs, info)
        if "hold_point" in view.goal:
            hold = np.array(view.goal["hold_point"], dtype=float)
            values["hold"] = hold.tolist()
            values["dist"] = float(np.linalg.norm(obs[:3] - hold))
        if "capture_speed" in view.goal:
            values["cap_speed"] = view.goal["capture_speed"]
        values["keep_out"] = self.keep_out
        return self.prompt.format(**values)

    def reply_format(self) -> str:
        return BELIEF_FORMAT if self.belief else REPLY_FORMAT

    def act(self, obs, info):
        messages: list[dict[str, Any]] = [{"role": "user", "content": self.prompt_text(obs, info)}]
        offered = {tool.name: tool for tool in self.tools}
        specs = [tool.spec() for tool in self.tools]
        replies, calls = [], []
        t0 = time.perf_counter()
        for round_ in range(self.max_tool_rounds + 1):
            # The last call offers no tools, so the model has to answer.
            last = round_ == self.max_tool_rounds or not specs
            reply = as_reply(self.complete(messages, [] if last else specs))
            replies.append(reply)
            if last or not reply.tool_calls:
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.text,
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": _plain(c.arguments)}
                        for c in reply.tool_calls
                    ],
                }
            )
            for c in reply.tool_calls:
                if isinstance(c.name, str) and isinstance(c.arguments, Mapping):
                    result = call_tool(offered, c.name, obs, self.view, c.arguments)
                else:
                    result = {"error": "a tool call needs a name and an object of arguments"}
                calls.append({"name": c.name, "arguments": _plain(c.arguments), "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": c.id,
                        "name": c.name,
                        "content": json.dumps(result, sort_keys=True),
                    }
                )
        latency = time.perf_counter() - t0
        return self._finish(obs, info, messages, replies, calls, latency)

    def _finish(
        self,
        obs: np.ndarray,
        info: Mapping[str, Any],
        messages: list[dict[str, Any]],
        replies: Sequence[Reply],
        calls: list[dict[str, Any]],
        latency: float,
    ) -> np.ndarray:
        """Parse the last reply, log the decision and return the action."""
        text = replies[-1].text
        action = parse_action(text)
        failed = action is None
        if action is None:
            action = np.zeros(3)  # safe default: coast
        action = action * self.scale
        tokens_in = sum(r.input_tokens for r in replies)
        tokens_out = sum(r.output_tokens for r in replies)
        belief = parse_belief(text) if self.belief else None
        self.log.append(
            {
                "time_s": info["time_s"],
                "prompt": messages[0]["content"],
                "messages": messages,
                "tool_calls": calls,
                "reply": text,
                "action": action.tolist(),
                "parse_failed": failed,
                "model_calls": len(replies),
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "cost_usd": None if self.price is None else self.price.cost(tokens_in, tokens_out),
                "latency_s": latency,
                "belief": belief,
                "faithful": belief_matches(belief, obs) if self.belief else None,
            }
        )
        return action

    def usage(self) -> dict[str, Any]:
        """Episode totals: decisions, model calls, tokens, cost, parse failures and the fraction
        of beliefs that matched the observation (None without `belief`)."""
        tokens_in = sum(e["input_tokens"] for e in self.log)
        tokens_out = sum(e["output_tokens"] for e in self.log)
        return {
            "decisions": len(self.log),
            "model_calls": sum(e["model_calls"] for e in self.log),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost_usd": None if self.price is None else self.price.cost(tokens_in, tokens_out),
            "parse_failures": sum(e["parse_failed"] for e in self.log),
            "faithful_fraction": (
                float(np.mean([e["faithful"] for e in self.log]))
                if self.belief and self.log
                else None
            ),
        }
