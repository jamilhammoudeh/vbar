"""LLM agent: asks a language model for a thrust command at each decision step.

The model is passed in as `complete(messages, tools) -> Reply | str`, so any hosted API or local
model works through a small adapter; `text_model` wraps a plain `prompt -> text` function.
`messages` is the decision's conversation as provider-neutral dicts with a "role" ("user",
"assistant" or "tool") and a "content" string; `tools` lists the JSON-schema descriptions of the
tools offered. A reply may call tools; the agent runs them on its own observation, adds the
results to the conversation and asks again, at most `max_tool_rounds` times, then asks once more
without tools. Each decision's messages, tool calls, reply, parsed action, parse failure, token
counts, cost and wall-clock latency are stored in `self.log`.

The simulation clock is paused during the model call unless the run is in real time. Every
number in the prompt comes from the observation or the scenario's public description, never
from the true state.
"""

from __future__ import annotations

import json
import re
import string
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

import numpy as np

from .base import AgentView, action_scale
from .tools import Tool, call_tool

PROMPT_TEMPLATE = """You are the guidance system of a chaser spacecraft near a target spacecraft.

Frame (centered on the target): x = radial (up, away from Earth), y = along-track
(direction of flight), z = cross-track. Units: meters and meters per second.

Goal: reach the hold point {hold} m and stop there (speed below {cap_speed} m/s).
Constraint: NEVER come within {keep_out} m of the target (the origin).
Orbital mechanics apply (Clohessy-Wiltshire dynamics, mean motion n = {n:.6f} rad/s).

Current state (t = {t:.0f} s):
  position = [{x:.1f}, {y:.1f}, {z:.1f}] m
  velocity = [{vx:.3f}, {vy:.3f}, {vz:.3f}] m/s
  distance to hold point = {dist:.1f} m, distance to target = {rng:.1f} m

Each command is held for {dt:.0f} s. Thrust per axis is a fraction of the maximum
({max_accel} m/s^2), between -1 and 1.

{reply_format}"""

REPLY_FORMAT = 'Reply with ONLY a JSON object, for example: {"ax": 0.1, "ay": -0.5, "az": 0.0}'

# Fields a prompt template may use.
PROMPT_FIELDS = (
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


def text_model(complete: Callable[[str], str]) -> Model:
    """Adapt a `prompt -> text` function: it receives the last message's content."""

    def model(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        return complete(messages[-1]["content"])

    return model


def as_reply(answer: Reply | str) -> Reply:
    return answer if isinstance(answer, Reply) else Reply(text=str(answer))


def parse_action(text: str) -> np.ndarray | None:
    """Extract {"ax":..,"ay":..,"az":..} from a model reply. Returns None on failure."""
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        a = np.array([float(data["ax"]), float(data["ay"]), float(data["az"])])
    except (ValueError, KeyError, TypeError):
        return None
    if not np.all(np.isfinite(a)):
        return None
    return np.clip(a, -1.0, 1.0)


@dataclass
class LLMAgent:
    """Agent that asks `complete(messages, tools)` for each command.

    `prompt` is a `str.format` template over `PROMPT_FIELDS`; `price` turns the reported tokens
    into dollars; `tools` are offered to the model for at most `max_tool_rounds` rounds of calls
    per decision. An unparseable reply is logged and the spacecraft coasts.
    """

    complete: Model
    _: KW_ONLY
    name: str = "llm"
    prompt: str = PROMPT_TEMPLATE
    price: Price | None = None
    tools: Sequence[Tool] = ()
    max_tool_rounds: int = 4
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        used = {f for _, f, _, _ in string.Formatter().parse(self.prompt) if f}
        unknown = used - set(PROMPT_FIELDS)
        if unknown:
            raise ValueError(f"unknown prompt fields {sorted(unknown)}, expected {PROMPT_FIELDS}")

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        self.view = view
        self.scale = action_scale(view)
        self.log = []

    def prompt_text(self, obs: np.ndarray, info: Mapping[str, Any]) -> str:
        """The prompt for observation `obs`, from the observation and the public view only."""
        view = self.view
        hold = np.array(view.goal["hold_point"], dtype=float)
        keep_out = next(c["radius"] for c in view.constraints if c["type"] == "keep_out")
        x, y, z, vx, vy, vz = obs[:6]
        return self.prompt.format(
            hold=hold.tolist(),
            cap_speed=view.goal["capture_speed"],
            keep_out=keep_out,
            n=view.n,
            t=info["time_s"],
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            dist=float(np.linalg.norm(obs[:3] - hold)),
            rng=float(np.linalg.norm(obs[:3])),
            dt=view.dt,
            max_accel=view.max_accel,
            reply_format=self.reply_format(),
        )

    def reply_format(self) -> str:
        return REPLY_FORMAT

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
                        {"id": c.id, "name": c.name, "arguments": dict(c.arguments)}
                        for c in reply.tool_calls
                    ],
                }
            )
            for c in reply.tool_calls:
                result = call_tool(offered, c.name, obs, self.view, c.arguments)
                calls.append({"name": c.name, "arguments": dict(c.arguments), "result": result})
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
            }
        )
        return action

    def usage(self) -> dict[str, Any]:
        """Totals over the episode: decisions, model calls, tokens, cost and parse failures."""
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
