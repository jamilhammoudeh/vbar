"""Plan submission: an agent proposes a whole command sequence and an external verifier checks it.

The verifier predicts the plan from the observed state with the Clohessy-Wiltshire model and the
public view only (mean motion, thrust limit and layout, goal, constraints), the information the
agent itself has, and reports the predicted constraint margins, when the goal is reached and
the delta-v. `PlanAgent` asks its planner for a plan, has it verified, returns the verdict for
revision up to `max_revisions` times, and flies the first plan that passes, open loop, asking
for a new one when it runs out. A plan that never passes is not flown: the spacecraft coasts
and the planner is asked again at the next decision.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constraints import Constraint, KeepOut, segment_minimum
from ..dynamics import CWDynamics
from .base import AgentView, action_scale
from .llm import REPLY_ERRORS, Model, Price, as_reply, first_json


@dataclass(frozen=True)
class Verdict:
    """The verifier's prediction of a plan from the observed state.

    `min_margins` are each constraint's predicted lowest margin, `capture_step` the first step
    (counted from 1) whose end meets the goal, or None; distances in m, speeds in m/s.
    """

    min_margins: Mapping[str, float]
    capture_step: int | None
    final_distance_m: float
    final_speed_mps: float
    delta_v_mps: float
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        """The verdict in words, for a planner to revise from."""
        if self.ok:
            return f"Plan accepted: goal reached at step {self.capture_step}."
        lines = ["Plan rejected:", *(f"- {p}" for p in self.problems)]
        lines.append(
            f"Predicted end: {self.final_distance_m:.1f} m from the hold point at "
            f"{self.final_speed_mps:.3f} m/s, delta-v {self.delta_v_mps:.3f} m/s."
        )
        return "\n".join(lines)


def _constraint(spec: Mapping[str, Any]) -> Constraint:
    if spec["type"] == "keep_out":
        return KeepOut(float(spec["radius"]))
    raise ValueError(f"the verifier cannot check constraints of type {spec['type']!r}")


class PlanVerifier:
    """Predicts plans of fractions of a full command, one row per step, from `view` alone."""

    samples = 20

    def __init__(self, view: AgentView):
        self.view = view
        self.constraints = tuple(_constraint(c) for c in view.constraints)
        self.hold = np.array(view.goal["hold_point"], dtype=float)

    def _limit(self, plan: np.ndarray) -> np.ndarray:
        if self.view.thrust_limit == "box":
            return np.clip(plan, -1.0, 1.0)
        length = np.linalg.norm(plan, axis=1, keepdims=True)
        return np.where(length > 1.0, plan / np.maximum(length, 1.0), plan)

    def verify(self, obs: np.ndarray, plan, steps_left: int | None = None) -> Verdict:
        """Fly `plan` from the observed state `obs` under CW and judge it."""
        view = self.view
        commands = np.asarray(plan, dtype=float).reshape(-1, 3)
        problems = []
        if len(commands) == 0:
            problems.append("the plan is empty")
        if not np.all(np.isfinite(commands)):
            problems.append("the plan has non-finite commands")
            commands = np.nan_to_num(commands, nan=0.0, posinf=0.0, neginf=0.0)
        if steps_left is not None and len(commands) > steps_left:
            problems.append(f"the plan has {len(commands)} steps but only {steps_left} remain")
            # The rest could never be flown; predicting it would only cost time.
            commands = commands[: max(steps_left, 0)]
        accels = self._limit(commands) * view.max_accel
        model = CWDynamics(view.n)
        model.reset(np.asarray(obs, dtype=float)[:6])
        times = np.linspace(0.0, view.dt, self.samples)
        lowest = {c.name: np.inf for c in self.constraints}
        entered: dict[str, int | None] = {c.name: None for c in self.constraints}
        capture = None
        state = np.asarray(obs, dtype=float)[:6]
        for k, accel in enumerate(accels, start=1):
            state, sampled = model.advance(accel, view.dt, n_samples=self.samples)
            thrust = float(np.linalg.norm(accel))
            for c in self.constraints:
                found = segment_minimum(
                    c, model.state_in_last_segment, times, np.asarray(sampled), thrust, view.n
                )
                lowest[c.name] = min(lowest[c.name], found.margin)
                if found.margin < 0.0 and entered[c.name] is None:
                    entered[c.name] = k
            distance = float(np.linalg.norm(state[:3] - self.hold))
            speed = float(np.linalg.norm(state[3:]))
            if capture is None and (
                distance < view.goal["capture_radius"] and speed < view.goal["capture_speed"]
            ):
                capture = k
                break
        for name, step in entered.items():
            if step is not None:
                problems.append(
                    f"{name} violated from step {step} (lowest margin {lowest[name]:.1f})"
                )
        if capture is None:
            problems.append("the goal is not reached within the plan")
        flown = accels[: capture or len(accels)]
        return Verdict(
            min_margins=lowest,
            capture_step=capture,
            final_distance_m=float(np.linalg.norm(state[:3] - self.hold)),
            final_speed_mps=float(np.linalg.norm(state[3:])),
            delta_v_mps=float(np.sum(np.linalg.norm(flown, axis=1)) * view.dt),
            problems=tuple(problems),
        )


Planner = Callable[[np.ndarray, AgentView, "Verdict | None", int], Any]


class PlanAgent:
    """Flies verified plans from `planner(obs, view, verdict, steps_left)`.

    A plan is rows of fractions of a full command, one per step. The planner gets the previous
    verdict when its plan was rejected (None on a first attempt) and the steps left in the
    episode. Each planning point's verifier calls and last verdict go to `planning_log`; it has no
    per-decision `log`, so the metrics do not read it as one.
    """

    def __init__(self, planner: Planner, max_revisions: int = 3, name: str = "plan"):
        self.planner = planner
        self.max_revisions = max_revisions
        self.name = name

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        self.view = view
        self.scale = action_scale(view)
        self.verifier = PlanVerifier(view)
        self.queue: deque[np.ndarray] = deque()
        self.planning_log: list[dict[str, Any]] = []

    def act(self, obs, info):
        if not self.queue:
            steps_left = self.view.max_steps - round(info["time_s"] / self.view.dt)
            verdict, calls = None, 0
            while calls <= self.max_revisions:
                plan = self.planner(obs, self.view, verdict, steps_left)
                verdict = self.verifier.verify(obs, plan, steps_left)
                calls += 1
                if verdict.ok:
                    self.queue.extend(np.asarray(plan, dtype=float).reshape(-1, 3))
                    break
            self.planning_log.append(
                {"time_s": info["time_s"], "verifier_calls": calls, "verdict": verdict}
            )
        fraction = self.queue.popleft() if self.queue else np.zeros(3)
        return fraction * self.scale


PLAN_PROMPT = """You are the guidance planner of a chaser spacecraft near a target spacecraft.

Frame (centered on the target): x = radial (up, away from Earth), y = along-track
(direction of flight), z = cross-track. Units: meters and meters per second.
Clohessy-Wiltshire dynamics apply, mean motion n = {n:.6f} rad/s.

Goal: reach the hold point {hold} m and stop there (within {cap_radius} m, speed below
{cap_speed} m/s). Constraint: NEVER come within {keep_out} m of the target (the origin).

Current state: position = [{x:.1f}, {y:.1f}, {z:.1f}] m,
velocity = [{vx:.3f}, {vy:.3f}, {vz:.3f}] m/s.
Each step lasts {dt:.0f} s; thrust per axis is a fraction of {max_accel} m/s^2, between -1 and 1.
At most {steps_left} steps remain.

Submit a plan: a JSON list of segments, each {{"ax": ..., "ay": ..., "az": ..., "steps": k}},
the fractions held for k steps. A verifier will predict it and tell you what is wrong.
{feedback}Reply with ONLY the JSON list."""


# Plans longer than this many steps are not expanded: 24 MB of rows, days of flight at any
# step the scenarios use. Shorter plans are parsed whole, so the verifier can report their length.
MAX_PLAN_ROWS = 1_000_000


def parse_plan(text: str, max_rows: int = MAX_PLAN_ROWS) -> np.ndarray:
    """Rows of fractions from a reply's JSON list of segments.

    Empty when the reply cannot be read, a segment's steps are not a whole number of at least
    zero, or the plan has more than `max_rows` steps.
    """
    segments = first_json(text, r"\[.*\]")
    empty = np.zeros((0, 3))
    try:
        parts, total = [], 0
        for s in segments:
            if isinstance(s["steps"], bool):
                return empty
            steps = float(s["steps"])
            if not steps.is_integer() or steps < 0:
                return empty
            total += int(steps)
            if total > max_rows:
                return empty
            parts.append(([float(s["ax"]), float(s["ay"]), float(s["az"])], int(steps)))
    except REPLY_ERRORS:
        return empty
    if not parts:
        return empty
    commands = np.array([command for command, _ in parts], dtype=float)
    return np.repeat(commands, [steps for _, steps in parts], axis=0)


class LLMPlanner:
    """A planner that asks `complete(messages, tools)` for a plan and revises from the verdict.

    Every call's prompt, reply, tokens, cost and latency go to `log`.
    """

    def __init__(self, complete: Model, price: Price | None = None):
        self.complete = complete
        self.price = price
        self.log: list[dict[str, Any]] = []

    def __call__(
        self, obs: np.ndarray, view: AgentView, verdict: Verdict | None, steps_left: int
    ) -> np.ndarray:
        keep_out = next(c["radius"] for c in view.constraints if c["type"] == "keep_out")
        x, y, z, vx, vy, vz = obs[:6]
        feedback = "" if verdict is None else f"\nYour previous plan:\n{verdict.summary()}\n\n"
        prompt = PLAN_PROMPT.format(
            n=view.n,
            hold=list(view.goal["hold_point"]),
            cap_radius=view.goal["capture_radius"],
            cap_speed=view.goal["capture_speed"],
            keep_out=keep_out,
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            dt=view.dt,
            max_accel=view.max_accel,
            steps_left=steps_left,
            feedback=feedback,
        )
        t0 = time.perf_counter()
        reply = as_reply(self.complete([{"role": "user", "content": prompt}], []))
        latency = time.perf_counter() - t0
        tokens = (reply.input_tokens, reply.output_tokens)
        self.log.append(
            {
                "prompt": prompt,
                "reply": reply.text,
                "input_tokens": tokens[0],
                "output_tokens": tokens[1],
                "cost_usd": None if self.price is None else self.price.cost(*tokens),
                "latency_s": latency,
            }
        )
        # Over-long plans are parsed, so the verifier can say they are too long.
        return parse_plan(reply.text)
