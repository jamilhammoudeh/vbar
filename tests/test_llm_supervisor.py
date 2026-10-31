"""The LLM supervisor: the reply it accepts, the prompt it builds from a situation, and the
supervision steps of whole flights when a stand-in model answers from the prompt text alone.

Besides the briefing and the situation lines of `vbar.agents.briefing`, the tests rely on this
layout of the prompt:

- the frame line names the axes ("x = radial", "y = along-track", "z = cross-track");
- the ladder is a list of numbered lines ("1." or "1)") that hold the rung's three coordinates;
- one line gives the phase ("phase: holding") and one the number of the current rung as the ladder
  numbers it ("rung 2"); "t = 60 s" gives the time;
- the line "Commands you may give now" is followed by the valid commands, up to the next blank line,
  and each unavailable command then starts a line of its own that carries its reason.

Replies are read case-insensitively, a command by its name only (a number or a different spelling
is not a command), and the first object in the reply counts. Whole flights are marked `slow`.
"""

import dataclasses
import json
import math
import re
from types import MappingProxyType

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from supervision_support import Flight, plain_step

from vbar.agents import agent_generator
from vbar.agents.briefing import TASK_KEYS, briefing, situation
from vbar.agents.llm import MAX_JSON_DEPTH, Price, RecordedModel, ReplayError, ReplayModel, Reply
from vbar.agents.llm_supervisor import (
    MAX_REPLY_CHARS,
    SUPERVISOR_PROMPT,
    LLMSupervisor,
    parse_order,
)
from vbar.agents.rules import RuleSupervisor
from vbar.agents.supervision import PHASES, Command, Order, Situation, SupervisedAgent, ladder
from vbar.scenarios import InertialPointing, Misbriefed, get, make, seeds
from vbar.scenarios.registry import entry_class

BASE = ("approach-vbar", "approach-rbar", "docking:axis", "docking-afrl", "inspection")
MODES = ("ladder", "waypoint", "velocity")
NAMES = [c.name for c in Command]
GO, HOLD, RETREAT, CAM, INHIBIT = (
    Command.GO,
    Command.HOLD,
    Command.RETREAT,
    Command.CAM,
    Command.THRUST_INHIBIT,
)

# A GO needs a destination in the modes that have one.
DESTINATION = {
    "ladder": {},
    "waypoint": {"point": [0.0, -100.0, 0.0], "speed": 0.3},
    "velocity": {"velocity": [0.0, 0.1, 0.0]},
}

# Observations (Hill, m and m/s) on the approach-vbar ladder: rung 0 is (0, -200, 0), rung 1
# (0, -100, 0), the commit radius of the second view below is 30 m.
FAR = (0.0, -300.0, 0.0, 0.0, 0.0, 0.0)
R0 = (0.0, -200.0, 0.0, 0.0, 0.0, 0.0)
INSIDE = (0.0, -20.0, 0.0, 0.0, 0.0, 0.0)
APPROACH = make("approach-vbar").agent_view()
COMMIT = dataclasses.replace(APPROACH, goal={**APPROACH.goal, "commit_radius": 30.0})


def reply_of(command, **fields):
    return json.dumps({"command": command, **fields})


# parse_order


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("command", list(Command), ids=lambda c: c.name)
def test_every_command_name_parses_in_every_mode(command, mode):
    extra = DESTINATION[mode] if command == GO else {}
    order = parse_order(reply_of(command.name, **extra), mode)
    assert isinstance(order, Order) and order.command == command
    assert order.unit is None and order.reason == ""


@pytest.mark.parametrize(
    ("text", "command"),
    [
        ('{"command": "go"}', GO),
        ('{"command": "Go"}', GO),
        ('{"command": "  GO\\n"}', GO),
        ('{"command": "thrust_inhibit"}', INHIBIT),
        ('{"command": "Thrust_Inhibit"}', INHIBIT),
        ('{"command": "cam"}', CAM),
    ],
)
def test_command_names_are_read_case_insensitively_and_trimmed(text, command):
    assert parse_order(text, "ladder").command == command


@pytest.mark.parametrize(
    "name",
    [
        "",
        "PROCEED",
        "THRUST INHIBIT",
        "THRUST-INHIBIT",
        "GO AHEAD",
        "GO NOW",
        "ABORT",
        "GOGO",
        "G O",
    ],
)
def test_an_unknown_command_name_is_not_an_order(name):
    assert parse_order(reply_of(name), "ladder") is None


@pytest.mark.parametrize("value", [1, 0, 2.0, True, None, ["GO"], {"name": "GO"}])
def test_a_command_that_is_not_a_name_is_not_an_order(value):
    assert parse_order(json.dumps({"command": value}), "ladder") is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "GO",
        "go ahead to the next rung",
        '"GO"',
        '["GO"]',
        "42",
        "{}",
        '{"cmd": "GO"}',
        '{"command": "GO"',
        '{"command": GO}',
        "{'command': 'GO'}",
    ],
)
def test_a_reply_without_a_command_object_is_not_an_order(text):
    assert parse_order(text, "ladder") is None


@pytest.mark.parametrize(
    "text",
    [
        'The chaser has settled, so I will go ahead. {"command": "GO"} Hope that helps.',
        '```json\n{"command": "GO"}\n```',
        'Thinking about the rung...\n\n{"command": "GO", "reason": "settled"}',
        '{"command": "GO"}\n\nExplanation: the arrival was held for a minute.',
    ],
)
def test_prose_around_the_json_object_is_ignored(text):
    assert parse_order(text, "ladder").command == GO


def test_the_first_object_in_a_reply_is_the_one_that_counts():
    order = parse_order('{"command": "HOLD"} and then {"command": "GO"}', "ladder")
    assert order.command == HOLD


def test_waypoint_mode_reads_the_point_and_the_speed():
    order = parse_order(reply_of("GO", point=[10, -100, 5.5], speed=0.25), "waypoint")
    assert tuple(order.point) == (10.0, -100.0, 5.5) and order.speed == 0.25
    assert order.velocity is None


def test_velocity_mode_reads_the_velocity():
    order = parse_order(reply_of("GO", velocity=[0, 0.1, -0.05]), "velocity")
    assert tuple(order.velocity) == (0.0, 0.1, -0.05)
    assert order.point is None and order.speed is None


def test_a_reason_is_kept_and_defaults_to_empty():
    order = parse_order(reply_of("HOLD", reason="closing too fast"), "ladder")
    assert order.command == HOLD and order.reason == "closing too fast"
    assert parse_order(reply_of("HOLD"), "ladder").reason == ""


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("reason", [3, 2.5, True, ["closing", "fast"], None], ids=repr)
def test_a_reason_that_is_not_a_string_gives_an_order_with_an_empty_reason(reason, mode):
    extra = DESTINATION[mode] if mode != "ladder" else {}
    order = parse_order(reply_of("GO", reason=reason, **extra), mode)
    assert isinstance(order, Order) and order.command == GO
    assert order.reason == ""


def test_a_reason_that_is_an_object_gives_an_order_with_an_empty_reason():
    order = parse_order(reply_of("HOLD", reason={"why": "closing"}), "ladder")
    assert isinstance(order, Order) and order.command == HOLD
    assert order.reason == ""


def test_fields_that_belong_to_another_mode_are_ignored():
    everything = {"point": [0, -100, 0], "speed": 0.3, "velocity": [0, 0.1, 0]}
    ladder_order = parse_order(reply_of("GO", **everything), "ladder")
    assert ladder_order.command == GO
    assert (ladder_order.point, ladder_order.velocity, ladder_order.speed) == (None, None, None)
    waypoint_order = parse_order(reply_of("GO", **everything), "waypoint")
    assert tuple(waypoint_order.point) == (0.0, -100.0, 0.0) and waypoint_order.velocity is None
    velocity_order = parse_order(reply_of("GO", **everything), "velocity")
    assert tuple(velocity_order.velocity) == (0.0, 0.1, 0.0)
    assert velocity_order.point is None and velocity_order.speed is None
    held = parse_order(reply_of("HOLD", point=[0, -100, 0], speed=0.3), "waypoint")
    assert held.command == HOLD and held.point is None and held.speed is None


GOOD_POINT, GOOD_SPEED = [0.0, -100.0, 0.0], 0.3


@pytest.mark.parametrize(
    "fields",
    [
        {"speed": GOOD_SPEED},
        {"point": GOOD_POINT},
        {},
        {"point": [0.0, -100.0], "speed": GOOD_SPEED},
        {"point": [0.0, -100.0, 0.0, 0.0], "speed": GOOD_SPEED},
        {"point": ["a", "b", "c"], "speed": GOOD_SPEED},
        {"point": ["0", "-100", "0"], "speed": GOOD_SPEED},
        {"point": [True, 0, 0], "speed": GOOD_SPEED},
        {"point": [None, 0, 0], "speed": GOOD_SPEED},
        {"point": [[0], [-100], [0]], "speed": GOOD_SPEED},
        {"point": {"x": 0, "y": -100, "z": 0}, "speed": GOOD_SPEED},
        {"point": -100.0, "speed": GOOD_SPEED},
        {"point": None, "speed": GOOD_SPEED},
        {"point": GOOD_POINT, "speed": "0.3"},
        {"point": GOOD_POINT, "speed": True},
        {"point": GOOD_POINT, "speed": [0.3]},
        {"point": GOOD_POINT, "speed": None},
    ],
)
def test_a_waypoint_go_without_a_well_formed_point_and_speed_is_not_an_order(fields):
    assert parse_order(reply_of("GO", **fields), "waypoint") is None


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"velocity": [0.0, 0.1]},
        {"velocity": [0.0, 0.1, 0.0, 0.0]},
        {"velocity": ["0", "0.1", "0"]},
        {"velocity": [False, 0, 0]},
        {"velocity": 0.1},
        {"velocity": None},
        {"point": GOOD_POINT, "speed": GOOD_SPEED},
    ],
)
def test_a_velocity_go_without_a_well_formed_velocity_is_not_an_order(fields):
    assert parse_order(reply_of("GO", **fields), "velocity") is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_number_makes_the_reply_not_an_order(bad):
    assert parse_order(reply_of("GO", point=[bad, 0.0, 0.0], speed=0.3), "waypoint") is None
    assert parse_order(reply_of("GO", point=[0.0, 0.0, 0.0], speed=bad), "waypoint") is None
    assert parse_order(reply_of("GO", velocity=[0.0, bad, 0.0]), "velocity") is None


@pytest.mark.parametrize("literal", ["1e999", "-1e999", "10" + "0" * 400])
def test_a_number_too_large_for_a_float_makes_the_reply_not_an_order(literal):
    text = '{"command": "GO", "point": [0, ' + literal + ', 0], "speed": 0.3}'
    assert parse_order(text, "waypoint") is None
    text = '{"command": "GO", "point": [0, 0, 0], "speed": ' + literal + "}"
    assert parse_order(text, "waypoint") is None


def test_a_destination_the_mode_cannot_fly_is_left_for_the_agent_to_refuse():
    # The agent checks speed against its tracker and the point against the keep-out sphere and
    # records the refusal; the parser has neither.
    order = parse_order(reply_of("GO", point=[0.0, 0.0, 0.0], speed=99.0), "waypoint")
    assert order.speed == 99.0 and tuple(order.point) == (0.0, 0.0, 0.0)
    order = parse_order(reply_of("GO", point=GOOD_POINT, speed=-1.0), "waypoint")
    assert order is not None and order.speed == -1.0


@pytest.mark.parametrize("mode", MODES)
def test_an_over_long_reply_does_not_raise(mode):
    assert parse_order("x" * 2_000_000, mode) is None
    assert parse_order("{" * 5_000 + "}", mode) is None
    nested = '{"command": "GO", "reason": ' + "[" * 5_000 + "]" * 5_000 + "}"
    padded = "I considered every option. " * 20_000 + reply_of("CONTINUE")
    long_reason = reply_of("CONTINUE", reason="because " * 100_000)
    for text in (nested, padded, long_reason):
        order = parse_order(text, mode)
        assert order is None or isinstance(order, Order)


JSON_SCALARS = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=True, allow_infinity=True)
    | st.text(max_size=12)
)
JSON_VALUES = st.recursive(
    JSON_SCALARS,
    lambda inner: (
        st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=6), inner, max_size=3)
    ),
    max_leaves=10,
)
REPLY_FIELDS = st.fixed_dictionaries(
    {},
    optional={
        "command": st.sampled_from(NAMES + [n.lower() for n in NAMES]) | JSON_VALUES,
        "point": JSON_VALUES,
        "speed": JSON_VALUES,
        "velocity": JSON_VALUES,
        "reason": JSON_VALUES,
    },
)
REPLIES = st.text(max_size=200) | st.tuples(
    st.text(max_size=20), REPLY_FIELDS.map(json.dumps), st.text(max_size=20)
).map("".join)


@given(text=REPLIES, mode=st.sampled_from(MODES))
def test_parse_order_never_raises_and_returns_only_orders_it_could_fly(text, mode):
    order = parse_order(text, mode)
    if order is None:
        return
    assert isinstance(order, Order) and isinstance(order.command, Command)
    assert isinstance(order.reason, str)
    for vector in (order.point, order.velocity):
        if vector is not None:
            assert len(vector) == 3 and all(math.isfinite(float(c)) for c in vector)
    if order.speed is not None:
        assert math.isfinite(float(order.speed))
    if order.command == GO and mode == "waypoint":
        assert order.point is not None and order.speed is not None
    if order.command == GO and mode == "velocity":
        assert order.velocity is not None


# The prompt


class Replies:
    """A stand-in model that answers from a list in turn (the last repeats) and keeps its
    requests."""

    def __init__(self, *replies):
        self.replies, self.requests = replies, []

    def __call__(self, messages, tools):
        self.requests.append(([dict(m) for m in messages], list(tools)))
        return self.replies[min(len(self.requests), len(self.replies)) - 1]

    @property
    def prompts(self):
        return [messages[-1]["content"] for messages, _ in self.requests]


class Capture(Replies):
    """A stand-in model that always gives one reply, CONTINUE unless told otherwise."""

    def __init__(self, text=None, tokens=(0, 0)):
        super().__init__(Reply(reply_of("CONTINUE") if text is None else text, *tokens))


_scenario_cache = {}


def scenario(name):
    """The public view and first observation of a base scenario on its first validation seed."""
    if name not in _scenario_cache:
        env = make(name)
        seed, _ = seeds(name, "validation", 1).pairs()[0]
        obs, _ = env.reset(seed=seed)
        _scenario_cache[name] = (env.agent_view(), obs)
    return _scenario_cache[name]


def a_situation(
    view, rungs, obs, *, t=0.0, phase="transfer", rung=0, valid=(), refused=None, info=None
):
    """A situation with the given commands valid (CONTINUE always is) and a reason for the rest."""
    mask = tuple(c == Command.CONTINUE or c in valid for c in Command)
    reasons = {c: f"{c.name.lower()} refused by rule 42" for c in Command if not mask[c]}
    return Situation(
        t=t,
        obs=np.array(obs, dtype=float),
        phase=phase,
        rung=rung,
        ladder=rungs,
        mask=mask,
        reasons=MappingProxyType(reasons),
        refused=refused,
        info=MappingProxyType(info if info is not None else {"time_s": t}),
    )


def consult(view, rungs, sit, model=None, **options):
    """The prompt the supervisor sends for one situation, and what it decided."""
    model = Capture() if model is None else model
    supervisor = LLMSupervisor(model, **options)
    supervisor.reset(view, rungs, None)
    answer = supervisor.decide(sit)
    return model.prompts[-1], answer


def order_of(answer):
    return Order(answer) if isinstance(answer, Command) else answer


COMMAND_LINE = re.compile(r"^\s*(?:[-*]\s+)?(" + "|".join(NAMES) + r")\b(.*)$")
LADDER_LINE = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def body(prompt, view):
    """The prompt without the briefing, which holds numbered and lettered lines of its own."""
    return prompt.replace(briefing(view), "", 1)


def read_commands(text):
    """The valid commands and, name to line, the unavailable ones, as the prompt lists them."""
    lines = text.splitlines()
    heading = "Commands you may give now"
    start = next(i for i, line in enumerate(lines) if heading in line)
    valid = set(re.findall(r"\b(" + "|".join(NAMES) + r")\b", lines[start].split(heading, 1)[1]))
    k = start + 1
    while k < len(lines) and lines[k].strip():
        valid |= set(re.findall(r"\b(" + "|".join(NAMES) + r")\b", lines[k]))
        k += 1
    unavailable = {}
    for line in lines[k:]:
        match = COMMAND_LINE.match(line)
        if match:
            unavailable[match.group(1)] = line
    return valid, unavailable


def read_ladder(text):
    """The numbers and the text of the ladder's numbered lines."""
    return [
        (int(m.group(1)), m.group(2))
        for line in text.splitlines()
        if (m := LADDER_LINE.match(line))
    ]


def other_lines(text):
    return "\n".join(line for line in text.splitlines() if not LADDER_LINE.match(line))


def read_phase(text):
    found = re.findall(
        r"\bphase\b[^\n]*?\b(" + "|".join(PHASES) + r")\b", other_lines(text), flags=re.I
    )
    return found[-1].lower()


def read_rung_number(text):
    return int(re.findall(r"\brung\b[^\d\n]{0,20}?(\d+)", other_lines(text), flags=re.I)[-1])


def read_time(text):
    return float(re.findall(r"\bt = (\d+(?:\.\d+)?) s", text)[-1])


def read_rung_index(text):
    """Index in the ladder of the rung the prompt names, whatever the first number is."""
    return [number for number, _ in read_ladder(text)].index(read_rung_number(text))


def has_run(values, target, atol):
    return any(
        all(abs(values[i + j] - target[j]) <= atol for j in range(3))
        for i in range(len(values) - 2)
    )


@pytest.mark.parametrize("name", BASE)
def test_the_prompt_holds_the_frame_the_briefing_the_ladder_and_the_situation_lines(name):
    view, obs0 = scenario(name)
    rungs = ladder(view, obs0)
    info = {"time_s": 120.0, "coverage": 0.25}
    prompt, _ = consult(view, rungs, a_situation(view, rungs, obs0, t=120.0, info=info))
    assert all(
        f"{axis} = {word}" in prompt
        for axis, word in zip("xyz", ("radial", "along-track", "cross-track"), strict=True)
    )
    assert briefing(view) in prompt
    text = body(prompt, view)
    for line in situation(view, obs0, info).splitlines():
        assert line.strip() in text
    numbered = read_ladder(text)
    assert [n - numbered[0][0] for n, _ in numbered] == list(range(len(rungs)))
    for (_, line), rung in zip(numbered, rungs, strict=True):
        assert has_run([float(x) for x in NUMBER.findall(line)], rung.position, 0.06)
    assert read_time(text) == 120.0
    assert read_phase(text) == "transfer" and read_rung_index(text) == 0


def test_the_prompt_names_the_phase_and_the_rung_of_the_situation():
    view, obs0 = scenario("approach-vbar")
    rungs = ladder(view, obs0)
    for phase in PHASES:
        for rung in range(len(rungs)):
            sit = a_situation(view, rungs, obs0, phase=phase, rung=rung)
            text = body(consult(view, rungs, sit)[0], view)
            assert read_phase(text) == phase and read_rung_index(text) == rung


def test_the_prompt_asks_for_a_json_command_and_the_model_gets_one_message_and_no_tools():
    view, obs0 = scenario("approach-vbar")
    rungs = ladder(view, obs0)
    model = Capture()
    consult(view, rungs, a_situation(view, rungs, obs0), model)
    ((messages, tools),) = model.requests
    assert tools == [] and [m["role"] for m in messages] == ["user"]
    assert '"command"' in messages[0]["content"]


VALID_SETS = [
    (),
    (GO,),
    (RETREAT,),
    (INHIBIT,),
    (HOLD, RETREAT, CAM, INHIBIT),
    (GO, HOLD, RETREAT, CAM, INHIBIT),
]


@pytest.mark.parametrize("valid", VALID_SETS, ids=lambda v: "+".join(c.name for c in v) or "none")
def test_the_prompt_lists_exactly_the_valid_commands_and_each_other_with_its_reason(valid):
    view, obs0 = scenario("approach-vbar")
    rungs = ladder(view, obs0)
    sit = a_situation(view, rungs, obs0, valid=valid)
    text = body(consult(view, rungs, sit)[0], view)
    listed, unavailable = read_commands(text)
    allowed = {Command.CONTINUE, *valid}
    assert listed == {c.name for c in allowed}
    assert set(unavailable) == {c.name for c in Command if c not in allowed}
    for name, line in unavailable.items():
        assert sit.reasons[Command[name]] in line
    assert Command.RECONFIGURE.name in unavailable
    assert "rule 42" not in text.split("Commands you may give now", 1)[1].split("\n\n", 1)[0]


def test_the_last_refusal_is_in_the_next_prompt_only():
    view, obs0 = scenario("approach-vbar")
    rungs = ladder(view, obs0)
    refusal = "GO: arrival at the rung not held"
    model = Capture()
    supervisor = LLMSupervisor(model)
    supervisor.reset(view, rungs, None)
    supervisor.decide(a_situation(view, rungs, obs0, refused=refusal))
    supervisor.decide(a_situation(view, rungs, obs0, refused=None))
    assert refusal in model.prompts[0]
    assert refusal not in model.prompts[1]


@pytest.mark.parametrize(
    ("view", "observations"),
    [
        (APPROACH, [FAR, FAR, R0, R0, R0, FAR]),
        (COMMIT, [FAR, R0, INSIDE, INSIDE]),
    ],
    ids=["ladder", "commit"],
)
def test_the_prompt_of_a_flown_agent_lists_the_commands_its_mask_allows(view, observations):
    model = Capture()
    agent = SupervisedAgent(LLMSupervisor(model), period_s=view.dt)
    agent.reset(view, None)
    phases = []
    for k, obs in enumerate(observations):
        agent.act(np.array(obs), {"time_s": k * view.dt})
        step = agent.last_step
        text = body(model.prompts[-1], view)
        assert read_commands(text)[0] == {c.name for c in Command if step.mask[c]}
        assert read_phase(text) == step.phase and read_rung_index(text) == step.rung
        assert read_time(text) == k * view.dt
        phases.append(step.phase)
    assert len(model.prompts) == len(observations)
    assert len(set(phases)) >= 2


def test_a_refused_order_reaches_the_next_prompt_of_a_flown_agent():
    model = Replies(Reply(reply_of("GO")), Reply(reply_of("CONTINUE")), Reply(reply_of("CONTINUE")))
    agent = SupervisedAgent(LLMSupervisor(model), period_s=APPROACH.dt)
    agent.reset(APPROACH, None)
    for k in range(3):
        agent.act(np.array(FAR), {"time_s": k * APPROACH.dt})
        if k == 0:
            assert agent.last_step.accepted is False
            refusal = agent.last_step.reason
    # The refused command also stays in the list of unavailable ones, so the count tells them apart.
    assert refusal
    assert model.prompts[1].count(refusal) == model.prompts[2].count(refusal) + 1


def test_a_goal_without_a_key_its_task_needs_is_refused_at_reset():
    view, obs0 = scenario("approach-vbar")
    key = TASK_KEYS[view.goal["task"]][0]
    goal = {k: v for k, v in view.goal.items() if k != key}
    broken = dataclasses.replace(view, goal=goal)
    with pytest.raises(ValueError, match=key):
        LLMSupervisor(Capture()).reset(broken, ladder(view, obs0), None)


# Log and usage


def test_usage_counts_decisions_tokens_cost_and_parse_failures():
    view, obs0 = scenario("approach-vbar")
    rungs = ladder(view, obs0)
    model = Replies(
        Reply(reply_of("CONTINUE", reason="waiting for the rung"), 100, 10),
        Reply("no idea what to do", 50, 5),
        Reply(reply_of("CONTINUE"), 100, 10),
    )
    supervisor = LLMSupervisor(model, price=Price(3.0, 15.0))
    supervisor.reset(view, rungs, None)
    assert supervisor.usage()["decisions"] == 0 and supervisor.usage()["parse_failures"] == 0
    for k in range(3):
        supervisor.decide(a_situation(view, rungs, obs0, t=10.0 * k))
    usage = supervisor.usage()
    assert usage["decisions"] == usage["model_calls"] == 3
    assert (usage["input_tokens"], usage["output_tokens"]) == (250, 25)
    assert usage["parse_failures"] == 1
    assert usage["cost_usd"] == pytest.approx((250 * 3.0 + 25 * 15.0) / 1e6)
    assert [e["parse_failed"] for e in supervisor.log] == [False, True, False]
    assert [e["time_s"] for e in supervisor.log] == [0.0, 10.0, 20.0]
    assert [e["prompt"] for e in supervisor.log] == model.prompts
    assert supervisor.log[1]["reply"] == "no idea what to do"
    assert supervisor.log[0]["reason"] == "waiting for the rung"
    supervisor.reset(view, rungs, None)
    assert supervisor.usage()["decisions"] == 0 and supervisor.log == []
    assert LLMSupervisor(Capture()).usage()["cost_usd"] is None


def test_an_unparseable_reply_is_continue():
    view, obs0 = scenario("approach-vbar")
    rungs = ladder(view, obs0)
    model = Capture("thrusting toward the target")
    _, answer = consult(view, rungs, a_situation(view, rungs, obs0, valid=(GO,)), model)
    assert order_of(answer).command == Command.CONTINUE


def test_an_unparseable_reply_leaves_the_tracker_reference_and_is_counted():
    outcomes = {}
    for label, text in (("garbled", "go to the next rung please"), ("valid", reply_of("GO"))):
        model = Replies(Reply(reply_of("CONTINUE")), Reply(text), Reply(reply_of("CONTINUE")))
        supervisor = LLMSupervisor(model)
        agent = SupervisedAgent(supervisor, period_s=APPROACH.dt)
        agent.reset(APPROACH, None)
        for k, obs in enumerate((FAR, R0)):
            agent.act(np.array(obs), {"time_s": k * APPROACH.dt})
        outcomes[label] = (agent.last_step, supervisor.usage()["parse_failures"])
    step, failures = outcomes["garbled"]
    assert failures == 1
    assert step.phase == "holding" and step.rung == 0 and step.order.command == Command.CONTINUE
    assert np.allclose(step.r_ref, R0[:3])
    # The same state with a valid GO moves the reference to the next rung, so the garbled reply
    # is what held it back.
    go_step, go_failures = outcomes["valid"]
    assert go_failures == 0 and go_step.rung == 1
    assert np.allclose(go_step.r_ref, [0.0, -100.0, 0.0])


def test_a_reply_longer_than_the_bound_is_a_parse_failure():
    # The bound keeps a hostile reply of many braces from blocking each consult for minutes.
    valid = reply_of("GO")
    at_bound = valid + " " * (MAX_REPLY_CHARS - len(valid))
    assert parse_order(at_bound, "ladder").command == Command.GO
    assert parse_order(at_bound + " ", "ladder") is None
    for text, failures in ((at_bound, 0), (at_bound + " ", 1)):
        supervisor = LLMSupervisor(Replies(Reply(text)))
        agent = SupervisedAgent(supervisor, period_s=APPROACH.dt)
        agent.reset(APPROACH, None)
        agent.act(np.array(FAR), {"time_s": 0.0})
        assert supervisor.usage()["parse_failures"] == failures
        assert supervisor.log[-1]["parse_failed"] == bool(failures)


# Flights with a stand-in model


class PromptRules:
    """A stand-in model that reads the time, phase, rung, range, closing speed and the valid
    commands from the prompt text and answers with the command RuleSupervisor gives for them.

    The rules read the public view, as the supervisor does; every number that changes during a
    flight is only what the prompt printed, rounded as printed.
    """

    def __init__(self, view, obs0, tokens=(100, 8)):
        self.view, self.tokens = view, tokens
        self.rungs = ladder(view, obs0)
        self.rules = RuleSupervisor()
        self.rules.reset(view, self.rungs)

    def __call__(self, messages, tools):
        text = body(messages[-1]["content"], self.view)
        valid, _ = read_commands(text)
        range_m, closing = map(
            float,
            re.search(
                r"distance to the target = (\S+) m, closing speed = (\S+) m/s", text
            ).groups(),
        )
        sit = Situation(
            t=read_time(text),
            obs=np.array([range_m, 0.0, 0.0, -closing, 0.0, 0.0]),
            phase=read_phase(text),
            rung=read_rung_index(text),
            ladder=self.rungs,
            mask=tuple(c.name in valid for c in Command),
            reasons=MappingProxyType({}),
            refused=None,
            info=MappingProxyType({}),
        )
        return Reply(reply_of(self.rules.decide(sit).name), *self.tokens)


def projection(step):
    """A supervision step as plain values, with the order reduced to its command."""
    plain = list(plain_step(step))
    plain[2] = None if step.order is None else Command(step.order.command)
    return tuple(plain)


def fly(name, supervisor_for, *, steps=None, **config):
    """One episode on the first validation seed under a supervised agent; its step records."""
    env = make(name, **config)
    seed, agent_seed = seeds(name, "validation", 1).pairs()[0]
    obs, _ = env.reset(seed=seed)
    view = env.agent_view()
    agent = SupervisedAgent(supervisor_for(view, obs))
    agent.reset(view, agent_generator(agent_seed))
    records, actions = [], []
    while steps is None or len(records) < steps:
        action = agent.act(obs, env.agent_info())
        records.append(projection(agent.last_step))
        actions.append(np.array(action))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return agent, records, np.array(actions)


def rules_supervisor(view, obs0):
    return RuleSupervisor()


def prompt_reading_supervisor(view, obs0):
    return LLMSupervisor(PromptRules(view, obs0))


def check_same_orders(name, steps=None, **config):
    rules, expected, expected_actions = fly(name, rules_supervisor, steps=steps, **config)
    llm, got, got_actions = fly(name, prompt_reading_supervisor, steps=steps, **config)
    consults = [r for r in got if r[1]]
    assert llm.supervisor.usage()["parse_failures"] == 0
    assert llm.supervisor.usage()["decisions"] == len(consults) > 3
    assert got == expected
    assert np.array_equal(got_actions, expected_actions)
    given = [r[2] for r in got if r[1] and r[2] != Command.CONTINUE]
    assert given, "the rules gave only CONTINUE, so the comparison proved nothing"


@pytest.mark.slow
def test_a_stand_in_reading_only_the_prompt_gives_the_rule_supervisors_orders_early_in_a_flight():
    check_same_orders("approach-vbar", steps=170)


@pytest.mark.slow
@pytest.mark.parametrize("name", BASE)
def test_a_stand_in_reading_only_the_prompt_gives_the_rule_supervisors_orders_on_a_drawn_seed(name):
    check_same_orders(name)


def recording_supervisor(holder):
    def build(view, obs0):
        holder["model"] = RecordedModel(PromptRules(view, obs0))
        return LLMSupervisor(holder["model"])

    return build


def replaying_supervisor(holder, **options):
    def build(view, obs0):
        holder["replay"] = ReplayModel(holder["model"].entries)
        return LLMSupervisor(holder["replay"], **options)

    return build


def test_a_recorded_run_replays_to_identical_supervision_steps_without_the_model():
    holder = {}
    first, records, actions = fly("approach-vbar", recording_supervisor(holder), steps=40)
    second, replayed, replayed_actions = fly(
        "approach-vbar", replaying_supervisor(holder), steps=40
    )
    assert replayed == records and np.array_equal(replayed_actions, actions)
    consults = len([r for r in records if r[1]])
    assert holder["replay"].position == len(holder["model"].entries) == consults
    assert second.supervisor.usage() == first.supervisor.usage()
    assert first.supervisor.usage()["input_tokens"] > 0


def one_byte_changed(template):
    """The template with its first literal letter changed, outside the replacement fields."""
    depth = 0
    for i, ch in enumerate(template):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0 and ch.isascii() and ch.isalpha():
            return template[:i] + ("Q" if ch != "Q" else "R") + template[i + 1 :]
    raise AssertionError("the prompt has no literal letter")


def test_a_replay_with_one_byte_of_the_prompt_changed_raises_replay_error():
    holder = {}
    fly("approach-vbar", recording_supervisor(holder), steps=20)
    changed = one_byte_changed(SUPERVISOR_PROMPT)
    assert len(changed) == len(SUPERVISOR_PROMPT) and changed != SUPERVISOR_PROMPT
    with pytest.raises(ReplayError):
        fly("approach-vbar", replaying_supervisor(holder, prompt=changed), steps=20)


def test_a_replay_with_one_byte_of_the_request_changed_raises_replay_error():
    holder = {}
    fly("approach-vbar", recording_supervisor(holder), steps=20)

    def build(view, obs0):
        replay = ReplayModel(holder["model"].entries)

        def altered(messages, tools):
            content = messages[-1]["content"]
            return replay(
                [{**messages[-1], "content": content[:-1] + chr(ord(content[-1]) ^ 1)}], tools
            )

        return LLMSupervisor(altered)

    with pytest.raises(ReplayError):
        fly("approach-vbar", build, steps=20)


# What the prompt may depend on


def printed_numbers(texts):
    return [float(x) for text in texts for x in NUMBER.findall(text)]


class Fixed:
    """A sensor that reports one observation sequence whatever the true state."""

    name = "fixed"

    def reset(self, rng):
        pass

    def observe(self, true_state, t):
        if t < 200.0:
            return np.array([0.0, -300.0 + 0.5 * t, 0.0, 0.0, 0.5, 0.0])
        if t < 500.0:
            return np.array(R0)
        if t < 700.0:
            return np.array([0.0, -200.0 + 0.5 * (t - 500.0), 0.0, 0.0, 0.5, 0.0])
        return np.array((0.0, -100.0, 0.0, 0.0, 0.0, 0.0))


def sensor_flight(pair):
    seed, agent_seed = pair
    model = Capture(reply_of("GO"))
    agent = SupervisedAgent(LLMSupervisor(model))
    flight = Flight("approach-vbar", agent, seed, agent_seed, max_steps=90, sensor=Fixed())
    flight.model = model
    return flight.run(90)


def test_runs_from_true_starts_more_than_10_m_apart_give_identical_prompts():
    pairs = seeds("approach-vbar", "validation", 2).pairs()
    a, b = sensor_flight(pairs[0]), sensor_flight(pairs[1])
    assert np.linalg.norm(a.truths[0][:3] - b.truths[0][:3]) > 10.0
    assert a.model.prompts == b.model.prompts
    assert len(a.model.prompts) > 5
    assert {read_phase(body(p, a.env.agent_view())) for p in a.model.prompts} >= {
        "transfer",
        "holding",
    }


def test_no_number_of_the_true_state_is_in_the_prompts_of_a_flight():
    flight = sensor_flight(seeds("approach-vbar", "validation", 2).pairs()[0])
    numbers = printed_numbers(flight.model.prompts)
    assert numbers
    truth = np.concatenate(flight.truths)
    for c in truth[np.abs(truth) > 1e-2]:
        assert not any(abs(x - c) <= 1e-6 * max(1.0, abs(c)) for x in numbers)


def test_the_prompt_of_a_misbriefed_target_is_the_prompt_of_its_briefing():
    told = InertialPointing(0.3)
    prompts = []
    for target in (Misbriefed(actual=InertialPointing(1.0), briefed=told), told):
        spec = dataclasses.replace(get("docking"), target=target)
        env = entry_class(spec)(spec)
        model = Capture()
        agent = SupervisedAgent(LLMSupervisor(model), period_s=env.agent_view().dt)
        obs, _ = env.reset(seed=5)
        agent.reset(env.agent_view(), None)
        for _ in range(12):
            obs = env.step(agent.act(obs, env.agent_info()))[0]
        prompts.append(model.prompts)
    assert prompts[0] == prompts[1] and len(prompts[0]) == 12


@pytest.mark.parametrize(("prose_braces", "found"), [(7, True), (8, False)])
def test_an_order_is_read_only_within_the_first_eight_opening_braces(prose_braces, found):
    text = "{ " * prose_braces + reply_of("GO")
    order = parse_order(text, "ladder")
    assert (order is not None and order.command == Command.GO) is found


@pytest.mark.parametrize(("depth", "found"), [(MAX_JSON_DEPTH, True), (MAX_JSON_DEPTH + 1, False)])
def test_an_order_nested_deeper_than_the_json_limit_is_not_read(depth, found):
    # The order object itself is one level; its reason holds the rest.
    reason = "[" * (depth - 1) + "]" * (depth - 1)
    text = '{"command": "GO", "reason": ' + reason + "}"
    assert (parse_order(text, "ladder") is not None) is found


def test_deep_brackets_after_the_order_do_not_hide_it():
    order = parse_order(reply_of("GO") + " " + "[" * (MAX_JSON_DEPTH + 10), "ladder")
    assert order is not None and order.command == Command.GO


@pytest.mark.parametrize("agent_mode", ["ladder", "waypoint", "velocity"])
@pytest.mark.parametrize("supervisor_mode", ["ladder", "waypoint", "velocity"])
def test_a_supervisor_whose_mode_is_not_the_agents_is_refused_before_the_model_is_called(
    agent_mode, supervisor_mode
):
    model = Replies(Reply(reply_of("CONTINUE")))
    agent = SupervisedAgent(LLMSupervisor(model, mode=supervisor_mode), mode=agent_mode)
    agent.reset(APPROACH, None)
    if agent_mode == supervisor_mode:
        agent.act(np.array(FAR), {"time_s": 0.0})
        assert agent.last_step.consulted
    else:
        with pytest.raises(ValueError, match="orders but the agent flies"):
            agent.act(np.array(FAR), {"time_s": 0.0})
        assert model.requests == []
