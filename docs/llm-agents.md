# LLM agents

Code: `vbar.agents.llm` (`LLMAgent`, `Reply`, `ToolCall`, `Price`, `text_model`, `parse_action`,
`RecordedModel`, `ReplayModel`), `vbar.agents.briefing` (`briefing`, `situation`, `check_goal`,
`target_mode`), `vbar.agents.tools` (`Tool`, `PREDICT_COAST`, `TRANSFER_BURN`, `CW_TOOLS`). An
LLM agent flies every task kind (hold, berthing, docking, inspection, departure); it asks a
language model for each command. V-BAR has no model client of its own: the model is any function,
so a hosted API, a local model or a test stand-in plugs in the same way.

## The model function

`LLMAgent(complete, ...)` calls `complete(messages, tools)` once per decision:

- `messages` is the decision's conversation as provider-neutral dicts with a `"role"`
  (`"user"`, `"assistant"` or `"tool"`) and a `"content"` string; it starts with one user
  message holding the prompt.
- `tools` lists the JSON-schema descriptions of the tools the agent offers (empty without tools).
- The function returns a `Reply(text, input_tokens, output_tokens, tool_calls)`, or just the
  text.

An adapter for a provider's client turns `messages` into its request and its response into a
`Reply`, with the token counts the provider reports. `text_model(fn)` adapts a function from a
prompt string to a reply string.

## Prompt

The default prompt (`PROMPT_TEMPLATE`) gives the frame and its axes, the mean motion, the briefing
of the task, the time, the observed position (to 1 cm) and velocity (to 0.1 mm/s), the situation
lines, the step length and the thrust limit, and ends with the reply format, a JSON object of
fractions such as `{"ax": 0.1, "ay": -0.5, "az": 0.0}`. `prompt=` replaces it with any
`str.format` template over the fields in `PROMPT_FIELDS`:

| Field | Value |
|---|---|
| `briefing` | the goal, the target's motion and every constraint, in sentences (below) |
| `situation` | the observed state measured against the goal, one line per quantity (below) |
| `thrust` | the thrust limit: per axis for box thrusters, the vector's length for a ball |
| `n`, `dt`, `max_accel` | mean motion (rad/s), step (s), maximum acceleration (m/s^2) |
| `t` | time since the start (s) |
| `x`, `y`, `z`, `vx`, `vy`, `vz` | the observed state (m, m/s) |
| `rng` | observed distance to the target (m) |
| `hold`, `cap_speed`, `dist` | hold point (m), capture speed (m/s), observed distance to the hold point (m) |
| `keep_out` | the keep-out radius (m) |
| `reply_format` | the instruction for the reply's JSON |

Every value comes from the observation, the agent's `info` or the scenario's public description
(`AgentView`), never from the true state. A template with an unknown field is refused when the
agent is built. `reset` refuses a goal that lacks a key the template needs: `hold`, `dist` and
`cap_speed` need a goal with a hold point and a capture speed, `keep_out` a `keep_out`
constraint with a radius, and the briefing the keys of its task (table below). The default
template uses none of the hold fields, so it flies every task kind.

### Briefing and situation

`vbar.agents.briefing` turns the public view into words. `briefing(view)` describes the goal of
each task kind from its keys (`TASK_KEYS`, which `check_goal(view)` requires). `check_goal` also
refuses a numeric goal entry the briefing puts in a sentence that is not finite numbers of the
right shape and sign: 3-vectors for points and directions, a nonzero port axis and Sun
direction, an axial speed window with 0 <= low < high, positive radii, limits and ranges, a
positive whole number of points, a coverage in (0, 1], a Sun exclusion angle in [0, pi], burn and
waypoint lists of 3-vectors, a station-keeping point at a positive distance with a dwell of 0 or
more, and inhibit windows that start at or after 0.
`target_mode` refuses a mode parameter that is not a single finite number, and a zero spin axis.
An entry the briefing does not know is printed as it is. The task keys:

| Task | Keys | Sentence |
|---|---|---|
| `hold` | `hold_point`, `capture_radius`, `capture_speed` | reach the hold point and stop within the radius below the speed, for `hold_time` without a break when it is set |
| `berth` | the hold keys and `capture_time` | rest in the capture box; the thrusters are inhibited for the capture window while the arm grapples |
| `dock` | `port`, `port_axis`, `lateral_limit`, `lateral_speed_limit`, `axial_speed_window` | contact is the first crossing of the port plane within `contact_range` of the port, inside the lateral offset, lateral speed and axial speed window |
| `inspect` | `points`, `target_radius`, `max_range`, `sun_exclusion`, `coverage` | when a point is imaged (range, facing side, lit, Sun out of the camera's view) and the coverage of the points lit at some time that succeeds |
| `depart` | `port`, `port_axis`, `clear_range`, `drift_horizon` | reach the clear range on a path whose free drift does not return into the keep-out zone within the horizon; arrival scored at the clear point |

A goal of another task is named and its keys are listed as they are. The other goal entries
each get a sentence: the target's attitude mode (passive, a fixed pitch with an optional limit
cycle, inertial pointing, a tumble, or a manoeuvring target with its burn schedule), suggested
waypoints and a docking hold point (not scored), thrust-inhibit windows, the range beyond which
passive safety is scored, the radius inside which abort availability is not, the plume range,
and the Sun direction at t = 0, which turns about z at -n. Each public constraint gets a line:
keep-out sphere (open at the port for a docking target, with the reception depth), corridor cone
(half-angle in degrees, apex, axis and active range), speed and closing-speed limits
v0 + v1 rho, a gate table, the passive-speed envelopes on the V-bar, n (rho - d) / 0.47755, and
the R-bar, n sqrt((4 rho - d)^2 - 9 rho^2) (Fehse, *Automated Rendezvous and Docking of
Spacecraft*, eqs. 3.28 and 3.22), the lighting window, and a zone that may not be entered before
the GO time. A constraint of an unknown type is shown with its parameters. Every goal entry and
constraint of the built-in catalogue has its own sentence.

Ports, corridors and the body keep-out are fixed to the target's body; `target_mode(entry)`
rebuilds the briefed attitude mode from the goal's `"target"` entry, and the briefing gives
their positions in Hill axes at t = 0. `situation(view, obs, info)` gives, from the observation:

- the distance to the target, the closing speed $-\mathbf r\cdot\mathbf v/\rho$ and the speed;
- for a hold or berthing, the distance to the hold point;
- for a docking or departure, the distance along the port axis $(\mathbf r - \mathbf p)\cdot\hat
  a$, the lateral offset from the axis, the speed along the axis toward the port (outward for a
  departure) and the lateral speed, measured as the contact check measures them, with the port
  and its axis where the briefed mode has turned them by now;
- for a departure, the distance to the clear point;
- for an inspection, the fraction of points imaged so far (from the agent's `info`) and the Sun
  direction now;
- on a turning target, each corridor's apex and axis now; inside a thrust-inhibit window, when it
  ends.

A misbriefed target (`Misbriefed`) is described as briefed: the prompt is the same as for a
scenario whose target really moves as the briefing says.

The first JSON object in the reply is read; each fraction is clipped to [-1, 1] and converted to
the action mode. A reply without a valid object (malformed, nested more than 32 levels deep, or
holding a number too large for a float) is a parse failure (metric A4) and the spacecraft coasts
for that step; `true` and `false` read as 1 and 0, as the environment reads them. The agent rejects
the discrete and burn-and-coast action modes.

## Tool calling

`LLMAgent(complete, tools=CW_TOOLS)` offers tools to the model. When a reply has tool calls
(`ToolCall(name, arguments, id)`), the agent appends the assistant message with its calls,
runs each tool, appends one `"tool"` message per call with the result as JSON (paired by
`tool_call_id`), and calls the model again. After `max_tool_rounds` (4) rounds of calls it asks
once more offering no tools, so every decision ends in an answer. Unknown tools, invalid
arguments (a missing or non-finite one, or one that is not a number: a quoted number or a boolean
is refused, as the schema asks for numbers), and calls whose name is not a string or whose
arguments are not an object come back to the model as `{"error": ...}`.

A tool's `run(obs, view, arguments)` sees only the observation the agent received and its
`AgentView`: tools do arithmetic for the model and cannot reveal the truth. The built-in ones
use the Clohessy-Wiltshire model with the view's mean motion:

- `predict_coast(seconds)`: the position and velocity after coasting from the observed state,
  and the closest range to the target on the way (found with the certified in-step minimum,
  [constraints.md](constraints.md)), for up to three orbits.
- `transfer_burn(x, y, z, seconds)`: the two impulses of a transfer from the observed state to
  rest at a point, $\Delta\mathbf v_1 = \Phi_{rv}^{-1}(\mathbf r_f - \Phi_{rr}\mathbf r_0) -
  \mathbf v_0$ and $\Delta\mathbf v_2 = -(\Phi_{vr}\mathbf r_0 + \Phi_{vv}\mathbf v_0^+)$, the
  total, and how many steps of full thrust the first burn takes. Durations where the targeting
  problem is singular, where $\sin nT = 0$ out of plane and at the roots of
  $8(1 - \cos nT) = 3nT\sin nT$ in plane (Alfriend et al., *Spacecraft Formation Flying*,
  p. 92), are refused. The impulses are ideal; a burn of several steps arrives off target.

Results are rounded (positions to 1 cm, velocities to 0.1 mm/s) to keep transcripts short; a
result with a number that is not finite, from extreme arguments, comes back as an error.

## Tokens, cost and the log

`price=Price(input_per_mtok, output_per_mtok)` is the declared price table of metric A3, in USD
per million tokens. Each decision's entry in `agent.log` holds the time, the prompt, the
messages, the final reply, the action, the parse-failure flag, the number of model calls, the
input and output tokens, the cost (or `None` without a price), the tool calls with their
arguments and results, and the wall-clock latency of the whole decision.
`agent.usage()` totals decisions, model calls, tokens, cost and parse failures over the episode.
The log restarts at every reset.

## Reasoning faithfulness

With `belief=True` the reply format asks the model to add what it believes about the target to
its JSON: `"range_m"`, its distance to the target, and `"closing"`, whether that distance is
shrinking. `belief_matches(belief, obs)` checks the belief against the observation the model was
given (metric A5): the range within 10% of $\lVert\mathbf r\rVert$ and `closing` equal to
$\mathbf r\cdot\mathbf v < 0$ for the observed position and velocity. It measures whether the
model's stated reasoning is consistent with its own inputs, not whether the inputs are true; a
missing or malformed belief (a non-number range, a quoted number, a closing flag that is not
`true` or `false`) does not match. `parse_belief` reads the first JSON object in the reply that
has both fields, so the belief may also come in its own object after the command. Each log entry
holds `belief` and `faithful`, and `usage()["faithful_fraction"]` is the fraction over the episode
(`None` without `belief`); it equals metric A5, which uses the same parser and rule.

## Transcripts and replay

`RecordedModel(model)` wraps a model and records, for each request, the SHA-256 digest of its
canonical JSON (messages and tools, keys sorted) with the reply, text, token counts and tool
calls; `save(path)` writes them as JSON. `ReplayModel(entries)` or `ReplayModel.load(path)`
answers the same requests in order with the recorded replies and never calls a model. Given the
same scenario, seeds and agent configuration, the prompts, tool results and therefore the whole
run repeat exactly. A request that differs from the recorded one raises `ReplayError` naming
the request, because the run has diverged (another seed, prompt or package version), and so
does a run that asks for more replies than were recorded.

In real time the model's latency changes the trajectory; replay it with the recorded latencies,
`RealTime(latencies=result["latencies_s"], time_scale=s)` ([agents.md](agents.md)).

## How it is tested

`tests/test_llm.py` with stand-in models:

- The model receives one user message per decision with the observed position and the time, and
  an empty tool list; the discrete and burn-and-coast modes are refused before the model is
  called.
- Tokens and cost follow the declared price per decision and in total; without a price the cost
  is `None`.
- A custom template is filled from the observation; a template with an unknown field is refused.
- Unparseable replies coast and are counted.
- With 5 m navigation noise, `predict_coast` returns the closed-form prediction from the
  observation (to its 1 cm rounding) and differs from the truth's by metres; the conversation
  carries the assistant's call and the paired tool result; both tools are offered.
- A model that calls tools forever is cut off after `max_tool_rounds` rounds and asked once more
  without tools.
- Unknown tools, missing or non-finite arguments, quoted numbers, booleans and out-of-range
  durations return errors.
- Applying `transfer_burn`'s first impulse to the observed state arrives at the target within the
  rounding (0.3 m over 1500 s) and the second impulse stops it; a half-orbit transfer is refused as
  singular; arguments of 1e308 give an error, not infinity, and so does a 400-digit integer; a reply
  whose fractions overflow a float, or nest 3000 levels deep, is a parse failure and the episode
  goes on; a belief nested 32 levels parses and one nested 33, 3000 or 50 000 levels does not, on
  every Python version; tool calls with a list, a string or null as arguments, or a list as the
  name, get errors; a reply whose text is null (an adapter passing a provider's empty content) is a
  parse failure.
- Through 20 m of navigation noise, a model that reads its position and velocity from the prompt
  and reports them as its belief is faithful at every step, and one that overstates the range by
  half never is; the belief must have both fields and the right closing sign; without `belief`
  the prompt does not ask for one and nothing is scored.
- A 25-step run with navigation noise, a model with its own random stream and tool calls, saved
  to a file and replayed, gives identical states, actions, messages and token counts; replay of
  another seed fails at request 0 and a truncated transcript fails when it runs out; a real-time
  run replayed with its recorded latencies gives identical states.

`tests/test_agents.py` checks that the prompt is built from the noisy observation, not the
truth, and that fractions are converted to each action mode.

`tests/test_briefing.py`, for the briefing:

- The briefing of every catalogue scenario has a sentence for each goal entry and one line per
  constraint, with no raw fallback; an unknown goal entry, constraint type and task are still
  shown.
- A goal without a key its task needs is refused, naming the key, for every task; an unknown
  target mode, or one missing a parameter, is refused.
- The target mode rebuilt from its public entry turns as the mode does, for every mode and every
  attitude a manoeuvring target keeps.
- Through 0.5 m and 5 mm/s of navigation noise, the docking offsets and speeds are those of the
  observation and differ from the truth's; a turning port and a corridor on a turning target are
  reported where the scenario puts them (to the 1 cm rounding), and a pitched target's corridor
  lies along its port axis in Hill axes; the inspection coverage is the one in `info`; a thrust
  inhibit window is announced only while it lasts; a misbriefed target's briefing and situation
  equal those of a scenario that flies the briefing; a departure's distance along the port axis,
  lateral offset, outward and lateral speeds and distance to the clear point are the observation's.

`tests/test_llm_tasks.py`, for the agent:

- The agent resets and flies two steps of every catalogue scenario and variant and of every suite
  entry, and its first prompt names the task.
- The catalogue's task kinds are the five of `TASK_KEYS`; for each kind, a goal without any one of
  its keys is refused at reset with that key named, and the full goal is accepted. The hold fields
  need a hold point and a capture speed and `keep_out` a keep-out radius (the run time assurance
  docking task has none, and the default prompt flies it); ball thrusters are described as a limit on the
  command's length; a misbriefed target's prompt equals the prompt of a scenario that flies the
  briefing; the prompt at t = 600 s holds the situation at 600 s, where a turning port has moved.
- Behind the `slow` marker: a stand-in model that reads only the time, position and velocity
  printed in the prompt, and answers with the command of the reference controller of the scenario
  audit ([scenarios.md](scenarios.md)), succeeds on three drawn validation seeds of each of 21
  scenarios, with no parse failure: every entry of the `core-v1` suite (the V-bar and R-bar holds,
  `docking:axis`, the run time assurance docking task and inspection, each also with `:nav1`), the
  comparison tasks `afrl-docking`, `afrl-inspection`, `spacegym-docking` and
  `spacegym-rendezvous`, docking on a still, lit, pitched and turning target, berthing, a hold
  under a turning target and a departure past a manoeuvring target. A test checks that these
  flights include every task kind.

## Example

```python
from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import LLMAgent, Price, Reply


def complete(messages, tools):
    """Replace with a call to a model; this one always coasts."""
    return Reply('{"ax": 0.0, "ay": 0.0, "az": 0.0}', input_tokens=850, output_tokens=25)


agent = LLMAgent(complete, name="stand-in", price=Price(3.0, 15.0))
run_episode(ApproachEnv(ApproachConfig(max_steps=10)), agent, seed=0)
print(agent.usage()["input_tokens"], round(agent.usage()["cost_usd"], 6))  # 8500 0.02925
```

The same agent docks; the briefing and the first prompt's situation lines read:

```python
from vbar.agents import LLMAgent, Reply
from vbar.agents.briefing import briefing
from vbar.scenarios import make

env = make("docking")
contact = briefing(env.agent_view()).splitlines()[1]
print(contact[:41])  # Contact is the first crossing of the port

agent = LLMAgent(lambda messages, tools: Reply('{"ax": 0.0, "ay": 0.0, "az": 0.0}'))
obs, _ = env.reset(seed=3)
agent.reset(env.agent_view())
agent.act(obs, env.agent_info())
print(agent.log[0]["prompt"].splitlines()[-6])  # the offsets from the port axis
```
