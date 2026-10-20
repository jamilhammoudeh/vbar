# LLM agents

Code: `vbar.agents.llm` (`LLMAgent`, `Reply`, `ToolCall`, `Price`, `text_model`, `parse_action`,
`RecordedModel`, `ReplayModel`), `vbar.agents.tools` (`Tool`, `PREDICT_COAST`, `TRANSFER_BURN`,
`CW_TOOLS`). An LLM agent asks a language model for each command. V-BAR has no model client of its
own: the model is any function, so a hosted API, a local model or a test stand-in plugs in the same
way.

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

The default prompt (`PROMPT_TEMPLATE`) gives the frame and its axes, the goal and capture speed,
the keep-out radius, the mean motion, the time, the observed position and velocity, the
distances to the hold point and the target, the step length and the thrust limit, and ends with
the reply format, a JSON object of fractions such as `{"ax": 0.1, "ay": -0.5, "az": 0.0}`.
`prompt=` replaces it with any `str.format` template over the fields in `PROMPT_FIELDS`:

| Field | Value |
|---|---|
| `hold`, `cap_speed`, `keep_out` | hold point (m), capture speed (m/s), keep-out radius (m) |
| `n`, `dt`, `max_accel` | mean motion (rad/s), step (s), maximum acceleration (m/s^2) |
| `t` | time since the start (s) |
| `x`, `y`, `z`, `vx`, `vy`, `vz` | the observed state (m, m/s) |
| `dist`, `rng` | observed distance to the hold point and to the target (m) |
| `reply_format` | the instruction for the reply's JSON |

Every value comes from the observation or the scenario's public description (`AgentView`),
never from the true state. A template with an unknown field is refused when the agent is built.

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
arguments, and calls whose name is not a string or whose arguments are not an object come back
to the model as `{"error": ...}`.

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
- Unknown tools, missing or non-finite arguments and out-of-range durations return errors.
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
