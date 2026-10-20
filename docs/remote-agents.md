# Remote agents

Code: `vbar.agents.remote` (`AgentServer`, `RemoteAgent`, `serve_agent`, `main`), `clients/matlab/`
(`vbar_agent_client.m`, `pd_agent.m`), `vbar.agents.mcp` (`ToolServer`). An agent can run in another
process, on another machine or in another language: the agent server waits for it to connect and
presents it to `run_episode` as an ordinary agent, so it is scored exactly like an in-process one.
Running submitted agents out of process also keeps them away from the environment's memory.

## Protocol

Version 1 is a lockstep exchange over one TCP connection: UTF-8 JSON objects, one per line
(`\n`). The client connects and speaks first.

| From | Message | When |
|---|---|---|
| client | `{"type": "hello", "name": <agent name>, "protocol": 1}` | once, on connecting |
| server | `{"type": "reset", "view": <AgentView>, "seed": <int>, "spawn_key": [<int>, ...]}` | each episode start |
| client | `{"type": "ready"}` | after resetting |
| server | `{"type": "act", "obs": [<float>, ...], "info": {"time_s": <float>}}` | each decision |
| client | `{"type": "action", "action": [<float>, ...]}` or an integer for `"disc27"` | the answer |
| server | `{"type": "close"}` | at the end of the session |

A client may answer any message with `{"type": "error", "message": ...}`, which the server raises as
`RemoteAgentError`, as it does for an action of the wrong form: three finite numbers, four in the
`impulse_coast` mode, and a whole number from 0 to 26 in the discrete mode. The `view` carries every
`AgentView` field (`scenario_id`, `dt`, `max_steps`, `n`, `action_mode`, `max_accel`,
`thrust_limit`, `mass_kg`, `goal`, `constraints`, `observation`, `frame`) with tuples as lists.
`seed` and `spawn_key` are the agent's own random stream (`numpy.random.SeedSequence(seed,
spawn_key=...)`), seeded from a hash of the agent seed and independently of the episode, so
they carry nothing about the start state or the noise; the agent seed can be found from them
only by guessing it, which a drawn 128-bit seed defeats ([environment.md](environment.md)).
Nothing else crosses: no true state, no reward, no metrics.

Floats are written in Python's shortest round-trip form and read back to the same double, so a
client computes on exactly the numbers an in-process agent would, and a deterministic agent flies
the same trajectory bit for bit. NaN and infinity are refused. The server waits `timeout`
seconds (60 by default) for each whole message, however slowly its bytes arrive, refuses lines
longer than 1 MiB, and raises `RemoteAgentError` if the client is silent, disconnects or answers
out of turn. The Python client sends a one-number action as a number and reports a non-finite
action as an error.

## Server

```
server = AgentServer(host="127.0.0.1", port=5555)
agent = server.accept()              # waits for the hello; agent.name is the client's name
result = run_episode(env, agent, seed=0)
agent.close()                        # sends close
```

`python -m vbar.agents.remote --port 5555 --episodes 3 --out results.jsonl` hosts the default
approach scenario for one client and writes one JSON line per episode with the result of
`run_episode` (arrays as lists). Each episode's seed is drawn (128 bits) and recorded in its result;
`--seeds` repeats given seeds instead. A client that can guess the seed can rebuild the true state
from its first observation, so scored sessions keep the drawn seeds. However the session fails (a
bad hello, a disconnect or reset, a line that is not JSON, not UTF-8 or too deeply nested to parse,
an action the server cannot use), the episode being flown is written with `completed` false and the
error, and so is every seed not yet flown; a failed hello records every seed unflown. The records
are also written when the host itself is interrupted (Ctrl+C) before the interrupt goes on.
Real-time runs ([agents.md](agents.md)) work through the server too; the measured latency then
includes the network.

## Clients

**Python:** `serve_agent(agent, host, port)` connects, runs any `vbar` agent for the server and
returns when the server closes the session. It rebuilds the view and the agent's random stream
from the reset message.

**MATLAB** (R2020b or later): `vbar_agent_client(host, port, agent)` in `clients/matlab/` runs an
agent given as a struct with `name`, `reset(view, seed)` returning the agent's state, and
`act(state, obs, info)` returning `[action, state]`. It uses only MATLAB's `tcpclient`,
`jsonencode` and `jsondecode`, and writes actions with 17 significant digits so they reach the
server exactly. `pd_agent()` is the PD controller with Clohessy-Wiltshire cancellation written in
MATLAB, for the normalized action mode:

```
vbar_agent_client("127.0.0.1", 5555, pd_agent())
```

The seed arrives in MATLAB as a double, so the agent's stream cannot be rebuilt bit for bit
there; MATLAB agents that need randomness seed their own generator from it.

## Tool server for LLM harnesses

LLM harnesses that call tools through the Model Context Protocol can fly V-BAR episodes directly:
`python -m vbar.agents.mcp --episodes 3 --out results.jsonl` serves the approach scenario over
stdio, with drawn seeds as above (`--seeds` repeats given ones), JSON-RPC 2.0 with one message per
line, and a harness adds it as a stdio tool server. `ToolServer(env, seeds)` is the same server in
Python; its `handle(message)` answers one message and `serve(stdin, stdout)` runs the loop. It
answers `initialize` (protocol versions 2025-06-18, 2025-03-26 and 2024-11-05; the client's choice
when supported), `ping`, `tools/list` and `tools/call`, ignores notifications, and returns JSON-RPC
errors for unknown methods (-32601), unknown tools and `params` or `arguments` that are not objects
(-32602), lines that are not JSON (-32700) and failures inside the server (-32603); after an error
it goes on answering.

| Tool | Arguments | Returns |
|---|---|---|
| `describe_task` | none | the frame in words, the `AgentView` and the number of episodes |
| `start_episode` | none | the next seed's first observation (by name), time, steps left, episode number |
| `observe` | none | the same for the current step |
| `command` | `ax`, `ay`, `az` fractions in [-1, 1] | the next observation, or `episode_over` and the episodes left |
| `predict_coast`, `transfer_burn` | as for the LLM agent | predictions from the current observation ([llm-agents.md](llm-agents.md)) |

Tool results come as JSON text and as `structuredContent`; a call that cannot be carried out (no
episode running, a new episode before the current one ends, no episodes left, non-numeric or
non-finite commands, invalid predictions) returns `isError` with the reason. Episodes run in the
order of the seeds and cannot be abandoned, so a client cannot restart until it likes a start. Each
command's wall time since the previous observation is the decision latency. Results stay on the
server: `ToolServer.results` holds `run_episode`'s result for each finished episode (with
a drawn `agent_seed` the client never uses), and the command line writes them to the
`--out` file as they finish. When the session ends, because the input ends, the client stops reading
the output or the server fails, an episode still running and every seed not yet flown are recorded
too, with `completed` false, so leaving does not remove a bad episode from the results. The client
never receives the true state, the reward or the outcome, though the time each step takes still
says a little about the true path ([environment.md](environment.md)).

## How it is tested

`tests/test_remote.py`, with the client in a thread of the test process:

- The PD agent (clear path and target in the way, normalized and acceleration modes), an agent
  driven by its own random stream through 3 m of navigation noise, and a bang-off-bang agent in
  the discrete mode fly exactly the trajectories and commands of the same agents in process.
- A hand-written client sees only the view, its seed (the hashed one behind an in-process
  agent's generator) and spawn key, the observation and the time, then the close.
- Without `--seeds` the host draws a different 128-bit seed for each episode and records it.
- An error message from the client, a wrong protocol version and a silent client are reported.
- The command-line host writes one result per seed; a client that answers two steps and then
  sends a line that is not JSON leaves an unfinished record of two steps, with the error, and an
  unfinished record of the seed it never flew; so does an action of two numbers, a string, a
  missing action or NaN (the trajectory stays finite), an abortive close, a send that fails
  mid-episode, a 401-digit integer, 200 000 levels of nesting and bytes that are not UTF-8; a
  hello with the wrong protocol records both seeds unflown; a message sent a byte every 0.25 s
  times out after the 1 s timeout, a line over 1 MiB is refused, and an interrupt of the host
  mid-episode still writes both records before it propagates.
- An agent returning `array([13])` in the discrete mode flies the in-process trajectory through
  the server; a NaN action is reported by the client as an error.
- Burn-and-coast actions cross the server exactly; an action of the wrong length for its mode,
  or a discrete index out of range or fractional, is refused, while 13.0 is index 13 as in
  process.
- With MATLAB installed and `VBAR_TEST_MATLAB=1`, the MATLAB PD agent flies exactly the Python PD
  agent's trajectories over two default episodes (checked on R2025a; not on CI, which has no
  MATLAB).

`tests/test_mcp.py`:

- The server refuses the discrete and burn-and-coast modes.
- The handshake negotiates the protocol version, notifications get no answer, the six tools are
  listed, an unknown method and a line that is not JSON get their error codes.
- A PD controller flying through `start_episode` and `command` calls produces exactly the
  in-process run's states and actions; command replies hold only the observation, time, steps
  left and episode number.
- Starting before the episode ends, observing with none running, a non-numeric command and starting
  past the last seed are tool errors; 200 000 levels of nesting are a parse error and the next
  request is answered; a NaN request id, or one that is a list, an object or a boolean, gets an
  error in strict JSON with id null; a result that is not finite keeps its id; a prediction of 1e308
  metres is a tool error that keeps its request id.
- `predict_coast` through the server equals the tool on the current observation.
- `params` given as a list, `arguments` given as a list or a string, and `initialize` with list
  params get -32602, as does a tool name that is not a string, and the next request is still
  answered; a tool that fails inside gets -32603; a session that ends after one command of three
  episodes records all three, unfinished, and so does a client that stops reading its replies.
- `python -m vbar.agents.mcp` in a subprocess answers five requests on stdio and writes the
  episode's result; without `--seeds` it records two episodes under two different drawn 128-bit
  seeds.

## Example

```python
import threading

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent
from vbar.agents.remote import AgentServer, serve_agent

server = AgentServer()  # a free local port
host, port = server.address
threading.Thread(target=serve_agent, args=(PDAgent(), host, port), daemon=True).start()
agent = server.accept()
result = run_episode(ApproachEnv(ApproachConfig()), agent, seed=0)
agent.close()
server.close()
print(agent.name, result["captured"])  # pd True
```
