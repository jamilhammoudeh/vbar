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
`impulse_coast` mode, and an integer from 0 to 26 in the discrete mode. The `view` carries every
`AgentView` field (`scenario_id`, `dt`, `max_steps`, `n`, `action_mode`, `max_accel`,
`thrust_limit`, `mass_kg`, `goal`, `constraints`, `observation`, `frame`) with tuples as lists.
`seed` and `spawn_key` are the agent's own random stream (`numpy.random.SeedSequence(seed,
spawn_key=...)`), seeded independently of the episode, so they carry nothing about the start state
or the noise. Nothing else crosses: no true state, no reward, no metrics.

Floats are written in Python's shortest round-trip form and read back to the same double, so a
client computes on exactly the numbers an in-process agent would, and a deterministic agent flies
the same trajectory bit for bit. NaN and infinity are refused. The server waits `timeout`
seconds (60 by default) for each message and raises `RemoteAgentError` if the client is silent,
disconnects or answers out of turn.

## Server

```
server = AgentServer(host="127.0.0.1", port=5555)
agent = server.accept()              # waits for the hello; agent.name is the client's name
result = run_episode(env, agent, seed=0)
agent.close()                        # sends close
```

`python -m vbar.agents.remote --port 5555 --seeds 0 1 2 --out results.jsonl` hosts the default
approach scenario for one client and writes one JSON line per episode with the result of
`run_episode` (arrays as lists). If the client fails, disconnects, resets the connection or sends
something that is not a valid message, the episode it was flying is written with `completed` false
and the error, and so is every seed not yet flown. Real-time runs ([agents.md](agents.md)) work
through the server too; the measured latency then includes the network.

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

LLM harnesses that call tools through the Model Context Protocol can fly V-BAR episodes
directly: `python -m vbar.agents.mcp --seeds 0 1 2 --out results.jsonl` serves the approach
scenario over stdio, JSON-RPC 2.0 with one message per line, and a harness adds it as a stdio
tool server. `ToolServer(env, seeds)` is the same server in Python; its `handle(message)` answers
one message and `serve(stdin, stdout)` runs the loop. It answers `initialize` (protocol versions
2025-06-18, 2025-03-26 and 2024-11-05; the client's choice when supported), `ping`, `tools/list`
and `tools/call`, ignores notifications, and returns JSON-RPC errors for unknown methods (-32601),
unknown tools and `params` or `arguments` that are not objects (-32602), lines that are not JSON
(-32700) and failures inside the server (-32603); after an error it goes on answering.

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
`agent_seed` 0, since the client has no stream of ours), and the command line writes them to the
`--out` file as they finish. When the session ends, because the input ends, the client stops reading
the output or the server fails, an episode still running and every seed not yet flown are recorded
too, with `completed` false, so leaving does not remove a bad episode from the results. The client
never sees the true state, the reward or the outcome.

## How it is tested

`tests/test_remote.py`, with the client in a thread of the test process:

- The PD agent (clear path and target in the way, normalized and acceleration modes), an agent
  driven by its own random stream through 3 m of navigation noise, and a bang-off-bang agent in
  the discrete mode fly exactly the trajectories and commands of the same agents in process.
- A hand-written client sees only the view, its seed (equal to the run's agent seed) and spawn
  key, the observation and the time, then the close.
- An error message from the client, a wrong protocol version and a silent client are reported.
- The command-line host writes one result per seed; a client that answers two steps and then
  sends a line that is not JSON leaves an unfinished record of two steps, with the error, and an
  unfinished record of the seed it never flew; so does an action of two numbers, a string, a
  missing action or NaN (the trajectory stays finite), an abortive close and a send that fails
  mid-episode.
- Burn-and-coast actions cross the server exactly; an action of the wrong length for its mode,
  or a discrete index out of range, is refused.
- With MATLAB installed and `VBAR_TEST_MATLAB=1`, the MATLAB PD agent flies exactly the Python PD
  agent's trajectories over two default episodes (checked on R2025a; not on CI, which has no
  MATLAB).

`tests/test_mcp.py`:

- The handshake negotiates the protocol version, notifications get no answer, the six tools are
  listed, an unknown method and a line that is not JSON get their error codes.
- A PD controller flying through `start_episode` and `command` calls produces exactly the
  in-process run's states and actions; command replies hold only the observation, time, steps
  left and episode number.
- Starting before the episode ends, observing with none running, a non-numeric command and
  starting past the last seed are tool errors.
- `predict_coast` through the server equals the tool on the current observation.
- `params` given as a list, `arguments` given as a list or a string, and `initialize` with list
  params get -32602, as does a tool name that is not a string, and the next request is still
  answered; a tool that fails inside gets -32603; a session that ends after one command of three
  episodes records all three, unfinished, and so does a client that stops reading its replies.
- `python -m vbar.agents.mcp` in a subprocess answers five requests on stdio and writes the
  episode's result.

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
