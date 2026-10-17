# Remote agents

Code: `vbar.agents.remote` (`AgentServer`, `RemoteAgent`, `serve_agent`, `main`),
`clients/matlab/` (`vbar_agent_client.m`, `pd_agent.m`). An agent can run in another process,
on another machine or in another language: the agent server waits for it to connect and presents
it to `run_episode` as an ordinary agent, so it is scored exactly like an in-process one. Running
submitted agents out of process also keeps them away from the environment's memory.

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

A client may answer any message with `{"type": "error", "message": ...}`, which the server raises
as `RemoteAgentError`. The `view` carries every `AgentView` field (`scenario_id`, `dt`,
`max_steps`, `n`, `action_mode`, `max_accel`, `thrust_limit`, `mass_kg`, `goal`, `constraints`,
`observation`, `frame`) with tuples as lists. `seed` and `spawn_key` are the agent's own random
stream (`numpy.random.SeedSequence(seed, spawn_key=...)`), seeded independently of the episode,
so they carry nothing about the start state or the noise. Nothing else crosses: no true state,
no reward, no metrics.

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
`run_episode` (arrays as lists). Real-time runs ([agents.md](agents.md)) work through the server
too; the measured latency then includes the network.

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

## How it is tested

`tests/test_remote.py`, with the client in a thread of the test process:

- The PD agent (clear path and target in the way, normalized and acceleration modes), an agent
  driven by its own random stream through 3 m of navigation noise, and a bang-off-bang agent in
  the discrete mode fly exactly the trajectories and commands of the same agents in process.
- A hand-written client sees only the view, its seed (equal to the run's agent seed) and spawn
  key, the observation and the time, then the close.
- An error message from the client, a wrong protocol version and a silent client are reported.
- The command-line host writes one result per seed.
- With MATLAB installed and `VBAR_TEST_MATLAB=1`, the MATLAB PD agent flies exactly the Python PD
  agent's trajectories over two default episodes (checked on R2025a; not on CI, which has no
  MATLAB).

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
