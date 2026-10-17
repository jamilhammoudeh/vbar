function vbar_agent_client(host, port, agent)
%VBAR_AGENT_CLIENT Run a MATLAB agent against a V-BAR agent server (protocol 1).
%   VBAR_AGENT_CLIENT(HOST, PORT, AGENT) connects to the server, then answers its messages until
%   the server closes the session. AGENT is a struct with fields
%     name   - the agent's name
%     reset  - function handle, state = reset(view, seed), called at the start of each episode
%     act    - function handle, [action, state] = act(state, obs, info), called every step
%   VIEW is the scenario's public description as decoded by JSONDECODE (view.dt, view.n,
%   view.max_accel, view.goal.hold_point, ...), OBS a column vector and INFO.time_s the time.
%   ACTION is a 3-vector in the scenario's action mode, or an index for "disc27".
%
%   Needs MATLAB R2020b or later (tcpclient with readline and writeline).

c = tcpclient(host, port, "Timeout", 60);
configureTerminator(c, "LF");
writeline(c, jsonencode(struct("type", "hello", "name", agent.name, "protocol", 1)));
state = [];
while true
    message = jsondecode(readline(c));
    switch message.type
        case "reset"
            state = agent.reset(message.view, message.seed);
            writeline(c, '{"type":"ready"}');
        case "act"
            [action, state] = agent.act(state, message.obs(:), message.info);
            writeline(c, action_message(action));
        case "close"
            return
        otherwise
            writeline(c, jsonencode(struct("type", "error", ...
                "message", "unknown message type " + string(message.type))));
            return
    end
end
end

function text = action_message(action)
if isscalar(action)
    text = sprintf('{"type":"action","action":%d}', round(action));
    return
end
% Seventeen significant digits write a double exactly, so the server flies the same command.
parts = arrayfun(@(a) sprintf("%.17g", a), action(:)', "UniformOutput", false);
text = "{""type"":""action"",""action"":[" + strjoin(string(parts), ",") + "]}";
end
