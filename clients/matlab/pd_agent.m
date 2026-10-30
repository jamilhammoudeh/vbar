function agent = pd_agent(wn, zeta)
%PD_AGENT The PD controller with Clohessy-Wiltshire cancellation, as a V-BAR remote agent.
%   AGENT = PD_AGENT(WN, ZETA) uses natural frequency WN (rad/s) and damping ratio ZETA; the
%   defaults 0.01 and 1 are those of vbar.agents.PDAgent. It commands fractions of the maximum
%   acceleration, so it suits the "normalized" action mode only.
%
%   It flies to a hold point, so its reset raises the error vbar:pd_agent for a goal whose task is
%   not "hold" or "berth" (a goal without a task is a hold), or that has no hold_point, as the
%   Python agents do.
%
%   With a server started by  python -m vbar.agents.remote --port 5555 --seeds 0 1 2
%       vbar_agent_client("127.0.0.1", 5555, pd_agent())

if nargin < 1
    wn = 0.01;
end
if nargin < 2
    zeta = 1;
end
agent.name = "matlab-pd";
agent.reset = @(view, seed) start(view);
agent.act = @(state, obs, info) deal(command(state, obs, wn, zeta), state);
end

function state = start(view)
task = "hold";
if isfield(view.goal, "task")
    task = string(view.goal.task);
end
if ~any(task == ["hold", "berth"])
    error("vbar:pd_agent", ...
        "the MATLAB PD agent flies to a hold point and cannot fly the %s task", task);
end
if ~isfield(view.goal, "hold_point")
    error("vbar:pd_agent", ...
        "the MATLAB PD agent flies to a hold point and this goal has none");
end
state = struct("n", view.n, "hold", view.goal.hold_point(:), "max_accel", view.max_accel);
end

function u = command(state, obs, wn, zeta)
n = state.n;
r = obs(1:3);
v = obs(4:6);
cw = [3 * n^2 * r(1) + 2 * n * v(2); -2 * n * v(1); -(n^2) * r(3)];
a = -cw - wn^2 * (r - state.hold) - 2 * zeta * wn * v;
u = min(max(a / state.max_accel, -1), 1);
end
