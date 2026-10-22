"""Closed-loop known answers of the keep-out barrier filters on the exact CW model.

Run with `python benchmarks/filter_known_answers.py`. Each table asserts its counts and prints
them with the filter's compute per decision; the full run takes several minutes.

All runs are at 500 km with a 50 m keep-out, the command held for each step, and a violation is
a certified in-step lower bound (`vbar.constraints.segment_minimum`, 20 samples) below -1 mm, the
keep-out resolution. Starts are drawn with seed 1903 in a fixed order, so the counts are exact.

1. 100 starts in the exponential barrier's start set, h0 ~ U(0, 150) m, velocity components
   ~ U(-0.3, 0.3) m/s, PD nominal, 1500 s, rows projected without a thrust limit: poles 0.05 1/s
   give no violation at dt 1, 5 and 10 s; poles 0.1 1/s none at 1 and 5 s and 100 of 100 at
   10 s, where 0.1 dt exceeds the sample-and-hold limit 0.787.
2. 40 starts inside each filter's set from the same region, box 0.02 m/s^2, PD nominal, 2000 s:
   the barrier row with poles 0.05 1/s violates in no run at dt 1, 5, 10 s and in all at 20 s; a
   first-order row on the braking barrier violates in every run at every dt; `BackupFilter`
   with the braking barrier, at a_b = 0.01044 m/s^2 and at its default a_b, in none at 1, 5, 10
   and 20 s.
3. 400 approach-vbar starts: at least 393 are certified passively safe, each confirmed by the
   drift sampled every second; on the first 40 of them `BackupFilter` with the free-drift barrier,
   dt 10 s, box 0.02 m/s^2, has no violation and all 40 reach the hold point.

Ames et al., *Control Barrier Functions: Theory and Applications*, ECC 2019: sec. IV (Def. 7)
for the exponential barrier, sec. III (eq. 12, Theorem 4) and sec. V.B for the backup barriers.
"""

import time

import numpy as np
from scipy.optimize import minimize_scalar

from vbar.agents.base import AgentView
from vbar.constraints import KeepOut, segment_minimum
from vbar.dynamics import cw_discrete, mean_motion
from vbar.dynamics.models import CWDynamics
from vbar.safety.barriers import BrakingBarrier, KeepOutEcbf, PassiveBarrier
from vbar.safety.cbf import BackupFilter
from vbar.solvers import ActuatorSet, Rows, project

N = mean_motion(6378137.0 + 500e3)
R = 50.0
A_MAX = 0.02
KEEP_OUT = KeepOut(R)
SAMPLES = 20
HOLD_POINT = np.array([0.0, -100.0, 0.0])
# a_max - (3 n^2 rho_cap + 2 n v_cap) for rho_cap = 2 km, v_cap = 1 m/s.
A_B = A_MAX - (3.0 * N * N * 2000.0 + 2.0 * N * 1.0)
BOX = ActuatorSet(A_MAX, "box")


def a_cw(x):
    return np.array([3 * N * N * x[0] + 2 * N * x[4], -2 * N * x[3], -N * N * x[2]])


def view(dt, max_steps):
    return AgentView(
        scenario_id="vbar:approach-vbar:cw:box:dv:v0",
        dt=dt,
        max_steps=max_steps,
        n=N,
        action_mode="accel",
        max_accel=A_MAX,
        thrust_limit="box",
        goal={"hold_point": tuple(HOLD_POINT), "capture_radius": 5.0, "capture_speed": 0.05},
        constraints=({"type": "keep_out", "radius": R},),
    )


def fly(command, x0, dt, t_end, capture=False):
    """Worst certified margin (m), capture and the decision times (s) of one closed loop."""
    dyn = CWDynamics(N)
    dyn.reset(x0)
    x = np.array(x0, dtype=float)
    times = np.linspace(0.0, dt, SAMPLES)
    worst, captured, compute = np.inf, False, []
    for k in range(int(np.ceil(t_end / dt - 1e-9))):
        t0 = time.perf_counter()
        u = command(x)
        compute.append(time.perf_counter() - t0)
        x_new, states = dyn.advance(u, dt, SAMPLES)
        segment = segment_minimum(
            KEEP_OUT, dyn.state_in_last_segment, times, states, float(np.linalg.norm(u)), N, k * dt
        )
        worst = min(worst, segment.lower_bound)
        x = x_new
        near = np.linalg.norm(x[:3] - HOLD_POINT) < 5.0 and np.linalg.norm(x[3:]) < 0.05
        captured = captured or (capture and near)
    return worst, captured, compute


def pd_target(x):
    return -4e-4 * x[:3] - 0.04 * x[3:]


def pd_hold(x):
    return np.clip(-2e-4 * (x[:3] - HOLD_POINT) - 0.03 * x[3:], -A_MAX, A_MAX)


def ecbf_rows(x, poles):
    a, b = KeepOutEcbf(radius=R, poles=poles).row(x, a_cw(x))
    return Rows(a[None], np.array([b]), ("keep_out",))


def braking_row(x, gamma):
    """First-order row on the braking barrier; no condition on u while not closing."""
    rho = np.linalg.norm(x[:3])
    rhat = x[:3] / rho
    rho_dot = float(rhat @ x[3:])
    c = max(0.0, -rho_dot)
    if c == 0.0:
        return Rows(np.zeros((0, 3)), np.zeros(0), ())
    cent = (x[3:] @ x[3:] - rho_dot**2) / rho
    h_b = rho - R - c * c / (2 * A_B)
    b = A_B * (-gamma * h_b + c) / c - cent - rhat @ a_cw(x)
    return Rows(rhat[None], np.array([b]), ("braking",))


def through_filter(safety_filter):
    def command(x):
        return safety_filter.filter(x.copy(), pd_target(x)).action

    return command


def table(label, runs, expected):
    worst = np.array([w for w, _, _ in runs])
    compute = np.concatenate([c for _, _, c in runs]) * 1e3
    violations = int(np.sum(worst < -KEEP_OUT.resolution))
    print(
        f"   {label:34s} violations {violations:3d}/{len(runs)}, worst {worst.min():+9.4f} m, "
        f"compute median {np.median(compute):6.3f} ms, p95 {np.percentile(compute, 95):6.3f} ms"
    )
    assert violations == expected, f"{label}: {violations} violations, expected {expected}"


def ecbf_starts(p, count, rng):
    starts = []
    while len(starts) < count:
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        h0 = rng.uniform(0.0, 150.0)
        v = rng.uniform(-0.3, 0.3, size=3)
        if d @ v + p * h0 >= 0.0:
            starts.append(np.concatenate([(R + h0) * d, v]))
    return starts


def region(rng, count):
    d = rng.normal(size=(count, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    h0 = rng.uniform(0.0, 150.0, size=count)
    v = rng.uniform(-0.3, 0.3, size=(count, 3))
    return np.hstack([(R + h0)[:, None] * d, v])


def unbounded_table():
    print("1. exponential barrier rows without a thrust limit, 100 starts, PD nominal, 1500 s")
    rng = np.random.default_rng(1903)
    starts = {p: ecbf_starts(p, 100, rng) for p in (0.05, 0.1)}
    expected = {(0.05, 1.0): 0, (0.05, 5.0): 0, (0.05, 10.0): 0}
    expected |= {(0.1, 1.0): 0, (0.1, 5.0): 0, (0.1, 10.0): 100}
    for (p, dt), count in expected.items():

        def command(x, p=p):
            return project(pd_target(x), ecbf_rows(x, (p, p))).u

        runs = [fly(command, x0, dt, 1500.0) for x0 in starts[p]]
        table(f"poles {p:4.2f} 1/s, dt {dt:4.1f} s", runs, count)


def box_table():
    print("2. box 0.02 m/s^2, 40 starts inside each filter's set, PD nominal, 2000 s")
    rng = np.random.default_rng(1903)
    region(rng, 200000)  # drawn first, as in the reference runs, to keep the stream in step
    starts = region(rng, 5000)

    def first_40(inside):
        chosen = [x for x in starts if inside(x)][:40]
        assert len(chosen) == 40
        return chosen

    def ecbf_start(x):
        return KeepOutEcbf(radius=R, poles=(0.05, 0.05)).contains(x)

    braking = BrakingBarrier(radius=R, a_b=A_B, rho_cap=2000.0, v_cap=1.0)
    for dt, count in ((1.0, 0), (5.0, 0), (10.0, 0), (20.0, 40)):

        def command(x):
            return project(np.clip(pd_target(x), -A_MAX, A_MAX), ecbf_rows(x, (0.05, 0.05)), BOX).u

        runs = [fly(command, x0, dt, 2000.0) for x0 in first_40(ecbf_start)]
        table(f"barrier row 0.05 1/s, dt {dt:4.1f} s", runs, count)
    for dt in (1.0, 5.0, 10.0, 20.0):

        def command(x):
            return project(np.clip(pd_target(x), -A_MAX, A_MAX), braking_row(x, 0.02), BOX).u

        runs = [fly(command, x0, dt, 2000.0) for x0 in first_40(braking.contains)]
        table(f"braking row 0.02 1/s, dt {dt:4.1f} s", runs, 40)
    for a_b, v_cap, label in ((A_B, 1.0, "a_b 0.01044"), (None, 0.6, "default a_b")):
        for dt in (1.0, 5.0, 10.0, 20.0):
            backup = BackupFilter(barrier="braking", a_b=a_b, rho_cap=2000.0, v_cap=v_cap)
            backup.reset(view(dt, int(2000.0 / dt)))
            runs = []
            for x0 in first_40(backup.contains):
                backup.reset(view(dt, int(2000.0 / dt)))
                runs.append(fly(through_filter(backup), x0, dt, 2000.0))
            table(f"braking backup, {label}, dt {dt:4.1f} s", runs, 0)


def drift_minimiser(x, tau, phi_r):
    """The drift minimiser of the reference runs, which decides which states they draw."""
    rho = np.linalg.norm(phi_r @ x, axis=1)
    k = int(np.argmin(rho))
    found = minimize_scalar(
        lambda t: np.linalg.norm(cw_discrete(N, t)[0][:3] @ x),
        bounds=(tau[max(k - 1, 0)], tau[min(k + 1, len(tau) - 1)]),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return float(found.x)


def passive_table():
    print("3. free-drift barrier, 400 approach-vbar starts, dt 10 s, box 0.02 m/s^2, 3600 s")
    rng = np.random.default_rng(1903)
    barrier = PassiveBarrier(radius=R, n=N)
    t_cap = barrier.t_cap
    tau = np.arange(0.0, t_cap + 5.0, 5.0)
    phi_r = cw_discrete(N, tau)[0][:, :3, :]
    # The reference runs first draw five states for a gradient check, rejecting those whose drift
    # minimiser lies within 50 s of the start or in the last tenth of the cap.
    for _ in range(5):
        while True:
            x = np.concatenate([rng.uniform(-300, 300, 3), rng.uniform(-0.2, 0.2, 3)])
            t_end, _ = barrier.horizon(x)
            m = int(np.searchsorted(tau, t_end)) + 1
            if 50.0 < drift_minimiser(x, tau[:m], phi_r[:m]) < 0.9 * t_cap:
                break
    starts = np.column_stack(
        [
            rng.uniform(-200, 200, 400),
            rng.uniform(200, 600, 400),
            rng.uniform(-50, 50, 400),
            rng.uniform(-0.1, 0.1, (400, 3)),
        ]
    )
    certified = np.array([barrier.value(x) >= 0.0 for x in starts])
    dense = cw_discrete(N, np.arange(0.0, t_cap + 1.0, 1.0))[0][:, :3, :]
    for x in starts[certified]:
        t_end, _ = barrier.horizon(x)
        rho = np.linalg.norm(dense[: int(t_end) + 1] @ x, axis=1)
        assert rho.min() >= R, "a certified start enters the keep-out in free drift"
    print(f"   certified passively safe: {int(certified.sum())} of 400, all confirmed at 1 s")
    assert certified.sum() >= 393

    backup = BackupFilter(barrier="passive")
    runs = []
    for x0 in starts[certified][:40]:
        backup.reset(view(10.0, 360))

        def command(x):
            return backup.filter(x.copy(), pd_hold(x)).action

        runs.append(fly(command, x0, 10.0, 3600.0, capture=True))
    table("passive backup, dt 10.0 s", runs, 0)
    captured = sum(c for _, c, _ in runs)
    print(f"   reached the hold point: {captured} of 40")
    assert captured == 40


if __name__ == "__main__":
    unbounded_table()
    box_table()
    passive_table()
