"""Certified bounds against a 50-digit Clohessy-Wiltshire reference, and the rounding they allow.

Run with `python benchmarks/rounding_bounds.py` from the repository root. The segments are the
seeded ones of the tests, coasting and thrusting: the 320 random segments and 200 thrusting flybys
of `tests/test_constraints.py`, the 282 steps of its radial hop, and the 180 speed-type segments
of `tests/test_constraint_types.py`. Each is sampled by `CWDynamics` and checked for `KeepOut`,
`SpeedLimit`, `ClosingSpeedLimit` and `GateTable`:

1. The certified lower bound of `segment_minimum` and of `segment_minima` never exceeds the exact
   margin, from the closed-form solution in 50-digit decimal arithmetic, at any sample or at the
   reported minimum's time.
2. The worst error of a sample's margin, in units of eps times its interval's rounding scale, is
   printed per constraint; the allowance is 2 of these units.
"""

import importlib
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np


def from_tests(name):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    return importlib.import_module(name)


segments_module = from_tests("test_constraints")
types_module = from_tests("test_constraint_types")

from vbar import constraints  # noqa: E402
from vbar.dynamics import CWDynamics, cw_states  # noqa: E402

N = segments_module.N
EPS = np.finfo(float).eps


def exact_state(state, accel, t):
    """State `t` s after `state` under constant `accel`, Hill frame, to 50 digits.

    The zero-order-hold closed form, taking the floats in `state`, `accel`, `t` and N as exact.
    """
    with localcontext() as ctx:
        ctx.prec = 50
        x, y, z, vx, vy, vz = (Decimal(float(c)) for c in state)
        ax, ay, az = (Decimal(float(c)) for c in accel)
        n, t = Decimal(float(N)), Decimal(float(t))
        s, c = segments_module.decimal_sin_cos(n * t)
        k, d, nt, n2 = 1 - c, n * t - s, n * t, n * n
        return (
            (1 + 3 * k) * x + s / n * vx + 2 * k / n * vy + k / n2 * ax + 2 * d / n2 * ay,
            -6 * d * x
            + y
            - 2 * k / n * vx
            + (nt - 4 * d) / n * vy
            - 2 * d / n2 * ax
            + (4 * k / n2 - Decimal(1.5) * t * t) * ay,
            c * z + s / n * vz + k / n2 * az,
            3 * n * s * x + c * vx + 2 * s * vy + s / n * ax + 2 * k / n * ay,
            -6 * n * k * x - 2 * s * vx + (1 - 4 * k) * vy - 2 * k / n * ax + (nt - 4 * d) / n * ay,
            -n * s * z + c * vz + s / n * az,
        )


def gate_speed(gates, rho):
    """The gate table's allowed speed at `rho`, interpolated in decimal."""
    ranges, speeds = (tuple(Decimal(float(q)) for q in column) for column in gates._table())
    if rho <= ranges[0]:
        return speeds[0]
    if rho >= ranges[-1]:
        return speeds[-1]
    j = next(i for i in range(len(ranges) - 1) if rho <= ranges[i + 1])
    return speeds[j] + (rho - ranges[j]) * (speeds[j + 1] - speeds[j]) / (ranges[j + 1] - ranges[j])


def exact_margin(constraint, e):
    """`constraint.margin` of the decimal state `e`, to 50 digits."""
    with localcontext() as ctx:
        ctx.prec = 50
        x, y, z, vx, vy, vz = e
        rho = (x * x + y * y + z * z).sqrt()
        if isinstance(constraint, constraints.KeepOut):
            return rho - Decimal(float(constraint.radius))
        if isinstance(constraint, constraints.SpeedLimit):
            allowed = Decimal(float(constraint.v0)) + Decimal(float(constraint.v1)) * rho
            return allowed - (vx * vx + vy * vy + vz * vz).sqrt()
        closing = max(Decimal(0), -(x * vx + y * vy + z * vz) / rho)
        if isinstance(constraint, constraints.ClosingSpeedLimit):
            allowed = Decimal(float(constraint.v0)) + Decimal(float(constraint.v1)) * rho
            return allowed - closing
        return gate_speed(constraint, rho) - closing


def segments():
    rows = [tuple(s) for s in segments_module.random_segments()]
    rows += [(*f[:3], 20) for f in segments_module.flyby_tables()["thrusting"]]
    rows += [(s, np.zeros(3), 10.0, 11) for s in segments_module.radial_hop()[0]]
    rows += [tuple(s) for s in types_module.speed_type_segments()]
    return rows


def check(constraint_list, rows):
    worst = {c.name: 0.0 for c in constraint_list}
    checked = 0
    for state, accel, duration, samples in rows:
        model = CWDynamics(N)
        model.reset(state)
        _, sampled = model.advance(accel, duration, n_samples=samples)
        # The model keys its matrices to the nanosecond, so its samples are at these times.
        times = np.linspace(0.0, round(duration, 9), samples)
        thrust = float(np.linalg.norm(accel))
        h = float(times[1])
        speed, accel_bound = constraints.interval_bounds(sampled, h, thrust, N)
        rho_min = constraints.range_lower_bounds(sampled, h, speed)
        exact_states = [exact_state(state, accel, t) for t in times]

        def batched_state_at(_rows, tau, state=state, accel=accel):
            return cw_states(
                N, tau, np.repeat(state[None], len(tau), 0), np.repeat(accel[None], len(tau), 0)
            )

        for c in constraint_list:
            exact = [exact_margin(c, e) for e in exact_states]
            found = constraints.segment_minimum(
                c, model.state_in_last_segment, times, sampled, thrust, N
            )
            t_batch, _, bound_batch = constraints.segment_minima(
                c, batched_state_at, times, sampled[None], np.array([thrust]), N, np.zeros(1)
            )
            # The scalar refinement reads the model, whose matrices are keyed to the nanosecond.
            at_found = exact_margin(c, exact_state(state, accel, round(found.t, 9)))
            at_batch = exact_margin(c, exact_state(state, accel, float(t_batch[0])))
            ceiling = min(min(exact), at_found, at_batch)
            for bound in (found.lower_bound, float(bound_batch[0])):
                if np.isfinite(bound):
                    assert Decimal(bound) <= ceiling, (c.name, state, accel, duration, samples)
            allowance = constraints._rounding_allowances(
                c, float(times[-1]), sampled, speed, accel_bound, rho_min
            )
            scale = allowance / (constraints._ROUNDING_ULPS * EPS)
            margins = c.margin(sampled)
            error = np.array(
                [abs(float(Decimal(float(m)) - e)) for m, e in zip(margins, exact, strict=True)]
            )
            ratio = np.maximum(error[:-1], error[1:]) / (EPS * scale)
            finite = np.isfinite(scale)
            if finite.any():
                worst[c.name] = max(worst[c.name], float(ratio[finite].max()))
            checked += 1
    return worst, checked


if __name__ == "__main__":
    start = time.perf_counter()
    rows = segments()
    constraint_list = [
        segments_module.KEEP_OUT,
        types_module.SPEED,
        types_module.CLOSING,
        types_module.GATES,
    ]
    worst, checked = check(constraint_list, rows)
    samples = sum(r[3] for r in rows)
    print(
        f"{len(rows)} segments, {samples} samples, {checked} segment-constraint pairs: "
        "no certified bound above the reference"
    )
    for name, ratio in worst.items():
        print(f"worst sample error, {name:19s}: {ratio:.3f} eps x scale (allowance 2)")
    print(f"{time.perf_counter() - start:.1f} s")
