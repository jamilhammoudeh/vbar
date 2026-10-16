"""Periodic reference orbits of the CR3BP and their stability.

Richardson's third-order halo approximation as a seed, differential correctors for planar
Lyapunov and halo orbits, pseudo-arclength continuation along a halo family with a check for
jumps to another family, and monodromy and Floquet analysis. Nondimensional synodic states as in
`vbar.dynamics.cr3bp`. Methods from Koon, Lo, Marsden and Ross, Dynamical Systems, the
Three-Body Problem and Space Mission Design, secs. 4.3 to 4.4 and 6.6 to 7.2, with their
corrections: Richardson's b22 is +3 c3 wp / d1 and a32 contains 4 c3 (k a24 - b22) (their sec.
6.6 prints both wrongly), and the correctors keep the crossing-time terms that their sec. 6.7
leaves out.

Halo orbits are named by libration point and by the sign of z where they are farthest from
the smaller primary: "northern" and "southern" swap meaning between Sun-Earth L1 and
Earth-Moon L2, where c3 changes sign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from . import cr3bp

_RTOL = 1e-13
_ATOL = 1e-15


@dataclass(frozen=True)
class RichardsonHalo:
    """Coefficients of Richardson's third-order halo solution about L1 or L2.

    Richardson, "Analytic Construction of Periodic Orbits about the Collinear Points", Celestial
    Mechanics 22, 1980; Koon et al. sec. 6.6. Lengths are in units of gamma, the point's
    distance from the smaller primary; `wp` is the in-plane frequency (Richardson's lambda).
    """

    mu: float
    point: int
    gamma: float
    c2: float
    c3: float
    c4: float
    wp: float
    k: float
    a21: float
    a22: float
    a23: float
    a24: float
    a31: float
    a32: float
    b21: float
    b22: float
    b31: float
    b32: float
    d21: float
    d31: float
    d32: float
    s1: float
    s2: float
    l1: float
    l2: float
    delta: float


def richardson_coefficients(mu: float, point: int) -> RichardsonHalo:
    """Richardson's coefficients at L1 or L2 (`point` 1 or 2) for mass ratio `mu`."""
    gamma = abs(cr3bp.lagrange_points(mu)[point - 1, 0] - (1.0 - mu))
    c2, c3, c4 = (cr3bp.collinear_coefficient(mu, point, n) for n in (2, 3, 4))
    _, wp, _, k = cr3bp.libration_frequencies(c2)
    lam = wp
    d1 = 3.0 * lam**2 / k * (k * (6.0 * lam**2 - 1.0) - 2.0 * lam)
    d2 = 8.0 * lam**2 / k * (k * (11.0 * lam**2 - 1.0) - 2.0 * lam)
    a21 = 3.0 * c3 * (k**2 - 2.0) / (4.0 * (1.0 + 2.0 * c2))
    a22 = 3.0 * c3 / (4.0 * (1.0 + 2.0 * c2))
    a23 = -3.0 * c3 * lam / (4.0 * k * d1) * (3.0 * k**3 * lam - 6.0 * k * (k - lam) + 4.0)
    a24 = -3.0 * c3 * lam / (4.0 * k * d1) * (2.0 + 3.0 * k * lam)
    b21 = -3.0 * c3 * lam / (2.0 * d1) * (3.0 * k * lam - 4.0)
    b22 = 3.0 * c3 * lam / d1
    d21 = -c3 / (2.0 * lam**2)
    f = 9.0 * lam**2 + 1.0
    a31 = -9.0 * lam / (4.0 * d2) * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k**2)) + (
        f - c2
    ) / (2.0 * d2) * (3.0 * c3 * (2.0 * a23 - k * b21) + c4 * (2.0 + 3.0 * k**2))
    a32 = (
        -(
            9.0 * lam / 4.0 * (4.0 * c3 * (k * a24 - b22) + k * c4)
            + 1.5 * (f - c2) * (c3 * (k * b22 + d21 - 2.0 * a24) - c4)
        )
        / d2
    )
    b31 = (
        3.0
        / (8.0 * d2)
        * (
            8.0 * lam * (3.0 * c3 * (k * b21 - 2.0 * a23) - c4 * (2.0 + 3.0 * k**2))
            + (f + 2.0 * c2) * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k**2))
        )
    )
    b32 = (
        9.0 * lam * (c3 * (k * b22 + d21 - 2.0 * a24) - c4)
        + 3.0 / 8.0 * (f + 2.0 * c2) * (4.0 * c3 * (k * a24 - b22) + k * c4)
    ) / d2
    d31 = 3.0 / (64.0 * lam**2) * (4.0 * c3 * a24 + c4)
    d32 = 3.0 / (64.0 * lam**2) * (4.0 * c3 * (a23 - d21) + c4 * (4.0 + k**2))
    scale = 1.0 / (2.0 * lam * (lam * (1.0 + k**2) - 2.0 * k))
    s1 = scale * (
        1.5 * c3 * (2.0 * a21 * (k**2 - 2.0) - a23 * (k**2 + 2.0) - 2.0 * k * b21)
        - 3.0 / 8.0 * c4 * (3.0 * k**4 - 8.0 * k**2 + 8.0)
    )
    s2 = scale * (
        1.5 * c3 * (2.0 * a22 * (k**2 - 2.0) + a24 * (k**2 + 2.0) + 2.0 * k * b22 + 5.0 * d21)
        + 3.0 / 8.0 * c4 * (12.0 - k**2)
    )
    l1 = (
        -1.5 * c3 * (2.0 * a21 + a23 + 5.0 * d21)
        - 3.0 / 8.0 * c4 * (12.0 - k**2)
        + 2.0 * lam**2 * s1
    )
    l2 = 1.5 * c3 * (a24 - 2.0 * a22) + 9.0 / 8.0 * c4 + 2.0 * lam**2 * s2
    return RichardsonHalo(
        mu, point, gamma, c2, c3, c4, wp, k, a21, a22, a23, a24, a31, a32, b21, b22, b31, b32,
        d21, d31, d32, s1, s2, l1, l2, lam**2 - c2,
    )  # fmt: skip


def richardson_halo(mu: float, point: int, az: float, branch: int = 1) -> tuple[np.ndarray, float]:
    """Seed state at the x-z plane crossing and period of a halo orbit, third order.

    `az` is the out-of-plane amplitude in nondimensional length; `branch` is Richardson's m,
    1 or 3, the two mirror-image families. The in-plane amplitude follows from the amplitude
    constraint l1 Ax^2 + l2 Az^2 + Delta = 0, so `az` must be large enough for a halo to exist.
    """
    h = richardson_coefficients(mu, point)
    az_n = az / h.gamma
    ax2 = -(h.l2 * az_n**2 + h.delta) / h.l1
    if ax2 <= 0.0:
        raise ValueError("no halo orbit has this out-of-plane amplitude")
    ax = math.sqrt(ax2)
    delta_m = 2.0 - branch
    nu = 1.0 + h.s1 * ax**2 + h.s2 * az_n**2
    rate = nu * h.wp
    x = (
        h.a21 * ax**2
        + h.a22 * az_n**2
        - ax
        + h.a23 * ax**2
        - h.a24 * az_n**2
        + h.a31 * ax**3
        - h.a32 * ax * az_n**2
    )
    z = delta_m * (az_n - 2.0 * h.d21 * ax * az_n + h.d32 * az_n * ax**2 - h.d31 * az_n**3)
    vy = rate * (
        h.k * ax
        + 2.0 * (h.b21 * ax**2 - h.b22 * az_n**2)
        + 3.0 * (h.b31 * ax**3 - h.b32 * ax * az_n**2)
    )
    origin = 1.0 - mu - h.gamma if point == 1 else 1.0 - mu + h.gamma
    state = np.array([origin + h.gamma * x, 0.0, h.gamma * z, 0.0, h.gamma * vy, 0.0])
    return state, 2.0 * math.pi / rate


def _variational_rhs(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    """State and transition-matrix derivatives, written out in scalars.

    The same equations as `cr3bp.rhs` and `cr3bp.jacobian`, five times faster; the correctors
    spend nearly all their time here.
    """
    x, yy, z, vx, vy, vz = y[0:6].tolist()
    d1, d2 = x + mu, x - 1.0 + mu
    rr = yy * yy + z * z
    r1s, r2s = d1 * d1 + rr, d2 * d2 + rr
    k1 = (1.0 - mu) / (r1s * math.sqrt(r1s))
    k2 = mu / (r2s * math.sqrt(r2s))
    k = k1 + k2
    q1, q2 = 3.0 * k1 / r1s, 3.0 * k2 / r2s
    q = q1 + q2
    hxy = (q1 * d1 + q2 * d2) * yy
    hxz = (q1 * d1 + q2 * d2) * z
    hessian = np.array(
        [
            [1.0 - k + q1 * d1 * d1 + q2 * d2 * d2, hxy, hxz],
            [hxy, 1.0 - k + q * yy * yy, q * yy * z],
            [hxz, q * yy * z, -k + q * z * z],
        ]
    )
    phi = y[6:42].reshape(6, 6)
    dy = np.empty(42)
    dy[0:6] = (vx, vy, vz, x + 2.0 * vy - k1 * d1 - k2 * d2, yy - 2.0 * vx - k * yy, -k * z)
    dphi = dy[6:42].reshape(6, 6)
    dphi[0:3] = phi[3:6]
    dphi[3:6] = hessian @ phi[0:3]
    dphi[3] += 2.0 * phi[4]
    dphi[4] -= 2.0 * phi[3]
    return dy


class _PlaneCrossing:
    """Event function for scipy: y returning to 0, from the side the start's vy points to."""

    terminal = True

    def __init__(self, vy: float):
        self.direction = -math.copysign(1.0, vy)

    def __call__(self, t: float, y: np.ndarray, mu: float) -> float:
        return y[1]


def half_period_crossing(state: ArrayLike, mu: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Time, state and transition matrix at the return to the x-z plane of a state on it.

    Event location refines the crossing to rounding. The start itself is not a crossing in
    the direction looked for.
    """
    state = np.asarray(state, dtype=float)
    sol = solve_ivp(
        _variational_rhs, (0.0, 20.0), np.concatenate([state, np.eye(6).ravel()]),
        method="DOP853", rtol=_RTOL, atol=_ATOL, args=(mu,), events=_PlaneCrossing(state[4]),
    )  # fmt: skip
    if not sol.t_events[0].size:
        raise RuntimeError("no crossing of the x-z plane found")
    end = sol.y_events[0][0]
    return float(sol.t_events[0][0]), end[:6], end[6:42].reshape(6, 6)


def _crossing_jacobian(end: np.ndarray, phi: np.ndarray, mu: float, free: list[int]) -> np.ndarray:
    """d(vx, vz) at the crossing / d(free initial components), with the crossing-time terms.

    The crossing time moves with the start: dt = -phi[1] / vy, so d(vx) = phi[3] - (ax / vy)
    phi[1] (Koon et al. sec. 4.4); likewise for vz.
    """
    accel = cr3bp.acceleration(end[:3], end[3:], mu)
    rows = [phi[3] - accel[0] / end[4] * phi[1], phi[5] - accel[2] / end[4] * phi[1]]
    return np.array(rows)[:, free]


@dataclass(frozen=True)
class PeriodicOrbit:
    """A symmetric periodic orbit: its state on the x-z plane, period and Jacobi constant."""

    state: np.ndarray
    period: float
    jacobi: float
    iterations: int


def correct_lyapunov(state: ArrayLike, mu: float, tol: float = 1e-12) -> PeriodicOrbit:
    """Planar Lyapunov orbit through x0 of `state`, correcting vy0 (Koon et al. sec. 4.4)."""
    s = np.array(state, dtype=float)
    for iteration in range(1, 21):
        t, end, phi = half_period_crossing(s, mu)
        if abs(end[3]) < tol:
            return PeriodicOrbit(s, 2.0 * t, float(cr3bp.jacobi_constant(s, mu)), iteration - 1)
        s[4] -= end[3] / _crossing_jacobian(end, phi, mu, [4])[0, 0]
    raise RuntimeError("lyapunov corrector did not converge")


def correct_halo(
    state: ArrayLike, mu: float, fixed: str = "z", tol: float = 1e-12
) -> PeriodicOrbit:
    """Halo orbit near `state`, holding x0 (`fixed="x"`) or z0 (`"z"`) and correcting the rest.

    Drives vx and vz at the next x-z plane crossing to zero; the orbit is then symmetric about
    that plane. Holding z0 fails where the family turns back in z0, as it does on the way from
    small Earth-Moon L2 halos to the near-rectilinear ones; `continue_halo` handles that.
    """
    s = np.array(state, dtype=float)
    free = {"z": [0, 4], "x": [2, 4]}[fixed]
    for iteration in range(1, 21):
        t, end, phi = half_period_crossing(s, mu)
        miss = np.array([end[3], end[5]])
        if np.max(np.abs(miss)) < tol:
            return PeriodicOrbit(s, 2.0 * t, float(cr3bp.jacobi_constant(s, mu)), iteration - 1)
        s[free] -= np.linalg.solve(_crossing_jacobian(end, phi, mu, free), miss)
    raise RuntimeError("halo corrector did not converge")


class FamilyJump(RuntimeError):
    """Continuation left the family it was following."""


def family_jump(
    previous: PeriodicOrbit, current: PeriodicOrbit, step: float, slope: float | None = None
) -> bool:
    """Whether `current` is not the neighbour of `previous` on one family.

    Along a family the start state and the period change smoothly: a step of `step` in the
    start state moves it by about `step`, and the period by about `slope` (its rate of change
    on the previous step) times `step`. A converged orbit that breaks either belongs to another
    family: continuing the Earth-Moon L2 halo family in z0, as Koon et al. do in sec. 6.7,
    lands past the family's largest z0 on an unrelated orbit of about twice the period.
    Without a slope, a period change of a fifth counts as a jump.
    """
    moved = float(np.linalg.norm(current.state - previous.state))
    change = abs(current.period - previous.period)
    if slope is None:
        expected = 0.2 * previous.period
    else:
        expected = 3.0 * slope * step + 0.01 * previous.period
    return moved > 2.0 * step or change > expected


def _tangent(orbit: PeriodicOrbit, mu: float, reference: np.ndarray) -> np.ndarray:
    """Unit tangent of the family at `orbit` in (x0, z0, vy0), on `reference`'s side."""
    _, end, phi = half_period_crossing(orbit.state, mu)
    null = np.linalg.svd(_crossing_jacobian(end, phi, mu, _FAMILY))[2][-1]
    return null if null @ reference > 0.0 else -null


def _arclength_step(
    orbit: PeriodicOrbit, tangent: np.ndarray, step: float, mu: float
) -> PeriodicOrbit:
    """The family member `step` along `tangent` from `orbit`, corrected on the normal plane.

    Raises RuntimeError if the corrector does not converge or has to move more than half a
    step off the prediction: the step is then too long for the family's curvature, and the
    corrector could land on another arm of a folded family.
    """
    predicted = orbit.state.copy()
    predicted[_FAMILY] += step * tangent
    s = predicted.copy()
    for iteration in range(1, 21):
        t, end, phi = half_period_crossing(s, mu)
        offset = tangent @ (s[_FAMILY] - orbit.state[_FAMILY]) - step
        residual = np.array([end[3], end[5], offset])
        if np.max(np.abs(residual)) < 1e-12:
            if np.linalg.norm(s - predicted) > 0.5 * abs(step):
                break
            return PeriodicOrbit(s, 2.0 * t, float(cr3bp.jacobi_constant(s, mu)), iteration - 1)
        full = np.vstack([_crossing_jacobian(end, phi, mu, _FAMILY), tangent])
        s[_FAMILY] -= np.linalg.solve(full, residual)
    raise RuntimeError("continuation step too long for the family's curvature")


_FAMILY = [0, 2, 4]


def continue_halo(
    start: PeriodicOrbit, mu: float, step: float, count: int, direction: ArrayLike = (0, 1, 0)
) -> list[PeriodicOrbit]:
    """Follow a halo family from `start` by pseudo-arclength continuation.

    The unknowns (x0, z0, vy0) satisfy two crossing conditions, so the family is a curve; each
    step moves `step` along its tangent and corrects on the plane normal to it, which passes
    turning points in x0 or z0 where holding either fails. The first step goes to the side of
    `direction` in (x0, z0, vy0); by default z0 grows. Raises `FamilyJump` if a step lands on
    another family.
    """
    orbits = [start]
    tangent = np.asarray(direction, dtype=float)
    slope = None
    for _ in range(count):
        tangent = _tangent(orbits[-1], mu, tangent)
        orbit = _arclength_step(orbits[-1], tangent, step, mu)
        if family_jump(orbits[-1], orbit, step, slope):
            raise FamilyJump(f"left the family after {len(orbits) - 1} steps")
        slope = abs(orbit.period - orbits[-1].period) / step
        orbits.append(orbit)
    return orbits


def halo_by_period(
    start: PeriodicOrbit,
    mu: float,
    period: float,
    step: float = 0.02,
    max_step: float = 0.1,
    direction: ArrayLike = (0, 1, 0),
) -> PeriodicOrbit:
    """The member of `start`'s halo family with nondimensional `period`, to 1e-12.

    Continues as `continue_halo`, doubling the step up to `max_step` while the corrector
    converges in four iterations or fewer and halving it when it fails or lands off the
    family, until the period is passed; then finds the step that lands on it by false position
    (Illinois variant) between the two bracketing members.
    """
    current = start
    tangent = np.asarray(direction, dtype=float)
    slope = None
    for _ in range(1000):
        tangent = _tangent(current, mu, tangent)
        try:
            orbit = _arclength_step(current, tangent, step, mu)
            if family_jump(current, orbit, step, slope):
                raise FamilyJump("landed off the family")
        except RuntimeError:
            if step < 1e-4:
                raise
            step *= 0.5
            continue
        if (orbit.period - period) * (current.period - period) <= 0.0:
            return _land_on_period(current, orbit, tangent, step, period, mu)
        slope = abs(orbit.period - current.period) / step
        current = orbit
        if orbit.iterations <= 4:
            step = min(2.0 * step, max_step)
    raise RuntimeError("period not reached")


def _land_on_period(
    near: PeriodicOrbit,
    far: PeriodicOrbit,
    tangent: np.ndarray,
    step: float,
    period: float,
    mu: float,
) -> PeriodicOrbit:
    lo, f_lo = 0.0, near.period - period
    hi, f_hi = step, far.period - period
    best = far
    for _ in range(40):
        s = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        best = _arclength_step(near, tangent, s, mu)
        f = best.period - period
        if abs(f) < 1e-12:
            return best
        if f * f_hi < 0.0:
            lo, f_lo = hi, f_hi
        else:
            f_lo *= 0.5
        hi, f_hi = s, f
    raise RuntimeError("period not reached")


def monodromy(state: ArrayLike, period: float, mu: float) -> np.ndarray:
    """Transition matrix of a periodic orbit over one period."""
    y0 = np.concatenate([np.asarray(state, dtype=float), np.eye(6).ravel()])
    sol = solve_ivp(
        _variational_rhs, (0.0, period), y0, method="DOP853", rtol=_RTOL, atol=_ATOL, args=(mu,)
    )
    return sol.y[6:42, -1].reshape(6, 6)


def stability_indices(monodromy_matrix: ArrayLike) -> np.ndarray:
    """(lambda + 1 / lambda) / 2 of the eigenvalues, in pairs; |index| > 1 is unstable.

    Eigenvalues of a Hamiltonian monodromy matrix come as lambda, 1 / lambda, plus a pair at 1
    (Koon et al. sec. 4.3).
    """
    eig = np.linalg.eigvals(np.asarray(monodromy_matrix, dtype=float))
    return np.sort_complex(0.5 * (eig + 1.0 / eig))


def floquet_modes(
    state: ArrayLike, period: float, mu: float, times: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues of the monodromy matrix and its periodic Floquet modes at `times`.

    e_i(t) = exp(-(t / T) log lambda_i) phi(t, 0) e_i(0) repeats every period (Koon et al.
    sec. 8.7). Returns the eigenvalues (6,) and the modes (len(times), 6, 6) as columns.
    """
    times = np.asarray(times, dtype=float)
    y0 = np.concatenate([np.asarray(state, dtype=float), np.eye(6).ravel()])
    sol = solve_ivp(
        _variational_rhs, (0.0, period), y0, method="DOP853", rtol=_RTOL, atol=_ATOL,
        args=(mu,), dense_output=True,
    )  # fmt: skip
    eig, vectors = np.linalg.eig(sol.y[6:42, -1].reshape(6, 6))
    logs = np.log(eig.astype(complex))
    phis = sol.sol(times)[6:42].T.reshape(-1, 6, 6)
    decay = np.exp(-np.outer(times / period, logs))
    return eig, (phis @ vectors) * decay[:, None, :]
