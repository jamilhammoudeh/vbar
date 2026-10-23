"""Error sets on the Hill-frame state and the margins they add to safety constraints.

An error set E bounds e = x - x_hat, the difference between the true state and the state a
filter acts on (m and m/s, Hill frame, order x, y, z, vx, vy, vz). A row a . x >= b holds for
every x in x_hat + E when a . x_hat >= b + h_E(a), with the support function h_E(a) = max over
e in E of a . e (Boyd and Vandenberghe, *Convex Optimization*, sec. 4.3.1). Every set here is
symmetric, so h_E(-a) = h_E(a).

The tube and delay margins bound the error a disturbance builds up over many steps of a linear
loop, with and without feedback, and the sampled margins bound it from draws of its law.
"""

from __future__ import annotations

import itertools
import math
import operator
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

from ..reach import LinearLoop, Zonotope, cw_loop, support

# A covariance may be asymmetric or have negative eigenvalues by this fraction of its largest
# entry, the rounding of a propagated covariance; beyond it the matrix is refused.
_COV_TOL = 1e-12
_VERTICES = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))
_VERTICES.flags.writeable = False
# A loop whose spectral radius rounds to just below 1 may never contract; stop the search here.
_MAX_TUBE_STEPS = 100_000
# A delay counts as a whole number of steps when delay / dt is that close to an integer, the
# rounding of a delay written as a decimal multiple of dt.
_STEP_TOL = 1e-9


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def _covariance(cov) -> np.ndarray:
    """Return a validated 6x6 covariance with its eigenvalues clipped at zero."""
    cov = np.array(cov, dtype=float)
    if cov.shape != (6, 6) or not np.all(np.isfinite(cov)):
        raise ValueError("cov must be a finite 6x6 matrix")
    scale = float(np.max(np.abs(cov)))
    if np.max(np.abs(cov - cov.T)) > _COV_TOL * scale:
        raise ValueError("cov must be symmetric")
    cov = 0.5 * (cov + cov.T)
    eigenvalues, vectors = np.linalg.eigh(cov)
    if eigenvalues[0] < -_COV_TOL * scale:
        raise ValueError("cov must be positive semidefinite")
    return (vectors * np.clip(eigenvalues, 0.0, None)) @ vectors.T


@dataclass(frozen=True, kw_only=True)
class BoxError:
    """Per-axis bounds |e_i| <= r_pos on position and |e_i| <= r_vel on velocity.

    Parameters
    ----------
    r_pos : float
        Bound on each position component, m, nonnegative.
    r_vel : float
        Bound on each velocity component, m/s, nonnegative.

    Raises
    ------
    ValueError
        If a bound is negative or not finite.
    """

    r_pos: float
    r_vel: float

    def __post_init__(self):
        object.__setattr__(self, "r_pos", _nonnegative(self.r_pos, "r_pos"))
        object.__setattr__(self, "r_vel", _nonnegative(self.r_vel, "r_vel"))


@dataclass(frozen=True, kw_only=True)
class BallError:
    """Separate balls |e_pos|_2 <= r_pos and |e_vel|_2 <= r_vel.

    Parameters
    ----------
    r_pos : float
        Radius of the position error ball, m, nonnegative.
    r_vel : float
        Radius of the velocity error ball, m/s, nonnegative.

    Raises
    ------
    ValueError
        If a radius is negative or not finite.
    """

    r_pos: float
    r_vel: float

    def __post_init__(self):
        object.__setattr__(self, "r_pos", _nonnegative(self.r_pos, "r_pos"))
        object.__setattr__(self, "r_vel", _nonnegative(self.r_vel, "r_vel"))


@dataclass(frozen=True, kw_only=True)
class EllipsoidError:
    """The ellipsoid E = {P z : |z|_2 <= 1}.

    Parameters
    ----------
    P : array_like, shape (6, 6)
        The shape matrix, rows in m (position) and m/s (velocity); singular for a flat
        ellipsoid.
    exceedance : float
        The probability per draw that the error lies outside E, in [0, 1]; 0 for a hard bound.

    Raises
    ------
    ValueError
        If `P` is not a finite 6x6 matrix or `exceedance` is outside [0, 1].
    """

    P: np.ndarray
    exceedance: float = 0.0

    def __post_init__(self):
        P = np.array(self.P, dtype=float)
        if P.shape != (6, 6) or not np.all(np.isfinite(P)):
            raise ValueError("P must be a finite 6x6 matrix")
        exceedance = float(self.exceedance)
        if not 0.0 <= exceedance <= 1.0:
            raise ValueError("exceedance must be in [0, 1]")
        P.flags.writeable = False
        object.__setattr__(self, "P", P)
        object.__setattr__(self, "exceedance", exceedance)

    @classmethod
    def from_gaussian(cls, cov, k: float) -> EllipsoidError:
        """Build the k-sigma ellipsoid of a zero-mean Gaussian error.

        Parameters
        ----------
        cov : array_like, shape (6, 6)
            The error covariance S, symmetric positive semidefinite, m^2, m^2/s and m^2/s^2.
        k : float
            The size of the ellipsoid in standard deviations, positive.

        Returns
        -------
        EllipsoidError
            P = k S^(1/2), the set {e : e^T S^-1 e <= k^2} for a nonsingular S, with
            `exceedance` = P(chi-square with 6 degrees of freedom > k^2).

        Raises
        ------
        ValueError
            If `cov` is not a symmetric positive semidefinite 6x6 matrix or `k` is not positive.

        Notes
        -----
        e^T S^-1 e of a Gaussian error is chi-square with 6 degrees of freedom. For a singular S
        it has as many degrees of freedom as the rank of S, and its tail beyond k^2 is smaller,
        so the stated exceedance is an upper bound then.
        """
        k = _positive(k, "k")
        eigenvalues, vectors = np.linalg.eigh(_covariance(cov))
        root = (vectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ vectors.T
        return cls(P=k * root, exceedance=float(chi2.sf(k**2, 6)))


ErrorSet = BoxError | BallError | EllipsoidError


def halfspace_margin(a, error: ErrorSet) -> float:
    """Return the support function h_E(a), the tightening of a row a . x >= b.

    Parameters
    ----------
    a : array_like, shape (6,)
        The row normal on the state, in row units per m (position) and per m/s (velocity).
    error : BoxError, BallError or EllipsoidError
        The error set E.

    Returns
    -------
    float
        max over e in E of a . e, in row units: r_pos |a_pos|_1 + r_vel |a_vel|_1 for a box,
        r_pos |a_pos|_2 + r_vel |a_vel|_2 for two balls, |P^T a|_2 for an ellipsoid
        (Boyd and Vandenberghe, sec. 4.3.1 and eq. 4.37).

    Raises
    ------
    ValueError
        If `a` is not a finite vector of length 6.
    TypeError
        If `error` is not one of the error sets.
    """
    a = np.asarray(a, dtype=float)
    if a.shape != (6,) or not np.all(np.isfinite(a)):
        raise ValueError("a must be a finite vector of length 6")
    match error:
        case BoxError():
            return error.r_pos * float(np.sum(np.abs(a[:3]))) + error.r_vel * float(
                np.sum(np.abs(a[3:]))
            )
        case BallError():
            return error.r_pos * float(np.linalg.norm(a[:3])) + error.r_vel * float(
                np.linalg.norm(a[3:])
            )
        case EllipsoidError():
            return float(np.linalg.norm(error.P.T @ a))
    raise TypeError(f"unknown error set {type(error).__name__}")


def chance_margin(a, cov, eta: float) -> float:
    """Return the margin that makes a row hold with probability eta under Gaussian error.

    Parameters
    ----------
    a : array_like, shape (6,)
        The row normal on the state, in row units per m and per m/s.
    cov : array_like, shape (6, 6)
        The covariance S of the zero-mean Gaussian error, symmetric positive semidefinite.
    eta : float
        The required probability, in [0.5, 1).

    Returns
    -------
    float
        Phi^-1(eta) |S^(1/2) a|_2, in row units: a . x >= b holds with probability eta when
        a . x_hat >= b plus this margin (Boyd and Vandenberghe, eq. 4.38).

    Raises
    ------
    ValueError
        If `eta` is outside [0.5, 1), where the constraint is not convex or cannot be met, or
        `a` or `cov` is malformed.
    """
    eta = float(eta)
    if not 0.5 <= eta < 1.0:
        raise ValueError("eta must be in [0.5, 1)")
    a = np.asarray(a, dtype=float)
    if a.shape != (6,) or not np.all(np.isfinite(a)):
        raise ValueError("a must be a finite vector of length 6")
    variance = max(0.0, float(a @ _covariance(cov) @ a))
    return float(norm.ppf(eta)) * math.sqrt(variance)


def keep_out_margin(phi_r, error: ErrorSet) -> float:
    """Return the largest range change max over e in E of |Phi_r e|_2.

    The range |r| is 1-Lipschitz in r and the Clohessy-Wiltshire map is linear, so with
    r = Phi_r x the true range is at least the range computed from x_hat minus this margin.

    Parameters
    ----------
    phi_r : array_like, shape (k, 6)
        The rows of the state transition matrix that give position, in m per m and m per m/s.
    error : BoxError, BallError or EllipsoidError
        The error set E.

    Returns
    -------
    float
        The margin in m. Exact for a box (the maximum over its 64 vertices, where a convex
        function on a polytope peaks) and for an ellipsoid (sigma_max(Phi_r P)). For two balls it
        is the bound r_pos sigma_max(Phi_r,pos) + r_vel sigma_max(Phi_r,vel), exact when the
        two blocks share their top left singular vector and otherwise above the maximum.

    Raises
    ------
    ValueError
        If `phi_r` is not a finite matrix with 6 columns.
    TypeError
        If `error` is not one of the error sets.
    """
    phi_r = np.asarray(phi_r, dtype=float)
    if phi_r.ndim != 2 or phi_r.shape[1] != 6 or not np.all(np.isfinite(phi_r)):
        raise ValueError("phi_r must be a finite matrix with 6 columns")
    match error:
        case BoxError():
            radii = np.array([error.r_pos] * 3 + [error.r_vel] * 3)
            return float(np.max(np.linalg.norm((_VERTICES * radii) @ phi_r.T, axis=1)))
        case BallError():
            return error.r_pos * float(np.linalg.norm(phi_r[:, :3], 2)) + error.r_vel * float(
                np.linalg.norm(phi_r[:, 3:], 2)
            )
        case EllipsoidError():
            return float(np.linalg.norm(phi_r @ error.P, 2))
    raise TypeError(f"unknown error set {type(error).__name__}")


def ecbf_bias(da_max: float, poles: tuple[float, float]) -> float:
    """Return the depth of the riding offset of an exponential CBF under a persistent bias.

    Parameters
    ----------
    da_max : float
        The largest unmodelled acceleration along the barrier gradient, m/s^2, nonnegative.
    poles : tuple of float
        The two real closed-loop poles p1, p2 of the barrier, 1/s, positive.

    Returns
    -------
    float
        da_max / (p1 p2), in m: h'' + (p1 + p2) h' + p1 p2 h = -da_max settles at
        h = -da_max / (p1 p2), so the barrier must be moved out by this amount.

    Raises
    ------
    ValueError
        If `da_max` is negative or a pole is not positive.
    """
    da_max = _nonnegative(da_max, "da_max")
    if len(poles) != 2:
        raise ValueError("poles must be two values")
    p1, p2 = (_positive(p, "poles") for p in poles)
    return da_max / (p1 * p2)


def first_order_bias(da_max: float, gamma: float) -> float:
    """Return the depth of the riding offset of a first-order CBF under a persistent bias.

    Parameters
    ----------
    da_max : float
        The largest unmodelled rate of the barrier value, in barrier units per s, nonnegative.
    gamma : float
        The class-K gain of h' >= -gamma h, 1/s, positive.

    Returns
    -------
    float
        da_max / gamma: h' = -gamma h - da_max settles at h = -da_max / gamma.

    Raises
    ------
    ValueError
        If `da_max` is negative or `gamma` is not positive.
    """
    return _nonnegative(da_max, "da_max") / _positive(gamma, "gamma")


def _count(value, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    count = operator.index(value)
    if count < 1:
        raise ValueError(f"{name} must be at least 1")
    return count


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return value


def box_hull(B, W) -> np.ndarray:
    """Return the half-widths of the smallest box that contains the image B W.

    Parameters
    ----------
    B : array_like, shape (n, m)
        How the disturbance enters the state; for the CW step Gamma, in m per m/s^2 (position
        rows) and m/s per m/s^2 (velocity rows).
    W : Zonotope, Polytope or ActuatorSet
        The disturbance set in the units `B` acts on; an `ActuatorSet` needs m = 3.

    Returns
    -------
    numpy.ndarray, shape (n,)
        h_BW(e_i) = h_W(B^T e_i) on each state axis, in state units. For W symmetric about the
        origin these are the half-widths of the box hull of B W.

    Raises
    ------
    ValueError
        If `B` is not a finite matrix or `W` does not have m dimensions.

    Notes
    -----
    A disturbance that enters through the thrusters, Gamma W with Gamma of shape (6, 3), has rank
    3 in six dimensions and no interior, so no power of a stable A_K maps it into alpha Gamma W,
    and the contraction test of Rawlings, Mayne and Diehl, sec. 3.5.3 (Example 3.13), never
    passes. Its box hull contains it and is full-dimensional, which `tube_margin` needs.
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or not np.all(np.isfinite(B)):
        raise ValueError("B must be a finite matrix")
    dim = B.shape[0]
    # One step of x+ = B w from the origin reaches exactly B W.
    one_step = LinearLoop(np.zeros((dim, dim)), B, np.zeros(dim))
    return np.array([support(one_step, np.zeros(dim), W, e, 1) for e in np.eye(dim)])


def tube_margin(A_K, w_bar, a, alpha: float = 0.5) -> float:
    """Return a bound on the support of the infinite tube of a stable loop under a box disturbance.

    The tube S = W + A_K W + A_K^2 W + ..., with W the box |w_i| <= w_bar_i, holds every error
    that x+ = A_K x + w builds from zero. Its support along `a` is
    phi_inf = sum over j >= 0 of |a . A_K^j| w_bar; phi_N is the sum of the first N terms.

    Parameters
    ----------
    A_K : array_like, shape (k, k)
        The closed-loop matrix A + B K of the feedback the tube assumes.
    w_bar : array_like, shape (k,)
        Half-widths of the disturbance box on the state per step, in state units, all positive;
        `box_hull` gives them for a disturbance that enters through B.
    a : array_like, shape (k,)
        The row normal, in row units per state unit.
    alpha : float
        The contraction to reach, in (0, 1].

    Returns
    -------
    float
        phi_N / (1 - alpha_N) >= phi_inf in row units, at the first N >= 1 with
        alpha_N = max_i (|A_K^N| w_bar)_i / w_bar_i < alpha. Then A_K^N W lies in alpha_N W,
        so S lies in S_N + alpha_N S with S_N the first N terms, and
        h_S(a) <= phi_N + alpha_N h_S(a) (Rawlings, Mayne and Diehl, 2nd ed., sec. 3.5.3 and
        Example 3.13; sec. 5.3.5, p. 347). The attained alpha_N, not the threshold, sets the
        factor. With alpha = 1 the first N at which A_K^N W lies strictly inside W is used; there
        alpha_N < 1, so the margin is finite, and it is the bound from the fewest terms. A row
        a . x <= d holds on the tube around a nominal state that meets a . x <= d minus this
        margin; the box is symmetric, so the same margin serves a . x >= b.

    Raises
    ------
    ValueError
        If the shapes disagree or are not finite, a half-width is not positive, `alpha` is
        outside (0, 1], the spectral radius of `A_K` is at least 1, or A_K^N W does not reach
        alpha W within 100000 steps.

    Notes
    -----
    Sec. 5.3.5 prints the state tightening as d - (1 - alpha^-1) phi_N. That factor is negative
    for alpha < 1 and would loosen the constraint; the factor here is (1 - alpha)^-1, as the same
    page prints for the input constraint. For A_K = 0.5 and W = [-1, 1] it gives the exact
    phi_inf = 2 at every N, where the printed one gives -1.
    """
    A_K = np.asarray(A_K, dtype=float)
    if A_K.ndim != 2 or A_K.shape[0] != A_K.shape[1] or not np.all(np.isfinite(A_K)):
        raise ValueError("A_K must be a finite square matrix")
    dim = A_K.shape[0]
    w_bar = np.asarray(w_bar, dtype=float)
    a = np.asarray(a, dtype=float)
    if w_bar.shape != (dim,) or a.shape != (dim,):
        raise ValueError(f"w_bar and a must have length {dim}")
    if not (np.all(np.isfinite(w_bar)) and np.all(w_bar > 0.0)):
        raise ValueError("w_bar must be finite and positive on every axis")
    if not np.all(np.isfinite(a)):
        raise ValueError("a must be finite")
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if float(np.max(np.abs(np.linalg.eigvals(A_K)))) >= 1.0:
        raise ValueError("A_K must have spectral radius below 1")
    power = A_K
    for big_n in range(1, _MAX_TUBE_STEPS + 1):
        alpha_n = float(np.max((np.abs(power) @ w_bar) / w_bar))
        if alpha_n < alpha:
            loop = LinearLoop(A_K, np.eye(dim), np.zeros(dim))
            box = Zonotope.box(np.zeros(dim), w_bar)
            return support(loop, np.zeros(dim), box, a, big_n) / (1.0 - alpha_n)
        power = power @ A_K
    raise ValueError(f"A_K^N W is not inside alpha W for any N up to {_MAX_TUBE_STEPS}")


def delay_margin(n: float, dt: float, delay_s: float, W, a) -> float:
    """Return the support of the error that held commands build up over a delay, open loop.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit, rad/s, positive.
    dt : float
        Hold time of each command, s, positive.
    delay_s : float
        The delay, s, a nonnegative whole number of steps `dt`.
    W : Zonotope, Polytope or ActuatorSet
        The error on each held acceleration, m/s^2, three-dimensional, independent per step.
    a : array_like, shape (6,)
        The row normal on the Hill-frame state, in row units per m and per m/s.

    Returns
    -------
    float
        max a . e over the reachable set of e+ = Phi e + Gamma w from e = 0 after delay_s / dt
        steps, in row units: the exact support of the sum of Phi^j Gamma W over the steps
        (Kochenderfer et al., *Algorithms for Validation*, eq. 8.20). Commands already sent run
        unchanged while they are pending, so no gain acts on this error, and along-track it
        grows without bound with the delay. A row a . x <= d holds on the delayed state when
        the prediction meets a . x <= d minus this margin; for a . x >= b pass -a.

    Raises
    ------
    ValueError
        If `n` or `dt` is not finite and positive, `delay_s` is negative or not a whole number
        of steps, or a shape is wrong.
    """
    dt = _positive(dt, "dt")
    ratio = _nonnegative(delay_s, "delay_s") / dt
    steps = round(ratio)
    if abs(ratio - steps) > _STEP_TOL:
        raise ValueError("delay_s must be a whole number of steps dt")
    return support(cw_loop(n, dt), np.zeros(6), W, a, steps)


def per_draw_sigma(eps: float, draws: int) -> float:
    """Return the width in standard deviations that keeps every one of many draws inside.

    Parameters
    ----------
    eps : float
        The allowed probability that any draw falls outside, in (0, 1).
    draws : int
        The number of Gaussian draws the bound covers, at least 1.

    Returns
    -------
    float
        k = Phi^-1(1 - eps / (2 draws)). Each draw leaves [-k sigma, k sigma] with probability
        2 Phi(-k) = eps / draws, so by Boole's inequality any of them does with probability at
        most eps, however the draws are correlated. `EllipsoidError.from_gaussian` takes this
        k, and `chance_margin` the matching eta = 1 - eps / (2 draws).

    Raises
    ------
    ValueError
        If `eps` is outside (0, 1) or `draws` is less than 1.
    TypeError
        If `draws` is not an integer.
    """
    eps = _probability(eps, "eps")
    draws = _count(draws, "draws")
    return float(norm.isf(eps / (2 * draws)))


def scenario_sample_count(*, eps: float, beta: float, n_theta: int = 1) -> int:
    """Return M*, the number of draws the scenario program needs.

    Parameters
    ----------
    eps : float
        The allowed probability that the constraint fails on a new draw, in (0, 1).
    beta : float
        The allowed probability that the draws are unrepresentative, in (0, 1).
    n_theta : int
        The number of decision variables of the scenario program, at least 1; one for a margin.

    Returns
    -------
    int
        ceil((2 / eps)(ln(1 / beta) + n_theta)) (Rawlings, Mayne and Diehl, 2nd ed., sec. 3.7.3,
        eq. 3.25): 113, 1122 and 11211 at eps = 0.1, 0.01 and 0.001 with beta = 0.01 and
        n_theta = 1. The bound is a least number of draws, so it is rounded up.

    Raises
    ------
    ValueError
        If `eps` or `beta` is outside (0, 1) or `n_theta` is less than 1.
    TypeError
        If `n_theta` is not an integer.
    """
    eps = _probability(eps, "eps")
    beta = _probability(beta, "beta")
    n_theta = _count(n_theta, "n_theta")
    return math.ceil(2.0 / eps * (math.log(1.0 / beta) + n_theta))


def scenario_margin(a, samples, *, eps: float, beta: float) -> float:
    """Return the scenario margin of a row, the largest a . e over sampled errors.

    Parameters
    ----------
    a : array_like, shape (k,)
        The row normal, in row units per state unit.
    samples : array_like, shape (M, k)
        Independent draws e(i) of the error, in state units, from the law the run uses.
    eps : float
        The allowed probability that a new error exceeds the margin, in (0, 1).
    beta : float
        The allowed probability that the draws are unrepresentative, in (0, 1).

    Returns
    -------
    float
        max_i a . e(i) in row units, the solution of the scenario program min theta subject to
        a . e(i) <= theta (Rawlings, Mayne and Diehl, 2nd ed., sec. 3.7.3, eq. 3.24). It has one
        decision variable, so with M >= M*(eps, beta, 1) draws a new error exceeds it with
        probability at most eps, at confidence 1 - beta over the draws (eq. 3.25). A row
        a . x <= d holds when the nominal state meets a . x <= d minus this margin; for
        a . x >= b pass -a, since sampled errors need not be symmetric.

    Raises
    ------
    ValueError
        If the shapes disagree or are not finite, `eps` or `beta` is outside (0, 1), or there
        are fewer than M* draws.
    """
    a = np.asarray(a, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if a.ndim != 1 or samples.ndim != 2 or samples.shape[1] != a.size:
        raise ValueError(f"samples must have shape (M, {a.size})")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(samples))):
        raise ValueError("a and samples must be finite")
    needed = scenario_sample_count(eps=eps, beta=beta, n_theta=1)
    if samples.shape[0] < needed:
        raise ValueError(f"needs at least {needed} samples, got {samples.shape[0]}")
    return float(np.max(samples @ a))
