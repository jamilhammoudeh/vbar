"""Impulse execution and coast prediction shared by the classical planners.

Planners work with impulsive velocity changes in the Hill frame. `impulse_action` turns one into
an action in the view's mode, and `predicted_margin` gives the keep-out margin over the coast
that follows, sampled and refined the way the environment scores it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..constraints import KeepOut, segment_minimum
from ..dynamics import CWDiscretizer
from .base import CONTINUOUS_MODES, AgentView, action_scale


@dataclass(frozen=True)
class Impulse:
    """An impulsive velocity change `dv` (Hill frame, m/s) at `t` (s since the episode start)."""

    t: float
    dv: np.ndarray


def _delivery(view: AgentView, dv: np.ndarray) -> float:
    """Size of `dv` in the norm the thruster layout bounds: max per axis (box) or 2-norm (ball)."""
    if view.thrust_limit == "box":
        return float(np.max(np.abs(dv)))
    if view.thrust_limit == "ball":
        return float(np.linalg.norm(dv))
    raise ValueError(f"unknown thrust limit {view.thrust_limit!r}")


def impulse_action(view: AgentView, dv: ArrayLike, coast_s: float) -> tuple[np.ndarray, bool]:
    """The action that starts delivering the impulse `dv` in the view's action mode.

    Parameters
    ----------
    view : AgentView
        Public scenario description; its action mode, `max_accel` (m/s^2), `dt` (s) and thrust
        layout set the action.
    dv : array_like, shape (3,)
        Impulse in the Hill frame (m/s).
    coast_s : float
        Coast after the burn step (s), used in "impulse_coast" mode only.

    Returns
    -------
    action : numpy.ndarray, shape (4,) or (3,)
        In "impulse_coast", fractions of a full command and the coast (s). An impulse within one
        step of delivery (`max_accel` `dt` per axis, or in 2-norm for a ball layout) is one step
        at fraction dv / (max_accel dt); a larger one is the first of k equal steps, k the
        ceiling of its size over one step's delivery, with no coast. In the continuous modes,
        dv / dt clipped to `max_accel` (per axis, or scaled in 2-norm for a ball layout), in the
        mode's units; the caller re-plans the remainder next step.
    saturated : bool
        Whether the impulse needs more than one step of delivery.

    Raises
    ------
    ValueError
        If the mode is "disc27" or not a known mode, the thrust layout is unknown, `dv` is not
        three finite numbers, or `coast_s` is negative or not finite (checked in every mode, so a
        bad plan fails the same way whatever the mode).
    """
    if view.action_mode == "disc27":
        raise ValueError("disc27 actions cannot deliver a planned impulse")
    if view.action_mode != "impulse_coast" and view.action_mode not in CONTINUOUS_MODES:
        raise ValueError(f"unknown action mode {view.action_mode!r}")
    dv = np.asarray(dv, dtype=float)
    if dv.shape != (3,) or not np.all(np.isfinite(dv)):
        raise ValueError("dv must be three finite numbers")
    if not (math.isfinite(coast_s) and coast_s >= 0.0):
        raise ValueError("coast_s must be finite and non-negative")
    step_dv = view.max_accel * view.dt
    size = _delivery(view, dv)
    saturated = size > step_dv
    if view.action_mode == "impulse_coast":
        if not saturated:
            return np.r_[dv / step_dv, coast_s], False
        k = math.ceil(size / step_dv)
        return np.r_[dv / (k * step_dv), 0.0], True
    fraction = dv / step_dv
    if saturated and view.thrust_limit == "box":
        fraction = np.clip(fraction, -1.0, 1.0)
    elif saturated:
        fraction = fraction / np.linalg.norm(fraction)
    return fraction * action_scale(view), saturated


def predicted_margin(
    view: AgentView, state: ArrayLike, coast_s: float, samples: int = 20
) -> tuple[float, float]:
    """Minimum keep-out margin over an unforced CW coast, and a certified lower bound on it.

    The coast is sampled at `samples` points per environment step, as the environment samples
    its constraint checks, and refined and bounded by `vbar.constraints.segment_minimum`: on
    each sample interval the bound is the larger of the speed bound and the keep-out sphere's
    second-order bound, and an interval is refined only when its bound lies more than
    `KeepOut.resolution` (1 mm) below the lowest margin found.

    Parameters
    ----------
    view : AgentView
        Public scenario description; its mean motion `n` (rad/s), `dt` (s) and keep-out
        constraint are used.
    state : array_like, shape (6,)
        State at the start of the coast, [x, y, z, vx, vy, vz] in the Hill frame (m, m/s).
    coast_s : float
        Coast duration (s).
    samples : int
        Samples per step of `dt`, at least 2.

    Returns
    -------
    margin : float
        Lowest range minus the keep-out radius found over the coast (m), at most 1 mm above
        the true minimum when each refined interval holds a single dip.
    bound : float
        Lower bound on the true minimum (m); -inf where the sample spacing is too long for the
        speed bound between samples to hold. Where the minimum falls on a sample the bound is
        that sample's margin, so it carries the propagation's rounding. Both are inf when the
        view has no keep-out constraint.

    Raises
    ------
    ValueError
        If `state` is not six finite numbers, `coast_s` is negative or not finite, or
        `samples` is below 2.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("state must be six finite numbers")
    if not (math.isfinite(coast_s) and coast_s >= 0.0):
        raise ValueError("coast_s must be finite and non-negative")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    radius = next((c["radius"] for c in view.constraints if c["type"] == "keep_out"), None)
    if radius is None:
        return math.inf, math.inf
    # The environment's rule for a segment of any length, so a one-step coast is sampled at
    # exactly the points the environment checks.
    count = max(2, round(samples * coast_s / view.dt))
    disc = CWDiscretizer(view.n)
    phis, _ = disc.sample_matrices(coast_s, count)
    found = segment_minimum(
        KeepOut(float(radius)),
        lambda tau: disc.matrices(tau)[0] @ state,
        np.linspace(0.0, coast_s, count),
        phis @ state,
        0.0,
        view.n,
    )
    return found.margin, found.lower_bound
