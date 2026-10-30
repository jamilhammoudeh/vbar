"""Published rendezvous results rebuilt from their papers, and the table that compares them.

`python -m vbar.reproductions` prints the match table quoted in `docs/validation.md`.

Ariba, Arzelier, Urbina and Louembet, "V-bar and R-bar glideslope guidance algorithms for
fixed-time rendezvous: a linear programming approach", IFAC-PapersOnLine 49(17), 385-390 (2016),
case 1, which they take from Hablani, Tapper and Dana-Bashian (2002): a V-bar transfer from 500 m
to 100 m behind the target, 20 m above it, in 540 s, at an orbital rate the paper gives as 0.001
rad/s. Their axes are x along-track, y out of plane and z toward the Earth, so their
[x, y, z] is V-BAR's [-z_p, x_p, -y_p] in [x, y, z]: the start [-500, 0, -20] m is (20, -500, 0) m
here. Costs are sums of the impulses' 1-norms, as in the paper's criterion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .agents.cw_targeting import two_impulse
from .agents.glideslope import glideslope_profile, glideslope_pulses
from .agents.glideslope_lp import lp_glideslope
from .agents.planning import Impulse
from .dynamics import cw_stm

__all__ = ["ARIBA_CASE_ONE", "MatchRow", "ariba_case_one", "excursion", "match_table", "matches"]


@dataclass(frozen=True)
class AribaCase:
    """Ariba et al.'s case 1 in V-BAR's Hill frame (m, s, rad/s)."""

    n: float = 1e-3
    start: tuple[float, float, float] = (20.0, -500.0, 0.0)
    aim: tuple[float, float, float] = (20.0, -100.0, 0.0)
    duration: float = 540.0


ARIBA_CASE_ONE = AribaCase()


@dataclass(frozen=True)
class MatchRow:
    """One published number, its V-BAR reproduction and how the two are related."""

    quantity: str
    paper: float
    vbar: float
    unit: str
    note: str


def _cost(impulses: tuple[Impulse, ...]) -> float:
    return float(sum(np.sum(np.abs(impulse.dv)) for impulse in impulses))


def excursion(
    n: float, start: np.ndarray, impulses: tuple[Impulse, ...], samples: int = 400
) -> float:
    """Largest radial distance from the start's V-bar line over an impulsive flight.

    The chaser starts at rest at `start`, takes each impulse at its time and coasts on exact
    Clohessy-Wiltshire arcs between them; each arc is sampled at `samples` points.

    Parameters
    ----------
    n : float
        Mean motion, rad/s, positive.
    start : numpy.ndarray
        Start position, m, Hill frame, shape (3,).
    impulses : tuple of Impulse
        The plan, at least one impulse, times not decreasing.
    samples : int
        Points per arc, at least 2.

    Returns
    -------
    float
        Largest |x - x_start| over the flight, m.

    Raises
    ------
    ValueError
        If an input is out of range or not finite.
    """
    if isinstance(n, bool) or not (math.isfinite(n) and n > 0.0):
        raise ValueError("n must be positive and finite")
    start = np.asarray(start, dtype=float)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError("start must be three finite numbers")
    if isinstance(samples, bool) or not isinstance(samples, int | np.integer) or samples < 2:
        raise ValueError("samples must be an integer of at least 2")
    if not impulses:
        raise ValueError("impulses must hold at least one impulse")
    times = np.array([impulse.t for impulse in impulses], dtype=float)
    kicks = np.array([impulse.dv for impulse in impulses], dtype=float)
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(kicks))) or kicks.shape[1:] != (3,):
        raise ValueError("impulse times and velocity changes must be finite, three per change")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("impulse times must not decrease")
    state = np.concatenate([start, np.zeros(3)])
    largest = 0.0
    for k, impulse in enumerate(impulses):
        state = state + np.concatenate([np.zeros(3), impulse.dv])
        if k + 1 == len(impulses):
            break
        span = impulses[k + 1].t - impulse.t
        for tau in np.linspace(0.0, span, samples)[1:]:
            largest = max(largest, abs(float((cw_stm(n, tau) @ state)[0]) - start[0]))
        state = cw_stm(n, span) @ state
    return largest


def _standard(case: AribaCase, ratio: float, pulses: int) -> tuple[Impulse, ...]:
    """Hablani's glideslope with closing rates in `ratio`, timed to the case's duration."""
    rho0 = float(np.linalg.norm(np.subtract(case.start, case.aim)))
    rate0 = rho0 * math.log(ratio) / (case.duration * (1.0 - ratio))
    profile = glideslope_profile(rho0, rate0, ratio * rate0)
    # A hair over duration / N, so rounding cannot add a pulse.
    interval = case.duration / pulses * (1.0 + 1e-9)
    state = np.concatenate([np.asarray(case.start), np.zeros(3)])
    return glideslope_pulses(case.n, state, np.asarray(case.aim), profile, interval)


def ariba_case_one(case: AribaCase = ARIBA_CASE_ONE) -> list[MatchRow]:
    """Ariba et al.'s case 1 numbers and their reproduction.

    The standard glideslope's closing rates are not given in the paper beyond the transfer time;
    their ratio is fitted to the N = 10 cost, and the other four costs are predictions.

    Parameters
    ----------
    case : AribaCase
        The case in V-BAR's Hill frame.

    Returns
    -------
    list of MatchRow
        One row per published number, in the paper's order.
    """
    start, aim = np.asarray(case.start), np.asarray(case.aim)
    rows = [
        MatchRow(
            "LP glideslope, N = 10, 1 m bound",
            2.31,
            _cost(lp_glideslope(case.n, start, aim, case.duration, 10, 1.0)),
            "m/s",
            "rebuilt from sec. 3",
        )
    ]
    lp_costs = (2.26, 2.29, 2.30, 2.31, 2.31)
    lp_excursions = (13.8, 6.2, 3.5, 0.56, 0.15)
    for pulses, cost, spread in zip((2, 3, 4, 10, 20), lp_costs, lp_excursions, strict=True):
        plan = lp_glideslope(case.n, start, aim, case.duration, pulses, 20.0)
        rows.append(
            MatchRow(f"LP glideslope, N = {pulses}, 20 m bound", cost, _cost(plan), "m/s", "")
        )
        rows.append(
            MatchRow(
                f"LP excursion, N = {pulses}, 20 m bound",
                spread,
                excursion(case.n, start, plan),
                "m",
                "",
            )
        )
    ratio = 10.0 ** brentq(lambda q: _cost(_standard(case, 10.0**q, 10)) - 8.09, -12.0, -0.01)
    std_costs = (3.61, 4.76, 5.63, 8.09, 9.32)
    std_excursions = (26.2, 16.2, 11.1, 2.67, 0.78)
    for pulses, cost, spread in zip((2, 3, 4, 10, 20), std_costs, std_excursions, strict=True):
        plan = _standard(case, ratio, pulses)
        note = "rate ratio fitted here" if pulses == 10 else "prediction"
        rows.append(MatchRow(f"Standard glideslope, N = {pulses}", cost, _cost(plan), "m/s", note))
        rows.append(
            MatchRow(
                f"Standard excursion, N = {pulses}",
                spread,
                excursion(case.n, start, plan),
                "m",
                "prediction",
            )
        )
    first, second = two_impulse(case.n, np.concatenate([start, np.zeros(3)]), aim, case.duration)
    pair = (Impulse(0.0, first), Impulse(case.duration, second))
    # The paper's two-impulse transfer is Fehse's tangential one, whose duration it does not
    # give; the transfer in the case's 540 s is a different manoeuvre, shown for scale.
    unknown = "not reproduced: the paper's transfer time is not stated; 540 s transfer shown"
    rows.append(MatchRow("Two-impulse transfer", 0.12, _cost(pair), "m/s", unknown))
    rows.append(
        MatchRow("Two-impulse excursion", 125.0, excursion(case.n, start, pair), "m", unknown)
    )
    return rows


def _decimals(value: float) -> int:
    text = f"{value:g}"
    return len(text.split(".")[1]) if "." in text else 0


def matches(row: MatchRow) -> bool:
    """Whether V-BAR's number rounds to the paper's at the paper's printed digits.

    Parameters
    ----------
    row : MatchRow
        The comparison.

    Returns
    -------
    bool
        True when |V-BAR - paper| is at most half a unit of the paper's last printed digit.
    """
    half = 0.5 * 10.0 ** -_decimals(row.paper)
    return abs(row.vbar - row.paper) <= half * (1.0 + 1e-9)


def match_table(rows: list[MatchRow]) -> str:
    """The rows as a Markdown table, V-BAR's numbers to one digit past the paper's.

    Parameters
    ----------
    rows : list of MatchRow
        The comparisons, in table order.

    Returns
    -------
    str
        The table, one line per row after the header, no trailing newline.
    """
    lines = [
        "| Quantity | Paper | V-BAR | Unit | Matches | Note |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        digits = _decimals(row.paper) + 1
        verdict = "yes" if matches(row) else "no"
        lines.append(
            f"| {row.quantity} | {row.paper:g} | {row.vbar:.{digits}f} | {row.unit} | {verdict} "
            f"| {row.note} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(match_table(ariba_case_one()))
