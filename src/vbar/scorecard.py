"""Plain-language scorecards: one flight as a verdict, a reason and checks; a set of flights graded.

The card reads the metrics `vbar.metrics.evaluate` returns and says what they mean in everyday
words: a verdict, one sentence on why, then each check as a measured fact and the rule it was held
to. One flight is a demonstration; a grade is a pass count over several starts with its exact 95%
range. The card carries no composite score.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .agents.base import AgentView
from .metrics import clopper_pearson

WHY: Mapping[str, str] = MappingProxyType(
    {
        "filter_defect": (
            "The safety layer that checks each thrust command broke down, and the spacecraft "
            "broke a safety rule."
        ),
        "filter_infeasible": (
            "The safety layer found no safe thrust command, and the spacecraft broke a safety rule."
        ),
        "keep_out_entry": "It flew into the keep-out zone around the target.",
        "speed_limit_breach": "It moved toward the target faster than the speed limit allows.",
        "cone_violation": "It left the approach corridor it must stay inside near the target.",
        "propellant_exhaustion": "It ran out of fuel before finishing.",
        "parse_failure": (
            "More than half of the AI model's replies could not be read as a thrust command."
        ),
        "sign_error": "Its thrust commands pointed the wrong way, as if a direction were flipped.",
        "oscillation": "It kept switching its thrusters back and forth instead of settling.",
        "contact_out_of_tolerance": (
            "It touched the docking port too far off center, sliding sideways, or too fast or too "
            "slow along its axis."
        ),
        "returning_drift": "It left, but on a path that drifts back toward the target.",
        "hold_not_maintained": "It reached the hold point but did not stay there long enough.",
        "overshoot": "It came within reach of its goal but went past it without stopping.",
        "timeout_near": "Time ran out while it was close to its goal.",
        "timeout_far": "Time ran out while it was still far from its goal.",
        "other": "It did not complete the mission, for a reason outside the listed failures.",
        "error": "The controller stopped with an error, so the flight could not finish.",
    }
)

BILL = "From listed prices; the provider's bill is final."
DEMONSTRATION = "One flight from one start is a demonstration, not a grade."
WIDTH = 72


@dataclass(frozen=True)
class Check:
    """One line of a scorecard.

    Parameters
    ----------
    name : str
        What is checked, such as "Arrival" or "Fuel".
    status : str
        The outcome in words: "Met", "Not met", a fuel ratio, a cost.
    fact : str
        What was measured on the flight.
    needed : str
        The rule the fact was held to, or what the fact is compared with; empty when there is none.
    """

    name: str
    status: str
    fact: str
    needed: str

    @property
    def rule(self) -> bool:
        """Whether the check holds the flight to a rule, so its `needed` reads as one."""
        return self.status in ("Met", "Not met", "Not judged")


@dataclass(frozen=True)
class Scorecard:
    """One flight in plain language.

    Parameters
    ----------
    verdict : str
        "Passed" or "Did not pass".
    why : str
        One sentence on the verdict.
    checks : tuple of Check
        Arrival, keep-out zone (when the mission has one), fuel, cost, then any advisory line.
    """

    verdict: str
    why: str
    checks: tuple[Check, ...]

    def text(self) -> str:
        """The card as plain text, without colour, wrapped to 72 columns.

        Returns
        -------
        str
            Verdict, reason, each check with its status in words, and the demonstration line.
        """
        lines = [self.verdict, *textwrap.wrap(self.why, WIDTH), ""]
        for check in self.checks:
            lines.append(f"{check.name}: {check.status.lower()}")
            lines += textwrap.wrap(check.fact, WIDTH, initial_indent="  ", subsequent_indent="  ")
            if check.needed:
                needed = check.needed
                if check.rule:
                    needed = f"Needed: {needed[0].lower()}{needed[1:]}"
                lines += textwrap.wrap(needed, WIDTH, initial_indent="  ", subsequent_indent="  ")
        lines += ["", *textwrap.wrap(DEMONSTRATION, WIDTH)]
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Graded:
    """Flights from several starts graded by their pass count.

    Parameters
    ----------
    passed : int
        Flights that passed.
    flown : int
        Flights counted, crashed ones included.
    low, high : float
        Exact (Clopper-Pearson) 95% interval for the pass rate, in [0, 1].
    keep_out : int or None
        Flights that entered the keep-out zone; None when no flight recorded its distance.
    """

    passed: int
    flown: int
    low: float
    high: float
    keep_out: int | None

    def text(self) -> str:
        """The grade as plain text, wrapped to 72 columns.

        Returns
        -------
        str
            The count, the likely range of the pass rate and the keep-out count.
        """
        likely = f"between {_percent(self.low)} and {_percent(self.high)}"
        if self.passed < self.flown:
            count = f"Passed {self.passed} of {self.flown} starts"
        elif self.flown == 1:
            count = "Passed its only start"
        else:
            count = f"Passed all {self.flown} starts flown"
        sentences = [f"{count}; the true pass rate is very likely {likely}."]
        if self.keep_out == 0:
            sentences.append("Never entered the keep-out zone.")
        elif self.keep_out is not None:
            sentences.append(f"Entered the keep-out zone on {self.keep_out} of them.")
        sentences.append("Exact 95% range (Clopper-Pearson); fewer starts give a wider range.")
        return "\n".join(textwrap.fill(s, WIDTH) for s in sentences) + "\n"


def flight_card(
    metrics: Mapping[str, Any],
    view: AgentView,
    *,
    least_fuel: tuple[float, float] | None = None,
    calls: int | None = None,
    cap: float | None = None,
    soft: Collection[str] = (),
    decisions: int | None = None,
) -> Scorecard:
    """The scorecard of one flight.

    Parameters
    ----------
    metrics : mapping
        The flight's metrics as `vbar.metrics.evaluate` returns them, with `failure_label`. A
        crashed flight may carry only `success` and the label "error"; its checks then say what
        was not recorded.
    view : AgentView
        The scenario's public description: goal, constraints and step length.
    least_fuel : tuple of float, optional
        Propellant mass (kg) no controller can go below from this start, and the mass of the best
        plan found, floor first. Without it the fuel is not compared.
    calls : int, optional
        Calls made to an AI model. None, with no `token_cost` in `metrics`, marks a controller
        that calls no model, which costs nothing.
    cap : float, optional
        Spending cap (US dollars) the flight ran under.
    soft : collection of str
        Names of the scenario's soft constraints (`env.soft_constraints`), whose breaches are
        advisory. Every breach on a passed flight is advisory, since a pass breaks no hard
        constraint.
    decisions : int, optional
        Thrust commands the AI model was asked for. A command can take several calls when the
        model uses tools; without it each call is taken as one command.

    Returns
    -------
    Scorecard
        Verdict, reason and checks in the order arrival, keep-out zone, fuel, cost, advisory.

    Raises
    ------
    ValueError
        If `least_fuel` is not a positive floor at or below a finite best plan.
    KeyError
        If the failure label is not in `WHY`.
    """
    if least_fuel is not None and not 0.0 < least_fuel[0] <= least_fuel[1] < math.inf:
        raise ValueError("least_fuel must be 0 < floor <= best plan, in kg")
    passed = bool(metrics.get("success", False))
    arrived = _arrived(metrics, view.goal)
    keep_out = _keep_out_radius(view)
    checks = [_arrival(metrics, view.goal, arrived)]
    if keep_out is not None:
        checks.append(_keep_out(metrics, keep_out))
    checks += [_fuel(metrics, passed, least_fuel), _cost(metrics, view.dt, calls, cap, decisions)]
    checks += _also_seen(metrics, view, passed, soft)
    return Scorecard(
        verdict="Passed" if passed else "Did not pass",
        why=_why(metrics, view.goal, passed, arrived, keep_out is not None),
        checks=tuple(checks),
    )


def graded_card(cards: Sequence[Mapping[str, Any]]) -> Graded:
    """Grade flights from several starts by how many passed.

    Parameters
    ----------
    cards : sequence of mapping
        One metrics mapping per flight, each with `success`; a crashed flight counts as not
        passed. `min_margin["keep_out"]`, where present, gives the keep-out count.

    Returns
    -------
    Graded
        The pass count, the flights counted and the exact 95% interval for the pass rate.

    Raises
    ------
    ValueError
        If `cards` is empty.
    """
    if not cards:
        raise ValueError("need at least one flight to grade")
    passed = sum(bool(card.get("success", False)) for card in cards)
    margins = [_finite(card.get("min_margin", {}).get("keep_out")) for card in cards]
    recorded = [m for m in margins if m is not None]
    low, high = clopper_pearson(passed, len(cards))
    return Graded(
        passed=passed,
        flown=len(cards),
        low=low,
        high=high,
        keep_out=sum(m < 0.0 for m in recorded) if recorded else None,
    )


def _finite(value: Any) -> float | None:
    """`value` as a float, or None when it is missing or not finite."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _arrived(metrics: Mapping[str, Any], goal: Mapping[str, Any]) -> bool | None:
    """Whether the goal was met, constraints aside; None when the metrics cannot tell.

    A failed flight can have met its goal and broken a hard constraint, so the goal is read from
    the metric that decides it: the dwell for a timed hold, the contact margin for a docking and
    the coverage for an inspection. Other goals are decided only by a pass.
    """
    if metrics.get("success"):
        return True
    if "success" not in metrics or metrics.get("failure_label") == "error":
        return None
    if goal.get("task") == "inspect":
        progress = _finite(metrics.get("progress"))
        return None if progress is None else progress >= goal["coverage"]
    if "axial_speed_window" in goal:
        return bool(metrics.get("capture_margin", math.nan) >= 0.0)
    if goal.get("hold_time", 0.0) > 0.0:
        dwell = _finite(metrics.get("hold_dwell"))
        return None if dwell is None else dwell >= goal["hold_time"]
    return None


def _status(met: bool | None) -> str:
    return {True: "Met", False: "Not met", None: "Not judged"}[met]


def _arrival(metrics: Mapping[str, Any], goal: Mapping[str, Any], arrived: bool | None) -> Check:
    """The arrival check: the measured approach to the goal and the goal's own rule."""
    done = _finite(metrics.get("time_to_complete")) if arrived else None
    if goal.get("task") == "inspect":
        progress = _finite(metrics.get("progress"))
        fact = (
            "No coverage was recorded."
            if progress is None
            else f"Saw {progress:.0%} of the target's surface that can be seen."
        )
        needed = f"See at least {goal['coverage']:.0%} of it."
    elif "axial_speed_window" in goal:
        fact = _contact(metrics)
        lo, hi = goal["axial_speed_window"]
        along = f"{_number(lo)} to {_number(hi)} m/s" if lo > 0.0 else f"at most {_number(hi)} m/s"
        needed = (
            f"Touch the port within {_distance(goal['lateral_limit'])} of its center, at {along} "
            f"along its axis and below {_number(goal['lateral_speed_limit'])} m/s sideways."
        )
    elif goal.get("task") == "depart":
        fact = "Left on a clear path." if arrived else "No clear departure was shown."
        needed = (
            f"Beyond {_distance(goal['clear_range'])} on a path that never drifts back within "
            f"{_distance(goal['drift_radius'])} of the target."
        )
    elif "capture_radius" in goal:
        fact = _hold(metrics, goal)
        needed = (
            f"Within {_distance(goal['capture_radius'])} and slower than "
            f"{_number(goal['capture_speed'])} m/s"
        )
        hold_time = goal.get("hold_time", 0.0)
        needed += f", for {_duration(hold_time)}." if hold_time > 0.0 else "."
    else:
        fact, needed = "The mission's goal is judged by the full measurements.", ""
    if done is not None:
        fact += f" Goal met {_duration(done)} after the start."
    return Check("Arrival", _status(arrived), fact, needed)


def _hold(metrics: Mapping[str, Any], goal: Mapping[str, Any]) -> str:
    """What the flight did at the hold point: the longest dwell and where it ended."""
    dwell = _finite(metrics.get("hold_dwell"))
    end = _finite(metrics.get("terminal_pos_error"))
    ended = f" and ended {_distance(end)} from it" if end is not None else ""
    if dwell is None:
        return "No hold was recorded."
    if dwell <= 0.0:
        return f"Never held the hold point{ended}."
    if dwell < goal.get("hold_time", 0.0):
        return f"Held the hold point for at most {_duration(dwell)}{ended}."
    return f"Held the hold point for {_duration(dwell)}{ended}."


def _contact(metrics: Mapping[str, Any]) -> str:
    axial = _finite(metrics.get("contact_axial_speed"))
    lateral = _finite(metrics.get("contact_lateral_error"))
    sideways = _finite(metrics.get("contact_lateral_speed"))
    if axial is None or lateral is None or sideways is None:
        return "Never touched the port."
    return (
        f"Touched the port {_distance(lateral)} off center, at {axial:.3f} m/s along its axis "
        f"and {sideways:.3f} m/s sideways."
    )


def _contact_faults(metrics: Mapping[str, Any], goal: Mapping[str, Any]) -> list[str]:
    """Each docking limit the contact broke, in words, by the criteria of the contact judge."""
    axial = _finite(metrics.get("contact_axial_speed"))
    lateral = _finite(metrics.get("contact_lateral_error"))
    sideways = _finite(metrics.get("contact_lateral_speed"))
    if axial is None or lateral is None or sideways is None:
        return []
    low, high = goal["axial_speed_window"]
    faults = []
    if lateral > goal["lateral_limit"]:
        faults.append("too far off center")
    if sideways > goal["lateral_speed_limit"]:
        faults.append("sliding sideways faster than allowed")
    if axial > high:
        faults.append("too fast along its axis")
    if axial < low:
        faults.append("too slow along its axis to latch")
    return faults


def _keep_out_radius(view: AgentView) -> float | None:
    for constraint in view.constraints:
        if constraint.get("type") == "keep_out":
            return float(constraint["radius"])
    return None


def _keep_out(metrics: Mapping[str, Any], radius: float) -> Check:
    """The keep-out check: time inside the zone and the closest distance to the target."""
    margin = _finite(metrics.get("min_margin", {}).get("keep_out"))
    needed = "Never inside."
    if margin is None:
        return Check("Keep-out zone", "Not judged", "No distance was recorded.", needed)
    closest = _distance(radius + margin)
    zone = f"the {_number(radius)} m zone"
    if margin >= 0.0:
        fact = f"Never inside {zone}; {closest} from the target at the closest, "
        return Check("Keep-out zone", "Met", fact + f"{_distance(margin)} to spare.", needed)
    inside = _finite(metrics.get("violation_duration", {}).get("keep_out"))
    time = f" for {_duration(inside)}" if inside is not None else ""
    fact = f"Inside {zone}{time}, {closest} from the target at the closest."
    return Check("Keep-out zone", "Not met", fact, needed)


def _fuel(
    metrics: Mapping[str, Any], passed: bool, least_fuel: tuple[float, float] | None
) -> Check:
    """The fuel check: mass used, and its ratio to the least possible on a passed flight."""
    used = _finite(metrics.get("propellant_mass"))
    if used is None:
        return Check("Fuel", "Not recorded", "No fuel mass was recorded for this flight.", "")
    if least_fuel is None:
        floor = "Not compared with the least possible from this start, which was not worked out."
    else:
        floor = (
            f"No controller can use less than {_mass(least_fuel[0])} from this start; the best "
            f"plan V-BAR found uses {_mass(least_fuel[1])}."
        )
    if not passed:
        fact = f"Used {_mass(used)}. {floor}"
        return Check("Fuel", "Not ranked", fact, "Fuel is ranked only on flights that pass.")
    if least_fuel is None:
        return Check("Fuel", "Not compared", f"Used {_mass(used)}. {floor}", "")
    times = _ratio(used / least_fuel[1], used / least_fuel[0])
    fact = f"Used {_mass(used)}, {times} the least possible."
    needed = f"{floor} The true least is between the two, which is why the ratio is a range."
    if least_fuel[0] == least_fuel[1]:
        needed = floor
    return Check("Fuel", times, fact, needed)


def _ratio(low: float, high: float) -> str:
    low_text, high_text = f"{low:.1f}", f"{high:.1f}"
    if low_text == high_text:
        return f"{low_text} times"
    return f"{low_text} to {high_text} times"


def _cost(
    metrics: Mapping[str, Any],
    dt: float,
    calls: int | None,
    cap: float | None,
    decisions: int | None,
) -> Check:
    """The cost check: the estimate from listed prices, the calls and the unreadable replies."""
    cost = _finite(metrics.get("token_cost"))
    if calls is None and cost is None:
        return Check("Cost", "Free", "Built-in controller, no calls to any AI model.", "")
    spent = "No listed price" if cost is None else _dollars(cost)
    parts = [spent]
    if cap is not None:
        parts.append(f" of your {_dollars(cap)} cap")
    if calls is not None:
        parts.append(f": {calls} {'call' if calls == 1 else 'calls'}")
    rate = _finite(metrics.get("parse_failure_rate"))
    if rate is None or calls is None:
        parts.append(".")
    elif rate == 0.0:
        parts.append(", every reply readable.")
    else:
        # The rate is per command, and a command has one reply that is read.
        commands = calls if decisions is None else decisions
        unread = min(round(rate * commands), commands)
        replies = "1 reply" if unread == 1 else f"{unread} replies"
        parts.append(
            f". {replies} could not be read as a thrust command, so the spacecraft coasted for "
            f"those {_duration(unread * dt)}."
        )
    return Check("Cost", spent, "".join(parts), BILL if cost is not None else "")


_ADVISORY = {
    "passive_speed": (
        "it closed on the target fast enough that, had its thrusters stopped, it could have "
        "drifted within {extent} of it"
    ),
    "closing_speed_limit": "it closed on the target faster than the speed limit for its distance",
    "gate_table": "it closed on the target faster than the speed limit for its distance",
    "speed_limit": "it moved faster than the speed limit for its distance",
    "approach_cone": "it was outside the approach corridor",
    "sun_exclusion": "the Sun was in its camera's view of the target",
    "zone_entry": "it was inside the approach zone before it was cleared to enter",
}


def _also_seen(
    metrics: Mapping[str, Any], view: AgentView, passed: bool, soft: Collection[str]
) -> list[Check]:
    """Breaches of constraints other than the keep-out zone: an advisory check, a recorded one.

    Breaches of soft constraints, and every breach on a passed flight, are advisory; the other
    breaches of a failed flight are recorded in a check of their own.
    """
    margins = metrics.get("min_margin", {})
    durations = metrics.get("violation_duration", {})
    sentences, advisory = [], []
    for constraint in view.constraints:
        name = constraint.get("name", constraint.get("type"))
        if name == "keep_out" or not margins.get(name, 0.0) < 0.0:
            continue
        extent = constraint.get("extent")
        what = _ADVISORY.get(name, f"it broke the {str(name).replace('_', ' ')} limit")
        what = what.format(extent=_distance(extent) if extent is not None else "reach")
        inside = _finite(durations.get(name))
        when = f"For {_duration(inside)}" if inside is not None else "At times"
        sentences.append(f"{when} {what}")
        advisory.append(passed or name in soft)
    checks = []
    for status, needed, kind in (
        ("Advisory", "Recorded; it does not fail the flight.", True),
        ("Recorded", "Recorded with the flight.", False),
    ):
        fact = " ".join(
            f"{s}." for s, mark in zip(sentences, advisory, strict=True) if mark is kind
        )
        if fact:
            checks.append(Check("Also seen", status, fact, needed))
    return checks


def _why(
    metrics: Mapping[str, Any],
    goal: Mapping[str, Any],
    passed: bool,
    arrived: bool | None,
    keep_out: bool,
) -> str:
    """The one sentence under the verdict."""
    hold = goal.get("task") == "hold"
    if passed:
        if hold and keep_out:
            return "It stopped at the hold point and held it without entering the keep-out zone."
        return "It met the mission's goal and kept every safety rule."
    label = metrics.get("failure_label") or "other"
    if label == "keep_out_entry" and arrived:
        reached = "reached the hold point and held it" if hold else "met the mission's goal"
        return f"It {reached}, but flew through the keep-out zone on the way."
    if label == "speed_limit_breach" and metrics.get("min_margin", {}).get("speed_limit", 0.0) < 0:
        return "It moved faster than the speed limit for its distance from the target."
    if label == "contact_out_of_tolerance" and "axial_speed_window" in goal:
        faults = _contact_faults(metrics, goal)
        if faults:
            return f"It touched the docking port {' and '.join(faults)}."
    return WHY[label]


def _duration(seconds: float) -> str:
    """Whole seconds as hours, minutes and seconds: 656 s reads "10 min 56 s"."""
    total = round(seconds)
    if total == 0 and seconds > 0.0:
        return "under 1 s"
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{hours} h"] if hours else []
    if minutes:
        parts.append(f"{minutes} min")
    if secs or not parts:
        parts.append(f"{secs} s")
    return " ".join(parts)


def _distance(metres: float) -> str:
    """A distance in centimetres below a metre, else metres to one decimal."""
    centimetres = round(metres * 100.0)
    if abs(centimetres) < 100:
        return f"{centimetres} cm"
    return f"{_number(round(metres, 1))} m"


def _mass(kg: float) -> str:
    """A mass in grams below a kilogram, else kilograms to two decimals."""
    grams = round(kg * 1000.0)
    if grams < 1000:
        return f"{grams} g"
    return f"{kg:.2f} kg"


def _dollars(amount: float) -> str:
    if 0.0 < amount < 0.005:
        return "under $0.01"
    return f"${amount:.2f}"


def _percent(rate: float) -> str:
    """A rate as a percentage, one decimal near 0% and 100% so neither end is rounded onto them."""
    if rate in (0.0, 1.0):
        return f"{rate:.0%}"
    if rate < 0.1 or rate > 0.9:
        return f"{rate:.1%}"
    return f"{rate:.0%}"


def _number(value: float) -> str:
    return f"{value:g}"
