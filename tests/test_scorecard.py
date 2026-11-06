"""The plain-language scorecard reads a flight's metrics as a verdict, a reason and checks."""

import inspect
import math
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import binomtest

from vbar.agents import DriftAgent, PDAgent
from vbar.metrics import LABELS, clopper_pearson, evaluate, record_episode
from vbar.scenarios import make, seeds
from vbar.scorecard import WHY, flight_card, graded_card

BANNED = re.compile(
    r"note:|here we|this function|simply|robust|comprehensive|seamless|leverage|elegant|"
    r"powerful|ensure that|!",
    re.I,
)
# Benchmark terms the card replaces with everyday words (glossary in the scorecard wording).
JARGON = re.compile(r"episode|seed|delta-v|optimality|oracle|agent|policy|clopper|score|_", re.I)
# Provider names that may appear in test files; the rest are kept out by the commit hooks.
PROVIDERS = re.compile(r"google|gemini|mistral|openrouter|ollama", re.I)
BILL = "From listed prices; the provider's bill is final."


@pytest.fixture(scope="module")
def pd_flight():
    """PD on approach-vbar, validation start 3: it holds the point but crosses the keep-out zone."""
    env = make("approach-vbar")
    seed, agent_seed = seeds("approach-vbar", "validation", 4).pairs()[3]
    ep = record_episode(env, PDAgent(), seed=seed, agent_seed=agent_seed)
    return evaluate(ep, env.metric_settings()), ep.view


def passing(metrics):
    """A stand-in passing flight built on the PD flight's figures."""
    return {
        **metrics,
        "success": True,
        "failure_label": None,
        "propellant_mass": 1.32,
        "hold_dwell": 605.0,
        "min_margin": {"keep_out": 0.4, "closing_speed_limit": 0.1, "passive_speed": 0.2},
        "violation_duration": {"keep_out": 0.0, "closing_speed_limit": 0.0, "passive_speed": 0.0},
        "parse_failure_rate": 0.0,
    }


def check(card, name):
    (found,) = [c for c in card.checks if c.name == name]
    return found


def words(text):
    return " ".join(text.split())


@pytest.mark.parametrize("label", LABELS)
def test_every_failure_label_has_one_everyday_sentence(label):
    sentence = WHY[label]
    assert sentence[0].isupper() and sentence.endswith(".")
    assert ". " not in sentence[:-1]
    assert not JARGON.search(sentence) and not BANNED.search(sentence)


def test_failure_sentences_are_distinct():
    assert len({WHY[label] for label in LABELS}) == len(LABELS)


def test_a_flight_that_crashed_has_a_sentence():
    # run_job records a controller exception as the label "error".
    assert WHY["error"].endswith(".")


@pytest.mark.parametrize("label", [label for label in LABELS if label != "keep_out_entry"])
def test_failed_flight_gives_the_label_sentence(pd_flight, label):
    metrics, view = pd_flight
    card = flight_card({**metrics, "failure_label": label}, view)
    assert card.verdict == "Did not pass"
    assert card.why == WHY[label]


def test_pd_through_keep_out_zone_does_not_pass(pd_flight):
    metrics, view = pd_flight
    assert metrics["failure_label"] == "keep_out_entry"
    card = flight_card(metrics, view)
    assert card.verdict == "Did not pass"
    assert "keep-out zone" in card.why
    keep_out = check(card, "Keep-out zone")
    assert keep_out.status == "Not met"
    # violation_duration 134.9 s and min margin -32.2 m on a 50 m zone for this start.
    assert "2 min 15 s" in keep_out.fact
    assert "17.8 m" in keep_out.fact
    assert "never inside" in keep_out.needed.lower()


def test_pd_flight_arrival_is_met_with_the_rule(pd_flight):
    metrics, view = pd_flight
    arrival = check(flight_card(metrics, view), "Arrival")
    assert arrival.status == "Met"
    # hold_dwell 605.8 s for this start.
    assert "10 min 6 s" in arrival.fact
    assert all(part in arrival.needed for part in ("5 m", "0.05 m/s", "10 min"))


def test_checks_come_in_the_card_order(pd_flight):
    metrics, view = pd_flight
    names = [c.name for c in flight_card(passing(metrics), view).checks]
    assert names[:4] == ["Arrival", "Keep-out zone", "Fuel", "Cost"]


def test_passing_flight_shows_fuel_as_a_range_of_the_least_possible(pd_flight):
    metrics, view = pd_flight
    card = flight_card(passing(metrics), view, least_fuel=(0.206, 0.214))
    assert card.verdict == "Passed"
    assert check(card, "Keep-out zone").status == "Met"
    fuel = check(card, "Fuel")
    # 1.32 / 0.214 = 6.17 and 1.32 / 0.206 = 6.41.
    assert fuel.status == "6.2 to 6.4 times"
    assert "1.32 kg" in fuel.fact
    assert "206 g" in fuel.fact + fuel.needed and "214 g" in fuel.fact + fuel.needed
    assert "m/s" not in fuel.fact + fuel.needed


def test_failed_flight_fuel_is_not_ranked(pd_flight):
    metrics, view = pd_flight
    fuel = check(flight_card(metrics, view, least_fuel=(0.206, 0.214)), "Fuel")
    assert fuel.status == "Not ranked"
    assert "1.50 kg" in fuel.fact
    assert "times" not in fuel.fact + fuel.needed
    assert "ranked only on flights that pass" in fuel.needed


@pytest.mark.parametrize("passed", [True, False])
def test_an_uncertified_floor_leaves_only_the_best_plan_on_the_card(pd_flight, passed):
    metrics, view = pd_flight
    flown = passing(metrics) if passed else metrics
    fuel = check(flight_card(flown, view, best_plan=0.214), "Fuel")
    text = words(fuel.fact + " " + fuel.needed)
    assert "the best plan V-BAR found uses 214 g" in text
    assert "no least possible is certified for this flight" in text
    assert "No controller can use less than" not in text
    assert "times" not in text


@pytest.mark.parametrize("best_plan", [-0.2, 0.0, math.nan, math.inf])
def test_a_best_plan_that_is_not_a_positive_finite_mass_is_refused(pd_flight, best_plan):
    metrics, view = pd_flight
    with pytest.raises(ValueError, match="best_plan"):
        flight_card(metrics, view, best_plan=best_plan)


def test_built_in_controller_costs_nothing(pd_flight):
    metrics, view = pd_flight
    cost = check(flight_card(metrics, view), "Cost")
    assert cost.status == "Free"
    assert "Built-in controller" in cost.fact


def test_ai_model_cost_is_an_estimate_from_listed_prices(pd_flight):
    metrics, view = pd_flight
    flown = {**passing(metrics), "token_cost": 0.14, "parse_failure_rate": 3 / 134}
    cost = check(flight_card(flown, view, calls=134, cap=1.0), "Cost")
    assert cost.status == "$0.14"
    assert "$0.14 of your $1.00 cap: 134 calls" in cost.fact
    assert "3 replies could not be read" in cost.fact
    assert cost.needed == BILL


def test_card_cannot_be_given_a_provider_name():
    names = set(inspect.signature(flight_card).parameters)
    assert not names & {"provider", "vendor", "company", "model"}


def test_soft_constraint_breach_is_advisory_and_does_not_fail(pd_flight):
    metrics, view = pd_flight
    flown = passing(metrics)
    flown["min_margin"] = {**flown["min_margin"], "passive_speed": -1.0}
    flown["violation_duration"] = {**flown["violation_duration"], "passive_speed": 656.0}
    card = flight_card(flown, view, soft=make("approach-vbar").soft_constraints)
    assert card.verdict == "Passed"
    seen = check(card, "Also seen")
    assert seen.status == "Advisory"
    assert "10 min 56 s" in seen.fact
    assert seen.needed == "Recorded; it does not fail the flight."


def test_no_advisory_line_without_a_soft_breach(pd_flight):
    metrics, view = pd_flight
    assert "Also seen" not in [c.name for c in flight_card(passing(metrics), view).checks]


@pytest.mark.parametrize("flown", ["failed", "passed", "ai"])
def test_card_text_is_plain(pd_flight, flown):
    metrics, view = pd_flight
    if flown == "failed":
        card = flight_card(metrics, view, least_fuel=(0.206, 0.214))
    elif flown == "passed":
        card = flight_card(passing(metrics), view, least_fuel=(0.206, 0.214))
    else:
        card = flight_card({**passing(metrics), "token_cost": 0.14}, view, calls=134, cap=1.0)
    text = card.text()
    lines = text.splitlines()
    assert lines[0] == card.verdict
    assert card.why in words(text)
    for c in card.checks:
        assert c.name.split()[0] in text
        assert words(c.fact) in words(text)
    assert "Needed:" in text
    assert not BANNED.search(text)
    assert not JARGON.search(text)
    assert not PROVIDERS.search(text)
    assert "\x1b" not in text
    assert "out of 100" not in text
    assert "not a grade" in words(text)


def test_card_text_marks_met_and_not_met_in_words(pd_flight):
    metrics, view = pd_flight
    text = flight_card(metrics, view).text().lower()
    assert "not met" in text
    assert re.search(r"(?<!not )\bmet\b", text)


def graded_cards(passed, flown, keep_out=None):
    keep_out = flown - passed if keep_out is None else keep_out
    cards = [{"success": True, "failure_label": None, "min_margin": {"keep_out": 1.0}}] * passed
    for k in range(flown - passed):
        entered = k < keep_out
        cards.append(
            {
                "success": False,
                "failure_label": "keep_out_entry" if entered else "timeout_far",
                "min_margin": {"keep_out": -1.0 if entered else 1.0},
            }
        )
    return cards


def test_two_of_a_hundred_gives_the_exact_range():
    graded = graded_card(graded_cards(2, 100))
    low, high = binomtest(2, 100).proportion_ci(confidence_level=0.95, method="exact")
    assert (graded.passed, graded.flown) == (2, 100)
    assert graded.low == pytest.approx(low, abs=1e-12)
    assert graded.high == pytest.approx(high, abs=1e-12)
    text = words(graded.text())
    assert "2 of 100" in text
    assert "between 0.2% and 7.0%" in text
    assert "keep-out zone on 98 of them" in text


def test_all_starts_passed_reaches_one_hundred_percent():
    graded = graded_card(graded_cards(40, 40))
    assert graded.high == 1.0
    text = words(graded.text())
    assert "all 40" in text and "100%" in text
    assert "Never entered the keep-out zone" in text


def test_graded_card_needs_a_flight():
    with pytest.raises(ValueError):
        graded_card([])


def test_crashed_flight_counts_as_not_passed():
    crashed = {"success": False, "failure_label": "error"}
    graded = graded_card(graded_cards(1, 1) + [crashed])
    assert (graded.passed, graded.flown) == (1, 2)


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 200).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))))
def test_graded_range_holds_the_observed_rate(counts):
    passed, flown = counts
    graded = graded_card(graded_cards(passed, flown))
    assert (graded.low, graded.high) == pytest.approx(clopper_pearson(passed, flown))
    assert 0.0 <= graded.low <= passed / flown <= graded.high <= 1.0
    assert f"{passed} of {flown}" in words(graded.text()) or passed == flown


@pytest.fixture(scope="module")
def short_flights():
    """Four-step drifting flights: a docking, an inspection and a departure."""
    flights = {}
    for name in ("docking", "inspection", "separation-flyaround"):
        env = make(name, max_steps=4)
        ep = record_episode(env, DriftAgent(), seed=1, agent_seed=2)
        flights[name] = evaluate(ep, env.metric_settings()), ep.view
    return flights


def test_docking_arrival_states_the_contact_rule(short_flights):
    metrics, view = short_flights["docking"]
    arrival = check(flight_card(metrics, view), "Arrival")
    assert arrival.status == "Not met"
    assert arrival.fact == "Never touched the port."
    # axial_speed_window (0.05, 0.1) m/s and lateral_limit 0.1 m in the docking goal.
    assert "0.05 to 0.1 m/s" in arrival.needed
    assert "10 cm" in arrival.needed


def test_inspection_arrival_states_the_coverage_rule(short_flights):
    metrics, view = short_flights["inspection"]
    arrival = check(flight_card(metrics, view), "Arrival")
    assert arrival.status == ("Met" if metrics["success"] else "Not met")
    assert "of the target's surface" in arrival.fact
    assert "95%" in arrival.needed


def test_departure_arrival_is_not_judged_without_a_pass(short_flights):
    metrics, view = short_flights["separation-flyaround"]
    assert not metrics["success"]
    arrival = check(flight_card(metrics, view), "Arrival")
    assert arrival.status == "Not judged"
    # clear_range 609.6 m and drift_radius 30 m in the departure goal.
    assert "609.6 m" in arrival.needed and "30 m" in arrival.needed


def test_not_judged_reads_in_the_text(short_flights):
    metrics, view = short_flights["separation-flyaround"]
    assert "not judged" in flight_card(metrics, view).text().lower()


def test_crashed_flight_judges_nothing(pd_flight):
    _, view = pd_flight
    card = flight_card({"success": False, "failure_label": "error"}, view)
    assert card.verdict == "Did not pass"
    assert card.why == WHY["error"]
    assert check(card, "Arrival").status == "Not judged"
    assert check(card, "Keep-out zone").status == "Not judged"


def test_short_hold_reads_as_at_most(pd_flight):
    metrics, view = pd_flight
    flown = {**metrics, "failure_label": "hold_not_maintained", "hold_dwell": 300.0}
    flown["min_margin"] = {**metrics["min_margin"], "keep_out": 1.0}
    card = flight_card(flown, view)
    arrival = check(card, "Arrival")
    assert arrival.status == "Not met"
    assert "at most 5 min" in arrival.fact
    assert card.why == WHY["hold_not_maintained"]


def test_never_held_reads_as_never(pd_flight):
    metrics, view = pd_flight
    flown = {**metrics, "failure_label": "timeout_far", "hold_dwell": 0.0}
    assert "Never held" in check(flight_card(flown, view), "Arrival").fact


def test_hard_breach_beside_a_failure_is_recorded(pd_flight):
    metrics, view = pd_flight
    # Without soft names, the PD flight's speed breaches are not advisory.
    seen = check(flight_card(metrics, view, soft=()), "Also seen")
    assert seen.status == "Recorded"
    assert "does not fail" not in seen.needed


def test_soft_breach_on_a_failed_flight_is_advisory(pd_flight):
    metrics, view = pd_flight
    assert metrics["min_margin"]["passive_speed"] < 0.0
    card = flight_card(metrics, view, soft=make("approach-vbar").soft_constraints)
    assert card.verdict == "Did not pass"
    seen = check(card, "Also seen")
    assert seen.status == "Advisory"
    assert seen.needed == "Recorded; it does not fail the flight."


@pytest.mark.parametrize("rate, unread", [(0.03, 3), (1.0, 100)])
def test_unreadable_replies_count_decisions_not_calls(pd_flight, rate, unread):
    metrics, view = pd_flight
    flown = {**passing(metrics), "token_cost": 0.4, "parse_failure_rate": rate}
    # Tool rounds make 300 calls for 100 decisions; at most 100 replies can go unread.
    cost = check(flight_card(flown, view, calls=300, cap=1.0, decisions=100), "Cost")
    assert f"{unread} replies could not be read" in cost.fact
    assert ": 300 calls" in cost.fact


def test_soft_and_other_breaches_are_separate_checks(pd_flight):
    metrics, view = pd_flight
    assert metrics["min_margin"]["closing_speed_limit"] < 0.0
    assert metrics["min_margin"]["passive_speed"] < 0.0
    card = flight_card(metrics, view, soft=("passive_speed",))
    seen = {c.status: c for c in card.checks if c.name == "Also seen"}
    assert sorted(seen) == ["Advisory", "Recorded"]
    assert seen["Advisory"].needed == "Recorded; it does not fail the flight."
    assert "does not fail" not in seen["Recorded"].needed
    assert "(advisory)" not in card.text().lower()


def docking_contact(metrics, lateral, axial, sideways):
    """A stand-in docking flight that touched the port with the given figures (m, m/s)."""
    return {
        **metrics,
        "success": False,
        "failure_label": "contact_out_of_tolerance",
        "capture_margin": -0.5,
        "contact_lateral_error": lateral,
        "contact_axial_speed": axial,
        "contact_lateral_speed": sideways,
    }


def test_whole_speed_limit_breach_is_not_called_closing(short_flights):
    metrics, view = short_flights["docking"]
    flown = {**metrics, "success": False, "failure_label": "speed_limit_breach"}
    flown["min_margin"] = {**metrics["min_margin"], "speed_limit": -0.1}
    # speed_limit bounds the whole relative speed, in any direction.
    why = flight_card(flown, view).why
    assert "fast" in why
    assert "toward the target" not in why


def test_closing_speed_breach_is_called_closing(pd_flight):
    metrics, view = pd_flight
    flown = {**metrics, "failure_label": "speed_limit_breach"}
    flown["min_margin"] = {**metrics["min_margin"], "keep_out": 1.0, "closing_speed_limit": -0.1}
    assert "toward the target" in flight_card(flown, view).why


def test_docking_rule_names_every_contact_limit(short_flights):
    metrics, view = short_flights["docking"]
    needed = check(flight_card(metrics, view), "Arrival").needed
    # lateral_speed_limit 0.03 m/s in the docking goal.
    assert "0.03 m/s" in needed and "sideways" in needed
    assert "0.05 to 0.1 m/s" in needed and "10 cm" in needed


def test_contact_failed_on_sideways_speed_says_so(short_flights):
    metrics, view = short_flights["docking"]
    # Within 10 cm of center and inside the 0.05 to 0.1 m/s window; 0.05 m/s sideways > 0.03.
    card = flight_card(docking_contact(metrics, 0.02, 0.07, 0.05), view)
    assert "sideways" in card.why
    assert "off center" not in card.why and "too fast" not in card.why
    assert "sideways" in check(card, "Arrival").fact


def test_contact_failed_by_being_too_slow_says_so(short_flights):
    metrics, view = short_flights["docking"]
    # 0.02 m/s is below the 0.05 m/s floor of the axial window; offset and sideways speed are fine.
    card = flight_card(docking_contact(metrics, 0.02, 0.02, 0.01), view)
    assert "slow" in card.why
    assert "too fast" not in card.why and "off center" not in card.why
