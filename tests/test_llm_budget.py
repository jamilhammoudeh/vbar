"""A budget on model calls: a call that would pass the cap is refused before it is made."""

import json

import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import (
    BudgetedModel,
    BudgetExceeded,
    LLMAgent,
    LLMSupervisor,
    Price,
    Reply,
    SetpointTracker,
    SupervisedAgent,
)
from vbar.scenarios import make, seeds

PRICE = Price(3.0, 15.0)
MESSAGES = [{"role": "user", "content": "range 400 m"}]


class Counting:
    """A stand-in model with fixed replies and token counts that counts its calls."""

    def __init__(self, text='{"ax": 0.0, "ay": 0.0, "az": 0.0}', tokens=(100, 10)):
        self.text, self.tokens, self.calls = text, tokens, 0

    def __call__(self, messages, tools):
        self.calls += 1
        return Reply(self.text, *self.tokens)


def worst(messages, tools, max_output_tokens):
    # One token is at least one UTF-8 byte of the request's JSON.
    size = len(json.dumps({"messages": messages, "tools": tools}).encode("utf-8"))
    return PRICE.cost(size, max_output_tokens)


def test_the_call_past_the_call_cap_is_refused_and_never_reaches_the_model():
    model = Counting()
    capped = BudgetedModel(model, max_calls=2)
    capped(MESSAGES, [])
    capped(MESSAGES, [])
    with pytest.raises(BudgetExceeded):
        capped(MESSAGES, [])
    assert model.calls == capped.calls == 2


def test_spend_is_the_declared_price_of_the_reported_tokens():
    capped = BudgetedModel(
        Counting(tokens=(120, 30)), price=PRICE, max_usd=1.0, max_output_tokens=50
    )
    capped(MESSAGES, [])
    capped(MESSAGES, [])
    assert capped.spent_usd == pytest.approx(2 * PRICE.cost(120, 30))


def test_a_call_whose_worst_case_would_pass_the_dollar_cap_is_refused():
    one = worst(MESSAGES, [], 50)
    model = Counting(tokens=(1, 1))
    # Room for the first call's worst case, not for a second after the first's actual cost.
    cap = PRICE.cost(1, 1) + one - 1e-12
    capped = BudgetedModel(model, price=PRICE, max_usd=cap, max_output_tokens=50)
    capped(MESSAGES, [])
    with pytest.raises(BudgetExceeded):
        capped(MESSAGES, [])
    assert model.calls == 1
    assert capped.spent_usd <= cap


def test_a_reply_without_token_counts_is_charged_its_worst_case():
    capped = BudgetedModel(Counting(tokens=(0, 0)), price=PRICE, max_usd=1.0, max_output_tokens=50)
    capped(MESSAGES, [])
    assert capped.spent_usd == pytest.approx(worst(MESSAGES, [], 50))


@pytest.mark.parametrize(
    "tokens",
    [(float("nan"), 1), (-100_000, 0), (0, 1), (100, -5), (1.5, 10), (True, 10), (100, None)],
)
def test_a_reply_with_unusable_token_counts_is_charged_its_worst_case(tokens):
    capped = BudgetedModel(Counting(tokens=tokens), price=PRICE, max_usd=1.0, max_output_tokens=50)
    capped(MESSAGES, [])
    assert capped.spent_usd == pytest.approx(worst(MESSAGES, [], 50))


def test_a_failed_call_is_charged_its_worst_case_and_the_error_passes_through():
    def failing(messages, tools):
        raise TimeoutError("no reply")

    capped = BudgetedModel(failing, price=PRICE, max_usd=1.0, max_output_tokens=50)
    with pytest.raises(TimeoutError):
        capped(MESSAGES, [])
    assert capped.calls == 1
    assert capped.spent_usd == pytest.approx(worst(MESSAGES, [], 50))


def test_a_text_reply_is_charged_its_worst_case_and_returned_as_a_reply():
    capped = BudgetedModel(
        lambda m, t: "plain text", price=PRICE, max_usd=1.0, max_output_tokens=50
    )
    reply = capped(MESSAGES, [])
    assert reply.text == "plain text"
    assert capped.spent_usd == pytest.approx(worst(MESSAGES, [], 50))


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"max_calls": 0},
        {"max_calls": 1.5},
        {"max_calls": True},
        {"max_usd": 1.0},
        {"max_usd": 1.0, "price": PRICE},
        {"max_usd": 1.0, "max_output_tokens": 50},
        {"max_usd": 0.0, "price": PRICE, "max_output_tokens": 50},
        {"max_usd": float("nan"), "price": PRICE, "max_output_tokens": 50},
        {"max_usd": True, "price": PRICE, "max_output_tokens": 50},
        {"max_usd": "1", "price": PRICE, "max_output_tokens": 50},
        {"max_usd": 1.0, "price": Price(float("nan"), 1.0), "max_output_tokens": 50},
        {"max_usd": 1.0, "price": Price(1.0, float("inf")), "max_output_tokens": 50},
        {"max_usd": 1.0, "price": Price(-5.0, -5.0), "max_output_tokens": 50},
        {"max_calls": 3, "price": Price(1.0, -1.0)},
        {"max_usd": 1.0, "price": PRICE, "max_output_tokens": 0},
    ],
)
def test_a_budget_needs_a_positive_cap_and_a_dollar_cap_needs_a_price_and_a_reply_limit(fields):
    with pytest.raises(ValueError):
        BudgetedModel(Counting(), **fields)


def test_an_llm_agent_stops_its_run_at_the_cap():
    model = Counting()
    env = ApproachEnv(ApproachConfig(max_steps=10, start_state=(0.0, 400.0, 0.0, 0.0, 0.0, 0.0)))
    with pytest.raises(BudgetExceeded):
        run_episode(env, LLMAgent(BudgetedModel(model, max_calls=3)))
    assert model.calls == 3


def test_a_supervised_flight_stops_at_the_cap_and_the_refusal_is_not_read_as_continue():
    model = Counting(text='{"command": "CONTINUE"}')
    agent = SupervisedAgent(
        LLMSupervisor(BudgetedModel(model, max_calls=3)), tracker=SetpointTracker()
    )
    seed, agent_seed = seeds("approach-vbar", "validation", 1).pairs()[0]
    with pytest.raises(BudgetExceeded):
        run_episode(make("approach-vbar"), agent, seed=seed, agent_seed=agent_seed)
    assert model.calls == 3
