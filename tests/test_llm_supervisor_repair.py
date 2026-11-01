"""The LLM supervisor's repair of an invalid reply, and its invalid-action and repair counts.

A reply is invalid when it is not an order or orders a command that is not valid now. With
`repairs=k` the supervisor sends the conversation back with the reason, at most k times.
"""

import pytest
from test_llm_supervisor import GO, RETREAT, Replies, a_situation, reply_of, scenario

from vbar.agents.llm import Price, Reply
from vbar.agents.llm_supervisor import LLMSupervisor
from vbar.agents.supervision import Command, Order, ladder


def flown(model, *, repairs, valid=(GO,), price=None):
    view, obs0 = scenario("approach-vbar")
    rungs = ladder(view, obs0)
    supervisor = LLMSupervisor(model, repairs=repairs, price=price)
    supervisor.reset(view, rungs, None)
    decision = supervisor.decide(a_situation(view, rungs, obs0, phase="holding", valid=valid))
    return supervisor, decision


def command_of(decision):
    return decision.command if isinstance(decision, Order) else decision


def test_without_repairs_an_unparseable_reply_is_one_call_and_one_invalid_action():
    model = Replies(Reply("no idea"))
    supervisor, decision = flown(model, repairs=0)
    assert command_of(decision) == Command.CONTINUE
    assert len(model.requests) == 1
    usage = supervisor.usage()
    assert (usage["invalid_actions"], usage["invalid_replies"], usage["repair_attempts"]) == (
        1,
        1,
        0,
    )


def test_without_repairs_a_command_not_valid_now_is_passed_on_and_counted_invalid():
    supervisor, decision = flown(Replies(Reply(reply_of("RETREAT"))), repairs=0)
    assert command_of(decision) == RETREAT
    assert supervisor.usage()["invalid_actions"] == 1
    assert supervisor.usage()["parse_failures"] == 0


def test_a_repaired_reply_is_the_order_and_the_model_saw_its_reply_and_the_reason():
    model = Replies(Reply("no idea", 100, 10), Reply(reply_of("GO"), 120, 12))
    supervisor, decision = flown(model, repairs=2, price=Price(3.0, 15.0))
    assert command_of(decision) == GO
    first, second = (messages for messages, _ in model.requests)
    assert len(first) == 1
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[0] == first[0] and second[1]["content"] == "no idea"
    assert "not a command" in second[2]["content"] and "GO" in second[2]["content"]
    usage = supervisor.usage()
    assert usage["model_calls"] == 2 and usage["decisions"] == 1
    assert (usage["invalid_actions"], usage["invalid_replies"], usage["repair_attempts"]) == (
        0,
        1,
        1,
    )
    assert usage["repaired"] == 1 and usage["parse_failures"] == 0
    assert (usage["input_tokens"], usage["output_tokens"]) == (220, 22)
    assert usage["cost_usd"] == pytest.approx((220 * 3.0 + 22 * 15.0) / 1e6)
    entry = supervisor.log[0]
    assert entry["reply"] == reply_of("GO") and entry["repair_replies"] == ["no idea"]


def test_the_repair_of_a_command_not_valid_now_names_it_and_its_reason():
    model = Replies(Reply(reply_of("RETREAT")), Reply(reply_of("GO")))
    supervisor, decision = flown(model, repairs=1)
    assert command_of(decision) == GO
    repair = model.requests[1][0][-1]["content"]
    assert "RETREAT" in repair and "retreat refused by rule 42" in repair
    assert supervisor.usage()["invalid_actions"] == 0


def test_a_reply_still_invalid_after_every_repair_is_continue_or_the_refused_order():
    garbage = Replies(Reply("no idea"))
    supervisor, decision = flown(garbage, repairs=2)
    assert command_of(decision) == Command.CONTINUE
    assert len(garbage.requests) == 3
    usage = supervisor.usage()
    assert (usage["invalid_actions"], usage["invalid_replies"], usage["repair_attempts"]) == (
        1,
        3,
        2,
    )
    assert usage["repaired"] == 0 and usage["parse_failures"] == 1

    stubborn = Replies(Reply(reply_of("RETREAT")))
    supervisor, decision = flown(stubborn, repairs=1)
    # The agent refuses it, as it refuses any command its mask does not allow.
    assert command_of(decision) == RETREAT
    assert supervisor.usage()["invalid_actions"] == 1


def test_a_valid_first_reply_is_not_repaired():
    model = Replies(Reply(reply_of("GO")))
    supervisor, _ = flown(model, repairs=3)
    assert len(model.requests) == 1
    assert supervisor.usage()["repair_attempts"] == 0
    assert supervisor.log[0]["repair_replies"] == []


@pytest.mark.parametrize("repairs", [-1, 1.5, True, "2"])
def test_repairs_must_be_a_non_negative_integer(repairs):
    with pytest.raises(ValueError):
        LLMSupervisor(Replies(Reply("x")), repairs=repairs)
