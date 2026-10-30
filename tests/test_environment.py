"""The mock database and its tools: validation, permissions, and what gets written."""

import copy

import pytest
from pydantic import ValidationError

from simbench.environment import READ_ONLY_TOOLS, TOOLS, Environment
from simbench.world import EnvironmentSpec

from conftest import ENV_DATA

ALL = list(TOOLS)


def call(env: Environment, tool: str, arguments: object, *, allowed=ALL, turn: int = 0, step: int = 0):
    """Shorthand for a tool call with sensible defaults."""
    return env.call(tool, arguments, allowed=allowed, turn_index=turn, step=step)


def test_lookup_returns_order_facts_but_never_the_email(env_spec) -> None:
    result = call(Environment(env_spec), "lookup_order", {"order_id": "1042"})

    assert result.ok is True
    assert result.data["status"] == "in_transit"
    assert result.data["days_late"] == 6
    assert result.data["tracking_number"] == "TRK-88213"
    assert result.data["total_usd"] == 49.99
    assert result.data["already_refunded"] is False
    assert "jordan.lee@example.com" not in str(result.data)
    assert "email" not in " ".join(result.data)


def test_days_late_uses_the_delivery_date_for_a_delivered_order(env_spec) -> None:
    result = call(Environment(env_spec), "lookup_order", {"order_id": "2087"})

    assert result.data["days_late"] == 0
    assert result.data["delivered_on"] == "2026-10-22"


def test_unknown_order_is_refused_with_a_reason(env_spec) -> None:
    result = call(Environment(env_spec), "lookup_order", {"order_id": "9999"})

    assert result.ok is False
    assert result.error.startswith("order_not_found")


def test_whole_number_order_ids_are_accepted(env_spec) -> None:
    assert call(Environment(env_spec), "lookup_order", {"order_id": 1042}).ok is True


def test_refund_writes_a_row_and_records_whether_the_email_matched(env_spec) -> None:
    env = Environment(env_spec)

    result = call(
        env,
        "issue_refund",
        {"order_id": "1042", "customer_email": " Jordan.Lee@Example.com ", "reason": "late"},
        turn=3,
    )

    assert result.ok is True
    assert result.data == {
        "refund_id": "R-001",
        "order_id": "1042",
        "amount_usd": 49.99,
        "refunded_to": "Visa ending 4242",
    }
    refund = env.snapshot().refunds[0]
    assert (refund.amount_cents, refund.email_matched, refund.turn_index) == (4999, True, 3)
    assert env.snapshot().orders[0].refunded is True


def test_a_wrong_email_is_recorded_not_refused_and_not_revealed(env_spec) -> None:
    env = Environment(env_spec)

    result = call(
        env,
        "issue_refund",
        {"order_id": "1042", "customer_email": "someone@else.com", "reason": "late"},
    )

    assert result.ok is True
    assert "email_matched" not in str(result.data)
    assert env.snapshot().refunds[0].email_matched is False


def test_the_same_order_cannot_be_refunded_twice(env_spec) -> None:
    env = Environment(env_spec)
    args = {"order_id": "1042", "customer_email": "jordan.lee@example.com", "reason": "late"}
    call(env, "issue_refund", args)

    second = call(env, "issue_refund", args)

    assert second.ok is False and second.error.startswith("already_refunded")
    assert len(env.snapshot().refunds) == 1
    assert call(env, "lookup_order", {"order_id": "1042"}).data["already_refunded"] is True


def test_tools_do_not_enforce_policy_so_an_ineligible_refund_goes_through(env_spec) -> None:
    result = call(
        Environment(env_spec),
        "issue_refund",
        {"order_id": "2087", "customer_email": "casey.morgan@example.com", "reason": "changed mind"},
    )

    assert result.ok is True


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ({}, "missing_argument"),
        ({"order_id": "1042", "customer_email": "a@b.co"}, "missing_argument"),
        ({"order_id": "1042", "customer_email": "a@b.co", "reason": "x", "extra": "y"}, "unexpected_argument"),
        ({"order_id": "   ", "customer_email": "a@b.co", "reason": "x"}, "invalid_argument"),
        ({"order_id": ["1042"], "customer_email": "a@b.co", "reason": "x"}, "invalid_argument"),
        ({"order_id": True, "customer_email": "a@b.co", "reason": "x"}, "invalid_argument"),
        ("1042", "invalid_argument"),
        (None, "invalid_argument"),
    ],
)
def test_bad_arguments_are_refused_and_change_nothing(env_spec, arguments, code) -> None:
    env = Environment(env_spec)

    result = call(env, "issue_refund", arguments)

    assert result.ok is False and result.error.startswith(code)
    assert env.snapshot().refunds == []


def test_unknown_tool_lists_the_real_ones(env_spec) -> None:
    result = call(Environment(env_spec), "delete_everything", {})

    assert result.error.startswith("unknown_tool")
    assert "issue_refund" in result.error


def test_a_tool_outside_the_agents_permissions_is_refused(env_spec) -> None:
    env = Environment(env_spec)

    result = call(
        env,
        "issue_refund",
        {"order_id": "1042", "customer_email": "jordan.lee@example.com", "reason": "late"},
        allowed=READ_ONLY_TOOLS,
    )

    assert result.error.startswith("tool_not_permitted")
    assert env.snapshot().refunds == []


def test_read_only_tools_are_exactly_the_ones_that_do_not_write() -> None:
    assert READ_ONLY_TOOLS == ["lookup_order"]


def test_ticket_is_created_and_may_name_an_order(env_spec) -> None:
    env = Environment(env_spec)

    assert call(env, "create_ticket", {"summary": "Customer wants a human."}).data == {"ticket_id": "T-001"}
    assert call(env, "create_ticket", {"summary": "Late order", "order_id": "1042"}).ok is True
    assert [t.order_id for t in env.snapshot().tickets] == ["", "1042"]
    assert call(env, "create_ticket", {"summary": "x", "order_id": "9999"}).error.startswith("order_not_found")


def test_every_call_is_logged_including_refused_ones(env_spec) -> None:
    env = Environment(env_spec)
    call(env, "lookup_order", {"order_id": "1042"}, turn=1, step=0)
    call(env, "issue_refund", {}, turn=1, step=1)
    call(env, "nope", {}, turn=3, step=0)

    assert [(c.turn_index, c.step, c.tool, c.ok) for c in env.calls] == [
        (1, 0, "lookup_order", True),
        (1, 1, "issue_refund", False),
        (3, 0, "nope", False),
    ]
    assert env.calls[0].result["order_id"] == "1042"
    assert env.calls[1].error.startswith("missing_argument")


def test_each_environment_starts_fresh_from_the_spec(env_spec) -> None:
    first = Environment(env_spec)
    call(first, "issue_refund", {"order_id": "1042", "customer_email": "a@b.co", "reason": "x"})

    second = Environment(env_spec)

    assert second.snapshot().refunds == []
    assert call(second, "lookup_order", {"order_id": "1042"}).data["already_refunded"] is False


# ---- Spec validation ----


def variant(mutate) -> dict:
    data = copy.deepcopy(ENV_DATA)
    mutate(data)
    return data


def test_duplicate_order_ids_are_rejected() -> None:
    data = variant(lambda d: d["orders"][1].update(order_id="1042"))
    with pytest.raises(ValidationError, match="duplicate order ids: 1042"):
        EnvironmentSpec.model_validate(data)


def test_expected_refund_must_name_a_real_order() -> None:
    data = variant(lambda d: d["expected_outcome"]["refunds"][0].update(order_id="7777"))
    with pytest.raises(ValidationError, match="unknown order '7777'"):
        EnvironmentSpec.model_validate(data)


def test_an_order_cannot_be_expected_to_be_refunded_twice() -> None:
    data = variant(lambda d: d["expected_outcome"]["refunds"].append(d["expected_outcome"]["refunds"][0]))
    with pytest.raises(ValidationError, match="refunded twice"):
        EnvironmentSpec.model_validate(data)


def test_delivered_status_and_delivery_date_must_agree() -> None:
    missing = variant(lambda d: d["orders"][1].pop("delivered_on"))
    with pytest.raises(ValidationError, match="no delivered_on"):
        EnvironmentSpec.model_validate(missing)
    extra = variant(lambda d: d["orders"][0].update(delivered_on="2026-10-21"))
    with pytest.raises(ValidationError, match="not delivered"):
        EnvironmentSpec.model_validate(extra)
