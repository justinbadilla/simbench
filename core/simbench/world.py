"""Data models for the mock support environment.

Everything here is plain data. ``EnvironmentSpec`` is what a scenario file
contains (the starting orders, the policy text, and the date "today" is
pinned to). ``EnvironmentState`` is what is left after a run. The behaviour
that changes one into the other lives in ``environment.py``.

Nothing here is real: the orders, customers, emails, and tracking numbers are
made up, and no external system is touched.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

OrderStatus = Literal["processing", "in_transit", "delivered", "lost"]


class OrderSpec(BaseModel):
    """One order in the starting database.

    ``customer_email`` is the account email. It is never returned by the
    ``lookup_order`` tool, so an agent can only confirm identity by asking the
    customer and passing what they say to ``issue_refund``.
    """

    order_id: str = Field(min_length=1)
    customer_email: str = Field(min_length=3)
    status: OrderStatus
    due_date: date
    delivered_on: date | None = None
    tracking_number: str = Field(min_length=1)
    amount_cents: int = Field(gt=0)
    payment_method: str = Field(min_length=1)

    @model_validator(mode="after")
    def delivered_orders_have_a_delivery_date(self) -> "OrderSpec":
        """Pydantic hook: a delivered order needs ``delivered_on``; any other status must not have one."""
        if self.status == "delivered" and self.delivered_on is None:
            raise ValueError(f"order {self.order_id!r} is delivered but has no delivered_on date")
        if self.status != "delivered" and self.delivered_on is not None:
            raise ValueError(f"order {self.order_id!r} has delivered_on but is not delivered")
        return self


class ExpectedRefund(BaseModel):
    """A refund that must exist in the final database for the run to count as a success."""

    order_id: str = Field(min_length=1)
    amount_cents: int = Field(gt=0)


class ExpectedOutcome(BaseModel):
    """What the final database must look like for the task to count as done.

    ``refunds`` is the exact list of refunds expected, in any order. An empty
    list means the correct outcome is that no refund is issued. Success is
    judged only from the database, never from what the agent said.
    """

    refunds: list[ExpectedRefund] = Field(default_factory=list)


class EnvironmentSpec(BaseModel):
    """The starting state of the mock world for one scenario.

    Attributes:
        as_of: The date "today" is pinned to, so days-late is the same on every run.
        policy_text: The refund policy in plain language. Given to agents that are
            configured to see it. The tools do not enforce it (see ``environment.py``).
        orders: The starting orders. Order ids must be unique.
        expected_outcome: What a successful run leaves in the database.
    """

    as_of: date
    policy_text: str = Field(min_length=1)
    orders: list[OrderSpec] = Field(min_length=1)
    expected_outcome: ExpectedOutcome

    @field_validator("orders")
    @classmethod
    def order_ids_are_unique(cls, orders: list[OrderSpec]) -> list[OrderSpec]:
        """Pydantic hook: reject duplicate order ids, which would make lookups ambiguous."""
        ids = [order.order_id for order in orders]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            raise ValueError(f"duplicate order ids: {', '.join(duplicates)}")
        return orders

    @model_validator(mode="after")
    def expected_refunds_point_at_real_orders(self) -> "EnvironmentSpec":
        """Pydantic hook: every expected refund must name an existing order, at most once.

        Catches a typo in a scenario file before a run, where it would otherwise
        make every run look like a failure.
        """
        known = {order.order_id for order in self.orders}
        seen: set[str] = set()
        for expected in self.expected_outcome.refunds:
            if expected.order_id not in known:
                raise ValueError(f"expected refund names unknown order {expected.order_id!r}")
            if expected.order_id in seen:
                raise ValueError(f"order {expected.order_id!r} is expected to be refunded twice")
            seen.add(expected.order_id)
        return self


class Refund(BaseModel):
    """A refund row written by the ``issue_refund`` tool.

    ``email_matched`` is computed by the environment by comparing the email the
    agent passed with the one on file. The agent does not report it, so it
    cannot claim a verification that did not happen.
    """

    refund_id: str
    order_id: str
    amount_cents: int
    email_matched: bool
    reason: str
    turn_index: int = Field(ge=0)


class Ticket(BaseModel):
    """A support ticket row written by the ``create_ticket`` tool."""

    ticket_id: str
    order_id: str = ""
    summary: str
    turn_index: int = Field(ge=0)


class ToolCallRecord(BaseModel):
    """One tool call as the environment saw it, including calls that were refused.

    Attributes:
        turn_index: Position the agent's next message would take, which is the
            number of messages said before the call.
        step: Zero-based order of the call within that agent turn.
        arguments: The arguments exactly as the agent supplied them after parsing.
        ok: Whether the tool ran. False for unknown tools, tools the agent is
            not permitted to use, invalid arguments, and refused operations.
        result: The data returned when ``ok`` is true.
        error: A short machine-readable reason when ``ok`` is false.
    """

    turn_index: int = Field(ge=0)
    step: int = Field(ge=0)
    tool: str
    arguments: dict[str, object] = Field(default_factory=dict)
    ok: bool
    result: dict[str, object] | None = None
    error: str = ""


class OrderState(BaseModel):
    """An order as it stands at the end of a run."""

    order_id: str
    refunded: bool


class EnvironmentState(BaseModel):
    """The database after a run: what was refunded and which tickets exist."""

    orders: list[OrderState]
    refunds: list[Refund] = Field(default_factory=list)
    tickets: list[Ticket] = Field(default_factory=list)
