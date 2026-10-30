"""Shared fixtures. The spec here is independent of data/scenarios so tests do not shift when a scenario is edited."""

import pytest

from simbench.models import Message, Role
from simbench.spec import SimUserSpec
from simbench.world import EnvironmentSpec

SPEC_DATA = {
    "goal": {
        "text": "Get a refund for order 1042.",
        "markers": ["refund"],
        "abandon_markers": ["never mind", "forget it"],
    },
    "style": "Brief.",
    "done_when": "End after the refund is confirmed.",
    "facts": [
        {
            "id": "order_number",
            "text": "The order number is 1042.",
            "markers": ["1042"],
            "required": True,
        },
        {
            "id": "due_date",
            "text": "The order was due last Tuesday.",
            "markers": ["tuesday"],
            "required": True,
        },
        {
            "id": "account_email",
            "text": "The email on the account is jordan.lee@example.com.",
            "markers": ["jordan.lee@example.com"],
            "disclosure": "on_request",
            "unlock_phrases": ["email", "e-mail"],
        },
    ],
    "unknown_facts": [
        {"id": "tracking_number", "label": "the tracking number", "markers": ["trk-88213"]},
    ],
}


@pytest.fixture
def spec() -> SimUserSpec:
    """A validated spec with volunteered, on-request, and unknown facts. Fresh for each test."""
    return SimUserSpec.model_validate(SPEC_DATA)


@pytest.fixture
def convo():
    """Build messages from ("u" | "a", text) pairs. "u" is the customer, "a" the agent."""

    def build(*pairs: tuple[str, str]) -> list[Message]:
        """Return numbered ``Message`` objects, alternating speakers as given."""
        return [
            Message(
                turn_index=index,
                role=Role.sim_user if who == "u" else Role.agent,
                content=text,
            )
            for index, (who, text) in enumerate(pairs)
        ]

    return build


ENV_DATA = {
    "as_of": "2026-10-26",
    "policy_text": "Refund orders at least 3 days late. Confirm the account email first.",
    "orders": [
        {
            "order_id": "1042",
            "customer_email": "jordan.lee@example.com",
            "status": "in_transit",
            "due_date": "2026-10-20",
            "tracking_number": "TRK-88213",
            "amount_cents": 4999,
            "payment_method": "Visa ending 4242",
        },
        {
            "order_id": "2087",
            "customer_email": "casey.morgan@example.com",
            "status": "delivered",
            "due_date": "2026-10-24",
            "delivered_on": "2026-10-22",
            "tracking_number": "TRK-55102",
            "amount_cents": 2999,
            "payment_method": "Mastercard ending 1881",
        },
    ],
    "expected_outcome": {"refunds": [{"order_id": "1042", "amount_cents": 4999}]},
}


@pytest.fixture
def env_spec() -> EnvironmentSpec:
    """A two-order world: a late order that should be refunded and an on-time one that should not."""
    return EnvironmentSpec.model_validate(ENV_DATA)
