"""Shared fixtures. The spec here is independent of data/scenarios so tests do not shift when a scenario is edited."""

import pytest

from simbench.models import Message, Role
from simbench.spec import SimUserSpec

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
