"""State tracking: what is hidden, asked for, said, and learned."""

from simbench.models import StateEventKind
from simbench.state import SimUserState, replay


def kinds(state: SimUserState) -> list[tuple[int, str, str]]:
    """Summarize the event log as ``(turn, kind, fact_id)`` tuples for easy comparison."""
    return [(event.turn_index, event.kind.value, event.fact_id) for event in state.events]


def test_on_request_fact_starts_hidden_and_is_unlocked_by_an_agent_request(spec, convo) -> None:
    state = SimUserState(spec)
    assert [fact.id for fact in state.shareable_facts()] == ["order_number", "due_date"]

    messages = convo(
        ("u", "Hi, order 1042 was due last Tuesday."),
        ("a", "Can you confirm the e-mail address on the account?"),
    )
    for message in messages:
        state.observe(message)

    assert [fact.id for fact in state.shareable_facts()] == ["account_email"]
    assert state.asked_for(spec.fact("account_email")) is True
    assert kinds(state) == [
        (0, "revealed", "order_number"),
        (0, "revealed", "due_date"),
        (1, "unlocked", "account_email"),
    ]


def test_revealed_facts_leave_the_shareable_list(spec, convo) -> None:
    messages = convo(
        ("u", "Order 1042."),
        ("a", "What is the email on the account?"),
        ("u", "jordan.lee@example.com"),
    )
    state = replay(spec, messages)

    assert [fact.id for fact in state.revealed_facts()] == ["order_number", "account_email"]
    assert [fact.id for fact in state.shareable_facts()] == ["due_date"]


def test_agent_request_must_come_before_the_reveal(spec, convo) -> None:
    messages = convo(
        ("u", "My email is jordan.lee@example.com and the order is 1042."),
        ("a", "Thanks, and what is the email on the account?"),
    )
    state = replay(spec, messages)

    assert ("early_reveal", "account_email") in [(e.kind.value, e.fact_id) for e in state.events]
    assert state.early_reveals == {"account_email"}
    # Asking afterward does not undo the early reveal.
    assert "account_email" in state.unlocked


def test_agent_stating_an_unknown_fact_makes_it_usable(spec, convo) -> None:
    messages = convo(
        ("u", "Order 1042 is late."),
        ("a", "I can see tracking number TRK-88213 at a depot."),
        ("u", "Yes, TRK-88213 has not moved."),
    )
    state = replay(spec, messages)

    assert state.learned == {"tracking_number": 1}
    assert state.unknown_used == {}
    assert StateEventKind.unknown_fact_used not in {event.kind for event in state.events}


def test_using_an_unknown_fact_before_the_agent_says_it_is_logged_once(spec, convo) -> None:
    messages = convo(
        ("u", "Tracking TRK-88213 says nothing."),
        ("a", "Let me look."),
        ("u", "Again, trk-88213 shows nothing."),
    )
    state = replay(spec, messages)

    used = [event for event in state.events if event.kind is StateEventKind.unknown_fact_used]
    assert [(e.turn_index, e.fact_id) for e in used] == [(0, "tracking_number")]


def test_matching_ignores_case_and_extra_whitespace(spec, convo) -> None:
    messages = convo(
        ("a", "Please confirm   the   E-Mail on the account."),
        ("u", "It is JORDAN.LEE@EXAMPLE.COM, for  Order\n1042 due TUESDAY."),
    )
    state = replay(spec, messages)

    assert set(state.revealed) == {"account_email", "order_number", "due_date"}
    assert state.early_reveals == set()


def test_replay_matches_live_incremental_updates(spec, convo) -> None:
    messages = convo(
        ("u", "Order 1042, due Tuesday."),
        ("a", "What email is on the account?"),
        ("u", "jordan.lee@example.com"),
        ("a", "Tracking number TRK-88213 is delayed."),
    )
    live = SimUserState(spec)
    for message in messages:
        live.observe(message)

    replayed = replay(spec, messages)

    assert live.events == replayed.events
    assert live.revealed == replayed.revealed
    assert live.unlocked == replayed.unlocked
    assert live.learned == replayed.learned
