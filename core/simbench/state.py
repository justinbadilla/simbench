"""SimUser state, kept apart from the chat text.

The harness updates this after every message. The model never edits it, and
the model's own account of what it said is never trusted. State changes only
because of what the messages actually contain, matched against the spec.

For each customer fact the state tracks hidden, available, or revealed. For
each unknown fact it tracks whether the agent has said it yet. The same
``observe`` code runs live during a run and again when checks replay a stored
transcript, so the two cannot drift apart.
"""

from simbench.models import Message, Role, StateEvent, StateEventKind
from simbench.spec import Fact, SimUserSpec, UnknownFact, normalize


def _contains_any(text: str, phrases: list[str]) -> bool:
    """Return True if any phrase appears as a substring of ``text``.

    Both arguments must already be normalized (see ``spec.normalize``). This is
    the only matching primitive in the state tracker.
    """
    return any(phrase in text for phrase in phrases)


class SimUserState:
    """Harness-owned record of what the SimUser has been asked, said, and learned.

    One instance is created per run. ``run_llm_simulation`` feeds it every
    message through ``observe`` and reads the ``shareable_facts`` and related
    views to build the stateful prompt. ``replay`` builds the same object from a
    finished transcript so the checks see exactly what a live run would.

    Attributes:
        unlocked: On-request fact id -> turn index of the agent message that asked for it.
        revealed: Fact id -> turn index of the first customer message that said it.
        learned: Unknown fact id -> turn index of the agent message that said it.
        early_reveals: Ids of on-request facts said before they were asked for.
        unknown_used: Unknown fact id -> turn the customer said it before the agent did.
        events: Ordered log of every state change, stored on the transcript.
    """

    def __init__(self, spec: SimUserSpec) -> None:
        """Start with nothing unlocked, revealed, or learned for the given spec."""
        self.spec = spec
        self.unlocked: dict[str, int] = {}  # on-request fact id -> turn the agent asked
        self.revealed: dict[str, int] = {}  # fact id -> first turn the customer said it
        self.learned: dict[str, int] = {}  # unknown fact id -> turn the agent said it
        self.early_reveals: set[str] = set()
        self.unknown_used: dict[str, int] = {}  # unknown fact id -> turn used before learned
        self.events: list[StateEvent] = []

    def observe(self, message: Message) -> None:
        """Update the state from one message that was actually sent.

        Routes by speaker: agent messages can unlock facts and teach unknown
        facts, customer messages can reveal facts or misuse unknown ones. Call
        this in conversation order, once per message.

        Called by:
            ``run_llm_simulation`` (live) and ``replay`` (checks and tests).
        """
        text = normalize(message.content)
        if message.role is Role.agent:
            self._observe_agent(message.turn_index, text)
        else:
            self._observe_customer(message.turn_index, text)

    def _observe_agent(self, turn: int, text: str) -> None:
        """Handle an agent message (``text`` is normalized).

        An on-request fact is unlocked the first time the agent's text contains
        one of its unlock phrases. An unknown fact is learned the first time the
        agent's text contains one of its markers. Each happens at most once.

        Heuristic caveat: any mention of an unlock phrase counts as a request,
        even if the agent was not really asking.
        """
        for fact in self.spec.facts:
            if (
                fact.disclosure == "on_request"
                and fact.id not in self.unlocked
                and _contains_any(text, fact.unlock_phrases)
            ):
                self.unlocked[fact.id] = turn
                self._log(turn, StateEventKind.unlocked, fact.id)
        for unknown in self.spec.unknown_facts:
            if unknown.id not in self.learned and _contains_any(text, unknown.markers):
                self.learned[unknown.id] = turn
                self._log(turn, StateEventKind.learned, unknown.id)

    def _observe_customer(self, turn: int, text: str) -> None:
        """Handle a customer message (``text`` is normalized).

        A fact is marked revealed the first time one of its markers appears. If
        it was on-request and not yet unlocked, the event is ``early_reveal``
        instead of ``revealed``. An unknown fact whose marker appears before the
        agent has said it is logged once as ``unknown_fact_used``.
        """
        for fact in self.spec.facts:
            if fact.id in self.revealed or not _contains_any(text, fact.markers):
                continue
            self.revealed[fact.id] = turn
            if fact.disclosure == "on_request" and fact.id not in self.unlocked:
                self.early_reveals.add(fact.id)
                self._log(turn, StateEventKind.early_reveal, fact.id)
            else:
                self._log(turn, StateEventKind.revealed, fact.id)
        for unknown in self.spec.unknown_facts:
            if (
                unknown.id not in self.learned
                and unknown.id not in self.unknown_used
                and _contains_any(text, unknown.markers)
            ):
                self.unknown_used[unknown.id] = turn
                self._log(turn, StateEventKind.unknown_fact_used, unknown.id)

    def _log(self, turn: int, kind: StateEventKind, fact_id: str) -> None:
        """Append one ``StateEvent`` to the ordered event log."""
        self.events.append(StateEvent(turn_index=turn, kind=kind, fact_id=fact_id))

    # Views used to build the stateful prompt.
    def shareable_facts(self) -> list[Fact]:
        """Facts the customer may say right now, in spec order.

        A fact qualifies if it has not been said yet and is either volunteered or
        has been unlocked by an agent request.

        Called by:
            ``_stateful_knowledge`` in ``sim_user.py`` to fill the "may share now" list.
        """
        return [
            fact
            for fact in self.spec.facts
            if fact.id not in self.revealed
            and (fact.disclosure == "volunteer" or fact.id in self.unlocked)
        ]

    def asked_for(self, fact: Fact) -> bool:
        """Return True if ``fact`` is on-request and the agent has asked for it.

        Called by:
            ``_stateful_knowledge`` to add the "(the agent asked for this)" note.
        """
        return fact.disclosure == "on_request" and fact.id in self.unlocked

    def revealed_facts(self) -> list[Fact]:
        """Facts the customer has already said, in spec order.

        Called by:
            ``_stateful_knowledge`` to build the "already told the agent" list.
        """
        return [fact for fact in self.spec.facts if fact.id in self.revealed]

    def unknown_fact(self, fact_id: str) -> UnknownFact:
        """Look up an unknown fact by id.

        Raises:
            KeyError: If the spec has no unknown fact with that id.

        Called by:
            ``_unknown_fact_not_used`` in ``checks.py`` to get the human label.
        """
        for unknown in self.spec.unknown_facts:
            if unknown.id == fact_id:
                return unknown
        raise KeyError(fact_id)


def replay(spec: SimUserSpec, messages: list[Message]) -> SimUserState:
    """Rebuild the state a finished transcript implies.

    Args:
        spec: The SimUser spec the run used.
        messages: The transcript messages, in order.

    Returns:
        A ``SimUserState`` as if every message had been observed live.

    Called by:
        ``checks.evaluate``, and the tests that compare live and replayed state.
    """
    state = SimUserState(spec)
    for message in messages:
        state.observe(message)
    return state
