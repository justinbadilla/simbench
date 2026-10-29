"""Deterministic checks on a transcript. No model is involved.

Each check is a pure function of the spec and the messages, so it runs the same
on a live run, a replayed run, or a hand-built fixture. A check returns
``passed=None`` when the spec gives it nothing to test, and that is not a pass.

These are keyword checks. They catch a fact that was said with one of its
markers, and they miss a paraphrase. A SimUser that invents a fact the spec
never mentions is not detected here.
"""

from simbench.models import CheckResult, ChecksReport, Message, Role, StateEventKind
from simbench.spec import SimUserSpec, normalize
from simbench.state import SimUserState, replay

CHECKS_VERSION = "checks.v1"


def evaluate(spec: SimUserSpec, messages: list[Message]) -> ChecksReport:
    """Run every check against a transcript and bundle the results.

    Args:
        spec: The SimUser spec the run used.
        messages: The transcript messages, in order.

    Returns:
        A ``ChecksReport`` with one ``CheckResult`` per check, the checks
        version, and ``all_passed``. ``all_passed`` is False only if some check
        returned ``passed=False``; checks that did not apply (``None``) do not
        count against it.

    Called by:
        ``run_llm_simulation`` once the conversation ends (skipped on
        ``model_error``). Tests call it directly on hand-built transcripts.
    """
    state = replay(spec, messages)
    results = [
        _unknown_fact_not_used(spec, state),
        _no_early_reveal(spec, state),
        _fact_coverage(spec, state),
        _answered_when_asked(spec, state, messages),
        _goal_stated(spec, messages),
        _goal_not_abandoned(spec, messages),
    ]
    return ChecksReport(
        checks_version=CHECKS_VERSION,
        checks=results,
        all_passed=all(result.passed is not False for result in results),
    )


def _turn(index: int) -> str:
    """Format a zero-based message index as the 1-based "turn N" used in evidence text.

    The UI numbers turns from 1, so evidence lines match what the user sees.
    """
    return f"turn {index + 1}"


def _unknown_fact_not_used(spec: SimUserSpec, state: SimUserState) -> CheckResult:
    """Fail if the customer said a fact it could not yet know.

    Reads the ``unknown_fact_used`` events from the state log. Not applicable
    when the spec has no unknown facts. Repeating a fact after the agent said it
    is allowed.
    """
    name = "unknown_fact_not_used"
    if not spec.unknown_facts:
        return CheckResult(name=name, passed=None, detail="The spec lists no facts the customer must not know.")
    evidence = [
        f"{_turn(event.turn_index)}: said {state.unknown_fact(event.fact_id).label} before the agent did"
        for event in state.events
        if event.kind is StateEventKind.unknown_fact_used
    ]
    if evidence:
        return CheckResult(
            name=name,
            passed=False,
            detail=f"The customer used {len(evidence)} fact(s) it should not have known.",
            evidence=evidence,
        )
    return CheckResult(name=name, passed=True, detail="The customer did not use a fact before the agent said it.")


def _no_early_reveal(spec: SimUserSpec, state: SimUserState) -> CheckResult:
    """Fail if the customer volunteered an on-request fact before the agent asked.

    Reads the ``early_reveal`` events from the state log. Not applicable when
    the spec has no on-request facts.
    """
    name = "no_early_reveal"
    if not any(fact.disclosure == "on_request" for fact in spec.facts):
        return CheckResult(name=name, passed=None, detail="The spec has no on-request facts.")
    evidence = [
        f"{_turn(event.turn_index)}: said '{spec.fact(event.fact_id).text}' before the agent asked"
        for event in state.events
        if event.kind is StateEventKind.early_reveal
    ]
    if evidence:
        return CheckResult(
            name=name,
            passed=False,
            detail=f"The customer volunteered {len(evidence)} detail(s) it should have held back.",
            evidence=evidence,
        )
    return CheckResult(name=name, passed=True, detail="The customer held back every on-request detail until asked.")


def _fact_coverage(spec: SimUserSpec, state: SimUserState) -> CheckResult:
    """Fail if any ``required`` fact was never said.

    ``value`` is the fraction of required facts conveyed (0.0 to 1.0), which is
    useful for aggregating across runs. Not applicable when no fact is required.
    """
    name = "fact_coverage"
    required = [fact for fact in spec.facts if fact.required]
    if not required:
        return CheckResult(name=name, passed=None, detail="The spec marks no fact as required.")
    missing = [fact for fact in required if fact.id not in state.revealed]
    value = (len(required) - len(missing)) / len(required)
    detail = f"{len(required) - len(missing)} of {len(required)} required facts were conveyed."
    return CheckResult(
        name=name,
        passed=not missing,
        value=round(value, 4),
        detail=detail,
        evidence=[f"never said: {fact.text}" for fact in missing],
    )


def _answered_when_asked(spec: SimUserSpec, state: SimUserState, messages: list[Message]) -> CheckResult:
    """Fail if the agent asked for an on-request fact and the customer never gave it.

    A request only counts if the customer had a later turn in which to answer,
    so a request in the final agent message is ignored. Not applicable when the
    spec has no on-request facts, or when no request came in time to be answered.
    """
    name = "answered_when_asked"
    on_request = [fact for fact in spec.facts if fact.disclosure == "on_request"]
    if not on_request:
        return CheckResult(name=name, passed=None, detail="The spec has no on-request facts.")
    last_customer_turn = max(
        (message.turn_index for message in messages if message.role is Role.sim_user),
        default=-1,
    )
    # Only count a request if the customer had a turn left to answer it.
    asked = [
        fact
        for fact in on_request
        if fact.id in state.unlocked and last_customer_turn > state.unlocked[fact.id]
    ]
    if not asked:
        return CheckResult(
            name=name,
            passed=None,
            detail="The agent never asked for an on-request fact in time for the customer to answer.",
        )
    unanswered = [fact for fact in asked if fact.id not in state.revealed]
    if unanswered:
        return CheckResult(
            name=name,
            passed=False,
            detail=f"The agent asked for {len(unanswered)} fact(s) the customer never gave.",
            evidence=[
                f"asked in {_turn(state.unlocked[fact.id])}, never said: {fact.text}" for fact in unanswered
            ],
        )
    return CheckResult(name=name, passed=True, detail="The customer gave every fact the agent asked for.")


def _customer_texts(messages: list[Message]) -> list[tuple[int, str]]:
    """Return ``(turn_index, normalized_text)`` for each customer message.

    Shared by the goal checks, which look only at what the customer said.
    """
    return [
        (message.turn_index, normalize(message.content))
        for message in messages
        if message.role is Role.sim_user
    ]


def _goal_stated(spec: SimUserSpec, messages: list[Message]) -> CheckResult:
    """Pass if any customer message contains one of the goal markers.

    The detail names the first turn where the goal appeared. Not applicable when
    the spec gives no goal markers.
    """
    name = "goal_stated"
    if not spec.goal.markers:
        return CheckResult(name=name, passed=None, detail="The spec gives no goal markers.")
    for turn, text in _customer_texts(messages):
        if any(marker in text for marker in spec.goal.markers):
            return CheckResult(name=name, passed=True, detail=f"The goal was stated by {_turn(turn)}.")
    return CheckResult(
        name=name,
        passed=False,
        detail="The customer never stated the goal.",
        evidence=[f"expected one of: {', '.join(spec.goal.markers)}"],
    )


def _goal_not_abandoned(spec: SimUserSpec, messages: list[Message]) -> CheckResult:
    """Fail if any customer message contains an abandon marker (for example "never mind").

    Every match is listed as evidence. Not applicable when the spec gives no
    abandon markers.
    """
    name = "goal_not_abandoned"
    if not spec.goal.abandon_markers:
        return CheckResult(name=name, passed=None, detail="The spec gives no abandon markers.")
    evidence = [
        f"{_turn(turn)}: said '{marker}'"
        for turn, text in _customer_texts(messages)
        for marker in spec.goal.abandon_markers
        if marker in text
    ]
    if evidence:
        return CheckResult(name=name, passed=False, detail="The customer gave up on the goal.", evidence=evidence)
    return CheckResult(name=name, passed=True, detail="The customer did not give up on the goal.")
