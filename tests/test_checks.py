"""Each check against a clean transcript and a deliberately broken one."""

import copy

from simbench.checks import CHECKS_VERSION, evaluate
from simbench.spec import SimUserSpec

from conftest import SPEC_DATA


def by_name(report) -> dict:
    """Index a report's results by check name so a test can look one up directly."""
    return {check.name: check for check in report.checks}


CLEAN = (
    ("u", "Hi, order 1042 was due last Tuesday. I want a refund."),
    ("a", "Sorry about that. Can you confirm the email address on the account?"),
    ("u", "It is jordan.lee@example.com."),
    ("a", "Thanks. I can see tracking number TRK-88213 stuck at a depot."),
    ("u", "Please refund it. TRK-88213 has not moved in days."),
    ("a", "Done."),
)


def test_clean_transcript_passes_every_check(spec, convo) -> None:
    report = evaluate(spec, convo(*CLEAN))
    checks = by_name(report)

    assert report.checks_version == CHECKS_VERSION
    assert report.all_passed is True
    assert {name: check.passed for name, check in checks.items()} == {
        "unknown_fact_not_used": True,
        "no_early_reveal": True,
        "fact_coverage": True,
        "answered_when_asked": True,
        "goal_stated": True,
        "goal_not_abandoned": True,
    }
    assert checks["fact_coverage"].value == 1.0


# ---- Deliberate violations: each one must be flagged by exactly the check meant to catch it. ----


def test_volunteering_an_on_request_detail_is_flagged(spec, convo) -> None:
    report = evaluate(
        spec,
        convo(
            ("u", "Order 1042, due last Tuesday, refund please. My email is jordan.lee@example.com."),
            ("a", "Can you confirm the email address on the account?"),
            ("u", "Yes, as I said."),
        ),
    )
    checks = by_name(report)

    assert checks["no_early_reveal"].passed is False
    assert "turn 1" in checks["no_early_reveal"].evidence[0]
    assert report.all_passed is False
    assert checks["unknown_fact_not_used"].passed is True


def test_using_a_fact_the_customer_should_not_know_is_flagged(spec, convo) -> None:
    report = evaluate(
        spec,
        convo(
            ("u", "Order 1042, due last Tuesday, refund please. Tracking TRK-88213 shows nothing."),
            ("a", "Let me check."),
        ),
    )
    checks = by_name(report)

    assert checks["unknown_fact_not_used"].passed is False
    assert checks["unknown_fact_not_used"].evidence == [
        "turn 1: said the tracking number before the agent did"
    ]
    assert report.all_passed is False


def test_repeating_a_fact_after_the_agent_said_it_is_not_flagged(spec, convo) -> None:
    report = evaluate(spec, convo(*CLEAN))

    assert by_name(report)["unknown_fact_not_used"].passed is True


def test_leaving_out_required_facts_lowers_coverage(spec, convo) -> None:
    report = evaluate(spec, convo(("u", "Hi, I would like a refund."), ("a", "Sure.")))
    coverage = by_name(report)["fact_coverage"]

    assert coverage.passed is False
    assert coverage.value == 0.0
    assert coverage.evidence == [
        "never said: The order number is 1042.",
        "never said: The order was due last Tuesday.",
    ]
    assert report.all_passed is False


def test_partial_coverage_reports_a_fraction(spec, convo) -> None:
    report = evaluate(spec, convo(("u", "Order 1042 is late, refund please."), ("a", "Sure.")))

    assert by_name(report)["fact_coverage"].value == 0.5


def test_ignoring_a_request_is_flagged(spec, convo) -> None:
    report = evaluate(
        spec,
        convo(
            ("u", "Order 1042, due last Tuesday, refund please."),
            ("a", "Can you confirm the email on the account?"),
            ("u", "I would rather not say."),
        ),
    )
    checks = by_name(report)

    assert checks["answered_when_asked"].passed is False
    assert checks["answered_when_asked"].evidence == [
        "asked in turn 2, never said: The email on the account is jordan.lee@example.com."
    ]


def test_a_request_with_no_customer_turn_left_is_not_counted(spec, convo) -> None:
    report = evaluate(
        spec,
        convo(
            ("u", "Order 1042, due last Tuesday, refund please."),
            ("a", "Can you confirm the email on the account?"),
        ),
    )

    assert by_name(report)["answered_when_asked"].passed is None
    assert report.all_passed is True


def test_never_stating_the_goal_is_flagged(spec, convo) -> None:
    report = evaluate(spec, convo(("u", "Order 1042 is late, due last Tuesday."), ("a", "Sure.")))
    goal = by_name(report)["goal_stated"]

    assert goal.passed is False
    assert goal.evidence == ["expected one of: refund"]


def test_giving_up_is_flagged_with_the_turn(spec, convo) -> None:
    report = evaluate(
        spec,
        convo(
            ("u", "Order 1042, due last Tuesday, refund please."),
            ("a", "That will take a while."),
            ("u", "Oh, never mind."),
        ),
    )
    abandoned = by_name(report)["goal_not_abandoned"]

    assert abandoned.passed is False
    assert abandoned.evidence == ["turn 3: said 'never mind'"]


def test_checks_with_nothing_to_test_are_not_applicable_and_do_not_block_all_passed(convo) -> None:
    data = copy.deepcopy(SPEC_DATA)
    data["facts"] = [data["facts"][0]]
    data["unknown_facts"] = []
    data["goal"] = {"text": "Get a refund."}
    bare = SimUserSpec.model_validate(data)

    report = evaluate(bare, convo(("u", "Order 1042."), ("a", "Ok.")))
    checks = by_name(report)

    assert checks["unknown_fact_not_used"].passed is None
    assert checks["no_early_reveal"].passed is None
    assert checks["answered_when_asked"].passed is None
    assert checks["goal_stated"].passed is None
    assert checks["goal_not_abandoned"].passed is None
    assert checks["fact_coverage"].passed is True
    assert report.all_passed is True


def test_no_customer_messages_fails_coverage_and_goal(spec, convo) -> None:
    report = evaluate(spec, convo(("a", "Hello, how can I help?")))
    checks = by_name(report)

    assert checks["fact_coverage"].passed is False
    assert checks["goal_stated"].passed is False
    assert report.all_passed is False
