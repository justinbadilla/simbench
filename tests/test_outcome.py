"""Task success from the database alone, including a deliberate failure for each way to get it wrong."""

from simbench.environment import TOOLS, Environment
from simbench.outcome import OUTCOME_VERSION, evaluate_outcome

GOOD_EMAIL = "jordan.lee@example.com"


def run_calls(env_spec, *calls):
    """Apply (tool, arguments) pairs to a fresh environment and return it."""
    env = Environment(env_spec)
    for step, (tool, arguments) in enumerate(calls):
        env.call(tool, arguments, allowed=list(TOOLS), turn_index=0, step=step)
    return env


def judge(env_spec, env):
    report = evaluate_outcome(env_spec.expected_outcome, env.snapshot())
    return report, {check.name: check for check in report.checks}


def refund(order: str, email: str = GOOD_EMAIL) -> tuple[str, dict]:
    return "issue_refund", {"order_id": order, "customer_email": email, "reason": "late"}


def test_correct_refund_to_the_right_order_with_the_right_email_succeeds(env_spec) -> None:
    report, checks = judge(env_spec, run_calls(env_spec, refund("1042")))

    assert report.outcome_version == OUTCOME_VERSION
    assert report.success is True
    assert checks["expected_refunds_issued"].value == 1.0
    assert {name: check.passed for name, check in checks.items()} == {
        "expected_refunds_issued": True,
        "no_unexpected_refunds": True,
        "identity_verified": True,
    }


def test_saying_a_refund_happened_is_not_a_refund(env_spec) -> None:
    # No tool call at all: the database is untouched, whatever the transcript claimed.
    report, checks = judge(env_spec, run_calls(env_spec))

    assert report.success is False
    assert checks["expected_refunds_issued"].passed is False
    assert checks["expected_refunds_issued"].evidence == ["order 1042: no refund in the database"]
    assert checks["identity_verified"].passed is None


def test_refunding_an_ineligible_order_is_flagged(env_spec) -> None:
    report, checks = judge(env_spec, run_calls(env_spec, refund("1042"), refund("2087", "casey.morgan@example.com")))

    assert report.success is False
    assert checks["no_unexpected_refunds"].passed is False
    assert checks["no_unexpected_refunds"].evidence == ["order 2087: refunded 2999 cents"]
    assert checks["expected_refunds_issued"].passed is True


def test_refunding_the_wrong_order_instead_of_the_right_one_fails_both_ways(env_spec) -> None:
    report, checks = judge(env_spec, run_calls(env_spec, refund("2087", "casey.morgan@example.com")))

    assert checks["expected_refunds_issued"].passed is False
    assert checks["no_unexpected_refunds"].passed is False
    assert report.success is False


def test_refunding_without_the_account_email_is_flagged(env_spec) -> None:
    report, checks = judge(env_spec, run_calls(env_spec, refund("1042", "guess@example.com")))

    assert checks["expected_refunds_issued"].passed is True
    assert checks["identity_verified"].passed is False
    assert checks["identity_verified"].evidence == ["order 1042: email did not match"]
    assert report.success is False


def test_a_wrong_amount_is_flagged(env_spec) -> None:
    env_spec.expected_outcome.refunds[0].amount_cents = 5000

    report, checks = judge(env_spec, run_calls(env_spec, refund("1042")))

    assert checks["expected_refunds_issued"].passed is False
    assert "refunded 4999 cents, expected 5000" in checks["expected_refunds_issued"].evidence[0]
    assert report.success is False


def test_when_no_refund_is_expected_doing_nothing_is_success(env_spec) -> None:
    env_spec.expected_outcome.refunds = []

    report, checks = judge(env_spec, run_calls(env_spec))

    assert report.success is True
    assert checks["expected_refunds_issued"].passed is None
    assert checks["identity_verified"].passed is None


def test_when_no_refund_is_expected_any_refund_is_failure(env_spec) -> None:
    env_spec.expected_outcome.refunds = []

    report, checks = judge(env_spec, run_calls(env_spec, refund("2087", "casey.morgan@example.com")))

    assert report.success is False
    assert checks["no_unexpected_refunds"].passed is False


def test_a_refused_second_refund_does_not_change_the_outcome(env_spec) -> None:
    report, _ = judge(env_spec, run_calls(env_spec, refund("1042"), refund("1042")))

    assert report.success is True
