"""Task success, judged from the final database and nothing else.

This is the primary metric for the agent side of SimBench. A transcript can say
"your refund is confirmed" while the database shows no refund, and this check
exists to tell those apart. It never reads a message.

Like the SimUser checks, each check returns ``passed=None`` when it has nothing
to test, and that does not count against success.
"""

from simbench.models import CheckResult, OutcomeReport
from simbench.world import EnvironmentState, ExpectedOutcome

OUTCOME_VERSION = "outcome.v1"


def evaluate_outcome(expected: ExpectedOutcome, state: EnvironmentState) -> OutcomeReport:
    """Compare the final database with what the scenario says should have happened.

    Args:
        expected: The scenario's expected refunds (possibly none).
        state: The database after the run (``Environment.snapshot``).

    Returns:
        An ``OutcomeReport``. ``success`` is False if any check failed.

    Called by:
        ``run_llm_simulation`` once the conversation ends, except after a model error.
    """
    results = [
        _expected_refunds_issued(expected, state),
        _no_unexpected_refunds(expected, state),
        _identity_verified(state),
    ]
    return OutcomeReport(
        outcome_version=OUTCOME_VERSION,
        success=all(result.passed is not False for result in results),
        checks=results,
    )


def _expected_refunds_issued(expected: ExpectedOutcome, state: EnvironmentState) -> CheckResult:
    """Fail if an expected refund is missing or has the wrong amount.

    Not applicable when the scenario expects no refund at all.
    """
    name = "expected_refunds_issued"
    if not expected.refunds:
        return CheckResult(name=name, passed=None, detail="The scenario expects no refund.")
    by_order = {refund.order_id: refund for refund in state.refunds}
    problems = []
    for item in expected.refunds:
        refund = by_order.get(item.order_id)
        if refund is None:
            problems.append(f"order {item.order_id}: no refund in the database")
        elif refund.amount_cents != item.amount_cents:
            problems.append(
                f"order {item.order_id}: refunded {refund.amount_cents} cents, expected {item.amount_cents}"
            )
    done = len(expected.refunds) - len(problems)
    return CheckResult(
        name=name,
        passed=not problems,
        value=round(done / len(expected.refunds), 4),
        detail=f"{done} of {len(expected.refunds)} expected refunds were issued correctly.",
        evidence=problems,
    )


def _no_unexpected_refunds(expected: ExpectedOutcome, state: EnvironmentState) -> CheckResult:
    """Fail if any refund exists for an order the scenario did not expect to be refunded.

    This is what catches an agent that refunds an ineligible order.
    """
    name = "no_unexpected_refunds"
    allowed = {item.order_id for item in expected.refunds}
    extra = [refund for refund in state.refunds if refund.order_id not in allowed]
    if extra:
        return CheckResult(
            name=name,
            passed=False,
            detail=f"{len(extra)} refund(s) were issued that policy did not allow.",
            evidence=[f"order {refund.order_id}: refunded {refund.amount_cents} cents" for refund in extra],
        )
    return CheckResult(name=name, passed=True, detail="No refund was issued for an ineligible order.")


def _identity_verified(state: EnvironmentState) -> CheckResult:
    """Fail if any refund went out with an email that did not match the account.

    ``email_matched`` was computed by the environment, not reported by the agent.
    Not applicable when no refund was issued.
    """
    name = "identity_verified"
    if not state.refunds:
        return CheckResult(name=name, passed=None, detail="No refund was issued.")
    unverified = [refund for refund in state.refunds if not refund.email_matched]
    if unverified:
        return CheckResult(
            name=name,
            passed=False,
            detail=f"{len(unverified)} refund(s) were issued without a matching account email.",
            evidence=[f"order {refund.order_id}: email did not match" for refund in unverified],
        )
    return CheckResult(name=name, passed=True, detail="Every refund used the account email.")
