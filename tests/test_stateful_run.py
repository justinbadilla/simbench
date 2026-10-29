"""Full runs of the bundled scenario with a scripted fake model, one per behaviour we want to detect."""

from simbench.llm import MockLLMClient, SimUserTurn
from simbench.models import Role, TerminationReason
from simbench.run import run_llm_simulation
from simbench.scenario import load_scenario

EMAIL = "jordan.lee@example.com"


def turns(*pairs: tuple[str, bool]) -> list[SimUserTurn]:
    """Build model replies from ``(message, end_conversation)`` pairs."""
    return [SimUserTurn(message=message, end_conversation=done) for message, done in pairs]


WELL_BEHAVED = turns(
    ("Hi, my order 1042 was due last Tuesday. I would like a refund.", False),
    (f"The email on the account is {EMAIL}.", False),
    ("A refund please. TRK-88213 has not moved.", False),
    ("Thanks, that is all I needed.", True),
)


def check_results(transcript) -> dict:
    """Map each check name to its pass/fail/None result. Asserts the run produced a report."""
    assert transcript.checks is not None
    return {check.name: check.passed for check in transcript.checks.checks}


def test_stateful_prompt_changes_as_the_agent_asks_and_the_customer_answers() -> None:
    transcript = run_llm_simulation(
        load_scenario("late_refund"),
        MockLLMClient(WELL_BEHAVED),
        variant="stateful",
        seed=1,
    )
    first, second, third, fourth = transcript.turn_prompts

    assert transcript.termination_reason is TerminationReason.sim_user_ended
    assert transcript.system_prompt == first
    assert EMAIL not in first
    assert f"{EMAIL}. (the agent asked for this)" in second
    assert "Already told the agent" in third and EMAIL in third
    assert "(the agent asked for this)" not in third
    assert "Nothing new right now." in fourth
    assert [(e.kind.value, e.fact_id) for e in transcript.state_log] == [
        ("revealed", "order_number"),
        ("revealed", "due_date"),
        ("revealed", "wants_refund"),
        ("unlocked", "account_email"),
        ("revealed", "account_email"),
        ("learned", "tracking_number"),
    ]
    assert transcript.sim_user_spec is not None


def test_well_behaved_run_passes_every_check() -> None:
    transcript = run_llm_simulation(
        load_scenario("late_refund"), MockLLMClient(WELL_BEHAVED), variant="stateful", seed=1
    )

    assert transcript.checks.all_passed is True
    assert set(check_results(transcript).values()) == {True}


def test_persona_variant_sends_the_same_prompt_every_turn() -> None:
    transcript = run_llm_simulation(
        load_scenario("late_refund"), MockLLMClient(WELL_BEHAVED), variant="persona", seed=1
    )

    assert transcript.variant == "persona"
    assert len(set(transcript.turn_prompts)) == 1
    assert transcript.usage.prompt_version == "sim_user.v2.persona+late_refund.v2"
    assert f"{EMAIL}. Do not mention this until the agent asks for it." in transcript.turn_prompts[0]


def test_a_customer_who_leaks_the_email_on_turn_one_is_caught() -> None:
    leaky = turns(
        (f"Order 1042, due last Tuesday, I want a refund. My email is {EMAIL}.", False),
        ("Thanks, that is all I needed.", True),
    )

    transcript = run_llm_simulation(
        load_scenario("late_refund"), MockLLMClient(leaky), variant="persona", seed=1
    )

    assert check_results(transcript)["no_early_reveal"] is False
    assert transcript.checks.all_passed is False
    assert ("early_reveal", "account_email") in [
        (e.kind.value, e.fact_id) for e in transcript.state_log
    ]


def test_a_customer_who_invents_the_tracking_number_is_caught() -> None:
    inventing = turns(
        ("Order 1042, due last Tuesday, refund please. Tracking TRK-88213 is stuck.", False),
        (f"The email is {EMAIL}.", True),
    )

    transcript = run_llm_simulation(
        load_scenario("late_refund"), MockLLMClient(inventing), variant="stateful", seed=1
    )

    assert check_results(transcript)["unknown_fact_not_used"] is False
    assert transcript.checks.all_passed is False


def test_a_customer_who_never_answers_the_email_request_is_caught() -> None:
    stonewalling = turns(
        ("Order 1042, due last Tuesday, refund please.", False),
        ("I would rather not say.", False),
        ("A refund please.", False),
        ("Thanks, bye.", True),
    )

    transcript = run_llm_simulation(
        load_scenario("late_refund"), MockLLMClient(stonewalling), variant="stateful", seed=1
    )

    assert check_results(transcript)["answered_when_asked"] is False


def test_checks_are_skipped_when_the_model_failed_mid_run() -> None:
    class AlwaysFails:
        """Client whose every call fails, to test the model_error path."""

        def complete(self, **kwargs: object):
            from simbench.llm import LLMError

            raise LLMError("The model request timed out.", model_name="m")

    transcript = run_llm_simulation(load_scenario("late_refund"), AlwaysFails(), seed=1)

    assert transcript.termination_reason is TerminationReason.model_error
    assert transcript.checks is None
    assert transcript.messages == []
    assert transcript.turn_prompts == [transcript.system_prompt]


def test_run_with_no_model_calls_still_records_a_prompt() -> None:
    transcript = run_llm_simulation(
        load_scenario("late_refund"), MockLLMClient([]), max_turns=0, seed=1
    )

    assert transcript.turn_prompts == []
    assert transcript.system_prompt
    assert [message.role for message in transcript.messages] == []
