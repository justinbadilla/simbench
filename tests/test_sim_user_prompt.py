"""What each variant puts in front of the model."""

import pytest

from simbench.scenario import Scenario, load_scenario
from simbench.sim_user import build_instructions, prompt_version
from simbench.state import SimUserState, replay

EMAIL = "jordan.lee@example.com"


@pytest.fixture
def scenario() -> Scenario:
    """The bundled ``late_refund`` scenario."""
    return load_scenario("late_refund")


def shareable_section(prompt: str) -> str:
    """Extract the "Facts you may share now" block from a stateful prompt."""
    return prompt.split("Facts you may share now:")[1].split("\n\n")[0]


def test_stateful_prompt_leaves_out_a_fact_the_customer_has_not_been_asked_for(scenario) -> None:
    prompt = build_instructions(scenario, "stateful", SimUserState(scenario.sim_user))

    assert EMAIL not in prompt
    assert "email" not in prompt.lower()
    assert "The order number is 1042." in shareable_section(prompt)


def test_neither_variant_ever_contains_a_fact_the_customer_should_not_know(scenario) -> None:
    state = SimUserState(scenario.sim_user)
    for prompt in (
        build_instructions(scenario, "stateful", state),
        build_instructions(scenario, "persona"),
    ):
        assert "trk-88213" not in prompt.lower()
        assert "You do not know the package tracking number unless the agent tells you." in prompt


def test_stateful_prompt_adds_the_fact_once_the_agent_asks(scenario, convo) -> None:
    messages = convo(
        ("u", "Order 1042, due last Tuesday, refund please."),
        ("a", "Can you confirm the email address on the account?"),
    )
    prompt = build_instructions(scenario, "stateful", replay(scenario.sim_user, messages))

    assert f"- The email on the account is {EMAIL}. (the agent asked for this)" in shareable_section(prompt)


def test_stateful_prompt_moves_said_facts_to_already_told(scenario, convo) -> None:
    messages = convo(
        ("u", "Order 1042, due last Tuesday, refund please."),
        ("a", "Can you confirm the email address on the account?"),
        ("u", f"It is {EMAIL}."),
        ("a", "Thanks."),
    )
    prompt = build_instructions(scenario, "stateful", replay(scenario.sim_user, messages))

    assert shareable_section(prompt).strip() == "- Nothing new right now."
    already_told = prompt.split("Already told the agent (do not repeat unless asked):")[1]
    assert EMAIL in already_told
    assert "The order number is 1042." in already_told


def test_persona_prompt_lists_every_fact_with_a_hold_back_instruction(scenario) -> None:
    prompt = build_instructions(scenario, "persona")

    assert f"The email on the account is {EMAIL}. Do not mention this until the agent asks for it." in prompt
    assert "The order number is 1042." in prompt
    assert "Facts you may share now" not in prompt


def test_persona_prompt_does_not_change_as_the_conversation_goes(scenario, convo) -> None:
    messages = convo(("u", "Order 1042."), ("a", "What is the email on the account?"))
    state = replay(scenario.sim_user, messages)

    assert build_instructions(scenario, "persona", state) == build_instructions(scenario, "persona")


def test_variants_share_the_same_header_and_tail(scenario) -> None:
    stateful = build_instructions(scenario, "stateful", SimUserState(scenario.sim_user))
    persona = build_instructions(scenario, "persona")

    header = stateful.split("Facts you may share now:")[0]
    assert persona.startswith(header)
    tail_start = "If the agent asks for something you have not been given"
    assert stateful[stateful.index(tail_start) :] == persona[persona.index(tail_start) :]


def test_stateful_variant_requires_a_state(scenario) -> None:
    with pytest.raises(ValueError, match="needs a SimUserState"):
        build_instructions(scenario, "stateful")


def test_prompt_version_names_the_variant_and_the_scenario_script(scenario) -> None:
    assert prompt_version(scenario, "stateful") == "sim_user.v2.stateful+late_refund.v2"
    assert prompt_version(scenario, "persona") == "sim_user.v2.persona+late_refund.v2"
