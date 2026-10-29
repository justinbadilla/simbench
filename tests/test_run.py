"""Turn order and termination for the scripted runner."""

from simbench.models import Role, TerminationReason
from simbench.run import run_simulation
from simbench.scenario import Scenario, load_scenario


def scenario(**overrides: object) -> Scenario:
    """A minimal scripted scenario (two lines each). Keyword arguments override fields."""
    data: dict[str, object] = {
        "id": "fixture",
        "title": "Fixture",
        "description": "Test scenario",
        "notes": "Not an evaluation result.",
        "opening_role": Role.sim_user,
        "max_turns": 10,
        "sim_user_id": "user",
        "agent_id": "agent",
        "script_version": "fixture.v0",
        "sim_user_lines": ["u1", "u2"],
        "agent_lines": ["a1", "a2"],
    }
    data.update(overrides)
    return Scenario.model_validate(data)


def assert_turn_order(transcript_messages: list, opening: Role) -> None:
    """Assert indexes count up from zero and speakers alternate, starting with ``opening``."""
    expected = opening
    for index, message in enumerate(transcript_messages):
        assert message.turn_index == index
        assert message.role is expected
        expected = Role.agent if expected is Role.sim_user else Role.sim_user


def test_user_opens_and_roles_alternate_until_the_script_ends() -> None:
    transcript = run_simulation(scenario())

    assert [message.content for message in transcript.messages] == ["u1", "a1", "u2", "a2"]
    assert_turn_order(transcript.messages, Role.sim_user)
    assert transcript.termination_reason is TerminationReason.script_complete
    assert transcript.usage.model_name is None
    assert transcript.usage.input_tokens == 0
    assert transcript.usage.estimated_cost_usd == 0.0
    assert transcript.usage.prompt_version == "fixture.v0"


def test_agent_can_open_when_the_scenario_says_so() -> None:
    transcript = run_simulation(scenario(opening_role=Role.agent))

    assert [message.role for message in transcript.messages] == [
        Role.agent,
        Role.sim_user,
        Role.agent,
        Role.sim_user,
    ]
    assert_turn_order(transcript.messages, Role.agent)
    assert transcript.termination_reason is TerminationReason.script_complete


def test_max_turns_stops_before_a_remaining_line() -> None:
    transcript = run_simulation(scenario(), max_turns=3)

    assert [message.content for message in transcript.messages] == ["u1", "a1", "u2"]
    assert_turn_order(transcript.messages, Role.sim_user)
    assert transcript.termination_reason is TerminationReason.max_turns
    assert transcript.max_turns == 3


def test_finished_script_is_script_complete_even_when_it_fills_the_cap() -> None:
    transcript = run_simulation(scenario(), max_turns=4)

    assert len(transcript.messages) == 4
    assert transcript.termination_reason is TerminationReason.script_complete


def test_zero_cap_emits_nothing_when_a_line_is_waiting() -> None:
    transcript = run_simulation(scenario(), max_turns=0)

    assert transcript.messages == []
    assert transcript.termination_reason is TerminationReason.max_turns


def test_shorter_agent_script_ends_the_run_on_the_agents_turn() -> None:
    transcript = run_simulation(scenario(agent_lines=["a1"]))

    assert [message.content for message in transcript.messages] == ["u1", "a1", "u2"]
    assert transcript.termination_reason is TerminationReason.script_complete


def test_empty_opening_script_ends_immediately() -> None:
    transcript = run_simulation(scenario(sim_user_lines=[]))

    assert transcript.messages == []
    assert transcript.termination_reason is TerminationReason.script_complete


def test_bundled_late_refund_scenario_completes_in_order() -> None:
    transcript = run_simulation(load_scenario("late_refund"))

    assert transcript.scenario_id == "late_refund"
    assert len(transcript.messages) == 8
    assert_turn_order(transcript.messages, Role.sim_user)
    assert transcript.messages[0].content.startswith("Hi, my order 1042")
    assert transcript.termination_reason is TerminationReason.script_complete
    assert "no order database" in transcript.notes
