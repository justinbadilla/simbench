"""LLM SimUser against a scripted agent. The client is a fake, so these tests stay offline."""

import json

from simbench.llm import LLMError, MockLLMClient, RecordingLLMClient, ReplayLLMClient, SimUserTurn
from simbench.models import Role, TerminationReason
from simbench.run import run_llm_simulation
from simbench.scenario import Scenario, load_scenario


def brief_scenario(**overrides: object) -> Scenario:
    data: dict[str, object] = {
        "id": "fixture",
        "title": "Fixture",
        "description": "Test scenario",
        "notes": "No order database is updated.",
        "opening_role": Role.sim_user,
        "max_turns": 8,
        "temperature": 0.7,
        "sim_user_id": "customer",
        "agent_id": "scripted_support",
        "script_version": "fixture.v1",
        "sim_user": {
            "goal": "Get a refund for order 1042.",
            "known_facts": ["The order number is 1042."],
            "style": "Brief.",
            "done_when": "End after the agent confirms the refund and you have thanked them.",
        },
        "agent_lines": [
            "Do you want a refund or a replacement?",
            "Confirmed. A refund for order 1042 is noted.",
        ],
    }
    data.update(overrides)
    return Scenario.model_validate(data)


def turns(*pairs: tuple[str, bool]) -> list[SimUserTurn]:
    return [SimUserTurn(message=message, end_conversation=done) for message, done in pairs]


def test_customer_opens_and_the_next_prompt_includes_the_agent_line() -> None:
    client = MockLLMClient(
        turns(
            ("I want a refund for order 1042.", False),
            ("Thanks, that is all.", True),
        )
    )

    transcript = run_llm_simulation(brief_scenario(), client, model="gpt-4o-mini", seed=7)

    assert [message.role for message in transcript.messages] == [
        Role.sim_user,
        Role.agent,
        Role.sim_user,
    ]
    assert transcript.messages[1].content.startswith("Do you want a refund")
    assert transcript.termination_reason is TerminationReason.sim_user_ended
    assert transcript.usage.seed == 7
    assert transcript.usage.model_name == "mock-sim-user"
    assert transcript.usage.provider == "mock"
    assert transcript.usage.input_tokens == 20
    assert transcript.usage.output_tokens == 10
    assert transcript.usage.latency_ms == 50.0
    assert transcript.usage.requests == 2
    assert transcript.usage.free_tier is False
    assert transcript.usage.prompt_version == "sim_user.v1+fixture.v1"
    assert "Get a refund for order 1042." in transcript.system_prompt
    assert len(client.calls) == 2
    second_prompt = client.calls[1]["messages"][1]["content"]
    assert "Agent: Do you want a refund or a replacement?" in second_prompt
    assert client.calls[0]["seed"] == 7


def test_max_turns_stops_before_another_model_call() -> None:
    client = MockLLMClient(turns(("Hello", False), ("Still here", False)))

    transcript = run_llm_simulation(brief_scenario(), client, max_turns=1, seed=1)

    assert [message.content for message in transcript.messages] == ["Hello"]
    assert transcript.termination_reason is TerminationReason.max_turns
    assert len(client.calls) == 1


def test_agent_script_ends_the_run_when_the_customer_does_not() -> None:
    client = MockLLMClient(turns(("One", False), ("Two", False), ("Three", False)))

    transcript = run_llm_simulation(brief_scenario(), client, seed=1)

    assert [message.role for message in transcript.messages] == [
        Role.sim_user,
        Role.agent,
        Role.sim_user,
        Role.agent,
        Role.sim_user,
    ]
    assert transcript.termination_reason is TerminationReason.script_complete


def test_model_error_keeps_the_messages_already_said() -> None:
    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, **kwargs: object):
            del kwargs
            self.calls += 1
            if self.calls == 1:
                return MockLLMClient(turns(("Hello", False))).complete(
                    messages=[],
                    model="gpt-4o-mini",
                    temperature=0.7,
                    seed=1,
                )
            raise LLMError(
                "The model request timed out.",
                model_name="gpt-4o-mini",
                input_tokens=4,
                output_tokens=0,
                latency_ms=30,
                estimated_cost_usd=0.0000006,
            )

    transcript = run_llm_simulation(brief_scenario(), Flaky(), seed=3)

    assert [message.role for message in transcript.messages] == [Role.sim_user, Role.agent]
    assert transcript.termination_reason is TerminationReason.model_error
    assert transcript.error == "The model request timed out."
    assert transcript.usage.input_tokens == 14
    assert transcript.usage.seed == 3
    assert transcript.system_prompt


def test_requests_and_free_tier_are_summed_across_turns() -> None:
    client = MockLLMClient(
        turns(("One", False), ("Two", True)),
        requests=2,
        free_tier=True,
        estimated_cost_usd=0.001,
    )

    transcript = run_llm_simulation(brief_scenario(), client, seed=1)

    assert transcript.usage.requests == 4
    assert transcript.usage.free_tier is True
    assert transcript.usage.estimated_cost_usd == 0.002


def test_unknown_model_cost_stays_null() -> None:
    client = MockLLMClient(turns(("Hello", True)), estimated_cost_usd=None, model_name="made-up")

    transcript = run_llm_simulation(brief_scenario(), client, model="made-up", seed=1)

    assert transcript.usage.estimated_cost_usd is None


def test_record_then_replay_returns_the_same_turn(tmp_path) -> None:
    path = tmp_path / "tape.json"
    turn = SimUserTurn(message="I want a refund.", end_conversation=False)
    RecordingLLMClient(MockLLMClient([turn], input_tokens=3), path).complete(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o-mini",
        temperature=0.2,
        seed=9,
    )

    replayed = ReplayLLMClient(path).complete(
        messages=[],
        model="ignored",
        temperature=0,
        seed=None,
    )

    assert replayed.turn.message == "I want a refund."
    assert replayed.input_tokens == 3
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["turn"]["message"] == "I want a refund."


def test_scenario_without_a_brief_cannot_start() -> None:
    scenario = brief_scenario(sim_user=None)
    try:
        run_llm_simulation(scenario, MockLLMClient([]), seed=1)
    except ValueError as error:
        assert "no SimUser brief" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_bundled_scenario_replay_ends_when_the_fixture_says_so() -> None:
    transcript = run_llm_simulation(
        load_scenario("late_refund"),
        ReplayLLMClient(fixture_path()),
        model="gpt-4o-mini",
        seed=1,
    )

    assert transcript.termination_reason is TerminationReason.sim_user_ended
    assert transcript.usage.provider == "replay"
    assert transcript.usage.free_tier is True
    assert transcript.usage.requests == 3
    assert [message.role for message in transcript.messages] == [
        Role.sim_user,
        Role.agent,
        Role.sim_user,
        Role.agent,
        Role.sim_user,
    ]
    assert "no order database" in transcript.notes


def fixture_path():
    from simbench.paths import repo_root

    return repo_root() / "data" / "cassettes" / "late_refund.fixture.json"
