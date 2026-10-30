"""The model-driven agent loop, with a scripted fake model. No network."""

import json

import pytest

from simbench.agent import AgentConfig, load_agent_config
from simbench.environment import Environment
from simbench.llm import AgentAction, LLMError, MockAgentClient
from simbench.llm_agent import LLMAgent
from simbench.models import Message, Role
from simbench.scenario import load_scenario

EMAIL = "jordan.lee@example.com"


def reply(text: str) -> AgentAction:
    return AgentAction(action="reply", message=text)


def tool(name: str, arguments: object) -> AgentAction:
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return AgentAction(action="tool_call", tool=name, arguments_json=text)


def customer(index: int, text: str) -> Message:
    return Message(turn_index=index, role=Role.sim_user, content=text)


def make(agent_id: str, actions: list[AgentAction], scenario_id: str = "late_refund", **kwargs):
    """Build an agent, its fake model, and a fresh environment for a scenario."""
    scenario = load_scenario(scenario_id)
    client = MockAgentClient(actions, **kwargs)
    agent = LLMAgent(load_agent_config(agent_id), client, scenario.environment, model="m")
    return agent, client, Environment(scenario.environment)


def user_text(client: MockAgentClient, call: int) -> str:
    return client.calls[call]["messages"][1]["content"]


def test_a_direct_reply_ends_the_turn_after_one_model_call() -> None:
    agent, client, env = make("agent_b", [reply("Hello! Which order is this about?")])

    result = agent.respond([customer(0, "Hi")], env, seed=1)

    assert result.message == "Hello! Which order is this about?" and result.step_limit is False
    assert len(client.calls) == 1 and env.calls == []


def test_a_tool_result_is_shown_to_the_model_before_its_next_step() -> None:
    agent, client, env = make(
        "agent_b", [tool("lookup_order", {"order_id": "1042"}), reply("It is six days late.")]
    )

    result = agent.respond([customer(0, "Where is order 1042?")], env, seed=1)

    assert result.message == "It is six days late."
    assert "What you have done this turn" not in user_text(client, 0)
    second = user_text(client, 1)
    assert 'You called lookup_order with {"order_id": "1042"}' in second
    assert '"days_late": 6' in second
    assert [(c.turn_index, c.step, c.tool, c.ok) for c in env.calls] == [(1, 0, "lookup_order", True)]


def test_invalid_json_arguments_are_fed_back_and_logged_as_a_failed_call() -> None:
    agent, client, env = make("agent_b", [tool("lookup_order", "{not json"), reply("Sorry, one moment.")])

    agent.respond([customer(0, "Hi")], env, seed=1)

    assert env.calls[0].ok is False and env.calls[0].error.startswith("invalid_argument")
    assert "arguments must be a JSON object" in user_text(client, 1)


def test_an_unknown_tool_is_refused_and_the_agent_can_recover() -> None:
    agent, client, env = make("agent_b", [tool("delete_order", {}), reply("I cannot do that.")])

    result = agent.respond([customer(0, "Delete it")], env, seed=1)

    assert result.message == "I cannot do that."
    assert env.calls[0].error.startswith("unknown_tool")


def test_a_write_tool_is_refused_for_a_read_only_version_and_changes_nothing() -> None:
    args = {"order_id": "1042", "customer_email": EMAIL, "reason": "late"}
    agent, client, env = make("agent_b", [tool("issue_refund", args), reply("I cannot refund.")])

    agent.respond([customer(0, "Refund me")], env, seed=1)

    assert env.calls[0].error.startswith("tool_not_permitted")
    assert env.snapshot().refunds == []
    assert "tool_not_permitted" in user_text(client, 1)


def test_the_turn_stops_with_step_limit_when_the_model_never_replies() -> None:
    looping = [tool("lookup_order", {"order_id": "1042"})] * 4
    agent, client, env = make("agent_b", looping)

    result = agent.respond([customer(0, "Hi")], env, seed=1)

    assert result.message is None and result.step_limit is True
    assert len(client.calls) == 4 == len(env.calls)
    assert [c.step for c in env.calls] == [0, 1, 2, 3]


def test_a_model_failure_keeps_its_usage_and_the_tool_calls_already_made() -> None:
    scenario = load_scenario("late_refund")

    class FailsSecond:
        def __init__(self) -> None:
            self.inner = MockAgentClient([tool("lookup_order", {"order_id": "1042"})])
            self.calls = 0

        def complete_agent(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return self.inner.complete_agent(**kwargs)
            raise LLMError("timed out", model_name="m", input_tokens=7, requests=2)

    agent = LLMAgent(load_agent_config("agent_c"), FailsSecond(), scenario.environment, model="m")
    env = Environment(scenario.environment)

    with pytest.raises(LLMError):
        agent.respond([customer(0, "Hi")], env, seed=1)

    usage = agent.usage(seed=1)
    assert usage.requests == 3 and usage.input_tokens == 17
    assert len(env.calls) == 1


def test_usage_sums_every_step_and_names_the_prompt_file() -> None:
    agent, _, env = make(
        "agent_b_prime",
        [tool("lookup_order", {"order_id": "1042"}), reply("Done.")],
        input_tokens=100,
        output_tokens=20,
    )

    agent.respond([customer(0, "Hi")], env, seed=5)
    usage = agent.usage(seed=5)

    assert (usage.requests, usage.input_tokens, usage.output_tokens) == (2, 200, 40)
    assert usage.prompt_version == "agent.v1.careful"
    assert usage.seed == 5


def test_the_agent_never_sees_the_simusers_hidden_instructions() -> None:
    scenario = load_scenario("late_refund")
    spec = scenario.sim_user
    secrets = [spec.style, spec.done_when, spec.goal.text, EMAIL] + [fact.text for fact in spec.facts]

    for agent_id in ("agent_a", "agent_b", "agent_c", "agent_d", "agent_b_prime"):
        agent = LLMAgent(load_agent_config(agent_id), MockAgentClient([reply("Hi")]), scenario.environment, model="m")
        for secret in secrets:
            assert secret not in agent.prompt, (agent_id, secret)

    agent, client, env = make("agent_c", [reply("Which order?")])
    agent.respond([customer(0, "Hello, I need a refund for order 1042.")], env, seed=1)
    sent = json.dumps(client.calls[0]["messages"])
    for secret in secrets:
        assert secret not in sent


def test_respond_does_not_modify_the_history_it_was_given() -> None:
    agent, _, env = make("agent_b", [reply("Hi")])
    history = [customer(0, "Hello")]

    agent.respond(history, env, seed=1)

    assert history == [customer(0, "Hello")]


def test_an_agent_with_tools_needs_a_scenario_with_an_environment() -> None:
    with pytest.raises(ValueError, match="no environment"):
        LLMAgent(load_agent_config("agent_c"), MockAgentClient([]), None, model="m")


def test_a_toolless_agent_is_allowed_without_an_environment() -> None:
    config = AgentConfig(id="agent_x", label="X", description="x", prompt="basic", tools=[])

    agent = LLMAgent(config, MockAgentClient([reply("Hi")]), None, model="m")

    assert "You have no tools." in agent.prompt
    assert agent.respond([customer(0, "Hi")], None, seed=1).message == "Hi"


def test_tool_calls_without_an_environment_are_refused_not_crashed() -> None:
    config = AgentConfig(id="agent_x", label="X", description="x", prompt="basic", tools=[])
    client = MockAgentClient([tool("lookup_order", {"order_id": "1"}), reply("Sorry.")])
    agent = LLMAgent(config, client, None, model="m")

    assert agent.respond([customer(0, "Hi")], None, seed=1).message == "Sorry."
    assert "tool_not_permitted" in user_text(client, 1)


def test_an_action_that_does_not_fit_the_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="no message"):
        AgentAction(action="reply", message="   ")
    with pytest.raises(ValueError, match="no tool name"):
        AgentAction(action="tool_call", tool="")
