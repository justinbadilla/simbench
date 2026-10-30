"""Whole conversations: a fake SimUser model against agents, judged by the database."""

import json

from simbench.agent import AgentConfig, load_agent_config
from simbench.llm import AgentAction, LLMError, MockAgentClient, MockLLMClient, SimUserTurn
from simbench.llm_agent import LLMAgent
from simbench.models import Role, TerminationReason
from simbench.run import run_llm_simulation
from simbench.scenario import load_scenario

EMAIL = "jordan.lee@example.com"
CASEY = "casey.morgan@example.com"


def reply(text: str) -> AgentAction:
    return AgentAction(action="reply", message=text)


def tool(name: str, arguments: dict) -> AgentAction:
    return AgentAction(action="tool_call", tool=name, arguments_json=json.dumps(arguments))


def customer(*pairs: tuple[str, bool]) -> MockLLMClient:
    return MockLLMClient([SimUserTurn(message=m, end_conversation=e) for m, e in pairs])


def agent_for(scenario, agent_id: str, actions: list[AgentAction], **kwargs):
    client = MockAgentClient(actions, **kwargs)
    return LLMAgent(load_agent_config(agent_id), client, scenario.environment, model="m"), client


LATE_CUSTOMER = (
    ("Hi, order 1042 was due last Tuesday and has not arrived. I would like a refund.", False),
    (f"The email on the account is {EMAIL}.", False),
    ("Thank you, that is all.", True),
)

CORRECT_C = [
    tool("lookup_order", {"order_id": "1042"}),
    reply("Sorry about that. Order 1042 is stuck at a depot (TRK-88213). What email is on the account?"),
    tool("issue_refund", {"order_id": "1042", "customer_email": EMAIL, "reason": "six days late"}),
    reply("Done. $49.99 goes back to your Visa ending 4242."),
    reply("You are welcome."),
]


def test_a_correct_agent_completes_the_task_and_every_part_is_recorded() -> None:
    scenario = load_scenario("late_refund")
    agent, _ = agent_for(scenario, "agent_c", CORRECT_C, input_tokens=100, output_tokens=10)

    transcript = run_llm_simulation(scenario, customer(*LATE_CUSTOMER), agent=agent, seed=3)

    assert transcript.termination_reason is TerminationReason.sim_user_ended
    assert transcript.agent_id == "agent_c"
    assert transcript.outcome.success is True
    assert [(c.turn_index, c.step, c.tool, c.ok) for c in transcript.tool_calls] == [
        (1, 0, "lookup_order", True),
        (3, 0, "issue_refund", True),
    ]
    assert transcript.final_state.refunds[0].email_matched is True
    assert transcript.final_state.refunds[0].turn_index == 3
    assert transcript.environment_spec is not None
    assert "agent_c" not in transcript.agent_prompt  # the prompt is the text the model saw
    assert "Tools you can use" in transcript.agent_prompt
    assert transcript.checks.all_passed is True


def test_simuser_and_agent_usage_are_kept_apart() -> None:
    scenario = load_scenario("late_refund")
    agent, _ = agent_for(scenario, "agent_c", CORRECT_C, input_tokens=100)

    transcript = run_llm_simulation(scenario, customer(*LATE_CUSTOMER), agent=agent, seed=3)

    assert transcript.usage.requests == 3 and transcript.usage.input_tokens == 30
    # Two tool calls and two replies: the customer ends the chat before the agent's last queued reply.
    assert transcript.agent_usage.requests == 4 and transcript.agent_usage.input_tokens == 400
    assert transcript.agent_usage.seed == transcript.usage.seed == 3
    assert transcript.agent_usage.prompt_version == "agent.v1.careful"


def test_state_tracking_sees_what_a_real_agent_says() -> None:
    scenario = load_scenario("late_refund")
    agent, _ = agent_for(scenario, "agent_c", CORRECT_C)

    transcript = run_llm_simulation(scenario, customer(*LATE_CUSTOMER), agent=agent, seed=1)

    kinds = [(e.kind.value, e.fact_id) for e in transcript.state_log]
    assert ("unlocked", "account_email") in kinds
    assert ("learned", "tracking_number") in kinds


def test_claiming_a_refund_without_issuing_it_is_not_success() -> None:
    # The common failure for a version with no refund tool: it says it is done.
    scenario = load_scenario("late_refund")
    agent, _ = agent_for(
        scenario,
        "agent_b",
        [reply("Sure, what email is on the account?"), reply("Your refund has been issued."), reply("Welcome.")],
    )

    transcript = run_llm_simulation(scenario, customer(*LATE_CUSTOMER), agent=agent, seed=1)

    assert "refund has been issued" in transcript.messages[3].content
    assert transcript.tool_calls == []
    assert transcript.outcome.success is False
    assert transcript.outcome.checks[0].passed is False


def test_a_read_only_agent_that_tries_to_refund_is_refused_and_still_fails() -> None:
    scenario = load_scenario("late_refund")
    args = {"order_id": "1042", "customer_email": EMAIL, "reason": "late"}
    agent, _ = agent_for(
        scenario,
        "agent_b",
        [tool("issue_refund", args), reply("I cannot issue refunds myself."), reply("Welcome.")],
    )

    transcript = run_llm_simulation(
        scenario, customer(("I want a refund for order 1042.", False), ("Thanks.", True)), agent=agent, seed=1
    )

    assert transcript.tool_calls[0].error.startswith("tool_not_permitted")
    assert transcript.final_state.refunds == []
    assert transcript.outcome.success is False


def test_the_flawed_agent_refunds_an_ineligible_order_and_the_database_check_catches_it() -> None:
    scenario = load_scenario("on_time_refund")
    flawed, _ = agent_for(
        scenario,
        "agent_d",
        [
            tool("issue_refund", {"order_id": "2087", "customer_email": "not provided", "reason": "changed mind"}),
            reply("Your refund is on its way."),
            reply("Welcome."),
        ],
    )
    asks = customer(("Order 2087, I changed my mind. I would like a refund.", False), ("Thanks!", True))

    transcript = run_llm_simulation(scenario, asks, agent=flawed, seed=1)

    checks = {c.name: c for c in transcript.outcome.checks}
    assert transcript.outcome.success is False
    assert checks["no_unexpected_refunds"].passed is False
    assert checks["identity_verified"].passed is False


def test_the_careful_agent_declines_the_same_request_and_succeeds() -> None:
    scenario = load_scenario("on_time_refund")
    careful, _ = agent_for(
        scenario,
        "agent_c",
        [
            tool("lookup_order", {"order_id": "2087"}),
            reply("That order arrived early, so it is not eligible. I can open a ticket."),
            reply("You are welcome."),
        ],
    )
    asks = customer(("Order 2087, I changed my mind. I would like a refund.", False), ("Okay, thanks.", True))

    transcript = run_llm_simulation(scenario, asks, agent=careful, seed=1)

    assert transcript.outcome.success is True
    assert transcript.final_state.refunds == []


def test_a_scripted_agent_that_says_refund_but_never_acts_fails_the_database_check() -> None:
    scenario = load_scenario("late_refund")

    four_turns = customer(*LATE_CUSTOMER[:2], ("A refund please.", False), ("Thanks, that is all.", True))

    transcript = run_llm_simulation(scenario, four_turns, seed=1)

    assert "refund for order 1042 will go back" in transcript.messages[5].content
    assert transcript.agent_id == "scripted_support"
    assert transcript.agent_usage is None and transcript.agent_prompt == ""
    assert transcript.tool_calls == []
    assert transcript.outcome.success is False


def test_every_run_gets_a_fresh_database() -> None:
    scenario = load_scenario("late_refund")
    outcomes = []
    for _ in range(2):
        agent, _ = agent_for(scenario, "agent_c", CORRECT_C)
        transcript = run_llm_simulation(scenario, customer(*LATE_CUSTOMER), agent=agent, seed=1)
        outcomes.append(transcript.outcome.success)

    assert outcomes == [True, True]


def test_using_every_step_without_replying_ends_the_run_with_step_limit() -> None:
    scenario = load_scenario("late_refund")
    config = AgentConfig(
        id="agent_x", label="X", description="x", prompt="careful", tools=["lookup_order"], max_steps=2
    )
    client = MockAgentClient([tool("lookup_order", {"order_id": "1042"})] * 2)
    agent = LLMAgent(config, client, scenario.environment, model="m")

    transcript = run_llm_simulation(scenario, customer(*LATE_CUSTOMER), agent=agent, seed=1)

    assert transcript.termination_reason is TerminationReason.step_limit
    assert "every tool step" in transcript.error
    assert [m.role for m in transcript.messages] == [Role.sim_user]
    assert len(transcript.tool_calls) == 2
    assert transcript.outcome is not None and transcript.checks is not None


def test_an_agent_model_failure_ends_the_run_with_model_error_and_keeps_the_record() -> None:
    scenario = load_scenario("late_refund")

    class Fails:
        def __init__(self) -> None:
            self.inner = MockAgentClient([tool("lookup_order", {"order_id": "1042"})], input_tokens=50)
            self.calls = 0

        def complete_agent(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return self.inner.complete_agent(**kwargs)
            raise LLMError("The model request timed out.", model_name="m", input_tokens=9, requests=3)

    agent = LLMAgent(load_agent_config("agent_c"), Fails(), scenario.environment, model="m")

    transcript = run_llm_simulation(scenario, customer(*LATE_CUSTOMER), agent=agent, seed=1)

    assert transcript.termination_reason is TerminationReason.model_error
    assert transcript.error == "The model request timed out."
    assert [m.role for m in transcript.messages] == [Role.sim_user]
    assert len(transcript.tool_calls) == 1
    assert transcript.agent_usage.requests == 4 and transcript.agent_usage.input_tokens == 59
    assert transcript.checks is None and transcript.outcome is None


def test_the_agent_gets_a_copy_so_it_cannot_rewrite_the_conversation() -> None:
    scenario = load_scenario("late_refund")

    class Vandal:
        id = "vandal"
        prompt = ""

        def exhausted(self) -> bool:
            return False

        def respond(self, history, env, *, seed):
            history.clear()
            from simbench.agent import AgentReply

            return AgentReply(message="Hello.")

        def usage(self, *, seed):
            return None

    transcript = run_llm_simulation(
        scenario, customer(("Refund please, order 1042.", False), ("Bye.", True)), agent=Vandal(), seed=1
    )

    assert [m.content for m in transcript.messages][:2] == ["Refund please, order 1042.", "Hello."]


def test_a_scenario_without_an_environment_has_no_outcome() -> None:
    scenario = load_scenario("late_refund").model_copy(update={"environment": None})

    transcript = run_llm_simulation(scenario, customer(*LATE_CUSTOMER), seed=1)

    assert transcript.outcome is None and transcript.final_state is None and transcript.tool_calls == []


def test_both_bundled_scenarios_load_with_consistent_environments() -> None:
    for scenario_id, expected_refunds in (("late_refund", 1), ("on_time_refund", 0)):
        scenario = load_scenario(scenario_id)
        assert scenario.sim_user is not None
        assert len(scenario.environment.expected_outcome.refunds) == expected_refunds
