"""The scripted agent, the agent version configs, and what makes the versions differ."""

import json

import pytest
from pydantic import ValidationError

from simbench.agent import (
    AgentConfig,
    AgentNotFoundError,
    ScriptedAgent,
    list_agent_configs,
    load_agent_config,
    load_prompt,
)
from simbench.environment import READ_ONLY_TOOLS, TOOLS
from simbench.llm_agent import build_agent_prompt
from simbench.scenario import load_scenario


def test_scripted_agent_plays_its_lines_in_order_then_is_exhausted() -> None:
    agent = ScriptedAgent(["one", "two"], agent_id="s")

    assert agent.exhausted() is False
    assert agent.respond([], None, seed=1).message == "one"
    assert agent.respond([], None, seed=1).message == "two"
    assert agent.exhausted() is True
    assert agent.id == "s" and agent.prompt == "" and agent.usage(seed=1) is None


def test_all_five_versions_load_and_ids_match_their_files() -> None:
    assert [config.id for config in list_agent_configs()] == [
        "agent_a",
        "agent_b",
        "agent_b_prime",
        "agent_c",
        "agent_d",
    ]


def test_a_and_b_can_only_read_while_c_and_d_can_act() -> None:
    tools = {config.id: config.tools for config in list_agent_configs()}

    assert tools["agent_a"] == tools["agent_b"] == tools["agent_b_prime"] == READ_ONLY_TOOLS
    assert set(tools["agent_c"]) == set(tools["agent_d"]) == set(TOOLS)


def test_b_prime_is_an_exact_copy_of_b_apart_from_its_name() -> None:
    b = load_agent_config("agent_b").model_dump()
    b_prime = load_agent_config("agent_b_prime").model_dump()
    for field in ("id", "label", "description"):
        b.pop(field)
        b_prime.pop(field)

    assert b == b_prime


def test_b_prime_and_b_get_byte_identical_prompts() -> None:
    env = load_scenario("late_refund").environment

    assert build_agent_prompt(load_agent_config("agent_b"), env) == build_agent_prompt(
        load_agent_config("agent_b_prime"), env
    )


def test_d_differs_from_c_only_by_its_prompt_and_says_so() -> None:
    c = load_agent_config("agent_c").model_dump()
    d = load_agent_config("agent_d").model_dump()

    differing = {key for key in c if c[key] != d[key]}

    assert differing == {"id", "label", "description", "prompt", "flaw"}
    assert d["flaw"] and not c["flaw"]
    assert all(not config.flaw for config in list_agent_configs() if config.id != "agent_d")


def test_the_flawed_prompt_changes_exactly_the_two_steps_that_hold_the_safeguards() -> None:
    careful = set(load_prompt("careful").splitlines())
    flawed = set(load_prompt("flawed").splitlines())

    assert sorted(careful - flawed) == [
        "2. Check the order against the company policy. Do not promise anything the policy does not allow.",
        "3. Before any refund, ask the customer for the email address on their account and use exactly what they tell you. Never guess it.",
    ]
    assert len(flawed - careful) == 2


def test_agent_a_is_not_shown_the_policy_but_b_and_c_are() -> None:
    env = load_scenario("late_refund").environment

    assert env.policy_text not in build_agent_prompt(load_agent_config("agent_a"), env)
    assert env.policy_text in build_agent_prompt(load_agent_config("agent_b"), env)


def test_the_prompt_lists_only_the_tools_the_version_may_use() -> None:
    env = load_scenario("late_refund").environment
    read_only = build_agent_prompt(load_agent_config("agent_b"), env)
    full = build_agent_prompt(load_agent_config("agent_c"), env)

    assert "lookup_order" in read_only and "issue_refund" not in read_only
    assert "issue_refund" in full and "create_ticket" in full


def test_config_rejects_unknown_and_repeated_tools_and_bad_ids() -> None:
    base = {"id": "agent_x", "label": "X", "description": "x", "prompt": "basic"}

    with pytest.raises(ValidationError, match="unknown tools: hack"):
        AgentConfig.model_validate({**base, "tools": ["hack"]})
    with pytest.raises(ValidationError, match="must not repeat"):
        AgentConfig.model_validate({**base, "tools": ["lookup_order", "lookup_order"]})
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({**base, "id": "Agent X"})


def test_missing_config_and_id_mismatch_are_reported(tmp_path) -> None:
    with pytest.raises(AgentNotFoundError):
        load_agent_config("nobody", tmp_path)

    (tmp_path / "agent_x.json").write_text(
        json.dumps({"id": "agent_y", "label": "Y", "description": "y", "prompt": "basic"})
    )
    with pytest.raises(ValueError, match="declares id 'agent_y'"):
        load_agent_config("agent_x", tmp_path)
