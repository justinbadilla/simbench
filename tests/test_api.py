"""POST /runs persists a transcript. LLM tests inject a fake client."""

import json

from fastapi.testclient import TestClient

from simbench.llm import AgentAction, MockAgentClient, MockLLMClient, SimUserTurn
from simbench.store import load_transcript
from simbench_api.main import create_app


def test_post_runs_returns_an_alternating_transcript_and_stores_it(tmp_path) -> None:
    db_path = tmp_path / "runs.sqlite"
    client = TestClient(create_app(db_path=db_path))

    response = client.post("/runs", json={"mode": "scripted"})

    assert response.status_code == 200
    body = response.json()
    assert body["scenario_id"] == "late_refund"
    assert body["termination_reason"] == "script_complete"
    assert body["usage"]["model_name"] is None
    roles = [message["role"] for message in body["messages"]]
    assert roles == ["sim_user", "agent"] * 4
    assert [message["turn_index"] for message in body["messages"]] == list(range(8))
    assert body["variant"] == "scripted"
    assert body["checks"] is None

    loaded = load_transcript(db_path, body["run_id"])
    assert loaded is not None
    assert loaded.model_dump(mode="json")["messages"] == body["messages"]


def test_post_runs_honors_a_turn_cap(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "runs.sqlite"))

    response = client.post(
        "/runs",
        json={"scenario_id": "late_refund", "max_turns": 1, "mode": "scripted"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["termination_reason"] == "max_turns"
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "sim_user"


def test_unknown_scenario_is_not_found(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "runs.sqlite"))

    response = client.post("/runs", json={"scenario_id": "missing", "mode": "scripted"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown scenario: missing"


def test_llm_run_is_stored_with_usage(tmp_path) -> None:
    db_path = tmp_path / "runs.sqlite"
    fake = MockLLMClient(
        [
            SimUserTurn(message="I want a refund for order 1042.", end_conversation=False),
            SimUserTurn(message="Thanks, that is all.", end_conversation=True),
        ],
        input_tokens=12,
        output_tokens=6,
        latency_ms=40,
        estimated_cost_usd=0.000005,
    )
    client = TestClient(create_app(db_path=db_path, llm_client=fake, model="gpt-4o-mini"))

    response = client.post("/runs", json={"seed": 11})

    assert response.status_code == 200
    body = response.json()
    assert body["termination_reason"] == "sim_user_ended"
    assert body["usage"]["provider"] == "mock"
    assert body["usage"]["seed"] == 11
    assert body["usage"]["input_tokens"] == 24
    assert body["usage"]["latency_ms"] == 80.0
    assert body["usage"]["requests"] == 2
    assert body["usage"]["model_name"] == "mock-sim-user"
    assert "Get a refund for order 1042." in body["system_prompt"]
    loaded = load_transcript(db_path, body["run_id"])
    assert loaded is not None
    assert loaded.messages[0].content == "I want a refund for order 1042."


def test_llm_run_returns_variant_checks_and_state(tmp_path) -> None:
    db_path = tmp_path / "runs.sqlite"
    fake = MockLLMClient(
        [
            SimUserTurn(
                message="Hi, order 1042 was due last Tuesday. I want a refund.",
                end_conversation=False,
            ),
            SimUserTurn(message="It is jordan.lee@example.com.", end_conversation=False),
            SimUserTurn(message="A refund please.", end_conversation=False),
            SimUserTurn(message="Thanks, that is all.", end_conversation=True),
        ]
    )
    client = TestClient(create_app(db_path=db_path, llm_client=fake, model="gpt-4o-mini"))

    response = client.post("/runs", json={"variant": "persona", "seed": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["variant"] == "persona"
    assert body["checks"]["all_passed"] is True
    assert body["checks"]["checks_version"] == "checks.v1"
    assert len(body["turn_prompts"]) == 4
    assert body["sim_user_spec"]["facts"][0]["id"] == "order_number"
    assert [event["kind"] for event in body["state_log"]][:3] == ["revealed"] * 3
    loaded = load_transcript(db_path, body["run_id"])
    assert loaded is not None and loaded.checks is not None
    assert loaded.checks.all_passed is True


def test_unknown_variant_is_rejected(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "runs.sqlite"))

    assert client.post("/runs", json={"variant": "chaotic"}).status_code == 422


def test_missing_api_key_is_a_server_error(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "runs.sqlite"))

    response = client.post("/runs", json={})

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
    assert "browser" in response.json()["detail"]


# ---- V3: agents, scenarios, and tool use through the API ----


class _BothSides:
    """One fake client that serves both the SimUser (``complete``) and the agent (``complete_agent``)."""

    def __init__(self, customer_turns, agent_actions) -> None:
        self._customer = MockLLMClient(customer_turns)
        self._agent = MockAgentClient(agent_actions)
        self.complete = self._customer.complete
        self.complete_agent = self._agent.complete_agent


def _refunding_client() -> _BothSides:
    email = "jordan.lee@example.com"
    return _BothSides(
        [
            SimUserTurn(message="Order 1042 was due last Tuesday. I want a refund.", end_conversation=False),
            SimUserTurn(message=f"The email on the account is {email}.", end_conversation=False),
            SimUserTurn(message="Thanks, that is all.", end_conversation=True),
        ],
        [
            AgentAction(action="tool_call", tool="lookup_order", arguments_json='{"order_id": "1042"}'),
            AgentAction(action="reply", message="Sorry about that. What email is on the account?"),
            AgentAction(
                action="tool_call",
                tool="issue_refund",
                arguments_json=json.dumps({"order_id": "1042", "customer_email": email, "reason": "late"}),
            ),
            AgentAction(action="reply", message="Done. Your refund is on its way."),
        ],
    )


def test_scenarios_and_agents_are_listed_for_the_ui(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "runs.sqlite"))

    scenarios = client.get("/scenarios").json()
    agents = client.get("/agents").json()

    assert [s["id"] for s in scenarios] == ["late_refund", "on_time_refund"]
    assert all(s["has_environment"] for s in scenarios)
    assert [a["id"] for a in agents] == ["agent_a", "agent_b", "agent_b_prime", "agent_c", "agent_d"]
    assert next(a for a in agents if a["id"] == "agent_c")["tools"] == [
        "lookup_order",
        "issue_refund",
        "create_ticket",
    ]


def test_a_tool_using_agent_run_is_judged_by_the_database_and_stored(tmp_path) -> None:
    db_path = tmp_path / "runs.sqlite"
    client = TestClient(create_app(db_path=db_path, llm_client=_refunding_client(), model="m"))

    response = client.post("/runs", json={"agent": "agent_c", "variant": "stateful", "seed": 9})

    assert response.status_code == 200
    body = response.json()
    assert body["agent_id"] == "agent_c"
    assert body["outcome"]["success"] is True
    assert [call["tool"] for call in body["tool_calls"]] == ["lookup_order", "issue_refund"]
    assert body["final_state"]["refunds"][0]["email_matched"] is True
    assert body["agent_usage"]["requests"] == 4 and body["usage"]["requests"] == 3
    assert "Tools you can use" in body["agent_prompt"]
    loaded = load_transcript(db_path, body["run_id"])
    assert loaded is not None and loaded.outcome is not None and loaded.outcome.success is True


def test_the_default_scripted_agent_run_reports_a_failed_outcome_when_it_only_talks(tmp_path) -> None:
    fake = MockLLMClient(
        [
            SimUserTurn(message="Order 1042, due last Tuesday. A refund please.", end_conversation=False),
            SimUserTurn(message="jordan.lee@example.com", end_conversation=False),
            SimUserTurn(message="A refund please.", end_conversation=False),
            SimUserTurn(message="Thanks.", end_conversation=True),
        ]
    )
    client = TestClient(create_app(db_path=tmp_path / "runs.sqlite", llm_client=fake, model="m"))

    body = client.post("/runs", json={}).json()

    assert "refund for order 1042 will go back" in body["messages"][5]["content"]
    assert body["agent_id"] == "scripted_support" and body["agent_usage"] is None
    assert body["outcome"]["success"] is False


def test_the_second_scenario_runs_and_expects_no_refund(tmp_path) -> None:
    fake = MockLLMClient(
        [
            SimUserTurn(message="Order 2087, I changed my mind. I want a refund.", end_conversation=False),
            SimUserTurn(message="casey.morgan@example.com", end_conversation=False),
            SimUserTurn(message="Okay, thanks anyway.", end_conversation=True),
        ]
    )
    client = TestClient(create_app(db_path=tmp_path / "runs.sqlite", llm_client=fake, model="m"))

    body = client.post("/runs", json={"scenario_id": "on_time_refund"}).json()

    assert body["scenario_id"] == "on_time_refund"
    assert body["outcome"]["success"] is True


def test_an_unknown_agent_is_not_found(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "runs.sqlite", llm_client=MockLLMClient([]), model="m"))

    response = client.post("/runs", json={"agent": "agent_z"})

    assert response.status_code == 404 and "agent_z" in response.json()["detail"]


def test_a_client_that_cannot_drive_an_agent_gets_a_clear_message(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "runs.sqlite", llm_client=MockLLMClient([]), model="m"))

    response = client.post("/runs", json={"agent": "agent_c"})

    assert response.status_code == 400
    assert "replay mode" in response.json()["detail"]


def test_scripted_mode_cannot_use_a_model_agent(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "runs.sqlite"))

    response = client.post("/runs", json={"mode": "scripted", "agent": "agent_c"})

    assert response.status_code == 400


def test_an_agent_run_without_a_key_still_says_to_set_one(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "runs.sqlite"))

    response = client.post("/runs", json={"agent": "agent_c"})

    assert response.status_code == 503
