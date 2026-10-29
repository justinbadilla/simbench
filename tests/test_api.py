"""POST /runs persists a transcript. LLM tests inject a fake client."""

from fastapi.testclient import TestClient

from simbench.llm import MockLLMClient, SimUserTurn
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
