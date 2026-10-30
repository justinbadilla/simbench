"""Provider adapters against a fake HTTP transport. No network."""

import json

import httpx

from simbench.llm import GeminiClient, LLMError, OpenAIClient, SimUserTurn
from simbench.pricing import estimate_cost_usd


def _body(content: str, *, model: str = "gpt-4o-mini", prompt_tokens: int = 11, completion_tokens: int = 7) -> dict:
    """Build a fake OpenAI chat-completions response body with the given text and token counts."""
    return {
        "model": model,
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _client(handler, *, key: str = "sk-test-secret", sleeper=None, max_attempts: int = 3) -> OpenAIClient:
    """Build an ``OpenAIClient`` wired to a fake transport; retries do not really sleep."""
    return OpenAIClient(
        api_key=key,
        transport=httpx.MockTransport(handler),
        sleeper=sleeper or (lambda _seconds: None),
        max_attempts=max_attempts,
    )


def test_successful_call_sends_a_schema_and_prices_the_tokens() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["authorization"]
        seen["payload"] = json.loads(request.content.decode())
        content = SimUserTurn(message="I want a refund.", end_conversation=False).model_dump_json()
        return httpx.Response(200, json=_body(content))

    result = _client(handler).complete(
        messages=[{"role": "system", "content": "Be a customer."}],
        model="gpt-4o-mini",
        temperature=0.7,
        seed=4,
    )

    assert seen["authorization"] == "Bearer sk-test-secret"
    assert seen["payload"]["response_format"]["json_schema"]["strict"] is True
    assert seen["payload"]["seed"] == 4
    assert result.turn.message == "I want a refund."
    assert result.provider == "openai"
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert result.estimated_cost_usd == estimate_cost_usd("gpt-4o-mini", 11, 7)
    assert "sk-test-secret" not in result.turn.message


def test_retries_a_server_error_then_prices_both_attempts() -> None:
    calls = {"n": 0}
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                500,
                headers={"retry-after": "2"},
                json=_body("unused", prompt_tokens=3, completion_tokens=1),
            )
        content = SimUserTurn(message="Hello", end_conversation=True).model_dump_json()
        return httpx.Response(200, json=_body(content, prompt_tokens=10, completion_tokens=2))

    result = _client(handler, sleeper=delays.append).complete(
        messages=[],
        model="gpt-4o-mini",
        temperature=0,
        seed=None,
    )

    assert calls["n"] == 2
    assert delays == [2.0]
    assert result.turn.end_conversation is True
    assert result.input_tokens == 13
    assert result.output_tokens == 3
    assert result.estimated_cost_usd == estimate_cost_usd("gpt-4o-mini", 13, 3)


def test_retries_invalid_json_then_accepts_the_next_reply() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=_body("not json", prompt_tokens=5, completion_tokens=1))
        content = SimUserTurn(message="A real message", end_conversation=False).model_dump_json()
        return httpx.Response(200, json=_body(content, prompt_tokens=6, completion_tokens=4))

    result = _client(handler).complete(messages=[], model="gpt-4o-mini", temperature=0, seed=1)

    assert result.turn.message == "A real message"
    assert result.input_tokens == 11
    assert result.output_tokens == 5


def test_timeout_stops_after_the_attempt_budget_without_leaking_the_key() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        raise httpx.ReadTimeout("timed out")

    try:
        _client(handler, key="sk-test-secret").complete(
            messages=[],
            model="gpt-4o-mini",
            temperature=0,
            seed=1,
        )
    except LLMError as error:
        assert str(error) == "The model request timed out."
        assert "sk-test-secret" not in str(error)
    else:
        raise AssertionError("expected LLMError")
    assert calls["n"] == 3


def test_bad_status_does_not_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request sk-test-secret"})

    try:
        _client(handler).complete(messages=[], model="gpt-4o-mini", temperature=0, seed=1)
    except LLMError as error:
        assert "400" in str(error)
        assert "sk-test-secret" not in str(error)
    else:
        raise AssertionError("expected LLMError")
    assert calls["n"] == 1


def test_unknown_model_has_no_invented_price() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        content = SimUserTurn(message="Hello", end_conversation=False).model_dump_json()
        return httpx.Response(200, json=_body(content, model="made-up"))

    result = _client(handler).complete(messages=[], model="made-up", temperature=0, seed=1)

    assert result.estimated_cost_usd is None
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == 0.15


# ---- Gemini ----


def _gemini_body(
    content: str,
    *,
    model: str = "gemini-3.5-flash-lite",
    prompt_tokens: int = 20,
    candidate_tokens: int = 6,
    thought_tokens: int = 0,
) -> dict:
    """Build a fake Gemini ``generateContent`` response body with the given text and token counts."""
    usage = {"promptTokenCount": prompt_tokens, "candidatesTokenCount": candidate_tokens}
    if thought_tokens:
        usage["thoughtsTokenCount"] = thought_tokens
    return {
        "candidates": [{"content": {"parts": [{"text": content}]}}],
        "usageMetadata": usage,
        "modelVersion": model,
    }


def _gemini(handler, *, key: str = "AIza-test-secret", free_tier: bool = True, sleeper=None, delays=None) -> GeminiClient:
    """Build a ``GeminiClient`` wired to a fake transport; ``delays`` collects the waits it would have slept."""
    return GeminiClient(
        api_key=key,
        free_tier=free_tier,
        transport=httpx.MockTransport(handler),
        sleeper=sleeper or (delays.append if delays is not None else (lambda _seconds: None)),
    )


def test_gemini_request_shape_and_usage() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key_header"] = request.headers.get("x-goog-api-key")
        seen["payload"] = json.loads(request.content.decode())
        content = SimUserTurn(message="I want a refund.", end_conversation=False).model_dump_json()
        return httpx.Response(200, json=_gemini_body(content, thought_tokens=4))

    result = _gemini(handler).complete(
        messages=[
            {"role": "system", "content": "Be a customer."},
            {"role": "user", "content": "Conversation so far: (none)"},
        ],
        model="gemini-3.5-flash-lite",
        temperature=0.7,
        seed=5,
    )

    assert seen["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
    )
    assert "AIza-test-secret" not in seen["url"]
    assert seen["key_header"] == "AIza-test-secret"
    payload = seen["payload"]
    assert payload["systemInstruction"] == {"parts": [{"text": "Be a customer."}]}
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "Conversation so far: (none)"}]}
    ]
    config = payload["generationConfig"]
    assert config["temperature"] == 0.7
    assert config["seed"] == 5
    assert config["responseMimeType"] == "application/json"
    assert "additionalProperties" not in config["responseJsonSchema"]
    assert result.turn.message == "I want a refund."
    assert result.provider == "gemini"
    assert result.model_name == "gemini-3.5-flash-lite"
    assert result.input_tokens == 20
    assert result.output_tokens == 10
    assert result.requests == 1
    assert result.free_tier is True
    assert result.estimated_cost_usd == estimate_cost_usd("gemini-3.5-flash-lite", 20, 10)


def test_gemini_free_tier_flag_can_be_turned_off() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        content = SimUserTurn(message="Hello", end_conversation=False).model_dump_json()
        return httpx.Response(200, json=_gemini_body(content))

    result = _gemini(handler, free_tier=False).complete(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-3.5-flash-lite",
        temperature=0,
        seed=None,
    )

    assert result.free_tier is False


def test_gemini_rate_limit_waits_for_the_provider_delay_then_succeeds() -> None:
    calls = {"n": 0}
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "status": "RESOURCE_EXHAUSTED",
                        "details": [{"retryDelay": "7s"}],
                    }
                },
            )
        content = SimUserTurn(message="Hello", end_conversation=True).model_dump_json()
        return httpx.Response(200, json=_gemini_body(content))

    result = _gemini(handler, delays=delays).complete(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-3.5-flash-lite",
        temperature=0,
        seed=1,
    )

    assert delays == [7.0]
    assert calls["n"] == 2
    assert result.requests == 2


def test_gemini_does_not_wait_for_a_delay_longer_than_the_cap() -> None:
    calls = {"n": 0}
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        return httpx.Response(
            429,
            json={"error": {"details": [{"retryDelay": "58s"}]}},
        )

    try:
        _gemini(handler, delays=delays).complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-3.5-flash-lite",
            temperature=0,
            seed=1,
        )
    except LLMError as error:
        assert "rate limited" in str(error)
        assert error.requests == 1
        assert error.free_tier is True
    else:
        raise AssertionError("expected LLMError")
    assert calls["n"] == 1
    assert delays == []


def test_gemini_daily_quota_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        return httpx.Response(
            429,
            json={"error": {"message": "Quota exceeded for GenerateRequestsPerDayPerProjectPerModel-FreeTier"}},
        )

    try:
        _gemini(handler).complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-3.5-flash-lite",
            temperature=0,
            seed=1,
        )
    except LLMError as error:
        assert "midnight Pacific" in str(error)
    else:
        raise AssertionError("expected LLMError")
    assert calls["n"] == 1


def test_gemini_invalid_key_is_reported_without_echoing_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, json={"error": {"message": "API key not valid. AIza-test-secret"}})

    try:
        _gemini(handler).complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-3.5-flash-lite",
            temperature=0,
            seed=1,
        )
    except LLMError as error:
        assert str(error) == "The model rejected the API key."
        assert "AIza-test-secret" not in str(error)
    else:
        raise AssertionError("expected LLMError")


def test_gemini_reply_without_text_is_retried_then_fails_cleanly() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "candidates": [{"finishReason": "SAFETY"}],
                "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 0},
                "modelVersion": "gemini-3.5-flash-lite",
            },
        )

    try:
        _gemini(handler).complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-3.5-flash-lite",
            temperature=0,
            seed=1,
        )
    except LLMError as error:
        assert "did not match" in str(error)
        assert error.requests == 3
    else:
        raise AssertionError("expected LLMError")
    assert calls["n"] == 3


def test_missing_gemini_key_names_the_variable() -> None:
    try:
        GeminiClient(api_key="")
    except LLMError as error:
        assert "GEMINI_API_KEY" in str(error)
    else:
        raise AssertionError("expected LLMError")


# ---- Agent steps (the same adapters, a different response shape) ----


def _agent_json(**fields: str) -> str:
    base = {"action": "reply", "tool": "", "arguments_json": "{}", "message": ""}
    return json.dumps({**base, **fields})


def test_gemini_agent_step_asks_for_the_agent_schema_and_parses_a_tool_call() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content.decode())
        content = _agent_json(action="tool_call", tool="lookup_order", arguments_json='{"order_id": "1042"}')
        return httpx.Response(200, json=_gemini_body(content, thought_tokens=3))

    call = _gemini(handler).complete_agent(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        model="gemini-3.5-flash-lite",
        temperature=0.2,
        seed=4,
    )

    schema = seen["payload"]["generationConfig"]["responseJsonSchema"]
    assert schema["properties"]["action"]["enum"] == ["tool_call", "reply"]
    assert "additionalProperties" not in schema
    assert (call.action.action, call.action.tool) == ("tool_call", "lookup_order")
    assert json.loads(call.action.arguments_json) == {"order_id": "1042"}
    assert call.provider == "gemini" and call.requests == 1 and call.free_tier is True
    assert call.output_tokens == 6 + 3


def test_openai_agent_step_uses_a_strict_named_schema() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_body(_agent_json(message="Hello there.")))

    call = _client(handler).complete_agent(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        model="gpt-4o-mini",
        temperature=0.2,
        seed=None,
    )

    spec = seen["payload"]["response_format"]["json_schema"]
    assert spec["name"] == "agent_action" and spec["strict"] is True
    assert sorted(spec["schema"]["required"]) == ["action", "arguments_json", "message", "tool"]
    assert call.action.message == "Hello there."
    assert call.estimated_cost_usd == estimate_cost_usd("gpt-4o-mini", 11, 7)


def test_both_kinds_of_call_use_the_same_retry_path_and_a_wrongly_shaped_step_is_retried() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            # A reply with no message does not fit the agent shape.
            return httpx.Response(200, json=_body(_agent_json(action="reply", message="")))
        return httpx.Response(200, json=_body(_agent_json(message="Second try works.")))

    call = _client(handler).complete_agent(messages=[], model="gpt-4o-mini", temperature=0.2, seed=1)

    assert call.action.message == "Second try works."
    assert call.requests == 2
    assert call.input_tokens == 22  # the failed attempt is still counted


def test_a_simuser_call_still_returns_a_simuser_turn_after_the_refactor() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = SimUserTurn(message="Hello.", end_conversation=True).model_dump_json()
        return httpx.Response(200, json=_body(content))

    call = _client(handler).complete(messages=[], model="gpt-4o-mini", temperature=0.7, seed=1)

    assert (call.turn.message, call.turn.end_conversation) == ("Hello.", True)
