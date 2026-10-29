"""LLM clients.

Tests use the mock and the replay client. GeminiClient and OpenAIClient are the
only ones that talk to a provider, and they share one retry path.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from simbench.pricing import estimate_cost_usd, has_price

# JSON schema for the one thing the model must return: the next customer message
# and whether that message ends the chat. Sent to the provider so the reply is
# structured, then validated again by ``SimUserTurn``.
SIM_USER_TURN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "message": {"type": "string"},
        "end_conversation": {"type": "boolean"},
    },
    "required": ["message", "end_conversation"],
}

# Gemini's JSON-schema subset is narrower, so leave out keywords it may reject.
_GEMINI_TURN_SCHEMA = {
    key: value for key, value in SIM_USER_TURN_SCHEMA.items() if key != "additionalProperties"
}

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """A model call failed after retries. The message is safe to show in the UI.

    Carries the usage the failed attempts still consumed (tokens, latency,
    requests, estimated cost) so the runner can include it in the run totals.
    Raised by the provider adapters; caught by ``run_llm_simulation``, which
    ends the run with ``model_error`` instead of crashing.
    """

    def __init__(
        self,
        message: str,
        *,
        model_name: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        estimated_cost_usd: float | None = 0.0,
        requests: int = 0,
        free_tier: bool = False,
    ) -> None:
        """Store the user-facing message plus any usage spent before the failure."""
        super().__init__(message)
        self.model_name = model_name
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.estimated_cost_usd = estimated_cost_usd
        self.requests = requests
        self.free_tier = free_tier


class SimUserTurn(BaseModel):
    """One structured reply from the model: the customer's next message and an end flag.

    Parsed from the provider's JSON text. Anything that does not fit (empty
    message, missing field) is rejected and treated as a failed attempt.
    """

    message: str = Field(min_length=1)
    end_conversation: bool

    @field_validator("message")
    @classmethod
    def message_has_text(cls, value: str) -> str:
        """Pydantic hook: strip whitespace and reject a message that is blank after stripping."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message is empty")
        return cleaned


class LLMCall(BaseModel):
    """The result of one successful ``complete`` call: the turn plus its usage.

    This is also the record format of replay fixtures, so a stored list of
    these can stand in for a provider.
    """

    turn: SimUserTurn
    model_name: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    # HTTP requests this turn took, retries included. Free-tier quotas count requests.
    requests: int = 1
    # True when the project is on a free tier, so nothing is billed.
    free_tier: bool = False
    # List-price estimate at paid rates. On the free tier this is not what you paid.
    estimated_cost_usd: float | None = None


class LLMClient(Protocol):
    """The only interface the runner knows about. Any object with ``complete`` qualifies.

    Implemented by ``GeminiClient`` and ``OpenAIClient`` (real providers) and by
    ``MockLLMClient``, ``ReplayLLMClient`` and ``RecordingLLMClient`` (offline).
    Keeping the runner on this protocol is what lets tests run without a network.
    """

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> LLMCall:
        """Ask the model for the customer's next message.

        Args:
            messages: Chat messages from ``build_model_messages`` (``system`` then ``user``).
            model: Provider model name.
            temperature: Sampling temperature.
            seed: Provider seed for best-effort repeatability, or ``None``.

        Returns:
            An ``LLMCall`` with the parsed turn and its usage.

        Raises:
            LLMError: If the call fails after retries.
        """


class MockLLMClient:
    """Returns scripted turns and records the prompts it was given."""

    def __init__(
        self,
        turns: list[SimUserTurn],
        *,
        model_name: str = "mock-sim-user",
        provider: str = "mock",
        input_tokens: int = 10,
        output_tokens: int = 5,
        latency_ms: float = 25.0,
        estimated_cost_usd: float | None = 0.0,
        requests: int = 1,
        free_tier: bool = False,
    ) -> None:
        """Queue the turns to return, and fix the usage numbers every call will report.

        Args:
            turns: Replies handed out in order, one per ``complete`` call.
            model_name, provider, input_tokens, output_tokens, latency_ms,
            estimated_cost_usd, requests, free_tier: Reported verbatim on every
                ``LLMCall``, so tests can assert on the aggregation in the runner.
        """
        self._turns = list(turns)
        self._model_name = model_name
        self._provider = provider
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._latency_ms = latency_ms
        self._estimated_cost_usd = estimated_cost_usd
        self._requests = requests
        self._free_tier = free_tier
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> LLMCall:
        """Record the arguments, then return the next queued turn.

        ``calls`` keeps every prompt it was given, so tests can check what the
        runner sent on each turn.

        Raises:
            LLMError: If the queue is empty.
        """
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "seed": seed,
            }
        )
        if not self._turns:
            raise LLMError("Mock LLM client has no turns left.")
        turn = self._turns.pop(0)
        return LLMCall(
            turn=turn,
            model_name=self._model_name,
            provider=self._provider,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=self._latency_ms,
            requests=self._requests,
            free_tier=self._free_tier,
            estimated_cost_usd=self._estimated_cost_usd,
        )


class ReplayLLMClient:
    """Plays back a JSON list of LLMCall records. Does not call a provider."""

    def __init__(self, path: Path) -> None:
        """Load and validate a JSON list of ``LLMCall`` records from ``path``.

        Raises:
            FileNotFoundError, ValueError: If the file is missing or malformed.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._calls = [LLMCall.model_validate(item) for item in payload]

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> LLMCall:
        """Return the next recorded call. The prompt, model, temperature and seed are ignored.

        Raises:
            LLMError: If every recorded call has been used.
        """
        del messages, model, temperature, seed
        if not self._calls:
            raise LLMError("Replay file has no calls left.")
        return self._calls.pop(0)


class RecordingLLMClient:
    """Appends each inner call to a JSON list so a later run can replay it."""

    def __init__(self, inner: LLMClient, path: Path) -> None:
        """Wrap ``inner`` and write each of its results to the JSON file at ``path``."""
        self._inner = inner
        self._path = path

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> LLMCall:
        """Delegate to the wrapped client, append the result to the file, and return it.

        Only successful calls are recorded. The file is created if missing and
        extended if it already exists.
        """
        result = self._inner.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            seed=seed,
        )
        existing: list[object] = []
        if self._path.is_file():
            existing = json.loads(self._path.read_text(encoding="utf-8"))
        existing.append(json.loads(result.model_dump_json()))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return result


class _Spent:
    """Everything one complete() call used across its attempts, failed ones included."""

    def __init__(self, free_tier: bool) -> None:
        """Start with zero usage; ``free_tier`` is copied onto the result or error."""
        self.free_tier = free_tier
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_ms = 0.0
        self._costs: list[float | None] = []

    def add_usage(self, input_tokens: int, output_tokens: int, cost: float | None) -> None:
        """Add one response's tokens and its price estimate (``None`` if the price is unknown)."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self._costs.append(cost)

    def cost(self) -> float | None:
        """Total estimated cost so far, or ``None`` if any attempt had an unknown price."""
        if not self._costs:
            return 0.0
        if any(item is None for item in self._costs):
            return None
        return round(sum(item for item in self._costs if item is not None), 8)

    def error(self, message: str, model: str) -> LLMError:
        """Build an ``LLMError`` that carries everything spent so far. The caller raises it."""
        return LLMError(
            message,
            model_name=model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=round(self.latency_ms, 1),
            estimated_cost_usd=self.cost(),
            requests=self.requests,
            free_tier=self.free_tier,
        )


class _HTTPClient:
    """One request loop with a timeout and retries. Subclasses shape the request and read the reply.

    The API key goes in a header and is never part of an error message.
    """

    provider = ""
    missing_key_hint = ""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        free_tier: bool = False,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        max_wait_seconds: float = 20.0,
        sleeper: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Configure the request loop.

        Args:
            api_key: Provider key, sent only in a header and never put in messages.
            base_url: API root, without a trailing slash.
            free_tier: Whether calls are unbilled; copied onto every result.
            timeout_seconds: Per-request timeout.
            max_attempts: Total tries per call, including the first.
            max_wait_seconds: Longest provider-requested pause that will be
                honored; a longer one stops retrying so a web request is not held open.
            sleeper: Replaceable for tests so retries do not really wait.
            transport: Replaceable ``httpx`` transport so tests use a fake server.

        Raises:
            LLMError: If ``api_key`` is empty, with the provider's setup hint.
        """
        if not api_key:
            raise LLMError(self.missing_key_hint)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._free_tier = free_tier
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._max_wait = max_wait_seconds
        self._sleeper = sleeper
        self._transport = transport

    # Subclass hooks.
    def _request(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> tuple[str, dict[str, str], dict[str, object]]:
        """Hook: build the provider request.

        Returns:
            ``(url, headers, json_payload)``.
        """
        raise NotImplementedError

    def _usage(self, body: dict, requested_model: str) -> tuple[str, int, int]:
        """Hook: read usage from a response body.

        Returns:
            ``(model name, input tokens, output tokens)``. The model name is the
            one the provider reports, falling back to ``requested_model``.
        """
        raise NotImplementedError

    def _text(self, body: dict) -> str:
        """Hook: pull the model's JSON text out of a response body (parsed later as ``SimUserTurn``)."""
        raise NotImplementedError

    def _key_rejected(self, response: httpx.Response) -> bool:
        """Hook: True if the response means the API key is invalid. Never retried."""
        return response.status_code in {401, 403}

    def _quota_exhausted(self, response: httpx.Response) -> bool:
        """Hook: True if a 429 means a daily quota is used up, so retrying is pointless."""
        del response
        return False

    def _retry_delay(self, response: httpx.Response) -> float | None:
        """Hook: seconds the provider asks us to wait before retrying, or ``None`` for default backoff."""
        return _retry_after_header(response)

    # Shared request loop.
    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> LLMCall:
        """Send the request, retrying transient failures, and return the parsed turn.

        Retries on timeouts, 429/5xx responses, and replies that are not valid
        turn JSON, with exponential backoff or the provider's requested delay.
        Does not retry a rejected key, a used-up daily quota, or other 4xx errors.
        Usage from failed attempts is accumulated and reported either way.

        Raises:
            LLMError: When attempts run out or a non-retryable error occurs.
        """
        url, headers, payload = self._request(messages, model, temperature, seed)
        spent = _Spent(self._free_tier)
        last_error = "The model request failed."
        with httpx.Client(timeout=self._timeout, transport=self._transport) as http:
            for attempt in range(self._max_attempts):
                result, error, wait = self._attempt(http, url, headers, payload, model, spent)
                if result is not None:
                    return result
                last_error = error or last_error
                if attempt + 1 == self._max_attempts:
                    break
                if wait is not None and wait > self._max_wait:
                    # The provider wants a longer pause than a web request should hold.
                    break
                self._sleeper(0.25 * (2**attempt) if wait is None else max(wait, 0.0))
        raise spent.error(last_error, model)

    def _attempt(
        self,
        http: httpx.Client,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        requested_model: str,
        spent: _Spent,
    ) -> tuple[LLMCall | None, str | None, float | None]:
        """Make one HTTP attempt.

        Returns:
            ``(result, error, wait)``. On success ``result`` is the ``LLMCall`` and
            the others are ``None``. On a retryable failure ``result`` is ``None``,
            ``error`` explains why, and ``wait`` is the provider's requested delay.

        Raises:
            LLMError: For failures that retrying cannot fix.

        Called by:
            ``complete``, once per attempt.
        """
        started = time.perf_counter()
        spent.requests += 1
        try:
            response = http.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            spent.latency_ms += (time.perf_counter() - started) * 1000
            return None, "The model request timed out.", None
        spent.latency_ms += (time.perf_counter() - started) * 1000

        status = response.status_code
        if self._key_rejected(response):
            raise spent.error("The model rejected the API key.", requested_model)
        if status == 429 and self._quota_exhausted(response):
            raise spent.error(
                "The free-tier daily request quota looks used up. It resets at midnight Pacific time.",
                requested_model,
            )
        if status in _RETRYABLE_STATUS:
            self._record_usage(response, requested_model, spent)
            message = (
                "The model request was rate limited (429)."
                if status == 429
                else f"The model request failed ({status})."
            )
            return None, message, self._retry_delay(response)
        if status != 200:
            raise spent.error(f"The model request failed ({status}).", requested_model)

        try:
            body = response.json()
            model_name, input_tokens, output_tokens = self._usage(body, requested_model)
            spent.add_usage(
                input_tokens,
                output_tokens,
                _price(model_name, requested_model, input_tokens, output_tokens),
            )
            turn = SimUserTurn.model_validate_json(self._text(body))
        except (ValidationError, KeyError, IndexError, TypeError, ValueError, AttributeError):
            return None, "The model returned a message that did not match the expected JSON.", None
        return (
            LLMCall(
                turn=turn,
                model_name=model_name,
                provider=self.provider,
                input_tokens=spent.input_tokens,
                output_tokens=spent.output_tokens,
                latency_ms=round(spent.latency_ms, 1),
                requests=spent.requests,
                free_tier=spent.free_tier,
                estimated_cost_usd=spent.cost(),
            ),
            None,
            None,
        )

    def _record_usage(self, response: httpx.Response, requested_model: str, spent: _Spent) -> None:
        """Best-effort: count the tokens of a failed response so retries are billed in the totals.

        Silently does nothing if the body has no readable usage.
        """
        try:
            model_name, input_tokens, output_tokens = self._usage(response.json(), requested_model)
        except (ValueError, KeyError, TypeError, AttributeError):
            return
        spent.add_usage(
            input_tokens,
            output_tokens,
            _price(model_name, requested_model, input_tokens, output_tokens),
        )


class GeminiClient(_HTTPClient):
    """Google Gemini generateContent over HTTPS.

    The free tier is per model and capped per minute and per day. Google no
    longer prints the exact limits, so read them in AI Studio. A project with a
    billing account linked is no longer on the free tier.

    Written from Google's REST reference. It has been exercised against a fake
    transport in tests, not against the live service.
    """

    provider = "gemini"
    missing_key_hint = "Set GEMINI_API_KEY on the server to run the SimUser."

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        free_tier: bool = True,
        **kwargs: object,
    ) -> None:
        """Create a Gemini client. ``free_tier`` defaults to True; set it False if billing is linked.

        Extra keyword arguments (timeout, retries, transport) pass through to ``_HTTPClient``.
        """
        super().__init__(api_key, base_url=base_url, free_tier=free_tier, **kwargs)  # type: ignore[arg-type]

    def _request(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> tuple[str, dict[str, str], dict[str, object]]:
        """Build a ``generateContent`` request.

        System messages become ``systemInstruction``; the rest become ``contents``.
        Asks for JSON matching the turn schema. The key goes in the
        ``x-goog-api-key`` header, not the URL, so it cannot leak into logs.
        """
        system_parts = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        generation: dict[str, object] = {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseJsonSchema": _GEMINI_TURN_SCHEMA,
        }
        if seed is not None:
            generation["seed"] = seed
        payload: dict[str, object] = {"contents": contents, "generationConfig": generation}
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        return f"{self._base_url}/models/{model}:generateContent", headers, payload

    def _usage(self, body: dict, requested_model: str) -> tuple[str, int, int]:
        """Read ``usageMetadata``. Output tokens include thinking tokens, which are billed as output."""
        usage = body.get("usageMetadata") or {}
        model_name = body.get("modelVersion")
        if not isinstance(model_name, str) or not model_name:
            model_name = requested_model
        input_tokens = int(usage.get("promptTokenCount") or 0)
        # Thinking tokens are billed as output, so count them there.
        output_tokens = int(usage.get("candidatesTokenCount") or 0) + int(
            usage.get("thoughtsTokenCount") or 0
        )
        return model_name, input_tokens, output_tokens

    def _text(self, body: dict) -> str:
        """Join the first candidate's text parts, skipping any marked as model "thoughts".

        Raises:
            ValueError: If no text is present.
            KeyError, IndexError: If the response shape is unexpected (handled by the caller as a bad reply).
        """
        parts = body["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        if not text:
            raise ValueError("empty model text")
        return text

    def _key_rejected(self, response: httpx.Response) -> bool:
        """True for 401/403, and for the 400 Gemini returns when the key is invalid."""
        if response.status_code in {401, 403}:
            return True
        # An invalid key comes back as 400 with a message about the key.
        return response.status_code == 400 and "api key" in response.text.lower()

    def _quota_exhausted(self, response: httpx.Response) -> bool:
        """True if a 429 body names a per-day limit. Best effort, based on the error text."""
        # Best effort: a daily-quota 429 names a per-day limit in the body.
        text = response.text.lower()
        return "perday" in text or "per day" in text

    def _retry_delay(self, response: httpx.Response) -> float | None:
        """Seconds to wait: the ``Retry-After`` header, else Gemini's ``retryDelay`` (like ``"12s"``) in the error body."""
        header = _retry_after_header(response)
        if header is not None:
            return header
        try:
            details = response.json()["error"]["details"]
        except (ValueError, KeyError, TypeError):
            return None
        for item in details:
            delay = item.get("retryDelay") if isinstance(item, dict) else None
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return float(delay[:-1])
                except ValueError:
                    return None
        return None


class OpenAIClient(_HTTPClient):
    """OpenAI chat completions over HTTPS. Kept to show the interface is not tied to one provider."""

    provider = "openai"
    missing_key_hint = "Set OPENAI_API_KEY on the server to run the SimUser."

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        **kwargs: object,
    ) -> None:
        """Create an OpenAI client. Calls are always treated as billed (``free_tier=False``)."""
        super().__init__(api_key, base_url=base_url, free_tier=False, **kwargs)  # type: ignore[arg-type]

    def _request(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> tuple[str, dict[str, str], dict[str, object]]:
        """Build a ``chat/completions`` request with strict JSON-schema output and a Bearer token."""
        payload: dict[str, object] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "sim_user_turn",
                    "strict": True,
                    "schema": SIM_USER_TURN_SCHEMA,
                },
            },
        }
        if seed is not None:
            payload["seed"] = seed
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        return f"{self._base_url}/chat/completions", headers, payload

    def _usage(self, body: dict, requested_model: str) -> tuple[str, int, int]:
        """Read ``usage.prompt_tokens`` and ``usage.completion_tokens`` from the response."""
        usage = body.get("usage") or {}
        model_name = body.get("model")
        if not isinstance(model_name, str) or not model_name:
            model_name = requested_model
        return (
            model_name,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )

    def _text(self, body: dict) -> str:
        """Return the first choice's message content.

        Raises:
            TypeError: If the content is not a string (for example a refusal object).
        """
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("model content is not text")
        return content


def _price(model_name: str, requested_model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate the cost of one response.

    Prefers the model name the provider reported, falls back to the name that
    was requested, and returns ``None`` if neither is in the price table.
    """
    priced_as = model_name if has_price(model_name) else requested_model
    if not has_price(priced_as):
        return None
    return estimate_cost_usd(priced_as, input_tokens, output_tokens)


def _retry_after_header(response: httpx.Response) -> float | None:
    """Parse a numeric ``Retry-After`` header into seconds. Returns ``None`` if absent or not a number."""
    header = response.headers.get("retry-after")
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def provider_from_environment() -> str:
    """Read ``SIMBENCH_PROVIDER`` (default ``gemini``).

    Raises:
        ValueError: If the value is neither ``gemini`` nor ``openai``.

    Called by:
        ``model_from_environment``, ``missing_key_message``, ``client_from_environment``.
    """
    provider = os.environ.get("SIMBENCH_PROVIDER", "gemini").strip().lower()
    if provider not in {"gemini", "openai"}:
        raise ValueError("SIMBENCH_PROVIDER must be 'gemini' or 'openai'.")
    return provider


def model_from_environment() -> str:
    """Pick the model name: ``SIMBENCH_MODEL`` if set, else the default for the chosen provider.

    Called by:
        ``simbench_api.main`` when building the app.
    """
    configured = os.environ.get("SIMBENCH_MODEL", "").strip()
    if configured:
        return configured
    return DEFAULT_GEMINI_MODEL if provider_from_environment() == "gemini" else DEFAULT_OPENAI_MODEL


def missing_key_message() -> str:
    """Return the text shown when no API key is set, naming the right variable for the provider.

    Called by:
        ``create_app``, which returns it in the HTTP 503 detail.
    """
    name = "GEMINI_API_KEY" if provider_from_environment() == "gemini" else "OPENAI_API_KEY"
    return (
        f"Set {name} in .env and restart the server. "
        "The key stays on the server and is not sent to the browser."
    )


def client_from_environment() -> LLMClient | None:
    """Build the server's LLM client from environment variables.

    Precedence: ``SIMBENCH_REPLAY`` (a fixture file, no provider call), then the
    provider chosen by ``SIMBENCH_PROVIDER`` with its key. Keys are read on the
    server only and never sent to the browser.

    Returns:
        A ready client, or ``None`` when no key is configured. The API turns
        ``None`` into a 503 with setup instructions rather than failing at startup.

    Called by:
        ``simbench_api.main`` at import time.
    """

    from simbench.paths import repo_root

    replay = os.environ.get("SIMBENCH_REPLAY")
    if replay:
        path = Path(replay)
        if not path.is_absolute():
            path = repo_root() / path
        return ReplayLLMClient(path)

    if provider_from_environment() == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None
        free_tier = os.environ.get("SIMBENCH_GEMINI_FREE_TIER", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        return GeminiClient(api_key=api_key, free_tier=free_tier)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAIClient(api_key=api_key)
