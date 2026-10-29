"""Scripted conversations, and conversations where only the SimUser is a model."""

import secrets
from datetime import datetime, timezone
from uuid import uuid4

from simbench.checks import evaluate
from simbench.llm import DEFAULT_GEMINI_MODEL, LLMCall, LLMClient, LLMError
from simbench.models import Message, Role, TerminationReason, Transcript, Usage, Variant
from simbench.scenario import Scenario
from simbench.sim_user import build_instructions, build_model_messages, prompt_version
from simbench.state import SimUserState


def run_simulation(
    scenario: Scenario,
    *,
    max_turns: int | None = None,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> Transcript:
    """Run one scripted conversation (V0 mode, no model involved).

    The opening role speaks first, then the roles alternate. Each side emits
    its next scripted line and does not read what the other said.

    Args:
        scenario: Supplies both scripts, the opener, and the default cap.
        max_turns: Override the scenario's message cap. Must be >= 0.
        run_id: Fixed id for tests; a UUID is generated when omitted.
        created_at: Fixed timestamp for tests; defaults to now (UTC).

    Returns:
        A ``Transcript`` with ``variant="scripted"``, no checks, and a ``Usage``
        of zeros.

    Raises:
        ValueError: If ``max_turns`` is negative.

    Called by:
        The API's ``create_run`` when ``mode == "scripted"``, and the V0 tests.

    Stop for ``script_complete`` when the speaker whose turn it is has no line
    left, including when the message cap has also been reached. Stop for
    ``max_turns`` only when that speaker still has a line and the transcript
    is already at the cap.

    The returned transcript is the whole record of the run. V0 does not score
    it and does not change an order database.
    """

    limit = scenario.max_turns if max_turns is None else max_turns
    if limit < 0:
        raise ValueError("max_turns must be >= 0")

    cursors = {Role.sim_user: 0, Role.agent: 0}
    role = scenario.opening_role
    messages: list[Message] = []

    while True:
        lines = scenario.lines_for(role)
        cursor = cursors[role]
        if cursor >= len(lines):
            reason = TerminationReason.script_complete
            break
        if len(messages) >= limit:
            reason = TerminationReason.max_turns
            break
        messages.append(
            Message(turn_index=len(messages), role=role, content=lines[cursor])
        )
        cursors[role] = cursor + 1
        role = Role.agent if role is Role.sim_user else Role.sim_user

    return Transcript(
        run_id=run_id or str(uuid4()),
        created_at=created_at or datetime.now(timezone.utc),
        scenario_id=scenario.id,
        scenario_title=scenario.title,
        notes=scenario.notes,
        sim_user_id=scenario.sim_user_id,
        agent_id=scenario.agent_id,
        script_version=scenario.script_version,
        opening_role=scenario.opening_role,
        max_turns=limit,
        messages=messages,
        termination_reason=reason,
        usage=Usage(prompt_version=scenario.script_version),
    )


def run_llm_simulation(
    scenario: Scenario,
    client: LLMClient,
    *,
    max_turns: int | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    variant: Variant = "stateful",
    temperature: float | None = None,
    seed: int | None = None,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> Transcript:
    """Run one conversation: an LLM SimUser against the scenario's scripted agent.

    The agent still speaks its next scripted line and does not read the customer.
    Stop for ``sim_user_ended`` when the model sets end_conversation. Stop for
    ``script_complete`` when the agent has no line left. Stop for ``max_turns``
    when the next speaker could still talk but the cap is full. Stop for
    ``model_error`` when the client fails, and keep the messages already said.

    The harness tracks SimUser state from the messages as they happen. With the
    ``stateful`` variant that state decides what the prompt contains each turn.
    With ``persona`` the state is still tracked, but only so the checks and the
    state log exist for both variants. Checks run on the finished transcript,
    except after a model error, where a cut-off run would only show an outage
    as missing facts.

    No order database is updated.

    Args:
        scenario: Must have a ``sim_user`` spec. Supplies the agent script.
        client: Any ``LLMClient`` (Gemini, OpenAI, mock, or replay). Injected so
            tests run offline.
        max_turns: Override the scenario's message cap. Must be >= 0.
        model: Model name sent to the client and used for cost lookup.
        variant: ``"stateful"`` or ``"persona"`` prompt style.
        temperature: Override the scenario's sampling temperature.
        seed: Seed passed to the provider; a random one is chosen and logged
            when omitted, so every run records the seed it used.
        run_id: Fixed id for tests; a UUID is generated when omitted.
        created_at: Fixed timestamp for tests; defaults to now (UTC).

    Returns:
        A ``Transcript`` holding the messages, the prompt for every turn, the
        spec, the state log, the checks (or ``None`` after a model error), the
        termination reason, any error text, and aggregated ``Usage``.

    Raises:
        ValueError: If the scenario has no SimUser spec or ``max_turns`` is negative.
            Model failures do not raise; they end the run with ``model_error``.

    Called by:
        The API's ``create_run`` when ``mode == "llm"``, and the LLM run tests.
    """

    if scenario.sim_user is None:
        raise ValueError(f"Scenario {scenario.id} has no SimUser brief.")
    spec = scenario.sim_user
    limit = scenario.max_turns if max_turns is None else max_turns
    if limit < 0:
        raise ValueError("max_turns must be >= 0")
    chosen_seed = secrets.randbelow(2**31) if seed is None else seed
    chosen_temperature = scenario.temperature if temperature is None else temperature

    state = SimUserState(spec)
    turn_prompts: list[str] = []
    messages: list[Message] = []
    agent_cursor = 0
    role = scenario.opening_role
    totals = _CallTotals()
    reason = TerminationReason.max_turns
    error = ""

    while True:
        if role is Role.agent:
            if agent_cursor >= len(scenario.agent_lines):
                reason = TerminationReason.script_complete
                break
            if len(messages) >= limit:
                reason = TerminationReason.max_turns
                break
            agent_message = Message(
                turn_index=len(messages),
                role=Role.agent,
                content=scenario.agent_lines[agent_cursor],
            )
            messages.append(agent_message)
            state.observe(agent_message)
            agent_cursor += 1
            role = Role.sim_user
            continue

        if len(messages) >= limit:
            reason = TerminationReason.max_turns
            break
        instructions = build_instructions(scenario, variant, state)
        turn_prompts.append(instructions)
        try:
            result = client.complete(
                messages=build_model_messages(instructions, messages),
                model=model,
                temperature=chosen_temperature,
                seed=chosen_seed,
            )
        except LLMError as exc:
            totals.add_error(exc)
            reason = TerminationReason.model_error
            error = str(exc)
            break
        totals.add_call(result)
        customer_message = Message(
            turn_index=len(messages), role=Role.sim_user, content=result.turn.message
        )
        messages.append(customer_message)
        state.observe(customer_message)
        if result.turn.end_conversation:
            reason = TerminationReason.sim_user_ended
            break
        role = Role.agent

    # With no model call (a zero cap), show the prompt the first turn would have used.
    first_prompt = turn_prompts[0] if turn_prompts else build_instructions(scenario, variant, state)
    checks = None if reason is TerminationReason.model_error else evaluate(spec, messages)

    return Transcript(
        run_id=run_id or str(uuid4()),
        created_at=created_at or datetime.now(timezone.utc),
        scenario_id=scenario.id,
        scenario_title=scenario.title,
        notes=scenario.notes,
        sim_user_id=scenario.sim_user_id,
        agent_id=scenario.agent_id,
        script_version=scenario.script_version,
        opening_role=scenario.opening_role,
        max_turns=limit,
        temperature=chosen_temperature,
        variant=variant,
        system_prompt=first_prompt,
        turn_prompts=turn_prompts,
        sim_user_spec=spec,
        state_log=state.events,
        checks=checks,
        messages=messages,
        termination_reason=reason,
        error=error,
        usage=totals.usage(model=model, prompt=prompt_version(scenario, variant), seed=chosen_seed),
    )


class _CallTotals:
    """Running totals of model usage across all calls in one run.

    Collects tokens, latency, request count, per-call costs, and free-tier
    flags from successful calls and failed ones alike, then folds them into one
    ``Usage`` record. Failed attempts are included because they still consume
    quota and time.
    """

    def __init__(self) -> None:
        """Start with all counters at zero and no model or provider known yet."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_ms = 0.0
        self.requests = 0
        self.costs: list[float | None] = []
        self.free_flags: list[bool] = []
        self.model_name: str | None = None
        self.provider: str | None = None

    def add_call(self, result: LLMCall) -> None:
        """Add the usage of one successful model call and remember its model and provider."""
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.latency_ms += result.latency_ms
        self.requests += result.requests
        self.costs.append(result.estimated_cost_usd)
        self.free_flags.append(result.free_tier)
        self.model_name = result.model_name
        self.provider = result.provider

    def add_error(self, error: LLMError) -> None:
        """Add the usage a failed call still incurred (retries, partial tokens).

        A cost or free-tier flag is only recorded if the failure actually spent
        tokens or sent a request, so an error before any request does not skew
        the totals.
        """
        self.input_tokens += error.input_tokens
        self.output_tokens += error.output_tokens
        self.latency_ms += error.latency_ms
        self.requests += error.requests
        if error.input_tokens or error.output_tokens:
            self.costs.append(error.estimated_cost_usd)
        if error.requests:
            self.free_flags.append(error.free_tier)
        if error.model_name:
            self.model_name = error.model_name

    def usage(self, *, model: str, prompt: str, seed: int) -> Usage:
        """Fold the totals into the ``Usage`` stored on the transcript.

        Cost rules: no priced calls gives 0.0; if any call had an unknown price
        the total is ``None`` rather than a partial sum. ``free_tier`` is true
        only if at least one call happened and every call was free.

        Args:
            model: Fallback name when no call reported one.
            prompt: The prompt version string (see ``prompt_version``).
            seed: The seed used for the run.
        """
        if not self.costs:
            cost: float | None = 0.0
        elif any(item is None for item in self.costs):
            cost = None
        else:
            cost = round(sum(item for item in self.costs if item is not None), 8)
        return Usage(
            model_name=self.model_name or model,
            provider=self.provider,
            prompt_version=prompt,
            seed=seed,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=round(self.latency_ms, 1),
            requests=self.requests,
            free_tier=bool(self.free_flags) and all(self.free_flags),
            estimated_cost_usd=cost,
        )
