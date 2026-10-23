"""Scripted conversations, and V1 conversations where only the SimUser is a model."""

import secrets
from datetime import datetime, timezone
from uuid import uuid4

from simbench.llm import DEFAULT_GEMINI_MODEL, LLMCall, LLMClient, LLMError
from simbench.models import Message, Role, TerminationReason, Transcript, Usage
from simbench.scenario import Scenario
from simbench.sim_user import build_instructions, build_model_messages, prompt_version


def run_simulation(
    scenario: Scenario,
    *,
    max_turns: int | None = None,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> Transcript:
    """Run one scripted conversation.

    The opening role speaks first, then the roles alternate. Each side emits
    its next scripted line and does not read what the other said.

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

    No order database is updated.
    """

    instructions = build_instructions(scenario)
    limit = scenario.max_turns if max_turns is None else max_turns
    if limit < 0:
        raise ValueError("max_turns must be >= 0")
    chosen_seed = secrets.randbelow(2**31) if seed is None else seed
    chosen_temperature = scenario.temperature if temperature is None else temperature

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
            messages.append(
                Message(
                    turn_index=len(messages),
                    role=Role.agent,
                    content=scenario.agent_lines[agent_cursor],
                )
            )
            agent_cursor += 1
            role = Role.sim_user
            continue

        if len(messages) >= limit:
            reason = TerminationReason.max_turns
            break
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
        messages.append(
            Message(turn_index=len(messages), role=Role.sim_user, content=result.turn.message)
        )
        if result.turn.end_conversation:
            reason = TerminationReason.sim_user_ended
            break
        role = Role.agent

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
        system_prompt=instructions,
        messages=messages,
        termination_reason=reason,
        error=error,
        usage=totals.usage(model=model, prompt=prompt_version(scenario), seed=chosen_seed),
    )


class _CallTotals:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_ms = 0.0
        self.requests = 0
        self.costs: list[float | None] = []
        self.free_flags: list[bool] = []
        self.model_name: str | None = None
        self.provider: str | None = None

    def add_call(self, result: LLMCall) -> None:
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.latency_ms += result.latency_ms
        self.requests += result.requests
        self.costs.append(result.estimated_cost_usd)
        self.free_flags.append(result.free_tier)
        self.model_name = result.model_name
        self.provider = result.provider

    def add_error(self, error: LLMError) -> None:
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
