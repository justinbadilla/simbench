"""Scripted conversations, and conversations where only the SimUser is a model."""

import secrets
from datetime import datetime, timezone
from uuid import uuid4

from simbench.agent import Agent, ScriptedAgent
from simbench.checks import evaluate
from simbench.environment import Environment
from simbench.llm import DEFAULT_GEMINI_MODEL, LLMClient, LLMError
from simbench.models import Message, Role, TerminationReason, Transcript, Usage, Variant
from simbench.outcome import evaluate_outcome
from simbench.scenario import Scenario
from simbench.sim_user import build_instructions, build_model_messages, prompt_version
from simbench.state import SimUserState
from simbench.usage import CallTotals


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
    agent: Agent | None = None,
    max_turns: int | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    variant: Variant = "stateful",
    temperature: float | None = None,
    seed: int | None = None,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> Transcript:
    """Run one conversation: an LLM SimUser against an agent.

    The agent is whatever implements ``Agent``. By default it is the scenario's
    scripted lines, which do not read the customer. With an ``LLMAgent`` it is a
    model that reads the conversation and can call tools in the scenario's
    mock environment. The environment is rebuilt from the scenario for every
    run, so runs never share state.

    Stop for ``sim_user_ended`` when the model sets end_conversation. Stop for
    ``script_complete`` when the agent has nothing left to say. Stop for
    ``max_turns`` when the next speaker could still talk but the cap is full.
    Stop for ``step_limit`` when the agent used every tool step in a turn
    without replying. Stop for ``model_error`` when either model call fails,
    and keep the messages already said.

    The harness tracks SimUser state from the messages as they happen. With the
    ``stateful`` variant that state decides what the prompt contains each turn.
    With ``persona`` the state is still tracked, but only so the checks and the
    state log exist for both variants. Checks run on the finished transcript,
    except after a model error, where a cut-off run would only show an outage
    as missing facts.

    No order database is updated.

    Args:
        scenario: Must have a ``sim_user`` spec. Supplies the agent script and the environment.
        client: The SimUser's ``LLMClient`` (Gemini, OpenAI, mock, or replay). Injected so
            tests run offline.
        agent: The agent under test. Defaults to ``ScriptedAgent`` playing the scenario's lines.
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
        spec, the state log, the SimUser checks, the tool calls, the final
        database state, the task outcome (checks and outcome are ``None`` after a
        model error), the termination reason, any error text, and ``Usage`` for
        the SimUser and for the agent separately.

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
    role = scenario.opening_role
    totals = CallTotals()
    reason = TerminationReason.max_turns
    error = ""
    agent = agent if agent is not None else ScriptedAgent(scenario.agent_lines, scenario.agent_id)
    env = Environment(scenario.environment) if scenario.environment is not None else None

    while True:
        if role is Role.agent:
            if agent.exhausted():
                reason = TerminationReason.script_complete
                break
            if len(messages) >= limit:
                reason = TerminationReason.max_turns
                break
            try:
                # The agent gets a copy of the conversation so it cannot edit the record.
                reply = agent.respond(list(messages), env, seed=chosen_seed)
            except LLMError as exc:
                reason = TerminationReason.model_error
                error = str(exc)
                break
            if reply.message is None:
                reason = TerminationReason.step_limit
                error = "The agent used every tool step in one turn without replying."
                break
            agent_message = Message(turn_index=len(messages), role=Role.agent, content=reply.message)
            messages.append(agent_message)
            state.observe(agent_message)
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
    finished = reason is not TerminationReason.model_error
    checks = evaluate(spec, messages) if finished else None
    outcome = None
    if env is not None and scenario.environment is not None and finished:
        outcome = evaluate_outcome(scenario.environment.expected_outcome, env.snapshot())

    return Transcript(
        run_id=run_id or str(uuid4()),
        created_at=created_at or datetime.now(timezone.utc),
        scenario_id=scenario.id,
        scenario_title=scenario.title,
        notes=scenario.notes,
        sim_user_id=scenario.sim_user_id,
        agent_id=agent.id,
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
        agent_prompt=agent.prompt,
        agent_usage=agent.usage(seed=chosen_seed),
        environment_spec=scenario.environment,
        tool_calls=list(env.calls) if env is not None else [],
        final_state=env.snapshot() if env is not None else None,
        outcome=outcome,
        messages=messages,
        termination_reason=reason,
        error=error,
        usage=totals.usage(model=model, prompt=prompt_version(scenario, variant), seed=chosen_seed),
    )
