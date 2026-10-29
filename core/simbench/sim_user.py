"""Prompts for the LLM SimUser.

Two variants are built from the same spec and differ only in how the customer's
knowledge reaches the model:

* ``persona`` puts every fact in the prompt and asks the model to hold some
  back. Discipline is up to the model.
* ``stateful`` puts in only what the state allows right now. A fact the
  customer has not been asked for is not in the prompt, so the model cannot
  repeat it. The prompt is rebuilt every turn from the state.

The header and tail are identical, so a comparison between the two is not
confounded by other wording. The rendered prompt for every turn is stored on
the transcript.
"""

from simbench.models import Message, Role, Variant
from simbench.scenario import Scenario
from simbench.spec import SimUserSpec
from simbench.state import SimUserState

PROMPT_TEMPLATE_VERSION = "sim_user.v2"


def prompt_version(scenario: Scenario, variant: Variant) -> str:
    """Build the identifier logged with a run, for example ``sim_user.v2.stateful+late_refund.v2``.

    It combines the template version, the variant, and the scenario's
    ``script_version``. Change the template version whenever the wording in this
    module changes, so stored runs stay comparable only to like runs.

    Called by:
        ``run_llm_simulation`` when it records ``Usage.prompt_version``.
    """
    return f"{PROMPT_TEMPLATE_VERSION}.{variant}+{scenario.script_version}"


def _header(spec: SimUserSpec) -> str:
    """Opening section shared by both variants: role, goal, and style."""
    return (
        "You are a simulated customer in a support chat. "
        "Write only the customer's next message.\n\n"
        f"Goal: {spec.goal.text}\n\n"
        f"Style: {spec.style}\n\n"
    )


def _unknowns(spec: SimUserSpec) -> str:
    """Section telling the model what it does not know, shared by both variants.

    Lists labels only, never markers, so the hidden values never reach the
    prompt. Returns an empty string when the spec has no unknown facts.
    """
    if not spec.unknown_facts:
        return ""
    lines = "\n".join(
        f"- You do not know {unknown.label} unless the agent tells you." for unknown in spec.unknown_facts
    )
    return f"Things you do not know:\n{lines}\n\n"


def _tail(spec: SimUserSpec) -> str:
    """Closing section shared by both variants: no inventing, when to end, and output rules."""
    return (
        "If the agent asks for something you have not been given, say you do not have it. "
        "Do not invent details.\n\n"
        f"When to end: {spec.done_when}\n"
        "Set end_conversation to true only on the message that ends the chat. "
        "That message is still something the customer would say.\n\n"
        "Do not write the agent's lines. Do not mention that you are a simulation."
    )


def _persona_knowledge(spec: SimUserSpec) -> str:
    """Knowledge section for the ``persona`` variant.

    Lists every fact. On-request facts carry an instruction to hold them back, so
    the model alone is responsible for that discipline. The output does not
    depend on conversation state, so it is identical on every turn.
    """
    lines = []
    for fact in spec.facts:
        if fact.disclosure == "on_request":
            lines.append(f"- {fact.text} Do not mention this until the agent asks for it.")
        else:
            lines.append(f"- {fact.text}")
    return "Facts you know:\n" + "\n".join(lines) + "\n\n"


def _stateful_knowledge(state: SimUserState) -> str:
    """Knowledge section for the ``stateful`` variant, built from the current state.

    Includes only facts the customer may say now (annotated when the agent asked
    for them) and, separately, facts already said so the model does not repeat
    them. Facts that are still hidden do not appear at all.
    """
    shareable = state.shareable_facts()
    if shareable:
        lines = [
            f"- {fact.text}" + (" (the agent asked for this)" if state.asked_for(fact) else "")
            for fact in shareable
        ]
    else:
        lines = ["- Nothing new right now."]
    text = "Facts you may share now:\n" + "\n".join(lines) + "\n\n"
    told = state.revealed_facts()
    if told:
        text += (
            "Already told the agent (do not repeat unless asked):\n"
            + "\n".join(f"- {fact.text}" for fact in told)
            + "\n\n"
        )
    return text


def build_instructions(scenario: Scenario, variant: Variant, state: SimUserState | None = None) -> str:
    """Render the system prompt for the next customer message.

    Layout: header, variant-specific knowledge, unknowns, tail. Only the
    knowledge section differs between variants.

    Args:
        scenario: Must have a ``sim_user`` spec.
        variant: ``"persona"`` or ``"stateful"``.
        state: Required for ``stateful`` (rebuild the prompt from it each turn);
            ignored by ``persona``.

    Returns:
        The full instruction text, stored verbatim in ``Transcript.turn_prompts``.

    Raises:
        ValueError: If the scenario has no SimUser spec, or ``stateful`` is
            requested without a state.

    Called by:
        ``run_llm_simulation`` before every model call.
    """
    if scenario.sim_user is None:
        raise ValueError(f"Scenario {scenario.id} has no SimUser brief.")
    spec = scenario.sim_user
    if variant == "stateful":
        if state is None:
            raise ValueError("The stateful variant needs a SimUserState.")
        knowledge = _stateful_knowledge(state)
    else:
        knowledge = _persona_knowledge(spec)
    return _header(spec) + knowledge + _unknowns(spec) + _tail(spec)


def build_model_messages(instructions: str, history: list[Message]) -> list[dict[str, str]]:
    """Package the prompt and the conversation so far as chat messages for an ``LLMClient``.

    The history is flattened into one user message ("Customer: ...", "Agent: ..."),
    rather than alternating roles, so the model always sees a single, explicit
    transcript and is asked for exactly one next customer message.

    Args:
        instructions: Output of ``build_instructions``.
        history: Messages said so far; may be empty on the opening turn.

    Returns:
        A two-item list: a ``system`` message and a ``user`` message.

    Called by:
        ``run_llm_simulation`` on each customer turn.
    """
    if history:
        lines = []
        for message in history:
            speaker = "Customer" if message.role is Role.sim_user else "Agent"
            lines.append(f"{speaker}: {message.content}")
        transcript = "\n".join(lines)
    else:
        transcript = "(no messages yet)"
    return [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{transcript}\n\n"
                "Write the customer's next message as JSON."
            ),
        },
    ]
