"""Prompt for the V1 SimUser.

The brief is pasted into the system prompt on every turn. Nothing here records
which facts have been said or withheld. That split is later work.

``done_when`` changes when the customer stops, which changes conversation
length. The rendered prompt is stored on the transcript.
"""

from simbench.models import Message, Role
from simbench.scenario import Scenario

PROMPT_TEMPLATE_VERSION = "sim_user.v1"


def prompt_version(scenario: Scenario) -> str:
    return f"{PROMPT_TEMPLATE_VERSION}+{scenario.script_version}"


def build_instructions(scenario: Scenario) -> str:
    if scenario.sim_user is None:
        raise ValueError(f"Scenario {scenario.id} has no SimUser brief.")
    brief = scenario.sim_user
    facts = "\n".join(f"- {fact}" for fact in brief.known_facts)
    return (
        "You are a simulated customer in a support chat. "
        "Write only the customer's next message.\n\n"
        f"Goal: {brief.goal}\n\n"
        f"Facts you know:\n{facts}\n\n"
        f"Style: {brief.style}\n\n"
        f"When to end: {brief.done_when}\n"
        "Set end_conversation to true only on the message that ends the chat. "
        "That message is still something the customer would say.\n\n"
        "Do not write the agent's lines. Do not mention that you are a simulation."
    )


def build_model_messages(instructions: str, history: list[Message]) -> list[dict[str, str]]:
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
