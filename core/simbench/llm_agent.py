"""An agent driven by a model, with tools, run as a hand-written loop.

Each time the customer speaks, the agent loops: ask the model for a step, and if
the step is a tool call, run it in the environment and show the model the
result, then ask again. The turn ends when the model replies to the customer or
runs out of steps. The loop is written out here instead of using a framework so
every step can be read, logged, and tested.

The agent sees the conversation and the tool results. It is never given the
SimUser's spec or state. ``build_agent_prompt`` takes only an agent config and
the environment spec, so there is no path for the customer's hidden facts to
reach it.
"""

import json
from pathlib import Path

from simbench.agent import AgentConfig, AgentReply, load_prompt
from simbench.environment import TOOLS, Environment, ToolResult
from simbench.llm import AgentLLMClient, LLMError
from simbench.models import Message, Role, Usage
from simbench.usage import CallTotals
from simbench.world import EnvironmentSpec

AGENT_TEMPLATE_VERSION = "agent.v1"

_PROTOCOL = """\
How to respond:
Every step, answer with JSON containing the fields action, tool, arguments_json, and message.
- To use a tool: set action to "tool_call", tool to the tool name, arguments_json to the arguments as a JSON object written as text, and message to "".
- To reply to the customer: set action to "reply", tool to "", arguments_json to "{}", and message to what you want to say.
After a tool call you will be shown its result, and then you choose your next step.
Write only the agent's side. Never write the customer's lines."""


def build_agent_prompt(
    config: AgentConfig,
    env_spec: EnvironmentSpec | None,
    agents_directory: Path | None = None,
) -> str:
    """Assemble the system prompt for an agent version.

    The prompt is the version's own prompt text, then (if the config says so)
    the company policy, then a description of the tools it may use, then the
    shared step protocol. Everything that differs between versions comes from
    the config, so two configs with the same prompt, policy flag, and tools
    produce byte-identical prompts. That is what makes B-prime a true copy of B.

    Args:
        config: The agent version.
        env_spec: The scenario's environment, which supplies the policy text. May be None.
        agents_directory: Override the ``data/agents`` folder the prompt file is read from (tests).

    Returns:
        The full prompt, stored verbatim on the transcript as ``agent_prompt``.

    Called by:
        ``LLMAgent.__init__``.
    """
    parts = [load_prompt(config.prompt, agents_directory)]
    if config.include_policy and env_spec is not None:
        parts.append(f"Company policy:\n{env_spec.policy_text}")
    if config.tools:
        lines = []
        for name in config.tools:
            tool = TOOLS[name]
            arguments = [f"{key} ({text})" for key, text in tool.required.items()]
            arguments += [f"{key} (optional: {text})" for key, text in tool.optional.items()]
            lines.append(f"- {name}: {tool.description} Arguments: {'; '.join(arguments)}")
        parts.append("Tools you can use:\n" + "\n".join(lines))
    else:
        parts.append("You have no tools.")
    parts.append(_PROTOCOL)
    return "\n\n".join(parts)


def _render_history(history: list[Message]) -> str:
    """Flatten the conversation into ``Customer: ...`` / ``Agent: ...`` lines."""
    if not history:
        return "(no messages yet)"
    lines = []
    for message in history:
        speaker = "Customer" if message.role is Role.sim_user else "Agent"
        lines.append(f"{speaker}: {message.content}")
    return "\n".join(lines)


def _render_result(result: ToolResult) -> str:
    """Format a tool result as the JSON text the model is shown."""
    if result.ok:
        return json.dumps({"ok": True, "data": result.data}, sort_keys=True)
    return json.dumps({"ok": False, "error": result.error})


def _parse_arguments(text: str) -> object:
    """Parse the model's ``arguments_json`` text.

    Returns the parsed value. If the text is not valid JSON, returns the raw
    text itself, which the environment then refuses as "arguments must be a
    JSON object" and logs, so the model gets feedback and the attempt is visible.
    """
    try:
        return json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        return text


class LLMAgent:
    """A model-driven support agent that can call tools in the mock environment.

    One instance is created per run, so its usage totals cover exactly that run.
    """

    def __init__(
        self,
        config: AgentConfig,
        client: AgentLLMClient,
        env_spec: EnvironmentSpec | None,
        *,
        model: str,
        agents_directory: Path | None = None,
    ) -> None:
        """Build the agent and its prompt.

        Args:
            config: Which agent version this is.
            client: The model behind the agent (a real provider or a mock).
            env_spec: The scenario's environment, for the policy text.
            model: Model name sent to the client.
            agents_directory: Override the ``data/agents`` folder prompts are read from (tests).

        Raises:
            ValueError: If the config allows tools but the scenario has no environment,
                since the tools would have nothing to act on.
        """
        if config.tools and env_spec is None:
            raise ValueError(f"Agent {config.id} uses tools but the scenario has no environment.")
        self.id = config.id
        self._config = config
        self._client = client
        self._model = model
        self.prompt = build_agent_prompt(config, env_spec, agents_directory)
        self._totals = CallTotals()

    def exhausted(self) -> bool:
        """A model agent can always say more, so this is always False."""
        return False

    def respond(self, history: list[Message], env: Environment | None, *, seed: int) -> AgentReply:
        """Run the tool loop for one customer turn and return the agent's reply.

        Each loop step asks the model for an action. A reply ends the turn. A
        tool call is executed in ``env`` (refused calls included, so the model
        can correct itself), its result is added to what the model sees, and the
        loop continues. If ``max_steps`` steps pass without a reply, the turn
        ends with ``step_limit=True`` and no message.

        Args:
            history: Messages said so far, ending with the customer's latest.
            env: The run's database, used to execute tool calls.
            seed: The run's seed, passed to the provider on every model call.

        Returns:
            An ``AgentReply``.

        Raises:
            LLMError: If a model call fails. Its usage is added to this agent's
                totals first, and tool calls already made stay in ``env.calls``.
        """
        turn_index = len(history)
        conversation = _render_history(history)
        activity: list[str] = []
        for step in range(self._config.max_steps):
            user_text = f"Conversation so far:\n{conversation}\n\n"
            if activity:
                user_text += "What you have done this turn:\n" + "\n".join(activity) + "\n\n"
            user_text += "Write your next step as JSON."
            try:
                call = self._client.complete_agent(
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": user_text},
                    ],
                    model=self._model,
                    temperature=self._config.temperature,
                    seed=seed,
                )
            except LLMError as error:
                self._totals.add_error(error)
                raise
            self._totals.add_call(call)

            action = call.action
            if action.action == "reply":
                return AgentReply(message=action.message)

            arguments = _parse_arguments(action.arguments_json)
            if env is None:
                result = ToolResult(ok=False, error="tool_not_permitted: no tools are available.")
            else:
                result = env.call(
                    action.tool,
                    arguments,
                    allowed=self._config.tools,
                    turn_index=turn_index,
                    step=step,
                )
            activity.append(f"You called {action.tool} with {action.arguments_json}")
            activity.append(f"Result: {_render_result(result)}")
        return AgentReply(message=None, step_limit=True)

    def usage(self, *, seed: int) -> Usage | None:
        """Return the model usage for this agent across the whole run.

        The prompt version names the prompt file, so two configs that share a
        prompt (B and B-prime) report the same version.
        """
        return self._totals.usage(
            model=self._model,
            prompt=f"{AGENT_TEMPLATE_VERSION}.{self._config.prompt}",
            seed=seed,
        )
