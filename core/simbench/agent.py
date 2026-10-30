"""The agent under test: the interface, the scripted agent, and agent version configs.

The runner only knows the ``Agent`` protocol. Today there are two kinds of
agent behind it: ``ScriptedAgent`` (fixed lines, no model) here, and
``LLMAgent`` (a model with tools) in ``llm_agent.py``. Later adapters, such as
a LangGraph agent or an HTTP endpoint, will implement the same protocol.

An agent only ever receives the conversation so far and the environment. It is
never given the SimUser's spec or state, so it cannot see what the customer has
been told to hide.
"""

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, field_validator

from simbench.environment import TOOLS, Environment
from simbench.models import Message, Usage
from simbench.paths import agents_dir


class AgentReply(BaseModel):
    """What an agent produced for one turn.

    ``message`` is the text the customer sees. It is None only when the agent
    ran out of tool steps without replying (``step_limit`` is then True), in
    which case the runner ends the run instead of adding a message.
    """

    message: str | None
    step_limit: bool = False


class Agent(Protocol):
    """The contract every agent satisfies.

    Attributes:
        id: Identifier stored on the transcript (``agent_c``, or the scenario's scripted id).
        prompt: The system prompt the agent runs on, logged on the transcript. Empty for scripted agents.
    """

    id: str
    prompt: str

    def exhausted(self) -> bool:
        """Return True if the agent has nothing left to say.

        A scripted agent is exhausted when its lines run out. A model agent never is.
        The runner asks this before the message cap so ``script_complete`` keeps its V0 meaning.
        """

    def respond(self, history: list[Message], env: Environment | None, *, seed: int) -> AgentReply:
        """Produce the agent's next message given the conversation so far.

        Args:
            history: Messages said so far, customer first. Nothing else about the customer is shared.
            env: The run's mock database, or None if the scenario has no environment.
            seed: The run's seed, chosen once by the runner. Model agents pass it to the provider.

        Raises:
            LLMError: If a model call fails after retries.
        """

    def usage(self, *, seed: int) -> Usage | None:
        """Return the agent's own model usage for the run, or None if it never used a model."""


class ScriptedAgent:
    """Plays a fixed list of lines in order. Does not read the customer or touch the environment.

    This is the V0 agent. It stays as an agent option because it is free, fast,
    and deterministic, which makes it the right baseline for tests.
    """

    prompt = ""

    def __init__(self, lines: list[str], agent_id: str = "scripted_support") -> None:
        """Remember the lines to play and the id to record on the transcript."""
        self.id = agent_id
        self._lines = list(lines)
        self._cursor = 0

    def exhausted(self) -> bool:
        """True once every line has been said."""
        return self._cursor >= len(self._lines)

    def respond(self, history: list[Message], env: Environment | None, *, seed: int) -> AgentReply:
        """Return the next scripted line. ``history``, ``env`` and ``seed`` are ignored.

        Raises:
            IndexError: If called when ``exhausted()`` is True. The runner never does this.
        """
        del history, env, seed
        line = self._lines[self._cursor]
        self._cursor += 1
        return AgentReply(message=line)

    def usage(self, *, seed: int) -> Usage | None:
        """A scripted agent uses no model, so there is no usage to report."""
        del seed
        return None


class AgentNotFoundError(LookupError):
    """Raised when no agent config exists for a requested id. The API maps it to HTTP 404."""

    def __init__(self, agent_id: str) -> None:
        """Remember the missing id on ``agent_id`` and build the error message."""
        self.agent_id = agent_id
        super().__init__(f"Unknown agent: {agent_id}")


class AgentConfig(BaseModel):
    """One agent version, as data.

    Agent versions A, B, C, D, and B-prime are files in ``data/agents``, not
    code, so changing or adding a version never touches the runner. The
    config says which prompt to run, which tools to allow, and whether the
    agent is shown the company policy.

    Attributes:
        id: Unique id, also the file name (``agent_c``).
        label: Short name for the UI.
        description: What is different about this version.
        prompt: Name of a prompt file in ``data/agents/prompts`` (without ``.md``).
        tools: Tool names this agent may call. Calls to any other tool are refused.
        include_policy: Whether the scenario's policy text is appended to the prompt.
        max_steps: Most tool-or-reply steps allowed in one turn before the run stops.
        temperature: Sampling temperature for the agent model.
        flaw: Empty for a normal agent. Otherwise a plain description of the flaw that was
            deliberately injected. It is documented here on purpose, so nothing is hidden.
    """

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    prompt: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    tools: list[str] = Field(default_factory=list)
    include_policy: bool = True
    max_steps: int = Field(default=4, ge=1, le=10)
    temperature: float = Field(default=0.2, ge=0, le=2)
    flaw: str = ""

    @field_validator("tools")
    @classmethod
    def tools_exist_once(cls, tools: list[str]) -> list[str]:
        """Pydantic hook: every tool must be a real tool and be listed only once."""
        unknown = sorted(set(tools) - set(TOOLS))
        if unknown:
            raise ValueError(f"unknown tools: {', '.join(unknown)}")
        if len(set(tools)) != len(tools):
            raise ValueError("tools must not repeat")
        return tools


def load_agent_config(agent_id: str, directory: Path | None = None) -> AgentConfig:
    """Read and validate an agent config by id.

    Args:
        agent_id: File stem under the agents directory (``agent_c``).
        directory: Override the folder; defaults to ``data/agents``. Tests use this.

    Raises:
        AgentNotFoundError: If the file does not exist.
        ValueError: If the file's ``id`` disagrees with its name, or validation fails.

    Called by:
        The API's ``create_run`` and ``list_agent_configs``.
    """
    directory = agents_dir() if directory is None else directory
    path = directory / f"{agent_id}.json"
    if not path.is_file():
        raise AgentNotFoundError(agent_id)
    config = AgentConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if config.id != agent_id:
        raise ValueError(f"Agent file {path.name} declares id {config.id!r}, expected {agent_id!r}")
    return config


def list_agent_configs(directory: Path | None = None) -> list[AgentConfig]:
    """Load every agent config in the folder, sorted by id. Used by ``GET /agents``."""
    directory = agents_dir() if directory is None else directory
    return [load_agent_config(path.stem, directory) for path in sorted(directory.glob("*.json"))]


def load_prompt(name: str, directory: Path | None = None) -> str:
    """Read an agent prompt file by name from ``prompts/``.

    Raises:
        FileNotFoundError: If the prompt file does not exist.

    Called by:
        ``build_agent_prompt`` in ``llm_agent.py``.
    """
    directory = (agents_dir() if directory is None else directory) / "prompts"
    return (directory / f"{name}.md").read_text(encoding="utf-8").strip()
