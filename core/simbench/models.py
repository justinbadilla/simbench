"""Transcript schema. Every run keeps its config and the raw messages."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from simbench.spec import SimUserSpec

# How the SimUser's knowledge reaches the model. Same spec, different presentation.
Variant = Literal["persona", "stateful"]


class Role(str, Enum):
    """Who sent a message. ``sim_user`` is the simulated customer; ``agent`` is the system under test."""

    sim_user = "sim_user"
    agent = "agent"


class TerminationReason(str, Enum):
    """Why a run stopped. Set once by the runner and stored on the transcript.

    ``script_complete``: the agent had no scripted line left.
    ``max_turns``: the message cap was reached while someone could still speak.
    ``sim_user_ended``: the model ended the chat on its own message.
    ``model_error``: the provider call failed; earlier messages are kept.
    """

    script_complete = "script_complete"
    max_turns = "max_turns"
    sim_user_ended = "sim_user_ended"
    model_error = "model_error"


class Usage(BaseModel):
    """Model accounting for one run.

    Scripted runs leave the model name empty and the counts at zero.
    ``requests`` counts HTTP requests including retries, because free-tier
    quotas are per request. ``estimated_cost_usd`` is a paid-tier list-price
    estimate. When ``free_tier`` is true nothing was billed and the estimate is
    what the same run would cost at paid rates. It is null when the model has
    no price in the local table. It is not an invoice.
    """

    model_name: str | None = None
    provider: str | None = None
    prompt_version: str | None = None
    seed: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    requests: int = 0
    free_tier: bool = False
    estimated_cost_usd: float | None = 0.0


class Message(BaseModel):
    """One utterance. ``turn_index`` is the zero-based position across both speakers."""

    turn_index: int = Field(ge=0)
    role: Role
    content: str = Field(min_length=1)


class StateEventKind(str, Enum):
    """The kinds of change the state tracker records. See ``SimUserState`` for when each fires."""

    unlocked = "unlocked"  # an agent message asked for an on-request fact
    revealed = "revealed"  # the customer said a fact
    early_reveal = "early_reveal"  # the customer said an on-request fact before it was asked for
    learned = "learned"  # the agent said a fact the customer did not know
    unknown_fact_used = "unknown_fact_used"  # the customer said an unknown fact before the agent did


class StateEvent(BaseModel):
    """One logged state change: what happened (``kind``), to which fact, and on which turn."""

    turn_index: int = Field(ge=0)
    kind: StateEventKind
    fact_id: str


class CheckResult(BaseModel):
    """One deterministic check. ``passed`` is null when the check does not apply."""

    name: str
    passed: bool | None
    value: float | None = None
    detail: str
    evidence: list[str] = Field(default_factory=list)


class ChecksReport(BaseModel):
    """All check results for a run, plus the rule-set version that produced them.

    ``all_passed`` is False only when some check failed; not-applicable checks do not count.
    """

    checks_version: str
    checks: list[CheckResult]
    all_passed: bool


class Transcript(BaseModel):
    """The complete record of one run, and the API's response body.

    Built by ``run_simulation`` or ``run_llm_simulation``, saved by
    ``save_transcript``, and rendered by the web UI. It carries the config,
    prompts, spec, state log, checks, raw messages, and usage, so a stored run
    can be understood and re-checked without the original scenario file.
    """

    run_id: str
    created_at: datetime
    scenario_id: str
    scenario_title: str
    notes: str
    sim_user_id: str
    agent_id: str
    script_version: str
    opening_role: Role
    max_turns: int = Field(ge=0)
    temperature: float | None = None
    # "scripted" for the V0 script; "persona" or "stateful" for an LLM SimUser.
    variant: str = "scripted"
    # The first model prompt. turn_prompts has every one, because the stateful
    # prompt changes as the state changes.
    system_prompt: str = ""
    turn_prompts: list[str] = Field(default_factory=list)
    # The spec the run used, so the transcript is self-describing.
    sim_user_spec: SimUserSpec | None = None
    state_log: list[StateEvent] = Field(default_factory=list)
    checks: ChecksReport | None = None
    messages: list[Message]
    termination_reason: TerminationReason
    error: str = ""
    usage: Usage


class RunRequest(BaseModel):
    """Request body for ``POST /runs``. Every field is optional; defaults give an LLM run of ``late_refund``."""

    scenario_id: str = "late_refund"
    max_turns: int | None = Field(default=None, ge=0)
    mode: Literal["llm", "scripted"] = "llm"
    variant: Variant = "stateful"
    seed: int | None = Field(default=None, ge=0)
