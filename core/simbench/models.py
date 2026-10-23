"""Transcript schema. Every run keeps its config and the raw messages."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Role(str, Enum):
    sim_user = "sim_user"
    agent = "agent"


class TerminationReason(str, Enum):
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
    turn_index: int = Field(ge=0)
    role: Role
    content: str = Field(min_length=1)


class Transcript(BaseModel):
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
    system_prompt: str = ""
    messages: list[Message]
    termination_reason: TerminationReason
    error: str = ""
    usage: Usage


class RunRequest(BaseModel):
    scenario_id: str = "late_refund"
    max_turns: int | None = Field(default=None, ge=0)
    mode: Literal["llm", "scripted"] = "llm"
    seed: int | None = Field(default=None, ge=0)
