"""Scenario files are the V0 configuration. The runner does not hard-code their lines."""

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from simbench.models import Role
from simbench.paths import scenarios_dir


class ScenarioNotFoundError(LookupError):
    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id
        super().__init__(f"Unknown scenario: {scenario_id}")


class SimUserBrief(BaseModel):
    """Prompt inputs for an LLM SimUser. This is not a memory store."""

    goal: str = Field(min_length=1)
    known_facts: list[str] = Field(min_length=1)
    style: str = Field(min_length=1)
    done_when: str = Field(min_length=1)

    @field_validator("known_facts")
    @classmethod
    def facts_have_text(cls, facts: list[str]) -> list[str]:
        for fact in facts:
            if not fact.strip():
                raise ValueError("known facts must contain text")
        return facts


class Scenario(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str
    notes: str = ""
    opening_role: Role = Role.sim_user
    max_turns: int = Field(ge=0)
    temperature: float = Field(default=0.7, ge=0, le=2)
    sim_user_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    script_version: str = Field(min_length=1)
    sim_user: SimUserBrief | None = None
    sim_user_lines: list[str] = Field(default_factory=list)
    agent_lines: list[str] = Field(default_factory=list)

    @field_validator("sim_user_lines", "agent_lines")
    @classmethod
    def lines_are_non_empty(cls, lines: list[str]) -> list[str]:
        for line in lines:
            if not line.strip():
                raise ValueError("script lines must contain text")
        return lines

    def lines_for(self, role: Role) -> list[str]:
        if role is Role.sim_user:
            return self.sim_user_lines
        return self.agent_lines


def load_scenario(scenario_id: str, directory: Path | None = None) -> Scenario:
    directory = scenarios_dir() if directory is None else directory
    path = directory / f"{scenario_id}.json"
    if not path.is_file():
        raise ScenarioNotFoundError(scenario_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenario = Scenario.model_validate(payload)
    if scenario.id != scenario_id:
        raise ValueError(
            f"Scenario file {path.name} declares id {scenario.id!r}, expected {scenario_id!r}"
        )
    return scenario
