"""Scenario files are the V0 configuration. The runner does not hard-code their lines."""

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from simbench.models import Role
from simbench.paths import scenarios_dir
from simbench.spec import SimUserSpec
from simbench.world import EnvironmentSpec


class ScenarioNotFoundError(LookupError):
    """Raised when no scenario file exists for a requested id. The API maps it to HTTP 404."""

    def __init__(self, scenario_id: str) -> None:
        """Remember the missing id on ``scenario_id`` and build the error message."""
        self.scenario_id = scenario_id
        super().__init__(f"Unknown scenario: {scenario_id}")


class Scenario(BaseModel):
    """One configurable test case: who talks first, turn cap, SimUser spec, agent script, and environment.

    Loaded from ``data/scenarios/<id>.json``. Nothing about a specific scenario
    is hard-coded in the runner, so a new scenario is a new file.
    """

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
    sim_user: SimUserSpec | None = None
    # The mock orders database, policy, and expected outcome. Needed by agents that use tools.
    environment: EnvironmentSpec | None = None
    sim_user_lines: list[str] = Field(default_factory=list)
    agent_lines: list[str] = Field(default_factory=list)

    @field_validator("sim_user_lines", "agent_lines")
    @classmethod
    def lines_are_non_empty(cls, lines: list[str]) -> list[str]:
        """Pydantic hook: reject blank script lines, which would produce empty messages."""
        for line in lines:
            if not line.strip():
                raise ValueError("script lines must contain text")
        return lines

    def lines_for(self, role: Role) -> list[str]:
        """Return the scripted lines for a speaker.

        Called by:
            ``run_simulation`` (both roles) and ``run_llm_simulation`` (agent
            only, since the customer is a model there).
        """
        if role is Role.sim_user:
            return self.sim_user_lines
        return self.agent_lines


def load_scenario(scenario_id: str, directory: Path | None = None) -> Scenario:
    """Read and validate a scenario file by id.

    Args:
        scenario_id: File stem under the scenarios directory (``late_refund``).
        directory: Override the search folder; defaults to ``data/scenarios``.
            Tests use this to load fixtures.

    Returns:
        The validated ``Scenario``.

    Raises:
        ScenarioNotFoundError: If the file does not exist.
        ValueError: If the file's ``id`` field disagrees with its filename, or
            validation fails (``pydantic.ValidationError`` is a ``ValueError``).

    Called by:
        The API's ``create_run`` and the test suite.
    """
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


def list_scenarios(directory: Path | None = None) -> list[Scenario]:
    """Load every scenario file in the folder, sorted by id. Used by ``GET /scenarios``."""
    directory = scenarios_dir() if directory is None else directory
    return [load_scenario(path.stem, directory) for path in sorted(directory.glob("*.json"))]
