"""Raw transcripts round-trip through SQLite."""

from simbench.models import Role
from simbench.run import run_simulation
from simbench.scenario import Scenario
from simbench.store import load_transcript, save_transcript


def test_save_and_load_keep_the_raw_transcript(tmp_path) -> None:
    scenario = Scenario(
        id="fixture",
        title="Fixture",
        description="Test scenario",
        notes="Keep this text.",
        opening_role=Role.sim_user,
        max_turns=4,
        sim_user_id="user",
        agent_id="agent",
        script_version="fixture.v0",
        sim_user_lines=["hello"],
        agent_lines=["hi"],
    )
    transcript = run_simulation(scenario, run_id="run-1")
    path = tmp_path / "nested" / "runs.sqlite"

    save_transcript(path, transcript)
    loaded = load_transcript(path, "run-1")

    assert loaded is not None
    assert loaded.model_dump() == transcript.model_dump()
    assert load_transcript(path, "missing") is None
