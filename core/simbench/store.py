"""SQLite log of raw transcripts. The API writes every run here."""

import sqlite3
from pathlib import Path

from simbench.models import Transcript


def init_db(path: Path) -> None:
    """Create the database file, its folder, and the ``runs`` table if missing.

    Safe to call repeatedly. The table stores the full transcript as JSON, with
    a few columns (id, time, scenario) kept separate for easy querying.

    Called by:
        ``save_transcript`` and ``load_transcript`` before every access.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                transcript_json TEXT NOT NULL
            )
            """
        )


def save_transcript(path: Path, transcript: Transcript) -> None:
    """Append one transcript to the log as raw JSON.

    Args:
        path: SQLite file to write.
        transcript: The finished run. Failed runs are saved too.

    Raises:
        sqlite3.IntegrityError: If a transcript with the same ``run_id`` exists.
            Runs are never overwritten.

    Called by:
        The API's ``create_run`` after every run.
    """
    init_db(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO runs (run_id, created_at, scenario_id, transcript_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                transcript.run_id,
                transcript.created_at.isoformat(),
                transcript.scenario_id,
                transcript.model_dump_json(),
            ),
        )


def load_transcript(path: Path, run_id: str) -> Transcript | None:
    """Read one stored transcript back by run id.

    Returns:
        The validated ``Transcript``, or ``None`` if no such run exists.

    Called by:
        The tests, to confirm what was stored. No endpoint reads runs back yet.
    """
    init_db(path)
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT transcript_json FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    return Transcript.model_validate_json(row[0])
