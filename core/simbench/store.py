"""SQLite log of raw transcripts. The API writes every run here."""

import sqlite3
from pathlib import Path

from simbench.models import Transcript


def init_db(path: Path) -> None:
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
    init_db(path)
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT transcript_json FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    return Transcript.model_validate_json(row[0])
