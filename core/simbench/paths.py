"""Repository paths. Resolved from this file so tests do not depend on the shell cwd."""

from pathlib import Path


def repo_root() -> Path:
    """Absolute path of the repository root (two levels above this package)."""
    return Path(__file__).resolve().parents[2]


def scenarios_dir() -> Path:
    """Folder holding scenario JSON files (``data/scenarios``). Used by ``load_scenario``."""
    return repo_root() / "data" / "scenarios"


def default_db_path() -> Path:
    """Default SQLite log location (``data/simbench.sqlite``, git-ignored). Used by the API."""
    return repo_root() / "data" / "simbench.sqlite"
