"""Repository paths. Resolved from this file so tests do not depend on the shell cwd."""

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def scenarios_dir() -> Path:
    return repo_root() / "data" / "scenarios"


def default_db_path() -> Path:
    return repo_root() / "data" / "simbench.sqlite"
