"""Load a local .env file without overriding variables already set in the shell."""

import os

from simbench.paths import repo_root


def load_local_env() -> None:
    """Read ``.env`` at the repo root into ``os.environ``.

    Blank lines, comments, and lines without ``=`` are skipped, and surrounding
    quotes are stripped from values. Variables already set in the shell win, so
    an exported value overrides the file. Does nothing if ``.env`` is missing.
    Values are never printed or logged.

    Called by:
        ``simbench_api.main`` at import time, before the LLM client is built.
    """
    path = repo_root() / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
