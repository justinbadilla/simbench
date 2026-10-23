#!/usr/bin/env bash
# Start the API and the web UI. Ctrl-C stops both.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/uvicorn" ]]; then
  echo "Missing .venv. From the repo root: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi

if [[ ! -d "$ROOT/web/node_modules" ]]; then
  echo "Missing web/node_modules. From the repo root: npm install --prefix web" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "$API_PID" 2>/dev/null || true
  fi
  if [[ -n "${WEB_PID:-}" ]]; then
    kill "$WEB_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$ROOT/.venv/bin/uvicorn" simbench_api.main:app --reload --port 8000 &
API_PID=$!

npm --prefix "$ROOT/web" run dev -- --host 127.0.0.1 --port 5173 &
WEB_PID=$!

wait
