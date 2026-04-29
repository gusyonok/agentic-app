#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export STREAMLIT_HOME="${ROOT_DIR}/.streamlit_runtime"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="poll"

mkdir -p "${STREAMLIT_HOME}"

PORT="${PORT:-8501}"

if lsof -ti "tcp:${PORT}" >/dev/null 2>&1; then
  echo "Port ${PORT} is busy. Stopping existing process(es)..."
  pids="$(lsof -ti "tcp:${PORT}" | tr '\n' ' ')"
  # Try graceful shutdown first, then force kill if needed.
  kill ${pids} >/dev/null 2>&1 || true
  sleep 1
  if lsof -ti "tcp:${PORT}" >/dev/null 2>&1; then
    kill -9 ${pids} >/dev/null 2>&1 || true
  fi
fi

if [[ ! -x "${ROOT_DIR}/.venv/bin/streamlit" ]]; then
  echo "Missing .venv or streamlit. Install dependencies first:"
  echo "  python3 -m venv .venv && .venv/bin/python -m pip install -e \".[dev]\""
  exit 1
fi

exec "${ROOT_DIR}/.venv/bin/streamlit" run src/app/main.py --server.port "${PORT}" --server.headless true
