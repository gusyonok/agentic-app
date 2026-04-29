"""Stable launcher for local Streamlit execution."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    streamlit_home = root / ".streamlit_runtime"
    streamlit_home.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("STREAMLIT_HOME", str(streamlit_home))
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")

    cmd = ["streamlit", "run", "src/app/main.py", "--server.port", "8501", "--server.headless", "true"]
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
