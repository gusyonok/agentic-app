import os
from pathlib import Path


def test_streamlit_runtime_defaults_are_writable():
    runtime_home = Path(".streamlit_runtime")
    runtime_home.mkdir(exist_ok=True)
    test_file = runtime_home / "write_test.txt"
    test_file.write_text("ok", encoding="utf-8")
    assert test_file.exists()

    os.environ.setdefault("STREAMLIT_HOME", str(runtime_home.resolve()))
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")

    assert os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"
    assert os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] == "poll"
