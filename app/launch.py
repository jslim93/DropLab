"""Console entry point for the DropLab sandbox.

Installed as ``droplab-app`` (see pyproject [project.scripts]); it launches the
Streamlit multipage app. Any extra CLI args are forwarded to ``streamlit run``
(e.g. ``droplab-app --server.port 8600``).

Intended for a local install (``pip install -e .[app]`` in the repo). Locates
``Home.py`` next to this module so it works from any working directory.
"""
import pathlib
import subprocess
import sys


def main():
    home = pathlib.Path(__file__).resolve().parent / "Home.py"
    if not home.exists():
        sys.exit(f"DropLab: could not find the app entry point at {home}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(home), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
