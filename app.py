#!/usr/bin/env python3
"""Proxy entrypoint so hosting platforms can run `streamlit run app.py`."""
from __future__ import annotations

from pathlib import Path
import os
import runpy
import sys


def main() -> None:
    root = Path(__file__).resolve().parent
    target = root / "services" / "python" / "app.py"
    if not target.exists():
        raise FileNotFoundError(f"Dashboard entrypoint missing at {target}")
    # Ensure the dashboard package directory is importable and mimic local cwd.
    services_dir = target.parent
    if str(services_dir) not in sys.path:
        sys.path.insert(0, str(services_dir))
    os.chdir(services_dir)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":  # pragma: no cover - Streamlit executes as a script
    main()
