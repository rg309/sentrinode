#!/usr/bin/env python3
"""Proxy entrypoint so hosting platforms can run `streamlit run app.py`."""
from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    root = Path(__file__).resolve().parent
    target = root / "services" / "python" / "app.py"
    if not target.exists():
        raise FileNotFoundError(f"Dashboard entrypoint missing at {target}")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":  # pragma: no cover - Streamlit executes as a script
    main()
