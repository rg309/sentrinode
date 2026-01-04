#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def timestamped_artifact_path(prefix: str, directory: str | Path = ".") -> Path:
    return Path(directory) / f"{prefix}_{_timestamp()}.pkl"


def save_artifact(obj: Any, prefix: str, directory: str | Path = ".") -> Path:
    path = timestamped_artifact_path(prefix, directory)
    joblib.dump(obj, path)
    return path


def latest_artifact_path(prefix: str, directory: str | Path = ".") -> Path:
    pattern = f"{prefix}_*.pkl"
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No artifacts matching pattern '{pattern}' were found in {directory}."
        )
    return files[-1]


def load_latest_artifact(prefix: str, directory: str | Path = ".") -> Any:
    path = latest_artifact_path(prefix, directory)
    print(f"Loading {prefix} artifact: {path.name}")
    return joblib.load(path)
