from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Mapping

import pandas as pd

from .config import FeatureStoreConfig

logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, config: FeatureStoreConfig) -> None:
        self.base_path = Path(config.output_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.file_prefix = config.file_prefix

    def _filename(self, window_end: str | None) -> Path:
        timestamp = window_end or datetime.utcnow().isoformat()
        safe_timestamp = timestamp.replace(":", "-")
        return self.base_path / f"{self.file_prefix}_{safe_timestamp}.parquet"

    def persist(self, frame: pd.DataFrame, metadata: Mapping[str, str]) -> Path:
        if frame.empty:
            raise ValueError("Cannot persist an empty feature frame.")

        parquet_path = self._filename(metadata.get("window_end"))
        frame.to_parquet(parquet_path, index=False)

        metadata_path = parquet_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2))
        logger.info("Feature set written to %s", parquet_path)
        return parquet_path
