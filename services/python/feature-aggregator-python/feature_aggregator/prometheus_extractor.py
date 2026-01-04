from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Iterable

import requests

from .config import PrometheusConfig

logger = logging.getLogger(__name__)


class PrometheusClient:
    def __init__(self, config: PrometheusConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.timeout = config.timeout_seconds

    def query_scalar(self, query: str, evaluation_time: datetime) -> float | None:
        url = f"{self.base_url}/api/v1/query"
        params = {"query": query, "time": evaluation_time.isoformat()}
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            result = data.get("data", {}).get("result", [])
            if not result:
                return None
            value = result[0].get("value")
            if value is None or len(value) < 2:
                return None
            return float(value[1])
        except Exception as exc:
            logger.exception("Prometheus query failed (%s): %s", query, exc)
            return None


class TimeSeriesFeatureExtractor:
    def __init__(self, config: PrometheusConfig) -> None:
        self.config = config
        self.client = PrometheusClient(config)

    def get_cpu_features(
        self, services: Iterable[str], window_minutes: int, evaluation_time: datetime
    ) -> Dict[str, float | None]:
        window = f"{window_minutes}m"
        features: Dict[str, float | None] = {}
        for service in services:
            if not service:
                continue
            query = (
                f"avg_over_time({self.config.cpu_metric}{{service='{service}'}}[{window}])"
            )
            features[service] = self.client.query_scalar(query, evaluation_time)
        return features

    def get_future_latency(
        self, services: Iterable[str], target_time: datetime
    ) -> Dict[str, float | None]:
        latency: Dict[str, float | None] = {}
        for service in services:
            if not service:
                continue
            query = (
                "histogram_quantile(0.99, "
                f"sum(rate({self.config.latency_metric}{{service='{service}'}}[5m])) by (le))"
            )
            latency[service] = self.client.query_scalar(query, target_time)
        return latency
