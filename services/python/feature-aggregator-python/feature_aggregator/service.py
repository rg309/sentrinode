from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .config import AppConfig
from .feature_store import FeatureStore
from .jobs import FeatureAggregationJob
from .neo4j_extractor import GraphFeatureExtractor
from .prometheus_extractor import TimeSeriesFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    output_path: str
    row_count: int


class FeatureAggregatorService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.graph_extractor = GraphFeatureExtractor(config.neo4j)
        self.ts_extractor = TimeSeriesFeatureExtractor(config.prometheus)
        self.store = FeatureStore(config.store)

    def shutdown(self) -> None:
        self.graph_extractor.close()

    def run_job(self, job: FeatureAggregationJob) -> Optional[AggregationResult]:
        logger.info(
            "Running feature aggregation job for entry=%s window_end=%s",
            job.entry_service,
            job.window_end.isoformat(),
        )

        graph_metrics = self.graph_extractor.extract_metrics(
            job.entry_service, job.high_risk_services
        )
        cpu_features = self.ts_extractor.get_cpu_features(
            job.high_risk_services, job.window_minutes, job.window_end
        )
        latency_targets = self.ts_extractor.get_future_latency(
            job.high_risk_services, job.target_timestamp
        )

        frame = self._build_frame(job, graph_metrics, cpu_features, latency_targets)
        if frame.empty:
            logger.warning("No feature rows generated for job %s", job)
            return None

        metadata = job.as_metadata() | {
            "critical_path": " > ".join(graph_metrics.critical_path),
        }
        output_path = self.store.persist(frame, metadata)
        logger.info("Feature aggregation complete (%d rows)", len(frame))
        return AggregationResult(output_path=str(output_path), row_count=len(frame))

    def run_once(self) -> Optional[AggregationResult]:
        job = FeatureAggregationJob.from_config(self.config.service)
        return self.run_job(job)

    def run_schedule(self) -> None:
        interval = max(60, self.config.scheduler.interval_seconds)
        logger.info("Starting scheduler loop (interval=%ss)", interval)
        try:
            while True:
                self.run_once()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted, shutting down...")
        finally:
            self.shutdown()

    def _build_frame(
        self,
        job: FeatureAggregationJob,
        graph_metrics,
        cpu_features,
        latency_targets,
    ) -> pd.DataFrame:
        rows: List[dict] = []
        critical_path = " > ".join(graph_metrics.critical_path)

        for service in job.high_risk_services:
            rows.append(
                {
                    "timestamp": job.window_end.isoformat(),
                    "entry_service": job.entry_service,
                    "service": service,
                    "cpu_rolling_avg": cpu_features.get(service),
                    "path_depth": graph_metrics.path_depth,
                    "fan_out": graph_metrics.fan_out_per_service.get(service, 0),
                    "critical_path": critical_path,
                    "target_latency_p99": latency_targets.get(service),
                }
            )
        return pd.DataFrame(rows)
