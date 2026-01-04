from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _env_list(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str | None = None


@dataclass
class PrometheusConfig:
    base_url: str
    cpu_metric: str = "service_cpu_utilization"
    latency_metric: str = "service_latency_ms"
    timeout_seconds: float = 5.0


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9093"
    topic: str = "feature_aggregation_requests"
    group_id: str = "feature-aggregator"
    enabled: bool = False
    auto_offset_reset: str = "latest"


@dataclass
class FeatureStoreConfig:
    output_path: Path = Path("./feature_store")
    file_prefix: str = "features"

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)


@dataclass
class ServiceConfig:
    entry_service: str
    high_risk_services: List[str] = field(default_factory=list)
    window_minutes: int = 5
    target_horizon_minutes: int = 5


@dataclass
class SchedulerConfig:
    mode: str = "once"  # once | schedule | kafka
    interval_seconds: int = 300


@dataclass
class AppConfig:
    neo4j: Neo4jConfig
    prometheus: PrometheusConfig
    kafka: KafkaConfig
    service: ServiceConfig
    store: FeatureStoreConfig
    scheduler: SchedulerConfig


def load_config() -> AppConfig:
    neo4j_cfg = Neo4jConfig(
        uri=_env("NEO4J_URI", "bolt://localhost:7687"),
        user=_env("NEO4J_USER", "neo4j"),
        password=_env("NEO4J_PASSWORD", "testpassword"),
        database=os.getenv("NEO4J_DATABASE"),
    )

    prom_cfg = PrometheusConfig(
        base_url=_env("PROMETHEUS_URL", "http://localhost:9090"),
        cpu_metric=os.getenv("CPU_METRIC", "service_cpu_utilization"),
        latency_metric=os.getenv("LATENCY_METRIC", "service_latency_ms"),
        timeout_seconds=_env_float("PROM_TIMEOUT_SECONDS", 5.0),
    )

    kafka_cfg = KafkaConfig(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9093"),
        topic=os.getenv("KAFKA_FEATURE_TOPIC", "feature_aggregation_requests"),
        group_id=os.getenv("KAFKA_GROUP_ID", "feature-aggregator"),
        enabled=_env_bool("KAFKA_TRIGGER_ENABLED", False),
        auto_offset_reset=os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest"),
    )

    service_cfg = ServiceConfig(
        entry_service=_env("ENTRY_SERVICE", "api-gateway"),
        high_risk_services=_env_list(
            "HIGH_RISK_SERVICES", "payments-service,checkout-service,db-writer"
        ),
        window_minutes=_env_int("FEATURE_WINDOW_MINUTES", 5),
        target_horizon_minutes=_env_int("TARGET_HORIZON_MINUTES", 5),
    )

    store_cfg = FeatureStoreConfig(
        output_path=Path(os.getenv("FEATURE_STORE_PATH", "./feature_store")),
        file_prefix=os.getenv("FEATURE_FILE_PREFIX", "features"),
    )

    scheduler_cfg = SchedulerConfig(
        mode=os.getenv("FEATURE_AGGREGATOR_MODE", "once"),
        interval_seconds=_env_int("SCHEDULE_INTERVAL_SECONDS", 300),
    )

    return AppConfig(
        neo4j=neo4j_cfg,
        prometheus=prom_cfg,
        kafka=kafka_cfg,
        service=service_cfg,
        store=store_cfg,
        scheduler=scheduler_cfg,
    )
