from __future__ import annotations

import random
import time
from prometheus_client import Counter, Gauge, Histogram

SERVICE_LATENCY = Histogram(
    "service_latency_ms",
    "Latency per request (ms)",
    ["service", "endpoint"],
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000),
)

REQUEST_COUNT = Counter(
    "service_request_total",
    "Total requests per service",
    ["service", "endpoint", "status"],
)

CPU_UTILIZATION = Gauge(
    "service_cpu_utilization",
    "Synthetic CPU utilization percentage",
    ["service"],
)

QUEUE_DEPTH = Gauge(
    "service_queue_depth",
    "Synthetic queue depth for background jobs",
    ["service"],
)

ERROR_RATE = Gauge(
    "service_error_rate",
    "Synthetic error rate",
    ["service"],
)


def track_request(service: str, endpoint: str, status: str, duration_ms: float) -> None:
    REQUEST_COUNT.labels(service=service, endpoint=endpoint, status=status).inc()
    SERVICE_LATENCY.labels(service=service, endpoint=endpoint).observe(duration_ms)


def update_synthetic_metrics(service: str, cpu: float | None = None) -> None:
    CPU_UTILIZATION.labels(service=service).set(cpu or random.uniform(20, 80))
    QUEUE_DEPTH.labels(service=service).set(random.uniform(0, 20))
    ERROR_RATE.labels(service=service).set(random.uniform(0, 0.05))
