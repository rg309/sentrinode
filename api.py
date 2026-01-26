from __future__ import annotations

import json
import logging
import threading
from typing import Any, Iterable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentrinode-api")

app = FastAPI(title="SentriNode API", version="0.1.0")

_METRICS_LOCK = threading.Lock()
_CALLS_TOTAL = 0
_DURATION_SUM_SECONDS = 0.0
_DURATION_COUNT = 0
_DURATION_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
_DURATION_BUCKET_COUNTS = [0] * len(_DURATION_BUCKETS)


def _extract_duration_seconds(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    candidates: Iterable[str] = (
        "duration_seconds",
        "duration_s",
        "duration",
        "duration_ms",
        "latency_seconds",
        "latency_s",
        "latency",
        "latency_ms",
    )
    for key in candidates:
        if key not in payload:
            continue
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            continue
        if value < 0:
            return None
        if key.endswith("_ms"):
            return value / 1000.0
        return value
    return None


def _observe_duration(seconds: float) -> None:
    global _DURATION_SUM_SECONDS, _DURATION_COUNT, _DURATION_BUCKET_COUNTS
    if seconds < 0:
        return
    _DURATION_SUM_SECONDS += seconds
    _DURATION_COUNT += 1
    for idx, bound in enumerate(_DURATION_BUCKETS):
        if seconds <= bound:
            _DURATION_BUCKET_COUNTS[idx] += 1
    # +Inf bucket is handled at render time.


def _render_metrics() -> str:
    with _METRICS_LOCK:
        calls_total = _CALLS_TOTAL
        duration_sum = _DURATION_SUM_SECONDS
        duration_count = _DURATION_COUNT
        bucket_counts = list(_DURATION_BUCKET_COUNTS)
    lines: list[str] = []
    lines.append("# HELP spanmetrics_calls_total Total spanmetric calls.")
    lines.append("# TYPE spanmetrics_calls_total counter")
    lines.append(f"spanmetrics_calls_total {calls_total}")
    lines.append("# HELP spanmetrics_duration_bucket Spanmetrics duration histogram buckets.")
    lines.append("# TYPE spanmetrics_duration_bucket histogram")
    cumulative = 0
    for bound, count in zip(_DURATION_BUCKETS, bucket_counts):
        cumulative += count
        lines.append(f'spanmetrics_duration_bucket{{le="{bound}"}} {cumulative}')
    lines.append(f'spanmetrics_duration_bucket{{le="+Inf"}} {duration_count}')
    lines.append("# HELP spanmetrics_duration_sum Total spanmetrics duration sum in seconds.")
    lines.append("# TYPE spanmetrics_duration_sum counter")
    lines.append(f"spanmetrics_duration_sum {duration_sum}")
    lines.append("# HELP spanmetrics_duration_count Total spanmetrics duration count.")
    lines.append("# TYPE spanmetrics_duration_count counter")
    lines.append(f"spanmetrics_duration_count {duration_count}")
    return "\n".join(lines) + "\n"


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/ingest")
async def ingest(request: Request) -> JSONResponse:
    global _CALLS_TOTAL
    try:
        payload: Any = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"accepted": False, "error": "invalid_json"})

    logger.info("ingest payload=%s", json.dumps(payload, separators=(",", ":"), sort_keys=True))
    duration = _extract_duration_seconds(payload)
    with _METRICS_LOCK:
        _CALLS_TOTAL += 1
        if duration is not None:
            _observe_duration(duration)
    return JSONResponse(status_code=202, content={"accepted": True})


@app.get("/metrics")
def metrics() -> Response:
    body = _render_metrics()
    return Response(content=body, media_type="text/plain; version=0.0.4")
