from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from ..common.observability import track_request, update_synthetic_metrics
from ..common.telemetry import setup_telemetry
from ..common.utils import artificial_cpu_work, simulate_io_latency

SERVICE_NAME = "db"
BASE_LATENCY_MS = int(os.getenv("DB_BASE_LATENCY_MS", "25"))

logger = logging.getLogger(SERVICE_NAME)

app = FastAPI(title="Mock DB Service")
setup_telemetry(app, SERVICE_NAME)

DATA_STORE: Dict[str, Any] = {}


async def simulate_latency(multiplier: float = 1.0) -> None:
    jitter = random.uniform(0, 15)
    await simulate_io_latency(
        BASE_LATENCY_MS, int(BASE_LATENCY_MS * multiplier + jitter)
    )
    artificial_cpu_work(random.randint(1, 3))


@app.post("/read")
async def read(payload: Dict[str, Any]):
    start = time.perf_counter()
    status = "error"
    try:
        await simulate_latency(1.3)
        context = payload.get("context", "default")
        value = DATA_STORE.get(context, {"latest": random.random()})
        status = "success"
        return {"status": status, "context": context, "value": value}
    except Exception as exc:
        logger.exception("DB read failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        track_request(SERVICE_NAME, "/read", status, duration_ms)
        update_synthetic_metrics(SERVICE_NAME)


@app.post("/write")
async def write(payload: Dict[str, Any]):
    start = time.perf_counter()
    status = "error"
    try:
        await simulate_latency(1.8)
        context = payload.get("context", "default")
        DATA_STORE[context] = {"latest": payload, "timestamp": time.time()}
        if random.random() < 0.01:
            raise RuntimeError("Transaction deadlock detected.")
        status = "success"
        return {"status": status, "context": context}
    except Exception as exc:
        logger.exception("DB write failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        track_request(SERVICE_NAME, "/write", status, duration_ms)
        update_synthetic_metrics(SERVICE_NAME)
