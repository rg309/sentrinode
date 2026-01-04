from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException

from ..common.downstream import call_service
from ..common.observability import track_request, update_synthetic_metrics
from ..common.telemetry import setup_telemetry
from ..common.utils import artificial_cpu_work, simulate_io_latency

SERVICE_NAME = "recommendation"
DB_URL = os.getenv("DB_URL", "http://db:8005")

logger = logging.getLogger(SERVICE_NAME)

app = FastAPI(title="Recommendation Service")
setup_telemetry(app, SERVICE_NAME)
client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global client
    client = httpx.AsyncClient()
    logger.info("Recommendation ready db=%s", DB_URL)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if client:
        await client.aclose()


@app.post("/score")
async def score(payload: Dict[str, Any]):
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready.")

    start = time.perf_counter()
    status = "error"
    try:
        await simulate_io_latency(5, 20)
        artificial_cpu_work(random.randint(1, 5))
        data = await call_service(
            client, "POST", f"{DB_URL}/read", json={"context": "recommendation", **payload}
        )
        status = "success"
        return {
            "status": status,
            "score": random.uniform(0, 1),
            "data": data,
        }
    except Exception as exc:
        logger.exception("Recommendation scoring failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        track_request(SERVICE_NAME, "/score", status, duration_ms)
        update_synthetic_metrics(SERVICE_NAME)
