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
from ..common.utils import artificial_cpu_work, cpu_saturation_budget, simulate_io_latency

SERVICE_NAME = "inventory"
DB_URL = os.getenv("DB_URL", "http://db:8005")

logger = logging.getLogger(SERVICE_NAME)

app = FastAPI(title="Inventory Service")
setup_telemetry(app, SERVICE_NAME)
client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global client
    client = httpx.AsyncClient()
    logger.info("Inventory ready db=%s", DB_URL)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if client:
        await client.aclose()


@app.post("/reserve")
async def reserve(payload: Dict[str, Any]):
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready.")

    start = time.perf_counter()
    status = "error"
    try:
        await simulate_io_latency(20, 70)
        artificial_cpu_work(cpu_saturation_budget() // 2)
        response = await call_service(
            client, "POST", f"{DB_URL}/read", json={"context": "inventory", **payload}
        )
        if random.random() < 0.05:
            raise RuntimeError("Inventory mismatch detected.")
        status = "success"
        return {"status": status, "db_read": response}
    except Exception as exc:
        logger.exception("Inventory reserve failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        track_request(SERVICE_NAME, "/reserve", status, duration_ms)
        update_synthetic_metrics(SERVICE_NAME, cpu=40.0 + random.random() * 30)
