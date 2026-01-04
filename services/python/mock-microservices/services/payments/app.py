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

SERVICE_NAME = "payments"
RECOMMENDATION_URL = os.getenv("RECOMMENDATION_URL", "http://recommendation:8004")
DB_URL = os.getenv("DB_URL", "http://db:8005")

logger = logging.getLogger(SERVICE_NAME)

app = FastAPI(title="Payments Service")
setup_telemetry(app, SERVICE_NAME)
client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global client
    client = httpx.AsyncClient()
    logger.info("Payments ready rec=%s db=%s", RECOMMENDATION_URL, DB_URL)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if client:
        await client.aclose()


@app.post("/process")
async def process_payment(payload: Dict[str, Any]):
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready.")

    start = time.perf_counter()
    status = "error"
    try:
        await simulate_io_latency(10, 40)
        artificial_cpu_work(cpu_saturation_budget())

        recommendation = await call_service(
            client, "POST", f"{RECOMMENDATION_URL}/score", json=payload
        )
        write_result = await call_service(
            client, "POST", f"{DB_URL}/write", json={"context": "payments", **payload}
        )

        if random.random() < 0.02:
            raise RuntimeError("Random payment gateway failure.")

        status = "success"
        return {
            "status": status,
            "recommendation": recommendation,
            "db_write": write_result,
        }
    except Exception as exc:
        logger.exception("Payment processing failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        track_request(SERVICE_NAME, "/process", status, duration_ms)
        update_synthetic_metrics(SERVICE_NAME, cpu=60.0 + random.random() * 20)
