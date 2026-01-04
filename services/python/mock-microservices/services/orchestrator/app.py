from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException

from ..common.downstream import fan_out_calls
from ..common.observability import track_request, update_synthetic_metrics
from ..common.telemetry import setup_telemetry

SERVICE_NAME = "orchestrator"
PAYMENTS_URL = os.getenv("PAYMENTS_URL", "http://payments:8002")
INVENTORY_URL = os.getenv("INVENTORY_URL", "http://inventory:8003")

logger = logging.getLogger(SERVICE_NAME)

app = FastAPI(title="Orchestrator Service")
setup_telemetry(app, SERVICE_NAME)
client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global client
    client = httpx.AsyncClient()
    logger.info("Orchestrator ready, payments=%s inventory=%s", PAYMENTS_URL, INVENTORY_URL)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if client:
        await client.aclose()


@app.post("/orchestrate")
async def orchestrate(payload: Dict[str, Any]):
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready.")

    start = time.perf_counter()
    status = "error"
    try:
        calls = [
            ("payments", "POST", f"{PAYMENTS_URL}/process", payload),
            ("inventory", "POST", f"{INVENTORY_URL}/reserve", payload),
        ]
        results = await fan_out_calls(client, calls)
        status = "success"
        return {"status": status, "results": results}
    except Exception as exc:
        logger.exception("Orchestration failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        track_request(SERVICE_NAME, "/orchestrate", status, duration_ms)
        update_synthetic_metrics(SERVICE_NAME)
