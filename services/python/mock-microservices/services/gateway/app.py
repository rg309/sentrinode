from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..common.observability import track_request, update_synthetic_metrics
from ..common.telemetry import setup_telemetry

SERVICE_NAME = "gateway"
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8001")

logger = logging.getLogger(SERVICE_NAME)

app = FastAPI(title="Gateway Service")
setup_telemetry(app, SERVICE_NAME)
client: httpx.AsyncClient | None = None


class CheckoutRequest(BaseModel):
    user_id: str
    cart_total: float
    correlation_id: str | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global client
    client = httpx.AsyncClient()
    logger.info("Gateway client ready, orchestrator=%s", ORCHESTRATOR_URL)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if client:
        await client.aclose()


@app.get("/healthz")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.post("/api/checkout")
async def checkout(payload: CheckoutRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready.")

    start = time.perf_counter()
    status = "error"
    try:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/orchestrate",
            json=payload.dict(),
            timeout=5.0,
        )
        response.raise_for_status()
        result = response.json()
        status = "success"
        return {
            "status": "accepted",
            "orchestrator_result": result,
        }
    except Exception as exc:
        logger.exception("Checkout failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        track_request(SERVICE_NAME, "/api/checkout", status, duration_ms)
        update_synthetic_metrics(SERVICE_NAME)
