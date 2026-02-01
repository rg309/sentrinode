from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentrinode-api")

app = FastAPI(title="SentriNode API", version="0.1.0")
print("SERVER=fastapi PORT=8080", flush=True)

# Module-scoped counters ensure values persist across requests for a single process.
SPANMETRICS_CALLS_TOTAL = Counter(
    "spanmetrics_calls_total",
    "Total spanmetric calls.",
)


def _counter_value(counter: Counter) -> float:
    """Safely read the current value from a Prometheus counter."""
    try:
        # prometheus_client stores the atomic value on the private _value field.
        return float(counter._value.get())
    except Exception:
        return float("nan")


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/health")
def health() -> PlainTextResponse:
    return PlainTextResponse("ok", status_code=200)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=302)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ingest")
async def ingest(request: Request) -> JSONResponse:
    # Increment immediately so every ingest call is counted, even if payload is invalid.
    SPANMETRICS_CALLS_TOTAL.inc()
    calls_total = _counter_value(SPANMETRICS_CALLS_TOTAL)
    logger.info("calls_total now %s", calls_total)

    try:
        payload: Any = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"accepted": False, "error": "invalid_json"})

    logger.info("ingest payload=%s", json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return JSONResponse(status_code=202, content={"accepted": True})
