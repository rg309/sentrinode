from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentrinode-api")

app = FastAPI(title="SentriNode API", version="0.1.0")
print("SERVER=fastapi PORT=8080", flush=True)


def _setup_tracing(fastapi_app: FastAPI) -> None:
    if os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in {"true", "1", "yes"}:
        logger.info("OTEL_SDK_DISABLED=true; tracing disabled.")
        return

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not otlp_endpoint:
        logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT not set; tracing disabled.")
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "sentrinode-api")
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(fastapi_app)
    logger.info("OTEL tracing enabled. endpoint=%s service=%s", otlp_endpoint, service_name)


_setup_tracing(app)

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


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


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
