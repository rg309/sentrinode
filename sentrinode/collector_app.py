from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("collector")

STORE_ENABLED = os.getenv("COLLECTOR_STORE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
MAX_STORED = int(os.getenv("COLLECTOR_MAX_STORED", "1000"))

_store: list[dict[str, Any]] = []

app = FastAPI(title="SentriNode Collector", version="0.1.0")


@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"


@app.post("/ingest")
async def ingest(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid_json"})

    entry = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    logger.info("ingest payload=%s", json.dumps(entry, separators=(",", ":"), sort_keys=True))

    if STORE_ENABLED and MAX_STORED > 0:
        _store.append(entry)
        if len(_store) > MAX_STORED:
            del _store[:-MAX_STORED]

    return {"ok": True}
