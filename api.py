from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentrinode-api")

app = FastAPI(title="SentriNode API", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/ingest")
async def ingest(request: Request) -> JSONResponse:
    try:
        payload: Any = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"accepted": False, "error": "invalid_json"})

    logger.info("ingest payload=%s", json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return JSONResponse(status_code=202, content={"accepted": True})
