from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, Tuple

import httpx

logger = logging.getLogger(__name__)


async def call_service(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    try:
        response = await client.request(method, url, json=json, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.exception("Downstream call failed (%s %s): %s", method, url, exc)
        return {"status": "error", "message": str(exc)}


async def fan_out_calls(
    client: httpx.AsyncClient,
    calls: Iterable[Tuple[str, str, str, Dict[str, Any] | None]],
) -> Dict[str, Any]:
    tasks = []
    labels = []
    for label, method, url, payload in calls:
        labels.append(label)
        tasks.append(call_service(client, method, url, json=payload))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    aggregated: Dict[str, Any] = {}
    for label, result in zip(labels, results):
        if isinstance(result, Exception):
            aggregated[label] = {"status": "error", "message": str(result)}
        else:
            aggregated[label] = result
    return aggregated
