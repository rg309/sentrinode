from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""

MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", 5 * 1024 * 1024))  # default 5MB
MAX_NODES_PER_TENANT = int(os.getenv("MAX_NODES_PER_TENANT", 500))
NODE_TTL_SECONDS = int(os.getenv("NODE_TTL_SECONDS", 900))
AUTH_CACHE_TTL_SECONDS = int(os.getenv("AUTH_CACHE_TTL_SECONDS", 60))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TTLCache:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                return None
            ts, value = entry
            if time.time() - ts > self.ttl_seconds:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)


class LatestStore:
    def __init__(self, max_nodes_per_tenant: int, ttl_seconds: int) -> None:
        self.max_nodes_per_tenant = max_nodes_per_tenant
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, dict[str, Any]]] = {}

    def _prune(self, tenant_slug: str) -> None:
        cutoff = _utcnow() - timedelta(seconds=self.ttl_seconds)
        nodes = self._data.get(tenant_slug, {})
        to_remove = [name for name, entry in nodes.items() if entry["last_seen"] < cutoff]
        for name in to_remove:
            nodes.pop(name, None)
        if len(nodes) > self.max_nodes_per_tenant:
            # Evict oldest
            sorted_nodes = sorted(nodes.items(), key=lambda item: item[1]["last_seen"])
            for name, _ in sorted_nodes[:-self.max_nodes_per_tenant]:
                nodes.pop(name, None)

    def upsert(
        self,
        tenant_slug: str,
        node_name: str,
        metrics: dict[str, Any] | None,
        attributes: dict[str, Any] | None,
        last_seen: datetime | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            nodes = self._data.setdefault(tenant_slug, {})
            now = last_seen or _utcnow()
            entry = nodes.get(node_name, {})
            entry.update(
                {
                    "node_name": node_name,
                    "last_seen": now,
                    "metrics": metrics or entry.get("metrics", {}),
                    "attributes": attributes or entry.get("attributes", {}),
                }
            )
            nodes[node_name] = entry
            self._prune(tenant_slug)
            return entry

    def list_nodes(self, tenant_slug: str) -> list[dict[str, Any]]:
        with self._lock:
            self._prune(tenant_slug)
            nodes = list(self._data.get(tenant_slug, {}).values())
            nodes.sort(key=lambda e: e.get("last_seen") or _utcnow(), reverse=True)
            return [
                {
                    "node_name": n.get("node_name"),
                    "last_seen": (n.get("last_seen") or _utcnow()).isoformat(),
                    "metrics": n.get("metrics") or {},
                    "attributes": n.get("attributes") or {},
                }
                for n in nodes
            ]

    def get_node(self, tenant_slug: str, node_name: str) -> dict[str, Any] | None:
        with self._lock:
            self._prune(tenant_slug)
            node = self._data.get(tenant_slug, {}).get(node_name)
            if not node:
                return None
            return {
                "node_name": node.get("node_name"),
                "last_seen": (node.get("last_seen") or _utcnow()).isoformat(),
                "metrics": node.get("metrics") or {},
                "attributes": node.get("attributes") or {},
            }


tenant_cache = TTLCache(AUTH_CACHE_TTL_SECONDS)
key_cache = TTLCache(AUTH_CACHE_TTL_SECONDS)
store = LatestStore(MAX_NODES_PER_TENANT, NODE_TTL_SECONDS)
app = FastAPI(title="SentriNode Ingest", version="0.1.0")


def _supabase_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }


def _require_supabase_env() -> None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=503, detail="Supabase credentials are not configured.")


def _http_error(msg: str, status: int = 401) -> HTTPException:
    return HTTPException(status_code=status, detail=msg)


def _fetch_tenant_id(tenant_slug: str) -> str | None:
    cache_key = f"tenant:{tenant_slug}"
    cached = tenant_cache.get(cache_key)
    if cached:
        return cached
    _require_supabase_env()
    url = f"{SUPABASE_URL}/rest/v1/tenants"
    try:
        res = requests.get(
            url,
            headers=_supabase_headers(),
            params={"select": "id,slug", "slug": f"eq.{tenant_slug}"},
            timeout=6,
        )
    except requests.RequestException:
        return None
    if res.status_code != 200:
        return None
    data = res.json() if res.text else []
    tenant_id = data[0].get("id") if data else None
    if tenant_id:
        tenant_cache.set(cache_key, tenant_id)
    return tenant_id


def _api_key_valid(tenant_id: str, key_hash: str) -> bool:
    cache_key = f"key:{tenant_id}:{key_hash}"
    cached = key_cache.get(cache_key)
    if cached is not None:
        return bool(cached)
    _require_supabase_env()
    url = f"{SUPABASE_URL}/rest/v1/api_keys"
    try:
        res = requests.get(
            url,
            headers=_supabase_headers(),
            params={
                "select": "id,tenant_id,key_hash,revoked_at",
                "tenant_id": f"eq.{tenant_id}",
                "key_hash": f"eq.{key_hash}",
                "revoked_at": "is.null",
                "limit": 1,
            },
            timeout=6,
        )
    except requests.RequestException:
        return False
    if res.status_code != 200:
        return False
    data = res.json() if res.text else []
    valid = bool(data)
    key_cache.set(cache_key, valid)
    return valid


def _normalize_attr_value(val: dict[str, Any] | Any) -> Any:
    if not isinstance(val, dict):
        return val
    if "stringValue" in val:
        return val.get("stringValue")
    if "intValue" in val:
        return val.get("intValue")
    if "doubleValue" in val:
        return val.get("doubleValue")
    if "boolValue" in val:
        return val.get("boolValue")
    if "arrayValue" in val and isinstance(val["arrayValue"], dict):
        return [_normalize_attr_value(v) for v in val["arrayValue"].get("values", [])]
    if "kvlistValue" in val and isinstance(val["kvlistValue"], dict):
        return {item.get("key"): _normalize_attr_value(item.get("value")) for item in val["kvlistValue"].get("values", [])}
    return val


def _attributes_to_dict(attrs: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not attrs:
        return result
    for item in attrs:
        key = item.get("key")
        value = _normalize_attr_value(item.get("value"))
        if key:
            result[key] = value
    return result


def _pick_node_name(attrs: dict[str, Any]) -> str:
    for key in ("sentri.node_name", "service.name", "host.name", "k8s.pod.name", "service.instance.id"):
        if attrs.get(key):
            return str(attrs[key])
    return "unknown-node"


async def _require_auth(
    x_tenant_id: str | None = Header(default=None, convert_underscores=False),
    x_sentrinode_key: str | None = Header(default=None, convert_underscores=False),
) -> dict[str, str]:
    tenant_slug = (x_tenant_id or "").strip()
    raw_key = (x_sentrinode_key or "").strip()
    if not tenant_slug or not raw_key:
        raise _http_error("Missing X-Tenant-Id or X-SentriNode-Key headers.")
    tenant_id = _fetch_tenant_id(tenant_slug)
    if not tenant_id:
        raise _http_error("Invalid tenant.", status=401)
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    if not _api_key_valid(tenant_id, key_hash):
        raise _http_error("Invalid API key.", status=401)
    return {"tenant_slug": tenant_slug, "tenant_id": tenant_id, "key_hash": key_hash}


@app.middleware("http")
async def enforce_body_limit(request: Request, call_next):
    if request.url.path == "/healthz":
        return await call_next(request)
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Payload too large."})
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Payload too large."})

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    request._body = body  # type: ignore[attr-defined]
    request.scope["receive"] = receive  # type: ignore[assignment]
    return await call_next(request)


class HeartbeatPayload(BaseModel):
    node_name: str
    ts_unix: float | None = None
    attributes: dict[str, Any] | None = None


def _persist_resource_state(
    tenant_slug: str, resources: Iterable[dict[str, Any]], default_ts: datetime
) -> int:
    accepted = 0
    for resource in resources:
        resource_block = resource.get("resource") or resource
        attrs = _attributes_to_dict(resource_block.get("attributes"))
        scope_spans = resource.get("scopeSpans") or resource.get("instrumentationLibrarySpans") or []
        spans = []
        for scope in scope_spans:
            spans.extend(scope.get("spans") or [])
        ts = default_ts
        node_name = _pick_node_name(attrs)
        if spans:
            # use latest span start time if present (nanoseconds)
            start_times = [span.get("startTimeUnixNano") for span in spans if span.get("startTimeUnixNano")]
            if start_times:
                latest_ns = max(int(x) for x in start_times)
                ts = datetime.fromtimestamp(latest_ns / 1_000_000_000, tz=timezone.utc)
        store.upsert(tenant_slug, node_name, metrics=None, attributes=attrs, last_seen=ts)
        accepted += 1
    return accepted


def _extract_metrics(resource_metrics: Iterable[dict[str, Any]], default_ts: datetime, tenant_slug: str) -> int:
    accepted = 0
    for resource in resource_metrics:
        attrs = _attributes_to_dict(resource.get("resource", {}).get("attributes") or resource.get("attributes"))
        rm = resource.get("scopeMetrics") or resource.get("instrumentationLibraryMetrics") or []
        metrics_collected: dict[str, Any] = {}
        last_seen_ts = default_ts
        for scope in rm:
            for metric in scope.get("metrics", []):
                name = metric.get("name")
                datapoints = []
                for collection_key in ("gauge", "sum", "summary", "histogram"):
                    block = metric.get(collection_key)
                    if block and "dataPoints" in block:
                        datapoints.extend(block.get("dataPoints") or [])
                if not datapoints or not name:
                    continue
                last_point = datapoints[-1]
                if last_point.get("timeUnixNano"):
                    try:
                        last_seen_ts = datetime.fromtimestamp(int(last_point["timeUnixNano"]) / 1_000_000_000, tz=timezone.utc)
                    except Exception:
                        last_seen_ts = default_ts
                value = (
                    last_point.get("asDouble")
                    or last_point.get("asInt")
                    or last_point.get("doubleValue")
                    or last_point.get("intValue")
                    or last_point.get("value")
                )
                metrics_collected[name] = value
        node_name = _pick_node_name(attrs)
        store.upsert(tenant_slug, node_name, metrics=metrics_collected, attributes=attrs, last_seen=last_seen_ts)
        accepted += 1
    return accepted


@app.get("/healthz")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/heartbeat")
async def heartbeat(payload: HeartbeatPayload, auth=Depends(_require_auth)) -> dict[str, Any]:
    last_seen = (
        datetime.fromtimestamp(payload.ts_unix, tz=timezone.utc)
        if payload.ts_unix
        else _utcnow()
    )
    store.upsert(
        auth["tenant_slug"],
        payload.node_name,
        metrics=None,
        attributes=payload.attributes or {},
        last_seen=last_seen,
    )
    return {"ok": True, "node": payload.node_name, "last_seen": last_seen.isoformat()}


@app.post("/v1/otlp/traces")
async def ingest_traces(request: Request, auth=Depends(_require_auth)) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    resource_spans = payload.get("resourceSpans") or []
    accepted = _persist_resource_state(auth["tenant_slug"], resource_spans, _utcnow())
    return {"ok": True, "accepted": accepted}


@app.post("/v1/otlp/metrics")
async def ingest_metrics(request: Request, auth=Depends(_require_auth)) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    resource_metrics = payload.get("resourceMetrics") or payload.get("resource_metrics") or []
    accepted = _extract_metrics(resource_metrics, _utcnow(), auth["tenant_slug"])
    return {"ok": True, "accepted": accepted}


def _validate_tenant_path(tenant_slug: str, auth_tenant: str) -> None:
    if tenant_slug != auth_tenant:
        raise HTTPException(status_code=403, detail="Tenant slug mismatch.")


@app.get("/v1/tenants/{tenant_slug}/nodes")
async def list_nodes(
    tenant_slug: str,
    auth=Depends(_require_auth),
) -> dict[str, Any]:
    _validate_tenant_path(tenant_slug, auth["tenant_slug"])
    nodes = store.list_nodes(tenant_slug)
    return {"nodes": nodes, "count": len(nodes)}


@app.get("/v1/tenants/{tenant_slug}/nodes/{node_name}")
async def get_node(
    tenant_slug: str,
    node_name: str,
    auth=Depends(_require_auth),
) -> dict[str, Any]:
    _validate_tenant_path(tenant_slug, auth["tenant_slug"])
    node = store.get_node(tenant_slug, node_name)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found.")
    return node
