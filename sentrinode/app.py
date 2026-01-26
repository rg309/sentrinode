import base64
import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from supabase import Client, create_client

try:
    from streamlit_agraph import agraph, Config, Edge, Node
except Exception:  # pragma: no cover - optional dependency
    agraph = Config = Edge = Node = None

st.set_page_config(layout="wide")

# ---------------------------------------------------------------------------
# Schema Inventory (auto-updated at runtime in render_schema_inventory panel)
# Labels & Relationships (assumptions until discovered):
#   - :Node {node_id, name, status, env, region, service, tags, last_heartbeat}
#   - :Metric {timestamp, latency_p50, latency_p95, latency_p99, error_rate,
#              throughput_rps, cpu, memory, disk, node_id}
#   - :Incident {incident_id, severity, opened_at, closed_at, summary}
#   - :Alert {rule, triggered_at, severity, node_id, details}
#   - Relationships: (:Node)-[:DEPENDS_ON]->(:Node),
#                    (:Node)-[:EMITS]->(:Metric),
#                    (:Incident)-[:AFFECTS]->(:Node),
#                    (:Alert)-[:ON_NODE]->(:Node)
# Time properties are expected as epoch millis or datetime objects.
# TODO: refine the inventory after running SCHEMA_DISCOVERY_QUERIES.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Metrics Dictionary:
#   Active Nodes:
#       Definition: Nodes with last_heartbeat >= selected time window.
#       Query:
#           MATCH (n:Node)
#           WHERE n.last_heartbeat >= datetime({epochMillis:$since})
#             AND $envs = [] OR n.env IN $envs
#             ... (filters)
#           RETURN count(DISTINCT n) AS active_nodes
#       Assumptions: last_heartbeat stored as datetime or epoch millis.
#
#   Nodes Down:
#       Nodes missing heartbeat during window.
#       Query (uses same MATCH with COUNT of nodes not in active set).
#
#   Incident Count:
#           MATCH (i:Incident)
#           WHERE i.opened_at >= datetime({epochMillis:$since})
#           RETURN count(distinct i)
#
#   Error Rate:
#           MATCH (m:Metric)
#           WHERE m.timestamp >= datetime({epochMillis:$since})
#           RETURN avg(m.error_rate)
#
#   P95 Latency:
#           MATCH (m:Metric) WHERE ...
#           RETURN percentileCont(m.latency_p95, 0.95) AS latency_p95
#
#   Throughput, Saturation, Cost Proxy, etc. follow similar structure.
#   TODO: add indexes: CREATE INDEX node_node_id IF NOT EXISTS FOR (n:Node) ON (n.node_id);
#                       CREATE INDEX metric_timestamp IF NOT EXISTS FOR (m:Metric) ON (m.timestamp);
# ---------------------------------------------------------------------------

# --- INITIAL STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False
if "user_role" not in st.session_state:
    st.session_state.user_role = "user"
if "username" not in st.session_state:
    st.session_state.username = ""
if "user" not in st.session_state:
    st.session_state.user = None
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "selected_node" not in st.session_state:
    st.session_state.selected_node = None
if "registration_attempted" not in st.session_state:
    st.session_state.registration_attempted = False
if "dashboard_time_range" not in st.session_state:
    st.session_state.dashboard_time_range = "Last 1 hour"
if "dashboard_demo" not in st.session_state:
    st.session_state.dashboard_demo = False
if "tenant_id" not in st.session_state:
    st.session_state.tenant_id = ""
if "active_tenant_slug" not in st.session_state:
    st.session_state.active_tenant_slug = ""
if "active_tenant_id" not in st.session_state:
    st.session_state.active_tenant_id = None
if "ingest_raw_key" not in st.session_state:
    st.session_state.ingest_raw_key = ""
if "tenant_memberships" not in st.session_state:
    st.session_state.tenant_memberships = []

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
INGEST_BASE_URL = (os.getenv("INGEST_BASE_URL") or "http://localhost:8000").rstrip("/")

def _resolve_pipeline_metrics_url(base_url: str) -> str:
    if not base_url:
        return ""
    if base_url.endswith("/metrics"):
        return base_url
    return f"{base_url}/metrics"


PIPELINE_BASE_URL = (os.getenv("PIPELINE_METRICS_URL") or "").strip().rstrip("/")
PIPELINE_METRICS_URL = _resolve_pipeline_metrics_url(PIPELINE_BASE_URL)
PIPELINE_INGEST_URL = f"{PIPELINE_BASE_URL}/ingest" if PIPELINE_BASE_URL else ""
LIVE_PIPELINE_METRICS_URL = PIPELINE_METRICS_URL
print("BOOT_OK", flush=True)
print("PIPELINE_METRICS_URL=", PIPELINE_BASE_URL, flush=True)
print("PIPELINE_METRICS_URL_RESOLVED=", LIVE_PIPELINE_METRICS_URL or "", flush=True)
print("PIPELINE_INGEST_URL_RESOLVED=", PIPELINE_INGEST_URL or "", flush=True)

if LIVE_PIPELINE_METRICS_URL:
    try:
        _bootstrap_r = requests.get(LIVE_PIPELINE_METRICS_URL, timeout=5)
        _head = (_bootstrap_r.text or "").replace("\n", "\\n")[:200]
        print(f"METRICS_STATUS={_bootstrap_r.status_code}", flush=True)
        print(f"METRICS_HEAD={_head}", flush=True)
    except Exception as _bootstrap_exc:  # pragma: no cover - startup probe
        print(f"METRICS_ERROR={repr(_bootstrap_exc)}", flush=True)
else:
    print("PIPELINE_METRICS_URL not set, skipping metrics fetch", flush=True)

_supabase_client_instance: Client | None = None

SCHEMA_DISCOVERY_QUERIES = [
    "CALL db.labels()",
    "CALL db.relationshipTypes()",
    "CALL db.schema.nodeTypeProperties()",
    "CALL db.schema.relTypeProperties()",
    "MATCH (n) RETURN labels(n) AS labels, count(*) AS cnt ORDER BY cnt DESC LIMIT 10",
    "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC LIMIT 10",
]

TIME_WINDOWS = {
    "Last 15 minutes": timedelta(minutes=15),
    "Last 1 hour": timedelta(hours=1),
    "Last 6 hours": timedelta(hours=6),
    "Last 24 hours": timedelta(hours=24),
    "Last 7 days": timedelta(days=7),
    "Last 30 days": timedelta(days=30),
}

MetricsPayload = dict[str, Any]

NODE_KPI_QUERY = """
MATCH (n:Node)
WHERE ($search = '' OR toLower(n.name) CONTAINS $search)
  AND (size($envs) = 0 OR coalesce(n.env,'') IN $envs)
  AND (size($services) = 0 OR coalesce(n.service,'') IN $services)
  AND (size($regions) = 0 OR coalesce(n.region,'') IN $regions)
WITH n, coalesce(n.last_heartbeat, datetime({epochMillis:0})) AS last_seen
RETURN
    count(n) AS total_nodes,
    count(CASE WHEN last_seen >= datetime({epochMillis:$since_epoch}) THEN 1 END) AS active_nodes,
    count(CASE WHEN last_seen < datetime({epochMillis:$since_epoch}) OR last_seen IS NULL THEN 1 END) AS nodes_down
"""

INCIDENT_KPI_QUERY = """
MATCH (i:Incident)
WHERE coalesce(i.opened_at, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND ($search = '' OR toLower(i.summary) CONTAINS $search)
RETURN count(i) AS incidents
"""

METRIC_AGG_QUERY = """
MATCH (m:Metric)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND ($search = '' OR toLower(m.node_name) CONTAINS $search)
RETURN
    avg(m.error_rate) AS error_rate,
    percentileCont(m.latency_p95, 0.95) AS latency_p95,
    avg(m.throughput_rps) AS throughput_rps,
    avg(m.cpu) AS cpu,
    avg(m.memory) AS memory,
    avg(m.disk) AS disk
"""

METRIC_TS_QUERY = """
MATCH (m:Metric)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND coalesce(m.timestamp, datetime({epochMillis:0})) <= datetime({epochMillis:$until_epoch})
  AND ($search = '' OR toLower(m.node_name) CONTAINS $search)
RETURN
    m.timestamp AS bucket_date,
    m.latency_p95 AS latency_p95,
    m.latency_p50 AS latency_p50,
    m.latency_p99 AS latency_p99,
    m.error_rate AS error_rate,
    m.throughput_rps AS throughput_rps,
    m.cpu AS cpu,
    m.memory AS memory,
    m.disk AS disk
ORDER BY bucket_date
"""

NODE_HEALTH_QUERY = """
MATCH (n:Node)
WHERE ($search = '' OR toLower(n.name) CONTAINS $search)
  AND (size($envs) = 0 OR coalesce(n.env,'') IN $envs)
  AND (size($services) = 0 OR coalesce(n.service,'') IN $services)
  AND (size($regions) = 0 OR coalesce(n.region,'') IN $regions)
WITH n
RETURN
    coalesce(n.node_id, id(n)) AS node_id,
    coalesce(n.name, 'Unnamed') AS name,
    coalesce(n.status, 'unknown') AS status,
    coalesce(toString(n.last_heartbeat), 'unknown') AS last_seen,
    coalesce(n.env, 'n/a') AS env,
    coalesce(n.region, 'n/a') AS region,
    coalesce(n.service, 'n/a') AS service,
    coalesce(n.tags, []) AS tags
ORDER BY last_seen DESC
LIMIT 200
"""

RELATIONSHIP_COUNTS_QUERY = """
CALL {
    MATCH (n) RETURN labels(n) AS labels, count(*) AS cnt LIMIT 50
}
RETURN labels, cnt
"""

GRAPH_SAMPLE_QUERY = """
MATCH (n:Node)-[r]->(m:Node)
RETURN n, r, m
LIMIT 50
"""

SUBGRAPH_QUERY = """
MATCH (n:Node)
WHERE ($node_name = '' AND ($search = '' OR toLower(n.name) CONTAINS $search))
   OR (toLower(n.name) = toLower($node_name))
WITH n
MATCH path = (n)-[r*1..2]-(m:Node)
RETURN nodes(path) AS nodes, relationships(path) AS rels
LIMIT 200
"""

TOP_ERRORS_QUERY = """
MATCH (m:Metric)-[:RECORDED_FOR]->(n:Node)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND ($search = '' OR toLower(n.name) CONTAINS $search)
RETURN coalesce(n.name, 'Unnamed') AS name,
       avg(m.error_rate) AS error_rate,
       percentileCont(m.latency_p95,0.95) AS latency_p95,
       avg(m.throughput_rps) AS throughput
ORDER BY error_rate DESC
LIMIT 10
"""

TOP_LATENCY_QUERY = """
MATCH (m:Metric)-[:RECORDED_FOR]->(n:Node)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND ($search = '' OR toLower(n.name) CONTAINS $search)
RETURN coalesce(n.name, 'Unnamed') AS name,
       percentileCont(m.latency_p95,0.95) AS latency_p95,
       percentileCont(m.latency_p99,0.99) AS latency_p99,
       avg(m.error_rate) AS error_rate
ORDER BY latency_p95 DESC
LIMIT 10
"""

TOP_INCIDENTS_QUERY = """
MATCH (i:Incident)-[:AFFECTS]->(n:Node)
WHERE coalesce(i.opened_at, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND ($search = '' OR toLower(n.name) CONTAINS $search)
RETURN coalesce(n.name, 'Unnamed') AS name,
       count(i) AS incidents,
       avg(duration.inSeconds(coalesce(i.closed_at, datetime({epochMillis:$until_epoch})), i.opened_at).seconds) AS mttr_seconds
ORDER BY incidents DESC
LIMIT 10
"""

TOP_RESOURCE_QUERY = """
MATCH (m:Metric)-[:RECORDED_FOR]->(n:Node)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
RETURN coalesce(n.name, 'Unnamed') AS name,
       avg(m.cpu) AS cpu,
       avg(m.memory) AS memory,
       avg(m.disk) AS disk
ORDER BY (coalesce(m.cpu,0) + coalesce(m.memory,0) + coalesce(m.disk,0)) DESC
LIMIT 10
"""

ALERT_SOURCE_QUERY = """
MATCH (m:Metric)-[:RECORDED_FOR]->(n:Node)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
RETURN coalesce(n.name,'Unnamed') AS name,
       coalesce(m.timestamp, datetime({epochMillis:0})) AS last_seen,
       avg(m.error_rate) AS error_rate,
       avg(m.latency_p95) AS latency_p95
"""


def _supabase_client() -> Client | None:
    global _supabase_client_instance
    if create_client is None:
        st.error("Supabase client library is not installed. Please add `supabase` to your environment.")
        return None
    if _supabase_client_instance is None:
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            return None
        _supabase_client_instance = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _supabase_client_instance


def _supabase_user_headers(access_token: str | None) -> dict[str, str]:
    if not access_token:
        return {}
    return {
        "Authorization": f"Bearer {access_token}",
        "apikey": SUPABASE_ANON_KEY or "",
    }


def _post_pipeline_event(payload: dict[str, Any]) -> tuple[bool, str | None]:
    if not PIPELINE_INGEST_URL:
        return False, "PIPELINE_METRICS_URL not set"
    try:
        res = requests.post(PIPELINE_INGEST_URL, json=payload, timeout=5)
    except Exception as exc:  # pragma: no cover - network
        return False, f"Failed to post event: {exc}"
    if res.status_code not in (200, 202):
        return False, f"Ingest responded with {res.status_code}: {res.text}"
    return True, None


def _show_pipeline_debug_sidebar() -> None:
    st.sidebar.header("Debug")

    # Auto refresh the whole app every 5 seconds so we can see live fetches.
    st_autorefresh(interval=5000, key="debug_panel_autorefresh")

    base_target = (PIPELINE_BASE_URL or "").strip()
    target = (LIVE_PIPELINE_METRICS_URL or "").strip()
    st.sidebar.write("PIPELINE_BASE_URL:", base_target or "(not set)")
    st.sidebar.write("PIPELINE_METRICS_URL:", target or "(not set)")

    manual_refresh = st.sidebar.button("Refresh metrics", key="debug_refresh_metrics")

    # Persist last fetch details in session_state so the debug panel proves what happened.
    dbg = st.session_state.setdefault(
        "pipeline_debug",
        {
            "last_fetch": None,
            "status": None,
            "error": None,
            "preview": "",
        },
    )

    def _has_metric(metric_name: str, blob: str) -> bool:
        if not blob:
            return False
        return re.search(rf"^{re.escape(metric_name)}(\{{|\s)", blob, flags=re.MULTILINE) is not None

    if target:
        try:
            r = requests.get(target, timeout=5)
            text = r.text or ""
            dbg["last_fetch"] = datetime.utcnow().isoformat() + "Z"
            dbg["status"] = r.status_code
            dbg["error"] = None
            dbg["preview"] = text[:300]
            dbg["has_calls"] = _has_metric("calls", text)
            dbg["has_duration_bucket"] = _has_metric("duration_bucket", text)
            dbg["has_duration_sum"] = _has_metric("duration_sum", text)
            dbg["has_duration_count"] = _has_metric("duration_count", text)

            # Keep the first 300 chars visible even if it contains newlines.
            st.sidebar.write("last fetch time:", dbg["last_fetch"])
            st.sidebar.write("status:", dbg["status"])
            st.sidebar.write(
                "spanmetrics present:",
                {
                    "calls": dbg.get("has_calls", False),
                    "duration_bucket": dbg.get("has_duration_bucket", False),
                    "duration_sum": dbg.get("has_duration_sum", False),
                    "duration_count": dbg.get("has_duration_count", False),
                },
            )
            st.sidebar.caption("metrics preview (first 300 chars)")
            st.sidebar.code(dbg["preview"])
            if manual_refresh:
                ok, err = _post_pipeline_event(
                    {
                        "source": "streamlit",
                        "event": "metrics_refresh",
                        "status": r.status_code,
                    }
                )
                if ok:
                    st.sidebar.success("Posted event to ingest.")
                else:
                    st.sidebar.warning(f"Ingest post failed: {err}")
        except Exception as exc:  # pragma: no cover - network
            dbg["last_fetch"] = datetime.utcnow().isoformat() + "Z"
            dbg["status"] = None
            dbg["error"] = repr(exc)
            dbg["preview"] = ""
            st.sidebar.write("last fetch time:", dbg["last_fetch"])
            st.sidebar.write("error:", dbg["error"])
    else:
        dbg["last_fetch"] = None
        dbg["status"] = None
        dbg["error"] = "Metrics source not configured"
        dbg["preview"] = ""
        st.sidebar.info("Metrics source not configured. Set PIPELINE_METRICS_URL.")
        st.sidebar.write("last fetch time:", "(not fetched)")
        st.sidebar.write("error:", dbg["error"])



def _fetch_user_tenants(access_token: str | None) -> tuple[list[dict[str, Any]], str | None]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return [], "Supabase credentials missing."
    if not access_token:
        return [], "No user session found."
    try:
        res = requests.get(
            f"{SUPABASE_URL}/rest/v1/tenant_members",
            headers=_supabase_user_headers(access_token),
            params={"select": "tenant_id,role,tenants(id,slug,name)"},
            timeout=6,
        )
    except Exception as exc:  # pragma: no cover - network
        return [], f"Failed to load tenants: {exc}"
    if res.status_code != 200:
        return [], f"Supabase responded with {res.status_code}: {res.text}"
    rows = res.json() if res.text else []
    memberships: list[dict[str, Any]] = []
    for row in rows:
        tenant = row.get("tenants") or {}
        memberships.append(
            {
                "tenant_id": row.get("tenant_id") or tenant.get("id"),
                "tenant_slug": tenant.get("slug"),
                "tenant_name": tenant.get("name") or tenant.get("slug"),
                "role": row.get("role"),
            }
        )
    return memberships, None


def _set_active_tenant(slug: str | None, tenant_id: Any | None) -> None:
    st.session_state.active_tenant_slug = (slug or "").strip()
    st.session_state.active_tenant_id = tenant_id


def _create_api_key_record(access_token: str | None, tenant_id: Any, name: str) -> tuple[bool, str | None, str | None]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return False, None, "Supabase credentials missing."
    if not access_token:
        return False, None, "No user session found."
    raw_key = _generate_raw_api_key()
    payload = {
        "tenant_id": tenant_id,
        "name": name or "Ingest key",
        "key_hash": _hash_api_key(raw_key),
        "key_last4": raw_key[-4:],
        "created_at": datetime.utcnow().isoformat(),
    }
    try:
        res = requests.post(
            f"{SUPABASE_URL}/rest/v1/api_keys",
            headers={**_supabase_user_headers(access_token), "Content-Type": "application/json"},
            json=payload,
            timeout=6,
        )
    except Exception as exc:  # pragma: no cover - network
        return False, None, f"Failed to create key: {exc}"
    if res.status_code not in (200, 201):
        return False, None, f"Supabase responded with {res.status_code}: {res.text}"
    return True, raw_key, None


def _fetch_api_keys_user(access_token: str | None, tenant_id: Any) -> tuple[list[dict[str, Any]], str | None]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return [], "Supabase credentials missing."
    if not access_token:
        return [], "No user session found."
    try:
        res = requests.get(
            f"{SUPABASE_URL}/rest/v1/api_keys",
            headers=_supabase_user_headers(access_token),
            params={"select": "*", "tenant_id": f"eq.{tenant_id}", "order": "created_at.desc"},
            timeout=6,
        )
    except Exception as exc:  # pragma: no cover - network
        return [], f"Failed to load keys: {exc}"
    if res.status_code != 200:
        return [], f"Supabase responded with {res.status_code}: {res.text}"
    return res.json() if res.text else [], None


def _revoke_api_key_user(access_token: str | None, key_id: Any) -> tuple[bool, str | None]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return False, "Supabase credentials missing."
    if not access_token:
        return False, "No user session found."
    try:
        res = requests.patch(
            f"{SUPABASE_URL}/rest/v1/api_keys",
            headers={**_supabase_user_headers(access_token), "Content-Type": "application/json"},
            params={"id": f"eq.{key_id}"},
            json={"revoked_at": datetime.utcnow().isoformat()},
            timeout=6,
        )
    except Exception as exc:  # pragma: no cover - network
        return False, f"Failed to revoke key: {exc}"
    if res.status_code not in (200, 204):
        return False, f"Supabase responded with {res.status_code}: {res.text}"
    return True, None


def _ingest_auth_headers() -> tuple[dict[str, str], str | None]:
    tenant_slug = (st.session_state.get("active_tenant_slug") or "").strip()
    api_key = (st.session_state.get("ingest_raw_key") or "").strip()
    if not tenant_slug:
        return {}, "Set an active tenant in Account Settings."
    if not api_key:
        return {}, "Provide a raw API key to query ingest."
    return {"X-Tenant-Id": tenant_slug, "X-SentriNode-Key": api_key}, None


def _ingest_get(path: str) -> tuple[dict[str, Any] | None, str | None]:
    headers, err = _ingest_auth_headers()
    if err:
        return None, err
    url = f"{INGEST_BASE_URL}{path}"
    try:
        res = requests.get(url, headers=headers, timeout=6)
    except Exception as exc:  # pragma: no cover - network
        return None, f"Ingest request failed: {exc}"
    if res.status_code != 200:
        return None, f"Ingest responded with {res.status_code}: {res.text}"
    return res.json() if res.text else {}, None


def _fetch_nodes_from_ingest() -> tuple[list[dict[str, Any]], str | None]:
    tenant_slug = (st.session_state.get("active_tenant_slug") or "").strip()
    if not tenant_slug:
        return [], "Select a tenant to load node data."
    data, err = _ingest_get(f"/v1/tenants/{tenant_slug}/nodes")
    if err:
        return [], err
    nodes = data.get("nodes") if isinstance(data, dict) else []
    return nodes or [], None


def _fetch_node_detail(node_name: str) -> tuple[dict[str, Any] | None, str | None]:
    tenant_slug = (st.session_state.get("active_tenant_slug") or "").strip()
    if not tenant_slug:
        return None, "Select a tenant to load node details."
    data, err = _ingest_get(f"/v1/tenants/{tenant_slug}/nodes/{node_name}")
    if err:
        return None, err
    return data or {}, None


def fetch_live_pipeline() -> str:
    url = (LIVE_PIPELINE_METRICS_URL or "").strip()
    if not url:
        return "Metrics endpoint not configured."
    try:
        response = requests.get(url, timeout=1)
        return response.text
    except Exception:
        return "Collector Offline"


def fetch_live_pipeline_raw(url: str | None = None) -> str:
    target = (url or LIVE_PIPELINE_METRICS_URL or "").strip()
    if not target:
        return "Metrics endpoint not configured."
    try:
        res = requests.get(target, timeout=5)
        res.raise_for_status()
        return res.text
    except Exception as exc:  # pragma: no cover - network
        return f"Error connecting to pipeline: {exc}"


def _parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{")[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        metrics[name] = value
    return metrics


def _fetch_live_pipeline_data(url: str | None = None) -> dict[str, Any]:
    target = (url or LIVE_PIPELINE_METRICS_URL or "").strip()
    if not target:
        return {
            "kpis": {"p50": 0.0, "p95": 0.0, "error_rate": 0.0, "rpm": 0.0},
            "latency": pd.DataFrame(),
            "top_services": pd.DataFrame(),
            "events": pd.DataFrame(
                {"timestamp": [datetime.utcnow()], "event": ["Metrics endpoint not configured."], "status": ["info"]}
            ),
        }
    raw_text = ""
    try:
        res = requests.get(target, timeout=5)
        res.raise_for_status()
        raw_text = res.text
    except Exception as exc:  # pragma: no cover - network
        return {
            "kpis": {"p50": 0.0, "p95": 0.0, "error_rate": 0.0, "rpm": 0.0},
            "latency": pd.DataFrame(),
            "top_services": pd.DataFrame(),
            "events": pd.DataFrame({"timestamp": [datetime.utcnow()], "event": [f"Pipeline fetch failed: {exc}"], "status": ["error"]}),
        }

    return _parse_spanmetrics(raw_text)


def get_live_pipeline_metrics(url: str | None = None) -> str:
    """Expose a simple accessor for the live pipeline metrics endpoint."""
    target = (url or LIVE_PIPELINE_METRICS_URL or "").strip()
    if not target:
        return "PIPELINE_METRICS_URL not set"
    try:
        res = requests.get(target, timeout=2)
        res.raise_for_status()
        return res.text
    except Exception as exc:  # pragma: no cover - network
        return f"Pipeline Offline: {exc}"


def fetch_from_pipeline() -> str:
    """Fetch live metrics directly from the collector in the same network."""
    url = (LIVE_PIPELINE_METRICS_URL or "").strip()
    if not url:
        return "PIPELINE_METRICS_URL not set"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.text or ""
    except Exception as exc:
        return f"No data in pipeline yet... ({exc})"


def fetch_pipeline_data() -> str:
    """Local helper to display the live pipeline stream."""
    return fetch_from_pipeline()

    """Local helper to display the live pipeline stream."""
    return fetch_from_pipeline()


def _parse_labels(label_blob: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not label_blob:
        return labels
    for pair in label_blob.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        labels[k.strip()] = v.strip().strip('"')
    return labels


def _parse_spanmetrics(text: str) -> dict[str, Any]:
    calls_entries: list[tuple[dict[str, str], float]] = []
    duration_buckets: dict[frozenset[tuple[str, str]], list[tuple[float, float]]] = {}
    duration_sum: dict[frozenset[tuple[str, str]], float] = {}
    duration_count: dict[frozenset[tuple[str, str]], float] = {}

    def _is_metric(found: str, canonical: str) -> bool:
        return found == canonical or found.endswith("_" + canonical)


    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        metric_part, value_part = line.split(None, 1)
        try:
            value = float(value_part.strip().split()[0])
        except ValueError:
            continue
        if "{" in metric_part:
            name, label_blob = metric_part.split("{", 1)
            label_blob = label_blob.rstrip("}")
            labels = _parse_labels(label_blob)
        else:
            name, labels = metric_part, {}

        key_items = tuple(sorted((k, v) for k, v in labels.items() if k != "le"))
        key = frozenset(key_items)

        if _is_metric(name, "calls"):
            calls_entries.append((labels, value))
        elif _is_metric(name, "duration_bucket"):
            le_val = labels.get("le")
            if le_val is None:
                continue
            try:
                le = float(le_val)
            except ValueError:
                continue
            duration_buckets.setdefault(key, []).append((le, value))
        elif _is_metric(name, "duration_sum"):
            duration_sum[key] = value
        elif _is_metric(name, "duration_count"):
            duration_count[key] = value

    now = datetime.utcnow()
    rows = []
    total_calls = 0.0
    total_errors = 0.0
    for labels, count in calls_entries:
        service = labels.get("service.name") or "unknown"
        span_name = labels.get("span.name") or labels.get("operation") or "span"
        status = labels.get("status.code") or labels.get("status_code") or ""
        total_calls += count
        if status and status.lower() != "ok":
            total_errors += count
        rows.append(
            {
                "service": service,
                "span": span_name,
                "status": status or "unset",
                "calls": count,
            }
        )

    p95_ms = 0.0
    for key, buckets in duration_buckets.items():
        total = duration_count.get(key, 0.0)
        if total <= 0:
            continue
        buckets_sorted = sorted(buckets, key=lambda x: x[0])
        threshold = 0.95 * total
        cumulative = 0.0
        est = buckets_sorted[-1][0]
        for le, val in buckets_sorted:
            cumulative = val
            if cumulative >= threshold:
                est = le
                break
        p95_ms = max(p95_ms, est * 1000.0)

    error_rate = (total_errors / total_calls * 100) if total_calls > 0 else 0.0
    kpis = {"p50": 0.0, "p95": p95_ms, "error_rate": error_rate, "rpm": total_calls}

    latency_df = pd.DataFrame({"timestamp": [now], "latency_p50": [0.0], "latency_p95": [p95_ms]})
    top_services = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["service", "span", "status", "calls"])
    events = pd.DataFrame(
        {
            "timestamp": [now],
            "event": ["Pipeline scrape"],
            "status": ["ok"],
        }
    )
    return {"kpis": kpis, "latency": latency_df, "top_services": top_services, "events": events}


def _generate_raw_api_key() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")


def _hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _persist_tenant_id(client: Client | None, tenant_id: str) -> str:
    user = st.session_state.get("user")
    user_id = getattr(user, "id", None)
    if not client:
        return "Supabase client unavailable."
    if not tenant_id:
        return "Tenant ID cleared locally."
    if not user_id:
        return "User session missing; cannot persist tenant."
    try:
        client.table("profiles").upsert({"id": user_id, "tenant_id": tenant_id}).execute()
        return "Tenant ID saved to Supabase profile."
    except Exception as exc:  # pragma: no cover - network
        return f"Could not persist tenant ID (table missing?): {exc}"


def _insert_api_key_record(
    client: Client | None, tenant_id: str, name: str, key_hash: str, key_last4: str
) -> tuple[bool, str]:
    if not client:
        return False, "Supabase client unavailable."
    payload = {
        "tenant_id": tenant_id,
        "name": name,
        "key_hash": key_hash,
        "key_last4": key_last4,
        "created_at": datetime.utcnow().isoformat(),
    }
    try:
        client.table("api_keys").insert(payload).execute()
        return True, ""
    except Exception as exc:  # pragma: no cover - network
        return False, str(exc)


def _fetch_api_keys(client: Client | None, tenant_id: str) -> tuple[list[dict[str, Any]], str | None]:
    if not client:
        return [], "Supabase client unavailable."
    if not tenant_id:
        return [], "Set a tenant ID to view keys."
    try:
        res = client.table("api_keys").select("*").eq("tenant_id", tenant_id).order("created_at", desc=True).execute()
        rows = getattr(res, "data", None) or []
        return rows, None
    except Exception as exc:  # pragma: no cover - network
        return [], str(exc)


def _revoke_api_key_record(client: Client | None, key_id: Any) -> tuple[bool, str]:
    if not client:
        return False, "Supabase client unavailable."
    try:
        client.table("api_keys").update({"revoked_at": datetime.utcnow().isoformat()}).eq("id", key_id).execute()
        return True, ""
    except Exception as exc:  # pragma: no cover - network
        return False, str(exc)


def _handle_auth_success(
    res: Any,
    identifier: str,
    password: str,
    mode: str,
    *,
    email: str | None = None,
    toast_message: str | None = None,
    toast_icon: str = "✅",
) -> None:
    resolved_email = (email or identifier or "").strip()
    username = (identifier or resolved_email).strip()
    st.session_state["user"] = res.user
    st.session_state["access_token"] = res.session.access_token
    st.session_state.username = username
    st.session_state.logged_in = True
    st.session_state.show_signup = False
    st.session_state.user_role = "user"
    message = toast_message or ("Console unlocked. Welcome back." if mode == "login" else "Account created and signed in.")
    st.toast(message, icon=toast_icon)
    st.session_state.registration_attempted = True


def is_strong_password(pw: str) -> tuple[bool, str]:
    if len(pw) < 10:
        return False, "Use at least 10 characters."
    if not re.search(r"[A-Z]", pw):
        return False, "Add at least one uppercase letter."
    if not re.search(r"[a-z]", pw):
        return False, "Add at least one lowercase letter."
    if not re.search(r"\d", pw):
        return False, "Add at least one number."
    if not re.search(r"[^A-Za-z0-9]", pw):
        return False, "Add at least one symbol."
    return True, ""


def _resolve_user_role(username: str) -> str:
    return st.session_state.get("user_role", "user")


@dataclass
class FilterContext:
    since: datetime
    until: datetime
    search: str
    envs: list[str]
    services: list[str]
    regions: list[str]
    tags: list[str]


@st.cache_data(ttl=60)
def _run_cypher(query: str, params: dict | None = None) -> list[dict[str, Any]]:
    """Legacy shim: scrape live pipeline metrics instead of querying a database."""
    fetch_live_pipeline_raw()
    return []


def _selected_time_range() -> tuple[str, datetime, datetime]:
    options = ["Last 15 minutes", "Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days"]
    default_label = st.session_state.get("dashboard_time_range", "Last 1 hour")
    default_index = options.index(default_label) if default_label in options else 1
    label = st.selectbox("Time Range", options, index=default_index, key="dashboard_time_range")
    now = datetime.utcnow()
    delta = TIME_WINDOWS.get(label, timedelta(hours=1))
    return label, now - delta, now


def _generate_demo_dashboard_data(start: datetime, end: datetime) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(start, end, periods=32)
    latency_p50 = 80 + rng.normal(0, 5, size=len(timestamps)).cumsum() / 10
    latency_p95 = latency_p50 + 40 + rng.normal(0, 8, size=len(timestamps))
    error_rate = np.clip(rng.normal(0.02, 0.005, size=len(timestamps)), 0, 0.08)
    rpm = np.clip(500 + rng.normal(0, 30, size=len(timestamps)).cumsum() / 5, 300, 900)
    df_latency = pd.DataFrame(
        {
            "timestamp": timestamps,
            "latency_p50": latency_p50,
            "latency_p95": latency_p95,
            "error_rate": error_rate,
            "rpm": rpm,
        }
    )
    kpis = {
        "p50": float(np.median(latency_p50)),
        "p95": float(np.median(latency_p95)),
        "error_rate": float(np.mean(error_rate) * 100),
        "rpm": float(np.median(rpm)),
    }
    services = [f"service-{i}" for i in range(1, 6)]
    service_p95 = np.clip(90 + rng.normal(0, 10, size=len(services)), 70, 140)
    top_services = pd.DataFrame({"Service": services, "p95_latency_ms": service_p95}).sort_values(
        by="p95_latency_ms", ascending=False
    )
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range(end - timedelta(hours=2), end, periods=6),
            "event": [
                "Health check OK",
                "Deploy completed",
                "Health check OK",
                "Auto-scaling event",
                "Health check OK",
                "Config sync",
            ],
            "status": ["ok", "info", "ok", "info", "ok", "info"],
        }
    )
    return {"kpis": kpis, "latency": df_latency, "top_services": top_services, "events": events}


def _render_kpi_cards(kpis: dict[str, float]) -> None:
    cols = st.columns(4)
    cols[0].metric("p50 latency (ms)", f"{kpis.get('p50', 0):.1f}")
    cols[1].metric("p95 latency (ms)", f"{kpis.get('p95', 0):.1f}")
    cols[2].metric("Error rate (%)", f"{kpis.get('error_rate', 0):.2f}")
    cols[3].metric("Requests / min", f"{kpis.get('rpm', 0):.0f}")


def _render_latency_chart(df: pd.DataFrame, demo_enabled: bool) -> None:
    st.subheader("Latency over time")
    if df is None or df.empty:
        if demo_enabled:
            st.info("No latency samples available.")
        else:
            st.info("Enable demo mode to see sample latency trends.")
        return
    fig = px.line(
        df,
        x="timestamp",
        y=["latency_p50", "latency_p95"],
        labels={"value": "Latency (ms)", "timestamp": "Time", "variable": "Series"},
    )
    fig.update_layout(height=300, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def _render_dashboard_tables(top_services: pd.DataFrame, events: pd.DataFrame, demo_enabled: bool) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top services by p95 latency")
        if top_services is None or top_services.empty:
            st.info("Enable demo mode to view sample service metrics.")
        else:
            st.dataframe(top_services, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Recent events")
        if events is None or events.empty:
            st.info("Enable demo mode to view recent events.")
        else:
            st.dataframe(events, use_container_width=True, hide_index=True)


def _sidebar_filters(container=None) -> FilterContext:
    container = container or st.sidebar
    with container:
        st.subheader("Global Filters")
        time_label = st.selectbox("Time Range", list(TIME_WINDOWS.keys()) + ["Custom"], key="time_range")
        custom_since = custom_until = None
        if time_label == "Custom":
            custom_until = st.datetime_input("Until", value=datetime.utcnow(), key="custom_until")
            custom_since = st.datetime_input(
                "Since", value=datetime.utcnow() - timedelta(hours=1), key="custom_since"
            )
        search = st.text_input("Search Node / Incident", "", key="search_filter")
        envs = st.text_input("Environments (comma separated)", "", key="env_filter")
        services = st.text_input("Services (comma separated)", "", key="service_filter")
        regions = st.text_input("Regions (comma separated)", "", key="region_filter")
        tags = st.text_input("Tags (comma separated)", "", key="tag_filter")

        since = custom_since or (datetime.utcnow() - TIME_WINDOWS.get(time_label, timedelta(hours=1)))
        until = custom_until or datetime.utcnow()

        st.markdown("### Export")
        st.checkbox("Enable CSV export", value=True, key="export_csv")
        st.checkbox("Enable JSON export", value=True, key="export_json")

    return FilterContext(
        since=since,
        until=until,
        search=(search or "").strip(),
        envs=[e.strip() for e in envs.split(",") if e.strip()],
        services=[s.strip() for s in services.split(",") if s.strip()],
        regions=[r.strip() for r in regions.split(",") if r.strip()],
        tags=[t.strip() for t in tags.split(",") if t.strip()],
    )


def _time_params(filters: FilterContext) -> dict[str, Any]:
    return {
        "since": filters.since.isoformat(),
        "until": filters.until.isoformat(),
        "search": filters.search.lower(),
        "envs": filters.envs,
        "services": filters.services,
        "regions": filters.regions,
        "tags": filters.tags,
        "since_epoch": int(filters.since.timestamp() * 1000),
        "until_epoch": int(filters.until.timestamp() * 1000),
    }


def show_login(client: Client | None = None) -> None:
    st.title("SentriNode Login")
    client = client or _supabase_client()
    if not client:
        st.error("Supabase credentials missing.")
        return
    with st.form("login_form"):
        email = st.text_input("Email", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        submitted = st.form_submit_button("Login")
    if submitted:
        with st.spinner("Syncing with SentriNode Network..."):
            try:
                res = client.auth.sign_in_with_password({"email": email, "password": password})
            except Exception as exc:  # pragma: no cover - network
                res = None
                st.error(f"Login failed: {exc}")
        if res and res.user and res.session:
            _handle_auth_success(
                res,
                identifier=email or "",
                password=password or "",
                mode="login",
                email=email or "",
                toast_message="Console unlocked. Welcome back.",
                toast_icon="✅",
            )
            st.rerun()
        else:
            st.error("Login failed.")


def show_signup(client: Client | None = None) -> None:
    st.title("Create SentriNode Account")
    client = client or _supabase_client()
    if not client:
        st.error("Supabase credentials missing.")
        return
    email = st.text_input("Email", key="su_email")
    pw1 = st.text_input("Password", type="password", key="su_pw1")
    pw2 = st.text_input("Re-enter Password", type="password", key="su_pw2")
    accept = st.checkbox(
        "I understand this account will be created with Supabase Auth.", key="su_accept"
    )
    if st.button("Register Node", key="su_btn"):
        username = (email or "").strip()
        if not accept:
            st.error("Please check the box to continue.")
        elif not email or "@" not in email:
            st.error("Enter a valid email.")
        elif pw1 != pw2:
            st.error("Passwords do not match.")
        else:
            ok, msg = is_strong_password(pw1)
            if not ok:
                st.error(msg)
                return
            with st.spinner("Syncing with SentriNode Network..."):
                try:
                    res = client.auth.sign_up({"email": email, "password": pw1})
                except Exception as exc:  # pragma: no cover - network
                    res = None
                    st.error(f"Sign up failed: {exc}")
            if res:
                if getattr(res, "session", None):
                    _handle_auth_success(
                        res,
                        identifier=username,
                        password=pw1 or "",
                        mode="signup",
                        email=email,
                        toast_message="Account created and signed in.",
                        toast_icon="🎉",
                    )
                    st.rerun()
                else:
                    st.success("Account created. Check your email to confirm, then sign in.")


def _render_registration_status() -> None:
    return


def render_auth_portal() -> None:
    render_hero("SENTRINODE")
    st.title("SentriNode Console Access")
    client = _supabase_client()
    if not client:
        st.error("Supabase credentials missing.")
        st.stop()
    tab_login, tab_signup = st.tabs(["Login", "Create account"])
    with tab_login:
        show_login(client)
    with tab_signup:
        show_signup(client)
    st.stop()


def authenticate_user(username: str, password: str) -> tuple[bool, str | None]:
    """Legacy compatibility hook – retained for backward compat, not used."""
    if not (username and password):
        return False, None
    return False, None


def fetch_user_nodes(username: str) -> list[dict[str, object]]:
    return []


def fetch_all_nodes() -> list[dict[str, object]]:
    return []


def _fetch_global_kpis(filters: FilterContext) -> MetricsPayload:
    params = _time_params(filters)
    node_stats = _run_cypher(NODE_KPI_QUERY, params)
    incident_stats = _run_cypher(INCIDENT_KPI_QUERY, params)
    metric_stats = _run_cypher(METRIC_AGG_QUERY, params)
    node_row = node_stats[0] if node_stats else {}
    incident_row = incident_stats[0] if incident_stats else {}
    metric_row = metric_stats[0] if metric_stats else {}
    return {
        "active_nodes": node_row.get("active_nodes", 0),
        "nodes_down": node_row.get("nodes_down", 0),
        "incident_count": incident_row.get("incidents", 0),
        "error_rate": metric_row.get("error_rate", 0.0),
        "latency_p95": metric_row.get("latency_p95", 0.0),
        "throughput": metric_row.get("throughput_rps", 0.0),
        "cpu": metric_row.get("cpu", 0.0),
        "memory": metric_row.get("memory", 0.0),
        "disk": metric_row.get("disk", 0.0),
    }


def _render_kpi_tiles(kpis: MetricsPayload) -> None:
    cols = st.columns(4)
    cols[0].metric("Active Nodes", kpis["active_nodes"])
    cols[1].metric("Nodes Down", kpis["nodes_down"], delta_color="inverse")
    cols[2].metric("Incidents", kpis["incident_count"])
    cols[3].metric("Error Rate", f"{kpis['error_rate']:.3f}")
    cols = st.columns(4)
    cols[0].metric("P95 Latency", f"{kpis['latency_p95'] or 0:.1f} ms")
    cols[1].metric("Throughput", f"{kpis['throughput'] or 0:.1f} rps")
    cols[2].metric("CPU Saturation", f"{kpis['cpu'] or 0:.1f}%")
    cols[3].metric("Memory Saturation", f"{kpis['memory'] or 0:.1f}%")


HERO_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@700&display=swap');
.sentri-hero {
    font-family: 'Syncopate', sans-serif;
    letter-spacing: 0.65rem;
    font-size: 2.75rem;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 1.5rem;
    background: linear-gradient(120deg, #0ea5e9, #38bdf8, #e0f2fe, #0ea5e9);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: sentriShift 6s ease-in-out infinite;
}
@keyframes sentriShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
</style>
"""


def render_hero(text: str) -> None:
    st.markdown(HERO_STYLE, unsafe_allow_html=True)
    st.markdown(f"<div class='sentri-hero'>{text}</div>", unsafe_allow_html=True)


# --- SCHEMA INVENTORY PANEL ---
def render_schema_inventory() -> None:
    st.info("Temporarily disabled while storage is being migrated.")


# --- UI LOGIC ---


def render_admin_dashboard() -> None:
    st_autorefresh(interval=2000, key="admin_dashboard_refresh")
    st.info("Temporarily disabled while storage is being migrated.")


def render_user_dashboard(username: str) -> None:
    st_autorefresh(interval=5000, key="dashboard_autorefresh")
    st.title("Dashboard")
    subtitle = getattr(st.session_state.get("user"), "email", None) or username or "user"
    st.caption(subtitle)
    time_label, start, end = _selected_time_range()
    demo_enabled = st.checkbox("Demo mode", value=st.session_state.get("dashboard_demo", False), key="dashboard_demo")
    st.button("Refresh metrics", key="refresh_metrics")
    if demo_enabled:
        data = _generate_demo_dashboard_data(start, end)
    else:
        data = _fetch_live_pipeline_data()
    _render_kpi_cards(data["kpis"])
    _render_latency_chart(data.get("latency"), demo_enabled)
    _render_dashboard_tables(data.get("top_services"), data.get("events"), demo_enabled)
    st.subheader("Live Pipeline Stream")
    st.text(fetch_pipeline_data())


def _fetch_time_series(filters: FilterContext) -> pd.DataFrame:
    params = _time_params(filters)
    rows = _run_cypher(METRIC_TS_QUERY, params)
    if not rows:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(filters.since, periods=6, freq="H"),
                "latency_p95": [0] * 6,
                "latency_p50": [0] * 6,
                "latency_p99": [0] * 6,
                "error_rate": [0] * 6,
                "throughput_rps": [0] * 6,
                "cpu": [0] * 6,
                "memory": [0] * 6,
                "disk": [0] * 6,
            }
        )
    else:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["bucket_date"])
    df = df.set_index("timestamp").resample("15T").mean().reset_index().fillna(0)
    df = df.rename(columns={"timestamp": "bucket_date"})
    return df


def _render_time_series(df: pd.DataFrame) -> None:
    st.subheader("Trends")
    st.info("Temporarily disabled while storage is being migrated.")


def _fetch_top_lists(filters: FilterContext) -> dict[str, pd.DataFrame]:
    params = _time_params(filters)
    data = {
        "errors": pd.DataFrame(_run_cypher(TOP_ERRORS_QUERY, params)),
        "latency": pd.DataFrame(_run_cypher(TOP_LATENCY_QUERY, params)),
        "incidents": pd.DataFrame(_run_cypher(TOP_INCIDENTS_QUERY, params)),
        "resources": pd.DataFrame(_run_cypher(TOP_RESOURCE_QUERY, params)),
    }
    return data


def _render_top_lists(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Top Lists")
    top_cols = st.tabs(
        ["Top Errors", "Top Latency", "Top Incidents", "Top Resource Usage"]
    )
    tables = ["errors", "latency", "incidents", "resources"]
    for tab, key in zip(top_cols, tables):
        with tab:
            df = data.get(key, pd.DataFrame())
            if df.empty:
                st.info("No data available for this view.")
            else:
                st.dataframe(df, hide_index=True, use_container_width=True)
                selected = st.selectbox(
                    "Focus node", [""] + df["name"].astype(str).tolist(), key=f"focus_{key}"
                )
                if selected:
                    st.session_state.selected_node = selected
                    st.success(f"Selected {selected} for drilldown.", icon="🔎")


def _render_graph_overview(filters: FilterContext) -> None:
    st.subheader("Graph Overview")
    render_schema_inventory()
    label_counts = _run_cypher("CALL db.labels() YIELD label RETURN label, count(*) AS count")
    rel_counts = _run_cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
    col1, col2 = st.columns(2)
    with col1:
        if label_counts:
            st.table(pd.DataFrame(label_counts))
        else:
            st.info("No label data available.")
    with col2:
        if rel_counts:
            st.table(pd.DataFrame(rel_counts))
        else:
            st.info("No relationship data available.")

    rows = _run_cypher(GRAPH_SAMPLE_QUERY, _time_params(filters))
    if rows and agraph and Node and Edge and Config:
        nodes_map: dict[str, Node] = {}
        edges: list[Edge] = []
        for row in rows:
            start = row["n"]
            end = row["m"]
            rel = row["r"]
            for entity in (start, end):
                if entity.id not in nodes_map:
                    nodes_map[entity.id] = Node(
                        id=str(entity.id),
                        label=str(entity.get("name", f"node-{entity.id}")),
                        size=18,
                        color="#60a5fa",
                    )
            edges.append(
                Edge(
                    source=str(start.id),
                    target=str(end.id),
                    title=rel.type,
                    color="#f97316",
                )
            )
        st.markdown("#### Sample Topology")
        config = Config(height=420, width=1000, directed=True, physics=True)
        agraph(list(nodes_map.values()), edges, config)
    else:
        st.info("Graph visualization unavailable (no data or streamlit-agraph missing).")
    st.markdown("#### Graph Explorer")
    search_node = st.text_input("Explore node (exact match or leave blank)", key="graph_focus")
    params = _time_params(filters)
    params["node_name"] = (search_node or "").strip()
    subgraph_rows = _run_cypher(SUBGRAPH_QUERY, params)
    adjacency: list[dict[str, str]] = []
    if subgraph_rows and agraph and Node and Edge and Config:
        graph_nodes: dict[str, Node] = {}
        graph_edges: list[Edge] = []
        for row in subgraph_rows:
            for n in row["nodes"]:
                if n.id not in graph_nodes:
                    graph_nodes[n.id] = Node(
                        id=str(n.id),
                        label=str(n.get("name", f"node-{n.id}")),
                        size=20,
                        color="#4ade80" if (search_node and str(n.get("name","")).lower() == search_node.lower()) else "#a855f7",
                    )
            for rel in row["rels"]:
                graph_edges.append(
                    Edge(
                        source=str(rel.start_node.id),
                        target=str(rel.end_node.id),
                        title=rel.type,
                        color="#facc15",
                    )
                )
                adjacency.append(
                    {
                        "source": rel.start_node.get("name", rel.start_node.id),
                        "relation": rel.type,
                        "target": rel.end_node.get("name", rel.end_node.id),
                    }
                )
        if graph_nodes:
            config = Config(height=480, width=1000, directed=True, physics=True)
            agraph(list(graph_nodes.values()), graph_edges, config)
    if adjacency:
        st.dataframe(pd.DataFrame(adjacency), hide_index=True, use_container_width=True)
        if st.button("Export Subgraph CSV"):
            csv = pd.DataFrame(adjacency).to_csv(index=False)
            st.download_button("Download Adjacency", csv, "subgraph.csv", mime="text/csv")
    else:
        st.info("No subgraph data available for current selection.")


def _render_node_health(filters: FilterContext) -> pd.DataFrame:
    st.subheader("Node Health")
    df = pd.DataFrame(_run_cypher(NODE_HEALTH_QUERY, _time_params(filters)))
    if df.empty:
        st.info("No node health data.")
        return df
    st.dataframe(df, use_container_width=True, hide_index=True)
    if st.session_state.get("export_csv") and not df.empty:
        st.download_button(
            "Download Node Health CSV", df.to_csv(index=False), "node_health.csv", mime="text/csv"
        )
    selection = st.selectbox(
        "Focus node for drilldown",
        [""] + df["name"].astype(str).tolist(),
        key="node_health_focus",
    )
    if selection:
        st.session_state.selected_node = selection
        st.toast(f"Drilldown focus set to {selection}", icon="🔎")
    return df


def _render_boards(filters: FilterContext, top_lists: dict[str, pd.DataFrame]) -> None:
    st.subheader("SentriNode Boards")
    board_tabs = st.tabs(["Reliability", "Performance", "Cost", "Security", "Change"])

    with board_tabs[0]:
        st.markdown("##### Reliability Board")
        incidents_df = top_lists.get("incidents") or pd.DataFrame()
        st.write("Top incident drivers")
        st.dataframe(incidents_df, hide_index=True, use_container_width=True)
        st.write("MTTR (seconds)")
        if not incidents_df.empty:
            st.bar_chart(incidents_df.set_index("name")["mttr_seconds"])
        else:
            st.info("No incident data available. TODO: instrument incident writes.")

    with board_tabs[1]:
        st.markdown("##### Performance Board")
        latency_df = top_lists.get("latency") or pd.DataFrame()
        st.write("Top latency outliers")
        st.dataframe(latency_df, hide_index=True, use_container_width=True)

    with board_tabs[2]:
        st.markdown("##### Cost Board")
        cost_df = top_lists.get("resources") or pd.DataFrame()
        st.write("Resource usage proxy (CPU+Memory+Disk)")
        st.dataframe(cost_df, hide_index=True, use_container_width=True)
        st.caption("Cost proxy derived from average resource usage. TODO: replace with actual cost metrics.")

    with board_tabs[3]:
        st.markdown("##### Security Board")
        st.info("TODO: surface auth failures or unusual patterns when available. Placeholder view.")

    with board_tabs[4]:
        st.markdown("##### Change Board")
        st.info("TODO: integrate deployment/config change data when available.")


def _generate_alerts(filters: FilterContext) -> list[dict[str, str]]:
    return []


def _render_alerts(filters: FilterContext) -> None:
    st.subheader("Alerts & Anomalies")
    st.info("Temporarily disabled while storage is being migrated.")


def _fetch_node_drilldown(node_name: str, filters: FilterContext) -> dict[str, Any]:
    params = _time_params(filters)
    params["node_name"] = node_name
    summary = _run_cypher(
        """
        MATCH (n:Node)
        WHERE toLower(n.name) = toLower($node_name)
        OPTIONAL MATCH (n)-[:DEPENDS_ON]->(dep:Node)
        RETURN n AS node, collect(dep.name) AS dependencies
        LIMIT 1
        """,
        params,
    )
    metrics = _run_cypher(
        """
        MATCH (m:Metric)-[:RECORDED_FOR]->(n:Node)
        WHERE toLower(n.name) = toLower($node_name)
          AND m.timestamp >= datetime({epochMillis:$since_epoch})
        RETURN m.timestamp AS timestamp,
               m.latency_p95 AS latency_p95,
               m.error_rate AS error_rate,
               m.cpu AS cpu,
               m.memory AS memory,
               m.disk AS disk
        ORDER BY timestamp DESC
        LIMIT 200
        """,
        params,
    )
    incidents = _run_cypher(
        """
        MATCH (i:Incident)-[:AFFECTS]->(n:Node)
        WHERE toLower(n.name) = toLower($node_name)
        RETURN i.incident_id AS incident_id,
               i.severity AS severity,
               i.opened_at AS opened_at,
               i.closed_at AS closed_at,
               i.summary AS summary
        ORDER BY i.opened_at DESC
        LIMIT 20
        """,
        params,
    )
    alerts = [alert for alert in _generate_alerts(filters) if node_name in alert["title"]]
    return {"summary": summary, "metrics": metrics, "incidents": incidents, "alerts": alerts}


def _render_node_drilldown(filters: FilterContext) -> None:
    st.subheader("Node Drilldown")
    st.info("Temporarily disabled while storage is being migrated.")


def show_node_manager():
    render_hero("SentriNode Node Manager")
    st_autorefresh(interval=2000, key="node_manager_refresh")
    st.text_input("Tenant slug", key="active_tenant_slug")
    st.text_input("Raw API key (header X-SentriNode-Key)", key="ingest_raw_key", type="password")

    nodes, err = _fetch_nodes_from_ingest()
    if err:
        st.warning(err)
        return
    if not nodes:
        st.info("No nodes observed yet for this tenant.")
        return

    for node in nodes:
        ts = node.get("last_seen")
        if ts:
            try:
                node["last_seen_readable"] = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                node["last_seen_readable"] = ts
        else:
            node["last_seen_readable"] = "n/a"

    df = pd.DataFrame(nodes)
    display_cols = [col for col in ["node_name", "last_seen_readable"] if col in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    node_names = [n.get("node_name") for n in nodes if n.get("node_name")]
    if not node_names:
        st.info("No node identifiers available yet.")
        return
    default_index = node_names.index(st.session_state.get("selected_node")) if st.session_state.get("selected_node") in node_names else 0
    selected_node = st.selectbox("Select node", node_names, index=default_index if node_names else 0)
    st.session_state.selected_node = selected_node

    if selected_node:
        detail, detail_err = _fetch_node_detail(selected_node)
        if detail_err:
            st.error(detail_err)
            return
        st.markdown(f"#### {selected_node}")
        st.caption(f"Last seen: {detail.get('last_seen', 'n/a')}")
        metrics = detail.get("metrics") or {}
        if metrics:
            metric_items = list(metrics.items())
            cols = st.columns(min(4, len(metric_items)))
            for idx, (key, value) in enumerate(metric_items):
                cols[idx % len(cols)].metric(key, value)
        else:
            st.info("No metrics reported yet for this node.")
        attrs = detail.get("attributes") or {}
        if attrs:
            st.markdown("##### Attributes")
            st.json(attrs)

    st.markdown("#### Live Pipeline Data (Direct)")
    metrics_text = fetch_live_pipeline_raw()
    st.text_area("Current Metrics Stream", metrics_text or "No metrics available yet.", height=240)


def show_api_keys() -> None:
    st.header("API Keys")
    st.caption("Generate tenant-scoped ingest keys. Raw keys are only shown once and stored hashed.")
    access_token = st.session_state.get("access_token")
    tenant_id = st.session_state.get("active_tenant_id")
    tenant_slug = (st.session_state.get("active_tenant_slug") or "").strip()

    if not tenant_id or not tenant_slug:
        memberships, err = _fetch_user_tenants(access_token)
        if memberships:
            first = memberships[0]
            _set_active_tenant(first.get("tenant_slug"), first.get("tenant_id"))
            tenant_id = first.get("tenant_id")
            tenant_slug = first.get("tenant_slug") or ""
        elif err:
            st.warning(err)

    st.text_input("Active tenant slug", key="active_tenant_slug")
    st.text_input("Raw API key (header X-SentriNode-Key)", key="ingest_raw_key", type="password")

    if not tenant_id or not tenant_slug:
        st.warning("Set an active tenant in Account Settings.")
        return

    st.subheader("Generate API key")
    with st.form("generate_key_form"):
        key_name = st.text_input("Name", value="Ingest key", key="api_keys_key_name")
        generate_clicked = st.form_submit_button("Generate API key")
    if generate_clicked:
        ok, raw_key, err = _create_api_key_record(access_token, tenant_id, key_name)
        if ok and raw_key:
            st.session_state["latest_raw_api_key"] = raw_key
            st.session_state.ingest_raw_key = raw_key
            st.success("API key created. Copy it now; it will only be shown once.")
        else:
            st.error(err or "Failed to create API key.")

    raw_key_once = st.session_state.pop("latest_raw_api_key", None)
    if raw_key_once:
        st.warning("Copy this key now. It is not stored and will disappear on refresh.", icon="⚠️")
        st.code(raw_key_once, language="")

    st.subheader("Existing keys")
    keys, fetch_err = _fetch_api_keys_user(access_token, tenant_id)
    if fetch_err:
        st.warning(fetch_err)
    elif not keys:
        st.info("No API keys found for this tenant yet.")
    else:
        for idx, row in enumerate(keys):
            status = "revoked" if row.get("revoked_at") else "active"
            key_id = row.get("id", idx)
            cols = st.columns([3, 2, 2, 2, 2])
            cols[0].markdown(f"**{row.get('name') or 'Unnamed key'}**")
            cols[1].text(f"Last 4: {row.get('key_last4') or '????'}")
            cols[2].text(f"Created: {row.get('created_at') or 'n/a'}")
            cols[3].text(f"Status: {status}")
            if status == "active" and row.get("id") is not None:
                if cols[4].button("Revoke", key=f"revoke_{key_id}"):
                    ok, err = _revoke_api_key_user(access_token, row["id"])
                    if ok:
                        st.success(f"Key ending {row.get('key_last4') or ''} revoked.")
                        st.rerun()
                    else:
                        st.error(f"Failed to revoke key: {err}")
            else:
                cols[4].text("Revoked")

    st.subheader("Example ingest headers")
    tenant_example = tenant_slug or "<tenant_slug>"
    st.code(f"X-Tenant-Id: {tenant_example}\nX-SentriNode-Key: <raw_key>", language="")


def show_dashboard():
    role = st.session_state.get("user_role", "user")
    username = st.session_state.get("username", "operator")
    if role == "admin":
        render_hero("SentriNode Operational Command")
        render_admin_dashboard()
    else:
        render_hero("Dashboard")
        render_user_dashboard(username)
    st.caption(f"Last synced: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")


# --- SETTINGS LOGIC ---
def show_settings():
    st.header("Account Settings")
    access_token = st.session_state.get("access_token")
    memberships, err = _fetch_user_tenants(access_token)
    if err:
        st.error(err)
        return
    if memberships != st.session_state.get("tenant_memberships"):
        st.session_state.tenant_memberships = memberships
    options = {m["tenant_slug"]: m for m in memberships if m.get("tenant_slug")}
    if options:
        default_slug = (
            st.session_state.get("active_tenant_slug")
            or next(iter(options.keys()))
        )
        selected_slug = st.selectbox(
            "Active tenant",
            list(options.keys()),
            index=list(options.keys()).index(default_slug) if default_slug in options else 0,
        )
        selected = options.get(selected_slug)
        _set_active_tenant(selected.get("tenant_slug"), selected.get("tenant_id"))
        st.caption(f"Role: {selected.get('role') or 'member'} • Tenant ID: {selected.get('tenant_id')}")
    else:
        st.info("No tenant memberships found for this account.")

    st.subheader("Ingest session credentials")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Tenant slug", key="active_tenant_slug")
    with col2:
        st.text_input("Raw API key (header X-SentriNode-Key)", key="ingest_raw_key", type="password")
    st.caption("Stored only in this session. Required for dashboard polling against ingest.")

    st.subheader("API Keys")
    tenant_id = st.session_state.get("active_tenant_id")
    tenant_slug = (st.session_state.get("active_tenant_slug") or "").strip()
    if not tenant_id or not tenant_slug:
        st.warning("Select a tenant to manage API keys.")
        return

    with st.form("generate_key_form_settings"):
        key_name = st.text_input("Key name", value="Ingest key", key="settings_key_name")
        gen_clicked = st.form_submit_button("Generate API key")
    if gen_clicked:
        ok, raw_key, msg = _create_api_key_record(access_token, tenant_id, key_name)
        if ok and raw_key:
            st.session_state["latest_raw_api_key"] = raw_key
            st.session_state.ingest_raw_key = raw_key
            st.success("API key created. Copy it now; it will only be shown once.")
        else:
            st.error(msg or "Failed to create API key.")

    raw_key_once = st.session_state.pop("latest_raw_api_key", None)
    if raw_key_once:
        st.warning("Copy this key now. It will disappear on refresh.", icon="⚠️")
        st.code(raw_key_once, language="")

    keys, fetch_err = _fetch_api_keys_user(access_token, tenant_id)
    if fetch_err:
        st.error(fetch_err)
    elif not keys:
        st.info("No API keys yet for this tenant.")
    else:
        for idx, row in enumerate(keys):
            status = "revoked" if row.get("revoked_at") else "active"
            cols = st.columns([3, 2, 2, 2, 2])
            cols[0].markdown(f"**{row.get('name') or 'Unnamed key'}**")
            cols[1].text(f"Last 4: {row.get('key_last4') or '????'}")
            cols[2].text(f"Created: {row.get('created_at') or 'n/a'}")
            cols[3].text(f"Status: {status}")
            if status == "active" and row.get("id") is not None:
                if cols[4].button("Revoke", key=f"settings_revoke_{row['id']}_{idx}"):
                    ok, err_msg = _revoke_api_key_user(access_token, row["id"])
                    if ok:
                        st.success("Key revoked.")
                        st.rerun()
                    else:
                        st.error(err_msg or "Failed to revoke key.")
            else:
                cols[4].text("Revoked")

    st.subheader("Example ingest headers")
    tenant_example = tenant_slug or "<tenant_slug>"
    st.code(f"X-Tenant-Id: {tenant_example}\nX-SentriNode-Key: <raw_key>", language="")


# --- MAIN NAVIGATION ---
if not st.session_state.get("user") or not st.session_state.get("access_token"):
    _show_pipeline_debug_sidebar()
    render_auth_portal()
else:
    _show_pipeline_debug_sidebar()
    sidebar_option = st.sidebar.radio("Navigation", ("Dashboard", "Node Manager", "API Keys", "Settings"))
    st.sidebar.caption("Session Controls")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = "user"
        st.session_state.username = ""
        st.session_state.user = None
        st.session_state.access_token = None
        st.session_state.show_signup = False
        st.rerun()
    if sidebar_option == "Dashboard":
        show_dashboard()
    elif sidebar_option == "Node Manager":
        show_node_manager()
    elif sidebar_option == "API Keys":
        show_api_keys()
    else:
        show_settings()
