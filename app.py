import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
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
if "node_registered" not in st.session_state:
    st.session_state.node_registered = False
if "registration_error" not in st.session_state:
    st.session_state.registration_error = None
if "pending_registration" not in st.session_state:
    st.session_state.pending_registration = None
if "dashboard_time_range" not in st.session_state:
    st.session_state.dashboard_time_range = "Last 24h"
if "dashboard_demo" not in st.session_state:
    st.session_state.dashboard_demo = False

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
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

DASHBOARD_TIME_RANGES = {
    "Last 15m": timedelta(minutes=15),
    "Last 1h": timedelta(hours=1),
    "Last 6h": timedelta(hours=6),
    "Last 24h": timedelta(hours=24),
    "Last 7d": timedelta(days=7),
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
MATCH (n:Node)
OPTIONAL MATCH (n)-[:EMITS]->(m:Metric)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
RETURN coalesce(n.name,'Unnamed') AS name,
       coalesce(n.last_heartbeat, datetime({epochMillis:0})) AS last_seen,
       avg(m.error_rate) AS error_rate,
       avg(m.latency_p95) AS latency_p95
"""


def _neo4j_driver():
    return None


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


def _neo4j_username_exists(user: str) -> tuple[bool, str | None]:
    return False, None


def _run_registration(show_spinner: bool = True) -> None:
    st.session_state.node_registered = True
    st.session_state.registration_error = None
    st.session_state.pending_registration = None


def _start_registration_flow(username: str, email: str, password: str, mode: str) -> None:
    st.session_state.pending_registration = {
        "username": (username or "").strip(),
        "email": (email or "").strip(),
        "password": password or "",
        "mode": mode,
    }
    st.session_state.registration_error = None
    st.session_state.node_registered = False
    _run_registration(show_spinner=True)


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
    st.session_state.user_role = _resolve_user_role(username)
    message = toast_message or ("Console unlocked. Welcome back." if mode == "login" else "Account created and signed in.")
    st.toast(message, icon=toast_icon)
    _start_registration_flow(username, resolved_email, password, mode)


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
    return []


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
            exists, exists_error = _neo4j_username_exists(username)
            if exists_error:
                st.error(exists_error)
                return
            if exists:
                st.error("Username already taken.")
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
    _render_registration_status()
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


def _selected_time_range() -> tuple[str, datetime, datetime]:
    label = st.session_state.get("dashboard_time_range", "Last 24h")
    delta = DASHBOARD_TIME_RANGES.get(label, timedelta(hours=24))
    end = datetime.utcnow()
    start = end - delta
    return label, start, end


def _generate_demo_dashboard_data(start: datetime, end: datetime) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    periods = max(12, min(60, int((end - start).total_seconds() // 900) or 12))
    timeline = pd.date_range(start, end, periods=periods)
    latency_p50 = rng.normal(120, 12, size=periods).clip(min=55)
    latency_p95 = latency_p50 + rng.normal(45, 10, size=periods).clip(min=25)
    error_rate = rng.uniform(0.002, 0.02, size=periods)
    rpm = rng.integers(160, 360, size=periods)
    kpis = {
        "latency_p50": float(pd.Series(latency_p50).median()),
        "latency_p95": float(pd.Series(latency_p95).quantile(0.95)),
        "error_rate": float(pd.Series(error_rate).mean() * 100),
        "rpm": int(pd.Series(rpm).mean()),
    }
    top_services = (
        pd.DataFrame(
            {
                "Service": ["api-gateway", "billing", "auth", "frontend", "analytics"],
                "p95 latency (ms)": rng.normal(230, 25, size=5).round(1),
                "Error rate (%)": rng.uniform(0.2, 2.0, size=5).round(2),
            }
        )
        .sort_values("p95 latency (ms)", ascending=False)
        .reset_index(drop=True)
    )
    events = pd.DataFrame(
        {
            "Time": timeline[-8:],
            "Event": [
                "Health check passed",
                "Deploy completed",
                "Scaling event",
                "Latency probe",
                "Cache warm",
                "Background job",
                "Synthetic check",
                "API heartbeat",
            ],
            "Status": ["OK", "OK", "OK", "Warning", "OK", "OK", "OK", "OK"],
        }
    )
    latency = pd.DataFrame(
        {
            "timestamp": timeline,
            "latency_p50": latency_p50,
            "latency_p95": latency_p95,
        }
    )
    return {"kpis": kpis, "latency": latency, "top_services": top_services, "events": events}


def _empty_dashboard_data() -> dict[str, Any]:
    return {
        "kpis": {"latency_p50": None, "latency_p95": None, "error_rate": None, "rpm": None},
        "latency": pd.DataFrame(columns=["timestamp", "latency_p50", "latency_p95"]),
        "top_services": pd.DataFrame(columns=["Service", "p95 latency (ms)", "Error rate (%)"]),
        "events": pd.DataFrame(columns=["Time", "Event", "Status"]),
    }


def _render_empty_state(message: str) -> None:
    st.info(message)


def _render_kpi_cards(kpis: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric(
        "p50 latency (ms)", "—" if kpis["latency_p50"] is None else f"{kpis['latency_p50']:.1f}"
    )
    cols[1].metric(
        "p95 latency (ms)", "—" if kpis["latency_p95"] is None else f"{kpis['latency_p95']:.1f}"
    )
    cols[2].metric(
        "Error rate (%)", "—" if kpis["error_rate"] is None else f"{kpis['error_rate']:.2f}"
    )
    cols[3].metric("Requests / min", "—" if kpis["rpm"] is None else f"{kpis['rpm']}")


def _render_latency_chart(df: pd.DataFrame) -> None:
    st.subheader("Latency over time")
    if df.empty:
        _render_empty_state("No latency data available for this range. Enable demo mode to explore the dashboard.")
        return
    fig = px.line(
        df,
        x="timestamp",
        y=["latency_p50", "latency_p95"],
        labels={"timestamp": "Time", "value": "Latency (ms)", "variable": "Percentile"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)


def _render_dashboard_tables(top_services: pd.DataFrame, events: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top services by p95 latency")
        if top_services.empty:
            _render_empty_state("No services to display. Enable demo mode or connect your data.")
        else:
            st.dataframe(top_services, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Recent events or health checks")
        if events.empty:
            _render_empty_state("No recent events to show. Enable demo mode to preview activity.")
        else:
            st.dataframe(events, use_container_width=True, hide_index=True)


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
    st.info("Temporarily disabled while storage is being migrated.")


def render_user_dashboard(username: str) -> None:
    user_obj = st.session_state.get("user")
    user_email = getattr(user_obj, "email", None) or st.session_state.get("username") or username
    header_cols = st.columns([3, 2, 1])
    with header_cols[0]:
        st.title("Dashboard")
        if user_email:
            st.caption(user_email)
    with header_cols[1]:
        st.selectbox("Time range", list(DASHBOARD_TIME_RANGES.keys()), key="dashboard_time_range")
    with header_cols[2]:
        st.checkbox("Demo mode", key="dashboard_demo")

    _, start, end = _selected_time_range()
    demo_mode = st.session_state.get("dashboard_demo", False)
    data = _generate_demo_dashboard_data(start, end) if demo_mode else _empty_dashboard_data()
    if not demo_mode:
        _render_empty_state(
            "Demo mode is off. Enable it to explore the dashboard while data sources are connected."
        )

    _render_kpi_cards(data["kpis"])
    _render_latency_chart(data["latency"])
    _render_dashboard_tables(data["top_services"], data["events"])


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
    latency_cols = st.columns(1)
    latency_cols[0].line_chart(
        df.set_index("bucket_date")[["latency_p50", "latency_p95", "latency_p99"]],
        height=220,
    )
    metric_cols = st.columns(3)
    metric_cols[0].line_chart(df.set_index("bucket_date")[["error_rate"]], height=180)
    metric_cols[1].line_chart(df.set_index("bucket_date")[["throughput_rps"]], height=180)
    metric_cols[2].line_chart(df.set_index("bucket_date")[["cpu", "memory", "disk"]], height=180)


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
    st.info("Temporarily disabled while storage is being migrated.")


def _render_node_health(filters: FilterContext) -> pd.DataFrame:
    st.subheader("Node Health")
    st.info("Temporarily disabled while storage is being migrated.")
    return pd.DataFrame()


def _render_boards(filters: FilterContext, top_lists: dict[str, pd.DataFrame]) -> None:
    st.subheader("SentriNode Boards")
    st.info("Temporarily disabled while storage is being migrated.")


def _generate_alerts(filters: FilterContext) -> list[dict[str, str]]:
    alerts = []
    data = _run_cypher(ALERT_SOURCE_QUERY, _time_params(filters))
    now = datetime.utcnow()
    for row in data:
        last_seen_str = row.get("last_seen")
        last_seen_dt = None
        if isinstance(last_seen_str, datetime):
            last_seen_dt = last_seen_str
        else:
            try:
                last_seen_dt = datetime.fromisoformat(str(last_seen_str))
            except Exception:
                last_seen_dt = None
        if last_seen_dt and now - last_seen_dt > timedelta(hours=1):
            alerts.append(
                {
                    "severity": "high",
                    "title": f"Node {row['name']} missing heartbeat",
                    "detail": f"Last seen {last_seen_dt.isoformat()}",
                    "suggestion": "Check node agent connectivity.",
                }
            )
        if row.get("error_rate", 0) and row["error_rate"] > 0.05:
            alerts.append(
                {
                    "severity": "medium",
                    "title": f"Error spike on {row['name']}",
                    "detail": f"Error rate {row['error_rate']:.2%}",
                    "suggestion": "Inspect recent deployments or upstream dependencies.",
                }
            )
        if row.get("latency_p95", 0) and row["latency_p95"] > 500:
            alerts.append(
                {
                    "severity": "medium",
                    "title": f"Latency spike on {row['name']}",
                    "detail": f"P95 latency {row['latency_p95']:.1f} ms",
                    "suggestion": "Check downstream services for saturation.",
                }
            )
    return alerts


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
    st.info("Temporarily disabled while storage is being migrated.")


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
def _update_user_profile(username: str, full_name: str, email: str) -> bool:
    return False


def _update_user_preferences(username: str, theme: str, desktop_notifications: bool) -> bool:
    return False


def _change_password(username: str, old_password: str, new_password: str) -> bool:
    return False


def show_settings():
    st.header("Account Settings")
    st.info("Temporarily disabled while storage is being migrated.")


# --- MAIN NAVIGATION ---
if (
    not st.session_state.get("user")
    or not st.session_state.get("access_token")
    or not st.session_state.get("node_registered")
):
    render_auth_portal()
else:
    sidebar_option = st.sidebar.radio("Navigation", ("Dashboard", "Node Manager", "Settings"))
    st.sidebar.caption("Session Controls")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = "user"
        st.session_state.username = ""
        st.session_state.user = None
        st.session_state.access_token = None
        st.session_state.show_signup = False
        st.session_state.node_registered = False
        st.session_state.registration_error = None
        st.session_state.pending_registration = None
        st.rerun()
    if sidebar_option == "Dashboard":
        show_dashboard()
    elif sidebar_option == "Node Manager":
        show_node_manager()
    else:
        show_settings()
