#!/usr/bin/env python3
"""SentriNode Industrial Pro console."""
from __future__ import annotations

from datetime import datetime
import os

import numpy as np
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

try:
    from streamlit_agraph import agraph, Node, Edge, Config
except Exception:  # pragma: no cover - optional dependency
    agraph = Node = Edge = Config = None


st.set_page_config(page_title="SentriNode Console", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    header, footer, #MainMenu {visibility: hidden;}
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 20px;
    }
    .log-panel, .table-panel {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 16px 20px;
    }
    .log-panel h3, .table-panel h3 {
        margin-top: 0;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #cbd5f5;
    }
    .login-wrapper {
        min-height: 90vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .login-box {
        width: 100%;
        max-width: 420px;
        background-color: #111a2c;
        border: 1px solid #1f2a3d;
        border-radius: 6px;
        padding: 36px 32px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.35);
    }
    .login-box h1 {
        margin-bottom: 24px;
        letter-spacing: 0.35em;
        font-size: 1.1rem;
        text-align: center;
        color: #f8fafc;
    }
    .login-subtext {
        color:#94a3b8;
        text-align:center;
        margin-bottom:18px;
        font-size:0.9rem;
        letter-spacing:0.1em;
        text-transform:uppercase;
    }
    .login-box button {
        width: 100%;
        background-color: #1d4ed8 !important;
        color: #f8fafc !important;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

NEO4J_BOLT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip().rstrip("/")
AUTH_USERNAME = os.getenv("NEO4J_USER", "neo4j")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
LICENSE_SERIAL = os.getenv("LICENSE_SERIAL") or AUTH_PASSWORD


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "disabled" not in st.session_state:
    st.session_state["disabled"] = True


@st.cache_data(ttl=45)
def _fetch_topology() -> tuple[bool, list[dict[str, object]]]:
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_BOLT_URI, auth=(AUTH_USERNAME, AUTH_PASSWORD))
        with driver.session() as session:
            records = session.run(
                """
                MATCH (s:Service)-[r]->(d:Service)
                RETURN s.name AS source,
                       d.name AS target,
                       type(r) AS rel,
                       coalesce(r.latency_ms, rand()*400) AS latency
                """
            )
            return True, [dict(record) for record in records]
    except (ServiceUnavailable, Neo4jError, ValueError):
        fallback = [
            {"source": "edge-lb", "target": "api-gateway", "rel": "ROUTES", "latency": 80.0},
            {"source": "api-gateway", "target": "auth-service", "rel": "CALLS", "latency": 210.0},
            {"source": "api-gateway", "target": "payments", "rel": "CALLS", "latency": 260.0},
            {"source": "payments", "target": "db-cluster", "rel": "WRITES", "latency": 420.0},
        ]
        return False, fallback
    finally:
        if driver:
            driver.close()


def _compute_metrics(records: list[dict[str, object]]) -> tuple[int, float, float, int]:
    node_names: set[str] = set()
    latencies: list[float] = []
    for entry in records:
        node_names.add(entry["source"])
        node_names.add(entry["target"])
        latency = entry.get("latency")
        if latency is not None:
            latencies.append(float(latency))
    active_nodes = len(node_names)
    p99_latency = float(np.percentile(latencies, 99)) if latencies else 0.0
    anomaly_count = sum(1 for latency in latencies if latency >= 400)
    system_health = max(0.0, 100.0 - anomaly_count * 8)
    return active_nodes, p99_latency, system_health, anomaly_count


def _event_log(records: list[dict[str, object]]) -> str:
    lines: list[str] = []
    now = datetime.utcnow()
    for idx, entry in enumerate(records[:6]):
        ts = (now.replace(microsecond=0)).strftime("%H:%M:%S")
        lines.append(
            f"{ts} - TRACE: {entry['source'].upper()} -> {entry['target'].upper()} · {entry.get('latency', 0):.0f}ms"
        )
        if idx == 0 and entry.get("latency", 0) >= 400:
            lines.append(f"{ts} - ALERT: Elevated latency detected on {entry['target'].upper()}")
    if not lines:
        lines.append("No recent events.")
    return "\n".join(lines)


def _dependency_table(records: list[dict[str, object]]) -> pd.DataFrame:
    dependencies: dict[str, set[str]] = {}
    for entry in records:
        dependencies.setdefault(entry["source"], set()).add(entry["target"])
        dependencies.setdefault(entry["target"], set())
    rows = []
    for service, deps in sorted(dependencies.items()):
        dep_count = len(deps)
        risk = "LOW"
        status = "Healthy"
        if dep_count >= 4:
            risk = "CRITICAL"
            status = "Latent"
        elif dep_count >= 2:
            risk = "MEDIUM"
        rows.append(
            {
                "Service": service,
                "Dependencies": dep_count,
                "Risk Level": risk,
                "Status": status,
            }
        )
    return pd.DataFrame(rows)


ADMIN_KEY = os.getenv("SENTRINODE_ADMIN_KEY")


def _ensure_license(driver: GraphDatabase.driver) -> dict[str, object] | None:
    if not LICENSE_SERIAL:
        return None
    with driver.session() as session:
        record = session.run(
            """
            MERGE (l:License {serial:$serial})
            ON CREATE SET l.status='active', l.type='trial', l.created_at=timestamp()
            SET l.created_at = coalesce(l.created_at, l.updated, timestamp()),
                l.last_seen = timestamp()
            RETURN l.status AS status, l.type AS type, l.created_at AS created_at
            """,
            serial=LICENSE_SERIAL,
        ).single()
    return record


def _license_is_active() -> bool:
    if ADMIN_KEY and LICENSE_SERIAL == ADMIN_KEY:
        return True
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_BOLT_URI, auth=(AUTH_USERNAME, AUTH_PASSWORD))
        record = _ensure_license(driver)
        if not record:
            return False
        status = str(record["status"] or "").lower()
        lic_type = str(record.get("type") or "").lower()
        created_at = record.get("created_at")
        if status == "paid":
            return True
        if created_at is None:
            return False
        age_seconds = (datetime.utcnow().timestamp() * 1000) - float(created_at)
        seven_days_ms = 7 * 24 * 60 * 60 * 1000
        if age_seconds <= seven_days_ms and status == "active":
            return True
        return False
    except Exception:
        return False
    finally:
        if driver:
            driver.close()


def _sync_license_state() -> None:
    st.session_state["disabled"] = not _license_is_active()


def _authenticate(username: str, password: str) -> bool:
    return bool(
        username
        and password
        and username.strip().lower() == AUTH_USERNAME.lower()
        and password == AUTH_PASSWORD
    )


def _render_login() -> None:
    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
    _, center_col, _ = st.columns([1.2, 0.8, 1.2])
    with center_col:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("<h1>SENTRINODE</h1>", unsafe_allow_html=True)
        st.markdown("<div class='login-subtext'>Causal Intelligence Console</div>", unsafe_allow_html=True)
        username = st.text_input("Username", value="", key="username-input")
        password = st.text_input("Password", value="", type="password", key="password-input")
        if st.button("Sign In", use_container_width=True):
            if _authenticate(username, password):
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Access denied. Check your credentials.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_disabled() -> None:
    st.markdown(
        """
        <div style="min-height:90vh;display:flex;align-items:center;justify-content:center;background:#0f172a;">
            <div style="border:1px solid #7f1d1d;background:#1c0f0f;padding:48px 60px;border-radius:8px;text-align:center;">
                <div style="letter-spacing:0.3em;color:#f87171;font-size:1.5rem;margin-bottom:16px;">LICENSE EXPIRED</div>
                <div style="color:#fecaca;font-size:1rem;">CONTACT SUPPORT TO RESTORE ACCESS</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


_sync_license_state()
if st.session_state.get("disabled", True):
    _render_disabled()
    st.stop()

if not st.session_state["logged_in"]:
    _render_login()
    st.stop()


connected, topology = _fetch_topology()
active_nodes, p99_latency, system_health, anomaly_count = _compute_metrics(topology)
log_output = _event_log(topology)
dep_table = _dependency_table(topology)

st.markdown("## SENTRINODE // **CAUSAL INTELLIGENCE**")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Nodes", str(active_nodes), "Live" if connected else "Fallback")
col2.metric("P99 Latency", f"{p99_latency:.0f}ms", "-4ms")
col3.metric("System Health", f"{system_health:.1f}%", "Stable")
col4.metric("Anomalies", str(anomaly_count), "Clear", delta_color="inverse")

st.write("---")

st.subheader("Infrastructure Topology")
if not agraph or not Node or not Edge or not Config:
    st.warning("Install streamlit-agraph to render the service graph.")
else:
    node_objs: list[Node] = []
    edge_objs: list[Edge] = []
    node_names: set[str] = set()
    for entry in topology:
        for name in (entry["source"], entry["target"]):
            if name not in node_names:
                node_objs.append(Node(id=name, label=name, size=20, color="#64748b"))
                node_names.add(name)
        edge_objs.append(
            Edge(
                source=entry["source"],
                target=entry["target"],
                color="#94a3b8",
                title=entry.get("rel", "link"),
            )
        )
    graph_config = Config(width=1200, height=500, directed=True, physics=True, hierarchical=True)
    agraph(nodes=node_objs, edges=edge_objs, config=graph_config)

st.write("---")

left, right = st.columns(2)
with left:
    st.subheader("Causal Event Log")
    st.code(log_output, language="bash")
with right:
    st.subheader("Dependency Risk")
    if dep_table.empty:
        st.write("No dependency data available.")
    else:
        st.table(dep_table)
