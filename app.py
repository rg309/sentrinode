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
        letter-spacing: 0.3em;
        font-size: 1.1rem;
        text-align: center;
        color: #f8fafc;
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

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


@st.cache_data(ttl=45)
def _fetch_topology() -> tuple[bool, list[dict[str, object]]]:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
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


AUTH_LICENSE = os.getenv("SENTRINODE_LICENSE", "GUILD-ACCESS-2026")
AUTH_PASSWORD = os.getenv("SENTRINODE_PASS", "sentri-ops")


def _authenticate(license_key: str, password: str) -> bool:
    return bool(
        license_key
        and password
        and license_key.strip() == AUTH_LICENSE
        and password == AUTH_PASSWORD
    )


def _render_login() -> None:
    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 0.8, 1.2])
    with c2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("<h1>SENTRINODE</h1>", unsafe_allow_html=True)
        license_key = st.text_input("License Key", value="", type="default")
        password = st.text_input("Password", value="", type="password")
        if st.button("Sign In"):
            if _authenticate(license_key, password):
                st.session_state["logged_in"] = True
                st.experimental_rerun()
            else:
                st.error("Access denied. Check your license and password.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


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
