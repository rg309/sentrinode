#!/usr/bin/env python3
"""SentriNode Local Dashboard Console."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import os

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

try:  # Optional for the topology map
    from streamlit_agraph import Config, Edge, Node, agraph
except Exception:  # pragma: no cover - optional dependency
    Config = Edge = Node = agraph = None

st.set_page_config(page_title="SentriNode Console", layout="wide", initial_sidebar_state="expanded")


def _clean_uri(raw_uri: str | None) -> str:
    uri = (raw_uri or "bolt://localhost:7687").strip().rstrip("/")
    if "://" not in uri:
        uri = f"bolt://{uri}"
    return uri


NEO4J_URI = _clean_uri(os.getenv("NEO4J_URI"))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


@dataclass
class GraphSnapshot:
    nodes: list[dict]
    edges: list[dict]


def _connect() -> tuple[bool, str, GraphSnapshot]:
    snapshot = GraphSnapshot(nodes=[], edges=[])
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            node_rows = session.run(
                """
                MATCH (s:Service)
                OPTIONAL MATCH (s)-[r:DEPENDS_ON]->(t:Service)
                RETURN DISTINCT s.name AS name, COALESCE(r.latency_ms, 0) AS latency
                """
            )
            edge_rows = session.run(
                """
                MATCH (s:Service)-[r:DEPENDS_ON]->(t:Service)
                RETURN s.name AS source, t.name AS target, r.latency_ms AS latency
                """
            )
            snapshot.nodes = [
                {"name": record["name"] or f"service-{idx}", "latency": record["latency"] or 0}
                for idx, record in enumerate(node_rows)
            ]
            snapshot.edges = [
                {
                    "source": record["source"],
                    "target": record["target"],
                    "latency": record["latency"] or 0,
                }
                for record in edge_rows
            ]
        driver.close()
        return True, "Connected to Neo4j", snapshot
    except ServiceUnavailable:
        return False, "Neo4j host unreachable. Check your tunnel or Railway networking.", snapshot
    except Neo4jError as exc:
        return False, f"Neo4j authentication/bolt handshake failed: {exc}", snapshot
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Unexpected Neo4j error: {exc}", snapshot


connected, connection_msg, snapshot = _connect()

# Sidebar status -------------------------------------------------------------
st.sidebar.title("SentriNode Console")
status_icon = "🟢" if connected else "🔴"
st.sidebar.markdown(f"### Connection Status {status_icon}")
st.sidebar.write(connection_msg)
st.sidebar.write(f"URI: `{NEO4J_URI}`")
st.sidebar.write(f"User: `{NEO4J_USER}`")
st.sidebar.metric("Services Detected", len(snapshot.nodes))
st.sidebar.metric("Dependencies", len(snapshot.edges))
st.sidebar.markdown("---")

# Header ---------------------------------------------------------------------
st.title("SentriNode Portable Dashboard")
st.caption("All-in-one observability cockpit for local demos (2026 refresh).")

# KPI cards ------------------------------------------------------------------
now = datetime.utcnow()
kpi_cols = st.columns(3)
kpi_cols[0].metric("Current p99 latency", "420 ms", "+15 vs last hour")
kpi_cols[1].metric("Error rate", "1.2%", "-0.3 vs yesterday")
kpi_cols[2].metric("Active alerts", "3", delta="2 critical")

# Topology -------------------------------------------------------------------
st.subheader("Topology Map")
if connected and snapshot.edges and agraph:
    graph_nodes = [
        Node(id=node["name"], label=node["name"], size=20, color="#00D4FF") for node in snapshot.nodes
    ]
    graph_edges = [
        Edge(
            source=edge["source"],
            target=edge["target"],
            label=f'{int(edge["latency"] or 0)} ms',
            color="#FF4B4B" if (edge["latency"] or 0) > 350 else "#00D4FF",
        )
        for edge in snapshot.edges
    ]
    config = Config(width=900, height=520, directed=True, physics=True, hierarchical=False)
    st.caption("Visualized via streamlit-agraph.")
    agraph(graph_nodes, graph_edges, config)
else:
    if not connected:
        st.warning("Topology unavailable until Neo4j connection succeeds.")
    elif not agraph:
        st.info("Install streamlit-agraph to render a graph view. Showing raw dataframe instead.")
    df = pd.DataFrame(snapshot.edges or snapshot.nodes)
    st.dataframe(df if not df.empty else pd.DataFrame([{"status": "No topology data detected"}]), width="stretch")

# Trend panels ---------------------------------------------------------------
st.subheader("Latency & Error Trends")
trend_cols = st.columns(2)

latency_data = pd.DataFrame(
    {
        "timestamp": [now - timedelta(minutes=idx) for idx in range(60)],
        "latency_ms": [420 + (idx % 10) * 6 for idx in range(60)],
    }
).sort_values("timestamp")
trend_cols[0].line_chart(latency_data.set_index("timestamp"), height=300, width="stretch")

error_data = pd.DataFrame(
    {
        "timestamp": [now - timedelta(minutes=idx) for idx in range(60)],
        "error_rate": [1.2 + (idx % 5) * 0.1 for idx in range(60)],
    }
).sort_values("timestamp")
trend_cols[1].area_chart(error_data.set_index("timestamp"), height=300, width="stretch")

# Alert stream ---------------------------------------------------------------
st.subheader("Latest Alerts")
alerts_df = pd.DataFrame(
    [
        {"time": now.isoformat(timespec="seconds"), "service": "payments-api", "severity": "Critical"},
        {"time": (now - timedelta(minutes=3)).isoformat(timespec="seconds"), "service": "inventory-api", "severity": "Warning"},
        {"time": (now - timedelta(minutes=8)).isoformat(timespec="seconds"), "service": "edge-cache", "severity": "Info"},
    ]
)
st.dataframe(alerts_df, width="stretch", height=210)

st.success("Ready to stream telemetry via docker compose (sentrinode-agent + sentrinode-ui).")
