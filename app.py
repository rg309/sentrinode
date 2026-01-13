#!/usr/bin/env python3
"""SentriNode Portable Console – Streamlit 2026 edition."""
from __future__ import annotations

from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable


def _normalize_neo4j_uri(raw_uri: str | None) -> str:
    uri = (raw_uri or "bolt://localhost:7687").strip().rstrip("/")
    if "://" not in uri:
        uri = f"bolt://{uri}"
    return uri


NEO4J_URI = _normalize_neo4j_uri(os.getenv("NEO4J_URI"))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

st.set_page_config(
    page_title="SentriNode Console",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=30)
def _stream_metrics() -> pd.DataFrame:
    now = datetime.utcnow()
    rows = [
        {
            "timestamp": now - timedelta(minutes=idx),
            "p99_latency": 130 + np.sin(idx / 4) * 15 + np.random.randn() * 3,
            "error_rate": max(0.0, 1.2 + np.cos(idx / 6) * 0.3),
        }
        for idx in range(60)
    ]
    return pd.DataFrame(rows).sort_values("timestamp")


@st.cache_data(ttl=15)
def _sample_alerts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts": datetime.utcnow().isoformat(timespec="seconds"),
                "service": "payments-api",
                "dependency": "postgres-core",
                "severity": "critical",
                "reason": "p99 > 450ms",
            },
            {
                "ts": (datetime.utcnow() - timedelta(minutes=5)).isoformat(timespec="seconds"),
                "service": "inventory-api",
                "dependency": "redis-edge",
                "severity": "warning",
                "reason": "error rate above 2%",
            },
        ]
    )


def _probe_neo4j() -> tuple[bool, str, dict[str, int]]:
    info: dict[str, int] = {"nodes": 0, "relationships": 0}
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            info["nodes"] = session.run("MATCH (n) RETURN count(n) AS c").single().value()
            info["relationships"] = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single().value()
        driver.close()
        return True, "Connected to Neo4j", info
    except ServiceUnavailable:
        return False, "Neo4j host unreachable – confirm Railway private networking or local tunnel.", info
    except Neo4jError as exc:
        return False, f"Neo4j authentication or bolt negotiation failed: {exc}", info
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Unexpected Neo4j error: {exc}", info


connected, connection_msg, graph_stats = _probe_neo4j()

# Sidebar --------------------------------------------------------------------
st.sidebar.title("SentriNode Console")
status_emoji = "🟢" if connected else "🔴"
st.sidebar.markdown(f"### Connection Status {status_emoji}")
st.sidebar.write(connection_msg)
st.sidebar.write(f"**URI** `{NEO4J_URI}`")
st.sidebar.write(f"**User** `{NEO4J_USER}`")
st.sidebar.metric("Graph Nodes", graph_stats["nodes"])
st.sidebar.metric("Relationships", graph_stats["relationships"])
st.sidebar.divider()

telemetry_enabled = st.sidebar.toggle("OTel Agent Enabled", value=True)
st.sidebar.caption("Requires the sentrinode-agent container from docker-compose.")

# Main content ---------------------------------------------------------------
st.title("SentriNode Portable Console")
st.caption("Edge observability in a suitcase – optimized for on-site investor demos.")

metrics_df = _stream_metrics()
alerts_df = _sample_alerts()

col_kpi = st.columns(3)
col_kpi[0].metric("Current p99 latency", f"{metrics_df['p99_latency'].iloc[-1]:.0f} ms")
col_kpi[1].metric("Error rate", f"{metrics_df['error_rate'].iloc[-1]:.2f}%")
col_kpi[2].metric("Active alerts", len(alerts_df))

fig_latency = px.line(
    metrics_df,
    x="timestamp",
    y="p99_latency",
    title="Latency trend (last 60 minutes)",
    labels={"timestamp": "UTC Time", "p99_latency": "p99 (ms)"},
)
fig_latency.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig_latency, config={"displayModeBar": False}, width="stretch")

fig_error = px.area(
    metrics_df,
    x="timestamp",
    y="error_rate",
    title="Error rate overview",
    labels={"timestamp": "UTC Time", "error_rate": "%"},
)
fig_error.update_layout(margin=dict(l=10, r=10, t=50, b=10), yaxis_tickformat=".2f")
st.plotly_chart(fig_error, config={"displayModeBar": False}, width="stretch")

st.subheader("Alerts & recommendations")
st.dataframe(alerts_df, width="stretch", height=260)

st.subheader("Synthetic span explorer")
span_rows = [
    {
        "span_id": f"span-{idx:04d}",
        "service": np.random.choice(["gateway", "payments", "inventory", "edge-cache"]),
        "latency_ms": np.random.randint(80, 420),
        "status": np.random.choice(["OK", "ANOMALY"], p=[0.78, 0.22]),
    }
    for idx in range(1, 41)
]
st.dataframe(pd.DataFrame(span_rows), width="stretch", height=360)

if telemetry_enabled:
    st.info("Telemetry forwarding to sentrinode-agent is enabled. Inspect collector logs for OTLP traffic.")
else:
    st.warning("Telemetry disabled. Toggle it on from the sidebar if you want OTLP export.")
