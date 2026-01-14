#!/usr/bin/env python3
"""SentriNode Cyberpunk SOC console."""
from __future__ import annotations

from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

try:
    from streamlit_agraph import Config, Edge, Node, agraph
except Exception:  # pragma: no cover - optional dependency
    Config = Edge = Node = agraph = None


def _normalize_neo4j_uri(raw_uri: str | None) -> str:
    uri = (raw_uri or "bolt://localhost:7687").strip().rstrip("/")
    if "://" not in uri:
        uri = f"bolt://{uri}"
    return uri


NEO4J_URI = _normalize_neo4j_uri(os.getenv("NEO4J_URI"))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

st.set_page_config(page_title="SentriNode SOC", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"], .main, [data-testid="block-container"] {
        background-color: #000000 !important;
        color: #c8ffe3 !important;
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace !important;
        letter-spacing: 0.5px;
    }
    #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .cyber-shell {
        padding: 18px 42px 80px 42px;
    }
    .cyber-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 32px;
        border-radius: 18px;
        border: 1px solid rgba(57, 255, 20, 0.4);
        background: linear-gradient(90deg, rgba(57,255,20,0.14), rgba(0,0,0,0.8), rgba(255,49,49,0.12));
        box-shadow: 0 0 40px rgba(57,255,20,0.25);
        margin-bottom: 32px;
    }
    .cyber-title {
        font-size: 2.2rem;
        text-transform: uppercase;
        color: #39FF14;
        text-shadow: 0 0 12px rgba(57,255,20,0.9);
    }
    .status-stack {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 6px;
    }
    .status-chip {
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        letter-spacing: 1px;
        border: 1px solid rgba(57,255,20,0.35);
        box-shadow: 0 0 18px rgba(57,255,20,0.35);
        text-shadow: 0 0 8px rgba(57,255,20,0.8);
    }
    .status-chip.alert {
        color: #FF3131;
        border-color: rgba(255,49,49,0.4);
        box-shadow: 0 0 25px rgba(255,49,49,0.5);
        text-shadow: 0 0 14px rgba(255,49,49,0.9);
    }
    .timestamp-chip {
        color: #7DF3FF;
        font-size: 0.85rem;
        text-shadow: 0 0 10px rgba(125,243,255,0.8);
    }
    .cyber-panel {
        background: rgba(2, 12, 2, 0.82);
        border: 1px solid rgba(57,255,20,0.25);
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 0 32px rgba(57,255,20,0.2);
    }
    .metric-panel { min-height: 120px; }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #7DF3FF;
        text-shadow: 0 0 9px rgba(125,243,255,0.7);
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.4rem;
        line-height: 1.1;
        text-shadow: 0 0 18px rgba(57,255,20,0.9);
    }
    .metric-green { color: #39FF14; }
    .metric-red {
        color: #FF3131;
        text-shadow: 0 0 22px rgba(255,49,49,0.9);
        animation: pulseRed 2.4s ease-in-out infinite;
    }
    .glow-header {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #39FF14;
        margin-bottom: 12px;
        text-shadow: 0 0 14px rgba(57,255,20,0.9);
    }
    .cyber-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .cyber-table th, .cyber-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(57,255,20,0.15);
    }
    .cyber-table th {
        color: #7DF3FF;
        font-weight: 600;
        text-transform: uppercase;
        text-shadow: 0 0 8px rgba(125,243,255,0.8);
    }
    .cyber-table td {
        color: #f4fff7;
    }
    .cyber-table tr:hover {
        background: rgba(255, 49, 49, 0.08);
    }
    @keyframes pulseRed {
        0% { text-shadow: 0 0 18px rgba(255,49,49,0.7); }
        50% { text-shadow: 0 0 30px rgba(255,49,49,1); }
        100% { text-shadow: 0 0 18px rgba(255,49,49,0.7); }
    }
    canvas {
        filter: drop-shadow(0 0 18px rgba(57,255,20,0.35));
    }
    </style>
    """,
    unsafe_allow_html=True,
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


@st.cache_data(ttl=20)
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


def _fallback_topology() -> tuple[list[dict[str, str]], list[dict[str, float]]]:
    nodes = [
        {"name": "gateway", "health": "HEALTHY"},
        {"name": "payments", "health": "ANOMALY"},
        {"name": "inventory", "health": "HEALTHY"},
        {"name": "edge-cache", "health": "HEALTHY"},
    ]
    edges = [
        {"source": "gateway", "target": "payments", "latency": 480.0, "anomaly": True},
        {"source": "gateway", "target": "inventory", "latency": 180.0, "anomaly": False},
        {"source": "payments", "target": "edge-cache", "latency": 320.0, "anomaly": False},
    ]
    return nodes, edges


@st.cache_data(ttl=25, show_spinner=False)
def _load_topology() -> tuple[bool, list[dict[str, str]], list[dict[str, float]]]:
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            node_records = session.run(
                """
                MATCH (s:Service)
                RETURN s.name AS name, coalesce(s.health, 'HEALTHY') AS health
                """
            )
            edge_records = session.run(
                """
                MATCH (s:Service)-[r:DEPENDS_ON]->(t:Service)
                RETURN s.name AS source,
                       t.name AS target,
                       coalesce(r.latency_ms, rand()*400) AS latency,
                       coalesce(r.anomaly, r.latency_ms > 400) AS anomaly
                """
            )
            nodes = [
                {"name": record["name"], "health": str(record["health"]).upper()}
                for record in node_records
            ]
            edges = [
                {
                    "source": record["source"],
                    "target": record["target"],
                    "latency": float(record["latency"] or 0.0),
                    "anomaly": bool(record["anomaly"]),
                }
                for record in edge_records
            ]
        return True, nodes, edges
    except (ServiceUnavailable, Neo4jError, ValueError):
        nodes, edges = _fallback_topology()
        return False, nodes, edges
    except Exception:  # pragma: no cover - defensive fallback
        nodes, edges = _fallback_topology()
        return False, nodes, edges
    finally:
        if driver:
            driver.close()


metrics_df = _stream_metrics()
alerts_df = _sample_alerts()
connected, nodes_data, edges_data = _load_topology()

system_state = "SYSTEM ACTIVE" if connected else "LINK OFFLINE"
status_class = "status-chip" if connected else "status-chip alert"
current_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

st.markdown('<div class="cyber-shell">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="cyber-banner">
        <div class="cyber-title">SENTRINODE // CAUSAL_INTELLIGENCE</div>
        <div class="status-stack">
            <div class="{status_class}">{system_state}</div>
            <div class="timestamp-chip">SYSTEM ACTIVE {current_timestamp}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


latest_latency = metrics_df["p99_latency"].iloc[-1]
latest_error = metrics_df["error_rate"].iloc[-1]
active_alerts = len(alerts_df)
graph_nodes = len(nodes_data)


def _metric_block(label: str, value: str, variant: str = "green") -> str:
    accent = "metric-green" if variant == "green" else "metric-red"
    return f"""
    <div class="cyber-panel metric-panel">
        <div class="metric-label">{label}</div>
        <div class="metric-value {accent}">{value}</div>
    </div>
    """


metric_cols = st.columns(4)
metric_cols[0].markdown(
    _metric_block("P99 Latency", f"{latest_latency:.0f} ms"), unsafe_allow_html=True
)
metric_cols[1].markdown(
    _metric_block("Error Rate", f"{latest_error:.2f}%", variant="red" if latest_error > 1.6 else "green"),
    unsafe_allow_html=True,
)
metric_cols[2].markdown(
    _metric_block("Active Alerts", str(active_alerts), variant="red" if active_alerts else "green"),
    unsafe_allow_html=True,
)
metric_cols[3].markdown(
    _metric_block("Graph Nodes", str(graph_nodes)),
    unsafe_allow_html=True,
)


def _style_figure(fig) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#7DF3FF"),
        margin=dict(l=0, r=0, t=40, b=0),
        title_font=dict(color="#39FF14", size=16),
        xaxis=dict(color="#7DF3FF", gridcolor="rgba(125,243,255,0.15)", showgrid=False),
        yaxis=dict(color="#7DF3FF", gridcolor="rgba(125,243,255,0.15)", showgrid=False),
    )


latency_fig = px.line(
    metrics_df,
    x="timestamp",
    y="p99_latency",
    title="Latency Dynamics",
    labels={"timestamp": "UTC", "p99_latency": "p99 (ms)"},
)
latency_fig.update_traces(line_color="#39FF14")
_style_figure(latency_fig)

error_fig = px.area(
    metrics_df,
    x="timestamp",
    y="error_rate",
    title="Error Rate Drift",
    labels={"timestamp": "UTC", "error_rate": "%"},
)
error_fig.update_traces(line_color="#FF3131", fillcolor="rgba(255,49,49,0.2)")
_style_figure(error_fig)


chart_cols = st.columns(2)
with chart_cols[0]:
    st.markdown('<div class="cyber-panel"><div class="glow-header">latency telemetry</div>', unsafe_allow_html=True)
    st.plotly_chart(latency_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
with chart_cols[1]:
    st.markdown('<div class="cyber-panel"><div class="glow-header">error overview</div>', unsafe_allow_html=True)
    st.plotly_chart(error_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


alerts_html = alerts_df.to_html(index=False, classes="cyber-table", border=0)
st.markdown(
    '<div class="cyber-panel"><div class="glow-header">alerts & recommendations</div>'
    + alerts_html
    + "</div>",
    unsafe_allow_html=True,
)


st.markdown('<div class="cyber-panel" style="margin-top: 30px;">', unsafe_allow_html=True)
st.markdown('<div class="glow-header">neo4j cyber graph</div>', unsafe_allow_html=True)

if not all([Config, Node, Edge, agraph]):
    st.warning("Install streamlit-agraph to render the Neo4j graph.")
    st.dataframe(pd.DataFrame(nodes_data))
else:
    node_index = {node["name"]: node for node in nodes_data}
    for edge in edges_data:
        if edge.get("anomaly"):
            node_index.setdefault(edge["source"], {"name": edge["source"], "health": "ANOMALY"})
            node_index.setdefault(edge["target"], {"name": edge["target"], "health": "ANOMALY"})
            node_index[edge["source"]]["health"] = "ANOMALY"
            node_index[edge["target"]]["health"] = "ANOMALY"

    node_objs = []
    for node in node_index.values():
        is_bad = node.get("health") == "ANOMALY"
        node_objs.append(
            Node(
                id=node["name"],
                label=node["name"].upper(),
                size=34 if is_bad else 24,
                color="#FF3131" if is_bad else "#39FF14",
                shadow={
                    "enabled": True,
                    "color": "rgba(255,49,49,0.9)" if is_bad else "rgba(57,255,20,0.5)",
                    "size": 42 if is_bad else 18,
                },
                borderWidth=5 if is_bad else 2,
                font={"color": "#ffffff", "size": 16},
            )
        )

    edge_objs = []
    for edge in edges_data:
        source_bad = node_index.get(edge["source"], {}).get("health") == "ANOMALY"
        target_bad = node_index.get(edge["target"], {}).get("health") == "ANOMALY"
        anomaly_link = source_bad and target_bad
        edge_objs.append(
            Edge(
                source=edge["source"],
                target=edge["target"],
                color="#FF3131" if anomaly_link else "rgba(57,255,20,0.35)",
                width=3 if anomaly_link else 1,
                smooth={"enabled": True, "type": "continuous"},
                title=f"{edge['source']} ▶ {edge['target']} · {edge['latency']:.0f} ms",
            )
        )

    graph_config = Config(
        width=1600,
        height=640,
        directed=True,
        physics=True,
        nodeHighlightBehavior=True,
        highlightColor="#FFFFFF",
        background="#000000",
    )
    agraph(node_objs, edge_objs, graph_config)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
