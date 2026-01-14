#!/usr/bin/env python3
"""SentriNode Enterprise Monitoring Console."""
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

st.set_page_config(page_title="SentriNode Operations", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    :root {
        --bg: #1e293b;
        --surface: #243049;
        --surface-alt: #2b374c;
        --border: #344155;
        --text: #f8fafc;
        --muted: #cbd5f5;
        --accent: #2563eb;
        --alert: #ef4444;
    }
    html, body, [data-testid="stAppViewContainer"], .main, [data-testid="block-container"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', 'Helvetica', sans-serif !important;
    }
    #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .app-shell {
        padding: 24px 48px 72px 48px;
    }
    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 28px;
    }
    .eyebrow {
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .title {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text);
    }
    .status-group {
        text-align: right;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid var(--border);
        color: var(--text);
    }
    .status-pill.alert {
        border-color: var(--alert);
        color: var(--alert);
    }
    .timestamp {
        font-size: 0.85rem;
        color: var(--muted);
        margin-top: 6px;
    }
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 18px;
        margin-bottom: 32px;
    }
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px;
        min-height: 120px;
    }
    .card-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: var(--muted);
        letter-spacing: 0.08em;
        margin-bottom: 12px;
    }
    .card-value {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text);
    }
    .card-subtext {
        margin-top: 6px;
        font-size: 0.9rem;
        color: var(--muted);
    }
    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 26px;
    }
    .panel h3 {
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 16px;
    }
    .alerts-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .alerts-table th,
    .alerts-table td {
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding: 10px 12px;
    }
    .alerts-table th {
        color: var(--muted);
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
    }
    .alerts-table td {
        color: var(--text);
    }
    .alerts-table tr:last-child td {
        border-bottom: none;
    }
    canvas {
        filter: none !important;
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

nodes_online = len(nodes_data)
anomaly_nodes = sum(1 for node in nodes_data if node.get("health") == "ANOMALY")
alerts_open = len(alerts_df)
mean_latency = metrics_df["p99_latency"].tail(12).mean()
latest_error = metrics_df["error_rate"].iloc[-1]

system_health = "Stable" if connected and anomaly_nodes == 0 and alerts_open == 0 else "Attention Required"
status_bad = system_health != "Stable"
current_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="header-bar">
        <div>
            <div class="eyebrow">SentriNode</div>
            <div class="title">Operations Control</div>
        </div>
        <div class="status-group">
            <div class="status-pill{' alert' if status_bad else ''}">
                {'Degraded Link' if status_bad else 'Operational'}
            </div>
            <div class="timestamp">Updated {current_timestamp}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

cards = [
    {
        "label": "Nodes Online",
        "value": str(nodes_online),
        "subtext": "Connected to Neo4j" if connected else "Fallback topology in use",
    },
    {
        "label": "System Health",
        "value": system_health,
        "subtext": f"{anomaly_nodes} anomaly nodes",
    },
    {
        "label": "Active Alerts",
        "value": str(alerts_open),
        "subtext": "Monitoring stack",
    },
    {
        "label": "Avg p99 (min 12)",
        "value": f"{mean_latency:.0f} ms",
        "subtext": f"Current error {latest_error:.2f}%",
    },
]

cards_html = "<div class='card-grid'>" + "".join(
    f"""
    <div class='stat-card'>
        <div class='card-label'>{card['label']}</div>
        <div class='card-value'>{card['value']}</div>
        <div class='card-subtext'>{card['subtext']}</div>
    </div>
    """
    for card in cards
) + "</div>"

st.markdown(cards_html, unsafe_allow_html=True)


def _style_figure(fig) -> None:
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter", color="#f8fafc"),
        margin=dict(l=0, r=0, t=30, b=0),
        title_font=dict(size=16, color="#f8fafc"),
        xaxis=dict(color="#cbd5f5", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(color="#cbd5f5", gridcolor="rgba(255,255,255,0.05)"),
    )


latency_fig = px.line(
    metrics_df,
    x="timestamp",
    y="p99_latency",
    title="Latency (p99) – Last 60 minutes",
    labels={"timestamp": "UTC", "p99_latency": "ms"},
)
latency_fig.update_traces(line_color="#38bdf8")
_style_figure(latency_fig)

error_fig = px.area(
    metrics_df,
    x="timestamp",
    y="error_rate",
    title="Error Rate",
    labels={"timestamp": "UTC", "error_rate": "%"},
)
error_fig.update_traces(line_color="#f97316", fillcolor="rgba(249,115,22,0.25)")
_style_figure(error_fig)

chart_cols = st.columns(2)
with chart_cols[0]:
    st.markdown('<div class="panel"><h3>latency telemetry</h3>', unsafe_allow_html=True)
    st.plotly_chart(latency_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
with chart_cols[1]:
    st.markdown('<div class="panel"><h3>error performance</h3>', unsafe_allow_html=True)
    st.plotly_chart(error_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

alerts_html = alerts_df.to_html(index=False, classes="alerts-table", border=0)
st.markdown(
    '<div class="panel"><h3>alerts & escalations</h3>' + alerts_html + '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<h3>service dependency graph</h3>', unsafe_allow_html=True)

if not all([Config, Node, Edge, agraph]):
    st.warning("Install streamlit-agraph to render the service graph.")
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
        is_anomaly = node.get("health") == "ANOMALY"
        color_block = {
            "border": "#ef4444" if is_anomaly else "#cbd5f5",
            "background": "#94a3b8",
            "highlight": {
                "border": "#ef4444" if is_anomaly else "#38bdf8",
                "background": "#cbd5f5",
            },
        }
        node_objs.append(
            Node(
                id=node["name"],
                label=node["name"],
                size=26,
                color=color_block,
                borderWidth=4 if is_anomaly else 2,
                font={"color": "#0f172a", "size": 16},
            )
        )

    edge_objs = []
    for edge in edges_data:
        source_bad = node_index.get(edge["source"], {}).get("health") == "ANOMALY"
        target_bad = node_index.get(edge["target"], {}).get("health") == "ANOMALY"
        anomaly_link = source_bad or target_bad
        edge_objs.append(
            Edge(
                source=edge["source"],
                target=edge["target"],
                color="#ef4444" if anomaly_link else "#475569",
                width=2,
                smooth={"enabled": True, "type": "continuous"},
                title=f"{edge['source']} ▶ {edge['target']} · {edge['latency']:.0f} ms",
            )
        )

    graph_config = Config(
        width=1600,
        height=600,
        directed=True,
        physics=True,
        nodeHighlightBehavior=True,
        highlightColor="#f8fafc",
        background="#1e293b",
    )
    agraph(node_objs, edge_objs, graph_config)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
