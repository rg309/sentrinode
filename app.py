#!/usr/bin/env python3
"""SentriNode Enterprise Monitoring Console."""
from __future__ import annotations

from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
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
        --bg: #0f172a;
        --panel: #162033;
        --panel-alt: #19253a;
        --border: #1f2937;
        --text: #f8fafc;
        --muted: #94a3b8;
        --accent: #38bdf8;
        --alert: #ef4444;
    }
    html, body, [data-testid="stAppViewContainer"], .main, [data-testid="block-container"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', 'Public Sans', sans-serif !important;
    }
    #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .layout-shell {
        padding: 24px 48px 72px 48px;
    }
    .control-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 18px 24px;
        margin-bottom: 28px;
    }
    .control-bar .brand {
        font-size: 1.6rem;
        font-weight: 600;
    }
    .control-bar .meta {
        text-align: right;
        color: var(--muted);
        font-size: 0.9rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 32px;
    }
    .metric-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px 18px;
        min-height: 110px;
    }
    .metric-card h4 {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: var(--muted);
        margin-bottom: 10px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 600;
    }
    .metric-card .value.alert {
        color: var(--alert);
    }
    .metric-card .subtext {
        font-size: 0.85rem;
        color: var(--muted);
        margin-top: 6px;
    }
    .panel-frame {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 28px;
    }
    .panel-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 18px;
    }
    .timeline {
        display: flex;
        flex-direction: column;
        gap: 14px;
    }
    .timeline-entry {
        display: flex;
        gap: 14px;
        border-left: 2px solid var(--border);
        padding-left: 12px;
    }
    .timeline-entry time {
        font-size: 0.85rem;
        color: var(--muted);
        min-width: 120px;
    }
    .timeline-entry p {
        margin: 0;
        font-size: 0.95rem;
    }
    .dependency-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .dependency-table th,
    .dependency-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .dependency-table th {
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        color: var(--muted);
    }
    .badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge.low {
        background: rgba(56,189,248,0.12);
        color: var(--accent);
    }
    .badge.critical {
        background: rgba(239,68,68,0.12);
        color: var(--alert);
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
                "dependency": "db-cluster-01",
                "severity": "critical",
                "reason": "High latency",
            },
            {
                "ts": (datetime.utcnow() - timedelta(minutes=2)).isoformat(timespec="seconds"),
                "service": "auth-service",
                "dependency": "redis-latency",
                "severity": "warning",
                "reason": "Error budget burn",
            },
        ]
    )


def _fallback_topology() -> tuple[list[dict[str, str]], list[dict[str, float]]]:
    nodes = [
        {"name": "edge-lb", "health": "HEALTHY"},
        {"name": "web-tier", "health": "HEALTHY"},
        {"name": "auth-service", "health": "ANOMALY"},
        {"name": "payments", "health": "HEALTHY"},
        {"name": "db-cluster-01", "health": "ANOMALY"},
    ]
    edges = [
        {"source": "edge-lb", "target": "web-tier", "latency": 120.0, "anomaly": False},
        {"source": "web-tier", "target": "auth-service", "latency": 220.0, "anomaly": True},
        {"source": "web-tier", "target": "payments", "latency": 180.0, "anomaly": False},
        {"source": "auth-service", "target": "db-cluster-01", "latency": 460.0, "anomaly": True},
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


def _causal_entropy(nodes: list[dict[str, str]], edges: list[dict[str, float]]) -> float:
    if not nodes or not edges:
        return 0.0
    anomaly_edges = sum(1 for edge in edges if edge.get("anomaly"))
    weighted_latency = sum(edge.get("latency", 0.0) for edge in edges)
    entropy = np.log1p(weighted_latency / max(1, len(nodes)) + anomaly_edges * 50)
    return round(min(entropy * 3, 99.9), 1)


def _timeline(alerts: pd.DataFrame) -> list[dict[str, datetime]]:
    events: list[dict[str, datetime]] = []
    for _, row in alerts.iterrows():
        ts_raw = row["ts"]
        try:
            ts_obj = datetime.fromisoformat(ts_raw.replace("Z", ""))
        except ValueError:
            ts_obj = datetime.utcnow()
        events.append({"time": ts_obj, "text": f"{row['reason']} detected in {row['dependency'].upper()}"})
        events.append({"time": ts_obj, "text": f"SentriNode identified '{row['service']}' as causal source."})
    return sorted(events, key=lambda item: item["time"], reverse=True)


def _dependency_matrix(nodes: list[dict[str, str]], edges: list[dict[str, float]]) -> pd.DataFrame:
    dep_count: dict[str, int] = {}
    for edge in edges:
        dep_count[edge["source"]] = dep_count.get(edge["source"], 0) + 1
    rows = []
    for node in nodes:
        name = node["name"]
        deps = dep_count.get(name, 0)
        is_critical = node.get("health") == "ANOMALY"
        risk = "CRITICAL" if is_critical else ("MEDIUM" if deps >= 3 else "LOW")
        status = "Latent" if is_critical else "Healthy"
        rows.append({
            "Service": name,
            "Dependencies": deps,
            "Risk Level": risk,
            "Status": status,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {"CRITICAL": 0, "MEDIUM": 1, "LOW": 2}
    df["risk_order"] = df["Risk Level"].map(order).fillna(3)
    return df.sort_values(by=["risk_order", "Dependencies"], ascending=[True, False]).drop(columns=["risk_order"])


metrics_df = _stream_metrics()
alerts_df = _sample_alerts()
connected, nodes_data, edges_data = _load_topology()

active_nodes = len(nodes_data)
latest_latency = metrics_df["p99_latency"].iloc[-1]
causal_entropy = _causal_entropy(nodes_data, edges_data)
anomaly_count = sum(1 for node in nodes_data if node.get("health") == "ANOMALY")

timeline_events = _timeline(alerts_df)
dependency_df = _dependency_matrix(nodes_data, edges_data)

current_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

st.markdown('<div class="layout-shell">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="control-bar">
        <div>
            <div class="brand">SentriNode // Enterprise Control</div>
            <div style="color: var(--muted); font-size:0.95rem;">Observability & causal intelligence</div>
        </div>
        <div class="meta">
            {'Connected' if connected else 'Fallback Topology'}<br/>
            Updated {current_timestamp}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cards = [
    {"label": "Active Nodes", "value": str(active_nodes), "sub": "Telemetry sources"},
    {"label": "System Latency (P99)", "value": f"{latest_latency:.0f} ms", "sub": "Global tail latency"},
    {"label": "Causal Entropy", "value": f"{causal_entropy:.1f}", "sub": "Instability index"},
    {"label": "Anomaly Count", "value": str(anomaly_count), "sub": "Root causes", "alert": anomaly_count > 0},
]

cards_html = "<div class='metric-grid'>" + "".join(
    f"""
    <div class='metric-card'>
        <h4>{card['label']}</h4>
        <div class='value{' alert' if card.get('alert') else ''}'>{card['value']}</div>
        <div class='subtext'>{card['sub']}</div>
    </div>
    """
    for card in metric_cards
) + "</div>"

st.markdown(cards_html, unsafe_allow_html=True)

# --- Graph ---
st.markdown('<div class="panel-frame">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">causal topology map</div>', unsafe_allow_html=True)

if not all([Config, Node, Edge, agraph]):
    st.warning("Install streamlit-agraph to render the service graph.")
    st.dataframe(pd.DataFrame(nodes_data))
else:
    node_index = {node["name"]: node.copy() for node in nodes_data}
    incoming_anomaly: dict[str, int] = {node["name"]: 0 for node in nodes_data}
    for edge in edges_data:
        if edge.get("anomaly"):
            incoming_anomaly[edge["target"]] = incoming_anomaly.get(edge["target"], 0) + 1
            node_index.setdefault(edge["source"], {"name": edge["source"], "health": "ANOMALY"})
            node_index.setdefault(edge["target"], {"name": edge["target"], "health": "ANOMALY"})
            node_index[edge["source"]]["health"] = "ANOMALY"
            node_index[edge["target"]]["health"] = "ANOMALY"

    root_candidates = [name for name, info in node_index.items() if info.get("health") == "ANOMALY"]
    root_causes = [name for name in root_candidates if incoming_anomaly.get(name, 0) == 0]
    if not root_causes:
        root_causes = root_candidates

    node_objs = []
    for name, info in node_index.items():
        is_root = name in root_causes
        color_block = {
            "background": "#1f2937" if not is_root else "#ef4444",
            "border": "#48607a" if not is_root else "#ef4444",
            "highlight": {"background": "#334155", "border": "#38bdf8"},
        }
        node_objs.append(
            Node(
                id=name,
                label=name,
                size=30,
                color=color_block,
                borderWidth=4 if is_root else 2,
                font={"color": "#f8fafc", "size": 16},
                shape="dot",
            )
        )

    edge_objs = []
    for edge in edges_data:
        latency = edge.get("latency", 0.0)
        width = max(1, min(8, latency / 80))
        color = "#ef4444" if edge.get("anomaly") else "#38bdf8"
        edge_objs.append(
            Edge(
                source=edge["source"],
                target=edge["target"],
                color=color,
                width=width,
                smooth={"enabled": False},
                arrows="to",
                title=f"{edge['source']} -> {edge['target']} · {latency:.0f} ms",
            )
        )

    graph_config = Config(
        width=1600,
        height=600,
        directed=True,
        physics=False,
        hierarchical=True,
        hierarchicalSpacing=180,
        hierarchicalDirection="LR",
        nodeHighlightBehavior=True,
        highlightColor="#38bdf8",
        collapsible=False,
    )
    agraph(node_objs, edge_objs, graph_config)

st.markdown('</div>', unsafe_allow_html=True)

bottom_cols = st.columns(2)
with bottom_cols[0]:
    st.markdown('<div class="panel-frame"><div class="panel-title">root cause timeline</div>', unsafe_allow_html=True)
    if timeline_events:
        entries_html = "<div class='timeline'>" + "".join(
            f"""
            <div class='timeline-entry'>
                <time>{event['time'].strftime('%H:%M:%S')}</time>
                <p>{event['text']}</p>
            </div>
            """
            for event in timeline_events
        ) + "</div>"
        st.markdown(entries_html, unsafe_allow_html=True)
    else:
        st.write("No recent events.")
    st.markdown('</div>', unsafe_allow_html=True)

with bottom_cols[1]:
    st.markdown('<div class="panel-frame"><div class="panel-title">dependency health</div>', unsafe_allow_html=True)
    if dependency_df.empty:
        st.write("No dependency data available.")
    else:
        table_html = "<table class='dependency-table'><thead><tr>" + "".join(
            f"<th>{col}</th>" for col in dependency_df.columns
        ) + "</tr></thead><tbody>"
        for _, row in dependency_df.iterrows():
            risk_class = "critical" if row["Risk Level"] == "CRITICAL" else "low"
            table_html += "<tr>"
            table_html += f"<td>{row['Service']}</td>"
            table_html += f"<td>{row['Dependencies']}</td>"
            table_html += f"<td><span class='badge {risk_class}'>{row['Risk Level']}</span></td>"
            status_icon = "⚠️" if row["Status"].lower() != "healthy" else "✅"
            table_html += f"<td>{status_icon} {row['Status']}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
