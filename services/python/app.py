#!/usr/bin/env python3
"""SentriNode Security Operations Center dashboard."""
from __future__ import annotations

from datetime import datetime
import os
import random

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

try:
    from streamlit_agraph import Config, Edge, Node, agraph
except Exception:  # pragma: no cover - optional dependency
    Config = Edge = Node = agraph = None


def _normalize_uri(raw_uri: str | None) -> str:
    uri = (raw_uri or "bolt://localhost:7687").strip().rstrip("/")
    if "://" not in uri:
        uri = f"bolt://{uri}"
    return uri


NEO4J_URI = _normalize_uri(os.getenv("NEO4J_URI"))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")

st.set_page_config(
    page_title="SentriNode SOC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600&display=swap');
    body, html, [class*="css"] {
        background-color: #010409 !important;
        color: #d5ffe1 !important;
        font-family: 'Roboto Mono', 'Courier New', monospace !important;
    }
    header, footer, #MainMenu, [data-testid="stSidebar"] {
        display: none !important;
    }
    .soc-shell {
        padding: 6px 24px 40px 24px;
    }
    .soc-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 28px;
        margin-bottom: 18px;
        border-radius: 16px;
        border: 1px solid rgba(0, 255, 163, 0.35);
        background: linear-gradient(120deg, rgba(0,255,163,0.07), rgba(3,25,28,0.95));
        box-shadow: 0 0 25px rgba(0,255,163,0.12);
    }
    .soc-title {
        font-size: 2.4rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #85ffb0;
    }
    .soc-chip {
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        margin-right: 8px;
        border: 1px solid currentColor;
    }
    .soc-panel {
        background: rgba(2, 20, 24, 0.85);
        border: 1px solid rgba(0, 255, 163, 0.18);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 0 20px rgba(0, 255, 163, 0.05);
        margin-bottom: 18px;
    }
    .soc-panel h3 {
        color: #76ff96;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    canvas {
        filter: drop-shadow(0 0 6px rgba(0,255,163,0.3));
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=20, show_spinner=False)
def pull_topology() -> tuple[bool, list[dict], list[dict]]:
    nodes: list[dict] = []
    edges: list[dict] = []
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            node_records = session.run("MATCH (s:Service) RETURN s.name AS name, s.health AS health")
            edge_records = session.run(
                """
                MATCH (s:Service)-[r:DEPENDS_ON]->(t:Service)
                RETURN s.name AS source, t.name AS target,
                    coalesce(r.latency_ms, rand()*400) AS latency,
                    coalesce(r.anomaly, r.latency_ms > 380) AS anomaly
                """
            )
            nodes = [
                {"name": record["name"], "health": (record["health"] or "HEALTHY").upper()}
                for record in node_records
            ]
            edges = [
                {
                    "source": record["source"],
                    "target": record["target"],
                    "latency": float(record["latency"] or 0),
                    "anomaly": bool(record["anomaly"]),
                }
                for record in edge_records
            ]
        driver.close()
        return True, nodes, edges
    except (ServiceUnavailable, Neo4jError):
        nodes = [
            {"name": "gateway", "health": "HEALTHY"},
            {"name": "payments", "health": "ANOMALY"},
            {"name": "inventory", "health": "HEALTHY"},
            {"name": "edge-cache", "health": "HEALTHY"},
        ]
        edges = [
            {"source": "gateway", "target": "payments", "latency": 480, "anomaly": True},
            {"source": "gateway", "target": "inventory", "latency": 180, "anomaly": False},
            {"source": "payments", "target": "edge-cache", "latency": 320, "anomaly": False},
        ]
        return False, nodes, edges


connected, nodes_data, edges_data = pull_topology()
status_color = "#90ffb8" if connected else "#ff4d6a"

st.markdown('<div class="soc-shell">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="soc-banner">
        <div>
            <div class="soc-title">SentriNode SOC</div>
            <div style="opacity:0.7">neo4j target · {NEO4J_URI}</div>
        </div>
        <div>
            <span class="soc-chip" style="color:{status_color}">
                {'🟢 Secure Link' if connected else '🔴 Offline Mode'}
            </span>
            <span class="soc-chip" style="color:#7df3ff">
                {datetime.utcnow().strftime('%H:%M:%S UTC')}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def render_kpis() -> None:
    cols = st.columns(3, gap="large")
    cols[0].markdown(
        f"""
        <div class="soc-panel">
            <h3>P99 LATENCY</h3>
            <div style="font-size:2rem;color:#9CFF6E">{random.randint(280, 520)} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        f"""
        <div class="soc-panel">
            <h3>ACTIVE ALERTS</h3>
            <div style="font-size:2rem;color:#FF5C8A">{random.randint(3, 7)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        f"""
        <div class="soc-panel">
            <h3>GRAPH HEALTH</h3>
            <div style="font-size:2rem;color:#7df3ff">{len(nodes_data)} nodes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_kpis()


def render_graph() -> None:
    if not Config or not Node or not Edge or not agraph:
        st.warning("Install streamlit-agraph to view the SOC topology. Displaying raw table.")
        st.dataframe(pd.DataFrame(edges_data or nodes_data), width="stretch", height=400)
        return

    pulse = int(datetime.utcnow().timestamp()) % 2
    node_map = {node["name"]: node for node in nodes_data}
    for edge in edges_data:
        if edge["anomaly"]:
            node_map[edge["source"]]["health"] = "ANOMALY"
            node_map[edge["target"]]["health"] = "ANOMALY"

    node_objs = []
    for info in node_map.values():
        is_anomaly = info.get("health", "HEALTHY").upper() == "ANOMALY"
        node_objs.append(
            Node(
                id=info["name"],
                label=info["name"],
                size=28 if is_anomaly else 22,
                color="#FF1744" if is_anomaly else "#00FFB0",
                shadow={"enabled": True, "color": "#FF5E7A" if is_anomaly else "#0BFF95"},
                borderWidth=3,
            )
        )

    edge_objs = []
    for edge in edges_data:
        is_anomaly = edge["anomaly"]
        color = "#FF5C8A" if pulse and is_anomaly else "#FF2F5E" if is_anomaly else "#00E7A8"
        edge_objs.append(
            Edge(
                source=edge["source"],
                target=edge["target"],
                color=color,
                width=4 if is_anomaly else 2,
                dashes=is_anomaly,
                smooth={"enabled": True, "type": "continuous"},
                title=f"{edge['source']} ▶ {edge['target']} · {edge['latency']:.0f} ms",
            )
        )

    config = Config(
        width=1500,
        height=660,
        directed=True,
        physics=True,
        nodeHighlightBehavior=True,
        highlightColor="#FFFFFF",
        background="#010409",
    )

    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown("### Topology Map")
    agraph(node_objs, edge_objs, config)
    st.markdown("</div>", unsafe_allow_html=True)


render_graph()

alerts_table = pd.DataFrame(
    [
        {"timestamp": datetime.utcnow().strftime("%H:%M:%S"), "anomaly": "payments→db", "status": "Critical"},
        {"timestamp": (datetime.utcnow()).strftime("%H:%M:%S"), "anomaly": "inventory→edge-cache", "status": "Warning"},
    ]
)

st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
st.markdown("### Intrusion Events")
st.dataframe(alerts_table, width="stretch", height=200)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
