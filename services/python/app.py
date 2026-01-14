#!/usr/bin/env python3
"""SentriNode Cyberpunk SOC dashboard."""
from __future__ import annotations

from datetime import datetime
import os

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

try:
    from streamlit_agraph import Config, Edge, Node, agraph
except Exception:  # pragma: no cover
    Config = Edge = Node = agraph = None


def _normalize_uri(raw_uri: str | None) -> str:
    uri = (raw_uri or "bolt://localhost:7687").strip().rstrip("/")
    if "://" not in uri:
        uri = f"bolt://{uri}"
    return uri


NEO4J_URI = _normalize_uri(os.getenv("NEO4J_URI"))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")

st.set_page_config(page_title="SentriNode SOC", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] {
        background-color: #000000 !important;
        color: #bbffc8 !important;
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace !important;
        letter-spacing: 0.5px;
    }
    #MainMenu, header, footer, [data-testid="stSidebar"] {
        display: none !important;
    }
    .soc-shell {
        padding: 12px 32px 60px 32px;
    }
    .cyber-banner {
        display:flex;
        justify-content: space-between;
        align-items:center;
        padding: 18px 28px;
        border-radius: 18px;
        border: 1px solid rgba(57, 255, 20, 0.4);
        background: linear-gradient(90deg, rgba(57, 255, 20, 0.1), rgba(0,0,0,0.85), rgba(255,49,49,0.08));
        box-shadow: 0 0 35px rgba(57, 255, 20, 0.25);
    }
    .cyber-title {
        font-size: 2.2rem;
        text-transform: uppercase;
        color: #39FF14;
        text-shadow: 0 0 8px rgba(57,255,20,0.9);
    }
    .glow-chip {
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.85rem;
        border: 1px solid rgba(255,255,255,0.2);
        margin-left: 10px;
        box-shadow: 0 0 12px rgba(57,255,20,0.35);
    }
    .glow {
        text-shadow: 0 0 14px rgba(57,255,20,0.8);
    }
    .panel {
        background: rgba(0,0,0,0.75);
        border: 1px solid rgba(57,255,20,0.18);
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 0 25px rgba(57,255,20,0.15);
        margin-bottom: 20px;
    }
    .metric-glow {
        font-size: 2.4rem;
        color: #39FF14;
        text-shadow: 0 0 12px rgba(57,255,20,0.8);
    }
    .metric-red {
        color: #FF3131;
        text-shadow: 0 0 16px rgba(255,49,49,0.9);
    }
    canvas {
        filter: drop-shadow(0 0 8px rgba(57,255,20,0.4));
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=15, show_spinner=False)
def pull_topology():
    nodes, edges = [], []
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
                {"name": r["name"], "health": (r["health"] or "HEALTHY").upper()}
                for r in node_records
            ]
            edges = [
                {
                    "source": r["source"],
                    "target": r["target"],
                    "latency": float(r["latency"] or 0),
                    "anomaly": bool(r["anomaly"]),
                }
                for r in edge_records
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
status_chip = "glow-chip glow" if connected else "glow-chip metric-red"
st.markdown('<div class="soc-shell">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="cyber-banner">
        <div class="cyber-title">SENTRINODE // CAUSAL_INTELLIGENCE</div>
        <div style="display:flex;align-items:center;color:#39FF14;">
            <div class="{status_chip}">
                {'SYSTEM ACTIVE' if connected else 'LINK FAILURE'}
            </div>
            <div class="glow-chip" style="color:#7DF3FF">
                {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_metrics():
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="panel"><div class="glow">P99 LATENCY</div><div class="metric-glow">412 ms</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="panel"><div class="glow">ACTIVE ANOMALIES</div><div class="metric-glow metric-red">5</div></div>', unsafe_allow_html=True)
    c3.markdown(
        f'<div class="panel"><div class="glow">GRAPH NODES</div><div class="metric-glow">{len(nodes_data)}</div></div>',
        unsafe_allow_html=True,
    )


render_metrics()


def render_graph():
    if not Config or not Node or not Edge or not agraph:
        st.warning("Install streamlit-agraph to render the cyber graph.")
        st.dataframe(pd.DataFrame(edges_data or nodes_data), width="stretch")
        return

    node_map = {node["name"]: node for node in nodes_data}
    for edge in edges_data:
        if edge["anomaly"]:
            node_map[edge["source"]]["health"] = "ANOMALY"
            node_map[edge["target"]]["health"] = "ANOMALY"

    node_objs = []
    for node in node_map.values():
        is_bad = node.get("health") == "ANOMALY"
        node_objs.append(
            Node(
                id=node["name"],
                label=node["name"],
                size=30 if is_bad else 22,
                color="#FF3131" if is_bad else "#39FF14",
                shadow={"enabled": True, "color": "#FF3131" if is_bad else "#39FF14"},
                borderWidth=4 if is_bad else 2,
                font={"color": "#ffffff"},
            )
        )

    edge_objs = []
    for edge in edges_data:
        source_bad = node_map.get(edge["source"], {}).get("health") == "ANOMALY"
        target_bad = node_map.get(edge["target"], {}).get("health") == "ANOMALY"
        anomaly_link = source_bad and target_bad
        color = "#FF3131" if anomaly_link else "rgba(57,255,20,0.4)"
        edge_objs.append(
            Edge(
                source=edge["source"],
                target=edge["target"],
                color=color,
                width=4 if anomaly_link else 1,
                smooth={"enabled": True, "type": "continuous"},
                dashes=anomaly_link,
                title=f"{edge['source']} ▶ {edge['target']} · {edge['latency']:.0f} ms",
            )
        )

    config = Config(
        width=1600,
        height=640,
        directed=True,
        physics=True,
        nodeHighlightBehavior=True,
        highlightColor="#FFFFFF",
        background="#000000",
    )
    st.markdown('<div class="panel"><div class="glow" style="margin-bottom:10px;">CYBER TOPOLOGY</div>', unsafe_allow_html=True)
    agraph(node_objs, edge_objs, config)
    st.markdown("</div>", unsafe_allow_html=True)


render_graph()
st.markdown("</div>", unsafe_allow_html=True)
