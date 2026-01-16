#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="SENTRINODE", layout="wide")

import os
from datetime import datetime

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

try:  # Optional service graph rendering
    from streamlit_agraph import agraph, Config, Edge, Node
except Exception:  # pragma: no cover - optional dependency
    agraph = Node = Edge = Config = None


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip().rstrip("/")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
LICENSE_SERIAL = NEO4J_PASSWORD  # Requirement: gate checks against the Neo4j password value


GLOBAL_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Syncopate:wght@700&display=swap');

.stApp, html, body {
    background-color: #010714;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
.block-container {
    padding-top: 0rem !important;
    padding-left: 3.25rem;
    padding-right: 3.25rem;
}
header[data-testid="stHeader"], footer, #MainMenu {
    display: none;
}
section[data-testid="stSidebar"] {
    background: #020c1f;
    border-right: 1px solid rgba(148, 163, 184, 0.2);
}
.sentri-logo {
    font-family: 'Syncopate', sans-serif;
    text-transform: uppercase;
    letter-spacing: 10px;
    background: linear-gradient(115deg, #03caff, #ffffff);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 0;
}
.sentri-logo.main {
    font-size: 2.4rem;
}
.sentri-logo.small {
    font-size: 1.2rem;
    letter-spacing: 8px;
}
.logo-left {
    margin: 32px 0 12px 0;
}
.logo-center {
    text-align: center;
    margin: 48px 0 16px 0;
}
.logo-caption {
    text-transform: uppercase;
    letter-spacing: 0.35em;
    font-size: 0.75rem;
    color: #94a3b8;
    margin-bottom: 32px;
}
.onboarding-stage {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem 1rem 3rem;
}
.onboarding-card {
    width: 100%;
    max-width: 460px;
    background: rgba(8, 15, 35, 0.95);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 35px 75px rgba(0, 0, 0, 0.55);
}
.onboarding-card h3 {
    margin: 0 0 12px 0;
    letter-spacing: 0.4em;
    font-size: 0.95rem;
    text-transform: uppercase;
}
.onboarding-card p {
    margin-top: 0;
    color: #94a3b8;
    letter-spacing: 0.05em;
}
.form-error {
    margin-top: 18px;
    padding: 10px 14px;
    border-radius: 8px;
    background: rgba(248, 113, 113, 0.08);
    border: 1px solid rgba(248, 113, 113, 0.35);
    color: #fecaca;
    font-size: 0.85rem;
}
.panel {
    background: rgba(13, 17, 31, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
}
.panel h4 {
    margin-top: 0;
    text-transform: uppercase;
    letter-spacing: 0.3em;
    font-size: 0.8rem;
    color: #cbd5f5;
}
.event-log, .table-panel {
    background: rgba(6, 9, 20, 0.9);
    border-radius: 14px;
    padding: 1.2rem;
    border: 1px solid rgba(148, 163, 184, 0.14);
}
.event-log pre {
    font-size: 0.85rem;
}
</style>
"""


def _inject_styles() -> None:
    st.markdown(GLOBAL_STYLE, unsafe_allow_html=True)


def _render_logo(*, centered: bool = False, caption: str | None = None) -> None:
    alignment = "logo-center" if centered else "logo-left"
    st.markdown(
        f"<div class='{alignment}'><div class='sentri-logo main'>SENTRINODE</div></div>",
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(f"<div class='logo-caption'>{caption}</div>", unsafe_allow_html=True)


def _neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _license_registered(serial: str) -> bool:
    if not serial:
        return False
    driver = None
    try:
        driver = _neo4j_driver()
        with driver.session() as session:
            record = session.run(
                "MATCH (l:License {serial:$serial}) RETURN l.serial AS serial LIMIT 1",
                serial=serial,
            ).single()
        return bool(record)
    except (ServiceUnavailable, Neo4jError, ValueError):
        return False
    finally:
        if driver:
            driver.close()


def _create_license(serial: str, admin: str, company: str, email: str) -> bool:
    driver = None
    try:
        driver = _neo4j_driver()
        with driver.session() as session:
            session.run(
                """
                MERGE (l:License {serial:$serial})
                ON CREATE SET l.created_at = timestamp()
                SET l.admin = $admin,
                    l.company = $company,
                    l.email = $email,
                    l.status = 'active',
                    l.updated = timestamp(),
                    l.last_seen = timestamp()
                """,
                serial=serial,
                admin=admin.strip(),
                company=company.strip(),
                email=email.strip(),
            )
        return True
    except (ServiceUnavailable, Neo4jError, ValueError):
        return False
    finally:
        if driver:
            driver.close()


@st.cache_data(ttl=30)
def _get_license_profile(serial: str) -> dict[str, object] | None:
    if not serial:
        return None
    driver = None
    try:
        driver = _neo4j_driver()
        with driver.session() as session:
            record = session.run(
                """
                MATCH (l:License {serial:$serial})
                RETURN l.admin AS admin,
                       l.company AS company,
                       l.email AS email,
                       l.status AS status,
                       l.type AS type
                """,
                serial=serial,
            ).single()
        return dict(record) if record else None
    except (ServiceUnavailable, Neo4jError, ValueError):
        return None
    finally:
        if driver:
            driver.close()


@st.cache_data(ttl=45)
def _fetch_topology() -> tuple[bool, list[dict[str, object]]]:
    driver = None
    try:
        driver = _neo4j_driver()
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
            {"source": "edge-lb", "target": "api-gateway", "rel": "ROUTES", "latency": 70.0},
            {"source": "api-gateway", "target": "auth", "rel": "CALLS", "latency": 210.0},
            {"source": "api-gateway", "target": "billing", "rel": "CALLS", "latency": 260.0},
            {"source": "billing", "target": "db", "rel": "WRITES", "latency": 430.0},
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
    system_health = max(0.0, 100.0 - anomaly_count * 7.5)
    return active_nodes, p99_latency, system_health, anomaly_count


def _event_log(records: list[dict[str, object]]) -> str:
    lines: list[str] = []
    now = datetime.utcnow()
    for idx, entry in enumerate(records[:6]):
        ts = now.strftime("%H:%M:%S")
        latency = entry.get("latency", 0)
        lines.append(
            f"{ts} · TRACE · {entry['source'].upper()} ➝ {entry['target'].upper()} · {latency:.0f}ms"
        )
        if idx == 0 and latency >= 400:
            lines.append(f"{ts} · ALERT · Elevated latency on {entry['target'].upper()}")
    if not lines:
        lines.append(f"{now.strftime('%H:%M:%S')} · INFO · No telemetry ingested")
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
        status = "Stable"
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


def _render_onboarding(serial: str) -> None:
    st.markdown("<div class='onboarding-stage'><div class='onboarding-card'>", unsafe_allow_html=True)
    _render_logo(centered=True)
    st.markdown("<h3>System Onboarding</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p>Authenticate this appliance with the licensing graph to unlock telemetry and analytics.</p>",
        unsafe_allow_html=True,
    )
    error_msg = ""
    with st.form("system-onboarding"):
        admin = st.text_input("Administrator Name")
        company = st.text_input("Company / Facility")
        email = st.text_input("Operations Email")
        submitted = st.form_submit_button("Authorize Node", use_container_width=True)
        if submitted:
            if not admin.strip() or not company.strip() or not email.strip():
                error_msg = "All fields are required."
            elif _create_license(serial, admin, company, email):
                st.success("License registered. Reloading console...")
                st.rerun()
            else:
                error_msg = "Unable to reach Neo4j to register this node."
    if error_msg:
        st.markdown(f"<div class='form-error'>{error_msg}</div>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


def _render_sidebar() -> str:
    st.sidebar.markdown(
        "<div class='sentri-logo small' style='display:block;text-align:center;padding:20px 10px 14px;'>SENTRINODE</div>",
        unsafe_allow_html=True,
    )
    return st.sidebar.radio("Navigation", ("Metrics", "Account"), label_visibility="collapsed")


def _render_service_graph(records: list[dict[str, object]]) -> None:
    if not (agraph and Node and Edge and Config):
        st.info("Install streamlit-agraph to visualize the real-time topology.")
        return
    nodes: list[Node] = []
    edges: list[Edge] = []
    known: set[str] = set()
    for entry in records:
        for name in (entry["source"], entry["target"]):
            if name not in known:
                nodes.append(Node(id=name, label=name, size=18, color="#64748b"))
                known.add(name)
        edges.append(
            Edge(
                source=entry["source"],
                target=entry["target"],
                color="#38bdf8",
                title=f"{entry.get('rel', 'link')} · {entry.get('latency', 0):.0f}ms",
            )
        )
    config = Config(width=1200, height=480, directed=True, physics=True, hierarchical=True)
    agraph(nodes=nodes, edges=edges, config=config)


def _render_dashboard() -> None:
    connected, topology = _fetch_topology()
    active_nodes, p99_latency, system_health, anomaly_count = _compute_metrics(topology)
    log_output = _event_log(topology)
    dep_table = _dependency_table(topology)

    _render_logo(caption="Industrial Dashboard Metrics")
    cols = st.columns(4)
    cols[0].metric("Active Nodes", str(active_nodes), "Live" if connected else "Fallback")
    cols[1].metric("P99 Latency", f"{p99_latency:.0f}ms", "-4ms")
    cols[2].metric("System Health", f"{system_health:.1f}%", "Stable")
    cols[3].metric("Anomalies", str(anomaly_count), "Clear", delta_color="inverse")

    st.divider()
    st.markdown("#### Infrastructure Topology")
    _render_service_graph(topology)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Causal Event Log")
        st.code(log_output, language="bash")
    with right:
        st.markdown("#### Dependency Risk")
        if dep_table.empty:
            st.info("No dependency data available.")
        else:
            st.table(dep_table)


def _render_account(serial: str) -> None:
    _render_logo(caption="Account Console")
    st.markdown(f"**Hardware Key:** `{serial or 'Unavailable'}`")
    profile = _get_license_profile(serial)
    if not profile:
        st.info("No account metadata stored yet. The license node will populate after registration.")
        return
    info = {
        "Administrator": profile.get("admin") or "—",
        "Company": profile.get("company") or "—",
        "Email": profile.get("email") or "—",
        "License Type": profile.get("type") or "trial",
        "Status": profile.get("status") or "active",
    }
    for label, value in info.items():
        st.markdown(f"**{label}**")
        st.write(value)
        st.divider()


def main() -> None:
    registered = _license_registered(LICENSE_SERIAL)
    _inject_styles()
    if not registered:
        _render_onboarding(LICENSE_SERIAL)
    view = _render_sidebar()
    if view == "Account":
        _render_account(LICENSE_SERIAL)
    else:
        _render_dashboard()


if __name__ == "__main__":
    main()
