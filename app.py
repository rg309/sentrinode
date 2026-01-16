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
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Syncopate:wght@700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" crossorigin="anonymous" referrerpolicy="no-referrer">
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    header, footer, #MainMenu {visibility: hidden;}
    .main .block-container {
        padding-top: 0 !important;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-bottom: 3rem;
    }
    section[data-testid="stSidebar"] {
        background: #080f1c;
        border-right: 1px solid #1b2440;
        font-family: 'Inter', sans-serif;
    }
    /* Hide sidebar during registration */
    .registration-mode section[data-testid="stSidebar"] {
        display: none !important;
    }
    .registration-mode .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0 !important;
    }
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
    .brand-title, .sidebar-brand {
        font-size: 1.1rem;
        display: inline-flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.25em;
        font-family: 'Syncopate', sans-serif;
    }
    .brand-title .brand-name,
    .sidebar-brand .brand-name {
        letter-spacing: 8px;
        background: linear-gradient(90deg, #06b6d4, #ffffff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: 700;
        text-transform: uppercase;
    }
    .brand-title .brand-divider,
    .sidebar-brand .brand-divider {
        letter-spacing: 0.4em;
        color: #38bdf8;
        font-weight: 400;
    }
    .brand-title .brand-tag,
    .sidebar-brand .brand-tag {
        letter-spacing: 0.35em;
        color: #cbd5f5;
        font-weight: 500;
    }
    .sidebar-brand {
        flex-direction: column;
        align-items: flex-start;
        gap: 4px;
        padding: 20px 16px 8px;
        border-bottom: 1px solid #1c2a4a;
        width: 100%;
    }
    .registration-wrapper {
        min-height: 100vh;
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 16px;
        background: radial-gradient(circle at top, rgba(14,165,233,0.08), transparent 60%);
        margin: 0;
        width: 100%;
        box-sizing: border-box;
        overflow-y: auto;
    }
    .registration-box {
        width: 100%;
        max-width: 420px;
        background-color: #111a2c;
        border: 1px solid #1f2a3d;
        border-radius: 10px;
        padding: 20px 18px;
        box-shadow: 0 25px 50px rgba(15,23,42,0.65);
        margin: auto;
        box-sizing: border-box;
    }
    .registration-eyebrow {
        letter-spacing: 0.6em;
        color: #38bdf8;
        font-size: 0.7rem;
        text-transform: uppercase;
        margin-bottom: 6px;
        margin-top: 0;
    }
    .registration-box h1 {
        margin-bottom: 6px;
        margin-top: 0;
        letter-spacing: 8px;
        font-size: 1.2rem;
        text-align: left;
        color: #f8fafc;
        font-family: 'Syncopate', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #06b6d4, #ffffff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-transform: uppercase;
    }
    .registration-subtext {
        color:#94a3b8;
        margin-bottom:14px;
        font-size:0.8rem;
        letter-spacing:0.08em;
        text-transform:none;
        line-height: 1.3;
    }
    .input-label {
        font-size: 0.75rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #cbd5f5;
        margin-bottom: 6px;
        margin-top: 12px;
        display: block;
    }
    .input-label:first-of-type {
        margin-top: 0;
    }
    .registration-box .stTextInput {
        margin-bottom: 0;
    }
    .registration-box .stTextInput>div>div>input {
        background: #0b1425;
        border: 1px solid #1f2a3d;
        border-radius: 6px;
        color: #f8fafc;
        padding: 8px 12px;
        font-size: 0.9rem;
        margin-bottom: 0;
    }
    .registration-box .stTextInput>div>div>input:focus {
        border-color: #38bdf8;
        outline: none;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1);
    }
    .registration-box .stTextInput>div>div>input::placeholder {
        color: #64748b;
    }
    .registration-box .stForm {
        margin-bottom: 0;
    }
    .registration-box button, .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #0ea5e9) !important;
        color: #f8fafc !important;
        border: none;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 10px !important;
        margin-top: 12px;
        margin-bottom: 0;
    }
    .nav-heading {
        font-size: 0.85rem;
        letter-spacing: 0.4em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 16px 0 8px;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 8px;
    }
    section[data-testid="stSidebar"] .stRadio label {
        background: #0f172a;
        border: 1px solid transparent;
        padding: 8px 12px;
        border-radius: 6px;
        transition: border 0.2s ease, background 0.2s ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        border-color: #1d4ed8;
    }
    section[data-testid="stSidebar"] .stRadio label span {
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 10px;
    }
    section[data-testid="stSidebar"] .stRadio label:nth-of-type(1) span:before {
        font-family: "Font Awesome 6 Free";
        font-weight: 900;
        content: "\\f080";
        color: #38bdf8;
    }
    section[data-testid="stSidebar"] .stRadio label:nth-of-type(2) span:before {
        font-family: "Font Awesome 6 Free";
        font-weight: 900;
        content: "\\f007";
        color: #fde68a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar branding - only show if not in registration mode
# This will be conditionally rendered after license check

NEO4J_BOLT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip().rstrip("/")
AUTH_USERNAME = os.getenv("NEO4J_USER", "neo4j")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
LICENSE_SERIAL = os.getenv("LICENSE_SERIAL") or AUTH_PASSWORD


if "registration_error" not in st.session_state:
    st.session_state["registration_error"] = ""
if "edit_profile" not in st.session_state:
    st.session_state["edit_profile"] = False


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


def _fetch_license_status() -> tuple[bool, str | None]:
    """Return (connected, status) for the local license node. Returns None if Hardware ID not found in Neo4j."""
    if not LICENSE_SERIAL:
        return False, None
    if ADMIN_KEY and LICENSE_SERIAL == ADMIN_KEY:
        return True, "paid"
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_BOLT_URI, auth=(AUTH_USERNAME, AUTH_PASSWORD))
        with driver.session() as session:
            record = session.run(
                """
                MATCH (l:License {serial:$serial})
                RETURN l.status AS status
                """,
                serial=LICENSE_SERIAL,
            ).single()
        if not record:
            # Hardware ID not found in Neo4j - needs registration
            return True, None
        status = str(record["status"] or "").lower() or "active"
        return True, status
    except Exception:
        return False, None
    finally:
        if driver:
            driver.close()


@st.cache_data(ttl=30)
def _get_license_profile(serial: str) -> dict[str, object] | None:
    """Return license node details for profile view."""
    if not serial:
        return None
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_BOLT_URI, auth=(AUTH_USERNAME, AUTH_PASSWORD))
        with driver.session() as session:
            record = session.run(
                """
                MATCH (l:License {serial:$serial})
                RETURN l.status AS status,
                       l.type AS type,
                       l.admin AS admin,
                       l.company AS company,
                       l.serial AS serial
                """,
                serial=serial,
            ).single()
        if not record:
            return None
        return {
            "status": record.get("status"),
            "type": record.get("type"),
            "admin": record.get("admin"),
            "company": record.get("company"),
            "serial": record.get("serial") or serial,
        }
    except Exception:
        return None
    finally:
        if driver:
            driver.close()


def _register_license(admin_name: str, company: str) -> bool:
    """Create the license node with provided metadata and unlock immediately."""
    if not LICENSE_SERIAL:
        st.session_state["registration_error"] = "Missing hardware identifier."
        return False
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_BOLT_URI, auth=(AUTH_USERNAME, AUTH_PASSWORD))
        with driver.session() as session:
            session.run(
                """
                MERGE (l:License {serial:$serial})
                ON CREATE SET l.created_at = timestamp()
                SET l.status='active',
                    l.type = coalesce(l.type, 'trial'),
                    l.admin = $admin,
                    l.company = $company,
                    l.updated = timestamp(),
                    l.last_seen = timestamp()
                """,
                serial=LICENSE_SERIAL,
                admin=admin_name.strip(),
                company=company.strip(),
            )
        return True
    except Exception as exc:  # pragma: no cover - Streamlit UI
        st.session_state["registration_error"] = f"Registration failed: {exc}"
        return False
    finally:
        if driver:
            driver.close()


def _update_license_profile(admin_name: str, company: str) -> bool:
    """Persist profile edits to Neo4j."""
    if not LICENSE_SERIAL:
        st.session_state["registration_error"] = "Missing hardware identifier."
        return False
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_BOLT_URI, auth=(AUTH_USERNAME, AUTH_PASSWORD))
        with driver.session() as session:
            session.run(
                """
                MATCH (l:License {serial:$serial})
                SET l.admin = $admin,
                    l.company = $company,
                    l.updated = timestamp()
                """,
                serial=LICENSE_SERIAL,
                admin=admin_name.strip(),
                company=company.strip(),
            )
        return True
    except Exception as exc:  # pragma: no cover - Streamlit UI
        st.error(f"Unable to update profile: {exc}")
        return False
    finally:
        if driver:
            driver.close()


def _reset_local_session() -> None:
    """Clear cached data so the node can re-register."""
    st.cache_data.clear()
    st.session_state.clear()
    st.success("Session reset. Reloading...")
    st.rerun()


def _render_registration() -> None:
    # Hide sidebar during registration and ensure full viewport
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    .stApp {
        overflow: hidden;
    }
    [data-testid="stAppViewContainer"] {
        padding: 0 !important;
    }
    .registration-box .stCaption {
        font-size: 0.75rem !important;
        margin-top: 4px !important;
        margin-bottom: 0 !important;
    }
    .registration-box form {
        margin-bottom: 0;
    }
    .registration-box [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    .registration-box [data-testid="stVerticalBlock"] > div {
        margin-bottom: 0 !important;
    }
    .registration-box .element-container {
        margin-bottom: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="registration-wrapper"><div class="registration-box">', unsafe_allow_html=True)
    st.markdown("<div class='registration-eyebrow'>System Registration</div>", unsafe_allow_html=True)
    st.markdown("<h1>SENTRINODE</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='registration-subtext'>Link this appliance to your SentriNode license. Provide admin contact and deployment location.</p>",
        unsafe_allow_html=True,
    )
    with st.form("registration-form"):
        st.markdown("<label class='input-label'>Admin Name</label>", unsafe_allow_html=True)
        admin_name = st.text_input("Admin Name", value="", label_visibility="collapsed", placeholder="Enter your full name")
        st.markdown("<label class='input-label'>Company / Location</label>", unsafe_allow_html=True)
        company = st.text_input("Company / Location", value="", label_visibility="collapsed", placeholder="Enter company or deployment location")
        submitted = st.form_submit_button("Register Node", use_container_width=True)
        if submitted:
            if not admin_name.strip():
                st.session_state["registration_error"] = "Admin Name is required."
            elif not company.strip():
                st.session_state["registration_error"] = "Company / Location is required."
            elif _register_license(admin_name, company):
                st.session_state["registration_error"] = ""
                st.success("Node registered. Loading console...")
                st.rerun()
    if st.session_state.get("registration_error"):
        st.markdown(
            f"<div style='color: #ef4444; padding: 8px 10px; background: #1c0f0f; border: 1px solid #7f1d1d; border-radius: 6px; margin-top: 10px; margin-bottom: 6px; font-size: 0.8rem;'>{st.session_state['registration_error']}</div>",
            unsafe_allow_html=True
        )
    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
    st.caption("Need assistance? Contact ops@sentrinode.io for enterprise provisioning.")
    st.markdown("</div></div>", unsafe_allow_html=True)


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


def _render_account_settings(license_status: str) -> None:
    st.markdown("## Account Settings")
    profile = _get_license_profile(LICENSE_SERIAL)
    if not profile:
        st.warning("Unable to load account details from Neo4j.")
    status = (profile.get("status") if profile else license_status) or "unknown"
    status_key = status.lower()
    badge_color = "#22c55e"
    if status_key not in ("active", "paid"):
        badge_color = "#fbbf24" if status_key in ("trial", "pending") else "#ef4444"
    
    # Display Name and Company prominently
    admin_name = (profile or {}).get("admin") or "—"
    company = (profile or {}).get("company") or "—"
    
    st.markdown(f"**Name:** {admin_name}")
    st.markdown(f"**Company:** {company}")
    st.markdown("")
    st.markdown(
        f"<span style='padding:6px 14px;border-radius:4px;background:{badge_color};color:#0f172a;font-weight:600;'>Status: Active</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    info = {
        "Admin Name": (profile or {}).get("admin") or "—",
        "Company / Location": (profile or {}).get("company") or "—",
        "Hardware ID": LICENSE_SERIAL or "Unavailable",
        "License Type": (profile or {}).get("type") or "trial",
    }
    for label, value in info.items():
        st.markdown(f"**{label}**")
        st.write(value)
        st.divider()

    if st.button("Edit Profile", disabled=st.session_state.get("edit_profile", False)):
        st.session_state["edit_profile"] = True
        st.rerun()

    if st.session_state.get("edit_profile", False):
        with st.form("edit-profile-form"):
            new_admin = st.text_input("Admin Name", value=(profile or {}).get("admin") or "")
            new_company = st.text_input("Company / Location", value=(profile or {}).get("company") or "")
            save = st.form_submit_button("Save Changes")
            if save:
                if _update_license_profile(new_admin, new_company):
                    st.session_state["edit_profile"] = False
                    st.cache_data.clear()
                    st.success("Profile updated.")
                    st.rerun()
        if st.button("Cancel Edit"):
            st.session_state["edit_profile"] = False
            st.rerun()

    st.write("")
    if st.button("Reset Local Session"):
        _reset_local_session()
connected_license, license_status = _fetch_license_status()
# Show registration form if Hardware ID not found in Neo4j or if license is expired
if license_status is None or license_status == "expired":
    _render_registration()
    st.stop()

# Show sidebar branding and navigation only after registration
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div>
            <span class="brand-name">SENTRINODE</span>
        </div>
        <div>
            <span class="brand-divider">//</span>
            <span class="brand-tag">CAUSAL INTELLIGENCE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
if not connected_license:
    st.sidebar.warning("Unable to reach licensing service. Running in offline mode.")
st.sidebar.markdown("<div class='nav-heading'>Console</div>", unsafe_allow_html=True)
view = st.sidebar.radio("Console", ("📊 Real-Time Metrics", "👤 Account Settings"), index=0, label_visibility="collapsed")
if view == "👤 Account Settings":
    _render_account_settings(license_status or "active")
    st.stop()


connected, topology = _fetch_topology()
active_nodes, p99_latency, system_health, anomaly_count = _compute_metrics(topology)
log_output = _event_log(topology)
dep_table = _dependency_table(topology)

st.markdown(
    """
    <div class="brand-title">
        <div class="brand-name">SENTRINODE</div>
        <div>
            <span class="brand-divider">//</span>
            <span class="brand-tag">CAUSAL INTELLIGENCE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
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
