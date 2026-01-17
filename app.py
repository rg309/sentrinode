import os

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

try:
    from streamlit_agraph import agraph, Config, Edge, Node
except Exception:  # pragma: no cover - optional dependency
    agraph = Config = Edge = Node = None

st.set_page_config(layout="wide")

# --- INITIAL STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False


def _neo4j_driver():
    try:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD")
        if not pwd:
            return None
        return GraphDatabase.driver(uri, auth=(user, pwd))
    except Exception as exc:
        st.warning(f"Neo4j connect failed: {exc}")
        return None


HERO_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@700&display=swap');
.sentri-hero {
    font-family: 'Syncopate', sans-serif;
    letter-spacing: 0.65rem;
    font-size: 2.75rem;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 1.5rem;
    background: linear-gradient(120deg, #0ea5e9, #38bdf8, #e0f2fe, #0ea5e9);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: sentriShift 6s ease-in-out infinite;
}
@keyframes sentriShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
</style>
"""


def render_hero(text: str) -> None:
    st.markdown(HERO_STYLE, unsafe_allow_html=True)
    st.markdown(f"<div class='sentri-hero'>{text}</div>", unsafe_allow_html=True)


# --- UI LOGIC ---
def show_login():
    render_hero("SENTRINODE")
    st.title("SentriNode Login")
    with st.form("login_form"):
        st.text_input("Username")
        st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            # We will add DB check here later
            st.session_state.logged_in = True
            st.rerun()
    if st.button("Create an account"):
        st.session_state.show_signup = True
        st.rerun()


def show_signup():
    st.title("Create SentriNode Account")
    with st.form("signup_form"):
        st.text_input("Email")
        st.text_input("New Username")
        st.text_input("New Password", type="password")
        if st.form_submit_button("Register Node"):
            # We will add DB save here later
            st.success("Account created! Please log in.")
            st.session_state.show_signup = False
            st.rerun()
    if st.button("Back to Login"):
        st.session_state.show_signup = False
        st.rerun()


def show_dashboard():
    render_hero("SentriNode Operational Command")
    nodes_count = 0
    rels_count = 0
    graph_nodes: list[Node] = []
    graph_edges: list[Edge] = []
    driver = _neo4j_driver()
    if driver:
        try:
            with driver.session() as session:
                nodes_record = session.run("MATCH (n) RETURN count(n)").single()
                rels_record = session.run("MATCH ()-->() RETURN count(*)").single()
                nodes_count = nodes_record[0] if nodes_record else 0
                rels_count = rels_record[0] if rels_record else 0

                if agraph and Node and Edge and Config:
                    node_map: dict[int, Node] = {}
                    edges: list[Edge] = []
                    records = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 15")
                    for record in records:
                        n_start = record["n"]
                        rel = record["r"]
                        n_end = record["m"]
                        for point in (n_start, n_end):
                            if point.id not in node_map:
                                label = point.get("name") or point.get("id") or f"node-{point.id}"
                                node_map[point.id] = Node(
                                    id=str(point.id),
                                    label=str(label),
                                    size=18,
                                    color="#38bdf8",
                                )
                        edges.append(
                            Edge(
                                source=str(n_start.id),
                                target=str(n_end.id),
                                title=rel.type,
                                color="#22d3ee",
                            )
                        )
                    graph_nodes = list(node_map.values())
                    graph_edges = edges
        except (ServiceUnavailable, Neo4jError, ValueError):
            pass
        finally:
            driver.close()

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Nodes", f"{nodes_count:,}", delta="+5")
    col2.metric("Total Connections", f"{rels_count:,}")
    col3.metric("System Health", "Optimal")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Node Distribution")
        chart_data = pd.DataFrame(
            {
                "north": [400, 420, 430, 445, 460],
                "south": [320, 330, 335, 350, 360],
                "edge": [510, 515, 520, 525, 530],
            }
        )
        st.area_chart(chart_data, height=240, use_container_width=True)
        st.subheader("Node Network Map")
        if graph_nodes and graph_edges and agraph and Config:
            config = Config(width=900, height=360, directed=True, physics=True, hierarchical=False)
            agraph(nodes=graph_nodes, edges=graph_edges, config=config)
        else:
            st.info("Graph data unavailable. Configure Neo4j to view live topology.")
    with right:
        st.subheader("Recent Alerts")
        with st.expander("Security Feed", expanded=True):
            alerts = [
                "02:10 - Edge gateway elevated CPU",
                "01:52 - Auth service latency spike resolved",
                "01:33 - New node authenticated (plant-west)",
                "01:05 - Billing service deployed revision v3.4.1",
            ]
            for alert in alerts:
                st.write(f"• {alert}")


# --- MAIN NAVIGATION ---
if not st.session_state.logged_in:
    if st.session_state.show_signup:
        show_signup()
    else:
        show_login()
else:
    show_dashboard()
    with st.sidebar:
        st.title("Console")
        st.caption("Session Controls")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
