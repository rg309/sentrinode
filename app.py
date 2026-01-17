import os
from datetime import datetime

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
if "user_role" not in st.session_state:
    st.session_state.user_role = "user"
if "username" not in st.session_state:
    st.session_state.username = ""


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


def authenticate_user(username: str, password: str) -> tuple[bool, str | None]:
    username = (username or "").strip()
    password = password or ""
    if not username or not password:
        return False, None
    driver = _neo4j_driver()
    if not driver:
        return False, None
    try:
        with driver.session() as session:
            record = session.run(
                """
                MATCH (u:User {username:$username})
                WHERE coalesce(u.password, '') = $password
                RETURN coalesce(u.role, 'user') AS role
                """,
                username=username,
                password=password,
            ).single()
        if record:
            return True, record["role"]
        return False, None
    except (ServiceUnavailable, Neo4jError, ValueError):
        return False, None
    finally:
        driver.close()


def fetch_user_nodes(username: str) -> list[dict[str, object]]:
    username = (username or "").strip()
    if not username:
        return []
    driver = _neo4j_driver()
    if not driver:
        return []
    try:
        with driver.session() as session:
            records = session.run(
                """
                MATCH (u:User {username:$username})-[:OWNS|MONITORS]->(n)
                RETURN coalesce(n.name, n.id) AS node,
                       coalesce(n.status, 'online') AS status,
                       coalesce(n.latency_ms, 0) AS latency
                LIMIT 15
                """,
                username=username,
            )
        return [
            {"Node": record["node"], "Status": record["status"], "Latency (ms)": record["latency"]}
            for record in records
        ]
    except (ServiceUnavailable, Neo4jError, ValueError):
        return []
    finally:
        driver.close()


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
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            with st.spinner("Syncing with SentriNode Network..."):
                success, role = authenticate_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user_role = role or "user"
                st.session_state.username = username.strip()
                st.toast("Console unlocked. Welcome back.", icon="✅")
                st.rerun()
            else:
                st.error("Invalid credentials or unable to verify role.")
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


def render_admin_dashboard() -> None:
    nodes_count = 0
    rels_count = 0
    graph_nodes: list = []
    graph_edges: list = []
    driver = _neo4j_driver()
    if driver:
        try:
            with st.spinner("Syncing with SentriNode Network..."):
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
            with st.spinner("Syncing with SentriNode Network..."):
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


def render_user_dashboard(username: str) -> None:
    st.caption("Personal Node Status")
    with st.spinner("Syncing with SentriNode Network..."):
        nodes = fetch_user_nodes(username)
    summary_cols = st.columns(2)
    summary_cols[0].metric("Assigned Nodes", len(nodes))
    offline = sum(1 for node in nodes if str(node["Status"]).lower() not in ("online", "healthy"))
    summary_cols[1].metric("Alerts", offline)
    if nodes:
        st.table(pd.DataFrame(nodes))
    else:
        st.info("No assigned nodes yet. Provision a node to begin monitoring.")


def show_dashboard():
    role = st.session_state.get("user_role", "user")
    username = st.session_state.get("username", "operator")
    if role == "admin":
        render_hero("SentriNode Operational Command")
        render_admin_dashboard()
    else:
        render_hero("SentriNode Node Status")
        render_user_dashboard(username)
    st.caption(f"Last synced: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")


# --- SETTINGS LOGIC ---
def _update_user_profile(username: str, full_name: str, email: str) -> bool:
    driver = _neo4j_driver()
    if not driver:
        return False
    try:
        with driver.session() as session:
            session.run(
                """
                MATCH (u:User {username:$username})
                SET u.full_name = $full_name,
                    u.notification_email = $email
                """,
                username=username,
                full_name=full_name.strip(),
                email=email.strip(),
            )
        return True
    except (ServiceUnavailable, Neo4jError, ValueError):
        return False
    finally:
        driver.close()


def _update_user_preferences(username: str, theme: str, desktop_notifications: bool) -> bool:
    driver = _neo4j_driver()
    if not driver:
        return False
    try:
        with driver.session() as session:
            session.run(
                """
                MATCH (u:User {username:$username})
                SET u.system_theme = $theme,
                    u.desktop_notifications = $notify
                """,
                username=username,
                theme=theme,
                notify=desktop_notifications,
            )
        return True
    except (ServiceUnavailable, Neo4jError, ValueError):
        return False
    finally:
        driver.close()


def _change_password(username: str, old_password: str, new_password: str) -> bool:
    driver = _neo4j_driver()
    if not driver:
        return False
    try:
        with driver.session() as session:
            record = session.run(
                """
                MATCH (u:User {username:$username})
                WHERE coalesce(u.password, '') = $old
                SET u.password = $new
                RETURN u.username AS username
                """,
                username=username,
                old=old_password,
                new=new_password,
            ).single()
        return bool(record)
    except (ServiceUnavailable, Neo4jError, ValueError):
        return False
    finally:
        driver.close()


def show_settings():
    username = st.session_state.get("username") or ""
    st.header("Account Settings")
    with st.form("profile_form"):
        full_name = st.text_input("Full Name")
        email = st.text_input("Notification Email")
        submitted = st.form_submit_button("Save Profile")
        if submitted:
            with st.spinner("Syncing with SentriNode Network..."):
                ok = _update_user_profile(username, full_name, email)
            if ok:
                st.success("Profile updated.")
                st.toast("Profile saved.", icon="✅")
            else:
                st.error("Unable to update profile.")

    st.divider()
    st.subheader("Preferences")
    themes = ["Auto", "Night Ops", "Day Ops"]
    theme = st.selectbox("System Theme", themes)
    desktop_notifications = st.checkbox("Enable Desktop Notifications")
    if st.button("Save Preferences"):
        with st.spinner("Syncing with SentriNode Network..."):
            ok = _update_user_preferences(username, theme, desktop_notifications)
        if ok:
            st.success("Preferences saved.")
            st.toast("Preferences updated.", icon="⚙️")
        else:
            st.error("Unable to save preferences.")

    st.divider()
    st.subheader("Security")
    with st.form("password_form"):
        old_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        change = st.form_submit_button("Change Password")
        if change:
            if not new_password or new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                with st.spinner("Syncing with SentriNode Network..."):
                    ok = _change_password(username, old_password, new_password)
                if ok:
                    st.success("Password updated.")
                    st.toast("Credentials rotated.", icon="🔐")
                else:
                    st.error("Unable to update password. Check your current password.")


# --- MAIN NAVIGATION ---
if not st.session_state.logged_in:
    if st.session_state.show_signup:
        show_signup()
    else:
        show_login()
else:
    sidebar_option = st.sidebar.radio("Navigation", ("Dashboard", "Settings"))
    st.sidebar.caption("Session Controls")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = "user"
        st.session_state.username = ""
        st.rerun()
    if sidebar_option == "Dashboard":
        show_dashboard()
    else:
        show_settings()
