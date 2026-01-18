import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from supabase import Client, create_client

try:
    from streamlit_agraph import agraph, Config, Edge, Node
except Exception:  # pragma: no cover - optional dependency
    agraph = Config = Edge = Node = None

st.set_page_config(layout="wide")

# ---------------------------------------------------------------------------
# Schema Inventory (auto-updated at runtime in render_schema_inventory panel)
# Labels & Relationships (assumptions until discovered):
#   - :Node {node_id, name, status, env, region, service, tags, last_heartbeat}
#   - :Metric {timestamp, latency_p50, latency_p95, latency_p99, error_rate,
#              throughput_rps, cpu, memory, disk, node_id}
#   - :Incident {incident_id, severity, opened_at, closed_at, summary}
#   - :Alert {rule, triggered_at, severity, node_id, details}
#   - Relationships: (:Node)-[:DEPENDS_ON]->(:Node),
#                    (:Node)-[:EMITS]->(:Metric),
#                    (:Incident)-[:AFFECTS]->(:Node),
#                    (:Alert)-[:ON_NODE]->(:Node)
# Time properties are expected as epoch millis or Neo4j datetime objects.
# TODO: refine the inventory after running SCHEMA_DISCOVERY_QUERIES.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Metrics Dictionary:
#   Active Nodes:
#       Definition: Nodes with last_heartbeat >= selected time window.
#       Query:
#           MATCH (n:Node)
#           WHERE n.last_heartbeat >= datetime({epochMillis:$since})
#             AND $envs = [] OR n.env IN $envs
#             ... (filters)
#           RETURN count(DISTINCT n) AS active_nodes
#       Assumptions: last_heartbeat stored as datetime or epoch millis.
#
#   Nodes Down:
#       Nodes missing heartbeat during window.
#       Query (uses same MATCH with COUNT of nodes not in active set).
#
#   Incident Count:
#           MATCH (i:Incident)
#           WHERE i.opened_at >= datetime({epochMillis:$since})
#           RETURN count(distinct i)
#
#   Error Rate:
#           MATCH (m:Metric)
#           WHERE m.timestamp >= datetime({epochMillis:$since})
#           RETURN avg(m.error_rate)
#
#   P95 Latency:
#           MATCH (m:Metric) WHERE ...
#           RETURN percentileCont(m.latency_p95, 0.95) AS latency_p95
#
#   Throughput, Saturation, Cost Proxy, etc. follow similar structure.
#   TODO: add indexes: CREATE INDEX node_node_id IF NOT EXISTS FOR (n:Node) ON (n.node_id);
#                       CREATE INDEX metric_timestamp IF NOT EXISTS FOR (m:Metric) ON (m.timestamp);
# ---------------------------------------------------------------------------

# --- INITIAL STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False
if "user_role" not in st.session_state:
    st.session_state.user_role = "user"
if "username" not in st.session_state:
    st.session_state.username = ""
if "user" not in st.session_state:
    st.session_state.user = None
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "selected_node" not in st.session_state:
    st.session_state.selected_node = None
if "node_registered" not in st.session_state:
    st.session_state.node_registered = False
if "registration_error" not in st.session_state:
    st.session_state.registration_error = None
if "pending_registration" not in st.session_state:
    st.session_state.pending_registration = None
if "neo4j_ok" not in st.session_state:
    st.session_state.neo4j_ok = False
if "neo4j_last_error" not in st.session_state:
    st.session_state.neo4j_last_error = None

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
_supabase_client_instance: Client | None = None

SCHEMA_DISCOVERY_QUERIES = [
    "CALL db.labels()",
    "CALL db.relationshipTypes()",
    "CALL db.schema.nodeTypeProperties()",
    "CALL db.schema.relTypeProperties()",
    "MATCH (n) RETURN labels(n) AS labels, count(*) AS cnt ORDER BY cnt DESC LIMIT 10",
    "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC LIMIT 10",
]

TIME_WINDOWS = {
    "Last 15 minutes": timedelta(minutes=15),
    "Last 1 hour": timedelta(hours=1),
    "Last 6 hours": timedelta(hours=6),
    "Last 24 hours": timedelta(hours=24),
    "Last 7 days": timedelta(days=7),
    "Last 30 days": timedelta(days=30),
}

MetricsPayload = dict[str, Any]

NODE_KPI_QUERY = """
MATCH (n:Node)
WHERE ($search = '' OR toLower(n.name) CONTAINS $search)
  AND (size($envs) = 0 OR coalesce(n.env,'') IN $envs)
  AND (size($services) = 0 OR coalesce(n.service,'') IN $services)
  AND (size($regions) = 0 OR coalesce(n.region,'') IN $regions)
WITH n, coalesce(n.last_heartbeat, datetime({epochMillis:0})) AS last_seen
RETURN
    count(n) AS total_nodes,
    count(CASE WHEN last_seen >= datetime({epochMillis:$since_epoch}) THEN 1 END) AS active_nodes,
    count(CASE WHEN last_seen < datetime({epochMillis:$since_epoch}) OR last_seen IS NULL THEN 1 END) AS nodes_down
"""

INCIDENT_KPI_QUERY = """
MATCH (i:Incident)
WHERE coalesce(i.opened_at, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND ($search = '' OR toLower(i.summary) CONTAINS $search)
RETURN count(i) AS incidents
"""

METRIC_AGG_QUERY = """
MATCH (m:Metric)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND ($search = '' OR toLower(m.node_name) CONTAINS $search)
RETURN
    avg(m.error_rate) AS error_rate,
    percentileCont(m.latency_p95, 0.95) AS latency_p95,
    avg(m.throughput_rps) AS throughput_rps,
    avg(m.cpu) AS cpu,
    avg(m.memory) AS memory,
    avg(m.disk) AS disk
"""

METRIC_TS_QUERY = """
MATCH (m:Metric)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND coalesce(m.timestamp, datetime({epochMillis:0})) <= datetime({epochMillis:$until_epoch})
  AND ($search = '' OR toLower(m.node_name) CONTAINS $search)
RETURN
    m.timestamp AS bucket_date,
    m.latency_p95 AS latency_p95,
    m.latency_p50 AS latency_p50,
    m.latency_p99 AS latency_p99,
    m.error_rate AS error_rate,
    m.throughput_rps AS throughput_rps,
    m.cpu AS cpu,
    m.memory AS memory,
    m.disk AS disk
ORDER BY bucket_date
"""

NODE_HEALTH_QUERY = """
MATCH (n:Node)
WHERE ($search = '' OR toLower(n.name) CONTAINS $search)
  AND (size($envs) = 0 OR coalesce(n.env,'') IN $envs)
  AND (size($services) = 0 OR coalesce(n.service,'') IN $services)
  AND (size($regions) = 0 OR coalesce(n.region,'') IN $regions)
WITH n
RETURN
    coalesce(n.node_id, id(n)) AS node_id,
    coalesce(n.name, 'Unnamed') AS name,
    coalesce(n.status, 'unknown') AS status,
    coalesce(toString(n.last_heartbeat), 'unknown') AS last_seen,
    coalesce(n.env, 'n/a') AS env,
    coalesce(n.region, 'n/a') AS region,
    coalesce(n.service, 'n/a') AS service,
    coalesce(n.tags, []) AS tags
ORDER BY last_seen DESC
LIMIT 200
"""

RELATIONSHIP_COUNTS_QUERY = """
CALL {
    MATCH (n) RETURN labels(n) AS labels, count(*) AS cnt LIMIT 50
}
RETURN labels, cnt
"""

GRAPH_SAMPLE_QUERY = """
MATCH (n:Node)-[r]->(m:Node)
RETURN n, r, m
LIMIT 50
"""

SUBGRAPH_QUERY = """
MATCH (n:Node)
WHERE ($node_name = '' AND ($search = '' OR toLower(n.name) CONTAINS $search))
   OR (toLower(n.name) = toLower($node_name))
WITH n
MATCH path = (n)-[r*1..2]-(m:Node)
RETURN nodes(path) AS nodes, relationships(path) AS rels
LIMIT 200
"""

TOP_ERRORS_QUERY = """
MATCH (m:Metric)-[:RECORDED_FOR]->(n:Node)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND ($search = '' OR toLower(n.name) CONTAINS $search)
RETURN coalesce(n.name, 'Unnamed') AS name,
       avg(m.error_rate) AS error_rate,
       percentileCont(m.latency_p95,0.95) AS latency_p95,
       avg(m.throughput_rps) AS throughput
ORDER BY error_rate DESC
LIMIT 10
"""

TOP_LATENCY_QUERY = """
MATCH (m:Metric)-[:RECORDED_FOR]->(n:Node)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND ($search = '' OR toLower(n.name) CONTAINS $search)
RETURN coalesce(n.name, 'Unnamed') AS name,
       percentileCont(m.latency_p95,0.95) AS latency_p95,
       percentileCont(m.latency_p99,0.99) AS latency_p99,
       avg(m.error_rate) AS error_rate
ORDER BY latency_p95 DESC
LIMIT 10
"""

TOP_INCIDENTS_QUERY = """
MATCH (i:Incident)-[:AFFECTS]->(n:Node)
WHERE coalesce(i.opened_at, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
  AND ($search = '' OR toLower(n.name) CONTAINS $search)
RETURN coalesce(n.name, 'Unnamed') AS name,
       count(i) AS incidents,
       avg(duration.inSeconds(coalesce(i.closed_at, datetime({epochMillis:$until_epoch})), i.opened_at).seconds) AS mttr_seconds
ORDER BY incidents DESC
LIMIT 10
"""

TOP_RESOURCE_QUERY = """
MATCH (m:Metric)-[:RECORDED_FOR]->(n:Node)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
RETURN coalesce(n.name, 'Unnamed') AS name,
       avg(m.cpu) AS cpu,
       avg(m.memory) AS memory,
       avg(m.disk) AS disk
ORDER BY (coalesce(m.cpu,0) + coalesce(m.memory,0) + coalesce(m.disk,0)) DESC
LIMIT 10
"""

ALERT_SOURCE_QUERY = """
MATCH (n:Node)
OPTIONAL MATCH (n)-[:EMITS]->(m:Metric)
WHERE coalesce(m.timestamp, datetime({epochMillis:0})) >= datetime({epochMillis:$since_epoch})
RETURN coalesce(n.name,'Unnamed') AS name,
       coalesce(n.last_heartbeat, datetime({epochMillis:0})) AS last_seen,
       avg(m.error_rate) AS error_rate,
       avg(m.latency_p95) AS latency_p95
"""


def _neo4j_driver():
    uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not pwd:
        st.warning("NEO4J_PASSWORD is not set; unable to connect to Neo4j.")
        print("[neo4j] missing NEO4J_PASSWORD; cannot create driver")
        st.session_state.neo4j_ok = False
        st.session_state.neo4j_last_error = "NEO4J_PASSWORD not set"
        return None
    last_error = None
    for attempt in range(1, 11):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, pwd))
            driver.verify_connectivity()
            if NEO4J_DATABASE:
                with driver.session(database=NEO4J_DATABASE) as session:
                    session.run("RETURN 1 AS ok").single()
            st.session_state.neo4j_ok = True
            st.session_state.neo4j_last_error = None
            return driver
        except Exception as exc:
            last_error = str(exc)
            st.session_state.neo4j_ok = False
            st.session_state.neo4j_last_error = last_error
            if attempt < 10:
                time.sleep(1.0)
    message = f"Neo4j connect failed ({uri}): {last_error}"
    print(message)
    return None


def _neo4j_session(driver):
    if NEO4J_DATABASE:
        return driver.session(database=NEO4J_DATABASE)
    return driver.session()


def _supabase_client() -> Client | None:
    global _supabase_client_instance
    if create_client is None:
        st.error("Supabase client library is not installed. Please add `supabase` to your environment.")
        return None
    if _supabase_client_instance is None:
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            return None
        _supabase_client_instance = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _supabase_client_instance


def _neo4j_username_exists(user: str) -> tuple[bool, str | None]:
    if st.session_state.get("neo4j_ok") is False and st.session_state.get("neo4j_last_error"):
        return False, None
    driver = _neo4j_driver()
    if not driver:
        st.session_state.neo4j_ok = False
        st.session_state.neo4j_last_error = st.session_state.get("neo4j_last_error") or "Unable to connect to Neo4j."
        return False, None
    try:
        with _neo4j_session(driver) as session:
            record = session.run(
                "MATCH (u:User {username:$user}) RETURN u LIMIT 1",
                user=(user or "").strip(),
            ).single()
        return bool(record), None
    except (ServiceUnavailable, Neo4jError, ValueError) as exc:
        st.session_state.neo4j_ok = False
        st.session_state.neo4j_last_error = str(exc)
        return False, None
    finally:
        driver.close()


def _run_registration(show_spinner: bool = True) -> None:
    payload = st.session_state.get("pending_registration")
    if not payload:
        return
    username = (payload.get("username") or "").strip()
    if not username:
        st.session_state.registration_error = "Missing username for registration."
        st.session_state.node_registered = False
        return

    def _execute() -> tuple[bool, str | None]:
        driver = _neo4j_driver()
        if not driver:
            message = f"Unable to connect to Neo4j at {os.getenv('NEO4J_URI') or 'bolt://localhost:7687'}."
            print(f"Neo4j registration failed: {message}")
            st.session_state.neo4j_ok = False
            st.session_state.neo4j_last_error = message
            return False, None
        try:
            with _neo4j_session(driver) as session:
                existing = session.run(
                    "MATCH (u:User {username:$user}) RETURN u LIMIT 1",
                    user=username,
                ).single()
                if existing:
                    if payload.get("mode") == "signup":
                        return False, "Username already taken."
                    return True, None
                session.run(
                    """
                    CREATE (u:User {
                        username:$user,
                        email:$email,
                        password:$password,
                        role:'user',
                        created_at:timestamp()
                    })
                    """,
                    user=username,
                    email=payload.get("email") or username,
                    password=payload.get("password") or "",
                )
                st.session_state.neo4j_ok = True
                st.session_state.neo4j_last_error = None
                return True, None
        except (ServiceUnavailable, Neo4jError, ValueError) as exc:
            message = str(exc)
            print(f"Neo4j registration failed: {message}")
            st.session_state.neo4j_ok = False
            st.session_state.neo4j_last_error = message
            return False, None
        finally:
            driver.close()

    def _execute_with_retries() -> tuple[bool, str | None]:
        attempts = 10
        delay_seconds = 1.0
        for attempt in range(1, attempts + 1):
            success, error = _execute()
            if success or attempt == attempts:
                return success, error
            time.sleep(delay_seconds)
        return False, "Registration failed."

    if show_spinner:
        with st.spinner("Registering node with Neo4j..."):
            success, error = _execute_with_retries()
    else:
        success, error = _execute_with_retries()

    if success:
        st.session_state.node_registered = True
        st.session_state.registration_error = None
        st.session_state.pending_registration = None
        st.session_state.neo4j_ok = True
        st.session_state.neo4j_last_error = None
        st.toast("Node registration complete.", icon="🛰️")
        print(f"Neo4j registration completed for user '{username}'")
        st.rerun()
    else:
        st.session_state.node_registered = False
        st.session_state.registration_error = (
            error or st.session_state.get("neo4j_last_error") or "Registration failed."
        )
        print(f"Neo4j registration failed: {st.session_state.registration_error}")


def _start_registration_flow(username: str, email: str, password: str, mode: str) -> None:
    st.session_state.pending_registration = {
        "username": (username or "").strip(),
        "email": (email or "").strip(),
        "password": password or "",
        "mode": mode,
    }
    st.session_state.registration_error = None
    st.session_state.node_registered = False
    st.session_state.neo4j_ok = False
    st.session_state.neo4j_last_error = None
    _run_registration(show_spinner=True)


def _handle_auth_success(
    res: Any,
    identifier: str,
    password: str,
    mode: str,
    *,
    email: str | None = None,
    toast_message: str | None = None,
    toast_icon: str = "✅",
) -> None:
    resolved_email = (email or identifier or "").strip()
    username = (identifier or resolved_email).strip()
    st.session_state["user"] = res.user
    st.session_state["access_token"] = res.session.access_token
    st.session_state.username = username
    st.session_state.logged_in = True
    st.session_state.show_signup = False
    st.session_state.user_role = _resolve_user_role(username)
    message = toast_message or ("Console unlocked. Welcome back." if mode == "login" else "Account created and signed in.")
    st.toast(message, icon=toast_icon)
    _start_registration_flow(username, resolved_email, password, mode)


def is_strong_password(pw: str) -> tuple[bool, str]:
    if len(pw) < 10:
        return False, "Use at least 10 characters."
    if not re.search(r"[A-Z]", pw):
        return False, "Add at least one uppercase letter."
    if not re.search(r"[a-z]", pw):
        return False, "Add at least one lowercase letter."
    if not re.search(r"\d", pw):
        return False, "Add at least one number."
    if not re.search(r"[^A-Za-z0-9]", pw):
        return False, "Add at least one symbol."
    return True, ""


def _resolve_user_role(username: str) -> str:
    role = st.session_state.get("user_role", "user")
    if st.session_state.get("neo4j_ok") is False:
        return role
    driver = _neo4j_driver()
    if not driver:
        return role
    try:
        with _neo4j_session(driver) as session:
            record = session.run(
                "MATCH (u:User {username:$username}) RETURN coalesce(u.role,'user') AS role LIMIT 1",
                username=username,
            ).single()
        if record and record["role"]:
            role = record["role"]
    except (ServiceUnavailable, Neo4jError, ValueError):
        pass
    finally:
        driver.close()
    return role


@dataclass
class FilterContext:
    since: datetime
    until: datetime
    search: str
    envs: list[str]
    services: list[str]
    regions: list[str]
    tags: list[str]


@st.cache_data(ttl=60)
def _run_cypher(query: str, params: dict | None = None) -> list[dict[str, Any]]:
    if st.session_state.get("neo4j_ok") is False and st.session_state.get("neo4j_last_error"):
        return []
    driver = _neo4j_driver()
    if not driver:
        return []
    params = params or {}
    try:
        with _neo4j_session(driver) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]
    except (ServiceUnavailable, Neo4jError, ValueError) as exc:
        st.session_state.neo4j_ok = False
        st.session_state.neo4j_last_error = str(exc)
        st.warning(f"Neo4j query failed: {exc}")
        return []
    finally:
        driver.close()


def _sidebar_filters(container=None) -> FilterContext:
    container = container or st.sidebar
    with container:
        st.subheader("Global Filters")
        time_label = st.selectbox("Time Range", list(TIME_WINDOWS.keys()) + ["Custom"], key="time_range")
        custom_since = custom_until = None
        if time_label == "Custom":
            custom_until = st.datetime_input("Until", value=datetime.utcnow(), key="custom_until")
            custom_since = st.datetime_input(
                "Since", value=datetime.utcnow() - timedelta(hours=1), key="custom_since"
            )
        search = st.text_input("Search Node / Incident", "", key="search_filter")
        envs = st.text_input("Environments (comma separated)", "", key="env_filter")
        services = st.text_input("Services (comma separated)", "", key="service_filter")
        regions = st.text_input("Regions (comma separated)", "", key="region_filter")
        tags = st.text_input("Tags (comma separated)", "", key="tag_filter")

        since = custom_since or (datetime.utcnow() - TIME_WINDOWS.get(time_label, timedelta(hours=1)))
        until = custom_until or datetime.utcnow()

        st.markdown("### Export")
        st.checkbox("Enable CSV export", value=True, key="export_csv")
        st.checkbox("Enable JSON export", value=True, key="export_json")

    return FilterContext(
        since=since,
        until=until,
        search=(search or "").strip(),
        envs=[e.strip() for e in envs.split(",") if e.strip()],
        services=[s.strip() for s in services.split(",") if s.strip()],
        regions=[r.strip() for r in regions.split(",") if r.strip()],
        tags=[t.strip() for t in tags.split(",") if t.strip()],
    )


def _time_params(filters: FilterContext) -> dict[str, Any]:
    return {
        "since": filters.since.isoformat(),
        "until": filters.until.isoformat(),
        "search": filters.search.lower(),
        "envs": filters.envs,
        "services": filters.services,
        "regions": filters.regions,
        "tags": filters.tags,
        "since_epoch": int(filters.since.timestamp() * 1000),
        "until_epoch": int(filters.until.timestamp() * 1000),
    }


def show_login(client: Client | None = None) -> None:
    st.title("SentriNode Login")
    client = client or _supabase_client()
    if not client:
        st.error("Supabase credentials missing.")
        return
    with st.form("login_form"):
        email = st.text_input("Email", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        submitted = st.form_submit_button("Login")
    if submitted:
        with st.spinner("Syncing with SentriNode Network..."):
            try:
                res = client.auth.sign_in_with_password({"email": email, "password": password})
            except Exception as exc:  # pragma: no cover - network
                res = None
                st.error(f"Login failed: {exc}")
        if res and res.user and res.session:
            _handle_auth_success(
                res,
                identifier=email or "",
                password=password or "",
                mode="login",
                email=email or "",
                toast_message="Console unlocked. Welcome back.",
                toast_icon="✅",
            )
        else:
            st.error("Login failed.")


def show_signup(client: Client | None = None) -> None:
    st.title("Create SentriNode Account")
    client = client or _supabase_client()
    if not client:
        st.error("Supabase credentials missing.")
        return
    email = st.text_input("Email", key="su_email")
    pw1 = st.text_input("Password", type="password", key="su_pw1")
    pw2 = st.text_input("Re-enter Password", type="password", key="su_pw2")
    accept = st.checkbox(
        "I understand this account will be created with Supabase Auth.", key="su_accept"
    )
    if st.button("Register Node", key="su_btn"):
        username = (email or "").strip()
        if not accept:
            st.error("Please check the box to continue.")
        elif not email or "@" not in email:
            st.error("Enter a valid email.")
        elif pw1 != pw2:
            st.error("Passwords do not match.")
        else:
            ok, msg = is_strong_password(pw1)
            if not ok:
                st.error(msg)
                return
            exists, exists_error = _neo4j_username_exists(username)
            if exists_error:
                st.error(exists_error)
                return
            if exists:
                st.error("Username already taken.")
                return
            with st.spinner("Syncing with SentriNode Network..."):
                try:
                    res = client.auth.sign_up({"email": email, "password": pw1})
                except Exception as exc:  # pragma: no cover - network
                    res = None
                    st.error(f"Sign up failed: {exc}")
            if res:
                if getattr(res, "session", None):
                    _handle_auth_success(
                        res,
                        identifier=username,
                        password=pw1 or "",
                        mode="signup",
                        email=email,
                        toast_message="Account created and signed in.",
                        toast_icon="🎉",
                    )
                else:
                    st.success("Account created. Check your email to confirm, then sign in.")


def _render_registration_status() -> None:
    if not st.session_state.get("user"):
        return
    error_msg = st.session_state.get("registration_error") or st.session_state.get("neo4j_last_error")
    if not error_msg and st.session_state.get("neo4j_ok") is False:
        error_msg = "Neo4j is unavailable."
    if error_msg:
        st.warning(f"Neo4j registration failed: {error_msg}")
        if st.button("Retry Neo4j registration", key="retry_registration_btn"):
            _run_registration(show_spinner=True)
    elif not st.session_state.get("node_registered"):
        st.info("Authenticated. Completing Neo4j registration...")


def render_auth_portal() -> None:
    render_hero("SENTRINODE")
    st.title("SentriNode Console Access")
    client = _supabase_client()
    if not client:
        st.error("Supabase credentials missing.")
        st.stop()
    tab_login, tab_signup = st.tabs(["Login", "Create account"])
    with tab_login:
        show_login(client)
    with tab_signup:
        show_signup(client)
    _render_registration_status()
    st.stop()


def authenticate_user(username: str, password: str) -> tuple[bool, str | None]:
    """Legacy compatibility hook – retained for backward compat, not used."""
    username = (username or "").strip()
    password = password or ""
    if not username or not password:
        return False, None
    driver = _neo4j_driver()
    if not driver:
        return False, None
    try:
        with _neo4j_session(driver) as session:
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
        with _neo4j_session(driver) as session:
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


def fetch_all_nodes() -> list[dict[str, object]]:
    driver = _neo4j_driver()
    if not driver:
        return []
    try:
        with _neo4j_session(driver) as session:
            records = session.run(
                """
                MATCH (n:Node)
                RETURN coalesce(n.name, n.id) AS name,
                       coalesce(n.status, 'online') AS status,
                       coalesce(n.ip_address, n.ip, '10.0.0.1') AS ip,
                       coalesce(toString(n.last_heartbeat), 'unsynced') AS heartbeat
                LIMIT 100
                """
            )
        return [
            {
                "Node Name": record["name"],
                "Status": record["status"],
                "IP Address": record["ip"],
                "Last Heartbeat": record["heartbeat"],
            }
            for record in records
        ]
    except (ServiceUnavailable, Neo4jError, ValueError):
        return []
    finally:
        driver.close()


def _fetch_global_kpis(filters: FilterContext) -> MetricsPayload:
    params = _time_params(filters)
    node_stats = _run_cypher(NODE_KPI_QUERY, params)
    incident_stats = _run_cypher(INCIDENT_KPI_QUERY, params)
    metric_stats = _run_cypher(METRIC_AGG_QUERY, params)
    node_row = node_stats[0] if node_stats else {}
    incident_row = incident_stats[0] if incident_stats else {}
    metric_row = metric_stats[0] if metric_stats else {}
    return {
        "active_nodes": node_row.get("active_nodes", 0),
        "nodes_down": node_row.get("nodes_down", 0),
        "incident_count": incident_row.get("incidents", 0),
        "error_rate": metric_row.get("error_rate", 0.0),
        "latency_p95": metric_row.get("latency_p95", 0.0),
        "throughput": metric_row.get("throughput_rps", 0.0),
        "cpu": metric_row.get("cpu", 0.0),
        "memory": metric_row.get("memory", 0.0),
        "disk": metric_row.get("disk", 0.0),
    }


def _render_kpi_tiles(kpis: MetricsPayload) -> None:
    cols = st.columns(4)
    cols[0].metric("Active Nodes", kpis["active_nodes"])
    cols[1].metric("Nodes Down", kpis["nodes_down"], delta_color="inverse")
    cols[2].metric("Incidents", kpis["incident_count"])
    cols[3].metric("Error Rate", f"{kpis['error_rate']:.3f}")
    cols = st.columns(4)
    cols[0].metric("P95 Latency", f"{kpis['latency_p95'] or 0:.1f} ms")
    cols[1].metric("Throughput", f"{kpis['throughput'] or 0:.1f} rps")
    cols[2].metric("CPU Saturation", f"{kpis['cpu'] or 0:.1f}%")
    cols[3].metric("Memory Saturation", f"{kpis['memory'] or 0:.1f}%")


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


# --- SCHEMA INVENTORY PANEL ---
def render_schema_inventory() -> None:
    with st.expander("Schema Inventory", expanded=False):
        st.write("Discovery queries executed at runtime:")
        st.code("\n".join(SCHEMA_DISCOVERY_QUERIES), language="cypher")
        labels = _run_cypher("CALL db.labels()")
        rels = _run_cypher("CALL db.relationshipTypes()")
        node_props = _run_cypher("CALL db.schema.nodeTypeProperties()")
        rel_props = _run_cypher("CALL db.schema.relTypeProperties()")
        inventory = {
            "labels": labels,
            "relationships": rels,
            "node_properties": node_props,
            "relationship_properties": rel_props,
        }
        st.json(inventory, expanded=False)


# --- UI LOGIC ---


def render_admin_dashboard() -> None:
    if st.session_state.get("neo4j_ok") is False:
        st.info(
            f"Neo4j unavailable. Dashboard data is disabled until registration succeeds. "
            f"{st.session_state.get('neo4j_last_error') or ''}"
        )
        return
    filters = _sidebar_filters()
    st.caption(
        f"Window: {filters.since.isoformat()} ➝ {filters.until.isoformat()} | Search: {filters.search or '—'}"
    )
    kpis = _fetch_global_kpis(filters)
    _render_kpi_tiles(kpis)
    df_ts = _fetch_time_series(filters)
    _render_time_series(df_ts)
    top_lists = _fetch_top_lists(filters)
    _render_top_lists(top_lists)
    _render_graph_overview(filters)
    node_health_df = _render_node_health(filters)
    _render_boards(filters, top_lists)
    _render_alerts(filters)
    _render_node_drilldown(filters)
    if st.session_state.get("export_json") and not node_health_df.empty:
        st.download_button(
            "Download Selected Node Metrics (JSON)",
            json.dumps(node_health_df.to_dict(orient="records")[:20], default=str),
            "node_metrics.json",
            mime="application/json",
        )


def render_user_dashboard(username: str) -> None:
    st.caption("Personal Node Status")
    if st.session_state.get("neo4j_ok") is False:
        st.info(
            f"Neo4j unavailable. Node status is paused until registration succeeds. "
            f"{st.session_state.get('neo4j_last_error') or ''}"
        )
        return
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


def _fetch_time_series(filters: FilterContext) -> pd.DataFrame:
    params = _time_params(filters)
    rows = _run_cypher(METRIC_TS_QUERY, params)
    if not rows:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(filters.since, periods=6, freq="H"),
                "latency_p95": [0] * 6,
                "latency_p50": [0] * 6,
                "latency_p99": [0] * 6,
                "error_rate": [0] * 6,
                "throughput_rps": [0] * 6,
                "cpu": [0] * 6,
                "memory": [0] * 6,
                "disk": [0] * 6,
            }
        )
    else:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["bucket_date"])
    df = df.set_index("timestamp").resample("15T").mean().reset_index().fillna(0)
    df = df.rename(columns={"timestamp": "bucket_date"})
    return df


def _render_time_series(df: pd.DataFrame) -> None:
    st.subheader("Trends")
    latency_cols = st.columns(1)
    latency_cols[0].line_chart(
        df.set_index("bucket_date")[["latency_p50", "latency_p95", "latency_p99"]],
        height=220,
    )
    metric_cols = st.columns(3)
    metric_cols[0].line_chart(df.set_index("bucket_date")[["error_rate"]], height=180)
    metric_cols[1].line_chart(df.set_index("bucket_date")[["throughput_rps"]], height=180)
    metric_cols[2].line_chart(df.set_index("bucket_date")[["cpu", "memory", "disk"]], height=180)


def _fetch_top_lists(filters: FilterContext) -> dict[str, pd.DataFrame]:
    params = _time_params(filters)
    data = {
        "errors": pd.DataFrame(_run_cypher(TOP_ERRORS_QUERY, params)),
        "latency": pd.DataFrame(_run_cypher(TOP_LATENCY_QUERY, params)),
        "incidents": pd.DataFrame(_run_cypher(TOP_INCIDENTS_QUERY, params)),
        "resources": pd.DataFrame(_run_cypher(TOP_RESOURCE_QUERY, params)),
    }
    return data


def _render_top_lists(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Top Lists")
    top_cols = st.tabs(
        ["Top Errors", "Top Latency", "Top Incidents", "Top Resource Usage"]
    )
    tables = ["errors", "latency", "incidents", "resources"]
    for tab, key in zip(top_cols, tables):
        with tab:
            df = data.get(key, pd.DataFrame())
            if df.empty:
                st.info("No data available for this view.")
            else:
                st.dataframe(df, hide_index=True, use_container_width=True)
                selected = st.selectbox(
                    "Focus node", [""] + df["name"].astype(str).tolist(), key=f"focus_{key}"
                )
                if selected:
                    st.session_state.selected_node = selected
                    st.success(f"Selected {selected} for drilldown.", icon="🔎")


def _render_graph_overview(filters: FilterContext) -> None:
    st.subheader("Graph Overview")
    render_schema_inventory()
    label_counts = _run_cypher("CALL db.labels() YIELD label RETURN label, count(*) AS count")
    rel_counts = _run_cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
    col1, col2 = st.columns(2)
    with col1:
        if label_counts:
            st.table(pd.DataFrame(label_counts))
        else:
            st.info("No label data available.")
    with col2:
        if rel_counts:
            st.table(pd.DataFrame(rel_counts))
        else:
            st.info("No relationship data available.")

    rows = _run_cypher(GRAPH_SAMPLE_QUERY, _time_params(filters))
    if rows and agraph and Node and Edge and Config:
        nodes_map: dict[str, Node] = {}
        edges: list[Edge] = []
        for row in rows:
            start = row["n"]
            end = row["m"]
            rel = row["r"]
            for entity in (start, end):
                if entity.id not in nodes_map:
                    nodes_map[entity.id] = Node(
                        id=str(entity.id),
                        label=str(entity.get("name", f"node-{entity.id}")),
                        size=18,
                        color="#60a5fa",
                    )
            edges.append(
                Edge(
                    source=str(start.id),
                    target=str(end.id),
                    title=rel.type,
                    color="#f97316",
                )
            )
        st.markdown("#### Sample Topology")
        config = Config(height=420, width=1000, directed=True, physics=True)
        agraph(list(nodes_map.values()), edges, config)
    else:
        st.info("Graph visualization unavailable (no data or streamlit-agraph missing).")
    st.markdown("#### Graph Explorer")
    search_node = st.text_input("Explore node (exact match or leave blank)", key="graph_focus")
    params = _time_params(filters)
    params["node_name"] = (search_node or "").strip()
    subgraph_rows = _run_cypher(SUBGRAPH_QUERY, params)
    adjacency: list[dict[str, str]] = []
    if subgraph_rows and agraph and Node and Edge and Config:
        graph_nodes: dict[str, Node] = {}
        graph_edges: list[Edge] = []
        for row in subgraph_rows:
            for n in row["nodes"]:
                if n.id not in graph_nodes:
                    graph_nodes[n.id] = Node(
                        id=str(n.id),
                        label=str(n.get("name", f"node-{n.id}")),
                        size=20,
                        color="#4ade80" if (search_node and str(n.get("name","")).lower() == search_node.lower()) else "#a855f7",
                    )
            for rel in row["rels"]:
                graph_edges.append(
                    Edge(
                        source=str(rel.start_node.id),
                        target=str(rel.end_node.id),
                        title=rel.type,
                        color="#facc15",
                    )
                )
                adjacency.append(
                    {
                        "source": rel.start_node.get("name", rel.start_node.id),
                        "relation": rel.type,
                        "target": rel.end_node.get("name", rel.end_node.id),
                    }
                )
        if graph_nodes:
            config = Config(height=480, width=1000, directed=True, physics=True)
            agraph(list(graph_nodes.values()), graph_edges, config)
    if adjacency:
        st.dataframe(pd.DataFrame(adjacency), hide_index=True, use_container_width=True)
        if st.button("Export Subgraph CSV"):
            csv = pd.DataFrame(adjacency).to_csv(index=False)
            st.download_button("Download Adjacency", csv, "subgraph.csv", mime="text/csv")
    else:
        st.info("No subgraph data available for current selection.")


def _render_node_health(filters: FilterContext) -> pd.DataFrame:
    st.subheader("Node Health")
    df = pd.DataFrame(_run_cypher(NODE_HEALTH_QUERY, _time_params(filters)))
    if df.empty:
        st.info("No node health data.")
        return df
    st.dataframe(df, use_container_width=True, hide_index=True)
    if st.session_state.get("export_csv") and not df.empty:
        st.download_button(
            "Download Node Health CSV", df.to_csv(index=False), "node_health.csv", mime="text/csv"
        )
    selection = st.selectbox(
        "Focus node for drilldown",
        [""] + df["name"].astype(str).tolist(),
        key="node_health_focus",
    )
    if selection:
        st.session_state.selected_node = selection
        st.toast(f"Drilldown focus set to {selection}", icon="🔎")
    return df


def _render_boards(filters: FilterContext, top_lists: dict[str, pd.DataFrame]) -> None:
    st.subheader("SentriNode Boards")
    board_tabs = st.tabs(["Reliability", "Performance", "Cost", "Security", "Change"])

    with board_tabs[0]:
        st.markdown("##### Reliability Board")
        incidents_df = top_lists.get("incidents") or pd.DataFrame()
        st.write("Top incident drivers")
        st.dataframe(incidents_df, hide_index=True, use_container_width=True)
        st.write("MTTR (seconds)")
        if not incidents_df.empty:
            st.bar_chart(incidents_df.set_index("name")["mttr_seconds"])
        else:
            st.info("No incident data available. TODO: instrument incident writes.")

    with board_tabs[1]:
        st.markdown("##### Performance Board")
        latency_df = top_lists.get("latency") or pd.DataFrame()
        st.write("Top latency outliers")
        st.dataframe(latency_df, hide_index=True, use_container_width=True)

    with board_tabs[2]:
        st.markdown("##### Cost Board")
        cost_df = top_lists.get("resources") or pd.DataFrame()
        st.write("Resource usage proxy (CPU+Memory+Disk)")
        st.dataframe(cost_df, hide_index=True, use_container_width=True)
        st.caption("Cost proxy derived from average resource usage. TODO: replace with actual cost metrics.")

    with board_tabs[3]:
        st.markdown("##### Security Board")
        st.info("TODO: surface auth failures or unusual patterns when available. Placeholder view.")

    with board_tabs[4]:
        st.markdown("##### Change Board")
        st.info("TODO: integrate deployment/config change data from Neo4j (e.g., :Deployment nodes).")


def _generate_alerts(filters: FilterContext) -> list[dict[str, str]]:
    alerts = []
    data = _run_cypher(ALERT_SOURCE_QUERY, _time_params(filters))
    now = datetime.utcnow()
    for row in data:
        last_seen_str = row.get("last_seen")
        last_seen_dt = None
        if isinstance(last_seen_str, datetime):
            last_seen_dt = last_seen_str
        else:
            try:
                last_seen_dt = datetime.fromisoformat(str(last_seen_str))
            except Exception:
                last_seen_dt = None
        if last_seen_dt and now - last_seen_dt > timedelta(hours=1):
            alerts.append(
                {
                    "severity": "high",
                    "title": f"Node {row['name']} missing heartbeat",
                    "detail": f"Last seen {last_seen_dt.isoformat()}",
                    "suggestion": "Check node agent connectivity.",
                }
            )
        if row.get("error_rate", 0) and row["error_rate"] > 0.05:
            alerts.append(
                {
                    "severity": "medium",
                    "title": f"Error spike on {row['name']}",
                    "detail": f"Error rate {row['error_rate']:.2%}",
                    "suggestion": "Inspect recent deployments or upstream dependencies.",
                }
            )
        if row.get("latency_p95", 0) and row["latency_p95"] > 500:
            alerts.append(
                {
                    "severity": "medium",
                    "title": f"Latency spike on {row['name']}",
                    "detail": f"P95 latency {row['latency_p95']:.1f} ms",
                    "suggestion": "Check downstream services for saturation.",
                }
            )
    return alerts


def _render_alerts(filters: FilterContext) -> None:
    st.subheader("Alerts & Anomalies")
    alerts = _generate_alerts(filters)
    if not alerts:
        st.success("No active alerts detected.")
        return
    for alert in alerts:
        st.error(
            f"[{alert['severity'].upper()}] {alert['title']}\n\n{alert['detail']}\n\nNext action: {alert['suggestion']}"
        )


def _fetch_node_drilldown(node_name: str, filters: FilterContext) -> dict[str, Any]:
    params = _time_params(filters)
    params["node_name"] = node_name
    summary = _run_cypher(
        """
        MATCH (n:Node)
        WHERE toLower(n.name) = toLower($node_name)
        OPTIONAL MATCH (n)-[:DEPENDS_ON]->(dep:Node)
        RETURN n AS node, collect(dep.name) AS dependencies
        LIMIT 1
        """,
        params,
    )
    metrics = _run_cypher(
        """
        MATCH (m:Metric)-[:RECORDED_FOR]->(n:Node)
        WHERE toLower(n.name) = toLower($node_name)
          AND m.timestamp >= datetime({epochMillis:$since_epoch})
        RETURN m.timestamp AS timestamp,
               m.latency_p95 AS latency_p95,
               m.error_rate AS error_rate,
               m.cpu AS cpu,
               m.memory AS memory,
               m.disk AS disk
        ORDER BY timestamp DESC
        LIMIT 200
        """,
        params,
    )
    incidents = _run_cypher(
        """
        MATCH (i:Incident)-[:AFFECTS]->(n:Node)
        WHERE toLower(n.name) = toLower($node_name)
        RETURN i.incident_id AS incident_id,
               i.severity AS severity,
               i.opened_at AS opened_at,
               i.closed_at AS closed_at,
               i.summary AS summary
        ORDER BY i.opened_at DESC
        LIMIT 20
        """,
        params,
    )
    alerts = [alert for alert in _generate_alerts(filters) if node_name in alert["title"]]
    return {"summary": summary, "metrics": metrics, "incidents": incidents, "alerts": alerts}


def _render_node_drilldown(filters: FilterContext) -> None:
    st.subheader("Node Drilldown")
    if st.session_state.get("neo4j_ok") is False:
        st.info(
            f"Neo4j unavailable. Drilldown is disabled until registration succeeds. "
            f"{st.session_state.get('neo4j_last_error') or ''}"
        )
        return
    node_name = st.session_state.get("selected_node")
    if not node_name:
        st.info("Select a node from any table to drill down.")
        return
    payload = _fetch_node_drilldown(node_name, filters)
    summary = payload["summary"][0] if payload["summary"] else None
    if not summary:
        st.warning(f"No details found for {node_name}.")
        return
    node_props = summary["node"]
    dependencies = summary.get("dependencies", [])
    st.markdown(f"### {node_props.get('name', node_name)}")
    st.json(node_props)
    if dependencies:
        st.markdown("**Dependencies:** " + ", ".join(dependencies))
    metrics_df = pd.DataFrame(payload["metrics"])
    if not metrics_df.empty:
        metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"])
        st.line_chart(metrics_df.set_index("timestamp")[["latency_p95", "error_rate"]])
    incidents_df = pd.DataFrame(payload["incidents"])
    if not incidents_df.empty:
        st.markdown("#### Recent Incidents")
        st.dataframe(incidents_df, use_container_width=True, hide_index=True)
    if payload["alerts"]:
        st.markdown("#### Alerts")
        for alert in payload["alerts"]:
            st.error(f"{alert['title']} - {alert['detail']}")


def show_node_manager():
    render_hero("SentriNode Node Manager")
    if st.session_state.get("neo4j_ok") is False:
        st.info(
            f"Neo4j unavailable. Node manager actions are disabled until registration succeeds. "
            f"{st.session_state.get('neo4j_last_error') or ''}"
        )
        return
    search = st.text_input("Search Nodes", placeholder="Filter by node name")
    with st.spinner("Syncing with SentriNode Network..."):
        nodes = fetch_all_nodes()
    if search:
        nodes = [node for node in nodes if search.lower() in str(node["Node Name"]).lower()]
    if nodes:
        st.dataframe(pd.DataFrame(nodes), use_container_width=True, hide_index=True)
        st.subheader("Actions")
        for node in nodes:
            cols = st.columns([3, 1])
            cols[0].markdown(f"**{node['Node Name']}** · {node['Status']} · {node['IP Address']}")
            if cols[1].button("Reboot Node", key=f"reboot-{node['Node Name']}"):
                st.success(f"Reboot signal sent to {node['Node Name']}.")
                st.toast(f"{node['Node Name']} reboot requested.", icon="♻️")
    else:
        st.info("No nodes to display. Check your connections or adjust the filter.")


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
        with _neo4j_session(driver) as session:
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
        with _neo4j_session(driver) as session:
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
        with _neo4j_session(driver) as session:
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
if not st.session_state.get("user") or not st.session_state.get("access_token"):
    render_auth_portal()
else:
    sidebar_option = st.sidebar.radio("Navigation", ("Dashboard", "Node Manager", "Settings"))
    st.sidebar.caption("Session Controls")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = "user"
        st.session_state.username = ""
        st.session_state.user = None
        st.session_state.access_token = None
        st.session_state.show_signup = False
        st.session_state.node_registered = False
        st.session_state.registration_error = None
        st.session_state.pending_registration = None
        st.session_state.neo4j_ok = False
        st.session_state.neo4j_last_error = None
        st.rerun()
    _render_registration_status()
    if sidebar_option == "Dashboard":
        show_dashboard()
    elif sidebar_option == "Node Manager":
        show_node_manager()
    else:
        show_settings()
