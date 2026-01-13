#!/usr/bin/env python3
"""SentriNode Cloudflare-style dashboard demo built with Streamlit."""
from __future__ import annotations

import copy
import os
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
import tomllib
import numpy as np
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from streamlit_autorefresh import st_autorefresh
try:
    from streamlit_agraph import Config, Edge, Node, agraph
except ImportError:  # pragma: no cover - optional dependency
    Config = Edge = Node = agraph = None

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - optional dependency for demo analytics
    px = None

import requests

from ai_artifacts import load_latest_artifact
import graph_engine as ge

# -----------------------------------------------------------------------------
# Page configuration + dark theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SentriNode Enterprise Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Secret helpers + credential bootstrap
# -----------------------------------------------------------------------------
SECRETS_FILE = Path(".streamlit") / "secrets.toml"
BOOTSTRAP_FLAG_FILE = Path(".streamlit") / ".bootstrap_complete"
ENV_CREDENTIAL_KEYS: dict[str, tuple[str, ...]] = {
    "admin_user": ("SENTRINODE_ADMIN_USER", "SENTINEL_ADMIN_USER"),
    "admin_pwd": ("SENTRINODE_ADMIN_PWD", "SENTINEL_ADMIN_PWD"),
    "viewer_user": ("SENTRINODE_VIEWER_USER", "SENTINEL_VIEWER_USER"),
    "viewer_pwd": ("SENTRINODE_VIEWER_PWD", "SENTINEL_VIEWER_PWD"),
}
_local_secrets_cache: dict[str, dict[str, Any]] | None = None


def _load_local_secrets() -> dict[str, dict[str, Any]]:
    global _local_secrets_cache
    if _local_secrets_cache is None:
        try:
            with SECRETS_FILE.open("rb") as file_obj:
                _local_secrets_cache = tomllib.load(file_obj)
        except FileNotFoundError:
            _local_secrets_cache = {}
    return _local_secrets_cache


def _format_toml_value(value: Any) -> str:
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_local_secrets(data: dict[str, dict[str, Any]]) -> None:
    global _local_secrets_cache
    try:
        SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for section, entries in data.items():
            lines.append(f"[{section}]")
            for key, value in entries.items():
                lines.append(f"{key} = {_format_toml_value(value)}")
            lines.append("")
        SECRETS_FILE.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        _local_secrets_cache = copy.deepcopy(data)
    except OSError:
        # Read-only environments (e.g., managed PaaS) can't persist secrets.
        _local_secrets_cache = copy.deepcopy(data)


def _section_to_dict(section: Mapping[str, Any] | None) -> dict[str, Any]:
    if not section:
        return {}
    return {key: section[key] for key in section}


def get_secret_section(name: str) -> dict[str, Any]:
    local = _load_local_secrets().get(name)
    if local:
        return dict(local)
    return _section_to_dict(st.secrets.get(name))


def _credentials_configured(creds: Mapping[str, Any] | None) -> bool:
    if not creds:
        return False
    return all(creds.get(field) for field in ("admin_user", "admin_pwd", "viewer_user", "viewer_pwd"))


def _persist_credentials(new_creds: dict[str, str]) -> None:
    cache = copy.deepcopy(_load_local_secrets())
    if not cache:
        try:
            cache = {section: _section_to_dict(st.secrets[section]) for section in st.secrets}
        except Exception:  # pragma: no cover - streamlit-only object
            cache = {}
    cache["credentials"] = new_creds
    try:
        _write_local_secrets(cache)
    except OSError:
        pass
    st.session_state["sentri_credentials"] = new_creds
    _mark_bootstrap_complete()


def _parse_authorized_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    stringified = str(value).strip()
    return [stringified] if stringified else []


def _load_authorized_keys() -> list[str]:
    bootstrap_section = get_secret_section("bootstrap")
    combined: list[str] = []
    secrets_value = bootstrap_section.get("authorized_keys") or bootstrap_section.get("api_key")
    env_value = os.getenv("AUTHORIZED_KEYS") or os.getenv("SENTRINODE_BOOTSTRAP_KEY")
    combined.extend(_parse_authorized_values(secrets_value))
    combined.extend(_parse_authorized_values(env_value))
    seen: set[str] = set()
    unique: list[str] = []
    for key in combined:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _load_env_credentials() -> dict[str, str]:
    env_creds: dict[str, str] = {}
    for field, options in ENV_CREDENTIAL_KEYS.items():
        for env_key in options:
            value = os.getenv(env_key)
            if value:
                env_creds[field] = value.strip()
                break
    if _credentials_configured(env_creds):
        return env_creds
    return {}


AUTHORIZED_BOOTSTRAP_KEYS = _load_authorized_keys()


def _bootstrap_is_complete() -> bool:
    if st.session_state.get("bootstrap_verified"):
        return True
    if BOOTSTRAP_FLAG_FILE.exists():
        st.session_state["bootstrap_verified"] = True
        return True
    return False


def _mark_bootstrap_complete() -> None:
    try:
        BOOTSTRAP_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        BOOTSTRAP_FLAG_FILE.write_text("ok", encoding="utf-8")
    except OSError:
        pass
    st.session_state["bootstrap_verified"] = True

st.markdown(
    """
    <style>
        :root {
            --sentri-bg: #0E1117;
            --sentri-surface: #141821;
            --sentri-text: #E0E3EB;
            --sentri-muted: #8A8F9F;
            --sentri-border: #262730;
            --sentri-primary: #00D4FF;
        }
        html, body, [class*="css"]  {
            background-color: var(--sentri-bg) !important;
            color: var(--sentri-text) !important;
            font-family: "Inter", sans-serif;
        }
        .sentri-card {
            background: var(--sentri-surface);
            border: 1px solid var(--sentri-border);
            border-radius: 12px;
            padding: 18px 22px;
        }
        .metric-label {
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.3px;
            color: var(--sentri-muted);
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 600;
            color: var(--sentri-text);
        }
        .metric-delta {
            font-size: 0.85rem;
            margin-top: 8px;
        }
        .metric-delta.positive {
            color: #00D4FF;
        }
        .metric-delta.negative {
            color: #FF4B4B;
        }
        .sentri-section {
            margin-bottom: 28px;
            padding: 12px 8px;
        }
        .top-header {
            border-bottom: 1px solid var(--sentri-border);
            padding-bottom: 12px;
            margin-bottom: 24px;
        }
        .env-pill {
            background: rgba(0, 212, 255, 0.15);
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.85rem;
            color: var(--sentri-primary);
            display: inline-block;
            margin-bottom: 6px;
        }
        .search-input input {
            border-radius: 999px !important;
            border: 1px solid var(--sentri-border) !important;
            background: var(--sentri-bg) !important;
            color: var(--sentri-text) !important;
        }
        .network-panel {
            border: 1px solid var(--sentri-border);
            border-radius: 12px;
            padding: 16px;
            background: var(--sentri-surface);
        }
        .edge {
            padding: 10px 14px;
            border: 1px solid var(--sentri-border);
            border-radius: 10px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.02);
            font-size: 0.95rem;
        }
        .edge-alert {
            border-color: #FF4B4B;
            box-shadow: 0 0 8px rgba(255, 75, 75, 0.4);
            animation: pulse 1.8s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 4px rgba(255, 75, 75, 0.4); }
            50% { box-shadow: 0 0 16px rgba(255, 75, 75, 0.7); }
            100% { box-shadow: 0 0 4px rgba(255, 75, 75, 0.4); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data helpers + AI enrichment
# -----------------------------------------------------------------------------
TRACE_SOURCE = Path("training_data.csv")
LOGO_PATH = Path("SentriNode_Production_PNG_Pack_v2_Transparent/SentriNode_Icon_Dark_64.png")
LOAD_LEVELS = ["Low", "Medium", "High", "Burst"]
BASE_REVENUE_PER_MINUTE = 25000.0

NODES = [
    "Auth-Service",
    "Payment-Gateway",
    "User-DB",
    "Inventory-API",
    "Legacy-Mainframe",
]


@st.cache_data(ttl=5, show_spinner=False)
def load_trace_data(demo: bool = False, tenant: str | None = None) -> pd.DataFrame:
    if TRACE_SOURCE.exists():
        base = pd.read_csv(TRACE_SOURCE).rename(
            columns={
                "parent_service": "service",
                "child_service": "dependency",
                "system_load": "system_load",
                "parent_ms": "actual_ms",
                "target_ms": "target_ms",
            }
        )
    else:
        base = pd.DataFrame(
            {
                "service": ["lets-go", "lets-go", "lets-go", "lets-go"],
                "dependency": ["okey-dokey-0", "okey-dokey-1", "okey-dokey-2", "okey-dokey-2"],
                "system_load": ["Low", "Medium", "High", "Low"],
                "actual_ms": [20.5, 55.0, 180.0, 24.0],
                "target_ms": [22.0, 45.0, 90.0, 20.0],
            }
        )
    if "tenant_id" not in base.columns:
        base["tenant_id"] = tenant or "sentri-labs"
    else:
        base["tenant_id"] = base["tenant_id"].fillna(tenant or "sentri-labs")
    total_rows = len(base)
    if "ts" in base.columns:
        base["ts"] = pd.to_datetime(base["ts"], utc=True, errors="coerce")
    else:
        start = datetime.utcnow() - timedelta(minutes=total_rows)
        base["ts"] = pd.date_range(start, periods=total_rows, freq="min")
    base["trace_id"] = [
        f"trace-{i:05d}" for i in range(total_rows)
    ]

    if demo:
        base = pd.concat([base, _generate_demo_rows(total_rows, tenant or "demo")], ignore_index=True)

    base = base.sort_values("ts").reset_index(drop=True)
    return base


def _generate_demo_rows(offset: int, tenant: str) -> pd.DataFrame:
    rng = np.random.default_rng()
    now = datetime.utcnow()
    rows = []
    for idx in range(140):
        service = rng.choice(NODES)
        dependency_candidates = [svc for svc in NODES if svc != service]
        dependency = rng.choice(dependency_candidates) if dependency_candidates else service
        load = rng.choice(LOAD_LEVELS, p=[0.25, 0.35, 0.25, 0.15])
        jitter = rng.normal(0, 35)
        actual = max(20.0, rng.normal(220 if load in ("High", "Burst") else 120, 60) + jitter)
        target = max(18.0, rng.normal(85, 18))
        rows.append(
            {
                "service": service,
                "dependency": dependency,
                "system_load": load,
                "actual_ms": actual,
                "target_ms": target,
                "tenant_id": tenant,
                "ts": now - timedelta(minutes=140 - idx),
                "trace_id": f"demo-{offset + idx:05d}",
            }
        )
    return pd.DataFrame(rows)


def _normalize_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "__missing__"
    return str(value)


def _encode_with_fallback(values: pd.Series, primary_lookup: dict[str, int]) -> np.ndarray:
    normalized_primary = {_normalize_label(key): idx for key, idx in primary_lookup.items()}
    normalized_values = [_normalize_label(val) for val in values]
    fallback = {
        label: code for code, label in enumerate(sorted(set(normalized_values)))
    }
    return np.array(
        [normalized_primary.get(label, fallback.get(label, -1)) for label in normalized_values]
    )


def enrich_trace_data(
    df: pd.DataFrame,
    model,
    service_encoder,
    load_encoder,
) -> pd.DataFrame:
    if df.empty:
        return df.assign(
            AI_Predicted_ms=[],
            status=[],
            reason=[],
            ai_recommendation=[],
            latency_gap=[],
        )

    service_lookup = {}
    load_lookup = {}
    if hasattr(service_encoder, "classes_"):
        service_lookup = {_normalize_label(cls): idx for idx, cls in enumerate(service_encoder.classes_)}
    if hasattr(load_encoder, "classes_"):
        load_lookup = {_normalize_label(cls): idx for idx, cls in enumerate(load_encoder.classes_)}

    df = df.copy()
    df["service_num"] = _encode_with_fallback(df["service"], service_lookup)
    df["load_num"] = _encode_with_fallback(df["system_load"], load_lookup)

    rolling_default = df["actual_ms"].rolling(window=5, min_periods=1).mean().fillna(df["actual_ms"].mean())
    predictions = rolling_default.to_numpy()
    if model is not None:
        feature_frame = df[["service_num", "load_num"]].to_numpy()
        try:
            predictions = model.predict(feature_frame)
        except Exception:
            # Fall back to rolling average if schema doesn't match.
            predictions = rolling_default.to_numpy()

    df["AI_Predicted_ms"] = np.round(predictions, 2)
    df["latency_gap"] = df["actual_ms"] - df["AI_Predicted_ms"]
    guard = np.maximum(df["AI_Predicted_ms"] * 1.35, df["target_ms"])
    df["status"] = np.where(df["actual_ms"] > guard, "ANOMALY", "HEALTHY")
    df["reason"] = df.apply(_build_reasoning, axis=1)
    df["ai_recommendation"] = df.apply(_build_recommendation, axis=1)
    return df


def _build_reasoning(row: pd.Series) -> str:
    predicted = max(row.get("AI_Predicted_ms", 0.1), 0.1)
    ratio = row["actual_ms"] / predicted
    ratio_str = f"{ratio:.1f}x"
    if row["status"] == "ANOMALY":
        return (
            f"Latency ({row['actual_ms']:.1f}ms) is {ratio_str} predicted "
            f"({row['AI_Predicted_ms']:.1f}ms) during {row['system_load']} load."
        )
    return "Operating within predicted envelope."


def _build_recommendation(row: pd.Series) -> str:
    if row["status"] != "ANOMALY":
        return "No action required."
    if row["system_load"] in ("High", "Burst"):
        return f"Scale {row['service']} by +2 nodes and shed 15% traffic to {row['dependency']}."
    return f"Warm cache on {row['dependency']} and retry {row['service']} tail requests."


def summarize_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "total_requests": 0,
            "mean_latency": 0.0,
            "anomaly_count": 0,
            "system_health": 100.0,
            "health_delta": 0.0,
            "latency_delta": 0.0,
            "request_delta": 0.0,
            "anomaly_delta": 0.0,
            "health_percentage": 100.0,
            "volatile_service": "n/a",
            "volatile_count": 0,
            "blast_radius": 0,
            "avg_prediction_error": 0.0,
        }

    df = df.sort_values("ts")
    total_requests = len(df) * 320
    mean_latency = df["actual_ms"].mean()
    anomalies = df[df["status"] == "ANOMALY"]
    anomaly_count = len(anomalies)
    system_health = max(0.0, 100.0 - (anomaly_count / max(len(df), 1) * 100.0))
    avg_error = float(np.abs(df["latency_gap"]).mean())

    latest_ts = df["ts"].max()
    recent_start = latest_ts - timedelta(minutes=10)
    prev_start = recent_start - timedelta(minutes=10)
    recent = df[df["ts"] >= recent_start]
    previous = df[(df["ts"] >= prev_start) & (df["ts"] < recent_start)]

    def percent_change(cur: float, prev: float) -> float:
        if prev == 0 or np.isnan(prev):
            return 0.0
        return ((cur - prev) / prev) * 100.0

    request_delta = percent_change(len(recent), len(previous))
    latency_delta = percent_change(
        recent["actual_ms"].mean() if not recent.empty else 0.0,
        previous["actual_ms"].mean() if not previous.empty else 0.0,
    )
    anomaly_delta = percent_change(
        (recent["status"] == "ANOMALY").sum(),
        (previous["status"] == "ANOMALY").sum(),
    )
    health_delta = percent_change(
        100.0 - ((recent["status"] == "ANOMALY").sum() / max(len(recent), 1) * 100.0)
        if not recent.empty
        else system_health,
        100.0 - ((previous["status"] == "ANOMALY").sum() / max(len(previous), 1) * 100.0)
        if not previous.empty
        else system_health,
    )

    volatile_group = (
        anomalies.groupby("service")
        .size()
        .sort_values(ascending=False)
    )
    volatile_service = volatile_group.index[0] if not volatile_group.empty else "n/a"
    volatile_count = int(volatile_group.iloc[0]) if not volatile_group.empty else 0
    blast_radius = anomalies["dependency"].nunique()

    return {
        "total_requests": total_requests,
        "mean_latency": mean_latency,
        "anomaly_count": anomaly_count,
        "system_health": system_health,
        "health_percentage": system_health,
        "health_delta": health_delta,
        "latency_delta": latency_delta,
        "request_delta": request_delta,
        "anomaly_delta": anomaly_delta,
        "volatile_service": volatile_service,
        "volatile_count": volatile_count,
        "blast_radius": blast_radius,
        "avg_prediction_error": avg_error,
    }


def estimate_revenue_risk(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    anomalies = df[df["status"] == "ANOMALY"]
    if anomalies.empty:
        return 0.0
    overage = np.clip(anomalies["latency_gap"], 0, None)
    percent_drop = (overage / 100.0) * 0.01
    return float(percent_drop.sum() * BASE_REVENUE_PER_MINUTE)


def build_latency_series(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"timestamp": [], "latency_ms": []})
    series = (
        df.set_index("ts")["actual_ms"]
        .resample("1min")
        .mean()
        .tail(60)
        .reset_index()
        .rename(columns={"ts": "timestamp", "actual_ms": "latency_ms"})
    )
    return series


def top_anomaly_table(df: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    anomalies = df[df["status"] == "ANOMALY"]
    if anomalies.empty:
        return pd.DataFrame(columns=["Node", "Anomalies (24h)", "Peak Latency (ms)", "Reason", "AI Recommendation"])
    grouped = (
        anomalies.groupby("service")
        .agg(
            anomalies=("service", "count"),
            peak=("actual_ms", "max"),
            reason=("reason", "last"),
            recommendation=("ai_recommendation", "last"),
        )
        .sort_values("anomalies", ascending=False)
        .head(limit)
    )
    grouped = grouped.reset_index().rename(
        columns={
            "service": "Node",
            "anomalies": "Anomalies (24h)",
            "peak": "Peak Latency (ms)",
            "reason": "Reason",
            "recommendation": "AI Recommendation",
        }
    )
    return grouped


def build_weekly_health_pdf(metrics: dict[str, float], df: pd.DataFrame) -> bytes:
    anomalies = df[df["status"] == "ANOMALY"]
    lines = [
        "SentriNode Weekly Health",
        f"Total Requests: {metrics['total_requests']:,}",
        f"Mean Latency: {metrics['mean_latency']:.1f} ms",
        f"System Health: {metrics['system_health']:.2f} %",
        f"Anomalies Flagged: {metrics['anomaly_count']}",
        f"Most Volatile Service: {metrics['volatile_service']} ({metrics['volatile_count']})",
        f"Blast Radius Insights: {metrics['blast_radius']} dependent services impacted",
        f"Avg Prediction Error: {metrics['avg_prediction_error']:.2f} ms",
        f"Anomaly Sample: {', '.join(anomalies['service'].unique()[:5]) or 'None'}",
    ]
    return _render_simple_pdf(lines)


def _render_simple_pdf(lines: list[str]) -> bytes:
    buffer = BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets: list[int] = []

    def write_object(obj: str) -> None:
        offsets.append(buffer.tell())
        buffer.write(obj.encode("latin-1"))

    def sanitize(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    text_lines = ["BT", "/F1 16 Tf", "72 720 Td"]
    for idx, line in enumerate(lines):
        if idx > 0:
            text_lines.append("T*")
        text_lines.append(f"({sanitize(line)}) Tj")
    text_lines.append("ET")
    stream = "\n".join(text_lines)
    stream_bytes = stream.encode("latin-1")

    write_object("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    write_object("2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    write_object(
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
    )
    write_object(f"4 0 obj << /Length {len(stream_bytes)} >> stream\n{stream}\nendstream\nendobj\n")
    write_object("5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")

    xref_start = buffer.tell()
    buffer.write(f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for offset in offsets:
        buffer.write(f"{offset:010} 00000 n \n".encode("latin-1"))
    buffer.write(
        f"trailer << /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF".encode(
            "latin-1"
        )
    )
    return buffer.getvalue()


def purge_old_artifacts(prefixes: list[str]) -> list[str]:
    removed: list[str] = []
    for prefix in prefixes:
        for path in Path(".").glob(f"{prefix}_*.pkl"):
            path.unlink(missing_ok=True)
            removed.append(path.name)
    return removed


def retrain_latest_model(window_hours: int, tenant_id: str | None = None) -> tuple[bool, str]:
    command = [sys.executable, "retrain_ai.py", "--window-hours", str(window_hours)]
    if tenant_id:
        command.extend(["--tenant-id", tenant_id])
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "retrain_ai.py not found in this workspace."

    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return result.returncode == 0, output or "Retrain command completed."


def extract_selected_node(agraph_response: object) -> str | None:
    if not agraph_response:
        return None
    if isinstance(agraph_response, str):
        return agraph_response
    if isinstance(agraph_response, list):
        return agraph_response[0] if agraph_response else None
    if isinstance(agraph_response, dict):
        for key in ("selected_node", "selected_nodes", "nodes"):
            value = agraph_response.get(key)
            if isinstance(value, list) and value:
                return value[0]
            if isinstance(value, str):
                return value
    return None


def send_sentri_alert(message: str, webhook_url: str | None) -> bool:
    if not webhook_url:
        st.warning(
            "Alert webhook not configured. Set SENTRINODE_ALERT_WEBHOOK or update the [webhooks] section in .streamlit/secrets.toml."
        )
        return False
    try:
        response = requests.post(
            webhook_url,
            json={"text": f"SENTRINODE ALERT: {message}"},
            timeout=5,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        st.warning(f"Alert delivery failed: {exc}")
    return False


def _get_active_credentials() -> dict[str, str]:
    cached = st.session_state.get("sentri_credentials")
    if cached and _credentials_configured(cached):
        return cached

    stored = get_secret_section("credentials")
    if _credentials_configured(stored):
        st.session_state["sentri_credentials"] = stored
        _mark_bootstrap_complete()
        return stored

    env_creds = _load_env_credentials()
    if env_creds:
        st.session_state["sentri_credentials"] = env_creds
        _mark_bootstrap_complete()
        return env_creds

    return {}


def _run_initial_setup() -> None:
    _ensure_bootstrap_unlocked()
    st.sidebar.subheader("Initial Setup")
    st.sidebar.success("Bootstrap key accepted. Create admin/viewer credentials.")
    with st.sidebar.form("sentri-bootstrap-credentials"):
        admin_user = st.text_input("Admin Username")
        admin_pwd = st.text_input("Admin Password", type="password")
        viewer_user = st.text_input("Viewer Username")
        viewer_pwd = st.text_input("Viewer Password", type="password")
        submitted = st.form_submit_button("Save Credentials")

    if submitted:
        proposed = {
            "admin_user": admin_user.strip(),
            "admin_pwd": admin_pwd,
            "viewer_user": viewer_user.strip(),
            "viewer_pwd": viewer_pwd,
        }
        if not _credentials_configured(proposed):
            st.sidebar.error("All fields are required to create credentials.")
        else:
            _persist_credentials(proposed)
            st.sidebar.success("Credentials saved. Sign in with your new account.")
            st.session_state.pop("role", None)
            st.rerun()
    st.stop()


def _ensure_bootstrap_unlocked() -> None:
    if _bootstrap_is_complete():
        return
    if not AUTHORIZED_BOOTSTRAP_KEYS:
        st.sidebar.error(
            "Set AUTHORIZED_KEYS (comma-separated) or configure [bootstrap] authorized_keys in .streamlit/secrets.toml to unlock the dashboard."
        )
        st.stop()

    st.sidebar.subheader("Bootstrap Access")
    login_instead = False
    unlocked = False
    with st.sidebar.form("sentri-bootstrap-key"):
        st.write("Enter an authorized bootstrap key to unlock the dashboard.")
        key_value = st.text_input("Setup API Key", type="password")
        buttons = st.columns([0.6, 0.4])
        with buttons[0]:
            unlocked = st.form_submit_button("Unlock Dashboard")
        with buttons[1]:
            login_instead = st.form_submit_button("I have credentials")
    if login_instead:
        st.sidebar.info("Continue with your dashboard credentials.")
        st.session_state["bootstrap_verified"] = True
        return
    if unlocked:
        if key_value.strip() in AUTHORIZED_BOOTSTRAP_KEYS:
            st.session_state["bootstrap_verified"] = True
            st.sidebar.success("Bootstrap verified. Continue to sign in.")
            st.rerun()
        else:
            st.sidebar.error("Invalid API key. Try again.")
    st.stop()


def check_neo4j_connection(uri: str, user: str, password: str) -> bool:
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception as exc:  # pragma: no cover - interactive feedback only
        st.error(f"Connection failed: {exc}")
        return False


def draw_demo_graph() -> None:
    if not all([Config, Node, Edge, agraph]):
        st.info("Install streamlit-agraph to render the mini network map.")
        return
    nodes = [
        Node(id="Gateway", label="API Gateway", color="#00D4FF", size=18),
        Node(id="Payments", label="Payments", color="#FF4B4B", size=22),
        Node(id="Inventory", label="Inventory", color="#00D4FF", size=18),
        Node(id="DB", label="Database", color="#FF4B4B", size=20),
    ]
    edges = [
        Edge(source="Gateway", target="Payments", label="Latency 180ms", color="#FF4B4B"),
        Edge(source="Gateway", target="Inventory", label="Latency 90ms"),
        Edge(source="Payments", target="DB", label="Latency 400ms", color="#FF4B4B"),
    ]
    config = Config(width=500, height=420, directed=True, physics=True)
    agraph(nodes=nodes, edges=edges, config=config)


def require_authentication() -> str:
    credentials = _get_active_credentials()
    if not credentials:
        _run_initial_setup()

    stored_role = st.session_state.get("role")

    if stored_role:
        st.sidebar.success(f"Signed in as {stored_role.title()}")
        if st.sidebar.button("Logout", key="sentri-logout"):
            st.session_state["role"] = None
            st.rerun()
        return stored_role

    with st.sidebar.form("sentri-login"):
        st.write("Sign in to unlock the dashboard.")
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
        if submitted:
            if (
                user == credentials.get("admin_user")
                and password == credentials.get("admin_pwd")
            ):
                st.session_state["role"] = "admin"
                st.rerun()
            elif (
                user == credentials.get("viewer_user")
                and password == credentials.get("viewer_pwd")
            ):
                st.session_state["role"] = "viewer"
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials. Try again.")

    st.stop()


# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
nav_items = [
    ("Overview", "[OV]"),
    ("Analytics", "[AN]"),
    ("Security Events", "[SE]"),
    ("Edge Network Map", "[NM]"),
    ("Graph Console", "[GC]"),
    ("Admin Settings", "[AD]"),
]

neo4j_secret = get_secret_section("neo4j")
webhook_secret = get_secret_section("webhooks")
NEO4J_URI = neo4j_secret.get("uri") or os.getenv("NEO4J_URI")
NEO4J_USER = neo4j_secret.get("user") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = neo4j_secret.get("password") or os.getenv("NEO4J_PASSWORD")
ALERT_WEBHOOK_URL = webhook_secret.get("alert_url") or os.getenv("SENTRINODE_ALERT_WEBHOOK")
driver = None
driver_error = None
if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
    try:
        driver = ge.get_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    except Exception as exc:
        driver_error = str(exc)
else:
    driver_error = "Missing Neo4j credentials. Populate .streamlit/secrets.toml or environment variables."

st.sidebar.title("SentriNode")
role = require_authentication()

available_nav = nav_items if role == "admin" else [item for item in nav_items if item[0] != "Admin Settings"]
nav_lookup = dict(available_nav)
nav_choice = st.sidebar.radio(
    "Navigation",
    options=[item[0] for item in available_nav],
    format_func=lambda name: f"{nav_lookup[name]} {name}",
)
st.sidebar.text_input(
    "Search",
    placeholder="Search services, traces, policies...",
    key="search",
)
tenant_default = st.session_state.get("tenant_field", "sentri-labs")
tenant_label = st.sidebar.text_input(
    "Tenant / Org ID",
    value=tenant_default,
    help="Tenant scope is centrally managed for demos to avoid accidental cross-tenant access.",
    key="tenant_field",
    disabled=True,
)
tenant_id = tenant_label.strip() or None

demo_mode = st.sidebar.checkbox(
    "Demo Mode",
    value=False,
    help="Inject synthetic high-activity traces into the dashboard (and mock Neo4j) for investor demos.",
)
if demo_mode:
    st.sidebar.success("Demo mode active: streaming synthetic anomalies into Neo4j and UI.")
    st.sidebar.caption("Synthetic spans are staged automatically; no customer data is used.")

if role == "admin":
    with st.sidebar.expander("AI Controls", expanded=False):
        st.caption("Retrain on recent spans or prune stale graph history.")
        if st.button("Retrain AI (last 24h)", key="sentri-retrain"):
            removed = purge_old_artifacts(["latency_model", "service_encoder", "load_encoder"])
            success, retrain_output = retrain_latest_model(24, tenant_id=tenant_id)
            if removed:
                st.write("Removed artifacts:", ", ".join(removed))
            if success:
                st.success("Retraining completed.")
            else:
                st.error("Retraining failed.")
            if retrain_output:
                st.code(retrain_output, language="text")
        prune_hours = st.slider(
            "Prune data older than (hours)",
            min_value=24,
            max_value=168,
            value=72,
            step=24,
            key="prune-slider",
        )
        if st.button("Prune Neo4j History", key="sentri-prune"):
            if driver:
                stats = ge.prune_history(driver, prune_hours)
                st.success(
                    f"Detached {stats.get('spans', 0)} spans and {stats.get('relationships', 0)} TRACE relationships."
                )
            else:
                st.warning("Neo4j driver unavailable.")

st.sidebar.caption("Auto-refresh heartbeat: 5s")
st.sidebar.caption(datetime.utcnow().strftime("UTC %Y-%m-%d %H:%M:%S"))

st.sidebar.subheader("Inspector")
if driver_error:
    st.sidebar.error(driver_error)
else:
    selected_node = st.session_state.get("selected_node")
    if driver and selected_node:
        metrics = ge.fetch_service_metrics(driver, selected_node, tenant_id=tenant_id)
        if metrics:
            st.sidebar.metric("Service", selected_node)
            st.sidebar.metric("Avg Latency", f"{metrics['avg_latency']:.1f} ms")
            st.sidebar.metric("Max Latency", f"{metrics['max_latency']:.1f} ms")
            st.sidebar.metric("Requests Observed", f"{metrics['trace_count']:,}")
            st.sidebar.metric("Error Rate", f"{metrics['error_rate']:.2f}%")
        else:
            st.sidebar.info("No traces found for this service yet.")
    else:
        st.sidebar.info("Select a service on the Edge Network Map to inspect details.")

st.sidebar.header("SentriNode Database")
if st.sidebar.button("Test Graph Connection", key="test-neo4j"):
    if check_neo4j_connection(NEO4J_URI or "bolt://sentrinode-db:7687", NEO4J_USER or "neo4j", NEO4J_PASSWORD or ""):
        st.sidebar.success("Connected to Neo4j.")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="sentri-refresh")

# -----------------------------------------------------------------------------
# Logo + model loading status
# -----------------------------------------------------------------------------
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=64)

model = None
service_encoder = None
load_encoder = None
try:
    model = load_latest_artifact("latency_model")
    service_encoder = load_latest_artifact("service_encoder")
    load_encoder = load_latest_artifact("load_encoder")
    st.success("AI Model Loaded Successfully")
except FileNotFoundError:
    st.warning("No AI model found. Waiting for data to train...")
except Exception as exc:
    st.error(f"AI model failed to load: {exc}")
    model = None
    service_encoder = None
    load_encoder = None

# -----------------------------------------------------------------------------
# Header bar with environment/org/search
# -----------------------------------------------------------------------------
header_col1, header_col2, header_col3 = st.columns([1.1, 1.0, 0.9])
with header_col1:
    st.markdown('<div class="env-pill">Environment</div>', unsafe_allow_html=True)
    environment = st.selectbox("Mode", ["Production", "Staging"], label_visibility="collapsed")
    st.write(f"Org: SentriNode Labs")
with header_col2:
    st.write("")
    st.write("")
with header_col3:
    st.write("**Active Session**")
    st.write(datetime.utcnow().strftime("Synced %H:%M:%S UTC"))
    st.write("Tier: Enterprise")

st.divider()

# breadcrumb
breadcrumb = f"SentriNode > {environment} > {nav_choice}"
st.markdown(f"<div style='color: var(--sentri-muted); margin-bottom:12px;'>{breadcrumb}</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Shared data prep
# -----------------------------------------------------------------------------
raw_traces = load_trace_data(demo=demo_mode, tenant=tenant_id)
if tenant_id:
    trace_scope = raw_traces[raw_traces["tenant_id"] == tenant_id]
else:
    trace_scope = raw_traces

search_term = st.session_state.get("search", "").strip()
if search_term:
    mask = (
        trace_scope["service"].str.contains(search_term, case=False, na=False)
        | trace_scope["dependency"].str.contains(search_term, case=False, na=False)
    )
    trace_scope = trace_scope[mask]
trace_df = enrich_trace_data(trace_scope, model, service_encoder, load_encoder)
metrics = summarize_metrics(trace_df)
revenue_risk = estimate_revenue_risk(trace_df)
latency_df = build_latency_series(trace_df)
anomaly_df = top_anomaly_table(trace_df)
trace_history = trace_df.sort_values("ts", ascending=False)

if demo_mode and driver:
    last_seed = st.session_state.get("demo_seeded_ts")
    if not last_seed or (datetime.utcnow() - last_seed).total_seconds() > 30:
        try:
            ge.seed_demo_traces(driver, tenant_id=tenant_id or "demo")
            st.session_state["demo_seeded_ts"] = datetime.utcnow()
        except Exception as exc:  # pragma: no cover - best effort
            st.sidebar.warning(f"Demo seed failed: {exc}")

# -----------------------------------------------------------------------------
# Main Overview content
# -----------------------------------------------------------------------------
if nav_choice == "Overview":
    cards = st.columns(4)
    delta_defs = [
        (
            "Total Requests",
            f"{metrics['total_requests']:,}",
            f"{metrics['request_delta']:+.1f}%",
            True,
        ),
        (
            "Mean Latency (ms)",
            f"{metrics['mean_latency']:.1f}",
            f"{metrics['latency_delta']:+.1f}%",
            False,
        ),
        (
            "Anomaly Count",
            str(metrics["anomaly_count"]),
            f"{metrics['anomaly_delta']:+.1f}%",
            False,
        ),
        (
            "System Health %",
            f"{metrics['system_health']:.2f}",
            f"{metrics['health_delta']:+.1f}%",
            True,
        ),
    ]
    for col, (label, value, delta, positive_is_good) in zip(cards, delta_defs):
        delta_value = float(delta.replace("%", "")) if "%" in delta else float(delta)
        is_positive = delta_value >= 0
        delta_class = "positive" if (is_positive and positive_is_good) or (not is_positive and not positive_is_good) else "negative"
        col.markdown(
            f"""
            <div class='sentri-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{value}</div>
                <div class='metric-delta {delta_class}'>{delta} vs last 10m</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if demo_mode:
        st.warning(
            "Demo Mode: displaying synthetic high-activity telemetry while mock spans are seeded into Neo4j."
        )

    col_top = st.columns([1, 1, 1])
    col_top[0].metric(
        "System Health",
        f"{metrics['system_health']:.2f}%",
        delta=f"{metrics['health_delta']:+.1f}%",
        delta_color="normal",
    )
    col_top[1].metric(
        "Total Anomalies",
        metrics["anomaly_count"],
        delta=f"{metrics['anomaly_delta']:+.1f}%",
        delta_color="inverse",
    )
    col_top[2].metric(
        "Avg Prediction Error",
        f"{metrics['avg_prediction_error']:.2f} ms",
    )

    st.markdown(
        f"""
        <div class='sentri-card'>
            <div class='metric-label'>Revenue at Risk (Latency drag @ 1% per 100ms)</div>
            <div class='metric-value'>${revenue_risk:,.0f}</div>
            <div class='metric-delta {"negative" if revenue_risk > 0 else "positive"}'>
                {'Investigate anomalies to reclaim revenue' if revenue_risk > 0 else 'Latency within conversion-safe band'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    blast_cols = st.columns(3)
    blast_cols[0].markdown(
        f"""
        <div class='sentri-card'>
            <div class='metric-label'>Most Volatile Service</div>
            <div class='metric-value'>{metrics['volatile_service']}</div>
            <div class='metric-delta negative'>{metrics['volatile_count']} anomalies in last 10m</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    blast_cols[1].markdown(
        f"""
        <div class='sentri-card'>
            <div class='metric-label'>Blast Radius</div>
            <div class='metric-value'>{metrics['blast_radius']}</div>
            <div class='metric-delta negative'>Services affected downstream</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    blast_cols[2].markdown(
        f"""
        <div class='sentri-card'>
            <div class='metric-label'>Tenant Scope</div>
            <div class='metric-value'>{tenant_id or 'All tenants'}</div>
            <div class='metric-delta positive'>Breadcrumb: SentriNode &gt; {tenant_id or 'Global'} &gt; Overview</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    layout_left, layout_right = st.columns([2.2, 1.1])
    with layout_left:
        st.subheader("Latency Over Time")
        if latency_df.empty:
            st.info("No latency samples available yet.")
        else:
            st.line_chart(
                data=latency_df.set_index("timestamp"),
                height=360,
                use_container_width=True,
            )
    with layout_right:
        st.subheader("Top Anomalous Nodes")
        if anomaly_df.empty:
            st.info("No anomalies detected in the selected window.")
        else:
            st.dataframe(
                anomaly_df,
                use_container_width=True,
                height=360,
            )

    st.subheader("Network Health Map (Demo)")
    st.caption("Visualize key services and their current latency posture.")
    draw_demo_graph()

    with st.expander("Trace Archive (Full history)"):
        st.caption("Use this expander to audit every trace - not just the latest 19.")
        st.dataframe(
            trace_history[
                [
                    "ts",
                    "service",
                    "dependency",
                    "system_load",
                    "actual_ms",
                    "AI_Predicted_ms",
                    "status",
                    "reason",
                    "ai_recommendation",
                ]
            ],
            use_container_width=True,
            height=360,
        )

elif nav_choice == "Analytics":
    st.subheader("Latency Distribution & Blast Radius")
    if demo_mode:
        st.info("Demo Mode: analytics reflect high-activity synthetic traces.")
    analytics_cols = st.columns((3, 2))
    with analytics_cols[0]:
        if trace_df.empty:
            st.info("No trace data available yet.")
        else:
            if px:
                fig = px.histogram(
                    trace_df,
                    x="actual_ms",
                    color="status",
                    title="Latency Distribution",
                    nbins=30,
                    barmode="overlay",
                )
                fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Plotly is not installed; showing aggregate counts instead.")
                counts = (
                    trace_df.groupby("status")["actual_ms"]
                    .value_counts()
                    .unstack(fill_value=0)
                )
                st.bar_chart(counts.T)
    with analytics_cols[1]:
        st.subheader("Blast Radius Matrix")
        if trace_df.empty:
            st.info("No data to visualize yet.")
        else:
            blast_matrix = (
                trace_df.groupby(["service", "dependency"])
                .agg(
                    traces=("service", "count"),
                    anomalies=("status", lambda s: int((s == "ANOMALY").sum())),
                    avg_latency=("actual_ms", "mean"),
                )
                .reset_index()
            )
            blast_matrix["anomaly_rate"] = (
                blast_matrix["anomalies"] / blast_matrix["traces"] * 100.0
            ).round(2)
            st.dataframe(
                blast_matrix.sort_values("anomaly_rate", ascending=False).head(15),
                use_container_width=True,
                height=360,
            )
    with st.expander("Explainable AI feed"):
        anomaly_feed = trace_df[trace_df["status"] == "ANOMALY"].head(25)
        if anomaly_feed.empty:
            st.info("No anomalies triggered in this window.")
        else:
            st.dataframe(
                anomaly_feed[
                    [
                        "ts",
                        "service",
                        "dependency",
                        "system_load",
                        "actual_ms",
                        "AI_Predicted_ms",
                        "reason",
                        "ai_recommendation",
                    ]
                ],
                use_container_width=True,
                height=380,
            )
    st.download_button(
        "Download Trace CSV",
        data=trace_history.to_csv(index=False),
        file_name="sentri_traces.csv",
        mime="text/csv",
    )

elif nav_choice == "Security Events":
    st.subheader("Security Events & Alerting")
    recent_events = trace_history.head(30).copy()
    if recent_events.empty:
        st.info("No events recorded for this tenant yet.")
    else:
        recent_events["severity"] = np.where(
            recent_events["status"] == "ANOMALY", "Warning", "Healthy"
        )
        st.dataframe(
            recent_events[
                [
                    "ts",
                    "service",
                    "dependency",
                    "system_load",
                    "actual_ms",
                    "AI_Predicted_ms",
                    "severity",
                    "reason",
                    "ai_recommendation",
                ]
            ],
            use_container_width=True,
            height=420,
        )
    st.info("Alerts push directly to Slack/Discord via the send_alert webhook in linker.py.")
    with st.expander("Bottleneck Cypher template"):
        st.code(
            """
MATCH (p:Span {span_id: $anomaly_id})-[:CALLS]->(c:Span)
RETURN c.name as service, c.duration_ns as duration
ORDER BY c.duration_ns DESC LIMIT 1
""".strip(),
            language="cypher",
        )
    if not recent_events.empty:
        sample = recent_events.iloc[0]
        with st.expander("Webhook payload preview"):
            st.json(
                {
                    "text": (
                        f"Anomaly detected! Service: {sample['service']} "
                        f"Latency: {sample['actual_ms']:.1f}ms "
                        f"Recommendation: {sample['ai_recommendation']}"
                    )
                }
            )
        if st.button("Send Immediate Alert", key="security-send-alert"):
            alert_message = (
                f"Service {sample['service']} anomaly at {sample['actual_ms']:.1f}ms "
                f"(expected {sample['AI_Predicted_ms']:.1f}ms) - action: {sample['ai_recommendation']}"
            )
            if send_sentri_alert(alert_message, ALERT_WEBHOOK_URL):
                st.success("Alert dispatched via configured webhook.")
            else:
                st.warning("Alert webhook not configured or request failed.")

elif nav_choice == "Edge Network Map":
    if driver_error:
        st.error(driver_error)
    elif not driver:
        st.error("Neo4j driver unavailable. Configure credentials to render the topology map.")
    elif Config is None or Edge is None or Node is None or agraph is None:
        st.error("streamlit-agraph is required for the topology view. Install it via `pip install streamlit-agraph`.")
    else:
        minutes_back = st.slider(
            "Time-travel replay (minutes ago)",
            0,
            30,
            value=0,
            help="Drag to inspect historical network states.",
        )
        focus_mode = st.checkbox("Focus Mode (hide unrelated nodes)")
        edges_data, node_stats = ge.fetch_topology(
            driver, minutes_back=minutes_back, limit=400, tenant_id=tenant_id
        )
        if not edges_data:
            st.info("No service relationships found for the selected time window.")
        else:
            if (
                "selected_node" not in st.session_state
                or st.session_state["selected_node"] not in node_stats
            ):
                st.session_state["selected_node"] = next(iter(node_stats.keys()), None)
            ts_label = datetime.utcnow() - timedelta(minutes=minutes_back)
            st.markdown(
                f"**Snapshot:** {ts_label.strftime('%H:%M:%S UTC')} ({minutes_back} minutes ago)"
            )
            focus_node = st.session_state.get("selected_node")
            filtered_edges = edges_data
            if focus_mode and focus_node:
                filtered_edges = [
                    edge
                    for edge in edges_data
                    if focus_node in (edge["source"], edge["target"])
                ]
                st.caption(
                    f"Focus Mode active: highlighting blast radius around {focus_node}."
                )
            graph_nodes = []
            for service, stats in node_stats.items():
                color = "#FF4B4B" if stats["max_latency"] > 500 else "#00D4FF"
                size = 26 if service == focus_node else 18
                graph_nodes.append(
                    Node(id=service, label=service, size=size, color=color)
                )
            graph_edges = [
                Edge(
                    source=edge["source"],
                    target=edge["target"],
                    label=f"{int(edge['latency'])} ms",
                    color="#FF4B4B" if edge["latency"] > 500 else "#00D4FF",
                )
                for edge in filtered_edges
            ]
            config = Config(
                width=900,
                height=520,
                directed=True,
                physics=True,
                nodeHighlightBehavior=True,
                collapsible=True,
                tooltipDelay=0,
                hierarchical=False,
            )
            map_col, info_col = st.columns((3, 1.2))
            with map_col:
                st.write(
                    "Zoom out to view the entire SentriNode mesh. The force-directed layout clusters related microservices so SREs can spot architectural drift instantly."
                )
                response = agraph(nodes=graph_nodes, edges=graph_edges, config=config)
                selection = extract_selected_node(response)
                if selection:
                    st.session_state["selected_node"] = selection
                    focus_node = selection
            with info_col:
                st.markdown("#### Command Center")
                st.markdown(
                    """
                    - **Causal Highlighting**: edges jump to #FF4B4B when latency >500ms.
                    - **Node Drill-down**: click nodes to refresh the Inspector sidebar instantly.
                    - **Time-Travel**: drag the slider to jump back 10 minutes and replay the blast radius.
                    - **Focus Mode**: hide unrelated nodes to avoid a spaghetti-ball map at scale.
                    """.strip()
                )
                if demo_mode:
                    st.warning(
                        "Investor demo mode enabled: Payment-Gateway -> User-DB edges pulse when anomalies are injected."
                    )
                st.markdown("#### Investigation Shortcut")
                current_node = st.session_state.get("selected_node")
                if current_node:
                    st.code(
                        f"MATCH p=(a:Service {{name:'{current_node}'}})-[r:TRACE]->(b) "
                        "RETURN a,b,r ORDER BY r.latency DESC LIMIT 5;",
                        language="cypher",
                    )
                    st.caption(
                        "One click surfaces the Cypher query behind the slowdown - cutting investigation time from 20 minutes to 3 seconds."
                    )
            with st.expander("Advanced Graph Query"):
                default_query = "MATCH (s:Service)-[t:TRACE]->(d:Service)\nRETURN s.name AS source, d.name AS target, t.latency AS latency\nLIMIT 25"
                cypher_text = st.text_area(
                    "Cypher Console",
                    value=default_query,
                    height=140,
                    key="custom_cypher",
                )
                if st.button("Execute", key="run-custom-cypher"):
                    try:
                        records = ge.run_custom_query(driver, cypher_text)
                        if records:
                            st.dataframe(pd.DataFrame(records))
                        else:
                            st.info("Query executed successfully but returned no rows.")
                    except Exception as exc:
                        st.error(f"Cypher execution failed: {exc}")

elif nav_choice == "Graph Console":
    st.subheader("Neo4j Command Console")
    if driver_error:
        st.error(driver_error)
    elif not driver:
        st.warning("Neo4j driver unavailable. Configure credentials before running Cypher.")
    else:
        st.caption(
            "Run Cypher directly against the connected graph. Queries execute with the same credentials powering the Edge Network Map."
        )
        default_console = st.session_state.get(
            "console_query",
            "MATCH (s:Service)-[t:TRACE]->(d:Service)\nRETURN s.name AS source, d.name AS target, t.latency AS latency\nLIMIT 25",
        )
        console_query = st.text_area(
            "Cypher Query",
            value=default_console,
            height=220,
            key="graph-console-input",
        )
        run_col, clear_col = st.columns([0.2, 0.8])
        with run_col:
            run_console = st.button("Run Query", key="graph-console-run")
        with clear_col:
            if st.button("Clear Output", key="graph-console-clear"):
                st.session_state.pop("graph_console_results", None)
                st.rerun()
        if run_console:
            st.session_state["console_query"] = console_query
            try:
                records = ge.run_custom_query(driver, console_query)
                st.session_state["graph_console_results"] = records
            except Exception as exc:
                st.error(f"Query failed: {exc}")
        output = st.session_state.get("graph_console_results")
        if output:
            st.success(f"Returned {len(output)} row(s).")
            st.dataframe(pd.DataFrame(output), use_container_width=True)
        else:
            st.info("Results will appear here after you run a query.")
        st.caption("Need inspiration? Try `CALL db.schema.visualization()` or `MATCH (n) RETURN n LIMIT 5`.")

elif nav_choice == "Admin Settings":
    if role != "admin":
        st.error("Administrator access required.")
    else:
        st.subheader("Admin Settings")
        st.info(
            "Use one of the AUTHORIZED_KEYS values for initial access. Credential values live in .streamlit/secrets.toml while environment overrides (e.g., AUTHORIZED_KEYS) belong in .env."
        )
        pdf_bytes = build_weekly_health_pdf(metrics, trace_df)
        st.download_button(
            "Download Weekly Health PDF",
            data=pdf_bytes,
            file_name="sentri_weekly_health.pdf",
            mime="application/pdf",
        )
        with st.expander("Rotate Dashboard Credentials"):
            current_creds = _get_active_credentials()
            st.caption("Updating credentials logs everyone out immediately.")
            with st.form("sentri-rotate-creds"):
                admin_user = st.text_input("Admin Username", value=current_creds.get("admin_user", ""))
                admin_pwd = st.text_input("Admin Password", type="password")
                viewer_user = st.text_input("Viewer Username", value=current_creds.get("viewer_user", ""))
                viewer_pwd = st.text_input("Viewer Password", type="password")
                rotate_submitted = st.form_submit_button("Update Credentials")
            if rotate_submitted:
                new_values = {
                    "admin_user": admin_user.strip(),
                    "admin_pwd": admin_pwd,
                    "viewer_user": viewer_user.strip(),
                    "viewer_pwd": viewer_pwd,
                }
                if not _credentials_configured(new_values):
                    st.error("All credential fields are required.")
                else:
                    _persist_credentials(new_values)
                    st.success("Credentials updated. Please sign in again.")
                    st.session_state.pop("role", None)
                    st.session_state.pop("sentri_credentials", None)
                    st.rerun()
        st.markdown(
            """
            **Env & Secrets**

            - `.env` stores local-only environment overrides (e.g., AUTHORIZED_KEYS, Neo4j URI/credentials).
            - `.streamlit/secrets.toml` is the canonical store for dashboard credentials and bootstrap API keys.
            - Update both when rotating secrets to avoid drift across environments.
            """
        )
        st.markdown(
            """
            **Maintenance Tips**

            - Use the sidebar controls to retrain the AI on the last 24 hours of traces.
            - Prune historical spans regularly to keep Neo4j lean (DETACH DELETE is automated via the sidebar).
            - Demo mode injects synthetic traces via `seed_demo_traces` so investors always see a "breathing" graph.
            """
        )

else:
    st.info("Select a navigation item to continue.")
