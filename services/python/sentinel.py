#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase, Session

from ai_artifacts import load_latest_artifact

load_dotenv()

WEBHOOK_URL = os.getenv("WATCHDOG_WEBHOOK_URL")
ALERT_LOG = Path("alerts.log")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
TENANT_ID = os.getenv("TENANT_ID") or os.getenv("DEFAULT_TENANT_ID", "public")

if not NEO4J_PASSWORD:
    raise SystemExit("Set the NEO4J_PASSWORD environment variable before running sentinel.py")

try:
    model = load_latest_artifact("latency_model")
    le_service = load_latest_artifact("service_encoder")
    le_load = load_latest_artifact("load_encoder")
except FileNotFoundError:
    print("Error: Could not find model files. Run train_ai.py first!")
    raise SystemExit(1)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

print("Sentinel active... monitoring Neo4j for new traces.")


def suggest_remediation(service: Optional[str], load: Optional[str]) -> str:
    svc = service or "the affected service"
    load_level = (load or "unknown").lower()
    if load_level == "high":
        return f"High load detected. Recommendation: spin up 2 additional instances of {svc}."
    if load_level == "medium":
        return f"Moderate load observed. Recommendation: route traffic through standby replicas for {svc}."
    return f"Recommendation: inspect recent deploys and dependency latency for {svc}."


def find_bottleneck_child(session: Session, parent_span_id: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    if not parent_span_id:
        return None, None
    query = """
    MATCH (p:Span {span_id: $anomaly_id})-[:CALLS]->(c:Span)
    WHERE p.tenant_id = $tenant_id AND c.tenant_id = $tenant_id
    RETURN c.name AS service, c.duration_ns AS duration
    ORDER BY c.duration_ns DESC
    LIMIT 1
    """
    record = session.run(query, anomaly_id=parent_span_id, tenant_id=TENANT_ID).single()
    if not record:
        return None, None
    duration_ns = record.get("duration") or 0
    duration_ms = float(duration_ns) / 1_000_000.0
    return record.get("service"), duration_ms


def format_alert_message(alert: dict) -> str:
    message = (
        f"Anomaly Detected! Service: {alert['child']} | Load: {alert['load']} | "
        f"Latency {alert['actual_ms']:.1f} ms (expected {alert['predicted_ms']:.1f} ms)."
    )
    suspect = alert.get("suspected_service")
    suspect_ms = alert.get("suspected_duration_ms")
    if suspect and suspect_ms is not None:
        message += f" Suspected bottleneck: {suspect} ~{suspect_ms:.1f} ms."
    recommendation = alert.get("recommendation")
    if recommendation:
        message += f" {recommendation}"
    return message


def emit_alert(payload: dict, message: Optional[str] = None) -> None:
    record = payload.copy()
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    if message:
        record["message"] = message
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ALERT_LOG.open("a", encoding="utf-8") as logfile:
        logfile.write(json.dumps(record) + "\n")

    if WEBHOOK_URL:
        try:
            body = {"text": message or record.get("status", "Anomaly"), "details": record}
            resp = requests.post(WEBHOOK_URL, json=body, timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"Failed to send webhook: {exc}")


def check_for_anomalies():
    query = """
    MATCH (p:Span)-[:CALLS]->(c:Span)
    WHERE p.tenant_id = $tenant_id AND c.tenant_id = $tenant_id
    RETURN p.name as p_name, c.name as c_name, p.system_load as load,
           p.duration_ns/1000000.0 as p_ms,
           c.duration_ns/1000000.0 as actual_ms,
           c.trace_id as trace_id,
           p.span_id as parent_span_id
    ORDER BY coalesce(c.timestamp, c.end_time_unix_nano, c.start_time_unix_nano, timestamp()) DESC
    LIMIT 1
    """
    with driver.session() as session:
        result = session.run(query, tenant_id=TENANT_ID).single()
        if not result:
            return

        try:
            input_df = pd.DataFrame(
                [[
                    le_service.transform([result["p_name"]])[0],
                    le_service.transform([result["c_name"]])[0],
                    le_load.transform([result["load"]])[0],
                    result["p_ms"],
                ]],
                columns=["parent_id", "child_id", "load_id", "parent_ms"],
            )

            predicted_ms = float(model.predict(input_df)[0])
            actual_ms = float(result["actual_ms"])
            parent_span_id = result.get("parent_span_id")

            if actual_ms > (predicted_ms * 1.5):
                suspect_service, suspect_ms = find_bottleneck_child(
                    session, parent_span_id
                )
                alert = {
                    "status": "ANOMALY",
                    "parent": result["p_name"],
                    "child": result["c_name"],
                    "actual_ms": actual_ms,
                    "predicted_ms": predicted_ms,
                    "load": result["load"],
                    "trace_id": result["trace_id"],
                    "suspected_service": suspect_service,
                    "suspected_duration_ms": suspect_ms,
                    "recommendation": suggest_remediation(
                        suspect_service or result["c_name"], result["load"]
                    ),
                }
                message = format_alert_message(alert)
                emit_alert(alert, message)
                print(f"{message} trace={result['trace_id']}")
            else:
                print(f"Healthy: {result['c_name']} ({actual_ms:.2f}ms)")
        except ValueError:
            print(
                f"New service detected: {result['c_name']}."
                " AI needs retraining to understand this."
            )


while True:
    check_for_anomalies()
    time.sleep(3)
