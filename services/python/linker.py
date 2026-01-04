#!/usr/bin/env python3
import os
import sys
import time

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
pwd = os.getenv("NEO4J_PASSWORD", "Delahrg12")
WEBHOOK_URL = os.getenv("SENTRINODE_ALERT_WEBHOOK")

BOTTLENECK_QUERY = """
MATCH (p:Span {span_id: $anomaly_id})-[:CALLS]->(c:Span)
RETURN c.name as service, c.duration_ns as duration
ORDER BY c.duration_ns DESC
LIMIT 1
"""

ANOMALY_SWEEP = """
MATCH (p:Span)
WHERE (coalesce(p.status,'') = 'ANOMALY' OR coalesce(p.is_anomaly,false) = true)
  AND coalesce(p.bottleneck_noted,false) = false
RETURN p.span_id AS span_id, coalesce(p.name,'unknown') AS service, coalesce(p.trace_id, p.span_id) AS trace_id
LIMIT 5
"""


def send_alert(message: str) -> None:
    if not WEBHOOK_URL:
        return
    try:
        response = requests.post(WEBHOOK_URL, json={"text": message}, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[alert] webhook delivery failed: {exc}")


def diagnose_anomalies(session) -> None:
    """Inspect anomaly parents and tag the worst child span."""
    records = session.run(ANOMALY_SWEEP).data()
    for record in records:
        anomaly_id = record["span_id"]
        bottleneck = session.run(BOTTLENECK_QUERY, anomaly_id=anomaly_id).single()
        if bottleneck:
            child_service = bottleneck["service"]
            duration = bottleneck["duration"]
            session.run(
                """
                MATCH (p:Span {span_id: $span_id})
                SET p.bottleneck_noted = true,
                    p.bottleneck_child = $child_service,
                    p.bottleneck_child_duration = $duration
                """,
                span_id=anomaly_id,
                child_service=child_service,
                duration=duration,
            )
            send_alert(
                f"Anomaly detected! Service: {record['service']} "
                f"Impact: Latency spike traced to {child_service} ({duration}ns)."
            )
        else:
            session.run(
                "MATCH (p:Span {span_id: $span_id}) SET p.bottleneck_noted = true",
                span_id=anomaly_id,
            )


def run_linker():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", pwd))

    try:
        driver.verify_connectivity()
        print("Connection verified. Starting auto-linker...")
    except Exception as exc:
        print(f"AUTH FAILURE: {exc}")
        print("Fix your password before running this again to avoid lockout!")
        sys.exit(1)

    query = """
    MATCH (child:Span), (parent:Span)
    WHERE child.parent_id = parent.id AND child.parent_id <> "" AND NOT (parent)-[:CALLS]->(child)
    MERGE (parent)-[:CALLS]->(child)
    RETURN count(*) as new_links
    """

    try:
        while True:
            with driver.session() as session:
                result = session.run(query).single()
                if result and result["new_links"] > 0:
                    print(f"Linked {result['new_links']} new spans.")
                diagnose_anomalies(session)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        driver.close()


if __name__ == "__main__":
    run_linker()
