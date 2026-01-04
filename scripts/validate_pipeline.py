#!/usr/bin/env python3
"""Pipeline integrity validation helper.

Usage:
  python scripts/validate_pipeline.py \
    --trace-id <trace> \
    --expected-path gateway,orchestrator,payments,recommendation,db \
    --service payments \
    --event-ts 1715020000

Requires kafka-python, neo4j, and requests (already used elsewhere in repo).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

import requests
from kafka import KafkaConsumer
from neo4j import GraphDatabase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Kafka -> Neo4j -> Prometheus alignment")
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--expected-path", required=True, help="Comma separated service order")
    parser.add_argument("--service", required=True, help="Service name for latency check")
    parser.add_argument("--event-ts", type=float, required=True, help="Epoch seconds of the span of interest")
    parser.add_argument("--kafka-bootstrap", default="localhost:9093")
    parser.add_argument("--kafka-topic", default="raw_spans_topic")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="testpassword")
    parser.add_argument("--prom-url", default="http://localhost:9090")
    parser.add_argument(
        "--latency-metric",
        default="service_latency_ms",
        help="Prometheus histogram/counter used for latency",
    )
    parser.add_argument("--threshold", type=float, default=500.0, help="p99 breach threshold")
    return parser.parse_args()


def verify_kafka(trace_id: str, bootstrap: str, topic: str, timeout_s: int = 10) -> bool:
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=[bootstrap],
        auto_offset_reset="earliest",
        group_id=None,
        consumer_timeout_ms=timeout_s * 1000,
        enable_auto_commit=False,
    )
    for message in consumer:
        key = (message.key or b"").decode("utf-8")
        if key == trace_id:
            print(f"Kafka: located trace {trace_id} at offset {message.offset}")
            consumer.close()
            return True
    consumer.close()
    print("Kafka: trace not found in allotted time.")
    return False


def verify_graph(trace_id: str, expected_path: List[str], uri: str, user: str, password: str) -> bool:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = """
    MATCH path = (start:Service {trace_id: $trace_id})-[:CALLS*]->(end)
    RETURN [n IN nodes(path) | n.name] AS nodes
    """
    with driver.session() as session:
        records = session.run(query, trace_id=trace_id).values()
    driver.close()
    if not records:
        print("Neo4j: no path found for trace", trace_id)
        return False
    longest = max(records, key=lambda rec: len(rec[0]))[0]
    print("Neo4j path:", " -> ".join(longest))
    mismatch = [a for a, b in zip(expected_path, longest) if a != b]
    if mismatch:
        print("Neo4j: mismatch vs expected path", expected_path)
        return False
    return True


def verify_latency(
    prom_url: str,
    latency_metric: str,
    service: str,
    event_ts: float,
    threshold: float,
) -> bool:
    start = event_ts - 120
    end = event_ts + 120
    query = (
        "histogram_quantile(0.99, sum(rate("
        f"{latency_metric}{{service='{service}'}}[5m])) by (le))"
    )
    resp = requests.get(
        f"{prom_url}/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": 15},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {}).get("result", [])
    if not data:
        print("Prometheus: no data returned for query")
        return False
    samples = data[0]["values"]
    breaches = [float(value) for _, value in samples if float(value) > threshold]
    print("Prometheus samples:", json.dumps(samples[-5:], indent=2))
    if not breaches:
        print("Prometheus: no p99 breach detected in window")
        return False
    print(f"Prometheus: detected {len(breaches)} breach samples above {threshold} ms")
    return True


def main() -> None:
    args = parse_args()
    expected_path = [seg.strip() for seg in args.expected_path.split(",")]

    ok_kafka = verify_kafka(args.trace_id, args.kafka_bootstrap, args.kafka_topic)
    ok_graph = verify_graph(
        args.trace_id, expected_path, args.neo4j_uri, args.neo4j_user, args.neo4j_password
    )
    ok_latency = verify_latency(
        args.prom_url,
        args.latency_metric,
        args.service,
        args.event_ts,
        args.threshold,
    )

    if all([ok_kafka, ok_graph, ok_latency]):
        print("Pipeline validation: SUCCESS")
        sys.exit(0)
    print("Pipeline validation: FAILURE")
    sys.exit(1)


if __name__ == "__main__":
    main()
