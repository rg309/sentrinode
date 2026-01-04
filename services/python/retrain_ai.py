#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from neo4j import GraphDatabase
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ai_artifacts import save_artifact


def fetch_recent_spans(
    uri: str,
    user: str,
    password: str,
    limit: int,
    tenant_id: str,
) -> List[Dict[str, object]]:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = """
    MATCH (p:Span)-[:CALLS]->(c:Span)
    WHERE p.tenant_id = $tenant_id AND c.tenant_id = $tenant_id
    RETURN p.name AS parent_service,
           c.name AS child_service,
           coalesce(p.system_load, 'Unknown') AS system_load,
           p.duration_ns / 1000000.0 AS parent_ms,
           c.duration_ns / 1000000.0 AS target_ms,
           c.timestamp AS child_timestamp,
           c.end_time_unix_nano AS child_end_ns,
           c.start_time_unix_nano AS child_start_ns
    ORDER BY coalesce(c.timestamp, c.end_time_unix_nano, c.start_time_unix_nano, timestamp()) DESC
    LIMIT $limit
    """
    try:
        with driver.session() as session:
            records = session.run(query, limit=limit, tenant_id=tenant_id)
            return [record.data() for record in records]
    finally:
        driver.close()


def train_from_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        sys.exit("Neo4j did not return any CALLS relationships to train on.")

    le_service = LabelEncoder()
    le_load = LabelEncoder()

    all_services = pd.concat([df["parent_service"], df["child_service"]])
    le_service.fit(all_services)

    df["parent_id"] = le_service.transform(df["parent_service"])
    df["child_id"] = le_service.transform(df["child_service"])
    df["load_id"] = le_load.fit_transform(df["system_load"])

    X = df[["parent_id", "child_id", "load_id", "parent_ms"]]
    y = df["target_ms"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    model_path = save_artifact(model, "latency_model")
    service_enc_path = save_artifact(le_service, "service_encoder")
    load_enc_path = save_artifact(le_load, "load_encoder")

    print("Retraining complete from live Neo4j data.")
    print(f"Dataset rows: {len(df)} | Accuracy: {model.score(X_test, y_test):.2f}")
    print(f"Artifacts -> {model_path}, {service_enc_path}, {load_enc_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain latency model from recent Neo4j spans.")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j bolt URI.")
    parser.add_argument("--user", default="neo4j", help="Neo4j username.")
    parser.add_argument("--password", default="testpassword", help="Neo4j password.")
    parser.add_argument("--limit", type=int, default=10000, help="Number of recent spans to fetch.")
    parser.add_argument(
        "--window-hours",
        type=int,
        default=0,
        help="Restrict training data to spans observed in the past N hours (0 = all data).",
    )
    parser.add_argument(
        "--tenant-id",
        default=os.getenv("TENANT_ID") or os.getenv("DEFAULT_TENANT_ID", "public"),
        help="Tenant/org identifier used to scope spans.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    since_ts: Optional[datetime] = None
    if args.window_hours > 0:
        since_ts = datetime.now(timezone.utc) - timedelta(hours=args.window_hours)

    rows = fetch_recent_spans(args.uri, args.user, args.password, args.limit, args.tenant_id)
    if not rows:
        sys.exit("No spans returned from Neo4j; aborting retrain.")

    df = pd.DataFrame(rows)
    timestamp_cols = ["child_timestamp", "child_end_ns", "child_start_ns"]
    if since_ts is not None:
        df["event_ts"] = pd.to_datetime(df["child_timestamp"], utc=True, errors="coerce")
        end_ts = pd.to_datetime(df["child_end_ns"], utc=True, errors="coerce", unit="ns")
        start_ts = pd.to_datetime(df["child_start_ns"], utc=True, errors="coerce", unit="ns")
        df["event_ts"] = df["event_ts"].fillna(end_ts).fillna(start_ts)
        df = df[df["event_ts"] >= pd.Timestamp(since_ts)]

    df = df.drop(columns=["event_ts"], errors="ignore")
    df = df.drop(columns=timestamp_cols, errors="ignore")
    df = df.dropna(subset=["parent_service", "child_service", "system_load", "parent_ms", "target_ms"])
    df["system_load"] = df["system_load"].fillna("Unknown")

    if df.empty:
        sys.exit("No spans remained after applying the requested time window.")

    train_from_dataframe(df)


if __name__ == "__main__":
    main()
