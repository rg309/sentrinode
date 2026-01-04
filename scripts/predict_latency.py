#!/usr/bin/env python3
"""
Fetches span relationships from Neo4j, trains a RandomForestRegressor on the CALLS edges,
and predicts latency for a supplied parent/child service pair.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
from neo4j import GraphDatabase
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

MODEL_BUNDLE: Optional[Dict[str, object]] = None


def fetch_span_relationships(uri: str, user: str, password: str, tenant_id: str) -> List[Dict[str, object]]:
    """
    Pulls all CALLS relationships and returns rows with caller/target names plus latency (ns).
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = """
    MATCH (parent:Span)-[:CALLS]->(child:Span)
    WHERE parent.tenant_id = $tenant_id AND child.tenant_id = $tenant_id
    RETURN parent.name AS caller_service,
           child.name AS target_service,
           child.duration_ns AS target_duration_ns
    """
    try:
        with driver.session() as session:
            records = session.run(query, tenant_id=tenant_id)
            return [record.data() for record in records]
    finally:
        driver.close()


def build_training_frame(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["caller_service", "target_service", "target_duration_ns"])
    if df.empty:
        return df
    df["target_latency_ms"] = (
        df["target_duration_ns"].astype(float) / 1_000_000.0
    )
    return df[["caller_service", "target_service", "target_latency_ms"]]


def train_model(df: pd.DataFrame) -> Dict[str, object]:
    parent_encoder = LabelEncoder()
    child_encoder = LabelEncoder()

    parent_encoded = parent_encoder.fit_transform(df["caller_service"])
    child_encoded = child_encoder.fit_transform(df["target_service"])
    features: List[Tuple[int, int]] = list(zip(parent_encoded, child_encoded))
    target = df["target_latency_ms"].astype(float)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(features, target)

    return {
        "model": model,
        "parent_encoder": parent_encoder,
        "child_encoder": child_encoder,
    }


def _predict_with_bundle(model_bundle: Dict[str, object], parent: str, child: str) -> float:
    parent_encoder: LabelEncoder = model_bundle["parent_encoder"]
    child_encoder: LabelEncoder = model_bundle["child_encoder"]
    model: RandomForestRegressor = model_bundle["model"]

    if parent not in parent_encoder.classes_ or child not in child_encoder.classes_:
        raise ValueError(
            f"Unknown service pair parent={parent!r}, child={child!r}; "
            "train data does not contain this combination."
        )

    parent_val = parent_encoder.transform([parent])[0]
    child_val = child_encoder.transform([child])[0]
    predicted = model.predict([(parent_val, child_val)])[0]
    return float(predicted)


def predict_latency(parent_name: str, child_name: str) -> float:
    """
    Predicts latency in milliseconds using the globally trained model bundle.
    """
    if MODEL_BUNDLE is None:
        raise RuntimeError("Model not trained. Call load_model(...) or run this script directly first.")
    return _predict_with_bundle(MODEL_BUNDLE, parent_name, child_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict span latency from Neo4j relationships.")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j bolt URI.")
    parser.add_argument("--user", default="neo4j", help="Neo4j username.")
    parser.add_argument("--password", default="testpassword", help="Neo4j password.")
    parser.add_argument("--parent", default="ServiceA", help="Caller service to predict from.")
    parser.add_argument("--child", default="ServiceB", help="Target service to predict to.")
    parser.add_argument(
        "--tenant-id",
        default=os.getenv("TENANT_ID") or os.getenv("DEFAULT_TENANT_ID", "public"),
        help="Tenant/org identifier to scope queries.",
    )
    return parser.parse_args()


def main() -> None:
    global MODEL_BUNDLE
    args = parse_args()
    rows = fetch_span_relationships(args.uri, args.user, args.password, args.tenant_id)

    if not rows:
        sys.exit("No CALLS relationships found in Neo4j; train data is empty.")

    df = build_training_frame(rows)
    if df.empty:
        sys.exit("Neo4j returned spans but no valid latency data.")

    MODEL_BUNDLE = train_model(df)

    try:
        latency = predict_latency(args.parent, args.child)
    except ValueError as exc:
        sys.exit(str(exc))

    print(f"Predicted latency for {args.parent} -> {args.child}: {latency:.2f} ms")


if __name__ == "__main__":
    main()
