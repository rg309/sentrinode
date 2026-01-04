#!/usr/bin/env python3
"""Shared Neo4j helpers for the SentriNode Streamlit dashboard."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import streamlit as st
from neo4j import GraphDatabase


@st.cache_resource(show_spinner=False)
def get_driver(uri: str, user: str, password: str):
    """Create or reuse a cached Neo4j driver."""
    return GraphDatabase.driver(uri, auth=(user, password))


def fetch_topology(
    driver,
    minutes_back: int = 0,
    limit: int = 200,
    tenant_id: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Return recent service-to-service traces plus aggregate node metrics."""
    edge_query = """
    MATCH (s:Service)-[t:TRACE]->(d:Service)
    WHERE ($tenant_id IS NULL OR (s.tenant_id = $tenant_id AND d.tenant_id = $tenant_id))
      AND ($window_ms = 0 OR coalesce(t.timestamp, timestamp()) >= timestamp() - $window_ms)
    RETURN s.name AS source,
           d.name AS target,
           coalesce(t.latency, 0) AS latency
    ORDER BY coalesce(t.timestamp, timestamp()) DESC
    LIMIT $limit
    """
    node_query = """
    MATCH (s:Service)-[t:TRACE]->()
    WHERE $tenant_id IS NULL OR s.tenant_id = $tenant_id
    RETURN s.name AS service,
           coalesce(avg(t.latency), 0) AS avg_latency,
           coalesce(max(t.latency), 0) AS max_latency,
           sum(CASE WHEN coalesce(t.error,false) THEN 1 ELSE 0 END) AS error_count,
           count(t) AS trace_count
    """
    window_ms = minutes_back * 60 * 1000
    edges: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, float]] = {}
    with driver.session() as session:
        for record in session.run(
            edge_query,
            limit=limit,
            window_ms=window_ms,
            tenant_id=tenant_id,
        ):
            edges.append(
                {
                    "source": record["source"],
                    "target": record["target"],
                    "latency": float(record["latency"] or 0),
                }
            )
        for record in session.run(node_query, tenant_id=tenant_id):
            traces = int(record["trace_count"] or 0)
            errors = int(record["error_count"] or 0)
            error_rate = (errors / traces * 100) if traces else 0.0
            stats[record["service"]] = {
                "avg_latency": float(record["avg_latency"] or 0),
                "max_latency": float(record["max_latency"] or 0),
                "trace_count": traces,
                "error_rate": error_rate,
            }
    return edges, stats


def fetch_service_metrics(driver, service: str, tenant_id: str | None = None) -> Dict[str, Any]:
    """Return per-service metrics for the inspector sidebar."""
    query = """
    MATCH (s:Service {name: $service})-[t:TRACE]->()
    WHERE $tenant_id IS NULL OR (s.tenant_id = $tenant_id AND coalesce(t.tenant_id, s.tenant_id) = $tenant_id)
    RETURN coalesce(avg(t.latency),0) AS avg_latency,
           coalesce(max(t.latency),0) AS max_latency,
           count(t) AS trace_count,
           sum(CASE WHEN coalesce(t.error,false) THEN 1 ELSE 0 END) AS error_count
    """
    with driver.session() as session:
        record = session.run(query, service=service, tenant_id=tenant_id).single()
        if not record:
            return {}
        traces = int(record["trace_count"] or 0)
        errors = int(record["error_count"] or 0)
        error_rate = (errors / traces * 100) if traces else 0.0
        return {
            "avg_latency": float(record["avg_latency"] or 0),
            "max_latency": float(record["max_latency"] or 0),
            "trace_count": traces,
            "error_rate": error_rate,
        }


def run_custom_query(driver, cypher: str) -> List[Dict[str, Any]]:
    """Execute an arbitrary Cypher query and return the raw records as dicts."""
    if not cypher.strip():
        return []
    with driver.session() as session:
        result = session.run(cypher)
        return [record.data() for record in result]


def seed_demo_traces(driver, tenant_id: str = "demo", batch: int = 50) -> None:
    """Populate Neo4j with synthetic traces for demo mode."""
    relations = [
        ("Auth-Service", "User-DB"),
        ("Payment-Gateway", "Inventory-API"),
        ("Inventory-API", "User-DB"),
        ("Auth-Service", "Legacy-Mainframe"),
    ]
    with driver.session() as session:
        for _ in range(batch):
            source, target = random.choice(relations)
            latency = random.uniform(20, 150)
            if random.random() > 0.9:
                latency = random.uniform(2000, 5000)
            session.run(
                """
                MERGE (a:Service {name: $source, tenant_id: $tenant})
                MERGE (b:Service {name: $target, tenant_id: $tenant})
                CREATE (a)-[:TRACE {
                    timestamp: timestamp(),
                    latency: $latency,
                    tenant_id: $tenant,
                    error: CASE WHEN $latency > 180 THEN true ELSE false END
                }]->(b)
                """,
                source=source,
                target=target,
                latency=latency,
                tenant=tenant_id,
            )


def prune_history(driver, older_than_hours: int) -> Dict[str, int]:
    """Detach delete aged spans and trace relationships."""
    window_ms = older_than_hours * 60 * 60 * 1000
    rel_query = """
    MATCH ()-[t:TRACE]->()
    WHERE coalesce(t.timestamp, timestamp()) < timestamp() - $window
    WITH t LIMIT 1000
    DELETE t
    RETURN count(*) AS deleted
    """
    span_query = """
    MATCH (s:Span)
    WHERE coalesce(s.timestamp, timestamp()) < timestamp() - $window
    WITH s LIMIT 500
    DETACH DELETE s
    RETURN count(*) AS deleted
    """
    with driver.session() as session:
        rel_deleted = session.run(rel_query, window=window_ms).single()["deleted"]
        span_deleted = session.run(span_query, window=window_ms).single()["deleted"]
    return {"relationships": int(rel_deleted or 0), "spans": int(span_deleted or 0)}
