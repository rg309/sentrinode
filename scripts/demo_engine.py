#!/usr/bin/env python3
"""Injects synthetic SentriNode demo traces into Neo4j for investor demos."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from neo4j import GraphDatabase

# Demo settings
NODES = [
    "Auth-Service",
    "Payment-Gateway",
    "User-DB",
    "Inventory-API",
    "Legacy-Mainframe",
]
RELATIONS = [
    ("Auth-Service", "User-DB"),
    ("Payment-Gateway", "Inventory-API"),
    ("Inventory-API", "User-DB"),
    ("Auth-Service", "Legacy-Mainframe"),
    ("Payment-Gateway", "Legacy-Mainframe"),
]


@dataclass
class DemoConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "testpassword"
    batch: int = 50
    anomaly_probability: float = 0.1


def generate_mock_traces(cfg: DemoConfig) -> None:
    driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
    injected = 0
    try:
        with driver.session() as session:
            for _ in range(cfg.batch):
                source, target = random.choice(RELATIONS)
                latency = random.uniform(20, 150)
                if random.random() < cfg.anomaly_probability:
                    latency = random.uniform(2000, 5000)
                session.run(
                    """
                    MERGE (a:Service {name: $source})
                    MERGE (b:Service {name: $target})
                    CREATE (a)-[:TRACE {
                        timestamp: timestamp(),
                        latency: $latency,
                        demo: true
                    }]->(b)
                    """,
                    source=source,
                    target=target,
                    latency=latency,
                )
                injected += 1
                time.sleep(0.05)
    finally:
        driver.close()
    print(f"SentriNode Engine: {injected} synthetic traces injected (demo mode).")


if __name__ == "__main__":
    config = DemoConfig()
    generate_mock_traces(config)
