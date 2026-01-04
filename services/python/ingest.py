#!/usr/bin/env python3
from __future__ import annotations

import random
import time

from neo4j import GraphDatabase

URI = "bolt://sentrinode-db:7687"
AUTH = ("neo4j", "Delahrg12")
SERVICES = ["Gateway", "Payments", "Inventory", "Auth", "DB"]


def update_service_latency(service_name: str) -> None:
    new_latency = random.randint(20, 500)
    status = "Healthy" if new_latency < 200 else "Warning"
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        query = """
        MERGE (s:Service {name: $name})
        SET s.latency = $latency,
            s.status = $status,
            s.last_updated = timestamp()
        """
        driver.execute_query(query, name=service_name, latency=new_latency, status=status)
        print(f"Updated {service_name}: {new_latency}ms ({status})")


def main() -> None:
    while True:
        update_service_latency(random.choice(SERVICES))
        time.sleep(5)


if __name__ == "__main__":
    main()
