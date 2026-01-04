#!/usr/bin/env python3
"""
Delete historical Span nodes that are older than the configured TTL.

Usage:
    python scripts/prune_history.py --uri bolt://localhost:7687 --user neo4j \
        --password testpassword --ttl-days 30 --batch-size 5000
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from typing import Optional

from neo4j import GraphDatabase


def prune_history(
    uri: str,
    user: str,
    password: str,
    ttl_days: int,
    batch_size: int,
) -> int:
    if ttl_days <= 0:
        raise ValueError("ttl_days must be positive")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    ttl_ms = int(timedelta(days=ttl_days).total_seconds() * 1000)
    total_deleted = 0
    try:
        with driver.session() as session:
            while True:
                deleted = session.write_transaction(
                    _delete_batch, ttl_ms=ttl_ms, batch_size=batch_size
                )
                if deleted == 0:
                    break
                total_deleted += deleted
                print(f"Deleted {deleted} spans older than {ttl_days} days...")
    finally:
        driver.close()
    return total_deleted


def _delete_batch(tx, ttl_ms: int, batch_size: int) -> int:
    query = """
    MATCH (s:Span)
    WHERE s.timestamp IS NOT NULL AND s.timestamp < (timestamp() - $ttl_ms)
    WITH s LIMIT $batch_size
    DETACH DELETE s
    RETURN count(s) AS deleted
    """
    result = tx.run(query, ttl_ms=ttl_ms, batch_size=batch_size)
    record = result.single()
    return int(record["deleted"]) if record else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune historical Span nodes.")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j bolt URI.")
    parser.add_argument("--user", default="neo4j", help="Neo4j username.")
    parser.add_argument("--password", default="testpassword", help="Neo4j password.")
    parser.add_argument(
        "--ttl-days",
        type=int,
        default=30,
        help="Delete spans older than this many days.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Spans to delete per transaction batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        deleted = prune_history(
            uri=args.uri,
            user=args.user,
            password=args.password,
            ttl_days=args.ttl_days,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        sys.exit(f"Prune failed: {exc}")
    print(f"Completed history pruning. Deleted {deleted} spans.")


if __name__ == "__main__":
    main()
