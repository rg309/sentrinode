from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from neo4j import GraphDatabase, Session

from .config import Neo4jConfig

logger = logging.getLogger(__name__)


@dataclass
class GraphMetrics:
    path_depth: int = 0
    critical_path: List[str] = field(default_factory=list)
    fan_out_per_service: Dict[str, int] = field(default_factory=dict)


LONGEST_PATH_QUERY = """
MATCH path = (entry:Service {name: $entry_service})-[:CALLS*..15]->(db:Database)
WITH path
ORDER BY length(path) DESC
LIMIT 1
RETURN length(path) AS depth, [node IN nodes(path) | node.name] AS nodes
"""

FAN_OUT_QUERY = """
MATCH (svc:Service)
WHERE svc.name IN $services
RETURN svc.name AS service, size((svc)-[:CALLS]->()) AS fan_out
"""


class GraphFeatureExtractor:
    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        self.driver = GraphDatabase.driver(
            config.uri, auth=(config.user, config.password)
        )

    def close(self) -> None:
        self.driver.close()

    def _session(self) -> Session:
        if self.config.database:
            return self.driver.session(database=self.config.database)
        return self.driver.session()

    def extract_metrics(
        self, entry_service: str, high_risk_services: Iterable[str]
    ) -> GraphMetrics:
        metrics = GraphMetrics()

        try:
            with self._session() as session:
                path_result = session.run(
                    LONGEST_PATH_QUERY, entry_service=entry_service
                ).single()
                if path_result:
                    metrics.path_depth = path_result.get("depth", 0) or 0
                    metrics.critical_path = path_result.get("nodes", []) or []
                else:
                    logger.warning(
                        "No graph path found from entry service '%s' to a database.",
                        entry_service,
                    )

                fan_out_result = session.run(
                    FAN_OUT_QUERY, services=list(high_risk_services)
                )
                metrics.fan_out_per_service = {
                    record["service"]: record["fan_out"]
                    for record in fan_out_result
                    if record and record.get("service")
                }

        except Exception as exc:
            logger.exception("Neo4j feature extraction failed: %s", exc)

        return metrics
