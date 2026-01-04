from __future__ import annotations

import logging
from typing import Callable

from kafka import KafkaConsumer

from .config import KafkaConfig, ServiceConfig
from .jobs import FeatureAggregationJob

logger = logging.getLogger(__name__)


class KafkaTrigger:
    def __init__(
        self,
        kafka_config: KafkaConfig,
        service_config: ServiceConfig,
        handler: Callable[[FeatureAggregationJob], None],
    ) -> None:
        self.kafka_config = kafka_config
        self.service_config = service_config
        self.handler = handler

    def start(self) -> None:
        logger.info(
            "Starting Kafka trigger on topic=%s bootstrap=%s",
            self.kafka_config.topic,
            self.kafka_config.bootstrap_servers,
        )
        consumer = KafkaConsumer(
            self.kafka_config.topic,
            bootstrap_servers=[
                server.strip()
                for server in self.kafka_config.bootstrap_servers.split(",")
                if server.strip()
            ],
            client_id=f"{self.kafka_config.group_id}-client",
            group_id=self.kafka_config.group_id,
            enable_auto_commit=True,
            value_deserializer=lambda v: v,
            auto_offset_reset=self.kafka_config.auto_offset_reset,
        )
        try:
            for message in consumer:
                logger.debug(
                    "Kafka trigger received message offset=%s", message.offset
                )
                defaults = FeatureAggregationJob.from_config(self.service_config)
                try:
                    job = FeatureAggregationJob.from_message(message.value, defaults)
                except Exception as exc:
                    logger.exception("Invalid feature job payload: %s", exc)
                    continue
                self.handler(job)
        except KeyboardInterrupt:
            logger.info("Kafka trigger interrupted, stopping consumer...")
        finally:
            consumer.close()
