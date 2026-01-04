from __future__ import annotations

import argparse
import logging

from feature_aggregator.config import load_config
from feature_aggregator.service import FeatureAggregatorService
from feature_aggregator.triggers import KafkaTrigger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature Aggregator Service for graph and time-series features."
    )
    parser.add_argument(
        "--mode",
        choices=["once", "schedule", "kafka"],
        help="Override FEATURE_AGGREGATOR_MODE environment variable.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        help="Override SCHEDULE_INTERVAL_SECONDS for schedule mode.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (INFO, DEBUG, ...).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config = load_config()
    if args.mode:
        config.scheduler.mode = args.mode
    if args.interval_seconds:
        config.scheduler.interval_seconds = args.interval_seconds

    service = FeatureAggregatorService(config)

    try:
        if config.scheduler.mode == "schedule":
            service.run_schedule()
        elif config.scheduler.mode == "kafka":
            if not config.kafka.enabled:
                logging.warning(
                    "Kafka trigger requested but KAFKA_TRIGGER_ENABLED is not true."
                )
            trigger = KafkaTrigger(config.kafka, config.service, service.run_job)
            trigger.start()
        else:
            service.run_once()
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
