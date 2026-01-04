from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Mapping


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp '{value}': {exc}") from exc


def _normalize_services(items: Iterable[str]) -> List[str]:
    return [item for item in (svc.strip() for svc in items) if item]


@dataclass
class FeatureAggregationJob:
    entry_service: str
    high_risk_services: List[str]
    window_minutes: int
    target_horizon_minutes: int
    timestamp: datetime

    @property
    def window_start(self) -> datetime:
        return self.timestamp - timedelta(minutes=self.window_minutes)

    @property
    def window_end(self) -> datetime:
        return self.timestamp

    @property
    def target_timestamp(self) -> datetime:
        return self.timestamp + timedelta(minutes=self.target_horizon_minutes)

    @classmethod
    def from_message(
        cls, message: bytes, defaults: "FeatureAggregationJob"
    ) -> "FeatureAggregationJob":
        payload = json.loads(message.decode("utf-8"))
        return cls(
            entry_service=payload.get("entry_service", defaults.entry_service),
            high_risk_services=_normalize_services(
                payload.get("high_risk_services", defaults.high_risk_services)
            ),
            window_minutes=int(payload.get("window_minutes", defaults.window_minutes)),
            target_horizon_minutes=int(
                payload.get("target_horizon_minutes", defaults.target_horizon_minutes)
            ),
            timestamp=_parse_timestamp(payload.get("timestamp"))
            if payload.get("timestamp")
            else defaults.timestamp,
        )

    @classmethod
    def from_config(cls, service_cfg) -> "FeatureAggregationJob":
        now = datetime.now(timezone.utc)
        return cls(
            entry_service=service_cfg.entry_service,
            high_risk_services=list(service_cfg.high_risk_services),
            window_minutes=service_cfg.window_minutes,
            target_horizon_minutes=service_cfg.target_horizon_minutes,
            timestamp=now,
        )

    def as_metadata(self) -> Mapping[str, str]:
        return {
            "entry_service": self.entry_service,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
        }
