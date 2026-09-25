"""Deterministic inputs for company-exposure tests.

Factories build typed inputs and valid database rows only. They never
return an expected business conclusion and never dispatch on a case ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid5

FIXED_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_NAMESPACE = UUID("6f5a3c2e-9d7b-4e2a-8c1f-0b3d5e7a9c11")


def fixed_uuid(label: str) -> UUID:
    """Stable UUID for a readable label, so failures name their inputs."""

    return uuid5(_NAMESPACE, label)


@dataclass
class FixedClock:
    """Injectable UTC clock; tests advance it explicitly."""

    now_value: datetime = field(default=FIXED_NOW)

    def now(self) -> datetime:
        return self.now_value

    def advance(self, **delta) -> datetime:
        self.now_value = self.now_value + timedelta(**delta)
        return self.now_value

    def advance_to(self, when: datetime) -> datetime:
        if when.tzinfo is None:
            raise ValueError("clock requires timezone-aware datetimes")
        self.now_value = when
        return self.now_value
