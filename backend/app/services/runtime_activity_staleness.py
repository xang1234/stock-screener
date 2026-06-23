"""Pure helpers for identifying stale runtime activity records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .runtime_activity_contract import RuntimeActivityRecord, progress_mode

RUNNING_ACTIVITY_STALE_AFTER_SECONDS = 30 * 60


def parse_activity_timestamp(value: Any) -> datetime | None:
    """Parse a runtime activity timestamp as a timezone-aware UTC datetime.

    Naive strings are treated as UTC for legacy/backwards compatibility with
    the runtime activity contract.
    """
    if not isinstance(value, str):
        return None

    raw_value = value.strip()
    if not raw_value:
        return None

    try:
        parsed = datetime.fromisoformat(raw_value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def activity_age_seconds(
    record: RuntimeActivityRecord,
    now: datetime | None = None,
) -> float | None:
    """Return the non-negative age in seconds for a record update timestamp."""
    updated_at = parse_activity_timestamp(record.updated_at)
    if updated_at is None:
        return None

    resolved_now = now or datetime.now(timezone.utc)
    if resolved_now.tzinfo is None:
        resolved_now = resolved_now.replace(tzinfo=timezone.utc)
    else:
        resolved_now = resolved_now.astimezone(timezone.utc)

    return max((resolved_now - updated_at).total_seconds(), 0.0)


def is_stale_running_activity(
    record: RuntimeActivityRecord,
    now: datetime | None = None,
    stale_after_seconds: int = RUNNING_ACTIVITY_STALE_AFTER_SECONDS,
) -> bool:
    """Return True when a running activity has exceeded the stale threshold."""
    if record.status != "running":
        return False

    age_seconds = activity_age_seconds(record, now=now)
    return age_seconds is not None and age_seconds >= stale_after_seconds


def stale_runtime_activity_payload(
    record: RuntimeActivityRecord,
    reason: str,
) -> dict[str, Any]:
    """Build a stale presentation payload without mutating the source record."""
    payload = dict(record.to_payload())
    payload["status"] = "stale"
    payload["message"] = f"{record.message or 'Runtime activity'} - stale: {reason}"
    payload["progress_mode"] = progress_mode(
        payload["status"],
        payload.get("percent"),
        payload.get("current"),
        payload.get("total"),
    )
    return payload
