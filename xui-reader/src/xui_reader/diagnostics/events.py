"""Structured debug event schema helpers with compatibility guards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from xui_reader.diagnostics.artifacts import redact_value
from xui_reader.errors import DiagnosticsError

DEBUG_EVENT_SCHEMA_VERSION = "v1"
DEBUG_EVENT_COMPATIBILITY_NOTES = (
    "Top-level fields are append-only within a schema major version. "
    "Consumers must ignore unknown top-level fields. "
    "Breaking removals or type changes require a schema major bump."
)

_REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "event_type",
    "occurred_at",
    "run_id",
    "source_id",
    "payload",
)


@dataclass(frozen=True)
class DebugEvent:
    schema_version: str
    event_type: str
    occurred_at: str
    run_id: str
    source_id: str | None
    payload: dict[str, Any]


class JsonlEventLogger:
    """Append schema-validated JSONL debug events."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(
        self,
        event_type: str,
        *,
        run_id: str,
        source_id: str | None = None,
        payload: dict[str, Any] | None = None,
        occurred_at: datetime | None = None,
    ) -> dict[str, Any]:
        event = build_debug_event(
            event_type,
            run_id=run_id,
            source_id=source_id,
            payload=payload,
            occurred_at=occurred_at,
        )
        line = json.dumps(event, sort_keys=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.write("\n")
        return event


def build_debug_event(
    event_type: str,
    *,
    run_id: str,
    source_id: str | None = None,
    payload: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
    schema_version: str = DEBUG_EVENT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build and validate a redacted event payload using the active schema."""
    resolved_payload = payload if payload is not None else {}
    if not isinstance(resolved_payload, dict):
        raise DiagnosticsError("payload must be a dictionary.")
    if not event_type.strip():
        raise DiagnosticsError("event_type must be non-empty.")
    if not run_id.strip():
        raise DiagnosticsError("run_id must be non-empty.")

    resolved_time = occurred_at or datetime.now(timezone.utc)
    if resolved_time.tzinfo is None:
        resolved_time = resolved_time.replace(tzinfo=timezone.utc)
    event = DebugEvent(
        schema_version=schema_version,
        event_type=event_type.strip(),
        occurred_at=resolved_time.isoformat(),
        run_id=run_id.strip(),
        source_id=source_id.strip() if isinstance(source_id, str) and source_id.strip() else None,
        payload=redact_value(resolved_payload),
    )
    serialized = {
        "schema_version": event.schema_version,
        "event_type": event.event_type,
        "occurred_at": event.occurred_at,
        "run_id": event.run_id,
        "source_id": event.source_id,
        "payload": event.payload,
    }
    validate_debug_event(serialized)
    return serialized


def validate_debug_event(event: dict[str, Any]) -> None:
    """Validate required fields and enforce schema compatibility policy."""
    for field in _REQUIRED_TOP_LEVEL_FIELDS:
        if field not in event:
            raise DiagnosticsError(f"Debug event missing required field '{field}'.")

    schema_version = event["schema_version"]
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise DiagnosticsError("schema_version must be a non-empty string.")
    ensure_schema_compatible(schema_version)

    if not isinstance(event["event_type"], str) or not event["event_type"].strip():
        raise DiagnosticsError("event_type must be a non-empty string.")
    if not isinstance(event["occurred_at"], str) or not event["occurred_at"].strip():
        raise DiagnosticsError("occurred_at must be a non-empty ISO timestamp string.")
    if not isinstance(event["run_id"], str) or not event["run_id"].strip():
        raise DiagnosticsError("run_id must be a non-empty string.")
    if event["source_id"] is not None and not isinstance(event["source_id"], str):
        raise DiagnosticsError("source_id must be a string or null.")
    if not isinstance(event["payload"], dict):
        raise DiagnosticsError("payload must be an object.")


def ensure_schema_compatible(schema_version: str) -> None:
    """Accept events only when schema major matches the current major."""
    current_major = _schema_major(DEBUG_EVENT_SCHEMA_VERSION)
    incoming_major = _schema_major(schema_version)
    if incoming_major != current_major:
        raise DiagnosticsError(
            f"Incompatible debug event schema '{schema_version}'. Expected major '{current_major}'."
        )


def _schema_major(version: str) -> str:
    raw = version.strip().lower()
    match = re.match(r"^v?(?P<major>\d+)(?:[._-]\d+)?$", raw)
    if match is None:
        raise DiagnosticsError(f"Invalid schema version '{version}'. Use forms like 'v1' or '1.0'.")
    return match.group("major")
