"""Structured debug event schema and compatibility behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xui_reader.diagnostics.events import (
    DEBUG_EVENT_COMPATIBILITY_NOTES,
    DEBUG_EVENT_SCHEMA_VERSION,
    JsonlEventLogger,
    build_debug_event,
    ensure_schema_compatible,
    validate_debug_event,
)
from xui_reader.errors import DiagnosticsError


def test_build_debug_event_includes_schema_version_and_required_fields() -> None:
    event = build_debug_event(
        "watch_cycle",
        run_id="run-123",
        source_id="list:1",
        payload={"status": "ok", "auth_token": "secret"},
    )
    assert event["schema_version"] == DEBUG_EVENT_SCHEMA_VERSION
    assert event["event_type"] == "watch_cycle"
    assert event["run_id"] == "run-123"
    assert event["source_id"] == "list:1"
    assert "occurred_at" in event
    assert event["payload"]["auth_token"] == "<redacted>"
    assert "append-only" in DEBUG_EVENT_COMPATIBILITY_NOTES


def test_validate_debug_event_rejects_missing_required_fields() -> None:
    with pytest.raises(DiagnosticsError, match="missing required field 'run_id'"):
        validate_debug_event(
            {
                "schema_version": "v1",
                "event_type": "watch_cycle",
                "occurred_at": "2026-03-01T00:00:00+00:00",
                "source_id": None,
                "payload": {},
            }
        )


def test_ensure_schema_compatible_accepts_current_major_forms() -> None:
    ensure_schema_compatible("v1")
    ensure_schema_compatible("1.1")


def test_ensure_schema_compatible_rejects_incompatible_major() -> None:
    with pytest.raises(DiagnosticsError, match="Incompatible debug event schema"):
        ensure_schema_compatible("v2")


def test_jsonl_event_logger_appends_valid_json_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(log_path)
    logger.append(
        "source_result",
        run_id="run-456",
        source_id="user:a",
        payload={"status": "ok", "authorization": "Bearer xyz"},
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event_type"] == "source_result"
    assert parsed["payload"]["authorization"] == "<redacted>"
