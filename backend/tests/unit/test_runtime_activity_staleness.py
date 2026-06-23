from __future__ import annotations

from datetime import datetime, timezone


def _record(**overrides):
    from app.services.runtime_activity_contract import RuntimeActivityRecord

    values = {
        "market": "US",
        "lifecycle": "daily_refresh",
        "stage_key": "prices",
        "status": "running",
        "task_name": "app.tasks.cache_tasks.smart_refresh_cache",
        "task_id": "task-old",
        "message": "Refreshing market prices",
        "updated_at": "2026-06-23T05:39:46+00:00",
    }
    values.update(overrides)
    return RuntimeActivityRecord.create(**values)


def test_parse_activity_timestamp_treats_naive_strings_as_utc():
    from app.services.runtime_activity_staleness import parse_activity_timestamp

    parsed = parse_activity_timestamp("2026-06-23T05:39:46")

    assert parsed == datetime(2026, 6, 23, 5, 39, 46, tzinfo=timezone.utc)
    assert (
        parsed.tzinfo == timezone.utc
        or parsed.utcoffset() == timezone.utc.utcoffset(parsed)
    )


def test_running_activity_is_stale_after_threshold():
    from app.services.runtime_activity_staleness import is_stale_running_activity

    record = _record(updated_at="2026-06-23T05:39:46+00:00")
    now = datetime(2026, 6, 23, 6, 10, 0, tzinfo=timezone.utc)

    assert is_stale_running_activity(record, now=now) is True


def test_running_activity_is_not_stale_before_threshold():
    from app.services.runtime_activity_staleness import is_stale_running_activity

    record = _record(updated_at="2026-06-23T05:39:46+00:00")
    now = datetime(2026, 6, 23, 5, 50, 0, tzinfo=timezone.utc)

    assert is_stale_running_activity(record, now=now) is False


def test_non_running_activity_is_not_stale():
    from app.services.runtime_activity_staleness import is_stale_running_activity

    record = _record(status="completed", updated_at="2026-06-23T05:00:00+00:00")
    now = datetime(2026, 6, 23, 7, 0, 0, tzinfo=timezone.utc)

    assert is_stale_running_activity(record, now=now) is False


def test_stale_payload_marks_running_record_stale_without_mutating_original():
    from app.services.runtime_activity_staleness import stale_runtime_activity_payload

    record = _record(updated_at="2026-06-23T05:39:46+00:00")
    payload = stale_runtime_activity_payload(record, reason="No live worker owns this task.")

    assert payload["status"] == "stale"
    assert payload["message"] == "Refreshing market prices - stale: No live worker owns this task."
    assert payload["task_id"] == "task-old"
    assert record.status == "running"
