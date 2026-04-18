"""Unit tests for the Operations job inventory service."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import MagicMock, patch

from app.services.operations_job_service import OperationsJobService, _JobRecord


def _queued_message(*, task_id: str, task_name: str, args: list | None = None, kwargs: dict | None = None) -> bytes:
    body = base64.b64encode(
        json.dumps([args or [], kwargs or {}, {"callbacks": None, "errbacks": None, "chain": None, "chord": None}]).encode()
    ).decode()
    payload = {
        "body": body,
        "headers": {
            "id": task_id,
            "task": task_name,
        },
        "properties": {
            "correlation_id": task_id,
            "body_encoding": "base64",
        },
    }
    return json.dumps(payload).encode()


class _FakeBroker:
    def __init__(self, messages_by_queue: dict[str, list[bytes]]) -> None:
        self.messages_by_queue = {
            queue: list(messages)
            for queue, messages in messages_by_queue.items()
        }

    def lrange(self, queue_name: str, _start: int, _end: int) -> list[bytes]:
        return list(self.messages_by_queue.get(queue_name, []))

    def lrem(self, queue_name: str, _count: int, raw: bytes) -> int:
        queue = self.messages_by_queue.get(queue_name, [])
        try:
            queue.remove(raw)
        except ValueError:
            return 0
        return 1


class _FakeInspect:
    def stats(self):
        return {}

    def active(self):
        return {}

    def reserved(self):
        return {}

    def scheduled(self):
        return {}

    def active_queues(self):
        return {}


def _lease_snapshot(*, external_holder=None, market_holders=None):
    return {
        "external_fetch_global": external_holder,
        "market_workload": market_holders or {"US": None, "HK": None, "JP": None, "TW": None},
    }


def test_list_jobs_marks_data_fetch_queue_as_waiting_for_global_external_lease():
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker(
        {
            "data_fetch_hk": [
                _queued_message(
                    task_id="fetch-hk-1",
                    task_name="app.tasks.cache_tasks.smart_refresh_cache",
                    kwargs={"market": "HK"},
                )
            ]
        }
    )
    service._inspect = lambda: _FakeInspect()
    service._runtime_activity_records = lambda _db: []

    lock = MagicMock()
    lock.get_current_task.return_value = None

    with patch("app.services.operations_job_service.get_workload_coordination") as mock_get_coordination, patch(
        "app.services.operations_job_service.get_data_fetch_lock",
        return_value=lock,
    ):
        mock_get_coordination.return_value.get_external_fetch_holder.return_value = {
            "task_id": "fetch-us-1",
            "task_name": "app.tasks.cache_tasks.smart_refresh_cache",
        }
        mock_get_coordination.return_value.get_market_workload_holders.return_value = {
            "US": None,
            "HK": None,
            "JP": None,
            "TW": None,
        }

        payload = service.list_jobs(MagicMock())

    assert payload["jobs"][0]["task_id"] == "fetch-hk-1"
    assert payload["jobs"][0]["state"] == "waiting"
    assert payload["jobs"][0]["wait_reason"] == "waiting_for_external_fetch_global"
    assert any(queue["queue"] == "data_fetch_hk" and queue["depth"] == 1 for queue in payload["queues"])


def test_list_jobs_surfaces_stuck_lock_holder_without_worker_inspect():
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker({})
    service._inspect = lambda: _FakeInspect()
    service._runtime_activity_records = lambda _db: []

    stale_heartbeat = (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat()
    started_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()

    lock = MagicMock()
    lock.get_current_task.side_effect = lambda market=None: {
        "task_id": "fetch-us-lock",
        "task_name": "app.tasks.cache_tasks.smart_refresh_cache",
        "started_at": started_at,
        "last_heartbeat": stale_heartbeat,
        "lock_key": "data_fetch_job_lock:us",
    } if market == "US" else None

    with patch("app.services.operations_job_service.get_workload_coordination") as mock_get_coordination, patch(
        "app.services.operations_job_service.get_data_fetch_lock",
        return_value=lock,
    ):
        mock_get_coordination.return_value.get_external_fetch_holder.return_value = None
        mock_get_coordination.return_value.get_market_workload_holders.return_value = {
            "US": None,
            "HK": None,
            "JP": None,
            "TW": None,
        }

        payload = service.list_jobs(MagicMock())

    stuck_job = next(job for job in payload["jobs"] if job["task_id"] == "fetch-us-lock")
    assert stuck_job["state"] == "stuck"
    assert stuck_job["cancel_strategy"] == "force_cancel_refresh"
    assert stuck_job["queue"] == "data_fetch_us"


def test_list_jobs_marks_orphaned_market_lease_as_stale():
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker({})
    service._inspect = lambda: _FakeInspect()
    service._runtime_activity_records = lambda _db: []

    lock = MagicMock()
    lock.get_current_task.return_value = None

    with patch("app.services.operations_job_service.get_workload_coordination") as mock_get_coordination, patch(
        "app.services.operations_job_service.get_data_fetch_lock",
        return_value=lock,
    ):
        mock_get_coordination.return_value.get_external_fetch_holder.return_value = None
        mock_get_coordination.return_value.get_market_workload_holders.return_value = {
            "US": {
                "task_id": "market-job-1",
                "task_name": "app.tasks.group_rank_tasks.calculate_daily_group_rankings",
                "started_at": (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat(),
                "ttl_seconds": 5400,
            },
            "HK": None,
            "JP": None,
            "TW": None,
        }

        payload = service.list_jobs(MagicMock())

    lease_job = next(job for job in payload["jobs"] if job["task_id"] == "market-job-1")
    assert lease_job["state"] == "stale"
    assert lease_job["queue"] == "market_jobs_us"


def test_list_jobs_marks_near_expiry_market_lease_as_stuck():
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker({})
    service._inspect = lambda: _FakeInspect()
    service._runtime_activity_records = lambda _db: []

    lock = MagicMock()
    lock.get_current_task.return_value = None

    with patch("app.services.operations_job_service.get_workload_coordination") as mock_get_coordination, patch(
        "app.services.operations_job_service.get_data_fetch_lock",
        return_value=lock,
    ):
        mock_get_coordination.return_value.get_external_fetch_holder.return_value = None
        mock_get_coordination.return_value.get_market_workload_holders.return_value = {
            "US": None,
            "HK": {
                "task_id": "scan-hk-1",
                "task_name": "app.tasks.scan_tasks.run_bulk_scan",
                "started_at": (datetime.now(timezone.utc) - timedelta(minutes=50)).isoformat(),
                "ttl_seconds": 120,
            },
            "JP": None,
            "TW": None,
        }

        payload = service.list_jobs(MagicMock())

    lease_job = next(job for job in payload["jobs"] if job["task_id"] == "scan-hk-1")
    assert lease_job["state"] == "stuck"
    assert lease_job["queue"] == "user_scans_hk"


def test_cancel_job_removes_queued_task_and_revokes():
    service = OperationsJobService()
    raw = _queued_message(
        task_id="queued-1",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        kwargs={"market": "US"},
    )
    broker = _FakeBroker({"data_fetch_us": [raw]})
    service._broker = lambda: broker
    service._record_cancel_action = lambda *args, **kwargs: None
    service._find_scan_record = lambda _db, _task_id: _JobRecord(
        task_id="queued-1",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        queue="data_fetch_us",
        market="US",
        state="queued",
        worker=None,
        age_seconds=None,
        wait_reason=None,
        heartbeat_lag_seconds=None,
        cancel_strategy="revoke_and_remove_from_queue",
    )

    with patch.object(service, "broker_queue_names", return_value=["data_fetch_us"]), patch(
        "app.services.operations_job_service.celery_app.control.revoke"
    ) as mock_revoke:
        result = service.cancel_job(MagicMock(), "queued-1")

    assert result["status"] == "accepted"
    assert broker.messages_by_queue["data_fetch_us"] == []
    mock_revoke.assert_called_once_with("queued-1", terminate=False)


def test_cancel_job_uses_scan_cancel_strategy_for_running_scan():
    service = OperationsJobService()
    service._record_cancel_action = lambda *args, **kwargs: None
    service._cancel_scan = lambda _db, scan_id: ("accepted", f"cancelled:{scan_id}")
    service._find_scan_record = lambda _db, _task_id: _JobRecord(
        task_id="scan-task-1",
        task_name="app.tasks.scan_tasks.run_bulk_scan",
        queue="user_scans_hk",
        market="HK",
        state="running",
        worker="userscans-hk@host",
        age_seconds=10.0,
        wait_reason=None,
        heartbeat_lag_seconds=None,
        cancel_strategy="scan_cancel",
        args=["scan-001", ["0700.HK"], {"min_price": 10}],
        kwargs={"market": "HK"},
    )

    result = service.cancel_job(MagicMock(), "scan-task-1")

    assert result["status"] == "accepted"
    assert result["cancel_strategy"] == "scan_cancel"
    assert result["message"] == "cancelled:scan-001"
