"""Unit tests for the Operations job inventory service."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from app.services.operations_job_service import OperationsJobService, _JobRecord
from app.services.runtime_activity_contract import progress_mode


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
    service._job_backend.get_status = MagicMock(return_value=None)

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
    service._job_backend.get_status = MagicMock(return_value=None)

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
    service._job_backend.get_status = MagicMock(return_value=None)

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
    service._job_backend.get_status = MagicMock(return_value=None)

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


def test_market_lease_for_data_fetch_task_keeps_data_fetch_queue_label():
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker({})
    service._inspect = lambda: _FakeInspect()
    service._runtime_activity_records = lambda _db: []
    service._job_backend.get_status = MagicMock(return_value=None)

    lock = MagicMock()
    lock.get_current_task.return_value = None

    with patch("app.services.operations_job_service.get_workload_coordination") as mock_get_coordination, patch(
        "app.services.operations_job_service.get_data_fetch_lock",
        return_value=lock,
    ):
        mock_get_coordination.return_value.get_external_fetch_holder.return_value = None
        mock_get_coordination.return_value.get_market_workload_holders.return_value = {
            "US": {
                "task_id": "fetch-us-1",
                "task_name": "app.tasks.cache_tasks.smart_refresh_cache",
                "started_at": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
                "ttl_seconds": 6000,
            },
            "HK": None,
            "JP": None,
            "TW": None,
        }

        payload = service.list_jobs(MagicMock())

    lease_job = next(job for job in payload["jobs"] if job["task_id"] == "fetch-us-1")
    assert lease_job["queue"] == "data_fetch_us"


def test_list_jobs_surfaces_runtime_activity_progress_fields():
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker({})
    service._inspect = lambda: _FakeInspect()
    service._runtime_activity_records = lambda _db: [
        {
            "market": "US",
            "lifecycle": "bootstrap",
            "stage_key": "prices",
            "stage_label": "Price Refresh",
            "status": "running",
            "progress_mode": "determinate",
            "percent": 42.0,
            "current": 420,
            "total": 1000,
            "message": "Batch 5/12 · refreshing prices",
            "task_name": "app.tasks.cache_tasks.smart_refresh_cache",
            "task_id": "task-us",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    service._job_backend.get_status = MagicMock(return_value=None)

    lock = MagicMock()
    lock.get_current_task.return_value = None

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

    job = next(job for job in payload["jobs"] if job["task_id"] == "task-us")
    assert job["progress_mode"] == "determinate"
    assert job["percent"] == 42.0
    assert job["current"] == 420
    assert job["total"] == 1000
    assert job["message"] == "Batch 5/12 · refreshing prices"


def test_progress_mode_keeps_active_100_percent_indeterminate():
    assert progress_mode("running", 100.0, 3750, 3750) == "indeterminate"
    assert progress_mode("waiting", None, 3750, 3750) == "indeterminate"
    assert progress_mode("stale", 100.0, 3750, 3750) == "indeterminate"
    assert progress_mode("completed", 100.0, 3750, 3750) == "determinate"


def test_list_jobs_falls_back_to_job_backend_progress_for_active_worker_task():
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker({})

    class _InspectWithRunningTask(_FakeInspect):
        def active(self):
            return {
                "datafetch-global@host": [
                    {
                        "id": "task-fetch-us",
                        "name": "app.tasks.cache_tasks.smart_refresh_cache",
                        "kwargs": {"market": "US"},
                        "delivery_info": {"routing_key": "data_fetch_us"},
                        "time_start": datetime.now(timezone.utc).timestamp(),
                    }
                ]
            }

        def active_queues(self):
            return {
                "datafetch-global@host": [
                    {"name": "data_fetch_us"},
                ]
            }

        def stats(self):
            return {"datafetch-global@host": {}}

    service._inspect = lambda: _InspectWithRunningTask()
    service._runtime_activity_records = lambda _db: []
    service._job_backend.get_status = MagicMock(
        return_value=SimpleNamespace(
            status="running",
            current=50,
            total=200,
            percent=25.0,
            message="Batch 1/4 · refreshing prices",
        )
    )

    lock = MagicMock()
    lock.get_current_task.return_value = None

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

    job = next(job for job in payload["jobs"] if job["task_id"] == "task-fetch-us")
    assert job["progress_mode"] == "determinate"
    assert job["percent"] == 25.0
    assert job["current"] == 50
    assert job["total"] == 200
    assert job["message"] == "Batch 1/4 · refreshing prices"


def test_list_jobs_recomputes_runtime_progress_mode_from_counts():
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker({})

    class _InspectWithRunningTask(_FakeInspect):
        def active(self):
            return {
                "datafetch-global@host": [
                    {
                        "id": "task-fetch-us",
                        "name": "app.tasks.cache_tasks.smart_refresh_cache",
                        "kwargs": {"market": "US"},
                        "delivery_info": {"routing_key": "data_fetch_us"},
                        "time_start": datetime.now(timezone.utc).timestamp(),
                    }
                ]
            }

        def active_queues(self):
            return {
                "datafetch-global@host": [
                    {"name": "data_fetch_us"},
                ]
            }

        def stats(self):
            return {"datafetch-global@host": {}}

    service._inspect = lambda: _InspectWithRunningTask()
    service._runtime_activity_records = lambda _db: [
        {
            "market": "US",
            "lifecycle": "bootstrap",
            "stage_key": "prices",
            "stage_label": "Price Refresh",
            "status": "running",
            "progress_mode": "indeterminate",
            "percent": None,
            "current": 50,
            "total": 200,
            "message": "Batch 1/4 · refreshing prices",
            "task_name": "app.tasks.cache_tasks.smart_refresh_cache",
            "task_id": "task-fetch-us",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    service._job_backend.get_status = MagicMock(return_value=None)

    lock = MagicMock()
    lock.get_current_task.return_value = None

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

    job = next(job for job in payload["jobs"] if job["task_id"] == "task-fetch-us")
    assert job["progress_mode"] == "determinate"
    assert job["current"] == 50
    assert job["total"] == 200
    assert job["message"] == "Batch 1/4 · refreshing prices"


def test_list_jobs_reads_backend_message_even_when_runtime_counts_are_determinate():
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker({})

    class _InspectWithRunningTask(_FakeInspect):
        def active(self):
            return {
                "datafetch-global@host": [
                    {
                        "id": "task-fetch-us",
                        "name": "app.tasks.cache_tasks.smart_refresh_cache",
                        "kwargs": {"market": "US"},
                        "delivery_info": {"routing_key": "data_fetch_us"},
                        "time_start": datetime.now(timezone.utc).timestamp(),
                    }
                ]
            }

        def active_queues(self):
            return {
                "datafetch-global@host": [
                    {"name": "data_fetch_us"},
                ]
            }

        def stats(self):
            return {"datafetch-global@host": {}}

    service._inspect = lambda: _InspectWithRunningTask()
    service._runtime_activity_records = lambda _db: [
        {
            "market": "US",
            "lifecycle": "bootstrap",
            "stage_key": "prices",
            "stage_label": "Price Refresh",
            "status": "running",
            "progress_mode": "determinate",
            "percent": None,
            "current": 50,
            "total": 200,
            "message": None,
            "task_name": "app.tasks.cache_tasks.smart_refresh_cache",
            "task_id": "task-fetch-us",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    service._job_backend.get_status = MagicMock(
        return_value=SimpleNamespace(
            status="running",
            current=50,
            total=200,
            percent=25.0,
            message="Batch 1/4 · refreshing prices",
        )
    )

    lock = MagicMock()
    lock.get_current_task.return_value = None

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

    job = next(job for job in payload["jobs"] if job["task_id"] == "task-fetch-us")
    assert job["progress_mode"] == "determinate"
    assert job["message"] == "Batch 1/4 · refreshing prices"


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
    service._find_job_record = lambda _db, _task_id: _JobRecord(
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
    service._find_job_record = lambda _db, _task_id: _JobRecord(
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


def test_cancel_scan_queues_bootstrap_publish_instead_of_building_inline():
    scans = MagicMock()
    scans.get_by_scan_id.return_value = SimpleNamespace(status="running")
    uow = MagicMock()
    uow.__enter__.return_value = uow
    uow.scans = scans

    with (
        patch("app.infra.db.uow.SqlUnitOfWork", return_value=uow),
        patch("app.services.ui_snapshot_service.safe_publish_scan_bootstrap") as mock_publish,
        patch("app.tasks.scan_tasks.publish_scan_bootstrap_snapshots.apply_async") as mock_enqueue,
    ):
        status, _ = OperationsJobService()._cancel_scan(MagicMock(), "scan-001")

    assert status == "accepted"
    scans.update_status.assert_called_once_with("scan-001", "cancelled")
    mock_publish.assert_not_called()
    mock_enqueue.assert_called_once_with(args=["scan-001"], retry=False, connection=ANY)


def test_cancel_job_force_releases_market_lease_for_stale_market_job():
    service = OperationsJobService()
    service._record_cancel_action = lambda *args, **kwargs: None
    service._find_job_record = lambda _db, _task_id: _JobRecord(
        task_id="market-job-1",
        task_name="app.tasks.group_rank_tasks.calculate_daily_group_rankings",
        queue="market_jobs_us",
        market="US",
        state="stale",
        worker=None,
        age_seconds=3600,
        wait_reason=None,
        heartbeat_lag_seconds=None,
        cancel_strategy="force_release_market_lease",
    )

    with patch("app.services.operations_job_service.get_workload_coordination") as mock_get_coordination:
        mock_get_coordination.return_value.release_market_workload.return_value = True

        result = service.cancel_job(MagicMock(), "market-job-1")

    assert result["status"] == "accepted"
    assert result["cancel_strategy"] == "force_release_market_lease"
    mock_get_coordination.return_value.release_market_workload.assert_called_once_with("market-job-1", market="US")


def test_force_cancel_refresh_releases_scoped_lock_and_coordination_leases():
    service = OperationsJobService()
    service._record_cancel_action = lambda *args, **kwargs: None
    service._find_job_record = lambda _db, _task_id: _JobRecord(
        task_id="fetch-us-lock",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        queue="data_fetch_us",
        market="US",
        state="stuck",
        worker=None,
        age_seconds=7200,
        wait_reason=None,
        heartbeat_lag_seconds=7200,
        cancel_strategy="force_cancel_refresh",
    )

    lock = MagicMock()
    lock.get_any_current_task.return_value = {
        "task_id": "fetch-us-lock",
        "task_name": "app.tasks.cache_tasks.smart_refresh_cache",
        "last_heartbeat": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        "lock_key": "data_fetch_job_lock:us",
    }
    price_cache = MagicMock()

    with patch("app.services.operations_job_service.get_data_fetch_lock", return_value=lock), patch(
        "app.services.operations_job_service.get_workload_coordination"
    ) as mock_get_coordination, patch(
        "app.wiring.bootstrap.get_price_cache", return_value=price_cache
    ):
        result = service.cancel_job(MagicMock(), "fetch-us-lock")

    assert result["status"] == "accepted"
    lock.force_release.assert_called_once_with(market="us")
    mock_get_coordination.return_value.release_market_workload.assert_called_once_with("fetch-us-lock", market="us")
    mock_get_coordination.return_value.release_external_fetch.assert_called_once_with("fetch-us-lock")
    price_cache.clear_warmup_heartbeat.assert_called_once_with(market="us")


def test_cancel_job_refuses_when_the_strategy_changed_since_the_operator_saw_it():
    service = OperationsJobService()
    audit = MagicMock()
    service._record_cancel_action = audit
    service._find_job_record = lambda _db, _task_id: _JobRecord(
        task_id="queued-1",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        queue="data_fetch_us",
        market="US",
        state="reserved",
        worker="datafetch@host",
        age_seconds=None,
        wait_reason=None,
        heartbeat_lag_seconds=None,
        cancel_strategy="revoke",
    )

    with patch("app.services.operations_job_service.celery_app.control.revoke") as mock_revoke:
        result = service.cancel_job(
            MagicMock(), "queued-1", expected_strategy="revoke_and_remove_from_queue"
        )

    assert result["status"] == "conflict"
    assert "reserved" in result["message"]
    mock_revoke.assert_not_called()
    audit.assert_not_called()


class _QueryOnlyInspect:
    """Fails the test if the lookup falls back to the full inventory."""

    def __init__(self, replies):
        self.replies = replies
        self.queried = []

    def query_task(self, task_id):
        self.queried.append(task_id)
        return self.replies

    def __getattr__(self, name):
        raise AssertionError(f"cancel lookup must not call inspect.{name}()")


def test_find_job_record_asks_workers_about_one_task_only():
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker({})
    inspect = _QueryOnlyInspect(
        {
            "general@host": {
                "task-1": [
                    "reserved",
                    {
                        "id": "task-1",
                        "name": "app.tasks.breadth_tasks.calculate_daily_breadth",
                        "args": [],
                        "kwargs": {"market": "US"},
                        "delivery_info": {"routing_key": "market_jobs_us"},
                    },
                ]
            },
            "other@host": {},
        }
    )
    service._inspect = lambda: inspect

    with patch("app.services.operations_job_service.get_workload_coordination") as mock_get_coordination:
        mock_get_coordination.return_value.get_external_fetch_holder.return_value = None
        mock_get_coordination.return_value.get_market_workload_holders.return_value = {"US": None}
        record = service._find_job_record(MagicMock(), "task-1")

    assert inspect.queried == ["task-1"]
    assert record.state == "reserved"
    assert record.cancel_strategy == "revoke"
    assert record.queue == "market_jobs_us"


def test_find_job_record_prefers_the_worker_state_over_a_broker_duplicate():
    """A redelivered task can be in the broker list and on a worker at once;
    list_jobs shows the worker state, so the lookup must too (no false 409)."""
    service = OperationsJobService()
    raw = _queued_message(task_id="task-1", task_name="app.tasks.cache_tasks.smart_refresh_cache")
    service._broker = lambda: _FakeBroker({"celery": [raw]})
    service._inspect = lambda: _QueryOnlyInspect(
        {"general@host": {"task-1": ["reserved", {
            "id": "task-1", "name": "app.tasks.cache_tasks.smart_refresh_cache",
            "args": [], "kwargs": {}, "delivery_info": {"routing_key": "celery"},
        }]}}
    )

    with patch.object(service, "broker_queue_names", return_value=["celery"]), patch(
        "app.services.operations_job_service.get_workload_coordination"
    ) as mock_get_coordination:
        mock_get_coordination.return_value.get_external_fetch_holder.return_value = None
        mock_get_coordination.return_value.get_market_workload_holders.return_value = {}
        record = service._find_job_record(MagicMock(), "task-1")

    assert record.state == "reserved"
    assert record.cancel_strategy == "revoke"


def test_list_jobs_marks_rows_with_an_accepted_cancel_request():
    service = OperationsJobService()
    raw = _queued_message(
        task_id="scan-task-1",
        task_name="app.tasks.scan_tasks.run_bulk_scan",
        kwargs={"market": "US"},
    )
    service._broker = lambda: _FakeBroker({"user_scans_us": [raw]})
    service._inspect = lambda: _FakeInspect()
    service._runtime_activity_records = lambda _db: []

    def _action(task_id, outcome, timestamp, strategy="scan_cancel"):
        return SimpleNamespace(value=json.dumps({
            "task_id": task_id, "strategy": strategy, "outcome": outcome, "timestamp": timestamp,
        }))

    db = MagicMock()
    db.query.return_value.filter.return_value.filter.return_value.all.return_value = [
        _action("scan-task-1", "accepted", "2026-10-13T10:00:00+00:00"),
        _action("scan-task-1", "accepted", "2026-10-13T10:05:00+00:00"),
        _action("scan-task-1", "blocked", "2026-10-13T10:09:00+00:00", strategy="revoke"),
        SimpleNamespace(value="not json"),
    ]
    lock = MagicMock()
    lock.get_current_task.return_value = None

    with patch("app.services.operations_job_service.get_workload_coordination") as mock_get_coordination, patch(
        "app.services.operations_job_service.get_data_fetch_lock", return_value=lock
    ):
        mock_get_coordination.return_value.get_external_fetch_holder.return_value = None
        mock_get_coordination.return_value.get_market_workload_holders.return_value = {"US": None}
        payload = service.list_jobs(db)

    job = next(job for job in payload["jobs"] if job["task_id"] == "scan-task-1")
    assert job["cancel_requested_at"] == "2026-10-13T10:05:00+00:00"
    assert job["cancel_requested_strategy"] == "scan_cancel"


def test_force_actions_do_not_mark_rows_cancel_requested():
    db = MagicMock()
    db.query.return_value.filter.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(value=json.dumps({
            "task_id": "lease-1", "strategy": "force_release_market_lease",
            "outcome": "accepted", "timestamp": "2026-10-13T10:00:00+00:00",
        })),
    ]

    assert OperationsJobService()._cancel_requests(db) == {}


_ETA_REQUEST = {
    "id": "eta-1",
    "name": "app.tasks.cache_tasks.smart_refresh_cache",
    "args": [],
    "kwargs": {"market": "US"},
    "delivery_info": {"routing_key": "data_fetch_us"},
}


def test_eta_scheduled_job_lists_and_resolves_as_revoke():
    """An ETA task is held by a worker, not in the broker list or worker
    requests: the listing and the cancel lookup must agree it is revoked."""

    class _EtaInspect(_FakeInspect):
        def scheduled(self):
            return {"datafetch@host": [{"eta": "2026-10-14T00:00:00", "request": _ETA_REQUEST}]}

        def query_task(self, _task_id):
            return {"datafetch@host": {}}

    service = OperationsJobService()
    service._broker = lambda: _FakeBroker({})
    service._inspect = lambda: _EtaInspect()
    service._runtime_activity_records = lambda _db: []
    lock = MagicMock()
    lock.get_current_task.return_value = None

    with patch("app.services.operations_job_service.get_workload_coordination") as mock_get_coordination, patch(
        "app.services.operations_job_service.get_data_fetch_lock", return_value=lock
    ):
        mock_get_coordination.return_value.get_external_fetch_holder.return_value = None
        mock_get_coordination.return_value.get_market_workload_holders.return_value = {"US": None}
        listed = next(job for job in service.list_jobs(MagicMock())["jobs"] if job["task_id"] == "eta-1")
        record = service._find_job_record(MagicMock(), "eta-1")

    assert listed["cancel_strategy"] == "revoke"
    assert record is not None
    assert record.cancel_strategy == "revoke"


def test_scan_cancel_resolves_scan_id_from_task_id_for_lease_only_rows():
    service = OperationsJobService()
    service._record_cancel_action = lambda *args, **kwargs: None
    cancelled = []
    service._cancel_scan = lambda _db, scan_id: cancelled.append(scan_id) or ("accepted", "ok")
    service._find_job_record = lambda _db, _task_id: _JobRecord(
        task_id="scan-task-9",
        task_name="app.tasks.scan_tasks.run_bulk_scan",
        queue="user_scans_us",
        market="US",
        state="running",
        worker=None,
        age_seconds=60,
        wait_reason=None,
        heartbeat_lag_seconds=None,
        cancel_strategy="scan_cancel",
        args=[],
        kwargs={"market": "US"},
        source="market_lease",
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.scalar.return_value = "scan-009"

    result = service.cancel_job(db, "scan-task-9", expected_strategy="scan_cancel")

    assert result["status"] == "accepted"
    assert cancelled == ["scan-009"]


RELEASE = "app.tasks.daily_market_pipeline_tasks.release_daily_market_pipeline_claim"


def _stage_message(task_id, task_name, *, kwargs=None, errbacks=None, chain=None) -> bytes:
    body = base64.b64encode(json.dumps(
        [[], kwargs or {}, {"callbacks": None, "errbacks": errbacks, "chain": chain, "chord": None}]
    ).encode()).decode()
    return json.dumps({
        "body": body,
        "headers": {"id": task_id, "task": task_name},
        "properties": {"correlation_id": task_id, "body_encoding": "base64"},
    }).encode()


def _release_sig(market="US", token="tok-1"):
    return {"task": RELEASE, "args": [], "kwargs": {"market": market, "token": token}, "immutable": True}


def _remove(raw, queue="data_fetch_us", task_id="stage-1"):
    service = OperationsJobService()
    broker = _FakeBroker({queue: [raw]})
    service._broker = lambda: broker
    with patch.object(service, "broker_queue_names", return_value=[queue]), patch(
        "app.services.operations_job_service.celery_app.control.revoke"
    ), patch("app.services.operations_job_service.get_workload_coordination") as coordination:
        removed, _ = service._remove_queued_task(task_id=task_id)
    return removed, coordination.return_value


def test_removing_a_queued_pipeline_stage_releases_its_chain_claim():
    # Revoking a queued stage fires no errback, so without this the market's
    # pipeline reads Skipped until the claim's 20h expiry (#602).
    raw = _stage_message(
        "stage-1", "app.tasks.cache_tasks.smart_refresh_cache",
        kwargs={"market": "US"}, errbacks=[_release_sig()],
    )

    removed, coordination = _remove(raw)

    assert removed is True
    coordination.release_pipeline_chain.assert_called_once_with("tok-1", market="US")


def test_chain_claim_is_read_from_a_real_celery_stage_message():
    """Guards the embed format against Celery itself, not a hand-built body:
    the first stage of a chain with link_error carries the release in errbacks."""
    from celery import chain
    from kombu.serialization import dumps

    from app.celery_app import celery_app
    from app.tasks.daily_market_pipeline_tasks import release_daily_market_pipeline_claim

    release = release_daily_market_pipeline_claim.si(market="US", token="a" * 32)
    pipeline = chain(
        celery_app.signature("app.tasks.cache_tasks.smart_refresh_cache", kwargs={"market": "US"}),
        celery_app.signature("app.tasks.group_rank_tasks.calculate_daily_group_rankings", kwargs={"market": "US"}),
        release,
    )
    steps, _ = pipeline.prepare_steps((), {}, pipeline.tasks, link_error=release)
    first = steps[-1]
    message = celery_app.amqp.as_task_v2(
        "stage-1", first.task, kwargs=first.kwargs,
        errbacks=first.options.get("link_error"), chain=list(reversed(steps[:-1])),
    )
    _, _, body = dumps(message.body, "json")
    raw = json.dumps({
        "body": base64.b64encode(body.encode() if isinstance(body, str) else body).decode(),
        "headers": message.headers,
        "properties": {"body_encoding": "base64"},
    }).encode()

    record = OperationsJobService()._parse_queue_message(raw, "data_fetch_us")

    assert record.chain_claim == ("US", "a" * 32)


def test_removing_the_queued_release_stage_itself_releases_the_claim():
    raw = _stage_message("stage-1", RELEASE, kwargs={"market": "HK", "token": "tok-9"})

    _, coordination = _remove(raw, queue="market_jobs_hk")

    coordination.release_pipeline_chain.assert_called_once_with("tok-9", market="HK")


def test_removing_an_ordinary_queued_task_releases_no_claim():
    raw = _stage_message("stage-1", "app.tasks.cache_tasks.cleanup_orphaned_scans")

    _, coordination = _remove(raw, queue="celery")

    coordination.release_pipeline_chain.assert_not_called()


def test_a_failed_claim_release_does_not_undo_the_removal():
    raw = _stage_message("stage-1", "app.tasks.cache_tasks.smart_refresh_cache", errbacks=[_release_sig()])
    service = OperationsJobService()
    broker = _FakeBroker({"data_fetch_us": [raw]})
    service._broker = lambda: broker
    with patch.object(service, "broker_queue_names", return_value=["data_fetch_us"]), patch(
        "app.services.operations_job_service.celery_app.control.revoke"
    ), patch("app.services.operations_job_service.get_workload_coordination") as coordination:
        coordination.return_value.release_pipeline_chain.side_effect = ConnectionError("redis down")
        removed, _ = service._remove_queued_task(task_id="stage-1")

    assert removed is True
    assert broker.messages_by_queue["data_fetch_us"] == []


def test_a_failed_revoke_after_removal_still_releases_the_claim():
    raw = _stage_message("stage-1", "app.tasks.cache_tasks.smart_refresh_cache", errbacks=[_release_sig()])
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker({"data_fetch_us": [raw]})
    with patch.object(service, "broker_queue_names", return_value=["data_fetch_us"]), patch(
        "app.services.operations_job_service.celery_app.control.revoke", side_effect=ConnectionError("broker")
    ), patch("app.services.operations_job_service.get_workload_coordination") as coordination:
        try:
            service._remove_queued_task(task_id="stage-1")
        except ConnectionError:
            pass

    coordination.return_value.release_pipeline_chain.assert_called_once_with("tok-1", market="US")


def test_operator_release_is_owner_checked_and_audited():
    service = OperationsJobService()
    audit = MagicMock()
    service._record_cancel_action = audit
    with patch("app.services.operations_job_service.get_workload_coordination") as coordination:
        coordination.return_value.release_pipeline_chain.side_effect = [True, False]
        first = service.release_pipeline_chain(MagicMock(), "US", "tok-1")
        second = service.release_pipeline_chain(MagicMock(), "US", "tok-1")

    assert (first["status"], second["status"]) == ("accepted", "blocked")
    coordination.return_value.release_pipeline_chain.assert_called_with("tok-1", market="US")
    # Each release is audited intent-first (#613).
    assert [c.kwargs["outcome"] for c in audit.call_args_list] == [
        "requested", "accepted", "requested", "blocked",
    ]
    assert audit.call_args.kwargs["strategy"] == "release_pipeline_chain"


def test_lease_snapshot_lists_only_held_chain_claims():
    holder = {"task_name": "daily_market_pipeline", "task_id": "tok-1", "started_at": "x", "ttl_seconds": 100}
    with patch("app.services.operations_job_service.get_workload_coordination") as coordination:
        coordination.return_value.get_pipeline_chain_holders.return_value = {"US": holder, "HK": None}
        snapshot = OperationsJobService()._lease_snapshot()

    assert snapshot["pipeline_chain"] == {"US": holder}


class _FakeHashRedis:
    """Bytes in, bytes out, like the binary-mode cache client."""

    def __init__(self):
        self.hashes: dict[bytes, dict[bytes, bytes]] = {}

    @staticmethod
    def _b(value):
        return value if isinstance(value, bytes) else str(value).encode()

    def hset(self, name, mapping):
        self.hashes.setdefault(self._b(name), {}).update(
            {self._b(k): self._b(v) for k, v in mapping.items()}
        )

    def hgetall(self, name):
        return dict(self.hashes.get(self._b(name), {}))

    def hdel(self, name, *keys):
        for key in keys:
            self.hashes.get(self._b(name), {}).pop(self._b(key), None)


_NO_REDIS = object()


def _list_with_workers(messages_by_queue, active_queues, market_holders=None, *, stats=None, memory=None):
    """list_jobs with the given queued messages and worker replies. memory is
    the cache Redis that remembers worker queues (_NO_REDIS: unavailable)."""
    service = OperationsJobService()
    service._broker = lambda: _FakeBroker(messages_by_queue)

    class _Inspect(_FakeInspect):
        def stats(self):
            return stats or {}

        def active_queues(self):
            return active_queues

    service._inspect = lambda: _Inspect()
    service._runtime_activity_records = lambda _db: []
    service._job_backend.get_status = MagicMock(return_value=None)
    lock = MagicMock()
    lock.get_current_task.return_value = None
    redis_client = None if memory is _NO_REDIS else (memory if memory is not None else _FakeHashRedis())
    with patch("app.services.operations_job_service.get_workload_coordination") as coordination, patch(
        "app.services.operations_job_service.get_data_fetch_lock",
        return_value=lock,
    ), patch("app.services.operations_job_service.get_redis_client", return_value=redis_client):
        coordination.return_value.get_external_fetch_holder.return_value = None
        coordination.return_value.get_market_workload_holders.return_value = market_holders or {}
        coordination.return_value.get_pipeline_chain_holders.return_value = {}
        return service.list_jobs(MagicMock())


def _jp_refresh():
    return _queued_message(
        task_id="fetch-jp-1",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        kwargs={"market": "JP"},
    )


def test_queued_job_on_a_queue_no_responding_worker_reads_says_so():
    payload = _list_with_workers(
        {"data_fetch_jp": [_jp_refresh()]},
        {"datafetch-global@h": [{"name": "data_fetch_us"}, {"name": "data_fetch_hk"}]},
    )

    job = payload["jobs"][0]
    assert job["task_id"] == "fetch-jp-1"
    assert job["wait_reason"] == "no_worker_for_queue"
    queues = {queue["queue"]: queue for queue in payload["queues"]}
    assert queues["data_fetch_jp"]["consumers"] == 0
    assert queues["data_fetch_us"]["consumers"] == 1


def test_no_worker_outranks_waiting_for_the_market_lease():
    # The lease frees eventually; with no worker the job still never runs.
    payload = _list_with_workers(
        {"data_fetch_jp": [_jp_refresh()]},
        {"datafetch-global@h": [{"name": "data_fetch_us"}]},
        market_holders={"JP": {"task_id": "other", "task_name": "x"}},
    )

    jobs = {job["task_id"]: job for job in payload["jobs"]}
    assert jobs["fetch-jp-1"]["state"] == "waiting"
    assert jobs["fetch-jp-1"]["wait_reason"] == "no_worker_for_queue"


def test_served_queue_keeps_no_wait_reason():
    payload = _list_with_workers(
        {"data_fetch_jp": [_jp_refresh()]},
        {"datafetch-jp@h": [{"name": "data_fetch_jp"}]},
    )

    assert payload["jobs"][0]["wait_reason"] is None


def test_no_inspect_reply_is_unknown_not_unserved():
    # A timed-out inspect (or a busy solo-pool worker) answers nothing; that
    # must not read as "no worker".
    payload = _list_with_workers({"data_fetch_jp": [_jp_refresh()]}, {})

    assert payload["jobs"][0]["wait_reason"] is None
    queues = {queue["queue"]: queue for queue in payload["queues"]}
    assert queues["data_fetch_jp"]["consumers"] is None


def test_a_busy_silent_worker_still_counts_by_its_remembered_queues():
    # Local solo pool: one datafetch worker reads every data_fetch queue and
    # answers nothing while it fetches; the other workers do answer.
    memory = _FakeHashRedis()
    _list_with_workers(
        {},
        {
            "datafetch-global@h": [{"name": "data_fetch_us"}, {"name": "data_fetch_jp"}],
            "general@h": [{"name": "celery"}],
        },
        memory=memory,
    )

    payload = _list_with_workers(
        {"data_fetch_jp": [_jp_refresh()]},
        {"general@h": [{"name": "celery"}]},
        memory=memory,
    )

    assert payload["jobs"][0]["wait_reason"] is None
    queues = {queue["queue"]: queue for queue in payload["queues"]}
    assert queues["data_fetch_jp"]["consumers"] == 1


def test_a_long_silent_worker_is_forgotten():
    memory = _FakeHashRedis()
    stale = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    memory.hset(
        "operations:worker_queues",
        {"datafetch-global@h": json.dumps({"queues": ["data_fetch_jp"], "seen_at": stale})},
    )

    payload = _list_with_workers(
        {"data_fetch_jp": [_jp_refresh()]},
        {"general@h": [{"name": "celery"}]},
        memory=memory,
    )

    assert payload["jobs"][0]["wait_reason"] == "no_worker_for_queue"
    # Pruned, not just skipped: Docker worker names change on every restart.
    assert b"datafetch-global@h" not in memory.hgetall("operations:worker_queues")
    assert b"general@h" in memory.hgetall("operations:worker_queues")


def test_a_worker_whose_queues_did_not_answer_makes_counts_unknown():
    # It answered stats but missed the active_queues broadcast, and nothing
    # remembers its queues: it may read data_fetch_jp.
    payload = _list_with_workers(
        {"data_fetch_jp": [_jp_refresh()]},
        {"general@h": [{"name": "celery"}]},
        stats={"general@h": {}, "datafetch-global@h": {}},
    )

    assert payload["jobs"][0]["wait_reason"] is None
    queues = {queue["queue"]: queue for queue in payload["queues"]}
    assert queues["data_fetch_jp"]["consumers"] is None


def test_without_the_cache_redis_counts_are_unknown():
    # No memory means a busy silent worker cannot be ruled out.
    payload = _list_with_workers(
        {"data_fetch_jp": [_jp_refresh()]},
        {"general@h": [{"name": "celery"}]},
        memory=_NO_REDIS,
    )

    assert payload["jobs"][0]["wait_reason"] is None


def test_a_corrupt_remembered_entry_is_ignored():
    memory = _FakeHashRedis()
    memory.hset("operations:worker_queues", {"datafetch-global@h": "not json"})

    payload = _list_with_workers(
        {"data_fetch_jp": [_jp_refresh()]},
        {"general@h": [{"name": "celery"}]},
        memory=memory,
    )

    assert payload["jobs"][0]["wait_reason"] == "no_worker_for_queue"
