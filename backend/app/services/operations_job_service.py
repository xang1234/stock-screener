"""Inventory and control helpers for the Operations job console."""

from __future__ import annotations

import ast
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from typing import Any, Iterable, Optional

try:
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    redis = None

from sqlalchemy.orm import Session

from app.celery_app import _MARKET_JOB_TASKS, celery_app
from app.config import settings
from app.models.app_settings import AppSetting
from app.services.job_backend import CeleryJobBackend
from app.services.market_activity_service import (
    MARKET_ACTIVITY_KEY_PREFIX,
    RUNTIME_ACTIVITY_CATEGORY,
)
from app.services.ui_snapshot_service import safe_publish_scan_bootstrap
from app.tasks.market_queues import (
    SHARED_DATA_FETCH_QUEUE,
    SHARED_USER_SCANS_QUEUE,
    SUPPORTED_MARKETS,
    all_data_fetch_queues,
    all_market_job_queues,
    all_user_scans_queues,
    data_fetch_queue_for_market,
    market_jobs_queue_for_market,
    normalize_market,
)
from app.wiring.bootstrap import get_data_fetch_lock, get_workload_coordination

logger = logging.getLogger(__name__)

OPERATIONS_JOB_ACTION_CATEGORY = "operations_job_actions"

RUNNING_STALE_AFTER_SECONDS = 30 * 60
QUEUED_STALE_AFTER_SECONDS = 20 * 60
HEARTBEAT_STUCK_AFTER_SECONDS = 30 * 60
LEASE_STUCK_TTL_SECONDS = 10 * 60

SCAN_TASK_NAME = "app.tasks.scan_tasks.run_bulk_scan"


@dataclass
class _JobRecord:
    task_id: str
    task_name: str
    queue: str | None
    market: str | None
    state: str
    worker: str | None
    age_seconds: float | None
    wait_reason: str | None
    heartbeat_lag_seconds: float | None
    cancel_strategy: str
    progress_mode: str = "indeterminate"
    percent: float | None = None
    current: int | None = None
    total: int | None = None
    message: str | None = None
    args: Any = None
    kwargs: dict[str, Any] | None = None
    source: str | None = None

    def to_api(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "queue": self.queue,
            "market": self.market,
            "state": self.state,
            "worker": self.worker,
            "age_seconds": self.age_seconds,
            "wait_reason": self.wait_reason,
            "heartbeat_lag_seconds": self.heartbeat_lag_seconds,
            "cancel_strategy": self.cancel_strategy,
            "progress_mode": self.progress_mode,
            "percent": self.percent,
            "current": self.current,
            "total": self.total,
            "message": self.message,
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds_from(value: Any) -> float | None:
    parsed = _parse_iso(value)
    if parsed is None:
        return None
    return max((_utcnow() - parsed).total_seconds(), 0.0)


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return value


def _coerce_args(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        parsed = _safe_json_loads(value)
        if isinstance(parsed, tuple):
            return list(parsed)
        if isinstance(parsed, list):
            return parsed
    return []


def _coerce_kwargs(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = _safe_json_loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _queue_market(queue_name: str | None, kwargs: dict[str, Any] | None) -> str | None:
    kwargs = kwargs or {}
    market = kwargs.get("market")
    if isinstance(market, str):
        return normalize_market(market)
    if not queue_name or "_" not in queue_name:
        return None
    suffix = queue_name.rsplit("_", 1)[-1]
    if suffix == "shared":
        return None
    try:
        return normalize_market(suffix)
    except ValueError:
        return None


def _queue_family(queue_name: str | None) -> str:
    if not queue_name:
        return "general"
    if queue_name.startswith("data_fetch_"):
        return "data_fetch"
    if queue_name.startswith("market_jobs_"):
        return "market_jobs"
    if queue_name.startswith("user_scans_"):
        return "user_scans"
    return "general"


def _extract_scan_id(task_name: str, args: list[Any], kwargs: dict[str, Any]) -> str | None:
    if task_name != SCAN_TASK_NAME:
        return None
    scan_id = kwargs.get("scan_id")
    if isinstance(scan_id, str) and scan_id:
        return scan_id
    if args and isinstance(args[0], str):
        return args[0]
    return None


def _cancel_strategy_for(record: _JobRecord) -> str:
    if record.state in {"queued", "waiting"}:
        return "revoke_and_remove_from_queue"
    if record.state == "reserved":
        return "revoke"
    if record.task_name == SCAN_TASK_NAME and record.state == "running":
        return "scan_cancel"
    if (
        record.state in {"stale", "stuck"}
        and _queue_family(record.queue) == "data_fetch"
    ):
        return "force_cancel_refresh"
    if (
        record.state in {"stale", "stuck"}
        and _queue_family(record.queue) in {"market_jobs", "user_scans"}
    ):
        return "force_release_market_lease"
    return "unsupported"


def _progress_mode(percent: float | None, current: int | None, total: int | None) -> str:
    if percent is not None:
        return "determinate"
    if current is not None and total is not None:
        return "determinate"
    return "indeterminate"


def _market_lease_state(holder: dict[str, Any]) -> tuple[str, float | None]:
    age_seconds = _age_seconds_from(holder.get("started_at"))
    ttl_seconds = holder.get("ttl_seconds")
    try:
        ttl_seconds = int(ttl_seconds) if ttl_seconds is not None else None
    except (TypeError, ValueError):
        ttl_seconds = None

    if age_seconds is not None and age_seconds >= RUNNING_STALE_AFTER_SECONDS:
        if ttl_seconds is not None and ttl_seconds <= LEASE_STUCK_TTL_SECONDS:
            return ("stuck", age_seconds)
        return ("stale", age_seconds)
    return ("running", age_seconds)


def _task_wait_reason(
    *,
    task_id: str,
    queue_name: str | None,
    market: str | None,
    lease_snapshot: dict[str, Any],
) -> tuple[str, str | None] | tuple[None, None]:
    queue_family = _queue_family(queue_name)
    market_holders = lease_snapshot["market_workload"]
    external_holder = lease_snapshot["external_fetch_global"]

    if market and queue_family in {"data_fetch", "market_jobs", "user_scans"}:
        holder = market_holders.get(market)
        if holder and holder.get("task_id") != task_id:
            return (
                "waiting",
                f"waiting_for_market_workload:{market}",
            )

    if queue_family == "data_fetch" and external_holder and external_holder.get("task_id") != task_id:
        return ("waiting", "waiting_for_external_fetch_global")

    return (None, None)


class OperationsJobService:
    """Aggregates live job, queue, and worker state for the Operations UI."""

    def __init__(self) -> None:
        self._job_backend = CeleryJobBackend()

    @staticmethod
    def broker_queue_names() -> list[str]:
        return [
            "celery",
            SHARED_DATA_FETCH_QUEUE,
            SHARED_USER_SCANS_QUEUE,
            *all_data_fetch_queues(),
            *all_market_job_queues(),
            *all_user_scans_queues(),
        ]

    def _broker(self):
        if redis is None:
            raise RuntimeError("Redis package is not installed; operations queue inspection unavailable")
        return redis.from_url(settings.celery_broker_url)

    def _lease_snapshot(self) -> dict[str, Any]:
        coordination = get_workload_coordination()
        return {
            "external_fetch_global": coordination.get_external_fetch_holder(),
            "market_workload": coordination.get_market_workload_holders(),
        }

    def _runtime_activity_records(self, db: Session) -> list[dict[str, Any]]:
        rows = (
            db.query(AppSetting)
            .filter(AppSetting.category == RUNTIME_ACTIVITY_CATEGORY)
            .filter(AppSetting.key.like(f"{MARKET_ACTIVITY_KEY_PREFIX}%"))
            .all()
        )
        records: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row.value)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def _parse_queue_message(self, raw: bytes, queue_name: str) -> _JobRecord | None:
        try:
            payload = json.loads(raw.decode())
        except Exception:
            logger.debug("Failed to parse Celery queue message for %s", queue_name, exc_info=True)
            return None

        headers = payload.get("headers") or {}
        properties = payload.get("properties") or {}
        task_name = headers.get("task")
        task_id = headers.get("id") or properties.get("correlation_id")
        if not task_name or not task_id:
            return None

        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        body = payload.get("body")
        if body:
            try:
                body_encoding = properties.get("body_encoding") or "base64"
                decoded = (
                    base64.b64decode(body)
                    if body_encoding == "base64"
                    else body.encode() if isinstance(body, str) else body
                )
                body_payload = json.loads(decoded)
                if isinstance(body_payload, list) and len(body_payload) >= 2:
                    args = _coerce_args(body_payload[0])
                    kwargs = _coerce_kwargs(body_payload[1])
            except Exception:
                logger.debug("Failed to decode Celery body for %s", task_name, exc_info=True)

        market = _queue_market(queue_name, kwargs)
        age_seconds = _age_seconds_from(headers.get("sent_at") or headers.get("timestamp"))
        return _JobRecord(
            task_id=task_id,
            task_name=task_name,
            queue=queue_name,
            market=market,
            state="queued",
            worker=None,
            age_seconds=age_seconds,
            wait_reason=None,
            heartbeat_lag_seconds=None,
            cancel_strategy="revoke_and_remove_from_queue",
            args=args,
            kwargs=kwargs,
            source="broker",
        )

    def _queued_jobs(self, lease_snapshot: dict[str, Any]) -> list[_JobRecord]:
        broker = self._broker()
        records: list[_JobRecord] = []
        for queue_name in self.broker_queue_names():
            for raw in broker.lrange(queue_name, 0, -1):
                record = self._parse_queue_message(raw, queue_name)
                if record is None:
                    continue
                state, wait_reason = _task_wait_reason(
                    task_id=record.task_id,
                    queue_name=record.queue,
                    market=record.market,
                    lease_snapshot=lease_snapshot,
                )
                if state:
                    record.state = state
                    record.wait_reason = wait_reason
                records.append(record)
        return records

    def _inspect(self) -> Any:
        return celery_app.control.inspect(timeout=1.0)

    def _worker_queue_map(self, active_queues: dict[str, Any]) -> dict[str, list[str]]:
        mapped: dict[str, list[str]] = {}
        for worker, entries in (active_queues or {}).items():
            queues = []
            for entry in entries or []:
                queue_name = entry.get("name")
                if queue_name:
                    queues.append(queue_name)
            mapped[worker] = queues
        return mapped

    def _inspect_record(
        self,
        request: dict[str, Any],
        *,
        state: str,
        worker: str,
        queue_name: str | None,
    ) -> _JobRecord | None:
        task_name = request.get("name") or request.get("type")
        task_id = request.get("id")
        if not task_name or not task_id:
            return None

        args = _coerce_args(request.get("args"))
        kwargs = _coerce_kwargs(request.get("kwargs"))
        market = _queue_market(queue_name, kwargs)
        age_seconds = None
        if request.get("time_start") is not None:
            try:
                age_seconds = max(_utcnow().timestamp() - float(request["time_start"]), 0.0)
            except Exception:
                age_seconds = None
        if age_seconds is None:
            age_seconds = _age_seconds_from(request.get("acknowledged") or request.get("timestamp"))

        record = _JobRecord(
            task_id=task_id,
            task_name=task_name,
            queue=queue_name,
            market=market,
            state=state,
            worker=worker,
            age_seconds=age_seconds,
            wait_reason=None,
            heartbeat_lag_seconds=None,
            cancel_strategy="unsupported",
            args=args,
            kwargs=kwargs,
            source="worker_inspect",
        )
        record.cancel_strategy = _cancel_strategy_for(record)
        return record

    def _inspect_jobs(self) -> tuple[list[_JobRecord], list[dict[str, Any]]]:
        inspect = self._inspect()
        stats = inspect.stats() or {}
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        scheduled = inspect.scheduled() or {}
        active_queues = inspect.active_queues() or {}
        queue_map = self._worker_queue_map(active_queues)

        workers: list[dict[str, Any]] = []
        for worker in sorted(set(stats) | set(queue_map) | set(active) | set(reserved) | set(scheduled)):
            workers.append(
                {
                    "worker": worker,
                    "status": "online" if worker in stats or worker in queue_map else "unknown",
                    "queues": queue_map.get(worker, []),
                    "active": len(active.get(worker, []) or []),
                    "reserved": len(reserved.get(worker, []) or []),
                    "scheduled": len(scheduled.get(worker, []) or []),
                }
            )

        jobs: list[_JobRecord] = []
        for worker, entries in active.items():
            worker_queues = queue_map.get(worker, [])
            default_queue = worker_queues[0] if len(worker_queues) == 1 else None
            for entry in entries or []:
                queue_name = (entry.get("delivery_info") or {}).get("routing_key") or default_queue
                record = self._inspect_record(entry, state="running", worker=worker, queue_name=queue_name)
                if record:
                    jobs.append(record)

        for worker, entries in reserved.items():
            worker_queues = queue_map.get(worker, [])
            default_queue = worker_queues[0] if len(worker_queues) == 1 else None
            for entry in entries or []:
                queue_name = (entry.get("delivery_info") or {}).get("routing_key") or default_queue
                record = self._inspect_record(entry, state="reserved", worker=worker, queue_name=queue_name)
                if record:
                    jobs.append(record)

        for worker, entries in scheduled.items():
            worker_queues = queue_map.get(worker, [])
            default_queue = worker_queues[0] if len(worker_queues) == 1 else None
            for entry in entries or []:
                request = entry.get("request") or {}
                queue_name = (request.get("delivery_info") or {}).get("routing_key") or default_queue
                record = self._inspect_record(request, state="queued", worker=worker, queue_name=queue_name)
                if record:
                    record.wait_reason = "scheduled_eta"
                    record.cancel_strategy = "revoke"
                    jobs.append(record)

        return jobs, workers

    def _lock_holder_record(self, market: str) -> _JobRecord | None:
        current_task = get_data_fetch_lock().get_current_task(market=market)
        if not current_task:
            return None
        heartbeat_lag = _age_seconds_from(current_task.get("last_heartbeat"))
        state = "stuck" if heartbeat_lag is not None and heartbeat_lag >= HEARTBEAT_STUCK_AFTER_SECONDS else "running"
        record = _JobRecord(
            task_id=current_task.get("task_id") or "unknown",
            task_name=current_task.get("task_name") or "unknown",
            queue=f"data_fetch_{market.lower()}",
            market=market,
            state=state,
            worker=None,
            age_seconds=_age_seconds_from(current_task.get("started_at")),
            wait_reason=None,
            heartbeat_lag_seconds=heartbeat_lag,
            cancel_strategy="force_cancel_refresh" if state == "stuck" else "unsupported",
            progress_mode=_progress_mode(current_task.get("progress"), current_task.get("current"), current_task.get("total")),
            percent=current_task.get("progress"),
            current=current_task.get("current"),
            total=current_task.get("total"),
            args=[],
            kwargs={"market": market},
            source="lock_holder",
        )
        return record

    def _market_lease_record(
        self,
        market: str,
        holder: dict[str, Any] | None,
        runtime_records: dict[str, dict[str, Any]],
    ) -> _JobRecord | None:
        if not holder:
            return None
        task_id = holder.get("task_id")
        task_name = holder.get("task_name")
        if not task_id or not task_name:
            return None
        runtime_record = runtime_records.get(task_id, {})
        queue_name = runtime_record.get("queue")
        if queue_name is None:
            if task_name == SCAN_TASK_NAME:
                queue_name = f"user_scans_{market.lower()}"
            elif task_name in _MARKET_JOB_TASKS:
                queue_name = market_jobs_queue_for_market(market)
            else:
                queue_name = data_fetch_queue_for_market(market)
        state, age_seconds = _market_lease_state(holder)
        record = _JobRecord(
            task_id=task_id,
            task_name=task_name,
            queue=queue_name,
            market=market,
            state=state,
            worker=None,
            age_seconds=age_seconds,
            wait_reason=None,
            heartbeat_lag_seconds=None,
            cancel_strategy="unsupported",
            progress_mode=_progress_mode(runtime_record.get("percent"), runtime_record.get("current"), runtime_record.get("total")),
            percent=runtime_record.get("percent"),
            current=runtime_record.get("current"),
            total=runtime_record.get("total"),
            args=[],
            kwargs={"market": market},
            message=runtime_record.get("message"),
            source="market_lease",
        )
        record.cancel_strategy = _cancel_strategy_for(record)
        return record

    def _apply_runtime_progress(self, record: _JobRecord, runtime_record: dict[str, Any]) -> None:
        if runtime_record.get("message"):
            record.message = runtime_record.get("message")
        if runtime_record.get("percent") is not None:
            record.percent = runtime_record.get("percent")
        if runtime_record.get("current") is not None:
            record.current = runtime_record.get("current")
        if runtime_record.get("total") is not None:
            record.total = runtime_record.get("total")
        record.progress_mode = _progress_mode(
            record.percent,
            record.current,
            record.total,
        )

    def _apply_job_backend_progress(self, record: _JobRecord) -> None:
        if record.state not in {"running", "reserved", "waiting"}:
            return
        try:
            snapshot = self._job_backend.get_status(record.task_id)
        except Exception:
            logger.debug("Failed to read job-backend progress for %s", record.task_id, exc_info=True)
            return
        if snapshot is None:
            return
        if record.message is None and getattr(snapshot, "message", None):
            record.message = snapshot.message
        if record.percent is None and getattr(snapshot, "percent", None) is not None:
            record.percent = snapshot.percent
        if record.current is None and getattr(snapshot, "current", None) is not None:
            record.current = snapshot.current
        if record.total is None and getattr(snapshot, "total", None) is not None:
            record.total = snapshot.total
        record.progress_mode = _progress_mode(record.percent, record.current, record.total)

    def _apply_heartbeat_progress(self, record: _JobRecord) -> None:
        if _queue_family(record.queue) != "data_fetch" or record.market is None:
            return
        current_task = get_data_fetch_lock().get_current_task(market=record.market)
        if not current_task or current_task.get("task_id") != record.task_id:
            return
        heartbeat_lag = _age_seconds_from(current_task.get("last_heartbeat"))
        if heartbeat_lag is not None:
            record.heartbeat_lag_seconds = heartbeat_lag
        if record.percent is None and current_task.get("progress") is not None:
            record.percent = current_task.get("progress")
        if record.current is None and current_task.get("current") is not None:
            record.current = current_task.get("current")
        if record.total is None and current_task.get("total") is not None:
            record.total = current_task.get("total")
        record.progress_mode = _progress_mode(record.percent, record.current, record.total)

    def _augment_with_runtime_activity(
        self,
        db: Session,
        records: dict[str, _JobRecord],
        lease_snapshot: dict[str, Any],
    ) -> None:
        runtime_records = {
            record.get("task_id"): record
            for record in self._runtime_activity_records(db)
            if isinstance(record, dict) and record.get("task_id")
        }

        for market in SUPPORTED_MARKETS:
            lock_record = self._lock_holder_record(market)
            if lock_record and lock_record.task_id not in records:
                records[lock_record.task_id] = lock_record

        for market, holder in lease_snapshot["market_workload"].items():
            lease_record = self._market_lease_record(market, holder, runtime_records)
            if lease_record and lease_record.task_id not in records:
                records[lease_record.task_id] = lease_record

        for task_id, runtime_record in runtime_records.items():
            if task_id in records:
                record = records[task_id]
                self._apply_runtime_progress(record, runtime_record)
                if record.market is None and runtime_record.get("market"):
                    record.market = runtime_record.get("market")
                if record.wait_reason is None and runtime_record.get("message") and record.state == "waiting":
                    record.wait_reason = runtime_record.get("message")
                continue

            task_name = runtime_record.get("task_name")
            if not task_name:
                continue

            backend_status = self._job_backend.get_status(task_id)
            if backend_status and backend_status.status in {"failed", "cancelled"}:
                state = backend_status.status
            else:
                raw_status = runtime_record.get("status")
                age_seconds = _age_seconds_from(runtime_record.get("updated_at"))
                if raw_status == "running":
                    state = "stale" if age_seconds and age_seconds >= RUNNING_STALE_AFTER_SECONDS else "running"
                elif raw_status == "queued":
                    state = "stale" if age_seconds and age_seconds >= QUEUED_STALE_AFTER_SECONDS else "queued"
                elif raw_status == "failed":
                    state = "failed"
                else:
                    continue

            market = runtime_record.get("market")
            queue_name = None
            if task_name == SCAN_TASK_NAME and market:
                queue_name = f"user_scans_{str(market).lower()}"
            elif market:
                stage_key = runtime_record.get("stage_key")
                if stage_key in {"breadth", "groups", "snapshot"}:
                    queue_name = market_jobs_queue_for_market(market)
                else:
                    queue_name = f"data_fetch_{str(market).lower()}"

            record = _JobRecord(
                task_id=task_id,
                task_name=task_name,
                queue=queue_name,
                market=market,
                state=state,
                worker=None,
                age_seconds=_age_seconds_from(runtime_record.get("updated_at")),
                wait_reason=None,
                heartbeat_lag_seconds=None,
                cancel_strategy="unsupported",
                progress_mode=_progress_mode(runtime_record.get("percent"), runtime_record.get("current"), runtime_record.get("total")),
                percent=runtime_record.get("percent"),
                current=runtime_record.get("current"),
                total=runtime_record.get("total"),
                args=[],
                kwargs={"market": market} if market else {},
                message=runtime_record.get("message"),
                source="runtime_activity",
            )
            if record.state in {"queued", "stale"}:
                _, wait_reason = _task_wait_reason(
                    task_id=record.task_id,
                    queue_name=record.queue,
                    market=record.market,
                    lease_snapshot=lease_snapshot,
                )
                record.wait_reason = wait_reason
            record.cancel_strategy = _cancel_strategy_for(record)
            records[task_id] = record

    def _queue_summaries(self, queued_jobs: Iterable[_JobRecord]) -> list[dict[str, Any]]:
        depths: dict[str, dict[str, Any]] = {
            queue_name: {
                "queue": queue_name,
                "depth": 0,
                "oldest_age_seconds": None,
            }
            for queue_name in self.broker_queue_names()
        }
        for job in queued_jobs:
            if not job.queue:
                continue
            summary = depths.setdefault(
                job.queue,
                {"queue": job.queue, "depth": 0, "oldest_age_seconds": None},
            )
            summary["depth"] += 1
            if job.age_seconds is not None:
                current_oldest = summary["oldest_age_seconds"]
                summary["oldest_age_seconds"] = (
                    job.age_seconds
                    if current_oldest is None
                    else max(current_oldest, job.age_seconds)
                )
        return sorted(depths.values(), key=lambda item: item["queue"])

    def list_jobs(self, db: Session) -> dict[str, Any]:
        lease_snapshot = self._lease_snapshot()
        queued_jobs = self._queued_jobs(lease_snapshot)
        inspect_jobs, workers = self._inspect_jobs()

        records: dict[str, _JobRecord] = {}
        for job in [*queued_jobs, *inspect_jobs]:
            existing = records.get(job.task_id)
            if existing is None:
                job.cancel_strategy = _cancel_strategy_for(job)
                records[job.task_id] = job
                continue
            # Prefer live worker state over queued broker state.
            state_rank = {"running": 4, "reserved": 3, "waiting": 2, "queued": 1}
            if state_rank.get(job.state, 0) >= state_rank.get(existing.state, 0):
                job.cancel_strategy = _cancel_strategy_for(job)
                records[job.task_id] = job

        self._augment_with_runtime_activity(db, records, lease_snapshot)
        for record in records.values():
            if record.progress_mode == "indeterminate" or not record.message:
                self._apply_job_backend_progress(record)
            self._apply_heartbeat_progress(record)

        jobs = sorted(
            (record.to_api() for record in records.values()),
            key=lambda item: (
                {"stuck": 0, "stale": 1, "running": 2, "reserved": 3, "waiting": 4, "queued": 5, "failed": 6, "cancelled": 7}.get(item["state"], 99),
                item["queue"] or "",
                item["task_name"] or "",
                item["task_id"],
            ),
        )

        return {
            "jobs": jobs,
            "queues": self._queue_summaries(queued_jobs),
            "workers": workers,
            "leases": lease_snapshot,
            "generated_at": _utcnow().isoformat(),
        }

    def _record_cancel_action(
        self,
        db: Session,
        *,
        task_id: str,
        strategy: str,
        outcome: str,
        message: str,
    ) -> None:
        timestamp = _utcnow().strftime("%Y%m%d%H%M%S%f")
        row = AppSetting(
            key=f"operations.job_action.{task_id}.{timestamp}",
            value=json.dumps(
                {
                    "task_id": task_id,
                    "strategy": strategy,
                    "outcome": outcome,
                    "message": message,
                    "actor": "operator",
                    "timestamp": _utcnow().isoformat(),
                }
            ),
            category=OPERATIONS_JOB_ACTION_CATEGORY,
            description=f"Operations job action for {task_id}",
        )
        db.add(row)
        db.commit()

    def _find_scan_record(self, db: Session, task_id: str) -> _JobRecord | None:
        inventory = self.list_jobs(db)
        for item in inventory["jobs"]:
            if item["task_id"] == task_id:
                for record in self._queued_jobs(self._lease_snapshot()) + self._inspect_jobs()[0]:
                    if record.task_id == task_id:
                        return record
                # Fallback: synthesize a minimal record for runtime-only rows.
                return _JobRecord(
                    task_id=item["task_id"],
                    task_name=item["task_name"],
                    queue=item["queue"],
                    market=item["market"],
                    state=item["state"],
                    worker=item["worker"],
                    age_seconds=item["age_seconds"],
                    wait_reason=item["wait_reason"],
                    heartbeat_lag_seconds=item["heartbeat_lag_seconds"],
                    cancel_strategy=item["cancel_strategy"],
                    kwargs={"market": item["market"]} if item["market"] else {},
                )
        return None

    def _remove_queued_task(self, *, task_id: str) -> tuple[bool, str | None]:
        broker = self._broker()
        for queue_name in self.broker_queue_names():
            for raw in broker.lrange(queue_name, 0, -1):
                record = self._parse_queue_message(raw, queue_name)
                if record and record.task_id == task_id:
                    removed = broker.lrem(queue_name, 1, raw)
                    celery_app.control.revoke(task_id, terminate=False)
                    return bool(removed), queue_name
        return False, None

    def _cancel_scan(self, db: Session, scan_id: str) -> tuple[str, str]:
        from app.infra.db.uow import SqlUnitOfWork
        from app.database import SessionLocal

        uow = SqlUnitOfWork(SessionLocal)
        with uow:
            scan = uow.scans.get_by_scan_id(scan_id)
            if not scan:
                return "blocked", f"Scan {scan_id} not found"
            if scan.status not in {"queued", "running"}:
                return "blocked", f"Cannot cancel scan with status '{scan.status}'"
            uow.scans.update_status(scan_id, "cancelled")
            uow.commit()
        safe_publish_scan_bootstrap(scan_id)
        safe_publish_scan_bootstrap()
        return "accepted", f"Scan {scan_id} cancelled successfully"

    def cancel_job(self, db: Session, task_id: str) -> dict[str, Any]:
        record = self._find_scan_record(db, task_id)
        if record is None:
            message = f"Task {task_id} was not found in the Operations inventory."
            return {"status": "blocked", "cancel_strategy": "unknown", "message": message}

        strategy = record.cancel_strategy or _cancel_strategy_for(record)

        if strategy == "revoke_and_remove_from_queue":
            removed, queue_name = self._remove_queued_task(task_id=task_id)
            if removed:
                message = f"Removed task {task_id} from queue {queue_name} and revoked it."
                self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome="accepted", message=message)
                return {"status": "accepted", "cancel_strategy": strategy, "message": message}
            message = f"Task {task_id} is no longer queued."
            self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome="blocked", message=message)
            return {"status": "blocked", "cancel_strategy": strategy, "message": message}

        if strategy == "revoke":
            celery_app.control.revoke(task_id, terminate=False)
            message = f"Revoked reserved task {task_id}."
            self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome="accepted", message=message)
            return {"status": "accepted", "cancel_strategy": strategy, "message": message}

        if strategy == "scan_cancel":
            scan_id = _extract_scan_id(record.task_name, record.args or [], record.kwargs or {})
            if not scan_id:
                message = f"Could not resolve scan id for task {task_id}."
                self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome="blocked", message=message)
                return {"status": "blocked", "cancel_strategy": strategy, "message": message}
            status, message = self._cancel_scan(db, scan_id)
            self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome=status, message=message)
            return {"status": status, "cancel_strategy": strategy, "message": message}

        if strategy == "force_cancel_refresh":
            lock = get_data_fetch_lock()
            current = lock.get_any_current_task()
            if not current or current.get("task_id") != task_id:
                message = f"Task {task_id} is not the current external fetch holder."
                self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome="blocked", message=message)
                return {"status": "blocked", "cancel_strategy": strategy, "message": message}
            heartbeat_lag = _age_seconds_from(current.get("last_heartbeat"))
            if heartbeat_lag is not None and heartbeat_lag < HEARTBEAT_STUCK_AFTER_SECONDS:
                message = f"Task {task_id} is still progressing; force cancel is blocked."
                self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome="blocked", message=message)
                return {"status": "blocked", "cancel_strategy": strategy, "message": message}
            lock_key = current.get("lock_key", "")
            suffix = lock_key.rsplit(":", 1)[-1] if ":" in lock_key else ""
            market = None if suffix in {"", "shared"} else suffix
            lock.force_release(market=market)
            coordination = get_workload_coordination()
            coordination.release_market_workload(task_id, market=market)
            coordination.release_external_fetch(task_id)
            from app.wiring.bootstrap import get_price_cache

            get_price_cache().clear_warmup_heartbeat(market=market)
            message = f"Force-cancelled stale external fetch task {task_id}."
            self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome="accepted", message=message)
            return {"status": "accepted", "cancel_strategy": strategy, "message": message}

        if strategy == "force_release_market_lease":
            if not record.market:
                message = f"Task {task_id} does not have a market-scoped lease to release."
                self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome="blocked", message=message)
                return {"status": "blocked", "cancel_strategy": strategy, "message": message}
            coordination = get_workload_coordination()
            released = coordination.release_market_workload(task_id, market=record.market)
            if not released:
                message = f"Market workload lease for task {task_id} is no longer held."
                self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome="blocked", message=message)
                return {"status": "blocked", "cancel_strategy": strategy, "message": message}
            message = f"Released stale market workload lease for task {task_id} ({record.market})."
            self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome="accepted", message=message)
            return {"status": "accepted", "cancel_strategy": strategy, "message": message}

        message = f"Task {task_id} cannot be cancelled safely while in state '{record.state}'."
        self._record_cancel_action(db, task_id=task_id, strategy=strategy, outcome="unsupported", message=message)
        return {"status": "unsupported", "cancel_strategy": strategy, "message": message}
