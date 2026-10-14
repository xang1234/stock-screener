"""Inventory and control helpers for the Operations job console."""

from __future__ import annotations

import ast
import base64
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from uuid import uuid4
from typing import Any, Callable, Iterable, Optional

try:
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    redis = None

from sqlalchemy.orm import Session

from app.celery_app import _MARKET_JOB_TASKS, celery_app
from app.config import settings
from app.models.app_settings import AppSetting
from app.models.scan_result import Scan
from app.services.job_backend import CeleryJobBackend
from app.services.redis_pool import get_redis_client
from app.services.market_activity_service import (
    MARKET_ACTIVITY_KEY_PREFIX,
    RUNTIME_ACTIVITY_CATEGORY,
)
from app.services.runtime_activity_contract import progress_mode
from app.services.task_registry_service import record_task_run
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
# A queued message whose queue no responding worker reads: it never runs.
NO_WORKER_WAIT_REASON = "no_worker_for_queue"
# Each worker's queues as it last answered inspect, so a worker silent this
# poll (a solo-pool worker busy with a task answers nothing) still counts.
WORKER_QUEUES_KEY = "operations:worker_queues"
# ponytail: a dead worker still counts this long, so its queue flags up to a
# day late; a busy solo-pool worker can stay silent for hours of fetching.
WORKER_QUEUES_MEMORY = timedelta(hours=24)

SCAN_TASK_NAME = "app.tasks.scan_tasks.run_bulk_scan"
RELEASE_PIPELINE_CLAIM_TASK = "app.tasks.daily_market_pipeline_tasks.release_daily_market_pipeline_claim"


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
    cancel_requested: dict[str, Any] | None = None
    # (market, token) of the daily pipeline chain this queued stage belongs to.
    chain_claim: tuple[str, str] | None = None

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
            "cancel_requested_at": (self.cancel_requested or {}).get("timestamp"),
            "cancel_requested_strategy": (self.cancel_requested or {}).get("strategy"),
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def remember_worker_queues(worker_queues: dict[str, list[str]], *, client: Any = None) -> bool:
    """Record each worker's queues with the time seen. A worker registers at
    startup too, so one silent from its first task is still known. False when
    the cache Redis is unavailable."""
    if client is None:
        client = get_redis_client()
    if client is None:
        return False
    seen_at = _utcnow().isoformat()
    try:
        client.hset(
            WORKER_QUEUES_KEY,
            mapping={
                name: json.dumps({"queues": queues, "seen_at": seen_at})
                for name, queues in worker_queues.items()
            },
        )
    except redis.RedisError:
        logger.warning("Could not record worker queues", exc_info=True)
        return False
    return True


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


def _pipeline_chain_claim(
    task_name: str, kwargs: dict[str, Any], embed: Any
) -> tuple[str, str] | None:
    """The (market, token) claim a queued pipeline stage belongs to.

    Dispatch links the claim's release as every stage's errback and as the
    chain's last stage, so a stage message carries it in its embed; the
    release stage carries it in its own kwargs.
    """
    candidates: list[Any] = []
    if task_name == RELEASE_PIPELINE_CLAIM_TASK:
        candidates.append({"task": task_name, "kwargs": kwargs})
    if isinstance(embed, dict):
        candidates += list(embed.get("errbacks") or []) + list(embed.get("chain") or [])
    for signature in candidates:
        if not isinstance(signature, dict) or signature.get("task") != RELEASE_PIPELINE_CLAIM_TASK:
            continue
        claim_kwargs = signature.get("kwargs") or {}
        market, token = claim_kwargs.get("market"), claim_kwargs.get("token")
        if isinstance(market, str) and isinstance(token, str):
            return market, token
    return None


def _cancel_strategy_for(record: _JobRecord) -> str:
    if record.wait_reason == "scheduled_eta":
        # A worker holds the message until its ETA; it is not in the broker list.
        return "revoke"
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
            # Daily pipeline chain claims (#598): held from dispatch while the
            # chain is alive (#605); a stage killed with its worker leaves one.
            "pipeline_chain": {
                market: holder
                for market, holder in coordination.get_pipeline_chain_holders().items()
                if holder
            },
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
        chain_claim: tuple[str, str] | None = None
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
                    embed = body_payload[2] if len(body_payload) >= 3 else {}
                    chain_claim = _pipeline_chain_claim(task_name, kwargs, embed)
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
            chain_claim=chain_claim,
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

    def _inspect_jobs(self) -> tuple[list[_JobRecord], list[dict[str, Any]], set[str]]:
        """Live jobs, the worker roster, and the workers that answered
        active_queues (an empty queue list there is a known zero)."""
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

        jobs.extend(self._scheduled_records(scheduled, queue_map))
        return jobs, workers, set(queue_map)

    def _scheduled_records(
        self, scheduled: dict[str, Any], queue_map: dict[str, list[str]]
    ) -> list[_JobRecord]:
        jobs: list[_JobRecord] = []
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
        return jobs

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
            progress_mode=progress_mode(
                state,
                current_task.get("progress"),
                current_task.get("current"),
                current_task.get("total"),
            ),
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
            progress_mode=progress_mode(
                state,
                runtime_record.get("percent"),
                runtime_record.get("current"),
                runtime_record.get("total"),
            ),
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
        record.progress_mode = progress_mode(
            record.state,
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
        record.progress_mode = progress_mode(
            record.state,
            record.percent,
            record.current,
            record.total,
        )

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
        record.progress_mode = progress_mode(
            record.state,
            record.percent,
            record.current,
            record.total,
        )

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
                progress_mode=progress_mode(
                    state,
                    runtime_record.get("percent"),
                    runtime_record.get("current"),
                    runtime_record.get("total"),
                ),
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

    @staticmethod
    def _queue_consumers(
        workers: list[dict[str, Any]], queue_repliers: set[str]
    ) -> dict[str, int] | None:
        """Workers per queue, counting a worker silent this poll by the queues
        it last recorded (here or at startup). None (unknown, which must not
        read as "nobody serves this queue") when no worker answered
        active_queues, the cache Redis holding those queues is unavailable, or
        a worker answered without its queues and none are remembered.

        ponytail: a worker never recorded (cache Redis flushed while it is
        silent) still reads as missing until it answers or restarts."""
        answered = {worker["worker"]: worker["queues"] for worker in workers if worker["worker"] in queue_repliers}
        client = get_redis_client()
        if not answered or client is None or not remember_worker_queues(answered, client=client):
            return None
        now = _utcnow()
        try:
            stored = client.hgetall(WORKER_QUEUES_KEY)
        except redis.RedisError:
            logger.warning("Could not read remembered worker queues", exc_info=True)
            return None

        queue_lists: dict[str, list[str]] = {}
        expired: list[bytes] = []
        for raw_name, raw_entry in stored.items():
            try:
                entry = json.loads(raw_entry)
                queues, seen_at = entry["queues"], _parse_iso(entry["seen_at"])
            except (ValueError, TypeError, KeyError):
                expired.append(raw_name)
                continue
            if (
                not isinstance(queues, list)
                or not all(isinstance(queue, str) for queue in queues)
                or seen_at is None
                or now - seen_at > WORKER_QUEUES_MEMORY
            ):
                expired.append(raw_name)
                continue
            queue_lists[raw_name.decode()] = queues
        if expired:
            # Worker names carry the container hostname, new on each restart.
            try:
                client.hdel(WORKER_QUEUES_KEY, *expired)
            except redis.RedisError:
                logger.warning("Could not prune remembered worker queues", exc_info=True)

        if any(worker["worker"] not in queue_lists for worker in workers):
            return None
        return Counter(queue for queues in queue_lists.values() for queue in queues)

    def _queue_summaries(
        self, queued_jobs: Iterable[_JobRecord], consumers: dict[str, int] | None
    ) -> list[dict[str, Any]]:
        def _summary(queue_name: str) -> dict[str, Any]:
            return {
                "queue": queue_name,
                "depth": 0,
                "oldest_age_seconds": None,
                "consumers": None if consumers is None else consumers.get(queue_name, 0),
            }

        depths: dict[str, dict[str, Any]] = {
            queue_name: _summary(queue_name) for queue_name in self.broker_queue_names()
        }
        for job in queued_jobs:
            if not job.queue:
                continue
            summary = depths.setdefault(job.queue, _summary(job.queue))
            summary["depth"] += 1
            if job.age_seconds is not None:
                current_oldest = summary["oldest_age_seconds"]
                summary["oldest_age_seconds"] = (
                    job.age_seconds
                    if current_oldest is None
                    else max(current_oldest, job.age_seconds)
                )
        return sorted(depths.values(), key=lambda item: item["queue"])

    @staticmethod
    def _merge_live_jobs(jobs: Iterable[_JobRecord]) -> dict[str, _JobRecord]:
        records: dict[str, _JobRecord] = {}
        for job in jobs:
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
        return records

    def _cancel_requests(self, db: Session) -> dict[str, dict[str, Any]]:
        """Latest cancel per task that was accepted, or whose outcome was never
        recorded (#613: the action may well have happened, so its button must
        not invite a repeat). A row reads "cancel requested" until the job
        actually leaves the inventory. Rows written before #613 carry no
        action id and count only when accepted."""
        # ponytail: reads the last day of audit rows; a job still listed after
        # a day loses the marker. Index by task id if the rows pile up.
        since = _utcnow() - timedelta(days=1)
        rows = (
            db.query(AppSetting)
            .filter(AppSetting.category == OPERATIONS_JOB_ACTION_CATEGORY)
            .filter(AppSetting.created_at >= since)
            .all()
        )
        actions = []
        for row in rows:
            try:
                action = json.loads(row.value)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(action, dict):
                actions.append(action)
        recorded = {
            action.get("action_id")
            for action in actions
            if action.get("action_id") and action.get("outcome") != "requested"
        }
        requests: dict[str, dict[str, Any]] = {}
        for action in actions:
            outcome = action.get("outcome")
            unrecorded = (
                outcome == "requested"
                and action.get("action_id")
                and action["action_id"] not in recorded
            )
            if outcome != "accepted" and not unrecorded:
                continue
            # A force action finishes on the spot; a leftover stale row after it
            # should keep its button rather than read "requested" for a day.
            if str(action.get("strategy", "")).startswith("force_"):
                continue
            task_id = action.get("task_id")
            previous = requests.get(task_id)
            if task_id and (
                previous is None
                or str(action.get("timestamp")) > str(previous.get("timestamp"))
            ):
                requests[task_id] = action
        return requests

    def list_jobs(self, db: Session) -> dict[str, Any]:
        lease_snapshot = self._lease_snapshot()
        queued_jobs = self._queued_jobs(lease_snapshot)
        inspect_jobs, workers, queue_repliers = self._inspect_jobs()

        records = self._merge_live_jobs([*queued_jobs, *inspect_jobs])
        self._augment_with_runtime_activity(db, records, lease_snapshot)
        consumers = self._queue_consumers(workers, queue_repliers)
        cancel_requests = self._cancel_requests(db)
        for record in records.values():
            # Only a message still sitting in a broker list: its queue is known
            # for sure. Outranks a lease wait, which ends; this one never does.
            if consumers is not None and record.source == "broker" and not consumers.get(record.queue):
                record.wait_reason = NO_WORKER_WAIT_REASON
            if record.progress_mode == "indeterminate" or not record.message:
                self._apply_job_backend_progress(record)
            self._apply_heartbeat_progress(record)
            record.cancel_requested = cancel_requests.get(record.task_id)

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
            "queues": self._queue_summaries(queued_jobs, consumers),
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
        action_id: str,
    ) -> None:
        row = AppSetting(
            # Unique per row: a timestamp is not, on a coarse clock (Windows
            # ticks ~15 ms) or for two clicks at once.
            key=f"operations.job_action.{uuid4().hex}",
            value=json.dumps(
                {
                    "task_id": task_id,
                    "action_id": action_id,
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

    def _query_task(self, task_id: str) -> list[_JobRecord]:
        """Ask workers about one task id: one broadcast instead of the five a
        full inventory sends."""
        replies = self._inspect().query_task(task_id) or {}
        records: list[_JobRecord] = []
        for worker, tasks in replies.items():
            state, request = (tasks or {}).get(task_id) or (None, None)
            # ETA-scheduled tasks are not in worker requests until their ETA;
            # _find_job_record falls back to inspect.scheduled() for them.
            mapped = {"active": "running", "reserved": "reserved"}.get(state)
            if not mapped or not isinstance(request, dict):
                continue
            queue_name = (request.get("delivery_info") or {}).get("routing_key")
            record = self._inspect_record(request, state=mapped, worker=worker, queue_name=queue_name)
            if record:
                records.append(record)
        return records

    def _find_job_record(self, db: Session, task_id: str) -> _JobRecord | None:
        """Resolve one task the way list_jobs would, without rebuilding the
        whole inventory."""
        lease_snapshot = self._lease_snapshot()
        live = [job for job in self._queued_jobs(lease_snapshot) if job.task_id == task_id]
        # Always ask the workers: a redelivered task can be in the broker list
        # and on a worker at once, and the merge prefers the worker state.
        live += self._query_task(task_id)
        if not live:
            scheduled = self._scheduled_records(self._inspect().scheduled() or {}, {})
            live = [job for job in scheduled if job.task_id == task_id]
        records = self._merge_live_jobs(live)
        if task_id not in records:
            self._augment_with_runtime_activity(db, records, lease_snapshot)
        return records.get(task_id)

    def _remove_queued_task(self, *, task_id: str) -> tuple[bool, str | None]:
        broker = self._broker()
        for queue_name in self.broker_queue_names():
            for raw in broker.lrange(queue_name, 0, -1):
                record = self._parse_queue_message(raw, queue_name)
                if record and record.task_id == task_id:
                    removed = broker.lrem(queue_name, 1, raw)
                    # Release before revoking: once removed, the stage can never
                    # run its errback, so a revoke error must not skip this.
                    if removed and record.chain_claim:
                        self._release_removed_chain(record)
                    celery_app.control.revoke(task_id, terminate=False)
                    return bool(removed), queue_name
        return False, None

    def _release_removed_chain(self, record: _JobRecord) -> None:
        """A chain has one live message at a time, so removing a queued stage
        ends the chain; its errback never fires, so free the claim here.
        Owner-checked: a newer chain's claim is left alone.

        A queued copy cannot coexist with the same stage running: acks_late is
        off, so a worker acks before running and kombu only restores
        prefetched messages that never started.
        """
        market, token = record.chain_claim
        try:
            get_workload_coordination().release_pipeline_chain(token, market=market)
        except Exception:
            # The stage is already removed; the claim then expires on its own.
            logger.warning(
                "Could not release the %s pipeline chain claim after removing %s",
                market, record.task_id, exc_info=True,
            )

    def release_pipeline_chain(self, db: Session, market: str, token: str) -> dict[str, Any]:
        """Operator release of a market's chain claim, only if ``token`` still
        holds it. For a claim left by a stage killed with its worker."""

        def release() -> tuple[str, str]:
            if get_workload_coordination().release_pipeline_chain(token, market=market):
                return "accepted", f"Released the {market} pipeline chain claim {token}."
            return "blocked", f"The {market} pipeline chain claim {token} is no longer held."

        return self._audited(db, task_id=token, strategy="release_pipeline_chain", act=release)

    def _audited(
        self,
        db: Session,
        *,
        task_id: str,
        strategy: str,
        act: Callable[[], tuple[str, str]],
    ) -> dict[str, Any]:
        """Run a job action with its audit trail written intent-first (#613).

        The ``requested`` row is committed before ``act`` changes Redis, the
        broker or a scan; if it cannot be, nothing is done. The outcome row
        follows (``error`` if ``act`` raised, which is then re-raised); if
        that write fails, the action's real result is still returned, since
        the intent row already shows the attempt.
        """
        action_id = uuid4().hex
        try:
            self._record_cancel_action(
                db, task_id=task_id, strategy=strategy, outcome="requested",
                message=f"Operator requested {strategy} for {task_id}.",
                action_id=action_id,
            )
        except Exception:
            db.rollback()
            logger.warning(
                "Could not record the %s request for %s; not acting", strategy, task_id,
                exc_info=True,
            )
            message = (
                f"Could not record this action in the audit log, so nothing was done "
                f"to {task_id}. Retry shortly."
            )
            return {"status": "blocked", "cancel_strategy": strategy, "message": message}

        try:
            status, message = act()
        except Exception as exc:
            # It may have changed state before failing (a queue removal that
            # then cannot revoke), so record that it broke, then re-raise.
            self._record_outcome(
                db, task_id, strategy, action_id, "error", f"{type(exc).__name__}: {exc}"
            )
            raise
        self._record_outcome(db, task_id, strategy, action_id, status, message)
        return {"status": status, "cancel_strategy": strategy, "message": message}

    def _record_outcome(
        self, db: Session, task_id: str, strategy: str, action_id: str, outcome: str, message: str
    ) -> None:
        """Best effort: the request row already shows the attempt."""
        try:
            self._record_cancel_action(
                db, task_id=task_id, strategy=strategy, outcome=outcome, message=message,
                action_id=action_id,
            )
        except Exception:
            db.rollback()
            logger.warning(
                "Could not record the outcome (%s) of %s for %s; its request row stands",
                outcome, strategy, task_id, exc_info=True,
            )

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
        from app.tasks.scan_tasks import queue_scan_bootstrap_publish

        queue_scan_bootstrap_publish(scan_id)
        return "accepted", f"Scan {scan_id} cancelled successfully"

    def cancel_job(
        self,
        db: Session,
        task_id: str,
        expected_strategy: str | None = None,
    ) -> dict[str, Any]:
        record = self._find_job_record(db, task_id)
        if record is None:
            message = f"Task {task_id} is no longer queued or running. Refresh to see its current state."
            return {"status": "blocked", "cancel_strategy": "unknown", "message": message}

        strategy = record.cancel_strategy or _cancel_strategy_for(record)
        # Compare-and-set on what the operator confirmed. The strategy decides
        # what the click does; state alone flickers between queued and waiting.
        if expected_strategy and expected_strategy != strategy:
            message = f"Job moved to {record.state}, so this action no longer applies. Refresh and retry."
            return {"status": "conflict", "cancel_strategy": strategy, "message": message}

        return self._audited(
            db,
            task_id=task_id,
            strategy=strategy,
            act=lambda: self._perform_job_action(db, record, task_id, strategy),
        )

    def _perform_job_action(
        self, db: Session, record: _JobRecord, task_id: str, strategy: str
    ) -> tuple[str, str]:
        """Carry out ``strategy`` for ``task_id``: (status, message). Writes no
        audit rows; ``_audited`` records the intent before and the outcome after."""
        if strategy == "revoke_and_remove_from_queue":
            removed, queue_name = self._remove_queued_task(task_id=task_id)
            if removed:
                # The message never reaches a worker, so no task signal will
                # close its run-history row; close it here.
                record_task_run(task_id, record.task_name, record.kwargs, "cancelled")
                message = f"Removed task {task_id} from queue {queue_name} and revoked it."
                return "accepted", message
            message = f"Task {task_id} is no longer queued."
            return "blocked", message

        if strategy == "revoke":
            celery_app.control.revoke(task_id, terminate=False)
            message = f"Revoked reserved task {task_id}."
            return "accepted", message

        if strategy == "scan_cancel":
            scan_id = _extract_scan_id(record.task_name, record.args or [], record.kwargs or {})
            if not scan_id:
                # Lease-only rows (a busy solo worker skips inspect) carry no
                # args; the scan row records its Celery task id.
                scan_id = db.query(Scan.scan_id).filter(Scan.task_id == task_id).scalar()
            if not scan_id:
                message = f"Could not resolve scan id for task {task_id}."
                return "blocked", message
            status, message = self._cancel_scan(db, scan_id)
            return status, message

        if strategy == "force_cancel_refresh":
            lock = get_data_fetch_lock()
            current = lock.get_any_current_task()
            if not current or current.get("task_id") != task_id:
                message = f"Task {task_id} is not the current external fetch holder."
                return "blocked", message
            heartbeat_lag = _age_seconds_from(current.get("last_heartbeat"))
            if heartbeat_lag is not None and heartbeat_lag < HEARTBEAT_STUCK_AFTER_SECONDS:
                message = f"Task {task_id} is still progressing; force cancel is blocked."
                return "blocked", message
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
            return "accepted", message

        if strategy == "force_release_market_lease":
            if not record.market:
                message = f"Task {task_id} does not have a market-scoped lease to release."
                return "blocked", message
            coordination = get_workload_coordination()
            released = coordination.release_market_workload(task_id, market=record.market)
            if not released:
                message = f"Market workload lease for task {task_id} is no longer held."
                return "blocked", message
            message = f"Released stale market workload lease for task {task_id} ({record.market})."
            return "accepted", message

        message = f"Task {task_id} cannot be cancelled safely while in state '{record.state}'."
        return "unsupported", message
