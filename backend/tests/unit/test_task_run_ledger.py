"""Scheduled-task run history is recorded by the worker, not by browser polling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from celery.exceptions import TimeLimitExceeded
from celery.signals import task_postrun, task_prerun

import app.celery_app  # noqa: F401  (connects the signal handlers)
from app.models.task_execution import TaskExecutionHistory
from app.schemas.task import TaskStatusResponse
from app.services.task_registry_service import TaskRegistryService, record_task_run

PIPELINE = "app.tasks.daily_market_pipeline_tasks.queue_daily_market_pipeline"
CLEANUP = "app.tasks.cache_tasks.cleanup_orphaned_scans"
SOCIAL = "app.interfaces.tasks.social_signal_tasks.refresh_social_signals"
OPTIONS = "app.interfaces.tasks.options_analytics_tasks.refresh_options_analytics"


class _Task:
    """Stands in for a Celery task as a signal sender (senders must be hashable)."""

    def __init__(self, name: str) -> None:
        self.name = name


def _rows(db, task_id):
    db.expire_all()
    return db.query(TaskExecutionHistory).filter_by(task_id=task_id).all()


def _start(name, task_id, kwargs):
    sender = _Task(name)
    task_prerun.send(sender=sender, task_id=task_id, task=sender, args=(), kwargs=kwargs)


def _finish(name, task_id, kwargs, *, state, retval=None):
    sender = _Task(name)
    task_postrun.send(
        sender=sender, task_id=task_id, task=sender, args=(), kwargs=kwargs, retval=retval, state=state,
    )


def _last_run(db, name):
    tasks = TaskRegistryService().get_all_scheduled_tasks(db)
    return next(t for t in tasks if t["name"] == name)["last_run"]


def test_worker_run_is_recorded_start_to_finish(db_session):
    task_id = str(uuid4())

    _start(PIPELINE, task_id, {"market": "HK"})
    [row] = _rows(db_session, task_id)
    assert (row.task_name, row.status, row.triggered_by) == ("daily-market-pipeline-hk", "running", "auto")

    _finish(PIPELINE, task_id, {"market": "HK"}, state="SUCCESS", retval={"status": "ok"})
    [row] = _rows(db_session, task_id)
    assert row.status == "completed"
    assert row.completed_at is not None
    assert row.duration_seconds is not None
    assert row.result_summary == {"status": "ok"}


def test_failed_run_records_error_message(db_session):
    task_id = str(uuid4())

    _start(CLEANUP, task_id, {})
    _finish(CLEANUP, task_id, {}, state="FAILURE", retval=RuntimeError("boom"))

    [row] = _rows(db_session, task_id)
    assert (row.task_name, row.status, row.error_message) == ("weekly-orphaned-scan-cleanup", "failed", "boom")


@pytest.mark.parametrize(
    ("retval", "expected_status", "expected_error"),
    [
        # weekly_full_refresh / cleanup_old_price_data / refresh_all_fundamentals
        # catch the exception and return it, so Celery reports SUCCESS.
        ({"error": "db locked", "completed_at": "x"}, "failed", "db locked"),
        # refresh_stock_universe tags the status as well.
        ({"status": "error", "error": "feed down"}, "failed", "feed down"),
        ({"status": "failed"}, "failed", "Task reported status 'failed'"),
        ({"status": "skipped", "reason": "not_trading_day"}, "skipped", None),
        # Options: a quality-gate outcome is a finished run, not a failure.
        ({"status": "failed_quality", "reason_codes": ["low_coverage"]}, "completed", None),
        ({"status": "success", "error": None}, "completed", None),
    ],
)
def test_error_results_returned_as_success_are_recorded_as_failed(
    db_session, retval, expected_status, expected_error,
):
    task_id = str(uuid4())
    _start(CLEANUP, task_id, {})
    _finish(CLEANUP, task_id, {}, state="SUCCESS", retval=retval)

    [row] = _rows(db_session, task_id)
    assert (row.status, row.error_message) == (expected_status, expected_error)


@pytest.mark.parametrize(
    ("retval", "expected_detail"),
    [
        # queue_daily_market_pipeline on a weekend / with a run in progress.
        ({"status": "skipped", "reason": "not_trading_day", "market": "US"}, "not_trading_day"),
        ({"status": "skipped", "reason": "market_pipeline_already_active"}, "market_pipeline_already_active"),
        # cache_tasks' older shape.
        ({"skipped": True, "reason": "Not a trading day"}, "Not a trading day"),
        # options / social tag reason codes.
        ({"status": "skipped", "reason_codes": ["market_unsupported", "no_cohort"]}, "market_unsupported, no_cohort"),
        ({"status": "skipped"}, None),
        # serialized_data_fetch_task when the data-fetch lock is held: the
        # weekly jobs' most common skip, with only a message.
        ({"status": "already_running", "skipped": True, "message": "Data fetch already in progress (x)"},
         "Data fetch already in progress (x)"),
    ],
)
def test_skipped_runs_are_not_reported_as_completed(db_session, retval, expected_detail):
    task_id = str(uuid4())
    _start(PIPELINE, task_id, {"market": "US"})
    _finish(PIPELINE, task_id, {"market": "US"}, state="SUCCESS", retval=retval)

    last_run = _last_run(db_session, "daily-market-pipeline-us")

    assert (last_run["status"], last_run["detail"]) == ("skipped", expected_detail)


def _social_result(processing_status, *reason_codes):
    # refresh_social_signals returns asdict(SocialRunResult): no 'status' key.
    return {
        "run_id": "r1", "mode": "live", "processing_status": processing_status,
        "published": False, "coverage_summary": [], "reason_codes": list(reason_codes),
    }


@pytest.mark.parametrize(
    ("retval", "expected"),
    [
        (_social_result("skipped", "social_runtime_unavailable"), ("skipped", "social_runtime_unavailable", None)),
        (_social_result("blocked", "minimum_two_enabled"), ("skipped", "minimum_two_enabled", None)),
        # Every list read failed: a failure, not a skip.
        (_social_result("blocked", "source_participation_failed"),
         ("failed", None, "Social source collection failed (source_participation_failed)")),
        # Analysis ended unsuccessfully (RefreshSocialSignals fail_analysis).
        (_social_result("failed", "analysis_failed"),
         ("failed", None, "Social Signals analysis failed (analysis_failed)")),
        # Collected; analysis resumes later on its own.
        (_social_result("deferred", "analysis_incomplete"), ("completed", None, None)),
        (_social_result("complete"), ("completed", None, None)),
    ],
)
def test_social_signal_outcomes_come_from_processing_status(db_session, retval, expected):
    task_id = str(uuid4())
    _start(SOCIAL, task_id, {"origin": "scheduled"})
    _finish(SOCIAL, task_id, {"origin": "scheduled"}, state="SUCCESS", retval=retval)

    last_run = _last_run(db_session, "social-signal-refresh")

    assert (last_run["status"], last_run["detail"], last_run["error_message"]) == expected


@pytest.mark.parametrize("count", [0, 5])
def test_a_count_of_skipped_symbols_is_not_a_skipped_run(db_session, count):
    # refresh_all_fundamentals reports how many symbols it skipped.
    task_id = str(uuid4())
    _start("app.tasks.fundamentals_tasks.refresh_all_fundamentals", task_id, {"market": "US"})
    _finish("app.tasks.fundamentals_tasks.refresh_all_fundamentals", task_id, {"market": "US"},
            state="SUCCESS", retval={"status": "success", "updated": 10, "skipped": count})

    [row] = _rows(db_session, task_id)
    assert row.status == "completed"


def test_status_fallback_writer_classifies_a_skipped_result(db_session):
    # An open run whose signal write was lost: the status endpoint reads
    # Celery, reports completed (Options contract) and records skipped.
    task_id = str(uuid4())
    _start(PIPELINE, task_id, {"market": "US"})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        async_result.return_value = SimpleNamespace(
            state="SUCCESS", result={"status": "skipped", "reason": "not_trading_day"}, info=None,
        )
        status = TaskRegistryService().get_task_status("daily-market-pipeline-us", task_id, db_session)

    assert status["status"] == "completed"
    [row] = _rows(db_session, task_id)
    assert row.status == "skipped"


def test_status_fallback_reports_a_social_failure_as_failed(db_session):
    # The response must agree with what the same call records.
    task_id = str(uuid4())
    _start(SOCIAL, task_id, {"origin": "scheduled"})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        async_result.return_value = SimpleNamespace(
            state="SUCCESS", result=_social_result("blocked", "source_participation_failed"), info=None,
        )
        status = TaskRegistryService().get_task_status("social-signal-refresh", task_id, db_session)

    assert (status["status"], status["error"]) == (
        "failed", "Social source collection failed (source_participation_failed)",
    )
    [row] = _rows(db_session, task_id)
    assert row.status == "failed"


def test_pipeline_handoff_is_reported_as_dispatched_not_completed(db_session):
    # queue_daily_market_pipeline only queues the stage chain and returns in
    # seconds; the real work (prices, breadth, groups, scan) is still ahead.
    task_id = str(uuid4())
    _start(PIPELINE, task_id, {"market": "US"})
    _finish(PIPELINE, task_id, {"market": "US"}, state="SUCCESS",
            retval={"status": "queued", "market": "US", "task_id": "chain-1"})

    last_run = _last_run(db_session, "daily-market-pipeline-us")

    assert (last_run["status"], last_run["detail"]) == ("dispatched", None)


def test_reconcile_classifies_skipped_results_too(db_session):
    task_id = str(uuid4())
    _start(PIPELINE, task_id, {"market": "US"})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        async_result.return_value = SimpleNamespace(state="SUCCESS", result={"status": "skipped", "reason": "x"})
        assert TaskRegistryService().reconcile_open_runs(db_session) == 1

    [row] = _rows(db_session, task_id)
    assert row.status == "skipped"


def test_status_endpoint_keeps_reporting_skipped_runs_as_completed(db_session):
    # The Options page polls this endpoint and reads result.status itself; it
    # only stops on completed/failed/cancelled.
    task_id = str(uuid4())
    _start(OPTIONS, task_id, {"market": "US"})
    _finish(OPTIONS, task_id, {"market": "US"}, state="SUCCESS",
            retval={"status": "skipped", "reason_codes": ["market_unsupported"]})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        status = TaskRegistryService().get_task_status("daily-us-options-analytics", task_id, db_session)

    assert status["status"] == "completed"
    assert status["result"] == {"status": "skipped", "reason_codes": ["market_unsupported"]}
    assert TaskStatusResponse(**status).celery_state == "SUCCESS"
    async_result.assert_not_called()


def test_reconcile_records_an_error_result_as_failed(db_session):
    task_id = str(uuid4())
    _start(CLEANUP, task_id, {})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        async_result.return_value = SimpleNamespace(state="SUCCESS", result={"error": "db locked"})
        assert TaskRegistryService().reconcile_open_runs(db_session) == 1

    [row] = _rows(db_session, task_id)
    assert (row.status, row.error_message) == ("failed", "db locked")


def test_status_endpoint_reports_an_error_result_as_failed(db_session):
    task_id = str(uuid4())
    _start(CLEANUP, task_id, {})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        async_result.return_value = SimpleNamespace(state="SUCCESS", result={"error": "db locked"}, info=None)
        status = TaskRegistryService().get_task_status("weekly-orphaned-scan-cleanup", task_id, db_session)

    assert (status["status"], status["error"]) == ("failed", "db locked")
    [row] = _rows(db_session, task_id)
    assert (row.status, row.error_message) == ("failed", "db locked")


def test_retry_does_not_rewind_a_running_row(db_session):
    task_id = str(uuid4())

    _start(CLEANUP, task_id, {})
    _finish(CLEANUP, task_id, {}, state="RETRY")

    [row] = _rows(db_session, task_id)
    assert row.status == "running"


def test_retry_keeps_the_first_start_time(db_session):
    # A retry fires task_prerun again with the same id; duration and the
    # lost cutoff must cover the whole run, not just the last attempt.
    task_id = str(uuid4())
    _start(CLEANUP, task_id, {})
    first_start = datetime.now(timezone.utc) - timedelta(minutes=10)
    db_session.query(TaskExecutionHistory).filter_by(task_id=task_id).update({"started_at": first_start})
    db_session.commit()

    _finish(CLEANUP, task_id, {}, state="RETRY")
    _start(CLEANUP, task_id, {})
    _finish(CLEANUP, task_id, {}, state="SUCCESS")

    [row] = _rows(db_session, task_id)
    assert row.status == "completed"
    assert row.duration_seconds >= 600


def test_manual_run_start_time_is_when_a_worker_picks_it_up(db_session):
    service = TaskRegistryService()
    service._task_imports["weekly-orphaned-scan-cleanup"] = SimpleNamespace(
        apply_async=lambda **kwargs: SimpleNamespace(id=kwargs["task_id"])
    )
    task_id = service.trigger_task("weekly-orphaned-scan-cleanup", db_session)["task_id"]
    queued_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    db_session.query(TaskExecutionHistory).filter_by(task_id=task_id).update({"started_at": queued_at})
    db_session.commit()

    _start(CLEANUP, task_id, {})

    [row] = _rows(db_session, task_id)
    assert datetime.now(timezone.utc) - row.started_at.replace(tzinfo=timezone.utc) < timedelta(minutes=1)


def test_unserializable_result_still_records_the_outcome(db_session):
    task_id = str(uuid4())

    _start(CLEANUP, task_id, {})
    _finish(CLEANUP, task_id, {}, state="SUCCESS", retval={"ratio": float("nan")})

    [row] = _rows(db_session, task_id)
    assert row.status == "completed"
    assert row.result_summary is None


def test_tasks_outside_the_registry_are_not_recorded(db_session):
    task_id = str(uuid4())

    _start("app.tasks.theme_intelligence_tasks.refresh_groups", task_id, {})

    assert _rows(db_session, task_id) == []


def test_ad_hoc_social_refreshes_are_not_recorded_but_scheduled_ones_are(db_session):
    # prepare_live_attachments can dispatch the same task every minute; those
    # runs are not the six-hourly scheduled job and would flood the history.
    ad_hoc, scheduled = str(uuid4()), str(uuid4())

    _start(SOCIAL, ad_hoc, {"origin": "attachment-evidence"})
    _start(SOCIAL, scheduled, {"origin": "scheduled"})

    assert _rows(db_session, ad_hoc) == []
    [row] = _rows(db_session, scheduled)
    assert (row.task_name, row.triggered_by) == ("social-signal-refresh", "scheduled")


def test_recording_failure_is_logged_not_raised(monkeypatch, caplog):
    from app.services import task_registry_service

    def broken_session():
        raise RuntimeError("db down")

    monkeypatch.setattr(task_registry_service, "SessionLocal", broken_session)

    # Called directly (Celery's Signal.send would swallow errors on its own):
    # the Operations cancel path relies on this never raising.
    record_task_run("x", CLEANUP, {}, "cancelled")

    assert "Failed to record" in caplog.text


@pytest.mark.parametrize(
    ("celery_state", "result", "expected_status", "expected_error"),
    [
        ("FAILURE", TimeLimitExceeded(2100), "failed", "TimeLimitExceeded(2100,)"),
        ("REVOKED", None, "cancelled", None),
    ],
)
def test_reconcile_closes_open_runs_from_the_celery_result(
    db_session, celery_state, result, expected_status, expected_error,
):
    # A hard time limit or a lost worker kills the child before task_postrun,
    # and revokes are announced in the parent; Celery still stores the outcome.
    task_id = str(uuid4())
    _start("app.tasks.cache_tasks.weekly_full_refresh", task_id, {"market": "US"})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        async_result.return_value = SimpleNamespace(state=celery_state, result=result)
        assert TaskRegistryService().reconcile_open_runs(db_session) == 1

    [row] = _rows(db_session, task_id)
    assert row.status == expected_status
    assert row.error_message == expected_error


def test_reconcile_leaves_runs_celery_still_reports_open(db_session):
    task_id = str(uuid4())
    _start("app.tasks.cache_tasks.weekly_full_refresh", task_id, {"market": "US"})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        async_result.return_value = SimpleNamespace(state="STARTED", result=None)
        assert TaskRegistryService().reconcile_open_runs(db_session) == 0

    [row] = _rows(db_session, task_id)
    assert row.status == "running"


def test_reconcile_keeps_going_when_one_lookup_fails(db_session, caplog):
    broken, closable = str(uuid4()), str(uuid4())
    _start("app.tasks.cache_tasks.weekly_full_refresh", broken, {"market": "US"})
    _start(CLEANUP, closable, {})

    def async_result(task_id):
        if task_id == broken:
            raise ConnectionError("redis down")
        return SimpleNamespace(state="SUCCESS", result={"deleted": 3})

    with patch("app.services.task_registry_service.AsyncResult", side_effect=async_result):
        assert TaskRegistryService().reconcile_open_runs(db_session) == 1

    assert _rows(db_session, closable)[0].status == "completed"
    assert "Could not read Celery state" in caplog.text


def test_reconcile_runs_in_a_worker_on_a_beat_schedule():
    from app.celery_app import celery_app

    entry = celery_app.conf.beat_schedule["task-run-reconcile"]
    assert entry["task"] == "app.tasks.task_history_tasks.reconcile_task_runs"
    assert entry["options"]["queue"] == "celery"
    assert "app.tasks.task_history_tasks" in celery_app.conf.include


def test_task_list_is_read_only_and_never_waits_on_the_result_backend(db_session):
    # The result backend retries for minutes when Redis is down; the list is
    # polled by the UI and the copilot, so it must only read the database.
    task_id = str(uuid4())
    _start("app.tasks.cache_tasks.weekly_full_refresh", task_id, {"market": "US"})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        assert _last_run(db_session, "weekly-full-refresh-us")["status"] == "running"
    async_result.assert_not_called()


def test_manual_trigger_writes_its_row_before_dispatch_and_worker_updates_it(db_session):
    seen_at_dispatch = []

    def apply_async(**kwargs):
        # A fast worker may start the task before trigger_task returns, so
        # the row must already be committed when the message is published.
        seen_at_dispatch.extend(_rows(db_session, kwargs["task_id"]))
        return SimpleNamespace(id=kwargs["task_id"])

    service = TaskRegistryService()
    service._task_imports["weekly-orphaned-scan-cleanup"] = SimpleNamespace(apply_async=apply_async)
    task_id = service.trigger_task("weekly-orphaned-scan-cleanup", db_session)["task_id"]

    assert [r.status for r in seen_at_dispatch] == ["queued"]

    _start(CLEANUP, task_id, {})
    _finish(CLEANUP, task_id, {}, state="SUCCESS")

    [row] = _rows(db_session, task_id)
    assert (row.status, row.triggered_by) == ("completed", "manual")


def _manual_dispatch_that_raises(db_session, *, worker_started: bool):
    def apply_async(**kwargs):
        if worker_started:
            # The publish landed and a worker picked it up before the
            # producer saw an error (e.g. a lost confirm during retry).
            _start(CLEANUP, kwargs["task_id"], {})
        raise RuntimeError("broker unavailable")

    service = TaskRegistryService()
    service._task_imports["weekly-orphaned-scan-cleanup"] = SimpleNamespace(apply_async=apply_async)
    with pytest.raises(RuntimeError, match="broker unavailable"):
        service.trigger_task("weekly-orphaned-scan-cleanup", db_session)

    db_session.expire_all()
    return db_session.query(TaskExecutionHistory).filter_by(task_name="weekly-orphaned-scan-cleanup").one()


def test_manual_trigger_marks_row_failed_when_dispatch_fails(db_session):
    row = _manual_dispatch_that_raises(db_session, worker_started=False)

    assert row.status == "failed"
    assert "broker unavailable" in row.error_message


def test_dispatch_error_does_not_overwrite_a_run_the_worker_already_started(db_session):
    row = _manual_dispatch_that_raises(db_session, worker_started=True)

    assert row.status == "running"


def _open_cleanup_run(db_session, *, age, status="running"):
    task_id = str(uuid4())
    db_session.add(TaskExecutionHistory(
        task_name="monthly-price-data-cleanup",
        task_function="app.tasks.cache_tasks.cleanup_old_price_data",
        task_id=task_id,
        status=status,
        started_at=datetime.now(timezone.utc) - age,
        triggered_by="auto" if status == "running" else "manual",
    ))
    db_session.commit()
    return task_id


class _Broker:
    def __init__(self, queues=None, unacked=None, error=None):
        self.queues, self.unacked, self.error = queues or {}, unacked or [], error

    def lrange(self, name, start, end):
        if self.error:
            raise self.error
        return self.queues.get(name, [])[start:end + 1]

    def hscan(self, name, cursor, count):
        assert name == "unacked"
        page = self.unacked[cursor:cursor + count]
        next_cursor = cursor + count if cursor + count < len(self.unacked) else 0
        return next_cursor, {str(cursor + i).encode(): raw for i, raw in enumerate(page)}

    def close(self):
        pass


def _message(task_id, **headers):
    return json.dumps({"headers": {"id": task_id, **headers}, "body": "e30="}).encode()


def _lost_run_with_broker(db_session, broker, *, status="queued", message_for=None):
    task_id = _open_cleanup_run(db_session, age=timedelta(hours=26), status=status)
    if message_for:
        message_for(broker, task_id)
    with patch("redis.from_url", return_value=broker) as read_broker:
        run = _last_run(db_session, "monthly-price-data-cleanup")
    return run, read_broker


def _in_queue(broker, task_id):
    broker.queues.setdefault("celery", []).append(_message(task_id))


def test_uncached_broker_search_neither_reads_nor_writes_the_cache():
    # The pipeline-chain heartbeat (#605): a cached "not found" from before a
    # stage was published would read a live chain as missing.
    from app.services import task_registry_service as registry

    task_id = str(uuid4())
    registry._broker_checks[task_id] = (datetime.now(timezone.utc).timestamp(), False)
    broker = _Broker()
    _in_queue(broker, task_id)
    try:
        with patch("redis.from_url", return_value=broker):
            assert registry.still_in_broker([task_id], cache=False) == {task_id: True}
        assert registry._broker_checks[task_id][1] is False
    finally:
        registry._broker_checks.pop(task_id, None)


def test_lost_run_that_never_started_and_is_still_queued_reports_it(db_session):
    run, read_broker = _lost_run_with_broker(db_session, _Broker(), message_for=_in_queue)

    assert (run["status"], run["lost_from"], run["still_queued"]) == ("lost", "queued", True)
    # A hung broker must not hang the polled endpoint.
    assert read_broker.call_args.kwargs["socket_timeout"] == 2
    assert read_broker.call_args.kwargs["socket_connect_timeout"] == 2


def test_lost_run_prefetched_by_a_dead_worker_counts_as_still_queued(db_session):
    # An unacked entry is [message, exchange, routing_key]; a live worker restores it.
    def _unacked(broker, task_id):
        broker.unacked.append(json.dumps([json.loads(_message(task_id)), "", "celery"]).encode())

    run, _ = _lost_run_with_broker(db_session, _Broker(), message_for=_unacked)

    assert run["still_queued"] is True


def test_a_child_message_does_not_count_as_its_parent_still_queued(db_session):
    def _child(broker, task_id):
        broker.queues["celery"] = [_message(str(uuid4()), parent_id=task_id, root_id=task_id)]

    run, _ = _lost_run_with_broker(db_session, _Broker(), message_for=_child)

    assert run["still_queued"] is False


def test_broker_check_pages_past_the_first_batch(db_session):
    def _deep(broker, task_id):
        broker.queues["celery"] = [_message(str(uuid4())) for _ in range(450)] + [_message(task_id)]

    run, _ = _lost_run_with_broker(db_session, _Broker(), message_for=_deep)

    assert run["still_queued"] is True


def test_broker_check_reports_unknown_when_the_backlog_outruns_its_budget(db_session):
    def _backlog(broker, _task_id):
        broker.queues["celery"] = [_message(str(uuid4())) for _ in range(450)]

    clock = iter(range(0, 1000, 5))  # every deadline check is 5s later
    with patch("app.services.task_registry_service.time.monotonic", side_effect=lambda: next(clock)):
        run, _ = _lost_run_with_broker(db_session, _Broker(), message_for=_backlog)

    assert (run["lost_from"], run["still_queued"]) == ("queued", None)


def test_several_lost_runs_share_one_broker_pass(db_session):
    # Each run used to open its own client with its own budget, so N lost runs
    # (or a broker outage) multiplied the request time by N.
    waiting = _open_cleanup_run(db_session, age=timedelta(hours=26), status="queued")
    gone = str(uuid4())
    db_session.add(TaskExecutionHistory(
        task_name="weekly-orphaned-scan-cleanup", task_function=CLEANUP, task_id=gone,
        status="queued", started_at=datetime.now(timezone.utc) - timedelta(hours=26), triggered_by="manual",
    ))
    db_session.commit()
    broker = _Broker()
    _in_queue(broker, waiting)

    with patch("redis.from_url", return_value=broker) as read_broker:
        tasks = {t["name"]: t["last_run"] for t in TaskRegistryService().get_all_scheduled_tasks(db_session)}

    read_broker.assert_called_once()
    assert tasks["monthly-price-data-cleanup"]["still_queued"] is True
    assert tasks["weekly-orphaned-scan-cleanup"]["still_queued"] is False


def test_a_message_restored_from_unacked_mid_pass_is_still_found(db_session):
    # kombu restores an unacked message into its queue after that queue was
    # read but before the unacked read: one pass misses it in both places.
    task_id = _open_cleanup_run(db_session, age=timedelta(hours=26), status="queued")

    class _Restoring(_Broker):
        def hscan(self, name, cursor, count):
            if not self.queues.get("celery"):
                self.queues["celery"] = [_message(task_id)]  # restored, after the queue read
            return super().hscan(name, cursor, count)

    with patch("redis.from_url", return_value=_Restoring()):
        run = _last_run(db_session, "monthly-price-data-cleanup")

    assert run["still_queued"] is True


def test_a_run_picked_up_during_the_broker_check_does_not_enable_run(db_session):
    # The worker acks the message (gone from the broker) and flips the row to
    # running after the history read; "absent" must not mean "safe to rerun".
    task_id = _open_cleanup_run(db_session, age=timedelta(hours=26), status="queued")

    class _PickedUp(_Broker):
        def lrange(self, name, start, end):
            row = db_session.query(TaskExecutionHistory).filter_by(task_id=task_id).one()
            if row.status == "queued":
                row.status = "running"
                db_session.commit()
            return super().lrange(name, start, end)

    with patch("redis.from_url", return_value=_PickedUp()):
        run = _last_run(db_session, "monthly-price-data-cleanup")

    assert (run["status"], run["lost_from"], run["still_queued"]) == ("lost", "queued", None)


def test_broker_check_is_cached_between_polls(db_session):
    task_id = _open_cleanup_run(db_session, age=timedelta(hours=26), status="queued")
    broker = _Broker()
    _in_queue(broker, task_id)

    with patch("redis.from_url", return_value=broker) as read_broker:
        _last_run(db_session, "monthly-price-data-cleanup")
        _last_run(db_session, "monthly-price-data-cleanup")

    read_broker.assert_called_once()


def test_lost_run_that_never_started_and_whose_message_is_gone_can_rerun(db_session):
    run, _ = _lost_run_with_broker(db_session, _Broker())

    assert (run["lost_from"], run["still_queued"]) == ("queued", False)


def test_lost_run_reports_unknown_queue_state_when_the_broker_is_unreadable(db_session):
    run, _ = _lost_run_with_broker(db_session, _Broker(error=ConnectionError("down")))

    assert (run["lost_from"], run["still_queued"]) == ("queued", None)


def test_lost_run_that_started_does_not_read_the_broker(db_session):
    run, read_broker = _lost_run_with_broker(db_session, _Broker(), status="running")

    assert (run["status"], run["lost_from"], run["still_queued"]) == ("lost", "running", None)
    read_broker.assert_not_called()


def test_run_killed_at_the_hard_time_limit_is_still_reconcilable_not_lost(db_session):
    # Celery stores FAILURE when it kills a task at task_time_limit (24h), so
    # the run must stay open a little past it for reconcile to record that.
    _open_cleanup_run(db_session, age=timedelta(hours=24, minutes=30))

    assert _last_run(db_session, "monthly-price-data-cleanup")["status"] == "running"

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        async_result.return_value = SimpleNamespace(state="FAILURE", result=TimeLimitExceeded(86400))
        assert TaskRegistryService().reconcile_open_runs(db_session) == 1


def test_run_open_past_the_hard_time_limit_and_margin_reads_as_lost(db_session):
    _open_cleanup_run(db_session, age=timedelta(hours=26))

    assert _last_run(db_session, "monthly-price-data-cleanup")["status"] == "lost"
    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        assert TaskRegistryService().reconcile_open_runs(db_session) == 0
    async_result.assert_not_called()


def test_status_endpoint_falls_back_to_celery_when_the_ledger_lost_the_result(db_session):
    # The Options page needs result.status; a NaN-bearing result is stored
    # without its summary, so the status endpoint must still return it.
    task_id = str(uuid4())
    _start(OPTIONS, task_id, {"market": "US"})
    _finish(OPTIONS, task_id, {"market": "US"}, state="SUCCESS",
            retval={"status": "published", "coverage": float("nan")})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        async_result.return_value = SimpleNamespace(
            state="SUCCESS", result={"status": "published"}, info=None,
        )
        status = TaskRegistryService().get_task_status("daily-us-options-analytics", task_id, db_session)

    assert status["status"] == "completed"
    assert status["result"] == {"status": "published"}


@pytest.mark.parametrize(
    "celery_lookup",
    [
        # Past result_expires Celery reports an unknown id as PENDING.
        lambda _task_id: SimpleNamespace(state="PENDING", result=None, info=None),
        lambda _task_id: (_ for _ in ()).throw(ConnectionError("redis down")),
    ],
    ids=["result-expired", "backend-down"],
)
def test_completed_run_without_a_summary_stays_completed(db_session, celery_lookup):
    task_id = str(uuid4())
    _start(OPTIONS, task_id, {"market": "US"})
    _finish(OPTIONS, task_id, {"market": "US"}, state="SUCCESS", retval={"coverage": float("nan")})

    with patch("app.services.task_registry_service.AsyncResult", side_effect=celery_lookup):
        status = TaskRegistryService().get_task_status("daily-us-options-analytics", task_id, db_session)

    assert (status["status"], status["result"]) == ("completed", None)
    assert TaskStatusResponse(**status).celery_state == "SUCCESS"


def test_celery_pending_does_not_rewind_a_run_the_ledger_has_seen_start(db_session):
    # task_prerun fires before Celery stores STARTED, so Celery still says
    # PENDING for a moment after the ledger has recorded the start.
    task_id = str(uuid4())
    _start(CLEANUP, task_id, {})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        async_result.return_value = SimpleNamespace(state="PENDING", result=None, info=None)
        status = TaskRegistryService().get_task_status("weekly-orphaned-scan-cleanup", task_id, db_session)

    assert status["status"] == "running"


def test_status_fallback_write_survives_a_nan_result(db_session):
    task_id = str(uuid4())
    _start(CLEANUP, task_id, {})

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        async_result.return_value = SimpleNamespace(state="SUCCESS", result={"ratio": float("nan")}, info=None)
        status = TaskRegistryService().get_task_status("weekly-orphaned-scan-cleanup", task_id, db_session)

    assert status["status"] == "completed"
    [row] = _rows(db_session, task_id)
    assert (row.status, row.result_summary) == ("completed", None)


def test_status_endpoint_reads_finished_runs_from_the_ledger(db_session):
    task_id = str(uuid4())
    _start(CLEANUP, task_id, {})
    record_task_run(task_id, CLEANUP, {}, "cancelled")

    with patch("app.services.task_registry_service.AsyncResult") as async_result:
        status = TaskRegistryService().get_task_status("weekly-orphaned-scan-cleanup", task_id, db_session)

    assert status["status"] == "cancelled"
    async_result.assert_not_called()
    # The API serializes through this schema; a missing field would be a 500.
    assert TaskStatusResponse(**status).celery_state == "REVOKED"


def test_operations_queue_removal_marks_the_run_cancelled(db_session):
    from app.services.operations_job_service import OperationsJobService, _JobRecord
    from tests.unit.test_operations_job_service import _FakeBroker, _queued_message

    service = TaskRegistryService()
    service._task_imports["weekly-orphaned-scan-cleanup"] = SimpleNamespace(
        apply_async=lambda **kwargs: SimpleNamespace(id=kwargs["task_id"])
    )
    task_id = service.trigger_task("weekly-orphaned-scan-cleanup", db_session)["task_id"]

    ops = OperationsJobService()
    raw = _queued_message(task_id=task_id, task_name=CLEANUP)
    broker = _FakeBroker({"celery": [raw]})
    ops._broker = lambda: broker
    ops._record_cancel_action = lambda *args, **kwargs: None
    ops._find_job_record = lambda _db, _task_id: _JobRecord(
        task_id=task_id,
        task_name=CLEANUP,
        queue="celery", market=None, state="queued", worker=None, age_seconds=None,
        wait_reason=None, heartbeat_lag_seconds=None,
        cancel_strategy="revoke_and_remove_from_queue", kwargs={},
    )
    with patch.object(ops, "broker_queue_names", return_value=["celery"]), patch(
        "app.services.operations_job_service.celery_app.control.revoke"
    ):
        assert ops.cancel_job(MagicMock(), task_id)["status"] == "accepted"

    [row] = _rows(db_session, task_id)
    assert row.status == "cancelled"


@pytest.mark.parametrize("function, entry", [
    ("app.tasks.cache_tasks.weekly_full_refresh", "weekly-full-refresh"),
    ("app.tasks.fundamentals_tasks.refresh_all_fundamentals", "weekly-fundamental-refresh"),
])
def test_a_later_success_in_one_market_does_not_hide_a_failure_in_another(db_session, function, entry):
    # #591: Beat fans these out per market; one shared entry showed whichever
    # market started last, so a later HK success hid a failed US run.
    us, hk = str(uuid4()), str(uuid4())
    _start(function, us, {"market": "US"})
    _finish(function, us, {"market": "US"}, state="FAILURE", retval=RuntimeError("provider down"))
    _start(function, hk, {"market": "HK"})
    _finish(function, hk, {"market": "HK"}, state="SUCCESS", retval={"status": "ok"})

    assert _last_run(db_session, f"{entry}-us")["status"] == "failed"
    assert _last_run(db_session, f"{entry}-hk")["status"] == "completed"


def test_every_weekly_beat_entry_has_a_matching_registry_entry():
    from app.celery_app import _build_cache_warmup_beat_schedule
    from app.services.task_registry_service import SCHEDULED_TASKS
    from app.tasks.market_queues import SUPPORTED_MARKETS

    beat = _build_cache_warmup_beat_schedule(list(SUPPORTED_MARKETS))
    weekly = {name: e for name, e in beat.items() if name.startswith(("weekly-full-refresh-", "weekly-fundamental-refresh-"))}

    assert len(weekly) == 2 * len(SUPPORTED_MARKETS)
    for name, beat_entry in weekly.items():
        registry = SCHEDULED_TASKS[name]
        assert registry["task_function"] == beat_entry["task"]
        assert registry["manual_dispatch_kwargs"] == beat_entry["kwargs"]
        assert registry["manual_dispatch_options"]["queue"] == beat_entry["options"]["queue"]
    # No shared fundamentals entry left to collect every market's runs.
    assert "weekly-fundamental-refresh" not in SCHEDULED_TASKS


def test_unscoped_full_refresh_keeps_its_own_entry_without_collecting_market_runs(db_session):
    # Only weekly_full_refresh(market=None) cleans orphaned cache keys and old
    # telemetry, and Beat never runs it unscoped, so its manual entry stays.
    # Market runs must still land on their own entries, not this one.
    from app.services.task_registry_service import SCHEDULED_TASKS

    function = "app.tasks.cache_tasks.weekly_full_refresh"
    assert not SCHEDULED_TASKS["weekly-full-refresh"].get("manual_dispatch_kwargs")

    unscoped, us = str(uuid4()), str(uuid4())
    _start(function, unscoped, {})
    _finish(function, unscoped, {}, state="SUCCESS", retval={"status": "ok"})
    _start(function, us, {"market": "US"})
    _finish(function, us, {"market": "US"}, state="FAILURE", retval=RuntimeError("provider down"))

    assert _last_run(db_session, "weekly-full-refresh")["task_id"] == unscoped
    assert _last_run(db_session, "weekly-full-refresh-us")["status"] == "failed"


def test_weekly_schedule_text_matches_the_beat_settings():
    from app.services.task_registry_service import SCHEDULED_TASKS

    with patch("app.services.task_registry_service.settings") as s:
        s.fundamental_refresh_day, s.fundamental_refresh_hour = 6, 8
        s.cache_weekly_day, s.cache_weekly_hour = 0, 14
        from app.services.task_registry_service import _weekly_refresh_task_definitions
        entries = _weekly_refresh_task_definitions()

    assert entries["weekly-fundamental-refresh-us"]["schedule_description"] == "Saturday 8:00 AM ET (US)"
    assert entries["weekly-full-refresh-hk"]["schedule_description"] == "Sunday 2:00 PM ET (HK)"
    assert SCHEDULED_TASKS["weekly-full-refresh-us"]["display_name"] == "Weekly Full Refresh (US)"


def test_scheduled_tasks_group_into_one_job_per_type(db_session):
    from collections import defaultdict

    from app.services.task_registry_service import TaskRegistryService
    from app.tasks.market_queues import SUPPORTED_MARKETS

    groups = defaultdict(list)
    for task in TaskRegistryService().get_all_scheduled_tasks(db_session):
        groups[(task["job_type"], task["job_name"], task["advanced"])].append(task["market"])

    per_market = sorted(SUPPORTED_MARKETS)
    assert {key: sorted(markets, key=str) for key, markets in groups.items()} == {
        ("daily-market-pipeline", "Daily Market Pipeline", False): per_market,
        ("weekly-full-refresh-market", "Weekly Full Refresh", False): per_market,
        ("weekly-fundamental-refresh", "Weekly Fundamental Refresh", False): per_market,
        ("weekly-universe-refresh", "Weekly Universe Refresh", False): per_market,
        ("social-signal-refresh", "Refresh Social Signals", False): [None],
        ("weekly-orphaned-scan-cleanup", "Weekly Scan Cleanup", False): [None],
        ("daily-us-options-analytics", "Daily US Options Analytics", False): ["US"],
        ("weekly-full-refresh", "Weekly Full Refresh (all markets)", True): [None],
        ("weekly-global-maintenance", "Weekly Cache & Telemetry Cleanup", False): [None],
        ("monthly-price-data-cleanup", "Monthly Price Data Cleanup", True): [None],
    }
