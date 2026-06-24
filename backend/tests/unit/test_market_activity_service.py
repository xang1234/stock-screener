"""Unit tests for runtime market activity aggregation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base

import app.models.app_settings  # noqa: F401


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _bootstrap_status(*, required, primary="US", enabled=None, state="ready"):
    return SimpleNamespace(
        bootstrap_required=required,
        empty_system=False,
        primary_market=primary,
        enabled_markets=enabled or [primary],
        bootstrap_state=state,
        supported_markets=["US", "HK", "JP", "TW"],
    )


class _FakeLock:
    def __init__(self, tasks=None):
        self._tasks = tasks or {}

    def get_current_task(self, market=None):
        return self._tasks.get(market)


def test_runtime_activity_status_reports_primary_bootstrap_progress(db_session, monkeypatch):
    from app.services import market_activity_service as module

    module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="bootstrap",
        task_name="smart_refresh_cache",
        task_id="task-us",
        message="Refreshing prices",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["US", "HK"], state="running"),
    )
    monkeypatch.setattr(
        module,
        "get_data_fetch_lock",
        lambda: _FakeLock(
            {
                "US": {
                    "task_id": "task-us",
                    "current": 50,
                    "total": 100,
                    "progress": 50.0,
                }
            }
        ),
    )

    payload = module.get_runtime_activity_status(db_session)

    assert payload["bootstrap"]["state"] == "running"
    assert payload["bootstrap"]["app_ready"] is False
    assert payload["bootstrap"]["current_stage"] == "Price Refresh"
    assert payload["bootstrap"]["progress_mode"] == "determinate"
    assert payload["bootstrap"]["percent"] == pytest.approx(12.5)
    assert payload["summary"]["active_market_count"] == 2
    assert payload["summary"]["active_markets"] == ["US", "HK"]
    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "running"
    assert us_market["progress_mode"] == "determinate"
    assert us_market["current"] == 50
    assert us_market["total"] == 100
    assert us_market["percent"] == 50.0


def test_runtime_activity_status_exposes_bootstrap_run_task_manifest(db_session, monkeypatch):
    from app.services import market_activity_service as module

    module.save_runtime_bootstrap_run(
        db_session,
        primary_market="US",
        enabled_markets=["US", "HK", "TW"],
        primary_task_id="primary-task-123",
        market_task_ids={
            "US": "primary-task-123",
            "HK": "background-task-2",
            "TW": "background-task-3",
        },
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["US", "HK", "TW"], state="running"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    assert payload["bootstrap"]["queue_state"] == "queued"
    assert payload["bootstrap"]["task_id"] == "primary-task-123"
    assert payload["bootstrap"]["market_task_ids"] == {
        "US": "primary-task-123",
        "HK": "background-task-2",
        "TW": "background-task-3",
    }
    hk_market = next(item for item in payload["markets"] if item["market"] == "HK")
    assert hk_market["task_id"] == "background-task-2"


def test_runtime_activity_status_marks_running_stage_without_real_percent_as_indeterminate(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="fundamentals",
        lifecycle="bootstrap",
        task_name="refresh_all_fundamentals",
        task_id="task-us",
        message="Refreshing fundamentals",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["US"], state="running"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    assert payload["bootstrap"]["current_stage"] == "Fundamentals Refresh"
    assert payload["bootstrap"]["progress_mode"] == "indeterminate"
    assert payload["bootstrap"]["percent"] is None
    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "running"
    assert us_market["progress_mode"] == "indeterminate"
    assert us_market["percent"] is None


def test_persisted_runtime_activity_derives_response_fields():
    from app.services.runtime_activity_contract import PersistedRuntimeActivity

    record = PersistedRuntimeActivity.from_payload(
        {
            "market": "jp",
            "lifecycle": "bootstrap",
            "stage_key": "prices",
            "status": "running",
            "percent": None,
            "current": 25,
            "total": 100,
            "message": "Waiting on provider",
            "task_name": "smart_refresh_cache",
            "task_id": "task-jp",
            "updated_at": "2026-06-09T01:02:03+00:00",
        }
    ).to_record()

    assert record.market == "JP"
    assert record.stage_label == "Price Refresh"
    assert record.progress_mode == "determinate"
    assert record.percent == 25.0
    assert record.current == 25
    assert record.total == 100
    assert record.message == "Waiting on provider"


def test_persisted_runtime_activity_rejects_missing_persisted_fields():
    from app.services.runtime_activity_contract import PersistedRuntimeActivity

    with pytest.raises(ValueError, match="missing required runtime activity fields"):
        PersistedRuntimeActivity.from_payload(
            {
                "market": "US",
                "stage_key": "prices",
                "status": "running",
                "progress_mode": "determinate",
                "percent": 50,
            }
        )

    record = PersistedRuntimeActivity.from_payload(
        {
            "market": "US",
            "lifecycle": "bootstrap",
            "stage_key": "prices",
            "stage_label": "Stale Label",
            "status": "running",
            "progress_mode": "mostly",
            "percent": 50,
            "current": 50,
            "total": 100,
            "message": "Refreshing prices",
            "task_name": "smart_refresh_cache",
            "task_id": "task-us",
            "updated_at": "2026-06-09T01:02:03+00:00",
        }
    ).to_record()
    assert record.stage_label == "Price Refresh"
    assert record.progress_mode == "determinate"


def test_market_activity_persists_only_canonical_state_fields(db_session):
    from app.models.app_settings import AppSetting
    from app.services import market_activity_service as module
    from app.services.market_activity_service import _activity_key

    returned = module.mark_market_activity_started(
        db_session,
        market="JP",
        stage_key="prices",
        lifecycle="bootstrap",
        task_name="smart_refresh_cache",
        task_id="task-jp",
        current=25,
        total=100,
        message="Refreshing prices",
    )

    setting = (
        db_session.query(AppSetting)
        .filter(AppSetting.key == _activity_key("JP"))
        .one()
    )
    persisted = json.loads(setting.value)

    assert returned["stage_label"] == "Price Refresh"
    assert returned["progress_mode"] == "determinate"
    assert "stage_label" not in persisted
    assert "progress_mode" not in persisted


def test_mark_market_activity_progress_updates_running_record_with_determinate_progress(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="fundamentals",
        lifecycle="bootstrap",
        task_name="refresh_all_fundamentals",
        task_id="task-us",
        message="Refreshing fundamentals",
    )
    module.mark_market_activity_progress(
        db_session,
        market="US",
        stage_key="fundamentals",
        task_name="refresh_all_fundamentals",
        task_id="task-us",
        current=25,
        total=100,
        percent=25.0,
        message="Refreshing fundamentals",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["US"], state="running"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "running"
    assert us_market["progress_mode"] == "determinate"
    assert us_market["current"] == 25
    assert us_market["total"] == 100
    assert us_market["percent"] == 25.0


def test_runtime_activity_counts_without_percent_are_still_determinate(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="fundamentals",
        lifecycle="bootstrap",
        task_name="refresh_all_fundamentals",
        task_id="task-us",
        message="Refreshing fundamentals",
    )
    module.mark_market_activity_progress(
        db_session,
        market="US",
        stage_key="fundamentals",
        task_name="refresh_all_fundamentals",
        task_id="task-us",
        current=25,
        total=100,
        percent=None,
        message="Refreshing fundamentals",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["US"], state="running"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["progress_mode"] == "determinate"
    assert us_market["percent"] == 25.0
    assert us_market["current"] == 25
    assert us_market["total"] == 100


def test_failed_errback_does_not_overwrite_specific_failure_message(db_session):
    from app.services import market_activity_service as module

    module.mark_market_activity_failed(
        db_session,
        market="HK",
        stage_key="scan",
        lifecycle="bootstrap",
        task_name=None,
        task_id=None,
        message="Bootstrap scan did not publish",
    )

    result = module.mark_current_market_activity_failed(
        db_session,
        market="HK",
        lifecycle="bootstrap",
        message="Bootstrap failed",
    )

    assert result["message"] == "Bootstrap scan did not publish"


def test_current_failure_does_not_inherit_owner_from_different_lifecycle(db_session):
    from app.services import market_activity_service as module

    module.mark_market_activity_started(
        db_session,
        market="HK",
        stage_key="prices",
        lifecycle="daily_refresh",
        task_name="smart_refresh_cache",
        task_id="daily-task",
        message="Refreshing daily prices",
    )

    result = module.mark_current_market_activity_failed(
        db_session,
        market="HK",
        lifecycle="bootstrap",
        message="Bootstrap failed",
    )

    assert result["status"] == "running"
    assert result["lifecycle"] == "daily_refresh"
    assert result["task_id"] == "daily-task"
    assert result["message"] == "Refreshing daily prices"


def test_runtime_activity_prefers_persisted_progress_over_heartbeat_overlay(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="bootstrap",
        task_name="smart_refresh_cache",
        task_id="task-us",
        message="Refreshing prices",
    )
    module.mark_market_activity_progress(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="bootstrap",
        task_name="smart_refresh_cache",
        task_id="task-us",
        current=40,
        total=100,
        percent=40.0,
        message="Batch 1/2 · refreshing prices",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["US"], state="running"),
    )
    monkeypatch.setattr(
        module,
        "get_data_fetch_lock",
        lambda: _FakeLock(
            {
                "US": {
                    "task_id": "task-us",
                    "current": 10,
                    "total": 100,
                    "progress": 10.0,
                }
            }
        ),
    )

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["progress_mode"] == "determinate"
    assert us_market["current"] == 40
    assert us_market["total"] == 100
    assert us_market["percent"] == 40.0
    assert us_market["message"] == "Batch 1/2 · refreshing prices"


def test_mark_market_activity_progress_does_not_overwrite_completed_record(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_completed(
        db_session,
        market="US",
        stage_key="fundamentals",
        lifecycle="bootstrap",
        task_name="refresh_all_fundamentals",
        task_id="task-us",
        current=100,
        total=100,
        message="Fundamentals refresh completed",
    )
    module.mark_market_activity_progress(
        db_session,
        market="US",
        stage_key="fundamentals",
        task_name="refresh_all_fundamentals",
        task_id="task-us",
        current=99,
        total=100,
        percent=99.0,
        message="Refreshing fundamentals",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US"], state="ready"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "completed"
    assert us_market["current"] == 100
    assert us_market["percent"] == 100.0


def test_mark_market_activity_progress_does_not_overwrite_newer_running_task(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="fundamentals",
        lifecycle="bootstrap",
        task_name="refresh_all_fundamentals",
        task_id="new-task",
        message="Refreshing fundamentals",
    )
    module.mark_market_activity_progress(
        db_session,
        market="US",
        stage_key="fundamentals",
        task_name="refresh_all_fundamentals",
        task_id="stale-task",
        current=25,
        total=100,
        percent=25.0,
        message="Refreshing fundamentals",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["US"], state="running"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "running"
    assert us_market["task_id"] == "new-task"
    assert us_market["current"] is None
    assert us_market["percent"] is None


def test_mark_market_activity_completed_still_advances_same_task_running_record(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="fundamentals",
        lifecycle="bootstrap",
        task_name="refresh_all_fundamentals",
        task_id="task-us",
        current=25,
        total=100,
        percent=25.0,
        message="Refreshing fundamentals",
    )
    module.mark_market_activity_completed(
        db_session,
        market="US",
        stage_key="fundamentals",
        lifecycle="bootstrap",
        task_name="refresh_all_fundamentals",
        task_id="task-us",
        current=100,
        total=100,
        message="Fundamentals refresh completed",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US"], state="ready"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "completed"
    assert us_market["task_id"] == "task-us"
    assert us_market["current"] == 100
    assert us_market["percent"] == 100.0


def test_mark_market_activity_started_allows_new_same_stage_run_after_completion(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_completed(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="daily_refresh",
        task_name="smart_refresh_cache",
        task_id="day-1-task",
        current=100,
        total=100,
        message="Price refresh completed",
    )
    module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="daily_refresh",
        task_name="smart_refresh_cache",
        task_id="day-2-task",
        current=5,
        total=100,
        percent=5.0,
        message="Refreshing prices",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US"], state="ready"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "running"
    assert us_market["task_id"] == "day-2-task"
    assert us_market["stage_key"] == "prices"
    assert us_market["current"] == 5
    assert us_market["percent"] == 5.0


def test_mark_market_activity_started_allows_next_stage_after_completion(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_completed(
        db_session,
        market="US",
        stage_key="universe",
        lifecycle="bootstrap",
        task_name="refresh_stock_universe",
        task_id="universe-task",
        current=100,
        total=100,
        message="Universe refresh completed",
    )
    module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="bootstrap",
        task_name="smart_refresh_cache",
        task_id="prices-task",
        current=10,
        total=200,
        percent=5.0,
        message="Refreshing prices",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["US"], state="running"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "running"
    assert us_market["task_id"] == "prices-task"
    assert us_market["stage_key"] == "prices"
    assert us_market["current"] == 10
    assert us_market["percent"] == 5.0


def test_failed_activity_is_sticky_against_later_stage_updates(db_session, monkeypatch):
    from app.services import market_activity_service as module

    module.mark_market_activity_failed(
        db_session,
        market="HK",
        stage_key="prices",
        lifecycle="bootstrap",
        task_name="smart_refresh_cache",
        task_id="prices-task",
        message="Price refresh failed",
    )
    module.mark_market_activity_started(
        db_session,
        market="HK",
        stage_key="groups",
        lifecycle="bootstrap",
        task_name="calculate_daily_group_rankings",
        task_id="groups-task",
        message="Calculating groups",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["HK"], primary="HK", state="failed"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    hk_market = next(item for item in payload["markets"] if item["market"] == "HK")
    assert hk_market["status"] == "failed"
    assert hk_market["stage_key"] == "prices"
    assert hk_market["message"] == "Price refresh failed"


def test_failed_group_activity_is_replaced_when_bootstrap_scan_starts(db_session, monkeypatch):
    from app.services import market_activity_service as module

    module.mark_market_activity_failed(
        db_session,
        market="HK",
        stage_key="groups",
        lifecycle="bootstrap",
        task_name="calculate_daily_group_rankings_with_gapfill",
        task_id="groups-task",
        message="Daily group ranking failed: Cache warmup not complete",
    )
    module.mark_market_activity_started(
        db_session,
        market="HK",
        stage_key="scan",
        lifecycle="bootstrap",
        task_name="build_daily_snapshot",
        task_id="scan-task",
        current=12,
        total=100,
        message="Running market scan",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["HK"], primary="HK", state="running"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    hk_market = next(item for item in payload["markets"] if item["market"] == "HK")
    assert hk_market["status"] == "running"
    assert hk_market["stage_key"] == "scan"
    assert hk_market["stage_label"] == "Scan"
    assert hk_market["message"] == "Running market scan"
    assert hk_market["percent"] == 12.0


@pytest.mark.parametrize("stage_key", ["universe", "prices"])
def test_failed_hard_activity_is_not_replaced_when_bootstrap_scan_starts(
    db_session,
    monkeypatch,
    stage_key,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_failed(
        db_session,
        market="HK",
        stage_key=stage_key,
        lifecycle="bootstrap",
        task_name=f"{stage_key}_task",
        task_id=f"{stage_key}-task",
        message=f"{stage_key.title()} failed",
    )
    module.mark_market_activity_started(
        db_session,
        market="HK",
        stage_key="scan",
        lifecycle="bootstrap",
        task_name="build_daily_snapshot",
        task_id="scan-task",
        current=12,
        total=100,
        message="Running market scan",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["HK"], primary="HK", state="failed"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    hk_market = next(item for item in payload["markets"] if item["market"] == "HK")
    assert hk_market["status"] == "failed"
    assert hk_market["stage_key"] == stage_key
    assert hk_market["message"] == f"{stage_key.title()} failed"


def test_failed_activity_can_replace_completed_record(db_session, monkeypatch):
    from app.services import market_activity_service as module

    module.mark_market_activity_completed(
        db_session,
        market="HK",
        stage_key="scan",
        lifecycle="bootstrap",
        task_name="build_daily_snapshot",
        task_id="scan-task",
        message="Market scan ready",
    )
    module.mark_market_activity_failed(
        db_session,
        market="HK",
        stage_key="scan",
        lifecycle="bootstrap",
        task_name="runtime_bootstrap",
        task_id=None,
        message="Bootstrap scan did not publish",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["HK"], primary="HK", state="failed"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    hk_market = next(item for item in payload["markets"] if item["market"] == "HK")
    assert hk_market["status"] == "failed"
    assert hk_market["stage_key"] == "scan"
    assert hk_market["message"] == "Bootstrap scan did not publish"


def test_runtime_activity_status_reports_active_background_bootstrap_progress_after_primary_ready(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_completed(
        db_session,
        market="US",
        stage_key="snapshot",
        lifecycle="bootstrap",
        task_name="build_daily_snapshot",
        task_id="task-us",
        message="Primary bootstrap complete",
    )
    module.mark_market_activity_started(
        db_session,
        market="HK",
        stage_key="fundamentals",
        lifecycle="bootstrap",
        task_name="refresh_all_fundamentals",
        task_id="task-hk",
        current=1200,
        total=3750,
        message="Refreshing fundamentals",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US", "HK"], state="ready"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    assert payload["bootstrap"]["state"] == "ready"
    assert payload["bootstrap"]["app_ready"] is True
    assert payload["bootstrap"]["progress_mode"] == "determinate"
    assert payload["bootstrap"]["percent"] == pytest.approx(38.67)
    assert payload["bootstrap"]["current"] == 1200
    assert payload["bootstrap"]["total"] == 3750
    assert payload["bootstrap"]["current_stage"] == "Fundamentals Refresh"
    assert payload["bootstrap"]["message"] == "Refreshing fundamentals"
    assert "background" in payload["bootstrap"]["background_warning"].lower()
    assert payload["summary"]["active_market_count"] == 1
    assert payload["summary"]["active_markets"] == ["HK"]
    hk_market = next(item for item in payload["markets"] if item["market"] == "HK")
    assert hk_market["lifecycle"] == "bootstrap"
    assert hk_market["status"] == "running"
    assert hk_market["progress_mode"] == "determinate"


def test_runtime_activity_status_does_not_stage_weight_running_price_refresh_at_100(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_completed(
        db_session,
        market="US",
        stage_key="snapshot",
        lifecycle="bootstrap",
        task_name="build_daily_snapshot",
        task_id="task-us",
        message="Primary bootstrap complete",
    )
    module.mark_market_activity_started(
        db_session,
        market="JP",
        stage_key="prices",
        lifecycle="bootstrap",
        task_name="smart_refresh_cache",
        task_id="task-jp",
        current=3750,
        total=3750,
        message="Refreshing market prices",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US", "JP"], state="ready"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    assert payload["bootstrap"]["state"] == "ready"
    assert payload["bootstrap"]["app_ready"] is True
    assert payload["bootstrap"]["current_stage"] == "Price Refresh"
    assert payload["bootstrap"]["progress_mode"] == "indeterminate"
    assert payload["bootstrap"]["percent"] is None
    assert payload["bootstrap"]["current"] == 3750
    assert payload["bootstrap"]["total"] == 3750
    jp_market = next(item for item in payload["markets"] if item["market"] == "JP")
    assert jp_market["status"] == "running"
    assert jp_market["progress_mode"] == "indeterminate"


def test_runtime_activity_status_ignores_non_bootstrap_secondary_progress_after_primary_ready(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_completed(
        db_session,
        market="US",
        stage_key="snapshot",
        lifecycle="bootstrap",
        task_name="build_daily_snapshot",
        task_id="task-us",
        message="Primary bootstrap complete",
    )
    module.mark_market_activity_started(
        db_session,
        market="HK",
        stage_key="prices",
        lifecycle="daily_refresh",
        task_name="smart_refresh_cache",
        task_id="task-hk",
        current=50,
        total=100,
        message="Refreshing daily prices",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US", "HK"], state="ready"),
    )
    monkeypatch.setattr(
        module,
        "get_data_fetch_lock",
        lambda: _FakeLock(
            {
                "HK": {
                    "task_id": "task-hk",
                    "current": 50,
                    "total": 100,
                    "progress": 50.0,
                }
            }
        ),
    )

    payload = module.get_runtime_activity_status(db_session)

    assert payload["bootstrap"]["state"] == "ready"
    assert payload["bootstrap"]["progress_mode"] == "determinate"
    assert payload["bootstrap"]["percent"] == 100.0
    assert payload["bootstrap"]["current"] is None
    assert payload["bootstrap"]["total"] is None
    assert payload["bootstrap"]["message"] == "Primary market is ready."
    assert payload["bootstrap"]["background_warning"] is None
    assert payload["summary"]["active_markets"] == ["HK"]


def test_runtime_activity_status_uses_bootstrap_queue_copy_for_secondary_markets(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(
            required=True,
            enabled=["US", "HK"],
            primary="US",
            state="running",
        ),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    hk_market = next(item for item in payload["markets"] if item["market"] == "HK")
    assert hk_market["status"] == "queued"
    assert hk_market["message"] == "Bootstrap queued."


def test_mark_market_activity_queued_does_not_overwrite_newer_state_for_same_task(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_started(
        db_session,
        market="HK",
        stage_key="universe",
        lifecycle="bootstrap",
        task_name="runtime_bootstrap",
        task_id="secondary-task-123",
        message="Refreshing official market universe",
    )
    module.mark_market_activity_queued(
        db_session,
        market="HK",
        stage_key="universe",
        lifecycle="bootstrap",
        task_name="runtime_bootstrap",
        task_id="secondary-task-123",
        message="Queued bootstrap for HK",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US", "HK"], state="ready"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    hk_market = next(item for item in payload["markets"] if item["market"] == "HK")
    assert hk_market["status"] == "running"
    assert hk_market["message"] == "Refreshing official market universe"
    assert hk_market["task_id"] == "secondary-task-123"


def test_runtime_activity_status_surfaces_concurrent_market_refreshes(db_session, monkeypatch):
    from app.services import market_activity_service as module

    module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="daily_refresh",
        task_name="smart_refresh_cache",
        task_id="task-us",
    )
    module.mark_market_activity_started(
        db_session,
        market="HK",
        stage_key="prices",
        lifecycle="daily_refresh",
        task_name="smart_refresh_cache",
        task_id="task-hk",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US", "HK"], state="ready"),
    )
    monkeypatch.setattr(
        module,
        "get_data_fetch_lock",
        lambda: _FakeLock(
            {
                "US": {"task_id": "task-us", "current": 20, "total": 100, "progress": 20.0},
                "HK": {"task_id": "task-hk", "current": 80, "total": 100, "progress": 80.0},
            }
        ),
    )

    payload = module.get_runtime_activity_status(db_session)

    assert payload["summary"]["status"] == "active"
    assert payload["summary"]["active_market_count"] == 2
    assert payload["summary"]["active_markets"] == ["US", "HK"]
    assert {
        item["market"]: item["percent"] for item in payload["markets"] if item["status"] == "running"
    } == {"US": 20.0, "HK": 80.0}


def test_runtime_activity_status_surfaces_failed_market_stage(db_session, monkeypatch):
    from app.services import market_activity_service as module

    module.mark_market_activity_failed(
        db_session,
        market="JP",
        stage_key="groups",
        lifecycle="daily_refresh",
        task_name="calculate_daily_group_rankings",
        task_id="task-jp",
        message="Group ranking failed",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["JP"], primary="JP", state="ready"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    assert payload["summary"]["status"] == "warning"
    jp_market = next(item for item in payload["markets"] if item["market"] == "JP")
    assert jp_market["status"] == "failed"
    assert jp_market["stage_label"] == "Group Rankings"
    assert jp_market["message"] == "Group ranking failed"


def test_runtime_activity_supports_scan_stage_progress(db_session, monkeypatch):
    from app.services import market_activity_service as module

    module.mark_market_activity_started(
        db_session,
        market="HK",
        stage_key="scan",
        lifecycle="daily_refresh",
        task_name="build_daily_snapshot",
        task_id="task-hk-scan",
        current=25,
        total=100,
        message="Running market scan",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["HK"], primary="HK", state="ready"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    hk_market = next(item for item in payload["markets"] if item["market"] == "HK")
    assert hk_market["stage_key"] == "scan"
    assert hk_market["stage_label"] == "Scan"
    assert hk_market["progress_mode"] == "determinate"
    assert hk_market["percent"] == 25.0


def test_runtime_activity_status_exposes_bootstrap_stage_metadata(db_session, monkeypatch):
    from app.services import market_activity_service as module

    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=True, enabled=["US"], state="running"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    assert payload["bootstrap"]["stages"] == [
        {"key": "universe", "label": "Universe Refresh"},
        {"key": "prices", "label": "Price Refresh"},
        {"key": "fundamentals", "label": "Fundamentals Refresh"},
        {"key": "breadth", "label": "Breadth Calculation"},
        {"key": "groups", "label": "Group Rankings"},
        {"key": "scan", "label": "Scan"},
    ]


def test_runtime_activity_status_returns_idle_markets_without_activity(db_session, monkeypatch):
    from app.services import market_activity_service as module

    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US", "HK"], state="ready"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    assert payload["summary"]["status"] == "idle"
    assert payload["summary"]["active_market_count"] == 0
    assert [item["status"] for item in payload["markets"]] == ["idle", "idle"]


def test_runtime_activity_status_hides_background_warning_when_secondary_work_is_idle(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    module.mark_market_activity_completed(
        db_session,
        market="US",
        stage_key="snapshot",
        lifecycle="bootstrap",
        task_name="build_daily_snapshot",
        task_id="task-us",
        message="Primary bootstrap complete",
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US", "HK"], state="ready"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())

    payload = module.get_runtime_activity_status(db_session)

    assert payload["bootstrap"]["background_warning"] is None
