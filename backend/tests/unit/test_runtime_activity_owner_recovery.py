"""Runtime activity owner recovery and stale-row tests."""

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


def _persist_activity_row(db_session, payload):
    from app.models.app_settings import AppSetting
    from app.services.market_activity_service import (
        MARKET_ACTIVITY_KEY_PREFIX,
        RUNTIME_ACTIVITY_CATEGORY,
    )

    db_session.add(
        AppSetting(
            key=f"{MARKET_ACTIVITY_KEY_PREFIX}{payload['market']}",
            value=json.dumps(payload),
            category=RUNTIME_ACTIVITY_CATEGORY,
            description=f"Latest runtime activity state for {payload['market']}",
        )
    )
    db_session.commit()


def _persisted_running_record(
    *,
    stage_key: str,
    task_name: str,
    task_id: str,
    message: str,
):
    from app.services.runtime_activity_contract import (
        PersistedRuntimeActivity,
        RuntimeActivityRecord,
    )

    record = RuntimeActivityRecord.create(
        market="US",
        lifecycle="daily_refresh",
        stage_key=stage_key,
        status="running",
        task_name=task_name,
        task_id=task_id,
        message=message,
        updated_at="2026-06-23T05:00:00+00:00",
    )
    return PersistedRuntimeActivity.from_record(record).to_payload()


def _persisted_running_price_record(*, task_id: str):
    return _persisted_running_record(
        stage_key="prices",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        task_id=task_id,
        message="Refreshing market prices",
    )


def test_stale_running_activity_allows_new_task_owner_to_start(db_session, monkeypatch):
    from app.services import market_activity_service as module

    _persist_activity_row(
        db_session,
        _persisted_running_price_record(task_id="old-task"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())
    monkeypatch.setattr(
        module,
        "_utcnow_iso",
        lambda: "2026-06-23T06:00:00+00:00",
    )

    result = module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="daily_refresh",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        task_id="new-task",
        message="Refreshing market prices",
    )

    assert result["status"] == "running"
    assert result["task_id"] == "new-task"
    assert result["updated_at"] == "2026-06-23T06:00:00+00:00"


def test_stale_running_activity_does_not_override_when_old_owner_is_live(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    _persist_activity_row(
        db_session,
        _persisted_running_price_record(task_id="old-task"),
    )
    monkeypatch.setattr(
        module,
        "get_data_fetch_lock",
        lambda: _FakeLock({"US": {"task_id": "old-task"}}),
    )
    monkeypatch.setattr(
        module,
        "_utcnow_iso",
        lambda: "2026-06-23T06:00:00+00:00",
    )

    result = module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="daily_refresh",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        task_id="new-task",
        message="Refreshing market prices",
    )

    assert result["status"] == "running"
    assert result["task_id"] == "old-task"


def test_stale_us_activity_does_not_override_when_old_owner_has_shared_lock(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    _persist_activity_row(
        db_session,
        _persisted_running_price_record(task_id="old-task"),
    )
    monkeypatch.setattr(
        module,
        "get_data_fetch_lock",
        lambda: _FakeLock({None: {"task_id": "old-task"}}),
    )
    monkeypatch.setattr(
        module,
        "_utcnow_iso",
        lambda: "2026-06-23T06:00:00+00:00",
    )

    result = module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="daily_refresh",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        task_id="new-task",
        message="Refreshing market prices",
    )

    assert result["status"] == "running"
    assert result["task_id"] == "old-task"


def test_runtime_activity_status_uses_shared_lock_progress_for_us_activity(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    _persist_activity_row(
        db_session,
        _persisted_running_price_record(task_id="shared-task"),
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US"], state="ready"),
    )
    monkeypatch.setattr(
        module,
        "get_data_fetch_lock",
        lambda: _FakeLock(
            {
                None: {
                    "task_id": "shared-task",
                    "current": 60,
                    "total": 100,
                    "progress": 60.0,
                    "last_heartbeat": "2026-06-23T05:59:00+00:00",
                }
            }
        ),
    )

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "running"
    assert us_market["current"] == 60
    assert us_market["total"] == 100
    assert us_market["percent"] == 60.0
    assert us_market["updated_at"] == "2026-06-23T05:59:00+00:00"


def test_runtime_activity_status_marks_orphaned_running_row_stale(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    _persist_activity_row(
        db_session,
        _persisted_running_price_record(task_id="old-task"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US"], state="ready"),
    )

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "stale"
    assert "No live data-fetch lock owns task old-task" in us_market["message"]
    assert payload["summary"]["status"] == "warning"
    assert payload["summary"]["active_market_count"] == 0


def test_runtime_activity_status_does_not_mark_workload_stage_stale_without_data_fetch_lock(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    _persist_activity_row(
        db_session,
        _persisted_running_record(
            stage_key="groups",
            task_name="app.tasks.group_rank_tasks.calculate_daily_group_rankings",
            task_id="group-task",
            message="Calculating group rankings",
        ),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US"], state="ready"),
    )

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "running"
    assert us_market["stage_key"] == "groups"
    assert us_market["message"] == "Calculating group rankings"


def test_stale_workload_stage_does_not_allow_new_owner_override_from_data_fetch_lock(
    db_session,
    monkeypatch,
):
    from app.services import market_activity_service as module

    _persist_activity_row(
        db_session,
        _persisted_running_record(
            stage_key="groups",
            task_name="app.tasks.group_rank_tasks.calculate_daily_group_rankings",
            task_id="old-group-task",
            message="Calculating group rankings",
        ),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())
    monkeypatch.setattr(
        module,
        "_utcnow_iso",
        lambda: "2026-06-23T06:00:00+00:00",
    )

    result = module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="groups",
        lifecycle="daily_refresh",
        task_name="app.tasks.group_rank_tasks.calculate_daily_group_rankings",
        task_id="new-group-task",
        message="Calculating group rankings",
    )

    assert result["status"] == "running"
    assert result["task_id"] == "old-group-task"


def test_runtime_activity_status_logs_lock_lookup_failure(
    db_session,
    monkeypatch,
    caplog,
):
    from app.services import market_activity_service as module

    class FailingLock:
        def get_current_task(self, market=None):
            raise RuntimeError("redis unavailable")

    _persist_activity_row(
        db_session,
        _persisted_running_price_record(task_id="old-task"),
    )
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: FailingLock())
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US"], state="ready"),
    )

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "running"
    assert "Runtime activity lock lookup failed" in caplog.text
    assert "old-task" in caplog.text
