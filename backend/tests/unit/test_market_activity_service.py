"""Unit tests for runtime market activity aggregation."""

from __future__ import annotations

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
    assert payload["bootstrap"]["percent"] == pytest.approx(25.0)
    assert payload["summary"]["active_market_count"] == 2
    assert payload["summary"]["active_markets"] == ["US", "HK"]
    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "running"
    assert us_market["progress_mode"] == "determinate"
    assert us_market["current"] == 50
    assert us_market["total"] == 100
    assert us_market["percent"] == 50.0


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


def test_runtime_activity_status_marks_primary_ready_with_secondary_bootstrap_running(
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
    assert payload["bootstrap"]["percent"] == 100.0
    assert "continues in the background" in payload["bootstrap"]["background_warning"].lower()
    assert payload["summary"]["active_market_count"] == 1
    assert payload["summary"]["active_markets"] == ["HK"]
    hk_market = next(item for item in payload["markets"] if item["market"] == "HK")
    assert hk_market["lifecycle"] == "bootstrap"
    assert hk_market["status"] == "running"
    assert hk_market["progress_mode"] == "indeterminate"


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
