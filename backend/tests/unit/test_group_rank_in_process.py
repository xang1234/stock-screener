from datetime import datetime
from unittest.mock import MagicMock

import pytest


def _patch_serialized_lock(monkeypatch):
    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (True, False)
    fake_lock.release.return_value = True
    fake_coordination = MagicMock()
    fake_coordination.acquire_market_workload.return_value = (
        True,
        False,
    )
    fake_coordination.release_market_workload.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: fake_lock,
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_workload_coordination",
        lambda: fake_coordination,
    )


def _patch_calendar_service(monkeypatch, now: datetime):
    fake = MagicMock()
    fake.is_trading_day.return_value = True
    fake.market_now.return_value = now
    fake.last_completed_trading_day.return_value = now.date()
    monkeypatch.setattr(
        "app.tasks.group_rank_tasks.get_market_calendar_service",
        lambda: fake,
    )


def test_orchestrator_rolls_back_before_publishing_failure(monkeypatch):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(
        monkeypatch,
        datetime(2026, 3, 20, 17, 40),
    )
    monkeypatch.setattr(
        module.settings,
        "group_rank_gapfill_enabled",
        True,
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )
    monkeypatch.setattr(
        "app.services.ibd_industry_service.IBDIndustryService.get_all_groups",
        staticmethod(lambda db, *, market=None: ["Software"]),
    )

    fake_service = MagicMock()
    fake_service.find_missing_dates.side_effect = RuntimeError(
        "session failed"
    )
    monkeypatch.setattr(
        module,
        "get_group_rank_service",
        lambda: fake_service,
    )
    rollback_seen = []
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        lambda db, **_kwargs: rollback_seen.append(db.rollback.called),
    )

    result = module.calculate_daily_group_rankings_with_gapfill.run(
        market="US"
    )

    assert "session failed" in result["error"]
    fake_db.rollback.assert_called_once()
    assert rollback_seen == [True]


def test_orchestrator_bypasses_lease_for_inner_daily_call(monkeypatch):
    import app.tasks.group_rank_tasks as module
    import app.tasks.workload_coordination as coordination

    seen_disabled_state: list[bool] = []
    task = MagicMock()
    task.__module__ = "app.tasks.group_rank_tasks"
    task.request = MagicMock()

    def fake_run(market=None, activity_lifecycle=None):
        seen_disabled_state.append(
            coordination._SERIALIZED_MARKET_WORKLOAD_DISABLED.get()
        )
        return {"date": "2026-03-20", "market": market}

    task.run = fake_run
    monkeypatch.setattr(module, "calculate_daily_group_rankings", task)

    result = module._calculate_daily_group_rankings_in_process(
        market="US",
        activity_lifecycle="daily_refresh",
    )

    assert result == {"date": "2026-03-20", "market": "US"}
    assert seen_disabled_state == [True]
    assert (
        coordination._SERIALIZED_MARKET_WORKLOAD_DISABLED.get()
        is False
    )


def test_in_process_daily_call_propagates_transient_without_inner_retry(
    monkeypatch,
):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(
        monkeypatch,
        datetime(2026, 3, 20, 17, 40),
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )

    fake_service = MagicMock()
    fake_service.price_cache.get_warmup_metadata.return_value = {
        "status": "completed",
        "count": 10000,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service.calculate_group_rankings.side_effect = ConnectionError(
        "network down"
    )
    monkeypatch.setattr(
        module,
        "get_group_rank_service",
        lambda: fake_service,
    )
    inner_retry = MagicMock(
        side_effect=AssertionError("inner retry scheduled")
    )
    failed = MagicMock()
    monkeypatch.setattr(
        module.calculate_daily_group_rankings,
        "retry",
        inner_retry,
    )
    monkeypatch.setattr(module, "mark_market_activity_failed", failed)

    with pytest.raises(ConnectionError):
        module._calculate_daily_group_rankings_in_process(
            market="HK",
            activity_lifecycle="daily_refresh",
        )

    fake_db.rollback.assert_called_once()
    inner_retry.assert_not_called()
    failed.assert_not_called()
    assert (
        module._PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS.get()
        is False
    )
