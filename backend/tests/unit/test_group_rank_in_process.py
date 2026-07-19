from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock


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


def test_derived_tasks_do_not_invoke_decorated_tasks_in_process():
    backend_root = Path(__file__).resolve().parents[2]
    for relative_path in (
        "app/tasks/breadth_tasks.py",
        "app/tasks/group_rank_tasks.py",
    ):
        source = (backend_root / relative_path).read_text()
        assert "_calculate_daily_breadth_in_process" not in source
        assert "_calculate_daily_group_rankings_in_process" not in source
        assert "_PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS" not in source
        assert "unittest.mock" not in source
        assert ".run(**kwargs)" not in source
