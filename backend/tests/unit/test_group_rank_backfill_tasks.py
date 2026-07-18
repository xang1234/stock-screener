from datetime import datetime
from unittest.mock import MagicMock

import pytest
from celery.exceptions import Retry
from sqlalchemy.exc import OperationalError


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


def _patch_calendar_service(monkeypatch, module, now: datetime):
    fake = MagicMock()
    fake.market_now.return_value = now
    monkeypatch.setattr(module, "get_market_calendar_service", lambda: fake)
    return fake


def _transient_database_error():
    return OperationalError(
        "select 1",
        {},
        Exception("database system is not yet accepting connections"),
    )


def _configure_failing_group_backfill(
    monkeypatch,
    task_name,
    exc,
):
    import app.tasks.group_rank_backfill_tasks as module

    fake_db = MagicMock()
    fake_service = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(
        module,
        "get_group_rank_service",
        lambda: fake_service,
    )
    _patch_serialized_lock(monkeypatch)

    if task_name == "backfill_group_rankings":
        fake_service.backfill_rankings_optimized.side_effect = exc
        args = ("2026-03-20", "2026-03-20")
        kwargs = {}
    elif task_name == "gapfill_group_rankings":
        fake_service.find_missing_dates.return_value = [
            datetime(2026, 3, 20).date()
        ]
        fake_service.fill_gaps_optimized.side_effect = exc
        args = ()
        kwargs = {"max_days": 1}
    else:
        fake_service.backfill_rankings_optimized.side_effect = exc
        _patch_calendar_service(
            monkeypatch,
            module,
            datetime(2026, 3, 20, 17, 40),
        )
        args = ()
        kwargs = {}

    return getattr(module, task_name), args, kwargs, fake_db


@pytest.mark.parametrize(
    "task_name",
    [
        "backfill_group_rankings",
        "gapfill_group_rankings",
        "backfill_group_rankings_1year",
    ],
)
def test_group_backfill_tasks_retry_transient_database_errors(
    monkeypatch,
    task_name,
):
    task, args, kwargs, fake_db = _configure_failing_group_backfill(
        monkeypatch,
        task_name,
        _transient_database_error(),
    )
    retry = MagicMock(side_effect=Retry("retry"))
    monkeypatch.setattr(task, "retry", retry)

    with pytest.raises(Retry):
        task.run(*args, **kwargs)

    retry.assert_called_once()
    fake_db.rollback.assert_called_once()
    fake_db.close.assert_called_once()


@pytest.mark.parametrize(
    "task_name",
    [
        "backfill_group_rankings",
        "gapfill_group_rankings",
        "backfill_group_rankings_1year",
    ],
)
def test_group_backfill_tasks_preserve_non_transient_error_payloads(
    monkeypatch,
    task_name,
):
    task, args, kwargs, fake_db = _configure_failing_group_backfill(
        monkeypatch,
        task_name,
        RuntimeError("calculation failed"),
    )
    retry = MagicMock()
    monkeypatch.setattr(task, "retry", retry)

    result = task.run(*args, **kwargs)

    assert result["error"] == "calculation failed"
    retry.assert_not_called()
    fake_db.rollback.assert_called_once()
    fake_db.close.assert_called_once()


def test_gapfill_group_rankings_passes_market_to_service(monkeypatch):
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.group_rank_backfill_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(
        snapshot_module,
        "safe_publish_groups_bootstrap",
        lambda: None,
    )

    fake_service = MagicMock()
    fake_service.find_missing_dates.return_value = [
        datetime(2026, 3, 19).date()
    ]
    fake_service.fill_gaps_optimized.return_value = {
        "processed": 1,
        "errors": 0,
    }
    monkeypatch.setattr(
        module,
        "get_group_rank_service",
        lambda: fake_service,
    )

    result = module.gapfill_group_rankings.run(
        max_days=30,
        market="HK",
    )

    assert result["status"] == "complete"
    fake_service.find_missing_dates.assert_called_once_with(
        fake_db,
        lookback_days=30,
        market="HK",
    )
    fake_service.fill_gaps_optimized.assert_called_once_with(
        fake_db,
        [datetime(2026, 3, 19).date()],
        market="HK",
    )


def test_backfill_group_rankings_passes_market_to_service(monkeypatch):
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.group_rank_backfill_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(
        snapshot_module,
        "safe_publish_groups_bootstrap",
        lambda: None,
    )

    fake_service = MagicMock()
    fake_service.backfill_rankings_optimized.return_value = {
        "total_dates": 1,
        "deleted": 0,
        "processed": 1,
        "skipped": 0,
        "errors": 0,
    }
    monkeypatch.setattr(
        module,
        "get_group_rank_service",
        lambda: fake_service,
    )

    result = module.backfill_group_rankings.run(
        "2026-03-17",
        "2026-03-17",
        market="HK",
    )

    assert result["processed"] == 1
    fake_service.backfill_rankings_optimized.assert_called_once_with(
        fake_db,
        datetime(2026, 3, 17).date(),
        datetime(2026, 3, 17).date(),
        market="HK",
    )


def test_backfill_group_rankings_1year_passes_market_to_service(
    monkeypatch,
):
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.group_rank_backfill_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(
        monkeypatch,
        module,
        datetime(2026, 3, 20, 17, 40),
    )
    monkeypatch.setattr(
        snapshot_module,
        "safe_publish_groups_bootstrap",
        lambda: None,
    )

    fake_service = MagicMock()
    fake_service.backfill_rankings_optimized.return_value = {
        "total_dates": 1,
        "deleted": 0,
        "processed": 1,
        "skipped": 0,
        "errors": 0,
    }
    monkeypatch.setattr(
        module,
        "get_group_rank_service",
        lambda: fake_service,
    )

    result = module.backfill_group_rankings_1year.run(market="HK")

    assert result["processed"] == 1
    fake_service.backfill_rankings_optimized.assert_called_once_with(
        fake_db,
        datetime(2025, 3, 20).date(),
        datetime(2026, 3, 20).date(),
        market="HK",
    )


def test_backfill_does_not_publish_when_no_date_was_replaced(monkeypatch):
    import app.services.group_rankings_cache as cache_module
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.group_rank_backfill_tasks as module

    fake_db = MagicMock()
    fake_service = MagicMock()
    fake_service.backfill_rankings_optimized.return_value = {
        "total_dates": 1,
        "deleted": 0,
        "processed": 0,
        "skipped": 0,
        "errors": 1,
    }
    bump_epoch = MagicMock()
    publish = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)
    monkeypatch.setattr(cache_module, "bump_group_rankings_epoch", bump_epoch)
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", publish)
    _patch_serialized_lock(monkeypatch)

    result = module.backfill_group_rankings.run(
        "2026-03-20",
        "2026-03-20",
    )

    assert result["processed"] == 0
    bump_epoch.assert_not_called()
    publish.assert_not_called()


def test_gapfill_does_not_publish_when_no_date_was_filled(monkeypatch):
    import app.services.group_rankings_cache as cache_module
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.group_rank_backfill_tasks as module

    fake_db = MagicMock()
    fake_service = MagicMock()
    fake_service.find_missing_dates.return_value = [
        datetime(2026, 3, 20).date()
    ]
    fake_service.fill_gaps_optimized.return_value = {
        "processed": 0,
        "errors": 1,
    }
    bump_epoch = MagicMock()
    publish = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)
    monkeypatch.setattr(cache_module, "bump_group_rankings_epoch", bump_epoch)
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", publish)
    _patch_serialized_lock(monkeypatch)

    result = module.gapfill_group_rankings.run(max_days=1)

    assert result["processed"] == 0
    bump_epoch.assert_not_called()
    publish.assert_not_called()


def test_one_year_backfill_does_not_publish_when_no_date_was_replaced(
    monkeypatch,
):
    import app.services.group_rankings_cache as cache_module
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.group_rank_backfill_tasks as module

    fake_db = MagicMock()
    fake_service = MagicMock()
    fake_service.backfill_rankings_optimized.return_value = {
        "total_dates": 1,
        "deleted": 0,
        "processed": 0,
        "skipped": 0,
        "errors": 1,
    }
    bump_epoch = MagicMock()
    publish = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)
    monkeypatch.setattr(cache_module, "bump_group_rankings_epoch", bump_epoch)
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", publish)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(
        monkeypatch,
        module,
        datetime(2026, 3, 20, 17, 40),
    )

    result = module.backfill_group_rankings_1year.run()

    assert result["processed"] == 0
    bump_epoch.assert_not_called()
    publish.assert_not_called()


def test_extracted_tasks_keep_registered_names():
    from app.tasks import group_rank_tasks

    assert group_rank_tasks.backfill_group_rankings.name == (
        "app.tasks.group_rank_tasks.backfill_group_rankings"
    )
    assert group_rank_tasks.gapfill_group_rankings.name == (
        "app.tasks.group_rank_tasks.gapfill_group_rankings"
    )
    assert group_rank_tasks.backfill_group_rankings_1year.name == (
        "app.tasks.group_rank_tasks.backfill_group_rankings_1year"
    )
