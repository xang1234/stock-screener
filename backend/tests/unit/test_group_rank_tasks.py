from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from app.services.ibd_group_rank_service import IncompleteGroupRankingCacheError


def _patch_serialized_lock(monkeypatch):
    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (True, False)
    fake_lock.release.return_value = True
    monkeypatch.setattr(
        "app.tasks.data_fetch_lock.DataFetchLock.get_instance",
        lambda: fake_lock,
    )


def test_daily_group_rankings_refuse_to_publish_when_warmup_incomplete(monkeypatch):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module, "is_trading_day", lambda d: True)
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: datetime(2026, 3, 20, 17, 40, 0),
    )

    fake_price_cache = MagicMock()
    fake_price_cache.get_warmup_metadata.return_value = {
        "status": "partial",
        "count": 9500,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service = MagicMock()
    fake_service.price_cache = fake_price_cache

    monkeypatch.setattr(
        module,
        "IBDGroupRankService",
        type(
            "FakeGroupRankServiceFacade",
            (),
            {"get_instance": staticmethod(lambda: fake_service)},
        ),
    )

    result = module.calculate_daily_group_rankings.run()

    assert "error" in result
    assert "warmup not complete" in result["error"].lower()
    fake_service.calculate_group_rankings.assert_not_called()
    fake_db.commit.assert_not_called()


def test_daily_group_rankings_refuse_to_publish_when_cache_only_inputs_missing(monkeypatch):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module, "is_trading_day", lambda d: True)
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: datetime(2026, 3, 20, 17, 40, 0),
    )

    fake_price_cache = MagicMock()
    fake_price_cache.get_warmup_metadata.return_value = {
        "status": "completed",
        "count": 10000,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service = MagicMock()
    fake_service.price_cache = fake_price_cache
    fake_service.calculate_group_rankings.side_effect = IncompleteGroupRankingCacheError(
        {
            "target_symbols": 100,
            "symbols_with_prices": 99,
            "cache_miss_symbols": 1,
            "spy_cached": True,
        }
    )

    monkeypatch.setattr(
        module,
        "IBDGroupRankService",
        type(
            "FakeGroupRankServiceFacade",
            (),
            {"get_instance": staticmethod(lambda: fake_service)},
        ),
    )

    result = module.calculate_daily_group_rankings.run()

    assert result["cache_only"] is True
    assert result["prefetch_stats"]["cache_miss_symbols"] == 1
    assert "missing cached price data" in result["error"].lower()
    fake_db.commit.assert_not_called()


def test_manual_group_rankings_keep_fetch_capable_behavior(monkeypatch):
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: datetime(2026, 3, 20, 17, 40, 0),
    )
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)

    fake_price_cache = MagicMock()
    fake_service = MagicMock()
    fake_service.price_cache = fake_price_cache
    fake_service.calculate_group_rankings.return_value = [
        {"industry_group": "Software", "avg_rs_rating": 95.0, "rank": 1}
    ]

    monkeypatch.setattr(
        module,
        "IBDGroupRankService",
        type(
            "FakeGroupRankServiceFacade",
            (),
            {"get_instance": staticmethod(lambda: fake_service)},
        ),
    )

    result = module.calculate_daily_group_rankings.run("2026-03-19")

    assert result["groups_ranked"] == 1
    assert result["cache_only"] is False
    fake_service.calculate_group_rankings.assert_called_once_with(
        fake_db,
        datetime(2026, 3, 19).date(),
        cache_only=False,
        require_complete_cache=False,
    )
