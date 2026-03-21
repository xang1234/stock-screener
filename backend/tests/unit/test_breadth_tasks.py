from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock


def test_daily_breadth_refuses_to_publish_when_same_day_warmup_incomplete(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (True, False)
    fake_lock.release.return_value = True
    monkeypatch.setattr(
        "app.tasks.data_fetch_lock.DataFetchLock.get_instance",
        lambda: fake_lock,
    )
    monkeypatch.setattr("app.utils.market_hours.is_trading_day", lambda d: True)
    monkeypatch.setattr("app.utils.market_hours.get_eastern_now", lambda: datetime(2026, 3, 20, 17, 35, 0))

    fake_price_cache = MagicMock()
    fake_price_cache.get_warmup_metadata.return_value = {
        "status": "partial",
        "count": 9500,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
        "error": None,
    }

    fake_calculator = MagicMock()
    fake_calculator.price_cache = fake_price_cache
    fake_calculator.calculate_daily_breadth.return_value = {
        "stocks_up_4pct": 1,
        "stocks_down_4pct": 0,
        "ratio_5day": 1.0,
        "ratio_10day": 1.0,
        "stocks_up_25pct_quarter": 1,
        "stocks_down_25pct_quarter": 0,
        "stocks_up_25pct_month": 1,
        "stocks_down_25pct_month": 0,
        "stocks_up_50pct_month": 0,
        "stocks_down_50pct_month": 0,
        "stocks_up_13pct_34days": 1,
        "stocks_down_13pct_34days": 0,
        "total_stocks_scanned": 9500,
        "skipped_stocks": 500,
        "cache_miss_stocks": 25,
        "error_stocks": 0,
    }
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda db: fake_calculator)

    result = module.calculate_daily_breadth.run()

    assert "error" in result
    assert "warmup not complete" in result["error"].lower()
    fake_db.add.assert_not_called()
    fake_db.commit.assert_not_called()
